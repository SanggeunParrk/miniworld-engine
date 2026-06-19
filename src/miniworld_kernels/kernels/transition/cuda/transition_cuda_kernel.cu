// vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/transition/transition_cuda_kernel.cu
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <cublas_v2.h>

namespace {

constexpr int kThreads = 256;

void check_cuda(cudaError_t error, const char* message) {
    TORCH_CHECK(error == cudaSuccess, message, ": ", cudaGetErrorString(error));
}

void check_cublas(cublasStatus_t status, const char* message) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, message, ": cuBLAS status=", status);
}

struct CublasHandleGuard {
    cublasHandle_t handle = nullptr;

    explicit CublasHandleGuard(cudaStream_t stream) {
        check_cublas(cublasCreate(&handle), "cublasCreate failed");
        check_cublas(cublasSetStream(handle, stream), "cublasSetStream failed");
    }

    ~CublasHandleGuard() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }
};

template <typename dst_t, typename src_t>
__global__ void cast_kernel(const src_t* src, dst_t* dst, int64_t numel) {
    const auto idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (auto i = idx; i < numel; i += stride) {
        dst[i] = static_cast<dst_t>(static_cast<float>(src[i]));
    }
}

__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <typename scalar_t>
__global__ void swish_mul_kernel(
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* out,
    int64_t numel
) {
    const auto idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (auto i = idx; i < numel; i += stride) {
        const float a_val = static_cast<float>(a[i]);
        const float b_val = static_cast<float>(b[i]);
        const float swish = a_val * sigmoid(a_val);
        out[i] = static_cast<scalar_t>(swish * b_val);
    }
}

template <typename scalar_t>
__global__ void transition_grad_kernel(
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* grad_expand,
    scalar_t* dA,
    scalar_t* dB,
    int64_t numel
) {
    const auto idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (auto i = idx; i < numel; i += stride) {
        const float a_val = static_cast<float>(a[i]);
        const float b_val = static_cast<float>(b[i]);
        const float grad_val = static_cast<float>(grad_expand[i]);
        const float sigmoid_val = sigmoid(a_val);
        const float swish_val = a_val * sigmoid_val;
        const float swish_grad = sigmoid_val + a_val * sigmoid_val * (1.0f - sigmoid_val);

        dA[i] = static_cast<scalar_t>(grad_val * b_val * swish_grad);
        dB[i] = static_cast<scalar_t>(grad_val * swish_val);
    }
}

template <typename scalar_t>
void launch_swish_mul(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    cudaStream_t stream
) {
    const auto numel = a.numel();
    if (numel == 0) {
        return;
    }
    const auto blocks = static_cast<int>((numel + kThreads - 1) / kThreads);

    swish_mul_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        numel
    );
    check_cuda(cudaGetLastError(), "swish_mul_kernel launch failed");
}

template <typename scalar_t>
void launch_transition_grad(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& grad_expand,
    torch::Tensor& dA,
    torch::Tensor& dB,
    cudaStream_t stream
) {
    const auto numel = a.numel();
    if (numel == 0) {
        return;
    }
    const auto blocks = static_cast<int>((numel + kThreads - 1) / kThreads);

    transition_grad_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        grad_expand.data_ptr<scalar_t>(),
        dA.data_ptr<scalar_t>(),
        dB.data_ptr<scalar_t>(),
        numel
    );
    check_cuda(cudaGetLastError(), "transition_grad_kernel launch failed");
}

torch::Tensor cast_tensor_to_dtype(
    const torch::Tensor& src,
    torch::ScalarType dst_type,
    cudaStream_t stream
) {
    if (src.scalar_type() == dst_type) {
        return src;
    }

    auto dst = torch::empty(src.sizes(), src.options().dtype(dst_type));
    const auto numel = src.numel();
    if (numel == 0) {
        return dst;
    }
    const auto blocks = static_cast<int>((numel + kThreads - 1) / kThreads);

    if (
        src.scalar_type() == torch::ScalarType::BFloat16 &&
        dst_type == torch::ScalarType::Float
    ) {
        cast_kernel<float, at::BFloat16><<<blocks, kThreads, 0, stream>>>(
            src.data_ptr<at::BFloat16>(),
            dst.data_ptr<float>(),
            numel
        );
    } else if (
        src.scalar_type() == torch::ScalarType::Float &&
        dst_type == torch::ScalarType::BFloat16
    ) {
        cast_kernel<at::BFloat16, float><<<blocks, kThreads, 0, stream>>>(
            src.data_ptr<float>(),
            dst.data_ptr<at::BFloat16>(),
            numel
        );
    } else {
        TORCH_CHECK(
            false,
            "Unsupported cast from ",
            src.scalar_type(),
            " to ",
            dst_type
        );
    }

    check_cuda(cudaGetLastError(), "cast_kernel launch failed");
    return dst;
}

void gemm_row_major(
    cublasHandle_t handle,
    bool trans_a,
    bool trans_b,
    int64_t m,
    int64_t n,
    int64_t k,
    const float* a,
    int64_t a_cols,
    const float* b,
    int64_t b_cols,
    float* c,
    float beta = 0.0f
) {
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    const auto op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    constexpr float alpha = 1.0f;

    check_cublas(
        cublasSgemm(
            handle,
            op_b,
            op_a,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            b,
            static_cast<int>(b_cols),
            a,
            static_cast<int>(a_cols),
            &beta,
            c,
            static_cast<int>(n)
        ),
        "cublasSgemm failed"
    );
}

void gemm_row_major(
    cublasHandle_t handle,
    bool trans_a,
    bool trans_b,
    int64_t m,
    int64_t n,
    int64_t k,
    const at::BFloat16* a,
    int64_t a_cols,
    const at::BFloat16* b,
    int64_t b_cols,
    at::BFloat16* c,
    float beta = 0.0f
) {
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    const auto op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    constexpr float alpha = 1.0f;

    check_cublas(
        cublasGemmEx(
            handle,
            op_b,
            op_a,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            b,
            CUDA_R_16BF,
            static_cast<int>(b_cols),
            a,
            CUDA_R_16BF,
            static_cast<int>(a_cols),
            &beta,
            c,
            CUDA_R_16BF,
            static_cast<int>(n),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ),
        "cublasGemmEx failed"
    );
}

template <typename scalar_t>
torch::Tensor transition_forward_impl(
    const torch::Tensor& x,
    const torch::Tensor& expand_a_weight,
    const torch::Tensor& expand_b_weight,
    const torch::Tensor& squeeze_weight,
    cudaStream_t stream,
    cublasHandle_t handle
) {
    const auto M = x.size(0);
    const auto N = x.size(1);
    const auto nN = expand_a_weight.size(0);

    auto A = torch::empty({M, nN}, x.options());
    auto B = torch::empty({M, nN}, x.options());
    auto expand = torch::empty({M, nN}, x.options());
    auto out = torch::empty({M, N}, x.options());

    gemm_row_major(
        handle,
        false,
        true,
        M,
        nN,
        N,
        x.data_ptr<scalar_t>(),
        N,
        expand_a_weight.data_ptr<scalar_t>(),
        N,
        A.data_ptr<scalar_t>()
    );
    gemm_row_major(
        handle,
        false,
        true,
        M,
        nN,
        N,
        x.data_ptr<scalar_t>(),
        N,
        expand_b_weight.data_ptr<scalar_t>(),
        N,
        B.data_ptr<scalar_t>()
    );

    launch_swish_mul<scalar_t>(A, B, expand, stream);

    gemm_row_major(
        handle,
        false,
        true,
        M,
        N,
        nN,
        expand.data_ptr<scalar_t>(),
        nN,
        squeeze_weight.data_ptr<scalar_t>(),
        nN,
        out.data_ptr<scalar_t>()
    );
    return out;
}

template <typename scalar_t>
std::vector<torch::Tensor> transition_backward_impl(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& expand_a_weight,
    const torch::Tensor& expand_b_weight,
    const torch::Tensor& squeeze_weight,
    cudaStream_t stream,
    cublasHandle_t handle
) {
    const auto M = x.size(0);
    const auto N = x.size(1);
    const auto nN = expand_a_weight.size(0);

    auto A = torch::empty({M, nN}, x.options());
    auto B = torch::empty({M, nN}, x.options());
    auto expand = torch::empty({M, nN}, x.options());
    auto grad_expand = torch::empty({M, nN}, x.options());
    auto dA = torch::empty({M, nN}, x.options());
    auto dB = torch::empty({M, nN}, x.options());
    auto grad_a_weight = torch::empty({nN, N}, x.options());
    auto grad_b_weight = torch::empty({nN, N}, x.options());
    auto grad_squeeze_weight = torch::empty({N, nN}, x.options());
    auto dx = torch::empty({M, N}, x.options());

    gemm_row_major(
        handle,
        false,
        true,
        M,
        nN,
        N,
        x.data_ptr<scalar_t>(),
        N,
        expand_a_weight.data_ptr<scalar_t>(),
        N,
        A.data_ptr<scalar_t>()
    );
    gemm_row_major(
        handle,
        false,
        true,
        M,
        nN,
        N,
        x.data_ptr<scalar_t>(),
        N,
        expand_b_weight.data_ptr<scalar_t>(),
        N,
        B.data_ptr<scalar_t>()
    );

    launch_swish_mul<scalar_t>(A, B, expand, stream);

    gemm_row_major(
        handle,
        false,
        false,
        M,
        nN,
        N,
        grad_output.data_ptr<scalar_t>(),
        N,
        squeeze_weight.data_ptr<scalar_t>(),
        nN,
        grad_expand.data_ptr<scalar_t>()
    );

    launch_transition_grad<scalar_t>(A, B, grad_expand, dA, dB, stream);

    gemm_row_major(
        handle,
        true,
        false,
        nN,
        N,
        M,
        dA.data_ptr<scalar_t>(),
        nN,
        x.data_ptr<scalar_t>(),
        N,
        grad_a_weight.data_ptr<scalar_t>()
    );
    gemm_row_major(
        handle,
        true,
        false,
        nN,
        N,
        M,
        dB.data_ptr<scalar_t>(),
        nN,
        x.data_ptr<scalar_t>(),
        N,
        grad_b_weight.data_ptr<scalar_t>()
    );
    gemm_row_major(
        handle,
        true,
        false,
        N,
        nN,
        M,
        grad_output.data_ptr<scalar_t>(),
        N,
        expand.data_ptr<scalar_t>(),
        nN,
        grad_squeeze_weight.data_ptr<scalar_t>()
    );
    gemm_row_major(
        handle,
        false,
        false,
        M,
        N,
        nN,
        dA.data_ptr<scalar_t>(),
        nN,
        expand_a_weight.data_ptr<scalar_t>(),
        N,
        dx.data_ptr<scalar_t>()
    );
    gemm_row_major(
        handle,
        false,
        false,
        M,
        N,
        nN,
        dB.data_ptr<scalar_t>(),
        nN,
        expand_b_weight.data_ptr<scalar_t>(),
        N,
        dx.data_ptr<scalar_t>(),
        1.0f
    );

    return {dx, grad_a_weight, grad_b_weight, grad_squeeze_weight};
}

}  // namespace

torch::Tensor transition_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& expand_a_weight,
    const torch::Tensor& expand_b_weight,
    const torch::Tensor& squeeze_weight
) {
    const c10::cuda::CUDAGuard device_guard(x.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    CublasHandleGuard cublas(stream);

    auto expand_a_work = cast_tensor_to_dtype(expand_a_weight, x.scalar_type(), stream);
    auto expand_b_work = cast_tensor_to_dtype(expand_b_weight, x.scalar_type(), stream);
    auto squeeze_work = cast_tensor_to_dtype(squeeze_weight, x.scalar_type(), stream);

    if (x.scalar_type() == torch::ScalarType::Float) {
        return transition_forward_impl<float>(
            x,
            expand_a_work,
            expand_b_work,
            squeeze_work,
            stream,
            cublas.handle
        );
    }
    if (x.scalar_type() == torch::ScalarType::BFloat16) {
        return transition_forward_impl<at::BFloat16>(
            x,
            expand_a_work,
            expand_b_work,
            squeeze_work,
            stream,
            cublas.handle
        );
    }

    TORCH_CHECK(false, "Unsupported dtype for transition_forward_cuda: ", x.scalar_type());
}

std::vector<torch::Tensor> transition_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& expand_a_weight,
    const torch::Tensor& expand_b_weight,
    const torch::Tensor& squeeze_weight
) {
    const c10::cuda::CUDAGuard device_guard(x.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    CublasHandleGuard cublas(stream);

    auto expand_a_work = cast_tensor_to_dtype(expand_a_weight, x.scalar_type(), stream);
    auto expand_b_work = cast_tensor_to_dtype(expand_b_weight, x.scalar_type(), stream);
    auto squeeze_work = cast_tensor_to_dtype(squeeze_weight, x.scalar_type(), stream);

    if (x.scalar_type() == torch::ScalarType::Float) {
        return transition_backward_impl<float>(
            grad_output,
            x,
            expand_a_work,
            expand_b_work,
            squeeze_work,
            stream,
            cublas.handle
        );
    }
    if (x.scalar_type() == torch::ScalarType::BFloat16) {
        return transition_backward_impl<at::BFloat16>(
            grad_output,
            x,
            expand_a_work,
            expand_b_work,
            squeeze_work,
            stream,
            cublas.handle
        );
    }

    TORCH_CHECK(false, "Unsupported dtype for transition_backward_cuda: ", x.scalar_type());
}
