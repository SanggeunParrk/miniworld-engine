// vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/layernorm/layer_norm_cuda_kernel.cu
/*
 * layer_norm_cuda_kernel.cu
 *
 * Forward LayerNorm in plain CUDA.
 * Matches the Triton layer_norm_fwd_fused algorithm:
 *   mean  = sum(x) / N
 *   var   = sum((x - mean)^2) / N
 *   rstd  = 1 / sqrt(var + eps)
 *   y     = (x - mean) * rstd * w + b
 *
 * One CUDA block per row.
 * Supports float32 / float16 / bfloat16 via template.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

// ─────────────────────────────────────────────
// 1.  Warp-level reduction helpers
// ─────────────────────────────────────────────

__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Full block reduction.
// smem must have at least (blockDim.x / 32) floats.
// Returns the reduced value in thread 0; other threads get garbage.
__inline__ __device__ float block_reduce_sum(float val, float* smem) {
    const int lane  = threadIdx.x & 31;          // lane within warp
    const int wid   = threadIdx.x >> 5;          // warp index

    val = warp_reduce_sum(val);                   // reduce within warp

    if (lane == 0) smem[wid] = val;              // each warp's partial
    __syncthreads();

    const int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.f;
    val = warp_reduce_sum(val);                   // reduce across warps
    return val;
}

// ─────────────────────────────────────────────
// 2.  Forward kernel  (one block = one row)
// ─────────────────────────────────────────────

template <typename scalar_t>
__global__ void layer_norm_fwd_kernel(
    const scalar_t* __restrict__ X,   // [M, N]
    scalar_t*       __restrict__ Y,   // [M, N]
    const scalar_t* __restrict__ W,   // [N]
    const scalar_t* __restrict__ B,   // [N]
    float*          __restrict__ Mean, // [M]
    float*          __restrict__ Rstd, // [M]
    const int N,
    const float eps)
{
    // Shared memory: enough for one float per warp
    extern __shared__ float smem[];   // size = ceil(blockDim.x/32) floats

    const int row = blockIdx.x;
    const scalar_t* __restrict__ x = X + row * N;
    scalar_t*       __restrict__ y = Y + row * N;

    // ── Pass 1: mean ─────────────────────────
    float sum = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += (float)x[i];

    sum = block_reduce_sum(sum, smem);

    float mean;
    if (threadIdx.x == 0) {
        mean    = sum / (float)N;
        smem[0] = mean;              // broadcast via smem
    }
    __syncthreads();
    mean = smem[0];

    // ── Pass 2: variance ─────────────────────
    float var = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float d = (float)x[i] - mean;
        var += d * d;
    }

    var = block_reduce_sum(var, smem);

    float rstd;
    if (threadIdx.x == 0) {
        rstd    = rsqrtf(var / (float)N + eps);
        smem[0] = rstd;              // broadcast via smem
        Mean[row] = mean;
        Rstd[row] = rstd;
    }
    __syncthreads();
    rstd = smem[0];

    // ── Pass 3: normalize + affine ────────────
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x_hat = ((float)x[i] - mean) * rstd;
        y[i] = (scalar_t)(x_hat * (float)W[i] + (float)B[i]);
    }
}

// ─────────────────────────────────────────────
// 3.  Host-side launcher
// ─────────────────────────────────────────────

// Next power of two, clamped to [32, 1024]
static int choose_block_size(int N) {
    int t = 32;
    while (t < N && t < 1024) t <<= 1;
    return t;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
layer_norm_cuda_fwd(
    torch::Tensor x,        // [..., N]  – any leading dims
    torch::Tensor weight,   // [N]
    torch::Tensor bias,     // [N]
    float eps)
{
    TORCH_CHECK(x.is_cuda(),      "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(),   "bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const int N = x.size(-1);
    const int M = x.numel() / N;

    auto y    = torch::empty_like(x);
    auto mean = torch::empty({M}, x.options().dtype(torch::kFloat32));
    auto rstd = torch::empty({M}, x.options().dtype(torch::kFloat32));

    const int block = choose_block_size(N);
    const int grid  = M;
    // smem: one float per warp
    const int smem_bytes = ((block + 31) / 32) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "layer_norm_fwd_cuda",
        [&]() {
            layer_norm_fwd_kernel<scalar_t><<<grid, block, smem_bytes>>>(
                x.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                mean.data_ptr<float>(),
                rstd.data_ptr<float>(),
                N, eps);
        });

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "layer_norm_fwd_cuda kernel failed");
    return {y, mean, rstd};
}

// ─────────────────────────────────────────────
// 4.  pybind11 module
// ─────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "layer_norm_fwd",
        &layer_norm_cuda_fwd,
        "LayerNorm forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("eps") = 1e-5f);
}
