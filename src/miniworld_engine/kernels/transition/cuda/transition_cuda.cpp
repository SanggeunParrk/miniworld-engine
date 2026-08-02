// vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/transition/transition_cuda.cpp
#include <torch/extension.h>

namespace {

void check_transition_inputs(
    const torch::Tensor& x,
    const torch::Tensor& expand_a_weight,
    const torch::Tensor& expand_b_weight,
    const torch::Tensor& squeeze_weight,
    int64_t n
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(expand_a_weight.is_cuda(), "expand_a_weight must be a CUDA tensor");
    TORCH_CHECK(expand_b_weight.is_cuda(), "expand_b_weight must be a CUDA tensor");
    TORCH_CHECK(squeeze_weight.is_cuda(), "squeeze_weight must be a CUDA tensor");
    TORCH_CHECK(
        x.get_device() == expand_a_weight.get_device() &&
        x.get_device() == expand_b_weight.get_device() &&
        x.get_device() == squeeze_weight.get_device(),
        "all tensors must be on the same CUDA device"
    );

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(expand_a_weight.is_contiguous(), "expand_a_weight must be contiguous");
    TORCH_CHECK(expand_b_weight.is_contiguous(), "expand_b_weight must be contiguous");
    TORCH_CHECK(squeeze_weight.is_contiguous(), "squeeze_weight must be contiguous");

    TORCH_CHECK(
        x.scalar_type() == torch::ScalarType::Float ||
        x.scalar_type() == torch::ScalarType::BFloat16,
        "x must be float32 or bfloat16"
    );
    TORCH_CHECK(
        expand_a_weight.scalar_type() == torch::ScalarType::Float ||
        expand_a_weight.scalar_type() == torch::ScalarType::BFloat16,
        "weights must be float32 or bfloat16"
    );
    TORCH_CHECK(
        expand_a_weight.scalar_type() == expand_b_weight.scalar_type() &&
        expand_a_weight.scalar_type() == squeeze_weight.scalar_type(),
        "all weights must share the same dtype"
    );

    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor of shape (M, N)");
    TORCH_CHECK(
        expand_a_weight.dim() == 2,
        "expand_a_weight must be a 2D tensor of shape (nN, N)"
    );
    TORCH_CHECK(
        expand_b_weight.dim() == 2,
        "expand_b_weight must be a 2D tensor of shape (nN, N)"
    );
    TORCH_CHECK(
        squeeze_weight.dim() == 2,
        "squeeze_weight must be a 2D tensor of shape (N, nN)"
    );

    const auto N = x.size(1);
    const auto nN = expand_a_weight.size(0);

    TORCH_CHECK(expand_a_weight.size(1) == N, "expand_a_weight.shape[1] must match x.shape[1]");
    TORCH_CHECK(expand_b_weight.sizes() == expand_a_weight.sizes(), "expand_b_weight must match expand_a_weight shape");
    TORCH_CHECK(squeeze_weight.size(0) == N, "squeeze_weight.shape[0] must match x.shape[1]");
    TORCH_CHECK(squeeze_weight.size(1) == nN, "squeeze_weight.shape[1] must match expand weight shape[0]");

    TORCH_CHECK(n > 0, "n must be positive");
    TORCH_CHECK(nN == n * N, "expand weight shape[0] must equal n * N");
}

}  // namespace

torch::Tensor transition_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& expand_a_weight,
    const torch::Tensor& expand_b_weight,
    const torch::Tensor& squeeze_weight
);

std::vector<torch::Tensor> transition_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& expand_a_weight,
    const torch::Tensor& expand_b_weight,
    const torch::Tensor& squeeze_weight
);

torch::Tensor transition_forward(
    const torch::Tensor& x,
    const torch::Tensor& expand_a_weight,
    const torch::Tensor& expand_b_weight,
    const torch::Tensor& squeeze_weight,
    int64_t n
) {
    check_transition_inputs(x, expand_a_weight, expand_b_weight, squeeze_weight, n);
    return transition_forward_cuda(x, expand_a_weight, expand_b_weight, squeeze_weight);
}

std::vector<torch::Tensor> transition_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& expand_a_weight,
    const torch::Tensor& expand_b_weight,
    const torch::Tensor& squeeze_weight,
    int64_t n
) {
    check_transition_inputs(x, expand_a_weight, expand_b_weight, squeeze_weight, n);
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.dim() == 2, "grad_output must be a 2D tensor of shape (M, N)");
    TORCH_CHECK(grad_output.sizes() == x.sizes(), "grad_output shape must match x shape");
    TORCH_CHECK(
        grad_output.scalar_type() == x.scalar_type(),
        "grad_output dtype must match x dtype"
    );

    return transition_backward_cuda(
        grad_output,
        x,
        expand_a_weight,
        expand_b_weight,
        squeeze_weight
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &transition_forward, "Transition forward (CUDA)");
    m.def("backward", &transition_backward, "Transition backward (CUDA)");
}
