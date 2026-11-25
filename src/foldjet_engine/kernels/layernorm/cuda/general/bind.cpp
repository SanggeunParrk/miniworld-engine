#include <torch/extension.h>
std::vector<torch::Tensor> ln_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, bool rms, int64_t tpr,
                                  int64_t chunks, int64_t vec);
std::vector<torch::Tensor> ln_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd,
                                  bool wgrad, bool rms, int64_t tpr, int64_t chunks, int64_t vec, int64_t grid_cap);
std::vector<torch::Tensor> ln_rows_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, int64_t mode);
std::vector<torch::Tensor> ln_rows_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd, bool wgrad,
                                       int64_t unused, int64_t ctas_per_sm);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &ln_fwd, "general LayerNorm / RMSNorm forward");
  m.def("bwd", &ln_bwd, "general LayerNorm / RMSNorm backward");
  m.def("rows_fwd", &ln_rows_fwd, "LayerNorm forward of rows whose width is not a 16-byte multiple (ln_rows.cu)");
  m.def("rows_bwd", &ln_rows_bwd, "LayerNorm backward of rows whose width is not a 16-byte multiple (ln_rows.cu)");
}
