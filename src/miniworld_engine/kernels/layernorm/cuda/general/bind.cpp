#include <torch/extension.h>
std::vector<torch::Tensor> ln_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, bool rms, int64_t tpr,
                                  int64_t chunks, int64_t vec, int64_t rpt, int64_t waves);
std::vector<torch::Tensor> ln_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd,
                                  bool wgrad, bool rms, int64_t tpr, int64_t chunks, int64_t vec, int64_t rpt, int64_t grid_cap,
                                  bool gbf16);
std::vector<torch::Tensor> ln_rows_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, bool rms, int64_t tpr,
                                       int64_t ch, int64_t rpt, int64_t warps, int64_t ctas);
std::vector<torch::Tensor> ln_rows_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd, bool wgrad,
                                       bool rms, int64_t tpr, int64_t ch, int64_t rpt, int64_t warps, int64_t ctas, bool gbf16);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rows_fwd", &ln_rows_fwd, "LayerNorm / RMSNorm forward, B200 sub-warp rows (ln_rows.cu)");
  m.def("rows_bwd", &ln_rows_bwd, "LayerNorm / RMSNorm backward, B200 sub-warp rows (ln_rows.cu)");
  m.def("fwd", &ln_fwd, "general LayerNorm / RMSNorm forward");
  m.def("bwd", &ln_bwd, "general LayerNorm / RMSNorm backward");
}
