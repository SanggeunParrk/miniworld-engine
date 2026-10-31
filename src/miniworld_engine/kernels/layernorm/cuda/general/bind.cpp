#include <torch/extension.h>
std::vector<torch::Tensor> ln_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, bool rms, int64_t tpr,
                                  int64_t chunks, int64_t vec, int64_t rpt, int64_t waves);
std::vector<torch::Tensor> ln_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd,
                                  bool wgrad, bool rms, int64_t tpr, int64_t chunks, int64_t vec, int64_t rpt, int64_t grid_cap,
                                  bool gbf16);
std::vector<torch::Tensor> ln_fwd_bulk(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, bool rms, int64_t tpr,
                                       int64_t chunks, int64_t rs, int64_t stages, int64_t ctas);
std::vector<torch::Tensor> ln_bwd_bulk(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd,
                                       bool wgrad, bool rms, int64_t tpr, int64_t chunks, int64_t rs, int64_t stages, int64_t ctas,
                                       bool gbf16);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd_bulk", &ln_fwd_bulk, "general LayerNorm / RMSNorm forward, bulk-copy tiles (sm_90+)");
  m.def("bwd_bulk", &ln_bwd_bulk, "general LayerNorm / RMSNorm backward, bulk-copy tiles (sm_90+)");
  m.def("fwd", &ln_fwd, "general LayerNorm / RMSNorm forward");
  m.def("bwd", &ln_bwd, "general LayerNorm / RMSNorm backward");
}
