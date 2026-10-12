#include <torch/extension.h>
void wide_fold_prep(torch::Tensor wp, torch::Tensor go, torch::Tensor bo, torch::Tensor wg, torch::Tensor gi, torch::Tensor bi,
                    torch::Tensor wpq, torch::Tensor wgq, torch::Tensor vec);
void wide_gate_bwd(torch::Tensor dy, torch::Tensor p, torch::Tensor g, c10::optional<torch::Tensor> ds, torch::Tensor vec,
                   torch::Tensor mu_o, torch::Tensor rs_o, torch::Tensor dpr, torch::Tensor dg_out, torch::Tensor S1,
                   torch::Tensor S2, torch::Tensor r01, int64_t L);
void wide_lnout_bwd(torch::Tensor dout, torch::Tensor t, torch::Tensor mu_o, torch::Tensor rs_o, torch::Tensor S1, torch::Tensor S2,
                    torch::Tensor go, torch::Tensor dt, torch::Tensor dgb);
void wide_lnin_bwd(torch::Tensor dxn, torch::Tensor x, torch::Tensor dy, torch::Tensor mu_i, torch::Tensor rs_i, torch::Tensor gi,
                   torch::Tensor dx, torch::Tensor dgb);
void wide_ln_apply(torch::Tensor x, torch::Tensor mean, torch::Tensor rstd, torch::Tensor g, torch::Tensor b, torch::Tensor xn);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fold", &wide_fold_prep);
  m.def("gate_bwd", &wide_gate_bwd);
  m.def("lnout_bwd", &wide_lnout_bwd);
  m.def("lnin_bwd", &wide_lnin_bwd);
  m.def("ln_apply", &wide_ln_apply);
}
