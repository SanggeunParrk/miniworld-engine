// Python bindings of the B200 TriMul kernels built as ONE extension (two extensions compiling the same sources interpose each
// other's host symbols -- a launch helper's static "attribute set" flag then belongs to the other library's kernel):
//   inference (b200_infer): k1w (front), k3g (fused output, D <= 128), k3w (fused output, LayerNorms folded, D >= 256), wide_aux
//   training (b200_train): D64 / D128 -- k3g with saves, b1s / b1g (output-side backward), b7m / b7g (input side), lnpart_sum;
//                           D >= 256 -- k3w with saves, k1wb (front backward), wide_bwd
#include <torch/extension.h>
void k1w_stats(torch::Tensor x, torch::Tensor mean, torch::Tensor rstd, double eps);
void k1w_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                 c10::optional<torch::Tensor> tokmask, c10::optional<torch::Tensor> mean,
                 c10::optional<torch::Tensor> rstd, torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid,
                 double eps);
void k1w_prep(torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, c10::optional<torch::Tensor> wcopy,
              c10::optional<torch::Tensor> wp, c10::optional<torch::Tensor> wpp);
void k3g_forward(torch::Tensor x, torch::Tensor tri, torch::Tensor wp_perm, torch::Tensor wg, torch::Tensor g_in, torch::Tensor b_in,
                 torch::Tensor g_out, torch::Tensor b_out, torch::Tensor y, int64_t L, double eps, int64_t save,
                 c10::optional<torch::Tensor> ds, c10::optional<torch::Tensor> xn_out, c10::optional<torch::Tensor> mean_out,
                 c10::optional<torch::Tensor> rs_out, int64_t grid, c10::optional<torch::Tensor> zero_buf);
void k3w_forward(torch::Tensor x, torch::Tensor t, torch::Tensor wpq, torch::Tensor wgq, torch::Tensor vec, torch::Tensor mu_o,
                 torch::Tensor rs_o, torch::Tensor mu_i, torch::Tensor rs_i, c10::optional<torch::Tensor> ds, torch::Tensor y,
                 int64_t L, int64_t grid, c10::optional<torch::Tensor> p_out, c10::optional<torch::Tensor> g_out);
void wide_ln_stats(torch::Tensor t, torch::Tensor mean, torch::Tensor rstd, double eps);
void wide_fold_prep(torch::Tensor wp, torch::Tensor go, torch::Tensor bo, torch::Tensor wg, torch::Tensor gi, torch::Tensor bi,
                    torch::Tensor wpq, torch::Tensor wgq, torch::Tensor vec);

void k1wb_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                  c10::optional<torch::Tensor> tokmask, torch::Tensor mean, torch::Tensor rstd, torch::Tensor gamma, torch::Tensor beta,
                  torch::Tensor da, torch::Tensor dpre, int64_t grid);
void wide_gate_bwd(torch::Tensor dy, torch::Tensor p, torch::Tensor g, c10::optional<torch::Tensor> ds, torch::Tensor vec,
                   torch::Tensor mu_o, torch::Tensor rs_o, torch::Tensor dpr, torch::Tensor dg_out, torch::Tensor S1,
                   torch::Tensor S2, torch::Tensor r01, int64_t L);
void wide_lnout_bwd(torch::Tensor dout, torch::Tensor t, torch::Tensor mu_o, torch::Tensor rs_o, torch::Tensor S1, torch::Tensor S2,
                    torch::Tensor go, torch::Tensor dt, torch::Tensor dgb);
void wide_lnin_bwd(torch::Tensor dxn, torch::Tensor x, torch::Tensor dy, torch::Tensor mu_i, torch::Tensor rs_i, torch::Tensor gi,
                   torch::Tensor dx, torch::Tensor dgb);
void wide_ln_apply(torch::Tensor x, torch::Tensor mean, torch::Tensor rstd, torch::Tensor g, torch::Tensor b, torch::Tensor xn);
void lnpart_sum(torch::Tensor a, torch::Tensor b, torch::Tensor out);
void b1s_backward(torch::Tensor dy, torch::Tensor xn, torch::Tensor tri, torch::Tensor ds, torch::Tensor mean_o, torch::Tensor rs_o,
                  torch::Tensor wg, torch::Tensor wp, torch::Tensor go, torch::Tensor bo, torch::Tensor dg, torch::Tensor dtri,
                  torch::Tensor dwg, torch::Tensor dwp, torch::Tensor lnpart, torch::Tensor ring, torch::Tensor flags,
                  int64_t L, int64_t nf);
void b7m_backward(torch::Tensor x, torch::Tensor xn, torch::Tensor dy, torch::Tensor dpl, torch::Tensor dgout, torch::Tensor mask,
                  torch::Tensor w1, torch::Tensor wg, torch::Tensor gi, torch::Tensor dx, torch::Tensor dw1, torch::Tensor lnpart,
                  double eps);
void b1g_backward(torch::Tensor dy, torch::Tensor xn, torch::Tensor tri, torch::Tensor ds, torch::Tensor mean_o, torch::Tensor rs_o,
                  torch::Tensor wg, torch::Tensor wp, torch::Tensor go, torch::Tensor bo, torch::Tensor dg, torch::Tensor dtri,
                  torch::Tensor dwg, torch::Tensor dwp, torch::Tensor lnpart, torch::Tensor ring, torch::Tensor flags,
                  int64_t L, int64_t nf);
void b7g_backward(torch::Tensor x, torch::Tensor xn, torch::Tensor dy, torch::Tensor dpl, torch::Tensor dgout, torch::Tensor mask,
                  torch::Tensor w1, torch::Tensor wg, torch::Tensor gi, torch::Tensor dx, torch::Tensor dw1, torch::Tensor lnpart,
                  torch::Tensor ring, torch::Tensor flags, double eps, int64_t sg);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k1w_stats", &k1w_stats);
  m.def("k1w_forward", &k1w_forward);
  m.def("k1w_prep", &k1w_prep);
  m.def("k3g_forward", &k3g_forward, py::arg("x"), py::arg("tri"), py::arg("wp_perm"), py::arg("wg"), py::arg("g_in"), py::arg("b_in"),
        py::arg("g_out"), py::arg("b_out"), py::arg("y"), py::arg("L"), py::arg("eps"), py::arg("save"), py::arg("ds"), py::arg("xn_out"),
        py::arg("mean_out"), py::arg("rs_out"), py::arg("grid"), py::arg("zero_buf") = py::none());
  m.def("k3w_forward", &k3w_forward);
  m.def("wide_ln_stats", &wide_ln_stats);
  m.def("wide_fold_prep", &wide_fold_prep);
  m.def("k1wb_forward", &k1wb_forward);
  m.def("wide_gate_bwd", &wide_gate_bwd);
  m.def("wide_lnout_bwd", &wide_lnout_bwd);
  m.def("wide_lnin_bwd", &wide_lnin_bwd);
  m.def("wide_ln_apply", &wide_ln_apply);
  m.def("lnpart_sum", &lnpart_sum);
  m.def("b1s_backward", &b1s_backward);
  m.def("b7m_backward", &b7m_backward);
  m.def("b1g_backward", &b1g_backward);
  m.def("b7g_backward", &b7g_backward);
}
