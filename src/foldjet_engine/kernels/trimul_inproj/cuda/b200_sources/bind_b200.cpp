// Python bindings of the B200 TriMul kernels built as ONE extension (two extensions compiling the same sources interpose each
// other's host symbols -- a launch helper's static "attribute set" flag then belongs to the other library's kernel):
//   inference (b200_infer): k1w (front), k3g (fused output, D <= 128), k3w (fused output, LayerNorms folded, D >= 256), wide_aux
//   training (b200_train): D64 / D128 -- trimul_forms (weight forms), k3g with saves, b1s / b1g (output-side backward), b7m / b7g
//                           (input side), grad_reduce (deterministic parameter gradients);
//                           D >= 256 -- k3w with saves, k1wb (front backward), wide_bwd
//                           D64 bidirectional hidden 128 -- k3g with saves, wide_bwd (hid_*, lnout), k1wb
#include <torch/extension.h>
void k1w_stats(torch::Tensor x, torch::Tensor mean, torch::Tensor rstd, double eps);
void k1w_forward_rf3(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                     c10::optional<torch::Tensor> tokmask, c10::optional<torch::Tensor> mean,
                     c10::optional<torch::Tensor> rstd, torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid,
                     double eps, double rf3);
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
                 int64_t L, int64_t grid, c10::optional<torch::Tensor> p_out, c10::optional<torch::Tensor> g_out, int64_t pdl);
void wide_ln_stats(torch::Tensor t, torch::Tensor mean, torch::Tensor rstd, double eps);
void ctr_forward(torch::Tensor planes, torch::Tensor t, int64_t d, int64_t dir, int64_t grid, int64_t pdl);
torch::Tensor b7m_prof();  // role probes of b7m (built with -DPROF=1; an empty tensor otherwise)
torch::Tensor b1u_tl();   // b1u phase trace (built with -DPT_TL=1; an empty tensor otherwise; b7m_prof returns b7m's then)
void wide_fold_prep(torch::Tensor wp, torch::Tensor go, torch::Tensor bo, torch::Tensor wg, torch::Tensor gi, torch::Tensor bi,
                    torch::Tensor wpq, torch::Tensor wgq, torch::Tensor vec);

void k1wb_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                  c10::optional<torch::Tensor> tokmask, torch::Tensor mean, torch::Tensor rstd, torch::Tensor gamma, torch::Tensor beta,
                  torch::Tensor da, torch::Tensor dpre, int64_t grid);
torch::Tensor wide_gate_bwd(torch::Tensor dy, torch::Tensor p, torch::Tensor g, c10::optional<torch::Tensor> ds, torch::Tensor vec,
                   torch::Tensor mu_o, torch::Tensor rs_o, torch::Tensor dpr, torch::Tensor dg_out, torch::Tensor S1,
                   torch::Tensor S2, torch::Tensor r01, int64_t L, bool defer, c10::optional<torch::Tensor> cnt);
torch::Tensor wide_lnout_bwd(torch::Tensor dout, torch::Tensor t, torch::Tensor mu_o, torch::Tensor rs_o, torch::Tensor S1, torch::Tensor S2,
                    torch::Tensor go, torch::Tensor dt, torch::Tensor dgb, bool defer);
torch::Tensor wide_lnin_bwd(torch::Tensor dxn, torch::Tensor x, torch::Tensor dy, torch::Tensor mu_i, torch::Tensor rs_i, torch::Tensor gi,
                   torch::Tensor dx, torch::Tensor dgb, bool defer);
void wide_colsum3(torch::Tensor pg, torch::Tensor po, torch::Tensor pi, torch::Tensor og, torch::Tensor oo, torch::Tensor oi, torch::Tensor cnt, int64_t S);
void hid_gate_bwd(torch::Tensor dy, torch::Tensor q, torch::Tensor g, c10::optional<torch::Tensor> ds, torch::Tensor vec,
                  torch::Tensor mu_o, torch::Tensor rs_o, torch::Tensor dpr, torch::Tensor dg_out, torch::Tensor S1, torch::Tensor S2,
                  torch::Tensor r01, int64_t L);
void hid_lnin_bwd(torch::Tensor dxn, torch::Tensor x, torch::Tensor dy, torch::Tensor gi, torch::Tensor dx, torch::Tensor dgb,
                  double eps);
void wide_ln_apply(torch::Tensor x, torch::Tensor mean, torch::Tensor rstd, torch::Tensor g, torch::Tensor b, torch::Tensor xn);
void lnpart_sum(torch::Tensor a, torch::Tensor b, torch::Tensor out);
void drop_scale(torch::Tensor r, torch::Tensor out, double p);
void trimul_forms(torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, torch::Tensor wg, torch::Tensor wp,
                  torch::Tensor go,
                  torch::Tensor out, c10::optional<torch::Tensor> rf3_mask, int64_t rf3_len, c10::optional<torch::Tensor> rf3_out,
                  c10::optional<torch::Tensor> drop_r, c10::optional<torch::Tensor> drop_out, double drop_p, bool drop_philox, int64_t gs_blocks,
                  int64_t gs_mode, c10::optional<torch::Tensor> gs_cnt);
void trimul_graph_rng_api(int64_t address);
void trimul_forms_gs_preload();
void wide_forms(torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, torch::Tensor wg, torch::Tensor wp,
                torch::Tensor out, c10::optional<torch::Tensor> drop_out, double drop_p);
void wide_grad_pack(torch::Tensor dw, torch::Tensor mm, torch::Tensor acc, torch::Tensor go, torch::Tensor bo, torch::Tensor out);
void grad_reduce(torch::Tensor pf, int64_t nf, torch::Tensor lo, int64_t nl, torch::Tensor pw, int64_t ns, torch::Tensor li, int64_t nc,
                 torch::Tensor out, c10::optional<torch::Tensor> wl_scale, bool skip_o);
void b1s_backward(torch::Tensor dy, torch::Tensor xn, torch::Tensor tri, torch::Tensor ds, torch::Tensor mean_o, torch::Tensor rs_o,
                  torch::Tensor wg, torch::Tensor wp, torch::Tensor wpf, torch::Tensor go, torch::Tensor bo, torch::Tensor dg, torch::Tensor dtri,
                  torch::Tensor part, torch::Tensor lnpart, torch::Tensor ring, torch::Tensor flags, int64_t L, int64_t nf);
void b7m_backward(torch::Tensor x, torch::Tensor xn, torch::Tensor dy, torch::Tensor dpl, torch::Tensor dgout,
                  c10::optional<torch::Tensor> tmask, torch::Tensor w1, torch::Tensor wg, torch::Tensor gi, torch::Tensor dx,
                  torch::Tensor dw1, torch::Tensor lnpart, double eps, int64_t L, c10::optional<torch::Tensor> xpf, int64_t xnf,
                  c10::optional<torch::Tensor> xlo, int64_t xnl, c10::optional<torch::Tensor> xout, int64_t xg);
void b1g_backward(torch::Tensor dy, torch::Tensor xn, torch::Tensor tri, torch::Tensor ds, torch::Tensor mean_o, torch::Tensor rs_o,
                  torch::Tensor wg, torch::Tensor wp, torch::Tensor wpf, torch::Tensor go, torch::Tensor bo, torch::Tensor dg, torch::Tensor dtri,
                  torch::Tensor part, torch::Tensor lnpart, torch::Tensor ring, torch::Tensor flags, int64_t L, int64_t nf);
void b7g_backward(torch::Tensor x, torch::Tensor xn, torch::Tensor dy, torch::Tensor dpl, torch::Tensor dgout,
                  c10::optional<torch::Tensor> tmask, torch::Tensor w1, torch::Tensor wg, torch::Tensor gi, torch::Tensor dx,
                  torch::Tensor dw1, torch::Tensor lnpart, torch::Tensor ring, torch::Tensor flags, double eps, int64_t sg, int64_t L);
void cgb_backward(torch::Tensor planes, torch::Tensor dt, torch::Tensor dpl, int64_t L, int64_t direction);
torch::Tensor b7g_tl();

torch::Tensor k1w_tl();
torch::Tensor k3g_tl();
torch::Tensor k3g_tt();
torch::Tensor k3w_tl();
torch::Tensor k3w_t2();
void tl_spin(int64_t cycles);
void k1h_forward_rf3(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, c10::optional<torch::Tensor> tokmask,
                     torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid, double eps, double rf3);
void k1h_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                 c10::optional<torch::Tensor> tokmask, torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid, double eps);
void k3h_forward(torch::Tensor x, torch::Tensor tri, torch::Tensor wp, torch::Tensor wg, torch::Tensor g_in, torch::Tensor b_in,
                 torch::Tensor g_out, torch::Tensor b_out, torch::Tensor y, double eps, int64_t grid, int64_t pdl);
void k3hf_forward(torch::Tensor x, torch::Tensor tri, torch::Tensor wpq, torch::Tensor wgq, torch::Tensor vec, torch::Tensor y,
                  double eps, int64_t grid, int64_t pdl);
torch::Tensor k1h_tl();
torch::Tensor k3h_tl();
torch::Tensor k3g_tlg();
torch::Tensor k3hf_tl();
torch::Tensor ph_k1h();
torch::Tensor ph_k3h();
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k1w_tl", &k1w_tl);
  m.def("k3g_tl", &k3g_tl);
  m.def("k3g_tt", &k3g_tt);
  m.def("k3w_tl", &k3w_tl);
  m.def("k3w_t2", &k3w_t2);
  m.def("tl_spin", &tl_spin);
  m.def("k1h_forward", &k1h_forward);
  m.def("k1h_forward_rf3", &k1h_forward_rf3);
  m.def("k3h_forward", &k3h_forward);
  m.def("k3hf_forward", &k3hf_forward);
  m.def("k1h_tl", &k1h_tl);
  m.def("k3h_tl", &k3h_tl);
  m.def("k3g_tlg", &k3g_tlg);
  m.def("k3hf_tl", &k3hf_tl);
  m.def("ph_k1h", &ph_k1h);
  m.def("ph_k3h", &ph_k3h);
  m.def("k1w_stats", &k1w_stats);
  m.def("k1w_forward", &k1w_forward);
  m.def("k1w_forward_rf3", &k1w_forward_rf3);
  m.def("k1w_prep", &k1w_prep);
  m.def("k3g_forward", &k3g_forward, py::arg("x"), py::arg("tri"), py::arg("wp_perm"), py::arg("wg"), py::arg("g_in"), py::arg("b_in"),
        py::arg("g_out"), py::arg("b_out"), py::arg("y"), py::arg("L"), py::arg("eps"), py::arg("save"), py::arg("ds"), py::arg("xn_out"),
        py::arg("mean_out"), py::arg("rs_out"), py::arg("grid"), py::arg("zero_buf") = py::none());
  m.def("k3w_forward", &k3w_forward);
  m.def("wide_ln_stats", &wide_ln_stats);
  m.def("ctr_forward", &ctr_forward);
  m.def("b7m_prof", &b7m_prof);
  m.def("b1u_tl", &b1u_tl);
  m.def("wide_fold_prep", &wide_fold_prep);
  m.def("k1wb_forward", &k1wb_forward);
  m.def("wide_gate_bwd", &wide_gate_bwd, py::arg("dy"), py::arg("p"), py::arg("g"), py::arg("ds"), py::arg("vec"), py::arg("mu_o"), py::arg("rs_o"),
        py::arg("dpr"), py::arg("dg_out"), py::arg("S1"), py::arg("S2"), py::arg("r01"), py::arg("L"), py::arg("defer") = false,
        py::arg("cnt") = py::none());
  m.def("wide_lnout_bwd", &wide_lnout_bwd, py::arg("dout"), py::arg("t"), py::arg("mu_o"), py::arg("rs_o"), py::arg("S1"), py::arg("S2"), py::arg("go"),
        py::arg("dt"), py::arg("dgb"), py::arg("defer") = false);
  m.def("wide_lnin_bwd", &wide_lnin_bwd, py::arg("dxn"), py::arg("x"), py::arg("dy"), py::arg("mu_i"), py::arg("rs_i"), py::arg("gi"), py::arg("dx"),
        py::arg("dgb"), py::arg("defer") = false);
  m.def("wide_colsum3", &wide_colsum3);
  m.def("wide_ln_apply", &wide_ln_apply);
  m.def("hid_gate_bwd", &hid_gate_bwd);
  m.def("hid_lnin_bwd", &hid_lnin_bwd);
  m.def("lnpart_sum", &lnpart_sum);
  m.def("drop_scale", &drop_scale);
  m.def("trimul_forms", &trimul_forms, py::arg("wl"), py::arg("wlg"), py::arg("wr"), py::arg("wrg"), py::arg("wg"), py::arg("wp"),
        py::arg("go"), py::arg("out"), py::arg("rf3_mask") = py::none(), py::arg("rf3_len") = 0, py::arg("rf3_out") = py::none(),
        py::arg("drop_r") = py::none(), py::arg("drop_out") = py::none(), py::arg("drop_p") = 0.0, py::arg("drop_philox") = false,
        py::arg("gs_blocks") = 0, py::arg("gs_mode") = 0, py::arg("gs_cnt") = py::none());
  m.def("trimul_graph_rng_api", &trimul_graph_rng_api);
  m.def("trimul_forms_gs_preload", &trimul_forms_gs_preload);
  m.def("wide_forms", &wide_forms, py::arg("wl"), py::arg("wlg"), py::arg("wr"), py::arg("wrg"), py::arg("wg"), py::arg("wp"), py::arg("out"),
        py::arg("drop_out") = py::none(), py::arg("drop_p") = 0.0);
  m.def("wide_grad_pack", &wide_grad_pack);
  m.def("grad_reduce", &grad_reduce, py::arg("pf"), py::arg("nf"), py::arg("lo"), py::arg("nl"), py::arg("pw"), py::arg("ns"), py::arg("li"),
        py::arg("nc"), py::arg("out"), py::arg("wl_scale") = py::none(), py::arg("skip_o") = false);
  m.def("b1s_backward", &b1s_backward);
  m.def("b7m_backward", &b7m_backward, py::arg("x"), py::arg("xn"), py::arg("dy"), py::arg("dpl"), py::arg("dgout"), py::arg("tmask"),
        py::arg("w1"), py::arg("wg"), py::arg("gi"), py::arg("dx"), py::arg("dw1"), py::arg("lnpart"), py::arg("eps"), py::arg("L"),
        py::arg("xpf") = py::none(), py::arg("xnf") = 0, py::arg("xlo") = py::none(), py::arg("xnl") = 0, py::arg("xout") = py::none(),
        py::arg("xg") = 0);
  m.def("b1g_backward", &b1g_backward);
  m.def("b7g_backward", &b7g_backward);
  m.def("cgb_backward", &cgb_backward);
  m.def("b7g_tl", &b7g_tl);
}
