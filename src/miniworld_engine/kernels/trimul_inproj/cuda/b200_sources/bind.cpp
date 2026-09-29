// Python bindings of the B200 (sm_100a) bidirectional D128 TriMul kernels: K1, K3, B1r, B7r.
#include <torch/extension.h>

void k1_forward(torch::Tensor x, torch::Tensor w1, torch::Tensor mask, torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes,
                double eps, int64_t grid);
torch::Tensor k3_pack_wp(torch::Tensor wp);
void k3_forward(torch::Tensor x, torch::Tensor tri, torch::Tensor wp_perm, torch::Tensor wg, torch::Tensor g_in, torch::Tensor b_in,
                torch::Tensor g_out, torch::Tensor b_out, torch::Tensor y, int64_t L, double eps, int64_t save,
                c10::optional<torch::Tensor> ds, c10::optional<torch::Tensor> xn_out, c10::optional<torch::Tensor> mean_out,
                c10::optional<torch::Tensor> rs_out, int64_t grid, c10::optional<torch::Tensor> zero_buf);
void b1r_backward(torch::Tensor dy, torch::Tensor xn, torch::Tensor tri, torch::Tensor ds, torch::Tensor mean_o, torch::Tensor rs_o,
                  torch::Tensor wg, torch::Tensor wp, torch::Tensor go, torch::Tensor bo, torch::Tensor dg, torch::Tensor dtri,
                  torch::Tensor dwg, torch::Tensor dwp, torch::Tensor dgo, torch::Tensor dbo, torch::Tensor ring, torch::Tensor flags,
                  int64_t L, int64_t nf);
void b7r_backward(torch::Tensor x, torch::Tensor xn, torch::Tensor dy, torch::Tensor dl, torch::Tensor dr, torch::Tensor dgout,
                  torch::Tensor mask, torch::Tensor w1, torch::Tensor wg, torch::Tensor gi, torch::Tensor dx, torch::Tensor dw1,
                  torch::Tensor dgi, torch::Tensor dbi, torch::Tensor ring, torch::Tensor flags, double eps, int64_t sg);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k1_forward", &k1_forward);
  m.def("k3_pack_wp", &k3_pack_wp);
  m.def("k3_forward", &k3_forward, py::arg("x"), py::arg("tri"), py::arg("wp_perm"), py::arg("wg"), py::arg("g_in"), py::arg("b_in"),
        py::arg("g_out"), py::arg("b_out"), py::arg("y"), py::arg("L"), py::arg("eps"), py::arg("save"), py::arg("ds"), py::arg("xn_out"),
        py::arg("mean_out"), py::arg("rs_out"), py::arg("grid"), py::arg("zero_buf") = py::none());
  m.def("b1r_backward", &b1r_backward);
  m.def("b7r_backward", &b7r_backward);
}
