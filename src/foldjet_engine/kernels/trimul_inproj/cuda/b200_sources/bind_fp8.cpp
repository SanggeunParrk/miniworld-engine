// Python bindings of the B200 FP8 TriMul kernels (b200_fp8.py): ONE extension of its own (the default extension is untouched).
#include <torch/extension.h>
void ctr8_forward(torch::Tensor planes, torch::Tensor t, torch::Tensor csc, int64_t d, int64_t dir, int64_t grid, int64_t pdl, int64_t cfg);
void k1w8_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                  c10::optional<torch::Tensor> tokmask, c10::optional<torch::Tensor> mean, c10::optional<torch::Tensor> rstd,
                  torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid, double eps, torch::Tensor qinv,
                  torch::Tensor amax, int64_t ngrp);
void k1w8_forward_rf3(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                      c10::optional<torch::Tensor> tokmask, c10::optional<torch::Tensor> mean, c10::optional<torch::Tensor> rstd,
                      torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid, double eps, double rf3,
                      torch::Tensor qinv, torch::Tensor amax, int64_t ngrp);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ctr8_forward", &ctr8_forward);
  m.def("k1w8_forward", &k1w8_forward);
  m.def("k1w8_forward_rf3", &k1w8_forward_rf3);
}
