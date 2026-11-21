// token_dit_fp8_bind_sm100.cu -- the python module of the B200 e4m3 token DiT rows: token_dit_fp8_rows.cu (arch-neutral CUDA: the quantising AdaLN row passes and
// the row quantizer) built for sm_100a without the sm_90a SwiGLU GEMM (gemm_swiglu_f8_sm100.cu is the B200 GEMM, a cubin launched by token_dit_fp8_b200.py).
#include <torch/extension.h>

void adaln_in_rows_cf_q8(at::Tensor xin, at::Tensor g, at::Tensor r, at::Tensor b, at::Tensor b2, at::Tensor c, at::Tensor stats,
                         at::Tensor out8, at::Tensor srow, double eps, double ceps, bool pdl);
void adaln_in_rows_cf_par_q8(at::Tensor xin, at::Tensor g, at::Tensor r, at::Tensor b, at::Tensor b2, at::Tensor c, at::Tensor stats,
                             at::Tensor out, at::Tensor out8, at::Tensor srow, double eps, double ceps, bool pdl, bool sig);
void resgate_adaln_cf_q8(at::Tensor xin, at::Tensor ya, at::Tensor g, at::Tensor r, at::Tensor b, at::Tensor b2, at::Tensor stats,
                         at::Tensor out8, at::Tensor srow, double eps, bool pdl);
void quant_rows(at::Tensor x, at::Tensor out8, at::Tensor srow, bool pdl);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adaln_in_rows_cf_q8", &adaln_in_rows_cf_q8);
  m.def("adaln_in_rows_cf_par_q8", &adaln_in_rows_cf_par_q8);
  m.def("resgate_adaln_cf_q8", &resgate_adaln_cf_q8);
  m.def("quant_rows", &quant_rows);
}
