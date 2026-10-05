// ops.cu -- torch bindings of the A100 (sm_80) ConditionedTransition kernels (ct_rows.cuh).  Built on first use by kernels/conditioned_transition/cuda/sm80.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <type_traits>

#include "adaln_launch.cuh"
#include "ct_atom_bwd.cuh"
#include "ct_atom_fwd.cuh"
#include "ct_atom_fwd_tf32.cuh"
#include "ct_rows.cuh"

namespace {
using namespace adl;

template <typename T> T* P_(const at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
template <typename T> const T* CP_(const at::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <typename T> const T* OP_(const c10::optional<at::Tensor>& t) { return t.has_value() ? reinterpret_cast<const T*>(t->data_ptr()) : nullptr; }
cudaStream_t S_() { return at::cuda::getCurrentCUDAStream(); }

template <typename F> void with_dtype(const at::Tensor& t, const char* what, F&& f) {
  if (t.scalar_type() == at::kFloat) f(float{});
  else if (t.scalar_type() == at::kBFloat16) f(bf{});
  else TORCH_CHECK(false, what, ": fp32 or bf16 rows");
}

void check_rows(const at::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.size(1) == cols && t.stride(1) == 1, name, ": a CUDA [rows, ", cols, "] view with unit column stride");
  const int64_t es = t.element_size();
  TORCH_CHECK(t.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % (4 * es) == 0, name, ": rows must start on a ", 4 * es, "-byte boundary");
}

unsigned flat_grid(int64_t vectors) { return (unsigned)std::min<int64_t>((vectors + 255) / 256, 108 * 32); }

// h [M, N] = silu(a) b, ab [M, 2 N]
void swiglu_fwd(at::Tensor ab, at::Tensor h) {
  const int64_t M = ab.size(0), N = ab.size(1) / 2;
  check_rows(ab, "swiglu_fwd ab", 2 * N);
  check_rows(h, "swiglu_fwd h", N);
  TORCH_CHECK(N % 4 == 0 && h.is_contiguous() && h.size(0) == M && h.scalar_type() == ab.scalar_type(), "swiglu_fwd: h [M, N] contiguous like ab");
  const at::cuda::CUDAGuard g(ab.device());
  with_dtype(ab, "swiglu_fwd", [&](auto t) {
    using T = decltype(t);
    swiglu_fwd_kernel<T><<<flat_grid(M * N / 4), 256, 0, S_()>>>(CP_<T>(ab), ab.stride(0), P_<T>(h), M, (int)(N / 4));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dab [M, 2 N] = [da | db], h [M, N] from dh [M, N] and ab [M, 2 N]
void swiglu_bwd(at::Tensor dh, at::Tensor ab, at::Tensor dab, at::Tensor h) {
  const int64_t M = dh.size(0), N = dh.size(1);
  check_rows(dh, "swiglu_bwd dh", N);
  check_rows(ab, "swiglu_bwd ab", 2 * N);
  TORCH_CHECK(N % 4 == 0 && dh.is_contiguous() && dab.is_contiguous() && h.is_contiguous() && dab.size(0) == M && dab.size(1) == 2 * N && h.size(0) == M && h.size(1) == N, "swiglu_bwd: shapes");
  TORCH_CHECK(dh.scalar_type() == ab.scalar_type() && dab.scalar_type() == ab.scalar_type() && h.scalar_type() == ab.scalar_type(), "swiglu_bwd: one dtype");
  const at::cuda::CUDAGuard g(dh.device());
  with_dtype(ab, "swiglu_bwd", [&](auto t) {
    using T = decltype(t);
    swiglu_bwd_kernel<T><<<flat_grid(M * N / 4), 256, 0, S_()>>>(CP_<T>(dh), CP_<T>(ab), ab.stride(0), P_<T>(dab), P_<T>(h), M, (int)(N / 4));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y [M, d] = (x) + sigmoid(g[r % P]) z; g [P, d]
void gate_res_fwd(c10::optional<at::Tensor> x, at::Tensor z, at::Tensor g, at::Tensor y) {
  const int64_t M = z.size(0), d = z.size(1), P = g.size(0);
  check_rows(z, "gate_res_fwd z", d);
  check_rows(g, "gate_res_fwd g", d);
  TORCH_CHECK(z.is_contiguous() && g.is_contiguous() && y.is_contiguous() && y.size(0) == M && y.size(1) == d, "gate_res_fwd: contiguous z, g, y [M, d]");
  TORCH_CHECK(P > 0 && M % P == 0 && z.scalar_type() == g.scalar_type() && y.scalar_type() == z.scalar_type(), "gate_res_fwd: g has `period` rows, one dtype");
  if (x) TORCH_CHECK(x->is_cuda() && x->is_contiguous() && x->scalar_type() == z.scalar_type() && x->size(0) == M && x->size(1) == d, "gate_res_fwd: x like z");
  const at::cuda::CUDAGuard gd(z.device());
  with_dtype(z, "gate_res_fwd", [&](auto t) {
    using T = decltype(t);
    gate_res_fwd_kernel<T><<<flat_grid(M * d / 4), 256, 0, S_()>>>(OP_<T>(x), CP_<T>(z), CP_<T>(g), P_<T>(y), M, (int)(d / 4), P);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dz, dg [M, d] and pb [blocks, d] fp32 (the column sums of dg) from dy, z, g (one gate row per row)
void gate_res_bwd(at::Tensor dy, at::Tensor z, at::Tensor g, at::Tensor dz, at::Tensor dg, at::Tensor pb) {
  const int64_t M = z.size(0), d = z.size(1);
  for (const auto* t : {&dy, &z, &g, &dz, &dg}) TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->size(0) == M && t->size(1) == d && t->scalar_type() == z.scalar_type(), "gate_res_bwd: contiguous [M, d] of one dtype");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(z.data_ptr()) % 16 == 0 && pb.is_cuda() && pb.is_contiguous() && pb.scalar_type() == at::kFloat && pb.dim() == 2 && pb.size(1) == d, "gate_res_bwd: pb [blocks, d] fp32");
  const at::cuda::CUDAGuard gd(z.device());
  auto run = [&](auto nt, auto r) {
    constexpr int NT = decltype(nt)::value, R = decltype(r)::value;
    with_dtype(z, "gate_res_bwd", [&](auto t) {
      using T = decltype(t);
      gate_res_bwd_kernel<NT, R, T><<<(unsigned)pb.size(0), dim3(NT, R), 0, S_()>>>(CP_<T>(dy), CP_<T>(z), CP_<T>(g), P_<T>(dz), P_<T>(dg), P_<float>(pb), M);
    });
  };
  if (d == 128) run(std::integral_constant<int, 32>{}, std::integral_constant<int, 8>{});
  else if (d == 384) run(std::integral_constant<int, 96>{}, std::integral_constant<int, 2>{});
  else if (d == 768) run(std::integral_constant<int, 192>{}, std::integral_constant<int, 1>{});
  else TORCH_CHECK(false, "gate_res_bwd: width 128, 384 or 768, got ", d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------------------------------------- atom width (128), bf16
void check_bf16_rows(const at::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dim() == 2 && t.size(1) == cols && t.scalar_type() == at::kBFloat16, name, ": a contiguous bf16 CUDA [rows, ", cols, "]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": 16-byte aligned");
}

// y [M, 128] = x + sigmoid(cond Wsc^T + bsc) (silu(xa Wa^T) (xa Wb^T)) Ws^T; wab [512, 128] = [Wa; Wb], wd [128, 256] = Ws; x null: the update alone; zs [M, 128] = rn(z) when given
void ct_atom_fwd(at::Tensor xa, c10::optional<at::Tensor> xin, at::Tensor cond, at::Tensor wab, at::Tensor wd, at::Tensor wsc, at::Tensor bsc, at::Tensor y,
                 c10::optional<at::Tensor> zs) {
  const int64_t M = xa.size(0), P = cond.size(0);
  check_bf16_rows(xa, "ct_atom_fwd xa", 128);
  check_bf16_rows(cond, "ct_atom_fwd cond", 128);
  check_bf16_rows(wab, "ct_atom_fwd wab", 128);
  check_bf16_rows(wsc, "ct_atom_fwd wsc", 128);
  check_bf16_rows(y, "ct_atom_fwd y", 128);
  TORCH_CHECK(wab.size(0) == 512 && wd.is_cuda() && wd.is_contiguous() && wd.dim() == 2 && wd.size(0) == 128 && wd.size(1) == 256 && wd.scalar_type() == at::kBFloat16 &&
              reinterpret_cast<uintptr_t>(wd.data_ptr()) % 16 == 0, "ct_atom_fwd: wab [512, 128], wd [128, 256] bf16");
  TORCH_CHECK(y.size(0) == M && P > 0 && M % P == 0, "ct_atom_fwd: y like xa, P divides M");
  TORCH_CHECK(bsc.is_cuda() && bsc.is_contiguous() && bsc.numel() == 128 && bsc.scalar_type() == at::kBFloat16 && reinterpret_cast<uintptr_t>(bsc.data_ptr()) % 16 == 0, "ct_atom_fwd: bsc [128] bf16");
  if (xin) check_bf16_rows(*xin, "ct_atom_fwd x", 128);
  if (zs) check_bf16_rows(*zs, "ct_atom_fwd zs", 128);
  const at::cuda::CUDAGuard g(xa.device());
  CtTailFwdParams p{CP_<bf>(xa), OP_<bf>(xin), CP_<bf>(cond), CP_<bf>(wab), CP_<bf>(wd), CP_<bf>(wsc), CP_<bf>(bsc), P_<bf>(y), zs ? P_<bf>(*zs) : nullptr, M, P};
  using G = CtTailFwdCfg<8>;
  launch_persistent<G>(ct_atom_fwd_kernel<G>, p, (M + 15) / 16, S_(), 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------------------------------------------- atom width (128), fp32 (TF32), inference
void check_f32_rows(const at::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dim() == 2 && t.size(1) == cols && t.scalar_type() == at::kFloat, name, ": a contiguous fp32 CUDA [rows, ", cols, "]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": 16-byte aligned");
}

// z [M, 128] = (silu(xa Wa^T) (xa Wb^T)) Ws^T on TF32 tensor cores (ct_atom_fwd_tf32.cuh); wab [512, 128] = TF32-rounded [Wa; Wb], wsp [128, 256] = the packed squeeze weight (sm80.py: pack_tf32)
void ct_tail_tf32(at::Tensor xa, at::Tensor wab, at::Tensor wsp, at::Tensor z) {
  const int64_t M = xa.size(0);
  check_f32_rows(xa, "ct_tail_tf32 xa", 128);
  check_f32_rows(wab, "ct_tail_tf32 wab", 128);
  check_f32_rows(z, "ct_tail_tf32 z", 128);
  TORCH_CHECK(wab.size(0) == 512 && wsp.is_cuda() && wsp.is_contiguous() && wsp.dim() == 2 && wsp.size(0) == 128 && wsp.size(1) == 256 && wsp.scalar_type() == at::kFloat &&
              reinterpret_cast<uintptr_t>(wsp.data_ptr()) % 16 == 0, "ct_tail_tf32: wab [512, 128], wsp [128, 256] fp32");
  TORCH_CHECK(z.size(0) == M, "ct_tail_tf32: z like xa");
  const at::cuda::CUDAGuard g(xa.device());
  CtTailTf32Params p{CP_<float>(xa), CP_<float>(wab), CP_<float>(wsp), P_<float>(z), M};
  using G = CtTailTf32Cfg<8>;
  launch_persistent<G>(ct_tail_tf32_kernel<G>, p, (M + 15) / 16, S_(), 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y [M, 128] = (x) + sigmoid(cond Wg^T + bg) z, cond row r % P (x null: the update alone)
void ct_gate_tf32(at::Tensor z, c10::optional<at::Tensor> xin, at::Tensor cond, at::Tensor wsc, at::Tensor bsc, at::Tensor y) {
  const int64_t M = z.size(0), P = cond.size(0);
  check_f32_rows(z, "ct_gate_tf32 z", 128);
  check_f32_rows(cond, "ct_gate_tf32 cond", 128);
  check_f32_rows(wsc, "ct_gate_tf32 wsc", 128);
  check_f32_rows(y, "ct_gate_tf32 y", 128);
  if (xin) check_f32_rows(*xin, "ct_gate_tf32 x", 128);
  TORCH_CHECK(bsc.is_cuda() && bsc.is_contiguous() && bsc.numel() == 128 && bsc.scalar_type() == at::kFloat && reinterpret_cast<uintptr_t>(bsc.data_ptr()) % 16 == 0, "ct_gate_tf32: bsc [128] fp32");
  TORCH_CHECK(y.size(0) == M && P > 0 && M % P == 0, "ct_gate_tf32: y like z, P divides M");
  const at::cuda::CUDAGuard g(z.device());
  CtGateTf32Params p{CP_<float>(z), OP_<float>(xin), CP_<float>(cond), CP_<float>(wsc), CP_<float>(bsc), P_<float>(y), M, P};
  using G = CtGateTf32Cfg<8, 1>;
  launch_persistent<G>(ct_gate_tf32_kernel<G>, p, (M + 15) / 16, S_());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the backward's gate kernel (ct_atom_bwd.cuh): dz, dg, dcond2 [M, 128], hh [M, 256], dab [M, 512] bf16 and pbsc [grid, 128] fp32 (per-CTA partial column sums of dg; grid = ct_bwd_grid(M))
int64_t ct_bwd_grid(int64_t M) {
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return std::max<int64_t>(1, std::min<int64_t>(((M + 15) / 16 + CtAtomBwdGateCfg<8>::NW - 1) / CtAtomBwdGateCfg<8>::NW, (int64_t)sms));
}
void ct_atom_bwd_gate(at::Tensor dy, at::Tensor z, at::Tensor cond, at::Tensor xa, at::Tensor wab, at::Tensor wdt, at::Tensor wsc, at::Tensor wsct, at::Tensor bsc, at::Tensor dz,
                      at::Tensor dg, at::Tensor dcond2, at::Tensor hh, at::Tensor dab, at::Tensor pbsc) {
  const int64_t M = xa.size(0);
  for (const auto* t : {&dy, &z, &cond, &xa, &wsc, &wsct, &dz, &dg, &dcond2}) check_bf16_rows(*t, "ct_atom_bwd_gate", 128);
  check_bf16_rows(wab, "ct_atom_bwd_gate wab", 128);
  check_bf16_rows(wdt, "ct_atom_bwd_gate wdt", 128);
  check_bf16_rows(hh, "ct_atom_bwd_gate hh", 256);
  check_bf16_rows(dab, "ct_atom_bwd_gate dab", 512);
  TORCH_CHECK(wab.size(0) == 512 && wdt.size(0) == 256 && dy.size(0) == M && z.size(0) == M && cond.size(0) == M && dz.size(0) == M && dg.size(0) == M && dcond2.size(0) == M && hh.size(0) == M && dab.size(0) == M,
              "ct_atom_bwd_gate: shapes");
  TORCH_CHECK(bsc.is_cuda() && bsc.is_contiguous() && bsc.numel() == 128 && bsc.scalar_type() == at::kBFloat16 && reinterpret_cast<uintptr_t>(bsc.data_ptr()) % 16 == 0, "ct_atom_bwd_gate: bsc [128] bf16");
  TORCH_CHECK(pbsc.is_cuda() && pbsc.is_contiguous() && pbsc.scalar_type() == at::kFloat && pbsc.dim() == 2 && pbsc.size(1) == 128 && pbsc.size(0) >= ct_bwd_grid(M), "ct_atom_bwd_gate: pbsc [grid, 128] fp32");
  const at::cuda::CUDAGuard g(xa.device());
  CtAtomBwdGateParams p{CP_<bf>(dy), CP_<bf>(z), CP_<bf>(cond), CP_<bf>(xa), CP_<bf>(wab), CP_<bf>(wdt), CP_<bf>(wsc), CP_<bf>(wsct), CP_<bf>(bsc), P_<bf>(dz), P_<bf>(dg), P_<bf>(dcond2),
                        P_<bf>(hh), P_<bf>(dab), P_<float>(pbsc), M};
  using G = CtAtomBwdGateCfg<8>;
  launch_persistent<G>(ct_atom_bwd_gate_kernel<G>, p, (M + 15) / 16, S_(), 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dxa [M, 128] = dab [M, 512] [Wa; Wb]; wabt [128, 512] = [Wa; Wb]^T
void ct_atom_bwd_dxa(at::Tensor dab, at::Tensor wabt, at::Tensor dxa) {
  const int64_t M = dab.size(0);
  check_bf16_rows(dab, "ct_atom_bwd_dxa dab", 512);
  check_bf16_rows(wabt, "ct_atom_bwd_dxa wabt", 512);
  check_bf16_rows(dxa, "ct_atom_bwd_dxa dxa", 128);
  TORCH_CHECK(wabt.size(0) == 128 && dxa.size(0) == M, "ct_atom_bwd_dxa: shapes");
  const at::cuda::CUDAGuard g(dab.device());
  CtAtomBwdDxaParams p{CP_<bf>(dab), CP_<bf>(wabt), P_<bf>(dxa), M};
  using G = CtAtomBwdDxaCfg<8>;
  launch_persistent<G>(ct_atom_bwd_dxa_kernel<G>, p, (M + 15) / 16, S_(), 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ct_atom_fwd", &ct_atom_fwd);
  m.def("ct_tail_tf32", &ct_tail_tf32);
  m.def("ct_gate_tf32", &ct_gate_tf32);
  m.def("ct_atom_bwd_gate", &ct_atom_bwd_gate);
  m.def("ct_atom_bwd_dxa", &ct_atom_bwd_dxa);
  m.def("ct_bwd_grid", &ct_bwd_grid);
  m.def("swiglu_fwd", &swiglu_fwd);
  m.def("swiglu_bwd", &swiglu_bwd);
  m.def("gate_res_fwd", &gate_res_fwd);
  m.def("gate_res_bwd", &gate_res_bwd);
}
