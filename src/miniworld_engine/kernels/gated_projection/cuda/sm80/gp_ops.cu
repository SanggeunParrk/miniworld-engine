// gp_ops.cu -- torch bindings of the A100 gated-projection kernels (gp_kernels.cuh), on the current CUDA stream.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <algorithm>

#include "gp_kernels.cuh"

namespace {

int num_sms() {
  static int n = 0;
  if (n == 0) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, at::cuda::current_device());
  return n;
}
inline const __nv_bfloat16* cbf(const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); }
inline __nv_bfloat16* mbf(torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); }
inline cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
template <class K>
void set_smem(K kernel, int bytes) { C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)); }
inline void check_bf16(const torch::Tensor& t, const char* what, int64_t rows, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.size(0) == rows && t.size(1) == cols && t.stride(1) == 1 && t.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, what, ": bf16 [", rows, ", ", cols, "], unit column stride, 16-byte aligned rows");
}

// BM = 64 when 128-row tiles would leave SMs idle (a small M): twice the CTAs for the same rows
template <int BM, int BN>
void launch_gp_fwd(const a100::GpFwdParams& p) {
  static bool set = false;
  using C = a100::GpFwdCfg<BM, BN>;
  if (!set) { set_smem(a100::gp_fwd_kernel<BM, BN>, C::Tile::SMEM); set = true; }
  const int grid = ((p.M + BM - 1) / BM) * (p.N / BN);
  a100::gp_fwd_kernel<BM, BN><<<grid, 128, C::Tile::SMEM, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BM, int BN>
void launch_gp_dgrad(const a100::GpDgradParams& p) {
  static bool set = false;
  using C = a100::GpDgradCfg<BM, BN>;
  if (!set) { set_smem(a100::gp_dgrad_kernel<BM, BN>, C::SMEM); set = true; }
  const int grid = ((p.M + BM - 1) / BM) * (p.K / BN);
  a100::gp_dgrad_kernel<BM, BN><<<grid, 128, C::SMEM, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// BM = 64 when 128-row tiles leave the SMs worse filled: the tiles run two CTAs per SM, so a grid of t tiles takes ceil(t / 2 SMs) waves; the 64-row tiling (twice the tiles, for
// the same rows) wins whenever its last wave is fuller by a margin (a small M, or a grid just past a multiple of the wave size)
static inline bool small_m(int M, int ncols, int bn) {
  const int64_t slots = 2 * (int64_t)num_sms();
  auto eff = [&](int bm) { const int64_t t = (int64_t)((M + bm - 1) / bm) * (ncols / bn); return (double)t / (double)(((t + slots - 1) / slots) * slots); };
  return eff(64) > eff(128) + 0.03;
}

template <int BN>
void launch_tm2(const a100::Tm2Params& p) {
  static bool set = false;
  using T = a100::WTile<128, BN, 4, 2, 4, false, false>;
  if (!set) { set_smem(a100::tm2_kernel<BN>, T::SMEM); set = true; }
  const int grid = ((p.M + 127) / 128) * (p.D / BN);
  a100::tm2_kernel<BN><<<grid, 256, T::SMEM, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// out [M, N] = (sigmoid(g) v) [M, K] . w^T,  w [N, K]
void gp_fwd(torch::Tensor v, torch::Tensor g, torch::Tensor w, torch::Tensor out) {
  const int M = (int)v.size(0), K = (int)v.size(1), N = (int)w.size(0);
  check_bf16(v, "gp_fwd v", M, K); check_bf16(g, "gp_fwd g", M, K); check_bf16(w, "gp_fwd w", N, K); check_bf16(out, "gp_fwd out", M, N);
  TORCH_CHECK(K % 32 == 0 && N % 64 == 0 && M > 0, "gp_fwd: K % 32, N % 64");
  a100::GpFwdParams p;
  p.v = cbf(v); p.ldv = v.stride(0); p.g = cbf(g); p.ldg = g.stride(0); p.w = cbf(w); p.out = mbf(out); p.ldo = out.stride(0);
  p.M = M; p.N = N; p.K = K;
  const int bn = N % 128 == 0 ? 128 : 64;
  if (bn == 128) { if (small_m(M, N, 128)) launch_gp_fwd<64, 128>(p); else launch_gp_fwd<128, 128>(p); }
  else { if (small_m(M, N, 64)) launch_gp_fwd<64, 64>(p); else launch_gp_fwd<128, 64>(p); }
}

// dv, dg, a [M, K] from dO [M, N], wt [K, N] (= W^T), g, v [M, K]
void gp_dgrad(torch::Tensor dO, torch::Tensor wt, torch::Tensor g, torch::Tensor v, torch::Tensor dv, torch::Tensor dg, torch::Tensor a) {
  const int M = (int)dO.size(0), N = (int)dO.size(1), K = (int)wt.size(0);
  check_bf16(dO, "gp_dgrad dO", M, N); check_bf16(wt, "gp_dgrad wt", K, N); check_bf16(g, "gp_dgrad g", M, K); check_bf16(v, "gp_dgrad v", M, K);
  for (auto* t : {&dv, &dg, &a}) { check_bf16(*t, "gp_dgrad out", M, K); TORCH_CHECK(t->stride(0) == K, "gp_dgrad: outputs row-major"); }
  TORCH_CHECK(N % 32 == 0 && K % 64 == 0 && M > 0, "gp_dgrad: N % 32, K % 64");
  a100::GpDgradParams p;
  p.dO = cbf(dO); p.ldo = dO.stride(0); p.wt = cbf(wt); p.g = cbf(g); p.ldg = g.stride(0); p.v = cbf(v); p.ldv = v.stride(0);
  p.dv = mbf(dv); p.dg = mbf(dg); p.a = mbf(a); p.M = M; p.N = N; p.K = K;
  const int bn = K % 128 == 0 ? 128 : 64;
  if (bn == 128) { if (small_m(M, K, 128)) launch_gp_dgrad<64, 128>(p); else launch_gp_dgrad<128, 128>(p); }
  else { if (small_m(M, K, 64)) launch_gp_dgrad<64, 64>(p); else launch_gp_dgrad<128, 64>(p); }
}

// tm2: out = sigmoid(x Wg) (y Wo); weights as [out][in] (W^T of the matmul form)
void tm2_fwd(torch::Tensor x, torch::Tensor y, torch::Tensor wgt, torch::Tensor wot, torch::Tensor out) {
  const int M = (int)x.size(0), D = (int)x.size(1);
  check_bf16(x, "tm2 x", M, D); check_bf16(y, "tm2 y", M, D); check_bf16(wgt, "tm2 wg", D, D); check_bf16(wot, "tm2 wo", D, D); check_bf16(out, "tm2 out", M, D);
  TORCH_CHECK(D % 64 == 0 && M > 0 && out.stride(0) == D, "tm2: D % 64, row-major output");
  a100::Tm2Params p;
  p.x = cbf(x); p.y = cbf(y); p.wgt = cbf(wgt); p.wot = cbf(wot); p.gout = nullptr; p.out = mbf(out); p.da = nullptr; p.db = nullptr; p.M = M; p.D = D;
  TORCH_CHECK(x.stride(0) == D && y.stride(0) == D, "tm2: contiguous inputs");
  if (D % 128 == 0) launch_tm2<128>(p); else launch_tm2<64>(p);
}
void tm2_bwd(torch::Tensor x, torch::Tensor y, torch::Tensor wgt, torch::Tensor wot, torch::Tensor gout, torch::Tensor da, torch::Tensor db) {
  const int M = (int)x.size(0), D = (int)x.size(1);
  check_bf16(x, "tm2 x", M, D); check_bf16(y, "tm2 y", M, D); check_bf16(wgt, "tm2 wg", D, D); check_bf16(wot, "tm2 wo", D, D); check_bf16(gout, "tm2 gout", M, D);
  check_bf16(da, "tm2 da", M, D); check_bf16(db, "tm2 db", M, D);
  TORCH_CHECK(D % 64 == 0 && M > 0 && x.stride(0) == D && y.stride(0) == D && gout.stride(0) == D && da.stride(0) == D && db.stride(0) == D, "tm2: D % 64, contiguous");
  a100::Tm2Params p;
  p.x = cbf(x); p.y = cbf(y); p.wgt = cbf(wgt); p.wot = cbf(wot); p.gout = cbf(gout); p.out = nullptr; p.da = mbf(da); p.db = mbf(db); p.M = M; p.D = D;
  if (D % 128 == 0) launch_tm2<128>(p); else launch_tm2<64>(p);
}

// tm1: w1 [4 D, D] packed rows (gate | projection of 8 channels per 16 rows; left channels first, then right)
void tm1_fwd(torch::Tensor x, torch::Tensor w1, torch::Tensor left, torch::Tensor right) {
  const int M = (int)x.size(0), D = (int)x.size(1);
  check_bf16(x, "tm1 x", M, D); check_bf16(w1, "tm1 w1", 4 * D, D); check_bf16(left, "tm1 left", M, D); check_bf16(right, "tm1 right", M, D);
  TORCH_CHECK(D % 64 == 0 && M > 0 && x.stride(0) == D && left.stride(0) == D && right.stride(0) == D, "tm1: D % 64, contiguous");
  a100::Tm1Params p;
  p.x = cbf(x); p.w1 = cbf(w1); p.left = mbf(left); p.right = mbf(right); p.gl = nullptr; p.gr = nullptr; p.dla = p.dlb = p.dra = p.drb = nullptr; p.M = M; p.D = D;
  static bool set = false;
  using T = a100::WTile<128, 128, 2, 2, 4, false, false>;
  if (!set) { set_smem(a100::tm1_kernel, T::SMEM); set = true; }
  a100::tm1_kernel<<<((M + 127) / 128) * (4 * D / 128), 128, T::SMEM, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tm1_bwd(torch::Tensor x, torch::Tensor w1, torch::Tensor gl, torch::Tensor gr, torch::Tensor dla, torch::Tensor dlb, torch::Tensor dra, torch::Tensor drb) {
  const int M = (int)x.size(0), D = (int)x.size(1);
  check_bf16(x, "tm1 x", M, D); check_bf16(w1, "tm1 w1", 4 * D, D); check_bf16(gl, "tm1 gl", M, D); check_bf16(gr, "tm1 gr", M, D);
  for (auto* t : {&dla, &dlb, &dra, &drb}) { check_bf16(*t, "tm1 out", M, D); TORCH_CHECK(t->stride(0) == D, "tm1: row-major outputs"); }
  TORCH_CHECK(D % 64 == 0 && M > 0 && x.stride(0) == D && gl.stride(0) == D && gr.stride(0) == D, "tm1: D % 64, contiguous");
  a100::Tm1Params p;
  p.x = cbf(x); p.w1 = cbf(w1); p.left = nullptr; p.right = nullptr; p.gl = cbf(gl); p.gr = cbf(gr);
  p.dla = mbf(dla); p.dlb = mbf(dlb); p.dra = mbf(dra); p.drb = mbf(drb); p.M = M; p.D = D;
  static bool set = false;
  using T = a100::WTile<128, 128, 2, 2, 4, false, false>;
  if (!set) { set_smem(a100::tm1_kernel, T::SMEM); set = true; }
  a100::tm1_kernel<<<((M + 127) / 128) * (4 * D / 128), 128, T::SMEM, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// elementwise (contiguous tensors of one bf16 shape)
static inline int ew_grid(int64_t n) { return (int)std::max<int64_t>(1, std::min<int64_t>((n / 8 + 255) / 256, 8 * (int64_t)num_sms())); }
#define EW_CHECK(...) for (auto* t : {__VA_ARGS__}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->is_contiguous() && t->numel() == n && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "elementwise gate: contiguous bf16 tensors of one size, 16-byte aligned")
void sigmul_fwd(torch::Tensor g, torch::Tensor o, torch::Tensor a) {
  const int64_t n = g.numel(); EW_CHECK(&g, &o, &a);
  if (n) { a100::sigmul_fwd_kernel<<<ew_grid(n), 256, 0, stream()>>>(cbf(g), cbf(o), mbf(a), (size_t)n); C10_CUDA_KERNEL_LAUNCH_CHECK(); }
}
void sigmul_bwd(torch::Tensor da, torch::Tensor g, torch::Tensor o, torch::Tensor dg, torch::Tensor dd) {
  const int64_t n = g.numel(); EW_CHECK(&da, &g, &o, &dg, &dd);
  if (n) { a100::sigmul_bwd_kernel<<<ew_grid(n), 256, 0, stream()>>>(cbf(da), cbf(g), cbf(o), mbf(dg), mbf(dd), (size_t)n); C10_CUDA_KERNEL_LAUNCH_CHECK(); }
}
void gres_fwd(torch::Tensor x, torch::Tensor g, torch::Tensor b, torch::Tensor y) {
  const int64_t n = x.numel(); EW_CHECK(&x, &g, &b, &y);
  if (n) { a100::gres_fwd_kernel<<<ew_grid(n), 256, 0, stream()>>>(cbf(x), cbf(g), cbf(b), mbf(y), (size_t)n); C10_CUDA_KERNEL_LAUNCH_CHECK(); }
}
void gres_bwd(torch::Tensor dy, torch::Tensor g, torch::Tensor b, torch::Tensor dg, torch::Tensor db) {
  const int64_t n = dy.numel(); EW_CHECK(&dy, &g, &b, &dg, &db);
  if (n) { a100::gres_bwd_kernel<<<ew_grid(n), 256, 0, stream()>>>(cbf(dy), cbf(g), cbf(b), mbf(dg), mbf(db), (size_t)n); C10_CUDA_KERNEL_LAUNCH_CHECK(); }
}

// the TriMul output gate: glogit / proj / res / y / gate [M, N] contiguous bf16 (N % 8 == 0), ds [L, N]
void gate_elem_fwd(torch::Tensor glogit, torch::Tensor proj, torch::Tensor res, torch::Tensor ds, torch::Tensor y, torch::Tensor gate) {
  const int64_t M = glogit.size(0), N = glogit.size(1);
  for (auto* t : {&glogit, &proj, &res, &y, &gate}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->is_contiguous() && t->sizes() == glogit.sizes() &&
                                                                 reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "gate_elem_fwd: contiguous bf16 [M, N]");
  TORCH_CHECK(N % 8 == 0 && ds.is_contiguous() && ds.scalar_type() == at::kBFloat16 && ds.dim() == 2 && ds.size(1) == N, "gate_elem_fwd: N % 8, ds [L, N] bf16");
  if (M) { a100::gate_elem_fwd_kernel<<<ew_grid(M * N), 256, 0, stream()>>>(cbf(glogit), cbf(proj), cbf(res), cbf(ds), mbf(y), mbf(gate), (size_t)M, (int)N, (int)ds.size(0)); C10_CUDA_KERNEL_LAUNCH_CHECK(); }
}
void gate_elem_bwd(torch::Tensor dy, torch::Tensor proj, torch::Tensor gate, torch::Tensor ds, torch::Tensor dproj, torch::Tensor dglogit, bool from_preact) {
  const int64_t M = dy.size(0), N = dy.size(1);
  for (auto* t : {&dy, &proj, &gate, &dproj, &dglogit}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->is_contiguous() && t->sizes() == dy.sizes() &&
                                                                     reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "gate_elem_bwd: contiguous bf16 [M, N]");
  TORCH_CHECK(N % 8 == 0 && ds.is_contiguous() && ds.scalar_type() == at::kBFloat16 && ds.dim() == 2 && ds.size(1) == N, "gate_elem_bwd: N % 8, ds [L, N] bf16");
  if (M) { a100::gate_elem_bwd_kernel<<<ew_grid(M * N), 256, 0, stream()>>>(cbf(dy), cbf(proj), cbf(gate), cbf(ds), mbf(dproj), mbf(dglogit), (size_t)M, (int)N, (int)ds.size(0), from_preact ? 1 : 0);
           C10_CUDA_KERNEL_LAUNCH_CHECK(); }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gp_fwd", &gp_fwd);
  m.def("gp_dgrad", &gp_dgrad);
  m.def("tm2_fwd", &tm2_fwd);
  m.def("tm2_bwd", &tm2_bwd);
  m.def("tm1_fwd", &tm1_fwd);
  m.def("tm1_bwd", &tm1_bwd);
  m.def("sigmul_fwd", &sigmul_fwd);
  m.def("sigmul_bwd", &sigmul_bwd);
  m.def("gres_fwd", &gres_fwd);
  m.def("gres_bwd", &gres_bwd);
  m.def("gate_elem_fwd", &gate_elem_fwd);
  m.def("gate_elem_bwd", &gate_elem_bwd);
}
