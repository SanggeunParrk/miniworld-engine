// wide_ops.cu -- torch bindings of the A100 "wide" TriMul kernels (any width D = 64 / 128 / 256 / 384, either direction), on the current CUDA stream.
// bf16 tensors run the bf16 kernels, fp32 tensors the TF32 ones (wide_gemm32.cuh / wide_k32.cuh); the row kernels are templates over the element type.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <climits>

#include "wide_k1.cuh"
#include "wide_k3.cuh"
#include "wide_k32.cuh"
#include "wide_rows.cuh"
#include "wide_bwd.cuh"
#include "wide_k1b.cuh"
#include "wide_fin.cuh"

namespace {

int num_sms() {
  static int n = 0;
  if (n == 0) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, at::cuda::current_device());
  return n;
}
inline cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
template <class E> inline const typename E::T* cp(const torch::Tensor& t) { return reinterpret_cast<const typename E::T*>(t.data_ptr()); }
template <class E> inline typename E::T* mp(torch::Tensor& t) { return reinterpret_cast<typename E::T*>(t.data_ptr()); }
inline const __nv_bfloat16* cbf(const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); }
inline __nv_bfloat16* mbf(torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); }
inline bool is_f32(const torch::Tensor& t) {
  TORCH_CHECK(t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat, "wide TriMul: bf16 or fp32 tensors");
  return t.scalar_type() == at::kFloat;
}
inline const float2* f2(const torch::Tensor& t) { return reinterpret_cast<const float2*>(t.data_ptr<float>()); }
inline float2* mf2(torch::Tensor& t) { return reinterpret_cast<float2*>(t.data_ptr<float>()); }

template <class K>
void set_smem(K kernel, int bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

template <class E> int rows_per_pass(int D) { const int g = D / E::VEC, lpr = g < 32 ? g : 32; return 8 * (32 / lpr); }

}  // namespace

// ------------------------------------------------------------------------------------------------------------------------------------ statistics
// z [T, D] -> st [T, 2] fp32 (mean, rstd)
template <class E>
void stats_rows_t(torch::Tensor z, torch::Tensor st, double eps) {
  const int T = (int)z.size(0), D = (int)z.size(1);
  const int rpp = rows_per_pass<E>(D);
  const int grid = std::max(1, std::min((T + rpp - 1) / rpp, num_sms() * 8));
  float2* o = mf2(st);
#define RS(DD) if (D == DD) { a100::ln_stats_rows_kernel<DD, E><<<grid, 256, 0, stream()>>>(cp<E>(z), o, T, (float)eps); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  RS(64) RS(128) RS(256) RS(384) RS(512) RS(768)
#undef RS
  TORCH_CHECK(false, "stats_rows: unsupported width ", D);
}
void stats_rows(torch::Tensor z, torch::Tensor st, double eps) {
  TORCH_CHECK(z.is_contiguous() && z.dim() == 2 && st.is_contiguous() && st.numel() == 2 * z.size(0), "stats_rows: z [T, D], st [T, 2]");
  if (is_f32(z)) stats_rows_t<a100::F32>(z, st, eps); else stats_rows_t<a100::Bf16>(z, st, eps);
}

// x [H, T] channel-major -> st [T, 2] fp32 (mean, rstd) over the H channels
template <class E>
void stats_cm_t(torch::Tensor x, torch::Tensor st, double eps) {
  const int H = (int)x.size(0), M = (int)x.size(1);
  constexpr int V = E::VEC;                                       // 16-byte vectors: 8 bf16 / 4 fp32 tokens per lane
  constexpr int TPL4 = 4 / E::BYTES;                              // the 4-byte variant (2 bf16 / 1 fp32 tokens per lane), for short sequences
  const bool wide = M % (32 * V) == 0 && M / (32 * V) >= 2 * num_sms();
  float2* o = mf2(st);
#define CM(HH) if (H == HH) { if (wide) a100::ln_stats_cm_kernel<HH, V, E><<<M / (32 * V), 256, 0, stream()>>>(cp<E>(x), o, M, (float)eps); \
                              else a100::ln_stats_cm_kernel<HH, TPL4, E><<<M / (32 * TPL4), 256, 0, stream()>>>(cp<E>(x), o, M, (float)eps); \
                              C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  CM(64) CM(128) CM(256) CM(384) CM(512) CM(768)
#undef CM
  TORCH_CHECK(false, "stats_cm: unsupported hidden width ", H);
}
void stats_cm(torch::Tensor x, torch::Tensor st, double eps) {
  TORCH_CHECK(x.is_contiguous() && x.dim() == 2 && st.is_contiguous() && st.numel() == 2 * x.size(1), "stats_cm: x [H, T], st [T, 2]");
  TORCH_CHECK(x.size(1) % 64 == 0, "stats_cm: T % 64");
  if (is_f32(x)) stats_cm_t<a100::F32>(x, st, eps); else stats_cm_t<a100::Bf16>(x, st, eps);
}

// -------------------------------------------------------------------------------------------------------------------------------------- weight packs
// wl, wlg, wr, wrg [Hs, D] (any strides), wg [D, D], wo [D, Hc]; gi, bi [D], go, bo [Hc] fp32; wdx / spx: the training outputs (empty = inference)
template <class E>
void wpack_t(torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, torch::Tensor wg, torch::Tensor wo, torch::Tensor gi,
             torch::Tensor bi, torch::Tensor go, torch::Tensor bo, torch::Tensor w1, torch::Tensor vs, torch::Tensor vb, torch::Tensor wo3,
             torch::Tensor so, torch::Tensor eo, torch::Tensor wg3, torch::Tensor sg, torch::Tensor eg, torch::Tensor wdx, torch::Tensor spx) {
  const int Hs = (int)wl.size(0), D = (int)wl.size(1), Hc = (int)wo.size(1);
  a100::WPackParams<E> p;
  p.wl = cp<E>(wl); p.wlg = cp<E>(wlg); p.wr = cp<E>(wr); p.wrg = cp<E>(wrg); p.wg = cp<E>(wg); p.wo = cp<E>(wo);
  int i = 0;
  for (auto* t : {&wl, &wlg, &wr, &wrg}) {
    TORCH_CHECK(t->stride(0) > 0 && t->stride(1) > 0 && t->stride(0) * (int64_t)(Hs - 1) + t->stride(1) * (int64_t)(D - 1) < INT_MAX, "wpack: front weight strides");
    p.rs[i] = (int)t->stride(0); p.ks[i] = (int)t->stride(1); ++i;
  }
  p.gi = gi.data_ptr<float>(); p.bi = bi.data_ptr<float>(); p.go = go.data_ptr<float>(); p.bo = bo.data_ptr<float>();
  p.w1 = mp<E>(w1); p.wo3 = mp<E>(wo3); p.wg3 = mp<E>(wg3);
  p.vs = vs.data_ptr<float>(); p.vb = vb.data_ptr<float>();
  p.so = so.data_ptr<float>(); p.eo = eo.data_ptr<float>(); p.sg = sg.data_ptr<float>(); p.eg = eg.data_ptr<float>();
  p.wdx = wdx.numel() ? mp<E>(wdx) : nullptr; p.spx = spx.numel() ? spx.data_ptr<float>() : nullptr;
  TORCH_CHECK(!wdx.numel() || wdx.numel() == (int64_t)(4 * Hs + D) * D, "wpack: wdx");
  p.Hs = Hs; p.D = D; p.Hc = Hc;
  TORCH_CHECK(w1.numel() == (int64_t)4 * Hs * D && wo3.numel() == (int64_t)D * Hc && wg3.numel() == (int64_t)D * D && vs.numel() == 4 * Hs && vb.numel() == 4 * Hs, "wpack: outputs");
  const int items = 4 * Hs + 2 * D;
  a100::wpack_kernel<E><<<(items + 7) / 8, 256, 0, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void wpack(torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, torch::Tensor wg, torch::Tensor wo, torch::Tensor gi,
           torch::Tensor bi, torch::Tensor go, torch::Tensor bo, torch::Tensor w1, torch::Tensor vs, torch::Tensor vb, torch::Tensor wo3,
           torch::Tensor so, torch::Tensor eo, torch::Tensor wg3, torch::Tensor sg, torch::Tensor eg, torch::Tensor wdx, torch::Tensor spx) {
  const int Hs = (int)wl.size(0), D = (int)wl.size(1), Hc = (int)wo.size(1);
  TORCH_CHECK(Hs % 8 == 0 && Hc == Hs && wo.size(0) == D && wg.size(0) == D && wg.size(1) == D, "wpack: shapes");
  const auto dt = wl.scalar_type();
  for (auto* t : {&wl, &wlg, &wr, &wrg}) TORCH_CHECK(t->dim() == 2 && t->sizes() == wl.sizes() && t->scalar_type() == dt, "wpack: front weights [Hs, D] of one dtype");
  for (auto* t : {&wg, &wo}) TORCH_CHECK(t->is_contiguous() && t->scalar_type() == dt, "wpack: weights of the same dtype");
  for (auto* t : {&gi, &bi, &go, &bo}) TORCH_CHECK(t->is_contiguous() && t->scalar_type() == at::kFloat, "wpack: fp32 LayerNorm affine");
  if (is_f32(wl)) wpack_t<a100::F32>(wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, w1, vs, vb, wo3, so, eo, wg3, sg, eg, wdx, spx);
  else wpack_t<a100::Bf16>(wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, w1, vs, vb, wo3, so, eo, wg3, sg, eg, wdx, spx);
}

// --------------------------------------------------------------------------------------------------------------------------------------------- k1w
// z [T, D], w1 [4 Hs, D], vs / vb [4 Hs], st [T, 2], mask [L] uint8 or empty -> planes [2 Hs, T].  fuse (bf16, D <= 128): st is an OUTPUT, the kernel computes (mean, rstd) of the rows
// itself (eps); otherwise st holds ln_stats_rows' result
void k1w(torch::Tensor z, torch::Tensor w1, torch::Tensor vs, torch::Tensor vb, torch::Tensor st, torch::Tensor mask, torch::Tensor planes, int64_t L, double eps, bool fuse) {
  const int T = (int)z.size(0), D = (int)z.size(1), ncol = (int)w1.size(0);
  TORCH_CHECK(T % 128 == 0 && D % 32 == 0 && ncol % 128 == 0 && planes.size(0) == ncol / 2 && planes.size(1) == T && z.scalar_type() == w1.scalar_type() &&
              planes.scalar_type() == z.scalar_type(), "k1w: shapes");
  TORCH_CHECK(!fuse || (!is_f32(z) && D <= 128 && st.numel() == 2 * T), "k1w: the in-kernel statistics are bf16 with D <= 128");
  if (is_f32(z)) {
    a100::K1w32Params p;
    p.z = z.data_ptr<float>(); p.w1 = w1.data_ptr<float>(); p.vs = vs.data_ptr<float>(); p.vb = vb.data_ptr<float>(); p.st = f2(st);
    p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr; p.planes = planes.data_ptr<float>();
    p.T = T; p.D = D; p.L = (int)L; p.ncol = ncol;
    static bool set = false;
    if (!set) { set_smem(a100::k1w32_kernel, a100::K1w32Tile::SMEM); set = true; }
    a100::k1w32_kernel<<<(T / 128) * (ncol / 128), 128, a100::K1w32Tile::SMEM, stream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  a100::K1wParams p;
  p.z = cbf(z); p.w1 = cbf(w1); p.vs = vs.data_ptr<float>(); p.vb = vb.data_ptr<float>(); p.st = fuse ? nullptr : f2(st); p.stw = fuse ? mf2(st) : nullptr; p.eps = (float)eps;
  p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
  p.planes = mbf(planes);
  p.T = T; p.D = D; p.L = (int)L; p.ncol = ncol;
  static bool set = false;
  static bool set1 = false;
  if (!set1) { set_smem(a100::k1w_kernel<false>, a100::K1wTile::SMEM); set_smem(a100::k1w_kernel<true>, a100::K1W_STATS_SMEM); set1 = true; }
  if (fuse) a100::k1w_kernel<true><<<(T / 128) * (ncol / 128), 128, a100::K1W_STATS_SMEM, stream()>>>(p);
  else a100::k1w_kernel<false><<<(T / 128) * (ncol / 128), 128, a100::K1wTile::SMEM, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// --------------------------------------------------------------------------------------------------------------------------------------------- k3w
template <int BN>
void launch_k3w(const a100::K3wParams& p, bool fuse) {
  static bool set = false;
  using Cfg = a100::K3wCfg<BN>;
  if (!set) { set_smem(a100::k3w_kernel<BN, false>, Cfg::SMEM); set_smem(a100::k3w_kernel<BN, true>, Cfg::SMEM_X); set = true; }
  if (fuse) a100::k3w_kernel<BN, true><<<(p.T / 128) * (p.D / BN), Cfg::NTHR, Cfg::SMEM_X, stream()>>>(p);
  else a100::k3w_kernel<BN, false><<<(p.T / 128) * (p.D / BN), Cfg::NTHR, Cfg::SMEM, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int BN>
void launch_k3w32(const a100::K3w32Params& p) {
  static bool set = false;
  using Cfg = a100::K3w32Cfg<BN>;
  if (!set) { set_smem(a100::k3w32_kernel<BN>, Cfg::SMEM); set = true; }
  a100::k3w32_kernel<BN><<<(p.T / 128) * (p.D / BN), Cfg::NTHR, Cfg::SMEM, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x [Hc, T], z [T, D], wo3 [D, Hc], wg3 [D, D], fold vectors [D], sto / sti [T, 2], ds [L, D] or empty -> out [T, D]; ps / gs [T, D] (training saves) or empty.  fuse (bf16, Hc <= 128 and
// D <= 128: one CTA per token tile): sto is an OUTPUT, the kernel computes LayerNorm_out's (mean, rstd) of X itself (eps)
void k3w(torch::Tensor x, torch::Tensor z, torch::Tensor wo3, torch::Tensor wg3, torch::Tensor so, torch::Tensor eo, torch::Tensor sg, torch::Tensor eg,
         torch::Tensor sto, torch::Tensor sti, torch::Tensor ds, torch::Tensor out, torch::Tensor ps, torch::Tensor gs, int64_t L, double eps, bool fuse) {
  const int T = (int)z.size(0), D = (int)z.size(1), Hc = (int)x.size(0);
  TORCH_CHECK(T % 128 == 0 && D % 64 == 0 && Hc % 32 == 0 && x.size(1) == T && wo3.size(0) == D && wo3.size(1) == Hc, "k3w: shapes");
  TORCH_CHECK((ps.numel() == 0) == (gs.numel() == 0), "k3w: ps and gs together");
  TORCH_CHECK(!fuse || (!is_f32(z) && Hc <= 128 && D <= 128 && sto.numel() == 2 * T), "k3w: the in-kernel statistics are bf16 with Hc, D <= 128");
  if (is_f32(z)) {
    a100::K3w32Params p;
    p.x = x.data_ptr<float>(); p.z = z.data_ptr<float>(); p.wo = wo3.data_ptr<float>(); p.wg = wg3.data_ptr<float>();
    p.so = so.data_ptr<float>(); p.eo = eo.data_ptr<float>(); p.sg = sg.data_ptr<float>(); p.eg = eg.data_ptr<float>(); p.sto = f2(sto); p.sti = f2(sti);
    p.ds = ds.numel() ? ds.data_ptr<float>() : nullptr; p.out = out.data_ptr<float>();
    p.ps = ps.numel() ? ps.data_ptr<float>() : nullptr; p.gs = gs.numel() ? gs.data_ptr<float>() : nullptr;
    p.T = T; p.D = D; p.Hc = Hc; p.L = (int)L;
    if (D % 128 == 0) launch_k3w32<128>(p); else launch_k3w32<64>(p);
    return;
  }
  a100::K3wParams p;
  p.x = cbf(x); p.z = cbf(z); p.wo = cbf(wo3); p.wg = cbf(wg3);
  p.so = so.data_ptr<float>(); p.eo = eo.data_ptr<float>(); p.sg = sg.data_ptr<float>(); p.eg = eg.data_ptr<float>(); p.sto = fuse ? nullptr : f2(sto); p.stw = fuse ? mf2(sto) : nullptr; p.eps = (float)eps; p.sti = f2(sti);
  p.ds = ds.numel() ? cbf(ds) : nullptr;
  p.out = mbf(out);
  p.ps = ps.numel() ? mbf(ps) : nullptr; p.gs = gs.numel() ? mbf(gs) : nullptr;
  p.T = T; p.D = D; p.Hc = Hc; p.L = (int)L;
  if (D % 128 == 0) launch_k3w<128>(p, fuse); else launch_k3w<64>(p, fuse);
}

// ------------------------------------------------------------------------------------------------------------------------------------ backward bindings
// grid of the blocks that write one row of per-block partials each (gate_bwd, lnin_bwd): a fixed function of the card, so replays are bit-identical
int64_t gate_bwd_blocks(int64_t T, int64_t D, bool f32) {
  const int rpp = f32 ? rows_per_pass<a100::F32>((int)D) : rows_per_pass<a100::Bf16>((int)D);
  return std::max<int64_t>(1, std::min<int64_t>((T + rpp - 1) / rpp, 4 * num_sms()));
}
int64_t lnin_bwd_blocks(int64_t T, int64_t D, bool f32) {
  const int rpp = 2 * (f32 ? rows_per_pass<a100::F32>((int)D) : rows_per_pass<a100::Bf16>((int)D));
  return std::max<int64_t>(1, std::min<int64_t>((T + rpp - 1) / rpp, 4 * num_sms()));
}

template <class E>
void gate_bwd_t(torch::Tensor dy, torch::Tensor ps, torch::Tensor gs, torch::Tensor ds, torch::Tensor spx, torch::Tensor epx, torch::Tensor sto, torch::Tensor dpr,
                torch::Tensor dcat, int64_t col0, torch::Tensor s12, torch::Tensor part, int64_t L) {
  const int T = (int)dy.size(0), D = (int)dy.size(1);
  const int grid = (int)gate_bwd_blocks(T, D, E::IS_F32);
  const typename E::T* dsp = ds.numel() ? cp<E>(ds) : nullptr;
  typename E::T* dg = mp<E>(dcat) + col0;
#define GB(DD) if (D == DD) { a100::gate_bwd_kernel<DD, E><<<grid, 256, 0, stream()>>>(cp<E>(dy), cp<E>(ps), cp<E>(gs), dsp, spx.data_ptr<float>(), epx.data_ptr<float>(), \
    f2(sto), mp<E>(dpr), dg, (int)dcat.stride(0), mf2(s12), part.data_ptr<float>(), T, (int)L); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  GB(64) GB(128) GB(256) GB(384)
#undef GB
  TORCH_CHECK(false, "gate_bwd: unsupported width ", D);
}
// dy, ps, gs [T, D]; ds [L, D] or empty; spx, epx [D] fp32; sto [T, 2] -> dpr [T, D]; dg into dcat[:, col0 : col0 + D] (row stride dcat.stride(0)); s12 [T, 2]; part [blocks, 2 D]
void gate_bwd(torch::Tensor dy, torch::Tensor ps, torch::Tensor gs, torch::Tensor ds, torch::Tensor spx, torch::Tensor epx, torch::Tensor sto, torch::Tensor dpr,
              torch::Tensor dcat, int64_t col0, torch::Tensor s12, torch::Tensor part, int64_t L) {
  const int T = (int)dy.size(0), D = (int)dy.size(1);
  const bool f = is_f32(dy);
  const int vec = f ? 4 : 8;
  TORCH_CHECK(dy.is_contiguous() && ps.is_contiguous() && gs.is_contiguous() && dpr.is_contiguous() && dcat.stride(1) == 1 && dcat.stride(0) % vec == 0 && col0 % vec == 0, "gate_bwd: layouts");
  TORCH_CHECK(part.numel() == gate_bwd_blocks(T, D, f) * 2 * D && s12.numel() == 2 * (int64_t)T, "gate_bwd: part / s12 sizes");
  if (f) gate_bwd_t<a100::F32>(dy, ps, gs, ds, spx, epx, sto, dpr, dcat, col0, s12, part, L);
  else gate_bwd_t<a100::Bf16>(dy, ps, gs, ds, spx, epx, sto, dpr, dcat, col0, s12, part, L);
}

// out[c] = sum_r part[r][c]
void colsum(torch::Tensor part, torch::Tensor out) {
  const int R = (int)part.size(0), C = (int)part.size(1);
  TORCH_CHECK(part.is_contiguous() && out.numel() == C && part.scalar_type() == at::kFloat, "colsum: shapes");
  a100::colsum_kernel<<<(C + 31) / 32, 256, 0, stream()>>>(part.data_ptr<float>(), out.data_ptr<float>(), R, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class E>
void lnout_bwd_t(torch::Tensor dor, torch::Tensor x, torch::Tensor sto, torch::Tensor s12, torch::Tensor go, torch::Tensor dt, torch::Tensor part) {
  const int H = (int)x.size(0), M = (int)x.size(1);
  a100::lnout_bwd_kernel<E><<<dim3(M / (32 * E::VEC), H / a100::LOB_CH), 256, 0, stream()>>>(cp<E>(dor), cp<E>(x), f2(sto), f2(s12), go.data_ptr<float>(), mp<E>(dt),
                                                                                              part.data_ptr<float>(), M, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// dor, x, dt [H, T] channel-major; sto, s12 [T, 2]; go [H] -> dt, part [T / (32 VEC), 2 H]
void lnout_bwd(torch::Tensor dor, torch::Tensor x, torch::Tensor sto, torch::Tensor s12, torch::Tensor go, torch::Tensor dt, torch::Tensor part) {
  const int H = (int)x.size(0), M = (int)x.size(1);
  const bool f = is_f32(x);
  const int tb = f ? 128 : 256;
  TORCH_CHECK(M % tb == 0 && H % 64 == 0 && dor.is_contiguous() && x.is_contiguous() && dt.is_contiguous() && part.numel() == (int64_t)(M / tb) * 2 * H, "lnout_bwd: shapes");
  if (f) lnout_bwd_t<a100::F32>(dor, x, sto, s12, go, dt, part); else lnout_bwd_t<a100::Bf16>(dor, x, sto, s12, go, dt, part);
}

template <class E>
void lnin_bwd_t(torch::Tensor dxn, torch::Tensor z, torch::Tensor dy, torch::Tensor sti, torch::Tensor gi, torch::Tensor dx, torch::Tensor part) {
  const int T = (int)z.size(0), D = (int)z.size(1);
  const int grid = (int)lnin_bwd_blocks(T, D, E::IS_F32);
#define LB(DD) if (D == DD) { a100::lnin_bwd_kernel<DD, E><<<grid, 256, 0, stream()>>>(cp<E>(dxn), cp<E>(z), cp<E>(dy), f2(sti), gi.data_ptr<float>(), mp<E>(dx), \
    part.data_ptr<float>(), T); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  LB(64) LB(128) LB(256) LB(384)
#undef LB
  TORCH_CHECK(false, "lnin_bwd: unsupported width ", D);
}
// dxn, z, dy [T, D]; sti [T, 2]; gi [D] -> dx [T, D], part [blocks, 2 D]
void lnin_bwd(torch::Tensor dxn, torch::Tensor z, torch::Tensor dy, torch::Tensor sti, torch::Tensor gi, torch::Tensor dx, torch::Tensor part) {
  const int T = (int)z.size(0), D = (int)z.size(1);
  const bool f = is_f32(z);
  TORCH_CHECK(dxn.is_contiguous() && z.is_contiguous() && dy.is_contiguous() && dx.is_contiguous() && part.numel() == lnin_bwd_blocks(T, D, f) * 2 * D, "lnin_bwd: shapes");
  if (f) lnin_bwd_t<a100::F32>(dxn, z, dy, sti, gi, dx, part); else lnin_bwd_t<a100::Bf16>(dxn, z, dy, sti, gi, dx, part);
}

template <class E>
void ln_apply_t(torch::Tensor z, torch::Tensor sti, torch::Tensor g, torch::Tensor b, torch::Tensor xn) {
  const int T = (int)z.size(0), D = (int)z.size(1);
#define LA(DD) if (D == DD) { a100::ln_apply_kernel<DD, E><<<num_sms() * 8, 256, 0, stream()>>>(cp<E>(z), f2(sti), g.data_ptr<float>(), b.data_ptr<float>(), mp<E>(xn), T); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  LA(64) LA(128) LA(256) LA(384)
#undef LA
  TORCH_CHECK(false, "ln_apply: unsupported width ", D);
}
void ln_apply(torch::Tensor z, torch::Tensor sti, torch::Tensor g, torch::Tensor b, torch::Tensor xn) {
  TORCH_CHECK(z.is_contiguous() && xn.is_contiguous(), "ln_apply: layouts");
  if (is_f32(z)) ln_apply_t<a100::F32>(z, sti, g, b, xn); else ln_apply_t<a100::Bf16>(z, sti, g, b, xn);
}

// k1wb: z [T, D], w1 [4 Hs, D], vs / vb, st [T, 2], mask [L] or empty, da [2 Hs, T] -> dpre [T, ldd] (columns [0, 4 Hs))
void k1wb(torch::Tensor z, torch::Tensor w1, torch::Tensor vs, torch::Tensor vb, torch::Tensor st, torch::Tensor mask, torch::Tensor da, torch::Tensor dpre, int64_t L) {
  const int T = (int)z.size(0), D = (int)z.size(1), ncol = (int)w1.size(0), ldd = (int)dpre.stride(0);
  const bool f = is_f32(z);
  TORCH_CHECK(T % 128 == 0 && D % 32 == 0 && ncol % 128 == 0 && da.size(0) == ncol / 2 && da.size(1) == T && dpre.stride(1) == 1 && ldd % (f ? 4 : 8) == 0 && dpre.size(1) >= ncol &&
              da.is_contiguous() && da.scalar_type() == z.scalar_type() && dpre.scalar_type() == z.scalar_type(), "k1wb: shapes");
  if (f) {
    a100::K1wb32Params p;
    p.z = z.data_ptr<float>(); p.w1 = w1.data_ptr<float>(); p.vs = vs.data_ptr<float>(); p.vb = vb.data_ptr<float>(); p.st = f2(st);
    p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr; p.da = da.data_ptr<float>(); p.dpre = dpre.data_ptr<float>();
    p.T = T; p.D = D; p.L = (int)L; p.ncol = ncol; p.ldd = ldd;
    static bool set = false;
    if (!set) { set_smem(a100::k1wb32_kernel, a100::K1wb32Tile::SMEM); set = true; }
    a100::k1wb32_kernel<<<(T / 128) * (ncol / 128), 128, a100::K1wb32Tile::SMEM, stream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  a100::K1wbParams p;
  p.z = cbf(z); p.w1 = cbf(w1); p.vs = vs.data_ptr<float>(); p.vb = vb.data_ptr<float>(); p.st = f2(st);
  p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
  p.da = cbf(da); p.dpre = mbf(dpre);
  p.T = T; p.D = D; p.L = (int)L; p.ncol = ncol; p.ldd = ldd;
  static bool set = false;
  if (!set) { set_smem(a100::k1wb_kernel, a100::K1WB_SMEM); set = true; }
  a100::k1wb_kernel<<<(T / 128) * (ncol / 128), 128, a100::K1WB_SMEM, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the end of the backward (wide_fin.cuh)
template <class E>
void wfin_t(torch::Tensor dw1, torch::Tensor G, torch::Tensor r01, torch::Tensor gout, torch::Tensor gin, torch::Tensor go, torch::Tensor bo,
            torch::Tensor dwl, torch::Tensor dwlg, torch::Tensor dwr, torch::Tensor dwrg, torch::Tensor dwg, torch::Tensor dwo,
            torch::Tensor dgo, torch::Tensor dbo, torch::Tensor dgi, torch::Tensor dbi) {
  const int Hs = (int)dwl.size(0), D = (int)dwl.size(1);
  a100::WFinParams<E> p;
  p.dw1 = dw1.data_ptr<float>(); p.G = G.data_ptr<float>(); p.r01 = r01.data_ptr<float>(); p.gout = gout.data_ptr<float>(); p.gin = gin.data_ptr<float>();
  p.go = go.data_ptr<float>(); p.bo = bo.data_ptr<float>();
  int i = 0;
  for (auto* t : {&dwl, &dwlg, &dwr, &dwrg}) {
    TORCH_CHECK(t->dim() == 2 && t->size(0) == Hs && t->size(1) == D && t->scalar_type() == dwl.scalar_type() && t->stride(0) > 0 && t->stride(1) > 0, "wfin: [Hs, D] outputs");
    p.w[i] = mp<E>(*t); p.rs[i] = (int)t->stride(0); p.ks[i] = (int)t->stride(1); ++i;
  }
  TORCH_CHECK(dwg.is_contiguous() && dwo.is_contiguous() && dwg.scalar_type() == dwl.scalar_type() && dwo.scalar_type() == dwl.scalar_type(), "wfin: dWg / dWo row-major, one dtype");
  p.dwg = mp<E>(dwg); p.dwo = mp<E>(dwo);
  int v = 0;
  for (auto* t : {&dgo, &dbo, &dgi, &dbi}) {
    TORCH_CHECK((t->scalar_type() == at::kBFloat16 || t->scalar_type() == at::kFloat) && t->is_contiguous() && t->numel() == (v < 2 ? Hs : D), "wfin: LayerNorm gradients [Hs] / [D], bf16 or fp32");
    p.ln[v] = t->data_ptr(); p.ln_bf16[v] = t->scalar_type() == at::kBFloat16 ? 1 : 0; ++v;
  }
  p.Hs = Hs; p.D = D;
  const int items = 4 * Hs + 2 * D + 4;
  a100::wfin_kernel<E><<<(items + 7) / 8, 256, 0, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void wfin(torch::Tensor dw1, torch::Tensor G, torch::Tensor r01, torch::Tensor gout, torch::Tensor gin, torch::Tensor go, torch::Tensor bo,
          torch::Tensor dwl, torch::Tensor dwlg, torch::Tensor dwr, torch::Tensor dwrg, torch::Tensor dwg, torch::Tensor dwo,
          torch::Tensor dgo, torch::Tensor dbo, torch::Tensor dgi, torch::Tensor dbi) {
  const int Hs = (int)dwl.size(0), D = (int)dwl.size(1);
  TORCH_CHECK(dw1.is_contiguous() && dw1.size(0) == 4 * Hs + D && dw1.size(1) == D && G.is_contiguous() && G.size(0) == D && G.size(1) == Hs, "wfin: GEMM results");
  if (is_f32(dwl)) wfin_t<a100::F32>(dw1, G, r01, gout, gin, go, bo, dwl, dwlg, dwr, dwrg, dwg, dwo, dgo, dbo, dgi, dbi);
  else wfin_t<a100::Bf16>(dw1, G, r01, gout, gin, go, bo, dwl, dwlg, dwr, dwrg, dwg, dwo, dgo, dbo, dgi, dbi);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stats_rows", &stats_rows);
  m.def("stats_cm", &stats_cm);
  m.def("wpack", &wpack);
  m.def("k1w", &k1w);
  m.def("k3w", &k3w);
  m.def("gate_bwd_blocks", &gate_bwd_blocks);
  m.def("lnin_bwd_blocks", &lnin_bwd_blocks);
  m.def("gate_bwd", &gate_bwd);
  m.def("colsum", &colsum);
  m.def("lnout_bwd", &lnout_bwd);
  m.def("lnin_bwd", &lnin_bwd);
  m.def("ln_apply", &ln_apply);
  m.def("k1wb", &k1wb);
  m.def("wfin", &wfin);
}
