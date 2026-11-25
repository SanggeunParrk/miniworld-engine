// apb_rows_sm100.cu -- the B200 (sm_100) build of AttentionPairBias's row / pair extension (2026-11-17): the shared source apb_rows.cu
// (compiled unchanged for H100) plus B200-only passes. Only the sm_100 build compiles this file (apb/__init__.py ``_ext("100")``), so the
// H100 extension, its source and its key are untouched. The shared file's module body is kept (its PYBIND11_MODULE renamed into a plain
// function, called from this file's module, which then adds the B200 entry points).
//
//   prep_dp   ``prep`` for any pair width dp (128 / 256 / 384 / 512) in ONE launch: W q | k | v | g, its bias, Wo, and Wf / Wb over all dp
//             columns (the wide pair's Wf / Wb were three to four torch launches after ``prep``). The same arithmetic as ``prep`` per element
//             (each parameter rounded to bf16 first, then scaled), so fp32 masters go in as they are: no cast launches.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) static void apb_shared_defs(pybind11::module_& variable)
// SM reserve (2026-11-17): the pair passes fill every SM (their rings take the SM's shared memory / thread slots), so a kernel launched
// beside them on another stream (the q | k | v | g GEMM, the backward's row passes) sat queued until they ended. ``sm_reserve(k)`` makes
// every grid this extension sizes by the SM count leave k SMs free (the shared source reads the count through cudaDeviceGetAttribute,
// wrapped here; the shared file itself is untouched). The caller sets it around the launches it wants to share the card and resets it.
#include <cuda_runtime.h>
namespace { int g_sm_reserve = 0; }
static inline cudaError_t apb_dev_attr(int* v, cudaDeviceAttr a, int d) {
  const cudaError_t e = cudaDeviceGetAttribute(v, a, d);
  if (a == cudaDevAttrMultiProcessorCount && g_sm_reserve > 0 && *v > 2 * g_sm_reserve) *v -= g_sm_reserve;
  return e;
}
#define cudaDeviceGetAttribute apb_dev_attr
#include "apb_rows.cu"
#pragma pop_macro("PYBIND11_MODULE")
#undef cudaDeviceGetAttribute
// PDL (2026-11-17, v20; v21: the hook moved out of the shared apb_rows.cu): B200 kernels open with APB_PDL_ENTRY -- griddepcontrol.wait
// (they read nothing before it), then let their own programmatic dependents be scheduled; launched plainly both are no-ops. The B200
// launchers below (``pdl_go``) start them as programmatic dependents of the previous kernel on the stream while ``pdl(1)`` is set. The
// shared row kernels' PDL forms (*_pk) are B200 copies in apb_rows_pdl_sm100.cuh.
#define APB_PDL_ENTRY() asm volatile("griddepcontrol.wait;\n\tgriddepcontrol.launch_dependents;" ::: "memory")
#include "apb_rows_pdl_sm100.cuh"
#include "apb_pair_fwd_rows_sm100.cuh"

namespace {

// PDL launches (2026-11-17, v20): ``pdl(1)`` makes the launchers below start their kernel as a programmatic dependent of the previous kernel
// on the stream (its griddepcontrol.wait, APB_PDL_ENTRY, holds every read); ``pdl(0)``: plain launches (the same kernels and values).
int g_pdl = 0;
template <typename... KA, typename... A>
void pdl_go(void (*k)(KA...), unsigned grid, unsigned block, size_t smem, A... a) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = S();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = g_pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, a...));
}
// the shared row launchers' B200 forms through pdl_go (the same kernels, grids and arguments as apb_rows.cu gate_rows / gate_bwd / qkv_bwd /
// ln_bwd / ln_rows; bf16 activations)
void gate_rows_p(at::Tensor o, at::Tensor qkvg, at::Tensor og) {
  const int64_t M = o.size(0), W = o.size(1);
  chk(o, at::kFloat, "o"); chk(qkvg, at::kBFloat16, "qkvg"); chk(og, at::kBFloat16, "og");
  TORCH_CHECK((W == 384 || W == 512) && qkvg.size(1) == 4 * W && og.size(1) == W, "gate_rows widths");
  const at::cuda::CUDAGuard gd(o.device());
  pdl_go(W == 384 ? gate_rows_pk<3> : gate_rows_pk<4>, blocks(M), NT, 0, CP<float>(o), CP<bf>(qkvg), P<bf>(og), (int)M);
}
void qkv_bwd_p(at::Tensor dq, at::Tensor dk, at::Tensor dv, at::Tensor dqkvg, at::Tensor acc, int64_t d) {
  const int64_t NCH = dq.dim() == 3 ? dq.size(0) : 1, M = dk.size(0), W = dk.size(1);
  TORCH_CHECK(dq.numel() == NCH * M * W, "qkv_bwd: dq [NCH, M, W]");
  chk(dq, at::kFloat, "dq"); chk(dk, at::kFloat, "dk"); chk(dv, at::kFloat, "dv"); chk(dqkvg, at::kBFloat16, "dqkvg"); chk(acc, at::kFloat, "acc");
  TORCH_CHECK(W == 384 || W == 512, "qkv_bwd width");
  const at::cuda::CUDAGuard gd(dq.device());
  pdl_go(W == 384 ? qkv_bwd_pk<3> : qkv_bwd_pk<4>, blocks(M), NT, 0, CP<float>(dq), CP<float>(dk), CP<float>(dv), P<bf>(dqkvg),
         P<float>(acc) + 2 * d, (int)M, (int)NCH, acc.dim() == 2 ? (int)acc.size(1) : 0);
}
void ln_bwd_p(at::Tensor dxa, at::Tensor x, at::Tensor st, at::Tensor lw, at::Tensor dy, at::Tensor dx, at::Tensor acc) {
  const int64_t D = lw.numel(), M = x.numel() / D;
  TORCH_CHECK(D == 384 || D == 512, "ln_bwd: d_single 384 or 512");
  chk(dxa, at::kBFloat16, "dxa"); chk(x, at::kBFloat16, "x"); chk(dy, at::kBFloat16, "dy"); chk(dx, at::kBFloat16, "dx");
  chk(lw, at::kFloat, "lw"); chk(st, at::kFloat, "stats"); chk(acc, at::kFloat, "acc");
  const at::cuda::CUDAGuard gd(x.device());
  pdl_go(D == 384 ? ln_bwd_pk<bf, bf, 3> : ln_bwd_pk<bf, bf, 4>, blocks(M), NT, 0, CP<bf>(dxa), CP<bf>(x), CP<float2>(st), CP<float>(lw),
         CP<bf>(dy), P<bf>(dx), P<float>(acc), P<float>(acc) + D, (int)M, acc.dim() == 2 ? (int)acc.size(1) : 0);
}
void gate_bwd_p(at::Tensor dog, at::Tensor o, at::Tensor qkvg, at::Tensor dob, at::Tensor dd, at::Tensor dqkvg) {
  const int64_t M = o.size(0), W = o.size(1), nh = dd.size(0);
  chk(dog, at::kBFloat16, "dog"); chk(o, at::kFloat, "o"); chk(dob, at::kBFloat16, "dob"); chk(dd, at::kFloat, "dd");
  chk(dqkvg, at::kBFloat16, "dqkvg");
  TORCH_CHECK((W == 384 && (nh == 8 || nh == 12 || nh == 24)) || (W == 512 && nh == 16), "gate_bwd widths");
  const at::cuda::CUDAGuard gd(o.device());
  auto k = nh == 24 ? gate_bwd_hl_pk<24, 16> : nh == 12 ? gate_bwd_hl_pk<12, 32> : W == 384 ? gate_bwd_pk<3, 4> : gate_bwd_pk<4, 2>;
  pdl_go(k, blocks(M), NT, 0, CP<bf>(dog), CP<float>(o), CP<bf>(qkvg), P<bf>(dob), P<float>(dd), P<bf>(dqkvg), (int)M);
}

// the pack's pointers and scales (prep_dp_k, and the pack blocks of prep_ln_k)
template <typename T> struct PrepArgs {
  const T *WQ, *BQ, *WK, *WV, *WG, *WO, *WB;
  const float* LPW;
  bf *WP, *BV, *WOP, *WF, *WBO;
  Geo G;
  int dp;
  float qs, ws, gate_bias;
};
template <typename T> long prep_items(const PrepArgs<T>& a) {
  const long W = a.G.nh * a.G.dhp, D = a.G.d;
  return 4L * W * D / 8 + (a.WOP ? D * W / 8 : 0) + 4 * W + (long)a.G.nh * a.dp;
}
template <typename T> __device__ __forceinline__ void prep_dp_body(const PrepArgs<T>& a, long i);
template <typename T> __global__ void __launch_bounds__(256) prep_dp_k(PrepArgs<T> a) {
  prep_dp_body<T>(a, (long)blockIdx.x * blockDim.x + threadIdx.x);
}
// prep_ln_k: the pack AND LayerNorm(single) of the rows (ln_rows_k's row math: its statements, a warp per row) in ONE launch -- blocks
// [0, nprep) pack, the rest take 8 rows each. A captured step packs at every replay, and both passes are a few microseconds of launch
// latency: one launch instead of two.
// v28: MB = the minimum blocks per SM in __launch_bounds__ (0: none). At D 512 the default took 64 registers -> 4 blocks per SM, 592
// resident, and the 16 x 32 pack + LayerNorm is 689 blocks: two waves after the flush. ``prep_mb(k)`` picks MB for the next launches.
int g_prep_mb = 0;
template <typename T, int V, int MB = 0>
__global__ void __launch_bounds__(256, MB > 0 ? MB : 1) prep_ln_k(PrepArgs<T> a, int nprep, const bf* __restrict__ X, const float* __restrict__ LW,
    const float* __restrict__ LB, bf* __restrict__ XA, float2* __restrict__ ST, bf* __restrict__ Y, int M, float eps) {
  if ((int)blockIdx.x < nprep) { prep_dp_body<T>(a, (long)blockIdx.x * 256 + threadIdx.x); return; }
  constexpr int D = 128 * V;
  const long r = (long)(blockIdx.x - nprep) * 8 + (threadIdx.x >> 5);
  if (r >= M) return;
  float4 x[V], w[V], b[V];
  ldv<V>(X + r * D, x); ldv<V>(LW, w); ldv<V>(LB, b);
  if (Y) stv<V>(Y + r * D, x);
  const float mean = warp_sum(sumv<V>(x)) * (1.f / D);
#pragma unroll
  for (int k = 0; k < V; ++k) x[k] = make_float4(x[k].x - mean, x[k].y - mean, x[k].z - mean, x[k].w - mean);
  const float rstd = rsqrtf(warp_sum(dotv<V>(x, x)) * (1.f / D) + eps);
#pragma unroll
  for (int k = 0; k < V; ++k)
    x[k] = make_float4(x[k].x * rstd * w[k].x + b[k].x, x[k].y * rstd * w[k].y + b[k].y, x[k].z * rstd * w[k].z + b[k].z,
                       x[k].w * rstd * w[k].w + b[k].w);
  stv<V>(XA + r * D, x);
  if (ST && (threadIdx.x & 31) == 0) ST[r] = make_float2(mean, rstd);
}
template <typename T> __device__ __forceinline__ void prep_dp_body(const PrepArgs<T>& a, long i) {
  const T *WQ = a.WQ, *BQ = a.BQ, *WK = a.WK, *WV = a.WV, *WG = a.WG, *WO = a.WO, *WB = a.WB;
  const float* LPW = a.LPW;
  bf *WP = a.WP, *BV = a.BV, *WOP = a.WOP, *WF = a.WF, *WBO = a.WBO;
  const Geo G = a.G;
  const int dp = a.dp;
  const float qs = a.qs, ws = a.ws, gate_bias = a.gate_bias;
  const int W = G.nh * G.dhp, D = G.d;
  const long n1 = 4L * W * D / 8, n2 = WOP ? (long)D * W / 8 : 0;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  if (i < n1) {                                                             // packed row p (of segment blk), 8 input columns
    const long e = i * 8, prow = e / D, col = e % D;
    const int blk = (int)(prow / W), rr = real_of(G, (int)(prow % W));
    uint4 u = zero;
    if (rr >= 0) {
      float f[8];
      load8<T>((blk == 0 ? WQ : blk == 1 ? WK : blk == 2 ? WV : WG) + (long)rr * D + col, f);
#pragma unroll
      for (int k = 0; k < 8; ++k) f[k] = rbf(f[k]);                         // the bf16 parameter, then its scale
      u = pack8(f, blk == 0 ? qs : 1.f);
    }
    *reinterpret_cast<uint4*>(WP + e) = u;
    return;
  }
  if (i < n1 + n2) {                                                        // Wo row r, 8 padded columns (dh % 8 == 0: one head)
    const long e = (i - n1) * 8, r = e / W, pc = e % W;
    const int rc = real_of(G, (int)pc);
    uint4 u = zero;
    if (rc >= 0) { float f[8]; load8<T>(WO + r * D + rc, f); u = pack8(f, 1.f); }
    *reinterpret_cast<uint4*>(WOP + e) = u;
    return;
  }
  const long j = i - n1 - n2;
  if (j < 4 * W) {
    const int rr = j < W ? real_of(G, (int)j) : -1;
    float v = 0.f;
    if (rr >= 0) { const T b = BQ[rr]; v = rbf(static_cast<float>(b)) * qs; }
    if (j >= 3 * W) v = gate_bias;                                          // sigmoid(g + c): the constant on the gate rows' bias
    BV[j] = __float2bfloat16_rn(v);
  } else if (j < 4 * W + (long)G.nh * dp) {
    const int k = (int)(j - 4 * W);
    const float wb = rbf(static_cast<float>(WB[k]));
    WF[k] = __float2bfloat16_rn(wb * LPW[k % dp] * ws);
    if (WBO) WBO[k] = __float2bfloat16_rn(wb);
  }
}

// ``prep`` at any pair width (wb [nh, dp], lpw [dp] fp32, wf / wbo [nh, dp] bf16), the gate constant written on the g rows of the bias.
// ``ln`` (prep_ln): (x, lw, lb, xa, st or none, y or none, eps) of ``ln_rows`` on bf16 rows, in the same launch.
struct LnArgs { at::Tensor x, lw, lb, xa; c10::optional<at::Tensor> st, y; double eps; };
template <typename T>
void prep_launch(const PrepArgs<T>& a, const LnArgs* ln) {
  const unsigned nprep = (unsigned)((prep_items(a) + 255) / 256);
  if (ln == nullptr) { prep_dp_k<T><<<nprep, 256, 0, S()>>>(a); return; }
  const int64_t D = ln->lw.numel(), M = ln->x.numel() / D;
  auto k = D == 384 ? prep_ln_k<T, 3> : g_prep_mb >= 5 ? prep_ln_k<T, 4, 5> : prep_ln_k<T, 4>;
  k<<<nprep + (unsigned)((M + 7) / 8), 256, 0, S()>>>(a, (int)nprep, CP<bf>(ln->x), CP<float>(ln->lw), CP<float>(ln->lb), P<bf>(ln->xa),
      ln->st ? P<float2>(*ln->st) : nullptr, ln->y ? P<bf>(*ln->y) : nullptr, (int)M, (float)ln->eps);
}
void prep_dp_impl(at::Tensor wq, at::Tensor bq, at::Tensor wk, at::Tensor wv, at::Tensor wg, c10::optional<at::Tensor> wo, at::Tensor wb,
                  at::Tensor lpw, at::Tensor wp, at::Tensor bv, c10::optional<at::Tensor> wop, at::Tensor wf, double qs, double ws,
                  c10::optional<at::Tensor> wbo, double gate_bias, const LnArgs* ln) {
  const auto dt = wq.scalar_type();
  TORCH_CHECK(dt == at::kBFloat16 || dt == at::kFloat, "prep_dp: bf16 or fp32 weights");
  for (auto* t : {&bq, &wk, &wv, &wg, &wb}) chk(*t, dt, "prep_dp: one weight dtype");
  for (auto* t : {&wp, &bv, &wf}) chk(*t, at::kBFloat16, "prep_dp: bf16 outputs");
  chk(lpw, at::kFloat, "ln_pair.weight");
  const int64_t dp = wb.size(1);
  const Geo G = geo(wb.size(0), wq.size(1));
  const int64_t W = width(G), D = G.d;
  TORCH_CHECK(lpw.numel() == dp && wf.numel() == G.nh * dp, "prep_dp: pair width");
  TORCH_CHECK(wq.numel() == D * D && wp.numel() == 4 * W * D && bv.numel() == 4 * W, "prep_dp shapes");
  TORCH_CHECK(wo.has_value() == wop.has_value(), "prep_dp: wo and wop together");
  if (wop) { chk(*wo, dt, "wo"); chk(*wop, at::kBFloat16, "wop"); TORCH_CHECK(wop->numel() == D * W, "wop [D, W]"); }
  if (wbo) { chk(*wbo, at::kBFloat16, "wbo"); TORCH_CHECK(wbo->numel() == G.nh * dp, "wbo [NH, dp]"); }
  if (ln) {
    chk(ln->x, at::kBFloat16, "prep_ln: bf16 x"); chk(ln->xa, at::kBFloat16, "xa"); chk(ln->lw, at::kFloat, "lw"); chk(ln->lb, at::kFloat, "lb");
    TORCH_CHECK(ln->lw.numel() == D && ln->lb.numel() == D && ln->x.numel() % D == 0 && ln->xa.numel() == ln->x.numel(), "prep_ln shapes");
    if (ln->st) { chk(*ln->st, at::kFloat, "st"); TORCH_CHECK(ln->st->numel() == 2 * ln->x.numel() / D, "st [M, 2]"); }
    if (ln->y) { chk(*ln->y, at::kBFloat16, "y"); TORCH_CHECK(ln->y->numel() == ln->x.numel(), "y like x"); }
  }
  const at::cuda::CUDAGuard gd(wq.device());
  if (dt == at::kBFloat16)
    prep_launch<bf>(PrepArgs<bf>{CP<bf>(wq), CP<bf>(bq), CP<bf>(wk), CP<bf>(wv), CP<bf>(wg), wo ? CP<bf>(*wo) : nullptr, CP<bf>(wb),
        CP<float>(lpw), P<bf>(wp), P<bf>(bv), wop ? P<bf>(*wop) : nullptr, P<bf>(wf), wbo ? P<bf>(*wbo) : nullptr, G, (int)dp, (float)qs,
        (float)ws, (float)gate_bias}, ln);
  else
    prep_launch<float>(PrepArgs<float>{CP<float>(wq), CP<float>(bq), CP<float>(wk), CP<float>(wv), CP<float>(wg),
        wo ? CP<float>(*wo) : nullptr, CP<float>(wb), CP<float>(lpw), P<bf>(wp), P<bf>(bv), wop ? P<bf>(*wop) : nullptr, P<bf>(wf),
        wbo ? P<bf>(*wbo) : nullptr, G, (int)dp, (float)qs, (float)ws, (float)gate_bias}, ln);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void prep_dp(at::Tensor wq, at::Tensor bq, at::Tensor wk, at::Tensor wv, at::Tensor wg, c10::optional<at::Tensor> wo, at::Tensor wb,
             at::Tensor lpw, at::Tensor wp, at::Tensor bv, c10::optional<at::Tensor> wop, at::Tensor wf, double qs, double ws,
             c10::optional<at::Tensor> wbo, double gate_bias) {
  prep_dp_impl(wq, bq, wk, wv, wg, wo, wb, lpw, wp, bv, wop, wf, qs, ws, wbo, gate_bias, nullptr);
}
// ``prep_dp`` and ``ln_rows(x, lw, lb, xa, st, eps, y)`` (bf16 rows) in one launch
void prep_ln(at::Tensor wq, at::Tensor bq, at::Tensor wk, at::Tensor wv, at::Tensor wg, c10::optional<at::Tensor> wo, at::Tensor wb,
             at::Tensor lpw, at::Tensor wp, at::Tensor bv, c10::optional<at::Tensor> wop, at::Tensor wf, double qs, double ws,
             c10::optional<at::Tensor> wbo, double gate_bias, at::Tensor x, at::Tensor lw, at::Tensor lb, at::Tensor xa,
             c10::optional<at::Tensor> st, double eps, c10::optional<at::Tensor> y) {
  const LnArgs ln{x, lw, lb, xa, st, y, eps};
  prep_dp_impl(wq, bq, wk, wv, wg, wo, wb, lpw, wp, bv, wop, wf, qs, ws, wbo, gate_bias, &ln);
}

// ------------------------------------------------------------------------------------------------ fp32 pair -> bias on the tensor cores
// pair_bias32_tc: what pair_bias32_fwd computes (bias[h, i, j] = Wf[h] . LN(pair[i, j]), fp32 head-major, ``neg`` on masked keys, the keys
// optionally in the TF32 core's key_perm order, optionally every row's (mean, rstd)), with the projection on the TF32 tensor cores
// (mma.sync m16n8k8: x^ = (x - mean) rstd and Wf rounded to TF32 to nearest, fp32 sums -- the reference's own TF32 Linear) instead of
// fp32 FMAs and lane shuffles. The fp32 pass was issue-bound (shuffle reduce-scatters, 18 FMAs per element: 1.8-2.3 TB/s at L 512).
// Persistent, one block per SM (6-8 warps); a tile is TK consecutive keys of one query row (TK DPW 4 bytes = 48-64 KB, contiguous),
// brought by DPW / 32 TMA boxes of 32 columns in the 128-B swizzle (conflict-free fragment reads), 3 stages (2 at 512). Warp w takes the 16-row group
// w % RG of the tile and the 128-column slice w / RG (RG x KS = 8 warps, 6 at 384): it reads its 64 values into registers once, the row statistics
// are sums about a pivot (the row's first value) combined over the slices in a fixed order, then the MMAs over its slice; the slices'
// partial products are added in slice order (deterministic). Each tile's planes are staged and leave in 16-byte stores per head row.
template <int DPW> constexpr int tc_rows() { return DPW <= 128 ? 128 : DPW <= 256 ? 64 : 32; }
template <int DPW> constexpr int tc_warps() { return (tc_rows<DPW>() / 16) * (DPW / 128); }   // row groups x 128-column slices (6 at 384)
template <int DPW> constexpr int tc_st() { return DPW >= 512 ? 2 : 3; }             // stages (512: 3 x 64 KB do not fit beside Wf)
template <int NH> constexpr int tc_nhp() { return (NH + 7) / 8 * 8; }
template <int NH, int DPW> constexpr int tc_tile() { return tc_rows<DPW>() * DPW * 4; }
template <int NH, int DPW> constexpr int tc_ks() { return DPW / 128; }                       // column slices = 8 warps / row groups
template <int NH, int DPW> constexpr int tc_smem() {
  return tc_st<DPW>() * tc_tile<NH, DPW>() + tc_nhp<NH>() * (DPW + 4) * 4                    // ring, Wf (TF32, padded rows)
         + tc_ks<NH, DPW>() * tc_nhp<NH>() * (tc_rows<DPW>() + 4) * 4                         // the slices' partial planes
         + tc_ks<NH, DPW>() * tc_rows<DPW>() * 2 * 4 + 1024 + 64;                            // slice statistics, alignment, barriers
}
__device__ __forceinline__ uint32_t tf32r(float v) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v)); return r; }
__device__ __forceinline__ void mma1688tf32(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// byte offset of (row r, column c) in a tile of DPW / 32 boxes [TK rows][32 fp32], each 128-B swizzled
template <int TK> __device__ __forceinline__ uint32_t tcoff(int r, int c) {
  return (uint32_t)((c >> 5) * TK * 128 + r * 128 + ((((c & 31) >> 2) ^ (r & 7)) << 4) + (c & 3) * 4);
}
template <int NH, int DPW>
__global__ void __launch_bounds__(256, 1) pair_bias32_tc_k(const __grid_constant__ CUtensorMap tmz, const float* __restrict__ WF,
    const bool* __restrict__ MASK, float* __restrict__ OUT, float2* __restrict__ ST, int L, float eps, float neg, int kperm) {
  constexpr int TK = tc_rows<DPW>(), NHP = tc_nhp<NH>(), NT = NHP / 8, RG = TK / 16, KS = tc_ks<NH, DPW>(), TB = tc_tile<NH, DPW>();
  constexpr int TC_ST = tc_st<DPW>();
  constexpr int TC_WARPS = tc_warps<DPW>();
  constexpr int WS = DPW + 4, PS = TK + 4;                                  // padded row strides (floats): Wf, partial planes
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  float* wsm = reinterpret_cast<float*>(smem + TC_ST * TB);                 // [NHP][WS] TF32 bit patterns
  float* part = wsm + NHP * WS;                                             // [KS][NHP][PS] (written after a tile's first barrier, read before the next one's)
  float* sst = part + KS * NHP * PS;                                    // [KS][TK][2]: per slice (sum, sum of squares) about the pivot
  uint64_t* full = reinterpret_cast<uint64_t*>(sst + KS * TK * 2);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int rg = warp % RG, ks = warp / RG, r0 = rg * 16 + g, c0 = ks * 128;
  const int ntj = L / TK;
  const long ntiles = (long)L * ntj;
  auto issue = [&](long n) {
    const long t = blockIdx.x + n * gridDim.x;
    if (t >= ntiles) return;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * TK;
    uint64_t* fb = &full[n % TC_ST];
    const uint32_t st = su32(smem + (n % TC_ST) * TB);
    mbar_expect_tx(fb, TB);
#pragma unroll
    for (int b = 0; b < DPW / 32; ++b) tma2d(st + b * TK * 128, &tmz, 32 * b, i * L + j0, fb);
  };
  if (threadIdx.x == 0) {
    for (int k = 0; k < TC_ST; ++k) mbar_init(&full[k], 1);
    mbar_fence_init();
  }
  for (int k = threadIdx.x; k < NHP * DPW; k += 32 * TC_WARPS) {
    const int h = k / DPW, c = k % DPW;
    wsm[h * WS + c] = __uint_as_float(h < NH ? tf32r(WF[h * DPW + c]) : 0u);
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int k = 0; k < TC_ST; ++k) issue(k);
  for (long n = 0;; ++n) {
    const long t = blockIdx.x + n * gridDim.x;
    if (t >= ntiles) break;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * TK, sti = (int)(n % TC_ST);
    mbar_wait(&full[sti], (uint32_t)((n / TC_ST) & 1));
    const uint8_t* tile = smem + sti * TB;
    // this warp's 16 rows x 128 columns: lane (g, q) holds rows r0, r0 + 8 at columns c0 + 8 kk + q (+ 4)
    float xa[16][2], xb[16][2];
    const float p0 = *reinterpret_cast<const float*>(tile + tcoff<TK>(r0, 0)), p1 = *reinterpret_cast<const float*>(tile + tcoff<TK>(r0 + 8, 0));
    float s0 = 0.f, s1 = 0.f, q0 = 0.f, q1 = 0.f;
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = c0 + 8 * kk + q + 4 * u;
        xa[kk][u] = *reinterpret_cast<const float*>(tile + tcoff<TK>(r0, c));
        xb[kk][u] = *reinterpret_cast<const float*>(tile + tcoff<TK>(r0 + 8, c));
        const float da = xa[kk][u] - p0, db = xb[kk][u] - p1;
        s0 += da; q0 = fmaf(da, da, q0); s1 += db; q1 = fmaf(db, db, q1);
      }
    }
    s0 = quad_sum(s0); q0 = quad_sum(q0); s1 = quad_sum(s1); q1 = quad_sum(q1);
    if (q == 0) {
      sst[(ks * TK + r0) * 2] = s0; sst[(ks * TK + r0) * 2 + 1] = q0;
      sst[(ks * TK + r0 + 8) * 2] = s1; sst[(ks * TK + r0 + 8) * 2 + 1] = q1;
    }
    __syncthreads();                                                        // every warp holds its values; the slices' sums are in
    if (threadIdx.x == 0) issue(n + TC_ST);
    float m0, rs0, m1, rs1;
    {
      float a = 0.f, b = 0.f, c = 0.f, d = 0.f;
#pragma unroll
      for (int k = 0; k < KS; ++k) {                                        // slice order: the same sums in every warp
        a += sst[(k * TK + r0) * 2]; b += sst[(k * TK + r0) * 2 + 1];
        c += sst[(k * TK + r0 + 8) * 2]; d += sst[(k * TK + r0 + 8) * 2 + 1];
      }
      const float e0 = a * (1.f / DPW), e1 = c * (1.f / DPW);
      m0 = p0 + e0; m1 = p1 + e1;
      rs0 = rsqrtf(fmaxf(b * (1.f / DPW) - e0 * e0, 0.f) + eps);
      rs1 = rsqrtf(fmaxf(d * (1.f / DPW) - e1 * e1, 0.f) + eps);
    }
    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      const uint32_t a0 = tf32r((xa[kk][0] - m0) * rs0), a1 = tf32r((xb[kk][0] - m1) * rs1);
      const uint32_t a2 = tf32r((xa[kk][1] - m0) * rs0), a3 = tf32r((xb[kk][1] - m1) * rs1);
      const int c = c0 + 8 * kk + q;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const float* wr = wsm + (nt * 8 + g) * WS + c;
        mma1688tf32(acc[nt], a0, a1, a2, a3, __float_as_uint(wr[0]), __float_as_uint(wr[4]));
      }
    }
    float* pp = part + ks * NHP * PS;                     // [NHP][PS]: head, key
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int h = nt * 8 + 2 * q;
      pp[h * PS + rg * 16 + g] = acc[nt][0]; pp[(h + 1) * PS + rg * 16 + g] = acc[nt][1];
      pp[h * PS + rg * 16 + g + 8] = acc[nt][2]; pp[(h + 1) * PS + rg * 16 + g + 8] = acc[nt][3];
    }
    if (ST != nullptr && ks == 0 && q == 0) {
      ST[(long)i * L + j0 + r0] = make_float2(m0, rs0);
      ST[(long)i * L + j0 + r0 + 8] = make_float2(m1, rs1);
    }
    __syncthreads();                                                        // the partial planes are in
    const float* pb = part;
    for (int task = threadIdx.x; task < NH * (TK / 4); task += 32 * TC_WARPS) {
      const int h = task / (TK / 4), p = (task % (TK / 4)) * 4;             // output positions p .. p + 3 of head h
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int pos = p + e, key = kperm ? (pos & ~7) + 4 * (pos & 1) + ((pos & 7) >> 1) : pos;
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < KS; ++k) s += pb[(k * NHP + h) * PS + key];
        v[e] = (MASK == nullptr || MASK[j0 + key]) ? s : neg;
      }
      *reinterpret_cast<float4*>(OUT + ((long)h * L + i) * L + j0 + p) = make_float4(v[0], v[1], v[2], v[3]);
    }
  }
}

template <int NH, int DPW>
void pair_bias32_tc_t(const at::Tensor& z, const at::Tensor& wf, const c10::optional<at::Tensor>& mask, at::Tensor& out,
                      const c10::optional<at::Tensor>& stats, int64_t L, double eps, double neg, bool kperm) {
  constexpr int TK = tc_rows<DPW>();
  const cuuint64_t zd[2] = {(cuuint64_t)DPW, (cuuint64_t)(L * L)}, zs[1] = {(cuuint64_t)DPW * 4};
  const cuuint32_t zb[2] = {32, (cuuint32_t)TK};
  const CUtensorMap mz = tensor_map(z.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, zd, zs, zb, CU_TENSOR_MAP_SWIZZLE_128B);
  auto k = pair_bias32_tc_k<NH, DPW>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, tc_smem<NH, DPW>()));
    attr = true;
  }
  const long ntiles = L * (L / TK);
  k<<<(unsigned)std::min<long>(nsm_of(z), ntiles), 32 * tc_warps<DPW>(), tc_smem<NH, DPW>(), S()>>>(mz, CP<float>(wf),
      mask ? CP<bool>(*mask) : nullptr, P<float>(out), stats ? P<float2>(*stats) : nullptr, (int)L, (float)eps, (float)neg, kperm ? 1 : 0);
}
// z [L L, dp] fp32, wf [NH, dp] fp32 -> out [NH, L, L] fp32 (natural key order, or ``kperm``: the TF32 core's key_perm order), stats
// (optional) [L L, 2]; L a multiple of 128
void pair_bias32_tc(at::Tensor z, at::Tensor wf, c10::optional<at::Tensor> mask, at::Tensor out, c10::optional<at::Tensor> stats, int64_t L,
                    double eps, double neg, bool kperm) {
  chk(z, at::kFloat, "pair"); chk(wf, at::kFloat, "wf"); chk(out, at::kFloat, "bias");
  const int64_t nh = wf.size(0), dp = wf.size(1);
  TORCH_CHECK(L % 128 == 0 && z.numel() == L * L * dp && out.numel() == nh * L * L, "pair_bias32_tc shapes");
  TORCH_CHECK(z.data_ptr() != nullptr && reinterpret_cast<uintptr_t>(z.data_ptr()) % 16 == 0, "pair_bias32_tc: 16-byte aligned pair");
  if (mask) { chk(*mask, at::kBool, "mask"); TORCH_CHECK(mask->numel() == L, "mask [L]"); }
  if (stats) { chk(*stats, at::kFloat, "stats"); TORCH_CHECK(stats->numel() == 2 * L * L, "stats [L L, 2]"); }
  const at::cuda::CUDAGuard gd(z.device());
  pair32_dispatch(nh, dp, [&](auto nh_t, auto dp_t) {
    constexpr int NH = decltype(nh_t)::value, DPW = decltype(dp_t)::value;
    pair_bias32_tc_t<NH, DPW>(z, wf, mask, out, stats, L, eps, neg, kperm);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------------ fp32 training glue (one launch each)
// The B200 TF32 training cores pair samples: a batch row runs as A = 2 with a zero second sample, its operands rounded to TF32 to nearest.
//   pack2_rna   Q2 [2 L, 3 W] = (rna(q | k | v of QKVG [L, ld]) ; zeros)              (was: zeros, a copy, rna_rows)
//   gate32      s = sigmoid(g) written over the g columns of QKVG, OG = s O             (was: sigmoid_, a multiply)
//   gate32_bwd  DO2 [2 L, W] = (rna(dog s) ; zeros), DD2 [2, H, L] = (sum_head(dog O s) ; zeros), DG = dog O s (1 - s) (ld 4 W)
//               (was: two fills, four multiplies, rna_rows, a head sum and its transpose copy)
__global__ void __launch_bounds__(256) pack2_rna_k(const float* __restrict__ QKVG, float* __restrict__ Q2, int L, int w3, int ld) {
  const int c4 = w3 / 4;
  const long n4 = 2L * L * c4;
  for (long e = (long)blockIdx.x * 256 + threadIdx.x; e < n4; e += (long)gridDim.x * 256) {
    const long r = e / c4;
    const int c = (int)(e - r * c4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < L) {
      v = *reinterpret_cast<const float4*>(QKVG + r * ld + c);
      v.x = rowmath::rna_tf32(v.x); v.y = rowmath::rna_tf32(v.y); v.z = rowmath::rna_tf32(v.z); v.w = rowmath::rna_tf32(v.w);
    }
    *reinterpret_cast<float4*>(Q2 + r * w3 + c) = v;
  }
}
__global__ void __launch_bounds__(256) gate32_k(float* __restrict__ QKVG, const float* __restrict__ O, float* __restrict__ OG, int L, int W, int ld) {
  const int c4 = W / 4;
  const long n4 = (long)L * c4;
  for (long e = (long)blockIdx.x * 256 + threadIdx.x; e < n4; e += (long)gridDim.x * 256) {
    const long r = e / c4;
    const int c = (int)(e - r * c4) * 4;
    float4 g = *reinterpret_cast<const float4*>(QKVG + r * ld + 3 * W + c);
    const float4 o = *reinterpret_cast<const float4*>(O + r * W + c);
    g.x = 1.f / (1.f + expf(-g.x)); g.y = 1.f / (1.f + expf(-g.y)); g.z = 1.f / (1.f + expf(-g.z)); g.w = 1.f / (1.f + expf(-g.w));
    *reinterpret_cast<float4*>(QKVG + r * ld + 3 * W + c) = g;
    *reinterpret_cast<float4*>(OG + r * W + c) = make_float4(g.x * o.x, g.y * o.y, g.z * o.z, g.w * o.w);
  }
}
// a thread per (row, head) of the live sample; the threads past L H write the second sample's zeros
__global__ void __launch_bounds__(256) gate32_bwd_k(const float* __restrict__ DOG, const float* __restrict__ O, const float* __restrict__ QKVG,
    float* __restrict__ DO2, float* __restrict__ DD2, float* __restrict__ DG, int L, int H, int dhp, int ld, int ldg) {
  const int W = H * dhp;
  const long e = (long)blockIdx.x * 256 + threadIdx.x;
  if (e < (long)L * H) {
    const int r = (int)(e / H), h = (int)(e % H);
    const long base = (long)r * W + h * dhp;
    const float* s = QKVG + (long)r * ld + 3 * W + h * dhp;
    float sum = 0.f;
    for (int c = 0; c < dhp; c += 4) {
      const float4 d = *reinterpret_cast<const float4*>(DOG + base + c), o = *reinterpret_cast<const float4*>(O + base + c);
      const float4 sv = *reinterpret_cast<const float4*>(s + c);
      *reinterpret_cast<float4*>(DO2 + base + c) = make_float4(rowmath::rna_tf32(d.x * sv.x), rowmath::rna_tf32(d.y * sv.y),
                                                               rowmath::rna_tf32(d.z * sv.z), rowmath::rna_tf32(d.w * sv.w));
      const float4 dO = make_float4(d.x * o.x, d.y * o.y, d.z * o.z, d.w * o.w);       // dog O
      sum += (dO.x * sv.x + dO.y * sv.y) + (dO.z * sv.z + dO.w * sv.w);
      *reinterpret_cast<float4*>(DG + (long)r * ldg + h * dhp + c) =
          make_float4(dO.x * (sv.x * (1.f - sv.x)), dO.y * (sv.y * (1.f - sv.y)), dO.z * (sv.z * (1.f - sv.z)), dO.w * (sv.w * (1.f - sv.w)));
    }
    DD2[(long)h * L + r] = sum;
  } else if (e < 2L * L * H) {                                              // the second sample: dO rows and D zero
    const long z = e - (long)L * H;
    const int r = (int)(z / H), h = (int)(z % H);
    for (int c = 0; c < dhp; c += 4) *reinterpret_cast<float4*>(DO2 + ((long)(L + r)) * W + h * dhp + c) = make_float4(0.f, 0.f, 0.f, 0.f);
    DD2[(long)H * L + (long)h * L + r] = 0.f;
  }
}
unsigned glue_grid(long n) { return (unsigned)std::min<long>((n + 255) / 256, 148L * 16); }
void pack2_rna(at::Tensor qkvg, at::Tensor q2, int64_t L) {
  chk(q2, at::kFloat, "q2");
  TORCH_CHECK(qkvg.scalar_type() == at::kFloat && qkvg.stride(1) == 1 && q2.size(0) == 2 * L && q2.size(1) % 4 == 0, "pack2_rna shapes");
  const at::cuda::CUDAGuard gd(qkvg.device());
  pack2_rna_k<<<glue_grid(2 * L * q2.size(1) / 4), 256, 0, S()>>>(CP<float>(qkvg), P<float>(q2), (int)L, (int)q2.size(1), (int)qkvg.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void gate32(at::Tensor qkvg, at::Tensor o, at::Tensor og, int64_t L) {
  chk(og, at::kFloat, "og");
  const int64_t W = og.size(1);
  TORCH_CHECK(qkvg.scalar_type() == at::kFloat && qkvg.stride(1) == 1 && o.stride(0) == W && o.stride(1) == 1 && W % 4 == 0, "gate32 shapes");
  const at::cuda::CUDAGuard gd(qkvg.device());
  gate32_k<<<glue_grid(L * W / 4), 256, 0, S()>>>(P<float>(qkvg), CP<float>(o), P<float>(og), (int)L, (int)W, (int)qkvg.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void gate32_bwd(at::Tensor dog, at::Tensor o, at::Tensor qkvg, at::Tensor do2, at::Tensor dd2, at::Tensor dg, int64_t L, int64_t H, int64_t dhp) {
  chk(dog, at::kFloat, "dog"); chk(do2, at::kFloat, "do2"); chk(dd2, at::kFloat, "dd2");
  TORCH_CHECK(o.scalar_type() == at::kFloat && o.stride(0) == H * dhp && o.stride(1) == 1 && dg.stride(1) == 1 && qkvg.stride(1) == 1
              && dhp % 4 == 0 && do2.size(0) == 2 * L, "gate32_bwd shapes");
  const at::cuda::CUDAGuard gd(dog.device());
  gate32_bwd_k<<<(unsigned)((2 * L * H + 255) / 256), 256, 0, S()>>>(CP<float>(dog), CP<float>(o), CP<float>(qkvg), P<float>(do2), P<float>(dd2),
      P<float>(dg), (int)L, (int)H, (int)dhp, (int)qkvg.stride(0), (int)dg.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ------------------------------------------------------------------------------------------------ wide pair backward, one launch (B200)
// pair_bias_bwd_wide1_k: pair_bias_bwd_wide_k (apb_rows.cu, one launch per 128-column chunk of a 256 / 384 / 512-wide pair) with the
// chunks in ONE launch: block b takes chunk b % NC and walks that chunk's tiles as block b / NC of gridDim / NC. The same tiles, fragments
// and products per (tile, chunk) -- the same dpair bits; a launch's fixed part (Wf's row sums and B words: ~200 dependent loads per
// block, ~12 us) is paid once instead of NC times (L 128, 512 wide: 4 x 15 us -> one pass).
template <int NH>
__global__ void __launch_bounds__(256, bwd_bps<NH>()) pair_bias_bwd_wide1_k(const __grid_constant__ CUtensorMap tmz,
    const __grid_constant__ CUtensorMap tmd, const bf* __restrict__ W, const float2* __restrict__ ST, const float* __restrict__ BU,
    bf* __restrict__ DZ, float* __restrict__ AWF, float* __restrict__ AHS, int L, int dpw, int acc_stride) {
  const int NC = dpw / DP, c0 = (int)(blockIdx.x % NC) * DP;                 // this block's chunk, its index and count among the chunk's blocks
  const long bi = blockIdx.x / NC, gd = gridDim.x / NC;
  constexpr int NT8 = nhp<NH>() / 8, BW = NT8, BT = bwd_tile<NH>(), NSB = bwd_stages<NH>();
  constexpr int RING = NSB * BT > 8 * (NH * DP + NH) * 4 ? NSB * BT : 8 * (NH * DP + NH) * 4;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + RING);
  uint32_t* bwt = reinterpret_cast<uint32_t*>(smem + RING + 64);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int ntj = (L + 127) / 128;
  const long ntiles = (long)L * ntj, LL = (long)L * L;
  const float inv = 1.f / (float)dpw;
  auto issue = [&](long n) {
    const long t = bi + n * gd;
    if (t >= ntiles) return;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * 128;
    uint64_t* fb = &full[n % NSB];
    const uint32_t st = su32(smem + (n % NSB) * BT);
    mbar_expect_tx(fb, BT);
    const int row = i * L + j0;
    tma2d(st, &tmz, c0, row, fb);
    tma2d(st + BT_PAIR / 2, &tmz, c0 + 64, row, fb);
    tma3d(st + BT_PAIR, &tmd, j0, i, 0, fb);
  };
  if (threadIdx.x == 0) {
    for (int k = 0; k < NSB; ++k) mbar_init(&full[k], 1);
    mbar_fence_init();
  }
  if (warp == 0) {                                                          // dx^ B words of the chunk's columns
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const int col = c0 + 32 * (t >> 2) + 8 * (g >> 1) + 2 * (t & 3) + (g & 1);
#pragma unroll
      for (int k = 0; k < BW; ++k)
        bwt[((t >> 2) * 32 + lane) * 4 * BW + (t & 3) * BW + k] =
            2 * q + 8 * k < NH ? pack2(__bfloat162float(W[(2 * q + 8 * k) * dpw + col]), __bfloat162float(W[(2 * q + 1 + 8 * k) * dpw + col])) : 0u;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int k = 0; k < NSB; ++k) issue(k);
  float sw0[NT8], sw1[NT8];
#pragma unroll
  for (int nt = 0; nt < NT8; ++nt) {                                        // sum_c Wf over the FULL row, heads 2q, 2q + 1 (+ 8 nt)
    float sw = 0.f;
    for (int c = 8 * q; c < dpw; c += 32)
#pragma unroll
      for (int e = 0; e < 8; ++e) sw += g + 8 * nt < NH ? __bfloat162float(W[(g + 8 * nt) * dpw + c + e]) : 0.f;
    sw = quad_sum(sw);
    sw0[nt] = __shfl_sync(0xffffffffu, sw, 8 * q); sw1[nt] = __shfl_sync(0xffffffffu, sw, 8 * q + 4);
  }
  float acc[8][NT8][4] = {};
  float hsum[NT8] = {}, corr[NT8] = {};
  const uint32_t lrow = (uint32_t)((lane & 7) + ((lane >> 4) << 3)), lcol = (uint32_t)(((lane >> 3) & 1) * 8);
  const int rb = warp * 16;
  for (long n = 0;; ++n) {
    const long t = bi + n * gd;
    if (t >= ntiles) break;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * 128, rows = min(128, L - j0), st = (int)(n % NSB);
    mbar_wait(&full[st], (uint32_t)((n / NSB) & 1));
    const uint8_t* tile = smem + st * BT;
    const uint32_t tb = su32(tile);
    const float* dbs = reinterpret_cast<const float*>(tile + BT_PAIR);
    if (rb < rows) {
      const long R = (long)i * L + j0 + rb;
      const float2 t0 = ST[R + g], t1 = ST[R + g + 8];
      const float mu0 = t0.x, rs0 = t0.y, mu1 = t1.x, rs1 = t1.y;
      float d0[NT8][2], d1[NT8][2], s1a = 0.f, s1b = 0.f, s2a = 0.f, s2b = 0.f;
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int h = 2 * q + j + 8 * nt;
          const bool live = h < NH;
          d0[nt][j] = live ? dbs[h * 128 + rb + g] : 0.f;
          d1[nt][j] = live ? dbs[h * 128 + rb + g + 8] : 0.f;
          if (live) {
            s2a = fmaf(d0[nt][j], __ldg(BU + h * LL + R + g), s2a);
            s2b = fmaf(d1[nt][j], __ldg(BU + h * LL + R + g + 8), s2b);
          }
        }
        s1a += d0[nt][0] * sw0[nt] + d0[nt][1] * sw1[nt]; s1b += d1[nt][0] * sw0[nt] + d1[nt][1] * sw1[nt];
      }
      uint32_t ad0[NT8], ad1[NT8];
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) { ad0[nt] = pack2(d0[nt][0], d0[nt][1]); ad1[nt] = pack2(d1[nt][0], d1[nt][1]); }
      auto dxhat = [&](int u, float (&c)[4][4], uint32_t (&u0)[4], uint32_t (&u1)[4]) {
        const uint4 x0v = lds128(tb + swz(rb + g, 32 * u + 8 * q)), x1v = lds128(tb + swz(rb + g + 8, 32 * u + 8 * q));
        u0[0] = x0v.x; u0[1] = x0v.y; u0[2] = x0v.z; u0[3] = x0v.w; u1[0] = x1v.x; u1[1] = x1v.y; u1[2] = x1v.z; u1[3] = x1v.w;
        uint32_t bw4[4 * BW];
#pragma unroll
        for (int k = 0; k < BW; ++k) {
          const uint4 bv = lds128(su32(bwt + (u * 32 + lane) * 4 * BW + 4 * k));
          bw4[4 * k] = bv.x; bw4[4 * k + 1] = bv.y; bw4[4 * k + 2] = bv.z; bw4[4 * k + 3] = bv.w;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          c[k][0] = c[k][1] = c[k][2] = c[k][3] = 0.f;
#pragma unroll
          for (int p = 0; p + 1 < NT8; p += 2)
            mma16816(c[k], ad0[p], ad1[p], ad0[p + 1], ad1[p + 1], bw4[BW * k + p], bw4[BW * k + p + 1]);
          if constexpr (NT8 & 1) mma1688(c[k], ad0[NT8 - 1], ad1[NT8 - 1], bw4[BW * k + NT8 - 1]);
        }
      };
      const float S1a = quad_sum(s1a), S1b = quad_sum(s1b), S2a = quad_sum(s2a), S2b = quad_sum(s2b);
      const float m1a = S1a * inv, m1b = S1b * inv, m2a = S2a * inv, m2b = S2b * inv;
      const float ka = -rs0 * rs0 * m2a, kb = -rs1 * rs1 * m2b;
      const float ca = rs0 * (rs0 * m2a * mu0 - m1a), cb = rs1 * (rs1 * m2b * mu1 - m1b);
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        float c[4][4]; uint32_t u0[4], u1[4];
        dxhat(u, c, u0, u1);
        uint32_t o0[4], o1[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 x0 = unpack2(u0[k]), x1 = unpack2(u1[k]);
          o0[k] = pack2(fmaf(rs0, c[k][0], fmaf(ka, x0.x, ca)), fmaf(rs0, c[k][1], fmaf(ka, x0.y, ca)));
          o1[k] = pack2(fmaf(rs1, c[k][2], fmaf(kb, x1.x, cb)), fmaf(rs1, c[k][3], fmaf(kb, x1.y, cb)));
        }
        *reinterpret_cast<uint4*>(DZ + (R + g) * dpw + c0 + 32 * u + 8 * q) = make_uint4(o0[0], o0[1], o0[2], o0[3]);
        *reinterpret_cast<uint4*>(DZ + (R + g + 8) * dpw + c0 + 32 * u + 8 * q) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
      }
      const float ra = __shfl_sync(0xffffffffu, rs0, 8 * q), rb_ = __shfl_sync(0xffffffffu, rs0, 8 * q + 4);
      const float rc = __shfl_sync(0xffffffffu, rs1, 8 * q), rd = __shfl_sync(0xffffffffu, rs1, 8 * q + 4);
      const float ma = __shfl_sync(0xffffffffu, mu0, 8 * q), mb = __shfl_sync(0xffffffffu, mu0, 8 * q + 4);
      const float mc = __shfl_sync(0xffffffffu, mu1, 8 * q), md = __shfl_sync(0xffffffffu, mu1, 8 * q + 4);
      uint32_t db0[NT8], db1[NT8];
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
        const bool live = g + 8 * nt < NH;
        const float2 e01 = live ? *reinterpret_cast<const float2*>(dbs + (g + 8 * nt) * 128 + rb + 2 * q) : make_float2(0.f, 0.f);
        const float2 e89 = live ? *reinterpret_cast<const float2*>(dbs + (g + 8 * nt) * 128 + rb + 2 * q + 8) : make_float2(0.f, 0.f);
        const float aa = ra * e01.x, ab = rb_ * e01.y, ac = rc * e89.x, ad = rd * e89.y;
        hsum[nt] += e01.x + e01.y + e89.x + e89.y;
        corr[nt] += aa * ma + ab * mb + ac * mc + ad * md;
        db0[nt] = pack2(aa, ab); db1[nt] = pack2(ac, ad);
      }
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) {
        uint32_t a[4];
        ldsm_x4_trans(a, tb + swz(rb + (int)lrow, 16 * mt + (int)lcol));
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt) mma16816(acc[mt][nt], a[0], a[1], a[2], a[3], db0[nt], db1[nt]);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) issue(n + NSB);
  }
  constexpr int RS = NH * DP + NH;
  float* red = reinterpret_cast<float*>(smem);
  float* pw = red + warp * RS;
#pragma unroll
  for (int nt = 0; nt < NT8; ++nt) {
    const float cq = quad_sum(corr[nt]), hq = quad_sum(hsum[nt]);
    const float cc0 = __shfl_sync(0xffffffffu, cq, 8 * q), cc1 = __shfl_sync(0xffffffffu, cq, 8 * q + 4);
    const int h0 = 2 * q + 8 * nt;
    if (h0 < NH) {
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) {
        pw[h0 * DP + 16 * mt + g] = acc[mt][nt][0] - cc0; pw[(h0 + 1) * DP + 16 * mt + g] = acc[mt][nt][1] - cc1;
        pw[h0 * DP + 16 * mt + g + 8] = acc[mt][nt][2] - cc0; pw[(h0 + 1) * DP + 16 * mt + g + 8] = acc[mt][nt][3] - cc1;
      }
    }
    if (q == 0 && g + 8 * nt < NH) pw[NH * DP + g + 8 * nt] = hq;
  }
  __syncthreads();
  for (int c = threadIdx.x; c < RS; c += 256) {
    if (c >= NH * DP && c0 != 0) continue;                                  // the head sums: chunk 0 only
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += red[w * RS + c];
    float* dst = c < NH * DP ? AWF + (long)(c / DP) * dpw + c0 + (c % DP) : AHS + (c - NH * DP);
    if (acc_stride) dst[(long)blockIdx.x * acc_stride] = v;
    else atomicAdd(dst, v);
  }
  if (acc_stride) {                                                         // block partial rows: the other chunks' columns (and, past chunk 0,
    float* row = AWF + (long)blockIdx.x * acc_stride;                       // the head sums) of this block's row are zero
    for (int c = threadIdx.x; c < NH * dpw + NH; c += 256)
      if (c < NH * dpw ? (c % dpw) / DP != c0 / DP : c0 != 0) row[c] = 0.f;
  }
}
// blocks per chunk, and the launch's grid (= the partial rows a 2-D accumulator needs)
long wide1_per(long bps, long nsm, long nc, long ntiles) { return std::max<long>(1, std::min<long>(bps * nsm / nc, ntiles)); }
int64_t pair_bwd_wide1_blocks(at::Tensor like, int64_t nh, int64_t L, int64_t dp) {
  return wide1_per(nh == 8 ? 2 : 1, nsm_of(like), dp / DP, L * ((L + 127) / 128)) * (dp / DP);
}
template <int NH>
void pair_bias_bwd_wide1_t(const at::Tensor& z, const at::Tensor& dbias, const at::Tensor& w, const at::Tensor& stats, const at::Tensor& bu,
                           at::Tensor& dz, at::Tensor& acc, int64_t L, const Geo& G) {
  const int dp = G.dp, NC = dp / DP;
  const cuuint64_t zd[2] = {(cuuint64_t)dp, (cuuint64_t)(L * L)}, zs[1] = {(cuuint64_t)dp * 2};
  const cuuint32_t zb[2] = {64, 128};
  const CUtensorMap mz = tensor_map(z.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, zd, zs, zb, CU_TENSOR_MAP_SWIZZLE_128B);
  const cuuint64_t dd[3] = {(cuuint64_t)L, (cuuint64_t)L, NH}, ds[2] = {(cuuint64_t)L * 4, (cuuint64_t)(L * L * 4)};
  const cuuint32_t db[3] = {128, 1, NH};
  const CUtensorMap md = tensor_map(dbias.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, dd, ds, db, CU_TENSOR_MAP_SWIZZLE_NONE);
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bias_bwd_wide1_k<NH>, cudaFuncAttributeMaxDynamicSharedMemorySize, bwd_smem<NH>()));
    attr = true;
  }
  const long ntiles = L * ((L + 127) / 128);
  float* awf = P<float>(acc) + 2 * G.d + width(G);
  const long per = wide1_per(bwd_bps<NH>(), nsm_of(z), NC, ntiles);                           // blocks per chunk
  pair_bias_bwd_wide1_k<NH><<<(unsigned)(per * NC), 256, bwd_smem<NH>(), S()>>>(mz, md, CP<bf>(w), CP<float2>(stats), CP<float>(bu), P<bf>(dz),
      awf, awf + NH * dp, (int)L, dp, acc.dim() == 2 ? (int)acc.size(1) : 0);
}
// pair_bias_bwd_wide (same arguments, same results) in one launch
void pair_bias_bwd_wide1(at::Tensor z, at::Tensor dbias, at::Tensor w, at::Tensor stats, at::Tensor unmasked, at::Tensor dz, at::Tensor acc,
                         int64_t L, int64_t d) {
  chk(z, at::kBFloat16, "pair"); chk(dbias, at::kFloat, "dbias"); chk(w, at::kBFloat16, "wf"); chk(dz, at::kBFloat16, "dpair");
  chk(stats, at::kFloat, "stats"); chk(unmasked, at::kFloat, "unmasked");
  const int64_t nh = w.size(0), dp = w.size(1);
  TORCH_CHECK(dp > DP && dp % DP == 0, "pair_bias_bwd_wide1: d_pair 256 / 384 / 512");
  const Geo G = geo(nh, d, dp);
  chk_acc(acc, G);
  TORCH_CHECK(L % 16 == 0 && z.numel() == L * L * dp && dz.numel() == L * L * dp && dbias.numel() == nh * L * L && stats.numel() == 2 * L * L
              && unmasked.numel() == nh * L * L, "pair_bias_bwd_wide1 shapes");
  TORCH_CHECK(nh == 8 || nh == 16, "pair_bias_bwd_wide1: 8 or 16 heads");
  TORCH_CHECK(acc.dim() == 1 || acc.size(0) >= pair_bwd_wide1_blocks(z, nh, L, dp), "pair_bias_bwd_wide1: a partial row per block");
  const at::cuda::CUDAGuard gd(z.device());
  if (nh == 8) pair_bias_bwd_wide1_t<8>(z, dbias, w, stats, unmasked, dz, acc, L, G);
  else pair_bias_bwd_wide1_t<16>(z, dbias, w, stats, unmasked, dz, acc, L, G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------------ key-split combine (B200)
// (from origin/b200/ladder-final apb_rows.cu, 2026-10-20; the cores built -DKSPLIT=1: sm100/attn_inf.cu)
// Key-split combine (the sm_100a forward cores built -DKSPLIT): row r of the [A L] rows (sample r / L, row r % L), head h. The NS
// chunks' normalised partial o_c [NS, M, W] and partial LSE_c [NS, A, NH, L] (log2 units) merge in chunk order as
// o = sum_c w_c o_c, w_c = 2^(LSE_c - max) / sum 2^(LSE_c' - max); LSE = max + log2(sum). A thread per four columns of a row
// (its head's weights from the NS LSE values, which every thread of the head reads: L1 hits). Templated on NS so the NS partial
// loads are unconditional and issue together (a runtime chunk count left them one latency each). Inference: sigmoid(g) o as bf16 over the q columns of qkvg; training: O (fp32), LSE and og = sigmoid(g) o (bf16),
// the gate_rows pass folded in.
constexpr int NSMAX = 16, CNT = 256;
template <int NS, bool TRAIN>
__global__ void __launch_bounds__(CNT) ksplit_combine_k(const float* __restrict__ OP, const float* __restrict__ LP, bf* __restrict__ QKVG,
    float* __restrict__ O, float* __restrict__ LSE, bf* __restrict__ OG, int M, int L, int W, int NH, int DHP) {
  APB_PDL_ENTRY();
  const long t = (long)blockIdx.x * CNT + threadIdx.x, q4 = W / 4;
  if (t >= (long)M * q4) return;
  const long r = t / q4;
  const int c = (int)(t - r * q4) * 4, h = c / DHP, A = M / L;
  const long a = r / L, i = r % L;
  float lse[NS], mx = -INFINITY;
  float4 p[NS];
#pragma unroll
  for (int k = 0; k < NS; ++k) {
    lse[k] = LP[(((long)k * A + a) * NH + h) * L + i];
    p[k] = *reinterpret_cast<const float4*>(OP + ((long)k * M + r) * W + c);
  }
#pragma unroll
  for (int k = 0; k < NS; ++k) mx = fmaxf(mx, lse[k]);
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < NS; ++k) { lse[k] = exp2f(lse[k] - mx); sum += lse[k]; }
  const float inv = 1.f / sum;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int k = 0; k < NS; ++k) {
    const float w = lse[k] * inv;
    acc.x = fmaf(w, p[k].x, acc.x); acc.y = fmaf(w, p[k].y, acc.y); acc.z = fmaf(w, p[k].z, acc.z); acc.w = fmaf(w, p[k].w, acc.w);
  }
  const float4 g = mul4(sig4(V4<bf>::load(QKVG + r * 4 * W + 3 * W + c)), acc);
  if (TRAIN) {
    *reinterpret_cast<float4*>(O + r * W + c) = acc;
    V4<bf>::store(OG + r * W + c, g);
    if (c % DHP == 0) LSE[(a * NH + h) * L + i] = mx + log2f(sum);
  } else {
    V4<bf>::store(QKVG + r * 4 * W + c, g);
  }
}

// op [NS, M, W] fp32, lp [NS, A, NH, L] fp32 -> inference (o, lse, og absent): sigmoid(g) o bf16 over qkvg[:, :W]; training: o
// [M, W] fp32, lse [A, NH, L] fp32, og [M, W] bf16.
void ksplit_combine(at::Tensor op, at::Tensor lp, at::Tensor qkvg, c10::optional<at::Tensor> o, c10::optional<at::Tensor> lse,
                    c10::optional<at::Tensor> og, int64_t L, int64_t nh, int64_t dhp) {
  chk(op, at::kFloat, "op"); chk(lp, at::kFloat, "lp"); chk(qkvg, at::kBFloat16, "qkvg");
  TORCH_CHECK(op.dim() == 3, "op [NS, M, W]");
  const int64_t NS = op.size(0), M = op.size(1), W = op.size(2);
  TORCH_CHECK(NS >= 1 && NS <= NSMAX && M % L == 0 && lp.numel() == NS * M * nh && qkvg.size(0) == M && qkvg.size(1) == 4 * W,
              "ksplit_combine shapes");
  TORCH_CHECK(W == nh * dhp && dhp % 4 == 0, "ksplit_combine widths");
  const bool train = o.has_value();
  TORCH_CHECK(train == lse.has_value() && train == og.has_value(), "ksplit_combine: o, lse and og together");
  if (train) { chk(*o, at::kFloat, "o"); chk(*lse, at::kFloat, "lse"); chk(*og, at::kBFloat16, "og"); }
  const at::cuda::CUDAGuard gd(op.device());
  const unsigned grid = (unsigned)((M * (W / 4) + CNT - 1) / CNT);
  float* po = train ? P<float>(*o) : nullptr;
  float* pl = train ? P<float>(*lse) : nullptr;
  bf* pg = train ? P<bf>(*og) : nullptr;
  auto run = [&](auto ns) {
    constexpr int N = decltype(ns)::value;
    auto k = train ? ksplit_combine_k<N, true> : ksplit_combine_k<N, false>;
    pdl_go(k, grid, CNT, 0, CP<float>(op), CP<float>(lp), P<bf>(qkvg), po, pl, pg, (int)M, (int)L, (int)W, (int)nh, (int)dhp);
  };
  switch (NS) {
    case 1: run(std::integral_constant<int, 1>{}); break;
    case 2: run(std::integral_constant<int, 2>{}); break;
    case 3: run(std::integral_constant<int, 3>{}); break;
    case 4: run(std::integral_constant<int, 4>{}); break;
    case 5: run(std::integral_constant<int, 5>{}); break;
    case 6: run(std::integral_constant<int, 6>{}); break;
    case 7: run(std::integral_constant<int, 7>{}); break;
    case 8: run(std::integral_constant<int, 8>{}); break;
    case 9: run(std::integral_constant<int, 9>{}); break;
    case 10: run(std::integral_constant<int, 10>{}); break;
    case 11: run(std::integral_constant<int, 11>{}); break;
    case 12: run(std::integral_constant<int, 12>{}); break;
    case 13: run(std::integral_constant<int, 13>{}); break;
    case 14: run(std::integral_constant<int, 14>{}); break;
    case 15: run(std::integral_constant<int, 15>{}); break;
    default: run(std::integral_constant<int, 16>{}); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ksplit_combine32 (2026-11-17, part 7): the fp32 twin of the inference combine, for sm100/attn_inf_tf32.cu -DKSPLIT=1 (one sample): the NS
// chunks' partial o [NS, L, W] and LSE [NS, NH, L] merge in chunk order (the weights as ksplit_combine_k's), then sigmoid(g) o is written as
// fp32 over q (q at column 0, g at column goff of the [L, ld] fp32 q | k | g rows). A thread per four columns of a row.
template <int NS>
__global__ void __launch_bounds__(CNT) ksplit_combine32_k(const float* __restrict__ OP, const float* __restrict__ LP, float* __restrict__ QKG,
    int L, int W, int NH, int DHP, int ld, int goff) {
  const long t = (long)blockIdx.x * CNT + threadIdx.x, q4 = W / 4;
  if (t >= (long)L * q4) return;
  const long i = t / q4;
  const int c = (int)(t - i * q4) * 4, h = c / DHP;
  float lse[NS], mx = -INFINITY;
  float4 p[NS];
#pragma unroll
  for (int k = 0; k < NS; ++k) {
    lse[k] = LP[((long)k * NH + h) * L + i];
    p[k] = *reinterpret_cast<const float4*>(OP + ((long)k * L + i) * W + c);
  }
#pragma unroll
  for (int k = 0; k < NS; ++k) mx = fmaxf(mx, lse[k]);
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < NS; ++k) { lse[k] = exp2f(lse[k] - mx); sum += lse[k]; }
  const float inv = 1.f / sum;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int k = 0; k < NS; ++k) {
    const float w = lse[k] * inv;
    acc.x = fmaf(w, p[k].x, acc.x); acc.y = fmaf(w, p[k].y, acc.y); acc.z = fmaf(w, p[k].z, acc.z); acc.w = fmaf(w, p[k].w, acc.w);
  }
  const float4 g = *reinterpret_cast<const float4*>(QKG + i * ld + goff + c);
  auto sg = [](float x) { return 1.f / (1.f + __expf(-x)); };
  *reinterpret_cast<float4*>(QKG + i * ld + c) = make_float4(sg(g.x) * acc.x, sg(g.y) * acc.y, sg(g.z) * acc.z, sg(g.w) * acc.w);
}
void ksplit_combine32(at::Tensor op, at::Tensor lp, at::Tensor qkg, int64_t goff, int64_t nh, int64_t dhp) {
  chk(op, at::kFloat, "op"); chk(lp, at::kFloat, "lp");
  TORCH_CHECK(qkg.is_cuda() && qkg.scalar_type() == at::kFloat && qkg.stride(1) == 1 && qkg.stride(0) % 4 == 0, "ksplit_combine32: qkg");
  TORCH_CHECK(op.dim() == 3, "op [NS, L, W]");
  const int64_t NS = op.size(0), L = op.size(1), W = op.size(2);
  TORCH_CHECK(NS >= 1 && NS <= NSMAX && lp.numel() == NS * nh * L && qkg.size(0) >= L && W == nh * dhp && dhp % 4 == 0 && goff % 4 == 0,
              "ksplit_combine32 shapes");
  const at::cuda::CUDAGuard gd(op.device());
  const unsigned grid = (unsigned)((L * (W / 4) + CNT - 1) / CNT);
  auto run = [&](auto ns) {
    constexpr int N = decltype(ns)::value;
    ksplit_combine32_k<N><<<grid, CNT, 0, S()>>>(CP<float>(op), CP<float>(lp), P<float>(qkg), (int)L, (int)W, (int)nh, (int)dhp,
                                                 (int)qkg.stride(0), (int)goff);
  };
  switch (NS) {
    case 1: run(std::integral_constant<int, 1>{}); break;
    case 2: run(std::integral_constant<int, 2>{}); break;
    case 3: run(std::integral_constant<int, 3>{}); break;
    case 4: run(std::integral_constant<int, 4>{}); break;
    case 5: run(std::integral_constant<int, 5>{}); break;
    case 6: run(std::integral_constant<int, 6>{}); break;
    case 8: run(std::integral_constant<int, 8>{}); break;
    case 10: run(std::integral_constant<int, 10>{}); break;
    case 12: run(std::integral_constant<int, 12>{}); break;
    default: TORCH_CHECK(NS == 16, "ksplit_combine32: NS 1-6, 8, 10, 12 or 16"); run(std::integral_constant<int, 16>{}); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------------ bias planes fix-ups, one launch (B200)
// planes_fix: DST [H, LP, LP] = SRC [H, L, L] (+ scale * EX), `neg` on the pad keys (>= L) of the real rows, 0 in the pad query rows.
// EX [HE, L, L] (HE = 1: shared by the heads; fp32 or bf16), natural key order; ``perm``: SRC / DST hold the TF32 core's key order (position
// p of a group of 8 = key p - r + 4 (r % 2) + r / 2, r = p % 8), so EX is read at that key. LP == L: in place (DST == SRC).
// Replaces, for extra_pair_bias, the cast + (fp32: sort / gather) + add launches, and for L % 128 != 0 the padded copy of the pair.
template <typename T, typename E>
__global__ void __launch_bounds__(256) planes_fix_k(const T* SRC, T* DST, const E* EX, int HE, int L, int LP, float scale, float neg, int perm) {
  const int i = blockIdx.x, h = blockIdx.y;
  T* d = DST + ((long)h * LP + i) * LP;
  if (i >= L) {
    for (int p = threadIdx.x; p < LP; p += 256) d[p] = T(0.f);
    return;
  }
  const T* s = SRC + ((long)h * L + i) * L;
  const E* e = EX ? EX + ((long)(HE > 1 ? h : 0) * L + i) * L : nullptr;
  for (int p = threadIdx.x; p < LP; p += 256) {
    if (p >= L) { d[p] = T(neg); continue; }
    float v = (float)s[p];
    if (e) { const int r = p & 7, key = perm ? p - r + 4 * (r & 1) + (r >> 1) : p; v += scale * (float)e[key]; }
    d[p] = T(v);
  }
}
void planes_fix(at::Tensor src, at::Tensor dst, c10::optional<at::Tensor> extra, int64_t L, int64_t LP, double scale, double neg, bool perm) {
  TORCH_CHECK(src.dim() == 3 && dst.dim() == 3 && src.is_contiguous() && dst.is_contiguous() && src.scalar_type() == dst.scalar_type()
              && src.size(1) == L && src.size(2) == L && dst.size(0) == src.size(0) && dst.size(1) == LP && dst.size(2) == LP && LP >= L
              && (LP > L || src.data_ptr() == dst.data_ptr()), "planes_fix shapes");
  const int64_t H = src.size(0);
  int64_t HE = 0;
  if (extra) {
    TORCH_CHECK(extra->dim() == 3 && extra->is_contiguous() && extra->size(1) == L && extra->size(2) == L
                && (extra->size(0) == 1 || extra->size(0) == H), "planes_fix extra [1 | H, L, L]");
    HE = extra->size(0);
  }
  const at::cuda::CUDAGuard gd(src.device());
  const dim3 grid((unsigned)LP, (unsigned)H);
  const bool tb = src.scalar_type() == at::kBFloat16, eb = extra && extra->scalar_type() == at::kBFloat16;
  TORCH_CHECK((tb || src.scalar_type() == at::kFloat) && (!extra || eb || extra->scalar_type() == at::kFloat), "planes_fix dtypes");
  const void* ex = extra ? extra->data_ptr() : nullptr;
#define PF_GO(T, E) planes_fix_k<T, E><<<grid, 256, 0, S()>>>(CP<T>(src), P<T>(dst), (const E*)ex, (int)HE, (int)L, (int)LP, (float)scale, (float)neg, (int)perm)
  if (tb) { if (eb) PF_GO(bf, bf); else PF_GO(bf, float); }
  else { if (eb) PF_GO(float, bf); else PF_GO(float, float); }
#undef PF_GO
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------------ backward tail (2026-11-17, part 5)
// finalize_red: acc_reduce_k + finalize_k in ONE launch (the H100 runs them as two; on the B200 they were the last two launches of every
// training step, ~3 us of launch latency each). Blocks [0, ncopy): finalize's copies (dW q | k | v | g real rows, dWo real columns), 4 values
// a thread. [ncopy, ncopy + nra): the row-kernel columns [0, 2 D + W) (dlnw | dlnb | dbq), 32 a block, summed over rows_a in acc_reduce_k's
// order (warp w: rows w, w + 16, ...; warp 0 adds the 16 sums in warp order). The rest: 8 pair channels c a block, dWf[h, c] for every head
// (64 row groups g: rows g, g + 64, ..., the upper 32 groups added onto the lower, then the 32 in order), dWb = dWf ln_pair.weight,
// dln_pair.weight[c] = sum_h dWf[h, c] Wb[h, c] in head order. Deterministic; every output written by one thread.
constexpr int FR_T = 512, FR_G = FR_T / 8;
template <int NH>
__global__ void __launch_bounds__(FR_T) finalize_red_k(const float* __restrict__ DWP, const float* __restrict__ DWO, const float* __restrict__ ACC,
    int width, int rows_a, int rows_b, const bf* __restrict__ WB, const float* __restrict__ LPW, float* __restrict__ OQ, float* __restrict__ OKe,
    float* __restrict__ OV, float* __restrict__ OG, float* __restrict__ OO, float* __restrict__ OLW, float* __restrict__ OLB,
    float* __restrict__ OBQ, float* __restrict__ OWB, float* __restrict__ OLPW, Geo G, int ncopy, int nra, int boff) {
  APB_PDL_ENTRY();
  constexpr int HG = FR_G / 2;
  __shared__ __align__(16) float red[NH * HG * 8];
  __shared__ float fin[NH * 8];
  static_assert(NH * HG * 8 >= FR_G * 8, "red holds the row-column sums");
  const int W = G.nh * G.dhp, D = G.d, nwf = G.dp / 8;
  int bid = (int)blockIdx.x + boff;  // v27: boff > 0 launches a later part of the grid alone
  // grid order: the dWf blocks, the row-column blocks, then the copies (the reduction chains start first)
  bid = bid < nwf ? ncopy + nra + bid : bid < nwf + nra ? ncopy + (bid - nwf) : bid - nwf - nra;
  const int tid = threadIdx.x;
  if (bid < ncopy) {
    const long nv = 5L * D * D / 4, i = (long)bid * FR_T + tid;
    if (i >= nv) return;
    const long e = i * 4, blk = e / (D * D), off = e % (D * D), r = off / D, c = off % D;
    if (blk < 4) {
      const long prow = (long)blk * W + (r / G.dh) * G.dhp + r % G.dh;
      float* dst = (blk == 0 ? OQ : blk == 1 ? OKe : blk == 2 ? OV : OG) + off;
      V4<float>::store(dst, V4<float>::load(DWP + prow * D + c));
    } else {
      V4<float>::store(OO + off, V4<float>::load(DWO + r * W + (c / G.dh) * G.dhp + c % G.dh));
    }
    return;
  }
  bid -= ncopy;
  if (bid < nra) {                                                          // 8 columns, 64 row groups: rows g, g + 64, ... (loads issued together)
    const int c = bid * 8 + (tid & 7), g = tid >> 3, split = 2 * D + W;
    float value = 0.f;
    if (c < split) {
#pragma unroll 4
      for (int row = g; row < rows_a; row += FR_G) value += ACC[(long)row * width + c];
    }
    red[g * 8 + (tid & 7)] = value;
    __syncthreads();
    if (tid < 8 && c < split) {
      value = red[tid];
      for (int k = 1; k < FR_G; ++k) value += red[k * 8 + tid];
      if (c < D) OLW[c] = value;
      else if (c < 2 * D) OLB[c - D] = value;
      else {
        const int rc = real_of(G, c - 2 * D);
        if (rc >= 0) OBQ[rc] = value;
      }
    }
    return;
  }
  bid -= nra;
  const int c0 = bid * 8, cc = tid & 7, g = tid >> 3;                       // g: row group 0 .. 63
  const float* base = ACC + 2 * D + W + c0 + cc;
  float s[NH];
#pragma unroll
  for (int h = 0; h < NH; ++h) s[h] = 0.f;
#pragma unroll 3
  for (int row = g; row < rows_b; row += FR_G) {
    float v[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) v[h] = base[(long)row * width + h * G.dp];
#pragma unroll
    for (int h = 0; h < NH; ++h) s[h] += v[h];
  }
  if (g >= HG) {
#pragma unroll
    for (int h = 0; h < NH; ++h) red[(h * HG + g - HG) * 8 + cc] = s[h];
  }
  __syncthreads();
  if (g < HG) {
#pragma unroll
    for (int h = 0; h < NH; ++h) red[(h * HG + g) * 8 + cc] = s[h] + red[(h * HG + g) * 8 + cc];
  }
  __syncthreads();
  if (tid < NH * 8) {
    const int h = tid >> 3;
    float t = red[(h * HG) * 8 + cc];
    for (int k = 1; k < HG; ++k) t += red[(h * HG + k) * 8 + cc];
    fin[h * 8 + cc] = t;
    OWB[h * G.dp + c0 + cc] = t * LPW[c0 + cc];
  }
  __syncthreads();
  if (tid < 8) {
    float t = 0.f;
    for (int h = 0; h < NH; ++h) t += fin[h * 8 + tid] * __bfloat162float(WB[h * G.dp + c0 + tid]);
    OLPW[c0 + tid] = t;
  }
}
void finalize_red_impl(at::Tensor dwp, at::Tensor dwo, at::Tensor acc, at::Tensor wb, at::Tensor lpw, std::vector<at::Tensor> out, int64_t rows_a,
                       int64_t rows_b, bool copy, int parts = 7) {
  chk(dwp, at::kFloat, "dwp"); chk(dwo, at::kFloat, "dwo"); chk(wb, at::kBFloat16, "wb"); chk(lpw, at::kFloat, "lpw");
  const Geo G = geo(wb.size(0), dwp.size(1), wb.size(1));
  const int64_t W = width(G), D = G.d;
  chk_acc(acc, G);
  TORCH_CHECK(acc.dim() == 2, "finalize_red: block partial rows [rows, n]");
  TORCH_CHECK(rows_a >= 1 && rows_b >= 1 && rows_a <= acc.size(0) && rows_b <= acc.size(0), "finalize_red: partial row counts");
  TORCH_CHECK(dwp.numel() == 4 * W * D && dwo.numel() == D * W && lpw.numel() == G.dp, "finalize_red: dwp [4 W, D], dwo [D, W], lpw [d_pair]");
  TORCH_CHECK(out.size() == 10, "finalize_red: 10 outputs");
  for (int i = 0; i < 10; ++i) chk(out[i], at::kFloat, "finalize_red output");
  const at::cuda::CUDAGuard gd(dwp.device());
  const int ncopy = copy ? (int)((5L * D * D / 4 + FR_T - 1) / FR_T) : 0, nra = (int)((2 * D + W + 7) / 8), nwf = G.dp / 8;
  // v27: parts (bits) 4 dWf, 2 the row columns, 1 the copies -- a contiguous range of the grid [dWf | rows | copies]
  const int lo = (parts & 4) ? 0 : (parts & 2) ? nwf : nwf + nra;
  const int hi = (parts & 1) && copy ? nwf + nra + ncopy : (parts & 2) ? nwf + nra : nwf;
  TORCH_CHECK(parts == 7 || parts == 4 || parts == 2 || parts == 1 || parts == 3, "finalize parts");
  if (hi <= lo) return;
#define FR_GO(NH)                                                                                                                       \
  pdl_go(finalize_red_k<NH>, (unsigned)(hi - lo), FR_T, 0, CP<float>(dwp), CP<float>(dwo), CP<float>(acc), (int)acc.size(1), \
      (int)rows_a, (int)rows_b, CP<bf>(wb), CP<float>(lpw), P<float>(out[2]), P<float>(out[4]), P<float>(out[5]), P<float>(out[6]),       \
      P<float>(out[7]), P<float>(out[0]), P<float>(out[1]), P<float>(out[3]), P<float>(out[9]), P<float>(out[8]), G, ncopy, nra, lo)
  switch (G.nh) {
    case 8: FR_GO(8); break;
    case 12: FR_GO(12); break;
    case 16: FR_GO(16); break;
    default: FR_GO(24); break;
  }
#undef FR_GO
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void finalize_red(at::Tensor dwp, at::Tensor dwo, at::Tensor acc, at::Tensor wb, at::Tensor lpw, std::vector<at::Tensor> out, int64_t rows_a,
                  int64_t rows_b) {
  finalize_red_impl(dwp, dwo, acc, wb, lpw, out, rows_a, rows_b, true);
}
// v25 (2026-11-18): finalize_red without its copy blocks -- at W == d_single (no head padding) the dW q | k | v | g rows and dWo ARE the
// parameter gradients (the caller hands out views of the GEMM outputs); out[2], out[4..7] are not written. It reads neither dwp nor dwo, so it
// need not wait for the dW GEMMs (the caller joins their stream after it).
void finalize_red_nc(at::Tensor dwp, at::Tensor dwo, at::Tensor acc, at::Tensor wb, at::Tensor lpw, std::vector<at::Tensor> out, int64_t rows_a,
                     int64_t rows_b) {
  finalize_red_impl(dwp, dwo, acc, wb, lpw, out, rows_a, rows_b, false);
}
// v27: one part of finalize_red (bits: 4 dWf | dWb | dln_pair.weight from the pair backward's rows, 2 dlnw | dlnb | dbq from the row kernels', 1
// the dW copies; copy = false: no copy blocks) -- each launched where its inputs are made (the same blocks and statements: the same bits)
void finalize_part(at::Tensor dwp, at::Tensor dwo, at::Tensor acc, at::Tensor wb, at::Tensor lpw, std::vector<at::Tensor> out, int64_t rows_a,
                   int64_t rows_b, bool copy, int64_t parts) {
  finalize_red_impl(dwp, dwo, acc, wb, lpw, out, rows_a, rows_b, copy, (int)parts);
}

// gate_bwd for 12 x 32 / 24 x 16 (W = 384) with every lane busy: a warp per row, lane l takes columns 4 l + 128 k (k < 3), so a head is
// HL = dh / 4 neighbouring lanes (D summed over them by shuffles). The shared gate_bwd_hl_k gives a head per lane (12 / 24 of 32 lanes,
// 8 / 4 vectors each in a chain). The other geometries go to the shared kernels.
template <int HL>
__global__ void __launch_bounds__(NT) gate_bwd_s_k(const bf* __restrict__ DOG, const float* __restrict__ O, const bf* __restrict__ QKVG,
    bf* __restrict__ DOB, float* __restrict__ DD, bf* __restrict__ DQKVG, int M) {
  constexpr int WD = 384, HC = 4 * HL;
  APB_PDL_ENTRY();
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (r >= M) return;
  const int lane = threadIdx.x & 31;
  float4 dog[3], o[3], s[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int c = 4 * lane + 128 * k;
    dog[k] = V4<bf>::load(DOG + r * WD + c); o[k] = V4<float>::load(O + r * WD + c); s[k] = V4<bf>::load(QKVG + r * 4 * WD + 3 * WD + c);
  }
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int c = 4 * lane + 128 * k;
    const float4 sg = sig4(s[k]), d_o = mul4(dog[k], sg), t = mul4(dog[k], o[k]);
    const float4 dg = make_float4(t.x * sg.x * (1.f - sg.x), t.y * sg.y * (1.f - sg.y), t.z * sg.z * (1.f - sg.z), t.w * sg.w * (1.f - sg.w));
    V4<bf>::store(DOB + r * WD + c, d_o);
    V4<bf>::store(DQKVG + r * 4 * WD + 3 * WD + c, dg);
    float ds = sum4(mul4(d_o, o[k]));
#pragma unroll
    for (int x = 1; x < HL; x <<= 1) ds += __shfl_xor_sync(0xffffffffu, ds, x);
    if (lane % HL == 0) DD[(long)(c / HC) * M + r] = ds;
  }
}
void gate_bwd2(at::Tensor dog, at::Tensor o, at::Tensor qkvg, at::Tensor dob, at::Tensor dd, at::Tensor dqkvg) {
  const int64_t M = o.size(0), W = o.size(1), nh = dd.size(0);
  if (W != 384 || (nh != 12 && nh != 24)) return gate_bwd_p(dog, o, qkvg, dob, dd, dqkvg);
  chk(dog, at::kBFloat16, "dog"); chk(o, at::kFloat, "o"); chk(dob, at::kBFloat16, "dob"); chk(dd, at::kFloat, "dd");
  chk(dqkvg, at::kBFloat16, "dqkvg"); chk(qkvg, at::kBFloat16, "qkvg");
  TORCH_CHECK(qkvg.size(1) == 4 * W && dqkvg.size(1) == 4 * W && dd.size(1) == M, "gate_bwd2 widths");
  const at::cuda::CUDAGuard gd(o.device());
  pdl_go(nh == 12 ? gate_bwd_s_k<8> : gate_bwd_s_k<4>, blocks(M), NT, 0, CP<bf>(dog), CP<float>(o), CP<bf>(qkvg), P<bf>(dob), P<float>(dd), P<bf>(dqkvg), (int)M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------------ fused inference front (2026-11-17, part 6)
// inf_front_k: LayerNorm(single) + the q | k | v | g GEMM + the pair -> bias pass of the bf16 inference step in ONE launch, with roles (the
// three are independent: the GEMM reads the packed weights, the pair pass Wf; neither reads the other's output). Blocks [0, G): GEMM tiles
// of S1_BM rows x S1_BN qkvg columns -- the tile's weight rows go to shared memory by bulk copies (one per row: one round trip), the
// tile's rows are normalised meanwhile with ln_rows_k's row math (a warp per row, its statements: the same xa bits) into shared memory,
// then mma.sync m16n8k16 (warp w: columns 8 w .. 8 w + 7 of the tile, both 16-row groups), fp32 sums + the packed bias, bf16 out (the
// cuBLAS addmm's form). Column tile 0 also writes the residual seed y = x. Blocks [G, grid): pair_bias_fwd_k's pass (its statements;
// NB = 1, no stats / unmasked planes) over the grid - G blocks. The GEMM blocks come first in the grid, so they are dispatched with the
// first pair blocks instead of queueing behind the persistent pass. The step was LN, then [pair beside GEMM], each a launch.
constexpr int S1_BM = 32, S1_BN = 64, S1_NT = 256;
// FR_TL (probe builds only, -DFR_TL=1): per-block stage stamps (globaltimer) of inf_front_k into g_frtl; 0 in every product build
#ifndef FR_TL
#define FR_TL 0
#endif
#if FR_TL
__device__ unsigned long long g_frtl[1024][8];
#define FRT(i, v) do { if (threadIdx.x == 0 && blockIdx.x < 1024) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_frtl[blockIdx.x][i] = (v) ? (unsigned long long)(v) : t_; } } while (0)
#else
#define FRT(i, v)
#endif
template <int V> constexpr int s1_ld() { return 128 * V + 8; }  // padded row (conflict-free fragment reads)
// TMW (v24) layout: [1024-B alignment pad][B: S1_BN weight rows as D / 64 chunks of 64 columns, 128-B swizzled][A rows][mbarrier]; the
// legacy layout [A][B padded rows][mbarrier] fits in the same bytes
template <int V> constexpr int s1_gemm_smem() { return (S1_BM + S1_BN) * s1_ld<V>() * 2 + 16; }  // + the copies' mbarrier
static_assert(1024 + S1_BN * 128 * 3 * 2 + S1_BM * s1_ld<3>() * 2 + 16 <= s1_gemm_smem<3>() && 1024 + S1_BN * 128 * 4 * 2 + S1_BM * s1_ld<4>() * 2 + 16 <= s1_gemm_smem<4>(), "TMW layout");
int g_front_tmw = 1;  // inf_front: the weight rows of a GEMM tile as ONE 3-D tensor-map box (1) or one bulk copy per row (0)
template <int NH, int DPW, int V> constexpr int s1_smem() {
  return fwd_smem<NH, DPW>() > s1_gemm_smem<V>() ? fwd_smem<NH, DPW>() : s1_gemm_smem<V>();
}
// Rows by bulk copies (one per row, the lanes of the calling warp share them; 16-B granular cp.async kept few bytes in flight per SM:
// 72 KB took ~5 us), all on one mbarrier. Lane 0 posts the expected bytes first. Rows past `live` are not copied (the caller zeroes them).
__device__ __forceinline__ void bulk_rows(void* dst, int dst_ld, const void* src, long src_ld, int row_bytes, int nrows, int live, uint64_t* bar) {
  const int lane = threadIdx.x & 31;
  if (lane == 0) mbar_expect_tx(bar, (uint32_t)(min(nrows, live) * row_bytes));
  __syncwarp();
  for (int r = lane; r < nrows && r < live; r += 32)
    bulk_g2s(static_cast<uint8_t*>(dst) + (long)r * dst_ld, static_cast<const uint8_t*>(src) + r * src_ld, (uint32_t)row_bytes, bar);
}
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// the GEMM tile's mbarrier (either layout)
template <int V> __device__ __forceinline__ uint8_t* s1_bar(uint8_t* smem, bool tmw) {
  constexpr int D = 128 * V, LD = s1_ld<V>();
  const uint32_t pad = (1024u - (su32(smem) & 1023u)) & 1023u;
  return tmw ? smem + pad + (S1_BN * D + S1_BM * LD) * 2 : smem + (S1_BM + S1_BN) * LD * 2;
}
template <int V>
__device__ __forceinline__ void s1_gemm(int tile, const bf* __restrict__ X, const float* __restrict__ LW, const float* __restrict__ LB,
    const bf* __restrict__ WP, const bf* __restrict__ BV, bf* __restrict__ Y, bf* __restrict__ QKVG, int M, int NC, float eps, uint8_t* smem,
    const CUtensorMap* tmw) {
  constexpr int D = 128 * V, LD = s1_ld<V>();
  const int ntn = NC / S1_BN, mt = tile / ntn, nt = tile - mt * ntn, r0 = mt * S1_BM, n0 = nt * S1_BN;
  // TMW: B at the first 1024-B boundary (the swizzle's atom), A after it; else the legacy layout
  const uint32_t pad = tmw ? ((1024u - (su32(smem) & 1023u)) & 1023u) : 0u;
  bf* Bs = reinterpret_cast<bf*>(smem + pad);
  bf* As = tmw ? Bs + S1_BN * D : reinterpret_cast<bf*>(smem);
  if (!tmw) Bs = As + S1_BM * LD;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint64_t* bar = reinterpret_cast<uint64_t*>(s1_bar<V>(smem, tmw != nullptr));
  // v24 (2026-11-18): every first-touch load is issued before anything waits (stamps after an L2 flush: a bulk-copy issue holds its
  // thread ~1 us, so the rows' loads issued after warp 0's copies started 2 us late): the rows, LayerNorm's weights and the epilogue's
  // bias into registers first, then the weight rows by bulk copies, one per lane over the eight warps (S1_BN = 64 rows).
  FRT(0, 0);
  constexpr int RPW = S1_BM / (S1_NT / 32);  // rows per warp (4)
  uint2 xr[RPW][V];  // the rows' raw bf16 words, every load issued before any use
  const int cl = 4 * V * lane;
#pragma unroll
  for (int rr = 0; rr < RPW; ++rr) {
    const long r = r0 + warp * RPW + rr;
#pragma unroll
    for (int k = 0; k < V; ++k) xr[rr][k] = r < M ? *reinterpret_cast<const uint2*>(X + r * D + cl + 4 * k) : make_uint2(0u, 0u);
  }
  float4 w[V], b[V];
  ldv<V>(LW, w); ldv<V>(LB, b);
  const uint32_t bvw = __ldcg(reinterpret_cast<const unsigned int*>(BV + n0 + 8 * warp + 2 * (lane & 3)));
  static_assert(S1_BN == 8 * (S1_NT / 32), "s1_gemm: one weight row per lane 0..7 of each warp");
  if (threadIdx.x == 0) { mbar_init(bar, 1); mbar_fence_init(); mbar_expect_tx(bar, (uint32_t)(S1_BN * D * 2)); }
  __syncthreads();
  if (tmw) {  // v24: ONE box of S1_BN rows x D columns (D / 64 chunks of 64, 128-B swizzle): a block issued 64 row copies, which queued
    if (threadIdx.x == 0) tma3d(su32(Bs), tmw, 0, n0, 0, bar);  // behind the pair tiles' loads at the same SM
  } else if (lane < 8) {
    bulk_g2s(Bs + (8 * warp + lane) * LD, WP + (long)(n0 + 8 * warp + lane) * D, (uint32_t)(D * 2), bar);
  }
  FRT(1, 0);
#pragma unroll
  for (int rr = 0; rr < RPW; ++rr) {  // ln_rows_k's statements
    const int lr = warp * RPW + rr;
    const long r = r0 + lr;
    float4 x[V];
#pragma unroll
    for (int k = 0; k < V; ++k) {
      const float2 lo = unpack2(xr[rr][k].x), hi = unpack2(xr[rr][k].y);
      x[k] = make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    if (Y && nt == 0 && r < M) {
#pragma unroll
      for (int k = 0; k < V; ++k) *reinterpret_cast<uint2*>(Y + r * D + cl + 4 * k) = xr[rr][k];
    }
    const float mean = warp_sum(sumv<V>(x)) * (1.f / D);
#pragma unroll
    for (int k = 0; k < V; ++k) x[k] = make_float4(x[k].x - mean, x[k].y - mean, x[k].z - mean, x[k].w - mean);
    const float rstd = rsqrtf(warp_sum(dotv<V>(x, x)) * (1.f / D) + eps);
#pragma unroll
    for (int k = 0; k < V; ++k)
      x[k] = make_float4(x[k].x * rstd * w[k].x + b[k].x, x[k].y * rstd * w[k].y + b[k].y, x[k].z * rstd * w[k].z + b[k].z,
                         x[k].w * rstd * w[k].w + b[k].w);
    stv<V>(As + lr * LD, x);
  }
  __syncthreads();  // the normalised rows are in shared memory
  FRT(2, 0);
  mbar_wait(bar, 0);  // the weight rows have landed
  FRT(3, 0);
  const int g = lane >> 2, q = lane & 3;
  float c[2][4] = {};
  if (tmw) {  // v24: the fragments by ldmatrix (the same words per lane as the scalar reads: the same products and bits). The scalar
    // form issued 10 shared loads per k step per warp (every warp reads all 32 A rows). Measured level (the tile's mma phase stays
    // ~1.3 us; two accumulator chains per m instead of one measured level as well, not kept).
    // B (row n, column k) sits at chunk k / 64, row n, 16-B unit ((k % 64) / 8) ^ (n % 8) (the box's 128-B swizzle).
    const int mi = lane >> 3, rr = lane & 7, n = 8 * warp + rr;
    const uint32_t bbase = su32(Bs) + (uint32_t)(n * 128), abase = su32(As) + (uint32_t)(((rr + 8 * (mi & 1)) * LD + 8 * (mi >> 1)) * 2);
#pragma unroll 4
    for (int k0 = 0; k0 < D; k0 += 32) {
      uint32_t bq[4];
      const uint32_t u = (uint32_t)(((k0 & 63) >> 3) + mi);
      ldsm4(bbase + (uint32_t)((k0 >> 6) * (S1_BN * 128)) + ((u ^ (uint32_t)rr) << 4), bq);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          uint32_t a[4];
          ldsm4(abase + (uint32_t)((16 * m * LD + k0 + 16 * h) * 2), a);
          mma16816(c[m], a[0], a[1], a[2], a[3], bq[2 * h], bq[2 * h + 1]);
        }
      }
    }
  } else {
#pragma unroll 8
    for (int k0 = 0; k0 < D; k0 += 16) {
      const uint32_t* bp = reinterpret_cast<const uint32_t*>(Bs + (8 * warp + g) * LD + k0 + 2 * q);
      const uint32_t b0 = bp[0], b1 = bp[4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const bf* ap = As + (16 * m + g) * LD + k0 + 2 * q;
        mma16816(c[m], *reinterpret_cast<const uint32_t*>(ap), *reinterpret_cast<const uint32_t*>(ap + 8 * LD),
                 *reinterpret_cast<const uint32_t*>(ap + 8), *reinterpret_cast<const uint32_t*>(ap + 8 * LD + 8), b0, b1);
      }
    }
  }
  const int col = n0 + 8 * warp + 2 * q;
  const float2 bb = unpack2(bvw);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const long ra = r0 + 16 * m + g, rb = ra + 8;
    if (ra < M) *reinterpret_cast<uint32_t*>(QKVG + ra * NC + col) = pack2(c[m][0] + bb.x, c[m][1] + bb.y);
    if (rb < M) *reinterpret_cast<uint32_t*>(QKVG + rb * NC + col) = pack2(c[m][2] + bb.x, c[m][3] + bb.y);
  }
  FRT(4, 0);
  FRT(7, 1);
}
// wf_frags (apb_rows.cu) in two halves, the same statements: the loads (issued before the pair tiles' bulk copies, v24) and the sums
template <int NT8, int NH, int DPW>
__device__ __forceinline__ void wf_raw(const bf* W, uint32_t (&b)[NT8][DPW / 16][2]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#pragma unroll
  for (int nt = 0; nt < NT8; ++nt) {
#pragma unroll
    for (int s = 0; s < DPW / 16; ++s) {
      const bool live = g + 8 * nt < NH;
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(W + (live ? g + 8 * nt : 0) * DPW + 32 * (s >> 1) + 8 * q + 4 * (s & 1));
      b[nt][s][0] = live ? wp[0] : 0u; b[nt][s][1] = live ? wp[1] : 0u;
    }
  }
}
template <int NT8, int DPW>
__device__ __forceinline__ void wf_sums(const uint32_t (&b)[NT8][DPW / 16][2], float (&sw0)[NT8], float (&sw1)[NT8]) {
  const int lane = threadIdx.x & 31, q = lane & 3;
#pragma unroll
  for (int nt = 0; nt < NT8; ++nt) {
    float sw = 0.f;
#pragma unroll
    for (int s = 0; s < DPW / 16; ++s) {
      const float2 u = unpack2(b[nt][s][0]), v = unpack2(b[nt][s][1]);
      sw += u.x + u.y + v.x + v.y;
    }
    sw = quad_sum(sw);
    sw0[nt] = __shfl_sync(0xffffffffu, sw, 8 * q); sw1[nt] = __shfl_sync(0xffffffffu, sw, 8 * q + 4);
  }
}
// pair_bias_fwd_k's pass (statement for statement; NB = 1, no ST / BU) as block `bid` of `nblk`
// v24: the tiles [t0, tend) (tend < 0: all), block `bid` of `nblk` taking t0 + bid, t0 + bid + nblk, ... (inf_fronto_k splits them)
template <int NH, int DPW>
__device__ __forceinline__ void s1_pair(int bid, int nblk, const bf* __restrict__ Z, const bf* __restrict__ W, const bool* __restrict__ MASK,
    bf* __restrict__ OUT, int L, float eps, float neg, uint8_t* smem, int* klen_s, long t0 = 0, long tend = -1) {
  constexpr int NT8 = nhp<NH>() / 8, NHP = nhp<NH>(), NSF = fwd_stages<DPW>(), FT = fwd_tile<DPW>(), UU = DPW / 32;
  constexpr int TR = fwd_rows<DPW>(), NTH = 32 * fwd_warps<DPW>(), TC = TR / 8;
  static_assert(NTH == S1_NT, "s1_pair: the 8-warp pair passes (d_pair 128 / 256)");
  float (*out_s)[NHP][128 + 4] = reinterpret_cast<float (*)[NHP][128 + 4]>(smem + NSF * FT);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + NSF * FT + 2 * NHP * (128 + 4) * 4);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int ntj = (L + TR - 1) / TR;
  const long ntiles = (long)L * ntj;
  const bool skip_dead = MASK != nullptr;
  auto dead_tile = [&](long n, int j0) { return skip_dead && n >= NSF && j0 >= klen_s[0]; };
  const long tlast = tend < 0 ? ntiles : tend;
  auto tile_of = [&](long n) -> long { return t0 + bid + n * nblk; };
  auto issue = [&](long n) {
    const long t = tile_of(n);
    if (t >= tlast) return;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * TR, rows = min(TR, L - j0);
    uint64_t* fb = &full[n % NSF];
    if (dead_tile(n, j0)) { mbar_expect_tx(fb, 0); return; }
    mbar_expect_tx(fb, (uint32_t)rows * DPW * 2);
    bulk_g2s(smem + (n % NSF) * FT, Z + ((long)i * L + j0) * DPW, (uint32_t)rows * DPW * 2, fb);
  };
  FRT(0, 0);
  // v24: Wf's words are loaded first and the ring's first NSF tiles are issued by NSF lanes of warp 0 (one copy each): after an L2
  // flush a bulk-copy issue holds its thread ~1 us, which delayed thread 0's own Wf loads (stamps: Wf ready 3 us after entry)
  uint32_t b[NT8][DPW / 16][2];
  float sw0[NT8], sw1[NT8];
  wf_raw<NT8, NH, DPW>(W, b);
  if (threadIdx.x == 0) {
    for (int k = 0; k < NSF; ++k) mbar_init(&full[k], 1);
    mbar_fence_init();
  }
  __syncthreads();
  if (threadIdx.x < NSF) issue(threadIdx.x);
  if (skip_dead && warp == 0) {
    const int kl = warp_key_len(reinterpret_cast<const uint8_t*>(MASK), L);
    if (lane == 0) klen_s[0] = kl;
  }
  wf_sums<NT8, DPW>(b, sw0, sw1);
#if FR_TL
  if (threadIdx.x == 0) g_frtl[blockIdx.x < 1024 ? blockIdx.x : 0][1] = (unsigned long long)(sw0[0] != 1.2345f ? 0 : 1);
  FRT(1, 0);
#endif
  for (long n = 0;; ++n) {
    const long t = tile_of(n);
    if (t >= tlast) break;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * TR, rows = min(TR, L - j0), st = (int)(n % NSF);
    mbar_wait(&full[st], (uint32_t)((n / NSF) & 1));
    if (n == 0) FRT(2, 0);
    const bool dead = dead_tile(n, j0);
    const uint8_t* tile = smem + st * FT;
    float (*os)[128 + 4] = out_s[n & 1];
    const int rb = warp * 16;
    if (!dead && rb < rows) {
      float c[NT8][4] = {}, g0[4] = {}, g1[4] = {}, cs[4] = {};
      uint4 r0[UU], r1[UU];
#pragma unroll
      for (int u = 0; u < UU; ++u) {
        r0[u] = *reinterpret_cast<const uint4*>(tile + (rb + g) * DPW * 2 + 64 * u + 16 * q);
        r1[u] = *reinterpret_cast<const uint4*>(tile + (rb + g + 8) * DPW * 2 + 64 * u + 16 * q);
      }
#pragma unroll
      for (int u = 0; u < UU; ++u) {
        const uint32_t u0[4] = {r0[u].x, r0[u].y, r0[u].z, r0[u].w}, u1[4] = {r1[u].x, r1[u].y, r1[u].z, r1[u].w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int nt = 0; nt < NT8; ++nt)
            mma16816(c[nt], u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1], b[nt][2 * u + h][0], b[nt][2 * u + h][1]);
          gram_step(g0, g1, cs, u0[2 * h], u1[2 * h], u0[2 * h + 1], u1[2 * h + 1]);
        }
      }
      float m0, rs0, m1, rs1;
      gram_stats<DPW>(g0, g1, cs, eps, m0, rs0, m1, rs1);
      const int jl = rb + g;
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
        const int h0 = 2 * q + 8 * nt;
        os[h0][jl] = rs0 * (c[nt][0] - m0 * sw0[nt]); os[h0 + 1][jl] = rs0 * (c[nt][1] - m0 * sw1[nt]);
        os[h0][jl + 8] = rs1 * (c[nt][2] - m1 * sw0[nt]); os[h0 + 1][jl + 8] = rs1 * (c[nt][3] - m1 * sw1[nt]);
      }
    }
    __syncthreads();
    if (n == 0) FRT(3, 0);
    if (threadIdx.x == 0) issue(n + NSF);
    for (int task = threadIdx.x; task < NH * TC; task += NTH) {
      const int h = task / TC, ch = task % TC, j = j0 + ch * 8;
      if (ch * 8 >= rows) continue;
      if (dead) {
        const uint32_t nn = pack2(neg, neg);
        *reinterpret_cast<uint4*>(OUT + ((long)h * L + i) * L + j) = make_uint4(nn, nn, nn, nn);
        continue;
      }
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float a = os[h][ch * 8 + 2 * e], c1 = os[h][ch * 8 + 2 * e + 1];
        if (MASK) { if (!MASK[j + 2 * e]) a = neg; if (!MASK[j + 2 * e + 1]) c1 = neg; }
        o[e] = pack2(a, c1);
      }
      *reinterpret_cast<uint4*>(OUT + ((long)h * L + i) * L + j) = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
  FRT(4, 0);
  FRT(7, 2);
}
template <int NH, int DPW, int V>
__global__ void __launch_bounds__(S1_NT, DPW == DP ? 2 : 1) inf_front_k(const bf* __restrict__ Z, const bf* __restrict__ WF,
    const bool* __restrict__ MASK, bf* __restrict__ BIAS, int L, float eps_p, float neg, int G, const bf* __restrict__ X,
    const float* __restrict__ LW, const float* __restrict__ LB, const bf* __restrict__ WP, const bf* __restrict__ BV, bf* __restrict__ Y,
    bf* __restrict__ QKVG, int NC, float eps_s, const __grid_constant__ CUtensorMap tmw, int use_tmw) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // the core (a programmatic dependent) may be scheduled now; it waits
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int klen_s[1];
#if FR_TL
  { unsigned s_; asm volatile("mov.u32 %0, %%smid;" : "=r"(s_)); FRT(6, s_ + 1); FRT(5, 0); }
#endif
  if ((int)blockIdx.x < G) { s1_gemm<V>((int)blockIdx.x, X, LW, LB, WP, BV, Y, QKVG, L, NC, eps_s, smem, use_tmw ? &tmw : nullptr); return; }
  s1_pair<NH, DPW>((int)blockIdx.x - G, (int)gridDim.x - G, Z, WF, MASK, BIAS, L, eps_p, neg, smem, klen_s);
}
// inf_fronto_k (v24): inf_front_k with the roles overlapped in one wave. At L 256 the GEMM tiles (192-256) and the pair blocks did not fit
// in one wave (2 blocks per SM), so most pair blocks started only after the GEMM blocks ended (stamps: pair work from 6 us on, the
// front 13-15 us against 8 / 8-10 us for either role alone). Here the grid is one wave (P blocks): blocks [0, G) take GEMM tile b
// (stride P if G > P), then join the pair walk; the P - G others walk pair tiles at once. The pair tiles are split statically: [0, TA)
// over the P - G pair-only blocks (KA tiles each), [TA, T) over the G GEMM blocks. Every tile is computed as in inf_front_k (the same
// bits). (A dynamic queue -- an atomic counter per launch -- was built first and lost 2-4 us: docs/b200/rejected/apb_v24_front_queue.diff.)
template <int NH, int DPW, int V>
__global__ void __launch_bounds__(S1_NT, DPW == DP ? 2 : 1) inf_fronto_k(const bf* __restrict__ Z, const bf* __restrict__ WF,
    const bool* __restrict__ MASK, bf* __restrict__ BIAS, int L, float eps_p, float neg, int G, const bf* __restrict__ X,
    const float* __restrict__ LW, const float* __restrict__ LB, const bf* __restrict__ WP, const bf* __restrict__ BV, bf* __restrict__ Y,
    bf* __restrict__ QKVG, int NC, float eps_s, int KA, const __grid_constant__ CUtensorMap tmw, int use_tmw) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int klen_s[1];
#if FR_TL
  { unsigned s_; asm volatile("mov.u32 %0, %%smid;" : "=r"(s_)); FRT(6, s_ + 1); FRT(5, 0); }
#endif
  const long ntiles = (long)L * ((L + fwd_rows<DPW>() - 1) / fwd_rows<DPW>());
  const int P = (int)gridDim.x, b = (int)blockIdx.x, N0 = P - min(G, P);
  const long TA = N0 > 0 ? min(ntiles, (long)N0 * KA) : 0;
  if (b < G) {
    for (int it = b; it < G; it += P) {
      s1_gemm<V>(it, X, LW, LB, WP, BV, Y, QKVG, L, NC, eps_s, smem, use_tmw ? &tmw : nullptr);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // the tile's generic shared accesses before later bulk copies land there
      __syncthreads();
      if (threadIdx.x == 0) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(su32(s1_bar<V>(smem, use_tmw != 0))) : "memory");
      __syncthreads();
    }
    if (TA < ntiles) s1_pair<NH, DPW>(b, min(G, P), Z, WF, MASK, BIAS, L, eps_p, neg, smem, klen_s, TA, ntiles);
  } else if (TA > 0) {
    s1_pair<NH, DPW>(b - G, N0, Z, WF, MASK, BIAS, L, eps_p, neg, smem, klen_s, 0, TA);
  }
}
template <int NH, int DPW, int V>
void inf_front_t(const at::Tensor& z, const at::Tensor& wf, const c10::optional<at::Tensor>& mask, at::Tensor& bias, int64_t L, double eps_p,
                 double neg, const at::Tensor& x, const at::Tensor& lw, const at::Tensor& lb, const at::Tensor& wp, const at::Tensor& bv,
                 const c10::optional<at::Tensor>& y, at::Tensor& qkvg, double eps_s, int64_t pblocks) {
  static bool attr = false;
  constexpr int SMB = s1_smem<NH, DPW, V>();
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(inf_front_k<NH, DPW, V>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    C10_CUDA_CHECK(cudaFuncSetAttribute(inf_fronto_k<NH, DPW, V>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    attr = true;
  }
  const int NC = (int)qkvg.size(1);
  int G = (int)((L + S1_BM - 1) / S1_BM) * (NC / S1_BN);
  // the pack wp [NC, D] as D / 64 chunks of 64 columns: dims (64, NC, D / 64), one box = a GEMM tile's S1_BN rows (s1_gemm, TMW)
  constexpr int D = 128 * V;
  const cuuint64_t wd[3] = {64, (cuuint64_t)NC, (cuuint64_t)(D / 64)}, ws[2] = {(cuuint64_t)D * 2, 128};
  const cuuint32_t wb[3] = {64, (cuuint32_t)S1_BN, (cuuint32_t)(D / 64)};
  const CUtensorMap tmw = tensor_map(wp.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, wd, ws, wb, CU_TENSOR_MAP_SWIZZLE_128B);
  const long ntiles = L * ((L + fwd_rows<DPW>() - 1) / fwd_rows<DPW>());
  const long per_sm = DPW == DP ? 2 : 1, nsm = nsm_of(z);
  if (pblocks <= -100) {  // the overlapped roles (inf_fronto_k), one wave; KA = -100 - pblocks pair tiles per pair-only block
    const long grid = per_sm * nsm;
    inf_fronto_k<NH, DPW, V><<<(unsigned)grid, S1_NT, SMB, S()>>>(CP<bf>(z), CP<bf>(wf), mask ? CP<bool>(*mask) : nullptr, P<bf>(bias),
        (int)L, (float)eps_p, (float)neg, G, CP<bf>(x), CP<float>(lw), CP<float>(lb), CP<bf>(wp), CP<bf>(bv), y ? P<bf>(*y) : nullptr,
        P<bf>(qkvg), NC, (float)eps_s, (int)(-100 - pblocks), tmw, g_front_tmw);
    return;
  }
  // default: the pair blocks that fit beside the GEMM blocks in the first wave (at least one SM's worth)
  long PB = pblocks > 0 ? pblocks : std::max<long>(per_sm * nsm - G, nsm);
  PB = std::min<long>(PB, ntiles);
  if (pblocks == -1) PB = 0;  // timing probes only (incomplete results): the GEMM role alone / the pair role alone
  if (pblocks == -2) { G = 0; PB = std::min<long>(per_sm * nsm, ntiles); }
  inf_front_k<NH, DPW, V><<<(unsigned)(G + PB), S1_NT, SMB, S()>>>(CP<bf>(z), CP<bf>(wf), mask ? CP<bool>(*mask) : nullptr, P<bf>(bias),
      (int)L, (float)eps_p, (float)neg, G, CP<bf>(x), CP<float>(lw), CP<float>(lb), CP<bf>(wp), CP<bf>(bv), y ? P<bf>(*y) : nullptr,
      P<bf>(qkvg), NC, (float)eps_s, tmw, g_front_tmw);
}
// z [L L, dp] bf16, wf [H, dp] bf16, mask [L] bool or None -> bias [H, L, L] bf16 (exp2 units as packed); x [L, D] bf16, lw / lb [D] fp32,
// wp [4 W, D] bf16 (the pack), bv [4 W] bf16 -> qkvg [L, 4 W] = LN(x) wp^T + bv, y [L, D] = x (residual seed). d_pair 128 (8 / 12 / 16 / 24
// heads) or 256 (8 / 16 heads); D 384 / 512. ``pblocks``: pair blocks (0: the default rule; <= -100: inf_fronto_k with KA = -100 - pblocks).
void inf_front(at::Tensor z, at::Tensor wf, c10::optional<at::Tensor> mask, at::Tensor bias, int64_t L, double eps_p, double neg, at::Tensor x,
               at::Tensor lw, at::Tensor lb, at::Tensor wp, at::Tensor bv, c10::optional<at::Tensor> y, at::Tensor qkvg, double eps_s,
               int64_t pblocks) {
  chk(z, at::kBFloat16, "pair"); chk(wf, at::kBFloat16, "wf"); chk(bias, at::kBFloat16, "bias"); chk(x, at::kBFloat16, "x");
  chk(lw, at::kFloat, "lw"); chk(lb, at::kFloat, "lb"); chk(wp, at::kBFloat16, "wp"); chk(bv, at::kBFloat16, "bv"); chk(qkvg, at::kBFloat16, "qkvg");
  const int64_t nh = wf.size(0), dp = wf.size(1), D = lw.numel(), NC = qkvg.size(1);
  TORCH_CHECK(L % 16 == 0 && z.numel() == L * L * dp && bias.numel() == nh * L * L, "inf_front: pair / bias shapes");
  TORCH_CHECK((dp == DP && (nh == 8 || nh == 12 || nh == 16 || nh == 24)) || (dp == 2 * DP && (nh == 8 || nh == 16)), "inf_front: d_pair 128 / 256");
  TORCH_CHECK((D == 384 || D == 512) && x.numel() == L * D && wp.size(1) == D && wp.size(0) == NC && bv.numel() == NC && qkvg.size(0) == L
              && NC % S1_BN == 0, "inf_front: single / pack shapes");
  if (mask) { chk(*mask, at::kBool, "mask"); TORCH_CHECK(mask->numel() == L, "mask [L]"); }
  if (y) { chk(*y, at::kBFloat16, "y"); TORCH_CHECK(y->numel() == L * D, "y [L, D]"); }
  const at::cuda::CUDAGuard gd(z.device());
#define IF_GO(NH, DPW)                                                                                     \
  do {                                                                                                     \
    if (D == 384) inf_front_t<NH, DPW, 3>(z, wf, mask, bias, L, eps_p, neg, x, lw, lb, wp, bv, y, qkvg, eps_s, pblocks); \
    else inf_front_t<NH, DPW, 4>(z, wf, mask, bias, L, eps_p, neg, x, lw, lb, wp, bv, y, qkvg, eps_s, pblocks);         \
  } while (0)
  if (dp == 2 * DP) { if (nh == 16) IF_GO(16, 2 * DP); else IF_GO(8, 2 * DP); }
  else if (nh == 8) IF_GO(8, DP);
  else if (nh == 12) IF_GO(12, DP);
  else if (nh == 16) IF_GO(16, DP);
  else IF_GO(24, DP);
#undef IF_GO
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------------------------------------------
// small_core_k (v23, 2026-11-18): the bf16 inference core AND to_out in ONE launch for short sequences (L 128 / 256), launched as a
// programmatic dependent of the front (inf_front: LN + q | k | v | g GEMM + pair pass). The step was front -> attn_inf (tcgen05, one CTA
// per head and query-tile pair: 16 work items for 148 SMs) -> cuBLAS to_out (its own launch).
// A CTA is (head h, 64-query block qb), 4 warps of 16 query rows, mma.sync; a CLUSTER is the NH heads of one query block (NH <= 16):
//   prologue (before griddepcontrol.wait; the packed Wo is a parameter form, made before the front): the CTA's to_out column slice
//            Wo[c0 : c0 + NC, :] (NC = D / NH) to shared memory, its mbarriers, one cluster barrier (every CTA's barriers exist);
//   wait;    the residual seed out[q0.., c0..] to registers; q / g / k / v of head h (cp.async), the bias rows [h, q0 : q0 + 64, :]
//            (bulk copies, one row per lane);
//   core     S = q k^T + bias (exp2 units), exact row maximum (the key row in registers), P = 2^(S - m) (bf16), l = sum P (fp32),
//            O = P v, o = sigmoid(g) O / l -> bf16 (attn_inf's form) into the CTA's slot h of the block's o [NH][64][DHP + 8];
//   exchange each CTA pushes its slot to the other NH - 1 CTAs of the cluster (bulk shared::cta -> shared::cluster copies, one lane
//            per destination, onto the destination's mbarrier): no global round trip, no counter;
//   to_out   out[q0 + r, c0 + c] = seed + sum_k o[r, k] Wo[c0 + c, k] (fixed order: deterministic), bf16.
// A last cluster barrier keeps every CTA's shared memory alive until all pushes have landed.
#ifndef SC_TL
#define SC_TL 0
#endif
#if SC_TL
__device__ unsigned long long g_sctl[128][8];
#define SCT(i) do { if (threadIdx.x == 0 && blockIdx.x < 128) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_sctl[blockIdx.x][i] = t_; } } while (0)
#else
#define SCT(i)
#endif
// v26: QB_ queries per CTA (2 QB_ threads), HPC_ heads per CTA (a cluster of NR = NH / HPC_ ranks; 24 heads: 2 per CTA, 12 ranks)
template <int NH, int DHP, int V, int L, bool TO, int QB_ = 64, int HPC_ = 1> struct SC {
  static constexpr int HPC = HPC_, NR = NH / HPC_;
  static constexpr int D = 128 * V, W = NH * DHP, NC = D / NR, QB = QB_, NTH = 2 * QB_, LDH = DHP + 8, LDW = W + 8, LDB = L + 8, OB = QB * LDH * 2;
  static constexpr int OFF_WO = 0, OFF_O = OFF_WO + (TO ? NC * LDW * 2 : 0), OFF_Q = OFF_O + (TO ? NH * OB : 0), OFF_G = OFF_Q + HPC * QB * LDH * 2,
                       OFF_K = OFF_G + HPC * QB * LDH * 2, OFF_V = OFF_K + HPC * L * LDH * 2, OFF_B = OFF_V + HPC * L * LDH * 2,
                       OFF_BAR = OFF_B + HPC * QB * LDB * 2;
  static constexpr int SMEM = OFF_BAR + 16;
  static_assert(NC % 8 == 0 && DHP % 16 == 0 && D % NR == 0 && NH % HPC == 0 && (NR <= 16 || !TO) && SMEM <= 232448, "small_core: shapes");
};
__device__ __forceinline__ void cp16(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(su32(s)), "l"(g) : "memory");
}
__device__ __forceinline__ float sc_sigmoid(float a) {  // sm100.cuh sigmoid_kit: 1 / (1 + 2^(-log2e a))
  float e, r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(__fmul_rn(-1.4426950408889634f, a)));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(__fadd_rn(1.f, e)));
  return r;
}
__device__ __forceinline__ float sc_ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ void ldsm_x2_t(uint32_t& b0, uint32_t& b1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(su32(p)));
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
// TR (v26, training forward, with TO): q unscaled and the bias in natural units (attn_fwd2's contract): x = (q k^T rsqd + bias) log2e; the
// pass also writes O = P v / l (fp32 [L, W]), LSE = m + log2 l ([NH, L], log2 units) and og = sigmoid(g) O (bf16 [L, W]): what the
// backward reads (attn_fwd2 + gate_rows + to_out's values up to rounding order).
template <int NH, int DHP, int V, int L, bool TO, int QB_ = 64, int HPC_ = 1, bool TR = false>
__global__ void __launch_bounds__(2 * QB_, 1) small_core_k(bf* __restrict__ QKVG, const bf* __restrict__ BIAS, const bf* __restrict__ WO,
                                                          bf* __restrict__ OUT, float* __restrict__ OF, float* __restrict__ LSEO,
                                                          bf* __restrict__ OGO, float rsqd) {
  using C = SC<NH, DHP, V, L, TO, QB_, HPC_>;
  constexpr int D = C::D, W = C::W, NC = C::NC, QB = C::QB, NTH = C::NTH, LDH = C::LDH, LDW = C::LDW, LDB = C::LDB, CH = DHP / 8, KC = L <= 256 ? L : 128, NT = KC / 8;
  constexpr int HPC = C::HPC, NR = C::NR;
  extern __shared__ __align__(128) uint8_t smem[];
  bf* wo_s = reinterpret_cast<bf*>(smem + C::OFF_WO);
  bf* o_s = reinterpret_cast<bf*>(smem + C::OFF_O);
  bf* q_s = reinterpret_cast<bf*>(smem + C::OFF_Q);
  bf* g_s = reinterpret_cast<bf*>(smem + C::OFF_G);
  bf* k_s = reinterpret_cast<bf*>(smem + C::OFF_K);
  bf* v_s = reinterpret_cast<bf*>(smem + C::OFF_V);
  bf* b_s = reinterpret_cast<bf*>(smem + C::OFF_B);
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::OFF_BAR);  // 0: bias rows landed, 1 (TO): every peer's o slot landed
  // h = the rank in the cluster: heads h HPC .. h HPC + HPC - 1, to_out columns c0 .. c0 + NC
  const int h = (int)blockIdx.x % NR, qb = (int)blockIdx.x / NR, q0 = qb * QB, c0 = h * NC;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3, rb = 16 * warp;
  SCT(0);
  for (int i = tid; TO && i < NC * (W / 8); i += NTH) {
    const int r = i / (W / 8), c = i % (W / 8);
    cp16(wo_s + r * LDW + 8 * c, WO + (long)(c0 + r) * W + 8 * c);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  if (tid == 0) {
    mbar_init(&bars[0], 1);
    if (TO) mbar_init(&bars[1], 1);
    mbar_fence_init();
    if (TO) mbar_expect_tx(&bars[1], (uint32_t)((NH - HPC) * C::OB));  // armed before the cluster arrive below: the peers' st.async count down
  }
  __syncthreads();
  if constexpr (TO) asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");  // waited on before the exchange
  asm volatile("griddepcontrol.wait;" ::: "memory");
  SCT(1);
  if (warp == 0) {  // the bias rows: one bulk copy per row, lanes in parallel
    if (lane == 0) mbar_expect_tx(&bars[0], (uint32_t)(HPC * QB * L * 2));
    __syncwarp();
    for (int i = lane; i < HPC * QB; i += 32) {
      const int j = i / QB, r = i - j * QB;
      bulk_g2s(b_s + i * LDB, BIAS + ((long)(h * HPC + j) * L + q0 + r) * L, (uint32_t)(L * 2), &bars[0]);
    }
  }
  constexpr long ld4 = 4L * W;
  for (int i = tid; i < HPC * QB * CH; i += NTH) {  // [j][r] rows of head h HPC + j
    const int jr = i / CH, c = i % CH, j = jr / QB, r = jr - j * QB;
    const bf* row = QKVG + (long)(q0 + r) * ld4 + (h * HPC + j) * DHP + 8 * c;
    cp16(q_s + jr * LDH + 8 * c, row);
    cp16(g_s + jr * LDH + 8 * c, row + 3 * W);
  }
  for (int i = tid; i < HPC * L * CH; i += NTH) {
    const int jr = i / CH, c = i % CH, j = jr / L, r = jr - j * L;
    const bf* row = QKVG + (long)r * ld4 + (h * HPC + j) * DHP + 8 * c;
    cp16(k_s + jr * LDH + 8 * c, row + W);
    cp16(v_s + jr * LDH + 8 * c, row + 2 * W);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  uint32_t ya[NC / 8], yb[NC / 8];  // the residual seed (written by the front), loaded under the core
#pragma unroll
  for (int nt = 0; nt < (TO ? NC / 8 : 0); ++nt) {
    const int col = c0 + 8 * nt + 2 * q;
    ya[nt] = __ldcg(reinterpret_cast<const unsigned int*>(OUT + (long)(q0 + rb + g) * D + col));
    yb[nt] = __ldcg(reinterpret_cast<const unsigned int*>(OUT + (long)(q0 + rb + g + 8) * D + col));
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  mbar_wait(&bars[0], 0);
  SCT(2);
  // ---- core: warp w owns query rows 16 w .. 16 w + 15 of the block
#pragma unroll 1
  for (int j = 0; j < HPC; ++j) {  // v26: this CTA's heads one after another (HPC 2 at 24 heads)
  const bf* qj = q_s + j * QB * LDH; const bf* gj = g_s + j * QB * LDH; const bf* kj = k_s + j * L * LDH; const bf* vj = v_s + j * L * LDH;
  const bf* bj = b_s + j * QB * LDB;
  const int hd = h * HPC + j;
  uint32_t qa[DHP / 16][4];
#pragma unroll
  for (int kk = 0; kk < DHP / 16; ++kk) {
    const bf* ap = qj + (rb + g) * LDH + 16 * kk + 2 * q;
    qa[kk][0] = *reinterpret_cast<const uint32_t*>(ap);
    qa[kk][1] = *reinterpret_cast<const uint32_t*>(ap + 8 * LDH);
    qa[kk][2] = *reinterpret_cast<const uint32_t*>(ap + 8);
    qa[kk][3] = *reinterpret_cast<const uint32_t*>(ap + 8 * LDH + 8);
  }
  // v26: keys in chunks of KC (L <= 256: one chunk -- v23's exact one-pass softmax, the same statements); longer L: 128-key chunks with
  // the running maximum (o and l rescaled by 2^(m_old - m_new) when a chunk raises it)
  float mx0 = -INFINITY, mx1 = -INFINITY;
  float l0 = 0.f, l1 = 0.f;
  float o[DHP / 8][4] = {};
#pragma unroll 1
  for (int k0 = 0; k0 < L; k0 += KC) {
  float s[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    if constexpr (TR) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
    } else {
      const float2 b0 = unpack2(*reinterpret_cast<const uint32_t*>(bj + (rb + g) * LDB + k0 + 8 * nt + 2 * q));
      const float2 b1 = unpack2(*reinterpret_cast<const uint32_t*>(bj + (rb + g + 8) * LDB + k0 + 8 * nt + 2 * q));
      s[nt][0] = b0.x; s[nt][1] = b0.y; s[nt][2] = b1.x; s[nt][3] = b1.y;
    }
  }
#pragma unroll
  for (int kk = 0; kk < DHP / 16; ++kk)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const uint32_t* bp = reinterpret_cast<const uint32_t*>(kj + (k0 + 8 * nt + g) * LDH + 16 * kk + 2 * q);
      mma16816(s[nt], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], bp[0], bp[4]);
    }
  if constexpr (TR) {  // x = (S rsqd + bias) log2e (attn_fwd2's t, in exp2 units)
    constexpr float L2E = 1.4426950408889634f;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const float2 b0 = unpack2(*reinterpret_cast<const uint32_t*>(bj + (rb + g) * LDB + k0 + 8 * nt + 2 * q));
      const float2 b1 = unpack2(*reinterpret_cast<const uint32_t*>(bj + (rb + g + 8) * LDB + k0 + 8 * nt + 2 * q));
      s[nt][0] = fmaf(s[nt][0], rsqd, b0.x) * L2E; s[nt][1] = fmaf(s[nt][1], rsqd, b0.y) * L2E;
      s[nt][2] = fmaf(s[nt][2], rsqd, b1.x) * L2E; s[nt][3] = fmaf(s[nt][3], rsqd, b1.y) * L2E;
    }
  }
  float cm0 = -INFINITY, cm1 = -INFINITY;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    cm0 = fmaxf(cm0, fmaxf(s[nt][0], s[nt][1]));
    cm1 = fmaxf(cm1, fmaxf(s[nt][2], s[nt][3]));
  }
  cm0 = fmaxf(cm0, __shfl_xor_sync(0xffffffffu, cm0, 1)); cm0 = fmaxf(cm0, __shfl_xor_sync(0xffffffffu, cm0, 2));
  cm1 = fmaxf(cm1, __shfl_xor_sync(0xffffffffu, cm1, 1)); cm1 = fmaxf(cm1, __shfl_xor_sync(0xffffffffu, cm1, 2));
  if (k0 == 0) {
    mx0 = cm0; mx1 = cm1;
  } else {
    const float n0 = fmaxf(mx0, cm0), n1 = fmaxf(mx1, cm1), a0 = sc_ex2(mx0 - n0), a1 = sc_ex2(mx1 - n1);
    mx0 = n0; mx1 = n1; l0 *= a0; l1 *= a1;
#pragma unroll
    for (int dn = 0; dn < DHP / 8; ++dn) { o[dn][0] *= a0; o[dn][1] *= a0; o[dn][2] *= a1; o[dn][3] *= a1; }
  }
#pragma unroll
  for (int kc = 0; kc < NT / 2; ++kc) {  // 16 keys per k-step of P V
    uint32_t pa[4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const float* t = s[2 * kc + hh];
      const float p0 = sc_ex2(t[0] - mx0), p1 = sc_ex2(t[1] - mx0), p2 = sc_ex2(t[2] - mx1), p3 = sc_ex2(t[3] - mx1);
      l0 += p0 + p1; l1 += p2 + p3;
      pa[2 * hh] = pack2(p0, p1); pa[2 * hh + 1] = pack2(p2, p3);
    }
#pragma unroll
    for (int dn = 0; dn < DHP / 8; ++dn) {
      uint32_t b0, b1;
      ldsm_x2_t(b0, b1, vj + (k0 + 16 * kc + (lane & 15)) * LDH + 8 * dn);
      mma16816(o[dn], pa[0], pa[1], pa[2], pa[3], b0, b1);
    }
  }
  }  // key chunks
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  bf* mine = o_s + hd * (C::OB / 2);
  if constexpr (TR) {
    if (q == 0) {
      LSEO[(long)hd * L + q0 + rb + g] = mx0 + __log2f(l0);
      LSEO[(long)hd * L + q0 + rb + g + 8] = mx1 + __log2f(l1);
    }
  }
#pragma unroll
  for (int dn = 0; dn < DHP / 8; ++dn) {
    const int col = 8 * dn + 2 * q;
    const float2 ga = unpack2(*reinterpret_cast<const uint32_t*>(gj + (rb + g) * LDH + col));
    const float2 gb = unpack2(*reinterpret_cast<const uint32_t*>(gj + (rb + g + 8) * LDH + col));
    uint32_t ua, ub;
    if constexpr (TR) {  // O fp32 (the backward's), og = sigmoid(g) O bf16 (saved for dWo, and to_out's operand)
      const float2 oa = make_float2(o[dn][0] * i0, o[dn][1] * i0), ob = make_float2(o[dn][2] * i1, o[dn][3] * i1);
      *reinterpret_cast<float2*>(OF + (long)(q0 + rb + g) * W + hd * DHP + col) = oa;
      *reinterpret_cast<float2*>(OF + (long)(q0 + rb + g + 8) * W + hd * DHP + col) = ob;
      ua = pack2(sc_sigmoid(ga.x) * oa.x, sc_sigmoid(ga.y) * oa.y);
      ub = pack2(sc_sigmoid(gb.x) * ob.x, sc_sigmoid(gb.y) * ob.y);
      *reinterpret_cast<uint32_t*>(OGO + (long)(q0 + rb + g) * W + hd * DHP + col) = ua;
      *reinterpret_cast<uint32_t*>(OGO + (long)(q0 + rb + g + 8) * W + hd * DHP + col) = ub;
    } else {
      ua = pack2(o[dn][0] * i0 * sc_sigmoid(ga.x), o[dn][1] * i0 * sc_sigmoid(ga.y));
      ub = pack2(o[dn][2] * i1 * sc_sigmoid(gb.x), o[dn][3] * i1 * sc_sigmoid(gb.y));
    }
    if constexpr (TO) {
      *reinterpret_cast<uint32_t*>(mine + (rb + g) * LDH + col) = ua;
      *reinterpret_cast<uint32_t*>(mine + (rb + g + 8) * LDH + col) = ub;
    } else {  // the gated o over q (attn_inf's contract): cuBLAS to_out follows
      *reinterpret_cast<uint32_t*>(QKVG + (long)(q0 + rb + g) * ld4 + hd * DHP + col) = ua;
      *reinterpret_cast<uint32_t*>(QKVG + (long)(q0 + rb + g + 8) * ld4 + hd * DHP + col) = ub;
    }
  }
  }  // heads of this CTA
  if constexpr (!TO) return;
  SCT(3);
  // ---- exchange (v26): my slot into every other CTA of the cluster (the same offset there) as st.async 16-byte stores that count down
  // the destination's mbarrier (complete_tx): each thread reads its chunks of the slot once and sends each to the NH - 1 peers, a warp's 32
  // lanes 512 contiguous bytes per destination per instruction, destinations rotated by rank. No cluster-wide release / acquire: a CTA
  // waits only for the bytes of its own slots (v23's st.shared::cluster + barrier.cluster: 4.6 us at L 128).
  __syncthreads();  // my slot is complete in shared memory
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // every peer's mbarrier is armed (its arrive in the prologue)
  constexpr int OC = HPC * C::OB / 16;  // 16-byte chunks of my HPC (contiguous) slots
  {
    const uint32_t bar = su32(&bars[1]), mine = su32(o_s + h * HPC * (C::OB / 2));
    for (int ch = tid; ch < OC; ch += NTH) {
      const uint32_t a = mine + 16 * ch;
      uint4 v;
      asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
#pragma unroll
      for (int rk = 1; rk < NR; ++rk) {
        const uint32_t dst = (uint32_t)((h + rk) % NR);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     ::"r"(mapa(a, dst)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(mapa(bar, dst)) : "memory");
      }
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");  // my stores are issued (waited on at the very end)
    uint32_t ok = 0;
    while (!ok)
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                   : "=r"(ok) : "r"(bar), "r"(0u) : "memory");
  }
  SCT(4);
  // ---- to_out: out[q0 + rb .. + 16, c0 .. c0 + NC] = seed + o Wo_slice^T (k over heads, then the head's channels)
  float acc[NC / 8][4] = {};
#pragma unroll 2
  for (int hk = 0; hk < NH; ++hk) {
    const bf* ob = o_s + hk * (C::OB / 2);
#pragma unroll
    for (int kk = 0; kk < DHP / 16; ++kk) {
      const bf* ap = ob + (rb + g) * LDH + 16 * kk + 2 * q;
      const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap), a1 = *reinterpret_cast<const uint32_t*>(ap + 8 * LDH),
                     a2 = *reinterpret_cast<const uint32_t*>(ap + 8), a3 = *reinterpret_cast<const uint32_t*>(ap + 8 * LDH + 8);
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) {
        const uint32_t* bp = reinterpret_cast<const uint32_t*>(wo_s + (8 * nt + g) * LDW + hk * DHP + 16 * kk + 2 * q);
        mma16816(acc[nt], a0, a1, a2, a3, bp[0], bp[4]);
      }
    }
  }
#pragma unroll
  for (int nt = 0; nt < NC / 8; ++nt) {
    const int col = c0 + 8 * nt + 2 * q;
    const float2 sa = unpack2(ya[nt]), sb = unpack2(yb[nt]);
    *reinterpret_cast<uint32_t*>(OUT + (long)(q0 + rb + g) * D + col) = pack2(acc[nt][0] + sa.x, acc[nt][1] + sa.y);
    *reinterpret_cast<uint32_t*>(OUT + (long)(q0 + rb + g + 8) * D + col) = pack2(acc[nt][2] + sb.x, acc[nt][3] + sb.y);
  }
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");  // no CTA exits while a peer's stores into it may be in flight
  SCT(5);
}
template <int NH, int DHP, int V, int L, bool TO, int QB_ = 64, int HPC_ = 1, bool TR = false>
void small_core_t(const at::Tensor& qkvg, const at::Tensor& bias, const at::Tensor& wo, at::Tensor& out, bool pdl, float* of = nullptr,
                  float* lse = nullptr, bf* og = nullptr, float rsqd = 0.f) {
  using C = SC<NH, DHP, V, L, TO, QB_, HPC_>;
  auto k = small_core_k<NH, DHP, V, L, TO, QB_, HPC_, TR>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    if (TO) C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C::NR * (L / C::QB));
  cfg.blockDim = dim3(C::NTH);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = S();
  cudaLaunchAttribute at[2];
  int na = 0;
  if (TO) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = C::NR; at[na].val.clusterDim.y = 1; at[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, P<bf>(const_cast<at::Tensor&>(qkvg)), CP<bf>(bias), CP<bf>(wo), P<bf>(out), of, lse, og, rsqd));
}
// v26: the core-only form at L 384 / 512 / 640 (head width <= 32: K / V of a head and the CTA's bias rows fit in shared memory)
template <int NH, int DHP, int V>
void sc_long(const at::Tensor& qkvg, const at::Tensor& bias, const at::Tensor& wo, at::Tensor& out, int64_t L, bool pdl, int64_t to_out) {
  if constexpr (DHP <= 32) {
    auto go = [&](auto lc) {
      constexpr int LL = decltype(lc)::value;
      if (to_out == -32) small_core_t<NH, DHP, V, LL, false, 32>(qkvg, bias, wo, out, pdl);
      else small_core_t<NH, DHP, V, LL, false, 64>(qkvg, bias, wo, out, pdl);
    };
    if (L == 384) go(std::integral_constant<int, 384>{});
    else if (L == 512) go(std::integral_constant<int, 512>{});
    else go(std::integral_constant<int, 640>{});
  } else {
    TORCH_CHECK(false, "small_core: L > 256 needs head width <= 32");
  }
}
// qkvg [L, 4 W] bf16 (q | k | v | g), bias [H, L, L] bf16 (exp2 units), wo [D, W] bf16 (the packed Wo), out [L, D] bf16 (the residual seed).
// to_out > 0: out = seed + o Wo^T in the same launch (a cluster per query block; v26: to_out = 16 / 32 / 64 (1: 64) queries per CTA, 24
// heads 2 per CTA); 0: the gated o over q (attn_inf's contract), out untouched (the caller's to_out follows), 64 queries per CTA; v26: -32 /
// -64 the same with 32 / 64 queries per CTA, also at L 384 / 512 / 640 (128-key chunks; head width 16 / 32 only). L 128 / 256 otherwise;
// (H, D) 8 / 12 / 16 / 24 x 384, 16 x 512.
void small_core(at::Tensor qkvg, at::Tensor bias, at::Tensor wo, at::Tensor out, int64_t L, bool pdl, int64_t to_out) {
  chk(qkvg, at::kBFloat16, "qkvg"); chk(bias, at::kBFloat16, "bias"); chk(wo, at::kBFloat16, "wo"); chk(out, at::kBFloat16, "out");
  const int64_t H = bias.size(0), D = out.size(-1), W = wo.size(1);
  const bool lng = L == 384 || L == 512 || L == 640;
  TORCH_CHECK((L == 128 || L == 256 || (lng && to_out < 0)) && qkvg.size(0) == L && qkvg.size(1) == 4 * W && bias.numel() == H * L * L
              && wo.size(0) == D && out.numel() == L * D
              && (to_out == 0 || to_out == 1 || to_out == 16 || to_out == 32 || to_out == 64 || to_out == -32 || to_out == -64),
              "small_core: shapes");
  const at::cuda::CUDAGuard gd(qkvg.device());
  // 24 heads fused: 2 heads per CTA (a cluster of 12), 16 or 32 queries per CTA (64 does not fit)
#define SC_GQ(NH, DHP, V, LL)                                                                                \
  do {                                                                                                       \
    constexpr int HP = NH > 16 ? 2 : 1;                                                                      \
    if (to_out == 16) small_core_t<NH, DHP, V, LL, true, 16, HP>(qkvg, bias, wo, out, pdl);                  \
    else if (to_out == 32 || HP > 1) small_core_t<NH, DHP, V, LL, true, 32, HP>(qkvg, bias, wo, out, pdl);   \
    else small_core_t<NH, DHP, V, LL, true, NH <= 16 ? 64 : 32, HP>(qkvg, bias, wo, out, pdl);               \
  } while (0)
#define SC_GC(NH, DHP, V, LL)                                                                                \
  do {                                                                                                       \
    if (to_out == -32) small_core_t<NH, DHP, V, LL, false, 32>(qkvg, bias, wo, out, pdl);                    \
    else small_core_t<NH, DHP, V, LL, false, 64>(qkvg, bias, wo, out, pdl);                                  \
  } while (0)
#define SC_GL(NH, DHP, V) sc_long<NH, DHP, V>(qkvg, bias, wo, out, L, pdl, to_out)
#define SC_GO(NH, DHP, V)                                                                                    \
  do {                                                                                                       \
    if (lng) SC_GL(NH, DHP, V);                                                                              \
    else if (to_out > 0) {                                                                                   \
      if (L == 128) SC_GQ(NH, DHP, V, 128);                                                                  \
      else SC_GQ(NH, DHP, V, 256);                                                                           \
    } else {                                                                                                 \
      if (L == 128) SC_GC(NH, DHP, V, 128);                                                                  \
      else SC_GC(NH, DHP, V, 256);                                                                           \
    }                                                                                                        \
  } while (0)
  if (D == 384 && H == 16 && W == 512) SC_GO(16, 32, 3);
  else if (D == 384 && H == 12 && W == 384) SC_GO(12, 32, 3);
  else if (D == 384 && H == 24 && W == 384) SC_GO(24, 16, 3);
  else if (D == 384 && H == 8 && W == 384) SC_GO(8, 48, 3);
  else if (D == 512 && H == 16 && W == 512) SC_GO(16, 32, 4);
  else TORCH_CHECK(false, "small_core: geometry");
#undef SC_GO
#undef SC_GQ
#undef SC_GC
#undef SC_GL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// v26: the training forward at L 128 in ONE launch (TR): qkvg [L, 4 W] (q unscaled), bias [H, L, L] bf16 natural units, wo [D, W],
// out [L, D] (the residual seed) -> out += og Wo^T, and the saved O [L, W] fp32, LSE [H, L] (log2), og [L, W] bf16. qb queries per CTA
// (16 / 32); 24 heads 2 per CTA.
void small_core_train(at::Tensor qkvg, at::Tensor bias, at::Tensor wo, at::Tensor out, at::Tensor o, at::Tensor lse, at::Tensor og,
                      int64_t L, bool pdl, double rsqd, int64_t qb) {
  chk(qkvg, at::kBFloat16, "qkvg"); chk(bias, at::kBFloat16, "bias"); chk(wo, at::kBFloat16, "wo"); chk(out, at::kBFloat16, "out");
  chk(o, at::kFloat, "o"); chk(lse, at::kFloat, "lse"); chk(og, at::kBFloat16, "og");
  const int64_t H = bias.size(0), D = out.size(-1), W = wo.size(1);
  TORCH_CHECK((L == 128 || L == 256) && qkvg.size(0) == L && qkvg.size(1) == 4 * W && bias.numel() == H * L * L && wo.size(0) == D && out.numel() == L * D
              && o.numel() == L * W && lse.numel() == H * L && og.numel() == L * W && (qb == 16 || qb == 32), "small_core_train: shapes");
  const at::cuda::CUDAGuard gd(qkvg.device());
  float* po = o.data_ptr<float>();
  float* pl = lse.data_ptr<float>();
  bf* pg = P<bf>(og);
  const float rs = (float)rsqd;
#define SC_GT(NH, DHP, V)                                                                                                        \
  do {                                                                                                                           \
    constexpr int HP = NH > 16 ? 2 : 1;                                                                                          \
    if (L == 256) small_core_t<NH, DHP, V, 256, true, 16, HP, true>(qkvg, bias, wo, out, pdl, po, pl, pg, rs);                  \
    else if (qb == 16) small_core_t<NH, DHP, V, 128, true, 16, HP, true>(qkvg, bias, wo, out, pdl, po, pl, pg, rs);             \
    else small_core_t<NH, DHP, V, 128, true, 32, HP, true>(qkvg, bias, wo, out, pdl, po, pl, pg, rs);                           \
  } while (0)
  if (D == 384 && H == 16 && W == 512) SC_GT(16, 32, 3);
  else if (D == 384 && H == 12 && W == 384) SC_GT(12, 32, 3);
  else if (D == 384 && H == 24 && W == 384) SC_GT(24, 16, 3);
  else if (D == 512 && H == 16 && W == 512) SC_GT(16, 32, 4);
  else TORCH_CHECK(false, "small_core_train: geometry");
#undef SC_GT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------------------------------------------
// dqb_small_k (v23, 2026-11-18): the training backward's dQ + dbias pass at L 128 (attn_dqb -DQPAIR=0 -DDQPART -DDET's values: one key chunk,
// one sample) with the QUERY rows split over CTAs: a CTA is (head h, 32-query block), 2 warps of 16 rows, mma.sync, so dQ needs no combine
// and the pass runs as NH L / 32 CTAs (attn_dqb: NH work items) -- it heads the side stream's dqb -> pair backward chain. Per row:
//   x = S rsqd log2e + bias log2e - LSE (S = q k^T), P = 2^x, dP = dO v^T, dS = P (dP - D), dbias = dS (fp32, plain stores),
//   dQ = bf16(dS) k rsqd (fp32, plain stores into DQ[0]).
template <int DHP, int L>
__global__ void __launch_bounds__(64, 1) dqb_small_k(const bf* __restrict__ Q, const bf* __restrict__ K, const bf* __restrict__ VV,
    const bf* __restrict__ DO, int ldq, int lddo, const bf* __restrict__ BIAS, const float* __restrict__ LSE, const float* __restrict__ DD,
    float* __restrict__ DQ, float* __restrict__ DB, int NH, int W, float rsqd) {
  constexpr int QB = 32, LDH = DHP + 8, LDB = L + 8, CH = DHP / 8, NT = L / 8;
  extern __shared__ __align__(128) uint8_t smem[];
  bf* q_s = reinterpret_cast<bf*>(smem);
  bf* o_s = q_s + QB * LDH;
  bf* k_s = o_s + QB * LDH;
  bf* v_s = k_s + L * LDH;
  bf* b_s = v_s + L * LDH;
  const int h = (int)blockIdx.x % NH, q0 = ((int)blockIdx.x / NH) * QB;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3, rb = 16 * warp;
  for (int i = tid; i < QB * CH; i += 64) {
    const int r = i / CH, c = i % CH;
    cp16(q_s + r * LDH + 8 * c, Q + (long)(q0 + r) * ldq + h * DHP + 8 * c);
    cp16(o_s + r * LDH + 8 * c, DO + (long)(q0 + r) * lddo + h * DHP + 8 * c);
  }
  for (int i = tid; i < L * CH; i += 64) {
    const int r = i / CH, c = i % CH;
    cp16(k_s + r * LDH + 8 * c, K + (long)r * ldq + h * DHP + 8 * c);
    cp16(v_s + r * LDH + 8 * c, VV + (long)r * ldq + h * DHP + 8 * c);
  }
  for (int i = tid; i < QB * (L / 8); i += 64) {
    const int r = i / (L / 8), c = i % (L / 8);
    cp16(b_s + r * LDB + 8 * c, BIAS + ((long)h * L + q0 + r) * L + 8 * c);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  const long ra = (long)h * L + q0 + rb + g, rbb = ra + 8;
  const float lse0 = LSE[ra], lse1 = LSE[rbb], dd0 = DD[ra], dd1 = DD[rbb];
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  uint32_t qa[DHP / 16][4], da[DHP / 16][4];
#pragma unroll
  for (int kk = 0; kk < DHP / 16; ++kk) {
    const bf* ap = q_s + (rb + g) * LDH + 16 * kk + 2 * q;
    qa[kk][0] = *reinterpret_cast<const uint32_t*>(ap); qa[kk][1] = *reinterpret_cast<const uint32_t*>(ap + 8 * LDH);
    qa[kk][2] = *reinterpret_cast<const uint32_t*>(ap + 8); qa[kk][3] = *reinterpret_cast<const uint32_t*>(ap + 8 * LDH + 8);
    const bf* dp = o_s + (rb + g) * LDH + 16 * kk + 2 * q;
    da[kk][0] = *reinterpret_cast<const uint32_t*>(dp); da[kk][1] = *reinterpret_cast<const uint32_t*>(dp + 8 * LDH);
    da[kk][2] = *reinterpret_cast<const uint32_t*>(dp + 8); da[kk][3] = *reinterpret_cast<const uint32_t*>(dp + 8 * LDH + 8);
  }
  constexpr float L2E = 1.4426950408889634f;
  const float cql = rsqd * L2E;
  float dq[DHP / 8][4] = {};
  float* db0 = DB + ra * L;
  float* db1 = DB + rbb * L;
#pragma unroll
  for (int kc = 0; kc < NT / 2; ++kc) {  // 16 keys: S / dP for two 8-key tiles, then one k-step of dQ
    uint32_t pa[4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int nt = 2 * kc + hh;
      float s[4] = {}, d[4] = {};
#pragma unroll
      for (int kk = 0; kk < DHP / 16; ++kk) {
        const uint32_t* kp = reinterpret_cast<const uint32_t*>(k_s + (8 * nt + g) * LDH + 16 * kk + 2 * q);
        const uint32_t* vp = reinterpret_cast<const uint32_t*>(v_s + (8 * nt + g) * LDH + 16 * kk + 2 * q);
        mma16816(s, qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], kp[0], kp[4]);
        mma16816(d, da[kk][0], da[kk][1], da[kk][2], da[kk][3], vp[0], vp[4]);
      }
      const float2 b0 = unpack2(*reinterpret_cast<const uint32_t*>(b_s + (rb + g) * LDB + 8 * nt + 2 * q));
      const float2 b1 = unpack2(*reinterpret_cast<const uint32_t*>(b_s + (rb + g + 8) * LDB + 8 * nt + 2 * q));
      const float p0 = sc_ex2(fmaf(s[0], cql, fmaf(b0.x, L2E, -lse0))), p1 = sc_ex2(fmaf(s[1], cql, fmaf(b0.y, L2E, -lse0)));
      const float p2 = sc_ex2(fmaf(s[2], cql, fmaf(b1.x, L2E, -lse1))), p3 = sc_ex2(fmaf(s[3], cql, fmaf(b1.y, L2E, -lse1)));
      const float ds0 = p0 * (d[0] - dd0), ds1 = p1 * (d[1] - dd0), ds2 = p2 * (d[2] - dd1), ds3 = p3 * (d[3] - dd1);
      *reinterpret_cast<float2*>(db0 + 8 * nt + 2 * q) = make_float2(ds0, ds1);
      *reinterpret_cast<float2*>(db1 + 8 * nt + 2 * q) = make_float2(ds2, ds3);
      pa[2 * hh] = pack2(ds0, ds1); pa[2 * hh + 1] = pack2(ds2, ds3);
    }
#pragma unroll
    for (int dn = 0; dn < DHP / 8; ++dn) {
      uint32_t b0, b1;
      ldsm_x2_t(b0, b1, k_s + (16 * kc + (lane & 15)) * LDH + 8 * dn);
      mma16816(dq[dn], pa[0], pa[1], pa[2], pa[3], b0, b1);
    }
    if ((kc + 1) % 8 == 0) {  // 128 keys: this key chunk's dQ partial into its slice DQ[chunk] (attn_dqb -DDQPART's layout)
      float* dqc = DQ + (long)(kc / 8) * L * W;
      const long r0 = q0 + rb + g;
#pragma unroll
      for (int dn = 0; dn < DHP / 8; ++dn) {
        const int col = h * DHP + 8 * dn + 2 * q;
        *reinterpret_cast<float2*>(dqc + r0 * W + col) = make_float2(dq[dn][0] * rsqd, dq[dn][1] * rsqd);
        *reinterpret_cast<float2*>(dqc + (r0 + 8) * W + col) = make_float2(dq[dn][2] * rsqd, dq[dn][3] * rsqd);
        dq[dn][0] = dq[dn][1] = dq[dn][2] = dq[dn][3] = 0.f;
      }
    }
  }
}
template <int DH, int L>
void dqb_small_t(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& dob, const at::Tensor& bias,
                 const at::Tensor& lse, const at::Tensor& dd, at::Tensor& dq, at::Tensor& db, int H, int W, float rsqd) {
  constexpr int SMB = (2 * 32 * (DH + 8) + 2 * L * (DH + 8) + 32 * (L + 8)) * 2;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(dqb_small_k<DH, L>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    attr = true;
  }
  dqb_small_k<DH, L><<<(unsigned)(H * (L / 32)), 64, SMB, S()>>>(CP<bf>(q), CP<bf>(k), CP<bf>(v), CP<bf>(dob), (int)q.stride(0),
      (int)dob.stride(0), CP<bf>(bias), CP<float>(lse), CP<float>(dd), P<float>(dq), P<float>(db), H, W, rsqd);
}
// q / k / v [L, W] bf16 (column views, one row stride), dob [L, W] bf16, bias [H, L, L] bf16 (natural units), lse / dd [H, L] fp32 (LSE in
// log2 units) -> dq [L / 128, L, W] fp32 (per 128-key chunk, x rsqd: attn_dqb -DDQPART's slices), db [H, L, L] fp32. L 128 / 256; head
// width 16 / 32 / 48.
void dqb_small(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor dob, at::Tensor bias, at::Tensor lse, at::Tensor dd, at::Tensor dq,
               at::Tensor db, int64_t L, int64_t dhp, double rsqd) {
  const int64_t H = bias.size(0), W = dob.size(1);
  TORCH_CHECK((L == 128 || L == 256) && q.stride(0) == k.stride(0) && q.stride(0) == v.stride(0) && q.stride(1) == 1 && dob.stride(1) == 1
              && dq.numel() == (L / 128) * L * W && db.numel() == H * L * L && H * dhp == W, "dqb_small: shapes");
  const at::cuda::CUDAGuard gd(q.device());
#define DQ_GO(DH)                                                                                                           \
  do {                                                                                                                      \
    if (L == 128) dqb_small_t<DH, 128>(q, k, v, dob, bias, lse, dd, dq, db, (int)H, (int)W, (float)rsqd);                  \
    else dqb_small_t<DH, 256>(q, k, v, dob, bias, lse, dd, dq, db, (int)H, (int)W, (float)rsqd);                           \
  } while (0)
  if (dhp == 16) DQ_GO(16);
  else if (dhp == 32) DQ_GO(32);
  else if (dhp == 48) DQ_GO(48);
  else TORCH_CHECK(false, "dqb_small: head width");
#undef DQ_GO
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------------------------------------------
// small_core_bwd_k (v27, 2026-11-18): the whole training attention backward at L 128 / 256 in ONE launch -- gate_bwd + attn_dkv + attn_dqb
// (dqb_small) + qkv_bwd. A CTA is (head h, block j of QB rows); it loads the head's q / k / v (all L rows), makes dO = dog sigmoid(g) and
// D = rowsum(dO o) for all L rows itself (redundant over the L / QB CTAs of a head: no exchange, no cluster), then
//   query warps (QB / 16 slabs x SP key splits): rows j QB ..: S, P, dP, dS over their L / SP keys -> dbias (fp32, the pair backward's
//            input), dQ partials;
//   key warps   (QB / 16 slabs x SP query splits): keys j QB ..: S^T, P^T, dP^T, dS^T over their L / SP queries -> dK / dV partials;
//   the SP partials summed in split order (deterministic) -> dQ rsqd, dK rsqd, dV bf16 into dqkvg's q / k / v columns, and dbq's 4-row
//   column sums of the rounded dQ into acc rows (qkv_bwd's layout and order).
// The gate gradient dg of rows j QB .. goes into dqkvg's g columns. Every output is written by one thread: deterministic.
#ifndef SBW_TL
#define SBW_TL 0
#endif
#if SBW_TL
__device__ unsigned long long g_sbtl[512][8];
#define SBT(i) do { if (threadIdx.x == 0 && blockIdx.x < 512) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_sbtl[blockIdx.x][i] = t_; } } while (0)
#else
#define SBT(i)
#endif
template <int DHP, int L, int QB, int SP> struct SB {
  static constexpr int LDH = DHP + 8, LDR = L + 8, LDC = QB + 8, NTH = 2 * (QB / 16) * SP * 32;
  static constexpr int OFF_Q = 0, OFF_K = OFF_Q + L * LDH * 2, OFF_V = OFF_K + L * LDH * 2, OFF_DO = OFF_V + L * LDH * 2,
                       OFF_BR = OFF_DO + L * LDH * 2, OFF_BC = OFF_BR + QB * LDR * 2, OFF_LSE = OFF_BC + L * LDC * 2,
                       OFF_DD = OFF_LSE + L * 4, OFF_END = OFF_DD + L * 4;
  static constexpr int PART = SP * QB * DHP;                                        // floats of one partial set (dq, dk or dv)
  static constexpr int SMEM = OFF_END > 3 * PART * 4 ? OFF_END : 3 * PART * 4;  // the partials alias the operand region (dead by then)
  static_assert(DHP % 16 == 0 && QB % 16 == 0 && L % QB == 0 && (L / SP) % 16 == 0 && SMEM <= 232448 && NTH <= 1024, "small_core_bwd: shapes");
};
template <int NH, int DHP, int L, int QB, int SP>
__global__ void __launch_bounds__(SB<DHP, L, QB, SP>::NTH, SB<DHP, L, QB, SP>::NTH <= 256 ? 2 : 1) small_core_bwd_k(const bf* __restrict__ DOG,
    const float* __restrict__ O, const bf* __restrict__ OG, const bf* __restrict__ QKVG, const bf* __restrict__ BIAS, const float* __restrict__ LSE,
    bf* __restrict__ DQKVG, float* __restrict__ DB, float* __restrict__ ABQ, int acc_stride, float rsqd) {
  using C = SB<DHP, L, QB, SP>;
  constexpr int W = NH * DHP, LDH = C::LDH, LDR = C::LDR, LDC = C::LDC, NTH = C::NTH, CH = DHP / 8, NS = QB / 16, LS = L / SP;
  constexpr int TPR = NTH >= 4 * L ? 4 : NTH >= 2 * L ? 2 : 1, CPT = DHP / TPR;  // gate pass: TPR threads a row, CPT columns each
  static_assert(CPT % 4 == 0 && NTH == TPR * L, "small_core_bwd: gate columns");
  constexpr long ld4 = 4L * W;
  extern __shared__ __align__(128) uint8_t smem[];
  bf* q_s = reinterpret_cast<bf*>(smem + C::OFF_Q);
  bf* k_s = reinterpret_cast<bf*>(smem + C::OFF_K);
  bf* v_s = reinterpret_cast<bf*>(smem + C::OFF_V);
  bf* do_s = reinterpret_cast<bf*>(smem + C::OFF_DO);
  bf* br_s = reinterpret_cast<bf*>(smem + C::OFF_BR);
  bf* bc_s = reinterpret_cast<bf*>(smem + C::OFF_BC);
  float* lse_s = reinterpret_cast<float*>(smem + C::OFF_LSE);
  float* dd_s = reinterpret_cast<float*>(smem + C::OFF_DD);
  float* part = reinterpret_cast<float*>(smem);  // after the loops: [3][SP][QB][DHP] (dq, dk, dv)
  SBT(0);
  const int h = (int)blockIdx.x % NH, j0 = ((int)blockIdx.x / NH) * QB;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  // the gate pass is one row part per thread (NTH / TPR == L); its forward operands (g, og, own rows' O) are issued first
  const int rr = tid / TPR, gc0 = (tid % TPR) * CPT;
  const bool own = rr >= j0 && rr < j0 + QB;
  float4 og4[CPT / 4], o4[CPT / 4], s4[CPT / 4], dg4[CPT / 4];
#pragma unroll
  for (int c = 0; c < CPT / 4; ++c) {
    s4[c] = V4<bf>::load(QKVG + (long)rr * ld4 + 3 * W + h * DHP + gc0 + 4 * c);
    og4[c] = V4<bf>::load(OG + (long)rr * W + h * DHP + gc0 + 4 * c);
    if (own) o4[c] = V4<float>::load(O + (long)rr * W + h * DHP + gc0 + 4 * c);
  }
  // the forward's tensors (q / k / v, the bias, LSE, O, og) were complete before the dO GEMM ran: loaded before griddepcontrol.wait
  for (int i = tid; i < L * CH; i += NTH) {
    const int r = i / CH, c = i % CH;
    const bf* row = QKVG + (long)r * ld4 + h * DHP + 8 * c;
    cp16(q_s + r * LDH + 8 * c, row);
    cp16(k_s + r * LDH + 8 * c, row + W);
    cp16(v_s + r * LDH + 8 * c, row + 2 * W);
  }
  for (int i = tid; i < QB * (L / 8); i += NTH) {  // bias rows j0 .. j0 + QB (query warps)
    const int r = i / (L / 8), c = i % (L / 8);
    cp16(br_s + r * LDR + 8 * c, BIAS + ((long)h * L + j0 + r) * L + 8 * c);
  }
  for (int i = tid; i < L * (QB / 8); i += NTH) {  // bias columns j0 .. j0 + QB of every row (key warps)
    const int r = i / (QB / 8), c = i % (QB / 8);
    cp16(bc_s + r * LDC + 8 * c, BIAS + ((long)h * L + r) * L + j0 + 8 * c);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  for (int r = tid; r < L; r += NTH) lse_s[r] = LSE[(long)h * L + r];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  SBT(1);
  // gate backward for every row of the head: dO = dog s (bf16 into do_s), D = sum dog og (og = s o, the forward's bf16), dg = dog o s (1 - s)
  // (rows j0 .. only, with o fp32)
  {
#pragma unroll
    for (int c = 0; c < CPT / 4; ++c) dg4[c] = V4<bf>::load(DOG + (long)rr * W + h * DHP + gc0 + 4 * c);
    float dsum = 0.f;
#pragma unroll
    for (int c = 0; c < CPT / 4; ++c) {
      const float4 s = sig4(s4[c]), d_o = mul4(dg4[c], s);
      V4<bf>::store(do_s + rr * LDH + gc0 + 4 * c, d_o);
      dsum += sum4(mul4(dg4[c], og4[c]));
      if (own) {
        const float4 t = mul4(dg4[c], o4[c]);
        const float4 dg = make_float4(t.x * s.x * (1.f - s.x), t.y * s.y * (1.f - s.y), t.z * s.z * (1.f - s.z), t.w * s.w * (1.f - s.w));
        V4<bf>::store(DQKVG + (long)rr * ld4 + 3 * W + h * DHP + gc0 + 4 * c, dg);
      }
    }
#pragma unroll
    for (int x = 1; x < TPR; x <<= 1) dsum += __shfl_xor_sync(0xffffffffu, dsum, x);
    if (tid % TPR == 0) dd_s[rr] = dsum;
  }
  SBT(2);
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  SBT(3);
  constexpr float L2E = 1.4426950408889634f;
  const float cql = rsqd * L2E;
  const bool qw = warp < NS * SP;
  const int wi = qw ? warp : warp - NS * SP, sl = wi / SP, sp = wi % SP, x0 = sp * LS;  // slab, split, the split's first key / query
  float acc0[DHP / 8][4] = {}, acc1[DHP / 8][4] = {};  // query warp: dq (acc0); key warp: dk (acc0), dv (acc1)
  if (qw) {
    // ---- query warp: rows j0 + 16 sl .., keys x0 .. x0 + LS
    const int rb = 16 * sl, r0 = j0 + rb + g;
    uint32_t qa[DHP / 16][4], da[DHP / 16][4];
#pragma unroll
    for (int kk = 0; kk < DHP / 16; ++kk) {
      const bf* ap = q_s + r0 * LDH + 16 * kk + 2 * q;
      qa[kk][0] = *reinterpret_cast<const uint32_t*>(ap); qa[kk][1] = *reinterpret_cast<const uint32_t*>(ap + 8 * LDH);
      qa[kk][2] = *reinterpret_cast<const uint32_t*>(ap + 8); qa[kk][3] = *reinterpret_cast<const uint32_t*>(ap + 8 * LDH + 8);
      const bf* dp = do_s + r0 * LDH + 16 * kk + 2 * q;
      da[kk][0] = *reinterpret_cast<const uint32_t*>(dp); da[kk][1] = *reinterpret_cast<const uint32_t*>(dp + 8 * LDH);
      da[kk][2] = *reinterpret_cast<const uint32_t*>(dp + 8); da[kk][3] = *reinterpret_cast<const uint32_t*>(dp + 8 * LDH + 8);
    }
    const float lse0 = lse_s[r0], lse1 = lse_s[r0 + 8], dd0 = dd_s[r0], dd1 = dd_s[r0 + 8];
    float* db0 = DB + ((long)h * L + r0) * L;
    float* db1 = db0 + 8L * L;
#pragma unroll
    for (int kc_ = 0; kc_ < LS / 16; ++kc_) {
      const int kc = x0 / 16 + kc_;
      uint32_t pa[4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int nt = 2 * kc + hh;
        float s[4] = {}, d[4] = {};
#pragma unroll
        for (int kk = 0; kk < DHP / 16; ++kk) {
          const uint32_t* kp = reinterpret_cast<const uint32_t*>(k_s + (8 * nt + g) * LDH + 16 * kk + 2 * q);
          const uint32_t* vp = reinterpret_cast<const uint32_t*>(v_s + (8 * nt + g) * LDH + 16 * kk + 2 * q);
          mma16816(s, qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], kp[0], kp[4]);
          mma16816(d, da[kk][0], da[kk][1], da[kk][2], da[kk][3], vp[0], vp[4]);
        }
        const float2 b0 = unpack2(*reinterpret_cast<const uint32_t*>(br_s + (rb + g) * LDR + 8 * nt + 2 * q));
        const float2 b1 = unpack2(*reinterpret_cast<const uint32_t*>(br_s + (rb + g + 8) * LDR + 8 * nt + 2 * q));
        const float p0 = sc_ex2(fmaf(s[0], cql, fmaf(b0.x, L2E, -lse0))), p1 = sc_ex2(fmaf(s[1], cql, fmaf(b0.y, L2E, -lse0)));
        const float p2 = sc_ex2(fmaf(s[2], cql, fmaf(b1.x, L2E, -lse1))), p3 = sc_ex2(fmaf(s[3], cql, fmaf(b1.y, L2E, -lse1)));
        const float ds0 = p0 * (d[0] - dd0), ds1 = p1 * (d[1] - dd0), ds2 = p2 * (d[2] - dd1), ds3 = p3 * (d[3] - dd1);
        *reinterpret_cast<float2*>(db0 + 8 * nt + 2 * q) = make_float2(ds0, ds1);
        *reinterpret_cast<float2*>(db1 + 8 * nt + 2 * q) = make_float2(ds2, ds3);
        pa[2 * hh] = pack2(ds0, ds1); pa[2 * hh + 1] = pack2(ds2, ds3);
      }
#pragma unroll
      for (int dn = 0; dn < DHP / 8; ++dn) {
        uint32_t b0, b1;
        ldsm_x2_t(b0, b1, k_s + (16 * kc + (lane & 15)) * LDH + 8 * dn);
        mma16816(acc0[dn], pa[0], pa[1], pa[2], pa[3], b0, b1);
      }
    }
  } else {
    // ---- key warp: keys j0 + 16 sl .., queries x0 .. x0 + LS
    const int kl = 16 * sl, k0 = j0 + kl + g;
    uint32_t ka[DHP / 16][4], va[DHP / 16][4];
#pragma unroll
    for (int kk = 0; kk < DHP / 16; ++kk) {
      const bf* ap = k_s + k0 * LDH + 16 * kk + 2 * q;
      ka[kk][0] = *reinterpret_cast<const uint32_t*>(ap); ka[kk][1] = *reinterpret_cast<const uint32_t*>(ap + 8 * LDH);
      ka[kk][2] = *reinterpret_cast<const uint32_t*>(ap + 8); ka[kk][3] = *reinterpret_cast<const uint32_t*>(ap + 8 * LDH + 8);
      const bf* vp = v_s + k0 * LDH + 16 * kk + 2 * q;
      va[kk][0] = *reinterpret_cast<const uint32_t*>(vp); va[kk][1] = *reinterpret_cast<const uint32_t*>(vp + 8 * LDH);
      va[kk][2] = *reinterpret_cast<const uint32_t*>(vp + 8); va[kk][3] = *reinterpret_cast<const uint32_t*>(vp + 8 * LDH + 8);
    }
    const bf* bcl = bc_s + kl + g;
#pragma unroll
    for (int qc_ = 0; qc_ < LS / 16; ++qc_) {  // 16 queries
      const int qc = x0 / 16 + qc_;
      uint32_t pa[4], sa[4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int nt = 2 * qc + hh, qa0 = 8 * nt + 2 * q;
        float s[4] = {}, d[4] = {};
#pragma unroll
        for (int kk = 0; kk < DHP / 16; ++kk) {
          const uint32_t* qp = reinterpret_cast<const uint32_t*>(q_s + (8 * nt + g) * LDH + 16 * kk + 2 * q);
          const uint32_t* op = reinterpret_cast<const uint32_t*>(do_s + (8 * nt + g) * LDH + 16 * kk + 2 * q);
          mma16816(s, ka[kk][0], ka[kk][1], ka[kk][2], ka[kk][3], qp[0], qp[4]);
          mma16816(d, va[kk][0], va[kk][1], va[kk][2], va[kk][3], op[0], op[4]);
        }
        const float la = lse_s[qa0], lb = lse_s[qa0 + 1], ea = dd_s[qa0], eb = dd_s[qa0 + 1];
        const float b00 = __bfloat162float(bcl[qa0 * LDC]), b01 = __bfloat162float(bcl[(qa0 + 1) * LDC]);
        const float b10 = __bfloat162float(bcl[qa0 * LDC + 8]), b11 = __bfloat162float(bcl[(qa0 + 1) * LDC + 8]);
        const float p0 = sc_ex2(fmaf(s[0], cql, fmaf(b00, L2E, -la))), p1 = sc_ex2(fmaf(s[1], cql, fmaf(b01, L2E, -lb)));
        const float p2 = sc_ex2(fmaf(s[2], cql, fmaf(b10, L2E, -la))), p3 = sc_ex2(fmaf(s[3], cql, fmaf(b11, L2E, -lb)));
        pa[2 * hh] = pack2(p0, p1); pa[2 * hh + 1] = pack2(p2, p3);
        sa[2 * hh] = pack2(p0 * (d[0] - ea), p1 * (d[1] - eb)); sa[2 * hh + 1] = pack2(p2 * (d[2] - ea), p3 * (d[3] - eb));
      }
#pragma unroll
      for (int dn = 0; dn < DHP / 8; ++dn) {
        uint32_t b0, b1;
        ldsm_x2_t(b0, b1, do_s + (16 * qc + (lane & 15)) * LDH + 8 * dn);
        mma16816(acc1[dn], pa[0], pa[1], pa[2], pa[3], b0, b1);
        ldsm_x2_t(b0, b1, q_s + (16 * qc + (lane & 15)) * LDH + 8 * dn);
        mma16816(acc0[dn], sa[0], sa[1], sa[2], sa[3], b0, b1);
      }
    }
  }
  SBT(4);
  __syncthreads();  // every operand read: the partials overwrite them
  {
    float* p0 = part + (qw ? 0 : C::PART) + (sp * QB + 16 * sl) * DHP;  // [set][sp][row][col]
    float* p1 = part + 2 * C::PART + (sp * QB + 16 * sl) * DHP;
#pragma unroll
    for (int dn = 0; dn < DHP / 8; ++dn) {
      const int col = 8 * dn + 2 * q;
      *reinterpret_cast<float2*>(p0 + g * DHP + col) = make_float2(acc0[dn][0], acc0[dn][1]);
      *reinterpret_cast<float2*>(p0 + (g + 8) * DHP + col) = make_float2(acc0[dn][2], acc0[dn][3]);
      if (!qw) {
        *reinterpret_cast<float2*>(p1 + g * DHP + col) = make_float2(acc1[dn][0], acc1[dn][1]);
        *reinterpret_cast<float2*>(p1 + (g + 8) * DHP + col) = make_float2(acc1[dn][2], acc1[dn][3]);
      }
    }
  }
  __syncthreads();
  // the split partials in split order; dQ / dK x rsqd, bf16 (qkv_bwd's rounding). dQ: a thread per (4-row group, column pair) -- the
  // group's rounded values summed in row order for dbq (qkv_bwd's acc rows: row = query / 4) in the same pass
  for (int i = tid; i < (QB / 4) * (DHP / 2); i += NTH) {
    const int b = i / (DHP / 2), c = 2 * (i % (DHP / 2));
    float2 cs = make_float2(0.f, 0.f);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const int r = 4 * b + w;
      const float* pp = part + r * DHP + c;
      float2 t = *reinterpret_cast<const float2*>(pp);
#pragma unroll
      for (int s = 1; s < SP; ++s) {
        const float2 u = *reinterpret_cast<const float2*>(pp + s * QB * DHP);
        t.x += u.x; t.y += u.y;
      }
      const uint32_t u = pack2(t.x * rsqd, t.y * rsqd);
      *reinterpret_cast<uint32_t*>(DQKVG + (long)(j0 + r) * ld4 + h * DHP + c) = u;
      const float2 f = unpack2(u);
      cs.x += f.x; cs.y += f.y;
    }
    *reinterpret_cast<float2*>(ABQ + (long)(j0 / 4 + b) * acc_stride + h * DHP + c) = cs;
  }
  for (int i = tid; i < 2 * QB * (DHP / 2); i += NTH) {
    const int set = 1 + i / (QB * (DHP / 2)), e = i % (QB * (DHP / 2)), r = e / (DHP / 2), c = 2 * (e % (DHP / 2));
    const float* pp = part + set * C::PART + r * DHP + c;
    float2 t = *reinterpret_cast<const float2*>(pp);
#pragma unroll
    for (int s = 1; s < SP; ++s) {
      const float2 u = *reinterpret_cast<const float2*>(pp + s * QB * DHP);
      t.x += u.x; t.y += u.y;
    }
    if (set == 1) { t.x *= rsqd; t.y *= rsqd; }
    *reinterpret_cast<uint32_t*>(DQKVG + (long)(j0 + r) * ld4 + set * W + h * DHP + c) = pack2(t.x, t.y);
  }
  SBT(5);
}
template <int NH, int DHP, int L, int QB>
void sbwd_t(const at::Tensor& dog, const at::Tensor& o, const at::Tensor& og, const at::Tensor& qkvg, const at::Tensor& bias, const at::Tensor& lse,
            at::Tensor& dqkvg, at::Tensor& db, float* abq, int stride, float rsqd, bool pdl) {
  constexpr int SP = NH > 16 && L == 256 && QB == 32 ? 2 : 4;  // 24 heads at L 256: 192 CTAs, two a SM (256 threads each)
  using C = SB<DHP, L, QB, SP>;
  auto k = small_core_bwd_k<NH, DHP, L, QB, SP>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NH * (L / QB));
  cfg.blockDim = dim3(C::NTH);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = S();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, CP<bf>(dog), CP<float>(o), CP<bf>(og), CP<bf>(qkvg), CP<bf>(bias), CP<float>(lse), P<bf>(dqkvg), P<float>(db),
                                    abq, stride, rsqd));
}
// v27: dog [L, W] bf16 (dy Wo), o [L, W] fp32, og [L, W] bf16 (sigmoid(g) o, saved by the forward), qkvg [L, 4 W] bf16 (q unscaled), bias
// [H, L, L] bf16 (natural units), lse [H, L] fp32 (log2) -> dqkvg [L, 4 W] bf16 (dq | dk | dv | dg), db [H, L, L] fp32 (dbias),
// acc[0 : L / 4, 2 d : 2 d + W] (dbq's 4-row partial sums). L 128 / 256; qb 16 / 32 query (and key) rows per CTA.
void small_core_bwd(at::Tensor dog, at::Tensor o, at::Tensor og, at::Tensor qkvg, at::Tensor bias, at::Tensor lse, at::Tensor dqkvg, at::Tensor db,
                    at::Tensor acc, int64_t d, int64_t L, double rsqd, bool pdl, int64_t qb) {
  chk(dog, at::kBFloat16, "dog"); chk(o, at::kFloat, "o"); chk(og, at::kBFloat16, "og"); chk(qkvg, at::kBFloat16, "qkvg"); chk(bias, at::kBFloat16, "bias");
  chk(lse, at::kFloat, "lse"); chk(dqkvg, at::kBFloat16, "dqkvg"); chk(db, at::kFloat, "db"); chk(acc, at::kFloat, "acc");
  const int64_t H = bias.size(0), W = o.size(1);
  TORCH_CHECK((L == 128 || L == 256) && (qb == 16 || qb == 32) && qkvg.size(0) == L && qkvg.size(1) == 4 * W && dqkvg.sizes() == qkvg.sizes()
              && dog.numel() == L * W && o.numel() == L * W && og.numel() == L * W && bias.numel() == H * L * L && db.numel() == H * L * L
              && lse.numel() == H * L && acc.dim() == 2 && acc.size(0) >= L / 4 && acc.size(1) >= 2 * d + W, "small_core_bwd: shapes");
  const at::cuda::CUDAGuard gd(o.device());
  float* abq = acc.data_ptr<float>() + 2 * d;
  const int st = (int)acc.size(1);
  const float rs = (float)rsqd;
#define SB_GO(NH, DHP)                                                                                            \
  do {                                                                                                            \
    if (L == 128 && qb == 16) sbwd_t<NH, DHP, 128, 16>(dog, o, og, qkvg, bias, lse, dqkvg, db, abq, st, rs, pdl); \
    else if (L == 128) sbwd_t<NH, DHP, 128, 32>(dog, o, og, qkvg, bias, lse, dqkvg, db, abq, st, rs, pdl);       \
    else if (qb == 16) sbwd_t<NH, DHP, 256, 16>(dog, o, og, qkvg, bias, lse, dqkvg, db, abq, st, rs, pdl);       \
    else sbwd_t<NH, DHP, 256, 32>(dog, o, og, qkvg, bias, lse, dqkvg, db, abq, st, rs, pdl);                      \
  } while (0)
  if (H == 16 && W == 512) SB_GO(16, 32);
  else if (H == 12 && W == 384) SB_GO(12, 32);
  else if (H == 24 && W == 384) SB_GO(24, 16);
  else if (H == 8 && W == 384) SB_GO(8, 48);
  else TORCH_CHECK(false, "small_core_bwd: geometry");
#undef SB_GO
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("small_core", &small_core);
  m.def("small_core_train", &small_core_train);
  m.def("dqb_small", &dqb_small);
  m.def("small_core_bwd", &small_core_bwd);
#if SBW_TL
  m.def("sb_tl", []() { std::vector<unsigned long long> v(512 * 8); cudaMemcpyFromSymbol(v.data(), g_sbtl, sizeof(unsigned long long) * 512 * 8); return v; });
#endif
#if FR_TL
  m.def("fr_tl", []() { std::vector<unsigned long long> v(1024 * 8); cudaMemcpyFromSymbol(v.data(), g_frtl, sizeof(unsigned long long) * 1024 * 8); return v; });
  m.def("fr_tl_clear", []() { std::vector<unsigned long long> v(1024 * 8, 0ull); cudaMemcpyToSymbol(g_frtl, v.data(), sizeof(unsigned long long) * 1024 * 8); });
#endif
#if SC_TL
  m.def("sc_tl", []() { std::vector<unsigned long long> v(128 * 8); cudaMemcpyFromSymbol(v.data(), g_sctl, sizeof(unsigned long long) * 128 * 8); return v; });
#endif
  m.def("planes_fix", &planes_fix);
  m.def("sm_reserve", [](int64_t k) { const int was = g_sm_reserve; g_sm_reserve = (int)k; return was; });
  m.def("ksplit_combine", &ksplit_combine);
  apb_shared_defs(m);
  m.def("prep_dp", &prep_dp);
  m.def("prep_ln", &prep_ln);
  m.def("pair_bias32_tc", &pair_bias32_tc);
  m.def("pack2_rna", &pack2_rna);
  m.def("gate32", &gate32);
  // the shared apb_rows.cu (3.0.6) now also defines a 7-argument gate32_bwd; pick this file's 9-argument overload (pybind overloads by arity)
  m.def("gate32_bwd", static_cast<void (*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t)>(&gate32_bwd));
  m.def("pair_bias_bwd_wide1", &pair_bias_bwd_wide1);
  m.def("pair_bwd_wide1_blocks", &pair_bwd_wide1_blocks);
  m.def("finalize_red", &finalize_red);
  m.def("finalize_red_nc", &finalize_red_nc);
  m.def("finalize_part", &finalize_part);
  m.def("ksplit_combine32", &ksplit_combine32);
  m.def("gate_bwd2", &gate_bwd2);
  m.def("inf_front", &inf_front);
  m.def("pdl", [](int64_t k) { const int was = g_pdl; g_pdl = (int)k; return was; });
  m.def("front_tmw", [](int64_t k) { const int was = g_front_tmw; g_front_tmw = (int)k; return was; });
  m.def("gate_rows_p", &gate_rows_p);
  m.def("gate_bwd_p", &gate_bwd_p);
  m.def("qkv_bwd_p", &qkv_bwd_p);
  m.def("ln_bwd_p", &ln_bwd_p);
  m.def("pair_bias_fwd2", &pair_bias_fwd2);
  m.def("prep_mb", [](int64_t k) { const int was = g_prep_mb; g_prep_mb = (int)k; return was; });
}
