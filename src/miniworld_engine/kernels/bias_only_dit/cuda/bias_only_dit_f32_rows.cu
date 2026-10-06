// Bias-only token DiT fp32 row kernels (CUDA): the passes between the GEMMs of the fp32 (TF32) path, inference and training.
// The bf16 path's kernels (bias_only_dit_rows.cu, bias_only_dit_train_rows.cu) keep bf16 intermediates; these are the same
// arithmetic with every activation, table and gradient fp32. Inference reuses the bf16 extension's dtype-generic rows (ln_rows,
// adaln_in_rows, resgate_*_rows take fp32), training its dtype-generic unfold / finalize; this file holds the rest:
//
//   inference + training  swiglu         h = silu(a) b, ab = [a | b]
//                         softmax_rows   p = softmax(bias row) over the keys (masked keys at the largest negative float: a fully
//                                        masked row is uniform, as the torch reference's finfo.min fill makes it)
//                         pair_bias      bias [H, R] = LN(pair) Wf^T and the row statistics (R = L^2 pair rows of 128): exact fp32 on
//                                        the FMA pipe (128 x H products per row; the pair stays in shared memory)
//   training forward      softmax_t      softmax_rows that also writes P^T (the backward's dV = P^T dO reads it as the K-major A)
//                         cond_ln, adaln_a, res_adaln_b, res_c   (as bias_only_dit_train_rows.cu, fp32 tables and outputs)
//   training backward     res_c_bwd, swiglu_bwd, res_adaln_b_bwd, gate_bwd, adaln_a_bwd, cond_bwd,
//                         pair_bias_bwd  d pair and per-block dWf partials (LN(pair) rebuilt from the saved statistics)
//
// Row layouts as in the bf16 training rows: the 768-wide rows take two warps (64 threads; lane l of warp half hw owns float4 chunks
// (3 hw + j) 32 + l, j < 3), the 384-wide conditioning rows one warp; persistent blocks walk the rows, per-column gradient sums
// accumulate per block and leave as one row of a [partial_rows, 768] buffer (finalize of the bf16 extension sums them). Every
// kernel is compiled for at most 128 registers (__launch_bounds__(256, 2) or smaller blocks).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"

namespace {
using namespace tdr;

constexpr int D = 768, DC = 384, NCC = 3, WPB = 8, RPB2 = 4;

__device__ __forceinline__ float sg(float v) { return 1.f / (1.f + __expf(-v)); }
__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 mul4(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ __forceinline__ float4 sig4(float4 a) { return make_float4(sg(a.x), sg(a.y), sg(a.z), sg(a.w)); }
__device__ __forceinline__ float sum4(float4 a) { return (a.x + a.y) + (a.z + a.w); }
__device__ __forceinline__ float4 dsig4(float4 v, float4 s) {   // v * s * (1 - s)
  return make_float4(v.x * s.x * (1.f - s.x), v.y * s.y * (1.f - s.y), v.z * s.z * (1.f - s.z), v.w * s.w * (1.f - s.w));
}
__device__ __forceinline__ float4 zero4() { return make_float4(0.f, 0.f, 0.f, 0.f); }
__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ float4 xhat4(float4 x, float mean, float rstd) {
  return make_float4((x.x - mean) * rstd, (x.y - mean) * rstd, (x.z - mean) * rstd, (x.w - mean) * rstd);
}
// the attention residual x1 = x + sigmoid(g1) y, rebuilt wherever it is needed (one fused multiply-add: the same bits everywhere)
__device__ __forceinline__ float4 resid4(float4 x, float4 sg1, float4 y) {
  return make_float4(__fmaf_rn(sg1.x, y.x, x.x), __fmaf_rn(sg1.y, y.y, x.y), __fmaf_rn(sg1.z, y.z, x.z), __fmaf_rn(sg1.w, y.w, x.w));
}

#define ROWS_BEGIN                                                                                   \
  const int lane = threadIdx.x % 32;                                                                 \
  const long wid = (long)blockIdx.x * WPB + threadIdx.x / 32, nw = (long)gridDim.x * WPB;
#define COL(j) (((j) * 32 + lane) * 4)
#define ROWS2_BEGIN                                                                                  \
  const int lane = threadIdx.x % 32, hw = (threadIdx.x / 32) & 1, slot = threadIdx.x / 64;           \
  const long sid = (long)blockIdx.x * RPB2 + slot, ns = (long)gridDim.x * RPB2;                     \
  __shared__ float2 xr_[2][RPB2][2];                                                                 \
  int par_ = 0;
#define COL2(j) ((((hw) * 3 + (j)) * 32 + lane) * 4)
#define ROWSUM2(a, b) row_sum2(make_float2((a), (b)), xr_, par_, slot, hw, lane)

__device__ __forceinline__ float2 row_sum2(float2 v, float2 (&xr)[2][RPB2][2], int& par, int slot, int hw, int lane) {
  v.x = warp_sum(v.x); v.y = warp_sum(v.y);
  if (lane == 0) xr[par][slot][hw] = v;
  asm volatile("bar.sync %0, 64;" :: "r"(1 + slot) : "memory");
  const float2 a = xr[par][slot][0], b = xr[par][slot][1];
  par ^= 1;
  return make_float2(a.x + b.x, a.y + b.y);
}

// LayerNorm statistics of a 768-row held as 12 values per thread of the slot's two warps, in one exchange (pairwise Chan merges:
// equal counts at every level). Returns (mean, rstd).
__device__ __forceinline__ float2 row_stats2(const float4 (&x)[3], float eps, float2 (&xr)[2][RPB2][2], int& par, int slot, int hw,
                                             int lane) {
  float m = (sum4(x[0]) + sum4(x[1]) + sum4(x[2])) * (1.f / 12);
  float q = 0.f;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const float4 d = make_float4(x[j].x - m, x[j].y - m, x[j].z - m, x[j].w - m);
    q += sum4(mul4(d, d));
  }
  float n = 12.f;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float mo = __shfl_xor_sync(0xffffffffu, m, o), qo = __shfl_xor_sync(0xffffffffu, q, o), dl = mo - m;
    q = q + qo + dl * dl * (0.5f * n);
    m = 0.5f * (m + mo);
    n *= 2.f;
  }
  if (lane == 0) xr[par][slot][hw] = make_float2(m, q);
  asm volatile("bar.sync %0, 64;" :: "r"(1 + slot) : "memory");
  const float2 a = xr[par][slot][0], b = xr[par][slot][1];
  par ^= 1;
  const float dl = b.x - a.x;
  return make_float2(0.5f * (a.x + b.x), rsqrtf((a.y + b.y + dl * dl * (0.5f * n)) * (1.f / D) + eps));
}

// the block's four row slots add their column sums in shared memory; the block leaves one row of the partial buffer
__device__ __forceinline__ void block_partial2(const float4 (&acc)[3], float (&red)[RPB2][768], float* part) {
  const int lane = threadIdx.x % 32, hw = (threadIdx.x / 32) & 1, slot = threadIdx.x / 64;
#pragma unroll
  for (int j = 0; j < 3; ++j) st4(&red[slot][COL2(j)], acc[j]);
  __syncthreads();
  for (int c = threadIdx.x; c < 768; c += blockDim.x) part[(long)blockIdx.x * 768 + c] = (red[0][c] + red[1][c]) + (red[2][c] + red[3][c]);
  __syncthreads();
}
__device__ __forceinline__ void block_partial_smem(float4 (&acc)[RPB2][D / 4], float* part) {
  __syncthreads();
  for (int c = threadIdx.x; c < D / 4; c += blockDim.x)
    st4(part + (long)blockIdx.x * D + 4 * c, add4(add4(acc[0][c], acc[1][c]), add4(acc[2][c], acc[3][c])));
}

// ------------------------------------------------------------------------------------------------------------- forward
__global__ void __launch_bounds__(WPB * 32, 2) cond_ln_k(const float* __restrict__ C, float* __restrict__ CHAT,
    float2* __restrict__ CST, long M, float eps) {
  ROWS_BEGIN
  for (long r = wid; r < M; r += nw) {
    float4 c[NCC];
#pragma unroll
    for (int j = 0; j < NCC; ++j) c[j] = ld4(C + r * DC + COL(j));
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < NCC; ++j) s += sum4(c[j]);
    const float mean = warp_sum(s) * (1.f / DC);
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < NCC; ++j) {
      const float4 d = make_float4(c[j].x - mean, c[j].y - mean, c[j].z - mean, c[j].w - mean);
      q += sum4(mul4(d, d));
    }
    const float rstd = rsqrtf(warp_sum(q) * (1.f / DC) + eps);
#pragma unroll
    for (int j = 0; j < NCC; ++j) st4(CHAT + r * DC + COL(j), xhat4(c[j], mean, rstd));
    if (lane == 0) CST[r] = make_float2(mean, rstd);
  }
}

// xa = sigmoid(G[:, 0:D] + bs1) LN(x) + G[:, D:2D]
__global__ void __launch_bounds__(256, 2) adaln_a_k(const float* __restrict__ X, const float* __restrict__ G, long sg_,
    const float* __restrict__ BS, float* __restrict__ XA, float2* __restrict__ XST, long M, float eps) {
  ROWS2_BEGIN
  for (long r = sid; r < M; r += ns) {
    float4 x[3], s[3], sh[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      x[j] = ld4(X + r * D + COL2(j)); s[j] = ld4(G + r * sg_ + COL2(j)); sh[j] = ld4(G + r * sg_ + D + COL2(j));
    }
    const float2 ms = row_stats2(x, eps, xr_, par_, slot, hw, lane);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float4 sj = sig4(add4(s[j], ld4(BS + COL2(j)))), xh = xhat4(x[j], ms.x, ms.y);
      st4(XA + r * D + COL2(j), add4(mul4(sj, xh), sh[j]));
    }
    if (hw == 0 && lane == 0) XST[r] = ms;
  }
}

// x1 = x + sigmoid(Gg[:, 0:D] + bg1) y;  xt = sigmoid(G[:, 2D:3D] + bs2) LN(x1) + G[:, 3D:4D]
__global__ void __launch_bounds__(256, 2) res_adaln_b_k(const float* __restrict__ X, const float* __restrict__ Y,
    const float* __restrict__ GG, long sgg, const float* __restrict__ BG1, const float* __restrict__ G, long sg_,
    const float* __restrict__ BS2, float* __restrict__ XT_, float2* __restrict__ X1ST, long M, float eps) {
  ROWS2_BEGIN
  for (long r = sid; r < M; r += ns) {
    float4 x[3], s[3], sh[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float4 xi = ld4(X + r * D + COL2(j)), y = ld4(Y + r * D + COL2(j)), g = ld4(GG + r * sgg + COL2(j));
      x[j] = resid4(xi, sig4(add4(g, ld4(BG1 + COL2(j)))), y);
      s[j] = ld4(G + r * sg_ + 2 * D + COL2(j)); sh[j] = ld4(G + r * sg_ + 3 * D + COL2(j));
    }
    const float2 ms = row_stats2(x, eps, xr_, par_, slot, hw, lane);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float4 sj = sig4(add4(s[j], ld4(BS2 + COL2(j)))), xh = xhat4(x[j], ms.x, ms.y);
      st4(XT_ + r * D + COL2(j), add4(mul4(sj, xh), sh[j]));
    }
    if (hw == 0 && lane == 0) X1ST[r] = ms;
  }
}

// out = x1 + sigmoid(Gg[:, D:2D] + bg2) z
__global__ void __launch_bounds__(256, 2) res_c_k(const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ Z,
    const float* __restrict__ GG, long sgg, const float* __restrict__ BG1, const float* __restrict__ BG2, float* __restrict__ OUT, long M) {
  ROWS2_BEGIN
  (void)par_; (void)xr_;
  for (long r = sid; r < M; r += ns) {
    float4 x[3], y[3], z[3], g1[3], g2[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      x[j] = ld4(X + r * D + COL2(j)); y[j] = ld4(Y + r * D + COL2(j)); z[j] = ld4(Z + r * D + COL2(j));
      g1[j] = ld4(GG + r * sgg + COL2(j)); g2[j] = ld4(GG + r * sgg + D + COL2(j));
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float4 x1 = resid4(x[j], sig4(add4(g1[j], ld4(BG1 + COL2(j)))), y[j]);
      st4(OUT + r * D + COL2(j), add4(x1, mul4(sig4(add4(g2[j], ld4(BG2 + COL2(j)))), z[j])));
    }
  }
}

// ------------------------------------------------------------------------------------------------------------ backward
__global__ void __launch_bounds__(256, 2) res_c_bwd_k(const float* __restrict__ DOUT, const float* __restrict__ Z, const float* __restrict__ GG,
    long sgg, const float* __restrict__ BG2, float* __restrict__ DZ, float* __restrict__ DGG, long sdg, float* __restrict__ PG2, long M) {
  ROWS2_BEGIN
  (void)par_; (void)xr_;
  __shared__ float red[RPB2][768];
  float4 acc[3] = {zero4(), zero4(), zero4()};
  for (long r = sid; r < M; r += ns) {
    float4 dout[3], z[3], g[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      dout[j] = ld4(DOUT + r * D + COL2(j)); z[j] = ld4(Z + r * D + COL2(j)); g[j] = ld4(GG + r * sgg + D + COL2(j));
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float4 s = sig4(add4(g[j], ld4(BG2 + COL2(j))));
      st4(DZ + r * D + COL2(j), mul4(dout[j], s));
      const float4 dg = dsig4(mul4(dout[j], z[j]), s);
      st4(DGG + r * sdg + D + COL2(j), dg);
      acc[j] = add4(acc[j], dg);
    }
  }
  block_partial2(acc, red, PG2);
}

__global__ void __launch_bounds__(256, 2) res_adaln_b_bwd_k(const float* __restrict__ DOUT, const float* __restrict__ DXT,
    const float* __restrict__ X, const float2* __restrict__ X1ST, const float* __restrict__ G, long sg_, const float* __restrict__ BS2,
    const float* __restrict__ GG, long sgg, const float* __restrict__ BG1, const float* __restrict__ Y, float* __restrict__ DX1,
    float* __restrict__ DY, float* __restrict__ DG, long sdg, float* __restrict__ DGG, long sdgg, float* __restrict__ PS2,
    float* __restrict__ PG1, long M) {
  ROWS2_BEGIN
  // the column sums accumulate in shared memory (each thread its own slot's columns; no barrier until the end): registers stay
  // below 128
  __shared__ float4 acc_s[RPB2][D / 4], acc_g[RPB2][D / 4];
#pragma unroll
  for (int j = 0; j < 3; ++j) { acc_s[slot][COL2(j) / 4] = zero4(); acc_g[slot][COL2(j) / 4] = zero4(); }
  for (long r = sid; r < M; r += ns) {
    float4 dxt[3], xh[3], s2[3], dout[3], g1[3], y[3];
    const float2 st = X1ST[r];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      dxt[j] = ld4(DXT + r * D + COL2(j));
      xh[j] = ld4(X + r * D + COL2(j));
      s2[j] = ld4(G + r * sg_ + 2 * D + COL2(j));
      dout[j] = ld4(DOUT + r * D + COL2(j)); g1[j] = ld4(GG + r * sgg + COL2(j)); y[j] = ld4(Y + r * D + COL2(j));
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {                                      // x1 rebuilt, then normalised; g1 becomes sigmoid(g1)
      g1[j] = sig4(add4(g1[j], ld4(BG1 + COL2(j))));
      xh[j] = xhat4(resid4(xh[j], g1[j], y[j]), st.x, st.y);
    }
    float m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      s2[j] = sig4(add4(s2[j], ld4(BS2 + COL2(j))));
      st4(DG + r * sdg + 3 * D + COL2(j), dxt[j]);
      const float4 ds2 = dsig4(mul4(dxt[j], xh[j]), s2[j]);
      st4(DG + r * sdg + 2 * D + COL2(j), ds2);
      acc_s[slot][COL2(j) / 4] = add4(acc_s[slot][COL2(j) / 4], ds2);
      dxt[j] = mul4(dxt[j], s2[j]);                                  // d xhat
      m1 += sum4(dxt[j]);
      m2 += sum4(mul4(dxt[j], xh[j]));
    }
    const float2 m = ROWSUM2(m1, m2);
    m1 = m.x * (1.f / D); m2 = m.y * (1.f / D);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float4 dx1 = make_float4(dout[j].x + st.y * (dxt[j].x - m1 - xh[j].x * m2), dout[j].y + st.y * (dxt[j].y - m1 - xh[j].y * m2),
                                     dout[j].z + st.y * (dxt[j].z - m1 - xh[j].z * m2), dout[j].w + st.y * (dxt[j].w - m1 - xh[j].w * m2));
      st4(DX1 + r * D + COL2(j), dx1);
      st4(DY + r * D + COL2(j), mul4(dx1, g1[j]));
      const float4 dg1 = dsig4(mul4(dx1, y[j]), g1[j]);
      st4(DGG + r * sdgg + COL2(j), dg1);
      acc_g[slot][COL2(j) / 4] = add4(acc_g[slot][COL2(j) / 4], dg1);
    }
  }
  block_partial_smem(acc_s, PS2);
  block_partial_smem(acc_g, PG1);
}

__global__ void __launch_bounds__(256, 2) adaln_a_bwd_k(const float* __restrict__ DXA, const float* __restrict__ X, const float2* __restrict__ XST,
    const float* __restrict__ G, long sg_, const float* __restrict__ BS1, const float* __restrict__ DX1, float* __restrict__ DX,
    float* __restrict__ DG, long sdg, float* __restrict__ PS1, long M) {
  ROWS2_BEGIN
  __shared__ float red[RPB2][768];
  float4 acc[3] = {zero4(), zero4(), zero4()};
  for (long r = sid; r < M; r += ns) {
    float4 dxa[3], xh[3], s[3], dx1[3];
    const float2 st = XST[r];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      dxa[j] = ld4(DXA + r * D + COL2(j));
      xh[j] = xhat4(ld4(X + r * D + COL2(j)), st.x, st.y);
      s[j] = ld4(G + r * sg_ + COL2(j));
      dx1[j] = ld4(DX1 + r * D + COL2(j));
    }
    float m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      s[j] = sig4(add4(s[j], ld4(BS1 + COL2(j))));
      st4(DG + r * sdg + D + COL2(j), dxa[j]);
      const float4 ds1 = dsig4(mul4(dxa[j], xh[j]), s[j]);
      st4(DG + r * sdg + COL2(j), ds1);
      acc[j] = add4(acc[j], ds1);
      dxa[j] = mul4(dxa[j], s[j]);                                   // d xhat
      m1 += sum4(dxa[j]);
      m2 += sum4(mul4(dxa[j], xh[j]));
    }
    const float2 m = ROWSUM2(m1, m2);
    m1 = m.x * (1.f / D); m2 = m.y * (1.f / D);
#pragma unroll
    for (int j = 0; j < 3; ++j)
      st4(DX + r * D + COL2(j), make_float4(dx1[j].x + st.y * (dxa[j].x - m1 - xh[j].x * m2), dx1[j].y + st.y * (dxa[j].y - m1 - xh[j].y * m2),
                                            dx1[j].z + st.y * (dxa[j].z - m1 - xh[j].z * m2), dx1[j].w + st.y * (dxa[j].w - m1 - xh[j].w * m2)));
  }
  block_partial2(acc, red, PS1);
}

__global__ void __launch_bounds__(WPB * 32, 2) cond_bwd_k(const float* __restrict__ DCHAT, const float* __restrict__ DCG,
    const float* __restrict__ C, const float2* __restrict__ CST, float* __restrict__ DCO, long M) {
  ROWS_BEGIN
  for (long r = wid; r < M; r += nw) {
    float4 dch[NCC], ch[NCC], dcg[NCC];
    const float2 st = CST[r];
#pragma unroll
    for (int j = 0; j < NCC; ++j) {
      dch[j] = ld4(DCHAT + r * DC + COL(j)); dcg[j] = ld4(DCG + r * DC + COL(j));
      ch[j] = xhat4(ld4(C + r * DC + COL(j)), st.x, st.y);
    }
    float m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int j = 0; j < NCC; ++j) { m1 += sum4(dch[j]); m2 += sum4(mul4(dch[j], ch[j])); }
    m1 = warp_sum(m1) * (1.f / DC);
    m2 = warp_sum(m2) * (1.f / DC);
#pragma unroll
    for (int j = 0; j < NCC; ++j)
      st4(DCO + r * DC + COL(j), make_float4(st.y * (dch[j].x - m1 - ch[j].x * m2) + dcg[j].x, st.y * (dch[j].y - m1 - ch[j].y * m2) + dcg[j].y,
                                             st.y * (dch[j].z - m1 - ch[j].z * m2) + dcg[j].z, st.y * (dch[j].w - m1 - ch[j].w * m2) + dcg[j].w));
  }
}

// do = da sigmoid(g), dg = da ao (1 - sigmoid(g)) (ao = sigmoid(g) o, the forward's gated output), dd[a, h, i] = sum over head h's
// channels of da ao. One warp per row of 768 (16 x 48, 24 x 32, 12 x 64) or 1024 (16 x 64) channels; lane l owns the float4 chunks
// l, l + 32, ...; the warp's chunk sums go through shared memory and lanes 0 .. nh - 1 add their head's.
template <int NCH>
__global__ void __launch_bounds__(WPB * 32, 2) gate_bwd_k(const float* __restrict__ DA, const float* __restrict__ AO,
    const float* __restrict__ G, float* __restrict__ DO, float* __restrict__ DG, float* __restrict__ DD, long R, int L, long gstride,
    long dgstride, int nh) {
  constexpr int W = NCH * 128, NQ = NCH * 32;
  __shared__ float hs[WPB][NQ];
  const long row = (long)blockIdx.x * WPB + threadIdx.x / 32;
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
  if (row >= R) return;
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    const int c = (j * 32 + lane) * 4;
    const float4 da = ld4(DA + row * W + c), ao = ld4(AO + row * W + c), s = sig4(ld4(G + row * gstride + c));
    st4(DO + row * W + c, mul4(da, s));
    st4(DG + row * dgstride + c, make_float4(da.x * ao.x * (1.f - s.x), da.y * ao.y * (1.f - s.y), da.z * ao.z * (1.f - s.z),
                                             da.w * ao.w * (1.f - s.w)));
    hs[w][j * 32 + lane] = sum4(mul4(da, ao));
  }
  __syncwarp();
  if (lane < nh) {
    const int qph = NQ / nh;
    float t = 0.f;
    for (int k = 0; k < qph; ++k) t += hs[w][lane * qph + k];
    DD[((row / L) * nh + lane) * L + row % L] = t;
  }
}

// ------------------------------------------------------------------------------------------------------------ SwiGLU
// h = silu(a) b over [M, N] (ab = [a | b] rows of row stride sab >= 2N), four elements per thread
__global__ void __launch_bounds__(256, 2) swiglu_k(const float* __restrict__ AB, float* __restrict__ H, long M, int N, long sab) {
  const int nc = N / 4;
  const long i = (long)blockIdx.x * 256 + threadIdx.x;
  if (i >= M * nc) return;
  const long r = i / nc;
  const int c = (int)(i - r * nc) * 4;
  const float4 a = ld4(AB + r * sab + c), b = ld4(AB + r * sab + N + c);
  st4(H + r * N + c, make_float4(a.x * sg(a.x) * b.x, a.y * sg(a.y) * b.y, a.z * sg(a.z) * b.z, a.w * sg(a.w) * b.w));
}

// da = dh b s (1 + a (1 - s)), db = dh a s (s = sigmoid(a))
__global__ void __launch_bounds__(256, 2) swiglu_bwd_k(const float* __restrict__ DH, const float* __restrict__ AB, float* __restrict__ DAB,
    long M, int N) {
  const int nc = N / 4;
  const long i = (long)blockIdx.x * 256 + threadIdx.x;
  if (i >= M * nc) return;
  const long r = i / nc;
  const int c = (int)(i - r * nc) * 4;
  const float4 dh = ld4(DH + r * N + c), a = ld4(AB + r * 2 * N + c), b = ld4(AB + r * 2 * N + N + c), s = sig4(a);
  st4(DAB + r * 2 * N + c, make_float4(dh.x * b.x * s.x * (1.f + a.x * (1.f - s.x)), dh.y * b.y * s.y * (1.f + a.y * (1.f - s.y)),
                                       dh.z * b.z * s.z * (1.f + a.z * (1.f - s.z)), dh.w * b.w * s.w * (1.f + a.w * (1.f - s.w))));
  st4(DAB + r * 2 * N + N + c, mul4(mul4(dh, a), s));
}

// ------------------------------------------------------------------------------------------------------------ softmax
// One warp per row of L keys (L a multiple of 128 here), float4 chunks lane, lane + 32, ... (CPL per lane); in place allowed
template <int CPL>
__device__ __forceinline__ void softmax_row(const float* __restrict__ b, const bool* __restrict__ mask, int L, int lane, float4 (&v)[CPL]) {
  constexpr float NEG = -3.0e38f;
  const int nch = L / 4;
  float mx = NEG;
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    const int ch = lane + 32 * c;
    v[c] = zero4();
    if (ch < nch) {
      float4 e = ld4(b + ch * 4);
      if (mask) {
        const uchar4 mk = *reinterpret_cast<const uchar4*>(mask + ch * 4);
        if (!mk.x) e.x = NEG;
        if (!mk.y) e.y = NEG;
        if (!mk.z) e.z = NEG;
        if (!mk.w) e.w = NEG;
      }
      v[c] = e;
      mx = fmaxf(mx, fmaxf(fmaxf(e.x, e.y), fmaxf(e.z, e.w)));
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sum = 0.f;
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    if (lane + 32 * c < nch) {
      v[c] = make_float4(__expf(v[c].x - mx), __expf(v[c].y - mx), __expf(v[c].z - mx), __expf(v[c].w - mx));
      sum += sum4(v[c]);
    }
  }
  const float inv = 1.f / warp_sum(sum);
#pragma unroll
  for (int c = 0; c < CPL; ++c) v[c] = make_float4(v[c].x * inv, v[c].y * inv, v[c].z * inv, v[c].w * inv);
}

template <int CPL>
__global__ void __launch_bounds__(256, 2) softmax_rows_k(const float* __restrict__ BIAS, float* P, const bool* __restrict__ MASK, long R, int L) {
  const long row = (long)blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= R) return;
  float4 v[CPL];
  softmax_row<CPL>(BIAS + row * L, MASK, L, lane, v);
#pragma unroll
  for (int c = 0; c < CPL; ++c)
    if (lane + 32 * c < L / 4) st4(P + row * L + (lane + 32 * c) * 4, v[c]);
}

// softmax_rows that also writes P^T: a block takes 32 query rows of one head (four per warp), keeps them in shared memory (row pitch
// L + 1 floats: the column gathers of the transpose hit distinct banks) and writes the [L keys x 32 queries] tile of P^T, 16 bytes
// (four queries) per thread store.
template <int CPL>
__global__ void __launch_bounds__(256, 2) softmax_t_k(const float* __restrict__ BIAS, float* P, float* __restrict__ PT,
    const bool* __restrict__ MASK, int L) {
  extern __shared__ float tile[];
  const int pitch = L + 1, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const long r0 = (long)blockIdx.x * 32;
  const long h = r0 / L;
  const int i0 = (int)(r0 - h * L);
  for (int rr = warp; rr < 32; rr += 8) {
    const long row = r0 + rr;
    float4 v[CPL];
    softmax_row<CPL>(BIAS + row * L, MASK, L, lane, v);
#pragma unroll
    for (int c = 0; c < CPL; ++c) {
      const int ch = lane + 32 * c;
      if (ch < L / 4) {
        st4(P + row * L + ch * 4, v[c]);
        float* t = tile + rr * pitch + ch * 4;
        t[0] = v[c].x; t[1] = v[c].y; t[2] = v[c].z; t[3] = v[c].w;
      }
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < 8 * L; idx += blockDim.x) {
    const int j = idx / 8, c = idx % 8;
    st4(PT + (h * L + j) * L + i0 + c * 4, make_float4(tile[(c * 4 + 0) * pitch + j], tile[(c * 4 + 1) * pitch + j],
                                                       tile[(c * 4 + 2) * pitch + j], tile[(c * 4 + 3) * pitch + j]));
  }
}

// ------------------------------------------------------------------------------------------------------------ pair bias
// forward: bias[h, r] = sum_c LN(pair)[r, c] Wf[h, c] (Wf = Wb diag(w): the LayerNorm weight folded in). A block of 128 threads
// takes 128 pair rows at a time into shared memory (row pitch 129 floats: a thread's walk along its row and the warp's rows hit
// distinct banks), each thread one row: its mean and variance (two passes), then NH dot products over the 128 channels, the
// weights read as float4 broadcasts of the transposed Wf ([c][NH]). Shared memory: 128 x 129 x 4 + 128 NH x 4 bytes (dynamic).
template <int NH>
__global__ void __launch_bounds__(128, 4) pair_bias_k(const float* __restrict__ Z, const float* __restrict__ WF, float* __restrict__ BIAS,
    float2* __restrict__ PST, long R, float eps) {
  constexpr int TR = 128, PITCH = 129;
  extern __shared__ float pbs[];
  float* zs = pbs;                                      // [TR][PITCH]
  float* wt = pbs + TR * PITCH;                         // [128][NH], 16-byte aligned (TR PITCH = 16512 floats)
  const int tid = threadIdx.x;
  for (int i = tid; i < NH * 128; i += 128) { const int h = i / 128, c = i % 128; wt[c * NH + h] = WF[i]; }
  for (long r0 = (long)blockIdx.x * TR; r0 < R; r0 += (long)gridDim.x * TR) {
    __syncthreads();                                    // the previous tile is read (and Wf^T is in)
    const int nr = (int)min((long)TR, R - r0);
#pragma unroll 4
    for (int k = 0; k < 32; ++k) {                      // 128 rows x 32 float4, coalesced
      const int idx = k * 128 + tid, row = idx >> 5, c4 = idx & 31;
      if (row < nr) {
        const float4 v = ld4(Z + (r0 + row) * 128 + c4 * 4);
        float* d = zs + row * PITCH + c4 * 4;
        d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
      }
    }
    __syncthreads();
    if (tid < nr) {
      const float* zr = zs + tid * PITCH;
      float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
      for (int c = 0; c < 128; ++c) s4[c & 3] += zr[c];
      const float mean = ((s4[0] + s4[1]) + (s4[2] + s4[3])) * (1.f / 128);
      float q4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
      for (int c = 0; c < 128; ++c) { const float d = zr[c] - mean; q4[c & 3] += d * d; }
      const float rstd = rsqrtf(((q4[0] + q4[1]) + (q4[2] + q4[3])) * (1.f / 128) + eps);
      if (PST) PST[r0 + tid] = make_float2(mean, rstd);
      float acc[NH];
#pragma unroll
      for (int h = 0; h < NH; ++h) acc[h] = 0.f;
#pragma unroll 2
      for (int c = 0; c < 128; ++c) {
        const float xh = (zr[c] - mean) * rstd;
        const float4* w4 = reinterpret_cast<const float4*>(wt + c * NH);
#pragma unroll
        for (int q = 0; q < NH / 4; ++q) {
          const float4 w = w4[q];
          acc[4 * q + 0] = fmaf(xh, w.x, acc[4 * q + 0]); acc[4 * q + 1] = fmaf(xh, w.y, acc[4 * q + 1]);
          acc[4 * q + 2] = fmaf(xh, w.z, acc[4 * q + 2]); acc[4 * q + 3] = fmaf(xh, w.w, acc[4 * q + 3]);
        }
      }
#pragma unroll
      for (int h = 0; h < NH; ++h) BIAS[(long)h * R + r0 + tid] = acc[h];
    }
  }
}

// backward: d LN(pair)[r, c] = sum_h dbias[h, r] Wf[h, c]; the LayerNorm backward with the saved (mean, rstd); dWf[h, c] += dbias[h, r]
// LN(pair)[r, c]. A block of 128 threads takes 32 rows at a time: thread c (a channel) forms d LN(pair)[r, c] for the 32 rows (Wf's
// column c in registers, dbias^T read as float4 broadcasts) and accumulates its column of dWf in registers; then each warp finishes 8
// rows of d pair (two warp sums per row). dWf leaves as one [NH, 128] partial per block (finalize of the bf16 extension sums them).
template <int NH>
__global__ void __launch_bounds__(128, 4) pair_bias_bwd_k(const float* __restrict__ DB, const float* __restrict__ Z,
    const float2* __restrict__ PST, const float* __restrict__ WF, float* __restrict__ DZ, float* __restrict__ DWF, long R) {
  constexpr int TR = 32, PITCH = 129;
  __shared__ float zs[TR * PITCH], dl[TR * PITCH];
  __shared__ __align__(16) float dbt[TR * NH];          // [row][head]
  __shared__ float2 st[TR];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, c = tid;
  float wf[NH], acc[NH];
#pragma unroll
  for (int h = 0; h < NH; ++h) { wf[h] = WF[h * 128 + c]; acc[h] = 0.f; }
  for (long r0 = (long)blockIdx.x * TR; r0 < R; r0 += (long)gridDim.x * TR) {
    __syncthreads();                                    // the previous tile is read
#pragma unroll
    for (int k = 0; k < 8; ++k) {                       // 32 rows x 32 float4
      const int idx = k * 128 + tid, row = idx >> 5, c4 = idx & 31;
      const float4 v = ld4(Z + (r0 + row) * 128 + c4 * 4);
      float* d = zs + row * PITCH + c4 * 4;
      d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
    }
    for (int i = tid; i < NH * TR; i += 128) { const int h = i / TR, rr = i % TR; dbt[rr * NH + h] = DB[(long)h * R + r0 + rr]; }
    if (tid < TR) st[tid] = PST[r0 + tid];
    __syncthreads();
#pragma unroll 2
    for (int rr = 0; rr < TR; ++rr) {
      const float2 s = st[rr];
      const float xh = (zs[rr * PITCH + c] - s.x) * s.y;
      const float4* d4 = reinterpret_cast<const float4*>(dbt + rr * NH);
      float d = 0.f;
#pragma unroll
      for (int q = 0; q < NH / 4; ++q) {
        const float4 b = d4[q];
        d = fmaf(b.x, wf[4 * q + 0], d); d = fmaf(b.y, wf[4 * q + 1], d); d = fmaf(b.z, wf[4 * q + 2], d); d = fmaf(b.w, wf[4 * q + 3], d);
        acc[4 * q + 0] = fmaf(b.x, xh, acc[4 * q + 0]); acc[4 * q + 1] = fmaf(b.y, xh, acc[4 * q + 1]);
        acc[4 * q + 2] = fmaf(b.z, xh, acc[4 * q + 2]); acc[4 * q + 3] = fmaf(b.w, xh, acc[4 * q + 3]);
      }
      dl[rr * PITCH + c] = d;
    }
    __syncthreads();
    // d pair = rstd (d LN - mean(d LN) - LN mean(d LN LN)); lane l owns channels l, l + 32, l + 64, l + 96 of the warp's rows
    for (int rr = w; rr < TR; rr += 4) {
      const float2 s = st[rr];
      float g[4], x[4], m1 = 0.f, m2 = 0.f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        g[k] = dl[rr * PITCH + lane + 32 * k];
        x[k] = (zs[rr * PITCH + lane + 32 * k] - s.x) * s.y;
        m1 += g[k]; m2 += g[k] * x[k];
      }
      m1 = warp_sum(m1) * (1.f / 128);
      m2 = warp_sum(m2) * (1.f / 128);
#pragma unroll
      for (int k = 0; k < 4; ++k) DZ[(r0 + rr) * 128 + lane + 32 * k] = s.y * (g[k] - m1 - x[k] * m2);
    }
  }
#pragma unroll
  for (int h = 0; h < NH; ++h) DWF[((long)blockIdx.x * NH + h) * 128 + c] = acc[h];
}

// -------------------------------------------------------------------------------------------------------------- host
template <typename C> C* P(const at::Tensor& t) { return reinterpret_cast<C*>(t.data_ptr()); }
template <typename C> const C* CP(const at::Tensor& t) { return reinterpret_cast<const C*>(t.data_ptr()); }
cudaStream_t S() { return at::cuda::getCurrentCUDAStream(); }
void f32c(const at::Tensor& t, const char* n) { TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous(), n, ": contiguous fp32"); }
void f32r(const at::Tensor& t, const char* n) {      // fp32 rows: unit column stride, 16-byte aligned rows
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.dim() == 2 && t.stride(1) == 1 && t.stride(0) % 4 == 0
              && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, n, ": fp32 rows, 16-byte aligned");
}

int nsm() { static int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount; return n; }
//: rows of every partial buffer: an upper bound of the blocks of any launch here (the kernels return how many they wrote); the
//: same bound as the bf16 training extension's, whose finalize sums them
int64_t partial_rows(int64_t) { return 8 * (int64_t)nsm(); }
// persistent blocks: as many as are resident on every SM at once, never more than the rows need nor than partial_rows
template <typename K>
unsigned grid(K kernel, int64_t rows, int rows_per_block, int threads, size_t smem = 0) {
  int per = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, threads, smem);
  const int64_t cap = std::min<int64_t>((int64_t)std::max(per, 1) * nsm(), partial_rows(0));
  return (unsigned)std::max<int64_t>(1, std::min<int64_t>((rows + rows_per_block - 1) / rows_per_block, cap));
}
const bool* mask_ptr(const c10::optional<at::Tensor>& mask, int64_t L, const char* n) {
  if (!mask.has_value()) return nullptr;
  TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && mask->numel() == L, n, ": bool mask [L]");
  return mask->data_ptr<bool>();
}

void cond_ln(at::Tensor c, at::Tensor chat, at::Tensor cst, double eps) {
  const int64_t M = c.size(0);
  f32c(c, "c"); f32c(chat, "chat"); f32c(cst, "cst");
  TORCH_CHECK(c.size(1) == DC && chat.sizes() == c.sizes() && cst.numel() == 2 * M, "cond_ln: [M, 384], stats [M, 2]");
  const at::cuda::CUDAGuard gd(c.device());
  cond_ln_k<<<grid(cond_ln_k, M, WPB, 256), 256, 0, S()>>>(CP<float>(c), P<float>(chat), P<float2>(cst), M, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void adaln_a(at::Tensor x, at::Tensor G, at::Tensor bs, at::Tensor xa, at::Tensor xst, double eps) {
  const int64_t M = x.size(0);
  f32c(x, "x"); f32r(G, "G"); f32c(bs, "bs"); f32c(xa, "xa"); f32c(xst, "xst");
  TORCH_CHECK(x.size(1) == D && G.size(1) >= 2 * D && xa.sizes() == x.sizes(), "adaln_a: x [M, 768], G [M, >= 1536]");
  const at::cuda::CUDAGuard gd(x.device());
  adaln_a_k<<<grid(adaln_a_k, M, RPB2, 256), 256, 0, S()>>>(CP<float>(x), CP<float>(G), G.stride(0), CP<float>(bs), P<float>(xa),
                                                             P<float2>(xst), M, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void res_adaln_b(at::Tensor x, at::Tensor y, at::Tensor Gg, at::Tensor bg1, at::Tensor G, at::Tensor bs2, at::Tensor xt,
                 at::Tensor x1st, double eps) {
  const int64_t M = x.size(0);
  f32c(x, "x"); f32c(y, "y"); f32r(Gg, "Gg"); f32c(bg1, "bg1"); f32r(G, "G"); f32c(bs2, "bs2"); f32c(xt, "xt"); f32c(x1st, "x1st");
  TORCH_CHECK(G.size(1) >= 4 * D && Gg.size(1) >= D, "res_adaln_b: G [M, 3072], Gg [M, >= 768]");
  const at::cuda::CUDAGuard gd(x.device());
  res_adaln_b_k<<<grid(res_adaln_b_k, M, RPB2, 256), 256, 0, S()>>>(CP<float>(x), CP<float>(y), CP<float>(Gg), Gg.stride(0),
      CP<float>(bg1), CP<float>(G), G.stride(0), CP<float>(bs2), P<float>(xt), P<float2>(x1st), M, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void res_c(at::Tensor x, at::Tensor y, at::Tensor z, at::Tensor Gg, at::Tensor bg1, at::Tensor bg2, at::Tensor out) {
  const int64_t M = x.size(0);
  f32c(x, "x"); f32c(y, "y"); f32c(z, "z"); f32r(Gg, "Gg"); f32c(bg1, "bg1"); f32c(bg2, "bg2"); f32c(out, "out");
  TORCH_CHECK(Gg.size(1) >= 2 * D, "res_c: Gg [M, 1536]");
  const at::cuda::CUDAGuard gd(x.device());
  res_c_k<<<grid(res_c_k, M, RPB2, 256), 256, 0, S()>>>(CP<float>(x), CP<float>(y), CP<float>(z), CP<float>(Gg), Gg.stride(0),
      CP<float>(bg1), CP<float>(bg2), P<float>(out), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_part(const at::Tensor& p, int64_t M, const char* n) {
  f32c(p, n);
  TORCH_CHECK(p.dim() == 2 && p.size(1) == D && p.size(0) >= partial_rows(M), n, ": partials [partial_rows, 768]");
}

int64_t res_c_bwd(at::Tensor dout, at::Tensor z, at::Tensor Gg, at::Tensor bg2, at::Tensor dz, at::Tensor dGg, at::Tensor pg2) {
  const int64_t M = dout.size(0);
  f32c(dout, "dout"); f32c(z, "z"); f32r(Gg, "Gg"); f32c(bg2, "bg2"); f32c(dz, "dz"); f32r(dGg, "dGg"); check_part(pg2, M, "pg2");
  const at::cuda::CUDAGuard gd(dout.device());
  const unsigned g = grid(res_c_bwd_k, M, RPB2, 256);
  res_c_bwd_k<<<g, 256, 0, S()>>>(CP<float>(dout), CP<float>(z), CP<float>(Gg), Gg.stride(0), CP<float>(bg2), P<float>(dz),
                                  P<float>(dGg), dGg.stride(0), P<float>(pg2), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return g;
}

int64_t res_adaln_b_bwd(at::Tensor dout, at::Tensor dxt, at::Tensor x, at::Tensor x1st, at::Tensor G, at::Tensor bs2, at::Tensor Gg,
                        at::Tensor bg1, at::Tensor y, at::Tensor dx1, at::Tensor dy, at::Tensor dG, at::Tensor dGg, at::Tensor ps2,
                        at::Tensor pg1) {
  const int64_t M = dout.size(0);
  f32c(dout, "dout"); f32c(dxt, "dxt"); f32c(x, "x"); f32c(x1st, "x1st"); f32r(G, "G"); f32c(bs2, "bs2"); f32r(Gg, "Gg");
  f32c(bg1, "bg1"); f32c(y, "y"); f32c(dx1, "dx1"); f32c(dy, "dy"); f32r(dG, "dG"); f32r(dGg, "dGg");
  check_part(ps2, M, "ps2"); check_part(pg1, M, "pg1");
  const at::cuda::CUDAGuard gd(dout.device());
  const unsigned g = grid(res_adaln_b_bwd_k, M, RPB2, 256);
  res_adaln_b_bwd_k<<<g, 256, 0, S()>>>(CP<float>(dout), CP<float>(dxt), CP<float>(x), CP<float2>(x1st), CP<float>(G), G.stride(0),
      CP<float>(bs2), CP<float>(Gg), Gg.stride(0), CP<float>(bg1), CP<float>(y), P<float>(dx1), P<float>(dy), P<float>(dG), dG.stride(0),
      P<float>(dGg), dGg.stride(0), P<float>(ps2), P<float>(pg1), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return g;
}

int64_t adaln_a_bwd(at::Tensor dxa, at::Tensor x, at::Tensor xst, at::Tensor G, at::Tensor bs1, at::Tensor dx1, at::Tensor dx,
                    at::Tensor dG, at::Tensor ps1) {
  const int64_t M = dxa.size(0);
  f32c(dxa, "dxa"); f32c(x, "x"); f32c(xst, "xst"); f32r(G, "G"); f32c(bs1, "bs1"); f32c(dx1, "dx1"); f32c(dx, "dx"); f32r(dG, "dG");
  check_part(ps1, M, "ps1");
  const at::cuda::CUDAGuard gd(dxa.device());
  const unsigned g = grid(adaln_a_bwd_k, M, RPB2, 256);
  adaln_a_bwd_k<<<g, 256, 0, S()>>>(CP<float>(dxa), CP<float>(x), CP<float2>(xst), CP<float>(G), G.stride(0), CP<float>(bs1),
                                    CP<float>(dx1), P<float>(dx), P<float>(dG), dG.stride(0), P<float>(ps1), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return g;
}

void cond_bwd(at::Tensor dchat, at::Tensor dcg, at::Tensor c, at::Tensor cst, at::Tensor dc) {
  const int64_t M = dchat.size(0);
  f32c(dchat, "dchat"); f32c(dcg, "dcg"); f32c(c, "c"); f32c(cst, "cst"); f32c(dc, "dc");
  TORCH_CHECK(c.size(1) == DC && dchat.sizes() == c.sizes() && dcg.sizes() == c.sizes() && dc.sizes() == c.sizes(), "cond_bwd: [M, 384]");
  const at::cuda::CUDAGuard gd(dchat.device());
  cond_bwd_k<<<grid(cond_bwd_k, M, WPB, 256), 256, 0, S()>>>(CP<float>(dchat), CP<float>(dcg), CP<float>(c), CP<float2>(cst),
                                                             P<float>(dc), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_bwd(at::Tensor da, at::Tensor ao, at::Tensor g, at::Tensor dout, at::Tensor dg, at::Tensor dd, int64_t L) {
  const int64_t M = da.size(0), W = da.size(1);
  TORCH_CHECK(W == 768 || W == 1024, "gate_bwd: 768 or 1024 attention channels");
  for (auto* t : {&da, &ao, &dout}) { f32c(*t, "gate_bwd"); TORCH_CHECK(t->size(1) == W && t->size(0) == M, "gate_bwd: [M, W]"); }
  f32r(g, "g"); f32r(dg, "dg");
  const int64_t nh = dd.numel() / M;
  TORCH_CHECK(dd.scalar_type() == at::kFloat && dd.is_contiguous() && (nh == 12 || nh == 16 || nh == 24) && dd.numel() == M * nh
              && M % L == 0 && (W / 4) % nh == 0, "gate_bwd: dd [A, 12, 16 or 24, L]");
  const at::cuda::CUDAGuard gd(da.device());
  auto k = W == 768 ? gate_bwd_k<6> : gate_bwd_k<8>;
  k<<<(unsigned)((M + WPB - 1) / WPB), WPB * 32, 0, S()>>>(CP<float>(da), CP<float>(ao), CP<float>(g), P<float>(dout), P<float>(dg),
                                                           P<float>(dd), M, (int)L, g.stride(0), dg.stride(0), (int)nh);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swiglu(at::Tensor ab, at::Tensor h) {
  const int64_t M = h.size(0), N = h.size(1);
  f32r(ab, "ab"); f32c(h, "h");
  TORCH_CHECK(ab.size(0) == M && ab.size(1) == 2 * N && N % 4 == 0, "swiglu: ab [M, 2N] view, h [M, N]");
  const at::cuda::CUDAGuard gd(ab.device());
  const int64_t n = M * (N / 4);
  if (n) swiglu_k<<<(unsigned)((n + 255) / 256), 256, 0, S()>>>(CP<float>(ab), P<float>(h), M, (int)N, ab.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swiglu_bwd(at::Tensor dh, at::Tensor ab, at::Tensor dab) {
  const int64_t M = dh.size(0), N = dh.size(1);
  f32c(dh, "dh"); f32c(ab, "ab"); f32c(dab, "dab");
  TORCH_CHECK(ab.size(0) == M && ab.size(1) == 2 * N && dab.sizes() == ab.sizes() && N % 4 == 0, "swiglu_bwd: dh [M, N], ab / dab [M, 2N]");
  const at::cuda::CUDAGuard gd(dh.device());
  const int64_t n = M * (N / 4);
  if (n) swiglu_bwd_k<<<(unsigned)((n + 255) / 256), 256, 0, S()>>>(CP<float>(dh), CP<float>(ab), P<float>(dab), M, (int)N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_rows(at::Tensor bias, at::Tensor p, c10::optional<at::Tensor> mask) {
  f32c(bias, "bias"); f32c(p, "p");
  TORCH_CHECK(bias.dim() == 2 && p.sizes() == bias.sizes(), "softmax_rows: p like bias [R, L]");
  const int64_t R = bias.size(0), L = bias.size(1);
  TORCH_CHECK(L % 4 == 0 && L <= 2048, "softmax_rows: L a multiple of 4, at most 2048");
  const bool* mk = mask_ptr(mask, L, "softmax_rows");
  const at::cuda::CUDAGuard gd(bias.device());
  const unsigned g = (unsigned)((R + 7) / 8);
  if (L <= 512) softmax_rows_k<4><<<g, 256, 0, S()>>>(CP<float>(bias), P<float>(p), mk, R, (int)L);
  else if (L <= 1024) softmax_rows_k<8><<<g, 256, 0, S()>>>(CP<float>(bias), P<float>(p), mk, R, (int)L);
  else softmax_rows_k<16><<<g, 256, 0, S()>>>(CP<float>(bias), P<float>(p), mk, R, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_t(at::Tensor bias, at::Tensor p, at::Tensor pt, c10::optional<at::Tensor> mask) {
  f32c(bias, "bias"); f32c(p, "p"); f32c(pt, "pt");
  TORCH_CHECK(bias.dim() == 2 && p.sizes() == bias.sizes() && pt.sizes() == bias.sizes() && pt.data_ptr() != bias.data_ptr(),
              "softmax_t: p, pt like bias [H L, L] (p may be bias, pt may not)");
  const int64_t R = bias.size(0), L = bias.size(1);
  TORCH_CHECK(L % 32 == 0 && L <= 1024 && R % L == 0, "softmax_t: L a multiple of 32, at most 1024; [H L, L]");
  const bool* mk = mask_ptr(mask, L, "softmax_t");
  const at::cuda::CUDAGuard gd(bias.device());
  const size_t smem = 32 * (L + 1) * 4;
  static bool attr = [] {
    return cudaFuncSetAttribute(softmax_t_k<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1025 * 4) == cudaSuccess
        && cudaFuncSetAttribute(softmax_t_k<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1025 * 4) == cudaSuccess;
  }();
  TORCH_CHECK(attr, "softmax_t: shared memory attribute");
  if (L <= 512) softmax_t_k<4><<<(unsigned)(R / 32), 256, smem, S()>>>(CP<float>(bias), P<float>(p), P<float>(pt), mk, (int)L);
  else softmax_t_k<8><<<(unsigned)(R / 32), 256, smem, S()>>>(CP<float>(bias), P<float>(p), P<float>(pt), mk, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NH> constexpr size_t pb_smem() { return (size_t)(128 * 129 + 128 * NH) * 4; }

void pair_bias(at::Tensor z, at::Tensor wf, at::Tensor bias, c10::optional<at::Tensor> pst, double eps) {
  f32c(z, "pair"); f32c(wf, "Wf"); f32c(bias, "bias");
  const int64_t R = z.size(0), nh = wf.size(0);
  TORCH_CHECK(z.dim() == 2 && z.size(1) == 128 && wf.size(1) == 128 && (nh == 12 || nh == 16 || nh == 24) && bias.numel() == nh * R,
              "pair_bias: pair [R, 128], Wf [12, 16 or 24, 128], bias [nh, R] fp32");
  float2* ps = nullptr;
  if (pst.has_value()) { f32c(*pst, "pst"); TORCH_CHECK(pst->numel() == 2 * R, "pair_bias: pst [R, 2]"); ps = P<float2>(*pst); }
  const at::cuda::CUDAGuard gd(z.device());
  static bool attr = [] {
    return cudaFuncSetAttribute(pair_bias_k<12>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pb_smem<12>()) == cudaSuccess
        && cudaFuncSetAttribute(pair_bias_k<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pb_smem<16>()) == cudaSuccess
        && cudaFuncSetAttribute(pair_bias_k<24>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pb_smem<24>()) == cudaSuccess;
  }();
  TORCH_CHECK(attr, "pair_bias: shared memory attribute");
  auto k = nh == 12 ? pair_bias_k<12> : nh == 16 ? pair_bias_k<16> : pair_bias_k<24>;
  const size_t smem = nh == 12 ? pb_smem<12>() : nh == 16 ? pb_smem<16>() : pb_smem<24>();
  int per = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 128, smem);
  const unsigned g = (unsigned)std::max<int64_t>(1, std::min<int64_t>((R + 127) / 128, (int64_t)std::max(per, 1) * nsm()));
  k<<<g, 128, smem, S()>>>(CP<float>(z), CP<float>(wf), P<float>(bias), ps, R, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t pair_bias_bwd(at::Tensor db, at::Tensor z, at::Tensor pst, at::Tensor wf, at::Tensor dz, at::Tensor dwf) {
  f32c(db, "dbias"); f32c(z, "pair"); f32c(pst, "pst"); f32c(wf, "Wf"); f32c(dz, "d pair"); f32c(dwf, "dWf partials");
  const int64_t R = z.size(0), nh = wf.size(0);
  TORCH_CHECK(z.dim() == 2 && z.size(1) == 128 && R % 32 == 0 && wf.size(1) == 128 && (nh == 12 || nh == 16 || nh == 24)
              && db.numel() == nh * R && pst.numel() == 2 * R && dz.sizes() == z.sizes(),
              "pair_bias_bwd: pair / d pair [R, 128] (R a multiple of 32), dbias [nh, R], Wf [12, 16 or 24, 128]");
  auto k = nh == 12 ? pair_bias_bwd_k<12> : nh == 16 ? pair_bias_bwd_k<16> : pair_bias_bwd_k<24>;
  const at::cuda::CUDAGuard gd(z.device());
  const unsigned g = grid(k, R, 32, 128);
  TORCH_CHECK(dwf.numel() >= (int64_t)g * nh * 128, "pair_bias_bwd: dWf partials [partial_rows, nh, 128]");
  k<<<g, 128, 0, S()>>>(CP<float>(db), CP<float>(z), CP<float2>(pst), CP<float>(wf), P<float>(dz), P<float>(dwf), R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return g;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("partial_rows", &partial_rows);
  m.def("cond_ln_cuda", &cond_ln);
  m.def("adaln_a_cuda", &adaln_a);
  m.def("res_adaln_b_cuda", &res_adaln_b);
  m.def("res_c_cuda", &res_c);
  m.def("res_c_bwd_cuda", &res_c_bwd);
  m.def("res_adaln_b_bwd_cuda", &res_adaln_b_bwd);
  m.def("adaln_a_bwd_cuda", &adaln_a_bwd);
  m.def("cond_bwd_cuda", &cond_bwd);
  m.def("gate_bwd_cuda", &gate_bwd);
  m.def("swiglu_cuda", &swiglu);
  m.def("swiglu_bwd_cuda", &swiglu_bwd);
  m.def("softmax_rows_cuda", &softmax_rows);
  m.def("softmax_t_cuda", &softmax_t);
  m.def("pair_bias_cuda", &pair_bias);
  m.def("pair_bias_bwd_cuda", &pair_bias_bwd);
}
