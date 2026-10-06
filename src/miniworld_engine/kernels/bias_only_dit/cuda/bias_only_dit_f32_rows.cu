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
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

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
// a shared-memory read the compiler may not hoist or keep (the operand is re-formed where it is used, to bound registers)
__device__ __forceinline__ float ldsv(const float* p) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
  return v;
}
// a second read of data this thread loaded moments ago (an L1 hit): volatile, so the compiler cannot fold it into the first load and
// keep that value live in registers -- the point of reading twice
__device__ __forceinline__ float4 reld4(const float* p) {
  float4 v;
  asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
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
// MB resident blocks per SM (MINIWORLD_BIAS_ONLY_DIT_F32_RESB_MINB, default 3; 2: the 128-register build): with two it moved five
// fp32 streams per row behind a two-warp barrier at 4.7 TB/s (L768 141 us); MB = 3 (<= 80 registers) issues the loads in two waves --
// x, y, g1 (the residual), then the AdaLN tables before the row statistics, so they fly during the reductions -- 107 us.
template <int MB>
__global__ void __launch_bounds__(256, MB) res_adaln_b_k(const float* __restrict__ X, const float* __restrict__ Y,
    const float* __restrict__ GG, long sgg, const float* __restrict__ BG1, const float* __restrict__ G, long sg_,
    const float* __restrict__ BS2, float* __restrict__ XT_, float2* __restrict__ X1ST, long M, float eps) {
  ROWS2_BEGIN
  for (long r = sid; r < M; r += ns) {
    float4 x[3], s[3], sh[3];
    if constexpr (MB >= 3) {
      float4 xi[3], y[3], g[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        xi[j] = ld4(X + r * D + COL2(j)); y[j] = ld4(Y + r * D + COL2(j)); g[j] = ld4(GG + r * sgg + COL2(j));
      }
#pragma unroll
      for (int j = 0; j < 3; ++j) x[j] = resid4(xi[j], sig4(add4(g[j], ld4(BG1 + COL2(j)))), y[j]);
#pragma unroll
      for (int j = 0; j < 3; ++j) { s[j] = ld4(G + r * sg_ + 2 * D + COL2(j)); sh[j] = ld4(G + r * sg_ + 3 * D + COL2(j)); }
    } else {
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const float4 xi = ld4(X + r * D + COL2(j)), y = ld4(Y + r * D + COL2(j)), g = ld4(GG + r * sgg + COL2(j));
        x[j] = resid4(xi, sig4(add4(g, ld4(BG1 + COL2(j)))), y);
        s[j] = ld4(G + r * sg_ + 2 * D + COL2(j)); sh[j] = ld4(G + r * sg_ + 3 * D + COL2(j));
      }
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

// res_adaln_b_bwd holds six streams of the row at once (the bf16 twin, bias_only_dit_train_rows.cu, keeps its bf16 operands packed,
// two registers per four values, until used): in fp32 the twin's two-warp layout needs 72 registers of row data per thread and
// spilled (24-64 bytes at 128). Here a 768-row takes THREE warps (96 threads, 8 columns each: two float4 per stream, 48 registers of
// row data), two rows per 192-thread block, two blocks per SM (at three ptxas capped it at 96 registers and spilled 80 bytes; at two
// it takes 135, no spill); the arithmetic and its order per element are the twin's, the row sums go through
// shared memory in a fixed order. d shift2 = dxt is read where the dxt GEMM wrote it (dG[:, 3D:4D]).
constexpr int RB3 = 2;                                         // rows per block of the three-warp kernels
#define COL3(j) ((((hw) * 2 + (j)) * 32 + lane) * 4)
__global__ void __launch_bounds__(96 * RB3, 2) res_adaln_b_bwd_k(const float* __restrict__ DOUT, const float* __restrict__ X,
    const float2* __restrict__ X1ST, const float* __restrict__ G, long sg_, const float* __restrict__ BS2, const float* __restrict__ GG,
    long sgg, const float* __restrict__ BG1, const float* __restrict__ Y, float* __restrict__ DX1, float* __restrict__ DY,
    float* __restrict__ DG, long sdg, float* __restrict__ DGG, long sdgg, float* __restrict__ PS2, float* __restrict__ PG1, long M) {
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32, hw = w % 3, slot = w / 3;
  const long sid = (long)blockIdx.x * RB3 + slot, ns = (long)gridDim.x * RB3;
  __shared__ float2 xr[2][RB3][3];
  int par = 0;
  // the column sums accumulate in shared memory (each thread its own slot's columns; no barrier until the end)
  __shared__ float4 acc_s[RB3][D / 4], acc_g[RB3][D / 4];
#pragma unroll
  for (int j = 0; j < 2; ++j) { acc_s[slot][COL3(j) / 4] = zero4(); acc_g[slot][COL3(j) / 4] = zero4(); }
  for (long r = sid; r < M; r += ns) {
    float4 dxt[2], xh[2], s2[2], dout[2], g1[2], y[2];
    const float2 st = X1ST[r];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      dxt[j] = ld4(DG + r * sdg + 3 * D + COL3(j));
      xh[j] = ld4(X + r * D + COL3(j));
      s2[j] = ld4(G + r * sg_ + 2 * D + COL3(j));
      dout[j] = ld4(DOUT + r * D + COL3(j)); g1[j] = ld4(GG + r * sgg + COL3(j)); y[j] = ld4(Y + r * D + COL3(j));
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {                                      // x1 rebuilt, then normalised; g1 becomes sigmoid(g1)
      g1[j] = sig4(add4(g1[j], ld4(BG1 + COL3(j))));
      xh[j] = xhat4(resid4(xh[j], g1[j], y[j]), st.x, st.y);
    }
    float m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      s2[j] = sig4(add4(s2[j], ld4(BS2 + COL3(j))));
      const float4 ds2 = dsig4(mul4(dxt[j], xh[j]), s2[j]);
      st4(DG + r * sdg + 2 * D + COL3(j), ds2);
      acc_s[slot][COL3(j) / 4] = add4(acc_s[slot][COL3(j) / 4], ds2);
      dxt[j] = mul4(dxt[j], s2[j]);                                  // d xhat
      m1 += sum4(dxt[j]);
      m2 += sum4(mul4(dxt[j], xh[j]));
    }
    m1 = warp_sum(m1); m2 = warp_sum(m2);                              // the row's three warps, in a fixed order
    if (lane == 0) xr[par][slot][hw] = make_float2(m1, m2);
    asm volatile("bar.sync %0, 96;" :: "r"(1 + slot) : "memory");
    {
      const float2 a = xr[par][slot][0], b = xr[par][slot][1], c = xr[par][slot][2];
      m1 = ((a.x + b.x) + c.x) * (1.f / D); m2 = ((a.y + b.y) + c.y) * (1.f / D);
    }
    par ^= 1;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float4 dx1 = make_float4(dout[j].x + st.y * (dxt[j].x - m1 - xh[j].x * m2), dout[j].y + st.y * (dxt[j].y - m1 - xh[j].y * m2),
                                     dout[j].z + st.y * (dxt[j].z - m1 - xh[j].z * m2), dout[j].w + st.y * (dxt[j].w - m1 - xh[j].w * m2));
      st4(DX1 + r * D + COL3(j), dx1);
      st4(DY + r * D + COL3(j), mul4(dx1, g1[j]));
      const float4 dg1 = dsig4(mul4(dx1, y[j]), g1[j]);
      st4(DGG + r * sdgg + COL3(j), dg1);
      acc_g[slot][COL3(j) / 4] = add4(acc_g[slot][COL3(j) / 4], dg1);
    }
  }
  __syncthreads();
  for (int c = threadIdx.x; c < D / 4; c += blockDim.x) {
    st4(PS2 + (long)blockIdx.x * D + 4 * c, add4(acc_s[0][c], acc_s[1][c]));
    st4(PG1 + (long)blockIdx.x * D + 4 * c, add4(acc_g[0][c], acc_g[1][c]));
  }
}

__global__ void __launch_bounds__(256, 2) adaln_a_bwd_k(const float* __restrict__ DXA, long sdxa, int copy_sh, const float* __restrict__ X,
    const float2* __restrict__ XST,
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
      dxa[j] = ld4(DXA + r * sdxa + COL2(j));
      xh[j] = xhat4(ld4(X + r * D + COL2(j)), st.x, st.y);
      s[j] = ld4(G + r * sg_ + COL2(j));
      dx1[j] = ld4(DX1 + r * D + COL2(j));
    }
    float m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      s[j] = sig4(add4(s[j], ld4(BS1 + COL2(j))));
      if (copy_sh) st4(DG + r * sdg + D + COL2(j), dxa[j]);           // d shift1 = dxa (no copy when the GEMM wrote it there)
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

// softmax_row for two rows of a warp: both rows' loads issued before any math (the per-row arithmetic is softmax_row's, in the same
// order: the same bits)
template <int CPL>
__device__ __forceinline__ void softmax_row2(const float* __restrict__ b0, const float* __restrict__ b1, const bool* __restrict__ mask,
                                             int L, int lane, float4 (&v)[2][CPL]) {
  constexpr float NEG = -3.0e38f;
  const int nch = L / 4;
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int c = 0; c < CPL; ++c) v[r][c] = lane + 32 * c < nch ? ld4((r ? b1 : b0) + (lane + 32 * c) * 4) : zero4();
  float mx[2] = {NEG, NEG};
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    const int ch = lane + 32 * c;
    if (ch < nch) {
      uchar4 mk = make_uchar4(1, 1, 1, 1);
      if (mask) mk = *reinterpret_cast<const uchar4*>(mask + ch * 4);
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float4 e = v[r][c];
        if (!mk.x) e.x = NEG;
        if (!mk.y) e.y = NEG;
        if (!mk.z) e.z = NEG;
        if (!mk.w) e.w = NEG;
        v[r][c] = e;
        mx[r] = fmaxf(mx[r], fmaxf(fmaxf(e.x, e.y), fmaxf(e.z, e.w)));
      }
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    mx[0] = fmaxf(mx[0], __shfl_xor_sync(0xffffffffu, mx[0], o));
    mx[1] = fmaxf(mx[1], __shfl_xor_sync(0xffffffffu, mx[1], o));
  }
  float sum[2] = {0.f, 0.f};
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    if (lane + 32 * c < nch) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        v[r][c] = make_float4(__expf(v[r][c].x - mx[r]), __expf(v[r][c].y - mx[r]), __expf(v[r][c].z - mx[r]), __expf(v[r][c].w - mx[r]));
        sum[r] += sum4(v[r][c]);
      }
    }
  }
  const float inv0 = 1.f / warp_sum(sum[0]), inv1 = 1.f / warp_sum(sum[1]);
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    v[0][c] = make_float4(v[0][c].x * inv0, v[0][c].y * inv0, v[0][c].z * inv0, v[0][c].w * inv0);
    v[1][c] = make_float4(v[1][c].x * inv1, v[1][c].y * inv1, v[1][c].z * inv1, v[1][c].w * inv1);
  }
}

// softmax_rows that also writes P^T: a block takes SR = 16 query rows of one head (two per warp, softmax_row2), keeps them in shared
// memory (row pitch L + 4 floats: 16-byte row stores without conflicts, the column gathers of the transpose at most 2-way) and
// writes the [L keys x 16 queries] tile of P^T, 16 bytes (four queries) per thread store, 64 contiguous bytes per key. (32 rows per block, four per warp one at a time:
// 58 us at L768 for 113 MB, latency-bound with 1.3 waves of 98 KB blocks.)
constexpr int SR = 16;
template <int CPL>
__global__ void __launch_bounds__(256, 2) softmax_t_k(const float* __restrict__ BIAS, float* P, float* __restrict__ PT,
    const bool* __restrict__ MASK, int L) {
  extern __shared__ float tile[];
  const int pitch = L + 4, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const long r0 = (long)blockIdx.x * SR;
  const long h = r0 / L;
  const int i0 = (int)(r0 - h * L);
  {
    const int rr = 2 * warp;
    const long row = r0 + rr;
    float4 v[2][CPL];
    softmax_row2<CPL>(BIAS + row * L, BIAS + (row + 1) * L, MASK, L, lane, v);
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int c = 0; c < CPL; ++c) {
        const int ch = lane + 32 * c;
        if (ch < L / 4) {
          st4(P + (row + r) * L + ch * 4, v[r][c]);
          st4(tile + (rr + r) * pitch + ch * 4, v[r][c]);
        }
      }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < (SR / 4) * L; idx += blockDim.x) {
    const int j = idx / (SR / 4), c = idx % (SR / 4);
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

// ------------------------------------------------------------------------------------- pair bias on TF32 tensor cores (training)
// The training step's pair bias and its backward run the R x 128 x H products on mma.sync m16n8k8 .tf32 (fp32 accumulate), as the
// bf16 path's kernels run them on bf16 mma.sync: the FMA-pipe kernels above are issue-bound (L768: 185 + 299 us against a memory
// floor of ~60 + ~105). TERMS = 1: the operands rounded to tf32 (cvt.rna, round to nearest: the TF32 recipe, as cuBLAS rounds the
// module's to_bias GEMM and its gradients); TERMS = 3: split fp32 (hi = tf32(x), lo = tf32(x - hi); hi hi + hi lo + lo hi), within
// ~1e-6 of exact fp32 (MINIWORLD_BIAS_ONLY_DIT_PAIR_EXACT=1). Fragments of m16n8k8 .tf32 (g = lane / 4, q = lane % 4):
// A a0 (g, q), a1 (g + 8, q), a2 (g, q + 4), a3 (g + 8, q + 4); B b0 (k q, n g), b1 (k q + 4, n g); C (g, 2q | 2q + 1), (g + 8, ...).
__device__ __forceinline__ void mma_tf32(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
__device__ __forceinline__ uint32_t tf32lo(float x, uint32_t hi) { return tf32r(x - __uint_as_float(hi)); }
__device__ __forceinline__ uint4 split2(float w0, float w1) {   // (hi w0, hi w1, lo w0, lo w1)
  const uint32_t h0 = tf32r(w0), h1 = tf32r(w1);
  return make_uint4(h0, h1, tf32lo(w0, h0), tf32lo(w1, h1));
}
// D += A B in TERMS products (a / b hi and lo; lo unused for TERMS = 1)
template <int TERMS>
__device__ __forceinline__ void mma_x(float (&d)[4], const uint32_t (&ah)[4], const uint32_t (&al)[4], uint32_t bh0, uint32_t bh1,
                                      uint32_t bl0, uint32_t bl1) {
  mma_tf32(d, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
  if constexpr (TERMS == 3) {
    mma_tf32(d, ah[0], ah[1], ah[2], ah[3], bl0, bl1);
    mma_tf32(d, al[0], al[1], al[2], al[3], bh0, bh1);
  }
}
template <int TERMS>
__device__ __forceinline__ void split_a(const float (&v)[4], uint32_t (&h)[4], uint32_t (&l)[4]) {
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    h[e] = tf32r(v[e]);
    l[e] = TERMS == 3 ? tf32lo(v[e], h[e]) : 0u;
  }
}
__device__ __forceinline__ float el4(const float4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
// forward K slot s (< 8) of k-step kk: lane s % 4 of a row holds columns 16 m + 4 (s % 4) + {0..3} of chunk m = kk / 2 (its
// float4 loads: 64 contiguous bytes per row across the four lanes), element 2 (kk % 2) + s / 4 -- the reduction order is free
__device__ __forceinline__ int fcol32(int kk, int s) { return 16 * (kk / 2) + 4 * (s % 4) + 2 * (kk % 2) + s / 4; }

// forward: a warp takes 16 pair rows at a time (rows g, g + 8 per lane: 8 float4 each, all loads first), the LayerNorm statistics
// over the row's four lanes (two passes, exact), LN(pair) Wf^T on mma (NT n-tiles of 8 heads; 12 heads padded to 16 with zero
// weights), the bias stored head-major straight from the accumulators (32 contiguous bytes per head and row group).
template <int NH, int TERMS>
__global__ void __launch_bounds__(256, 2) pair_bias_tc_k(const float* __restrict__ Z, const float* __restrict__ WF,
    float* __restrict__ BIAS, float2* __restrict__ PST, long R, float eps) {
  constexpr int NT = (NH + 7) / 8;
  __shared__ uint4 bw[16][NT][32];                     // Wf^T fragments per k-step, n-tile, lane: (b0, b1) hi, (b0, b1) lo
  const int lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
  const long wid = (long)blockIdx.x * WPB + threadIdx.x / 32, nw = (long)gridDim.x * WPB;
  for (int i = threadIdx.x; i < 16 * NT * 32; i += blockDim.x) {
    const int l = i % 32, t = (i / 32) % NT, kk = i / (32 * NT), h = 8 * t + l / 4;
    bw[kk][t][l] = split2(h < NH ? WF[h * 128 + fcol32(kk, l % 4)] : 0.f, h < NH ? WF[h * 128 + fcol32(kk, l % 4 + 4)] : 0.f);
  }
  __syncthreads();
  for (long rb = wid * 16; rb < R; rb += nw * 16) {
    float4 u[2][8];
#pragma unroll
    for (int hr = 0; hr < 2; ++hr)
#pragma unroll
      for (int m = 0; m < 8; ++m) u[hr][m] = ld4(Z + (rb + g + 8 * hr) * 128 + 4 * (q + 4 * m));
    float mean[2], rs[2];
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      float s1 = 0.f;
#pragma unroll
      for (int m = 0; m < 8; ++m) s1 += sum4(u[hr][m]);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 1); s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
      mean[hr] = s1 * (1.f / 128);
      float s2 = 0.f;
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        const float4 d = make_float4(u[hr][m].x - mean[hr], u[hr][m].y - mean[hr], u[hr][m].z - mean[hr], u[hr][m].w - mean[hr]);
        s2 += sum4(mul4(d, d));
      }
      s2 += __shfl_xor_sync(0xffffffffu, s2, 1); s2 += __shfl_xor_sync(0xffffffffu, s2, 2);
      rs[hr] = rsqrtf(s2 * (1.f / 128) + eps);
    }
    float acc[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      const int m = kk / 2, e = 2 * (kk % 2);
      // TERMS = 3 holds A hi / lo beside the accumulators: it re-reads its row chunk (an L1 hit, loaded just above for the statistics)
      // instead of holding all 64 row values through the products (it spilled: 8 bytes at 128 registers, 24 heads)
      const float4 c0 = TERMS == 3 ? reld4(Z + (rb + g) * 128 + 4 * (q + 4 * m)) : u[0][m];
      const float4 c1 = TERMS == 3 ? reld4(Z + (rb + g + 8) * 128 + 4 * (q + 4 * m)) : u[1][m];
      const float v[4] = {(el4(c0, e) - mean[0]) * rs[0], (el4(c1, e) - mean[1]) * rs[1],
                          (el4(c0, e + 1) - mean[0]) * rs[0], (el4(c1, e + 1) - mean[1]) * rs[1]};
      uint32_t ah[4], al[4];
      split_a<TERMS>(v, ah, al);
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const uint4 b = bw[kk][t][lane];
        mma_x<TERMS>(acc[t], ah, al, b.x, b.y, b.z, b.w);
      }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int h = 8 * t + 2 * q + e;
        if (h < NH) { BIAS[(long)h * R + rb + g] = acc[t][e]; BIAS[(long)h * R + rb + g + 8] = acc[t][2 + e]; }
      }
    if (q == 0) { PST[rb + g] = make_float2(mean[0], rs[0]); PST[rb + g + 8] = make_float2(mean[1], rs[1]); }
  }
}

// backward: 128 pair rows per block item (16 per warp), streamed into shared memory by cp.async (double buffer: the next item loads
// while this one is worked on). d LN(pair) = dbias^T Wf on mma (M = the warp's 16 rows, N = 16 channel tiles of 8, K = the heads in
// NK k-steps of 8), the LayerNorm backward with two row sums over the row's four lanes, d pair out; LN(pair) written back in place
// for dWf^T[16 w .., heads] += LN(pair)^T dbias^T over the item's rows (M = the warp's 16 channels, N = heads, K = rows). dWf leaves
// as one [NH, 128] partial per block (finalize sums). TERMS = 1 keeps d LN of the warp's rows in registers between the two passes
// (64 values per lane); TERMS = 3 forms it twice instead (registers).
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
template <int NH> struct PB32 {
  static constexpr int NK = NH <= 16 ? 2 : 3;          // 8-head k-steps (12 heads padded to 16 with zeros)
  static constexpr int NHP = 8 * NK;
  // row pitches (floats): pair rows 136 (the lanes' float2 reads of rows g and the dWf operand reads hit distinct banks), dbias 132
  // (the dWf B reads ds[head g][row q] distinct; the A reads once per item take a 2-way conflict)
  static constexpr int ZP = 136, DP = 132;
  static constexpr int ZS = 128 * ZP * 4, DS = NHP * DP * 4, BW = NK * 16 * 32 * 16;
  static constexpr int WS = 8 * NK * 4 * 32 * 4;       // TERMS = 3: the dWf sums, one slot per warp, n-tile, element and lane
  static constexpr int SMEM = 2 * ZS + 2 * DS + BW + WS;
};

template <int NH, int TERMS>
__global__ void __launch_bounds__(256, 2) pair_bias_bwd_tc_k(const float* __restrict__ DB, const float* __restrict__ Z,
    const float2* __restrict__ PST, const float* __restrict__ WF, float* __restrict__ DZ, float* __restrict__ DWF, long R) {
  using C = PB32<NH>;
  constexpr int NK = C::NK, ZP = C::ZP, DP = C::DP;
  constexpr bool KEEP = TERMS == 1;
  static_assert(NH == 12 || NH == 16 || NH == 24, "12, 16 or 24 heads");
  extern __shared__ __align__(16) uint8_t smem[];
  uint4 (*bw)[16][32] = reinterpret_cast<uint4 (*)[16][32]>(smem + 2 * C::ZS + 2 * C::DS);   // Wf fragments [k-step][channel tile][lane]
  const uint32_t s_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32, g = lane / 4, q = lane % 4;
  auto prefetch = [&](long b0, int b) {
    for (int i = threadIdx.x; i < 128 * 32; i += blockDim.x) {
      const int row = i / 32, c = i % 32;
      cp_async16(s_base + b * C::ZS + (row * ZP + 4 * c) * 4, Z + (b0 + row) * 128 + 4 * c);
    }
    for (int i = threadIdx.x; i < NH * 32; i += blockDim.x) {
      const int h = i / 32, c = i % 32;
      cp_async16(s_base + 2 * C::ZS + b * C::DS + (h * DP + 4 * c) * 4, DB + (long)h * R + b0 + 4 * c);
    }
  };
  for (int i = threadIdx.x; i < NK * 16 * 32; i += blockDim.x) {    // B = Wf [heads (k) x channels (n)]: b0 (head 8 ks + q, channel 8 j + g)
    const int l = i % 32, j = (i / 32) % 16, ks = i / 512, k = 8 * ks + l % 4, c = 8 * j + l / 4;
    bw[ks][j][l] = split2(k < NH ? WF[k * 128 + c] : 0.f, k + 4 < NH ? WF[(k + 4) * 128 + c] : 0.f);
  }
  for (int i = threadIdx.x; i < 2 * (C::NHP - NH) * DP; i += blockDim.x) {   // the padding heads of both dbias buffers weigh 0
    const int b = i / ((C::NHP - NH) * DP), k = i % ((C::NHP - NH) * DP);
    reinterpret_cast<float*>(smem + 2 * C::ZS + b * C::DS)[NH * DP + k] = 0.f;
  }
  // dWf sums: TERMS = 1 in registers; TERMS = 3 in this lane's own shared-memory slots (registers spilled 24-56 bytes there)
  constexpr bool FLUSH = TERMS == 3;
  float dacc[FLUSH ? 1 : NK][4];
  float* wsum = reinterpret_cast<float*>(smem + 2 * C::ZS + 2 * C::DS + C::BW) + (w * NK * 4) * 32 + lane;   // [w][t][e][lane]
#pragma unroll
  for (int t = 0; t < (FLUSH ? 1 : NK); ++t) dacc[t][0] = dacc[t][1] = dacc[t][2] = dacc[t][3] = 0.f;
  if constexpr (FLUSH) {
#pragma unroll
    for (int i = 0; i < NK * 4; ++i) wsum[i * 32] = 0.f;
  }
  // TERMS = 3 forms d LN twice and its A fragments per use: its loops over the channel tiles / K steps stay rolled (partly), so
  // their MMA chains are not all in flight at once
  constexpr int JU = KEEP ? 16 : 2, KU = KEEP ? 4 : 1;
  const long step = (long)gridDim.x * 128;
  if ((long)blockIdx.x * 128 < R) prefetch((long)blockIdx.x * 128, 0);
  cp_async_commit();
  int b = 0;
  for (long b0 = (long)blockIdx.x * 128; b0 < R; b0 += step, b ^= 1) {
    if (b0 + step < R) prefetch(b0 + step, b ^ 1);
    cp_async_commit();
    const int wr = 16 * w;
    const long rb = b0 + wr;
    const float2 st0 = PST[rb + g], st1 = PST[rb + g + 8];
    cp_async_wait1();                                  // this item's group (the next one may still be in flight)
    __syncthreads();
    float* zs = reinterpret_cast<float*>(smem + b * C::ZS);
    const float* ds = reinterpret_cast<const float*>(smem + 2 * C::ZS + b * C::DS);
    // A = dbias^T [16 rows x heads]: a0 (row g, head 8 ks + q), a1 (row g + 8, ..), a2 (row g, head 8 ks + q + 4), a3
    // TERMS = 1 holds the A fragments (hi only) for the item; TERMS = 3 forms hi / lo inside dln from shared memory (holding them,
    // with the dWf flush accumulators, spilled 24-64 bytes at 128 registers)
    constexpr int NKH = KEEP ? NK : 1;
    uint32_t ah[NKH][4], al[NKH][4];
    auto frag = [&](int ks, uint32_t (&h4)[4], uint32_t (&l4)[4]) {
      const float* d0 = ds + (8 * ks + q) * DP + wr + g;
      const float v[4] = {ldsv(d0), ldsv(d0 + 8), ldsv(d0 + 4 * DP), ldsv(d0 + 4 * DP + 8)};
      split_a<TERMS>(v, h4, l4);
    };
    if constexpr (KEEP) {
#pragma unroll
      for (int ks = 0; ks < NK; ++ks) frag(ks, ah[ks], al[ks]);
    }
    auto dln = [&](float (&d)[4], int j) {             // d LN for channels 8 j ..: (row g, 8 j + 2q | + 1), (row g + 8, ..)
      d[0] = d[1] = d[2] = d[3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < NK; ++ks) {
        const uint4 bb = bw[ks][j][lane];
        if constexpr (KEEP) {
          mma_x<TERMS>(d, ah[ks], al[ks], bb.x, bb.y, bb.z, bb.w);
        } else {
          uint32_t h4[4], l4[4];
          frag(ks, h4, l4);
          mma_x<TERMS>(d, h4, l4, bb.x, bb.y, bb.z, bb.w);
        }
      }
    };
    const float* z0 = zs + (wr + g) * ZP + 2 * q;
    const float* z1 = z0 + 8 * ZP;
    // pass 1: the row sums of d LN and of d LN (pair - mean)
    float dk[KEEP ? 16 : 1][4];
    float sd[2] = {0.f, 0.f}, sdv[2] = {0.f, 0.f};
#pragma unroll JU
    for (int j = 0; j < 16; ++j) {
      float d[4];
      dln(d, j);
      if constexpr (KEEP) { dk[j][0] = d[0]; dk[j][1] = d[1]; dk[j][2] = d[2]; dk[j][3] = d[3]; }
      const float2 x0 = *reinterpret_cast<const float2*>(z0 + 8 * j), x1 = *reinterpret_cast<const float2*>(z1 + 8 * j);
      sd[0] += d[0] + d[1]; sdv[0] = fmaf(d[0], x0.x - st0.x, fmaf(d[1], x0.y - st0.x, sdv[0]));
      sd[1] += d[2] + d[3]; sdv[1] = fmaf(d[2], x1.x - st1.x, fmaf(d[3], x1.y - st1.x, sdv[1]));
    }
    // d pair = rstd (d LN - mean(d LN) - LN mean(d LN LN)) = rstd d LN - c1 (pair - mean) + c0
    float c1[2], c0[2];
    const float2 st[2] = {st0, st1};
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      sd[k] += __shfl_xor_sync(0xffffffffu, sd[k], 1); sd[k] += __shfl_xor_sync(0xffffffffu, sd[k], 2);
      sdv[k] += __shfl_xor_sync(0xffffffffu, sdv[k], 1); sdv[k] += __shfl_xor_sync(0xffffffffu, sdv[k], 2);
      c1[k] = st[k].y * st[k].y * st[k].y * sdv[k] * (1.f / 128);
      c0[k] = -st[k].y * sd[k] * (1.f / 128);
    }
    // pass 2: d pair out; LN(pair) back in place (this warp's rows only: the dWf product reads them after the barrier)
#pragma unroll JU
    for (int j = 0; j < 16; ++j) {
      float d[4];
      if constexpr (KEEP) { d[0] = dk[j][0]; d[1] = dk[j][1]; d[2] = dk[j][2]; d[3] = dk[j][3]; } else { dln(d, j); }
      float2 x0 = *reinterpret_cast<const float2*>(z0 + 8 * j), x1 = *reinterpret_cast<const float2*>(z1 + 8 * j);
      x0.x -= st0.x; x0.y -= st0.x; x1.x -= st1.x; x1.y -= st1.x;
      *reinterpret_cast<float2*>(DZ + (rb + g) * 128 + 8 * j + 2 * q) =
          make_float2(fmaf(st0.y, d[0], fmaf(-c1[0], x0.x, c0[0])), fmaf(st0.y, d[1], fmaf(-c1[0], x0.y, c0[0])));
      *reinterpret_cast<float2*>(DZ + (rb + g + 8) * 128 + 8 * j + 2 * q) =
          make_float2(fmaf(st1.y, d[2], fmaf(-c1[1], x1.x, c0[1])), fmaf(st1.y, d[3], fmaf(-c1[1], x1.y, c0[1])));
      *reinterpret_cast<float2*>(zs + (wr + g) * ZP + 8 * j + 2 * q) = make_float2(x0.x * st0.y, x0.y * st0.y);
      *reinterpret_cast<float2*>(zs + (wr + g + 8) * ZP + 8 * j + 2 * q) = make_float2(x1.x * st1.y, x1.y * st1.y);
    }
    __syncthreads();                                   // LN of the item's rows
    // dWf^T [16 w + m, head 8 t + n] over the item's rows: A = LN^T (m channel, k row), B = dbias^T (k row, n head). TERMS = 3 sums
    // each item's 128 rows in a fresh accumulator and adds it to dacc with IEEE adds: an mma accumulator chained over a block's
    // thousands of rows drifts (the tensor core's fp32 accumulation truncates; L768, ~4000 rows per block: 2.8e-5 relative,
    // linear in the rows), which TF32 products (TERMS = 1) never resolve
    float dit[FLUSH ? NK : 1][4];
    if constexpr (FLUSH) {
#pragma unroll
      for (int t = 0; t < NK; ++t) dit[t][0] = dit[t][1] = dit[t][2] = dit[t][3] = 0.f;
    }
#pragma unroll KU
    for (int kk = 0; kk < 16; ++kk) {
      const int k0 = 8 * kk;
      const float* a0p = zs + (k0 + q) * ZP + 16 * w + g;
      const float va[4] = {a0p[0], a0p[8], a0p[4 * ZP], a0p[4 * ZP + 8]};
      uint32_t xh[4], xl[4];
      split_a<TERMS>(va, xh, xl);
#pragma unroll
      for (int t = 0; t < NK; ++t) {
        const float* bp = ds + (8 * t + g) * DP + k0 + q;
        const uint4 bb = split2(bp[0], bp[4]);
        if constexpr (FLUSH) mma_x<TERMS>(dit[t], xh, xl, bb.x, bb.y, bb.z, bb.w);
        else mma_x<TERMS>(dacc[t], xh, xl, bb.x, bb.y, bb.z, bb.w);
      }
    }
    if constexpr (FLUSH) {
#pragma unroll
      for (int t = 0; t < NK; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) wsum[(t * 4 + e) * 32] += dit[t][e];
    }
    __syncthreads();                                   // before this buffer is refilled
  }
  // dacc[t] (or its shared slots): d0, d1 = channel 16 w + g, heads 8 t + 2 q + {0, 1}; d2, d3 = channel 16 w + g + 8
#pragma unroll
  for (int t = 0; t < NK; ++t)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int h = 8 * t + 2 * q + e;
      if (h < NH) {
        DWF[((long)blockIdx.x * NH + h) * 128 + 16 * w + g] = FLUSH ? wsum[(t * 4 + e) * 32] : dacc[FLUSH ? 0 : t][e];
        DWF[((long)blockIdx.x * NH + h) * 128 + 16 * w + g + 8] = FLUSH ? wsum[(t * 4 + 2 + e) * 32] : dacc[FLUSH ? 0 : t][2 + e];
      }
    }
}

// ------------------------------------------------------------------------------------------------------------ weight pack
// The fp32 training step's weight pack (integrations/bias_only_dit_train.py _pack32) in one launch instead of eight torch cats and
// products: up to 16 segments dst = src (o scale, broadcast over the rows: the folded LayerNorm weights), float4 granular.
// Segment blockIdx.y per grid row (the segment table stays in parameter space: __grid_constant__, a uniform index), 32-bit indices.
// (A per-element search over the table with a 64-bit modulo ran 42 us for ~45 MB: slower than the torch ops it replaced.)
struct PackSeg { const float* src; float* dst; const float* scale; int n4; int period; };
struct PackSegs { PackSeg s[16]; int n; };

__global__ void __launch_bounds__(256) pack_k(const __grid_constant__ PackSegs S) {
  const PackSeg& sg = S.s[blockIdx.y];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < sg.n4; i += gridDim.x * blockDim.x) {
    float4 v = ld4(sg.src + 4 * i);
    if (sg.scale) v = mul4(v, ld4(sg.scale + (4 * i) % sg.period));
    st4(sg.dst + 4 * i, v);
  }
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
// the resident-block build of a row kernel: 2 or 3 from the environment variable (read per call: A/B in one process), else dflt
int minb(const char* var, int dflt) {
  const char* e = std::getenv(var);
  return e && *e ? (std::atoi(e) >= 3 ? 3 : 2) : dflt;
}
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
  auto k = minb("MINIWORLD_BIAS_ONLY_DIT_F32_RESB_MINB", 3) == 3 ? res_adaln_b_k<3> : res_adaln_b_k<2>;
  k<<<grid(k, M, RPB2, 256), 256, 0, S()>>>(CP<float>(x), CP<float>(y), CP<float>(Gg), Gg.stride(0),
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
  f32c(dout, "dout"); f32r(dxt, "dxt"); f32c(x, "x"); f32c(x1st, "x1st"); f32r(G, "G"); f32c(bs2, "bs2"); f32r(Gg, "Gg");
  f32c(bg1, "bg1"); f32c(y, "y"); f32c(dx1, "dx1"); f32c(dy, "dy"); f32r(dG, "dG"); f32r(dGg, "dGg");
  check_part(ps2, M, "ps2"); check_part(pg1, M, "pg1");
  const at::cuda::CUDAGuard gd(dout.device());
  auto k = res_adaln_b_bwd_k;
  const unsigned g = grid(k, M, RB3, 96 * RB3);
  // dxt is read from dG[:, 3D:4D] (d shift2 = dxt): the dxt GEMM writes it there; anything else is copied there first
  TORCH_CHECK(dxt.size(0) == M && dxt.size(1) == D, "res_adaln_b_bwd: dxt [M, 768]");
  if (!(dxt.data_ptr() == (void*)(P<float>(dG) + 3 * D) && dxt.stride(0) == dG.stride(0))) dG.narrow(1, 3 * D, D).copy_(dxt);
  k<<<g, 96 * RB3, 0, S()>>>(CP<float>(dout), CP<float>(x), CP<float2>(x1st), CP<float>(G), G.stride(0),
      CP<float>(bs2), CP<float>(Gg), Gg.stride(0), CP<float>(bg1), CP<float>(y), P<float>(dx1), P<float>(dy), P<float>(dG), dG.stride(0),
      P<float>(dGg), dGg.stride(0), P<float>(ps2), P<float>(pg1), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return g;
}

int64_t adaln_a_bwd(at::Tensor dxa, at::Tensor x, at::Tensor xst, at::Tensor G, at::Tensor bs1, at::Tensor dx1, at::Tensor dx,
                    at::Tensor dG, at::Tensor ps1) {
  const int64_t M = dxa.size(0);
  f32r(dxa, "dxa"); f32c(x, "x"); f32c(xst, "xst"); f32r(G, "G"); f32c(bs1, "bs1"); f32c(dx1, "dx1"); f32c(dx, "dx"); f32r(dG, "dG");
  check_part(ps1, M, "ps1");
  const at::cuda::CUDAGuard gd(dxa.device());
  const unsigned g = grid(adaln_a_bwd_k, M, RPB2, 256);
  // dxa may be the dG[:, D:2D] view the dxa GEMM wrote (d shift1 = dxa): then no copy
  const int copy_sh = !(dxa.data_ptr() == (void*)(P<float>(dG) + D) && dxa.stride(0) == dG.stride(0));
  TORCH_CHECK(dxa.size(0) == M && dxa.size(1) == D, "adaln_a_bwd: dxa [M, 768]");
  adaln_a_bwd_k<<<g, 256, 0, S()>>>(CP<float>(dxa), dxa.stride(0), copy_sh, CP<float>(x), CP<float2>(xst), CP<float>(G), G.stride(0), CP<float>(bs1),
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
  const size_t smem = SR * (L + 4) * 4;
  static bool attr = [] {
    return cudaFuncSetAttribute(softmax_t_k<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SR * 1028 * 4) == cudaSuccess
        && cudaFuncSetAttribute(softmax_t_k<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SR * 1028 * 4) == cudaSuccess;
  }();
  TORCH_CHECK(attr, "softmax_t: shared memory attribute");
  if (L <= 512) softmax_t_k<4><<<(unsigned)(R / SR), 256, smem, S()>>>(CP<float>(bias), P<float>(p), P<float>(pt), mk, (int)L);
  else softmax_t_k<8><<<(unsigned)(R / SR), 256, smem, S()>>>(CP<float>(bias), P<float>(p), P<float>(pt), mk, (int)L);
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

// the training pair bias / its backward on TF32 tensor cores (terms 1: TF32; 3: split fp32, ~exact)
using PbFwd = void (*)(const float*, const float*, float*, float2*, long, float);
using PbBwd = void (*)(const float*, const float*, const float2*, const float*, float*, float*, long);
PbFwd pb_fwd(int64_t nh, int64_t terms) {
  TORCH_CHECK(terms == 1 || terms == 3, "pair bias: terms 1 or 3");
  if (terms == 1) return nh == 12 ? pair_bias_tc_k<12, 1> : nh == 16 ? pair_bias_tc_k<16, 1> : pair_bias_tc_k<24, 1>;
  return nh == 12 ? pair_bias_tc_k<12, 3> : nh == 16 ? pair_bias_tc_k<16, 3> : pair_bias_tc_k<24, 3>;
}
PbBwd pb_bwd(int64_t nh, int64_t terms) {
  TORCH_CHECK(terms == 1 || terms == 3, "pair bias: terms 1 or 3");
  if (terms == 1) return nh == 12 ? pair_bias_bwd_tc_k<12, 1> : nh == 16 ? pair_bias_bwd_tc_k<16, 1> : pair_bias_bwd_tc_k<24, 1>;
  return nh == 12 ? pair_bias_bwd_tc_k<12, 3> : nh == 16 ? pair_bias_bwd_tc_k<16, 3> : pair_bias_bwd_tc_k<24, 3>;
}

void pair_bias_tc(at::Tensor z, at::Tensor wf, at::Tensor bias, at::Tensor pst, double eps, int64_t terms) {
  f32c(z, "pair"); f32c(wf, "Wf"); f32c(bias, "bias"); f32c(pst, "pst");
  const int64_t R = z.size(0), nh = wf.size(0);
  TORCH_CHECK(z.dim() == 2 && z.size(1) == 128 && R % 16 == 0 && wf.size(1) == 128 && (nh == 12 || nh == 16 || nh == 24)
              && bias.numel() == nh * R && pst.numel() == 2 * R,
              "pair_bias_tc: pair [R, 128] (R a multiple of 16), Wf [12, 16 or 24, 128], bias [nh, R], pst [R, 2] fp32");
  const at::cuda::CUDAGuard gd(z.device());
  const PbFwd k = pb_fwd(nh, terms);
  int per = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 256, 0);
  const unsigned g = (unsigned)std::max<int64_t>(1, std::min<int64_t>((R / 16 + WPB - 1) / WPB, (int64_t)std::max(per, 1) * nsm()));
  k<<<g, 256, 0, S()>>>(CP<float>(z), CP<float>(wf), P<float>(bias), P<float2>(pst), R, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NH> size_t pb32_smem() { return (size_t)PB32<NH>::SMEM; }

int64_t pair_bias_bwd_tc(at::Tensor db, at::Tensor z, at::Tensor pst, at::Tensor wf, at::Tensor dz, at::Tensor dwf, int64_t terms) {
  f32c(db, "dbias"); f32c(z, "pair"); f32c(pst, "pst"); f32c(wf, "Wf"); f32c(dz, "d pair"); f32c(dwf, "dWf partials");
  const int64_t R = z.size(0), nh = wf.size(0);
  TORCH_CHECK(z.dim() == 2 && z.size(1) == 128 && R % 128 == 0 && wf.size(1) == 128 && (nh == 12 || nh == 16 || nh == 24)
              && db.numel() == nh * R && pst.numel() == 2 * R && dz.sizes() == z.sizes(),
              "pair_bias_bwd_tc: pair / d pair [R, 128] (R a multiple of 128), dbias [nh, R], Wf [12, 16 or 24, 128]");
  const at::cuda::CUDAGuard gd(z.device());
  static bool attr = [] {
    bool ok = true;
#define PB_ATTR(NH_, T_) ok = ok && cudaFuncSetAttribute(pair_bias_bwd_tc_k<NH_, T_>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                                         (int)pb32_smem<NH_>()) == cudaSuccess;
    PB_ATTR(12, 1) PB_ATTR(16, 1) PB_ATTR(24, 1) PB_ATTR(12, 3) PB_ATTR(16, 3) PB_ATTR(24, 3)
#undef PB_ATTR
    return ok;
  }();
  TORCH_CHECK(attr, "pair_bias_bwd_tc: shared memory attribute");
  const PbBwd k = pb_bwd(nh, terms);
  const size_t smem = nh == 12 ? pb32_smem<12>() : nh == 16 ? pb32_smem<16>() : pb32_smem<24>();
  int per = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 256, smem);
  const unsigned g = (unsigned)std::max<int64_t>(1, std::min<int64_t>({R / 128, (int64_t)std::max(per, 1) * nsm(), partial_rows(0)}));
  TORCH_CHECK(dwf.numel() >= (int64_t)g * nh * 128, "pair_bias_bwd_tc: dWf partials [partial_rows, nh, 128]");
  k<<<g, 256, smem, S()>>>(CP<float>(db), CP<float>(z), CP<float2>(pst), CP<float>(wf), P<float>(dz), P<float>(dwf), R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return g;
}

// dst[i] = src[i] (o scale[i % scale.numel()] when scale is not empty), fp32 contiguous, sizes multiples of 4; at most 16 segments
void pack(std::vector<at::Tensor> src, std::vector<at::Tensor> dst, std::vector<at::Tensor> scale) {
  TORCH_CHECK(src.size() == dst.size() && src.size() == scale.size() && !src.empty() && src.size() <= 16, "pack: 1..16 segments");
  PackSegs segs{};
  long longest = 0;
  for (size_t k = 0; k < src.size(); ++k) {
    f32c(src[k], "pack src"); f32c(dst[k], "pack dst");
    TORCH_CHECK(src[k].numel() == dst[k].numel() && src[k].numel() % 4 == 0, "pack: src / dst of one segment alike, 4 | numel");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(src[k].data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(dst[k].data_ptr()) % 16 == 0,
                "pack: 16-byte aligned");
    const float* sc = nullptr;
    int period = 4;
    if (scale[k].numel()) {
      f32c(scale[k], "pack scale");
      TORCH_CHECK(scale[k].numel() % 4 == 0 && src[k].numel() % scale[k].numel() == 0, "pack: scale period divides the segment");
      sc = CP<float>(scale[k]);
      period = (int)scale[k].numel();
    }
    TORCH_CHECK(src[k].numel() < (1L << 31), "pack: segments below 2^31 elements");
    segs.s[k] = PackSeg{CP<float>(src[k]), P<float>(dst[k]), sc, (int)(src[k].numel() / 4), period};
    longest = std::max<long>(longest, src[k].numel() / 4);
  }
  segs.n = (int)src.size();
  const at::cuda::CUDAGuard gd(src[0].device());
  const dim3 g((unsigned)std::max<long>(1, std::min<long>((longest + 255) / 256, 2L * nsm())), (unsigned)segs.n);
  pack_k<<<g, 256, 0, S()>>>(segs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// (name, registers, local-memory bytes) of every kernel of this extension: the tests hold them to <= 128 registers without spills
std::vector<std::tuple<std::string, int64_t, int64_t>> func_attrs() {
  std::vector<std::tuple<std::string, int64_t, int64_t>> out;
  auto add = [&](const char* n, const void* f) {
    cudaFuncAttributes a{};
    TORCH_CHECK(cudaFuncGetAttributes(&a, f) == cudaSuccess, "cudaFuncGetAttributes ", n);
    out.emplace_back(n, (int64_t)a.numRegs, (int64_t)a.localSizeBytes);
  };
#define FA(...) add(#__VA_ARGS__, (const void*)__VA_ARGS__)
  FA(cond_ln_k); FA(adaln_a_k); FA(res_adaln_b_k<2>); FA(res_adaln_b_k<3>); FA(res_c_k); FA(res_c_bwd_k); FA(res_adaln_b_bwd_k);
  FA(adaln_a_bwd_k); FA(cond_bwd_k); FA(gate_bwd_k<6>); FA(gate_bwd_k<8>); FA(swiglu_k); FA(swiglu_bwd_k); FA(softmax_rows_k<4>);
  FA(softmax_rows_k<8>); FA(softmax_rows_k<16>); FA(softmax_t_k<4>); FA(softmax_t_k<8>); FA(pair_bias_k<12>); FA(pair_bias_k<16>);
  FA(pair_bias_k<24>); FA(pair_bias_bwd_k<12>); FA(pair_bias_bwd_k<16>); FA(pair_bias_bwd_k<24>);
  FA(pair_bias_tc_k<12, 1>); FA(pair_bias_tc_k<16, 1>); FA(pair_bias_tc_k<24, 1>); FA(pair_bias_tc_k<12, 3>); FA(pair_bias_tc_k<16, 3>);
  FA(pair_bias_tc_k<24, 3>); FA(pair_bias_bwd_tc_k<12, 1>); FA(pair_bias_bwd_tc_k<16, 1>); FA(pair_bias_bwd_tc_k<24, 1>);
  FA(pair_bias_bwd_tc_k<12, 3>); FA(pair_bias_bwd_tc_k<16, 3>); FA(pair_bias_bwd_tc_k<24, 3>); FA(pack_k);
#undef FA
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("func_attrs", &func_attrs);
  m.def("pair_bias_tc_cuda", &pair_bias_tc);
  m.def("pair_bias_bwd_tc_cuda", &pair_bias_bwd_tc);
  m.def("pack_cuda", &pack);
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
