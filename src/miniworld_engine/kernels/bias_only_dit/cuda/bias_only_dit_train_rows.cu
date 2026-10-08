// Bias-only token DiT TRAINING row kernels (CUDA): the elementwise / row steps between the GEMMs of the fused training block
// (integrations/bias_only_dit_train.py), forward and backward. Same arithmetic as the token DiT's training rows
// (kernels/conditioned_transition/cuda/token_dit_train_rows.cu), laid out for the memory rate:
//
//   * the 768-wide rows take two warps (64 threads) each, the 384-wide conditioning rows one warp; a lane owns float4 chunks lane,
//     lane + 32, ... (a warp access is 512 contiguous bytes of fp32, 256 of bf16); every load of a row is issued before its math;
//   * persistent blocks walk rows with a grid stride (as many as are resident), so per-column gradient sums (the biases)
//     accumulate in registers and leave as one row per block of a partial buffer, summed by finalize;
//   * the block input is read as it comes (bf16) and the fp32 residual x1 = x + sigmoid(g1) y is never stored: each kernel
//     that needs it rebuilds it with one FMA (the same bits everywhere).
//
//   forward   pair_bias    bias [16, R] = LN(pair) Wf^T on mma.sync, (mean, rstd)                    (R = L^2 pair rows of 128)
//             cond_ln      c_hat = LN(c) bf16, (mean, rstd)                                          (d_cond = 384)
//             adaln_a      xa = sigmoid(G[:, 0:D] + bs1) LN(x) + G[:, D:2D]                          (x = the input, D = 768)
//             res_adaln_b  x1 = x + sigmoid(Gg[:, 0:D] + bg1) y;  xt = sigmoid(G[:, 2D:3D] + bs2) LN(x1) + G[:, 3D:4D]
//             res_c        out = x1 + sigmoid(Gg[:, D:2D] + bg2) z
//             swiglu       h = silu(a) b (the path uses the expand GEMM's epilogue; kept for probes)
//   backward  res_c_bwd, swiglu_bwd, res_adaln_b_bwd, adaln_a_bwd, cond_bwd, unfold (AdaLN / cond-LN weights), finalize (the small gradients),
//             pair_bias_bwd  d pair and dWf = dbias LN(pair) on mma.sync (LN(pair) rebuilt on chip)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"

namespace {
using namespace tdr;
using bf = __nv_bfloat16;

constexpr int D = 768, NC = 6, DC = 384, NCC = 3, WPB = 8;
#ifndef MINB
#define MINB 3                     // resident blocks per SM the D-wide forward kernels are compiled for (<= 85 registers)
#endif
#ifndef MINB_LN
#define MINB_LN 4                  // the two forward LayerNorm kernels: their loads stay packed until used (<= 64 registers)
#endif
#ifndef MINB_BWD
#define MINB_BWD 2                 // the two LayerNorm backward kernels hold more per row (<= 128 registers)
#endif

__device__ __forceinline__ float sg(float v) { return __fdividef(1.f, 1.f + __expf(-v)); }
__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 mul4(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ __forceinline__ float4 sig4(float4 a) { return make_float4(sg(a.x), sg(a.y), sg(a.z), sg(a.w)); }
__device__ __forceinline__ float sum4(float4 a) { return (a.x + a.y) + (a.z + a.w); }
__device__ __forceinline__ float4 bfround(float4 a) {       // the value a bf16 store keeps (sums use what was stored)
  return make_float4(__bfloat162float(__float2bfloat16_rn(a.x)), __bfloat162float(__float2bfloat16_rn(a.y)),
                     __bfloat162float(__float2bfloat16_rn(a.z)), __bfloat162float(__float2bfloat16_rn(a.w)));
}
__device__ __forceinline__ float4 dsig4(float4 v, float4 s) {   // v * s * (1 - s)
  return make_float4(v.x * s.x * (1.f - s.x), v.y * s.y * (1.f - s.y), v.z * s.z * (1.f - s.z), v.w * s.w * (1.f - s.w));
}
__device__ __forceinline__ float4 zero4() { return make_float4(0.f, 0.f, 0.f, 0.f); }
// eight bf16 (16 bytes) <-> float
__device__ __forceinline__ void unpack8(uint4 u, float (&f)[8]) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int k = 0; k < 4; ++k) { const float2 t = __bfloat1622float2(h[k]); f[2 * k] = t.x; f[2 * k + 1] = t.y; }
}
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int k = 0; k < 4; ++k) h[k] = __floats2bfloat162_rn(f[2 * k], f[2 * k + 1]);
  return u;
}
__device__ __forceinline__ uint4 ld8(const bf* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ void st8(bf* p, uint4 v) { *reinterpret_cast<uint4*>(p) = v; }

// the warp's first row, the stride, and this lane's column of chunk j
#define ROWS_BEGIN                                                                                   \
  const int lane = threadIdx.x % 32;                                                                 \
  const long wid = (long)blockIdx.x * WPB + threadIdx.x / 32, nw = (long)gridDim.x * WPB;
#define COL(j) (((j) * 32 + lane) * 4)

template <int N>
__device__ __forceinline__ void stats(const float4 (&x)[N], float eps, float& mean, float& rstd) {
  constexpr float inv = 1.f / (N * 128);
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < N; ++j) s += sum4(x[j]);
  mean = warp_sum(s) * inv;
  float q = 0.f;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    const float4 d = make_float4(x[j].x - mean, x[j].y - mean, x[j].z - mean, x[j].w - mean);
    q += sum4(mul4(d, d));
  }
  rstd = rsqrtf(warp_sum(q) * inv + eps);
}

// ------------------------------------------------------------------------------------------------------------- forward
__global__ void __launch_bounds__(WPB * 32) cond_ln_k(const bf* __restrict__ C, bf* __restrict__ CHAT, float2* __restrict__ CST,
    long M, float eps) {
  ROWS_BEGIN
  for (long r = wid; r < M; r += nw) {
    float4 c[NCC];
#pragma unroll
    for (int j = 0; j < NCC; ++j) c[j] = V4<bf>::load(C + r * DC + COL(j));
    float mean, rstd;
    stats<NCC>(c, eps, mean, rstd);
#pragma unroll
    for (int j = 0; j < NCC; ++j)
      V4<bf>::store(CHAT + r * DC + COL(j), make_float4((c[j].x - mean) * rstd, (c[j].y - mean) * rstd, (c[j].z - mean) * rstd,
                                                        (c[j].w - mean) * rstd));
    if (lane == 0) CST[r] = make_float2(mean, rstd);
  }
}

// ---- D-wide rows: two warps per row (64 threads; lane l of warp half hw owns chunks hw * 3 + j, j < 3, of the six), four rows per
// block at a time. A thread holds 12 of the row's 768 values per tensor, so the backward kernels fit their live state and column
// sums in registers without spilling (one warp per row needed ~200). The row's sums combine the two warps through shared memory
// under a named barrier of the pair (ids 1..4), double-buffered by call parity.
constexpr int RPB2 = 4;
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

// four values loaded and kept as they are in memory (bf16: 8 bytes, two registers) until converted where used: the forward
// LayerNorm kernels hold a whole row slice per thread, and packed operands leave room for more resident blocks
template <typename T> struct R4;
template <> struct R4<bf> {
  using T = uint2;
  static __device__ __forceinline__ T ld(const bf* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }
  static __device__ __forceinline__ float4 f(T u) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
  }
};
template <> struct R4<float> {
  using T = float4;
  static __device__ __forceinline__ T ld(const float* p) { return V4<float>::load(p); }
  static __device__ __forceinline__ float4 f(T u) { return u; }
};

// LayerNorm statistics of a 768-row held as 12 values per thread of the slot's two warps, in ONE exchange: each thread's (mean,
// sum of squared deviations) over its values, merged pairwise (Chan et al.; equal counts at every level, so exact and stable)
// over the warp by shuffles and over the two warps through shared memory. Returns (mean, rstd).
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
    q = q + qo + dl * dl * (0.5f * n);                 // nA nB / (nA + nB) with nA = nB = n
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

// the attention residual x1 = x + sigmoid(g1) y, rebuilt wherever it is needed instead of kept in fp32 (one fused multiply-add,
// so every kernel gets the same bits and the saved LayerNorm statistics stay exact for it)
__device__ __forceinline__ float4 resid4(float4 x, float4 sg1, float4 y) {
  return make_float4(__fmaf_rn(sg1.x, y.x, x.x), __fmaf_rn(sg1.y, y.y, x.y), __fmaf_rn(sg1.z, y.z, x.z), __fmaf_rn(sg1.w, y.w, x.w));
}

// the block's four row slots add their column sums in shared memory; the block leaves one row of the partial buffer
__device__ __forceinline__ void block_partial2(const float4 (&acc)[3], float (&red)[RPB2][768], float* part) {
  const int lane = threadIdx.x % 32, hw = (threadIdx.x / 32) & 1, slot = threadIdx.x / 64;
#pragma unroll
  for (int j = 0; j < 3; ++j) V4<float>::store(&red[slot][COL2(j)], acc[j]);
  __syncthreads();
  for (int c = threadIdx.x; c < 768; c += blockDim.x) part[(long)blockIdx.x * 768 + c] = (red[0][c] + red[1][c]) + (red[2][c] + red[3][c]);
  __syncthreads();
}

__device__ __forceinline__ void block_partial_smem(float4 (&acc)[RPB2][D / 4], float* part) {
  __syncthreads();
  for (int c = threadIdx.x; c < D / 4; c += blockDim.x)
    V4<float>::store(part + (long)blockIdx.x * D + 4 * c, add4(add4(acc[0][c], acc[1][c]), add4(acc[2][c], acc[3][c])));
}

template <typename XT>
__global__ void __launch_bounds__(256, MINB_LN) adaln_a_k(const XT* __restrict__ X, const bf* __restrict__ G, long sg_,
    const float* __restrict__ BS, bf* __restrict__ XA, float2* __restrict__ XST, long M, float eps) {
  ROWS2_BEGIN
  for (long r = sid; r < M; r += ns) {
    float4 x[3];
    typename R4<XT>::T xr[3];
    R4<bf>::T sr[3], shr[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      xr[j] = R4<XT>::ld(X + r * D + COL2(j));
      sr[j] = R4<bf>::ld(G + r * sg_ + COL2(j)); shr[j] = R4<bf>::ld(G + r * sg_ + D + COL2(j));
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) x[j] = R4<XT>::f(xr[j]);
    const float2 ms = row_stats2(x, eps, xr_, par_, slot, hw, lane);
    const float mean = ms.x, rstd = ms.y;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float4 sj = sig4(add4(R4<bf>::f(sr[j]), V4<float>::load(BS + COL2(j)))), shj = R4<bf>::f(shr[j]);
      V4<bf>::store(XA + r * D + COL2(j), make_float4(sj.x * (x[j].x - mean) * rstd + shj.x, sj.y * (x[j].y - mean) * rstd + shj.y,
                                                      sj.z * (x[j].z - mean) * rstd + shj.z, sj.w * (x[j].w - mean) * rstd + shj.w));
    }
    if (hw == 0 && lane == 0) XST[r] = make_float2(mean, rstd);
  }
}

template <typename XT>
__global__ void __launch_bounds__(256, MINB_LN) res_adaln_b_k(const XT* __restrict__ X, const bf* __restrict__ Y,
    const bf* __restrict__ GG, long sgg, const float* __restrict__ BG1, const bf* __restrict__ G, long sg_,
    const float* __restrict__ BS2, bf* __restrict__ XT_, float2* __restrict__ X1ST, long M, float eps) {
  ROWS2_BEGIN
  for (long r = sid; r < M; r += ns) {
    float4 x[3];
    typename R4<XT>::T xr[3];
    R4<bf>::T yr[3], gr[3], sr[3], shr[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      xr[j] = R4<XT>::ld(X + r * D + COL2(j)); yr[j] = R4<bf>::ld(Y + r * D + COL2(j)); gr[j] = R4<bf>::ld(GG + r * sgg + COL2(j));
      sr[j] = R4<bf>::ld(G + r * sg_ + 2 * D + COL2(j)); shr[j] = R4<bf>::ld(G + r * sg_ + 3 * D + COL2(j));
    }
#pragma unroll
    for (int j = 0; j < 3; ++j)
      x[j] = resid4(R4<XT>::f(xr[j]), sig4(add4(R4<bf>::f(gr[j]), V4<float>::load(BG1 + COL2(j)))), R4<bf>::f(yr[j]));
    const float2 ms = row_stats2(x, eps, xr_, par_, slot, hw, lane);
    const float mean = ms.x, rstd = ms.y;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float4 sj = sig4(add4(R4<bf>::f(sr[j]), V4<float>::load(BS2 + COL2(j)))), shj = R4<bf>::f(shr[j]);
      V4<bf>::store(XT_ + r * D + COL2(j), make_float4(sj.x * (x[j].x - mean) * rstd + shj.x, sj.y * (x[j].y - mean) * rstd + shj.y,
                                                       sj.z * (x[j].z - mean) * rstd + shj.z, sj.w * (x[j].w - mean) * rstd + shj.w));
    }
    if (hw == 0 && lane == 0) X1ST[r] = make_float2(mean, rstd);
  }
}

template <typename XT, typename OT>
__global__ void __launch_bounds__(256, MINB) res_c_k(const XT* __restrict__ X, const bf* __restrict__ Y, const bf* __restrict__ Z,
    const bf* __restrict__ GG, long sgg, const float* __restrict__ BG1, const float* __restrict__ BG2, OT* __restrict__ OUT, long M) {
  ROWS2_BEGIN
  (void)par_; (void)xr_;
  for (long r = sid; r < M; r += ns) {
    float4 x[3], y[3], z[3], g1[3], g2[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      x[j] = V4<XT>::load(X + r * D + COL2(j)); y[j] = V4<bf>::load(Y + r * D + COL2(j)); z[j] = V4<bf>::load(Z + r * D + COL2(j));
      g1[j] = V4<bf>::load(GG + r * sgg + COL2(j)); g2[j] = V4<bf>::load(GG + r * sgg + D + COL2(j));
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float4 x1 = resid4(x[j], sig4(add4(g1[j], V4<float>::load(BG1 + COL2(j)))), y[j]);
      V4<OT>::store(OUT + r * D + COL2(j), add4(x1, mul4(sig4(add4(g2[j], V4<float>::load(BG2 + COL2(j)))), z[j])));
    }
  }
}

// ------------------------------------------------------------------------------------------------------------ backward
template <typename OT>
__global__ void __launch_bounds__(256, MINB) res_c_bwd_k(const OT* __restrict__ DOUT, const bf* __restrict__ Z, const bf* __restrict__ GG,
    long sgg, const float* __restrict__ BG2, bf* __restrict__ DZ, bf* __restrict__ DGG, long sdg, float* __restrict__ PG2, long M) {
  ROWS2_BEGIN
  (void)par_; (void)xr_;
  __shared__ float red[RPB2][768];
  float4 acc[3] = {zero4(), zero4(), zero4()};
  for (long r = sid; r < M; r += ns) {
    float4 dout[3], z[3], g[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      dout[j] = V4<OT>::load(DOUT + r * D + COL2(j)); z[j] = V4<bf>::load(Z + r * D + COL2(j)); g[j] = V4<bf>::load(GG + r * sgg + D + COL2(j));
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float4 s = sig4(add4(g[j], V4<float>::load(BG2 + COL2(j))));
      V4<bf>::store(DZ + r * D + COL2(j), mul4(dout[j], s));
      const float4 dg = bfround(dsig4(mul4(dout[j], z[j]), s));
      V4<bf>::store(DGG + r * sdg + D + COL2(j), dg);
      acc[j] = add4(acc[j], dg);
    }
  }
  block_partial2(acc, red, PG2);
}

template <typename XT, typename OT>
__global__ void __launch_bounds__(256, MINB_BWD) res_adaln_b_bwd_k(const OT* __restrict__ DOUT, const bf* __restrict__ DXT, long sdxt,
    bool cpy, const XT* __restrict__ X, const float2* __restrict__ X1ST, const bf* __restrict__ G, long sg_, const float* __restrict__ BS2,
    const bf* __restrict__ GG, long sgg, const float* __restrict__ BG1, const bf* __restrict__ Y, float* __restrict__ DX1,
    bf* __restrict__ DY, bf* __restrict__ DG, long sdg, bf* __restrict__ DGG, long sdgg, float* __restrict__ PS2,
    float* __restrict__ PG1, long M) {
  ROWS2_BEGIN
  // the column sums accumulate in shared memory (each thread its own slot's columns; no barrier until the end): in registers
  // they pushed the kernel past 128 and into spills
  __shared__ float4 acc_s[RPB2][D / 4], acc_g[RPB2][D / 4];
#pragma unroll
  for (int j = 0; j < 3; ++j) { acc_s[slot][COL2(j) / 4] = zero4(); acc_g[slot][COL2(j) / 4] = zero4(); }
  for (long r = sid; r < M; r += ns) {
    float4 dxt[3], xh[3], s2[3], dout[3], g1[3], y[3];
    const float2 st = X1ST[r];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      dxt[j] = V4<bf>::load(DXT + r * sdxt + COL2(j));
      xh[j] = V4<XT>::load(X + r * D + COL2(j));
      s2[j] = V4<bf>::load(G + r * sg_ + 2 * D + COL2(j));
      dout[j] = V4<OT>::load(DOUT + r * D + COL2(j)); g1[j] = V4<bf>::load(GG + r * sgg + COL2(j)); y[j] = V4<bf>::load(Y + r * D + COL2(j));
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {                                      // x1 rebuilt, then normalised; g1 becomes sigmoid(g1)
      g1[j] = sig4(add4(g1[j], V4<float>::load(BG1 + COL2(j))));
      const float4 x1 = resid4(xh[j], g1[j], y[j]);
      xh[j] = make_float4((x1.x - st.x) * st.y, (x1.y - st.x) * st.y, (x1.z - st.x) * st.y, (x1.w - st.x) * st.y);
    }
    float m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      s2[j] = sig4(add4(s2[j], V4<float>::load(BS2 + COL2(j))));
      if (cpy) V4<bf>::store(DG + r * sdg + 3 * D + COL2(j), dxt[j]);   // d shift2 = dxt (unless the GEMM wrote it there)
      const float4 ds2 = bfround(dsig4(mul4(dxt[j], xh[j]), s2[j]));
      V4<bf>::store(DG + r * sdg + 2 * D + COL2(j), ds2);
      acc_s[slot][COL2(j) / 4] = add4(acc_s[slot][COL2(j) / 4], ds2);
      dxt[j] = mul4(dxt[j], s2[j]);                                  // dxh
      m1 += sum4(dxt[j]);
      m2 += sum4(mul4(dxt[j], xh[j]));
    }
    const float2 m = ROWSUM2(m1, m2);
    m1 = m.x * (1.f / D); m2 = m.y * (1.f / D);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float4 dx1 = make_float4(dout[j].x + st.y * (dxt[j].x - m1 - xh[j].x * m2), dout[j].y + st.y * (dxt[j].y - m1 - xh[j].y * m2),
                                     dout[j].z + st.y * (dxt[j].z - m1 - xh[j].z * m2), dout[j].w + st.y * (dxt[j].w - m1 - xh[j].w * m2));
      V4<float>::store(DX1 + r * D + COL2(j), dx1);
      V4<bf>::store(DY + r * D + COL2(j), mul4(dx1, g1[j]));
      const float4 dg1 = bfround(dsig4(mul4(dx1, y[j]), g1[j]));
      V4<bf>::store(DGG + r * sdgg + COL2(j), dg1);
      acc_g[slot][COL2(j) / 4] = add4(acc_g[slot][COL2(j) / 4], dg1);
    }
  }
  block_partial_smem(acc_s, PS2);
  block_partial_smem(acc_g, PG1);
}

template <typename XT, typename OT>
__global__ void __launch_bounds__(256, MINB_BWD) adaln_a_bwd_k(const bf* __restrict__ DXA, long sdxa, bool cpy, const XT* __restrict__ X,
    const float2* __restrict__ XST,
    const bf* __restrict__ G, long sg_, const float* __restrict__ BS1, const float* __restrict__ DX1, OT* __restrict__ DX,
    bf* __restrict__ DG, long sdg, float* __restrict__ PS1, long M) {
  ROWS2_BEGIN
  __shared__ float red[RPB2][768];
  float4 acc[3] = {zero4(), zero4(), zero4()};
  for (long r = sid; r < M; r += ns) {
    float4 dxa[3], xh[3], s[3], dx1[3];
    const float2 st = XST[r];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      dxa[j] = V4<bf>::load(DXA + r * sdxa + COL2(j));
      const float4 x = V4<XT>::load(X + r * D + COL2(j));
      xh[j] = make_float4((x.x - st.x) * st.y, (x.y - st.x) * st.y, (x.z - st.x) * st.y, (x.w - st.x) * st.y);
      s[j] = V4<bf>::load(G + r * sg_ + COL2(j));
      dx1[j] = V4<float>::load(DX1 + r * D + COL2(j));
    }
    float m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      s[j] = sig4(add4(s[j], V4<float>::load(BS1 + COL2(j))));
      if (cpy) V4<bf>::store(DG + r * sdg + D + COL2(j), dxa[j]);       // d shift1 = dxa (unless the GEMM wrote it there)
      const float4 ds1 = bfround(dsig4(mul4(dxa[j], xh[j]), s[j]));
      V4<bf>::store(DG + r * sdg + COL2(j), ds1);
      acc[j] = add4(acc[j], ds1);
      dxa[j] = mul4(dxa[j], s[j]);                                   // dxh
      m1 += sum4(dxa[j]);
      m2 += sum4(mul4(dxa[j], xh[j]));
    }
    const float2 m = ROWSUM2(m1, m2);
    m1 = m.x * (1.f / D); m2 = m.y * (1.f / D);
#pragma unroll
    for (int j = 0; j < 3; ++j)
      V4<OT>::store(DX + r * D + COL2(j), make_float4(dx1[j].x + st.y * (dxa[j].x - m1 - xh[j].x * m2), dx1[j].y + st.y * (dxa[j].y - m1 - xh[j].y * m2),
                                                      dx1[j].z + st.y * (dxa[j].z - m1 - xh[j].z * m2), dx1[j].w + st.y * (dxa[j].w - m1 - xh[j].w * m2)));
  }
  block_partial2(acc, red, PS1);
}

template <typename OT>
__global__ void __launch_bounds__(WPB * 32) cond_bwd_k(const bf* __restrict__ DCHAT, const bf* __restrict__ DCG, const bf* __restrict__ C,
    const float2* __restrict__ CST, OT* __restrict__ DCO, long M) {
  ROWS_BEGIN
  for (long r = wid; r < M; r += nw) {
    float4 dch[NCC], ch[NCC], dcg[NCC];
    const float2 st = CST[r];
#pragma unroll
    for (int j = 0; j < NCC; ++j) {
      dch[j] = V4<bf>::load(DCHAT + r * DC + COL(j)); dcg[j] = V4<bf>::load(DCG + r * DC + COL(j));
      const float4 c = V4<bf>::load(C + r * DC + COL(j));
      ch[j] = make_float4((c.x - st.x) * st.y, (c.y - st.x) * st.y, (c.z - st.x) * st.y, (c.w - st.x) * st.y);
    }
    float m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int j = 0; j < NCC; ++j) { m1 += sum4(dch[j]); m2 += sum4(mul4(dch[j], ch[j])); }
    m1 = warp_sum(m1) * (1.f / DC);
    m2 = warp_sum(m2) * (1.f / DC);
#pragma unroll
    for (int j = 0; j < NCC; ++j)
      V4<OT>::store(DCO + r * DC + COL(j), make_float4(st.y * (dch[j].x - m1 - ch[j].x * m2) + dcg[j].x, st.y * (dch[j].y - m1 - ch[j].y * m2) + dcg[j].y,
                                                       st.y * (dch[j].z - m1 - ch[j].z * m2) + dcg[j].z, st.y * (dch[j].w - m1 - ch[j].w * m2) + dcg[j].w));
  }
}

// ------------------------------------------------------------------------------------------------------------ pair bias
// pair_ln_bwd with d LN(pair)[r] = sum_h dbias[h, r] Wf[h] formed in the kernel (no [R, 128] fp32 dph in memory), and LN(pair)
// written back (bf16) for the dWf = dbias LN(pair) GEMM. A lane holds 4 of a row's 128 channels and the 16 x 4 Wf values; a warp
// takes 4 rows at a time so their reduction chains interleave.

// ------------------------------------------------------------------------------------------ pair bias on tensor cores
// The attention's logits come from the pair alone: bias[h, r] = sum_c LN(pair)[r, c] Wf[h, c] (Wf = Wb diag(w), the LN affine
// folded in), r = i L + j over the R = L^2 pair rows of 128. Both directions run the 16 x 128 x 16 products on mma.sync
// (m16n8k16 bf16, fp32 accumulate), so LN(pair) never reaches HBM: the forward writes only the bias and the row statistics, the
// backward only d pair and a [16, 128] partial of dWf per warp.
//
// Column order: a lane (g = lane / 4, q = lane % 4) of a 16-row tile loads rows g and g + 8, 16-byte chunks q + 4 m (m < 4) --
// 64 contiguous bytes per row and load across the four lanes of a row -- and the product's K (forward) / N (backward) slots are
// assigned to those columns: the reduction order inside an mma is free, and the weight fragments follow the same map.
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// four 8 x 8 b16 matrices from shared memory; lane l gives the row address of row l % 8 of matrix l / 8
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm4t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma1688(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void ldsm2t(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void ldsm2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t pack2(bf lo, bf hi) {
  __nv_bfloat162 v; v.x = lo; v.y = hi;
  return *reinterpret_cast<const uint32_t*>(&v);
}
// the column of K slot (q, s) (s < 4: slots 2q, 2q + 1, 2q + 8, 2q + 9) of forward k-step kk
__device__ __forceinline__ int fcol(int kk, int q, int s) { return 8 * (q + 4 * (kk / 2)) + 4 * (kk % 2) + s; }
// the column of N slot n of backward n-tile t
__device__ __forceinline__ int bcol(int t, int n) { return 8 * (n / 2 + 4 * (t / 4)) + 2 * (t % 4) + n % 2; }

// forward: 64 rows per warp item (four 16-row tiles), the bias staged per warp and stored as 128 bytes per head. NH heads (16 or
// 24; 12 padded to 16 with zero weights) are NHP / 8 n-tiles of the product.
template <int NH>
__global__ void __launch_bounds__(WPB * 32, 2) pair_bias_k(const bf* __restrict__ Z, const bf* __restrict__ WF, bf* __restrict__ BIAS,
    float2* __restrict__ PST, long R, float eps) {
  constexpr int NHP = (NH + 7) / 8 * 8, NT = NHP / 8;
  __shared__ __align__(16) bf stg[WPB][NHP][64 + 8];
  __shared__ uint2 bw[8][NT][32];                      // Wf^T fragments per k-step, n-tile (heads 8 t ..) and lane: (b0, b1)
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32, g = lane / 4, q = lane % 4;
  const long wid = (long)blockIdx.x * WPB + w, nw = (long)gridDim.x * WPB;
  for (int i = threadIdx.x; i < 8 * NT * 32; i += blockDim.x) {
    const int l = i % 32, t = (i / 32) % NT, kk = i / (32 * NT);
    auto f = [&](int h) {
      return 8 * t + l / 4 < NH ? *reinterpret_cast<const uint32_t*>(WF + (8 * t + l / 4) * 128 + fcol(kk, l % 4, 2 * h)) : 0u;
    };
    bw[kk][t][l] = make_uint2(f(0), f(1));
  }
  __syncthreads();
  for (long r0 = wid * 64; r0 < R; r0 += nw * 64) {
#pragma unroll 1
    for (int tt = 0; tt < 4; ++tt) {
      const long rb = r0 + tt * 16;
      uint4 u[2][4];
#pragma unroll
      for (int hr = 0; hr < 2; ++hr)
#pragma unroll
        for (int m = 0; m < 4; ++m) u[hr][m] = *reinterpret_cast<const uint4*>(Z + (rb + g + 8 * hr) * 128 + 8 * (q + 4 * m));
      float mean[2], rs[2];
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        float v[8], s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int m = 0; m < 4; ++m) { unpack8(u[hr][m], v); for (int e = 0; e < 8; ++e) s1 += v[e]; }
        s1 += __shfl_xor_sync(0xffffffffu, s1, 1); s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
        mean[hr] = s1 * (1.f / 128);
#pragma unroll
        for (int m = 0; m < 4; ++m) { unpack8(u[hr][m], v); for (int e = 0; e < 8; ++e) s2 += (v[e] - mean[hr]) * (v[e] - mean[hr]); }
        s2 += __shfl_xor_sync(0xffffffffu, s2, 1); s2 += __shfl_xor_sync(0xffffffffu, s2, 2);
        rs[hr] = rsqrtf(s2 * (1.f / 128) + eps);
      }
      float acc[NT][4];
#pragma unroll
      for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
      const float nmr[2] = {-mean[0] * rs[0], -mean[1] * rs[1]};  // LN = pair rstd - mean rstd: one FMA per element
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        float v0[8], v1[8];
        unpack8(u[0][kk / 2], v0); unpack8(u[1][kk / 2], v1);
        const int e = 4 * (kk % 2);
        auto xh = [&](const float* v, int hr, int k) { return fmaf(v[e + k], rs[hr], nmr[hr]); };
        const uint32_t a0 = pack2(xh(v0, 0, 0), xh(v0, 0, 1)), a1 = pack2(xh(v1, 1, 0), xh(v1, 1, 1));
        const uint32_t a2 = pack2(xh(v0, 0, 2), xh(v0, 0, 3)), a3 = pack2(xh(v1, 1, 2), xh(v1, 1, 3));
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          const uint2 b = bw[kk][t][lane];
          mma16816(acc[t], a0, a1, a2, a3, b.x, b.y);
        }
      }
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          stg[w][8 * t + 2 * q + e][tt * 16 + g] = __float2bfloat16_rn(acc[t][e]);
          stg[w][8 * t + 2 * q + e][tt * 16 + g + 8] = __float2bfloat16_rn(acc[t][2 + e]);
        }
      if (q == 0) { PST[rb + g] = make_float2(mean[0], rs[0]); PST[rb + g + 8] = make_float2(mean[1], rs[1]); }
    }
    __syncwarp();
#pragma unroll
    for (int i = 0; i < NH / 4; ++i) {
      const int h = 4 * i + lane / 8, c = lane % 8;
      *reinterpret_cast<uint4*>(BIAS + (long)h * R + r0 + 8 * c) = *reinterpret_cast<const uint4*>(&stg[w][h][8 * c]);
    }
    __syncwarp();
  }
}

// backward: 128 rows per block item (a 16-row tile per warp). d LN(pair) = dbias^T Wf on mma (K = the heads: a k16 step, plus
// a k8 step for heads 16-23), the LayerNorm backward in registers, d pair out; LN(pair) (bf16, as the forward's GEMM operand)
// goes back into the item's own pair buffer (each lane rewrites the chunks it loaded), where warp w then adds
// dWf[:, 16 w .. 16 w + 15]^T += LN(pair)^T dbias^T over the item's 128 rows on mma (M = the warp's 16 columns, N = the heads in
// n-tiles of 8, K = rows: no padding). dWf leaves as one [NH, 128] partial per block (every block writes its row; finalize sums).
// The next item's pair rows and dbias stream into the other half of a double buffer (cp.async) while this one is worked on;
// the pair rows sit in 16-byte chunks XOR-swizzled by row so the per-lane chunk reads and the ldmatrix rows are conflict-free.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
template <int NH> struct PB {
  static constexpr int NHP = NH <= 16 ? 16 : 24;       // dbias rows in shared memory: 12 heads padded to 16 with zeros
  static constexpr int ZS = 128 * 256, DS = NHP * 136 * 2, BW = 8 * 32 * 16 + (NH > 16 ? 8 * 32 * 8 : 0);
  static constexpr int SMEM = 2 * ZS + 2 * DS + BW;
};

template <int NH>
__device__ __forceinline__ void pair_bias_bwd_body(const bf* __restrict__ DB, const bf* __restrict__ Z,
    const float2* __restrict__ PST, const bf* __restrict__ WF, bf* __restrict__ DZ, float* __restrict__ DWF, long R) {
  using C = PB<NH>;
  constexpr int NHP = C::NHP, NT = NHP / 8;           // head n-tiles of the dWf product
  static_assert(NH == 12 || NH == 16 || NH == 24, "12, 16 or 24 heads");
  extern __shared__ __align__(16) uint8_t smem[];
  // Wf fragments per lane: heads 0-15 for n-tiles 2 p, 2 p + 1 ({b0, b1, b0', b1'}); heads 16-23 (k8) for n-tiles 2 p, 2 p + 1
  uint4 (*bw)[32] = reinterpret_cast<uint4 (*)[32]>(smem + 2 * C::ZS + 2 * C::DS);
  uint2 (*bw8)[32] = reinterpret_cast<uint2 (*)[32]>(smem + 2 * C::ZS + 2 * C::DS + 8 * 32 * 16);
  const uint32_t s_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32, g = lane / 4, q = lane % 4;
  auto zs_off = [](int b, int row, int chunk) { return b * C::ZS + row * 256 + ((chunk ^ (row & 7)) * 16); };
  auto prefetch = [&](long b0, int b) {               // the item's pair rows (8 chunks per thread) and dbias
    for (int i = threadIdx.x; i < 128 * 16; i += blockDim.x) {
      const int row = i / 16, c = i % 16;
      cp_async16(s_base + zs_off(b, row, c), Z + (b0 + row) * 128 + 8 * c);
    }
    for (int i = threadIdx.x; i < NH * 16; i += blockDim.x) {
      const int h = i / 16, c = i % 16;
      cp_async16(s_base + 2 * C::ZS + b * C::DS + (h * 136 + 8 * c) * 2, DB + (long)h * R + b0 + 8 * c);
    }
  };
  for (int i = threadIdx.x; i < 8 * 32; i += blockDim.x) {
    const int l = i % 32, p = i / 32;
    auto f = [&](int t, int k) {                       // heads >= NH (12 padded to 16) weigh 0
      const int c = bcol(t, l / 4);
      return k < NH ? pack2(WF[k * 128 + c], WF[(k + 1) * 128 + c]) : 0u;
    };
    const int k = 2 * (l % 4);
    bw[p][l] = make_uint4(f(2 * p, k), f(2 * p, k + 8), f(2 * p + 1, k), f(2 * p + 1, k + 8));
    if (NH > 16) bw8[p][l] = make_uint2(f(2 * p, 16 + k), f(2 * p + 1, 16 + k));
  }
  for (int i = threadIdx.x; i < 2 * (NHP - NH) * 136; i += blockDim.x) {   // the padding heads of both dbias buffers: zero
    const int b = i / ((NHP - NH) * 136), k = i % ((NHP - NH) * 136);
    reinterpret_cast<bf*>(smem + 2 * C::ZS + b * C::DS)[NH * 136 + k] = __float2bfloat16(0.f);
  }
  float dacc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t) dacc[t][0] = dacc[t][1] = dacc[t][2] = dacc[t][3] = 0.f;
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
    const uint32_t ds_s = s_base + 2 * C::ZS + b * C::DS;
    uint4 u[2][4];
#pragma unroll
    for (int hr = 0; hr < 2; ++hr)
#pragma unroll
      for (int m = 0; m < 4; ++m) u[hr][m] = *reinterpret_cast<const uint4*>(smem + zs_off(b, wr + g + 8 * hr, q + 4 * m));
    // A = dbias^T [16 rows x heads]: the transposes of ds[heads 8 (i / 2) ..][rows wr + 8 (i % 2) ..]; heads 16-23 as a k8 step
    uint32_t a0, a1, a2, a3, e0 = 0, e1 = 0;
    ldsm4t(ds_s + ((lane % 8 + 8 * (lane / 16)) * 136 + wr + 8 * ((lane / 8) % 2)) * 2, a0, a1, a2, a3);
    if (NH > 16) ldsm2t(ds_s + ((16 + lane % 8) * 136 + wr + 8 * ((lane / 8) % 2)) * 2, e0, e1);
    auto dln = [&](float (&d)[4], int t) {             // d LN(pair) for n-tile t
      d[0] = d[1] = d[2] = d[3] = 0.f;
      const uint4 bb = bw[t / 2][lane];
      mma16816(d, a0, a1, a2, a3, t % 2 ? bb.z : bb.x, t % 2 ? bb.w : bb.y);
      if (NH > 16) {
        const uint2 b8 = bw8[t / 2][lane];
        mma1688(d, e0, e1, t % 2 ? b8.y : b8.x);
      }
    };
    // pass 1: the row sums of d LN and of d LN * pair (row g: d[0], d[1]; row g + 8: d[2], d[3]; columns chunk q + 4 (t / 4),
    // elements 2 (t % 4) + {0, 1}); sum d LN LN = rstd (sum d LN pair - mean sum d LN)
    float sd[2] = {0.f, 0.f}, sdv[2] = {0.f, 0.f};
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      float v0[8], v1[8];
      unpack8(u[0][m], v0); unpack8(u[1][m], v1);
#pragma unroll
      for (int t4 = 0; t4 < 4; ++t4) {
        float d[4];
        dln(d, 4 * m + t4);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          sd[0] += d[e]; sdv[0] = fmaf(d[e], v0[2 * t4 + e], sdv[0]);
          sd[1] += d[2 + e]; sdv[1] = fmaf(d[2 + e], v1[2 * t4 + e], sdv[1]);
        }
      }
    }
    // d pair = rstd (d LN - m1 - LN m2) = rstd d LN - c1 (pair - mean) - rstd m1, with m1 = mean d LN, m2 = mean d LN LN
    float c1[2], c0[2];
    const float2 st[2] = {st0, st1};
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      sd[k] += __shfl_xor_sync(0xffffffffu, sd[k], 1); sd[k] += __shfl_xor_sync(0xffffffffu, sd[k], 2);
      sdv[k] += __shfl_xor_sync(0xffffffffu, sdv[k], 1); sdv[k] += __shfl_xor_sync(0xffffffffu, sdv[k], 2);
      const float m1 = sd[k] * (1.f / 128), m2 = st[k].y * (sdv[k] - st[k].x * sd[k]) * (1.f / 128);
      c1[k] = st[k].y * st[k].y * m2;
      c0[k] = -st[k].y * m1;
    }
    // pass 2: d pair, one 16-byte chunk per row and m; LN back into the chunks this lane loaded
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      float v0[8], v1[8], o0[8], o1[8];
      unpack8(u[0][m], v0); unpack8(u[1][m], v1);
#pragma unroll
      for (int e = 0; e < 8; ++e) { v0[e] -= st0.x; v1[e] -= st1.x; }
#pragma unroll
      for (int t4 = 0; t4 < 4; ++t4) {
        float d[4];
        dln(d, 4 * m + t4);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int k = 2 * t4 + e;
          o0[k] = fmaf(st0.y, d[e], fmaf(-c1[0], v0[k], c0[0]));
          o1[k] = fmaf(st1.y, d[2 + e], fmaf(-c1[1], v1[k], c0[1]));
        }
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) { v0[e] *= st0.y; v1[e] *= st1.y; }
      const int col = 8 * (q + 4 * m);
      *reinterpret_cast<uint4*>(DZ + (rb + g) * 128 + col) = pack8(o0);
      *reinterpret_cast<uint4*>(DZ + (rb + g + 8) * 128 + col) = pack8(o1);
      *reinterpret_cast<uint4*>(smem + zs_off(b, wr + g, q + 4 * m)) = pack8(v0);
      *reinterpret_cast<uint4*>(smem + zs_off(b, wr + g + 8, q + 4 * m)) = pack8(v1);
    }
    __syncthreads();                                   // LN of the item's rows
    // dWf^T[16 w + m, head 8 t + n] over the item's rows: A = LN^T [16 columns x 16 rows], B = dbias^T [16 rows x 8 heads]
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int k0 = 16 * kk, i = lane / 8, r8 = lane % 8;
      uint32_t c0, c1, c2, c3;
      // A: transposes of LN[rows k0 + r8 + 8 (i / 2)][columns 16 w + 8 (i % 2) ..] (chunk 2 w + i % 2 of the swizzled buffer)
      ldsm4t(s_base + zs_off(b, k0 + r8 + 8 * (i / 2), 2 * w + i % 2), c0, c1, c2, c3);
      // B: ds[heads 8 t + r8][rows k0 + 8 (i % 2) ..] for n-tiles t, t + 1 (i / 2)
#pragma unroll
      for (int t = 0; t + 1 < NT; t += 2) {
        uint32_t b0, b1, b2, b3;
        ldsm4(ds_s + ((8 * (t + i / 2) + r8) * 136 + k0 + 8 * (i % 2)) * 2, b0, b1, b2, b3);
        mma16816(dacc[t], c0, c1, c2, c3, b0, b1);
        mma16816(dacc[t + 1], c0, c1, c2, c3, b2, b3);
      }
      if (NT % 2) {
        uint32_t b0, b1;
        ldsm2(ds_s + ((8 * (NT - 1) + r8) * 136 + k0 + 8 * (i % 2)) * 2, b0, b1);
        mma16816(dacc[NT - 1], c0, c1, c2, c3, b0, b1);
      }
    }
    __syncthreads();                                   // before this buffer is refilled
  }
  // dacc[t]: d0, d1 = column 16 w + g, heads 8 t + 2 q + {0, 1}; d2, d3 = column 16 w + g + 8
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int h = 8 * t + 2 * q + e;
      if (h < NH) {
        DWF[((long)blockIdx.x * NH + h) * 128 + 16 * w + g] = dacc[t][e];
        DWF[((long)blockIdx.x * NH + h) * 128 + 16 * w + g + 8] = dacc[t][2 + e];
      }
    }
}
template <int NH>
__global__ void __launch_bounds__(WPB * 32, 2) pair_bias_bwd_k(const bf* __restrict__ DB, const bf* __restrict__ Z,
    const float2* __restrict__ PST, const bf* __restrict__ WF, bf* __restrict__ DZ, float* __restrict__ DWF, long R) {
  pair_bias_bwd_body<NH>(DB, Z, PST, WF, DZ, DWF, R);
}

// -------------------------------------------------------------------------------------------- small weight gradients
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ bf from_f<bf>(float v) { return __float2bfloat16_rn(v); }

// The four AdaLN projections see LN(cond) w (w the cond-LN weight of their block), folded into Wn = Wraw w in the forward:
// dWraw = dWn w (per column), written in the parameters' dtype, and dw = sum over the block's 2 x 768 rows of dWn o Wraw,
// left as one partial row per block of 16 rows (192 blocks: 0-95 the attention's, 96-191 the transition's). 96 column quads x
// 4 row lanes, every load of a thread issued first.
constexpr int UNF_ROWS = 16, UNF_BLOCKS = 4 * 768 / UNF_ROWS;
template <typename OT>
__global__ void __launch_bounds__(384) unfold_k(const float* __restrict__ DWN, const float* __restrict__ WRAW, const float* __restrict__ W1,
    const float* __restrict__ W2, OT* __restrict__ DWU, float* __restrict__ PW) {
  __shared__ float4 red[4][96];
  const int cq = threadIdx.x % 96, rl = threadIdx.x / 96, col = cq * 4;
  const long i0 = (long)blockIdx.x * UNF_ROWS;
  const float4 lw = V4<float>::load((i0 < 2 * D ? W1 : W2) + col);
  float4 d[UNF_ROWS / 4], w[UNF_ROWS / 4];
#pragma unroll
  for (int k = 0; k < UNF_ROWS / 4; ++k) {
    const long i = i0 + rl + 4 * k;
    d[k] = V4<float>::load(DWN + i * DC + col); w[k] = V4<float>::load(WRAW + i * DC + col);
  }
  float4 acc = zero4();
#pragma unroll
  for (int k = 0; k < UNF_ROWS / 4; ++k) {
    V4<OT>::store(DWU + (i0 + rl + 4 * k) * DC + col, mul4(d[k], lw));
    acc = add4(acc, mul4(d[k], w[k]));
  }
  red[rl][cq] = acc;
  __syncthreads();
  if (rl == 0) V4<float>::store(PW + (long)blockIdx.x * DC + col, add4(add4(red[0][cq], red[1][cq]), add4(red[2][cq], red[3][cq])));
}

// The step's last kernel: every per-block partial summed and written in the parameters' dtype into the small-gradient buffer
// (bias sums, cond-LN weights, ln_pair / to_bias), 248 blocks:
//   0-95     the four bias sums (768 columns, 32 per block, over the rows the row kernels wrote)
//   96-119   the two cond-LN weights (384 columns over unfold's 96 rows of each block)
//   120-247  dWf [NH, 128] over pair_bias_bwd's per-block rows (one column of every head per block), then
//                   ln_pair = sum_h dWf o Wb, to_bias = dWf diag(wp)
// 16 row groups per block, eight independent loads in flight per thread: a few hundred partial rows in a few L2 round trips.
struct Counts { int n[4]; };
// OUT: bias sums [4, 768] | to_bias [NH, 128] in the projections' dtype; NOUT: cond-LN weights [2, 384] | ln_pair [128] in the
// LayerNorm weights' dtype (the engine's LayerNorms may keep fp32 weights in a bf16 block)
constexpr int FIN_PB = 0, FIN_TOB = 4 * D, FIN_DW = 0, FIN_LNP = 2 * DC, FIN_NOUT = FIN_LNP + 128;
// rows g, g + RG, ... of a column, U (8 or 16) loads in flight
template <int RG, int U = 8>
__device__ __forceinline__ float colsum_rg(const float* base, long stride, int n, int g) {
  static_assert(U == 8 || U == 16, "8 or 16 loads in flight");
  float t[U];
#pragma unroll
  for (int k = 0; k < U; ++k) t[k] = 0.f;
  int r = g;
  for (; r + (U - 1) * RG < n; r += U * RG) {
#pragma unroll
    for (int k = 0; k < U; ++k) t[k] += __ldcg(base + (long)(r + k * RG) * stride);
  }
  for (; r < n; r += RG) t[0] += __ldcg(base + (long)r * stride);
#pragma unroll
  for (int k = 8; k < U; ++k) t[k - 8] += t[k];
  return ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
}
// one finalize job (0 .. FIN_JOBS - 1) by a block of RG warps: 4 x 24 bias-sum column groups, 2 x 12 cond-LN column groups, 128 dWf
// columns; RG row groups (16 in finalize_k's 512 threads, 8 in pair_bias_bwd_fin_k's 256)
constexpr int FIN_JOBS = 96 + 24 + 128;
template <int NH, typename OT, typename NT, int RG, int U = 8>
__device__ __forceinline__ void finalize_job(int x, const float* __restrict__ PART, long prow, const Counts& N,
    const float* __restrict__ PW, const float* __restrict__ PWF, int nwf, const float* __restrict__ WB, const float* __restrict__ WP,
    OT* __restrict__ OUT, NT* __restrict__ NOUT) {
  __shared__ float red[RG][33];
  const int y = x < 96 ? x / 24 : x < 120 ? 4 + (x - 96) / 12 : 6;
  const int bx = x < 96 ? x % 24 : x < 120 ? (x - 96) % 12 : x - 120;
  const int lane = threadIdx.x % 32, g = threadIdx.x / 32;
  if (y < 4) {
    const int c = bx * 32 + lane;
    red[g][lane] = colsum_rg<RG, U>(PART + (long)y * prow * D + c, D, N.n[y], g);
  } else if (y < 6) {
    const int c = bx * 32 + lane;
    red[g][lane] = colsum_rg<RG, U>(PW + (long)(y - 4) * (UNF_BLOCKS / 2) * DC + c, DC, UNF_BLOCKS / 2, g);
  } else {
    // dWf: column bx of every head; thread (row group, head) sums rows rg, rg + RG, ...
    const int c = bx, h = lane, rg = g;
    red[rg][h] = h < NH ? colsum_rg<RG, U>(PWF + h * 128 + c, NH * 128, nwf, rg) : 0.f;
  }
  __syncthreads();
  if (g == 0) {
    float t = 0.f;
#pragma unroll
    for (int k = 0; k < RG; ++k) t += red[k][lane];
    if (y < 4) {
      OUT[FIN_PB + y * D + bx * 32 + lane] = from_f<OT>(t);
    } else if (y < 6) {
      NOUT[FIN_DW + (y - 4) * DC + bx * 32 + lane] = from_f<NT>(t);
    } else {
      const int c = bx, h = lane;                                       // t = dWf[h, c]
      if (h < NH) OUT[FIN_TOB + h * 128 + c] = from_f<OT>(t * WP[c]);
      const float u = warp_sum(h < NH ? t * WB[h * 128 + c] : 0.f);    // ln_pair[c] = sum_h dWf[h, c] Wb[h, c]
      if (h == 0) NOUT[FIN_LNP + c] = from_f<NT>(u);
    }
  }
  __syncthreads();                                                      // red free for the block's next job
}
template <int NH, typename OT, typename NT>
__global__ void __launch_bounds__(512) finalize_k(const float* __restrict__ PART, long prow, const Counts N, const float* __restrict__ PW,
    const float* __restrict__ PWF, int nwf, const float* __restrict__ WB, const float* __restrict__ WP, OT* __restrict__ OUT,
    NT* __restrict__ NOUT) {
  finalize_job<NH, OT, NT, 16>(blockIdx.x, PART, prow, N, PW, PWF, nwf, WB, WP, OUT, NOUT);
}

// pair_bias_bwd + finalize in ONE launch (MINIWORLD_BIAS_ONLY_DIT_BWD_PBFIN): the persistent pair_bias_bwd blocks (grid <= the
// resident blocks, so all are on the GPU at once) write their dWf partials, meet at a grid barrier (BAR[0]: thread 0 of every block
// __threadfence + atomicAdd, then spins until gridDim.x arrived; __threadfence; __syncthreads), and then take finalize's 248 jobs in
// turn (job = blockIdx.x + k gridDim.x; the partials read through L2, __ldcg, 16 rows in flight per thread: half finalize_k's
// threads per job). The last block to finish its jobs (BAR[1]) zeroes
// BAR for the next launch. The sums run over the same partial rows in a fixed order: bit-identical reruns (finalize_k's 16-way row
// split becomes 8-way here, so the result differs from finalize_k in the last bits).
template <int NH, typename OT, typename NT>
__global__ void __launch_bounds__(WPB * 32, 2) pair_bias_bwd_fin_k(const bf* __restrict__ DB, const bf* __restrict__ Z,
    const float2* __restrict__ PST, const bf* __restrict__ WF, bf* __restrict__ DZ, float* __restrict__ DWF, long R,
    const float* __restrict__ PART, long prow, const Counts N, const float* __restrict__ PW, const float* __restrict__ WB,
    const float* __restrict__ WP, OT* __restrict__ OUT, NT* __restrict__ NOUT, unsigned* __restrict__ BAR) {
  pair_bias_bwd_body<NH>(DB, Z, PST, WF, DZ, DWF, R);
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(BAR, 1u);
    while (*reinterpret_cast<volatile unsigned*>(BAR) < gridDim.x) __nanosleep(32);
    __threadfence();
  }
  __syncthreads();
  for (int x = blockIdx.x; x < FIN_JOBS; x += gridDim.x)
    finalize_job<NH, OT, NT, WPB, 16>(x, PART, prow, N, PW, DWF, (int)gridDim.x, WB, WP, OUT, NOUT);   // 16 loads in flight: 8 warps
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(BAR + 1, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (last && threadIdx.x == 0) { BAR[0] = 0u; BAR[1] = 0u; __threadfence(); }
}

// ------------------------------------------------------------------------------------------------------------ weight pack
// The bf16 training step's weight pack (integrations/bias_only_dit_train.py _pack, MINIWORLD_BIAS_ONLY_DIT_TRAIN_PACK1=1) in ONE
// launch: up to 48 segments, each dst = src (o scale[i % period], broadcast over the rows: the folded LayerNorm weights) with bf16
// or fp32 on either side and on the scale (fp32 arithmetic, RN to the destination: the torch casts / products bit for bit), or
// dst = (src o scale[col])^T of a [rows, cols] matrix, the scale optional (32 x 32 tiles through shared memory: the K-major weights
// of the fused backward GEMMs; the scaled transpose is the folded one's transpose bit for bit).
// A captured training step repacks every replay (the weights change between steps); as torch ops that was 19 kernels of ~3 us.
// One flat grid, each segment its own run of blocks (b0[k] .. b0[k + 1]): a block finds its segment by a scan of the (uniform) table
// and does PACK_F4 float4 groups or PACK_TILES transpose tiles. ~35 MB move per pack (16 x 48: reads 16.5, writes 18.9 incl. the fp32
// Wraw 4.7). Measured: one kernel, 10.3 us hot / 20.5 cold either way the grid was laid out (27 x 296 blocks, then a flat grid): the
// limit was one load in flight per thread, so every thread now issues all its loads first.
constexpr int PACK_F4 = 2048, PACK_TILES = 4;
struct Pack16Seg { const void* src; void* dst; const void* scale; int n; int period; int rows; int cols; int sdt, ddt, cdt, tr, b0; };
constexpr int PACK_SEGS = 48;
struct Pack16Segs { Pack16Seg s[PACK_SEGS]; int n; };
__device__ __forceinline__ float4 ldany4(const void* p, int f32, long i) {
  return f32 ? V4<float>::load(static_cast<const float*>(p) + i) : V4<bf>::load(static_cast<const bf*>(p) + i);
}
__device__ __forceinline__ void stany4(void* p, int f32, long i, float4 v) {
  if (f32) V4<float>::store(static_cast<float*>(p) + i, v); else V4<bf>::store(static_cast<bf*>(p) + i, v);
}
__device__ __forceinline__ float ldany(const void* p, int f32, long i) {
  return f32 ? static_cast<const float*>(p)[i] : __bfloat162float(static_cast<const bf*>(p)[i]);
}
__device__ __forceinline__ void stany(void* p, int f32, long i, float v) {
  if (f32) static_cast<float*>(p)[i] = v; else static_cast<bf*>(p)[i] = __float2bfloat16_rn(v);
}
__global__ void __launch_bounds__(256) pack16_k(const __grid_constant__ Pack16Segs S) {
  int k = 0;
  while (k + 1 < S.n && (int)blockIdx.x >= S.s[k + 1].b0) ++k;
  const Pack16Seg& g = S.s[k];
  const int lb = (int)blockIdx.x - g.b0;
  if (g.tr) {                                          // dst [cols, rows] = src [rows, cols]^T
    __shared__ float t[32][33];
    const int tx = threadIdx.x % 32, ty = threadIdx.x / 32, tc = g.cols / 32, nt = (g.rows / 32) * tc;
    for (int kk = lb * PACK_TILES; kk < min(nt, (lb + 1) * PACK_TILES); ++kk) {
      const int r0 = (kk / tc) * 32, c0 = (kk % tc) * 32;
      if (g.scale) {
        const float sc = ldany(g.scale, g.cdt, c0 + tx);
        for (int i = ty; i < 32; i += 8) t[i][tx] = ldany(g.src, g.sdt, (long)(r0 + i) * g.cols + c0 + tx) * sc;
      } else {
        for (int i = ty; i < 32; i += 8) t[i][tx] = ldany(g.src, g.sdt, (long)(r0 + i) * g.cols + c0 + tx);
      }
      __syncthreads();
      for (int i = ty; i < 32; i += 8) stany(g.dst, g.ddt, (long)(c0 + i) * g.rows + r0 + tx, t[tx][i]);
      __syncthreads();
    }
    return;
  }
  // every load of the thread's PACK_F4 / 256 groups issued before any store (round 2 looped load -> store: one load in flight per
  // thread, ~3 TB/s hot)
  constexpr int NV = PACK_F4 / 256;
  const int i0 = lb * PACK_F4 + threadIdx.x, n4 = g.n / 4;
  float4 v[NV];
#pragma unroll
  for (int j = 0; j < NV; ++j) {
    const int i = i0 + j * 256;
    v[j] = i < n4 ? ldany4(g.src, g.sdt, 4L * i) : zero4();
  }
#pragma unroll
  for (int j = 0; j < NV; ++j) {
    const int i = i0 + j * 256;
    if (i < n4) stany4(g.dst, g.ddt, 4L * i, g.scale ? mul4(v[j], ldany4(g.scale, g.cdt, (4L * i) % g.period)) : v[j]);
  }
}

// ------------------------------------------------------------------------------------------------------------ SwiGLU
// elementwise over [M, N] with [a | b] rows of 2N, eight bf16 (16 bytes) per thread per tensor

// h = silu(a) b
__global__ void __launch_bounds__(256) swiglu_k(const bf* __restrict__ AB, bf* __restrict__ Hh, long M, int N) {
  const int nc = N / 8;
  const long i = (long)blockIdx.x * 256 + threadIdx.x;
  if (i >= M * nc) return;
  const long r = i / nc;
  const int c = (int)(i - r * nc) * 8;
  float a[8], b[8], h[8];
  unpack8(ld8(AB + r * 2 * N + c), a); unpack8(ld8(AB + r * 2 * N + N + c), b);
#pragma unroll
  for (int k = 0; k < 8; ++k) h[k] = a[k] * sg(a[k]) * b[k];
  st8(Hh + r * N + c, pack8(h));
}

// da = dh b s (1 + a (1 - s)), db = dh a s (s = sigmoid(a)); h is the forward's, not rebuilt
__global__ void __launch_bounds__(256) swiglu_bwd_k(const bf* __restrict__ DH, const bf* __restrict__ AB, bf* __restrict__ DAB, long M, int N) {
  const int nc = N / 8;
  const long i = (long)blockIdx.x * 256 + threadIdx.x;
  if (i >= M * nc) return;
  const long r = i / nc;
  const int c = (int)(i - r * nc) * 8;
  float dh[8], a[8], b[8], da[8], db[8];
  unpack8(ld8(DH + r * N + c), dh); unpack8(ld8(AB + r * 2 * N + c), a); unpack8(ld8(AB + r * 2 * N + N + c), b);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float sk = sg(a[k]);
    da[k] = dh[k] * b[k] * sk * (1.f + a[k] * (1.f - sk));
    db[k] = dh[k] * a[k] * sk;
  }
  st8(DAB + r * 2 * N + c, pack8(da)); st8(DAB + r * 2 * N + N + c, pack8(db));
}

// -------------------------------------------------------------------------------------------------------------- host
#define TDT_DISPATCH(T, NAME, ...)                                                                   \
  [&] {                                                                                              \
    if ((T) == at::kFloat) { using NAME = float; return __VA_ARGS__(); }                             \
    TORCH_CHECK((T) == at::kBFloat16, "fp32 or bf16");                                                 \
    using NAME = bf; return __VA_ARGS__();                                                           \
  }()

template <typename C> C* P(const at::Tensor& t) { return reinterpret_cast<C*>(t.data_ptr()); }
template <typename C> const C* CP(const at::Tensor& t) { return reinterpret_cast<const C*>(t.data_ptr()); }
cudaStream_t S() { return at::cuda::getCurrentCUDAStream(); }
void bf16c(const at::Tensor& t, const char* n) { TORCH_CHECK(t.scalar_type() == at::kBFloat16 && t.stride(-1) == 1, n, ": bf16 rows"); }
void f32c(const at::Tensor& t, const char* n) { TORCH_CHECK(t.scalar_type() == at::kFloat && t.is_contiguous(), n, ": contiguous fp32"); }

// persistent warps: as many blocks of WPB warps as are resident on every SM at once (never more warps than rows); a larger grid
// would run the persistent blocks in waves
int nsm() { static int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount; return n; }
template <typename K>
unsigned grid(K kernel, int64_t M, int rows_per_block = WPB, size_t smem = 0) {
  int per = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, WPB * 32, smem);
  if (const char* e = std::getenv("MINIWORLD_BO_ROWS_BLOCKS_PER_SM")) per = std::min(per, std::atoi(e));   // experiments
  return (unsigned)std::max<int64_t>(1, std::min<int64_t>((M + rows_per_block - 1) / rows_per_block, (int64_t)std::max(per, 1) * nsm()));
}

//: rows of every partial buffer: an upper bound of the blocks of any launch here (the kernels return how many they wrote)
int64_t partial_rows(int64_t) { return 8 * (int64_t)nsm(); }

void cond_ln(at::Tensor c, at::Tensor chat, at::Tensor cst, double eps) {
  const int64_t M = c.size(0);
  bf16c(c, "c"); TORCH_CHECK(c.is_contiguous() && c.size(1) == DC, "cond_ln: [M, 384]");
  cond_ln_k<<<grid(cond_ln_k, M), WPB * 32, 0, S()>>>(CP<bf>(c), P<bf>(chat), P<float2>(cst), M, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void adaln_a(at::Tensor x, at::Tensor G, at::Tensor bs, at::Tensor xa, at::Tensor xst, double eps) {
  const int64_t M = x.size(0);
  bf16c(G, "G"); f32c(bs, "bs");
  TDT_DISPATCH(x.scalar_type(), XT, [&] {
    adaln_a_k<XT><<<grid(adaln_a_k<XT>, M, RPB2), WPB * 32, 0, S()>>>(CP<XT>(x), CP<bf>(G), G.stride(0), CP<float>(bs), P<bf>(xa), P<float2>(xst), M, (float)eps);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void res_adaln_b(at::Tensor x, at::Tensor y, at::Tensor Gg, at::Tensor bg1, at::Tensor G, at::Tensor bs2, at::Tensor xt,
                 at::Tensor x1st, double eps) {
  const int64_t M = x.size(0);
  bf16c(y, "y"); bf16c(Gg, "Gg"); bf16c(G, "G");
  TDT_DISPATCH(x.scalar_type(), XT, [&] {
    res_adaln_b_k<XT><<<grid(res_adaln_b_k<XT>, M, RPB2), WPB * 32, 0, S()>>>(CP<XT>(x), CP<bf>(y), CP<bf>(Gg), Gg.stride(0), CP<float>(bg1), CP<bf>(G), G.stride(0),
        CP<float>(bs2), P<bf>(xt), P<float2>(x1st), M, (float)eps);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void res_c(at::Tensor x, at::Tensor y, at::Tensor z, at::Tensor Gg, at::Tensor bg1, at::Tensor bg2, at::Tensor out) {
  const int64_t M = x.size(0);
  bf16c(y, "y"); bf16c(z, "z"); bf16c(Gg, "Gg");
  TDT_DISPATCH(x.scalar_type(), XT, [&] {
    TDT_DISPATCH(out.scalar_type(), OT, [&] {
      res_c_k<XT, OT><<<grid(res_c_k<XT, OT>, M, RPB2), WPB * 32, 0, S()>>>(CP<XT>(x), CP<bf>(y), CP<bf>(z), CP<bf>(Gg), Gg.stride(0),
          CP<float>(bg1), CP<float>(bg2), P<OT>(out), M);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t res_c_bwd(at::Tensor dout, at::Tensor z, at::Tensor Gg, at::Tensor bg2, at::Tensor dz, at::Tensor dGg, at::Tensor pg2) {
  const int64_t M = dout.size(0);
  TORCH_CHECK(pg2.size(0) >= partial_rows(M) && pg2.stride(0) == D && pg2.is_contiguous(), "res_c_bwd: partials [partial_rows, 768]");
  unsigned g = 0;
  TDT_DISPATCH(dout.scalar_type(), OT, [&] {
    g = grid(res_c_bwd_k<OT>, M, RPB2);
    res_c_bwd_k<OT><<<g, WPB * 32, 0, S()>>>(CP<OT>(dout), CP<bf>(z), CP<bf>(Gg), Gg.stride(0), CP<float>(bg2), P<bf>(dz), P<bf>(dGg),
        dGg.stride(0), P<float>(pg2), M);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return g;
}

int64_t res_adaln_b_bwd(at::Tensor dout, at::Tensor dxt, at::Tensor x, at::Tensor x1st, at::Tensor G, at::Tensor bs2, at::Tensor Gg,
                     at::Tensor bg1, at::Tensor y, at::Tensor dx1, at::Tensor dy, at::Tensor dG, at::Tensor dGg, at::Tensor ps2,
                     at::Tensor pg1) {
  const int64_t M = dout.size(0);
  TORCH_CHECK(ps2.is_contiguous() && pg1.is_contiguous() && ps2.size(0) >= partial_rows(M) && pg1.size(0) >= partial_rows(M), "partials");
  bf16c(dxt, "dxt"); bf16c(dG, "dG");
  // dxt may be dG's d-shift2 columns themselves (the GEMM wrote it there): then the copy is not written again
  const bool cpy = !(dxt.data_ptr() == static_cast<void*>(P<bf>(dG) + 3 * D) && dxt.stride(0) == dG.stride(0));
  unsigned g = 0;
  TDT_DISPATCH(x.scalar_type(), XT, [&] {
    TDT_DISPATCH(dout.scalar_type(), OT, [&] {
      g = grid(res_adaln_b_bwd_k<XT, OT>, M, RPB2);
      res_adaln_b_bwd_k<XT, OT><<<g, WPB * 32, 0, S()>>>(CP<OT>(dout), CP<bf>(dxt), dxt.stride(0), cpy, CP<XT>(x), CP<float2>(x1st), CP<bf>(G), G.stride(0),
          CP<float>(bs2), CP<bf>(Gg), Gg.stride(0), CP<float>(bg1), CP<bf>(y), P<float>(dx1), P<bf>(dy), P<bf>(dG), dG.stride(0),
          P<bf>(dGg), dGg.stride(0), P<float>(ps2), P<float>(pg1), M);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return g;
}

int64_t adaln_a_bwd(at::Tensor dxa, at::Tensor x, at::Tensor xst, at::Tensor G, at::Tensor bs1, at::Tensor dx1, at::Tensor dx,
                 at::Tensor dG, at::Tensor ps1) {
  const int64_t M = dxa.size(0);
  TORCH_CHECK(ps1.is_contiguous() && ps1.size(0) >= partial_rows(M), "partials");
  bf16c(dxa, "dxa"); bf16c(dG, "dG");
  const bool cpy = !(dxa.data_ptr() == static_cast<void*>(P<bf>(dG) + D) && dxa.stride(0) == dG.stride(0));   // dxa in dG already
  unsigned g = 0;
  TDT_DISPATCH(x.scalar_type(), XT, [&] {
    TDT_DISPATCH(dx.scalar_type(), OT, [&] {
      g = grid(adaln_a_bwd_k<XT, OT>, M, RPB2);
      adaln_a_bwd_k<XT, OT><<<g, WPB * 32, 0, S()>>>(CP<bf>(dxa), dxa.stride(0), cpy, CP<XT>(x), CP<float2>(xst), CP<bf>(G), G.stride(0), CP<float>(bs1),
          CP<float>(dx1), P<OT>(dx), P<bf>(dG), dG.stride(0), P<float>(ps1), M);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return g;
}

void cond_bwd(at::Tensor dchat, at::Tensor dcg, at::Tensor c, at::Tensor cst, at::Tensor dc) {
  const int64_t M = dchat.size(0);
  bf16c(c, "c");
  TDT_DISPATCH(dc.scalar_type(), OT, [&] {
    cond_bwd_k<OT><<<grid(cond_bwd_k<OT>, M), WPB * 32, 0, S()>>>(CP<bf>(dchat), CP<bf>(dcg), CP<bf>(c), CP<float2>(cst), P<OT>(dc), M);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}



void finalize(at::Tensor part, std::vector<int64_t> n, at::Tensor pw, at::Tensor pwf, int64_t nwf, at::Tensor wb, at::Tensor wp,
              at::Tensor out, at::Tensor nout) {
  f32c(part, "part"); f32c(pw, "pw"); f32c(pwf, "pwf"); f32c(wb, "Wb"); f32c(wp, "wp");
  TORCH_CHECK(part.dim() == 3 && part.size(0) == 4 && part.size(2) == D && n.size() == 4, "finalize: part [4, rows, 768], 4 counts");
  const int64_t nh = wb.size(0);
  TORCH_CHECK((nh == 12 || nh == 16 || nh == 24) && wb.numel() == nh * 128 && pw.numel() == UNF_BLOCKS * DC && pwf.numel() >= nwf * nh * 128
              && wp.numel() == 128, "finalize: unfold partials [192, 384], dWf partials [>= nwf, nh, 128], Wb [nh, 128], wp [128]");
  TORCH_CHECK(out.is_contiguous() && out.numel() == FIN_TOB + nh * 128 && nout.is_contiguous() && nout.numel() == FIN_NOUT,
              "finalize: out = bias sums | to_bias, nout = cond-LN weights | ln_pair");
  Counts c{};
  for (int i = 0; i < 4; ++i) { TORCH_CHECK(n[i] <= part.size(1), "finalize: row count"); c.n[i] = (int)n[i]; }
  TDT_DISPATCH(out.scalar_type(), OT, [&] {
    TDT_DISPATCH(nout.scalar_type(), NT, [&] {
      auto k = nh == 12 ? finalize_k<12, OT, NT> : nh == 16 ? finalize_k<16, OT, NT> : finalize_k<24, OT, NT>;
      k<<<96 + 24 + 128, 512, 0, S()>>>(CP<float>(part), part.size(1), c, CP<float>(pw), CP<float>(pwf), (int)nwf, CP<float>(wb),
                                       CP<float>(wp), P<OT>(out), P<NT>(nout));
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// pair_bias_bwd + finalize in one launch (pair_bias_bwd_fin_k); bar: int32 [2], zero (left zero)
void pair_bias_bwd_fin(at::Tensor db, at::Tensor z, at::Tensor pst, at::Tensor wf, at::Tensor dz, at::Tensor dwf, at::Tensor part,
                       std::vector<int64_t> n, at::Tensor pw, at::Tensor wb, at::Tensor wp, at::Tensor out, at::Tensor nout,
                       at::Tensor bar) {
  const int64_t R = z.size(0), nh = wf.size(0);
  bf16c(z, "pair"); bf16c(wf, "Wf"); f32c(dwf, "dWf partials");
  TORCH_CHECK(z.is_contiguous() && z.size(1) == 128 && R % 128 == 0 && wf.is_contiguous() && (nh == 12 || nh == 16 || nh == 24)
              && wf.size(1) == 128, "pair_bias_bwd_fin: pair [R, 128] (R a multiple of 128), Wf [12, 16 or 24, 128] bf16");
  TORCH_CHECK(db.scalar_type() == at::kBFloat16 && db.is_contiguous() && db.numel() == nh * R, "pair_bias_bwd_fin: dbias [nh, R]");
  TORCH_CHECK(dz.is_contiguous() && dz.scalar_type() == at::kBFloat16 && pst.is_contiguous(), "pair_bias_bwd_fin: d pair bf16");
  f32c(part, "part"); f32c(pw, "pw"); f32c(wb, "Wb"); f32c(wp, "wp");
  TORCH_CHECK(part.dim() == 3 && part.size(0) == 4 && part.size(2) == D && n.size() == 4, "pair_bias_bwd_fin: part [4, rows, 768], 4 counts");
  TORCH_CHECK(wb.numel() == nh * 128 && pw.numel() == UNF_BLOCKS * DC && wp.numel() == 128 && out.is_contiguous()
              && out.numel() == FIN_TOB + nh * 128 && nout.is_contiguous() && nout.numel() == FIN_NOUT, "pair_bias_bwd_fin: finalize operands");
  TORCH_CHECK(bar.scalar_type() == at::kInt && bar.is_contiguous() && bar.numel() >= 2, "pair_bias_bwd_fin: bar int32 [2]");
  Counts c{};
  for (int i = 0; i < 4; ++i) { TORCH_CHECK(n[i] <= part.size(1), "pair_bias_bwd_fin: row count"); c.n[i] = (int)n[i]; }
  TDT_DISPATCH(out.scalar_type(), OT, [&] {
    TDT_DISPATCH(nout.scalar_type(), NT, [&] {
      auto k = nh == 12 ? pair_bias_bwd_fin_k<12, OT, NT> : nh == 16 ? pair_bias_bwd_fin_k<16, OT, NT> : pair_bias_bwd_fin_k<24, OT, NT>;
      const int smem = nh == 12 ? PB<12>::SMEM : nh == 16 ? PB<16>::SMEM : PB<24>::SMEM;
      TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "pair_bias_bwd_fin: smem");
      const unsigned g = grid(k, R / 128, 1, smem);              // <= the resident blocks: the grid barrier needs them all on the GPU
      TORCH_CHECK(dwf.numel() >= (int64_t)g * nh * 128, "pair_bias_bwd_fin: dWf partials [partial_rows, nh, 128]");
      k<<<g, WPB * 32, smem, S()>>>(CP<bf>(db), CP<bf>(z), CP<float2>(pst), CP<bf>(wf), P<bf>(dz), P<float>(dwf), R, CP<float>(part),
                                    part.size(1), c, CP<float>(pw), CP<float>(wb), CP<float>(wp), P<OT>(out), P<NT>(nout),
                                    reinterpret_cast<unsigned*>(bar.data_ptr()));
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void unfold(at::Tensor dwn, at::Tensor wraw, at::Tensor w1, at::Tensor w2, at::Tensor dwu, at::Tensor pw) {
  for (auto* t : {&dwn, &wraw, &w1, &w2, &pw}) f32c(*t, "unfold: fp32");
  TORCH_CHECK(dwn.numel() == 4 * D * DC && wraw.numel() == 4 * D * DC && dwu.numel() == 4 * D * DC && dwu.is_contiguous()
              && w1.numel() == DC && w2.numel() == DC && pw.numel() == UNF_BLOCKS * DC,
              "unfold: dWn / Wraw / dWraw [3072, 384], w [384], partials [192, 384]");
  TDT_DISPATCH(dwu.scalar_type(), OT, [&] {
    unfold_k<OT><<<UNF_BLOCKS, 384, 0, S()>>>(CP<float>(dwn), CP<float>(wraw), CP<float>(w1), CP<float>(w2), P<OT>(dwu), P<float>(pw));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pair_bias(at::Tensor z, at::Tensor wf, at::Tensor bias, at::Tensor pst, double eps) {
  const int64_t R = z.size(0);
  bf16c(z, "pair"); bf16c(wf, "Wf");
  const int64_t nh = wf.size(0);
  TORCH_CHECK(z.is_contiguous() && z.size(1) == 128 && R % 128 == 0 && wf.is_contiguous() && (nh == 12 || nh == 16 || nh == 24)
              && wf.size(1) == 128, "pair_bias: pair [R, 128] (R a multiple of 128), Wf [12, 16 or 24, 128] bf16");
  TORCH_CHECK(bias.is_contiguous() && bias.scalar_type() == at::kBFloat16 && bias.numel() == nh * R && pst.is_contiguous()
              && pst.numel() == 2 * R, "pair_bias: bias [nh, R] bf16, pst [R, 2]");
  auto k = nh == 12 ? pair_bias_k<12> : nh == 16 ? pair_bias_k<16> : pair_bias_k<24>;
  k<<<grid(k, R / 64, 1), WPB * 32, 0, S()>>>(CP<bf>(z), CP<bf>(wf), P<bf>(bias), P<float2>(pst), R, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t pair_bias_bwd(at::Tensor db, at::Tensor z, at::Tensor pst, at::Tensor wf, at::Tensor dz, at::Tensor dwf) {
  const int64_t R = z.size(0), nh = wf.size(0);
  bf16c(z, "pair"); bf16c(wf, "Wf"); f32c(dwf, "dWf partials");
  TORCH_CHECK(z.is_contiguous() && z.size(1) == 128 && R % 128 == 0 && wf.is_contiguous() && (nh == 12 || nh == 16 || nh == 24)
              && wf.size(1) == 128, "pair_bias_bwd: pair [R, 128] (R a multiple of 128), Wf [12, 16 or 24, 128] bf16");
  TORCH_CHECK(db.scalar_type() == at::kBFloat16 && db.is_contiguous() && db.numel() == nh * R, "pair_bias_bwd: dbias [nh, R]");
  TORCH_CHECK(dz.is_contiguous() && dz.scalar_type() == at::kBFloat16 && pst.is_contiguous(), "pair_bias_bwd: d pair bf16");
  static bool attr = [] {
    return cudaFuncSetAttribute(pair_bias_bwd_k<12>, cudaFuncAttributeMaxDynamicSharedMemorySize, PB<12>::SMEM) == cudaSuccess
        && cudaFuncSetAttribute(pair_bias_bwd_k<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, PB<16>::SMEM) == cudaSuccess
        && cudaFuncSetAttribute(pair_bias_bwd_k<24>, cudaFuncAttributeMaxDynamicSharedMemorySize, PB<24>::SMEM) == cudaSuccess;
  }();
  TORCH_CHECK(attr, "pair_bias_bwd: shared memory attribute");
  auto k = nh == 12 ? pair_bias_bwd_k<12> : nh == 16 ? pair_bias_bwd_k<16> : pair_bias_bwd_k<24>;
  const int smem = nh == 12 ? PB<12>::SMEM : nh == 16 ? PB<16>::SMEM : PB<24>::SMEM;
  const unsigned g = grid(k, R / 128, 1, smem);
  TORCH_CHECK(dwf.numel() >= (int64_t)g * nh * 128, "pair_bias_bwd: dWf partials [partial_rows, nh, 128]");
  k<<<g, WPB * 32, smem, S()>>>(CP<bf>(db), CP<bf>(z), CP<float2>(pst), CP<bf>(wf), P<bf>(dz), P<float>(dwf), R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return g;
}

// dst[k] = src[k] (o scale[k][i % scale[k].numel()] when scale[k] is not empty), or dst[k] = src[k]^T when tr[k]; bf16 / fp32 contiguous
void pack16(std::vector<at::Tensor> src, std::vector<at::Tensor> dst, std::vector<at::Tensor> scale, std::vector<int64_t> tr) {
  const size_t n = src.size();
  TORCH_CHECK(n >= 1 && n <= (size_t)PACK_SEGS && dst.size() == n && scale.size() == n && tr.size() == n, "pack16: 1..48 segments");
  auto f32 = [](const at::Tensor& t, const char* w) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && (t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16)
                && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "pack16 ", w, ": bf16 / fp32 contiguous, 16-byte aligned");
    return t.scalar_type() == at::kFloat ? 1 : 0;
  };
  Pack16Segs segs{};
  long blocks = 0;
  for (size_t k = 0; k < n; ++k) {
    Pack16Seg& g = segs.s[k];
    g.sdt = f32(src[k], "src"); g.ddt = f32(dst[k], "dst");
    TORCH_CHECK(src[k].numel() == dst[k].numel() && src[k].numel() % 4 == 0 && src[k].numel() < (1L << 31), "pack16: segment sizes");
    g.src = src[k].data_ptr(); g.dst = dst[k].data_ptr(); g.n = (int)src[k].numel(); g.tr = tr[k] ? 1 : 0;
    g.scale = nullptr; g.period = 4; g.cdt = 1;
    if (g.tr) {
      TORCH_CHECK(src[k].dim() == 2 && dst[k].dim() == 2 && dst[k].size(0) == src[k].size(1) && dst[k].size(1) == src[k].size(0)
                  && src[k].size(0) % 32 == 0 && src[k].size(1) % 32 == 0
                  && (scale[k].numel() == 0 || scale[k].numel() == src[k].size(1)),
                  "pack16: transpose [r, c] -> [c, r], 32 | r, c, scale empty or [c]");
      g.rows = (int)src[k].size(0); g.cols = (int)src[k].size(1);
      if (scale[k].numel()) { g.cdt = f32(scale[k], "scale"); g.scale = scale[k].data_ptr(); g.period = (int)scale[k].numel(); }
      g.b0 = (int)blocks;
      blocks += ((long)(g.rows / 32) * (g.cols / 32) + PACK_TILES - 1) / PACK_TILES;
    } else {
      if (scale[k].numel()) {
        g.cdt = f32(scale[k], "scale");
        TORCH_CHECK(scale[k].numel() % 4 == 0 && src[k].numel() % scale[k].numel() == 0, "pack16: scale period divides the segment");
        g.scale = scale[k].data_ptr(); g.period = (int)scale[k].numel();
      }
      g.b0 = (int)blocks;
      blocks += std::max<long>(1, (src[k].numel() / 4 + PACK_F4 - 1) / PACK_F4);
    }
  }
  segs.n = (int)n;
  const at::cuda::CUDAGuard gd(src[0].device());
  pack16_k<<<(unsigned)blocks, 256, 0, S()>>>(segs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swiglu(at::Tensor ab, at::Tensor h) {
  const int64_t M = h.size(0), N = h.size(1);
  TORCH_CHECK(ab.scalar_type() == at::kBFloat16 && h.scalar_type() == at::kBFloat16 && ab.is_contiguous() && h.is_contiguous()
              && ab.size(0) == M && ab.size(1) == 2 * N && N % 8 == 0, "swiglu: ab [M, 2N], h [M, N] bf16");
  const int64_t n = M * (N / 8);
  if (n) swiglu_k<<<(unsigned)((n + 255) / 256), 256, 0, S()>>>(CP<bf>(ab), P<bf>(h), M, (int)N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swiglu_bwd(at::Tensor dh, at::Tensor ab, at::Tensor dab) {
  const int64_t M = dh.size(0), N = dh.size(1);
  TORCH_CHECK(dh.scalar_type() == at::kBFloat16 && ab.scalar_type() == at::kBFloat16 && dab.scalar_type() == at::kBFloat16
              && dh.is_contiguous() && ab.is_contiguous() && dab.is_contiguous() && ab.size(0) == M && ab.size(1) == 2 * N
              && dab.sizes() == ab.sizes() && N % 8 == 0, "swiglu_bwd: dh [M, N], ab / dab [M, 2N] bf16");
  const int64_t n = M * (N / 8);
  if (n) swiglu_bwd_k<<<(unsigned)((n + 255) / 256), 256, 0, S()>>>(CP<bf>(dh), CP<bf>(ab), P<bf>(dab), M, (int)N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cond_ln_cuda", &cond_ln);
  m.def("swiglu_cuda", &swiglu);
  m.def("pair_bias_cuda", &pair_bias);
  m.def("unfold_cuda", &unfold);
  m.def("swiglu_bwd_cuda", &swiglu_bwd);
  m.def("adaln_a_cuda", &adaln_a);
  m.def("res_adaln_b_cuda", &res_adaln_b);
  m.def("res_c_cuda", &res_c);
  m.def("res_c_bwd_cuda", &res_c_bwd);
  m.def("res_adaln_b_bwd_cuda", &res_adaln_b_bwd);
  m.def("adaln_a_bwd_cuda", &adaln_a_bwd);
  m.def("cond_bwd_cuda", &cond_bwd);
  m.def("partial_rows", &partial_rows);
  m.def("pair_bias_bwd_cuda", &pair_bias_bwd);
  m.def("finalize_cuda", &finalize);
  m.def("pair_bias_bwd_fin_cuda", &pair_bias_bwd_fin);
  m.def("pack16_cuda", &pack16);
}
