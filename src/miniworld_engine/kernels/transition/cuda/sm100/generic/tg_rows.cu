// tg_rows.cu -- the row passes of the generic sm_100a Transition (around the tcgen05 GEMMs of tg_gemm.cu), -DF32=1 for fp32 activations:
//
//   tg_ln_fwd_*   xn = LN(x) gamma + beta (bf16, or fp32 rounded to the TF32 grid: the expand GEMM's operand), and the row's mean / rstd
//   tg_ln_bwd_*   dx = dy + LN backward of d_xn (fp32) through gamma; per-block partial sums of d_xn xhat (dgamma) and d_xn (dbeta);
//                 tg_ln_bwds_* the same over the sum of a split-K GEMM's partial slabs, added in slab order
//   tg_colsum     column sums of [P, N] fp32 partials in a fixed order (dgamma | dbeta; the ConditionedTransition's gate bias)
//   tg_splitsum   out = res + the sum of a split-K GEMM's fp32 partial slabs in slab order (the squeeze + residual at small M)
//   tg_prep       the weights in the activation dtype (fp32: on the TF32 grid), and for the backward Ws^T and [Wa; Wb]^T
//
// A row is TPR consecutive lanes (8 / 16 / 32: a warp holds 32 / TPR rows, so a narrow row does not leave lanes idle), a lane loading
// 16-B vectors where the width allows (VEC 8 bf16 / 4 fp32). D a multiple of 64 up to 1024; the host picks the instantiation. Statistics in fp32 from the row held in registers (two passes, no E[x^2] - E[x]^2).
// Every reduction has a fixed order: a replay is bit-identical.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef F32
#define F32 0
#endif
#if F32
typedef float T;
#else
typedef __nv_bfloat16 T;
#endif


__device__ __forceinline__ float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
__device__ __forceinline__ float2 ld2(const T* p) {
#if F32
  return *reinterpret_cast<const float2*>(p);
#else
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
#endif
}
__device__ __forceinline__ void st2(T* p, float a, float b) {
#if F32
  *reinterpret_cast<float2*>(p) = make_float2(a, b);
#else
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
#endif
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// A row is TPR consecutive lanes (a warp holds 32 / TPR rows); a lane owns NV = D / TPR columns as VEC-wide vectors (16 B where the width
// allows) at columns VEC (TPR i + l): one access of a row group is a contiguous run. (VEC, TPR) are chosen per width by the host.
template <int VEC> __device__ __forceinline__ void ldv(const T* p, float* v) {
#pragma unroll
  for (int e = 0; e < VEC; e += 2) { const float2 t = ld2(p + e); v[e] = t.x; v[e + 1] = t.y; }
}
template <int VEC> __device__ __forceinline__ void stv(T* p, const float* v) {
#if F32
  if constexpr (VEC == 4) { *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]); return; }
#else
  if constexpr (VEC == 8) {
    uint4 w;
    __nv_bfloat162 h[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) h[e] = __floats2bfloat162_rn(v[2 * e], v[2 * e + 1]);
    w = *reinterpret_cast<uint4*>(h);
    *reinterpret_cast<uint4*>(p) = w;
    return;
  }
#endif
#pragma unroll
  for (int e = 0; e < VEC; e += 2) st2(p + e, v[e], v[e + 1]);
}
template <int VEC> __device__ __forceinline__ void ldvx(const T* p, float* v) {   // one 16-B load where VEC fills it
#if F32
  if constexpr (VEC == 4) { const float4 t = *reinterpret_cast<const float4*>(p); v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w; return; }
#else
  if constexpr (VEC == 8) {
    const uint4 w = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float2 t = __bfloat1622float2(h[e]); v[2 * e] = t.x; v[2 * e + 1] = t.y; }
    return;
  }
#endif
  ldv<VEC>(p, v);
}
template <int TPR> __device__ __forceinline__ float gsum(float v) {
#pragma unroll
  for (int o = TPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int VEC, int TPR, int NV>
__device__ __forceinline__ void ln_fwd(const T* __restrict__ X, const float* __restrict__ G, const float* __restrict__ Bt, T* __restrict__ XN,
                                       float* __restrict__ MEAN, float* __restrict__ RSTD, int M, int D, float eps, int save) {
  constexpr int RPW = 32 / TPR;
  const int lane = threadIdx.x & 31, l = lane % TPR, sub = lane / TPR;
  const int nw = gridDim.x * (blockDim.x >> 5);
  // the next row group's x is loaded before this one is reduced and stored: two row groups in flight per warp
  int base = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RPW;
  float nx[NV];
  auto load = [&](int b, float* v) {
    const int rr = b + sub;
    const T* x = X + (long)(rr < M ? rr : 0) * D;
#pragma unroll
    for (int i = 0; i < NV / VEC; ++i) ldvx<VEC>(x + VEC * (TPR * i + l), v + VEC * i);
  };
  if (base < M) load(base, nx);
  for (; base < M; base += nw * RPW) {
    const int row = base + sub;
    const bool ok = row < M;
    float v[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) v[i] = nx[i];
    if (base + nw * RPW < M) load(base + nw * RPW, nx);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < NV; ++i) s += v[i];
    const float mean = gsum<TPR>(s) / (float)D;
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < NV; ++i) { const float a = v[i] - mean; q += a * a; }
    const float rstd = rsqrtf(gsum<TPR>(q) / (float)D + eps);
    if (!ok) continue;
    T* xn = XN + (long)row * D;
#pragma unroll
    for (int i = 0; i < NV / VEC; ++i) {
      const int c = VEC * (TPR * i + l);
      float o[VEC];
#pragma unroll
      for (int e = 0; e < VEC; ++e) {
        o[e] = (v[VEC * i + e] - mean) * rstd * __ldg(G + c + e) + __ldg(Bt + c + e);
#if F32
        o[e] = tf32r(o[e]);
#endif
      }
      stv<VEC>(xn + c, o);
    }
    if (save && l == 0) { MEAN[row] = mean; RSTD[row] = rstd; }
  }
}

// dx = dy + rstd (g - mean(g) - xhat mean(g xhat)), g = d_xn gamma; PART[block] = [sum d_xn xhat | sum d_xn] over the block's rows
template <int VEC, int TPR, int NV, bool SPLIT>
__device__ __forceinline__ void ln_bwd(const float* __restrict__ DXN, const T* __restrict__ X, const T* __restrict__ DY,
                                       const float* __restrict__ MEAN, const float* __restrict__ RSTD, const float* __restrict__ G,
                                       T* __restrict__ DX, float* __restrict__ PART, int M, int D, int NS, long SST, float* red) {
  constexpr int RPW = 32 / TPR;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, l = lane % TPR, sub = lane / TPR;
  const int nwb = blockDim.x >> 5, nw = gridDim.x * nwb;
  float dg[NV], db[NV];
#pragma unroll
  for (int i = 0; i < NV; ++i) { dg[i] = 0.f; db[i] = 0.f; }
  for (int base = (blockIdx.x * nwb + warp) * RPW; base < M; base += nw * RPW) {
    const int row = base + sub;
    const bool ok = row < M;
    const long ro = (long)(ok ? row : 0) * D;
    const float mean = ok ? MEAN[row] : 0.f, rstd = ok ? RSTD[row] : 0.f;
    float gx[NV], xh[NV];
    float s1 = 0.f, s2 = 0.f;
    if constexpr (SPLIT) {
      // the row's d_xn: slab 0, then the split-K slabs added in slab order with SB slabs' whole row in flight at a time (a single track:
      // one row per warp, so the slab loads are the latency that counts); x
      float dd[NV], xv[NV];
      auto ldrow = [&](const float* P, float* v) {
#pragma unroll
        for (int i = 0; i < NV / VEC; ++i) {
          const float* q = P + ro + VEC * (TPR * i + l);
#pragma unroll
          for (int e = 0; e < VEC; e += 2) {
            if constexpr (VEC % 4 == 0) {
              if (e % 4 == 0) { const float4 t = *reinterpret_cast<const float4*>(q + e); v[VEC * i + e] = t.x; v[VEC * i + e + 1] = t.y; v[VEC * i + e + 2] = t.z; v[VEC * i + e + 3] = t.w; }
            } else {
              const float2 t = *reinterpret_cast<const float2*>(q + e); v[VEC * i + e] = t.x; v[VEC * i + e + 1] = t.y;
            }
          }
        }
      };
      ldrow(DXN, dd);
#pragma unroll
      for (int i = 0; i < NV / VEC; ++i) ldvx<VEC>(X + ro + VEC * (TPR * i + l), xv + VEC * i);
      constexpr int SB = NV <= 16 ? 4 : 2;
      for (int s0 = 1; s0 < NS; s0 += SB) {
        float t[SB][NV];
#pragma unroll
        for (int u = 0; u < SB; ++u) ldrow(DXN + (s0 + u < NS ? s0 + u : 0) * SST, t[u]);
#pragma unroll
        for (int u = 0; u < SB; ++u)
          if (s0 + u < NS) {
#pragma unroll
            for (int k = 0; k < NV; ++k) dd[k] += t[u][k];
          }
      }
#pragma unroll
      for (int i = 0; i < NV / VEC; ++i) {
        const int c = VEC * (TPR * i + l);
#pragma unroll
        for (int e = 0; e < VEC; ++e) {
          const int k = VEC * i + e;
          const float d = ok ? dd[k] : 0.f;
          xh[k] = (xv[k] - mean) * rstd;
          gx[k] = d * __ldg(G + c + e);
          s1 += gx[k];
          s2 += gx[k] * xh[k];
          dg[k] += d * xh[k];
          db[k] += d;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < NV / VEC; ++i) {
        const int c = VEC * (TPR * i + l);
        float d[VEC], xv[VEC];
#pragma unroll
        for (int e = 0; e < VEC; e += 4) {
          if (e + 4 <= VEC) { const float4 t = *reinterpret_cast<const float4*>(DXN + ro + c + e); d[e] = t.x; d[e + 1] = t.y; d[e + 2] = t.z; d[e + 3] = t.w; }
        }
        if constexpr (VEC < 4) {
#pragma unroll
          for (int e = 0; e < VEC; e += 2) { const float2 t = *reinterpret_cast<const float2*>(DXN + ro + c + e); d[e] = t.x; d[e + 1] = t.y; }
        }
        ldvx<VEC>(X + ro + c, xv);
#pragma unroll
        for (int e = 0; e < VEC; ++e) {
          if (!ok) d[e] = 0.f;
          const int k = VEC * i + e;
          xh[k] = (xv[e] - mean) * rstd;
          gx[k] = d[e] * __ldg(G + c + e);
          s1 += gx[k];
          s2 += gx[k] * xh[k];
          dg[k] += d[e] * xh[k];
          db[k] += d[e];
        }
      }
    }
    const float m1 = gsum<TPR>(s1) / (float)D, m2 = gsum<TPR>(s2) / (float)D;
    if (!ok) continue;
#pragma unroll
    for (int i = 0; i < NV / VEC; ++i) {
      const int c = VEC * (TPR * i + l);
      float r[VEC], o[VEC];
      ldvx<VEC>(DY + ro + c, r);
#pragma unroll
      for (int e = 0; e < VEC; ++e) { const int k = VEC * i + e; o[e] = r[e] + rstd * (gx[k] - m1 - xh[k] * m2); }
      stv<VEC>(DX + ro + c, o);
    }
  }
  // the block's partials: per 128 / 256-column chunk (a warp row group's i-th vectors), every (warp, row slot) writes its column sums to
  // its own shared-memory row, then each column adds the rows in (warp, slot) order -- the order a slot-by-slot accumulation would take,
  // without its 8-32 serial read-modify-write rounds
  constexpr int CW = VEC * TPR;                              // columns of one chunk
  const int slot = warp * RPW + sub, nslot = nwb * RPW;
#pragma unroll 1
  for (int pass = 0; pass < 2; ++pass)
#pragma unroll
    for (int i = 0; i < NV / VEC; ++i) {
      __syncthreads();
#pragma unroll
      for (int e = 0; e < VEC; ++e) red[slot * CW + VEC * l + e] = pass ? db[VEC * i + e] : dg[VEC * i + e];
      __syncthreads();
      for (int c = threadIdx.x; c < CW; c += blockDim.x) {
        float s = 0.f;
#pragma unroll 8
        for (int q = 0; q < nslot; ++q) s += red[q * CW + c];
        PART[(long)blockIdx.x * 2 * D + pass * D + CW * i + c] = s;
      }
    }
}

// the instantiations the host picks from: tg_ln_{fwd,bwd}_v<VEC>t<TPR>n<NV> (NV = D / TPR, at most 32)
#define LN_INST(VEC, TPR, NV)                                                                                                             \
  extern "C" __global__ void __launch_bounds__(256) tg_ln_fwd_v##VEC##t##TPR##n##NV(                                                     \
      const T* X, const float* G, const float* Bt, T* XN, float* MEAN, float* RSTD, int M, int D, float eps, int save) {                 \
    ln_fwd<VEC, TPR, NV>(X, G, Bt, XN, MEAN, RSTD, M, D, eps, save);                                                                     \
  }                                                                                                                                       \
  extern "C" __global__ void __launch_bounds__(256) tg_ln_bwd_v##VEC##t##TPR##n##NV(                                                     \
      const float* DXN, const T* X, const T* DY, const float* MEAN, const float* RSTD, const float* G, T* DX, float* PART, int M, int D,  \
      int NS, int SST) {                                                                                                                  \
    __shared__ float red[2048];   /* (8 warps x RPW slots) x VEC TPR columns */                                                          \
    ln_bwd<VEC, TPR, NV, false>(DXN, X, DY, MEAN, RSTD, G, DX, PART, M, D, 1, 0, red);                                                   \
  }                                                                                                                                       \
  extern "C" __global__ void __launch_bounds__(256) tg_ln_bwds_v##VEC##t##TPR##n##NV(                                                    \
      const float* DXN, const T* X, const T* DY, const float* MEAN, const float* RSTD, const float* G, T* DX, float* PART, int M, int D,  \
      int NS, int SST) {                                                                                                                  \
    __shared__ float red[2048];                                                                                                           \
    ln_bwd<VEC, TPR, NV, true>(DXN, X, DY, MEAN, RSTD, G, DX, PART, M, D, NS, (long)SST, red);                                           \
  }
#if F32
// fp32: 16-B vectors (VEC 4); D = 64 TPR 16, D = 128 TPR 32 n4, and every multiple of 128 up to 1024 at TPR 32 (n = D / 32); D = 64 (2k+1)
// (192, 320, ...) at VEC 2, TPR 32
LN_INST(4, 16, 4)
LN_INST(4, 32, 4) LN_INST(4, 32, 8) LN_INST(4, 32, 12) LN_INST(4, 32, 16) LN_INST(4, 32, 20) LN_INST(4, 32, 24) LN_INST(4, 32, 28) LN_INST(4, 32, 32)
LN_INST(2, 32, 6) LN_INST(2, 32, 10) LN_INST(2, 32, 14) LN_INST(2, 32, 18) LN_INST(2, 32, 22) LN_INST(2, 32, 26) LN_INST(2, 32, 30)
#else
// bf16: 16-B vectors (VEC 8); D = 64 TPR 8, D = 128 TPR 16, multiples of 256 at TPR 32; D = 64 (2k+1) / 128 (2k+1) at VEC 2 / 4, TPR 32
LN_INST(8, 8, 8) LN_INST(8, 16, 8)
LN_INST(8, 32, 8) LN_INST(8, 32, 16) LN_INST(8, 32, 24) LN_INST(8, 32, 32)
LN_INST(4, 32, 12) LN_INST(4, 32, 20) LN_INST(4, 32, 28)
LN_INST(2, 32, 6) LN_INST(2, 32, 10) LN_INST(2, 32, 14) LN_INST(2, 32, 18) LN_INST(2, 32, 22) LN_INST(2, 32, 26) LN_INST(2, 32, 30)
#endif

// fp32 values -> the TF32 grid (round to nearest, ties away), 4 per thread: the backward's dy, the A operand of the GATE GEMM, whose MMA
// would otherwise truncate it (a biased error: measured 1.5x the PyTorch TF32 error on dgamma at a single-track shape)
extern "C" __global__ void __launch_bounds__(256) tg_round_tf32(const float4* __restrict__ X, float4* __restrict__ Y, int n4) {   // ints are 32-bit at the launch surface
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x) {
    const float4 v = X[i];
    Y[i] = make_float4(tf32r(v.x), tf32r(v.y), tf32r(v.z), tf32r(v.w));
  }
}

// 32 columns per block, the rows dealt round robin over the 8 warps (row r to warp r % 8, each warp adding its rows in order), then the
// 8 warp sums added in warp order: a fixed order, and R / 8 loads in flight per thread instead of R in sequence
extern "C" __global__ void __launch_bounds__(256) tg_colsum(const float* __restrict__ P, float* __restrict__ OUT, int R, int N) {
  __shared__ float red[8][33];
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, c = blockIdx.x * 32 + l;
  float s = 0.f;
  if (c < N) {
#pragma unroll 4
    for (int r = w; r < R; r += 8) s += P[(long)r * N + c];
  }
  red[w][l] = s;
  __syncthreads();
  if (w == 0 && c < N) {
    float t = red[0][l];
#pragma unroll
    for (int q = 1; q < 8; ++q) t += red[q][l];
    OUT[c] = t;
  }
}

// out [M, N] (T) = res + sum over the NS fp32 slabs P[s] (slab stride SST elements, row stride N), the slabs added in order first; 4 columns
// per thread (N a multiple of 4)
extern "C" __global__ void __launch_bounds__(256) tg_splitsum(const float* __restrict__ P, const T* __restrict__ RES, T* __restrict__ OUT, int M,
                                                             int N, int NS, int SST) {
  const long n4 = (long)M * N / 4;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x) {
    float4 a = reinterpret_cast<const float4*>(P)[i];
    for (int sl = 1; sl < NS; ++sl) {
      const float4 t = reinterpret_cast<const float4*>(P + (long)sl * SST)[i];
      a.x += t.x; a.y += t.y; a.z += t.z; a.w += t.w;
    }
    float r[4], o[4];
    ldv<4>(RES + 4 * i, r);
    o[0] = r[0] + a.x; o[1] = r[1] + a.y; o[2] = r[2] + a.z; o[3] = r[3] + a.w;
    stv<4>(OUT + 4 * i, o);
  }
}

// Wa, Wb [H, D] and Ws [D, H] (fp32 when src_f32, else bf16) -> wa, wb, ws in T (fp32: TF32 grid); with `bwd` also Ws^T [H, D] and
// [Wa; Wb]^T [D, 2H]. Without `bwd` a grid-stride loop over the elements; with it one block per 32 x 32 tile (H, D multiples of 32): the
// first (H / 32)(D / 32) blocks take a tile of Wa and Wb, the rest a tile of Ws, and the transposes go through shared memory, so every
// global access is a coalesced 32-element row run (the per-element scattered transposes took 14 us at [1536, 384]; now 4).
__device__ __forceinline__ float ldw(const void* W, long e, int src_f32) {
  return src_f32 ? reinterpret_cast<const float*>(W)[e] : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(W)[e]);
}
__device__ __forceinline__ T cvw(float v) {
#if F32
  return tf32r(v);
#else
  return __float2bfloat16_rn(v);
#endif
}
extern "C" __global__ void __launch_bounds__(256) tg_prep(const void* __restrict__ WA, const void* __restrict__ WB, const void* __restrict__ WS,
                                                         T* __restrict__ OA, T* __restrict__ OB, T* __restrict__ OS, T* __restrict__ OST,
                                                         T* __restrict__ OABT, int H, int D, int src_f32, int bwd) {
  if (!bwd) {                                                // no transposes: a grid-stride loop over the elements
    const long n = (long)H * D;
    for (long e = (long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long)gridDim.x * blockDim.x) {
      OA[e] = cvw(ldw(WA, e, src_f32)); OB[e] = cvw(ldw(WB, e, src_f32)); OS[e] = cvw(ldw(WS, e, src_f32));
    }
    return;
  }
  __shared__ T ta[32][33], tb[32][33];
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5;      // 8 warps, 4 rows of the tile each
  const int nab = (H / 32) * (D / 32);
  if ((int)blockIdx.x < nab) {                               // Wa, Wb tile: rows i0.. of H, columns j0.. of D
    const int i0 = (blockIdx.x / (D / 32)) * 32, j0 = (blockIdx.x % (D / 32)) * 32;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int r = w + 8 * q;
      const long e = (long)(i0 + r) * D + j0 + l;
      const T a = cvw(ldw(WA, e, src_f32)), b = cvw(ldw(WB, e, src_f32));
      OA[e] = a; OB[e] = b;
      ta[r][l] = a; tb[r][l] = b;
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < 4; ++q) {                            // [Wa; Wb]^T [D, 2H]: row j0 + r, columns i0.. and H + i0..
      const int r = w + 8 * q;
      const long o = (long)(j0 + r) * 2 * H + i0 + l;
      OABT[o] = ta[l][r];
      OABT[o + H] = tb[l][r];
    }
  } else {                                                   // Ws tile: rows s0.. of D, columns t0.. of H
    const int b = blockIdx.x - nab;
    const int s0 = (b / (H / 32)) * 32, t0 = (b % (H / 32)) * 32;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int r = w + 8 * q;
      const long e = (long)(s0 + r) * H + t0 + l;
      const T v = cvw(ldw(WS, e, src_f32));
      OS[e] = v;
      ta[r][l] = v;
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < 4; ++q) {                            // Ws^T [H, D]: row t0 + r, columns s0..
      const int r = w + 8 * q;
      OST[(long)(t0 + r) * D + s0 + l] = ta[l][r];
    }
  }
}
