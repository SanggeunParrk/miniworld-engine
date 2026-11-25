// tg_roits tile's max |w| and (Wa, Wb tiles) the largest
//                sum of squares of one of its rows over its 32 columns (PART[t] as a float4);s.cu -- the row passes of the generic sm_100a Transition (around the tcgen05 GEMMs of tg_gemm.cu), -DF32=1 for fp32 activations:
//
//   tg_ln_fwd_*   xn = LN(x) gamma + beta (bf16, or fp32 rounded to the TF32 grid: the expand GEMM's operand), and the row's mean / rstd
//   tg_ln_bwd_*   dx = dy + LN backward of d_xn (fp32) through gamma; per-block partial sums of d_xn xhat (dgamma) and d_xn (dbeta);
//                 tg_ln_bwds_* the same over the sum of a split-K GEMM's partial slabs, added in slab order
//   tg_colsum     column sums of [P, N] fp32 partials in a fixed order (dgamma | dbeta; the ConditionedTransition's gate bias)
//   tg_splitsum   out = res + the sum of a split-K GEMM's fp32 partial slabs in slab order (the squeeze's split route, kept for A/B runs:
//                 at small M the host takes cuBLAS addmm, measured faster than the split GEMM plus this launch)
//   tg_prep       the weights in the activation dtype (fp32: on the TF32 grid), and for the backward Ws^T and [Wa; Wb]^T
//   tg_prep16     (-DF32=1, round 3 part 7) tg_prep's copies plus the fp16 forward forms and their device scales in ONE launch (the fp32
//                 training forward on the fp16 tensor cores; a software grid barrier between the bounds pass and the conversion pass)
//
// A row is TPR consecutive lanes (8 / 16 / 32: a warp holds 32 / TPR rows, so a narrow row does not leave lanes idle), a lane loading
// 16-B vectors where the width allows (VEC 8 bf16 / 4 fp32). D a multiple of 64 up to 1024; the host picks the instantiation. Statistics in fp32 from the row held in registers (two passes, no E[x^2] - E[x]^2).
// Every reduction has a fixed order: a replay is bit-identical.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef F32
#define F32 0
#endif
#if F32
typedef float T;
#else
typedef __nv_bfloat16 T;
#endif


// (round 3 part 5) every kernel here but tg_prep (whose copies the small-M chains read before their griddepcontrol.wait) starts with PDL_ENTRY: launched as a programmatic dependent launch it waits for the previous launch's
// outputs, then lets the next launch start its prologue (a no-op without the launch attribute)
#define PDL_ENTRY() asm volatile("griddepcontrol.wait;\n\tgriddepcontrol.launch_dependents;" ::: "memory")
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
// the lane's NV columns of an fp32 vector (gamma, beta), loaded once per kernel with 8 / 16-B loads: a per-element __ldg inside the row
// loop is one scalar L1 request per element per lane (32 sectors per warp instruction for 1 sector of data), which bound the row passes
// at ~2 TB/s (bf16 [147 456, 768]: fwd 233 us against a 74 us copy)
template <int VEC, int TPR, int NV> __device__ __forceinline__ void ldcols(const float* __restrict__ P, int l, float* v) {
#pragma unroll
  for (int i = 0; i < NV / VEC; ++i) {
    const float* q = P + VEC * (TPR * i + l);
#pragma unroll
    for (int e = 0; e < VEC; e += 2) {
      if constexpr (VEC % 4 == 0) {
        if (e % 4 == 0) { const float4 t = __ldg(reinterpret_cast<const float4*>(q + e)); v[VEC * i + e] = t.x; v[VEC * i + e + 1] = t.y; v[VEC * i + e + 2] = t.z; v[VEC * i + e + 3] = t.w; }
      } else {
        const float2 t = __ldg(reinterpret_cast<const float2*>(q + e)); v[VEC * i + e] = t.x; v[VEC * i + e + 1] = t.y;
      }
    }
  }
}
template <int TPR> __device__ __forceinline__ float gsum(float v) {
#pragma unroll
  for (int o = TPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int VEC, int TPR, int NV>
__device__ __forceinline__ void ln_fwd(const T* __restrict__ X, const float* __restrict__ G, const float* __restrict__ Bt, T* __restrict__ XN,
                                       float* __restrict__ MEAN, float* __restrict__ RSTD, int M, int D, float eps, int save, T* __restrict__ XC, int copy) {
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
  float gc[NV], bc[NV];
  ldcols<VEC, TPR, NV>(G, l, gc);
  ldcols<VEC, TPR, NV>(Bt, l, bc);
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
        o[e] = (v[VEC * i + e] - mean) * rstd * gc[VEC * i + e] + bc[VEC * i + e];
#if F32
        o[e] = tf32r(o[e]);
#endif
      }
      stv<VEC>(xn + c, o);
      if (copy) stv<VEC>(XC + (long)row * D + c, v + VEC * i);   // x itself (exact): the in-place cuBLAS squeeze accumulates onto it
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
  float dg[NV], db[NV], gc[NV];
#pragma unroll
  for (int i = 0; i < NV; ++i) { dg[i] = 0.f; db[i] = 0.f; }
  ldcols<VEC, TPR, NV>(G, l, gc);
  for (int base = (blockIdx.x * nwb + warp) * RPW; base < M; base += nw * RPW) {
    const int row = base + sub;
    const bool ok = row < M;
    const long ro = (long)(ok ? row : 0) * D;
    const float mean = ok ? MEAN[row] : 0.f, rstd = ok ? RSTD[row] : 0.f;
    float gx[NV], xh[NV], ry[NV];
    // a warp-wide row (TPR 32): dy (the residual branch) is issued with the row's other loads, not after its reductions -- one memory
    // round trip per row (bf16 [147 456, 768] 384 -> 210 us). Rows of 8 / 16 lanes already keep 2-4 rows in flight per warp and lose by it.
    constexpr bool PREF = TPR == 32;
    if constexpr (PREF) {
#pragma unroll
      for (int i = 0; i < NV / VEC; ++i) ldvx<VEC>(DY + ro + VEC * (TPR * i + l), ry + VEC * i);
    }
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
          gx[k] = d * gc[k];
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
          gx[k] = d[e] * gc[k];
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
      float o[VEC];
      if constexpr (!PREF) ldvx<VEC>(DY + ro + c, ry + VEC * i);
#pragma unroll
      for (int e = 0; e < VEC; ++e) { const int k = VEC * i + e; o[e] = ry[k] + rstd * (gx[k] - m1 - xh[k] * m2); }
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
      const T* X, const float* G, const float* Bt, T* XN, float* MEAN, float* RSTD, int M, int D, float eps, int save, T* XC, int copy) {                \
    PDL_ENTRY();                                                                                                                          \
    ln_fwd<VEC, TPR, NV>(X, G, Bt, XN, MEAN, RSTD, M, D, eps, save, XC, copy);                                                                 \
  }                                                                                                                                       \
  extern "C" __global__ void __launch_bounds__(256) tg_ln_bwd_v##VEC##t##TPR##n##NV(                                                     \
      const float* DXN, const T* X, const T* DY, const float* MEAN, const float* RSTD, const float* G, T* DX, float* PART, int M, int D,  \
      int NS, int SST) {                                                                                                                  \
    __shared__ float red[2048];   /* (8 warps x RPW slots) x VEC TPR columns */                                                          \
    PDL_ENTRY();                                                                                                                          \
    ln_bwd<VEC, TPR, NV, false>(DXN, X, DY, MEAN, RSTD, G, DX, PART, M, D, 1, 0, red);                                                   \
  }                                                                                                                                       \
  extern "C" __global__ void __launch_bounds__(256) tg_ln_bwds_v##VEC##t##TPR##n##NV(                                                    \
      const float* DXN, const T* X, const T* DY, const float* MEAN, const float* RSTD, const float* G, T* DX, float* PART, int M, int D,  \
      int NS, int SST) {                                                                                                                  \
    __shared__ float red[2048];                                                                                                           \
    PDL_ENTRY();                                                                                                                          \
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
  PDL_ENTRY();
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x) {
    const float4 v = X[i];
    Y[i] = make_float4(tf32r(v.x), tf32r(v.y), tf32r(v.z), tf32r(v.w));
  }
}

// (round 3 part 5) tg_round_tf32 that also writes the TF32 residual L = tf32(x - Y): the weight-gradient GEMM (tg_dw.cu -DSPL) forms
// dWs = (dy_hi + dy_lo)^T h, as cuBLAS' fp32 GEMM saw dy unrounded (dy rounded once to TF32 measured 1.2-1.4x the YELLOW limit on
// grad:squeeze.weight at the fp32 single-track rows 217 / 219)
extern "C" __global__ void __launch_bounds__(256) tg_split_tf32(const float4* __restrict__ X, float4* __restrict__ Y, float4* __restrict__ L,
                                                               int n4) {
  PDL_ENTRY();
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x) {
    const float4 v = X[i];
    const float4 h = make_float4(tf32r(v.x), tf32r(v.y), tf32r(v.z), tf32r(v.w));
    Y[i] = h;
    L[i] = make_float4(tf32r(v.x - h.x), tf32r(v.y - h.y), tf32r(v.z - h.z), tf32r(v.w - h.w));
  }
}

// 32 columns per block, the rows dealt round robin over the NW = blockDim / 32 warps (row r to warp r % NW, each warp adding its rows in
// order), then the NW warp sums added in warp order: a fixed order, and R / NW loads in flight per thread instead of R in sequence. 256
// threads (8 warps) as before for every caller; the Transition backward launches 1024 (round 3 part 4: its ~600 partial rows, 16 blocks, took
// 8 us with 8 warps)
extern "C" __global__ void __launch_bounds__(1024) tg_colsum(const float* __restrict__ P, float* __restrict__ OUT, int R, int N) {
  __shared__ float red[32][33];
  PDL_ENTRY();
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, c = blockIdx.x * 32 + l, nw = blockDim.x >> 5;
  float s = 0.f;
  if (c < N) {
#pragma unroll 4
    for (int r = w; r < R; r += nw) s += P[(long)r * N + c];
  }
  red[w][l] = s;
  __syncthreads();
  if (w == 0 && c < N) {
    float t = red[0][l];
    for (int q = 1; q < nw; ++q) t += red[q][l];
    OUT[c] = t;
  }
}

// (round 3 part 5) the generic backward's last launch when the weight gradients ran split over the rows (tg_dw.cu, S slabs):
// blocks [0, nbw): sum over s = 0 .. S - 1 of WS[s][i] (16-B vectors, slab order) into O1 (the first N1
// vectors: dWs) / O2 (dWab); blocks [nbw, ...): tg_colsum of the LayerNorm backward's [R, NC] dgamma | dbeta partials (32
// columns per block, 8 warps over the rows, warps added in order). Vector i < N1 of slab s is WS[s SA + i], vector i >= N1 is
// WS2[s SB + i - N1] (16-B units): tg_dw's slabs are WS2 = WS + N1 vectors, SA = SB = N4; the cuBLAS slab products of the fp32
// backward two separate [S][...] buffers (round 3, 2026-11-17 night).
extern "C" __global__ void __launch_bounds__(256) tg_wsum(const float* __restrict__ WS, float* __restrict__ O1, float* __restrict__ O2, int S,
                                                         int N1, int N4, const float* __restrict__ P, float* __restrict__ OUTC, int R, int NC, int nbw,
                                                         const float* __restrict__ WS2, int SA, int SB) {
  __shared__ float red[8][33];
  PDL_ENTRY();
  if ((int)blockIdx.x < nbw) {
    const float4* w = reinterpret_cast<const float4*>(WS);
    const float4* w2 = reinterpret_cast<const float4*>(WS2) - N1;
    float4* o1 = reinterpret_cast<float4*>(O1);
    float4* o2 = reinterpret_cast<float4*>(O2) - N1;
    for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < N4; i += (long)nbw * 256) {
      const float4* src = i < N1 ? w + i : w2 + i;
      const long st = i < N1 ? SA : SB;
      float4 a = src[0];
      // the slabs in slab order, 8 loads in flight at a time (a runtime-length loop of dependent adds kept one in flight)
      for (int s0 = 1; s0 < S; s0 += 8) {
        float4 b[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
          if (s0 + k < S) b[k] = src[(long)(s0 + k) * st];
#pragma unroll
        for (int k = 0; k < 8; ++k)
          if (s0 + k < S) { a.x += b[k].x; a.y += b[k].y; a.z += b[k].z; a.w += b[k].w; }
      }
      (i < N1 ? o1 : o2)[i] = a;
    }
    return;
  }
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, c = ((int)blockIdx.x - nbw) * 32 + l;
  float s = 0.f;
  if (c < NC) {                                              // rows w, w + 8, ... in order, 16 loads in flight
    for (int r0 = w; r0 < R; r0 += 8 * 16) {
      float t[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) t[k] = r0 + 8 * k < R ? P[(long)(r0 + 8 * k) * NC + c] : 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k)
        if (r0 + 8 * k < R) s += t[k];
    }
  }
  red[w][l] = s;
  __syncthreads();
  if (w == 0 && c < NC) {
    float t = red[0][l];
    for (int q = 1; q < 8; ++q) t += red[q][l];
    OUTC[c] = t;
  }
}

// out [M, N] (T) = res + sum over the NS fp32 slabs P[s] (slab stride SST elements, row stride N), the slabs added in order first; 4 columns
// per thread (N a multiple of 4)
extern "C" __global__ void __launch_bounds__(256) tg_splitsum(const float* __restrict__ P, const T* __restrict__ RES, T* __restrict__ OUT, int M,
                                                             int N, int NS, int SST) {
  PDL_ENTRY();
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

#if F32
// ---------------------------------------------------------------------------------------------------------------------------------
// tg_prep16 (round 3 part 7): the fp32 TRAINING forward on the fp16 tensor cores needs, per step, the backward's TF32 copies (tg_prep's
// Ws^T, [Wa; Wb]^T, Wa, Wb; not Ws: the forward is fp16) AND the forward's fp16 weight forms with their device scales
// (fused_generic_sm100a._h16_forms: ~10 torch launches; the H100's f32_forms_stats + f32_forms_apply: two). Here ONE launch with
// tg_prep's grid (one block per 32 x 32 tile, no residency assumption):
//   every block  its tile's TF32 copies / transposes, the fp16 forms at scale 1 (Wa16 = fp16(Wa), ...), its tile's max |w| and (Wa, Wb tiles)
//                the largest sum of squares of one of its rows over its 32 columns (PART[t], a float4);
//   last block   (a ticket counter, BAR[0], back at 0 when it is done: the buffer is zeroed once per device) reduces those in a fixed
//                order -- max |Wa|, |Wb|, |Ws|, for the largest row L2 norm of Wa / Wb the bound
//                max over 32-row tiles of the sum over its column tiles of those maxima (rigorous, <= ~2x the norm; inference's
//                _h16_forms takes the exact norms) -- and the
//                gamma / beta bounds, and stores sc = [1 / s_x, s_x s_w, 1 / s_h, s_h s_ws] (_h16_forms' H16_BOUND_L2 scales, each
//                s = max(bound / 30000, 1)). Where s_w or s_ws is not 1 (weights above 30000: never for trained weights) it rewrites the
//                fp16 forms divided by their scale itself (one block: slow, but exact and deterministic).
extern "C" __global__ void __launch_bounds__(256) tg_prep16(const void* __restrict__ WA, const void* __restrict__ WB, const void* __restrict__ WS,
                                                           const float* __restrict__ G, const float* __restrict__ BT, float* __restrict__ OA,
                                                           float* __restrict__ OB, float* __restrict__ OST, float* __restrict__ OABT,
                                                           __half* __restrict__ A16, __half* __restrict__ B16, __half* __restrict__ S16,
                                                           float* __restrict__ SC, float* PART, unsigned* BAR, int H, int D,
                                                           int src_f32) {
  __shared__ float ta[32][33], tb[32][33];
  __shared__ float red[8][4];
  __shared__ unsigned ticket;
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int nab = (H / 32) * (D / 32), nt = 2 * nab, dc = D / 32;
  float m0 = 0.f, m1 = 0.f, n0 = 0.f, n1 = 0.f;            // max |w| of the tile; the largest 32-column row sum of squares (Wa, Wb)
  if ((int)blockIdx.x < nab) {                               // Wa, Wb tile: rows i0.. of H, columns j0.. of D
    const int i0 = (blockIdx.x / dc) * 32, j0 = (blockIdx.x % dc) * 32;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int r = w + 8 * q;
      const long e = (long)(i0 + r) * D + j0 + l;
      const float va = ldw(WA, e, src_f32), vb = ldw(WB, e, src_f32);
      const float a = tf32r(va), b = tf32r(vb);
      OA[e] = a; OB[e] = b;
      A16[e] = __float2half_rn(va); B16[e] = __float2half_rn(vb);
      ta[r][l] = a; tb[r][l] = b;
      m0 = fmaxf(m0, fabsf(va)); m1 = fmaxf(m1, fabsf(vb));
      n0 = fmaxf(n0, wsum(va * va)); n1 = fmaxf(n1, wsum(vb * vb));   // the row's squares over these 32 columns
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
      const float vs = ldw(WS, e, src_f32);
      S16[e] = __float2half_rn(vs);
      ta[r][l] = tf32r(vs);
      m0 = fmaxf(m0, fabsf(vs));
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < 4; ++q) {                            // Ws^T [H, D]: row t0 + r, columns s0..
      const int r = w + 8 * q;
      OST[(long)(t0 + r) * D + s0 + l] = ta[l][r];
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) { m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o)); }
  if (l == 0) { red[w][0] = m0; red[w][1] = m1; red[w][2] = n0; red[w][3] = n1; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i < 8; ++i)
      for (int k = 0; k < 4; ++k) v[k] = fmaxf(v[k], red[i][k]);
    reinterpret_cast<float4*>(PART)[blockIdx.x] = make_float4(v[0], v[1], v[2], v[3]);
    __threadfence();
    ticket = atomicAdd(BAR, 1u);
  }
  __syncthreads();
  if (ticket != (unsigned)nt - 1) return;
  // ------------------------------------------------------------------ the last block: the scales
  __threadfence();
  float pa = 0.f, pb = 0.f, ps = 0.f, na = 0.f, nb = 0.f;
  const float4* P4 = reinterpret_cast<const float4*>(PART);
#pragma unroll 4
  for (int t = threadIdx.x; t < nt; t += blockDim.x) {
    const float4 p = __ldcg(P4 + t);
    if (t < nab) { pa = fmaxf(pa, p.x); pb = fmaxf(pb, p.y); } else ps = fmaxf(ps, p.x);
  }
  // row L2 norm bound per 32-row tile i: sum over its column tiles c (in c order: the butterfly) of the tile's largest 32-column row
  // sum of squares -- at least every row's squared norm (rigorous, within ~2x of the largest; the exact per-row sums would need
  // a second pass over [2H][D / 32] partials in this one block)
  const int nri = H / 32;
  for (int i0 = w; i0 < nri; i0 += 32) {
    float qa[4], qb[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = i0 + 8 * k;
      const float4 p = (i < nri && l < dc) ? __ldcg(P4 + i * dc + l) : make_float4(0.f, 0.f, 0.f, 0.f);
      qa[k] = p.z; qb[k] = p.w;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) { na = fmaxf(na, wsum(qa[k])); nb = fmaxf(nb, wsum(qb[k])); }
  }
  float mg = 0.f, mbt = 0.f, sb = 0.f;
  for (int c = threadIdx.x; c < D; c += blockDim.x) { const float bt = BT[c]; mg = fmaxf(mg, fabsf(G[c])); mbt = fmaxf(mbt, fabsf(bt)); sb = fmaf(bt, bt, sb); }
  __shared__ float sbs[256];
  sbs[threadIdx.x] = sb;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    pa = fmaxf(pa, __shfl_xor_sync(0xffffffffu, pa, o)); pb = fmaxf(pb, __shfl_xor_sync(0xffffffffu, pb, o));
    ps = fmaxf(ps, __shfl_xor_sync(0xffffffffu, ps, o)); na = fmaxf(na, __shfl_xor_sync(0xffffffffu, na, o));
    nb = fmaxf(nb, __shfl_xor_sync(0xffffffffu, nb, o)); mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, o));
    mbt = fmaxf(mbt, __shfl_xor_sync(0xffffffffu, mbt, o));
  }
  __shared__ float r7[8][7];
  __shared__ float scs[2];
  if (l == 0) { r7[w][0] = pa; r7[w][1] = pb; r7[w][2] = ps; r7[w][3] = na; r7[w][4] = nb; r7[w][5] = mg; r7[w][6] = mbt; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float v[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i < 8; ++i)
      for (int k = 0; k < 7; ++k) v[k] = fmaxf(v[k], r7[i][k]);
    float sb2 = 0.f;
    for (int i = 0; i < 256; ++i) sb2 += sbs[i];             // ||beta||^2: the threads' partials in thread order
    const float bound = 30000.f, sd = sqrtf((float)D);
    const float s_x = fmaxf((sd * v[5] + v[6]) / bound, 1.f);
    const float s_w = fmaxf(fmaxf(v[0], v[1]) / bound, 1.f);
    const float x2 = sd * v[5] + sqrtf(sb2);
    const float s_h = fmaxf(x2 * sqrtf(v[3]) * x2 * sqrtf(v[4]) / bound, 1.f);
    const float s_ws = fmaxf(v[2] / bound, 1.f);
    SC[0] = 1.f / s_x; SC[1] = s_x * s_w; SC[2] = 1.f / s_h; SC[3] = s_h * s_ws;
    scs[0] = s_w; scs[1] = s_ws;
    BAR[0] = 0u;                                             // the next launch's tickets start at 0
  }
  __syncthreads();
  const float s_w = scs[0], s_ws = scs[1];
  if (s_w != 1.f) {
    const long n = (long)H * D;
    for (long e = threadIdx.x; e < n; e += blockDim.x) {
      A16[e] = __float2half_rn(__fdiv_rn(ldw(WA, e, src_f32), s_w)); B16[e] = __float2half_rn(__fdiv_rn(ldw(WB, e, src_f32), s_w));
    }
  }
  if (s_ws != 1.f) {
    const long n = (long)H * D;
    for (long e = threadIdx.x; e < n; e += blockDim.x) S16[e] = __float2half_rn(__fdiv_rn(ldw(WS, e, src_f32), s_ws));
  }
}
#endif
