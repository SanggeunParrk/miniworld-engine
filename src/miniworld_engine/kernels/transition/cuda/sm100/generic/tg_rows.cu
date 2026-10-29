// tg_rows.cu -- the row passes of the generic sm_100a Transition (around the tcgen05 GEMMs of tg_gemm.cu), -DF32=1 for fp32 activations:
//
//   tg_ln_fwd_*   xn = LN(x) gamma + beta (bf16, or fp32 rounded to the TF32 grid: the expand GEMM's operand), and the row's mean / rstd
//   tg_ln_bwd_*   dx = dy + LN backward of d_xn (fp32) through gamma; per-block partial sums of d_xn xhat (dgamma) and d_xn (dbeta)
//   tg_colsum     column sums of [P, N] fp32 partials in a fixed order (dgamma | dbeta; the ConditionedTransition's gate bias)
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
template <int VEC, int TPR, int NV>
__device__ __forceinline__ void ln_bwd(const float* __restrict__ DXN, const T* __restrict__ X, const T* __restrict__ DY,
                                       const float* __restrict__ MEAN, const float* __restrict__ RSTD, const float* __restrict__ G,
                                       T* __restrict__ DX, float* __restrict__ PART, int M, int D, float* red) {
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
  // the block's partials: the RPW row slots of a warp, then the warps, added in order
  for (int c = threadIdx.x; c < 2 * D; c += blockDim.x) red[c] = 0.f;
  for (int w = 0; w < nwb; ++w)
    for (int q = 0; q < RPW; ++q) {
      __syncthreads();
      if (warp == w && sub == q) {
#pragma unroll
        for (int i = 0; i < NV / VEC; ++i)
#pragma unroll
          for (int e = 0; e < VEC; ++e) {
            const int c = VEC * (TPR * i + l) + e;
            red[c] += dg[VEC * i + e];
            red[D + c] += db[VEC * i + e];
          }
      }
    }
  __syncthreads();
  for (int c = threadIdx.x; c < 2 * D; c += blockDim.x) PART[(long)blockIdx.x * 2 * D + c] = red[c];
}

// the instantiations the host picks from: tg_ln_{fwd,bwd}_v<VEC>t<TPR>n<NV> (NV = D / TPR, at most 32)
#define LN_INST(VEC, TPR, NV)                                                                                                             \
  extern "C" __global__ void __launch_bounds__(256) tg_ln_fwd_v##VEC##t##TPR##n##NV(                                                     \
      const T* X, const float* G, const float* Bt, T* XN, float* MEAN, float* RSTD, int M, int D, float eps, int save) {                 \
    ln_fwd<VEC, TPR, NV>(X, G, Bt, XN, MEAN, RSTD, M, D, eps, save);                                                                     \
  }                                                                                                                                       \
  extern "C" __global__ void __launch_bounds__(256) tg_ln_bwd_v##VEC##t##TPR##n##NV(                                                     \
      const float* DXN, const T* X, const T* DY, const float* MEAN, const float* RSTD, const float* G, T* DX, float* PART, int M, int D) { \
    __shared__ float red[2 * 1024];                                                                                                       \
    ln_bwd<VEC, TPR, NV>(DXN, X, DY, MEAN, RSTD, G, DX, PART, M, D, red);                                                                \
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

extern "C" __global__ void __launch_bounds__(256) tg_colsum(const float* __restrict__ P, float* __restrict__ OUT, int R, int N) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= N) return;
  float s = 0.f;
  for (int r = 0; r < R; ++r) s += P[(long)r * N + c];
  OUT[c] = s;
}

// Wa, Wb [H, D] and Ws [D, H] (fp32 when src_f32, else bf16) -> wa, wb, ws in T (fp32: TF32 grid); with `bwd` also Ws^T [H, D] and
// [Wa; Wb]^T [D, 2H]. One thread per (Wa element, Ws element) pair: H D threads.
extern "C" __global__ void __launch_bounds__(256) tg_prep(const void* __restrict__ WA, const void* __restrict__ WB, const void* __restrict__ WS,
                                                         T* __restrict__ OA, T* __restrict__ OB, T* __restrict__ OS, T* __restrict__ OST,
                                                         T* __restrict__ OABT, int H, int D, int src_f32, int bwd) {
  const long n = (long)H * D;
  for (long e = (long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long)gridDim.x * blockDim.x) {
    float a, b, s;
    if (src_f32) {
      a = reinterpret_cast<const float*>(WA)[e]; b = reinterpret_cast<const float*>(WB)[e]; s = reinterpret_cast<const float*>(WS)[e];
    } else {
      a = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(WA)[e]);
      b = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(WB)[e]);
      s = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(WS)[e]);
    }
#if F32
    a = tf32r(a); b = tf32r(b); s = tf32r(s);
    const T ta = a, tb = b, ts = s;
#else
    const T ta = __float2bfloat16_rn(a), tb = __float2bfloat16_rn(b), ts = __float2bfloat16_rn(s);
#endif
    OA[e] = ta; OB[e] = tb; OS[e] = ts;
    if (bwd) {
      const long i = e / D, j = e % D;                       // Wa[i][j] -> [Wa; Wb]^T[j][i], [j][H + i]
      OABT[j * 2 * H + i] = ta;
      OABT[j * 2 * H + H + i] = tb;
      const long si = e / H, sj = e % H;                     // Ws[si][sj] -> Ws^T[sj][si]
      OST[sj * D + si] = ts;
    }
  }
}
