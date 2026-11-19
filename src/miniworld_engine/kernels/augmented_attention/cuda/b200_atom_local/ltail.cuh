// ltail.cuh — the TAIL of AlphaFold 3's slid key window (``LocalDiTBlock(key_window="slide" / "slide_qblock")``) on the B200 (2026-11-17),
// run by the group-1 edge CTAs of the attention launches themselves (slide.cuh), after their own windows: no launch of its own.
//
// The tail: every query window w >= t0 = slide_tail0(n_real) (t0 = max(2, last - 2) + 4, last = the last window holding a real atom). Its
// queries are padding atoms only, and the reference gives each of them the slid keys [kb, kb + 128), kb = max(n_real - 128, 0) (the keys
// of the last real windows). The tcgen05 part of every attention kernel skips those windows (slide_moved: their O / LSE / dQ rows are not
// written, their cells masked in dK / dV / dbias); an edge CTA of group 1 (one head, a sample range) computes them on CUDA cores in fp32:
//   tail_fwd   O and LSE (log2 units) of the tail rows (one rounding of O)
//   tail_dq    dQ of the tail rows
//   tail_dkv   dbias of the tail windows (the CTA's samples summed in order, then added into dbias: at most two sample ranges per head, onto
//              the zeroed tensor, so 0 + a + b in either order -- exact and deterministic), and the slid keys' dK / dV from every tail
//              window, added into the CTA's own rows of group 1's fp32 partial (spart, which lslide.cu then adds).
// Without a tail window (every requirement case: R = N - 24 lies in the last window) nothing runs. A block of any size (a multiple of 32,
// >= 256 threads); ``scr``: >= 2 x 128 x 33 floats of shared memory nobody else uses any more. Element types: T = bf16 or float. q, k, v
// share the row / sample strides (sN, sA, elements), dO has (dN, dA); O is contiguous [A, N, 128].
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace ltail {
constexpr int NH = 4, DH = 32, WQ = 32, WK = 128, KS = DH + 1;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;

__device__ __forceinline__ float ldf(const float* p) { return *p; }
__device__ __forceinline__ float ldf(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void stf(float* p, float v) { *p = v; }
__device__ __forceinline__ void stf(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

__device__ __forceinline__ bool key_ok(const uint8_t* km, int key, int N) { return key < N && (km == nullptr || km[key]); }

// K / V tiles of the slid keys of (row sample a, head h) into scr [2][128][33] fp32 (zero past N)
template <typename T>
__device__ __forceinline__ void load_kv(const T* __restrict__ k, const T* __restrict__ v, long sN, long sA, int a, int h, int kb, int N,
                                        float* scr) {
  for (int e = threadIdx.x; e < WK * DH; e += blockDim.x) {
    const int j = e >> 5, d = e & 31, key = kb + j;
    float x = 0.f, y = 0.f;
    if (key < N) {
      const long o = (long)a * sA + (long)key * sN + h * DH + d;
      x = ldf(k + o);
      y = ldf(v + o);
    }
    scr[j * KS + d] = x;
    scr[WK * KS + j * KS + d] = y;
  }
}

// scores of query row (lane = channel d holds q[d]) against the 128 slid keys: s[c] for key lane + 32 c, in log2 units (-inf off the valid)
__device__ __forceinline__ void scores(float qd, const float* sk, const float* brow, const bool (&ok)[4], float (&s)[4]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int c = 0; c < 4; ++c) s[c] = 0.f;
#pragma unroll 8
  for (int d = 0; d < DH; ++d) {
    const float qq = __shfl_sync(0xffffffffu, qd, d);
#pragma unroll
    for (int c = 0; c < 4; ++c) s[c] = fmaf(qq, sk[(lane + 32 * c) * KS + d], s[c]);
  }
#pragma unroll
  for (int c = 0; c < 4; ++c) s[c] = ok[c] ? fmaf(s[c], QSCALE * LOG2E, brow[lane + 32 * c] * LOG2E) : -INFINITY;
}

// O / LSE of the tail rows of head h, samples [a_lo, a_hi) (row sample a NF + f; the bias of fold f)
template <typename T>
__device__ __noinline__ void tail_fwd(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, T* __restrict__ o,
                                      float* __restrict__ lse, const float* __restrict__ gbias, const uint8_t* __restrict__ km, int nr,
                                      int N, int nwin, int NF, int f, int h, int a_lo, int a_hi, long sN, long sA, float* scr) {
  const int t0 = slide_tail0(nr, nwin);
  if (t0 >= nwin || a_lo >= a_hi) return;
  const int kb = max(nr - WK, 0), warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwp = blockDim.x >> 5;
  bool ok[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) ok[c] = key_ok(km, kb + lane + 32 * c, N);
  for (int a0 = a_lo; a0 < a_hi; ++a0) {
    const int a = a0 * NF + f;
    __syncthreads();
    load_kv(k, v, sN, sA, a, h, kb, N, scr);
    __syncthreads();
    for (int r = warp; r < (nwin - t0) * WQ; r += nwp) {
      const int w = t0 + r / WQ, i = r % WQ, row = 32 * w + i;
      float s[4];
      scores(ldf(q + (long)a * sA + (long)row * sN + h * DH + lane), scr,
             gbias + (((size_t)(h * NF + f) * nwin + w) * WQ + i) * WK, ok, s);
      float m = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3]));
#pragma unroll
      for (int x = 16; x; x >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, x));
      const float mm = m == -INFINITY ? 0.f : m;
      float l = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) { s[c] = exp2f(s[c] - mm); l += s[c]; }
#pragma unroll
      for (int x = 16; x; x >>= 1) l += __shfl_xor_sync(0xffffffffu, l, x);
      float acc = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
#pragma unroll 8
        for (int jj = 0; jj < 32; ++jj) acc = fmaf(__shfl_sync(0xffffffffu, s[c], jj), scr[WK * KS + (32 * c + jj) * KS + lane], acc);
      }
      stf(o + ((size_t)a * N + row) * (NH * DH) + h * DH + lane, l > 0.f ? acc / l : 0.f);
      if (lane == 0) lse[((size_t)a * NH + h) * N + row] = l > 0.f ? mm + __log2f(l) : 1e30f;
    }
  }
}

// dS of tail row (w, i) of sample a, head h against the slid keys (P = exp2(s - LSE), dP = dO V^T): ds[c] for key lane + 32 c
template <typename T>
__device__ __forceinline__ void tail_ds(const T* __restrict__ q, const T* __restrict__ dO, const float* __restrict__ lse,
                                        const float* __restrict__ dd, const float* __restrict__ gbias, const bool (&ok)[4], int N, int nwin,
                                        int h, int a, int w, int i, long sN, long sA, long dN, long dA, const float* scr, float (&ds)[4]) {
  const int lane = threadIdx.x & 31, row = 32 * w + i;
  float s[4], dp[4];
  scores(ldf(q + (long)a * sA + (long)row * sN + h * DH + lane), scr, gbias + (((size_t)h * nwin + w) * WQ + i) * WK, ok, s);
  const float od = ldf(dO + (long)a * dA + (long)row * dN + h * DH + lane);
#pragma unroll
  for (int c = 0; c < 4; ++c) dp[c] = 0.f;
#pragma unroll 8
  for (int d = 0; d < DH; ++d) {
    const float oo = __shfl_sync(0xffffffffu, od, d);
#pragma unroll
    for (int c = 0; c < 4; ++c) dp[c] = fmaf(oo, scr[WK * KS + (lane + 32 * c) * KS + d], dp[c]);
  }
  const float L = lse[((size_t)a * NH + h) * N + row], D = dd[((size_t)a * NH + h) * N + row];
#pragma unroll
  for (int c = 0; c < 4; ++c) ds[c] = ok[c] ? exp2f(s[c] - L) * (dp[c] - D) : 0.f;
}

// dQ of the tail rows of head h, samples [a_lo, a_hi): dq[row, h 32 + d] (rows ldq elements apart)
template <typename T>
__device__ __noinline__ void tail_dq(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, const T* __restrict__ dO,
                                     const float* __restrict__ lse, const float* __restrict__ dd, const float* __restrict__ gbias,
                                     const uint8_t* __restrict__ km, T* __restrict__ dq, int nr, int N, int nwin, int h, int a_lo, int a_hi,
                                     long sN, long sA, long dN, long dA, int ldq, float* scr) {
  const int t0 = slide_tail0(nr, nwin);
  if (t0 >= nwin || a_lo >= a_hi) return;
  const int kb = max(nr - WK, 0), warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwp = blockDim.x >> 5;
  bool ok[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) ok[c] = key_ok(km, kb + lane + 32 * c, N);
  for (int a = a_lo; a < a_hi; ++a) {
    __syncthreads();
    load_kv(k, v, sN, sA, a, h, kb, N, scr);
    __syncthreads();
    for (int r = warp; r < (nwin - t0) * WQ; r += nwp) {
      const int w = t0 + r / WQ, i = r % WQ;
      float ds[4];
      tail_ds(q, dO, lse, dd, gbias, ok, N, nwin, h, a, w, i, sN, sA, dN, dA, scr, ds);
      float acc = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
#pragma unroll 8
        for (int jj = 0; jj < 32; ++jj) acc = fmaf(__shfl_sync(0xffffffffu, ds[c], jj), scr[(32 * c + jj) * KS + lane], acc);
      }
      stf(dq + ((size_t)a * N + 32 * w + i) * ldq + h * DH + lane, acc * QSCALE);
    }
  }
}

// dbias of the tail windows (head h, this CTA's samples [a_lo, a_hi), added into dbias) and the slid keys' dK / dV from the tail windows,
// added into group 1's partial spart [2][A][4][dK 32 | dV 32][128] at the CTA's samples (rows its own drain wrote: call after a __syncthreads)
template <typename T>
__device__ __noinline__ void tail_dkv(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, const T* __restrict__ dO,
                                      const float* __restrict__ lse, const float* __restrict__ dd, const float* __restrict__ gbias,
                                      const uint8_t* __restrict__ km, float* __restrict__ dbias, float* __restrict__ spart, int nr, int N,
                                      int A, int nwin, int h, int a_lo, int a_hi, long sN, long sA, long dN, long dA, float* scr) {
  const int t0 = slide_tail0(nr, nwin);
  if (t0 >= nwin || a_lo >= a_hi) return;
  const int kb = max(nr - WK, 0), warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwp = blockDim.x >> 5, tid = threadIdx.x;
  {                                                                   // (1) dbias: a warp per (window, query) row, the samples in order
    bool ok[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) ok[c] = key_ok(km, kb + lane + 32 * c, N);
    const int rows = (nwin - t0) * WQ;
    for (int r0 = 0; r0 < rows; r0 += nwp) {                          // a pass: one row per warp
      const int r = r0 + warp, w = t0 + r / WQ, i = r % WQ;
      float db[4] = {0.f, 0.f, 0.f, 0.f};
      for (int a = a_lo; a < a_hi; ++a) {
        __syncthreads();
        load_kv(k, v, sN, sA, a, h, kb, N, scr);
        __syncthreads();
        if (r < rows) {
          float ds[4];
          tail_ds(q, dO, lse, dd, gbias, ok, N, nwin, h, a, w, i, sN, sA, dN, dA, scr, ds);
#pragma unroll
          for (int c = 0; c < 4; ++c) db[c] += ds[c];
        }
      }
      if (r < rows) {
#pragma unroll
        for (int c = 0; c < 4; ++c) atomicAdd(dbias + (((size_t)h * nwin + w) * WQ + i) * WK + lane + 32 * c, db[c]);
      }
    }
  }
  // (2) dK / dV: thread (key j, half) for tid < 256 (half 0: dK, 1: dV; both form P / dS of their key); q / dO rows of one window in scr
  const bool act = tid < 2 * WK;
  const int j = tid & (WK - 1), half = (tid >> 7) & 1, key = kb + j;
  const bool ok = act && key_ok(km, key, N);
  float* sq = scr;                                                    // [32][33] q, [32][33] dO, [32] LSE, [32] D
  float* so = scr + WQ * KS;
  float* sl = scr + 2 * WQ * KS;
  for (int a = a_lo; a < a_hi; ++a) {
    float kr[DH], vr[DH], acc[DH];
#pragma unroll
    for (int d = 0; d < DH; ++d) {
      const long off = (long)a * sA + (long)min(key, N - 1) * sN + h * DH + d;
      kr[d] = ok ? ldf(k + off) : 0.f;
      vr[d] = ok ? ldf(v + off) : 0.f;
      acc[d] = 0.f;
    }
    for (int w = t0; w < nwin; ++w) {
      __syncthreads();
      for (int e = tid; e < WQ * DH; e += blockDim.x) {
        const int i = e >> 5, d = e & 31, row = 32 * w + i;
        sq[i * KS + d] = ldf(q + (long)a * sA + (long)row * sN + h * DH + d);
        so[i * KS + d] = ldf(dO + (long)a * dA + (long)row * dN + h * DH + d);
      }
      if (tid < 2 * WQ) sl[tid] = (tid < WQ ? lse : dd)[((size_t)a * NH + h) * N + 32 * w + (tid & 31)];
      __syncthreads();
      if (!ok) continue;
      const float* bw = gbias + ((size_t)h * nwin + w) * WQ * WK + j;
      for (int i = 0; i < WQ; ++i) {
        float s = 0.f, dp = 0.f;
#pragma unroll
        for (int d = 0; d < DH; ++d) {
          s = fmaf(sq[i * KS + d], kr[d], s);
          dp = fmaf(so[i * KS + d], vr[d], dp);
        }
        const float pr = exp2f(fmaf(s, QSCALE * LOG2E, bw[i * WK] * LOG2E) - sl[i]);
        const float x = half == 0 ? pr * (dp - sl[WQ + i]) : pr;     // dS (dK += dS q) or P (dV += P dO)
        const float* src = half == 0 ? sq + i * KS : so + i * KS;
#pragma unroll
        for (int d = 0; d < DH; ++d) acc[d] = fmaf(x, src[d], acc[d]);
      }
    }
    if (ok) {
      float* dst = spart + ((((size_t)A + a) * NH + h) * 64 + 32 * half) * WK + j;
      const float sc = half == 0 ? QSCALE : 1.f;
#pragma unroll
      for (int d = 0; d < DH; ++d) dst[(size_t)d * WK] += acc[d] * sc;
    }
  }
}
}  // namespace ltail
