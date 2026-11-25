// lattn_dq.cu — dQ of the AF3 windowed atom attention (window-local: the 32 queries of a window own their dq entirely), sm_100a.
//
//   P = exp2(S - LSE)    dP = dO V^T    dS = P (dP - D)    dQ = dS K / sqrt(32)     (bf16 into dq[row, h * 32 + d], row stride ldd)
//
// dbias partials: dS summed over the samples a CTA visits (the two sample slots add through shared memory at the end) -> dbp[z][h][w][32][128] fp32,
// the sum over z is the caller's (local_bias_bwd). Same CTA structure as lattn_fwd.cu: (window, head, sample chunk), 4 warps = 2 samples x 2 query tiles, the window's bias (log2 units, key mask folded in)
// in shared memory, one stage of 2 samples (q | K | V | dO), per-thread precomputed cp.async offsets; 58 KB: two CTAs per SM.
// SPDX-License-Identifier: Apache-2.0
#include "local.cuh"
#ifndef ATT_PF
#define ATT_PF 1
#endif

constexpr int BS = 136;
constexpr int TQB = WQ * 64, TKB = WK * 64;
constexpr int SLOT = 2 * TQB + 2 * TKB;                           // q | K | V | dO
constexpr int O_BIAS = 0, O_ST = WQ * BS * 4;
constexpr int SMEM_BYTES = O_ST + 2 * SLOT;

extern "C" __global__ void __launch_bounds__(128, 2)
local_attn_dq(const __nv_bfloat16* __restrict__ gq, const __nv_bfloat16* __restrict__ gk, const __nv_bfloat16* __restrict__ gv,
              const __nv_bfloat16* __restrict__ gdo, const float* __restrict__ gbias, const uint8_t* __restrict__ kmask,
              const float* __restrict__ LSE, const float* __restrict__ Dd, __nv_bfloat16* __restrict__ dq, float* __restrict__ dbp, int ldd,
              int N, int A, int nwin, int psplit) {
  extern __shared__ __align__(1024) uint8_t sm[];
  float* sbias = reinterpret_cast<float*>(sm + O_BIAS);
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int w = blockIdx.x, h = blockIdx.y, z = blockIdx.z;
  const int npair = (A + 1) >> 1;
  const int kbase = WQ * w - KOFF;
  {
    const float* src = gbias + ((size_t)h * nwin + w) * (WQ * WK);
    for (int i = tid; i < WQ * WK / 4; i += 128) {
      const int r = i >> 5, c = (i & 31) * 4;
      float4 b = *reinterpret_cast<const float4*>(src + i * 4);
      float ka[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = kbase + c + e;
        ka[e] = (key >= 0 && key < N && (kmask == nullptr || kmask[key])) ? 0.f : -INFINITY;
      }
      b.x = fmaf(b.x, LOG2E, ka[0]); b.y = fmaf(b.y, LOG2E, ka[1]); b.z = fmaf(b.z, LOG2E, ka[2]); b.w = fmaf(b.w, LOG2E, ka[3]);
      *reinterpret_cast<float4*>(sbias + r * BS + c) = b;
    }
  }
  // copy mapping: thread -> (row tid / 4 + 32 k, 16-byte chunk tid % 4)
  const int crow = tid >> 2, cch = tid & 3;
  const uint32_t cswz = crow * 64 + ((cch ^ ((crow >> 1) & 3)) << 4);
  const bool qok = WQ * w + crow < N;
  const int qoff = (qok ? WQ * w + crow : 0) * DM + h * DH + cch * 8;
  int koff[4];
  uint32_t kvalid = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int key = kbase + crow + 32 * k;
    const bool ok = key >= 0 && key < N;
    kvalid |= (ok ? 1u : 0u) << k;
    koff[k] = (ok ? key : 0) * DM + h * DH + cch * 8;
  }
  auto issue = [&](int p) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      const int a = 2 * p + slot;
      const bool ok_a = p < npair && a < A;
      const size_t sbase = (size_t)(ok_a ? a : 0) * N * DM;
      const uint32_t st = su + O_ST + slot * SLOT + cswz;
      cp_async16(st, gq + sbase + qoff, ok_a && qok);
      cp_async16(st + TQB + 2 * TKB, gdo + sbase + qoff, ok_a && qok);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const bool ok = ok_a && ((kvalid >> k) & 1u);
        cp_async16(st + TQB + 32 * k * 64, gk + sbase + koff[k], ok);
        cp_async16(st + TQB + TKB + 32 * k * 64, gv + sbase + koff[k], ok);
      }
    }
    cp_commit();
  };
  const int mt = warp & 1, sl = warp >> 1;
  const int r0 = lane >> 2, c0 = 2 * (lane & 3);
  float dbacc[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) dbacc[i][0] = dbacc[i][1] = dbacc[i][2] = dbacc[i][3] = 0.f;
  for (int p = z; p < npair; p += psplit) {
    issue(p);
#if ATT_PF
    if (cch == 0 && p + psplit < npair) {                           // the next sample pair's rows into L2
#pragma unroll
      for (int slot = 0; slot < 2; ++slot) {
        const int a = 2 * (p + psplit) + slot;
        if (a < A) {
          const size_t sb = (size_t)a * N * DM;
          if (qok) { prefetch_l2(gq + sb + qoff); prefetch_l2(gdo + sb + qoff); }
#pragma unroll
          for (int k = 0; k < 4; ++k)
            if ((kvalid >> k) & 1u) { prefetch_l2(gk + sb + koff[k]); prefetch_l2(gv + sb + koff[k]); }
        }
      }
    }
#endif
    cp_wait<0>();
    __syncthreads();
    const int a = 2 * p + sl;
    if (a < A) {
      const uint32_t sbase = su + O_ST + sl * SLOT, tq = sbase, tk = sbase + TQB, tv = tk + TKB, td = tv + TKB;
      const int q0 = WQ * w + 16 * mt + r0, q1 = q0 + 8;
      const float lse0 = q0 < N ? LSE[((size_t)a * NH + h) * N + q0] : 1e30f, lse1 = q1 < N ? LSE[((size_t)a * NH + h) * N + q1] : 1e30f;
      const float d0 = q0 < N ? Dd[((size_t)a * NH + h) * N + q0] : 0.f, d1 = q1 < N ? Dd[((size_t)a * NH + h) * N + q1] : 0.f;
      uint32_t qa[2][4], da[2][4];
      load_a(qa, tq, 16 * mt, lane);
      load_a(da, td, 16 * mt, lane);
      float s[16][4], dp[16][4];
#pragma unroll
      for (int i = 0; i < 16; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = dp[i][0] = dp[i][1] = dp[i][2] = dp[i][3] = 0.f;
#pragma unroll
      for (int np = 0; np < 8; ++np)
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          uint32_t b0, b1, b2, b3;
          load_b_rows(b0, b1, b2, b3, tk, 16 * np, ks, lane);
          mma16816(s[2 * np], qa[ks], b0, b1);
          mma16816(s[2 * np + 1], qa[ks], b2, b3);
          load_b_rows(b0, b1, b2, b3, tv, 16 * np, ks, lane);
          mma16816(dp[2 * np], da[ks], b0, b1);
          mma16816(dp[2 * np + 1], da[ks], b2, b3);
        }
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const int c = 8 * nt + c0;
        const float2 b_lo = *reinterpret_cast<const float2*>(sbias + (16 * mt + r0) * BS + c);
        const float2 b_hi = *reinterpret_cast<const float2*>(sbias + (16 * mt + r0 + 8) * BS + c);
        const float p0 = ex2f(fmaf(s[nt][0], QSCALE * LOG2E, b_lo.x) - lse0);
        const float p1 = ex2f(fmaf(s[nt][1], QSCALE * LOG2E, b_lo.y) - lse0);
        const float p2 = ex2f(fmaf(s[nt][2], QSCALE * LOG2E, b_hi.x) - lse1);
        const float p3 = ex2f(fmaf(s[nt][3], QSCALE * LOG2E, b_hi.y) - lse1);
        s[nt][0] = p0 * (dp[nt][0] - d0);
        s[nt][1] = p1 * (dp[nt][1] - d0);
        s[nt][2] = p2 * (dp[nt][2] - d1);
        s[nt][3] = p3 * (dp[nt][3] - d1);
        dbacc[nt][0] += s[nt][0]; dbacc[nt][1] += s[nt][1]; dbacc[nt][2] += s[nt][2]; dbacc[nt][3] += s[nt][3];
      }
      float acc[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t sa[4] = {pack_bf16(s[2 * ks][0], s[2 * ks][1]), pack_bf16(s[2 * ks][2], s[2 * ks][3]),
                          pack_bf16(s[2 * ks + 1][0], s[2 * ks + 1][1]), pack_bf16(s[2 * ks + 1][2], s[2 * ks + 1][3])};
#pragma unroll
        for (int dpair = 0; dpair < 2; ++dpair) {
          uint32_t b0, b1, b2, b3;
          load_b_cols(b0, b1, b2, b3, tk, 16 * ks, dpair, lane);
          mma16816(acc[2 * dpair], sa, b0, b1);
          mma16816(acc[2 * dpair + 1], sa, b2, b3);
        }
      }
#pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        if (q0 < N) *reinterpret_cast<uint32_t*>(dq + ((size_t)a * N + q0) * ldd + h * DH + 8 * dt + c0) = pack_bf16(acc[dt][0] * QSCALE, acc[dt][1] * QSCALE);
        if (q1 < N) *reinterpret_cast<uint32_t*>(dq + ((size_t)a * N + q1) * ldd + h * DH + 8 * dt + c0) = pack_bf16(acc[dt][2] * QSCALE, acc[dt][3] * QSCALE);
      }
    }
    __syncthreads();
  }
  // dbias partial of this CTA's samples: slot 1 hands its sums to slot 0 through the (free) stage memory, slot 0 adds and stores
  float* xch = reinterpret_cast<float*>(sm + O_ST);
  if (sl == 1) {
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) xch[((mt * 16 + nt) * 4 + e) * 32 + lane] = dbacc[nt][e];
  }
  __syncthreads();
  if (sl == 0) {
    float* dst = dbp + ((((size_t)z * NH + h) * nwin + w) * WQ + 16 * mt + r0) * WK + c0;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) v[e] = dbacc[nt][e] + xch[((mt * 16 + nt) * 4 + e) * 32 + lane];
      *reinterpret_cast<float2*>(dst + 8 * nt) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(dst + 8 * WK + 8 * nt) = make_float2(v[2], v[3]);
    }
  }
}
