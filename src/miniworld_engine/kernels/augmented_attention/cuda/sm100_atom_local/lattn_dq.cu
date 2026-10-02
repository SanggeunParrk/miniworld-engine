// lattn_dq.cu — dQ of the AF3 windowed atom attention (window-local: the 32 queries of a window own their dq entirely), sm_100a.
//
//   P = exp2(S - LSE)    dP = dO V^T    dS = P (dP - D)    dQ = dS K / sqrt(32)     (bf16 into dq[row, h * 32 + d], row stride ldd)
//
// Same CTA structure as lattn_fwd.cu: (window, head, sample chunk), 4 warps = 2 samples x 2 query tiles, the window's bias in shared memory.
// SPDX-License-Identifier: Apache-2.0
#include "local.cuh"

constexpr int BS = 136;
constexpr int TQB = WQ * 64, TKB = WK * 64;
constexpr int SLOT = 2 * TQB + 2 * TKB;                           // q | K | V | dO
constexpr int O_BIAS = 0, O_KADD = WQ * BS * 4, O_ST = O_KADD + WK * 4;
constexpr int SMEM_BYTES = O_ST + 2 * 2 * SLOT;

extern "C" __global__ void __launch_bounds__(128)
local_attn_dq(const __nv_bfloat16* __restrict__ gq, const __nv_bfloat16* __restrict__ gk, const __nv_bfloat16* __restrict__ gv,
              const __nv_bfloat16* __restrict__ gdo, const float* __restrict__ gbias, const uint8_t* __restrict__ kmask,
              const float* __restrict__ LSE, const float* __restrict__ Dd, __nv_bfloat16* __restrict__ dq, int ldd,
              int N, int A, int nwin, int psplit) {
  extern __shared__ __align__(1024) uint8_t sm[];
  float* sbias = reinterpret_cast<float*>(sm + O_BIAS);
  float* kadd = reinterpret_cast<float*>(sm + O_KADD);
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int w = blockIdx.x, h = blockIdx.y, z = blockIdx.z;
  const int npair = (A + 1) >> 1;
  const int kbase = WQ * w - KOFF;
  {
    const float* src = gbias + ((size_t)h * nwin + w) * (WQ * WK);
    for (int i = tid; i < WQ * WK / 4; i += 128) {
      const int r = i >> 5, c = (i & 31) * 4;
      *reinterpret_cast<float4*>(sbias + r * BS + c) = *reinterpret_cast<const float4*>(src + i * 4);
    }
    for (int j = tid; j < WK; j += 128) {
      const int key = kbase + j;
      kadd[j] = (key >= 0 && key < N && (kmask == nullptr || kmask[key])) ? 0.f : -INFINITY;
    }
  }
  constexpr int PER = 2 * WQ * 4 + 2 * WK * 4;                    // 16-byte copies per sample: q, dO, K, V
  auto issue = [&](int stage, int p) {
    const uint32_t st = su + O_ST + stage * 2 * SLOT;
    for (int idx = tid; idx < 2 * PER; idx += 128) {
      const int slot = idx / PER, rem = idx - slot * PER;
      const int a = 2 * p + slot;
      const bool ok_a = a < A;
      const uint32_t sbase = st + slot * SLOT;
      if (rem < 2 * WQ * 4) {
        const int which = rem >= WQ * 4, rr = which ? rem - WQ * 4 : rem;
        const int row = rr >> 2, ch = rr & 3, q = WQ * w + row;
        const bool ok = ok_a && q < N;
        const __nv_bfloat16* src = (which ? gdo : gq) + ((size_t)a * N + (ok ? q : 0)) * DM + h * DH + ch * 8;
        cp_async16(tile_addr(sbase + (which ? TQB + 2 * TKB : 0), row, ch), src, ok);
      } else {
        const int r2 = rem - 2 * WQ * 4, which = r2 >= WK * 4, rr = which ? r2 - WK * 4 : r2;
        const int row = rr >> 2, ch = rr & 3, key = kbase + row;
        const bool ok = ok_a && key >= 0 && key < N;
        const __nv_bfloat16* src = (which ? gv : gk) + ((size_t)a * N + (ok ? key : 0)) * DM + h * DH + ch * 8;
        cp_async16(tile_addr(sbase + TQB + which * TKB, row, ch), src, ok);
      }
    }
    cp_commit();
  };
  int p = z;
  if (p < npair) issue(0, p);
  __syncthreads();
  const int mt = warp & 1, sl = warp >> 1;
  const int r0 = lane >> 2, c0 = 2 * (lane & 3);
  for (int stage = 0; p < npair; p += psplit, stage ^= 1) {
    if (p + psplit < npair) { issue(stage ^ 1, p + psplit); cp_wait<1>(); } else { cp_wait<0>(); }
    __syncthreads();
    const int a = 2 * p + sl;
    if (a < A) {
      const uint32_t sbase = su + O_ST + stage * 2 * SLOT + sl * SLOT, tq = sbase, tk = sbase + TQB, tv = tk + TKB, td = tv + TKB;
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
        const float2 ka = *reinterpret_cast<const float2*>(kadd + c);
        const float2 b_lo = *reinterpret_cast<const float2*>(sbias + (16 * mt + r0) * BS + c);
        const float2 b_hi = *reinterpret_cast<const float2*>(sbias + (16 * mt + r0 + 8) * BS + c);
        const float p0 = ex2f(fmaf(s[nt][0], QSCALE * LOG2E, b_lo.x * LOG2E + ka.x) - lse0);
        const float p1 = ex2f(fmaf(s[nt][1], QSCALE * LOG2E, b_lo.y * LOG2E + ka.y) - lse0);
        const float p2 = ex2f(fmaf(s[nt][2], QSCALE * LOG2E, b_hi.x * LOG2E + ka.x) - lse1);
        const float p3 = ex2f(fmaf(s[nt][3], QSCALE * LOG2E, b_hi.y * LOG2E + ka.y) - lse1);
        s[nt][0] = p0 * (dp[nt][0] - d0);
        s[nt][1] = p1 * (dp[nt][1] - d0);
        s[nt][2] = p2 * (dp[nt][2] - d1);
        s[nt][3] = p3 * (dp[nt][3] - d1);
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
}
