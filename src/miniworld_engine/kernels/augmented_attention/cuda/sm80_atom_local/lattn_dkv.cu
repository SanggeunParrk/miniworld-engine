// lattn_dkv.cu — dK and dV of the AF3 windowed atom attention, key-centric, sm_80 (mma.sync).
//
// 16 keys (a "unit" u: keys 16 u .. 16 u + 15) are seen by exactly four query windows, i.e. 128 consecutive queries starting at 32 i0 with
// i0 = u even ? u/2 - 2 : (u-1)/2 - 1, and for every one of those queries the unit's keys sit at window offsets 16 u - 32 i + 48 + [0, 16).
// A warp owns one unit and computes for its 128 queries  S^T = K Q^T,  P^T = exp2(S^T - LSE),  dV = P^T dO,  dP^T = V dO^T,
// dS^T = P^T (dP^T - D),  dK = dS^T Q / sqrt(32).  A CTA = 4 consecutive units (their windows span 6 windows = 192 queries, which the four
// warps share through shared memory), one head, the samples z, z + psplit, ... (one per iteration, a single stage: two CTAs per SM).
// The per-warp bias fragments (log2 units, key mask folded in as -inf) stay in shared memory for all the CTA's samples. Queries outside
// [0, N) get LSE 1e30 (P = 0) in the CTAs that touch the edges of the sequence. dbias is not computed here: lattn_dq.cu sums dS over the
// samples (its tile is the bias window).
// dk / dv go to dkv[row, 128 + h * 32 ..] / dkv[row, 256 + h * 32 ..] (row stride ldd).
// SPDX-License-Identifier: Apache-2.0
#include "local.cuh"
#ifndef ATT_PF
#define ATT_PF 1
#endif

constexpr int QROWS = 192, KROWS = 64;
constexpr int O_QS = 0, O_DS = QROWS * 64, O_KS = 2 * QROWS * 64, O_VS = O_KS + KROWS * 64, O_LS = O_VS + KROWS * 64;
constexpr int O_DD = O_LS + QROWS * 4, STAGE = O_DD + QROWS * 4;
constexpr int O_BF = (STAGE + 15) / 16 * 16;                       // per-warp bias fragments: [warp][nt][lane] float4 (log2 units + key term)
constexpr int SMEM_BYTES = O_BF + 4 * 16 * 32 * 16;

extern "C" __global__ void __launch_bounds__(128, 2)
local_attn_dkv(const __nv_bfloat16* __restrict__ gq, const __nv_bfloat16* __restrict__ gk, const __nv_bfloat16* __restrict__ gv,
               const __nv_bfloat16* __restrict__ gdo, const float* __restrict__ gbias, const uint8_t* __restrict__ kmask,
               const float* __restrict__ LSE, const float* __restrict__ Dd, __nv_bfloat16* __restrict__ dkv,
               int ldd, int N, int A, int nwin, int psplit) {
  extern __shared__ __align__(1024) uint8_t sm[];
  float4* sbf = reinterpret_cast<float4*>(sm + O_BF);
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int c = blockIdx.x, h = blockIdx.y, z = blockIdx.z;
  const int u = 4 * c + warp, m = u >> 1;
  const int i0 = (u & 1) ? m - 1 : m - 2, ib = 2 * c - 2;           // this warp's first window / the CTA's first window
  const int qoff = 32 * (i0 - ib);                                    // this warp's first query row in the 192-row tile
  const int qbase = 32 * ib, kbase = 64 * c;
  const int r0 = lane >> 2, c0 = 2 * (lane & 3);
  const int key0 = 16 * u + r0, key1 = key0 + 8;
  const float kv0 = (key0 < N && (kmask == nullptr || kmask[key0])) ? 0.f : -INFINITY;
  const float kv1 = (key1 < N && (kmask == nullptr || kmask[key1])) ? 0.f : -INFINITY;

  // bias fragments of this warp's (key row, query column) elements
  for (int nt = 0; nt < 16; ++nt) {
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int n = 8 * nt + c0 + (e & 1), key = (e >> 1) ? key1 : key0;
      const int i = i0 + (n >> 5), qq = n & 31, r = key - (32 * i - KOFF);
      float b = 0.f;
      if (i >= 0 && i < nwin && r >= 0 && r < WK) b = gbias[(((size_t)h * nwin + i) * WQ + qq) * WK + r] * LOG2E;
      v[e] = b + ((e >> 1) ? kv1 : kv0);
    }
    sbf[(warp * 16 + nt) * 32 + lane] = make_float4(v[0], v[1], v[2], v[3]);
  }

  // copy mapping: thread -> (row tid / 4 + 32 k, 16-byte chunk tid % 4); the swizzle depends on bits 1-2 of the row only
  const int crow = tid >> 2, cch = tid & 3;
  const uint32_t cswz = crow * 64 + ((cch ^ ((crow >> 1) & 3)) << 4);
  int qo[6], ko[2];
  uint32_t qvalid = 0, kvalid = 0;
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    const int q = qbase + crow + 32 * k;
    const bool ok = q >= 0 && q < N;
    qvalid |= (ok ? 1u : 0u) << k;
    qo[k] = (ok ? q : 0) * DM + h * DH + cch * 8;
  }
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int key = kbase + crow + 32 * k;
    const bool ok = key < N;
    kvalid |= (ok ? 1u : 0u) << k;
    ko[k] = (ok ? key : 0) * DM + h * DH + cch * 8;
  }
  const bool edge = qbase < 0 || qbase + QROWS > N;                  // CTA-uniform: some of the 192 queries lie outside the sequence
  const int lq0 = tid, lq1 = tid + 128;                               // the two LSE / D entries of this thread (the second only for tid < 64)
  const int lr0 = qbase + lq0, lr1 = qbase + lq1;
  const bool lok0 = lr0 >= 0 && lr0 < N, lok1 = lq1 < QROWS && lr1 >= 0 && lr1 < N;

  for (int a = z; a < A; a += psplit) {
#if ATT_PF
    if (cch == 0 && a + psplit < A) {                                 // the next sample's rows into L2
      const size_t sb = (size_t)(a + psplit) * N * DM;
#pragma unroll
      for (int k = 0; k < 6; ++k)
        if ((qvalid >> k) & 1u) { prefetch_l2(gq + sb + qo[k]); prefetch_l2(gdo + sb + qo[k]); }
#pragma unroll
      for (int k = 0; k < 2; ++k)
        if ((kvalid >> k) & 1u) { prefetch_l2(gk + sb + ko[k]); prefetch_l2(gv + sb + ko[k]); }
    }
#endif
    {                                                                 // sample a -> the stage
      const size_t sb = (size_t)a * N * DM;
      const uint32_t st = su + cswz;
#pragma unroll
      for (int k = 0; k < 6; ++k) {
        const bool ok = (qvalid >> k) & 1u;
        cp_async16(st + O_QS + 32 * k * 64, gq + sb + qo[k], ok);
        cp_async16(st + O_DS + 32 * k * 64, gdo + sb + qo[k], ok);
      }
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const bool ok = (kvalid >> k) & 1u;
        cp_async16(st + O_KS + 32 * k * 64, gk + sb + ko[k], ok);
        cp_async16(st + O_VS + 32 * k * 64, gv + sb + ko[k], ok);
      }
      const size_t lb = ((size_t)a * NH + h) * N;
      cp_async4(su + O_LS + lq0 * 4, LSE + lb + (lok0 ? lr0 : 0), lok0);
      cp_async4(su + O_DD + lq0 * 4, Dd + lb + (lok0 ? lr0 : 0), lok0);
      if (lq1 < QROWS) {
        cp_async4(su + O_LS + lq1 * 4, LSE + lb + (lok1 ? lr1 : 0), lok1);
        cp_async4(su + O_DD + lq1 * 4, Dd + lb + (lok1 ? lr1 : 0), lok1);
      }
      cp_commit();
    }
    cp_wait<0>();
    if (edge) {                                                       // P = 0 for the queries outside the sequence
      float* ls = reinterpret_cast<float*>(sm + O_LS);
      if (!lok0) ls[lq0] = 1e30f;
      if (lq1 < QROWS && !lok1) ls[lq1] = 1e30f;
    }
    __syncthreads();

    const uint32_t tq = su + O_QS, td = su + O_DS, tk = su + O_KS, tv = su + O_VS;
    const float* sl = reinterpret_cast<const float*>(sm + O_LS);
    const float* sd = reinterpret_cast<const float*>(sm + O_DD);
    uint32_t ka[2][4], va[2][4];
    load_a(ka, tk, 16 * warp, lane);
    load_a(va, tv, 16 * warp, lane);
    float dv[4][4], dk[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) dv[i][0] = dv[i][1] = dv[i][2] = dv[i][3] = dk[i][0] = dk[i][1] = dk[i][2] = dk[i][3] = 0.f;
#pragma unroll
    for (int nh = 0; nh < 2; ++nh) {                                   // two halves of 64 queries
      float s[8][4], dp[8][4];
#pragma unroll
      for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = dp[i][0] = dp[i][1] = dp[i][2] = dp[i][3] = 0.f;
#pragma unroll
      for (int np = 0; np < 4; ++np)
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          uint32_t b0, b1, b2, b3;
          load_b_rows(b0, b1, b2, b3, tq, qoff + 64 * nh + 16 * np, ks, lane);
          mma16816(s[2 * np], ka[ks], b0, b1);
          mma16816(s[2 * np + 1], ka[ks], b2, b3);
          load_b_rows(b0, b1, b2, b3, td, qoff + 64 * nh + 16 * np, ks, lane);
          mma16816(dp[2 * np], va[ks], b0, b1);
          mma16816(dp[2 * np + 1], va[ks], b2, b3);
        }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int nt = 8 * nh + j, col = qoff + 64 * nh + 8 * j + c0;   // column of the 192-row tile
        const float2 ls = *reinterpret_cast<const float2*>(sl + col), ds = *reinterpret_cast<const float2*>(sd + col);
        const float4 bf = sbf[(warp * 16 + nt) * 32 + lane];
        const float p0 = ex2f(fmaf(s[j][0], QSCALE * LOG2E, bf.x) - ls.x);
        const float p1 = ex2f(fmaf(s[j][1], QSCALE * LOG2E, bf.y) - ls.y);
        const float p2 = ex2f(fmaf(s[j][2], QSCALE * LOG2E, bf.z) - ls.x);
        const float p3 = ex2f(fmaf(s[j][3], QSCALE * LOG2E, bf.w) - ls.y);
        dp[j][0] = p0 * (dp[j][0] - ds.x);
        dp[j][1] = p1 * (dp[j][1] - ds.y);
        dp[j][2] = p2 * (dp[j][2] - ds.x);
        dp[j][3] = p3 * (dp[j][3] - ds.y);                              // dS^T
        s[j][0] = p0; s[j][1] = p1; s[j][2] = p2; s[j][3] = p3;         // P^T
      }
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {                                 // 16 queries a step
        const uint32_t pa[4] = {pack_bf16(s[2 * ks][0], s[2 * ks][1]), pack_bf16(s[2 * ks][2], s[2 * ks][3]),
                                pack_bf16(s[2 * ks + 1][0], s[2 * ks + 1][1]), pack_bf16(s[2 * ks + 1][2], s[2 * ks + 1][3])};
        const uint32_t sa[4] = {pack_bf16(dp[2 * ks][0], dp[2 * ks][1]), pack_bf16(dp[2 * ks][2], dp[2 * ks][3]),
                                pack_bf16(dp[2 * ks + 1][0], dp[2 * ks + 1][1]), pack_bf16(dp[2 * ks + 1][2], dp[2 * ks + 1][3])};
#pragma unroll
        for (int dpair = 0; dpair < 2; ++dpair) {
          uint32_t b0, b1, b2, b3;
          load_b_cols(b0, b1, b2, b3, td, qoff + 64 * nh + 16 * ks, dpair, lane);
          mma16816(dv[2 * dpair], pa, b0, b1);
          mma16816(dv[2 * dpair + 1], pa, b2, b3);
          load_b_cols(b0, b1, b2, b3, tq, qoff + 64 * nh + 16 * ks, dpair, lane);
          mma16816(dk[2 * dpair], sa, b0, b1);
          mma16816(dk[2 * dpair + 1], sa, b2, b3);
        }
      }
    }
#pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      if (key0 < N) {
        __nv_bfloat16* base = dkv + ((size_t)a * N + key0) * ldd + h * DH + 8 * dt + c0;
        *reinterpret_cast<uint32_t*>(base + DM) = pack_bf16(dk[dt][0] * QSCALE, dk[dt][1] * QSCALE);
        *reinterpret_cast<uint32_t*>(base + 2 * DM) = pack_bf16(dv[dt][0], dv[dt][1]);
      }
      if (key1 < N) {
        __nv_bfloat16* base = dkv + ((size_t)a * N + key1) * ldd + h * DH + 8 * dt + c0;
        *reinterpret_cast<uint32_t*>(base + DM) = pack_bf16(dk[dt][2] * QSCALE, dk[dt][3] * QSCALE);
        *reinterpret_cast<uint32_t*>(base + 2 * DM) = pack_bf16(dv[dt][2], dv[dt][3]);
      }
    }
    __syncthreads();
  }
}
