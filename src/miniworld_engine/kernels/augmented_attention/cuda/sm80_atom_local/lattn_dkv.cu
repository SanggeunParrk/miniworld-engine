// lattn_dkv.cu — dK, dV and dbias of the AF3 windowed atom attention, key-centric, sm_100a.
//
// 16 keys (a "unit" u: keys 16 u .. 16 u + 15) are seen by exactly four query windows, i.e. 128 consecutive queries starting at 32 i0 with
// i0 = u even ? u/2 - 2 : (u-1)/2 - 1, and for every one of those queries the unit's keys sit at window offsets 16 u - 32 i + 48 + [0, 16).
// A warp owns one unit and computes for its 128 queries  S^T = K Q^T,  P^T = exp2(S^T - LSE),  dV += P^T dO,  dP^T = V dO^T,
// dS^T = P^T (dP^T - D),  dK += dS^T Q / sqrt(32),  and sums dS^T over the samples into dbias[h][i][i % 32][j] (every dbias element is
// owned by exactly one key, hence one warp: no atomics). A CTA = 4 consecutive units (their windows span 6 windows = 192 queries,
// which the four warps share through shared memory), one head, ALL samples (double-buffered cp.async).
// dk / dv go to dkv[row, 128 + h * 32 ..] / dkv[row, 256 + h * 32 ..] (row stride ldd), dbias is fp32 [4, nwin, 32, 128] (zero-initialised
// by the caller: elements of keys >= N are never written).
// SPDX-License-Identifier: Apache-2.0
#include "local.cuh"

constexpr int QROWS = 192, KROWS = 64;
constexpr int O_QS = 0, O_DS = QROWS * 64, O_KS = 2 * QROWS * 64, O_VS = O_KS + KROWS * 64, O_LS = O_VS + KROWS * 64;
constexpr int O_DD = O_LS + QROWS * 4, STAGE = O_DD + QROWS * 4;
constexpr int O_BF = 2 * STAGE;                                    // per-warp bias fragments: [warp][nt][e][lane] (log2 units)
constexpr int SMEM_BYTES = O_BF + 4 * 16 * 4 * 32 * 4;

extern "C" __global__ void __launch_bounds__(128)
local_attn_dkv(const __nv_bfloat16* __restrict__ gq, const __nv_bfloat16* __restrict__ gk, const __nv_bfloat16* __restrict__ gv,
               const __nv_bfloat16* __restrict__ gdo, const float* __restrict__ gbias, const uint8_t* __restrict__ kmask,
               const float* __restrict__ LSE, const float* __restrict__ Dd, __nv_bfloat16* __restrict__ dkv, float* __restrict__ dbias,
               int ldd, int N, int A, int nwin) {
  extern __shared__ __align__(1024) uint8_t sm[];
  float* sbf = reinterpret_cast<float*>(sm + O_BF);
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int c = blockIdx.x, h = blockIdx.y;
  const int u = 4 * c + warp, m = u >> 1;
  const int i0 = (u & 1) ? m - 1 : m - 2, ib = 2 * c - 2;           // this warp's first window / the CTA's first window
  const int qoff = 32 * (i0 - ib);                                    // this warp's first query row in the 192-row tile
  const int qbase = 32 * ib, kbase = 64 * c;
  const int r0 = lane >> 2, c0 = 2 * (lane & 3);
  const int key0 = 16 * u + r0, key1 = key0 + 8;
  const float kv0 = (key0 < N && (kmask == nullptr || kmask[key0])) ? 0.f : -INFINITY;
  const float kv1 = (key1 < N && (kmask == nullptr || kmask[key1])) ? 0.f : -INFINITY;

  // bias fragments of this warp's (key row, query column) elements
  for (int nt = 0; nt < 16; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int n = 8 * nt + c0 + (e & 1), key = (e >> 1) ? key1 : key0;
      const int i = i0 + (n >> 5), qq = n & 31, r = key - (32 * i - KOFF);
      float v = 0.f;
      if (i >= 0 && i < nwin && r >= 0 && r < WK) v = gbias[(((size_t)h * nwin + i) * WQ + qq) * WK + r] * LOG2E;
      sbf[((warp * 16 + nt) * 4 + e) * 32 + lane] = v;
    }
  float dbacc[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) dbacc[i][0] = dbacc[i][1] = dbacc[i][2] = dbacc[i][3] = 0.f;

  auto issue = [&](int stage, int a) {
    const uint32_t st = su + stage * STAGE;
    for (int idx = tid; idx < 2 * QROWS * 4 + 2 * KROWS * 4; idx += 128) {
      if (idx < 2 * QROWS * 4) {
        const int which = idx >= QROWS * 4, rr = which ? idx - QROWS * 4 : idx;
        const int row = rr >> 2, ch = rr & 3, q = qbase + row;
        const bool ok = a < A && q >= 0 && q < N;
        const __nv_bfloat16* src = (which ? gdo : gq) + ((size_t)a * N + (ok ? q : 0)) * DM + h * DH + ch * 8;
        cp_async16(tile_addr(st + (which ? O_DS : O_QS), row, ch), src, ok);
      } else {
        const int r2 = idx - 2 * QROWS * 4, which = r2 >= KROWS * 4, rr = which ? r2 - KROWS * 4 : r2;
        const int row = rr >> 2, ch = rr & 3, key = kbase + row;
        const bool ok = a < A && key < N;
        const __nv_bfloat16* src = (which ? gv : gk) + ((size_t)a * N + (ok ? key : 0)) * DM + h * DH + ch * 8;
        cp_async16(tile_addr(st + (which ? O_VS : O_KS), row, ch), src, ok);
      }
    }
    for (int idx = tid; idx < 2 * QROWS; idx += 128) {
      const int which = idx >= QROWS, row = which ? idx - QROWS : idx, q = qbase + row;
      const bool ok = a < A && q >= 0 && q < N;
      cp_async4(st + (which ? O_DD : O_LS) + row * 4, (which ? Dd : LSE) + ((size_t)a * NH + h) * N + (ok ? q : 0), ok);
    }
    cp_commit();
  };

  issue(0, 0);
  __syncthreads();
  for (int a = 0, stage = 0; a < A; ++a, stage ^= 1) {
    if (a + 1 < A) { issue(stage ^ 1, a + 1); cp_wait<1>(); } else { cp_wait<0>(); }
    __syncthreads();
    const uint32_t st = su + stage * STAGE, tq = st + O_QS, td = st + O_DS, tk = st + O_KS, tv = st + O_VS;
    const float* sl = reinterpret_cast<const float*>(sm + stage * STAGE + O_LS);
    const float* sd = reinterpret_cast<const float*>(sm + stage * STAGE + O_DD);
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
        const int qa0 = qbase + col, qa1 = qa0 + 1;
        const float2 ls = *reinterpret_cast<const float2*>(sl + col), ds = *reinterpret_cast<const float2*>(sd + col);
        const float l0 = (qa0 >= 0 && qa0 < N) ? ls.x : 1e30f, l1 = (qa1 >= 0 && qa1 < N) ? ls.y : 1e30f;
        const float p0 = ex2f(fmaf(s[j][0], QSCALE * LOG2E, sbf[((warp * 16 + nt) * 4 + 0) * 32 + lane] + kv0) - l0);
        const float p1 = ex2f(fmaf(s[j][1], QSCALE * LOG2E, sbf[((warp * 16 + nt) * 4 + 1) * 32 + lane] + kv0) - l1);
        const float p2 = ex2f(fmaf(s[j][2], QSCALE * LOG2E, sbf[((warp * 16 + nt) * 4 + 2) * 32 + lane] + kv1) - l0);
        const float p3 = ex2f(fmaf(s[j][3], QSCALE * LOG2E, sbf[((warp * 16 + nt) * 4 + 3) * 32 + lane] + kv1) - l1);
        const float t0 = p0 * (dp[j][0] - ds.x), t1 = p1 * (dp[j][1] - ds.y), t2 = p2 * (dp[j][2] - ds.x), t3 = p3 * (dp[j][3] - ds.y);
        dbacc[nt][0] += t0; dbacc[nt][1] += t1; dbacc[nt][2] += t2; dbacc[nt][3] += t3;
        s[j][0] = p0; s[j][1] = p1; s[j][2] = p2; s[j][3] = p3;       // P^T
        dp[j][0] = t0; dp[j][1] = t1; dp[j][2] = t2; dp[j][3] = t3;   // dS^T
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
  // dbias: dS summed over the samples; element (key row, query column) -> [h][i][i % 32][key - (32 i - 48)]
#pragma unroll
  for (int nt = 0; nt < 16; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int n = 8 * nt + c0 + (e & 1), key = (e >> 1) ? key1 : key0;
      const int i = i0 + (n >> 5), qq = n & 31, r = key - (32 * i - KOFF);
      if (key < N && i >= 0 && i < nwin && r >= 0 && r < WK) dbias[(((size_t)h * nwin + i) * WQ + qq) * WK + r] = dbacc[nt][e];
    }
}
