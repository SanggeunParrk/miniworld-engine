// qkvg_fwd1t.cu — the SWA atom block's first forward stage on sm_100a for FEW tiles (inference, one modulation row per sequence row:
// A = 1), with the adaLN modulation inside. Same math and rounding points as mod_fwd.cu + qkvg_fwd2.cu:
//   a = rn(silu(c));  shift_a | scale_a = a Wmod[0:256]^T (fp32);
//   x = rn(RMS(q) (1 + scale_a) + shift_a);  p_{q,k,v,g} = x W_{q,k,v,g}^T;
//   Q = rn(rope(rn(headRMS(rn(p_q)))))  (same for K);  V = rn(p_v);  G = rn(p_g)      Q / K / V head-major [N, H, S, D]; G row-major.
// Why a build of its own: on one sample of a few thousand atoms every CTA of qkvg_fwd2 has ONE 32-row tile, and the kernel's time was its
// prologue (128 KB of W through TMA -> shared memory -> tcgen05.cp -> TMEM, only then the tile's inputs) behind a modulation launch that
// wrote an fp32 [R, 768] to HBM for it. Here: one tile per CTA (grid = tile count <= SMs), every load issued at once (the weights before
// the wait for the previous kernel), W = [Wq; Wk; Wv; Wg] and Wmod[0:256] are bf16 A operands read in place from shared memory (no TMEM
// copy), the modulation is two more MMAs on the c tile (silu in place) whose fp32 results the x phase reads straight from TMEM (thread =
// channel = TMEM lane, column = row), x is written over q, and the output staging tiles overlay Wmod once its MMAs are done.
// Threads as qkvg_fwd2: warp 0 lane 0 TMA loads, warp 1 MMAs, warp 2 lane 0 TMA stores, warps 4 - 11 one channel per thread (warpgroup
// wg: rows 16 wg .. for a / x; the epilogue: warpgroup 0 -> Q, V, warpgroup 1 -> K, G, warp = head).
// TMEM (256 columns): projection p at 32 p; shift_a at 128, scale_a at 160.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;
#ifndef TRACE
#define TRACE 0
#endif
#ifndef SAVE
#define SAVE 0        // 1 (round 3 part 8, training): also store x, rn(p_q), rn(p_k) (qkvg_fwd2's saves) for the backward
#endif
#ifndef LATE_WAIT
#define LATE_WAIT 0   // 1: no wait before the input loads -- the kernel before (mod_fwd LATE_TRIGGER) let this one start only after its own wait,
                      // so every input is final; the stores still wait (this kernel ends after the modulation kernel beside it)
#endif
#ifndef IN_FIRST
#define IN_FIRST 1                         // the tile inputs before Wmod in the TMA queue (0: Wmod first, before the wait for the previous kernel)
#endif
#define TRP(i, cond) do { if (TRACE && blockIdx.x == 0 && (cond)) TRb[i] = clock64(); } while (0)

constexpr int C = 128, H = 4, D = 32, AT = 32;
constexpr int KBLK = 4096;                                                 // [32 rows][64] bf16, SW128
constexpr int T_ = 2 * KBLK;                                               // [32][128] row-major tile (8 KB)
// W (4 x 32 KB) | Wmod[0:256] (2 x 32 KB; later the staging tiles Q | K | V (head-major: 4 heads x [32][32], SW64) | G) | q -> x | c -> a | cos | sin
constexpr int O_W = 0, O_WM = 4 * 32768, O_HQ = O_WM, O_HK = O_HQ + T_, O_HV = O_HK + T_, O_G = O_HV + T_, O_X = O_WM + 2 * 32768,
              O_CT = O_X + T_, O_CS = O_CT + T_, O_SN = O_CS + AT * 64;
constexpr int O_PQ = O_G + T_, O_PK = O_PQ + T_;                          // SAVE staging, also over Wmod (after its MMAs)
constexpr int O_RED = O_SN + AT * 64, O_RS = O_RED + 2 * 4 * 16 * 4, O_RR = O_RS + 2 * 16 * 4, O_BAR = O_RR + 2 * 4 * 32 * 4;
constexpr int SMEM = O_BAR + 256;
static_assert(SMEM <= 232448, "shared memory");
constexpr uint32_t T_ACC = 0, T_SH = 128, T_SC = 160;
constexpr uint32_t I_32 = idesc_bf16(128, 32);

struct Bars {
  uint64_t wfull, wmfull, infull, cfull, modfull, xfull, accfull, ofull;
  uint32_t tmem;
};
DEVI void tma_store_5d(const CUtensorMap* m, uint32_t src, int c0, int c1, int c2, int c3, int c4) {
  asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];"
               :: "l"(m), "r"(src), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory");
}
DEVI float lds_bf16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a) : "memory"); return __uint_as_float((uint32_t)v << 16); }
DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEVI void sts_bf16(uint32_t a, float x) {
  const unsigned short v = __bfloat16_as_ushort(__float2bfloat16_rn(x));
  asm volatile("st.shared.u16 [%0], %1;" :: "r"(a), "h"(v) : "memory");
}
DEVI float rscatter32(float (&v)[32], int lane) {
#pragma unroll
  for (int i = 0; i < 16; ++i) { const bool up = lane & 16; const float k = up ? v[i + 16] : v[i], s = up ? v[i] : v[i + 16]; v[i] = k + __shfl_xor_sync(~0u, s, 16); }
#pragma unroll
  for (int i = 0; i < 8; ++i) { const bool up = lane & 8; const float k = up ? v[i + 8] : v[i], s = up ? v[i] : v[i + 8]; v[i] = k + __shfl_xor_sync(~0u, s, 8); }
#pragma unroll
  for (int i = 0; i < 4; ++i) { const bool up = lane & 4; const float k = up ? v[i + 4] : v[i], s = up ? v[i] : v[i + 4]; v[i] = k + __shfl_xor_sync(~0u, s, 4); }
#pragma unroll
  for (int i = 0; i < 2; ++i) { const bool up = lane & 2; const float k = up ? v[i + 2] : v[i], s = up ? v[i] : v[i + 2]; v[i] = k + __shfl_xor_sync(~0u, s, 2); }
  { const bool up = lane & 1; const float k = up ? v[1] : v[0], s = up ? v[0] : v[1]; v[0] = k + __shfl_xor_sync(~0u, s, 1); }
  return v[0];
}

// mq: q [C, S, N, 1] box (64, 32, 1, 1) SW128; mw: [Wq; Wk; Wv; Wg] [C, 4C] box (32, 128) SW64; mwm: Wmod [C, 6C] box (32, 128) SW64;
// mc: c [C, N S] box (64, 32) SW128; mcos / msin: [16, N S] fp32 box (16, 32); mQ / mK / mV: head-major 5-D box (32, 32, 1, 1, 1) SW64;
// mG: as mq. Grid = N S / 32 tiles (<= the SM count), tile t = (sample t / nab, atoms 32 (t % nab) ..).
extern "C" __global__ void __launch_bounds__(384, 1)
swa_qkvg_fwd1t_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mwm,
                     const __grid_constant__ CUtensorMap mc, const __grid_constant__ CUtensorMap mcos, const __grid_constant__ CUtensorMap msin,
                     const __grid_constant__ CUtensorMap mQ, const __grid_constant__ CUtensorMap mK, const __grid_constant__ CUtensorMap mV,
                     const __grid_constant__ CUtensorMap mG, const __grid_constant__ CUtensorMap mX, const __grid_constant__ CUtensorMap mPQ,
                     const __grid_constant__ CUtensorMap mPK, int S, int nab, float eps, float qk_eps, int pair,
                     unsigned long long* __restrict__ TRb) {
  extern __shared__ __align__(1024) uint8_t sm[];
  TRP(0, threadIdx.x == 0);
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // pair (B200 2026-11-18, round 3 part 12): two CTAs per tile (grid = 2 x tiles <= SMs), rank r makes the projections p = r and r + 2
  // (rank 0: Q, V; rank 1: K, G) -- half of W loaded, half the MMAs, one epilogue pass per warpgroup instead of two. Both make a and x
  // (rank 0 stores the x save). No exchange between the two: the same bits as one CTA per tile.
  const int rank = pair ? (int)(blockIdx.x & 1u) : 0, tile = (int)blockIdx.x >> pair;
  const int b = tile / nab, s0 = (tile % nab) * AT;
  auto own = [&](int p) { return !pair || (p & 1) == rank; };

  if (tid == 0) {
    mbar_init(&B.wfull, 1); mbar_init(&B.wmfull, 1); mbar_init(&B.infull, 1); mbar_init(&B.cfull, 2); mbar_init(&B.modfull, 1); mbar_init(&B.xfull, 2);
    mbar_init(&B.accfull, 1); mbar_init(&B.ofull, 2);
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  TRP(1, tid == 0);

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA loads
    // One load per lane (a TMA instruction costs its issuing thread ~50 ns). A CTA takes ~83 GB/s through TMA whatever the source (L2-hot
    // or not, multicast or not: measured), so the order is the order of need: the tile's inputs (IN_FIRST; else Wmod before the wait
    // for the previous kernel), Wmod[0:256] (64 KB), then W (128 KB), which lands while a, the modulation and x are made.
    if (lane == 0) { mbar_expect_tx(&B.wmfull, 2 * 32768); mbar_expect_tx(&B.wfull, (pair ? 2 : 4) * 32768); }   // K-major SW64 [128][32] atoms
    __syncwarp();
    if (!IN_FIRST && lane < 8) tma_load_2d(su + O_WM + (lane >> 2) * 32768 + (lane & 3) * 8192, &mwm, &B.wmfull, 32 * (lane & 3), 128 * (lane >> 2));
    if (!LATE_WAIT) pdl_wait();                                            // q / c come from the previous kernels
    if (lane == 0) pdl_launch();                                           // the attention / FFN kernels may start now: q and c are final
    TRP(10, lane == 0);
    if (lane == 0) mbar_expect_tx(&B.infull, 2 * T_ + 2 * AT * 64);
    __syncwarp();
    if (lane < 2) tma_load_4d(su + O_X + lane * KBLK, &mq, &B.infull, lane * 64, s0, b, 0);
    else if (lane < 4) tma_load_2d(su + O_CT + (lane - 2) * KBLK, &mc, &B.infull, (lane - 2) * 64, b * S + s0);
    else if (lane == 4) tma_load_2d(su + O_CS, &mcos, &B.infull, 0, b * S + s0);
    else if (lane == 5) tma_load_2d(su + O_SN, &msin, &B.infull, 0, b * S + s0);
    __syncwarp();
    if (IN_FIRST && lane < 8) tma_load_2d(su + O_WM + (lane >> 2) * 32768 + (lane & 3) * 8192, &mwm, &B.wmfull, 32 * (lane & 3), 128 * (lane >> 2));
    __syncwarp();
    if (!pair) {
      if (lane < 16) tma_load_2d(su + O_W + (lane >> 2) * 32768 + (lane & 3) * 8192, &mw, &B.wfull, 32 * (lane & 3), 128 * (lane >> 2));
    } else if (lane < 8) {                                                 // W_p of p = rank, rank + 2
      const int p = rank + 2 * (lane >> 2);
      tma_load_2d(su + O_W + p * 32768 + (lane & 3) * 8192, &mw, &B.wfull, 32 * (lane & 3), 128 * p);
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    auto wdesc = [&](uint32_t base, int ks) { return desc_sw64(base + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2); };
    auto bdesc = [&](uint32_t base, int ks) { return desc_k128(base + (ks >> 2) * KBLK) + (uint64_t)((ks & 3) * 2); };
    mbar_wait(&B.wmfull, 0);
    mbar_wait(&B.cfull, 0);
    tc_fence_after();
    if (elect_one()) {                                                     // shift_a^T | scale_a^T = Wmod[128 k ..] a^T
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
          umma_ss(tmem + T_SH + 32 * k, wdesc(su + O_WM + k * 32768, ks), bdesc(su + O_CT, ks), I_32, ks > 0 ? 1u : 0u);
      tc_commit(&B.modfull);
    }
    __syncwarp();
    mbar_wait(&B.xfull, 0);
    mbar_wait(&B.wfull, 0);
    TRP(2, lane == 0);
    tc_fence_after();
    if (elect_one()) {                                                     // p^T = W_p x^T
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        if (!own(p)) continue;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
          umma_ss(tmem + T_ACC + 32 * p, wdesc(su + O_W + p * 32768, ks), bdesc(su + O_X, ks), I_32, ks > 0 ? 1u : 0u);
      }
      tc_commit(&B.accfull);
    }
    __syncwarp();
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA stores
    // one store per lane (a TMA instruction costs its issuing thread ~50 ns: 14 of them in a row were 0.8 us of the kernel)
    pdl_wait();
    if (lane == 0) mbar_wait(&B.ofull, 0);
    __syncwarp();
    if (lane < 12) {
      const int h = lane & 3, o = lane >> 2;                               // o: 0 Q, 1 K, 2 V
      if (own(o)) tma_store_5d(o == 0 ? &mQ : (o == 1 ? &mK : &mV), su + (o == 0 ? O_HQ : (o == 1 ? O_HK : O_HV)) + h * 2048, 0, s0, h, b, 0);
    } else if (lane < 14) {
      if (own(3)) tma_store_4d(&mG, su + O_G + (lane - 12) * KBLK, (lane - 12) * 64, s0, b, 0);
    } else if (SAVE && lane < 20) {
      const int o = (lane - 14) >> 1, kb = (lane - 14) & 1;              // 0 x, 1 rn(p_q), 2 rn(p_k)
      if (o == 0 ? rank == 0 : own(o - 1))
        tma_store_4d(o == 0 ? &mX : (o == 1 ? &mPQ : &mPK), su + (o == 0 ? O_X : (o == 1 ? O_PQ : O_PK)) + kb * KBLK, kb * 64, s0, b, 0);
    }
    if (lane < (SAVE ? 20 : 14)) { tma_store_commit(); tma_store_wait0(); }
    TRP(9, lane == 0);
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ compute threads
    const int wg = (warp - 4) >> 2, qw = warp & 3;
    const uint32_t lb = (uint32_t)qw * 32;
    const int c = (int)lb + lane;
    const uint32_t cofs = (uint32_t)(c >> 6) * KBLK + (uint32_t)(c & 7) * 2, cch = (uint32_t)((c & 63) >> 3);
    uint32_t xo[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) xo[k] = (cch ^ (uint32_t)k) << 4;
    float* red = reinterpret_cast<float*>(sm + O_RED);                     // [2 wg][4 warps][16 rows]
    float* rsv = reinterpret_cast<float*>(sm + O_RS);                      // [2 wg][16 rows] rstd
    float* rrv = reinterpret_cast<float*>(sm + O_RR);                      // [2 wg][4 warps (heads)][32 rows] head-RMS factor
    const uint32_t tl = tmem + (lb << 16) + 16 * wg;                       // this thread's lane (channel), the warpgroup's 16 columns (rows)
    mbar_wait(&B.infull, 0);
  TRP(3, tid == 128);
    // ---- a = rn(silu(c)) in place, rows 16 wg .. of channel c (the kit sigmoid, as mod_fwd.cu)
    {
      float cv[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) { const int r = 16 * wg + j; cv[j] = lds_bf16(su + O_CT + cofs + (uint32_t)r * 128u + xo[r & 7]); }
#pragma unroll
      for (int j = 0; j < 16; ++j) { const int r = 16 * wg + j; sts_bf16(su + O_CT + cofs + (uint32_t)r * 128u + xo[r & 7], cv[j] * sigmoid_kit(cv[j])); }
    }
    fence_proxy_async();
    named_bar_sync(1 + wg, 128);
    if (qw == 0 && lane == 0) mbar_arrive(&B.cfull);
  TRP(4, tid == 128);
    // ---- x = rn(RMS(q) (1 + scale_a) + shift_a) over q
    {
      float qv[16], v[32];
#pragma unroll
      for (int j = 0; j < 16; ++j) { const int r = 16 * wg + j; qv[j] = lds_bf16(su + O_X + cofs + (uint32_t)r * 128u + xo[r & 7]); }
#pragma unroll
      for (int j = 0; j < 16; ++j) { v[j] = qv[j] * qv[j]; v[16 + j] = 0.f; }
      const float ps = rscatter32(v, lane);
      if (lane < 16) red[(wg * 4 + qw) * 16 + lane] = ps;
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane < 16)
        rsv[wg * 16 + lane] = rsqrtf((red[(wg * 4) * 16 + lane] + red[(wg * 4 + 1) * 16 + lane] + red[(wg * 4 + 2) * 16 + lane] +
                                      red[(wg * 4 + 3) * 16 + lane]) * (1.f / C) + eps);
      named_bar_sync(1 + wg, 128);
      mbar_wait(&B.modfull, 0);
  TRP(5, tid == 128);
      tc_fence_after();
      uint32_t shv[16], scv[16];
      tmem_ld16(tl + T_SH, shv); tmem_ld16(tl + T_SC, scv);
      tmem_wait_ld();
      float rv[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) rv[j] = rsv[wg * 16 + j];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int r = 16 * wg + j;
        sts_bf16(su + O_X + cofs + (uint32_t)r * 128u + xo[r & 7], qv[j] * rv[j] * (1.f + __uint_as_float(scv[j])) + __uint_as_float(shv[j]));
      }
    }
    fence_proxy_async();
    tc_fence_before();
    named_bar_sync(1 + wg, 128);
    if (qw == 0 && lane == 0) mbar_arrive(&B.xfull);
  TRP(6, tid == 128);
    // ---- epilogue (qkvg_fwd2's): one output channel per thread: warpgroup 0 -> Q, V; warpgroup 1 -> K, G; warp qw = head qw, lane = d
    mbar_wait(&B.accfull, 0);
  TRP(7, tid == 128);
    tc_fence_after();
    const int dd = lane & 15;
    const uint32_t sw_l = (uint32_t)(lane & 7) * 2, ck = (uint32_t)lane >> 3;
#pragma unroll 1
    for (int pp = 0; pp < 2 - pair; ++pp) {
      const int p = pair ? rank + 2 * wg : wg + 2 * pp;                   // wg 0: Q (0), V (2); wg 1: K (1), G (3) (pair: rank r -> r, r + 2)
      const uint32_t tacc = tmem + (lb << 16) + T_ACC + 32 * p;
      uint32_t av[32];
      tmem_ld32(tacc, av);
      tmem_wait_ld();
      if (p < 2) {
        if (SAVE) {                                                        // rn(p_q) / rn(p_k), row-major staging
          const uint32_t sb = su + (p == 0 ? O_PQ : O_PK) + cofs;
#pragma unroll
          for (int j = 0; j < 32; ++j) sts_bf16(sb + (uint32_t)j * 128u + xo[j & 7], __uint_as_float(av[j]));
        }
        float v[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) { const float y = rnb(__uint_as_float(av[j])); v[j] = y * y; }
        const float ss = rscatter32(v, lane);
        rrv[(wg * 4 + qw) * 32 + lane] = rsqrtf(ss * (1.f / D) + qk_eps);
        __syncwarp();
        const uint32_t hb = su + (p == 0 ? O_HQ : O_HK) + (uint32_t)qw * 2048u + sw_l;
#pragma unroll
        for (int j8 = 0; j8 < 4; ++j8) {
          float cs[8], sn[8], rr[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int at = 8 * j8 + e;
            rr[e] = rrv[(wg * 4 + qw) * 32 + at];
            cs[e] = *reinterpret_cast<const float*>(sm + O_CS + at * 64 + dd * 4);
            sn[e] = *reinterpret_cast<const float*>(sm + O_SN + at * 64 + dd * 4);
          }
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int j = 8 * j8 + e;
            const float yn = rnb(rnb(__uint_as_float(av[j])) * rr[e]);
            const float pr = __shfl_xor_sync(~0u, yn, 16);
            const float z = lane < 16 ? yn * cs[e] - pr * sn[e] : yn * cs[e] + pr * sn[e];
            sts_bf16(hb + (uint32_t)j * 64u + ((ck ^ (((uint32_t)j >> 1) & 3u)) << 4), z);
          }
        }
        __syncwarp();
      } else if (p == 2) {                                                 // V -> head-major staging
        const uint32_t hb = su + O_HV + (uint32_t)qw * 2048u + sw_l;
#pragma unroll
        for (int j = 0; j < 32; ++j) sts_bf16(hb + (uint32_t)j * 64u + ((ck ^ (((uint32_t)j >> 1) & 3u)) << 4), __uint_as_float(av[j]));
      } else {                                                             // G -> row-major staging
        const uint32_t gb = su + O_G + cofs;
#pragma unroll
        for (int j = 0; j < 32; ++j) sts_bf16(gb + (uint32_t)j * 128u + xo[j & 7], __uint_as_float(av[j]));
      }
    }
    fence_proxy_async();
    tc_fence_before();
    named_bar_sync(1 + wg, 128);
    if (qw == 0 && lane == 0) mbar_arrive(&B.ofull);
  TRP(8, tid == 128);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 256); }
}
