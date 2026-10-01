// qkvg_fwd2.cu — the SWA atom block's first forward stage on sm_100a, transposed (same math and rounding points as qkvg_fwd.cu / the Triton
// _qkvg_fwd):
//   x = rn(RMS(q) (1 + scale_a) + shift_a);  p_{q,k,v,g} = x W_{q,k,v,g}^T;
//   Q = rn(rope(rn(headRMS(rn(p_q)))))  (same for K);  V = rn(p_v);  G = rn(p_g)
//   Q / K / V head-major [N, H, S, D]; G row-major [M, C]; optionally x, rn(p_q), rn(p_k) saved for the backward.
// Structure (as ffn_fwd2.cu): 32-row tiles (SP = min(A, 8) augments x AT = 32 / SP atoms, AT <= ATM: 8 by default, 32 for A = 1 - 3); W = [Wq; Wk; Wv; Wg] (4 x 128 x 128) lives in
// TMEM as the bf16 A operand (TMA into the idle stages, tcgen05.cp by the MMA warp); p^T = W_p x^T (M = 128 output channels, N = 32 rows)
// into a double-buffered set of 4 accumulators, so tile T + 1's projections run under tile T's epilogue.
// Threads: x one channel per thread (rows 16 w .. per warpgroup; the row RMS by reduce-scatter + 4-warp exchange). Epilogue one output
// channel per thread: warp q of a warpgroup is head q, so the per-head RMS is a warp sum (reduce-scatter, then the 32 row factors through
// shared memory) and the RoPE partner d +- 16 is lane ^ 16; warpgroup 0 takes Q and V, warpgroup 1 K and G. Every output leaves through a
// staging tile and a TMA store (head-major Q / K / V by a 5-D box per head).
// TMEM: W_p at 64 p (256 columns); accumulator set b at 256 + 128 b, projection p at + 32 p.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef ATM
#define ATM 8                              // modulation / RoPE rows a tile stage holds (AT = 32 / min(A, 8) <= ATM)
#endif
constexpr int C = 128, H = 4, D = 32;
constexpr int KBLK = 4096;                                                 // [32 rows][64] bf16, SW128
constexpr int T_ = 2 * KBLK;                                               // [32][128] row-major tile (8 KB)
constexpr int HT = 4 * 2048;                                               // head-major staging: 4 heads x [32 rows][32] (SW64), 8 KB
// stage: q | x | Q | K | V | G | PQ | PK | mod (shift_a, scale_a: 8 blocks x AT <= ATM rows) | cos | sin
constexpr int O_Q = 0, O_X = T_, O_HQ = 2 * T_, O_HK = O_HQ + HT, O_HV = O_HK + HT, O_G = O_HV + HT, O_PQ = O_G + T_, O_PK = O_PQ + T_,
              O_MOD = O_PK + T_, O_CS = O_MOD + 8 * ATM * 128, O_SN = O_CS + ATM * 64, STG = (O_SN + ATM * 64 + 1023) / 1024 * 1024;
constexpr int O_RED = 2 * STG, O_RS = O_RED + 2 * 4 * 16 * 4, O_RR = O_RS + 2 * 16 * 4, O_BAR = O_RR + 2 * 4 * 32 * 4;
constexpr int SMEM = O_BAR + 256;
static_assert(SMEM <= 232448, "shared memory");
static_assert(2 * STG >= 4 * 32768, "the weight units stage through the tile stages");
constexpr uint32_t T_W = 0, T_ACC = 256;
constexpr uint32_t I_32 = idesc_bf16(128, 32);

struct Bars {
  uint64_t wfull, wfree, infull[2], infree[2], xfull, accfull[2], accfree[2], ofull;
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

extern "C" __global__ void __launch_bounds__(384, 1)
swa_qkvg_fwd2_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mmod,
                    const __grid_constant__ CUtensorMap mcos, const __grid_constant__ CUtensorMap msin, const __grid_constant__ CUtensorMap mQ,
                    const __grid_constant__ CUtensorMap mK, const __grid_constant__ CUtensorMap mV, const __grid_constant__ CUtensorMap mG,
                    const __grid_constant__ CUtensorMap mX, const __grid_constant__ CUtensorMap mPQ, const __grid_constant__ CUtensorMap mPK,
                    int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, float eps, float qk_eps, int save) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto coords = [&](int T, int& b, int& a0, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    const int ab = t % nab, r = t / nab, ag = r % nag;
    b = r / nag; a0 = ag * SP; s0 = ab * AT;
  };

  if (tid == 0) {
    mbar_init(&B.wfull, 1); mbar_init(&B.wfree, 1);
    for (int i = 0; i < 2; ++i) { mbar_init(&B.infull[i], 1); mbar_init(&B.infree[i], 1); mbar_init(&B.accfull[i], 1); mbar_init(&B.accfree[i], 2); }
    mbar_init(&B.xfull, 2); mbar_init(&B.ofull, 2);
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      mbar_expect_tx(&B.wfull, 4 * 32768);                                // W_p: K-major SW64 [128][32] atoms, 32 KB per projection
      for (int p = 0; p < 4; ++p)
        for (int ka = 0; ka < 4; ++ka) tma_load_2d(su + p * 32768 + ka * 8192, &mw, &B.wfull, 32 * ka, 128 * p);
      mbar_wait(&B.wfree, 0);
      pdl_wait();                                                          // q / mod come from the previous kernels
      const uint32_t tb = (uint32_t)(SP * AT * 128);
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        const int xs = T & 1;
        if (T >= 2) mbar_wait(&B.infree[xs], ((T >> 1) - 1) & 1);
        const uint32_t st = su + xs * STG;
        mbar_expect_tx(&B.infull[xs], 2 * tb + (uint32_t)(8 * AT * 128 + 2 * AT * 64));
        for (int kb = 0; kb < 2; ++kb) tma_load_4d(st + O_Q + kb * KBLK, &mq, &B.infull[xs], kb * 64, s0, b, a0);
        tma_load_3d(st + O_MOD, &mmod, &B.infull[xs], 0, b * S + s0, 0);      // blocks 0..7: shift_a | scale_a
        tma_load_2d(st + O_CS, &mcos, &B.infull[xs], 0, b * S + s0);
        tma_load_2d(st + O_SN, &msin, &B.infull[xs], 0, b * S + s0);
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    mbar_wait(&B.wfull, 0);
    tc_fence_after();
    if (elect_one()) {
      for (int p = 0; p < 4; ++p)
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + T_W + 64 * p + ks * 8, desc_sw64(su + p * 32768 + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
      tc_commit(&B.wfree);
    }
    __syncwarp();
    for (int T = 0; T < ntT; ++T) {
      const int ab = T & 1;
      const uint32_t xa = su + (T & 1) * STG + O_X;
      if (T >= 2) mbar_wait(&B.accfree[ab], ((T >> 1) - 1) & 1);
      mbar_wait(&B.xfull, T & 1);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int p = 0; p < 4; ++p)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ts(tmem + T_ACC + 128 * ab + 32 * p, tmem + T_W + 64 * p + ks * 8, desc_k128(xa + (ks >> 2) * KBLK) + (uint64_t)((ks & 3) * 2), I_32,
                    ks > 0 ? 1u : 0u);
        tc_commit(&B.accfull[ab]);
      }
      __syncwarp();
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA stores
    if (lane == 0) {
      pdl_wait();
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        const uint32_t st = su + (T & 1) * STG;
        mbar_wait(&B.ofull, T & 1);
        for (int h = 0; h < H; ++h) {
          tma_store_5d(&mQ, st + O_HQ + h * 2048, 0, s0, h, b, a0);
          tma_store_5d(&mK, st + O_HK + h * 2048, 0, s0, h, b, a0);
          tma_store_5d(&mV, st + O_HV + h * 2048, 0, s0, h, b, a0);
        }
        for (int kb = 0; kb < 2; ++kb) {
          tma_store_4d(&mG, st + O_G + kb * KBLK, kb * 64, s0, b, a0);
          if (save) {
            tma_store_4d(&mX, st + O_X + kb * KBLK, kb * 64, s0, b, a0);
            tma_store_4d(&mPQ, st + O_PQ + kb * KBLK, kb * 64, s0, b, a0);
            tma_store_4d(&mPK, st + O_PK + kb * KBLK, kb * 64, s0, b, a0);
          }
        }
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.infree[T & 1]);
      }
      tma_store_wait0();
    }
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
    int mrow0[2];                                                          // first modulation row of kind k (0 shift_a, 1 scale_a) for channel c
    const uint32_t cq = (uint32_t)((c & 31) >> 2), cl4 = (uint32_t)(c & 3) * 4;
    auto xphase = [&](int T) {
      // x = rn(RMS(q) (1 + scale_a) + shift_a) over rows 16 wg .. of the tile, one channel per thread -> x staging (row-major)
      const uint32_t st = su + (T & 1) * STG, sto = (T & 1) * STG;
      mbar_wait(&B.infull[T & 1], (T >> 1) & 1);
      mrow0[0] = (c >> 5) * AT; mrow0[1] = (4 + (c >> 5)) * AT;
      float qv[16], v[32];
      int at = (16 * wg) % AT, atr[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) { atr[j] = at; at = at + 1 == AT ? 0 : at + 1; }
#pragma unroll
      for (int j = 0; j < 16; ++j) { const int r = 16 * wg + j; qv[j] = lds_bf16(st + O_Q + cofs + (uint32_t)r * 128u + xo[r & 7]); }
#pragma unroll
      for (int j = 0; j < 16; ++j) { v[j] = qv[j] * qv[j]; v[16 + j] = 0.f; }
      const float ps = rscatter32(v, lane);
      if (lane < 16) red[(wg * 4 + qw) * 16 + lane] = ps;
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane < 16)
        rsv[wg * 16 + lane] = rsqrtf((red[(wg * 4) * 16 + lane] + red[(wg * 4 + 1) * 16 + lane] + red[(wg * 4 + 2) * 16 + lane] +
                                      red[(wg * 4 + 3) * 16 + lane]) * (1.f / C) + eps);
      named_bar_sync(1 + wg, 128);
      float shv[16], scv[16], rv[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int r0 = mrow0[0] + atr[j], r1 = mrow0[1] + atr[j];
        shv[j] = *reinterpret_cast<const float*>(sm + sto + O_MOD + r0 * 128 + ((cq ^ (uint32_t)(r0 & 7)) << 4) + cl4);
        scv[j] = *reinterpret_cast<const float*>(sm + sto + O_MOD + r1 * 128 + ((cq ^ (uint32_t)(r1 & 7)) << 4) + cl4);
        rv[j] = rsv[wg * 16 + j];
      }
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int r = 16 * wg + j;
        sts_bf16(st + O_X + cofs + (uint32_t)r * 128u + xo[r & 7], qv[j] * rv[j] * (1.f + scv[j]) + shv[j]);
      }
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) mbar_arrive(&B.xfull);
    };
    auto epi = [&](int T) {
      // one output channel per thread: warpgroup 0 -> Q, V; warpgroup 1 -> K, G; warp qw = head qw, lane = d; all 32 rows
      const uint32_t st = su + (T & 1) * STG, sto = (T & 1) * STG;
      const int ab = T & 1;
      mbar_wait(&B.accfull[ab], (T >> 1) & 1);
      tc_fence_after();
      const int dd = lane & 15;
      const uint32_t sw_l = (uint32_t)(lane & 7) * 2, ck = (uint32_t)lane >> 3;
#pragma unroll 1
      for (int pp = 0; pp < 2; ++pp) {
        const int p = wg + 2 * pp;                                         // wg 0: Q (0), V (2); wg 1: K (1), G (3)
        const uint32_t tacc = tmem + (lb << 16) + T_ACC + 128 * ab + 32 * p;
        uint32_t av[32];
        tmem_ld32(tacc, av);
        tmem_wait_ld();
        if (p < 2) {
          // pass A: rn(p), its save, the per-(row, head) sum of squares by a warp reduce-scatter (warp = head)
          float v[32];
          const uint32_t pst = st + (p == 0 ? O_PQ : O_PK) + cofs;
#pragma unroll
          for (int j = 0; j < 32; ++j) {
            const float y = rnb(__uint_as_float(av[j]));
            if (save) sts_bf16(pst + (uint32_t)j * 128u + xo[j & 7], y);
            v[j] = y * y;
          }
          const float ss = rscatter32(v, lane);
          rrv[(wg * 4 + qw) * 32 + lane] = rsqrtf(ss * (1.f / D) + qk_eps);
          __syncwarp();
          // pass B: headRMS, RoPE (partner d +- 16 = lane ^ 16), 8 rows at a time: loads first, then math and stores
          const uint32_t hb = st + (p == 0 ? O_HQ : O_HK) + (uint32_t)qw * 2048u + sw_l;
          int at = 0;
#pragma unroll
          for (int j8 = 0; j8 < 4; ++j8) {
            float cs[8], sn[8], rr[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              rr[e] = rrv[(wg * 4 + qw) * 32 + 8 * j8 + e];
              cs[e] = *reinterpret_cast<const float*>(sm + sto + O_CS + at * 64 + dd * 4);
              sn[e] = *reinterpret_cast<const float*>(sm + sto + O_SN + at * 64 + dd * 4);
              at = at + 1 == AT ? 0 : at + 1;
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
        } else if (p == 2) {                                               // V -> head-major staging
          const uint32_t hb = st + O_HV + (uint32_t)qw * 2048u + sw_l;
#pragma unroll
          for (int j = 0; j < 32; ++j) sts_bf16(hb + (uint32_t)j * 64u + ((ck ^ (((uint32_t)j >> 1) & 3u)) << 4), __uint_as_float(av[j]));
        } else {                                                           // G -> row-major staging
          const uint32_t gb = st + O_G + cofs;
#pragma unroll
          for (int j = 0; j < 32; ++j) sts_bf16(gb + (uint32_t)j * 128u + xo[j & 7], __uint_as_float(av[j]));
        }
      }
      tc_fence_before();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) mbar_arrive(&B.accfree[ab]);
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) mbar_arrive(&B.ofull);
    };
    if (ntT > 0) xphase(0);
    for (int T = 0; T < ntT; ++T) {
      if (T + 1 < ntT) xphase(T + 1);                                     // tile T + 1's projections run under this tile's epilogue
      epi(T);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
