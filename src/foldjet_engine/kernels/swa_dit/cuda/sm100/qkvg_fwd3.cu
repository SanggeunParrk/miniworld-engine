// qkvg_fwd3.cu — the SWA atom block's first forward stage on sm_100a for MANY tiles per CTA (inference, one modulation row per sequence
// row: A = 1), with the adaLN modulation inside. Same math and rounding points as mod_fwd.cu + qkvg_fwd2.cu (the same bits):
//   a = rn(silu(c));  shift_a | scale_a = a Wmod[0:256]^T (fp32);
//   x = rn(RMS(q) (1 + scale_a) + shift_a);  p_{q,k,v,g} = x W_{q,k,v,g}^T;
//   Q = rn(rope(rn(headRMS(rn(p_q)))))  (same for K);  V = rn(p_v);  G = rn(p_g)      Q / K / V head-major [N, H, S, D]; G row-major.
// qkvg_fwd1t.cu's design (no modulation launch, no fp32 [R, 768] in HBM: a tile reads 20 KB instead of 44 KB) as a loop over the CTA's
// 32-row tiles (grid = min(tiles, SMs); tile t = blockIdx + T gridDim):
//   - W = [Wq; Wk; Wv; Wg] (128 KB) is a bf16 A operand read in place from shared memory (no TMEM copy on the first tile's path);
//   - Wmod[0:256] (64 KB) lives in TMEM (128 columns): unit 0 staged through the output staging tiles, unit 1 through W's last slot (Wg,
//     loaded once that unit is in TMEM), tcgen05.cp by the MMA warp;
//   - the tile inputs are loaded two tiles ahead (a cold TMA load takes ~2.3 us to land: with the stage released by the tile's
//     epilogue the row threads waited that long per tile): q -> x and c -> a in place in two buffers, each free again when the tile's
//     projection MMAs have read it (their commit arrives xcfree), cos / sin in four small buffers (tile T - 4's epilogue is over by
//     then); the projections go into two accumulator sets, so tile T + 1's a / modulation / x / projections run under tile T's
//     epilogue; the modulation results have their own columns;
//   - two output staging sets (Q, K, V head-major, G; a TMA store is read out of shared memory ~1.9 us after its issue): the even
//     tiles' set staged Wmod unit 0, the odd tiles' set is W's last slot -- Wg moves on into TMEM (64 columns) before the first
//     projection, so the first tile's path keeps Wq / Wk / Wv in place.
// Every barrier is waited by its one consumer before its producer can complete the next phase (no parity ABA): cfull -> modfull -> xfull
// alternate strictly between the row threads and the MMA warp; accfull / infull / xcfree / ofull / ofree are per buffer.
// Threads as qkvg_fwd2: warp 0 TMA loads (one per lane), warp 1 MMAs, warp 2 TMA stores (one per lane), warps 4 - 11 one channel per
// thread (warpgroup wg: rows 16 wg .. for a / x; the epilogue: warpgroup 0 -> Q, V, warpgroup 1 -> K, G, warp = head).
// TMEM (512 columns): Wmod units at 0 / 64; accumulator set b at 128 + 128 b (projection p at + 32 p); shift_a at 384, scale_a at 416; Wg at 448.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;
#ifndef TRACE
#define TRACE 0
#endif
#define TRP(i, cond) do { if (TRACE && blockIdx.x == 0 && (cond)) TRb[i] = clock64(); } while (0)
#define TR2(T, e) do { if (TRACE && blockIdx.x == 0 && tid == 128 && (T) < 12) TRb[32 + (T) * 16 + (e)] = clock64(); } while (0)

constexpr int C = 128, H = 4, D = 32, AT = 32;
constexpr int KBLK = 4096;                                                 // [32 rows][64] bf16, SW128
constexpr int T_ = 2 * KBLK;                                               // [32][128] row-major tile (8 KB)
constexpr int UNIT = 32768;                                                // one [128][128] bf16 weight unit: K-major SW64 [128][32] atoms
// W (4 units) | output staging: Q | K | V (head-major: 4 heads x [32][32], SW64) | G (row-major) | 2 input stages: q -> x | c -> a | 4 x (cos | sin)
constexpr int O_W = 0, O_HQ = 4 * UNIT, O_IN = O_HQ + 4 * T_;
constexpr int I_X = 0, I_CT = T_, ISTG = 2 * T_, NCS = 4, CSTG = 2 * AT * 64, O_CSN = O_IN + 2 * ISTG;   // cos | sin: NCS buffers of 4 KB
constexpr int O_RED = O_CSN + NCS * CSTG, O_RS = O_RED + 2 * 4 * 16 * 4, O_RR = O_RS + 2 * 16 * 4, O_BAR = O_RR + 2 * 4 * 32 * 4;
constexpr int SMEM = O_BAR + 256;
static_assert(O_IN - O_HQ == UNIT, "the output staging tiles stage Wmod unit 0");
static_assert(SMEM <= 232448, "shared memory");
constexpr int J_HK = T_, J_HV = 2 * T_, J_G = 3 * T_;                     // inside an output staging set (Q at 0)
DEVI uint32_t outb(int T) { return (uint32_t)((T & 1) ? O_W + 3 * UNIT : O_HQ); }   // tile T's set: even O_HQ .., odd W's last slot
constexpr uint32_t T_WM = 0, T_ACC = 128, T_SH = 384, T_SC = 416, T_WG = 448;
constexpr uint32_t I_32 = idesc_bf16(128, 32);

struct Bars {
  uint64_t wmfull[2], wmfree, wfull3, wfullg, infull[2], xcfree[2], cfull, modfull, xfull, accfull[2], ofull[2], ofree[2];
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
// mG: as mq. ntile = N S / 32 tiles, tile t = (sample t / nab, atoms 32 (t % nab) ..); CTA i takes tiles i, i + grid, ...
extern "C" __global__ void __launch_bounds__(384, 1)
swa_qkvg_fwd3_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mwm,
                    const __grid_constant__ CUtensorMap mc, const __grid_constant__ CUtensorMap mcos, const __grid_constant__ CUtensorMap msin,
                    const __grid_constant__ CUtensorMap mQ, const __grid_constant__ CUtensorMap mK, const __grid_constant__ CUtensorMap mV,
                    const __grid_constant__ CUtensorMap mG, int S, int nab, int ntile, float eps, float qk_eps,
                    unsigned long long* __restrict__ TRb) {
  extern __shared__ __align__(1024) uint8_t sm[];
  TRP(0, threadIdx.x == 0);
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto coords = [&](int T, int& b, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    b = t / nab; s0 = (t % nab) * AT;
  };

  if (tid == 0) {
    mbar_init(&B.wmfull[0], 1); mbar_init(&B.wmfull[1], 1); mbar_init(&B.wmfree, 1); mbar_init(&B.wfull3, 1); mbar_init(&B.wfullg, 1);
    for (int i = 0; i < 2; ++i) { mbar_init(&B.infull[i], 1); mbar_init(&B.xcfree[i], 1); mbar_init(&B.accfull[i], 1); }
    mbar_init(&B.cfull, 2); mbar_init(&B.modfull, 1); mbar_init(&B.xfull, 2); for (int i = 0; i < 2; ++i) { mbar_init(&B.ofull[i], 2); mbar_init(&B.ofree[i], 1); }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  TRP(1, tid == 0);

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA loads (one per lane: a TMA
    // instruction costs its issuing thread ~50 ns). A CTA takes ~83 GB/s through TMA, so the order is the order of need: the first
    // tile's inputs, Wmod[0:256], Wq / Wk / Wv, Wg (its slot staged Wmod unit 1), then the next tiles' inputs.
    if (lane == 0) {
      mbar_expect_tx(&B.wmfull[0], UNIT); mbar_expect_tx(&B.wmfull[1], UNIT); mbar_expect_tx(&B.wfull3, 3 * UNIT); mbar_expect_tx(&B.wfullg, UNIT);
    }
    __syncwarp();
    pdl_wait();                                                            // q / c come from the previous kernels
    if (lane == 0) pdl_launch();                                           // the attention / FFN kernels may start now: q and c are final
    auto load_in = [&](int T) {                                            // lanes 0 - 5; lane 0 has armed infull
      int b, s0; coords(T, b, s0);
      const uint32_t st = su + O_IN + (T & 1) * ISTG;
      uint64_t* bar = &B.infull[T & 1];
      if (lane < 2) tma_load_4d(st + I_X + lane * KBLK, &mq, bar, lane * 64, s0, b, 0);
      else if (lane < 4) tma_load_2d(st + I_CT + (lane - 2) * KBLK, &mc, bar, (lane - 2) * 64, b * S + s0);
      else if (lane == 4) tma_load_2d(su + O_CSN + (T % NCS) * CSTG, &mcos, bar, 0, b * S + s0);
      else if (lane == 5) tma_load_2d(su + O_CSN + (T % NCS) * CSTG + AT * 64, &msin, bar, 0, b * S + s0);
    };
    if (lane == 0 && ntT > 0) mbar_expect_tx(&B.infull[0], 2 * T_ + 2 * AT * 64);
    __syncwarp();
    if (ntT > 0) load_in(0);
    __syncwarp();
    if (lane < 8) tma_load_2d(su + (lane < 4 ? O_HQ : O_W + 3 * UNIT) + (lane & 3) * 8192, &mwm, &B.wmfull[lane >> 2], 32 * (lane & 3), 128 * (lane >> 2));
    __syncwarp();
    if (lane < 12) tma_load_2d(su + O_W + (lane >> 2) * UNIT + (lane & 3) * 8192, &mw, &B.wfull3, 32 * (lane & 3), 128 * (lane >> 2));
    else if (lane < 16) {
      mbar_wait(&B.wmfree, 0);                                             // Wmod unit 1 has left the slot
      tma_load_2d(su + O_W + 3 * UNIT + (lane & 3) * 8192, &mw, &B.wfullg, 32 * (lane & 3), 128 * 3);
    }
    __syncwarp();
    for (int T = 1; T < ntT; ++T) {
      if (lane == 0) {
        if (T >= 2) mbar_wait(&B.xcfree[T & 1], ((T >> 1) - 1) & 1);   // tile T - 2's x / a have been read by its MMAs
        mbar_expect_tx(&B.infull[T & 1], 2 * T_ + 2 * AT * 64);
      }
      __syncwarp();
      load_in(T);
      __syncwarp();
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    auto wdesc = [&](uint32_t base, int ks) { return desc_sw64(base + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2); };
    auto bdesc = [&](uint32_t base, int ks) { return desc_k128(base + (ks >> 2) * KBLK) + (uint64_t)((ks & 3) * 2); };
    for (int u = 0; u < 2; ++u) {                                          // Wmod unit u -> TMEM (128 lanes x 32 B per K16 step)
      mbar_wait(&B.wmfull[u], 0);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + T_WM + 64 * u + ks * 8, wdesc(su + (u == 0 ? O_HQ : O_W + 3 * UNIT), ks));
        if (u == 1) tc_commit(&B.wmfree);
      }
      __syncwarp();
    }
    for (int T = 0; T < ntT; ++T) {
      const int ab = T & 1;
      const uint32_t st = su + O_IN + (T & 1) * ISTG;
      mbar_wait(&B.cfull, T & 1);
      tc_fence_after();
      if (elect_one()) {                                                   // shift_a^T | scale_a^T = Wmod[128 k ..] a^T
#pragma unroll
        for (int k = 0; k < 2; ++k)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ts(tmem + T_SH + 32 * k, tmem + T_WM + 64 * k + ks * 8, bdesc(st + I_CT, ks), I_32, ks > 0 ? 1u : 0u);
        tc_commit(&B.modfull);
      }
      __syncwarp();
      mbar_wait(&B.xfull, T & 1);
      if (T == 0) { mbar_wait(&B.wfull3, 0); mbar_wait(&B.wfullg, 0); }
      TRP(2, lane == 0 && T == 0);
      tc_fence_after();
      if (elect_one()) {                                                   // p^T = W_p x^T
        if (T == 0) {                                                      // Wg -> TMEM: its slot becomes the odd tiles' output staging
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + T_WG + ks * 8, wdesc(su + O_W + 3 * UNIT, ks));
        }
#pragma unroll
        for (int p = 0; p < 3; ++p)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ss(tmem + T_ACC + 128 * ab + 32 * p, wdesc(su + O_W + p * UNIT, ks), bdesc(st + I_X, ks), I_32, ks > 0 ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) umma_ts(tmem + T_ACC + 128 * ab + 96, tmem + T_WG + ks * 8, bdesc(st + I_X, ks), I_32, ks > 0 ? 1u : 0u);
        tc_commit(&B.accfull[ab]);
        tc_commit(&B.xcfree[ab]);
      }
      __syncwarp();
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA stores (one per lane)
    pdl_wait();
    for (int T = 0; T < ntT; ++T) {
      int b, s0; coords(T, b, s0);
      const uint32_t ob = su + outb(T);
      if (lane == 0) mbar_wait(&B.ofull[T & 1], (T >> 1) & 1);
      __syncwarp();
      if (lane < 12) {
        const int h = lane & 3, o = lane >> 2;                             // o: 0 Q, 1 K, 2 V
        tma_store_5d(o == 0 ? &mQ : (o == 1 ? &mK : &mV), ob + o * T_ + h * 2048, 0, s0, h, b, 0);
      } else if (lane < 14) {
        tma_store_4d(&mG, ob + J_G + (lane - 12) * KBLK, (lane - 12) * 64, s0, b, 0);
      }
      if (lane < 14) { tma_store_commit(); tma_store_wait_read0(); }
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.ofree[T & 1]);
    }
    if (lane < 14) tma_store_wait0();
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
    const bool lead = qw == 0 && lane == 0;
    auto xphase = [&](int T) {
      const uint32_t st = su + O_IN + (T & 1) * ISTG;
      TR2(T, 0);
      mbar_wait(&B.infull[T & 1], (T >> 1) & 1);
      TR2(T, 1);
      TRP(3, tid == 128 && T == 0);
      // ---- a = rn(silu(c)) in place, rows 16 wg .. of channel c (the kit sigmoid, as mod_fwd.cu)
      {
        float cv[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) { const int r = 16 * wg + j; cv[j] = lds_bf16(st + I_CT + cofs + (uint32_t)r * 128u + xo[r & 7]); }
#pragma unroll
        for (int j = 0; j < 16; ++j) { const int r = 16 * wg + j; sts_bf16(st + I_CT + cofs + (uint32_t)r * 128u + xo[r & 7], cv[j] * sigmoid_kit(cv[j])); }
      }
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (lead) mbar_arrive(&B.cfull);
      TR2(T, 2);
      // ---- x = rn(RMS(q) (1 + scale_a) + shift_a) over q
      {
        float qv[16], v[32];
#pragma unroll
        for (int j = 0; j < 16; ++j) { const int r = 16 * wg + j; qv[j] = lds_bf16(st + I_X + cofs + (uint32_t)r * 128u + xo[r & 7]); }
#pragma unroll
        for (int j = 0; j < 16; ++j) { v[j] = qv[j] * qv[j]; v[16 + j] = 0.f; }
        const float ps = rscatter32(v, lane);
        if (lane < 16) red[(wg * 4 + qw) * 16 + lane] = ps;
        named_bar_sync(1 + wg, 128);
        if (qw == 0 && lane < 16)
          rsv[wg * 16 + lane] = rsqrtf((red[(wg * 4) * 16 + lane] + red[(wg * 4 + 1) * 16 + lane] + red[(wg * 4 + 2) * 16 + lane] +
                                        red[(wg * 4 + 3) * 16 + lane]) * (1.f / C) + eps);
        named_bar_sync(1 + wg, 128);
        TR2(T, 3);
        mbar_wait(&B.modfull, T & 1);
        TR2(T, 4);
        TRP(5, tid == 128 && T == 0);
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
          sts_bf16(st + I_X + cofs + (uint32_t)r * 128u + xo[r & 7], qv[j] * rv[j] * (1.f + __uint_as_float(scv[j])) + __uint_as_float(shv[j]));
        }
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + wg, 128);
      if (lead) mbar_arrive(&B.xfull);
      TR2(T, 5);
      TRP(6, tid == 128 && T == 0);
    };
    auto epi = [&](int T) {
      // qkvg_fwd2's epilogue: one output channel per thread: warpgroup 0 -> Q, V; warpgroup 1 -> K, G; warp qw = head qw, lane = d
      const uint32_t sto = (uint32_t)(O_CSN + (T % NCS) * CSTG);
      const int ab = T & 1;
      TR2(T, 6);
      mbar_wait(&B.accfull[ab], (T >> 1) & 1);
      TR2(T, 7);
      if (T >= 2) mbar_wait(&B.ofree[T & 1], ((T >> 1) - 1) & 1);          // tile T - 2's stores have read the staging set
      const uint32_t ob = su + outb(T);
      TR2(T, 8);
      TRP(7, tid == 128 && T == 0);
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
          float v[32];
#pragma unroll
          for (int j = 0; j < 32; ++j) { const float y = rnb(__uint_as_float(av[j])); v[j] = y * y; }
          const float ss = rscatter32(v, lane);
          rrv[(wg * 4 + qw) * 32 + lane] = rsqrtf(ss * (1.f / D) + qk_eps);
          __syncwarp();
          const uint32_t hb = ob + (p == 0 ? 0 : J_HK) + (uint32_t)qw * 2048u + sw_l;
#pragma unroll
          for (int j8 = 0; j8 < 4; ++j8) {
            float cs[8], sn[8], rr[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              const int at = 8 * j8 + e;
              rr[e] = rrv[(wg * 4 + qw) * 32 + at];
              cs[e] = *reinterpret_cast<const float*>(sm + sto + at * 64 + dd * 4);
              sn[e] = *reinterpret_cast<const float*>(sm + sto + AT * 64 + at * 64 + dd * 4);
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
          const uint32_t hb = ob + J_HV + (uint32_t)qw * 2048u + sw_l;
#pragma unroll
          for (int j = 0; j < 32; ++j) sts_bf16(hb + (uint32_t)j * 64u + ((ck ^ (((uint32_t)j >> 1) & 3u)) << 4), __uint_as_float(av[j]));
        } else {                                                           // G -> row-major staging
          const uint32_t gb = ob + J_G + cofs;
#pragma unroll
          for (int j = 0; j < 32; ++j) sts_bf16(gb + (uint32_t)j * 128u + xo[j & 7], __uint_as_float(av[j]));
        }
        TR2(T, 9 + pp);
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + wg, 128);
      if (lead) mbar_arrive(&B.ofull[T & 1]);
      TR2(T, 11);
      TRP(8, tid == 128 && T == 0);
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
