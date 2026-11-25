// ffn_fwd2.cu — the SWA atom block's second forward stage on sm_100a, transposed (same math and rounding points as ffn_fwd.cu / the
// Triton _oproj_ffn_fwd):
//   gated = rn(sigmoid(g) o);  att = rn(gated Wo^T);  q1 = rn(q + rn(gate_a) att);
//   y = rn(RMS(q1) (1 + scale_f) + shift_f);  a|b = y Wu^T;  h = rn(silu(a) b);  ffn = rn(h Wd^T);  out = rn(q1 + rn(gate_f) ffn)
// optionally saving q1, att, y, ffn for the backward.
// Why transposed: 32-row tiles (the MMA N is the row count), so a small inference batch (A = 5, S = 1024 atoms: 5120 rows) still spreads
// over every SM, and no weight is streamed per tile: Wu (512 x 128) and Wd (128 x 256) live in TMEM as bf16 A operands (384 columns;
// TMA into the idle tile stages, then tcgen05.cp by the MMA warp),
// Wo in shared memory (32 KB, A of att^T = Wo gated^T). Per tile: att^T, a^T | b^T (two 128-hidden chunks), ffn^T (M = 128, N = 32).
// Threads: one channel per thread (the TMEM lane) for gated / q1 / RMS / y / out (rows 16 w .. per warpgroup; the row RMS by a 32-lane
// reduce-scatter + 4-warp exchange), one hidden unit per thread for h. Every activation feeding an MMA is written into a row-major tile in
// shared memory (the K-major B operand), in place over an input where it can be (gated over g, q1 over q, out over o), and leaves by TMA.
// Tiles: SP = min(A, 8) augments x AT = 32 / SP atoms (AT <= ATM); the tile stage (q | g | o | y | h | att | ffn | modulation) is
// double-buffered (NSTG = 2, ATM = 8: A >= 4) or single (NSTG = 1, ATM = 32: A = 1 - 3, whose modulation is not shared by augments;
// such small inference batches have about one tile per CTA anyway).
// TMEM: Wu tiles (a 0-127, a 128-255, b 0-127, b 128-255) at 0 / 64 / 128 / 192, Wd at 256 (128 columns), att^T 384, a^T 416, b^T 448, ffn^T 480.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef ATM
#define ATM 8                              // modulation rows a stage holds (AT = 32 / min(A, 8) <= ATM)
#endif
#ifndef NSTG
#define NSTG 2                             // tile stages
#endif
#ifndef CL
#define CL 1                              // cluster size: the weights are multicast, each CTA of the cluster issues 1 / CL of the loads
#endif

constexpr int C = 128;
constexpr int KBLK = 4096;                                                 // [32 rows][64] bf16, SW128
constexpr int T_ = 2 * KBLK;                                               // one [32][128] tile
constexpr int O_Q = 0, O_G = T_, O_O = 2 * T_, O_Y = 3 * T_, O_H = 4 * T_, O_ATT = 6 * T_, O_FF = 7 * T_, O_MOD = 8 * T_;
constexpr int MODB = 16 * ATM * 128;                                       // modulation: 16 blocks (gate_a | shift_f | scale_f | gate_f) x AT rows
constexpr int STG = O_MOD + MODB;                                          // 80 KB per stage (ATM 8)
constexpr int O_WO = NSTG * STG, O_RED = O_WO + 2 * 16384, O_RS = O_RED + 2 * 4 * 16 * 4, O_BAR = O_RS + 2 * 16 * 4;
// the six 32-KB weight units stage through the tile stages, then the Wo slot, then (if still short) O_WX
constexpr int NU_ST = NSTG * STG / 32768, O_WX = (O_BAR + 256 + 1023) / 1024 * 1024;
constexpr int SMEM = NU_ST >= 5 ? O_BAR + 256 : O_WX + (5 - NU_ST) * 32768;
static_assert(NU_ST >= 4, "weight staging");
DEVI int stg_of(int T) { return NSTG == 2 ? (T & 1) : T % NSTG; }                   // tile T's stage and its use count
DEVI int use_of(int T) { return NSTG == 2 ? (T >> 1) : T / NSTG; }
static_assert(SMEM <= 232448, "shared memory");
constexpr uint32_t T_WU = 0, T_WD = 256, T_ATT = 384, T_A = 416, T_B = 448, T_FF = 480;
constexpr uint32_t I_32 = idesc_bf16(128, 32);

struct Bars {
  uint64_t wofull, wfull, wfree, infull[2], infree[2], gfull, attfull, yfull, abfull, abfree, hfull, ffull, ofull;
  uint32_t tmem;
};
#ifndef TRACE
#define TRACE 0
#endif
#define TR(T, e) do { if (TRACE && blockIdx.x == 0 && (T) < 8) TRb[(T) * 16 + (e)] = clock64(); } while (0)
DEVI void mbar_wait_h(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok) : "r"(smem_u32(b)), "r"(parity), "r"(10000000u) : "memory");
  } while (!ok);
}
DEVI float lds_bf16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a) : "memory"); return __uint_as_float((uint32_t)v << 16); }
DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEVI void sts_bf16(uint32_t a, float x) {
  const unsigned short v = __bfloat16_as_ushort(__float2bfloat16_rn(x));
  asm volatile("st.shared.u16 [%0], %1;" :: "r"(a), "h"(v) : "memory");
}
DEVI float sigm(float x) { return rcpf(1.f + ex2f(-1.4426950408889634f * x)); }
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
swa_ffn_fwd2_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mo,
                   const __grid_constant__ CUtensorMap mmod, const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mout,
                   const __grid_constant__ CUtensorMap mq1, const __grid_constant__ CUtensorMap matt, const __grid_constant__ CUtensorMap my,
                   const __grid_constant__ CUtensorMap mff, const __grid_constant__ CUtensorMap mwu, const __grid_constant__ CUtensorMap mwd, int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, float eps, int save,
                   const __nv_bfloat16* __restrict__ WU, const __nv_bfloat16* __restrict__ WD, unsigned long long* __restrict__ TRb) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto wunit = [&](int u) -> uint32_t { return u < NU_ST ? su + u * 32768 : (u == NU_ST ? su + O_WO : su + O_WX + (u - NU_ST - 1) * 32768); };
  auto coords = [&](int T, int& b, int& a0, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    const int ab = t % nab, r = t / nab, ag = r % nag;
    b = r / nag; a0 = ag * SP; s0 = ab * AT;
  };

  if (tid == 0) {
    mbar_init(&B.wofull, 1); mbar_init(&B.wfull, 1); mbar_init(&B.wfree, 1);
    for (int i = 0; i < 2; ++i) { mbar_init(&B.infull[i], 1); mbar_init(&B.infree[i], 1); }
    mbar_init(&B.gfull, 2); mbar_init(&B.attfull, 1); mbar_init(&B.yfull, 2); mbar_init(&B.abfull, 1); mbar_init(&B.abfree, 8);
    mbar_init(&B.hfull, 2); mbar_init(&B.ffull, 1); mbar_init(&B.ofull, 2);
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (CL > 1) cluster_sync();                                             // every CTA's barriers exist before the multicast
  const uint32_t tmem = B.tmem;
  pdl_launch();
  if (tid == 0 && TRACE && blockIdx.x == 0) TRb[8 * 16] = clock64();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (tid == 0 && TRACE && blockIdx.x == 0) TRb[8 * 16 + 1] = clock64();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      // Wu (units 0-3: rows 128 u ..) and Wd (units 4-5: k 128 (u - 4) ..) as K-major SW64 [128][32] atoms, 32 KB per unit, into the
      // (still unused) tile stages, then the Wo slot (wunit); multicast over the cluster (this CTA issues atoms i with i % CL == rank).
      // The MMA warp moves them into TMEM with tcgen05.cp; Wo and the tiles load after that.
      {
        const uint32_t rk = CL > 1 ? cluster_rank() : 0;
        mbar_expect_tx(&B.wfull, 6 * 32768);
        for (int i = (int)rk; i < 24; i += CL) {
          const int u = i >> 2, ka = i & 3;
          const uint32_t dst = wunit(u) + ka * 8192;
          const CUtensorMap* m = u < 4 ? &mwu : &mwd;
          const int c0 = u < 4 ? 32 * ka : 128 * (u - 4) + 32 * ka, c1 = u < 4 ? 128 * u : 0;
          if (CL > 1) tma_load_2d_mc(dst, m, &B.wfull, c0, c1, (uint16_t)((1u << CL) - 1));
          else tma_load_2d(dst, m, &B.wfull, c0, c1);
        }
        mbar_wait_h(&B.wfree, 0);
      }
      pdl_wait();                                                          // tile inputs come from the previous kernel
      mbar_expect_tx(&B.wofull, 2 * 16384);
      for (int kb = 0; kb < 2; ++kb) tma_load_2d(su + O_WO + kb * 16384, &mwo, &B.wofull, kb * 64, 0);
      const uint32_t tb = (uint32_t)(SP * AT * 128);
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        const int xs = stg_of(T);
        if (T >= NSTG) mbar_wait_h(&B.infree[xs], (use_of(T) - 1) & 1);
        const uint32_t st = su + xs * STG;
        mbar_expect_tx(&B.infull[xs], 6 * tb + (uint32_t)(16 * AT * 128));
        for (int kb = 0; kb < 2; ++kb) {
          tma_load_4d(st + O_Q + kb * KBLK, &mq, &B.infull[xs], kb * 64, s0, b, a0);
          tma_load_4d(st + O_G + kb * KBLK, &mg, &B.infull[xs], kb * 64, s0, b, a0);
          tma_load_4d(st + O_O + kb * KBLK, &mo, &B.infull[xs], kb * 64, s0, b, a0);
        }
        tma_load_3d(st + O_MOD, &mmod, &B.infull[xs], 0, b * S + s0, 8);     // blocks 8..23: gate_a | shift_f | scale_f | gate_f
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    mbar_wait_h(&B.wfull, 0);                                              // weights -> TMEM (tcgen05.cp, 128 lanes x 32 B per K16 step)
    tc_fence_after();
    if (elect_one()) {
      for (int u = 0; u < 6; ++u) {
        const uint32_t src = wunit(u), col = u < 4 ? T_WU + 64 * u : T_WD + 64 * (u - 4);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + col + ks * 8, desc_sw64(src + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
      }
      tc_commit(&B.wfree);
    }
    __syncwarp();
    mbar_wait_h(&B.wofull, 0);
    for (int T = 0; T < ntT; ++T) {
      const uint32_t st = su + stg_of(T) * STG;
      auto bdesc = [&](uint32_t base, int ks) { return desc_k128(base + (ks >> 2) * KBLK) + (uint64_t)((ks & 3) * 2); };
      mbar_wait_h(&B.gfull, T & 1);                                          // gated written over g
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
          umma_ss(tmem + T_ATT, desc_k128(su + O_WO + (ks >> 2) * 16384) + (uint64_t)((ks & 3) * 2), bdesc(st + O_G, ks), I_32, ks > 0 ? 1u : 0u);
        tc_commit(&B.attfull);
      }
      __syncwarp();
      mbar_wait_h(&B.yfull, T & 1);
      for (int hc = 0; hc < 2; ++hc) {                                      // chunk 1 -> the att / ffn columns (free until M3): no wait
        tc_fence_after();
        const uint32_t ta = hc ? T_ATT : T_A, tb = hc ? T_FF : T_B;
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            umma_ts(tmem + ta, tmem + T_WU + 64 * hc + ks * 8, bdesc(st + O_Y, ks), I_32, ks > 0 ? 1u : 0u);
            umma_ts(tmem + tb, tmem + T_WU + 64 * (2 + hc) + ks * 8, bdesc(st + O_Y, ks), I_32, ks > 0 ? 1u : 0u);
          }
          tc_commit(&B.abfull);
        }
        __syncwarp();
      }
      mbar_wait_h(&B.hfull, T & 1);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 16; ++ks)
          umma_ts(tmem + T_FF, tmem + T_WD + ks * 8, bdesc(st + O_H + (ks >> 3) * 2 * KBLK, ks & 7), I_32, ks > 0 ? 1u : 0u);
        tc_commit(&B.ffull);
      }
      __syncwarp();
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA stores: out (+ q1, att, y, ffn)
    if (lane == 0) {
      pdl_wait();
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        const uint32_t st = su + stg_of(T) * STG;
        mbar_wait_h(&B.ofull, T & 1);
        for (int kb = 0; kb < 2; ++kb) {
          tma_store_4d(&mout, st + O_O + kb * KBLK, kb * 64, s0, b, a0);
          if (save) {
            tma_store_4d(&mq1, st + O_Q + kb * KBLK, kb * 64, s0, b, a0);
            tma_store_4d(&matt, st + O_ATT + kb * KBLK, kb * 64, s0, b, a0);
            tma_store_4d(&my, st + O_Y + kb * KBLK, kb * 64, s0, b, a0);
            tma_store_4d(&mff, st + O_FF + kb * KBLK, kb * 64, s0, b, a0);
          }
        }
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.infree[stg_of(T)]);
      }
      tma_store_wait0();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ row-stage threads
    const int wg = (warp - 4) >> 2, qw = warp & 3;
    const uint32_t lb = (uint32_t)qw * 32;
    const int c = (int)lb + lane;                                          // channel (or hidden unit within a chunk for h)
    const uint32_t cofs = (uint32_t)(c >> 6) * KBLK + (uint32_t)(c & 7) * 2, cch = (uint32_t)((c & 63) >> 3);
    float* red = reinterpret_cast<float*>(sm + O_RED);                     // [2 wg][4 warps][16 rows]
    float* rs = reinterpret_cast<float*>(sm + O_RS);                       // [2 wg][16 rows] rstd
    const uint32_t tl = tmem + (lb << 16) + 16 * wg;                       // this thread's lane, the warpgroup's 16 columns (rows)
    int mrow0[4];                                                          // first modulation row of kind k for channel c
#pragma unroll
    for (int k = 0; k < 4; ++k) mrow0[k] = (k * 4 + (c >> 5)) * AT;
    const uint32_t cq = (uint32_t)((c & 31) >> 2), cl4 = (uint32_t)(c & 3) * 4;
    int atr[16];                                                           // atom of row 16 wg + j
    {
      int a_ = (16 * wg) % AT;
#pragma unroll
      for (int j = 0; j < 16; ++j) { atr[j] = a_; a_ = a_ + 1 == AT ? 0 : a_ + 1; }
    }
    for (int T = 0; T < ntT; ++T) {
      const uint32_t st = su + stg_of(T) * STG, mo_ = stg_of(T) * STG + O_MOD;
      auto ea = [&](uint32_t base, int j) {                                // element (row 16 wg + j, channel c) of a row-major tile
        const int r = 16 * wg + j;
        return base + cofs + (uint32_t)r * 128u + ((cch ^ (uint32_t)(r & 7)) << 4);
      };
      auto modv = [&](int kind, int at) {                                  // kind 0 gate_a, 1 shift_f, 2 scale_f, 3 gate_f; channel c, atom at
        const int row = mrow0[kind] + at;
        return *reinterpret_cast<const float*>(sm + mo_ + row * 128 + ((cq ^ (uint32_t)(row & 7)) << 4) + cl4);
      };
      const bool tl0 = warp == 4 && lane == 0;
      if (tl0) TR(T, 0);
      mbar_wait_h(&B.infull[stg_of(T)], use_of(T) & 1);
      if (tl0) TR(T, 1);
      auto atj = [&](int j) { return atr[j]; };
      // ---- P1: gated = rn(sigmoid(g) o) over g (all loads first, then the math and the stores)
      {
        float gv[16], ov[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) { gv[j] = lds_bf16(ea(st + O_G, j)); ov[j] = lds_bf16(ea(st + O_O, j)); }
#pragma unroll
        for (int j = 0; j < 16; ++j) sts_bf16(ea(st + O_G, j), sigm(gv[j]) * ov[j]);
      }
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) mbar_arrive(&B.gfull);
      if (tl0) TR(T, 2);
      // ---- P2: q1 = rn(q + rn(gate_a) att) over q (kept in registers), row RMS, y
      mbar_wait_h(&B.attfull, T & 1);
      if (tl0) TR(T, 3);
      tc_fence_after();
      float q1[16];
      {
        uint32_t av[16];
        tmem_ld16(tl + T_ATT, av);
        float qv[16], gav[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) { qv[j] = lds_bf16(ea(st + O_Q, j)); gav[j] = modv(0, atj(j)); }
        tmem_wait_ld();
        float v[32];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const float att = rnb(__uint_as_float(av[j]));
          q1[j] = rnb(qv[j] + rnb(gav[j]) * att);
          sts_bf16(ea(st + O_Q, j), q1[j]);
          if (save) sts_bf16(ea(st + O_ATT, j), att);
          v[j] = q1[j] * q1[j]; v[16 + j] = 0.f;
        }
        const float ps = rscatter32(v, lane);                              // lane l < 16: row 16 wg + l, this warp's 32 channels
        if (lane < 16) red[(wg * 4 + qw) * 16 + lane] = ps;
      }
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane < 16) {
        const float ss = red[(wg * 4) * 16 + lane] + red[(wg * 4 + 1) * 16 + lane] + red[(wg * 4 + 2) * 16 + lane] + red[(wg * 4 + 3) * 16 + lane];
        rs[wg * 16 + lane] = rsqrtf(ss * (1.f / C) + eps);
      }
      named_bar_sync(1 + wg, 128);
      {
        float scv[16], shv[16], rv[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) { const int at = atj(j); scv[j] = modv(2, at); shv[j] = modv(1, at); rv[j] = rs[wg * 16 + j]; }
#pragma unroll
        for (int j = 0; j < 16; ++j) sts_bf16(ea(st + O_Y, j), q1[j] * rv[j] * (1.f + scv[j]) + shv[j]);
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) mbar_arrive(&B.yfull);
      if (tl0) TR(T, 4);
      // ---- P3: h = rn(silu(a) b) for hidden unit 128 hc + c
      for (int hc = 0; hc < 2; ++hc) {
        mbar_wait_h(&B.abfull, (2 * T + hc) & 1);
        if (tl0) TR(T, 5 + hc);
        tc_fence_after();
        uint32_t aa[16], bb[16];
        tmem_ld16(tl + (hc ? T_ATT : T_A), aa); tmem_ld16(tl + (hc ? T_FF : T_B), bb);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.abfree);
        const int h = 128 * hc + c;
        const uint32_t hofs = (uint32_t)(h >> 6) * KBLK + (uint32_t)(h & 7) * 2, hch = (uint32_t)((h & 63) >> 3);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int r = 16 * wg + j;
          const float a = __uint_as_float(aa[j]);
          sts_bf16(st + O_H + hofs + (uint32_t)r * 128u + ((hch ^ (uint32_t)(r & 7)) << 4), a * sigm(a) * __uint_as_float(bb[j]));
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) mbar_arrive(&B.hfull);
      if (tl0) TR(T, 7);
      // ---- P4: out = rn(q1 + rn(gate_f) ffn) over o
      {
        float gfv[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) gfv[j] = modv(3, atj(j));
        mbar_wait_h(&B.ffull, T & 1);
        if (tl0) TR(T, 8);
        tc_fence_after();
        uint32_t fv[16];
        tmem_ld16(tl + T_FF, fv);
        tmem_wait_ld();
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const float ff = rnb(__uint_as_float(fv[j]));
          if (save) sts_bf16(ea(st + O_FF, j), ff);
          sts_bf16(ea(st + O_O, j), q1[j] + rnb(gfv[j]) * ff);
        }
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) mbar_arrive(&B.ofull);
      if (tl0) TR(T, 9);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
