// ffn_fwd1t.cu — the SWA atom block's second forward stage on sm_100a for FEW tiles (inference, one modulation row per sequence row:
// A = 1), with the adaLN modulation inside. Same math and rounding points as mod_fwd.cu + ffn_fwd2.cu (the same bits):
//   a = rn(silu(c));  gate_a | shift_f | scale_f | gate_f = a Wmod[256:768]^T (fp32);
//   gated = rn(sigmoid(g) o);  att = rn(gated Wo^T);  q1 = rn(q + rn(gate_a) att);
//   y = rn(RMS(q1) (1 + scale_f) + shift_f);  a|b = y Wu^T;  h = rn(silu(a) b);  ffn = rn(h Wd^T);  out = rn(q1 + rn(gate_f) ffn)
// One 32-row tile per CTA (grid = tile count <= SMs; see qkvg_fwd1t.cu for why). Everything that does not depend on the attention runs
// before the wait for it (the kernel is launched while the attention kernel runs): a CTA takes ~83 GB/s through TMA whatever the source,
// so the 352 KB of weights are loaded in the order of need -- Wmod[256:768] (its slot: the bf16 A operand, in place) and the c tile, then
// the six 32-KB units of Wu / Wd through two staging slots into TMEM (tcgen05.cp of a unit under the next unit's load), then Wo -- and the
// modulation is made there too: a = silu(c) in place, four MMAs: gate_a^T / shift_f^T / scale_f^T into the a / b / ffn columns, gate_f^T
// into the att columns. After the wait: q, g, o; gate_f is read out (kept rounded in registers) before the att MMA reuses its columns,
// the q1 / y phase reads gate_a / shift_f / scale_f straight from TMEM (thread = channel = lane, column = row) before the a | b MMAs
// reuse theirs. y is written over the c tile. No fp32 [R, 768] modulation in HBM, no modulation launch.
// Contract: c is final when this kernel starts (it may be read before the wait): the kernel before the attention kernel triggers its
// dependents only after its own wait (qkvg_fwd1t.cu), and a kernel that does not trigger early lets this one start only when it is done.
// TMEM: Wu tiles at 0 / 64 / 128 / 192, Wd at 256 (128 columns), att^T 384, a^T 416, b^T 448, ffn^T 480.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;
#ifndef TRACE
#define TRACE 0
#endif
#define TRP(i, cond) do { if (TRACE && blockIdx.x == 0 && (cond)) TRb[i] = clock64(); } while (0)

constexpr int C = 128, AT = 32;
constexpr int KBLK = 4096;                                                 // [32 rows][64] bf16, SW128
constexpr int T_ = 2 * KBLK;                                               // one [32][128] tile
// Wmod[256:768] | Wo (first: staging slot B) | q -> q1 | g -> gated | o -> out | h (2 tiles) (q .. first h tile: staging slot A) | c -> a -> y
constexpr int O_WM = 0, O_WO = 4 * 32768, O_Q = O_WO + 2 * 16384, O_G = O_Q + T_, O_O = O_G + T_, O_H = O_O + T_, O_CT = O_H + 2 * T_, O_Y = O_CT;
constexpr int O_RED = O_CT + T_, O_RS = O_RED + 2 * 4 * 16 * 4, O_BAR = O_RS + 2 * 16 * 4;
constexpr int SMEM = O_BAR + 256;
static_assert(O_CT - O_Q >= 32768, "staging slot A in the tile area, the c tile outside it");
static_assert(SMEM <= 232448, "shared memory");
constexpr uint32_t T_WU = 0, T_WD = 256, T_ATT = 384, T_A = 416, T_B = 448, T_FF = 480;
constexpr uint32_t I_32 = idesc_bf16(128, 32);

struct Bars {
  uint64_t ufull[6], ufree[6], wofull, wmfull, cinfull, infull, cfull, modfull, gfull, attfull, yfull, abfull, hfull, ffull, ofull;
  uint32_t tmem;
};
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

// mq / mg / mo / mout: [C, S, N, 1] box (64, 32, 1, 1) SW128; mc: c [C, N S] box (64, 32) SW128; mwo: Wo [C, C] box (64, 128) SW128;
// mwu: Wu [C, 512] box (32, 128) SW64; mwd: Wd [256, C] box (32, 128) SW64; mwm: Wmod [C, 6C] box (32, 128) SW64.
extern "C" __global__ void __launch_bounds__(384, 1)
swa_ffn_fwd1t_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mo,
                    const __grid_constant__ CUtensorMap mc, const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mout,
                    const __grid_constant__ CUtensorMap mwu, const __grid_constant__ CUtensorMap mwd, const __grid_constant__ CUtensorMap mwm,
                    int S, int nab, float eps, unsigned long long* __restrict__ TRb) {
  extern __shared__ __align__(1024) uint8_t sm[];
  TRP(0, threadIdx.x == 0);
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int b = (int)blockIdx.x / nab, s0 = ((int)blockIdx.x % nab) * AT;
  auto wslot = [&](int u) -> uint32_t { return (u & 1) ? su + O_WO : su + O_Q; };   // staging slot of Wu / Wd unit u

  if (tid == 0) {
    for (int u = 0; u < 6; ++u) { mbar_init(&B.ufull[u], 1); mbar_init(&B.ufree[u], 1); }
    mbar_init(&B.wofull, 1); mbar_init(&B.wmfull, 1); mbar_init(&B.cinfull, 1); mbar_init(&B.infull, 1);
    mbar_init(&B.cfull, 2); mbar_init(&B.modfull, 1); mbar_init(&B.gfull, 2); mbar_init(&B.attfull, 1); mbar_init(&B.yfull, 2);
    mbar_init(&B.abfull, 1); mbar_init(&B.hfull, 2); mbar_init(&B.ffull, 1); mbar_init(&B.ofull, 2);
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  TRP(1, tid == 0);
  pdl_launch();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA loads (one per lane: a TMA
    // instruction costs its issuing thread ~50 ns)
    if (lane == 0) { mbar_expect_tx(&B.wmfull, 4 * 32768); mbar_expect_tx(&B.cinfull, T_); }   // Wmod[128 k ..], k = 2 .. 5; the c tile
    __syncwarp();
    if (lane < 16) tma_load_2d(su + O_WM + (lane >> 2) * 32768 + (lane & 3) * 8192, &mwm, &B.wmfull, 32 * (lane & 3), 128 * ((lane >> 2) + 2));
    else if (lane < 18) tma_load_2d(su + O_CT + (lane - 16) * KBLK, &mc, &B.cinfull, (lane - 16) * 64, b * S + s0);
    for (int u = 0; u < 6; ++u) {                                          // Wu (units 0 - 3) / Wd (4, 5): K-major SW64 [128][32] atoms
      if (lane == 0) {
        if (u >= 2) mbar_wait(&B.ufree[u - 2], 0);                         // the slot's previous unit is in TMEM
        mbar_expect_tx(&B.ufull[u], 32768);
      }
      __syncwarp();
      if (lane < 4) {
        if (u < 4) tma_load_2d(wslot(u) + lane * 8192, &mwu, &B.ufull[u], 32 * lane, 128 * u);
        else tma_load_2d(wslot(u) + lane * 8192, &mwd, &B.ufull[u], 128 * (u - 4) + 32 * lane, 0);
      }
    }
    if (lane == 0) {
      mbar_wait(&B.ufree[4], 0); mbar_wait(&B.ufree[5], 0);
      TRP(3, true);
      mbar_expect_tx(&B.wofull, 2 * 16384);
    }
    __syncwarp();
    if (lane < 2) tma_load_2d(su + O_WO + lane * 16384, &mwo, &B.wofull, lane * 64, 0);
    pdl_wait();                                                            // q, g, o come from the previous kernels
    TRP(16, lane == 0);
    if (lane == 0) mbar_expect_tx(&B.infull, 3 * T_);
    __syncwarp();
    if (lane < 2) tma_load_4d(su + O_Q + lane * KBLK, &mq, &B.infull, lane * 64, s0, b, 0);
    else if (lane < 4) tma_load_4d(su + O_G + (lane - 2) * KBLK, &mg, &B.infull, (lane - 2) * 64, s0, b, 0);
    else if (lane < 6) tma_load_4d(su + O_O + (lane - 4) * KBLK, &mo, &B.infull, (lane - 4) * 64, s0, b, 0);
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    auto wdesc = [&](uint32_t base, int ks) { return desc_sw64(base + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2); };
    auto bdesc = [&](uint32_t base, int ks) { return desc_k128(base + (ks >> 2) * KBLK) + (uint64_t)((ks & 3) * 2); };
    mbar_wait(&B.wmfull, 0);
    TRP(4, lane == 0);
    mbar_wait(&B.cfull, 0);
    tc_fence_after();
    if (elect_one()) {                                                     // gate_a^T -> a cols, shift_f^T -> b, scale_f^T -> ffn, gate_f^T -> att
      const uint32_t slot[4] = {T_A, T_B, T_FF, T_ATT};
#pragma unroll
      for (int k = 0; k < 4; ++k)
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) umma_ss(tmem + slot[k], wdesc(su + O_WM + k * 32768, ks), bdesc(su + O_CT, ks), I_32, ks > 0 ? 1u : 0u);
      tc_commit(&B.modfull);
    }
    __syncwarp();
    for (int u = 0; u < 6; ++u) {                                          // Wu / Wd unit u -> TMEM (tcgen05.cp, 128 lanes x 32 B per K16 step)
      mbar_wait(&B.ufull[u], 0);
      tc_fence_after();
      if (elect_one()) {
        const uint32_t col = u < 4 ? T_WU + 64 * u : T_WD + 64 * (u - 4);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + col + ks * 8, wdesc(wslot(u), ks));
        tc_commit(&B.ufree[u]);
      }
      __syncwarp();
    }
    TRP(2, lane == 0);
    mbar_wait(&B.wofull, 0);
    mbar_wait(&B.gfull, 0);                                                // gated written over g; gate_f read out of the att columns
    tc_fence_after();
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < 8; ++ks)
        umma_ss(tmem + T_ATT, desc_k128(su + O_WO + (ks >> 2) * 16384) + (uint64_t)((ks & 3) * 2), bdesc(su + O_G, ks), I_32, ks > 0 ? 1u : 0u);
      tc_commit(&B.attfull);
    }
    __syncwarp();
    mbar_wait(&B.yfull, 0);                                                // y written; gate_a / shift_f / scale_f read out
    for (int hc = 0; hc < 2; ++hc) {
      tc_fence_after();
      const uint32_t ta = hc ? T_ATT : T_A, tb = hc ? T_FF : T_B;
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          umma_ts(tmem + ta, tmem + T_WU + 64 * hc + ks * 8, bdesc(su + O_Y, ks), I_32, ks > 0 ? 1u : 0u);
          umma_ts(tmem + tb, tmem + T_WU + 64 * (2 + hc) + ks * 8, bdesc(su + O_Y, ks), I_32, ks > 0 ? 1u : 0u);
        }
        tc_commit(&B.abfull);
      }
      __syncwarp();
    }
    mbar_wait(&B.hfull, 0);
    tc_fence_after();
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < 16; ++ks)
        umma_ts(tmem + T_FF, tmem + T_WD + ks * 8, bdesc(su + O_H + (ks >> 3) * 2 * KBLK, ks & 7), I_32, ks > 0 ? 1u : 0u);
      tc_commit(&B.ffull);
    }
    __syncwarp();
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA store: out
    pdl_wait();
    if (lane == 0) mbar_wait(&B.ofull, 0);
    __syncwarp();
    if (lane < 2) { tma_store_4d(&mout, su + O_O + lane * KBLK, lane * 64, s0, b, 0); tma_store_commit(); tma_store_wait0(); }
    TRP(15, lane == 0);
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ row-stage threads
    const int wg = (warp - 4) >> 2, qw = warp & 3;
    const uint32_t lb = (uint32_t)qw * 32;
    const int c = (int)lb + lane;                                          // channel (or hidden unit within a chunk for h)
    const uint32_t cofs = (uint32_t)(c >> 6) * KBLK + (uint32_t)(c & 7) * 2, cch = (uint32_t)((c & 63) >> 3);
    float* red = reinterpret_cast<float*>(sm + O_RED);                     // [2 wg][4 warps][16 rows]
    float* rs = reinterpret_cast<float*>(sm + O_RS);                       // [2 wg][16 rows] rstd
    const uint32_t tl = tmem + (lb << 16) + 16 * wg;                       // this thread's lane, the warpgroup's 16 columns (rows)
    auto ea = [&](uint32_t base, int j) {                                  // element (row 16 wg + j, channel c) of a row-major tile
      const int r = 16 * wg + j;
      return base + cofs + (uint32_t)r * 128u + ((cch ^ (uint32_t)(r & 7)) << 4);
    };
    // ---- a = rn(silu(c)) in place (the kit sigmoid, as mod_fwd.cu)
    mbar_wait(&B.cinfull, 0);
    {
      float cv[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) cv[j] = lds_bf16(ea(su + O_CT, j));
#pragma unroll
      for (int j = 0; j < 16; ++j) sts_bf16(ea(su + O_CT, j), cv[j] * sigmoid_kit(cv[j]));
    }
    fence_proxy_async();
    named_bar_sync(1 + wg, 128);
    if (qw == 0 && lane == 0) mbar_arrive(&B.cfull);
    TRP(6, tid == 128);
    // ---- rn(gate_f) out of the att columns (kept in registers: the att MMA reuses the columns)
    float gfr[16];
    {
      mbar_wait(&B.modfull, 0);
      TRP(7, tid == 128);
      tc_fence_after();
      uint32_t gf[16];
      tmem_ld16(tl + T_ATT, gf);
      tmem_wait_ld();
#pragma unroll
      for (int j = 0; j < 16; ++j) gfr[j] = rnb(__uint_as_float(gf[j]));
    }
    // ---- P1: gated = rn(sigmoid(g) o) over g
    mbar_wait(&B.infull, 0);
    TRP(5, tid == 128);
    {
      float gv[16], ov[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) { gv[j] = lds_bf16(ea(su + O_G, j)); ov[j] = lds_bf16(ea(su + O_O, j)); }
#pragma unroll
      for (int j = 0; j < 16; ++j) sts_bf16(ea(su + O_G, j), sigm(gv[j]) * ov[j]);
    }
    fence_proxy_async();
    tc_fence_before();
    named_bar_sync(1 + wg, 128);
    if (qw == 0 && lane == 0) mbar_arrive(&B.gfull);
    TRP(8, tid == 128);
    // ---- P2: q1 = rn(q + rn(gate_a) att) over q (kept in registers), row RMS, y over the c tile
    mbar_wait(&B.attfull, 0);
    TRP(9, tid == 128);
    tc_fence_after();
    float q1[16];
    {
      uint32_t av[16], ga[16];
      tmem_ld16(tl + T_ATT, av); tmem_ld16(tl + T_A, ga);
      float qv[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) qv[j] = lds_bf16(ea(su + O_Q, j));
      tmem_wait_ld();
      float v[32];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float att = rnb(__uint_as_float(av[j]));
        q1[j] = rnb(qv[j] + rnb(__uint_as_float(ga[j])) * att);
        v[j] = q1[j] * q1[j]; v[16 + j] = 0.f;
      }
      const float ps = rscatter32(v, lane);                                // lane l < 16: row 16 wg + l, this warp's 32 channels
      if (lane < 16) red[(wg * 4 + qw) * 16 + lane] = ps;
    }
    named_bar_sync(1 + wg, 128);
    if (qw == 0 && lane < 16) {
      const float ss = red[(wg * 4) * 16 + lane] + red[(wg * 4 + 1) * 16 + lane] + red[(wg * 4 + 2) * 16 + lane] + red[(wg * 4 + 3) * 16 + lane];
      rs[wg * 16 + lane] = rsqrtf(ss * (1.f / C) + eps);
    }
    named_bar_sync(1 + wg, 128);
    {
      uint32_t shv[16], scv[16];
      tmem_ld16(tl + T_B, shv); tmem_ld16(tl + T_FF, scv);
      tmem_wait_ld();
      float rv[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) rv[j] = rs[wg * 16 + j];
#pragma unroll
      for (int j = 0; j < 16; ++j) sts_bf16(ea(su + O_Y, j), q1[j] * rv[j] * (1.f + __uint_as_float(scv[j])) + __uint_as_float(shv[j]));
    }
    fence_proxy_async();
    tc_fence_before();
    named_bar_sync(1 + wg, 128);
    if (qw == 0 && lane == 0) mbar_arrive(&B.yfull);
    TRP(10, tid == 128);
    // ---- P3: h = rn(silu(a) b) for hidden unit 128 hc + c
    for (int hc = 0; hc < 2; ++hc) {
      mbar_wait(&B.abfull, hc & 1);
      TRP(11, tid == 128 && hc == 0);
      tc_fence_after();
      uint32_t aa[16], bb[16];
      tmem_ld16(tl + (hc ? T_ATT : T_A), aa); tmem_ld16(tl + (hc ? T_FF : T_B), bb);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      const int h = 128 * hc + c;
      const uint32_t hofs = (uint32_t)(h >> 6) * KBLK + (uint32_t)(h & 7) * 2, hch = (uint32_t)((h & 63) >> 3);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int r = 16 * wg + j;
        const float a = __uint_as_float(aa[j]);
        sts_bf16(su + O_H + hofs + (uint32_t)r * 128u + ((hch ^ (uint32_t)(r & 7)) << 4), a * sigm(a) * __uint_as_float(bb[j]));
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + wg, 128);
    if (qw == 0 && lane == 0) mbar_arrive(&B.hfull);
    TRP(12, tid == 128);
    // ---- P4: out = rn(q1 + rn(gate_f) ffn) over o
    {
      mbar_wait(&B.ffull, 0);
      TRP(13, tid == 128);
      tc_fence_after();
      uint32_t fv[16];
      tmem_ld16(tl + T_FF, fv);
      tmem_wait_ld();
#pragma unroll
      for (int j = 0; j < 16; ++j) sts_bf16(ea(su + O_O, j), q1[j] + gfr[j] * rnb(__uint_as_float(fv[j])));
    }
    fence_proxy_async();
    tc_fence_before();
    named_bar_sync(1 + wg, 128);
    if (qw == 0 && lane == 0) mbar_arrive(&B.ofull);
    TRP(14, tid == 128);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
