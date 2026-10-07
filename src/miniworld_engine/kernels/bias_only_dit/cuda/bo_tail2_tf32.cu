// bo_tail2_tf32.cu -- K3 of the bias-only token DiT's three-kernel fp32 (TF32) inference step, the PAIR form (round 10, A/B switch
// MINIWORLD_BIAS_ONLY_DIT_INF3_2CTA=1): the same block tail as bo_tail_tf32.cu,
//
//   y = a Wo^T,  x1 = x + gate1 y,  xt = LN(x1) s2 + sh2 (TF32),  a|b = xt [Wa; Wb]^T,  h = silu(a) b (TF32),  out = x1 + gate2 h Wsq^T
//
// on tcgen05 cta_group::2: a cluster of 8 CTAs carries TWO 128-row tiles (2 p, 2 p + 1) x 4 column groups. Cluster rank 2 g + s:
// column group g (output columns [192 g, + 192), hidden units [384 g, + 384)), tile s of the pair. The two CTAs of a column group
// (ranks 2 g, 2 g + 1) are a 2-CTA pair: the leader (s = 0) issues every product as M = 256 (its tile's 128 rows and the peer's), B
// split by N -- each CTA loads its own A rows and HALF of every weight box:
//   y  N = 192: leader Wo rows 192 g + [0, 96), peer + [96, 192) (the CL 8 pair-packed Wo, indexed by cluster rank)
//   a|b N = 256 per pass: leader Wa rows 384 g + 128 pass + [0, 128) (TMEM a), peer the same Wb rows (TMEM b); three passes
//   z  N = 192: as y, with the CL 8 pair-packed Wsq
// Why (round-10 estimate from the round-9 trace): at CL 8, A = 5, L640 / L768 take 25 / 30 tiles = two rounds of the 15 resident
// clusters of 8 (tail 84-86 us). Here a cluster of 8 is 2 tiles, so 30 tiles are ONE round of 15; per CTA twice the columns but half
// of each weight, ~4.8 MB of TMA in (xt read three times) and ~59-68 us. A = 5, L <= 512 keeps CL 8 / CL 6 (no round saved).
//
// x [M, 768], out [M, 768], TAB [T, 6, 768] (0 gate1, 1 gate2, 2 s1, 3 s2, 4 sh1, 5 sh2; tok = row % T, T a multiple of 128).
// Scratch, padded to an even number of tiles: XT [2 ceil(tiles / 2) 128, 768] rows (row stride xstride), H blocked k-block-major
// [tile][48][128][32]. A missing odd tile (tiles odd: the last cluster's s = 1 CTAs) loads the last real tile's inputs, writes its
// own padding rows of XT / H, and stores no output.
//
// Exchanges, among the 4 CTAs of a tile (ranks 2 g' + s): the LN statistics through DSMEM (Chan); xt and h through L2 with st.global
// and one release (xtready / hready, count 4); the A producers wait once and stream the operand. Pair synchronisation (the leader's
// barriers, count 2: one local and one remote arrival): p4done (both CTAs staged their own xt blocks and read Y / the s2 boxes),
// abfree (both read a | b of a pass out of TMEM), zfree (both read the last pass). The leader's MMA commits arrive on both CTAs'
// barriers (tcgen05.commit.cta_group::2 ... multicast, mask 3 << 2 g): aempty / wempty (slot reuse), ydone, abdone, zdone.
//
// A ring (NAR = 6 x 16 KB, own rows only), one sequence in both CTAs of a pair (same position -> same slot):
//   [0, NKA)            a k-blocks                     MMA (pair: leader barrier, both CTAs' boxes)
//   [I_X, +2 KO)        x_q, gate1_q                   own epilogue, P2   (KO = 6: twelve boxes -- twice CL 8's)
//   [I_S, +2 KO)        s2_q, sh2_q                    own epilogue, P4
//   [I_XT, +3 x 24)     xt k-blocks, pass 0 1 2        MMA. Pass 0 starts with the own 6 blocks (staged by P4, never loaded)
//   [I_H, +48)          h k-blocks, two per TMA box    MMA, after hready
//   [I_G, +KO)          gate2_q                        own epilogue, P8
// Barrier phases: afull / aempty of a slot complete once per position in BOTH CTAs (the peer arrives on its own afull for a pair
// position, whose data lands on the leader's; aempty gets the leader's multicast commit for MMA positions, the own epilogue's
// release for epilogue positions). A pair position's TMA is never issued while its slot's previous position is an epilogue box
// (static_assert below): otherwise a fast peer's bytes could land on the leader's barrier while it still counts that box.
// W ring (NWS = 4 x 24 KB): Wo / Wsq: two k-blocks of the CTA's 96 rows as one pair-packed box (24 KB); Wab: [128][32] (16 KB).
// Shared memory: A ring 96 KB | W ring 96 KB | statistics 4 + 2 KB | h transpose tiles 16 KB | barriers -> 219648 B.
// TMEM (512, cta_group::2 allocation in both CTAs): Y [0, 192) | a [192, 320) | b [320, 448); Z = [192, 384) after the last pass.
// Phases: P1 y | P2 x1 = x + gate1 y -> Y, statistics | P3 all-reduce | P4 xt per block, staged into the own xt slots as their s2 /
// sh2 boxes are consumed (at most 4 blocks in registers), st.global to XT, p4done + xtready | P5 three a | b passes | P6 per pass h
// -> H (st.global through transpose tiles), abfree / after the last zfree + hready | P7 z | P8 out = x1 + gate2 z.
// PDL as bo_tail_tf32.cu. Bit-identical reruns: fixed-order reductions, no atomics. -DTRACE: the tail's g_trace layout (below).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DATT
#define DATT 768
#endif
static_assert(DATT == 768 || DATT == 1024, "768 or 1024 attention channels");

DEVI uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI float silu32(float a) { return a * rcpf(1.f + ex2f(-1.4426950408889634f * a)); }
DEVI uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
DEVI void push_bulk(uint32_t dst_local, uint32_t src, uint32_t bytes, uint32_t bar_local, uint32_t rank) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(mapa(dst_local, rank)), "r"(src), "r"(bytes), "r"(mapa(bar_local, rank)) : "memory");
}
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI void fence_acq_rel_cluster() { asm volatile("fence.acq_rel.cluster;" ::: "memory"); }
DEVI void umma_ss2_tf32(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
DEVI void tmem_ld16w(uint32_t taddr, uint32_t (&r)[16]) {    // tcgen05.ld fused with its wait (one asm statement)
  asm volatile("{\n\ttcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n\t"
               "tcgen05.wait::ld.sync.aligned;\n\t}"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
                 "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEVI void u4x4(const uint32_t (&v)[16], int k, uint4& o) { o = make_uint4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]); }

// -DTRACE: g_trace[16 CTAs][2048] %globaltimer ns (CTA 0 = the leader of column group 0, tile 0). Markers 0 .. 255: 0 start |
// epilogue (tid 128): 1 ydone seen, 2 P2 done, 3 stats pushed, 4 xch seen, 5 xt stored + p4done arrived, 6 xtready signalled,
// 7 / 10 / 11 abdone seen (pass 0 / 1 / 2), 20 / 21 / 22 P6 of pass 0 / 1 / 2 done (abfree / zfree arrived), 8 h stored, 9 hready
// signalled, 13 zdone seen, 14 P8 done | A producer: 15 pdl_wait passed, 16 xtready seen, 17 hready seen | MMA (leader): 23 / 24 /
// 25 pass 0 / 1 / 2 may start (p4done / abfree passed) -- the P6 bubble of pass p is 24 + p - (10 or 11) -- 26 zfree passed.
// Per ring position i: 256 + i seen (MMA warp; for the epilogue's x / gate1 / s2 / sh2 / gate2 boxes: the epilogue, tid 128),
// 512 + i load issued, 768 + i wait begun, 1792 + i the producer saw the slot free. Per W slot wj: 1024 / 1280 / 1536 + wj (seen,
// issued, wait begun; the leader's MMA warp and each CTA's producer).
#ifdef TRACE
__device__ unsigned long long g_trace[16 * 2048];
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TR(i) do { if (blockIdx.x < 16 && (i) < 2048) g_trace[blockIdx.x * 2048 + (i)] = gtime(); } while (0)
#else
#define TR(i) do { } while (0)
#endif
constexpr int EV_AS = 256, EV_AI = 512, EV_AW = 768, EV_WS = 1024, EV_WI = 1280, EV_WW = 1536, EV_AR = 1792;

constexpr int D = 768, QM = 128, NCL = 8, CLT = 4;            // cluster CTAs; CTAs per tile
constexpr int NC = D / CLT, KO = NC / 32, NH = 1536 / CLT, HO = NH / 32, NP = 3, NHP = NH / NP, HOP = NHP / 32;
constexpr int NHALF = NC / 2;                                  // the CTA's half of a y / z B box: 96 rows
constexpr int NKA = DATT / 32, NXT = D / 32, NHK = 1536 / 32;
constexpr int I_X = NKA, I_S = I_X + 2 * KO, I_XT = I_S + 2 * KO, I_H = I_XT + NP * NXT, I_G = I_H + NHK, NI = I_G + KO;
constexpr int SLOT = QM * 128, NAR = 6;
constexpr int WSLOT = 2 * NHALF * 128, WHALF = NHALF * 128, WABB = NHP * 128, NWS = 4;
constexpr int O_A = 0, O_W = NAR * SLOT, O_RED = O_W + NWS * WSLOT, O_LOC = O_RED + CLT * 1024, O_TILE = O_LOC + 2048;
constexpr int O_BAR = O_TILE + 8 * 2048, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(O_W % 1024 == 0 && WSLOT % 1024 == 0 && WHALF % 1024 == 0 && O_TILE % 1024 == 0 && WABB <= WSLOT, "alignment");
static_assert(NKA % 2 == 0 && KO <= NAR && NI <= 256 && I_H % 2 == 0 && NAR % 2 == 0 && NHK % 2 == 0, "tiling");
static_assert(NP * NHP == NH && HOP * 32 == NHP, "passes");
constexpr uint32_t T_Y = 0, T_AB = NC, T_Z = NC;               // a: [T_AB, + 128), b: [T_AB + 128, + 128)
static_assert(T_AB + 2 * NHP <= 512 && T_Z + NC <= T_AB + 2 * NHP, "TMEM");
constexpr uint32_t I_Y = idesc_tf32(2 * QM, NC), I_AB = idesc_tf32(2 * QM, 2 * NHP);

__host__ __device__ constexpr bool mma_pos(int i) { return i < NKA || (i >= I_XT && i < I_G); }
__host__ __device__ constexpr bool own_pos(int i) { return i >= I_XT && i < I_XT + KO; }
// every pair (TMA) position's slot predecessor is an MMA position (committed by the leader, so the leader's barrier phase is over)
__host__ __device__ constexpr bool pair_slots_safe() {
  for (int i = NAR; i < NI; ++i)
    if (mma_pos(i) && !own_pos(i) && !mma_pos(i - NAR)) return false;
  return true;
}
static_assert(pair_slots_safe(), "a pair position after an epilogue box in the same slot");
// P4: own xt block q goes into the slot of position I_XT + q, whose previous box (position I_XT + q - 6) is an s2 / sh2 box of
// block (pp - I_S) / 2 or an x / gate1 box (read in P2): stage q once that block of P4 is done
__host__ __device__ constexpr int stage_at(int q) {
  const int pp = I_XT + q - NAR, b = pp >= I_S ? (pp - I_S) / 2 : -1;
  return b > q ? b : q;
}
DEVI int kb_xt(int t, int g, int pass) {
  if (pass) return t;
  if (t < KO) return KO * g + t;
  const int j = t - KO;
  return j < KO * g ? j : j + KO;
}

struct Bars {
  uint64_t afull[NAR], aempty[NAR], wfull[NWS], wempty[NWS];
  uint64_t xch, xtready, hready, ydone, p4done, abdone, abfree, zfree, zdone;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

extern "C" __global__ void __launch_bounds__(384, 1)
bo_tail2_tf32_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mtab,
                    const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwab, const __grid_constant__ CUtensorMap mwsq,
                    const __grid_constant__ CUtensorMap mxt, const __grid_constant__ CUtensorMap mh,
                    float* __restrict__ OUT, float* __restrict__ XT, float* __restrict__ HP, int xstride, int T, int ntiles, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t crank = cluster_rank(), lead = crank & ~1u;
  const int g = (int)(crank >> 1), s = (int)(crank & 1u);
  const bool leader = s == 0;
  const uint16_t pmask = (uint16_t)(3u << (2 * g));
  const int tile = 2 * (int)(blockIdx.x / NCL) + s;            // scratch rows: padded to the pair
  const bool valid = tile < ntiles;
  const int m0 = tile * QM, min_ = (valid ? tile : ntiles - 1) * QM, tr0 = min_ % T;   // min_: the rows the inputs are read from
  auto peer = [&](int gg) { return (uint32_t)(2 * gg + s); };  // the CTA of this tile in column group gg
  auto h_elem = [&](int kb, int row, int col) -> float* { return HP + ((size_t)(tile * NHK + kb) * QM + row) * 32 + col; };
  auto h_row = [&](int kb) { return (tile * NHK + kb) * QM; };

  if (tid == 0) {
    for (int i = 0; i < NAR; ++i) { mbar_init(&B.afull[i], 1); mbar_init(&B.aempty[i], 1); }
    for (int i = 0; i < NWS; ++i) { mbar_init(&B.wfull[i], 1); mbar_init(&B.wempty[i], 1); }
    mbar_init(&B.xch, 1);
    mbar_init(&B.xtready, CLT); mbar_init(&B.hready, CLT);
    mbar_init(&B.ydone, 1); mbar_init(&B.abdone, 1); mbar_init(&B.zdone, 1);
    mbar_init(&B.p4done, 2); mbar_init(&B.abfree, 2); mbar_init(&B.zfree, 2);   // the pair's two CTAs (used in the leader)
    mbar_expect_tx(&B.xch, (CLT - 1) * 1024);                  // the tile peers' statistics
    fence_barrier_init();
    prefetch_map(&ma); prefetch_map(&mx); prefetch_map(&mtab); prefetch_map(&mwo); prefetch_map(&mwab); prefetch_map(&mwsq);
    prefetch_map(&mxt); prefetch_map(&mh);
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();
  if (tid == 0) TR(0);

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ A producer (both CTAs)
    if (lane == 0) {
      pdl_wait();                                              // a: the core's output; x: the previous block's
      TR(15);
      const uint64_t once = pol_evict_first();                 // x / table boxes: read once
      for (int i = 0; i < NI; ++i) {
        const int slot = i % NAR;
        if (own_pos(i)) continue;                              // staged by the epilogue
        if (i == I_XT + KO) { mbar_wait_cl(&B.xtready, 0); TR(16); fence_proxy_async_global(); }
        if (i == I_H) { mbar_wait_cl(&B.hready, 0); TR(17); fence_proxy_async_global(); }
        const uint32_t dst = su + O_A + slot * SLOT;
        if (i >= I_H && i < I_G) {                             // two h k-blocks per box into slots slot, slot + 1
          if ((i - I_H) & 1) continue;
          if (i >= NAR) { mbar_wait(&B.aempty[slot], ((i / NAR) - 1) & 1); mbar_wait(&B.aempty[slot + 1], (((i + 1) / NAR) - 1) & 1); }
          TR(EV_AR + i);
          if (leader) mbar_expect_tx(&B.afull[slot], 2 * 2 * SLOT);   // both CTAs' 32-KB boxes land on the leader's barrier
          tma_load_2d_2sm(dst, &mh, &B.afull[slot], 0, h_row(i - I_H));
          if (!leader) mbar_arrive(&B.afull[slot]);            // the peer's own phase of this position
          mbar_arrive(&B.afull[slot + 1]);
          TR(EV_AI + i);
          continue;
        }
        if (i >= NAR) mbar_wait(&B.aempty[slot], ((i / NAR) - 1) & 1);
        TR(EV_AR + i);
        if (mma_pos(i)) {                                      // a / xt: the pair's product operand, own rows
          if (leader) mbar_expect_tx(&B.afull[slot], 2 * SLOT);
          if (i < NKA) {
            tma_load_2d_2sm(dst, &ma, &B.afull[slot], 32 * i, min_);
          } else {
            const int t = (i - I_XT) % NXT, pass = (i - I_XT) / NXT;
            tma_load_2d_2sm(dst, &mxt, &B.afull[slot], 32 * kb_xt(t, g, pass), m0);
          }
          if (!leader) mbar_arrive(&B.afull[slot]);
        } else {                                               // the own epilogue's boxes
          mbar_expect_tx(&B.afull[slot], SLOT);
          if (i < I_S) {                                       // x_q (even), gate1_q (odd)
            const int q = (i - I_X) >> 1, col = g * NC + 32 * q;
            if (((i - I_X) & 1) == 0) tma_load_2d_h(dst, &mx, &B.afull[slot], col, min_, once);
            else tma_load_2d_h(dst, &mtab, &B.afull[slot], 0 * D + col, tr0, once);
          } else if (i < I_XT) {                               // s2_q (even), sh2_q (odd)
            const int q = (i - I_S) >> 1, col = g * NC + 32 * q;
            tma_load_2d_h(dst, &mtab, &B.afull[slot], (((i - I_S) & 1) ? 5 : 3) * D + col, tr0, once);
          } else {                                             // gate2_q
            tma_load_2d_h(dst, &mtab, &B.afull[slot], 1 * D + g * NC + 32 * (i - I_G), tr0, once);
          }
        }
        TR(EV_AI + i);
      }
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ W producer (both CTAs: own half)
    if (lane == 0) {
      int wj = 0;
      auto slot = [&](uint32_t bytes) -> uint32_t {
        const int ws = wj % NWS;
        if (wj >= NWS) mbar_wait(&B.wempty[ws], ((wj / NWS) - 1) & 1);
        if (leader) mbar_expect_tx(&B.wfull[ws], 2 * bytes);   // both halves land on the leader's barrier
        return (uint32_t)ws;
      };
      for (int s2 = 0; s2 < NKA / 2; ++s2, ++wj) {             // Wo: k-blocks 2 s2, 2 s2 + 1 of this CTA's 96 rows, one box
        const uint32_t ws = slot(WSLOT);
        tma_load_2d_2sm(su + O_W + ws * WSLOT, &mwo, &B.wfull[ws], 0, ((int)crank * (NKA / 2) + s2) * 2 * NHALF);
        TR(EV_WI + wj);
      }
      for (int pass = 0; pass < NP; ++pass)
        for (int t = 0; t < NXT; ++t, ++wj) {                  // Wab: leader the pass's a rows, peer its b rows
          const uint32_t ws = slot(WABB);
          tma_load_2d_2sm(su + O_W + ws * WSLOT, &mwab, &B.wfull[ws], 32 * kb_xt(t, g, pass), s * 1536 + g * NH + pass * NHP);
          TR(EV_WI + wj);
        }
      for (int pr = 0; pr < NHK / 2; ++pr, ++wj) {             // Wsq: k-blocks 2 pr, 2 pr + 1 of this CTA's 96 rows
        const uint32_t ws = slot(WSLOT);
        tma_load_2d_2sm(su + O_W + ws * WSLOT, &mwsq, &B.wfull[ws], 0, ((int)crank * (NHK / 2) + pr) * 2 * NHALF);
        TR(EV_WI + wj);
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer (the leader)
    if (leader) {
      int wj = 0;
      auto wwait = [&]() -> uint32_t {
        const int ws = wj % NWS;
        if (lane == 0) TR(EV_WW + wj);
        mbar_wait(&B.wfull[ws], (wj / NWS) & 1);
        tc_fence_after();
        if (lane == 0) TR(EV_WS + wj);
        return (uint32_t)ws;
      };
      auto await_ = [&](int i) -> uint32_t {
        const int slot = i % NAR;
        if (lane == 0) TR(EV_AW + i);
        mbar_wait(&B.afull[slot], (i / NAR) & 1);
        tc_fence_after();
        if (lane == 0) TR(EV_AS + i);
        return (uint32_t)slot;
      };
      // ---- P1: Y = a Wo^T, M = 256 (both tiles), N = 192
      for (int s2 = 0; s2 < NKA / 2; ++s2, ++wj) {
        const uint32_t ws = wwait();
        for (int h2 = 0; h2 < 2; ++h2) {
          const int i = 2 * s2 + h2;
          const uint32_t slot = await_(i);
          if (elect_one()) {
            const uint64_t da = desc_k128(su + O_A + slot * SLOT), dw = desc_k128(su + O_W + ws * WSLOT + h2 * WHALF);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) umma_ss2_tf32(tmem + T_Y, da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_Y, (i | ks) ? 1u : 0u);
            tc_commit2_mc(&B.aempty[slot], pmask);
          }
          __syncwarp();
        }
        if (elect_one()) {
          tc_commit2_mc(&B.wempty[ws], pmask);
          if (s2 == NKA / 2 - 1) tc_commit2_mc(&B.ydone, pmask);
        }
        __syncwarp();
      }
      // ---- P5: [a | b] = xt [Wa; Wb]^T per pass of 128 hidden units, N = 256 (a from the leader's B half, b from the peer's)
      for (int pass = 0; pass < NP; ++pass) {
        if (pass == 0) mbar_wait_cl(&B.p4done, 0);             // both CTAs staged their own xt and read Y / the stash boxes
        else mbar_wait_cl(&B.abfree, (pass - 1) & 1);          // both CTAs read the previous pass out of TMEM
        tc_fence_after();
        if (lane == 0) TR(23 + pass);
        for (int t = 0; t < NXT; ++t, ++wj) {
          const uint32_t slot = await_(I_XT + pass * NXT + t);
          const uint32_t ws = wwait();
          if (elect_one()) {
            const uint64_t da = desc_k128(su + O_A + slot * SLOT), dw = desc_k128(su + O_W + ws * WSLOT);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ss2_tf32(tmem + T_AB, da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_AB, (t | ks) ? 1u : 0u);
            tc_commit2_mc(&B.wempty[ws], pmask);
            tc_commit2_mc(&B.aempty[slot], pmask);
            if (t == NXT - 1) tc_commit2_mc(&B.abdone, pmask);
          }
          __syncwarp();
        }
      }
      // ---- P7: Z = h Wsq^T, N = 192, over the last pass's columns once both CTAs read them
      mbar_wait_cl(&B.zfree, 0);
      tc_fence_after();
      if (lane == 0) TR(26);
      for (int pr = 0; pr < NHK / 2; ++pr, ++wj) {
        const uint32_t ws = wwait();
        for (int h2 = 0; h2 < 2; ++h2) {
          const uint32_t slot = await_(I_H + 2 * pr + h2);
          if (elect_one()) {
            const uint64_t da = desc_k128(su + O_A + slot * SLOT), dw = desc_k128(su + O_W + ws * WSLOT + h2 * WHALF);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ss2_tf32(tmem + T_Z, da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_Y, (pr | h2 | ks) ? 1u : 0u);
            tc_commit2_mc(&B.aempty[slot], pmask);
          }
          __syncwarp();
        }
        if (elect_one()) {
          tc_commit2_mc(&B.wempty[ws], pmask);
          if (pr == NHK / 2 - 1) tc_commit2_mc(&B.zdone, pmask);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ epilogue (both CTAs)
    pdl_wait();                                                // out / XT / H may still be read by earlier kernels
    const int q4 = warp & 3, hh = (warp - 4) >> 2;
    const uint32_t r = (uint32_t)(32 * q4 + lane), trow = tmem + ((uint32_t)(32 * q4) << 16);
    const uint32_t cofs = 16 * hh;
    const uint32_t rr0 = (uint32_t)(32 * q4) + (lane >> 2), qq = lane & 3;   // read-back: rows rr0 + 8 i, 16-B chunk qq of the half
    auto box = [&](int i) { return su + O_A + (uint32_t)(i % NAR) * SLOT; };
    auto bwait = [&](int i) {
      if (tid == 128) TR(EV_AW + i);
      mbar_wait(&B.afull[i % NAR], (i / NAR) & 1);
      if (tid == 128) TR(EV_AS + i);
    };
    auto release = [&](int i) { if (tid == 128) mbar_arrive(&B.aempty[i % NAR]); };
    auto pair_arrive = [&](uint64_t* bar) {                    // tid 128: the leader's pair barrier (count 2)
      if (leader) mbar_arrive(bar);
      else mbar_arrive_remote(bar, lead);
    };
    // ---- P2: x1 = x + gate1 y -> Y; the row's statistics over this thread's NC / 2 columns
    mbar_wait(&B.ydone, 0);
    tc_fence_after();
    if (tid == 128) TR(1);
    float n_ = 0.f, mean_ = 0.f, m2_ = 0.f;
#pragma unroll 1
    for (int q = 0; q < KO; ++q) {
      const int ix = I_X + 2 * q;
      bwait(ix);
      bwait(ix + 1);
      uint32_t yv[16];
      tmem_ld16w(trow + T_Y + 32 * q + cofs, yv);
      uint4 xs[4], gs[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) { xs[e] = lds128(box(ix) + sw128(r, 4 * hh + e)); gs[e] = lds128(box(ix + 1) + sw128(r, 4 * hh + e)); }
      float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t xw[4] = {xs[e].x, xs[e].y, xs[e].z, xs[e].w}, gw[4] = {gs[e].x, gs[e].y, gs[e].z, gs[e].w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float v = fmaf(__uint_as_float(gw[k]), __uint_as_float(yv[4 * e + k]), __uint_as_float(xw[k]));
          yv[4 * e + k] = __float_as_uint(v);
          s4[k] += v;
        }
      }
      const float pm = ((s4[0] + s4[1]) + (s4[2] + s4[3])) * (1.f / 16);
      float q4s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int k = 0; k < 16; ++k) { const float d = __uint_as_float(yv[k]) - pm; q4s[k & 3] += d * d; }
      const float pq = (q4s[0] + q4s[1]) + (q4s[2] + q4s[3]);
      const float nn = n_ + 16.f, dl = pm - mean_;
      mean_ += dl * (16.f / nn);
      m2_ += pq + dl * dl * (n_ * 16.f / nn);
      n_ = nn;
      tmem_st16(trow + T_Y + 32 * q + cofs, yv);               // x1 over y
      fence_proxy_async();                                     // the boxes' next TMA writes come after these reads
      named_bar_sync(1, 256);
      release(ix);
      release(ix + 1);
    }
    float* loc = reinterpret_cast<float*>(sm + O_LOC);         // [2 halves][2][128]
    loc[(2 * hh) * QM + r] = mean_;
    loc[(2 * hh + 1) * QM + r] = m2_;
    named_bar_sync(1, 256);
    if (tid == 128) TR(2);
    // ---- P3: (mean, M2) over the NC columns into red[g], pushed to the tile's 3 other CTAs; all 4 combined
    float* red = reinterpret_cast<float*>(sm + O_RED);         // [CLT][2][128]
    if (hh == 0) {
      const float ma_ = loc[r], qa = loc[QM + r], mb = loc[2 * QM + r], qb = loc[3 * QM + r], dl = mb - ma_;
      red[g * 2 * QM + r] = ma_ + 0.5f * dl;
      red[g * 2 * QM + QM + r] = qa + qb + dl * dl * (float)(NC / 4);   // n_a n_b / (n_a + n_b), n_a = n_b = NC / 2
      fence_proxy_async();
    }
    named_bar_sync(1, 256);
    if (tid == 128) {
      for (int p = 1; p < CLT; ++p) {
        const int gg = (g + p) % CLT;
        push_bulk(su + O_RED + g * 1024, su + O_RED + g * 1024, 1024, smem_u32(&B.xch), peer(gg));
      }
      TR(3);
    }
    mbar_wait_cl(&B.xch, 0);
    if (tid == 128) TR(4);
    float mean, rstd;
    {
      float mu = 0.f;
#pragma unroll
      for (int p = 0; p < CLT; ++p) mu += red[p * 2 * QM + r];
      mu *= 1.f / CLT;
      float M2 = 0.f;
#pragma unroll
      for (int p = 0; p < CLT; ++p) { const float d = red[p * 2 * QM + r] - mu; M2 += red[p * 2 * QM + QM + r] + (float)NC * d * d; }
      mean = mu;
      rstd = rsqrtf(M2 * (1.f / D) + eps);
    }
    // ---- P4: xt per block (its s2 / sh2 boxes released once read); own block q is staged into the slot of position I_XT + q as
    // soon as that slot's previous box is consumed (stage_at), then this warp's rows / half go to XT by st.global
    tmem_wait_st();
    uint32_t xr[KO][16];
#pragma unroll
    for (int b = 0; b < KO; ++b) {
      const int is = I_S + 2 * b;
      bwait(is);
      bwait(is + 1);
      tmem_ld16w(trow + T_Y + 32 * b + cofs, xr[b]);
      uint4 s4v[4], h4v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) { s4v[e] = lds128(box(is) + sw128(r, 4 * hh + e)); h4v[e] = lds128(box(is + 1) + sw128(r, 4 * hh + e)); }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t sw[4] = {s4v[e].x, s4v[e].y, s4v[e].z, s4v[e].w}, hw[4] = {h4v[e].x, h4v[e].y, h4v[e].z, h4v[e].w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
          xr[b][4 * e + k] = tf32r(fmaf((__uint_as_float(xr[b][4 * e + k]) - mean) * rstd, __uint_as_float(sw[k]), __uint_as_float(hw[k])));
      }
      fence_proxy_async();
      named_bar_sync(1, 256);                                  // block b's s2 / sh2 read by all
      release(is);
      release(is + 1);
#pragma unroll
      for (int q = 0; q < KO; ++q) {
        if (stage_at(q) != b) continue;
        const uint32_t dst = box(I_XT + q);
#pragma unroll
        for (int e = 0; e < 4; ++e) { uint4 o; u4x4(xr[q], e, o); sts128(dst + sw128(r, 4 * hh + e), o); }
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t rr = rr0 + 8 * i;
          stg128(XT + (size_t)(m0 + (int)rr) * xstride + g * NC + 32 * q + (int)cofs + 4 * (int)qq, lds128(dst + sw128(rr, 4 * hh + qq)));
        }
      }
    }
    fence_proxy_async_global();                                // this thread's XT stores -> the tile peers' TMA loads
    fence_proxy_async();                                       // the staged blocks -> the pair's MMA
    tc_fence_before();                                         // Y read out before the pair's MMAs go on (p4done)
    named_bar_sync(1, 256);
    if (tid == 128) {
      for (int q = 0; q < KO; ++q) mbar_arrive(&B.afull[(I_XT + q) % NAR]);   // the own xt positions (the leader's: the MMA waits)
      pair_arrive(&B.p4done);
      TR(5);
      fence_acq_rel_cluster();                                 // one release for the tile peers' relaxed arrivals
      for (int p = 0; p < CLT; ++p) mbar_arrive_remote_relaxed(&B.xtready, peer((g + p) % CLT));
      TR(6);
    }
    // ---- P6: h = silu(a) b per pass -> H (st.global through this warp's transpose tile)
    const uint32_t wtile = su + O_TILE + (uint32_t)(warp - 4) * 2048;
#pragma unroll 1
    for (int pass = 0; pass < NP; ++pass) {
      mbar_wait(&B.abdone, pass & 1);
      tc_fence_after();
      if (tid == 128) TR(pass == 0 ? 7 : pass == 1 ? 10 : 11);
#pragma unroll 1
      for (int j = 0; j < HOP; ++j) {
        uint32_t av[16], bv[16];
        tmem_ld16w(trow + T_AB + 32 * j + cofs, av);
        tmem_ld16w(trow + T_AB + NHP + 32 * j + cofs, bv);
#pragma unroll
        for (int k = 0; k < 16; ++k) av[k] = tf32r(silu32(__uint_as_float(av[k])) * __uint_as_float(bv[k]));
        const int kb = HO * g + pass * HOP + j;
#pragma unroll
        for (int e = 0; e < 4; ++e) { uint4 o; u4x4(av, e, o); sts128(wtile + sw64((uint32_t)lane, (uint32_t)e), o); }
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 4; ++i) {                          // rows 8 i .. 8 i + 7 of this warp's 32, 64 B each
          const uint32_t rr = 8 * i + (lane >> 2);
          stg128(h_elem(kb, 32 * q4 + (int)rr, (int)cofs + 4 * (int)qq), lds128(wtile + sw64(rr, qq)));
        }
        __syncwarp();
      }
      fence_proxy_async_global();                              // this thread's H stores, before the release below
      tc_fence_before();                                       // a | b of this pass read out: the pair's MMAs may overwrite them
      named_bar_sync(1, 256);
      if (tid == 128) {
        TR(20 + pass);
        if (pass + 1 < NP) {
          pair_arrive(&B.abfree);
        } else {
          pair_arrive(&B.zfree);
          TR(8);
          fence_acq_rel_cluster();
          for (int p = 0; p < CLT; ++p) mbar_arrive_remote_relaxed(&B.hready, peer((g + p) % CLT));
          TR(9);
        }
      }
    }
    // ---- P8: out = x1 + gate2 z, through this warp's [32 rows][64 B] tile (the h transpose tile: P6 is over) -> st.global.v4
    mbar_wait(&B.zdone, 0);
    tc_fence_after();
    if (tid == 128) TR(13);
#pragma unroll 1
    for (int q = 0; q < KO; ++q) {
      bwait(I_G + q);
      uint32_t zv[16], xv[16];
      tmem_ld16w(trow + T_Z + 32 * q + cofs, zv);
      tmem_ld16w(trow + T_Y + 32 * q + cofs, xv);
      uint4 g4[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) g4[e] = lds128(box(I_G + q) + sw128(r, 4 * hh + e));
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t gw[4] = {g4[e].x, g4[e].y, g4[e].z, g4[e].w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
          zv[4 * e + k] = __float_as_uint(fmaf(__uint_as_float(gw[k]), __uint_as_float(zv[4 * e + k]), __uint_as_float(xv[4 * e + k])));
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) { uint4 o; u4x4(zv, e, o); sts128(wtile + sw64((uint32_t)lane, (uint32_t)e), o); }
      __syncwarp();
      if (valid) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t rr = 8 * i + (lane >> 2);
          stg128(OUT + (size_t)(m0 + 32 * q4 + (int)rr) * D + g * NC + 32 * q + (int)cofs + 4 * (int)qq, lds128(wtile + sw64(rr, qq)));
        }
      }
      __syncwarp();
    }
    if (tid == 128) TR(14);
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();                                              // no CTA leaves while a pair / tile peer may still signal it
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}
