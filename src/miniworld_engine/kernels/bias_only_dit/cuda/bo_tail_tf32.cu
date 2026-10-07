// bo_tail_tf32.cu -- K3 of the bias-only token DiT's three-kernel fp32 (TF32) inference step on sm_100a (MINIWORLD_BIAS_ONLY_DIT_INF3=1):
// everything after the attention core, in ONE kernel per 128-row tile:
//
//   y   = a Wo^T                              (a [M, DA] = sigmoid(g) (P v), from pv_gate_tf32 -DPDL_INF, TF32-valued)
//   x1  = x + gate1 y                         (x = the block's input, fp32 residual; gate1 = sigmoid(to_scale_attn(c)), hoisted)
//   xt  = LN(x1) s2 + sh2                     (rounded to TF32: the expand GEMM's operand)
//   a|b = xt [Wa; Wb]^T,  h = silu(a) b       (h rounded to TF32: the squeeze GEMM's operand)
//   z   = h Wsq^T,        out = x1 + gate2 z  (fp32)
//
// x [M, 768], out [M, 768], TAB [T, 6, 768] (one block's slice of the hoisted tables: 0 gate1, 1 gate2, 2 s1, 3 s2, 4 sh1, 5 sh2,
// sigmoids applied; tok = row % T); weights fp32 rounded to TF32 at pack time: WOP / WSQP = Wo / Wsq pair-packed for this CL (below),
// Wab = [Wa; Wb] [3072, 768]; XT [M, 768] rows / H blocked (below) fp32 scratch, L2-resident (also by pointer: XT, HP).
//
// A cluster of CL CTAs per 128-row tile (grid = CL x the row tiles); CTA c owns output columns [NC c, NC c + NC) of y, x1, xt, z, out
// (NC = 768 / CL) and hidden units [NH c, NH c + NH) of a, b, h (NH = 1536 / CL). A tile's activations are 384 KB fp32, a | b alone
// 3072 accumulator columns, and A = 5, L768 gives only 30 tiles: one CTA per tile would not fit and would leave 118 SMs idle.
//   CL = 8 (default): NC 96, NH 192; a | b in one pass. At most 15 clusters of 8 are resident (measured), so 16-30 tiles take two rounds.
//   CL = 6 (round 5, for 16..22 tiles -- A = 5, L512: 20 tiles = 120 CTAs, ONE round instead of two at CL 8): NC 128, NH 256; TMEM
//          Y 128 + a | b 256 + Z 128 = 512, so a | b runs in TWO passes of 128 hidden units each, re-streaming xt from L2 (its W
//          slots hold the pass's a and b rows: [128][32] + [128][32]); h is written by st.global through per-warp transpose tiles
//          (its 8 own blocks do not fit the 6-slot ring) and z waits for the cluster's h before its first k-block.
//
// Exchanges inside the cluster: the LN statistics (1 KB per CTA, pushed into every peer's shared memory, Chan's combination); xt and
// h, the next GEMM's full-K A operand, written by their owners to XT / H, then ONE release per exchange (stores complete,
// fence.proxy.async.global, fence.acq_rel.cluster, CL relaxed remote arrivals on xtready / hready; CL 8: st.global, below); the A producer waits once
// (acquire.cluster) and streams the operand from L2 like any TMA-fed GEMM. Round 1's DSMEM push ring (one round trip per k-block,
// ~1.3 us each) was deleted.
//
// The A ring (NAR = 6 x 16 KB, [128 rows][32 fp32] SW128 boxes) carries one sequence (position i, slot i % 6):
//   [0, NKA)             a k-blocks                         MMA, P1 (y)
//   [I_X, +2 KO)         x_q, gate1_q                       epilogue, P2 (released per 32-column block)
//   [I_S, +2 KO)         s2_q, sh2_q                        epilogue, P4 (released per 32-column block)
//   [I_XT, +NPASS 24)    xt k-blocks, per pass              MMA, P5. Pass 0 starts with this CTA's own KO blocks, staged in their
//                                                           ring slots by P4 itself (never loaded); the rest load after xtready.
//   [I_H, +48)           h k-blocks                         MMA, P7. CL 8: the own 6 first, staged by P6 (never loaded); the rest
//                                                           after hready. CL 6: all after hready.
//   [I_G, +KO)           gate2_q                            epilogue, P8 (they load as the last h blocks retire)
// The ring stalls only where it waits for xtready / hready, and no CTA signals before its own stores have completed -- so staging
// tiles may live in ring slots (a TMA load never overwrites one still being stored).
// W ring (NWS slots of WSLOT = 2 NC x 128 B): Wo / Wsq slots hold TWO k-blocks as ONE pair-packed TMA box (CL 8: 24 KB, CL 6: 32 KB);
// a Wab slot one [192][32] half of a k-block (CL 8) or the pass's [128][32] a rows + [128][32] b rows (CL 6).
//   pair-packed W [768, K] -> P [(CL (K / 64) 2) NC, 32]:  P[((c (K / 64) + p) 2 + h) NC + n, k] = W[NC c + n, 64 p + 32 h + k]
// H is BLOCKED k-block-major -- [row tile][48 k-blocks][128 rows][32], every k-block a contiguous 16 KB -- and the z GEMM's L2 h
// blocks load TWO per TMA box (one 32-KB box into two adjacent ring slots; the second slot's afull completed by a plain arrive):
// round 6, CL 8, L768: z's H stall 7.6 -> 3.7 us, the tile 50.3 -> 46.2 us. XT is row-major (row stride xstride). Deleted after
// round 7's A/B (each lost at every L): row-major H, blocked XT with paired xt loads, the weight prefetch into L2.
// Round 8 loaded x / gate1 / s2 / sh2 with per-thread ld.global while y ran (into a TMEM stash): the y GEMM's TMA intake collapsed
// (a period 0.25 -> 0.96 us, Wo 0.57 -> 2.4 us per pair; tile 46.3 -> 66.7 us at L768) -- reverted: they come through the ring.
// Round 9 put round 8's other changes behind switches, one A/B each (whole step, round 7 = base). Kept: at CL 8 (XSTG) the own
// xt (P4) and h (P6) blocks go to XT / H with st.global, read back from the staged slots (8 rows x 64 B per warp instruction), then
// one release -- instead of TMA store + cp.async.bulk.wait_group 0 (2.3-3 us from issue to completion, queued in the TMA unit
// behind the W boxes): tile 46.9 -> 45.0 us at L768, whole step -1.7 .. -3.7 us at L256-768 except L512. CL 6 keeps the TMA store
// (L512 tile 60.5 -> 61.7 with st.global). Not kept: z's own h blocks under P6 (+0.3 .. +0.8 us), launch_dependents as the first
// instruction (within +-0.4 us).//
// Phases per tile and CTA (one tile per CTA):
//   P1  y over NKA a k-blocks -> TMEM Y
//   P2  x1 = x + gate1 y -> TMEM Y (over y); the row's statistics over this thread's NC / 2 columns
//   P3  statistics: own (mean, M2) into red[c], pushed to the CL - 1 peers' red[c]; all CL combined
//   P4  xt = (x1 - mean) rstd s2 + sh2 (all KO blocks in registers), staged in the ring slots of positions I_XT .., TMA-stored
//       (CL 8: st.global), xtready
//   P5  a | b over the 24 xt k-blocks per pass -> TMEM A | B
//   P6  h = silu(a) b per pass: CL 8 staged in ring slots, st.global; CL 6 st.global through transpose tiles;
//       hready after all
//   P7  z over the 48 h k-blocks -> TMEM Z
//   P8  out = x1 + gate2 z -> per-warp [32][64 B] transpose tile (64-B swizzle) -> st.global.v4
// TMEM (512): CL 8: Y [0, 96) | A [96, 288) | B [288, 480) | Z = [96, 192) (after h_0..h_2 are read out); CL 6: Y [0, 128) | A [128, 256)
// | B [256, 384) | Z [384, 512).
// Warps: 0 A producer (lane 0); 1 MMA (whole warp waits, elect_one() issues); 2 TMEM allocator + W producer (lane 0); 3 idle; 4-11
// epilogue (thread = TMEM lane = tile row; warpgroup hh takes columns 16 hh .. 16 hh + 15 of every 32-column k-block).
//   shared memory  CL 8: A ring 96 KB | W ring 5 x 24 KB | statistics 8 + 2 KB | barriers -> 231936 B
//                  CL 6: A ring 96 KB | W ring 3 x 32 KB | statistics 6 + 2 KB | transpose tiles 16 KB | barriers -> 221696 B
// Per CTA and tile (DA 768): CL 8 132 MFLOP, TMA in 3.8 MB; CL 6 176 MFLOP, TMA in ~5.2 MB (xt twice).
// GEMM k-block orders (the W producer's): a | b pass 0 own xt k-blocks first, then the others in order; z (CL 8) own h k-blocks
// 6 c .. 6 c + 5 first, then 0 .. 47 without them (pairs stay aligned: 6 c is even).
// PDL: pdl_launch() after setup; the A producer pdl_wait()s before loading a and x; the epilogue before its first store; the W
// producer loads and prefetches weights before the wait (not written in the step). Bit-identical reruns: fixed-order reductions,
// no atomics. -DTRACE: %globaltimer per role into g_trace (layout below; bench_scripts/bo32_inf3_trace.py).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef CL
#define CL 8
#endif
#ifndef DATT
#define DATT 768
#endif
static_assert(CL == 8 || CL == 6, "cluster of 8 or 6");
#define XSTG (CL == 8)                                         // the own xt blocks to L2 by st.global (CL 6: TMA store)
static_assert(DATT == 768 || DATT == 1024, "768 or 1024 attention channels");

DEVI uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI float silu32(float a) { return a * rcpf(1.f + ex2f(-1.4426950408889634f * a)); }
DEVI void cluster_sync_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned; barrier.cluster.wait.aligned;" ::: "memory"); }
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
DEVI void tmem_ld16w(uint32_t taddr, uint32_t (&r)[16]) {    // tcgen05.ld fused with its wait (one asm statement)
  asm volatile("{\n\ttcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n\t"
               "tcgen05.wait::ld.sync.aligned;\n\t}"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
                 "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEVI void u4x4(const uint32_t (&v)[16], int k, uint4& o) { o = make_uint4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]); }

// -DTRACE: g_trace[16 CTAs][2048] %globaltimer ns. Markers 0 .. 255: 0 start | epilogue (tid 128): 1 ydone seen, 2 P2 done, 3 stats
// pushed, 4 xch seen, 5 xt stores issued, 6 xtready signalled, 7 abdone seen (pass 0), 10 abdone seen (pass 1), 8 h stores issued,
// 9 hready signalled, 13 zdone seen, 14 P8 done | A producer: 15 pdl_wait passed, 16 xtready seen, 17 hready seen.
// Per ring position i: 256 + i the MMA warp saw afull, 512 + i the producer issued the load, 768 + i the MMA warp began waiting.
// Per W slot wj: 1024 + wj seen, 1280 + wj issued, 1536 + wj the MMA warp began waiting. (wait begin -> seen > 0: that stream stalled
// the MMAs.) 1792 + i: the A producer saw position i's slot free (aempty) -- the commit of position i - 6, i.e. when the tensor
// core finished that block: release(i + 6) - max(seen(i), release(i + 5)) is the tensor core's time per block.
#ifdef TRACE
__device__ unsigned long long g_trace[16 * 2048];
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TR(i) do { if (blockIdx.x < 16 && (i) < 2048) g_trace[blockIdx.x * 2048 + (i)] = gtime(); } while (0)
#else
#define TR(i) do { } while (0)
#endif
constexpr int EV_AS = 256, EV_AI = 512, EV_AW = 768, EV_WS = 1024, EV_WI = 1280, EV_WW = 1536, EV_AR = 1792;

constexpr int D = 768, QM = 128, NC = D / CL, KO = NC / 32, NH = 1536 / CL, HO = NH / 32;
constexpr int NPASS = CL == 6 ? 2 : 1, NHP = NH / NPASS, HOP = NHP / 32;
constexpr bool OWNH = CL == 8;                                 // own h blocks staged in the ring and consumed first
constexpr int NKA = DATT / 32, NXT = D / 32, NHK = 1536 / 32;
constexpr int I_X = NKA, I_S = I_X + 2 * KO, I_XT = I_S + 2 * KO, I_H = I_XT + NPASS * NXT, I_G = I_H + NHK, NI = I_G + KO;
constexpr int SLOT = QM * 128, NAR = 6;
constexpr int WSLOT = 2 * NC * 128, WHALF = NC * 128, NWS = CL == 8 ? 5 : 3;
constexpr int TILEB = CL == 6 ? 8 * 2048 : 0;                  // CL 6: the h transpose tiles
constexpr int O_A = 0, O_W = NAR * SLOT, O_RED = O_W + NWS * WSLOT, O_LOC = O_RED + CL * 1024, O_TILE = O_LOC + 2048;
constexpr int O_BAR = O_TILE + TILEB, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(O_W % 1024 == 0 && WSLOT % 1024 == 0 && WHALF % 1024 == 0 && O_TILE % 1024 == 0, "1 KB alignment of the swizzled tiles");
static_assert(NKA % 2 == 0 && KO <= NAR && (!OWNH || HO <= NAR) && KO + 1 <= NAR && NI <= 256, "tiling");
static_assert(2 * NC <= 256 && NHP <= 256, "TMA box rows");
constexpr int H_L2 = I_H + (OWNH ? HO : 0);                    // the first h position loaded from L2 (pairs from here on)
static_assert(NAR % 2 == 0 && H_L2 % 2 == 0 && (I_G - H_L2) % 2 == 0, "h pairs: two positions in adjacent slots");
constexpr uint32_t T_Y = 0, T_A = NC, T_B = NC + NHP, T_Z = OWNH ? NC : NC + 2 * NHP;
static_assert((OWNH ? T_B + NHP : T_Z + NC) <= 512, "TMEM");
constexpr uint32_t I_N = idesc_tf32(QM, NC), I_HP = idesc_tf32(QM, NHP);
// the GEMMs' k-block orders: own blocks first where they are staged locally, then the others in order
DEVI int kb_xt(int t, int c, int pass) {
  if (pass) return t;                                          // CL 6 pass 1: every block from L2, in order
  if (t < KO) return KO * c + t;
  const int j = t - KO;
  return j < KO * c ? j : j + KO;
}
DEVI int kb_h(int t, int c) {
  if (!OWNH) return t;
  if (t < HO) return HO * c + t;
  const int j = t - HO;
  return j < HO * c ? j : j + HO;
}
DEVI bool own_pos(int i) { return (i >= I_XT && i < I_XT + KO) || (OWNH && i >= I_H && i < I_H + HO); }

struct Bars {
  uint64_t afull[NAR], aempty[NAR], wfull[NWS], wempty[NWS];
  uint64_t xch, xtready, hready, ydone, p4done, abdone, abfree, zfree, zdone;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

extern "C" __global__ void __launch_bounds__(384, 1)
bo_tail_tf32_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mtab,
                   const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwab, const __grid_constant__ CUtensorMap mwsq,
                   const __grid_constant__ CUtensorMap mxt, const __grid_constant__ CUtensorMap mh,
                   float* __restrict__ OUT, float* __restrict__ XT, float* __restrict__ HP, int xstride, int T, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t c = cluster_rank();
  const int rtile = (int)(blockIdx.x / CL), m0 = rtile * QM, tr0 = m0 % T;
  // H element (k-block kb, tile row row, column col of the k-block): blocked [row tile][48][128][32]; its TMA row coordinate
  auto h_elem = [&](int kb, int row, int col) -> float* { return HP + ((size_t)(rtile * NHK + kb) * QM + row) * 32 + col; };
  auto h_row = [&](int kb) { return (rtile * NHK + kb) * QM; };

  if (tid == 0) {
    for (int s = 0; s < NAR; ++s) { mbar_init(&B.afull[s], 1); mbar_init(&B.aempty[s], 1); }
    for (int s = 0; s < NWS; ++s) { mbar_init(&B.wfull[s], 1); mbar_init(&B.wempty[s], 1); }
    mbar_init(&B.xch, 1);
    mbar_init(&B.xtready, CL); mbar_init(&B.hready, CL);
    mbar_init(&B.ydone, 1); mbar_init(&B.p4done, 1); mbar_init(&B.abdone, 1); mbar_init(&B.abfree, 1);
    mbar_init(&B.zfree, 1); mbar_init(&B.zdone, 1);
    mbar_expect_tx(&B.xch, (CL - 1) * 1024);                   // the peers' statistics (this CTA's own are written in place)
    fence_barrier_init();
    prefetch_map(&ma); prefetch_map(&mx); prefetch_map(&mtab); prefetch_map(&mwo); prefetch_map(&mwab); prefetch_map(&mwsq);
    prefetch_map(&mxt); prefetch_map(&mh);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();
  if (tid == 0) TR(0);

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ A producer: the ring sequence
    if (lane == 0) {
      pdl_wait();                                              // a: the core's output; x: the previous block's
      TR(15);
      const uint64_t once = pol_evict_first();                 // x / table boxes: read once by this CTA
      for (int i = 0; i < NI; ++i) {
        const int slot = i % NAR;
        if (own_pos(i)) continue;                              // own blocks: staged here by the epilogue
        if (i == I_XT + KO) { mbar_wait_cl(&B.xtready, 0); TR(16); fence_proxy_async_global(); }
        if (i == H_L2) { mbar_wait_cl(&B.hready, 0); TR(17); fence_proxy_async_global(); }
        if (i >= H_L2 && i < I_G) {                            // two h k-blocks per TMA box into slots slot, slot + 1
          if ((i - H_L2) & 1) continue;                        // the pair's second position: loaded with the first
          if (i >= NAR) { mbar_wait(&B.aempty[slot], ((i / NAR) - 1) & 1); mbar_wait(&B.aempty[slot + 1], (((i + 1) / NAR) - 1) & 1); }
          TR(EV_AR + i);
          mbar_expect_tx(&B.afull[slot], 2 * SLOT);
          tma_load_2d(su + O_A + slot * SLOT, &mh, &B.afull[slot], 0, h_row(kb_h(i - I_H, (int)c)));
          mbar_arrive(&B.afull[slot + 1]);                     // phase only: the data lands with afull[slot]'s transaction
          TR(EV_AI + i);
          continue;
        }
        if (i >= NAR) mbar_wait(&B.aempty[slot], ((i / NAR) - 1) & 1);
        TR(EV_AR + i);
        mbar_expect_tx(&B.afull[slot], SLOT);
        const uint32_t dst = su + O_A + slot * SLOT;
        if (i < NKA) {
          tma_load_2d(dst, &ma, &B.afull[slot], 32 * i, m0);
        } else if (i < I_S) {                                  // x_q (even), gate1_q (odd)
          const int q = (i - I_X) >> 1, col = (int)c * NC + 32 * q;
          if (((i - I_X) & 1) == 0) tma_load_2d_h(dst, &mx, &B.afull[slot], col, m0, once);
          else tma_load_2d_h(dst, &mtab, &B.afull[slot], 0 * D + col, tr0, once);
        } else if (i < I_XT) {                                 // s2_q (even), sh2_q (odd)
          const int q = (i - I_S) >> 1, col = (int)c * NC + 32 * q;
          tma_load_2d_h(dst, &mtab, &B.afull[slot], (((i - I_S) & 1) ? 5 : 3) * D + col, tr0, once);
        } else if (i < I_H) {
          const int t = (i - I_XT) % NXT, pass = (i - I_XT) / NXT;
          tma_load_2d(dst, &mxt, &B.afull[slot], 32 * kb_xt(t, (int)c, pass), m0);
        } else {                                               // gate2_q
          tma_load_2d_h(dst, &mtab, &B.afull[slot], 1 * D + (int)c * NC + 32 * (i - I_G), tr0, once);
        }
        TR(EV_AI + i);
      }
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ W producer
    if (lane == 0) {
      int wj = 0;
      const uint64_t keep = pol_evict_last();                  // the weights: re-read by every tile and every replay
      auto slot = [&]() -> uint32_t {
        const int ws = wj % NWS;
        if (wj >= NWS) mbar_wait(&B.wempty[ws], ((wj / NWS) - 1) & 1);
        mbar_expect_tx(&B.wfull[ws], WSLOT);
        return (uint32_t)ws;
      };
      auto wab_row = [&](int pass, int hf) { return hf * 1536 + (int)c * NH + pass * NHP; };
      for (int s2 = 0; s2 < NKA / 2; ++s2, ++wj) {             // Wo: k-blocks 2 s2, 2 s2 + 1 as one pair-packed box
        const uint32_t ws = slot();
        tma_load_2d_h(su + O_W + ws * WSLOT, &mwo, &B.wfull[ws], 0, ((int)c * (NKA / 2) + s2) * 2 * NC, keep);
        TR(EV_WI + wj);
      }
      for (int pass = 0; pass < NPASS; ++pass)
        for (int t = 0; t < NXT; ++t) {                        // Wab per xt k-block: CL 8 a half, b half; CL 6 a + b in one slot
          const int kb = kb_xt(t, (int)c, pass);
          if constexpr (CL == 8) {
            for (int hf = 0; hf < 2; ++hf, ++wj) {
              const uint32_t ws = slot();
              tma_load_2d_h(su + O_W + ws * WSLOT, &mwab, &B.wfull[ws], 32 * kb, wab_row(pass, hf), keep);
              TR(EV_WI + wj);
            }
          } else {
            const uint32_t ws = slot(), dst = su + O_W + ws * WSLOT;
            tma_load_2d_h(dst, &mwab, &B.wfull[ws], 32 * kb, wab_row(pass, 0), keep);
            tma_load_2d_h(dst + WSLOT / 2, &mwab, &B.wfull[ws], 32 * kb, wab_row(pass, 1), keep);
            TR(EV_WI + wj);
            ++wj;
          }
        }
      for (int pr = 0; pr < NHK / 2; ++pr, ++wj) {             // Wsq: one pair-packed box per two k-blocks, in the h order
        const uint32_t ws = slot();
        tma_load_2d_h(su + O_W + ws * WSLOT, &mwsq, &B.wfull[ws], 0, ((int)c * (NHK / 2) + kb_h(2 * pr, (int)c) / 2) * 2 * NC, keep);
        TR(EV_WI + wj);
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
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
    // ---- P1: Y = a Wo^T
    for (int s2 = 0; s2 < NKA / 2; ++s2, ++wj) {
      const uint32_t ws = wwait();
      for (int h2 = 0; h2 < 2; ++h2) {
        const int i = 2 * s2 + h2;
        const uint32_t slot = await_(i);
        if (elect_one()) {
          const uint64_t da = desc_k128(su + O_A + slot * SLOT), dw = desc_k128(su + O_W + ws * WSLOT + h2 * WHALF);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) umma_ss_tf32(tmem + T_Y, da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_N, (i | ks) ? 1u : 0u);
          tc_commit(&B.aempty[slot]);
        }
        __syncwarp();
      }
      if (elect_one()) {
        tc_commit(&B.wempty[ws]);
        if (s2 == NKA / 2 - 1) tc_commit(&B.ydone);
      }
      __syncwarp();
    }
    // ---- P5: A | B = xt [Wa; Wb]^T per pass (pass 0 after this CTA's P4; pass 1 after P6 has read pass 0 out of TMEM)
    for (int pass = 0; pass < NPASS; ++pass) {
      if (pass == 0) mbar_wait(&B.p4done, 0);
      else mbar_wait(&B.abfree, (pass - 1) & 1);
      tc_fence_after();
      for (int t = 0; t < NXT; ++t) {
        const uint32_t slot = await_(I_XT + pass * NXT + t);
        const uint64_t da = desc_k128(su + O_A + slot * SLOT);
        if constexpr (CL == 8) {
          for (int hf = 0; hf < 2; ++hf, ++wj) {
            const uint32_t ws = wwait();
            if (elect_one()) {
              const uint64_t dw = desc_k128(su + O_W + ws * WSLOT);
#pragma unroll
              for (int ks = 0; ks < 4; ++ks)
                umma_ss_tf32(tmem + (hf ? T_B : T_A), da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_HP, (t | ks) ? 1u : 0u);
              tc_commit(&B.wempty[ws]);
            }
            __syncwarp();
          }
        } else {
          const uint32_t ws = wwait();
          if (elect_one()) {
            const uint64_t dwa = desc_k128(su + O_W + ws * WSLOT), dwb = desc_k128(su + O_W + ws * WSLOT + WSLOT / 2);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
              umma_ss_tf32(tmem + T_A, da + (uint64_t)(2 * ks), dwa + (uint64_t)(2 * ks), I_HP, (t | ks) ? 1u : 0u);
              umma_ss_tf32(tmem + T_B, da + (uint64_t)(2 * ks), dwb + (uint64_t)(2 * ks), I_HP, (t | ks) ? 1u : 0u);
            }
            tc_commit(&B.wempty[ws]);
          }
          __syncwarp();
          ++wj;
        }
        if (elect_one()) {
          tc_commit(&B.aempty[slot]);
          if (t == NXT - 1) tc_commit(&B.abdone);
        }
        __syncwarp();
      }
    }
    // ---- P7: Z = h Wsq^T (CL 8: after h_0..h_2 have been read out of Z's columns)
    mbar_wait(&B.zfree, 0);
    tc_fence_after();
    for (int pr = 0; pr < NHK / 2; ++pr, ++wj) {
      const uint32_t ws = wwait();
      for (int h2 = 0; h2 < 2; ++h2) {
        const uint32_t slot = await_(I_H + 2 * pr + h2);
        if (elect_one()) {
          const uint64_t da = desc_k128(su + O_A + slot * SLOT), dw = desc_k128(su + O_W + ws * WSLOT + h2 * WHALF);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ss_tf32(tmem + T_Z, da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_N, (pr | h2 | ks) ? 1u : 0u);
          tc_commit(&B.aempty[slot]);
        }
        __syncwarp();
      }
      if (elect_one()) {
        tc_commit(&B.wempty[ws]);
        if (pr == NHK / 2 - 1) tc_commit(&B.zdone);
      }
      __syncwarp();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ epilogue
    pdl_wait();                                                // out / XT / H may still be read by earlier kernels
    const int q4 = warp & 3, hh = (warp - 4) >> 2;
    const uint32_t r = (uint32_t)(32 * q4 + lane), trow = tmem + ((uint32_t)(32 * q4) << 16);
    const uint32_t cofs = 16 * hh;
    const uint32_t rr0 = (uint32_t)(32 * q4) + (lane >> 2), qq = lane & 3;   // read-back (CL 8): rows rr0 + 8 i, 16-B chunk qq of the half
    auto box = [&](int i) { return su + O_A + (uint32_t)(i % NAR) * SLOT; };
    auto bwait = [&](int i) { mbar_wait(&B.afull[i % NAR], (i / NAR) & 1); };
    auto release = [&](int i) { if (tid == 128) mbar_arrive(&B.aempty[i % NAR]); };
    // ---- P2: x1 = x + gate1 y -> Y; the row's statistics over this thread's NC / 2 columns; each block's two boxes released at once
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
    // ---- P3: (mean, M2) over the NC columns into red[c] (this CTA's own row of the exchange), pushed to the peers; all CL combined
    float* red = reinterpret_cast<float*>(sm + O_RED);         // [CL][2][128]
    if (hh == 0) {
      const float ma = loc[r], qa = loc[QM + r], mb = loc[2 * QM + r], qb = loc[3 * QM + r], dl = mb - ma;
      red[c * 2 * QM + r] = ma + 0.5f * dl;
      red[c * 2 * QM + QM + r] = qa + qb + dl * dl * (float)(NC / 4);   // n_a n_b / (n_a + n_b), n_a = n_b = NC / 2
      fence_proxy_async();
    }
    named_bar_sync(1, 256);
    if (tid == 128) {
      for (int p = 1; p < CL; ++p) {
        const uint32_t pr = (c + (uint32_t)p) % CL;
        push_bulk(su + O_RED + c * 1024, su + O_RED + c * 1024, 1024, smem_u32(&B.xch), pr);
      }
      TR(3);
    }
    mbar_wait_cl(&B.xch, 0);
    if (tid == 128) TR(4);
    float mean, rstd;
    {
      float mu = 0.f;
#pragma unroll
      for (int p = 0; p < CL; ++p) mu += red[p * 2 * QM + r];
      mu *= 1.f / CL;
      float M2 = 0.f;
#pragma unroll
      for (int p = 0; p < CL; ++p) { const float d = red[p * 2 * QM + r] - mu; M2 += red[p * 2 * QM + QM + r] + (float)NC * d * d; }
      mean = mu;
      rstd = rsqrtf(M2 * (1.f / D) + eps);
    }
    // ---- P4: xt (all KO blocks in registers; each block's s2 / sh2 boxes released once read), then staged in the ring slots of
    // positions I_XT .. I_XT + KO - 1 (all their boxes read by then; the producer skips those positions and waits for xtready
    // before the next), TMA-stored; one release to the cluster
    tmem_wait_st();
    uint32_t xr[KO][16];
#pragma unroll
    for (int q = 0; q < KO; ++q) {
      const int is = I_S + 2 * q;
      bwait(is);
      bwait(is + 1);
      tmem_ld16w(trow + T_Y + 32 * q + cofs, xr[q]);
      uint4 s4v[4], h4v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) { s4v[e] = lds128(box(is) + sw128(r, 4 * hh + e)); h4v[e] = lds128(box(is + 1) + sw128(r, 4 * hh + e)); }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t sw[4] = {s4v[e].x, s4v[e].y, s4v[e].z, s4v[e].w}, hw[4] = {h4v[e].x, h4v[e].y, h4v[e].z, h4v[e].w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
          xr[q][4 * e + k] = tf32r(fmaf((__uint_as_float(xr[q][4 * e + k]) - mean) * rstd, __uint_as_float(sw[k]), __uint_as_float(hw[k])));
      }
      fence_proxy_async();
      named_bar_sync(1, 256);                                  // block q's s2 / sh2 read by all
      release(is);
      release(is + 1);
    }
#pragma unroll
    for (int q = 0; q < KO; ++q) {
      const uint32_t dst = box(I_XT + q);
#pragma unroll
      for (int e = 0; e < 4; ++e) { uint4 o; u4x4(xr[q], e, o); sts128(dst + sw128(r, 4 * hh + e), o); }
    }
#if XSTG
    __syncwarp();                                              // this warp's rows / half of every own block -> XT (st.global)
#pragma unroll
    for (int q = 0; q < KO; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t rr = rr0 + 8 * i;
        stg128(XT + (size_t)(m0 + (int)rr) * xstride + (int)c * NC + 32 * q + (int)cofs + 4 * (int)qq, lds128(box(I_XT + q) + sw128(rr, 4 * hh + qq)));
      }
    fence_proxy_async_global();                                // this thread's XT stores -> the peers' TMA loads (after the release)
#endif
    fence_proxy_async();                                       // generic stores -> the MMA and the TMA stores (async proxy) read them
    tc_fence_before();                                         // TMEM Y read out before the MMA warp proceeds (p4done)
    named_bar_sync(1, 256);
    if (tid == 128) {
      mbar_arrive(&B.p4done);
      if constexpr (!OWNH) mbar_arrive(&B.zfree);              // CL 6: Z has its own columns
      for (int q = 0; q < KO; ++q) mbar_arrive(&B.afull[(I_XT + q) % NAR]);   // the own xt blocks: the a | b GEMM starts on them
#if XSTG
      TR(5);
#else
      for (int q = 0; q < KO; ++q) tma_store_2d(&mxt, box(I_XT + q), 32 * (KO * (int)c + q), m0);
      tma_store_commit();
      TR(5);
      tma_store_wait0();
      fence_proxy_async_global();
#endif
      fence_acq_rel_cluster();                                 // one release for the CL relaxed arrivals below
      for (int p = 0; p < CL; ++p) mbar_arrive_remote_relaxed(&B.xtready, (c + (uint32_t)p) % CL);
      TR(6);
    }
    // ---- P6: h = silu(a) b, per pass
#pragma unroll 1
    for (int pass = 0; pass < NPASS; ++pass) {
      mbar_wait(&B.abdone, pass & 1);
      tc_fence_after();
      if (tid == 128) TR(pass ? 10 : 7);
#pragma unroll 1
      for (int j = 0; j < HOP; ++j) {
        uint32_t av[16], bv[16];
        tmem_ld16w(trow + T_A + 32 * j + cofs, av);
        tmem_ld16w(trow + T_B + 32 * j + cofs, bv);
#pragma unroll
        for (int k = 0; k < 16; ++k) av[k] = tf32r(silu32(__uint_as_float(av[k])) * __uint_as_float(bv[k]));
        if constexpr (OWNH) {                                  // CL 8: the slot of ring position I_H + j (idle: the ring waits for hready)
          const uint32_t dst = box(I_H + j);
#pragma unroll
          for (int e = 0; e < 4; ++e) { uint4 o; u4x4(av, e, o); sts128(dst + sw128(r, 4 * hh + e), o); }
          __syncwarp();                                        // this warp's rows / half of block j -> H (st.global)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint32_t rr = rr0 + 8 * i;
            stg128(h_elem(HO * (int)c + j, (int)rr, (int)cofs + 4 * (int)qq), lds128(dst + sw128(rr, 4 * hh + qq)));
          }
        } else {                                               // CL 6: h k-block HO c + HOP pass + j, columns 16 hh .., via the warp's tile
          const uint32_t wtile = su + O_TILE + (uint32_t)(warp - 4) * 2048;
          const int kb = HO * (int)c + pass * HOP + j;
#pragma unroll
          for (int e = 0; e < 4; ++e) { uint4 o; u4x4(av, e, o); sts128(wtile + sw64((uint32_t)lane, (uint32_t)e), o); }
          __syncwarp();
#pragma unroll
          for (int i = 0; i < 4; ++i) {                        // rows 8 i .. 8 i + 7 of this warp's 32, 64 B each
            const uint32_t rr = 8 * i + (lane >> 2);
            stg128(h_elem(kb, 32 * q4 + (int)rr, (int)cofs + 4 * (int)qq), lds128(wtile + sw64(rr, qq)));
          }
          __syncwarp();
        }
      }
      if constexpr (OWNH) {
        fence_proxy_async_global();                            // this thread's H stores -> the peers' TMA loads (after the release)
        fence_proxy_async();
        tc_fence_before();                                     // a of hidden 0..95 read out: Z may accumulate there
        named_bar_sync(1, 256);
        if (tid == 128) {
          mbar_arrive(&B.zfree);
          for (int j = 0; j < HO; ++j) mbar_arrive(&B.afull[(I_H + j) % NAR]);    // the own h blocks: z starts on them
          TR(8);
          fence_acq_rel_cluster();                             // one release (every epilogue thread's H stores: before the barrier)
          for (int p = 0; p < CL; ++p) mbar_arrive_remote_relaxed(&B.hready, (c + (uint32_t)p) % CL);
          TR(9);
        }
      } else {
        fence_proxy_async_global();                            // this thread's H stores, before the release below
        tc_fence_before();                                     // TMEM A | B read out: the next pass may accumulate there
        named_bar_sync(1, 256);
        if (tid == 128) {
          if (pass + 1 < NPASS) {
            mbar_arrive(&B.abfree);
          } else {
            TR(8);
            fence_acq_rel_cluster();
            for (int p = 0; p < CL; ++p) mbar_arrive_remote_relaxed(&B.hready, (c + (uint32_t)p) % CL);
            TR(9);
          }
        }
      }
    }
    // ---- P8: out = x1 + gate2 z, through this warp's [32 rows][64 B] tile in the slot after the gate2 boxes -> st.global.v4
    mbar_wait(&B.zdone, 0);
    tc_fence_after();
    if (tid == 128) TR(13);
    const uint32_t wtile = box(I_G + KO) + (uint32_t)(warp - 4) * 2048;  // idle: every h block consumed, not a gate2 box
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
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t rr = 8 * i + (lane >> 2), qq = lane & 3;
        const uint4 val = lds128(wtile + sw64(rr, qq));
        stg128(OUT + (size_t)(m0 + 32 * q4 + (int)rr) * D + (int)c * NC + 32 * q + (int)cofs + 4 * (int)qq, val);
      }
      __syncwarp();
    }
    if (tid == 128) TR(14);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
