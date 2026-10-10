// bo_tail2_bf16.cu -- the pair tail of the bias-only token DiT's bf16-mixed three-kernel inference step on sm_100a
// (MINIWORLD_BIAS_ONLY_DIT_INF3_BF16=1; where it takes fewer rounds than bo_tail_bf16.cu at CL 8 / 6): bo_tail2_tf32.cu's geometry
// with bo_tail_bf16.cu's operands,
//
//   y = a Wo^T,  x1 = x + gate1 y,  xt = LN(x1) s2 + sh2 (bf16),  a|b = xt [Wa; Wb]^T,  h = silu(a) b (bf16),  out = x1 + gate2 h Wsq^T
//
// A cluster of 8 CTAs = TWO 128-row tiles (2 p, 2 p + 1) x 4 column groups, rank 2 g + s: column group g (output columns [192 g,
// + 192), hidden units [384 g, + 384)), tile s of the pair. The two CTAs of a column group (ranks 2 g, 2 g + 1) are a tcgen05
// cta_group::2 pair: the leader (s = 0) issues every product as M = 256, kind::f16, B split by N -- each CTA loads its own A rows
// and HALF of every weight box: y / z N = 192 (96 weight rows per CTA: the CL 8 pair-packed Wo / Wsq, natural k order, indexed by
// cluster rank), a | b N = 256 per pass of 128 hidden units (the leader's half a, the peer's b). A k-block is 64 bf16 columns.
// x [M, 768] fp32 or bf16 (-DXBF), out [M, 768] fp32 or bf16 (-DOBF), TAB [T, 6, 768] bf16 (sigmoids applied; tok = row % T, T a
// multiple of 128). Scratch padded to whole tile pairs: XT [2 ceil(tiles / 2) 128, 768] bf16 rows, H [tile][24][128][64] bf16. A
// missing odd tile (the last cluster's s = 1 CTAs) reads the last real tile's inputs, writes only its own padding rows of XT / H,
// and stores no output.
// Exchanges among the 4 CTAs of a tile (ranks 2 g' + s): the LN statistics through DSMEM; xt / h through L2 (st.global, one
// release, xtready / hready count 4). Pair barriers on the leader (count 2): p4done, abfree, zfree; the leader's MMA commits arrive
// on both CTAs (tcgen05.commit.cta_group::2 ... multicast, mask 3 << 2 g): aempty, wempty, ydone, abdone, zdone.
//
// A ring (NAR = 6 x 16 KB, own rows), one sequence in both CTAs of a pair:
//   [0, NKA)        a k-blocks                         MMA (pair: the leader's barrier, both CTAs' boxes)
//   [I_X, +NPX KO)  x_q, gate1_q (32 columns)          own epilogue, P2   (KO = 6)
//   [I_S, +KO)      s2_q | sh2_q                       own epilogue, P4
//   Two 8-KB epilogue boxes share ONE slot (+0 / +8 KB): s2_q | sh2_q always, x_q | gate1_q when x is bf16 (PACK = XBF: NPX 1, else
//   2 positions x_q, gate1_q) -- twice the boxes in flight (opt2 r2: P2 / P4 waited on one-box refills, 5.3 / 6.0 us at L768).
//   [I_XT, +3 x 12) xt k-blocks, pass 0 1 2            MMA; pass 0 starts with the own 3 blocks (staged by P4, never loaded)
//   [I_H, +24)      h k-blocks, two per TMA box        MMA, after hready
//   [I_G, +KO)      gate2_q                            own epilogue, P8
// Slot phases (both CTAs complete afull / aempty of a slot once per position): a pair position's data lands on the leader's afull
// and the peer arrives on its own; epilogue boxes are local. A pair TMA must not be issued into a slot whose previous position was
// the LEADER's epilogue box while the leader may still count it: the 3 own xt k-blocks cover only half of the 6 slots, so the first
// three L2 xt positions follow s2 / sh2 boxes -- the peer's producer waits `pairgo` (the leader's P4 done: every box consumed)
// before them (static_assert pair_slots_safe).
// Shared memory: A ring 96 KB | W ring 4 x 24 KB | statistics 4 + 2 KB | h transpose tiles 16 KB | barriers -> 219648 B.
// TMEM (512, cta_group::2 allocation in both CTAs): Y [0, 192) | a [192, 320) | b [320, 448); Z = [192, 384) after the last pass.
// PDL as bo_tail_bf16.cu. Bit-identical reruns: fixed-order reductions, no atomics. -DTRACE: bo_tail2_tf32.cu's markers.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DATT
#define DATT 768
#endif
#ifndef XBF
#define XBF 0
#endif
#ifndef OBF
#define OBF 0
#endif
static_assert(DATT == 768 || DATT == 1024, "768 or 1024 attention channels");

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
DEVI void tmem_ld16w(uint32_t taddr, uint32_t (&r)[16]) {    // tcgen05.ld fused with its wait (one asm statement)
  asm volatile("{\n\ttcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n\t"
               "tcgen05.wait::ld.sync.aligned;\n\t}"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
                 "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEVI void ld16_bf(uint32_t bx, uint32_t r, int hh, float (&v)[16]) {   // 16 values of a [128][32] bf16 SW64 box (row r, cols 16 hh ..)
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const uint4 u = lds128(bx + sw64(r, (uint32_t)(2 * hh + e)));
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) { v[8 * e + 2 * k] = bf16lo(w[k]); v[8 * e + 2 * k + 1] = bf16hi(w[k]); }
  }
}
DEVI void ld16_f32(uint32_t bx, uint32_t r, int hh, float (&v)[16]) {  // ... of a [128][32] fp32 SW128 box
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const uint4 u = lds128(bx + sw128(r, (uint32_t)(4 * hh + e)));
    v[4 * e] = __uint_as_float(u.x); v[4 * e + 1] = __uint_as_float(u.y); v[4 * e + 2] = __uint_as_float(u.z); v[4 * e + 3] = __uint_as_float(u.w);
  }
}
DEVI uint4 pk8(const float* v) {
  return make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7]));
}

#ifdef TRACE
__device__ unsigned long long g_trace[16 * 2048];
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TR(i) do { if (blockIdx.x < 16 && (i) < 2048) g_trace[blockIdx.x * 2048 + (i)] = gtime(); } while (0)
#else
#define TR(i) do { } while (0)
#endif
constexpr int EV_AS = 256, EV_AI = 512, EV_AW = 768, EV_WS = 1024, EV_WI = 1280, EV_WW = 1536, EV_AR = 1792;

constexpr int D = 768, QM = 128, KB = 64, NCL = 8, CLT = 4;    // KB: bf16 columns per k-block; cluster CTAs; CTAs per tile
constexpr int NC = D / CLT, KO = NC / 32, NH = 1536 / CLT, HO = NH / KB, NP = 3, NHP = NH / NP, HOP = NHP / KB;
constexpr int KX = NC / KB;                                    // own xt k-blocks (3)
constexpr int NHALF = NC / 2;                                  // the CTA's half of a y / z B box: 96 rows
constexpr int NKA = DATT / KB, NXT = D / KB, NHK = 1536 / KB;
constexpr bool PACK = XBF;                                     // x_q | gate1_q in one slot (a bf16 x)
constexpr int NPX = PACK ? 1 : 2;                              // ring positions per 32-column block of P2 (P4: 1, s2_q | sh2_q)
constexpr int I_X = NKA, I_S = I_X + NPX * KO, I_XT = I_S + KO, I_H = I_XT + NP * NXT, I_G = I_H + NHK, NI = I_G + KO;
constexpr int SLOT = QM * 128, NAR = 6;
constexpr int XBOX = XBF ? QM * 64 : QM * 128, TBOX = QM * 64;
constexpr int WSLOT = 2 * NHALF * 128, WHALF = NHALF * 128, WABB = NHP * 128, NWS = 4;
constexpr int O_A = 0, O_W = NAR * SLOT, O_RED = O_W + NWS * WSLOT, O_LOC = O_RED + CLT * 1024, O_TILE = O_LOC + 2048;
constexpr int O_BAR = O_TILE + 8 * 2048, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(O_W % 1024 == 0 && WSLOT % 1024 == 0 && WHALF % 1024 == 0 && O_TILE % 1024 == 0 && WABB <= WSLOT, "alignment");
static_assert(NKA % 2 == 0 && KO <= NAR && KX <= NAR && NI <= 256 && I_H % 2 == 0 && NAR % 2 == 0 && NHK % 2 == 0, "tiling");
static_assert(NP * NHP == NH && HOP * KB == NHP && KX * KB == NC, "passes / own blocks");
constexpr uint32_t T_Y = 0, T_AB = NC, T_Z = NC;               // a: [T_AB, + 128), b: [T_AB + 128, + 128)
static_assert(T_AB + 2 * NHP <= 512 && T_Z + NC <= T_AB + 2 * NHP, "TMEM");
constexpr uint32_t I_Y = idesc_bf16(2 * QM, NC), I_AB = idesc_bf16(2 * QM, 2 * NHP);

__host__ __device__ constexpr bool mma_pos(int i) { return i < NKA || (i >= I_XT && i < I_G); }
__host__ __device__ constexpr bool own_pos(int i) { return i >= I_XT && i < I_XT + KX; }
__host__ __device__ constexpr bool guarded(int i) { return i >= I_XT + KX && i < I_XT + NAR; }   // after pairgo (the peer)
// every pair position loaded by TMA has an MMA position before it in its slot, or is guarded by pairgo; the h pairs start at even
// positions and stay inside a slot pair
__host__ __device__ constexpr bool pair_slots_safe() {
  for (int i = NAR; i < NI; ++i)
    if (mma_pos(i) && !own_pos(i) && !mma_pos(i - NAR) && !guarded(i)) return false;
  for (int i = I_H; i < I_G; i += 2)
    if ((i % NAR) + 1 >= NAR) return false;
  return true;
}
static_assert(pair_slots_safe(), "slot-phase invariant of the pair ring");
// P4: own xt k-block k (epilogue blocks 2 k, 2 k + 1) goes into the slot of position I_XT + k, whose previous box is an s2 | sh2 box of
// block pp - I_S: stage it after P4's block max(2 k + 1, that block)
__host__ __device__ constexpr int stage_at(int k) {
  const int pp = I_XT + k - NAR, b = pp >= I_S ? pp - I_S : -1;
  return b > 2 * k + 1 ? b : 2 * k + 1;
}
static_assert(stage_at(KX - 1) < KO && I_XT - NAR >= I_X, "own xt staging");
DEVI int kb_xt(int t, int g, int pass) {
  if (pass) return t;
  if (t < KX) return KX * g + t;
  const int j = t - KX;
  return j < KX * g ? j : j + KX;
}

struct Bars {
  uint64_t afull[NAR], aempty[NAR], wfull[NWS], wempty[NWS];
  uint64_t xch, xtready, hready, ydone, p4done, abdone, abfree, zfree, zdone, pairgo;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

extern "C" __global__ void __launch_bounds__(384, 1)
bo_tail2_bf16_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mtab,
                    const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwab, const __grid_constant__ CUtensorMap mwsq,
                    const __grid_constant__ CUtensorMap mxt, const __grid_constant__ CUtensorMap mh,
                    void* __restrict__ OUTV, __nv_bfloat16* __restrict__ XT, __nv_bfloat16* __restrict__ HP, int xstride, int T,
                    int ntiles, float eps) {
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
  auto h_elem = [&](int kb, int row, int col) -> __nv_bfloat16* { return HP + ((size_t)(tile * NHK + kb) * QM + row) * KB + col; };
  auto h_row = [&](int kb) { return (tile * NHK + kb) * QM; };

  if (tid == 0) {
    for (int i = 0; i < NAR; ++i) { mbar_init(&B.afull[i], 1); mbar_init(&B.aempty[i], 1); }
    for (int i = 0; i < NWS; ++i) { mbar_init(&B.wfull[i], 1); mbar_init(&B.wempty[i], 1); }
    mbar_init(&B.xch, 1);
    mbar_init(&B.xtready, CLT); mbar_init(&B.hready, CLT);
    mbar_init(&B.ydone, 1); mbar_init(&B.abdone, 1); mbar_init(&B.zdone, 1);
    mbar_init(&B.p4done, 2); mbar_init(&B.abfree, 2); mbar_init(&B.zfree, 2);   // the pair's two CTAs (used in the leader)
    mbar_init(&B.pairgo, 1);                                   // the leader's P4 done (used in the peer)
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
      pdl_wait();                                              // a: the core's output; x: an earlier kernel's
      TR(15);
      const uint64_t once = pol_evict_first();                 // x / table boxes: read once
      for (int i = 0; i < NI; ++i) {
        const int slot = i % NAR;
        if (own_pos(i)) continue;                              // staged by the epilogue
        if (i == I_XT + KX) {
          mbar_wait_cl(&B.xtready, 0);
          TR(16);
          if (!leader) mbar_wait_cl(&B.pairgo, 0);             // the leader's s2 / sh2 boxes consumed (pair_slots_safe)
          fence_proxy_async_global();
        }
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
            tma_load_2d_2sm(dst, &ma, &B.afull[slot], KB * i, min_);
          } else {
            const int t = (i - I_XT) % NXT, pass = (i - I_XT) / NXT;
            tma_load_2d_2sm(dst, &mxt, &B.afull[slot], KB * kb_xt(t, g, pass), m0);
          }
          if (!leader) mbar_arrive(&B.afull[slot]);
        } else {                                               // the own epilogue's boxes
          if (i < I_S && PACK) {                               // x_q | gate1_q in one slot
            const int col = g * NC + 32 * (i - I_X);
            mbar_expect_tx(&B.afull[slot], XBOX + TBOX);
            tma_load_2d_h(dst, &mx, &B.afull[slot], col, min_, once);
            tma_load_2d_h(dst + XBOX, &mtab, &B.afull[slot], 0 * D + col, tr0, once);
          } else if (i < I_S) {                                // x_q (even), gate1_q (odd)
            const int q = (i - I_X) >> 1, col = g * NC + 32 * q;
            if (((i - I_X) & 1) == 0) { mbar_expect_tx(&B.afull[slot], XBOX); tma_load_2d_h(dst, &mx, &B.afull[slot], col, min_, once); }
            else { mbar_expect_tx(&B.afull[slot], TBOX); tma_load_2d_h(dst, &mtab, &B.afull[slot], 0 * D + col, tr0, once); }
          } else if (i < I_XT) {                               // s2_q | sh2_q in one slot
            const int col = g * NC + 32 * (i - I_S);
            mbar_expect_tx(&B.afull[slot], 2 * TBOX);
            tma_load_2d_h(dst, &mtab, &B.afull[slot], 3 * D + col, tr0, once);
            tma_load_2d_h(dst + TBOX, &mtab, &B.afull[slot], 5 * D + col, tr0, once);
          } else {                                             // gate2_q
            mbar_expect_tx(&B.afull[slot], TBOX);
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
          tma_load_2d_2sm(su + O_W + ws * WSLOT, &mwab, &B.wfull[ws], KB * kb_xt(t, g, pass), s * 1536 + g * NH + pass * NHP);
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
      // ---- P1: Y = a Wo^T, M = 256 (both tiles), N = 192, K = 16 per MMA
      for (int s2 = 0; s2 < NKA / 2; ++s2, ++wj) {
        const uint32_t ws = wwait();
        for (int h2 = 0; h2 < 2; ++h2) {
          const int i = 2 * s2 + h2;
          const uint32_t slot = await_(i);
          if (elect_one()) {
            const uint64_t da = desc_k128(su + O_A + slot * SLOT), dw = desc_k128(su + O_W + ws * WSLOT + h2 * WHALF);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) umma_ss2(tmem + T_Y, da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_Y, (i | ks) ? 1u : 0u);
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
        if (pass == 0) mbar_wait_cl(&B.p4done, 0);             // both CTAs staged their own xt and read Y / the s2 boxes
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
              umma_ss2(tmem + T_AB, da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_AB, (t | ks) ? 1u : 0u);
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
              umma_ss2(tmem + T_Z, da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_Y, (pr | h2 | ks) ? 1u : 0u);
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
    const uint32_t qq = lane & 3;
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
      const int ix = I_X + NPX * q;
      const uint32_t xb = box(ix), gb = PACK ? box(ix) + XBOX : box(ix + 1);   // x_q, gate1_q
      bwait(ix);
      if (!PACK) bwait(ix + 1);
      uint32_t yv[16];
      tmem_ld16w(trow + T_Y + 32 * q + cofs, yv);
      float xv[16], gv[16];
#if XBF
      ld16_bf(xb, r, hh, xv);
#else
      ld16_f32(xb, r, hh, xv);
#endif
      ld16_bf(gb, r, hh, gv);
      float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const float v = fmaf(gv[k], __uint_as_float(yv[k]), xv[k]);
        yv[k] = __float_as_uint(v);
        s4[k & 3] += v;
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
      if (!PACK) release(ix + 1);
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
    // ---- P4: xt per 32-column block -> XT (st.global, 2 x 16 B of this thread's row); own k-block k into the slot of position
    // I_XT + k once that slot's previous box is consumed (stage_at)
    tmem_wait_st();
    uint4 xs[KO][2];
#pragma unroll
    for (int q = 0; q < KO; ++q) {
      const int is = I_S + q;
      bwait(is);
      uint32_t yv[16];
      tmem_ld16w(trow + T_Y + 32 * q + cofs, yv);
      float sv[16], hv[16], o[16];
      ld16_bf(box(is), r, hh, sv);
      ld16_bf(box(is) + TBOX, r, hh, hv);
#pragma unroll
      for (int k = 0; k < 16; ++k) o[k] = fmaf((__uint_as_float(yv[k]) - mean) * rstd, sv[k], hv[k]);
      xs[q][0] = pk8(o);
      xs[q][1] = pk8(o + 8);
      __nv_bfloat16* xg = XT + (size_t)(m0 + (int)r) * xstride + g * NC + 32 * q + (int)cofs;
      stg128(xg, xs[q][0]);
      stg128(xg + 8, xs[q][1]);
      fence_proxy_async();
      named_bar_sync(1, 256);                                  // block q's s2 / sh2 read by all
      release(is);
#pragma unroll
      for (int k = 0; k < KX; ++k) {
        if (stage_at(k) != q) continue;
        const uint32_t dst = box(I_XT + k);
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2)
#pragma unroll
          for (int e = 0; e < 2; ++e) sts128(dst + sw128(r, (uint32_t)(4 * h2 + 2 * hh + e)), xs[2 * k + h2][e]);
      }
    }
    fence_proxy_async_global();                                // this thread's XT stores -> the tile peers' TMA loads
    fence_proxy_async();                                       // the staged blocks -> the pair's MMA
    tc_fence_before();                                         // Y read out before the pair's MMAs go on (p4done)
    named_bar_sync(1, 256);
    if (tid == 128) {
      for (int k = 0; k < KX; ++k) mbar_arrive(&B.afull[(I_XT + k) % NAR]);   // the own xt positions (the leader's: the MMA waits)
      pair_arrive(&B.p4done);
      if (leader) mbar_arrive_remote(&B.pairgo, crank | 1u);   // the peer may load into this CTA's slot phases (pair_slots_safe)
      TR(5);
      fence_acq_rel_cluster();                                 // one release for the tile peers' relaxed arrivals
      for (int p = 0; p < CLT; ++p) mbar_arrive_remote_relaxed(&B.xtready, peer((g + p) % CLT));
      TR(6);
    }
    // ---- P6: h = silu(a) b per pass, per 64-column k-block -> H (st.global through this warp's transpose tile)
    const uint32_t wtile = su + O_TILE + (uint32_t)(warp - 4) * 2048;
#pragma unroll 1
    for (int pass = 0; pass < NP; ++pass) {
      mbar_wait(&B.abdone, pass & 1);
      tc_fence_after();
      if (tid == 128) TR(pass == 0 ? 7 : pass == 1 ? 10 : 11);
#pragma unroll 1
      for (int j = 0; j < HOP; ++j) {
        uint4 hq[4];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          uint32_t av[16], bv[16];
          tmem_ld16w(trow + T_AB + KB * j + 32 * hh + 16 * u, av);
          tmem_ld16w(trow + T_AB + NHP + KB * j + 32 * hh + 16 * u, bv);
          float hv[16];
#pragma unroll
          for (int k = 0; k < 16; ++k) hv[k] = silu32(__uint_as_float(av[k])) * __uint_as_float(bv[k]);
          hq[2 * u] = pk8(hv);
          hq[2 * u + 1] = pk8(hv + 8);
        }
        const int kb = HO * g + pass * HOP + j;
#pragma unroll
        for (int e = 0; e < 4; ++e) sts128(wtile + sw64((uint32_t)lane, (uint32_t)e), hq[e]);
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 4; ++i) {                          // rows 8 i .. 8 i + 7 of this warp's 32, 64 B each
          const uint32_t rr = 8 * i + (lane >> 2);
          stg128(h_elem(kb, 32 * q4 + (int)rr, 32 * hh + 8 * (int)qq), lds128(wtile + sw64(rr, qq)));
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
    // ---- P8: out = x1 + gate2 z; fp32 out through this warp's [32 rows][64 B] tile (the h transpose tile: P6 is over), bf16 out
    // straight from the registers
    mbar_wait(&B.zdone, 0);
    tc_fence_after();
    if (tid == 128) TR(13);
#pragma unroll 1
    for (int q = 0; q < KO; ++q) {
      bwait(I_G + q);
      uint32_t zv[16], xv[16];
      tmem_ld16w(trow + T_Z + 32 * q + cofs, zv);
      tmem_ld16w(trow + T_Y + 32 * q + cofs, xv);
      float gv[16], o[16];
      ld16_bf(box(I_G + q), r, hh, gv);
#pragma unroll
      for (int k = 0; k < 16; ++k) o[k] = fmaf(gv[k], __uint_as_float(zv[k]), __uint_as_float(xv[k]));
#if OBF
      if (valid) {
        __nv_bfloat16* og = reinterpret_cast<__nv_bfloat16*>(OUTV) + (size_t)(m0 + (int)r) * D + g * NC + 32 * q + (int)cofs;
        stg128(og, pk8(o));
        stg128(og + 8, pk8(o + 8));
      }
#else
#pragma unroll
      for (int e = 0; e < 4; ++e)
        sts128(wtile + sw64((uint32_t)lane, (uint32_t)e), make_uint4(__float_as_uint(o[4 * e]), __float_as_uint(o[4 * e + 1]),
                                                                       __float_as_uint(o[4 * e + 2]), __float_as_uint(o[4 * e + 3])));
      __syncwarp();
      if (valid) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t rr = 8 * i + (lane >> 2);
          stg128(reinterpret_cast<float*>(OUTV) + (size_t)(m0 + 32 * q4 + (int)rr) * D + g * NC + 32 * q + (int)cofs + 4 * (int)qq,
                 lds128(wtile + sw64(rr, qq)));
        }
      }
      __syncwarp();
#endif
    }
    if (tid == 128) TR(14);
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();                                              // no CTA leaves while a pair / tile peer may still signal it
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}
