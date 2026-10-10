// bo_tail_bf16.cu -- K3 of the bias-only token DiT's bf16-mixed inference step on sm_100a (MINIWORLD_BIAS_ONLY_DIT_INF3_BF16=1):
// everything after the attention core in ONE kernel per 128-row tile, the bf16 twin of bo_tail_tf32.cu:
//
//   y   = a Wo^T                              (a [M, DA] bf16 = sigmoid(g) (P v), from pv_gate_inf -DPDL_INF)
//   x1  = x + gate1 y                         (fp32: x the block's residual input, fp32 or bf16 (-DXBF); y fp32 in TMEM)
//   xt  = LN(x1) s2 + sh2                     (fp32 statistics, rounded to bf16: the expand GEMM's operand)
//   a|b = xt [Wa; Wb]^T,  h = silu(a) b       (fp32 accumulators; h rounded to bf16: the squeeze GEMM's operand)
//   z   = h Wsq^T,        out = x1 + gate2 z  (fp32, stored fp32 or bf16 (-DOBF): the next block's residual or the step output)
//
// tcgen05.mma kind::f16 (bf16 operands, fp32 accumulation), K = 16 per instruction. An operand k-block is 64 bf16 columns: one
// 128-byte swizzle row, [128 rows][64] = 16 KB per box (bo_tail_tf32.cu's k-block is 32 fp32 columns -- the same 128 B).
// TAB [T, 6, 768] bf16 (one block's slice of the hoisted tables: 0 gate1, 1 gate2, 2 s1, 3 s2, 4 sh1, 5 sh2, the four sigmoids
// applied; tok = row % T, T a multiple of 128); weights bf16: WOP = Wo pair-packed (two 64-column k-blocks of a CTA's NC output
// rows per box, k order), WSQP = Wsq pair-packed in this CTA's z k-block order (below), Wab = [Wa; Wb] [3072, 768]. Scratch,
// L2-resident: XT [M, 768] bf16 rows (row stride xstride), H blocked k-block-major [row tile][24][128][64] bf16.
//
// Geometry as bo_tail_tf32.cu: a cluster of CL CTAs per row tile, CTA c owns output columns [NC c, + NC) (NC = 768 / CL) and hidden
// units [NH c, + NH) (NH = 1536 / CL).
//   CL = 8: NC 96, NH 192; one a | b pass. NC is 1.5 k-blocks, so the own xt columns are not whole k-blocks: every CTA writes its xt
//           to XT and the a | b GEMM streams all 12 xt k-blocks from L2 after xtready (no own-first xt). The own h (NH = 3 k-blocks)
//           is staged in the ring and z takes it first.
//   CL = 6: NC 128, NH 256; a | b in two passes of 128 hidden units (TMEM Y 128 + A 128 + B 128 + Z 128); the own 2 xt k-blocks are
//           staged in the ring and pass 0 takes them first; h goes out through per-warp transpose tiles, z waits for the cluster's h.
// Exchanges inside the cluster: the LN statistics through DSMEM (Chan); xt / h through L2 with st.global and ONE release
// (fence.proxy.async.global, fence.acq_rel.cluster, CL relaxed remote arrivals on xtready / hready); the A producer waits once.
//
// The A ring (NAR = 6 x 16 KB) carries one sequence (position i, slot i % 6):
//   [0, NKA)             a k-blocks (64 columns)            MMA, P1 (y)
//   [I_X, +NPX KO)       x_q, gate1_q (32 columns)          epilogue, P2 (x: [128][32] fp32 SW128 16 KB or bf16 SW64 8 KB; tables
//                                                           [128][32] bf16 SW64 8 KB)
//   [I_S, +KO)           s2_q | sh2_q                       epilogue, P4
//   Two 8-KB epilogue boxes share ONE slot (+0 / +8 KB): s2_q | sh2_q always, x_q | gate1_q when x is bf16 (PACK = XBF: NPX 1, else
//   2 positions x_q, gate1_q) -- twice the boxes in flight; at CL 8 (bf16 x) every P2 and P4 box of the tile is in the ring at once,
//   loaded while the y GEMM frees its slots (opt2 r2: P2 / P4 waited on one-box refills, 3.1 / 2.6 us at L384)
//   [I_XT, +NPASS 12)    xt k-blocks per pass               MMA, P5 (CL 6: pass 0's first 2 = the own blocks, staged by P4)
//   [I_H, +24)           h k-blocks in z order              MMA, P7 (CL 8: the first 3 = the own blocks, staged by P6; the L2
//                                                           ones two per 32-KB box into adjacent slots, CL 8's first alone)
//   [I_G, +KO)           gate2_q                            epilogue, P8
// Each own (staged) position's slot is written only after its previous box is consumed: CL 6 xt -- an epilogue box (an x /
// gate1 box, released in P2, at every DATT / XBF today), P4 stages the block after that box's barrier (stage_at); CL 8 h -- an xt
// block consumed by a | b (P6 runs after abdone).
// W ring (NWS slots of WSLOT = 2 NC x 128 B): Wo / Wsq slots hold TWO k-blocks of the CTA's rows as one pair-packed box (CL 8 24 KB,
// CL 6 32 KB); a Wab slot one [192][64] half of a k-block (CL 8) or the pass's [128][64] a rows + [128][64] b rows (CL 6).
// Phases, TMEM and warp roles as bo_tail_tf32.cu: P1 y | P2 x1 -> TMEM Y, statistics | P3 all-reduce | P4 xt -> XT (st.global) |
// P5 a | b | P6 h | P7 z | P8 out. TMEM: CL 8 Y [0, 96) | A [96, 288) | B [288, 480), Z = [96, 192) after P6; CL 6 Y | A | B | Z.
//   shared memory  CL 8: A ring 96 KB | W ring 5 x 24 KB | statistics 8 + 2 KB | barriers -> 231936 B
//                  CL 6: A ring 96 KB | W ring 3 x 32 KB | statistics 6 + 2 KB | transpose tiles 16 KB | barriers -> 221696 B
// PDL: pdl_launch() after setup; the A producer waits before loading a and x; the epilogue before its first store; the W producer
// loads before the wait (weights are packed once, never written in a step). Bit-identical reruns: fixed-order reductions, no
// atomics. -DTRACE: %globaltimer markers into g_trace (bo_tail_tf32.cu's layout).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef CL
#define CL 8
#endif
#ifndef DATT
#define DATT 768
#endif
#ifndef XBF
#define XBF 0                                                  // 1: the residual input x is bf16 (the step's input), else fp32
#endif
#ifndef OBF
#define OBF 0                                                  // 1: out is bf16 (the step's output), else fp32 (the next block's x)
#endif
static_assert(CL == 8 || CL == 6, "cluster of 8 or 6");
static_assert(DATT == 768 || DATT == 1024, "768 or 1024 attention channels");

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
// this thread's 16 consecutive values of a [128][32] bf16 SW64 box (row r, columns 16 hh ..): two 16-byte chunks, as fp32
DEVI void ld16_bf(uint32_t bx, uint32_t r, int hh, float (&v)[16]) {
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const uint4 u = lds128(bx + sw64(r, (uint32_t)(2 * hh + e)));
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) { v[8 * e + 2 * k] = bf16lo(w[k]); v[8 * e + 2 * k + 1] = bf16hi(w[k]); }
  }
}
// ... of a [128][32] fp32 SW128 box: four 16-byte chunks
DEVI void ld16_f32(uint32_t bx, uint32_t r, int hh, float (&v)[16]) {
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const uint4 u = lds128(bx + sw128(r, (uint32_t)(4 * hh + e)));
    v[4 * e] = __uint_as_float(u.x); v[4 * e + 1] = __uint_as_float(u.y); v[4 * e + 2] = __uint_as_float(u.z); v[4 * e + 3] = __uint_as_float(u.w);
  }
}
DEVI uint4 pk8(const float* v) {                              // 8 fp32 -> 8 bf16 (round to nearest even), 16 bytes
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

constexpr int D = 768, QM = 128, KB = 64;                      // KB: bf16 columns per operand k-block (one 128-B swizzle row)
constexpr int NC = D / CL, KO = NC / 32, NH = 1536 / CL, HO = NH / KB;
constexpr int NPASS = CL == 6 ? 2 : 1, NHP = NH / NPASS, HOP = NHP / KB;
constexpr bool OWNH = CL == 8;                                 // own h k-blocks staged in the ring and consumed first
constexpr int KX = (CL == 6) ? NC / KB : 0;                    // own xt k-blocks staged in the ring (CL 6: 2; CL 8: NC = 1.5 k-blocks)
constexpr int NKA = DATT / KB, NXT = D / KB, NHK = 1536 / KB;
constexpr bool PACK = XBF;                                     // two 8-KB epilogue boxes per slot (a bf16 x)
constexpr int NPX = PACK ? 1 : 2, NPS = 1;                     // ring positions per 32-column block: P2 (x, gate1), P4 (s2 | sh2)
constexpr int I_X = NKA, I_S = I_X + NPX * KO, I_XT = I_S + NPS * KO, I_H = I_XT + NPASS * NXT, I_G = I_H + NHK, NI = I_G + KO;
constexpr int SLOT = QM * 128, NAR = 6;
constexpr int XBOX = XBF ? QM * 64 : QM * 128, TBOX = QM * 64;  // the x box, a table box ([128][32] bf16)
constexpr int WSLOT = 2 * NC * 128, WHALF = NC * 128, NWS = CL == 8 ? 5 : 3;
constexpr int TILEB = CL == 6 ? 8 * 2048 : 0;                  // CL 6: the h transpose tiles
constexpr int O_A = 0, O_W = NAR * SLOT, O_RED = O_W + NWS * WSLOT, O_LOC = O_RED + CL * 1024, O_TILE = O_LOC + 2048;
constexpr int O_BAR = O_TILE + TILEB, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(O_W % 1024 == 0 && WSLOT % 1024 == 0 && WHALF % 1024 == 0 && O_TILE % 1024 == 0, "1 KB alignment of the swizzled tiles");
static_assert(NKA % 2 == 0 && NHK % 2 == 0 && KO <= NAR && HO <= NAR && KX <= NAR && NI <= 256 && NC % 32 == 0 && NH % KB == 0, "tiling");
static_assert(2 * NC <= 256 && NHP <= 256 && KB * 2 == 128, "TMA box rows / the 128-B k-block");
constexpr uint32_t T_Y = 0, T_A = NC, T_B = NC + NHP, T_Z = OWNH ? NC : NC + 2 * NHP;
static_assert((OWNH ? T_B + NHP : T_Z + NC) <= 512, "TMEM");
constexpr uint32_t I_N = idesc_bf16(QM, NC), I_HP = idesc_bf16(QM, NHP);
// CL 6: own xt k-block k (32-column epilogue blocks 2 k, 2 k + 1) goes into the slot of position I_XT + k, whose previous box is an
// epilogue box: an s2 | sh2 box of block pp - I_S -> stage it after P4's block max(2 k + 1, that block); an x | gate1 box
// (released in P2) -> after P4's block 2 k + 1
__host__ __device__ constexpr int stage_at(int k) {
  const int pp = I_XT + k - NAR, b = pp >= I_S ? pp - I_S : -1;
  return b > 2 * k + 1 ? b : 2 * k + 1;
}
__host__ __device__ constexpr bool staging_safe() {
  for (int k = 0; k < KX; ++k)
    if (I_XT + k - NAR < I_X || I_XT + k - NAR >= I_XT || stage_at(k) >= KO) return false;   // an epilogue box before it
  for (int j = 0; OWNH && j < HO; ++j)
    if (I_H + j - NAR < I_XT || I_H + j - NAR >= I_H) return false;                       // an xt block (MMA) before it
  return true;
}
static_assert(staging_safe(), "own blocks staged into slots whose previous box is consumed first");
// the GEMMs' k-block orders: own blocks first where they are staged locally, then the others in order
DEVI int kb_xt(int t, int c, int pass) {
  if (pass || KX == 0) return t;
  if (t < KX) return KX * c + t;
  const int j = t - KX;
  return j < KX * c ? j : j + KX;
}
__host__ __device__ constexpr int kb_h(int t, int c) {        // z's k-block t (the order of the WSQP pack: tf32.py)
  return !OWNH ? t : t < HO ? HO * c + t : (t - HO < HO * c ? t - HO : t);
}
DEVI bool own_pos(int i) { return (i >= I_XT && i < I_XT + KX) || (OWNH && i >= I_H && i < I_H + HO); }
// z's L2 h k-blocks load two per 32-KB TMA box into adjacent ring slots (an even position and the next; a first L2 block at an odd
// position or a last one at an even position alone -- CL 8 with a bf16 / fp32 x); a pair split by the own blocks in the z order loads as two 16-KB boxes on one barrier
constexpr int I_HL = I_H + (OWNH ? HO : 0);                    // the first L2 h position
__host__ __device__ constexpr bool hpair(int i) { return i >= I_HL && i + 1 < I_G && (i % 2) == 0; }
__host__ __device__ constexpr bool hpairs_ok() {
  for (int i = I_HL; i < I_G; ++i) {
    if (hpair(i) && (i % NAR) + 1 >= NAR) return false;          // the pair's two slots adjacent
    if (i > I_HL && i + 1 < I_G && !hpair(i) && !hpair(i - 1)) return false;   // between the first and the last, all in pairs
  }
  return NAR % 2 == 0;
}
static_assert(hpairs_ok(), "z's h pairs");

struct Bars {
  uint64_t afull[NAR], aempty[NAR], wfull[NWS], wempty[NWS];
  uint64_t xch, xtready, hready, ydone, p4done, abdone, abfree, zfree, zdone;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

extern "C" __global__ void __launch_bounds__(384, 1)
bo_tail_bf16_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mtab,
                   const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwab, const __grid_constant__ CUtensorMap mwsq,
                   const __grid_constant__ CUtensorMap mxt, const __grid_constant__ CUtensorMap mh, const __grid_constant__ CUtensorMap mh2,
                   void* __restrict__ OUTV, __nv_bfloat16* __restrict__ XT, __nv_bfloat16* __restrict__ HP, int xstride, int T, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t c = cluster_rank();
  const int rtile = (int)(blockIdx.x / CL), m0 = rtile * QM, tr0 = m0 % T;
  // H element (k-block kb, tile row row, column col of the k-block): blocked [row tile][24][128][64]; its TMA row coordinate
  auto h_elem = [&](int kb, int row, int col) -> __nv_bfloat16* { return HP + ((size_t)(rtile * NHK + kb) * QM + row) * KB + col; };
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
    prefetch_map(&mxt); prefetch_map(&mh); prefetch_map(&mh2);
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
      pdl_wait();                                              // a: the core's output; x: an earlier kernel's
      TR(15);
      const uint64_t once = pol_evict_first();                 // x / table boxes: read once by this CTA
      for (int i = 0; i < NI; ++i) {
        const int slot = i % NAR;
        if (own_pos(i)) continue;                              // own blocks: staged here by the epilogue
        if (i == I_XT + KX) { mbar_wait_cl(&B.xtready, 0); TR(16); fence_proxy_async_global(); }
        if (i == I_HL) { mbar_wait_cl(&B.hready, 0); TR(17); fence_proxy_async_global(); }
        if (hpair(i)) {                                        // h k-blocks t, t + 1 into slots slot, slot + 1, one barrier
          mbar_wait(&B.aempty[slot], ((i / NAR) - 1) & 1);
          mbar_wait(&B.aempty[slot + 1], (((i + 1) / NAR) - 1) & 1);
          TR(EV_AR + i);
          const uint32_t dst = su + O_A + slot * SLOT;
          const int k0 = kb_h(i - I_H, (int)c), k1 = kb_h(i + 1 - I_H, (int)c);
          mbar_expect_tx(&B.afull[slot], 2 * SLOT);
          if (k1 == k0 + 1) {
            tma_load_2d(dst, &mh2, &B.afull[slot], 0, h_row(k0));
          } else {
            tma_load_2d(dst, &mh, &B.afull[slot], 0, h_row(k0));
            tma_load_2d(dst + SLOT, &mh, &B.afull[slot], 0, h_row(k1));
          }
          mbar_arrive(&B.afull[slot + 1]);                     // position i + 1's phase (its bytes counted on slot's barrier)
          TR(EV_AI + i);
          ++i;
          continue;
        }
        if (i >= NAR) mbar_wait(&B.aempty[slot], ((i / NAR) - 1) & 1);
        TR(EV_AR + i);
        const uint32_t dst = su + O_A + slot * SLOT;
        if (i < NKA) {
          mbar_expect_tx(&B.afull[slot], SLOT);
          tma_load_2d(dst, &ma, &B.afull[slot], KB * i, m0);
        } else if (i < I_S && PACK) {                          // x_q | gate1_q in one slot
          const int q = i - I_X, col = (int)c * NC + 32 * q;
          mbar_expect_tx(&B.afull[slot], XBOX + TBOX);
          tma_load_2d_h(dst, &mx, &B.afull[slot], col, m0, once);
          tma_load_2d_h(dst + XBOX, &mtab, &B.afull[slot], 0 * D + col, tr0, once);
        } else if (i < I_S) {                                  // x_q (even), gate1_q (odd)
          const int q = (i - I_X) >> 1, col = (int)c * NC + 32 * q;
          if (((i - I_X) & 1) == 0) {
            mbar_expect_tx(&B.afull[slot], XBOX);
            tma_load_2d_h(dst, &mx, &B.afull[slot], col, m0, once);
          } else {
            mbar_expect_tx(&B.afull[slot], TBOX);
            tma_load_2d_h(dst, &mtab, &B.afull[slot], 0 * D + col, tr0, once);
          }
        } else if (i < I_XT) {                                 // s2_q | sh2_q in one slot
          const int q = i - I_S, col = (int)c * NC + 32 * q;
          mbar_expect_tx(&B.afull[slot], 2 * TBOX);
          tma_load_2d_h(dst, &mtab, &B.afull[slot], 3 * D + col, tr0, once);
          tma_load_2d_h(dst + TBOX, &mtab, &B.afull[slot], 5 * D + col, tr0, once);
        } else if (i < I_H) {
          const int t = (i - I_XT) % NXT, pass = (i - I_XT) / NXT;
          mbar_expect_tx(&B.afull[slot], SLOT);
          tma_load_2d(dst, &mxt, &B.afull[slot], KB * kb_xt(t, (int)c, pass), m0);
        } else if (i < I_G) {
          mbar_expect_tx(&B.afull[slot], SLOT);
          tma_load_2d(dst, &mh, &B.afull[slot], 0, h_row(kb_h(i - I_H, (int)c)));
        } else {                                               // gate2_q
          mbar_expect_tx(&B.afull[slot], TBOX);
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
      auto slot = [&](uint32_t bytes) -> uint32_t {
        const int ws = wj % NWS;
        if (wj >= NWS) mbar_wait(&B.wempty[ws], ((wj / NWS) - 1) & 1);
        mbar_expect_tx(&B.wfull[ws], bytes);
        return (uint32_t)ws;
      };
      auto wab_row = [&](int pass, int hf) { return hf * 1536 + (int)c * NH + pass * NHP; };
      for (int s2 = 0; s2 < NKA / 2; ++s2, ++wj) {             // Wo: k-blocks 2 s2, 2 s2 + 1 as one pair-packed box
        const uint32_t ws = slot(WSLOT);
        tma_load_2d_h(su + O_W + ws * WSLOT, &mwo, &B.wfull[ws], 0, ((int)c * (NKA / 2) + s2) * 2 * NC, keep);
        TR(EV_WI + wj);
      }
      for (int pass = 0; pass < NPASS; ++pass)
        for (int t = 0; t < NXT; ++t) {                        // Wab per xt k-block: CL 8 a half, b half; CL 6 a + b in one slot
          const int kb = kb_xt(t, (int)c, pass);
          if constexpr (CL == 8) {
            for (int hf = 0; hf < 2; ++hf, ++wj) {
              const uint32_t ws = slot(NHP * 128);
              tma_load_2d_h(su + O_W + ws * WSLOT, &mwab, &B.wfull[ws], KB * kb, wab_row(pass, hf), keep);
              TR(EV_WI + wj);
            }
          } else {
            const uint32_t ws = slot(2 * NHP * 128), dst = su + O_W + ws * WSLOT;
            tma_load_2d_h(dst, &mwab, &B.wfull[ws], KB * kb, wab_row(pass, 0), keep);
            tma_load_2d_h(dst + NHP * 128, &mwab, &B.wfull[ws], KB * kb, wab_row(pass, 1), keep);
            TR(EV_WI + wj);
            ++wj;
          }
        }
      for (int pr = 0; pr < NHK / 2; ++pr, ++wj) {             // Wsq: z k-blocks 2 pr, 2 pr + 1 (z order, packed so) as one box
        const uint32_t ws = slot(WSLOT);
        tma_load_2d_h(su + O_W + ws * WSLOT, &mwsq, &B.wfull[ws], 0, ((int)c * (NHK / 2) + pr) * 2 * NC, keep);
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
    // ---- P1: Y = a Wo^T (K = 16 per MMA, 4 per 64-column k-block)
    for (int s2 = 0; s2 < NKA / 2; ++s2, ++wj) {
      const uint32_t ws = wwait();
      for (int h2 = 0; h2 < 2; ++h2) {
        const int i = 2 * s2 + h2;
        const uint32_t slot = await_(i);
        if (elect_one()) {
          const uint64_t da = desc_k128(su + O_A + slot * SLOT), dw = desc_k128(su + O_W + ws * WSLOT + h2 * WHALF);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) umma_ss(tmem + T_Y, da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_N, (i | ks) ? 1u : 0u);
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
                umma_ss(tmem + (hf ? T_B : T_A), da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_HP, (t | ks) ? 1u : 0u);
              tc_commit(&B.wempty[ws]);
            }
            __syncwarp();
          }
        } else {
          const uint32_t ws = wwait();
          if (elect_one()) {
            const uint64_t dwa = desc_k128(su + O_W + ws * WSLOT), dwb = desc_k128(su + O_W + ws * WSLOT + NHP * 128);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
              umma_ss(tmem + T_A, da + (uint64_t)(2 * ks), dwa + (uint64_t)(2 * ks), I_HP, (t | ks) ? 1u : 0u);
              umma_ss(tmem + T_B, da + (uint64_t)(2 * ks), dwb + (uint64_t)(2 * ks), I_HP, (t | ks) ? 1u : 0u);
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
    // ---- P7: Z = h Wsq^T (CL 8: after P6 has read a and b out of the columns Z reuses)
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
            umma_ss(tmem + T_Z, da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_N, (pr | h2 | ks) ? 1u : 0u);
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
    const uint32_t cofs = 16 * hh;                             // this warpgroup's 16 columns of every 32-column block
    const uint32_t rr0 = (uint32_t)(32 * q4) + (lane >> 2), qq = lane & 3;   // read-back: rows rr0 + 8 i, 16-B chunk qq of the half
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
    // ---- P3: (mean, M2) over the NC columns into red[c] (this CTA's own row of the exchange), pushed to the peers; all CL combined
    float* red = reinterpret_cast<float*>(sm + O_RED);         // [CL][2][128]
    if (hh == 0) {
      const float ma_ = loc[r], qa = loc[QM + r], mb = loc[2 * QM + r], qb = loc[3 * QM + r], dl = mb - ma_;
      red[c * 2 * QM + r] = ma_ + 0.5f * dl;
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
    // ---- P4: xt per 32-column block (its s2 / sh2 boxes released once read) -> XT (st.global: 2 x 16 B of this thread's row); CL 6:
    // also into the slot of own k-block q / 2 (position I_XT + q / 2), once that slot's previous box is consumed (stage_at)
    tmem_wait_st();
    uint4 xs[KO][2];                                           // xt of block q, bf16: this thread's 16 columns
#pragma unroll
    for (int q = 0; q < KO; ++q) {
      const int is = I_S + q;
      const uint32_t sb = box(is), hb = box(is) + TBOX;       // s2_q | sh2_q
      bwait(is);
      uint32_t yv[16];
      tmem_ld16w(trow + T_Y + 32 * q + cofs, yv);
      float sv[16], hv[16], o[16];
      ld16_bf(sb, r, hh, sv);
      ld16_bf(hb, r, hh, hv);
#pragma unroll
      for (int k = 0; k < 16; ++k) o[k] = fmaf((__uint_as_float(yv[k]) - mean) * rstd, sv[k], hv[k]);
      xs[q][0] = pk8(o);
      xs[q][1] = pk8(o + 8);
      __nv_bfloat16* xg = XT + (size_t)(m0 + (int)r) * xstride + (int)c * NC + 32 * q + (int)cofs;
      stg128(xg, xs[q][0]);
      stg128(xg + 8, xs[q][1]);
      fence_proxy_async();
      named_bar_sync(1, 256);                                  // block q's s2 / sh2 read by all
      release(is);
#pragma unroll
      for (int k = 0; k < KX; ++k) {
        if (stage_at(k) != q) continue;
        const uint32_t dst = box(I_XT + k);                    // 64-column k-block k = epilogue blocks 2 k, 2 k + 1
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2)
#pragma unroll
          for (int e = 0; e < 2; ++e) sts128(dst + sw128(r, (uint32_t)(4 * h2 + 2 * hh + e)), xs[2 * k + h2][e]);
      }
    }
    fence_proxy_async_global();                                // this thread's XT stores -> the peers' TMA loads (after the release)
    fence_proxy_async();                                       // the staged blocks -> the MMA
    tc_fence_before();                                         // TMEM Y read out before the MMA warp proceeds (p4done)
    named_bar_sync(1, 256);
    if (tid == 128) {
      mbar_arrive(&B.p4done);
      if constexpr (!OWNH) mbar_arrive(&B.zfree);              // CL 6: Z has its own columns
      for (int k = 0; k < KX; ++k) mbar_arrive(&B.afull[(I_XT + k) % NAR]);   // the own xt blocks: the a | b GEMM starts on them
      TR(5);
      fence_acq_rel_cluster();                                 // one release for the CL relaxed arrivals below
      for (int p = 0; p < CL; ++p) mbar_arrive_remote_relaxed(&B.xtready, (c + (uint32_t)p) % CL);
      TR(6);
    }
    // ---- P6: h = silu(a) b per pass, per 64-column h k-block (this thread's 32 columns: 4 chunks of the 128-B row)
#pragma unroll 1
    for (int pass = 0; pass < NPASS; ++pass) {
      mbar_wait(&B.abdone, pass & 1);
      tc_fence_after();
      if (tid == 128) TR(pass ? 10 : 7);
#pragma unroll 1
      for (int j = 0; j < HOP; ++j) {
        uint4 hq[4];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          uint32_t av[16], bv[16];
          tmem_ld16w(trow + T_A + KB * j + 32 * hh + 16 * u, av);
          tmem_ld16w(trow + T_B + KB * j + 32 * hh + 16 * u, bv);
          float hv[16];
#pragma unroll
          for (int k = 0; k < 16; ++k) hv[k] = silu32(__uint_as_float(av[k])) * __uint_as_float(bv[k]);
          hq[2 * u] = pk8(hv);
          hq[2 * u + 1] = pk8(hv + 8);
        }
        if constexpr (OWNH) {                                  // CL 8: the slot of ring position I_H + j (idle: the ring waits for hready)
          const uint32_t dst = box(I_H + j);
#pragma unroll
          for (int e = 0; e < 4; ++e) sts128(dst + sw128(r, (uint32_t)(4 * hh + e)), hq[e]);
          __syncwarp();                                        // this warp's rows / half of block j -> H (st.global)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint32_t rr = rr0 + 8 * i;
            stg128(h_elem(HO * (int)c + j, (int)rr, 32 * hh + 8 * (int)qq), lds128(dst + sw128(rr, 4 * hh + qq)));
          }
        } else {                                               // CL 6: h k-block HO c + HOP pass + j through this warp's tile
          const uint32_t wtile = su + O_TILE + (uint32_t)(warp - 4) * 2048;
          const int kb = HO * (int)c + pass * HOP + j;
#pragma unroll
          for (int e = 0; e < 4; ++e) sts128(wtile + sw64((uint32_t)lane, (uint32_t)e), hq[e]);
          __syncwarp();
#pragma unroll
          for (int i = 0; i < 4; ++i) {                        // rows 8 i .. 8 i + 7 of this warp's 32, 64 B each
            const uint32_t rr = 8 * i + (lane >> 2);
            stg128(h_elem(kb, 32 * q4 + (int)rr, 32 * hh + 8 * (int)qq), lds128(wtile + sw64(rr, qq)));
          }
          __syncwarp();
        }
      }
      fence_proxy_async_global();                              // this thread's H stores, before the release below
      if constexpr (OWNH) fence_proxy_async();                 // the staged own blocks -> the MMA
      tc_fence_before();                                       // TMEM A | B read out: Z / the next pass may accumulate there
      named_bar_sync(1, 256);
      if (tid == 128) {
        if (pass + 1 < NPASS) {
          mbar_arrive(&B.abfree);
        } else {
          if constexpr (OWNH) {
            mbar_arrive(&B.zfree);
            for (int j = 0; j < HO; ++j) mbar_arrive(&B.afull[(I_H + j) % NAR]);  // the own h blocks: z starts on them
          }
          TR(8);
          fence_acq_rel_cluster();
          for (int p = 0; p < CL; ++p) mbar_arrive_remote_relaxed(&B.hready, (c + (uint32_t)p) % CL);
          TR(9);
        }
      }
    }
    // ---- P8: out = x1 + gate2 z; fp32 out through this warp's [32 rows][64 B] tile in an idle slot -> st.global.v4, bf16 out
    // straight from the registers (32 B of this thread's row)
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
      float gv[16], o[16];
      ld16_bf(box(I_G + q), r, hh, gv);
#pragma unroll
      for (int k = 0; k < 16; ++k) o[k] = fmaf(gv[k], __uint_as_float(zv[k]), __uint_as_float(xv[k]));
#if OBF
      __nv_bfloat16* og = reinterpret_cast<__nv_bfloat16*>(OUTV) + (size_t)(m0 + (int)r) * D + (int)c * NC + 32 * q + (int)cofs;
      stg128(og, pk8(o));
      stg128(og + 8, pk8(o + 8));
#else
#pragma unroll
      for (int e = 0; e < 4; ++e)
        sts128(wtile + sw64((uint32_t)lane, (uint32_t)e), make_uint4(__float_as_uint(o[4 * e]), __float_as_uint(o[4 * e + 1]),
                                                                       __float_as_uint(o[4 * e + 2]), __float_as_uint(o[4 * e + 3])));
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t rr = 8 * i + (lane >> 2);
        stg128(reinterpret_cast<float*>(OUTV) + (size_t)(m0 + 32 * q4 + (int)rr) * D + (int)c * NC + 32 * q + (int)cofs + 4 * (int)qq,
               lds128(wtile + sw64(rr, qq)));
      }
      __syncwarp();
#endif
    }
    if (tid == 128) TR(14);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
