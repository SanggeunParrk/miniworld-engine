// bo_bwd_tail.cu -- the bias-only token DiT training backward from d out to the attention core's inputs on CTA PAIRS, sm_100a
// (integrations/bias_only_dit_train.py, MINIWORLD_BIAS_ONLY_DIT_BWD_TAIL=split3): what res_c_bwd, the dh GEMM, swiglu_bwd, the dxt
// GEMM, res_adaln_b_bwd, the d(og) GEMM and gate_bwd did as seven launches, in three launches of one kernel.
//
// Work items per PAIR TILE p (the 128-row tiles 2p and 2p + 1 of the A L rows; T tiles, the last pair's second tile absent when T
// is odd): a cluster of two CTAs takes each item, rank r working on tile rt = 2p + r (bf16 operands, fp32 accumulation, row math).
//   R1(p)     row pass   per CTA, its tile: s2g = sigmoid(Gg[:, D:] + bg2); dz = dout s2g -> DZ; dg2 = dout z s2g (1 - s2g) -> DGG[:, D:];
//                        bg2 partials
//   G1(p, j)  GEMM       dh[:, 256 j ..] = dz Wsq[:, 256 j ..] (K 768) for both tiles, SwiGLU backward in each CTA's epilogue (a, b
//                        of those units from the saved a | b) -> DAB[:, 256 j ..], DAB[:, 1536 + 256 j ..]                 (j < 6)
//   G2(p, n)  GEMM       dxt[:, 256 n ..] = dab Wab[:, 256 n ..] (K 3072) -> DG[:, 3D + 256 n ..] (bf16)            (n < 3)
//   R2(p)     row pass   per CTA, its tile: the transition's LayerNorm / AdaLN backward (res_adaln_b_bwd): ds2 -> DG[:, 2D:], dx1
//                        (fp32) -> DX1, dy = dx1 sigmoid(g1) -> DY, dg1 -> DGG[:, :D]; bs2, bg1 partials; rows on chip
//   G4(p, n)  GEMM       dog[:, NG4 n ..] = dy Wo[:, NG4 n ..] (K 768), the gate backward in each CTA's epilogue: do = dog sigmoid(g)
//                        -> DO, dg = dog og (1 - sigmoid(g)) -> DVG[:, DA:], D[a, h, i] = sum over head h of dog og -> DD  (n < DA / NG4)
// GEMM items are M 256 x N (256, or NG4) products on the pair (tcgen05 cta_group::2, as cuBLAS's pair kernels): each CTA loads its
// own tile's A rows [128][64] and HALF of the B tile ([N / 2][64]: rank 0 rows n0 .. n0 + N/2, rank 1 the rest), so a k-block moves
// 16 + N / 2 x 128 bytes per CTA instead of 16 + N x 128 KB; the leader issues the products, each CTA's TMEM gets its own 128 rows x
// N columns, and each CTA's epilogue is the single-CTA one on its rows. B operands are the weights transposed once per pack (WsqT
// [1536, 768], WaT / WbT [768, 1536] -- G2's k-blocks 0-23 / 24-47 --, WoT [DA, 768]): K-major.
//
// Launches (the host, bwd_fused.tail_launches): [R1, G1, G2] (DEPMASK: G1 / G2 wait on per-tile counters), [R2], [G4]. Row items
// always stream through the ROW ring (the whole operand + E area as 27 x 8 KB slots), GEMM items' epilogue inputs through the GEMM E
// ring (NE slots after the operand stages): in launch 1 a CTA's row items all come before its GEMM items (the table is phase-major),
// and it hands the area over once (rfree below). Items are taken by index: pair c of the grid's P pairs takes table entries
// c, c + P, ...; the table is phase-major (every R1, then every G1, then every G2, pair-tile order inside), so an item waits only on
// globally earlier ones and a pair runs its items in table order: the globally earliest unfinished item can always run -- no deadlock
// (grid <= SMs, one CTA per SM, all resident). Both CTAs of a pair walk the same item sequence (no exchange of item codes). Results do
// not depend on which pair runs what: bit-identical reruns.
//
// PDL: launched with programmatic dependent launch; every thread griddepcontrol.wait after the setup (barriers, TMEM, cluster
// barrier), before any global access, and launch_dependents right before it, so only the next launch's setup overlaps this one.
//
// SYNCHRONIZATION (per CTA unless said otherwise; phase of a barrier = its use count & 1):
//   item ring  B.item[NIR]   filler: producer (warp 0 lane 0) writes the code, mbarrier.arrive ifull[s] (count 1)
//                            consumers: MMA warp, E-loader (warp 2 lane 0), epilogue (warps 4-11) wait ifull[s]; MMA and E-loader
//                            arrive iempty[s] once they have read the code, the epilogue (thread 128) after the item (count 3); the
//                            producer waits iempty[s] before it writes the slot again
//   operands   NST stages    filler: producer of EACH CTA: j >= NST: wait its own oempty[s] parity ((j / NST) - 1) & 1; the leader
//              (A | B half)  arrive.expect_tx on ITS ofull[s] for both CTAs' bytes (2 (A + B half)); each CTA's TMA loads (cta_group::2)
//                            complete on the LEADER's ofull[s]. consumer: the leader's MMA warp waits ofull[s] parity (j / NST) & 1,
//                            tcgen05.fence::after_thread_sync, 4 MMAs of K 16 (cta_group::2, reading both CTAs' stage s),
//                            tcgen05.commit cta_group::2 multicast {0, 1} -> oempty[s] of both CTAs (count 1 each)
//   TMEM       2 x 256 cols  the GEMM items alternate buffers (GEMM count g, buffer g & 1): the leader's MMA waits ITS tempty[b]
//              (cta_group::2 (acquire.cluster) parity ((g >> 1) - 1) & 1 for g >= 2 (count 2: one arrival per CTA),
//               alloc)       tcgen05.fence::after_thread_sync, products, tcgen05.commit multicast {0, 1} -> tfull[b] of both CTAs after
//                            the item's last stage; each epilogue waits its own tfull[b] parity (g >> 1) & 1, tcgen05.fence::
//                            after_thread_sync, tcgen05.ld ...; after its last read every epilogue thread tcgen05.fence::
//                            before_thread_sync, named barrier 1 (256), thread 128 mbarrier.arrive.release.cluster on the LEADER's
//                            tempty[b]
//   row ring   27 x 8 KB     each CTA, row items' inputs ([4 rows][768] boxes, un-swizzled) from offset 0 (over the operand stages and
//                            the E slots): filler E-loader, arrive.expect_tx efull[e] + TMA; consumer every epilogue thread waits
//                            efull[e]; after reading a slot each warp fence.proxy.async, __syncwarp, lane 0 arrives eempty[e] (count
//                            8); the E-loader waits eempty[e] before reusing the slot
//   rfree      1 barrier     launch 1, a CTA that ran row items: at its first GEMM item the E-loader waits eempty for the last use of
//                            every row-ring slot (all row reads done), then arrives rfree (count 1); the producer waits rfree (parity
//                            0) before its first operand load; only then does anything write the area as stages / GEMM E slots
//   GEMM ring  NE x 8 KB     each CTA, GEMM items' epilogue inputs (SW64 [128][32] boxes of a | b, og, g) after the operand stages:
//                            filler E-loader, arrive.expect_tx gfull[e] + TMA; consumer every epilogue thread waits gfull[e]; the
//                            slots are freed by the stores below (gempty, count 8)
//   E stores   G1 / G4       the epilogue writes its results over its own inputs in the same slot (each thread its own 32 B of the
//                            SW64 box), fence.proxy.async, named barrier 6 + q (64 threads: the two warps of rows 32 q .. 32 q + 31);
//                            lane 0 of the first of them TMA-stores those 32 rows of both boxes ([32][32], dab a | b, do | dg),
//                            commit_group, and after cp.async.bulk.wait_group.read 1 frees the PREVIOUS chunk's two slots for its
//                            pair of warps (mbarrier.arrive count 2 each: four storing lanes make the 8); at the item's end each
//                            storing lane wait_group.read 0 + frees its last slots + wait_group 0 (the writes performed) before the
//                            item-done barrier, so the tile count follows every store
//   tiles      CNT[r]        global, one per 128-row tile: the epilogue of R1 / G1 / G2 / R2 items of tile r (the CTA that owns it),
//                            after its stores, each thread fence.proxy.async.global, named barrier 1, thread 128 fence.acq_rel.gpu +
//                            red.relaxed.gpu.add 1. Waiters (producer: G1 >= 1, G2 >= 7, G4 >= 11 before its A loads, its own tile;
//                            E-loader: R2 >= 10 before the dxt boxes) spin on ld.acquire.gpu, then fence.proxy.async.global, then TMA
//   done       CNT[T]        the CTA-done count; the last CTA zeroes CNT[0 .. T] for the next launch (stream order; graph replays too)
//   row pass   named barriers 2 + slot (64 threads: the two warps of a row) exchange the row sums (double-buffered by parity)
//   setup / exit: barrier init by thread 0, fence.mbarrier_init, cta_group::2 TMEM alloc, __syncthreads + cluster barrier before any
//                            peer arrival; exit: tcgen05.fence::before_thread_sync, __syncthreads, cluster barrier (no CTA leaves while
//                            its peer may still arrive on its barriers), cta_group::2 dealloc.
//   odd T: the second CTA of the last pair has no tile -- its A rows and E boxes come back zero-filled (out of bounds), its stores are
//          skipped (TMA stores clipped), its row items do nothing, and it counts no tile.
//
// Warps: 0 producer (lane 0), 1 MMA (the leader's: whole warp waits, elect_one() issues), 2 TMEM allocator + E-loader (lane 0), 3 idle,
// 4-11 epilogue: a GEMM item's thread = TMEM lane (row 32 (warp & 3) + lane of this CTA's tile) and column half (warp - 4) >> 2 of every
// 32-column chunk; a row item's 4-row group: row slot (warp - 4) >> 1, two warps per row, a lane owning 4-column chunks hw * 3 + j (j < 3).
// -DTRACE: per item (indexed by QBASE + its table index q < TRQ; the two CTAs of a pair write the same slots, last writer wins)
// %globaltimer events of every role into g_ev[q][16] (bo_bwd_trace.py):
//   0 claimed, 1 producer: dependency met, 2 producer: last load issued, 3 MMA: first stage seen, 4 MMA: last commit issued,
//   5 E-loader: item seen, 6 E-loader: dependency met / first box issued, 7 epilogue: item seen, 8 epilogue: first input (TMEM full or
//   E slot), 9 epilogue: item done; 10 CTA, 11 item code; and per item the ns spent waiting: 12 the leader's MMA warp on full operand
//   stages, 13 the producer on empty stages, 14 the epilogue (thread 128) on full E slots, 15 the E-loader on free E slots; g_cta0[CTA]
//   = the CTA's start. -DTAIL_NST=3 / 4 / 5: operand stages (the E ring gets the rest: 15 / 11 / 7 slots; the row ring stays 27).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DATT
#define DATT 768
#endif
#ifndef NHEAD
#define NHEAD 16
#endif
constexpr int DHD = DATT / NHEAD;
static_assert(DHD == 32 || DHD == 48 || DHD == 64, "head width 32, 48 or 64");
static_assert(DATT == 768 || DATT == 1024, "768 or 1024 attention channels");

constexpr int D = 768, QM = 128, KB = 64;
constexpr int NG1 = 256, NI1 = 1536 / NG1, NK1 = D / KB;
constexpr int NG2 = 256, NI2 = D / NG2, NK2 = 3072 / KB;
constexpr int NG4 = DATT == 768 ? 192 : 256, NK4 = D / KB;
static_assert(NG4 % DHD == 0 && NG4 % 32 == 0 && DHD % 16 == 0, "G4 tiles hold whole heads; 16-column groups inside a head");
enum { T_R1 = 0, T_G1 = 1, T_G2 = 2, T_R2 = 3, T_G4 = 4, T_STOP = 7 };
constexpr uint32_t NEED_G1 = 1, NEED_G2 = 1 + NI1, NEED_R2 = 1 + NI1 + NI2, NEED_G4 = NEED_R2 + 1;

#ifndef TAIL_NST
#define TAIL_NST 4                                                          // bwdf9: 4 beat 3 and 5 in the graph node A/B
#endif
constexpr int ASZ = QM * 128, BHZ = 128 * 128, STG = ASZ + BHZ;            // operand stage: A [128][64] | B half [<= 128][64] bf16
constexpr int ESZ = 8192, NIR = 2;                                        // NIR: items a CTA holds (taken, not done)
constexpr int NST = TAIL_NST, NE = (27 * ESZ - NST * STG) / ESZ;          // stages + E slots = the 27-slot row ring's 216 KB
static_assert(NST >= 3 && NST <= 5 && NE >= 7, "3 to 5 operand stages");
constexpr int RG = 4, NGR = QM / RG, RGB = RG * D * 2;                     // row passes: 4-row groups, one [4][768] bf16 box per tensor
constexpr int HSP = 17;                                                     // G4 head-sum scratch: [128 rows][16 (+1) column groups]
constexpr int O_OP = 0, O_E = NST * STG, O_HS = O_E + NE * ESZ, O_BAR = O_HS + QM * HSP * 4, SMEM_BYTES = O_BAR + 1024;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(O_E % 1024 == 0 && STG % 1024 == 0 && ESZ % 1024 == 0 && RGB <= ESZ, "alignment");
constexpr int NE_ROW = O_HS / ESZ;                                          // the row ring: the operand stages + the E slots
static_assert(NE_ROW * ESZ == O_HS && NE_ROW >= 2 * 6, "row ring: two R2 groups in flight at least");
constexpr uint32_t I_256 = idesc_bf16(2 * QM, 256), I_G4 = idesc_bf16(2 * QM, NG4);   // M 256: the pair's two tiles

struct Bars {
  uint64_t ifull[NIR], iempty[NIR], ofull[NST], oempty[NST], tfull[2], tempty[2], efull[NE_ROW], eempty[NE_ROW], gfull[NE], gempty[NE];
  uint64_t rfree;
  int item[NIR], qid[NIR];
  uint32_t tmem;
  int last;
};
static_assert(sizeof(Bars) <= 1024 - 64, "barriers");

#ifdef TRACE
constexpr int TRQ = 32768;                                                  // ten graph copies of the L768 tail
__device__ unsigned long long g_ev[TRQ * 16];
__device__ unsigned long long g_cta0[1024];
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define EV(q, k) do { if ((unsigned)(q) < (unsigned)TRQ) g_ev[(q) * 16 + (k)] = gtime(); } while (0)
#define EVV(q, k, v) do { if ((unsigned)(q) < (unsigned)TRQ) g_ev[(q) * 16 + (k)] = (unsigned long long)(v); } while (0)
#define WT0(t) const unsigned long long t = gtime()
#define WT1(acc, t) acc += gtime() - t
#else
#define WT0(t) do { } while (0)
#define WT1(acc, t) do { } while (0)
#define EV(q, k) do { } while (0)
#define EVV(q, k, v) do { } while (0)
#endif

DEVI uint32_t ld_acquire(const unsigned* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
DEVI void wait_count(const unsigned* p, uint32_t need) {
  while (ld_acquire(p) < need) __nanosleep(128);
}
DEVI void fence_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
DEVI void red_add(unsigned* p, unsigned v) { asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI uint2 lds64u(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v; }
DEVI void stg64(void* p, uint2 v) { asm volatile("st.global.v2.b32 [%0], {%1,%2};" :: "l"(p), "r"(v.x), "r"(v.y) : "memory"); }
DEVI float4 f4bf(uint2 u) { return make_float4(bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y)); }
DEVI uint2 bf4(float4 v) { return make_uint2(pack_bf16(v.x, v.y), pack_bf16(v.z, v.w)); }
DEVI float sg(float v) { return __fdividef(1.f, 1.f + __expf(-v)); }
DEVI float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }      // the value a bf16 store keeps
DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEVI void tmem_ld16w(uint32_t taddr, uint32_t (&r)[16]) { tmem_ld16(taddr, r); tmem_wait_ld(); }
// 16 values of a [128][32] bf16 SW64 box: row r, columns 16 hh .. 16 hh + 15
DEVI void ld16_bf(uint32_t bx, uint32_t r, int hh, float (&v)[16]) {
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const uint4 u = lds128(bx + sw64(r, (uint32_t)(2 * hh + e)));
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) { v[8 * e + 2 * k] = bf16lo(w[k]); v[8 * e + 2 * k + 1] = bf16hi(w[k]); }
  }
}
DEVI void mbar_arrive_cnt(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(n) : "memory");
}
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }
DEVI uint4 pk8(const float* v) {
  return make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7]));
}

extern "C" __global__ void __launch_bounds__(384, 1)
bo_bwd_tail_sm100(const __grid_constant__ CUtensorMap mr_dout, const __grid_constant__ CUtensorMap mr_z,
                  const __grid_constant__ CUtensorMap mr_g2, const __grid_constant__ CUtensorMap mr_dxt,
                  const __grid_constant__ CUtensorMap mr_x, const __grid_constant__ CUtensorMap mr_y,
                  const __grid_constant__ CUtensorMap mr_g1, const __grid_constant__ CUtensorMap mr_s2,
                  const __grid_constant__ CUtensorMap me_ab, const __grid_constant__ CUtensorMap me_og,
                  const __grid_constant__ CUtensorMap me_vg, const __grid_constant__ CUtensorMap ma_dz,
                  const __grid_constant__ CUtensorMap ma_dab, const __grid_constant__ CUtensorMap ma_dy,
                  const __grid_constant__ CUtensorMap mb_wsq, const __grid_constant__ CUtensorMap mb_wa,
                  const __grid_constant__ CUtensorMap mb_wb, const __grid_constant__ CUtensorMap mb_wo,
                  const __grid_constant__ CUtensorMap ms_dab, const __grid_constant__ CUtensorMap ms_do,
                  const __grid_constant__ CUtensorMap ms_dvg,
                  const int* __restrict__ ITEMS, unsigned* __restrict__ CNT,
                  const float2* __restrict__ X1ST,
                  const float* __restrict__ BG2, const float* __restrict__ BS2, const float* __restrict__ BG1,
                  __nv_bfloat16* __restrict__ DZ, __nv_bfloat16* __restrict__ DGG, __nv_bfloat16* __restrict__ DAB,
                  __nv_bfloat16* __restrict__ DG, float* __restrict__ DX1, __nv_bfloat16* __restrict__ DY,
                  __nv_bfloat16* __restrict__ DO, __nv_bfloat16* __restrict__ DVG, float* __restrict__ DD, float* __restrict__ PART,
                  int T, int NITEMS, int L, int PROW, int DEPMASK, int QBASE) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int crank = (int)cluster_rank(), pair = (int)blockIdx.x >> 1, npairs = (int)gridDim.x >> 1;
  const bool leader = crank == 0;
  unsigned* const done = CNT + T;

  if (tid == 0) {
    for (int i = 0; i < NIR; ++i) { mbar_init(&B.ifull[i], 1); mbar_init(&B.iempty[i], 3); }
    for (int i = 0; i < NST; ++i) { mbar_init(&B.ofull[i], 1); mbar_init(&B.oempty[i], 1); }
    for (int i = 0; i < 2; ++i) { mbar_init(&B.tfull[i], 1); mbar_init(&B.tempty[i], 2); }   // tempty: one arrival per CTA
    for (int i = 0; i < NE_ROW; ++i) { mbar_init(&B.efull[i], 1); mbar_init(&B.eempty[i], 8); }
    for (int i = 0; i < NE; ++i) { mbar_init(&B.gfull[i], 1); mbar_init(&B.gempty[i], 8); }
    mbar_init(&B.rfree, 1);
    fence_barrier_init();
    prefetch_map(&mr_dout); prefetch_map(&mr_z); prefetch_map(&mr_g2); prefetch_map(&mr_dxt); prefetch_map(&mr_x);
    prefetch_map(&mr_y); prefetch_map(&mr_g1); prefetch_map(&mr_s2); prefetch_map(&me_ab); prefetch_map(&me_og);
    prefetch_map(&me_vg); prefetch_map(&ma_dz); prefetch_map(&ma_dab); prefetch_map(&ma_dy); prefetch_map(&mb_wsq);
    prefetch_map(&mb_wa); prefetch_map(&mb_wb); prefetch_map(&mb_wo); prefetch_map(&ms_dab); prefetch_map(&ms_do);
    prefetch_map(&ms_dvg);
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();                                            // the peer's barriers initialized before any arrival on them
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  // PDL: the next launch may start its setup now; this one waits for its predecessor (the previous tail launch, whose last CTA also
  // zeroes CNT) before any global access
  pdl_launch();
  pdl_wait();
#ifdef TRACE
  if (tid == 0 && blockIdx.x < 1024) g_cta0[blockIdx.x] = gtime();
#endif

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ producer: items, operands
    // both CTAs: the pair's items by index; GEMM items: this CTA's A tile and its half of B, completing on the leader's ofull
    if (lane == 0) {
      uint32_t j = 0;
      bool rows = false, handed = false;                     // row items run here / the row ring handed over (rfree)
      for (uint32_t it = 0;; ++it) {
        const int is = (int)(it % NIR);
        if (it >= NIR) mbar_wait(&B.iempty[is], ((it / NIR) - 1) & 1);
        const unsigned q0 = (unsigned)(pair + (int)it * npairs);
        const int code = q0 < (unsigned)NITEMS ? __ldg(ITEMS + q0) : T_STOP;
        const unsigned q = q0 + (unsigned)QBASE;
        B.item[is] = code;
        B.qid[is] = (int)q;
        mbar_arrive(&B.ifull[is]);
        const int ty = code & 7;
        if (ty == T_STOP) break;
        EV(q, 0); EVV(q, 10, blockIdx.x); EVV(q, 11, code);
        if (ty == T_R1 || ty == T_R2) { rows = true; EV(q, 1); continue; }
        if (rows && !handed) { mbar_wait(&B.rfree, 0); handed = true; }   // the row ring drained: the area is the stages now
        const int n = (code >> 3) & 15, rt = 2 * (code >> 8) + crank;
        if (((DEPMASK >> ty) & 1) && rt < T) wait_count(CNT + rt, ty == T_G1 ? NEED_G1 : ty == T_G2 ? NEED_G2 : NEED_G4);
        EV(q, 1);
        fence_proxy_async_global();                            // the A rows were stored by other CTAs' generic proxy
        const CUtensorMap* ma = ty == T_G1 ? &ma_dz : ty == T_G2 ? &ma_dab : &ma_dy;
        const int nk = ty == T_G2 ? NK2 : ty == T_G4 ? NK4 : NK1, nb = ty == T_G4 ? NG4 : 256, nbh = nb / 2;
#ifdef TRACE
        unsigned long long wsum = 0;
#endif
        for (int kb = 0; kb < nk; ++kb, ++j) {
          const int s = (int)(j % NST);
          if (j >= NST) { WT0(tw); mbar_wait(&B.oempty[s], ((j / NST) - 1) & 1); WT1(wsum, tw); }
          if (leader) mbar_expect_tx(&B.ofull[s], (uint32_t)(2 * (ASZ + nbh * 128)));
          const uint32_t dst = su + O_OP + (uint32_t)s * STG;
          tma_load_2d_2sm(dst, ma, &B.ofull[s], KB * kb, rt * QM);
          const CUtensorMap* mb = ty == T_G1 ? &mb_wsq : ty == T_G4 ? &mb_wo : kb < NK2 / 2 ? &mb_wa : &mb_wb;
          tma_load_2d_2sm(dst + ASZ, mb, &B.ofull[s], KB * (ty == T_G2 && kb >= NK2 / 2 ? kb - NK2 / 2 : kb), n * nb + crank * nbh);
        }
        EV(q, 2);
#ifdef TRACE
        EVV(q, 13, wsum);
#endif
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer (the leader's warp)
    uint32_t j = 0, g = 0;
    for (uint32_t it = 0;; ++it) {
      const int is = (int)(it % NIR);
      mbar_wait(&B.ifull[is], (it / NIR) & 1);
      const int code = B.item[is];
#ifdef TRACE
      const int qi = B.qid[is];
#endif
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.iempty[is]);
      const int ty = code & 7;
      if (ty == T_STOP) break;
      if (ty == T_R1 || ty == T_R2) continue;
      const int nk = ty == T_G2 ? NK2 : ty == T_G4 ? NK4 : NK1;
      if (leader) {
        const int tb = (int)(g & 1);
        if (g >= 2) mbar_wait_cl(&B.tempty[tb], ((g >> 1) - 1) & 1);     // both CTAs' epilogues done with buffer tb
        tc_fence_after();
        const uint32_t idesc = ty == T_G4 ? I_G4 : I_256, d = tmem + (uint32_t)tb * 256;
#ifdef TRACE
        unsigned long long wsum = 0;
#endif
        for (int kb = 0; kb < nk; ++kb, ++j) {
          const int s = (int)(j % NST);
          { WT0(tw); mbar_wait(&B.ofull[s], (j / NST) & 1); WT1(wsum, tw); }
          tc_fence_after();
#ifdef TRACE
          if (lane == 0 && kb == 0) EV(qi, 3);
#endif
          if (elect_one()) {
            const uint64_t da = desc_k128(su + O_OP + (uint32_t)s * STG), db = desc_k128(su + O_OP + (uint32_t)s * STG + ASZ);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) umma_ss2(d, da + (uint64_t)(2 * ks), db + (uint64_t)(2 * ks), idesc, (kb | ks) ? 1u : 0u);
            tc_commit2_mc(&B.oempty[s], 3);
            if (kb == nk - 1) tc_commit2_mc(&B.tfull[tb], 3);
          }
          __syncwarp();
        }
#ifdef TRACE
        if (lane == 0) { EV(qi, 4); EVV(qi, 12, wsum); }
#endif
      }
      ++g;
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ E-loader: the epilogue's inputs
    if (lane == 0) {
      const uint64_t once = pol_evict_first();
      uint32_t er = 0, eg = 0;                                 // uses of the row ring / of the GEMM ring
      bool rows = false, handed = false;
#ifdef TRACE
      unsigned long long lsum = 0;
#endif
      auto slot = [&](uint32_t bytes) -> uint32_t {            // the row ring's next slot, free, armed
        const int es = (int)(er % (uint32_t)NE_ROW);
        if (er >= (uint32_t)NE_ROW) { WT0(tw); mbar_wait(&B.eempty[es], ((er / NE_ROW) - 1) & 1); WT1(lsum, tw); }
        mbar_expect_tx(&B.efull[es], bytes);
        ++er;
        return su + O_OP + (uint32_t)es * ESZ;
      };
      auto bar_of = [&](uint32_t dst) { return &B.efull[(dst - su - O_OP) / ESZ]; };
      auto gslot = [&](uint32_t bytes) -> uint32_t {           // the GEMM ring's next slot, free, armed
        const int es = (int)(eg % (uint32_t)NE);
        if (eg >= (uint32_t)NE) { WT0(tw); mbar_wait(&B.gempty[es], ((eg / NE) - 1) & 1); WT1(lsum, tw); }
        mbar_expect_tx(&B.gfull[es], bytes);
        ++eg;
        return su + O_E + (uint32_t)es * ESZ;
      };
      auto gbar_of = [&](uint32_t dst) { return &B.gfull[(dst - su - O_E) / ESZ]; };
      for (uint32_t it = 0;; ++it) {
        const int is = (int)(it % NIR);
        mbar_wait(&B.ifull[is], (it / NIR) & 1);
        const int code = B.item[is];
#ifdef TRACE
        const int qi = B.qid[is];
#endif
        mbar_arrive(&B.iempty[is]);
        const int ty = code & 7;
        if (ty == T_STOP) break;
        const int n = (code >> 3) & 15, rt = 2 * (code >> 8) + crank, row0 = rt * QM;
        if (ty == T_R1 || ty == T_R2) {
          rows = true;
          if (rt >= T) continue;                                 // the absent tile of an odd T: no row work
        } else if (rows && !handed) {                            // first GEMM item after row items: drain the row ring, hand it over
          for (uint32_t k = er > (uint32_t)NE_ROW ? er - NE_ROW : 0; k < er; ++k) mbar_wait(&B.eempty[k % NE_ROW], (k / NE_ROW) & 1);
          mbar_arrive(&B.rfree);
          handed = true;
        }
        EV(qi, 5);
#ifdef TRACE
        lsum = 0;
#endif
        if (ty != T_R2) EV(qi, 6);
        if (ty == T_R1) {
          for (int gq = 0; gq < NGR; ++gq) {
            const CUtensorMap* ms[3] = {&mr_dout, &mr_z, &mr_g2};
#pragma unroll
            for (int k = 0; k < 3; ++k) { const uint32_t dst = slot(RGB); tma_load_3d(dst, ms[k], bar_of(dst), 0, 0, row0 + RG * gq); }
          }
        } else if (ty == T_R2) {
          if ((DEPMASK >> T_R2) & 1) wait_count(CNT + rt, NEED_R2);   // dxt: the G2 items' stores (in this launch)
          fence_proxy_async_global();
          EV(qi, 6);
          for (int gq = 0; gq < NGR; ++gq) {
            const CUtensorMap* ms[6] = {&mr_dxt, &mr_x, &mr_y, &mr_g1, &mr_s2, &mr_dout};
#pragma unroll
            for (int k = 0; k < 6; ++k) { const uint32_t dst = slot(RGB); tma_load_3d(dst, ms[k], bar_of(dst), 0, 0, row0 + RG * gq); }
          }
        } else if (ty == T_G1) {
          for (int q = 0; q < NG1 / 32; ++q) {
            uint32_t dst = gslot(QM * 64);
            tma_load_2d_h(dst, &me_ab, gbar_of(dst), NG1 * n + 32 * q, row0, once);
            dst = gslot(QM * 64);
            tma_load_2d_h(dst, &me_ab, gbar_of(dst), 1536 + NG1 * n + 32 * q, row0, once);
          }
        } else if (ty == T_G4) {
          for (int q = 0; q < NG4 / 32; ++q) {
            uint32_t dst = gslot(QM * 64);
            tma_load_2d_h(dst, &me_og, gbar_of(dst), NG4 * n + 32 * q, row0, once);
            dst = gslot(QM * 64);
            tma_load_2d_h(dst, &me_vg, gbar_of(dst), DATT + NG4 * n + 32 * q, row0, once);
          }
        }
#ifdef TRACE
        EVV(qi, 15, lsum);
#endif
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ epilogue / row passes
    const int ew = warp - 4, q4 = warp & 3, hh = ew >> 2;
    const uint32_t r = (uint32_t)(32 * q4 + lane), trow = tmem + ((uint32_t)(32 * q4) << 16);
    const int rs = ew >> 1, hw = ew & 1;                       // row passes: row slot of the group, warp half of the row
    float* const hs = reinterpret_cast<float*>(sm + O_HS);
    float2* const xr = reinterpret_cast<float2*>(sm + O_HS);   // row passes' exchange [2][4][2] (the G4 scratch is idle then)
    int par = 0;
    uint32_t er = 0, eg = 0, g = 0;                           // uses of the row ring / of the GEMM ring; GEMM items
    bool rr = false;                                          // this item reads the row ring (else the GEMM ring)
#ifdef TRACE
    unsigned long long esum = 0;
#endif
    auto ewait = [&]() -> uint32_t {                          // the next slot of this item's ring, full
#ifdef TRACE
      const unsigned long long tw = gtime();
#endif
      uint32_t a;
      if (rr) {
        const int es = (int)(er % (uint32_t)NE_ROW);
        mbar_wait(&B.efull[es], (er / NE_ROW) & 1);
        ++er;
        a = su + O_OP + (uint32_t)es * ESZ;
      } else {
        const int es = (int)(eg % (uint32_t)NE);
        mbar_wait(&B.gfull[es], (eg / NE) & 1);
        ++eg;
        a = su + O_E + (uint32_t)es * ESZ;
      }
#ifdef TRACE
      if (tid == 128) esum += gtime() - tw;
#endif
      return a;
    };
    auto erelease = [&](uint32_t addr) {                     // this warp is done reading the row-ring slot
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.eempty[(addr - su - O_OP) / ESZ]);
    };
    auto gslot = [&](uint32_t addr) { return &B.gempty[(addr - su - O_E) / ESZ]; };
    const bool storer = hh == 0 && lane == 0;                 // stores rows 32 q4 .. + 31 of each chunk for its pair of warps
    uint32_t pa = 0, pb = 0;                                  // a storer: the slots of its last TMA-stored chunk, not yet freed
    auto store_chunk = [&](const CUtensorMap* m1, uint32_t s1, int c1, const CUtensorMap* m2, uint32_t s2, int c2, int row0) {
      fence_proxy_async();                                    // this thread's st.shared before the async-proxy store reads them
      named_bar_sync(6 + q4, 64);                             // the two warps of rows 32 q4 .. + 31
      if (storer) {
        const uint32_t off = (uint32_t)q4 * 32 * 64;          // their [32][32] part of the SW64 box (512-byte aligned)
        tma_store_2d(m1, s1 + off, c1, row0 + 32 * q4);
        tma_store_2d(m2, s2 + off, c2, row0 + 32 * q4);
        tma_store_commit();
        if (pa) { tma_store_wait_read1(); mbar_arrive_cnt(gslot(pa), 2); mbar_arrive_cnt(gslot(pb), 2); }
        pa = s1; pb = s2;
      }
    };
    auto store_done = [&]() {                                 // every epilogue thread, at the item's end (the storers act)
      if (storer) {
        if (pa) { tma_store_wait_read0(); mbar_arrive_cnt(gslot(pa), 2); mbar_arrive_cnt(gslot(pb), 2); pa = pb = 0; }
        tma_store_wait0();                                    // the writes performed before the tile count
      }
    };
    auto col = [&](int jj) { return ((hw * 3 + jj) * 32 + lane) * 4; };
    auto rowsum2 = [&](float a, float b) -> float2 {          // over the 64 threads of this row slot
      a = warp_sum(a); b = warp_sum(b);
      if (lane == 0) xr[(par * 4 + rs) * 2 + hw] = make_float2(a, b);
      named_bar_sync(2 + rs, 64);
      const float2 u = xr[(par * 4 + rs) * 2], v = xr[(par * 4 + rs) * 2 + 1];
      par ^= 1;
      return make_float2(u.x + v.x, u.y + v.y);
    };
    for (uint32_t it = 0;; ++it) {
      const int is = (int)(it % NIR);
      mbar_wait(&B.ifull[is], (it / NIR) & 1);
      const int code = B.item[is];
#ifdef TRACE
      const int qi = B.qid[is];
#endif
      const int ty = code & 7;
      if (ty == T_STOP) break;
      const int n = (code >> 3) & 15, rt = 2 * (code >> 8) + crank;
      const bool valid = rt < T;                               // false: the absent second tile of an odd T
      if (tid == 128) EV(qi, 7);
#ifdef TRACE
      esum = 0;
#endif
      rr = ty == T_R1 || ty == T_R2;
      if ((ty == T_R1 || ty == T_R2) && !valid) {
        // nothing: the E-loader issued nothing for it
      } else if (ty == T_R1) {
        // ---- R1: dz, dg2 per 4-row group; the bg2 column sums of this row slot over the item
        float4 bg[3], acc[3];
#pragma unroll
        for (int jj = 0; jj < 3; ++jj) { bg[jj] = __ldg(reinterpret_cast<const float4*>(BG2 + col(jj))); acc[jj] = make_float4(0.f, 0.f, 0.f, 0.f); }
        for (int gq = 0; gq < NGR; ++gq) {
          const uint32_t s_do = ewait(), s_z = ewait(), s_g = ewait();
          if (tid == 128 && gq == 0) EV(qi, 8);
          const long R = (long)rt * QM + RG * gq + rs;
#pragma unroll
          for (int jj = 0; jj < 3; ++jj) {
            const uint32_t o = (uint32_t)(rs * D + col(jj)) * 2;
            const float4 dout = f4bf(lds64u(s_do + o)), z = f4bf(lds64u(s_z + o)), gl = f4bf(lds64u(s_g + o));
            const float4 s = make_float4(sg(gl.x + bg[jj].x), sg(gl.y + bg[jj].y), sg(gl.z + bg[jj].z), sg(gl.w + bg[jj].w));
            stg64(DZ + R * D + col(jj), bf4(make_float4(dout.x * s.x, dout.y * s.y, dout.z * s.z, dout.w * s.w)));
            const float4 dg = make_float4(bfr(dout.x * z.x * s.x * (1.f - s.x)), bfr(dout.y * z.y * s.y * (1.f - s.y)),
                                          bfr(dout.z * z.z * s.z * (1.f - s.z)), bfr(dout.w * z.w * s.w * (1.f - s.w)));
            stg64(DGG + R * 1536 + D + col(jj), bf4(dg));
            acc[jj].x += dg.x; acc[jj].y += dg.y; acc[jj].z += dg.z; acc[jj].w += dg.w;
          }
          erelease(s_do); erelease(s_z); erelease(s_g);
        }
#pragma unroll
        for (int jj = 0; jj < 3; ++jj)
          *reinterpret_cast<float4*>(PART + ((long)0 * PROW + 4L * rt + rs) * D + col(jj)) = acc[jj];
      } else if (ty == T_R2) {
        // ---- R2: the transition's LayerNorm / AdaLN backward (res_adaln_b_bwd), rows on chip
        float4 b1[3], b2[3], accs[3], accg[3];
#pragma unroll
        for (int jj = 0; jj < 3; ++jj) {
          b1[jj] = __ldg(reinterpret_cast<const float4*>(BG1 + col(jj)));
          b2[jj] = __ldg(reinterpret_cast<const float4*>(BS2 + col(jj)));
          accs[jj] = make_float4(0.f, 0.f, 0.f, 0.f); accg[jj] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        for (int gq = 0; gq < NGR; ++gq) {
          const long R = (long)rt * QM + RG * gq + rs;
          const float2 st = X1ST[R];
          const uint32_t s_dxt = ewait(), s_x = ewait(), s_y = ewait(), s_g1 = ewait(), s_s2 = ewait(), s_do = ewait();
          if (tid == 128 && gq == 0) EV(qi, 8);
          // pass 1: the two sigmoids, xh and dxh of this thread's 12 columns, kept in registers for pass 2 (one sigmoid each, not two)
          float gsr[12], xhr[12], dxr[12];
          float m1 = 0.f, m2 = 0.f;
#pragma unroll
          for (int jj = 0; jj < 3; ++jj) {
            const uint32_t o = (uint32_t)(rs * D + col(jj)) * 2;
            const float4 dxt = f4bf(lds64u(s_dxt + o)), x = f4bf(lds64u(s_x + o)), y = f4bf(lds64u(s_y + o));
            const float4 g1l = f4bf(lds64u(s_g1 + o)), s2l = f4bf(lds64u(s_s2 + o));
            const float g1v[4] = {g1l.x + b1[jj].x, g1l.y + b1[jj].y, g1l.z + b1[jj].z, g1l.w + b1[jj].w};
            const float s2v[4] = {s2l.x + b2[jj].x, s2l.y + b2[jj].y, s2l.z + b2[jj].z, s2l.w + b2[jj].w};
            const float xv[4] = {x.x, x.y, x.z, x.w}, yv[4] = {y.x, y.y, y.z, y.w}, dv[4] = {dxt.x, dxt.y, dxt.z, dxt.w};
            float ds[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const float gs = sg(g1v[k]), ss = sg(s2v[k]);
              const float xh = (__fmaf_rn(gs, yv[k], xv[k]) - st.x) * st.y;
              ds[k] = bfr(dv[k] * xh * ss * (1.f - ss));
              const float dxh = dv[k] * ss;
              m1 += dxh;
              m2 += dxh * xh;
              gsr[4 * jj + k] = gs; xhr[4 * jj + k] = xh; dxr[4 * jj + k] = dxh;
            }
            stg64(DG + R * 3072 + 2 * D + col(jj), bf4(make_float4(ds[0], ds[1], ds[2], ds[3])));
            accs[jj].x += ds[0]; accs[jj].y += ds[1]; accs[jj].z += ds[2]; accs[jj].w += ds[3];
          }
          erelease(s_dxt); erelease(s_x); erelease(s_g1); erelease(s_s2);   // pass 2 reads only y and dout
          const float2 m = rowsum2(m1, m2);
          m1 = m.x * (1.f / D); m2 = m.y * (1.f / D);
#pragma unroll
          for (int jj = 0; jj < 3; ++jj) {
            const uint32_t o = (uint32_t)(rs * D + col(jj)) * 2;
            const float4 y = f4bf(lds64u(s_y + o)), dout = f4bf(lds64u(s_do + o));
            const float yv[4] = {y.x, y.y, y.z, y.w}, ov[4] = {dout.x, dout.y, dout.z, dout.w};
            float dx1[4], dy[4], dg1[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const float gs = gsr[4 * jj + k];
              dx1[k] = ov[k] + st.y * (dxr[4 * jj + k] - m1 - xhr[4 * jj + k] * m2);
              dy[k] = dx1[k] * gs;
              dg1[k] = bfr(dx1[k] * yv[k] * gs * (1.f - gs));
            }
            *reinterpret_cast<float4*>(DX1 + R * D + col(jj)) = make_float4(dx1[0], dx1[1], dx1[2], dx1[3]);
            stg64(DY + R * D + col(jj), bf4(make_float4(dy[0], dy[1], dy[2], dy[3])));
            stg64(DGG + R * 1536 + col(jj), bf4(make_float4(dg1[0], dg1[1], dg1[2], dg1[3])));
            accg[jj].x += dg1[0]; accg[jj].y += dg1[1]; accg[jj].z += dg1[2]; accg[jj].w += dg1[3];
          }
          erelease(s_y); erelease(s_do);
        }
#pragma unroll
        for (int jj = 0; jj < 3; ++jj) {
          *reinterpret_cast<float4*>(PART + ((long)1 * PROW + 4L * rt + rs) * D + col(jj)) = accs[jj];
          *reinterpret_cast<float4*>(PART + ((long)2 * PROW + 4L * rt + rs) * D + col(jj)) = accg[jj];
        }
      } else {
        // ---- GEMM epilogues: thread = row r of the tile, columns 16 hh .. + 15 of every 32-column chunk
        const int tb = (int)(g & 1);
        mbar_wait(&B.tfull[tb], (g >> 1) & 1);
        tc_fence_after();
        if (tid == 128) EV(qi, 8);
        const long R = (long)rt * QM + r;
        const uint32_t tacc = trow + (uint32_t)tb * 256 + (uint32_t)(16 * hh);
        if (ty == T_G1) {
          for (int q = 0; q < NG1 / 32; ++q) {
            const uint32_t sa = ewait(), sb = ewait();
            uint32_t dh[16];
            tmem_ld16w(tacc + 32 * q, dh);
            float a[16], b[16], da[16], db[16];
            ld16_bf(sa, r, hh, a);
            ld16_bf(sb, r, hh, b);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
              const float d = __uint_as_float(dh[k]), s = sg(a[k]);
              da[k] = d * b[k] * s * (1.f + a[k] * (1.f - s));
              db[k] = d * a[k] * s;
            }
            sts128(sa + sw64(r, (uint32_t)(2 * hh)), pk8(da)); sts128(sa + sw64(r, (uint32_t)(2 * hh + 1)), pk8(da + 8));
            sts128(sb + sw64(r, (uint32_t)(2 * hh)), pk8(db)); sts128(sb + sw64(r, (uint32_t)(2 * hh + 1)), pk8(db + 8));
            store_chunk(&ms_dab, sa, NG1 * n + 32 * q, &ms_dab, sb, 1536 + NG1 * n + 32 * q, rt * QM);
          }
        } else if (ty == T_G2) {
#pragma unroll 1
          for (int q = 0; q < NG2 / 32; ++q) {
            uint32_t v[16];
            tmem_ld16w(tacc + 32 * q, v);
            float f[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) f[k] = __uint_as_float(v[k]);
            __nv_bfloat16* p = DG + R * 3072 + 3 * D + NG2 * n + 32 * q + 16 * hh;
            if (valid) { stg128(p, pk8(f)); stg128(p + 8, pk8(f + 8)); }
          }
        } else {                                                // T_G4
          for (int q = 0; q < NG4 / 32; ++q) {
            const uint32_t so = ewait(), sgl = ewait();
            uint32_t v[16];
            tmem_ld16w(tacc + 32 * q, v);
            float og[16], gl[16], dov[16], dgv[16];
            ld16_bf(so, r, hh, og);
            ld16_bf(sgl, r, hh, gl);
            float hsum = 0.f;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
              const float dg = __uint_as_float(v[k]), s = sg(gl[k]);
              dov[k] = dg * s;
              dgv[k] = dg * og[k] * (1.f - s);
              hsum += dg * og[k];
            }
            hs[r * HSP + 2 * q + hh] = hsum;
            sts128(so + sw64(r, (uint32_t)(2 * hh)), pk8(dov)); sts128(so + sw64(r, (uint32_t)(2 * hh + 1)), pk8(dov + 8));
            sts128(sgl + sw64(r, (uint32_t)(2 * hh)), pk8(dgv)); sts128(sgl + sw64(r, (uint32_t)(2 * hh + 1)), pk8(dgv + 8));
            store_chunk(&ms_do, so, NG4 * n + 32 * q, &ms_dvg, sgl, DATT + NG4 * n + 32 * q, rt * QM);
          }
          named_bar_sync(1, 256);                               // every 16-column group sum of the tile in hs
          constexpr int NH4 = NG4 / DHD, GPH = DHD / 16;
          const long a = R / L, i = R % L;
          for (int h = hh; h < NH4; h += 2) {
            float t = 0.f;
#pragma unroll
            for (int k = 0; k < GPH; ++k) t += hs[r * HSP + h * GPH + k];
            if (valid) DD[(a * NHEAD + (long)n * NH4 + h) * L + i] = t;
          }
        }
        tc_fence_before();                                      // this item's TMEM reads are complete
        ++g;
      }
      // ---- item done: publish the stores, free TMEM / the ring slot, count the tile
      if (ty == T_G1 || ty == T_G4) store_done();
      fence_proxy_async_global();
      named_bar_sync(1, 256);
      if (tid == 128) {
        if (ty != T_R1 && ty != T_R2) mbar_arrive_remote(&B.tempty[(g - 1) & 1], 0);   // the leader's buffer, release.cluster
        if (ty != T_G4 && valid) { fence_gpu(); red_add(CNT + rt, 1u); }
        mbar_arrive(&B.iempty[is]);
        EV(qi, 9);
#ifdef TRACE
        EVV(qi, 14, esum);
#endif
      }
    }
  }
  // ------------------------------------------------------------------------------------------------ teardown; the last CTA resets
  tc_fence_before();
  __syncthreads();
  cluster_sync();                                            // no CTA leaves while its peer may still arrive on its barriers
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
  if (tid == 0) {
    fence_gpu();
    B.last = atomicAdd(done, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (B.last) {
    fence_gpu();
    for (int i = tid; i < T + 1; i += blockDim.x) CNT[i] = 0u;
  }
}
