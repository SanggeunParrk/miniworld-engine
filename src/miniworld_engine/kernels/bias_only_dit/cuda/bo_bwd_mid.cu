// bo_bwd_mid.cu -- the bias-only token DiT training backward from the attention core's input gradients to d single / d cond on CTA
// PAIRS, sm_100a (integrations/bias_only_dit_train.py, MINIWORLD_BIAS_ONLY_DIT_BWD_MID=1): what the dxa GEMM, adaln_a_bwd, the dchat
// and dcg GEMMs and cond_bwd did as five launches, in three launches of one kernel (PHASE 0, 1, 2). Items per PAIR TILE p (the
// 128-row tiles 2p, 2p + 1; the last pair's second tile absent when T is odd), rank r of the cluster working on tile rt = 2p + r:
//
//   PHASE 0  G5(p, n)  GEMM      dxa[:, 256 n ..] = dvg Wvg[:, 256 n ..] (K 2 DA) -> DG[:, D + 256 n ..] (bf16)            (n < 3)
//            G6(p, n)  GEMM      dcg[:, 192 n ..] = dGg Wg[:, 192 n ..] (K 1536) -> DCG[:, 192 n ..] (bf16 scratch)       (n < 2)
//   PHASE 1  R3(p)     row pass  per CTA, its tile: the attention's AdaLN backward (adaln_a_bwd, rows on chip): ds1 = dxa xh s1 (1 - s1)
//                                -> DG[:, :D], dx = dx1 + rstd (dxh - mean dxh - xh mean(dxh xh)), dxh = dxa s1 -> DX; bs1 partials
//   PHASE 2  G7(p)     GEMM      dchat = dG Wn (K 3072, N 384: two N 192 pair products per K step into one 384-column accumulator per
//                                CTA), the cond LayerNorm backward in each CTA's epilogue: dc = rstd_c (dchat - mean dchat - ch mean(dchat
//                                ch)) + dcg -> DC (dchat stays fp32 on chip)
// GEMM items are M 256 products on the pair (tcgen05 cta_group::2, as cuBLAS's pair kernels): each CTA loads its own tile's A rows and
// HALF of each B tile by N (G5: rows n0 + 128 r .. of 256; G6: n0 + 96 r .. of 192; G7: 96 r .. and 192 + 96 r .. of 384), the
// leader issues the products, each CTA's TMEM gets its 128 rows, and each CTA's epilogue is the single-CTA one on its rows. A launch
// only reads what earlier launches wrote (stream order + programmatic dependent launch): no item waits on another, and pair c of the
// grid's P pairs takes items c, c + P, ... (PHASE 0: [G5 x 3, G6 x 2] per pair tile, in order). B operands are the weights transposed
// once per pack (K-major): WvT / WgtT [768, DA] (G5's k-blocks below / above DA / 64), WsT0 / WsT1 [384, 768] (G6's k-blocks 0-11 /
// 12-23), WnT0..3 [384, 768] (the cond projections with the cond-LN weights folded: G7's k-blocks by twelves).
// -DSF32: x and dx fp32 (else bf16); -DCF32: dc fp32 (else bf16). bf16 operands, fp32 accumulation and row math.
//
// PDL: launched with programmatic dependent launch; every thread griddepcontrol.wait after the setup (barriers, TMEM, cluster
// barrier), before any global access, and launch_dependents right before it, so only the next launch's setup overlaps this one.
//
// SYNCHRONIZATION (per CTA unless said otherwise; phase of a barrier = its use count & 1):
//   item ring  B.item[NIR]   filler: producer (warp 0 lane 0) writes the code, mbarrier.arrive ifull[s] (count 1)
//                            consumers: MMA warp, E-loader (warp 2 lane 0), epilogue (warps 4-11) wait ifull[s]; MMA and E-loader
//                            arrive iempty[s] once they have read the code, the epilogue (thread 128) after the item (count 3); the
//                            producer waits iempty[s] before it writes the slot again
//   operands   NST stages    PHASE 0: 7 x 32 KB, PHASE 2: 5 x 40 KB (bwdf9: the MMA waited on full stages with 4).
//                            filler: producer of EACH CTA: j >= NST: wait its own oempty[s] parity ((j / NST) - 1) & 1; the leader
//              (A | B half)  arrive.expect_tx on ITS ofull[s] for both CTAs' bytes; each CTA's TMA loads (cta_group::2) complete on the
//                            LEADER's ofull[s]. consumer: the leader's MMA warp waits ofull[s] parity (j / NST) & 1,
//                            tcgen05.fence::after_thread_sync, 4 MMAs of K 16 (G7: 8, two N 192 halves; cta_group::2),
//                            tcgen05.commit cta_group::2 multicast {0, 1} -> oempty[s] of both CTAs (count 1 each)
//   TMEM       PHASE 0: 2 x 256 columns, the items alternate (GEMM count g, buffer g & 1, use g >> 1); PHASE 2: one buffer of
//              384 columns (buffer 0, use g); cta_group::2 alloc. The leader's MMA waits ITS tempty[b] (acquire.cluster, count 2:
//                            one arrival per CTA) parity (use - 1) & 1 for use >= 1, tcgen05.fence::after_thread_sync, products,
//                            tcgen05.commit multicast {0, 1} -> tfull[b] of both CTAs after the item's last stage; each epilogue
//                            waits its own tfull[b] parity use & 1, tcgen05.fence::after_thread_sync, tcgen05.ld ...; after its
//                            last read every epilogue thread tcgen05.fence::before_thread_sync, named barrier 1 (256), thread 128
//                            mbarrier.arrive.release.cluster on the LEADER's tempty[b]
//   E ring     NE x 8 KB     PHASE 1 only, each CTA, over the whole operand region: filler E-loader, arrive.expect_tx efull[e] + TMA
//                            (un-swizzled [4 rows][768] bf16 boxes, [4 rows][384] fp32 half-row boxes), in the order the epilogue
//                            consumes them; consumer every epilogue thread waits efull[e]; after reading a slot each warp
//                            fence.proxy.async, __syncwarp, lane 0 arrives eempty[e] (count 8: every epilogue warp, also for a half-row
//                            box it did not read); the E-loader waits eempty[e] before reusing the slot
//   row pass   named barriers 2 + slot (64 threads: the two warps of a row) exchange the row sums (double-buffered by parity)
//   G7 rows    the two epilogue threads of a row (hh 0 / 1) exchange their (sum dchat, sum dchat ch) through scratch xs[hh][r]
//              around named barrier 1 (256); the item-done barrier 1 orders the next item's writes after both reads
//   G7 staging the cond-LN epilogue's inputs (this CTA's c and dcg rows, 6 + 6 SW128 [128][64] boxes, 192 KB) TMA-loaded into the
//              operand region once the item's products are done: the E-loader waits tfull[0] (its own count of G7 items: parity w & 1),
//              arrive.expect_tx cfull (192 KB) + 12 loads; the epilogue waits cfull (its count of tile-holding G7 items) and reads them
//              from shared memory; after the item-done barrier thread 128 arrives stfree (count 1, every G7 item); the producer waits
//              stfree for G7 item k - 1 (parity (k - 1) & 1) before its first stage load of item k, so no stage write meets the
//              staging (the absent tile of an odd T: neither the loads nor the cfull wait)
//   setup / exit: barrier init by thread 0, fence.mbarrier_init, cta_group::2 TMEM alloc, __syncthreads + cluster barrier before any
//              peer arrival; exit: tcgen05.fence::before_thread_sync, __syncthreads, cluster barrier, cta_group::2 dealloc.
//   odd T: the second CTA of the last pair has no tile: its A rows come back zero-filled, its stores and its row / cond work are
//          skipped (uniformly over its epilogue threads, barriers included).
//
// Warps: 0 producer (lane 0), 1 MMA (the leader's: whole warp waits, elect_one() issues), 2 TMEM allocator + E-loader (lane 0), 3 idle,
// 4-11 epilogue: a GEMM item's thread = TMEM lane (row 32 (warp & 3) + lane) and column half hh = (warp - 4) >> 2 of every 32-column
// chunk; a row item's 4-row group: row slot (warp - 4) >> 1, two warps per row, a lane owning 4-column chunks hw * 3 + j (j < 3) of
// 32 (warp half hw = columns [384 hw, 384 hw + 384): the fp32 half-row box hw).
// -DTRACE: per item (indexed by QBASE + its index q < TRQ; both CTAs of a pair write the slot) %globaltimer events into g_ev[q][16] as
// bo_bwd_tail.cu's (bo_bwd_trace.py): 0 claimed, 1 producer: first load, 2 producer: last load issued, 3 MMA: first stage seen, 4 MMA:
// last commit issued, 5 E-loader: item seen, 6 E-loader: first box issued, 7 epilogue: item seen, 8 epilogue: first input, 9 epilogue:
// item done; 10 CTA, 11 code; per GEMM item the ns spent waiting: 12 the leader's MMA warp on full operand stages, 13 the producer
// on empty stages.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DATT
#define DATT 768
#endif
static_assert(DATT == 768 || DATT == 1024, "768 or 1024 attention channels");
#ifdef SF32
typedef float XT;                                                          // x, dx
#else
typedef __nv_bfloat16 XT;
#endif
#ifdef CF32
typedef float CT;                                                          // dc
#else
typedef __nv_bfloat16 CT;
#endif

constexpr int D = 768, DCN = 384, QM = 128, KB = 64;
constexpr int NI5 = 3, NK5 = 2 * DATT / KB, NKV = DATT / KB;              // G5: 3 x N 256, K 2 DA (value | gate halves of B)
constexpr int NI6 = 2, NK6 = 1536 / KB;                                    // G6: 2 x N 192, K 1536 (two to_scale^T of 12 k-blocks)
constexpr int NK7 = 3072 / KB;                                             // G7: N 384, K 3072 (four projections of 12 k-blocks)
constexpr int PER0 = NI5 + NI6;                                            // PHASE 0 items per tile
enum { T_G5 = 0, T_G6 = 1, T_R3 = 2, T_G7 = 3, T_STOP = 7 };

constexpr int ASZ = QM * 128;                                              // A [128][64] bf16
constexpr int STG0 = ASZ + 128 * 128, NST0 = 7;                            // PHASE 0 stage: A | B half [<= 128][64] (G5 128, G6 96 rows)
constexpr int STG2 = ASZ + 192 * 128, NST2 = 5;                            // PHASE 2 stage: A | two B boxes [96][64] (G7)
constexpr int NSTM = NST0 > NST2 ? NST0 : NST2;
constexpr int B7H = 96 * 128;                                              // G7: this CTA's second B box (rows 192 + 96 r ..) at + 12 KB
constexpr int ESZ = 8192, NIR = 2;
constexpr int RG = 4, NGR = QM / RG, RGB = RG * D * 2;                     // row passes: 4-row groups, [4][768] bf16 boxes (6 KB)
constexpr int REGION = 229376;                                             // the stages (PHASES 0, 2) / the row ring (PHASE 1) / G7 staging
constexpr int CSZ = QM * 128, O_CG = 6 * CSZ, CBYTES = 12 * CSZ;           // G7 staging: c boxes at 0, dcg boxes at 96 KB
constexpr int O_OP = 0, O_XS = REGION, O_BAR = O_XS + 2048, SMEM_BYTES = O_BAR + 1024;
static_assert(SMEM_BYTES <= 232448 && NST0 * STG0 <= REGION && NST2 * STG2 <= REGION && CBYTES <= REGION, "shared memory");
static_assert(STG0 % 1024 == 0 && STG2 % 1024 == 0 && B7H % 1024 == 0 && ESZ % 1024 == 0 && RGB <= ESZ, "alignment");
constexpr int NE = O_XS / ESZ;                                             // PHASE 1's E ring: the operand stages as 8 KB slots
#ifdef SF32
constexpr int SLOTS_R3 = 6;                                                // dxa, s1, x (2 halves), dx1 (2 halves)
#else
constexpr int SLOTS_R3 = 5;
#endif
static_assert(NE >= 3 * SLOTS_R3, "row ring: three groups in flight at least");
constexpr uint32_t I_256 = idesc_bf16(2 * QM, 256), I_192 = idesc_bf16(2 * QM, 192);   // M 256: the pair's two tiles

struct Bars {
  uint64_t ifull[NIR], iempty[NIR], ofull[NSTM], oempty[NSTM], tfull[2], tempty[2], efull[NE], eempty[NE], cfull, stfree;
  int item[NIR], qid[NIR];
  uint32_t tmem;
  int last;
};
static_assert(sizeof(Bars) <= 1024 - 64, "barriers");

#ifdef TRACE
constexpr int TRQ = 32768;                                                  // ten graph copies at L768
__device__ unsigned long long g_ev[TRQ * 16];
__device__ unsigned long long g_cta0[1024];
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define EV(q, k) do { if ((unsigned)(q) < (unsigned)TRQ) g_ev[(q) * 16 + (k)] = gtime(); } while (0)
#define EVV(q, k, v) do { if ((unsigned)(q) < (unsigned)TRQ) g_ev[(q) * 16 + (k)] = (unsigned long long)(v); } while (0)
#else
#define EV(q, k) do { } while (0)
#define EVV(q, k, v) do { } while (0)
#endif

DEVI void fence_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI uint2 lds64u(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v; }
DEVI void stg64(void* p, uint2 v) { asm volatile("st.global.v2.b32 [%0], {%1,%2};" :: "l"(p), "r"(v.x), "r"(v.y) : "memory"); }
DEVI float4 f4bf(uint2 u) { return make_float4(bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y)); }
DEVI uint2 bf4(float4 v) { return make_uint2(pack_bf16(v.x, v.y), pack_bf16(v.z, v.w)); }
DEVI float4 f4u(uint4 u) { return make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
DEVI float sg(float v) { return __fdividef(1.f, 1.f + __expf(-v)); }
DEVI float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }      // the value a bf16 store keeps
DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEVI void tmem_ld16w(uint32_t taddr, uint32_t (&r)[16]) { tmem_ld16(taddr, r); tmem_wait_ld(); }
DEVI uint4 pk8(const float* v) {
  return make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7]));
}
// four values of the x row in a slot: bf16 [4][768] box, or (SF32) the fp32 [4][384] half-row box of this warp half
DEVI float4 ld_x4(uint32_t s_x0, uint32_t s_x1, int rs, int c, int hw) {
#ifdef SF32
  return f4u(lds128((hw ? s_x1 : s_x0) + (uint32_t)(rs * DCN + c - DCN * hw) * 4));
#else
  (void)s_x1; (void)hw;
  return f4bf(lds64u(s_x0 + (uint32_t)(rs * D + c) * 2));
#endif
}
DEVI void st_x4(XT* p, float4 v) {
#ifdef SF32
  *reinterpret_cast<float4*>(p) = v;
#else
  stg64(p, bf4(v));
#endif
}

extern "C" __global__ void __launch_bounds__(384, 1)
bo_bwd_mid_sm100(const __grid_constant__ CUtensorMap ma_dvg, const __grid_constant__ CUtensorMap ma_dgg,
                 const __grid_constant__ CUtensorMap ma_dg, const __grid_constant__ CUtensorMap mb_wv,
                 const __grid_constant__ CUtensorMap mb_wgt, const __grid_constant__ CUtensorMap mb_s0,
                 const __grid_constant__ CUtensorMap mb_s1, const __grid_constant__ CUtensorMap mb_n0,
                 const __grid_constant__ CUtensorMap mb_n1, const __grid_constant__ CUtensorMap mb_n2,
                 const __grid_constant__ CUtensorMap mb_n3, const __grid_constant__ CUtensorMap mr_dxa,
                 const __grid_constant__ CUtensorMap mr_s1, const __grid_constant__ CUtensorMap mr_x,
                 const __grid_constant__ CUtensorMap mr_dx1, const __grid_constant__ CUtensorMap mc_c,
                 const __grid_constant__ CUtensorMap mc_g,
                 const float2* __restrict__ XST, const float* __restrict__ BS1,
                 const float2* __restrict__ CST, const __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ DCG,
                 __nv_bfloat16* __restrict__ DG, XT* __restrict__ DX, CT* __restrict__ DC, float* __restrict__ PART3,
                 int T, int PHASE, int QBASE) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int crank = (int)cluster_rank(), pair = (int)blockIdx.x >> 1, npairs = (int)gridDim.x >> 1;
  const bool leader = crank == 0;
  const int TP = (T + 1) >> 1, nitems = PHASE == 0 ? PER0 * TP : TP;   // pair tiles
  const bool wide = PHASE == 2;                                // G7: one 384-column TMEM buffer
  const int nst = wide ? NST2 : NST0;                          // this launch's operand stages
  const uint32_t stg = wide ? (uint32_t)STG2 : (uint32_t)STG0;

  if (tid == 0) {
    for (int i = 0; i < NIR; ++i) { mbar_init(&B.ifull[i], 1); mbar_init(&B.iempty[i], 3); }
    for (int i = 0; i < NSTM; ++i) { mbar_init(&B.ofull[i], 1); mbar_init(&B.oempty[i], 1); }
    mbar_init(&B.cfull, 1); mbar_init(&B.stfree, 1);
    for (int i = 0; i < 2; ++i) { mbar_init(&B.tfull[i], 1); mbar_init(&B.tempty[i], 2); }   // tempty: one arrival per CTA
    for (int i = 0; i < NE; ++i) { mbar_init(&B.efull[i], 1); mbar_init(&B.eempty[i], 8); }
    fence_barrier_init();
    prefetch_map(&ma_dvg); prefetch_map(&ma_dgg); prefetch_map(&ma_dg); prefetch_map(&mb_wv); prefetch_map(&mb_wgt);
    prefetch_map(&mb_s0); prefetch_map(&mb_s1); prefetch_map(&mb_n0); prefetch_map(&mb_n1); prefetch_map(&mb_n2);
    prefetch_map(&mb_n3); prefetch_map(&mr_dxa); prefetch_map(&mr_s1); prefetch_map(&mr_x); prefetch_map(&mr_dx1);
    prefetch_map(&mc_c); prefetch_map(&mc_g);
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();                                            // the peer's barriers initialized before any arrival on them
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();
  pdl_wait();
#ifdef TRACE
  if (tid == 0 && blockIdx.x < 1024) g_cta0[blockIdx.x] = gtime();
#endif

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ producer: items, operands
    // both CTAs: the pair's items by index; GEMM items: this CTA's A tile and its half of B, completing on the leader's ofull
    if (lane == 0) {
      uint32_t j = 0, k7 = 0;
      for (uint32_t it = 0;; ++it) {
        const int is = (int)(it % NIR);
        if (it >= NIR) mbar_wait(&B.iempty[is], ((it / NIR) - 1) & 1);
        const unsigned q0 = (unsigned)(pair + (int)it * npairs);
        int code = T_STOP;
        if (q0 < (unsigned)nitems) {
          if (PHASE == 0) {
            const int tp = (int)q0 / PER0, k = (int)q0 % PER0;
            code = (k < NI5 ? T_G5 | (k << 3) : T_G6 | ((k - NI5) << 3)) | (tp << 8);
          } else {
            code = (PHASE == 1 ? T_R3 : T_G7) | ((int)q0 << 8);
          }
        }
        const unsigned q = q0 + (unsigned)QBASE;
        B.item[is] = code;
        B.qid[is] = (int)q;
        mbar_arrive(&B.ifull[is]);
        const int ty = code & 7;
        if (ty == T_STOP) break;
        EV(q, 0); EVV(q, 10, blockIdx.x); EVV(q, 11, code);
        if (ty == T_R3) continue;
        const int n = (code >> 3) & 15, rt = 2 * (code >> 8) + crank;
        if (ty == T_G7) {                                       // the previous G7 item's staging read: the region is stages again
          if (k7 >= 1) mbar_wait(&B.stfree, (k7 - 1) & 1);
          ++k7;
        }
        EV(q, 1);
        const CUtensorMap* ma = ty == T_G5 ? &ma_dvg : ty == T_G6 ? &ma_dgg : &ma_dg;
        const int nk = ty == T_G5 ? NK5 : ty == T_G6 ? NK6 : NK7;
        const uint32_t bytes = (uint32_t)(2 * (ASZ + (ty == T_G5 ? 128 : ty == T_G6 ? 96 : 192) * 128));   // both CTAs' loads
#ifdef TRACE
        unsigned long long wsum = 0;
#endif
        for (int kb = 0; kb < nk; ++kb, ++j) {
          const int s = (int)(j % (uint32_t)nst);
          const uint32_t u = j / (uint32_t)nst;
#ifdef TRACE
          if (u >= 1) { const unsigned long long tw = gtime(); mbar_wait(&B.oempty[s], (u - 1) & 1); wsum += gtime() - tw; }
#else
          if (u >= 1) mbar_wait(&B.oempty[s], (u - 1) & 1);
#endif
          if (leader) mbar_expect_tx(&B.ofull[s], bytes);
          const uint32_t dst = su + O_OP + (uint32_t)s * stg;
          tma_load_2d_2sm(dst, ma, &B.ofull[s], KB * kb, rt * QM);
          if (ty == T_G5) {
            tma_load_2d_2sm(dst + ASZ, kb < NKV ? &mb_wv : &mb_wgt, &B.ofull[s], KB * (kb < NKV ? kb : kb - NKV), 256 * n + 128 * crank);
          } else if (ty == T_G6) {
            tma_load_2d_2sm(dst + ASZ, kb < 12 ? &mb_s0 : &mb_s1, &B.ofull[s], KB * (kb % 12), 192 * n + 96 * crank);
          } else {
            const int pj = kb / 12;
            const CUtensorMap* mb = pj == 0 ? &mb_n0 : pj == 1 ? &mb_n1 : pj == 2 ? &mb_n2 : &mb_n3;
            tma_load_2d_2sm(dst + ASZ, mb, &B.ofull[s], KB * (kb % 12), 96 * crank);
            tma_load_2d_2sm(dst + ASZ + B7H, mb, &B.ofull[s], KB * (kb % 12), 192 + 96 * crank);
          }
        }
        EV(q, 2);
        EVV(q, 13, wsum);
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
      if (ty == T_R3) continue;
      if (leader) {
        const int tb = wide ? 0 : (int)(g & 1);
        const uint32_t use = wide ? g : g >> 1;
        if (use >= 1) mbar_wait_cl(&B.tempty[tb], (use - 1) & 1);      // both CTAs' epilogues done with the buffer
        tc_fence_after();
        const int nk = ty == T_G5 ? NK5 : ty == T_G6 ? NK6 : NK7;
        const uint32_t d = tmem + (uint32_t)tb * 256;
#ifdef TRACE
        unsigned long long wsum = 0;
#endif
        for (int kb = 0; kb < nk; ++kb, ++j) {
          const int s = (int)(j % (uint32_t)nst);
#ifdef TRACE
          { const unsigned long long tw = gtime(); mbar_wait(&B.ofull[s], (j / (uint32_t)nst) & 1); wsum += gtime() - tw; }
#else
          mbar_wait(&B.ofull[s], (j / (uint32_t)nst) & 1);
#endif
          tc_fence_after();
#ifdef TRACE
          if (lane == 0 && kb == 0) EV(qi, 3);
#endif
          if (elect_one()) {
            const uint32_t sa = su + O_OP + (uint32_t)s * stg;
            const uint64_t da = desc_k128(sa), db = desc_k128(sa + ASZ);
            if (ty == T_G7) {
              const uint64_t db2 = desc_k128(sa + ASZ + B7H);
#pragma unroll
              for (int ks = 0; ks < 4; ++ks) {
                const uint32_t acc = (kb | ks) ? 1u : 0u;
                umma_ss2(d, da + (uint64_t)(2 * ks), db + (uint64_t)(2 * ks), I_192, acc);         // columns 0 .. 191
                umma_ss2(d + 192, da + (uint64_t)(2 * ks), db2 + (uint64_t)(2 * ks), I_192, acc);  // columns 192 .. 383
              }
            } else {
              const uint32_t idesc = ty == T_G5 ? I_256 : I_192;
#pragma unroll
              for (int ks = 0; ks < 4; ++ks) umma_ss2(d, da + (uint64_t)(2 * ks), db + (uint64_t)(2 * ks), idesc, (kb | ks) ? 1u : 0u);
            }
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
    // ------------------------------------------------------------------------------------------------ E-loader: R3's row boxes
    if (lane == 0) {
      uint32_t e = 0, w7 = 0;
      auto slot = [&](uint32_t bytes) -> uint32_t {
        const int es = (int)(e % (uint32_t)NE);
        if (e >= (uint32_t)NE) mbar_wait(&B.eempty[es], ((e / NE) - 1) & 1);
        mbar_expect_tx(&B.efull[es], bytes);
        ++e;
        return su + O_OP + (uint32_t)es * ESZ;
      };
      auto bar_of = [&](uint32_t dst) { return &B.efull[(dst - su - O_OP) / ESZ]; };
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
        if (ty == T_G7) {                                        // the item's products done (its stages read): stage c, dcg rows
          mbar_wait(&B.tfull[0], w7 & 1);
          ++w7;
          const int rt = 2 * (code >> 8) + crank;
          if (rt < T) {
            mbar_expect_tx(&B.cfull, (uint32_t)CBYTES);
#pragma unroll
            for (int b = 0; b < 6; ++b) {
              tma_load_2d(su + O_OP + (uint32_t)(b * CSZ), &mc_c, &B.cfull, 64 * b, rt * QM);
              tma_load_2d(su + O_OP + (uint32_t)(O_CG + b * CSZ), &mc_g, &B.cfull, 64 * b, rt * QM);
            }
          }
          continue;
        }
        if (ty != T_R3) continue;
        const int rt = 2 * (code >> 8) + crank, row0 = rt * QM;
        if (rt >= T) continue;                                   // the absent tile of an odd T: no row work
        EV(qi, 5);
        EV(qi, 6);
        for (int gq = 0; gq < NGR; ++gq) {
          const int rr = row0 + RG * gq;
          uint32_t dst = slot(RGB); tma_load_3d(dst, &mr_dxa, bar_of(dst), 0, 0, rr);
          dst = slot(RGB); tma_load_3d(dst, &mr_s1, bar_of(dst), 0, 0, rr);
#ifdef SF32
          dst = slot(RGB); tma_load_3d(dst, &mr_x, bar_of(dst), 0, 0, rr);
          dst = slot(RGB); tma_load_3d(dst, &mr_x, bar_of(dst), 0, 3, rr);
#else
          dst = slot(RGB); tma_load_3d(dst, &mr_x, bar_of(dst), 0, 0, rr);
#endif
          dst = slot(RGB); tma_load_3d(dst, &mr_dx1, bar_of(dst), 0, 0, rr);
          dst = slot(RGB); tma_load_3d(dst, &mr_dx1, bar_of(dst), 0, 3, rr);
        }
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ epilogue / row pass
    const int ew = warp - 4, q4 = warp & 3, hh = ew >> 2;
    const uint32_t r = (uint32_t)(32 * q4 + lane), trow = tmem + ((uint32_t)(32 * q4) << 16);
    const int rs = ew >> 1, hw = ew & 1;                       // row pass: row slot of the group, warp half of the row
    float2* const xr = reinterpret_cast<float2*>(sm + O_XS);   // row pass exchange [2][4][2]
    float2* const xs = reinterpret_cast<float2*>(sm + O_XS);   // G7 exchange [2][128] (the phases never share a launch)
    int par = 0;
    uint32_t e = 0, g = 0, c7 = 0;                            // c7: tile-holding G7 items (cfull uses)
    auto ewait = [&]() -> uint32_t {                          // the next E slot, full
      const int es = (int)(e % (uint32_t)NE);
      mbar_wait(&B.efull[es], (e / NE) & 1);
      ++e;
      return su + O_OP + (uint32_t)es * ESZ;
    };
    auto erelease = [&](uint32_t addr) {                     // this warp is done with the slot
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.eempty[(addr - su - O_OP) / ESZ]);
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
      const bool valid = rt < T;                               // false: the absent second tile of an odd T (uniform over the CTA)
      if (tid == 128) EV(qi, 7);
      if (ty == T_R3 && !valid) {
        // nothing: the E-loader issued nothing for it
      } else if (ty == T_R3) {
        // ---- R3: the attention's AdaLN backward (adaln_a_bwd), rows on chip
        float4 b1[3], acc[3];
#pragma unroll
        for (int jj = 0; jj < 3; ++jj) { b1[jj] = __ldg(reinterpret_cast<const float4*>(BS1 + col(jj))); acc[jj] = make_float4(0.f, 0.f, 0.f, 0.f); }
        for (int gq = 0; gq < NGR; ++gq) {
          const long R = (long)rt * QM + RG * gq + rs;
          const float2 st = XST[R];
          const uint32_t s_dxa = ewait(), s_s1 = ewait();
#ifdef SF32
          const uint32_t s_x0 = ewait(), s_x1 = ewait();
#else
          const uint32_t s_x0 = ewait(), s_x1 = s_x0;
#endif
          const uint32_t s_d0 = ewait(), s_d1 = ewait();
          if (tid == 128 && gq == 0) EV(qi, 8);
          // pass 1: ds1, and xh / dxh of this thread's 12 columns kept for pass 2
          float xhr[12], dxr[12];
          float m1 = 0.f, m2 = 0.f;
#pragma unroll
          for (int jj = 0; jj < 3; ++jj) {
            const int c = col(jj);
            const uint32_t o = (uint32_t)(rs * D + c) * 2;
            const float4 dxa = f4bf(lds64u(s_dxa + o)), s1l = f4bf(lds64u(s_s1 + o)), x = ld_x4(s_x0, s_x1, rs, c, hw);
            const float av[4] = {dxa.x, dxa.y, dxa.z, dxa.w}, xv[4] = {x.x, x.y, x.z, x.w};
            const float sv[4] = {s1l.x + b1[jj].x, s1l.y + b1[jj].y, s1l.z + b1[jj].z, s1l.w + b1[jj].w};
            float ds[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const float s = sg(sv[k]), xh = (xv[k] - st.x) * st.y;
              ds[k] = bfr(av[k] * xh * s * (1.f - s));
              const float dxh = av[k] * s;
              m1 += dxh;
              m2 += dxh * xh;
              xhr[4 * jj + k] = xh; dxr[4 * jj + k] = dxh;
            }
            stg64(DG + R * 3072 + c, bf4(make_float4(ds[0], ds[1], ds[2], ds[3])));
            acc[jj].x += ds[0]; acc[jj].y += ds[1]; acc[jj].z += ds[2]; acc[jj].w += ds[3];
          }
          erelease(s_dxa); erelease(s_s1); erelease(s_x0);
#ifdef SF32
          erelease(s_x1);
#endif
          const float2 m = rowsum2(m1, m2);
          m1 = m.x * (1.f / D); m2 = m.y * (1.f / D);
#pragma unroll
          for (int jj = 0; jj < 3; ++jj) {
            const int c = col(jj);
            const float4 d1 = f4u(lds128((hw ? s_d1 : s_d0) + (uint32_t)(rs * DCN + c - DCN * hw) * 4));
            const float dv[4] = {d1.x, d1.y, d1.z, d1.w};
            float dx[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) dx[k] = dv[k] + st.y * (dxr[4 * jj + k] - m1 - xhr[4 * jj + k] * m2);
            st_x4(DX + R * D + c, make_float4(dx[0], dx[1], dx[2], dx[3]));
          }
          erelease(s_d0); erelease(s_d1);
        }
#pragma unroll
        for (int jj = 0; jj < 3; ++jj) *reinterpret_cast<float4*>(PART3 + (4L * rt + rs) * D + col(jj)) = acc[jj];
      } else {
        // ---- GEMM epilogues: thread = row r of the tile, columns 16 hh .. + 15 of every 32-column chunk
        const long R = (long)rt * QM + r;
        const int tb = wide ? 0 : (int)(g & 1);
        const uint32_t use = wide ? g : g >> 1;
        mbar_wait(&B.tfull[tb], use & 1);
        tc_fence_after();
        if (tid == 128) EV(qi, 8);
        const uint32_t tacc = trow + (uint32_t)tb * 256 + (uint32_t)(16 * hh);
        if (ty == T_G5 || ty == T_G6) {
          const int nq = ty == T_G5 ? 256 / 32 : 192 / 32;
          __nv_bfloat16* const base = ty == T_G5 ? DG + R * 3072 + D + 256 * n : DCG + R * DCN + 192 * n;
#pragma unroll 1
          for (int q = 0; q < nq; ++q) {
            uint32_t v[16];
            tmem_ld16w(tacc + 32 * q, v);
            float f[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) f[k] = __uint_as_float(v[k]);
            __nv_bfloat16* p = base + 32 * q + 16 * hh;
            if (valid) { stg128(p, pk8(f)); stg128(p + 8, pk8(f + 8)); }
          }
        } else if (valid) {                                     // T_G7: dchat, then the cond LayerNorm backward
          mbar_wait(&B.cfull, c7 & 1);                          // this tile's c and dcg rows staged (SW128 [128][64] boxes)
          ++c7;
          const float2 cs = CST[R];
          const uint32_t cb = su + O_OP, gb = su + O_OP + (uint32_t)O_CG, rsw = r & 7;
          // 16-byte unit k (0 / 1) of columns c0 .. c0 + 15 of row r in a staged tensor
          auto unit = [&](uint32_t base, int c0, int k) -> uint4 {
            const uint32_t b = (uint32_t)c0 >> 6, u = (((uint32_t)c0 & 63u) >> 3) + (uint32_t)k;
            return lds128(base + b * (uint32_t)CSZ + r * 128u + ((u ^ rsw) << 4));
          };
          float m1 = 0.f, m2 = 0.f;
          // pass 1: sum dchat, sum dchat ch over this thread's 192 columns, two TMEM chunks per tcgen05.wait::ld
#pragma unroll 1
          for (int qp = 0; qp < 12; qp += 2) {
            uint32_t vv[2][16];
            tmem_ld16(tacc + 32 * qp, vv[0]);
            tmem_ld16(tacc + 32 * (qp + 1), vv[1]);
            tmem_wait_ld();
#pragma unroll
            for (int e2 = 0; e2 < 2; ++e2) {
              const int c0 = 32 * (qp + e2) + 16 * hh;
              const uint4 x0 = unit(cb, c0, 0), x1 = unit(cb, c0, 1);
              const uint32_t w[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
              for (int k = 0; k < 8; ++k) {
                const float d0 = __uint_as_float(vv[e2][2 * k]), d1 = __uint_as_float(vv[e2][2 * k + 1]);
                const float h0 = (bf16lo(w[k]) - cs.x) * cs.y, h1 = (bf16hi(w[k]) - cs.x) * cs.y;
                m1 += d0 + d1;
                m2 += d0 * h0 + d1 * h1;
              }
            }
          }
          xs[hh * QM + r] = make_float2(m1, m2);
          named_bar_sync(1, 256);
          const float2 o = xs[(hh ^ 1) * QM + r];
          m1 = (m1 + o.x) * (1.f / DCN); m2 = (m2 + o.y) * (1.f / DCN);
          // pass 2: dc = rstd (dchat - m1 - ch m2) + dcg
#pragma unroll 1
          for (int qp = 0; qp < 12; qp += 2) {
            uint32_t vv[2][16];
            tmem_ld16(tacc + 32 * qp, vv[0]);
            tmem_ld16(tacc + 32 * (qp + 1), vv[1]);
            tmem_wait_ld();
#pragma unroll
            for (int e2 = 0; e2 < 2; ++e2) {
              const int c0 = 32 * (qp + e2) + 16 * hh;
              const uint4 x0 = unit(cb, c0, 0), x1 = unit(cb, c0, 1), y0 = unit(gb, c0, 0), y1 = unit(gb, c0, 1);
              const uint32_t w[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
              const uint32_t u8[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
              float out[16];
#pragma unroll
              for (int k = 0; k < 8; ++k) {
                const float h0 = (bf16lo(w[k]) - cs.x) * cs.y, h1 = (bf16hi(w[k]) - cs.x) * cs.y;
                out[2 * k] = cs.y * (__uint_as_float(vv[e2][2 * k]) - m1 - h0 * m2) + bf16lo(u8[k]);
                out[2 * k + 1] = cs.y * (__uint_as_float(vv[e2][2 * k + 1]) - m1 - h1 * m2) + bf16hi(u8[k]);
              }
              CT* p = DC + R * DCN + c0;
#ifdef CF32
#pragma unroll
              for (int k = 0; k < 4; ++k) reinterpret_cast<float4*>(p)[k] = make_float4(out[4 * k], out[4 * k + 1], out[4 * k + 2], out[4 * k + 3]);
#else
              stg128(p, pk8(out)); stg128(p + 8, pk8(out + 8));
#endif
            }
          }
          fence_proxy_async();                                  // the staging reads before the next stage loads (async proxy)
        }
        tc_fence_before();                                      // this item's TMEM reads are complete
        ++g;
      }
      // ---- item done: publish the stores (generic proxy, read by the next launches' TMA), free TMEM and the item slot
      fence_proxy_async_global();
      named_bar_sync(1, 256);
      if (tid == 128) {
        if (ty != T_R3) mbar_arrive_remote(&B.tempty[wide ? 0 : (int)((g - 1) & 1)], 0);   // the leader's buffer, release.cluster
        if (ty == T_G7) mbar_arrive(&B.stfree);               // the staging read (every G7 item): the producer may refill
        mbar_arrive(&B.iempty[is]);
        EV(qi, 9);
      }
    }
  }
  // ------------------------------------------------------------------------------------------------ teardown
  tc_fence_before();
  __syncthreads();
  cluster_sync();                                            // no CTA leaves while its peer may still arrive on its barriers
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}
