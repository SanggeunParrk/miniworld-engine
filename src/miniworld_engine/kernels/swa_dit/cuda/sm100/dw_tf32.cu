// dw_tf32.cu — the five weight-gradient GEMMs of the fp32 SWA atom block's backward in one launch on sm_100a (TF32 tcgen05.mma, fp32 accumulation):
//   dWqkvg [4C, C] = dP^T x           (dP [R, 4C] = [dpq | dpk | dpv | dg],  x [R, C])
//   dWo    [C, C]  = datt^T gated
//   dWu    [2 NHID, C] = dab^T y      (dab [R, 2 NHID] = [da | db])
//   dWd    [C, NHID]   = dffn^T hh
// with R = N S rows (the reduction), every operand a row-major [R, cols] fp32 tensor (TF32-valued, or truncated by the MMA like the kernels' own
// operands): D[m][n] = sum_r A[r][m] B[r][n], both operands MN-major in the 128-B swizzle with 32-B atoms (TMA "128a32", UMMA layout type 1).
// The reduction runs split-K: a CTA takes a contiguous slice of rows of ONE problem (CTAs per problem proportional to its bytes), streams it by TMA
// through a ring of 16-row stages (A atoms | B atoms: one [16 rows][32 fp32] box each), accumulates the whole [m_total, n] tile in TMEM (m_total / 128
// accumulators of N = n) and writes its partial sum to a workspace; dw_reduce_tf32 adds the slices of every element in slice order (deterministic).
// These GEMMs read ~1.5 GB: the kernel is bound by memory, not by the tensor pipe.
// Warps: 0 TMA producer, 1 MMA issuer, 2 TMEM allocator, 4-7 epilogue (thread = row m = TMEM lane).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

namespace tf {
__host__ __device__ constexpr uint32_t idesc(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
// MN-major operand in the 128-B swizzle with 32-B atoms: SBO 512, LBO = distance between 32-element MN atoms
DEVI uint64_t desc_mn32b(uint32_t saddr, uint32_t lbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)(512 >> 4) << 32) |
         ((uint64_t)1 << 46) | ((uint64_t)1 << 61);
}
DEVI void mma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
}  // namespace tf

#ifndef TRACE
#define TRACE 0
#endif
#if TRACE
#define TR(e, cond) do { if (blockIdx.x == 0 && (cond)) TRb[e] = clock64(); } while (0)
#else
#define TR(e, cond) do { } while (0)
#endif
#ifndef DWM
#define DWM 0
#endif
#ifndef DW_CL
#define DW_CL 0
#endif
#ifndef DW_BKR
#define DW_BKR 16                                                          // reduction rows a stage (the cluster build: 16 / 32 / 64)
#endif
#ifndef NS
#define NS (DW_CL ? 204800 / (12 * DW_BKR * 128) : 5)                      // ring stages (cluster build: <= 4 + 8 boxes a stage)
#endif
#ifndef DW_BOX3
#define DW_BOX3 0                                                          // round 3 part 14: 3-D operand maps, two boxes a stage (cubin dw_tf32_b3)
#endif
#ifndef DW_DIRECT
#define DW_DIRECT 0                                                        // round 3 part 14: coalesced direct stores of the partial sums (measured level: off)
#endif
// round 3 part 15: DW_CL > 1 -- one cluster of DW_CL CTAs per 128-row output tile (dWqkvg m-blocks 0-3, dWo, dWu m-blocks 0-3, dWd [128 x
// 256]: 10 tiles), rank k the k-th row slice of the reduction; the partials are summed in the cluster: every CTA sends column group o of
// its accumulator (nn / DW_CL columns) to rank o's shared memory (slot = its rank, st.shared::cluster), rank o adds the DW_CL slots in rank
// order and writes the final dW columns -- no workspace, no partial stores to global, no reduce launch. Deterministic (fixed order).
#ifndef DW_CL
#define DW_CL 0
#endif
#ifndef DW_CLX
#define DW_CLX 0                                                           // development: 1 no DSMEM sends, 2 no reduction rows (epilogue only)
#endif
// r3-swa part 21 (DW_GX): the cluster's partials meet in a global workspace (L2) instead of DSMEM -- every rank stores its [128][nn] fp32
// accumulator to WS + (tile DW_CL + rank) 128 x 256 floats by its 128 epilogue threads (a row a thread, whole 128-B lines), one cluster
// barrier (release / acquire), then rank o adds the DW_CL partials of its column group nn / DW_CL in rank order (0 + s0 + s1 + ...: the
// DSMEM form's order, the same bits) and writes the final dW. A trace of the DSMEM form (clock64, CTA 0, L 1024, 2 tiles x 8 CTAs) had the
// exchange at ~9700 of the kernel's ~14000 clocks after the MMAs.
#ifndef DW_GX
#define DW_GX 0
#endif
#if DW_CL
static_assert(DW_BOX3, "the cluster build loads 3-D-map boxes");
static_assert(DW_CL == 2 || DW_CL == 4 || DW_CL == 8 || DW_CL == 16, "cluster size");
#endif
constexpr int BKR = DW_BKR;                                                // reduction rows per stage
static_assert(DW_CL || BKR == 16, "BKR");
constexpr int ABOX = BKR * 128;                                            // one [16 rows][32 fp32] box: 2 KB
constexpr int NBOX = DW_CL ? 12 : 20;                                                 // boxes of a stage (<= 16 A atoms + <= 8 B atoms: 4 + 8 at most together)
constexpr int STG = NBOX * ABOX;                                           // 40 KB
constexpr int O_BAR = NS * STG, SMEM_BYTES = O_BAR + 256;
constexpr int TILE = 128 * 128, NTILE = (NS * STG) / TILE;                 // the epilogue's staging tiles ([128 rows][32 fp32] SW128) over the ring
static_assert(SMEM_BYTES <= 232448, "shared memory");
// problem p: m_total = 128 MBf(p), n = NNf(p), A columns = 32 AAf(p), B columns = 32 BBf(p)
DEVI int MBf(int p) { return (p & 1) ? 1 : 4; }
DEVI int NNf(int p) { return p == 3 ? 256 : 128; }
DEVI int AAf(int p) { return (p & 1) ? 4 : 16; }
DEVI int BBf(int p) { return p == 3 ? 8 : 4; }

struct Bars {
  uint64_t full[NS], empty[NS], done, rcv;
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(256, 1)
swa_dw_tf32_sm100(const __grid_constant__ CUtensorMap ma0, const __grid_constant__ CUtensorMap mb0, const __grid_constant__ CUtensorMap ma1,
                  const __grid_constant__ CUtensorMap mb1, const __grid_constant__ CUtensorMap ma2, const __grid_constant__ CUtensorMap mb2,
                  const __grid_constant__ CUtensorMap ma3, const __grid_constant__ CUtensorMap mb3, float* __restrict__ WS, int R,
                  int ns0, int ns1, int ns2, int ns3, int cta0,
                  const __grid_constant__ CUtensorMap mws128, const __grid_constant__ CUtensorMap mws256
#if DW_CL
                  , float* __restrict__ o0, float* __restrict__ o1, float* __restrict__ o2, float* __restrict__ o3, float* __restrict__ og
#endif
#if TRACE
                  , long long* TRb
#endif
                  ) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
#if DW_CL
  // tile of this cluster (cta0: the launch's first tile): 0-3 dWqkvg m-block, 4 dWo, 5-8 dWu m-block, 9 dWd; the slice = the cluster rank
  const int tile = (int)blockIdx.x / DW_CL + cta0;
#if DWM
  // N2 (DWM=1): the modulation's weight gradient dWmod [6C, C] = dmod^T silu(c) -- ONE problem of six 128-row output tiles (A = dmod [R, 6C], the A map's box
  // 4 column atoms at 4 mbq; B = silu(c) [R, C]); the outputs o0 + 128 mbq rows
  const int p = 0;
  const int mbq = tile;
#else
  const int p = tile < 4 ? 0 : tile == 4 ? 1 : tile < 9 ? 2 : 3;
  const int mbq = p == 0 ? tile : p == 2 ? tile - 5 : 0;                  // the A operand's m-block (column atoms 4 mbq ..)
#endif
  const int slice = (int)cluster_rank(), nsp = DW_CL;
  (void)ns0; (void)ns1; (void)ns2; (void)ns3; (void)WS;
#else
  // problem and slice of this CTA (scalars: an indexed array would live in local memory)
  int p = 0, slice = (int)blockIdx.x + cta0, nsp = ns0;   // cta0: a launch of a sub-range of the problems (the grid = its CTAs)
  if (slice >= ns0) {
    slice -= ns0; p = 1; nsp = ns1;
    if (slice >= ns1) {
      slice -= ns1; p = 2; nsp = ns2;
      if (slice >= ns2) { slice -= ns2; p = 3; nsp = ns3; }
    }
  }
#endif
  const int rows_per = (((R + nsp - 1) / nsp) + BKR - 1) / BKR * BKR;
  const int r0 = min(slice * rows_per, R), r1 = min(R, r0 + rows_per);
  // (cluster build: the last slice's last stage may run past R -- TMA fills those rows with zeros; the other slices are whole stages)
  const int nblk = DW_CLX == 2 ? 0 : (r1 - r0 + (DW_CL ? BKR - 1 : 0)) / BKR;
#if DW_CL
  const int mbn = 1, nn = NNf(p), aa = 4, bb = BBf(p);
#else
  const int mbn = MBf(p), nn = NNf(p), aa = AAf(p), bb = BBf(p);
#endif
  const CUtensorMap* mA = p == 0 ? &ma0 : p == 1 ? &ma1 : p == 2 ? &ma2 : &ma3;
  const CUtensorMap* mB = p == 0 ? &mb0 : p == 1 ? &mb1 : p == 2 ? &mb2 : &mb3;
#if !DW_CL
  // workspace of this slice: [ns][m_total][n] fp32 after the earlier problems'
  size_t wso = 0;
  if (p > 0) wso += (size_t)ns0 * (4 * 128 * 128);
  if (p > 1) wso += (size_t)ns1 * (1 * 128 * 128);
  if (p > 2) wso += (size_t)ns2 * (4 * 128 * 128);
  float* ws = WS + wso + (size_t)slice * mbn * 128 * nn;
#endif

  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.done, 1);
    mbar_init(&B.rcv, 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();
  TR(0, tid == 0);

  if (warp == 0) {
    // a stage's boxes one per lane (round 3 part 4: one thread issuing the 20 loads of every stage spent ~37 ns on each -- 3 us for the 80
    // boxes of a 64-row slice before the last one was even requested)
    if (lane == 0) pdl_wait();                                             // the operands come from the previous kernels
    __syncwarp();
    TR(1, lane == 0);
    for (int blk = 0; blk < nblk; ++blk) {
      const int s = blk % NS;
      const uint32_t st = su + s * STG;
      if (lane == 0) {
        if (blk >= NS) mbar_wait(&B.empty[s], ((blk / NS) - 1) & 1);
        mbar_expect_tx(&B.full[s], (uint32_t)(aa + bb) * ABOX);
      }
      __syncwarp();
      const int row = r0 + blk * BKR;
#if DW_BOX3
      // round 3 part 14: the stage as TWO boxes of 3-D maps ([32 fp32][16 rows][aa | bb column atoms]: the same smem layout as the
      // aa + bb 2-KB boxes -- the SM's TMA cost is per box, not per byte)
#if DW_CL
      if (lane == 0) tma_load_3d(st, mA, &B.full[s], 0, row, 4 * mbq);     // (the A maps of dWqkvg / dWu: boxes of 4 column atoms)
#else
      if (lane == 0) tma_load_3d(st, mA, &B.full[s], 0, row, 0);
#endif
      else if (lane == 1) tma_load_3d(st + aa * ABOX, mB, &B.full[s], 0, row, 0);
#else
      if (lane < aa) tma_load_2d(st + lane * ABOX, mA, &B.full[s], 32 * lane, row);
      else if (lane < aa + bb) tma_load_2d(st + lane * ABOX, mB, &B.full[s], 32 * (lane - aa), row);
#endif
    }
    TR(2, lane == 0);
  } else if (warp == 1) {
    const uint32_t id = tf::idesc(128, nn, 1, 1);
    for (int blk = 0; blk < nblk; ++blk) {
      const int s = blk % NS;
      mbar_wait(&B.full[s], (blk / NS) & 1);
      tc_fence_after();
      TR(3 + min(blk, 3), lane == 0);
      const uint32_t st = su + s * STG;
      if (elect_one()) {
        for (int mb = 0; mb < mbn; ++mb) {
          const uint64_t da = tf::desc_mn32b(st + 4 * mb * ABOX, ABOX), db = tf::desc_mn32b(st + aa * ABOX, ABOX);
#pragma unroll
          for (int ks = 0; ks < BKR / 8; ++ks)                              // K = 8 reduction rows: 1 KB of every atom
            tf::mma_ss(tmem + mb * nn, da + (uint64_t)((ks * 1024) >> 4), db + (uint64_t)((ks * 1024) >> 4), id, (blk > 0 || ks > 0) ? 1u : 0u);
        }
        tc_commit(&B.empty[s]);
        if (blk == nblk - 1) tc_commit(&B.done);
      }
      __syncwarp();
    }
  } else if (warp >= 4) {
    const int lq = warp & 3, r = lq * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16);
    if (nblk > 0) { mbar_wait(&B.done, 0); tc_fence_after(); }
    TR(8, tid == 128);
#if DW_CL
    (void)r; (void)trow;
  }
  // the partial leaves in rounds of 128 accumulator columns (dWd: two): the epilogue threads stage it in this CTA's ring as DW_CL blocks
  // [128 rows][cw] (block o = rank o's columns), one bulk copy (cp.async.bulk shared::cta -> shared::cluster) per block lands it in slot
  // `slice` of rank o's receive area, completing on rank o's `rcv` mbarrier; rank o then adds the DW_CL slots in rank order. (Round 3
  // part 15, first form: the rows' own st.shared::cluster stores -- ~10 us for the 64 KB of a CTA: DSMEM wants bulk transfers.)
  constexpr int cw = 128 / DW_CL, cq = cw / 4;                             // columns of a rank's group, float4s of its row
  constexpr uint32_t slot = 128u * cw * 4u;                                // one block: [128 rows][cw] fp32 (64 KB / DW_CL)
  constexpr uint32_t RECV = 0, STAGE = 65536;                              // receive area | staging area (in the ring: every MMA is done)
  static_assert(STAGE + 65536 <= (uint32_t)O_BAR, "ring");
  float* base = p == 0 ? ((mbq == 3 && og != nullptr) ? og : o0 + (size_t)mbq * 128 * 128) : p == 1 ? o1 : p == 2 ? o2 + (size_t)mbq * 128 * 128 : o3;
#if DW_GX
  {
    // a partial = its 32-column chunks [128 rows][32 fp32] in the 128-B swizzle (sw128), chunk-major: PB bytes, one bulk store; rank o
    // then bulk-loads the o-th contiguous 1/DW_CL of every rank's partial (any partition adds the same elements in the same order)
    const uint32_t PB = 128u * (uint32_t)nn * 4u, SL = PB / DW_CL;
    static_assert(DW_CL <= 8, "DW_GX: 8 partial slots a tile");
    char* wst = reinterpret_cast<char*>(WS) + (size_t)tile * 8 * 128 * 256 * 4;    // this tile's DW_CL partials (8 slots a tile: any CL <= 8)
    if (warp >= 4) {
      const int lq = warp & 3, r = lq * 32 + lane;
      const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16);
      tc_fence_after();
#pragma unroll 1
      for (int c = 0; c < nn; c += 32) {
        uint32_t v[32];
        if (nblk > 0) { tmem_ld32(trow + c, v); tmem_wait_ld(); }
        else {
#pragma unroll
          for (int e = 0; e < 32; ++e) v[e] = 0u;
        }
#pragma unroll
        for (int q = 0; q < 8; ++q)
          sts128(su + (uint32_t)(c >> 5) * 16384u + sw128((uint32_t)r, (uint32_t)q), make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
      }
      tc_fence_before();
      fence_proxy_async();
      named_bar_sync(1, 128);
      if (tid == 128) {
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(wst + (size_t)slice * PB), "r"(su), "r"(PB) : "memory");
        tma_store_commit();
        tma_store_wait0();
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
    }
    if (tid == 0) mbar_expect_tx(&B.rcv, PB);
    TR(10, tid == 0);
    cluster_sync();                                                        // (release / acquire: every rank's partial is in global memory)
    TR(11, tid == 0);
    if (tid == 0) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      for (int s = 0; s < DW_CL; ++s)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(su + (uint32_t)s * SL), "l"(wst + (size_t)s * PB + (size_t)slice * SL), "r"(SL), "r"(smem_u32(&B.rcv)) : "memory");
    }
    mbar_wait(&B.rcv, 0);
    TR(13, tid == 0);
    const int len4 = (int)(SL / 16);
    for (int i = tid; i < len4; i += 256) {
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int s = 0; s < DW_CL; ++s) {
        const uint4 u = lds128(su + (uint32_t)s * SL + (uint32_t)i * 16u);
        acc.x += __uint_as_float(u.x); acc.y += __uint_as_float(u.y); acc.z += __uint_as_float(u.z); acc.w += __uint_as_float(u.w);
      }
      const int f = slice * len4 * 4 + 4 * i;                              // flat float index in the partial
      const int ch = f >> 12, rem = f & 4095, r = rem >> 5, pq = (rem & 31) >> 2;
      *reinterpret_cast<float4*>(base + (size_t)r * nn + ch * 32 + ((pq ^ (r & 7)) << 2)) = acc;
    }
    TR(14, tid == 0);
    __syncthreads();
    if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
    return;
  }
#endif
  const int rounds = nn / 128;
  if (tid == 0) mbar_expect_tx(&B.rcv, DW_CLX == 1 ? 0u : DW_CL * slot);
  __syncthreads();
  TR(10, tid == 0);
  cluster_sync();                                                          // every CTA's MMAs are done (rings free), every rcv armed
  TR(11, tid == 0);
  for (int h = 0; h < rounds; ++h) {
    if (warp >= 4) {
      const int lq = warp & 3, r = lq * 32 + lane;
      const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16) + (uint32_t)(128 * h);
      tc_fence_after();
#pragma unroll 1
      for (int c = 0; c < 128; c += 32) {
        uint32_t v[32];
        if (nblk > 0) { tmem_ld32(trow + c, v); tmem_wait_ld(); }
        else {
#pragma unroll
          for (int e = 0; e < 32; ++e) v[e] = 0u;
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int col = c + 4 * q, o = col / cw, qi = (col - o * cw) >> 2;
          sts128(su + STAGE + (uint32_t)o * slot + (uint32_t)r * (cw * 4u) + ((uint32_t)(qi ^ (r & (cq - 1))) << 4),
                 make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
        }
      }
      tc_fence_before();
      fence_proxy_async();                                                 // the staged blocks are read by the bulk copies (async proxy)
      named_bar_sync(1, 128);
      if (DW_CLX != 1 && tid >= 128 && tid < 128 + DW_CL) {
        const uint32_t o = (uint32_t)(tid - 128);
        uint32_t d, b;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(su + RECV + (uint32_t)slice * slot), "r"(o));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b) : "r"(smem_u32(&B.rcv)), "r"(o));
        asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(d), "r"(su + STAGE + o * slot), "r"(slot), "r"(b) : "memory");
      }
    }
    TR(12 + 4 * h, tid == 0);
    mbar_wait_cl(&B.rcv, (uint32_t)(h & 1));                               // every sender's block of this rank's columns has landed
    TR(13 + 4 * h, tid == 0);
    // rank `slice` owns the columns 128 h + slice cw .. + cw - 1: the DW_CL blocks added in rank order (fixed order: deterministic)
    for (int idx = tid; idx < 128 * cq; idx += 256) {
      const int row = idx / cq, qi = idx - row * cq;
      const uint32_t off = (uint32_t)row * (cw * 4u) + ((uint32_t)(qi ^ (row & (cq - 1))) << 4);
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int s = 0; s < DW_CL; ++s) {
        const uint4 u = lds128(su + RECV + (uint32_t)s * slot + off);
        acc.x += __uint_as_float(u.x); acc.y += __uint_as_float(u.y); acc.z += __uint_as_float(u.z); acc.w += __uint_as_float(u.w);
      }
      *reinterpret_cast<float4*>(base + (size_t)row * nn + 128 * h + slice * cw + 4 * qi) = acc;
    }
    TR(14 + 4 * h, tid == 0);
    if (h + 1 < rounds) {
      __syncthreads();
      if (tid == 0) mbar_expect_tx(&B.rcv, DW_CLX == 1 ? 0u : DW_CL * slot);
      __syncthreads();
    }
    cluster_sync();   // every rank has read its receive area and every copy out of our staging area is complete (reuse / exit)
    TR(15 + 4 * h, tid == 0);
  }
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
#else
    // the partial sum leaves by TMA stores from [128 rows][32 fp32] SW128 tiles staged in the ring's stages (free: every MMA is done) --
    // round 3 part 4: the rows' own 16-B global stores (one row per lane, 512 B apart) were 10 of a 64-row slice's 14.5 us
    const CUtensorMap* mW = p == 3 ? &mws256 : &mws128;
    const int row0 = p == 3 ? slice * 128 : (int)(wso / 128) + slice * mbn * 128;   // in rows of nn floats of the map
    int i = 0;
#if DW_DIRECT
    // round 3 part 14: the staged tile leaves by coalesced 16-B global stores of all 128 epilogue threads (a warp: 4 rows of 128 B per
    // instruction) instead of one thread's TMA stores -- one thread issued the 16 stores of a 512-row problem-0 slice ~600 clocks apart
    (void)row0;
    for (int mb = 0; mb < mbn; ++mb) {
      for (int c = 0; c < nn; c += 32, ++i) {
        uint32_t v[32];
        if (nblk > 0) { tmem_ld32(trow + mb * nn + c, v); tmem_wait_ld(); }
        else {
#pragma unroll
          for (int e = 0; e < 32; ++e) v[e] = 0u;
        }
        const uint32_t tile = su + (uint32_t)(i & 1) * TILE;
#pragma unroll
        for (int q = 0; q < 8; ++q) sts128(tile + sw128((uint32_t)r, (uint32_t)q), make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
        named_bar_sync(1, 128);
        float* gbase = ws + (size_t)(mb * 128) * nn + c;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int idx = k * 128 + r, rr = idx >> 3, q = idx & 7;
          *reinterpret_cast<uint4*>(gbase + (size_t)rr * nn + 4 * q) = lds128(tile + sw128((uint32_t)rr, (uint32_t)q));
        }
      }
    }
#else
    for (int mb = 0; mb < mbn; ++mb) {
      for (int c = 0; c < nn; c += 32, ++i) {
        if (i >= NTILE) { if (tid == 128) tma_store_wait_read0(); named_bar_sync(1, 128); }   // the tile's previous store has read it
        uint32_t v[32];
        if (nblk > 0) { tmem_ld32(trow + mb * nn + c, v); tmem_wait_ld(); }
        else {
#pragma unroll
          for (int e = 0; e < 32; ++e) v[e] = 0u;
        }
        const uint32_t tile = su + (uint32_t)(i % NTILE) * TILE;
#pragma unroll
        for (int q = 0; q < 8; ++q) sts128(tile + sw128((uint32_t)r, (uint32_t)q), make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
        fence_proxy_async();
        named_bar_sync(1, 128);
        if (tid == 128) { tma_store_2d(mW, tile, c, row0 + mb * 128); tma_store_commit(); }
      }
    }
    if (tid == 128) tma_store_wait0();
#endif
    TR(9, tid == 128);
    tc_fence_before();
  }
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
#endif  // DW_CL

// out_p[e] = sum over slices s (in order) of WS[offset_p + s m_total n + e]; one thread per four consecutive elements
DEVI void dw_sum(const float* __restrict__ w, float* __restrict__ out, int ns, size_t n4, size_t i) {
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  const float4* w4 = reinterpret_cast<const float4*>(w);
  for (int s = 0; s < ns; ++s) {
    const float4 v = w4[(size_t)s * n4 + i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  reinterpret_cast<float4*>(out)[i] = acc;
}
extern "C" __global__ void __launch_bounds__(256)
swa_dw_reduce_tf32_sm100(const float* __restrict__ WS, float* __restrict__ o0, float* __restrict__ o1, float* __restrict__ o2,
                         float* __restrict__ o3, int ns0, int ns1, int ns2, int ns3) {
  pdl_wait();                                                              // the partial sums come from the previous kernel
  constexpr size_t E0 = 4 * 128 * 128, E1 = 1 * 128 * 128, E2 = 4 * 128 * 128, E3 = 1 * 128 * 256;      // elements of the four outputs
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;                // float4 index over the four outputs, back to back
  size_t off = 0;
  if (i < E0 / 4) { dw_sum(WS + off, o0, ns0, E0 / 4, i); return; }
  i -= E0 / 4; off += (size_t)ns0 * E0;
  if (i < E1 / 4) { dw_sum(WS + off, o1, ns1, E1 / 4, i); return; }
  i -= E1 / 4; off += (size_t)ns1 * E1;
  if (i < E2 / 4) { dw_sum(WS + off, o2, ns2, E2 / 4, i); return; }
  i -= E2 / 4; off += (size_t)ns2 * E2;
  if (i < E3 / 4) dw_sum(WS + off, o3, ns3, E3 / 4, i);
}

// B200 round 3 part 14: the same reduction with dWg (rows 3C .. 4C of dP^T x) written into a buffer of its own (OG [C, C]): the block
// returns dWqkv and dWg as separate tensors without a copy (was dwqkvg[3C:].clone(), one more launch per block backward). Same sums.
extern "C" __global__ void __launch_bounds__(256)
swa_dw_reduce5_tf32_sm100(const float* __restrict__ WS, float* __restrict__ o0, float* __restrict__ o1, float* __restrict__ o2,
                          float* __restrict__ o3, int ns0, int ns1, int ns2, int ns3, float* __restrict__ og) {
  pdl_wait();
  constexpr size_t E0 = 4 * 128 * 128, E1 = 1 * 128 * 128, E2 = 4 * 128 * 128, E3 = 1 * 128 * 256;
  constexpr size_t Q0 = 3 * 128 * 128 / 4;                                 // float4s of the dWqkv rows
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t off = 0;
  if (i < E0 / 4) {
    if (i < Q0) { dw_sum(WS + off, o0, ns0, E0 / 4, i); return; }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    const float4* w4 = reinterpret_cast<const float4*>(WS + off);
    for (int s = 0; s < ns0; ++s) {
      const float4 v = w4[(size_t)s * (E0 / 4) + i];
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    reinterpret_cast<float4*>(og)[i - Q0] = acc;
    return;
  }
  i -= E0 / 4; off += (size_t)ns0 * E0;
  if (i < E1 / 4) { dw_sum(WS + off, o1, ns1, E1 / 4, i); return; }
  i -= E1 / 4; off += (size_t)ns1 * E1;
  if (i < E2 / 4) { dw_sum(WS + off, o2, ns2, E2 / 4, i); return; }
  i -= E2 / 4; off += (size_t)ns2 * E2;
  if (i < E3 / 4) dw_sum(WS + off, o3, ns3, E3 / 4, i);
}
