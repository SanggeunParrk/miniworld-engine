// tsx_expand2p.cu -- the PERSISTENT CTA-pair expand + SwiGLU of the small-M Transition chain, sm_100a, bf16, inference
// (-DDIM=<D> -DHID=<H> -DUH=<hidden units per CTA: 32 / 64> -DKBX=<K-blocks per A ring stage>).
// tsx_expand2.cu launches one pair per (128-row tile, 2 UH-unit chunk) with M = 128 pair products (64 rows a CTA: the tensor cores at
// half rate) and, past one wave (tiles x H / (2 UH) pairs > 74), every pair loads its weight panel again. Here the grid is (chunk j,
// group g) pairs, at most one wave, and the products are M = 256: CTA r holds the 128 rows of tile 2 rb + r of a 256-row block rb and
// its B half [Wa_r; Wb_r] (units [h0, h0 + UH)), and receives in its tensor memory its 128 rows x [a_0 | b_0 | a_1 | b_1] (columns 0 .. 2 UH
// the leader's B rows, 2 UH .. 4 UH the peer's). Pair (j, g) keeps its weight panel (the whole K) RESIDENT in shared memory, loaded once
// before griddepcontrol.wait (parameters), and loops over the row blocks g, g + tg, ... The blocks' xn K-blocks stream through an A ring
// of STG stages (KBX K-blocks per 3-D box; lane s of the producer warp issues ring slot s across block boundaries, so the next block's
// xn lands during the current block's products); the accumulator is double-buffered in tensor memory (2 x 4 UH columns): the SwiGLU of
// block k overlaps the products of block k + 1. Rows past m: zero-filled by TMA, not stored.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include "tsx_store.cuh"
using namespace s100;

#ifndef DIM
#define DIM 768
#endif
#ifndef HID
#define HID (2 * DIM)
#endif
#ifndef UH
#define UH 32
#endif
#ifndef KBX
#define KBX 1
#endif
constexpr int D_ = DIM, H_ = HID, ROWS = 128, NKB = D_ / 64, NP = H_ / (2 * UH);
static_assert((UH == 32 || UH == 64) && H_ % (2 * UH) == 0 && D_ % 64 == 0, "32- / 64-unit halves, whole K-blocks");
static_assert(NKB % KBX == 0, "whole ring stages");
constexpr int AKB = ROWS * 128;                                 // one K-block of this CTA's 128 xn rows, 16 KB
constexpr int WKB = 2 * UH * 128;                                // one K-block of [Wa_r; Wb_r]
constexpr int NS = NKB / KBX, AST = KBX * AKB, WBYTES = NKB * WKB;
constexpr int SMEM_MAX = 232448 - 1024, BARB = 512;
constexpr int CAP = (SMEM_MAX - WBYTES - BARB) / AST;
constexpr int STG = CAP < 32 ? CAP : 32;
static_assert(STG >= 2, "at least two A ring stages beside the resident weights");
constexpr int O_W = 0, O_A = WBYTES, O_BAR = O_A + STG * AST, SMEM_BYTES = O_BAR + BARB;
constexpr uint32_t TC = 4 * UH;                                  // columns of one accumulator buffer [a_0 | b_0 | a_1 | b_1]
constexpr uint32_t TCOLS = 2 * TC;                               // double buffer (256 / 512)
constexpr uint32_t IDESC = idesc_bf16(256, 4 * UH);
// -DESIG: the SwiGLU sigmoid. 1 (default): 0.5 tanh.approx(a / 2) + 0.5, one MUFU op per unit (the D 64 / 128 kernels' TR_SIG 1 form):
// the epilogue is MUFU-bound (128 rows x 2 UH units a CTA and block); 0: the kit form (ex2 + rcp, two MUFU ops); 2: f16x2 tanh (A/B)
#ifndef ESIG
#define ESIG 1
#endif
typedef __nv_bfloat16 ACT;
constexpr int NEPI = 2 * 8;                                      // epilogue warps of the pair (release of an accumulator buffer)

struct Bars {
  uint64_t full[STG], empty[STG], wfull[NS], acc_full[2], acc_empty[2];
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= BARB, "barrier block");

DEVI void tma_load_3d_2sm(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar) & 0xFEFFFFFFu), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
DEVI void load_a(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int st, int row) {
  if (KBX == 1) tma_load_2d_2sm(dst, m, bar, st * 64, row); else tma_load_3d_2sm(dst, m, bar, 0, row, st * KBX);
}

// maps: xn [M][D] (box 64 x 128 x KBX, the kb_box map); wa, wb [H][D] (box 64 x UH). h [M][H]. Grid: 2 x NP x tg CTAs (pairs (j, g),
// j fastest); m rows (256-row blocks: (m + 255) / 256).
extern "C" __global__ void __launch_bounds__(384, 1)
transition_sx_expand2p(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mwa,
                       const __grid_constant__ CUtensorMap mwb, ACT* __restrict__ hs, int m, int tg TSX_STAMP_ARG) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x >> 1, rank = (int)cluster_rank();
  const bool leader = rank == 0;
  const int j = pair % NP, g = pair / NP, h0 = j * 2 * UH + rank * UH;
  const int nrb = (m + 2 * ROWS - 1) / (2 * ROWS);
  const int nt = nrb > g ? (nrb - g + tg - 1) / tg : 0;         // this pair's row blocks g, g + tg, ...
  const int nit = nt * NS;                                       // A ring iterations (block, stage)
  if (tid == 0) TSX_ST(0);

  if (tid == 0) {
    for (int s = 0; s < STG; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int s = 0; s < NS; ++s) mbar_init(&B.wfull[s], 1);
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], NEPI); }
    fence_barrier_init();
    prefetch_map(&mxn); prefetch_map(&mwa); prefetch_map(&mwb);
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), TCOLS); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();                                      // the peer's TMA completes on the leader's barriers
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // the resident weight panel, before the wait (parameters): lane st < NS issues the K-blocks of ring stage st (Wa_r, Wb_r boxes) on
    // a barrier of their own, so the first block's products start as the panel's first K-blocks land
    if (lane < NS) {
      if (leader) mbar_expect_tx(&B.wfull[lane], 2 * KBX * WKB);
#pragma unroll
      for (int i = 0; i < KBX; ++i) {
        const int kb = lane * KBX + i;
        tma_load_2d_2sm(su + O_W + kb * WKB, &mwa, &B.wfull[lane], kb * 64, h0);
        tma_load_2d_2sm(su + O_W + kb * WKB + UH * 128, &mwb, &B.wfull[lane], kb * 64, h0);
      }
    }
    pdl_wait();                                                // xn is complete and visible
    if (lane == 0) TSX_ST(1);
    pdl_launch();
    if (lane < STG)
      for (int it = lane; it < nit; it += STG) {
        if (it >= STG) mbar_wait(&B.empty[lane], ((it / STG) - 1) & 1);
        if (leader) mbar_expect_tx(&B.full[lane], 2 * AST);
        const int k = it / NS, st = it - k * NS;
        load_a(su + O_A + lane * AST, &mxn, &B.full[lane], st, ((g + k * tg) * 2 + rank) * ROWS);
      }
    if (lane == 0) TSX_ST(2);
  } else if (warp == 1) {
    pdl_wait();
    pdl_launch();
    if (leader) {
      if (lane == 0) TSX_ST(3);
#pragma unroll 1
      for (int k = 0, it = 0; k < nt; ++k) {
        const int b = k & 1;
        if (k >= 2) mbar_wait_cl(&B.acc_empty[b], ((k >> 1) - 1) & 1);
        tc_fence_after();
        const uint32_t acc = tmem + b * TC;
#pragma unroll 1
        for (int st = 0; st < NS; ++st, ++it) {
          const int s = it % STG;
          if (k == 0) mbar_wait(&B.wfull[st], 0);
          mbar_wait(&B.full[s], (it / STG) & 1);
          if (lane == 0 && st == 0 && k < 4) TSX_ST(9 + k);
          tc_fence_after();
          if (elect_one()) {
#pragma unroll
            for (int i = 0; i < KBX; ++i) {
              const uint64_t da = desc_k128(su + O_A + s * AST + i * AKB), db = desc_k128(su + O_W + (st * KBX + i) * WKB);
#pragma unroll
              for (int ks = 0; ks < 4; ++ks)
                umma_ss2(acc, da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), IDESC, (st > 0 || i > 0 || ks > 0) ? 1u : 0u);
            }
            tc_commit2_mc(&B.empty[s], 3);
            if (st == NS - 1) tc_commit2_mc(&B.acc_full[b], 3);
          }
          __syncwarp();
        }
      }
      if (lane == 0) TSX_ST(4);
    }
  } else if (warp >= 4) {
    pdl_wait();
    pdl_launch();
    // ------------------------------------------------------------------------------------------ SwiGLU, one row per thread: warps 4-7
    // the leader's units (columns [a_0 | b_0]), warps 8-11 the peer's ([a_1 | b_1]); UH units in passes of 32
    const int q2 = (warp - 4) >> 2, r = (warp & 3) * 32 + lane, hb = j * 2 * UH + q2 * UH;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16) + q2 * 2 * UH;
    constexpr int ES = 32, NW = ES / 2, NPASS = UH / ES;
#pragma unroll 1
    for (int k = 0; k < nt; ++k) {
      const int b = k & 1, arow = ((g + k * tg) * 2 + rank) * ROWS;
      const bool whole = arow + ROWS <= m;
      mbar_wait(&B.acc_full[b], (k >> 1) & 1);
      if (warp == 4 && lane == 0 && k < 3) TSX_ST(13 + k);
      tc_fence_after();
      const uint32_t tb = trow + b * TC;
#pragma unroll
      for (int p = 0; p < NPASS; ++p) {
        uint32_t av[ES], bv[ES], hp[NW];
        tmem_ld32(tb + p * ES, av);
        tmem_ld32(tb + UH + p * ES, bv);
        tmem_wait_ld();
        if (p == NPASS - 1) {                                    // the buffer's last read of this warp: release it to the leader
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive_remote_relaxed(&B.acc_empty[b], 0);
        }
#pragma unroll
        for (int kk = 0; kk < NW; ++kk) {
          const float a0 = __uint_as_float(av[2 * kk]), a1 = __uint_as_float(av[2 * kk + 1]);
#if ESIG == 0
          const float s0 = sigmoid_kit(a0), s1 = sigmoid_kit(a1);
#elif ESIG == 2
          float s0, s1;
          sigmoid2_tanh_h2(a0, a1, s0, s1);
#else
          const f2 s2 = sigmoid2(mk2(a0, a1));
          const float s0 = lo2(s2), s1 = hi2(s2);
#endif
          hp[kk] = pack_bf16(a0 * s0 * __uint_as_float(bv[2 * kk]), a1 * s1 * __uint_as_float(bv[2 * kk + 1]));
        }
        if (whole) {
          store_t(hs + (size_t)arow * H_ + hb + p * ES + store_t_off<NW>(r, lane, H_, 2), H_, hp, lane);
        } else if (arow + r < m) {                               // the last, partial block: own row only
          ACT* o = hs + (size_t)(arow + r) * H_ + hb + p * ES;
#pragma unroll
          for (int c = 0; c < NW / 4; ++c) stg128(o + 8 * c, make_uint4(hp[4 * c], hp[4 * c + 1], hp[4 * c + 2], hp[4 * c + 3]));
        }
      }
    }
    if (warp == 4 && lane == 0) TSX_ST(5);
  } else {
    pdl_wait();
    pdl_launch();
  }
  __syncthreads();
  cluster_sync_relaxed();                                      // the leader's products read the peer's shared memory until the last commit
  if (tid == 0) TSX_ST(8);
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, TCOLS); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
