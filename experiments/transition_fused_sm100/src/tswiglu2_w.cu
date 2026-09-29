// tswiglu2_w.cu — tswiglu_w.cu (the expand + SwiGLU of the two-kernel Transition forward, D = 384 / 512 / 256; -DDIM=<D>; writes h and,
// with save_ab, a and b) with the 128-row xn tile RESIDENT in shared memory: it is loaded once per tile and reused by all H / 128 hidden
// items, so only the weights stream through the ring. tswiglu_w re-loaded xn for every item, and its loads alone asked ~64 B/clk of an
// SM whose measured TMA intake is ~52 B/clk; the h / a / b stores share that budget.
// Staging: per warpgroup (h | a | b of its 64 units, 48 KB each) when it fits, else one 48 KB buffer the two warpgroups take in turn
// (-DSHARED_STG; D = 512). SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DIM
#define DIM 384
#endif
constexpr int D_ = DIM, H_ = 4 * DIM, HC = 128, NCH = H_ / HC, ROWS = 128, NKB = D_ / 64;
constexpr int KBT = ROWS * 128;                                // [128][64] bf16 tile, 16 KB
#ifndef NST_
#define NST_ 2
#endif
constexpr int NST = NST_;
#ifdef SHARED_STG
constexpr int NSB = 1;
#else
constexpr int NSB = 2;
#endif
constexpr int O_XN = 0, O_RING = NKB * KBT, O_STG = O_RING + NST * KBT;   // staging buffer b: h at +0, a at +KBT, b at +2 KBT
constexpr int O_BAR = O_STG + NSB * 3 * KBT;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t IDESC = idesc_bf16(256, 256);

struct Bars {
  uint64_t full[NST], xn_full;                                 // leader: both CTAs' transactions
  uint64_t empty[NST], acc_full[2], xn_free;                   // both CTAs (leader's multicast commits)
  uint64_t acc_empty[2];                                       // leader: 8 epilogue warps per CTA
  uint64_t staged[2], stage_free[2];                           // local
  uint32_t tmem;
};

// maps: xn [M][D] (box 64 x 64); wa, wb [H][D] (box 64 x 128); h, a, b [M][H] (box 64 x 64)
extern "C" __global__ void __launch_bounds__(512, 1)
transition_swiglu2_w_sm100(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mwa,
                           const __grid_constant__ CUtensorMap mwb, const __grid_constant__ CUtensorMap mh,
                           const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, int save_ab, int tiles) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  auto count = [&](int k) { return (tiles > k) ? (tiles - k + G - 1) / G : 0; };
  const int n_valid = count(cta), n_local = count(cta & ~1);
  const int nitem = n_local * NCH;
  auto tile_of = [&](int i) { return i < n_valid ? cta + i * G : (n_valid > 0 ? cta + (n_valid - 1) * G : 0); };

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.acc_full[s], 1); mbar_init(&B.acc_empty[s], 16); }
    for (int g = 0; g < 2; ++g) { mbar_init(&B.staged[g], 4); mbar_init(&B.stage_free[g], 1); }
    mbar_init(&B.xn_full, 1); mbar_init(&B.xn_free, 1);
    fence_barrier_init();
    prefetch_map(&mxn); prefetch_map(&mwa); prefetch_map(&mwb); prefetch_map(&mh); prefetch_map(&ma); prefetch_map(&mb);
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    if (lane == 0) {
      const CUtensorMap* mab = leader ? &mwa : &mwb;
      int st = 0;
      for (int i = 0; i < n_local; ++i) {
        const int row = tile_of(i) * ROWS;
        if (i >= 1) mbar_wait(&B.xn_free, (i - 1) & 1);
        if (leader) mbar_expect_tx(&B.xn_full, 2 * NKB * KBT);
        for (int kb = 0; kb < NKB; ++kb)
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_load_2d_2sm(su + O_XN + kb * KBT + h * 8192, &mxn, &B.xn_full, kb * 64, row + h * 64);
        for (int j = 0; j < NCH; ++j)
          for (int kb = 0; kb < NKB; ++kb, ++st) {
            const int s = st % NST;
            if (st >= NST) mbar_wait(&B.empty[s], ((st / NST) - 1) & 1);
            if (leader) mbar_expect_tx(&B.full[s], 2 * KBT);
            tma_load_2d_2sm(su + O_RING + s * KBT, mab, &B.full[s], kb * 64, j * HC);
          }
      }
    }
  } else if (warp == 1) {
    if (leader) {
      int st = 0;
      for (int q = 0; q < nitem; ++q) {
        const int i = q / NCH, j = q % NCH, s = q & 1, u = q >> 1;
        if (j == 0) mbar_wait(&B.xn_full, i & 1);
        if (q >= 2) mbar_wait_cl(&B.acc_empty[s], (u - 1) & 1);
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int sg = st % NST;
          mbar_wait(&B.full[sg], (st / NST) & 1);
          tc_fence_after();
          const uint64_t dxn = desc_k128(su + O_XN + kb * KBT), dw = desc_k128(su + O_RING + sg * KBT);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ss2(tmem + s * 256, dxn + (uint64_t)(ks * 2), dw + (uint64_t)(ks * 2), IDESC, (kb > 0 || ks > 0) ? 1u : 0u);
            tc_commit2_mc(&B.empty[sg], 3);
            if (kb == NKB - 1) {
              tc_commit2_mc(&B.acc_full[s], 3);
              if (j == NCH - 1) tc_commit2_mc(&B.xn_free, 3);
            }
          }
          __syncwarp();
        }
      }
    }
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    setmaxnreg_inc<152>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int half = warp >= 12 ? 1 : 0;
    for (int q = 0; q < nitem; ++q) {
      const int s = q & 1, u = q >> 1;
      mbar_wait(&B.acc_full[s], u & 1);
      tc_fence_after();
      uint32_t hp[32], ap[32], bp[32];
#pragma unroll
      for (int st2 = 0; st2 < 2; ++st2) {
        uint32_t av[32], bv[32];
        tmem_ld32(trow + s * 256 + half * 64 + st2 * 32, av);
        tmem_ld32(trow + s * 256 + 128 + half * 64 + st2 * 32, bv);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
          const float b0 = __uint_as_float(bv[2 * k]), b1 = __uint_as_float(bv[2 * k + 1]);
          ap[st2 * 16 + k] = pack_bf16(a0, a1);
          bp[st2 * 16 + k] = pack_bf16(b0, b1);
          hp[st2 * 16 + k] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_kit(a1) * b1);
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_relaxed(&B.acc_empty[s], 0);
#ifdef SHARED_STG
      // one buffer, taken in the order (item q, warpgroup 0), (q, 1), (q + 1, 0), ...: use n = 2 q + half
      const int n = 2 * q + half;
      if (half == 1) mbar_wait(&B.stage_free[0], q & 1);        // warpgroup 0's store of this item has been read
      else if (q >= 1) mbar_wait(&B.stage_free[1], (q - 1) & 1);
      const uint32_t ob = su + O_STG;
      (void)n;
#else
      if (q >= 1) mbar_wait(&B.stage_free[half], (q - 1) & 1);
      const uint32_t ob = su + O_STG + half * 3 * KBT;
#endif
#pragma unroll
      for (int qq = 0; qq < 8; ++qq) {
        sts128(ob + sw128(r, qq), make_uint4(hp[4 * qq], hp[4 * qq + 1], hp[4 * qq + 2], hp[4 * qq + 3]));
        if (save_ab) {
          sts128(ob + KBT + sw128(r, qq), make_uint4(ap[4 * qq], ap[4 * qq + 1], ap[4 * qq + 2], ap[4 * qq + 3]));
          sts128(ob + 2 * KBT + sw128(r, qq), make_uint4(bp[4 * qq], bp[4 * qq + 1], bp[4 * qq + 2], bp[4 * qq + 3]));
        }
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.staged[half]);
    }
  } else if (warp == 8 || warp == 9) {
#ifdef SHARED_STG
    if (warp == 8 && lane == 0) {                               // one store thread, the two warpgroups in turn
      for (int q = 0; q < nitem; ++q) {
        for (int t = 0; t < 2; ++t) {
          const uint32_t ob = su + O_STG;
#else
    if (lane == 0) {                                            // one store thread per warpgroup
      const int t = warp - 8;
      for (int q = 0; q < nitem; ++q) {
        {
          const uint32_t ob = su + O_STG + t * 3 * KBT;
#endif
          const int i = q / NCH, j = q % NCH;
          mbar_wait(&B.staged[t], q & 1);
          if (i < n_valid) {
            const int row = tile_of(i) * ROWS, col = j * HC + t * 64;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              tma_store_2d(&mh, ob + h * 8192, col, row + h * 64);
              if (save_ab) {
                tma_store_2d(&ma, ob + KBT + h * 8192, col, row + h * 64);
                tma_store_2d(&mb, ob + 2 * KBT + h * 8192, col, row + h * 64);
              }
            }
          }
          tma_store_commit();
          tma_store_wait_read0();
          mbar_arrive(&B.stage_free[t]);
        }
      }
      tma_store_wait0();
    }
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}
