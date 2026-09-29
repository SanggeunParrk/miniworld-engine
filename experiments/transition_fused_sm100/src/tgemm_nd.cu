// tgemm_nd.cu — out[M][D] = A[M][K] B[K][D] with a fused epilogue, sm_100a, bf16 operands, fp32 accumulation (-DDIM=<D> -DKDIM=<K>):
//   EPI_RESID (default): out = bf16(acc + x)      -- the squeeze + residual of the two-kernel Transition forward (A = h, K = 4D, B^T = Ws)
//   EPI_PLAIN:           out = bf16(acc)          -- d_xn of the wide backward (A = [dA | dB], K = 8D, B^T = [Wa; Wb]^T); x unused
// The B operand is passed transposed and K-major, Bt[D][K] (for the squeeze that is Ws itself).
//
// 2-CTA tcgen05.mma: the two CTAs of a cluster run their own 128-row tiles in lockstep; the leader issues M = 256 products; N = D is cut
// into parts of at most 256 columns (D 384 = 256 + 128, D 512 = 256 + 256) and each CTA holds half of every part's B rows. A and B
// stream through a ring in 64-wide K-blocks. The accumulator (D columns) is double-buffered when D <= 256. The epilogue runs on two
// warpgroups that take alternate 64-column blocks, each prefetching its x blocks through a 2-slot ring and TMA-storing the result.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DIM
#define DIM 384
#endif
#ifndef KDIM
#define KDIM (4 * DIM)
#endif
constexpr int D_ = DIM, K_ = KDIM, ROWS = 128, NKB = K_ / 64, NCB = D_ / 64;
constexpr int NP = (D_ + 255) / 256;                           // N parts
__host__ __device__ constexpr int part_n0(int p) { return p * 256; }
__host__ __device__ constexpr int part_len(int p) { return (D_ - p * 256) < 256 ? (D_ - p * 256) : 256; }
constexpr int KBT = ROWS * 128;                                // one K-block of a 128-row tile: 16 KB
constexpr int BHALF = (D_ / 2) * 128;                          // this CTA's B rows (half of every part) of one K-block
constexpr int S_A = 0, S_B = KBT, STAGE = KBT + BHALF;
// epilogue warpgroups: 2 (warps 4-7, 12-15), or 3 with -DEPI3 (warps 8-11 too): with 3 at D = 384 every warpgroup owns two column
// blocks and two x slots, so no slot is refilled inside a tile and the accumulator is released as soon as it is read
#ifdef EPI3
constexpr int NEPI = 3;
#else
constexpr int NEPI = 2;
#endif
#ifdef NST_
constexpr int NST = NST_;
#else
constexpr int NST = D_ >= 512 ? 3 : (NEPI == 3 ? 3 : 4);
#endif
constexpr int NACC = D_ <= 256 ? 2 : 1;
constexpr int O_ST = 0, O_X = NST * STAGE;                     // epilogue x-block rings: NEPI warpgroups x 2 slots x 16 KB
constexpr int O_BAR = O_X + NEPI * 2 * KBT;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
static_assert(STAGE % 1024 == 0 && D_ % 128 == 0 && K_ % 64 == 0, "tile shapes");
static_assert(NCB >= 2 * NEPI, "every epilogue warpgroup owns at least two column blocks (the x-slot refill order assumes it)");

struct Bars {
  uint64_t full[NST];                                          // leader
  uint64_t empty[NST], acc_full[2];                            // both CTAs (multicast commits)
  uint64_t acc_empty[2];                                       // leader: 8 epilogue warps per CTA
  uint64_t xfull[NEPI][2];                                     // local: per epilogue warpgroup, per slot
  uint32_t tmem;
};

// maps: A [M][K] (box 64 x 64); Bt [D][K] (box 64 x 64); x, out [M][D] (box 64 x 64)
extern "C" __global__ void __launch_bounds__(512, 1)
transition_gemm_nd_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mbt,
                         const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mout, int tiles) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  auto count = [&](int k) { return (tiles > k) ? (tiles - k + G - 1) / G : 0; };
  const int n_valid = count(cta), n_local = count(cta & ~1);
  auto tile_of = [&](int i) { return i < n_valid ? cta + i * G : (n_valid > 0 ? cta + (n_valid - 1) * G : 0); };

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.acc_full[s], 1); mbar_init(&B.acc_empty[s], 8 * NEPI); }
    for (int g = 0; g < NEPI; ++g) for (int s = 0; s < 2; ++s) mbar_init(&B.xfull[g][s], 1);
    fence_barrier_init();
    prefetch_map(&ma); prefetch_map(&mbt); prefetch_map(&mx); prefetch_map(&mout);
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
#ifdef OLD_SYNC
  cluster_sync();
#else
  cluster_sync_relaxed();                                      // the barriers were published by fence_barrier_init
#endif
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    if (lane == 0) {
      int st = 0;
      for (int i = 0; i < n_local; ++i) {
        const int row = tile_of(i) * ROWS;
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int s = st % NST;
          if (st >= NST) mbar_wait(&B.empty[s], ((st / NST) - 1) & 1);
          if (leader) mbar_expect_tx(&B.full[s], 2 * STAGE);
          const uint32_t base = su + O_ST + s * STAGE;
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_load_2d_2sm(base + S_A + h * 8192, &ma, &B.full[s], kb * 64, row + h * 64);
          uint32_t boff = S_B;
#pragma unroll
          for (int p = 0; p < NP; ++p) {
            const int half = part_len(p) / 2, r0 = part_n0(p) + crank * half;
            for (int b = 0; b < half / 64; ++b) tma_load_2d_2sm(base + boff + b * 8192, &mbt, &B.full[s], kb * 64, r0 + b * 64);
            boff += half * 128;
          }
        }
      }
    }
  } else if (warp == 1) {
    if (leader) {
      int st = 0;
      for (int i = 0; i < n_local; ++i) {
        const int s = i % NACC, u = i / NACC;
        if (i >= NACC) mbar_wait_cl(&B.acc_empty[s], (u - 1) & 1);
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int sg = st % NST;
          mbar_wait(&B.full[sg], (st / NST) & 1);
          tc_fence_after();
          const uint32_t base = su + O_ST + sg * STAGE;
          const uint64_t da = desc_k128(base + S_A);
          if (elect_one()) {
            uint32_t boff = S_B;
#pragma unroll
            for (int p = 0; p < NP; ++p) {
              const uint64_t db = desc_k128(base + boff);
              const uint32_t idesc = idesc_bf16(256, part_len(p));
#pragma unroll
              for (int ks = 0; ks < 4; ++ks)
                umma_ss2(tmem + s * 256 + part_n0(p), da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), idesc, (kb > 0 || ks > 0) ? 1u : 0u);
              boff += (part_len(p) / 2) * 128;
            }
            tc_commit2_mc(&B.empty[sg], 3);
            if (kb == NKB - 1) tc_commit2_mc(&B.acc_full[s], 3);
          }
          __syncwarp();
        }
      }
    }
#ifdef EPI3
  } else if (warp >= 4) {
#else
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
#endif
    // ------------------------------------------------------------------------------------------ epilogue: warpgroup g takes column blocks
    // cb = g, g + 2, ...; out = bf16(acc + x), staged in the x slot and TMA-stored
    setmaxnreg_inc<152>();
#ifdef EPI3
    const int g = (warp >> 2) - 1;
#else
    const int g = warp >= 12 ? 1 : 0;
#endif
    const bool lead = (warp & 3) == 0 && lane == 0;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int nblk = (NCB - g + NEPI - 1) / NEPI;                // this warpgroup's blocks per tile
    int slot = 0;                                               // running x-slot use count of this warpgroup
    auto issue_x = [&](int i, int k, int sl) {                  // x block k of tile i into slot sl
#ifdef EPI_PLAIN
      return;
#endif
      const int row = tile_of(i) * ROWS, cb = g + NEPI * k;
      const uint32_t dst = su + O_X + (g * 2 + sl) * KBT;
      mbar_expect_tx(&B.xfull[g][sl], KBT);
#pragma unroll
      for (int h = 0; h < 2; ++h) tma_load_2d(dst + h * 8192, &mx, &B.xfull[g][sl], cb * 64, row + h * 64);
    };
#ifndef ABL_NOEPI
    if (lead && n_local > 0) { issue_x(0, 0, 0); if (nblk > 1) issue_x(0, 1, 1); }
#endif
    for (int i = 0; i < n_local; ++i) {
      const int s = i % NACC, u = i / NACC;
      mbar_wait(&B.acc_full[s], u & 1);
      tc_fence_after();
      for (int k = 0; k < nblk; ++k, ++slot) {
        const int sl = slot & 1, cb = g + NEPI * k;
        uint32_t acc[64];
        tmem_ld32(trow + s * 256 + cb * 64, *reinterpret_cast<uint32_t(*)[32]>(acc));
        tmem_ld32(trow + s * 256 + cb * 64 + 32, *reinterpret_cast<uint32_t(*)[32]>(acc + 32));
        tmem_wait_ld();
        if (k == nblk - 1) {                                    // this warpgroup has read all its columns of the buffer
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive_remote_relaxed(&B.acc_empty[s], 0);
        }
#ifdef ABL_NOEPI                                                 // ablation: mainloop + TMEM reads only (no x, no store)
        if (acc[0] == 0x7fffffffu && acc[63] == 0x7fffffffu) sts32(su + O_X, acc[1]);
        continue;
#endif
        const uint32_t xb = su + O_X + (g * 2 + sl) * KBT;
#ifdef EPI_PLAIN
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          uint32_t o[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) o[e] = pack_bf16(__uint_as_float(acc[q * 8 + 2 * e]), __uint_as_float(acc[q * 8 + 2 * e + 1]));
          sts128(xb + sw128(r, q), make_uint4(o[0], o[1], o[2], o[3]));
        }
        if (false)
#else
        mbar_wait(&B.xfull[g][sl], (slot >> 1) & 1);
#endif
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint32_t ad = xb + sw128(r, q);
          const uint4 xv = lds128(ad);
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
          uint32_t o[4];
#pragma unroll
          for (int e = 0; e < 4; ++e)
            o[e] = pack_bf16(bf16lo(xw[e]) + __uint_as_float(acc[q * 8 + 2 * e]), bf16hi(xw[e]) + __uint_as_float(acc[q * 8 + 2 * e + 1]));
          sts128(ad, make_uint4(o[0], o[1], o[2], o[3]));
        }
        fence_proxy_async();
        named_bar_sync(1 + g, 128);
        if (lead) {
          const int row = tile_of(i) * ROWS;
          if (i < n_valid) {
#pragma unroll
            for (int h = 0; h < 2; ++h) tma_store_2d(&mout, xb + h * 8192, cb * 64, row + h * 64);
          }
          tma_store_commit();
          // the next block for this slot: k + 2 of this tile, or the start of the next tile
          int ni = i, nk = k + 2;
          if (nk >= nblk) { ni = i + 1; nk -= nblk; }
#ifdef LATE_REFILL
          // a refill into this tile waits for this store's read now; the refills into the next tile wait until the tile's last block
          // (the accumulator is released by then), so no store read sits between two TMEM reads of one tile
          if (ni == i) { tma_store_wait_read0(); issue_x(ni, nk, sl); }
          else if (k == nblk - 1) {
            tma_store_wait_read0();
            if (i + 1 < n_local)
              for (int kk = (nblk >= 2 ? k - 1 : k); kk <= k; ++kk) issue_x(i + 1, kk + 2 - nblk, (slot - (k - kk)) & 1);
          }
#else
          tma_store_wait_read0();                               // the slot is refilled next
          if (ni < n_local) issue_x(ni, nk, sl);
#endif
        }
        named_bar_sync(1 + g, 128);                             // nobody reads the slot before the refill was issued in order
      }
    }
    if (lead) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
#ifdef OLD_SYNC
  cluster_sync();
#else
  cluster_sync_relaxed();
#endif
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}
