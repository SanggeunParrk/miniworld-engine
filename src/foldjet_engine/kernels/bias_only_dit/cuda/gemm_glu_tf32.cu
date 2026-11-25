// gemm_glu_tf32.cu -- the bias-only token DiT's transition GEMMs with the SwiGLU in their epilogue, fp32 path, sm_100a, TF32 tensor
// cores (tcgen05 kind::tf32, fp32 operands, fp32 accumulation), D = 768, H = 2D:
//
//   forward  (default)  [a | b] = X [Wa; Wb]^T, h = silu(a) b; writes h and a | b (the backward's inputs): the fp32 twin of
//                       gemm_swiglu2_sm100.cu with SAVE_AB. Replaces cuBLAS + the fp32 swiglu row pass (which read a | b back).
//   backward (-DBWD)    dh = dZ Wsq (B = Wsq^T [H, D], K-major), never stored: the epilogue reads a, b of the same rows / hidden units
//                       and writes da = dh b s (1 + a (1 - s)), db = dh a s (s = sigmoid(a)) into dab. Replaces cuBLAS dh + swiglu_bwd
//                       (which wrote and read dh back).
//
// The MMA side is gemm_swiglu2_sm100.cu's: 2-CTA tcgen05 (cluster of two): the pair runs its own 128-row tiles in lockstep over the
// same chunk sequence, the leader issues M = 256, N = 256 products whose B is split by N (forward: the leader holds Wa_j, the peer Wb_j,
// 128 rows each; backward: Wsq^T rows j 256 .. + 127 / + 128 .. + 255), so each SM's TMA intake carries half the weights. A K-block
// is 32 fp32 (one 128-B swizzle row: 16 KB per 128-row operand tile, as the bf16 kernel's 64-column blocks), 4 MMAs of K = 8 each.
// Accumulators [128 rows x 256 columns] double-buffered in tensor memory (512 columns).
//
// Epilogue: two warpgroups (warps 4-7: units / columns 0 .. half, 8-11: the other half), each with an IO thread (warp 12 / 13 lane 0)
// that owns the warpgroup's ring of NB staging buffers. A tile is 16 columns x 128 rows (64-B rows, 64-B swizzle, 8 KB per tensor);
// a thread owns one row (its TMEM lane). Forward: the warps write h, a, b of a tile into a buffer; the IO thread stores it (TMA) and
// frees the buffer once the store has read it. Backward: the IO thread loads a, b of tile u + NB into the buffer tile u's stores
// have read; the warps overwrite a, b in place with da, db; the IO thread stores them. A tile's sequence number u counts over the
// warpgroup's items, so buffers, barriers and parities follow u alone.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int D_ = 768, H_ = 2 * D_, ROWS = 128, KB = 32, NKB = D_ / KB;
constexpr int KBT = ROWS * 128;                                // one K-block of a 128-row operand tile: 16 KB
constexpr int S_X = 0, S_W = KBT, STAGE = 2 * KBT;
constexpr int TILE = ROWS * 64;                                // a 16-column fp32 tile [128 rows][64 B]: 8 KB
#ifdef BWD
constexpr int NCH = H_ / 256;                                  // dh column chunks of 256 (the pair's N)
constexpr int CPW = 128;                                       // columns per warpgroup and item
constexpr int NB = 4, NT_BUF = 2;                              // staging buffers per warpgroup, tiles per buffer (a / da, b / db)
constexpr int NST = 3;
#else
constexpr int NCH = H_ / 128;                                  // hidden-unit chunks of 128 (a and b: N = 256)
constexpr int CPW = 64;                                        // hidden units per warpgroup and item
constexpr int NB = 2, NT_BUF = 3;                              // h, a, b
constexpr int NST = 4;
#endif
constexpr int TPI = CPW / 16;                                  // tiles per warpgroup and item
constexpr int BUF = NT_BUF * TILE;
constexpr int O_ST = 0, O_EP = NST * STAGE, O_BAR = O_EP + 2 * NB * BUF;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
static_assert(O_EP % 1024 == 0 && BUF % 1024 == 0, "tile alignment");
constexpr uint32_t IDESC = idesc_tf32(256, 256);

DEVI void umma_ss2_tf32(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }
// the fp32 rows' sigmoid (bias_only_dit_f32_rows.cu sg): the same bits as the swiglu / swiglu_bwd row passes these replace
DEVI float sgf(float v) { return 1.f / (1.f + __expf(-v)); }
DEVI uint32_t u4el(const uint4& v, int e) { return e == 0 ? v.x : e == 1 ? v.y : e == 2 ? v.z : v.w; }   // e a constant after unrolling

struct Bars {
  uint64_t full[NST], empty[NST], acc_full[2], acc_empty[2];
  uint64_t comp[2][NB];                                        // the warpgroup's 4 warps have written buffer b
  uint64_t ready[2][NB];                                       // forward: buffer b is free again; backward: its a, b are loaded
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(512, 1)
#ifdef BWD
bo_glu_bwd_tf32_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mw,
                      const __grid_constant__ CUtensorMap la, const __grid_constant__ CUtensorMap lb,
                      const __grid_constant__ CUtensorMap sa, const __grid_constant__ CUtensorMap sb, int tiles) {
#else
bo_glu_fwd_tf32_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mwa,
                      const __grid_constant__ CUtensorMap mwb, const __grid_constant__ CUtensorMap mh,
                      const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, int tiles) {
#endif
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  // (tile pair, chunk) items dealt round-robin over the pairs of the grid (gemm_swiglu2_sm100.cu ITEM_SCHED): the pair's CTAs take
  // tiles 2 t and 2 t + 1 of tile pair t; a missing odd tile is loaded clamped and not stored
  const int npairs = G >> 1, pair = cta >> 1, ntp = (tiles + 1) >> 1, ntot = ntp * NCH;
  const int nitem = (ntot > pair) ? (ntot - pair + npairs - 1) / npairs : 0;
  auto item_j = [&](int q) { return (pair + q * npairs) % NCH; };
  auto item_tile = [&](int q) { return 2 * ((pair + q * npairs) / NCH) + crank; };
  auto item_valid = [&](int q) { return item_tile(q) < tiles; };
  auto item_row = [&](int q) { const int t = item_tile(q); return (t < tiles ? t : tiles - 1) * ROWS; };

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.acc_full[s], 1); mbar_init(&B.acc_empty[s], 16); }   // 8 warps in each CTA of the pair
    for (int g = 0; g < 2; ++g)
      for (int b = 0; b < NB; ++b) { mbar_init(&B.comp[g][b], 4); mbar_init(&B.ready[g][b], 1); }
    fence_barrier_init();
    prefetch_map(&mx);
#ifdef BWD
    prefetch_map(&mw); prefetch_map(&la); prefetch_map(&lb); prefetch_map(&sa); prefetch_map(&sb);
#else
    prefetch_map(&mwa); prefetch_map(&mwb); prefetch_map(&mh); prefetch_map(&ma); prefetch_map(&mb);
#endif
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer (both CTAs)
    if (lane == 0) {
#ifdef BWD
      const CUtensorMap* mwp = &mw;
#else
      const CUtensorMap* mwp = leader ? &mwa : &mwb;
#endif
      int st = 0;
      for (int q = 0; q < nitem; ++q) {
        const int j = item_j(q), row = item_row(q);
#ifdef BWD
        const int wrow = j * 256 + crank * 128;
#else
        const int wrow = j * 128;
#endif
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int s = st % NST;
          if (st >= NST) mbar_wait(&B.empty[s], ((st / NST) - 1) & 1);
          if (leader) mbar_expect_tx(&B.full[s], 2 * STAGE);
          const uint32_t base = su + O_ST + s * STAGE;
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_load_2d_2sm(base + S_X + h * (KBT / 2), &mx, &B.full[s], kb * KB, row + h * 64);
          tma_load_2d_2sm(base + S_W, mwp, &B.full[s], kb * KB, wrow);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer (the leader)
    if (leader) {
      int st = 0;
      for (int q = 0; q < nitem; ++q) {
        const int s = q & 1, u = q >> 1;
        if (q >= 2) mbar_wait_cl(&B.acc_empty[s], (u - 1) & 1);
        tc_fence_after();
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int sg = st % NST;
          mbar_wait(&B.full[sg], (st / NST) & 1);
          tc_fence_after();
          const uint32_t base = su + O_ST + sg * STAGE;
          const uint64_t dx = desc_k128(base + S_X), dw = desc_k128(base + S_W);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < KB / 8; ++ks)          // K = 8 fp32 = 32 B per MMA
              umma_ss2_tf32(tmem + s * 256, dx + (uint64_t)(ks * 2), dw + (uint64_t)(ks * 2), IDESC, (kb > 0 || ks > 0) ? 1u : 0u);
            tc_commit2_mc(&B.empty[sg], 3);
            if (kb == NKB - 1) tc_commit2_mc(&B.acc_full[s], 3);
          }
          __syncwarp();
        }
      }
    }
  } else if (warp >= 4 && warp < 12) {
    // ------------------------------------------------------------------------------------------------ epilogue warps
    const int wg = (warp - 4) >> 2;
    const uint32_t lb_ = (uint32_t)(warp & 3) * 32, r = lb_ + lane, trow = tmem + (lb_ << 16);
    int u = 0;
    for (int q = 0; q < nitem; ++q) {
      const int s = q & 1;
      mbar_wait(&B.acc_full[s], (q >> 1) & 1);
      tc_fence_after();
#pragma unroll 1
      for (int sub = 0; sub < TPI; ++sub, ++u) {
        const int b = u % NB;
        const uint32_t buf = su + O_EP + (uint32_t)((wg * NB + b) * BUF);
#ifdef BWD
        uint32_t dv[16];
        tmem_ld16(trow + (uint32_t)(s * 256 + wg * CPW + sub * 16), dv);
        tmem_wait_ld();
        if (sub == TPI - 1) {                                    // the item's accumulator is read: the MMAs may reuse it
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive_remote_relaxed(&B.acc_empty[s], 0);
        }
        mbar_wait(&B.ready[wg][b], (u / NB) & 1);                // a, b of this tile loaded
        uint4 av[4], bv[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) { av[c] = lds128(buf + sw64(r, c)); bv[c] = lds128(buf + TILE + sw64(r, c)); }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          float da[4], db[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float dh = __uint_as_float(dv[4 * c + e]), a = __uint_as_float(u4el(av[c], e)), bb = __uint_as_float(u4el(bv[c], e));
            const float sa_ = sgf(a);
            da[e] = dh * bb * sa_ * (1.f + a * (1.f - sa_));
            db[e] = dh * a * sa_;
          }
          sts128(buf + sw64(r, c), make_uint4(__float_as_uint(da[0]), __float_as_uint(da[1]), __float_as_uint(da[2]), __float_as_uint(da[3])));
          sts128(buf + TILE + sw64(r, c), make_uint4(__float_as_uint(db[0]), __float_as_uint(db[1]), __float_as_uint(db[2]), __float_as_uint(db[3])));
        }
#else
        uint32_t av[16], bv[16];
        tmem_ld16(trow + (uint32_t)(s * 256 + wg * CPW + sub * 16), av);
        tmem_ld16(trow + (uint32_t)(s * 256 + 128 + wg * CPW + sub * 16), bv);
        tmem_wait_ld();
        if (sub == TPI - 1) {
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive_remote_relaxed(&B.acc_empty[s], 0);
        }
        if (u >= NB) mbar_wait(&B.ready[wg][b], ((u / NB) - 1) & 1);   // the store of tile u - NB has read the buffer
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          uint32_t hv[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float a = __uint_as_float(av[4 * c + e]), bb = __uint_as_float(bv[4 * c + e]);
            hv[e] = __float_as_uint(a * sgf(a) * bb);
          }
          sts128(buf + sw64(r, c), make_uint4(hv[0], hv[1], hv[2], hv[3]));
          sts128(buf + TILE + sw64(r, c), make_uint4(av[4 * c], av[4 * c + 1], av[4 * c + 2], av[4 * c + 3]));
          sts128(buf + 2 * TILE + sw64(r, c), make_uint4(bv[4 * c], bv[4 * c + 1], bv[4 * c + 2], bv[4 * c + 3]));
        }
#endif
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.comp[wg][b]);
      }
    }
  } else if (warp == 12 || warp == 13) {
    // ------------------------------------------------------------------------------------------------ IO thread of warpgroup wg
    if (lane == 0) {
      const int wg = warp - 12, total = nitem * TPI;
      auto coords = [&](int t, int& col, int& row) {           // tile t's first column (of the H-wide halves) and row
        const int q = t / TPI, sub = t % TPI;
#ifdef BWD
        col = item_j(q) * 256 + wg * CPW + sub * 16;
#else
        col = item_j(q) * 128 + wg * CPW + sub * 16;
#endif
        row = item_row(q);
      };
#ifdef BWD
      auto load = [&](int t) {
        const int b = t % NB;
        const uint32_t buf = su + O_EP + (uint32_t)((wg * NB + b) * BUF);
        int col, row; coords(t, col, row);
        mbar_expect_tx(&B.ready[wg][b], 2 * TILE);
        tma_load_2d(buf, &la, &B.ready[wg][b], col, row);
        tma_load_2d(buf + TILE, &lb, &B.ready[wg][b], col, row);
      };
      for (int t = 0; t < NB && t < total; ++t) load(t);
#endif
      // tile t's stores are committed as one bulk group (empty for a missing tile); the IO thread then waits only for tile t - 1's
      // group to have read its buffer (wait_group.read 1: tile t's stores stay in flight) and refills / frees THAT buffer. (Waiting
      // for tile t's own stores before going on serialised the ring: the backward ran at 57 % of its HBM floor, L768 256 us.)
      for (int t = 0; t < total; ++t) {
        const int b = t % NB;
        const uint32_t buf = su + O_EP + (uint32_t)((wg * NB + b) * BUF);
        mbar_wait(&B.comp[wg][b], (t / NB) & 1);
        if (item_valid(t / TPI)) {
          int col, row; coords(t, col, row);
#ifdef BWD
          tma_store_2d(&sa, buf, col, row);
          tma_store_2d(&sb, buf + TILE, col, row);
#else
          tma_store_2d(&mh, buf, col, row);
          tma_store_2d(&ma, buf + TILE, col, row);
          tma_store_2d(&mb, buf + 2 * TILE, col, row);
#endif
        }
        tma_store_commit();
        if (t >= 1) {
          tma_store_wait_read1();                                // tile t - 1's buffer is read
#ifdef BWD
          if (t - 1 + NB < total) load(t - 1 + NB);
#else
          mbar_arrive(&B.ready[wg][(t - 1) % NB]);
#endif
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
