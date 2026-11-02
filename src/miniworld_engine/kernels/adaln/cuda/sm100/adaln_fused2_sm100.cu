// adaln_fused2_sm100.cu -- the B200 (sm_100a) fp32 adaptive LayerNorm without a backward at the atom width (d = cond = 128), in ONE
// launch over CTA PAIRS (cta_group::2):
//
//   y = sigmoid(cn Ws^T + bs) * LN(x) + cn Wb^T,      cn = LN(cond) * wc,      TF32 tensor cores, fp32 statistics and epilogue
//
// adaln_fused_sm100.cu's fp32 build keeps the whole 128-column fp32 weight tile resident (128 KB), which leaves no room to stage x: its
// producers and epilogue read x one row a thread from global memory (a cache line a lane and instruction), and the one A buffer
// serialises TMA -> LayerNorm -> MMA a tile (~6.5 us a tile at [40960, 128], slower than the row kernels around cuBLAS). Here the two
// CTAs of a cluster run their own 128-row tiles of a 256-row pair tile in lockstep and the leader (rank 0) issues M = 256 products whose
// B operand is split by N across the pair: each CTA holds 64 of the 128 weight rows of each projection (64 KB for both), so its own
// cond tile (A, 64 KB) AND its own x tile (64 KB) fit beside them, both by TMA:
//
//   warp 0       TMA: this CTA's cond tile (A) and x tile per pair tile, its half of [Wb ; Ws] once (completing on the leader's barrier)
//   warp 1       leader: MMA issuer (tcgen05.mma.cta_group::2.kind::tf32, both accumulators, N = 128); both: TMEM (cta_group::2) owner
//   warps 2-5    one thread a row: LN(cond) * wc in place in A (fp32 on the TF32 grid, four partial sums a statistic), then a remote
//                arrive on the leader's A barrier
//   warps 6-9    one thread a row: x's statistics from the staged x (two-pass), y = sigmoid(acc1 + bs) (x rstd - mean rstd) + acc0 written
//                over x in the staging (``rna``: on the TF32 grid), then one TMA store of the tile; the staging is free for the next x
//
// The leader's MMA commits arrive on both CTAs' barriers (multicast); the completions of both CTAs' producers / epilogues are counted on
// the leader's. Accumulators double-buffered in tensor memory (2 x 2 x 128 columns): the epilogue of tile i overlaps the MMA of tile i + 1.
// Any M (TMA zero-fills the tail rows and clips them on the store).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int KC = 128, DX = 128, BN = 128;          // cond width, x width, output columns (both projections: N = 128 each)
constexpr int NACC = 2;
constexpr int BKE = 32;                              // fp32 elements per 128-byte K-block row
constexpr int NKB = KC / BKE;                        // 4 K-blocks
constexpr int KB_A = 128 * 128;                      // one K-block of a 128-row tile: 16 KB
constexpr int KB_BH = (BN / 2) * 128;                // one K-block of this CTA's half of a weight tile: 64 rows, 8 KB
constexpr int A_BYTES = NKB * KB_A, BH_BYTES = NKB * KB_BH, X_BYTES = (DX / BKE) * KB_A;
constexpr int O_B = 0, O_A = NACC * BH_BYTES, O_X = O_A + A_BYTES, O_WC = O_X + X_BYTES, O_BS = O_WC + KC * 4, O_BAR = O_BS + BN * 4;
constexpr int SMEM_BYTES = O_BAR + 256;
constexpr int TCOLS = NACC * BN;                     // one accumulator buffer: 256 TMEM columns
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(O_A % 1024 == 0 && O_X % 1024 == 0, "128-B swizzle atoms");

// kind::tf32, TF32 A / B, fp32 D, K-major both, N = 128, M = 256 (the pair)
constexpr uint32_t IDESC2 = (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)(256 >> 4) << 24);
DEVI void umma2(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC2), "r"(acc) : "memory");
}
DEVI float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }

struct Bars {
  uint64_t bfull, afull, acce[2];                    // leader: weight bytes of both CTAs; 4 producer / 4 epilogue warps of each CTA
  uint64_t aempty, accf[2];                          // both: the leader's multicast commits
  uint64_t rawfull, xfull, xempty;                   // local
  uint32_t tmem;
};

// maps: B0 Wb, B1 Ws [128][KC] fp32 (TF32 grid) (box 32 x 64), C cond [M][KC], X x [M][DX], Y y [M][DX] (box 32 x 128). WC wc (fp32),
// BS the gate bias (fp32). One pair tile = 256 rows; this CTA's rows: pair tile * 256 + 128 rank.
extern "C" __global__ void __launch_bounds__(320, 1)
adaln_fused2_sm100(const __grid_constant__ CUtensorMap mB0, const __grid_constant__ CUtensorMap mB1, const __grid_constant__ CUtensorMap mC,
                   const __grid_constant__ CUtensorMap mX, const __grid_constant__ CUtensorMap mY, const float* __restrict__ WC,
                   const float* __restrict__ BS, int M, float eps_c, float eps_x, int rna) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  float* WCs = reinterpret_cast<float*>(sm + O_WC);
  float* BSs = reinterpret_cast<float*>(sm + O_BS);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  const int pair = (int)blockIdx.x >> 1, npairs = (int)gridDim.x >> 1;
  const int ptiles = (M + 255) / 256;

  if (threadIdx.x == 0) {
    mbar_init(&B.bfull, 1); mbar_init(&B.afull, 8); mbar_init(&B.aempty, 1);
    for (int s = 0; s < 2; ++s) { mbar_init(&B.acce[s], 8); mbar_init(&B.accf[s], 1); }
    mbar_init(&B.rawfull, 1); mbar_init(&B.xfull, 1); mbar_init(&B.xempty, 1);
    fence_barrier_init();
    prefetch_map(&mB0); prefetch_map(&mB1); prefetch_map(&mC); prefetch_map(&mX); prefetch_map(&mY);
  }
  if (threadIdx.x >= 64 && threadIdx.x < 64 + KC) WCs[threadIdx.x - 64] = __ldg(WC + threadIdx.x - 64);
  if (threadIdx.x >= 192 && threadIdx.x < 192 + BN) BSs[threadIdx.x - 192] = __ldg(BS + threadIdx.x - 192);
  if (warp == 1) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();                                    // both CTAs' barriers initialised before any remote arrive / TMA completion
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA
    if (lane == 0) {
      int tc = 0;
      for (int t = pair; t < ptiles; t += npairs, ++tc) {
        const int m0 = t * 256 + crank * 128;
        if (tc > 0) mbar_wait(&B.aempty, (tc - 1) & 1);   // the pair's MMA of the last tile has read A
        mbar_expect_tx(&B.rawfull, A_BYTES);
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) tma_load_2d(su + O_A + kb * KB_A, &mC, &B.rawfull, kb * BKE, m0);
        if (tc == 0) {                                   // the weights after the first cond tile (the LayerNorm comes first)
          if (leader) mbar_expect_tx(&B.bfull, 2 * NACC * BH_BYTES);
#pragma unroll
          for (int kb = 0; kb < NKB; ++kb) {
            tma_load_2d_2sm(su + O_B + kb * KB_BH, &mB0, &B.bfull, kb * BKE, crank * (BN / 2));
            tma_load_2d_2sm(su + O_B + BH_BYTES + kb * KB_BH, &mB1, &B.bfull, kb * BKE, crank * (BN / 2));
          }
        }
        if (tc > 0) mbar_wait(&B.xempty, (tc - 1) & 1);   // the epilogue of the last tile has stored y out of the staging
        mbar_expect_tx(&B.xfull, X_BYTES);
#pragma unroll
        for (int kb = 0; kb < DX / BKE; ++kb) tma_load_2d(su + O_X + kb * KB_A, &mX, &B.xfull, kb * BKE, m0);
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA (leader)
    if (leader && lane == 0) {
      mbar_wait(&B.bfull, 0);
      int tc = 0;
      for (int t = pair; t < ptiles; t += npairs, ++tc) {
        const int buf = tc & 1;
        mbar_wait_cl(&B.afull, tc & 1);                  // both CTAs' A tiles normalised
        mbar_wait_cl(&B.acce[buf], ((tc >> 1) & 1) ^ 1); // both CTAs' epilogues done with this accumulator buffer
        tc_fence_after();
        const uint32_t dbase = tmem + buf * TCOLS;
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) {
          const uint64_t da = desc_k128(su + O_A + kb * KB_A);
#pragma unroll
          for (int j = 0; j < NACC; ++j) {
            const uint64_t db = desc_k128(su + O_B + j * BH_BYTES + kb * KB_BH);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              umma2(dbase + j * BN, da + (uint64_t)(kk * 2), db + (uint64_t)(kk * 2), (kb > 0 || kk > 0) ? 1u : 0u);
          }
        }
        tc_commit2_mc(&B.aempty, 3);
        tc_commit2_mc(&B.accf[buf], 3);
      }
    }
  } else if (warp < 6) {
    // ------------------------------------------------------------------------------------------------ LN(cond) (warps 2-5)
    const uint32_t r = (uint32_t)((warp - 2) * 32 + lane);
    int tc = 0;
    for (int t = pair; t < ptiles; t += npairs, ++tc) {
      mbar_wait(&B.rawfull, tc & 1);
      float ps[4] = {0.f, 0.f, 0.f, 0.f};
      uint4 raw[32];
#pragma unroll
      for (int c = 0; c < 32; ++c) {
        raw[c] = lds128(su + O_A + (c / 8) * KB_A + sw128(r, c % 8));
        ps[c & 3] += (__uint_as_float(raw[c].x) + __uint_as_float(raw[c].y)) + (__uint_as_float(raw[c].z) + __uint_as_float(raw[c].w));
      }
      const float mc = ((ps[0] + ps[1]) + (ps[2] + ps[3])) * (1.f / KC);
#pragma unroll
      for (int j = 0; j < 4; ++j) ps[j] = 0.f;
#pragma unroll
      for (int c = 0; c < 32; ++c) {
        const float a = __uint_as_float(raw[c].x) - mc, b = __uint_as_float(raw[c].y) - mc, d = __uint_as_float(raw[c].z) - mc,
                    e = __uint_as_float(raw[c].w) - mc;
        ps[c & 3] += (a * a + b * b) + (d * d + e * e);
      }
      const float rc = rsqrtf(((ps[0] + ps[1]) + (ps[2] + ps[3])) * (1.f / KC) + eps_c);
#pragma unroll
      for (int c = 0; c < 32; ++c) {
        const float4 w = *reinterpret_cast<const float4*>(WCs + 4 * c);
        sts128(su + O_A + (c / 8) * KB_A + sw128(r, c % 8),
               make_uint4(__float_as_uint(tf32r((__uint_as_float(raw[c].x) - mc) * rc * w.x)), __float_as_uint(tf32r((__uint_as_float(raw[c].y) - mc) * rc * w.y)),
                          __float_as_uint(tf32r((__uint_as_float(raw[c].z) - mc) * rc * w.z)), __float_as_uint(tf32r((__uint_as_float(raw[c].w) - mc) * rc * w.w))));
      }
      fence_proxy_async();                               // generic-proxy stores of A -> the pair's MMA (async proxy)
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_cta(&B.afull, 0);
    }
  } else {
    // ------------------------------------------------------------------------------------------------ epilogue (warps 6-9)
    const int q4 = warp & 3;
    const uint32_t rl = (uint32_t)(q4 * 32 + lane);    // this thread's row (TMEM lane)
    int tc = 0;
    for (int t = pair; t < ptiles; t += npairs, ++tc) {
      const int buf = tc & 1;
      const int m0 = t * 256 + crank * 128;
      mbar_wait(&B.xfull, tc & 1);
      // x's statistics, two passes over the staged row, four partial sums
      float ps[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c = 0; c < 32; ++c) {
        const uint4 u = lds128(su + O_X + (c / 8) * KB_A + sw128(rl, c % 8));
        ps[c & 3] += (__uint_as_float(u.x) + __uint_as_float(u.y)) + (__uint_as_float(u.z) + __uint_as_float(u.w));
      }
      const float mx = ((ps[0] + ps[1]) + (ps[2] + ps[3])) * (1.f / DX);
#pragma unroll
      for (int j = 0; j < 4; ++j) ps[j] = 0.f;
#pragma unroll
      for (int c = 0; c < 32; ++c) {
        const uint4 u = lds128(su + O_X + (c / 8) * KB_A + sw128(rl, c % 8));
        const float a = __uint_as_float(u.x) - mx, b = __uint_as_float(u.y) - mx, d = __uint_as_float(u.z) - mx, e = __uint_as_float(u.w) - mx;
        ps[c & 3] += (a * a + b * b) + (d * d + e * e);
      }
      const float rx = rsqrtf(((ps[0] + ps[1]) + (ps[2] + ps[3])) * (1.f / DX) + eps_x), cx = mx * rx;
      mbar_wait(&B.accf[buf], (tc >> 1) & 1);
      tc_fence_after();
      const uint32_t tb = tmem + ((uint32_t)(q4 * 32) << 16) + buf * TCOLS;
#pragma unroll 1
      for (int c = 0; c < BN / 32; ++c) {
        uint32_t acc[NACC][32];
        tmem_ld32(tb + c * 32, acc[0]);
        tmem_ld32(tb + BN + c * 32, acc[1]);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint32_t a = su + O_X + c * KB_A + sw128(rl, q);
          const uint4 u = lds128(a);
          float o[4] = {__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int col = c * 32 + q * 4 + e;
            const float sp = __uint_as_float(acc[1][q * 4 + e]) + BSs[col];
            const float y = sigmoid_kit(sp) * (o[e] * rx - cx) + __uint_as_float(acc[0][q * 4 + e]);
            o[e] = rna ? tf32r(y) : y;
          }
          sts128(a, make_uint4(__float_as_uint(o[0]), __float_as_uint(o[1]), __float_as_uint(o[2]), __float_as_uint(o[3])));
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_relaxed(&B.acce[buf], 0);
      fence_proxy_async();                               // y in the staging -> the TMA store (async proxy)
      named_bar_sync(2, 128);
      if (warp == 6 && lane == 0) {
#pragma unroll
        for (int kb = 0; kb < DX / BKE; ++kb) tma_store_2d(&mY, su + O_X + kb * KB_A, kb * BKE, m0);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.xempty);
      }
    }
    if (warp == 6 && lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 1) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
