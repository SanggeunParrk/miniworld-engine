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
//                over x in the staging (``rna``: on the TF32 grid), each 32-column block TMA-stored as soon as it is written; the next tile's
//                x block lands in its place as soon as that store has read it (the staging is single-buffered, so whole-tile handoffs
//                serialised the store, the next x load and its statistics: ~7 us of chain for a pair's second tile)
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
constexpr int O_B = 0, O_A = NACC * BH_BYTES, O_X = O_A + A_BYTES, O_WC = O_X + X_BYTES, O_BS = O_WC + KC * 4, O_XS = O_BS + BN * 4, O_BAR = O_XS + 128 * 8;
// the tail tile (``mini``): 16 rows a CTA of cond and of x (4 K-blocks of 16 rows x 128 B = 2 KB each), kept clear of the big tile's buffers; the
// MMA reads its A operand as a 128-row tile, the 112 rows past the 16 being whatever lies behind them in shared memory (rows are independent)
constexpr int MR = 16, MKB = MR * 128;
constexpr int O_MC = (O_BAR + 256 + 1023) / 1024 * 1024, O_MX = O_MC + NKB * MKB;
constexpr int SMEM_BYTES = O_MC + 24576;             // >= O_MC + 3 MKB + the 128-row read of the last K-block
constexpr int TCOLS = NACC * BN;                     // one accumulator buffer: 256 TMEM columns
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(O_A % 1024 == 0 && O_X % 1024 == 0, "128-B swizzle atoms");

// kind::tf32, TF32 A / B, fp32 D, K-major both, N = 128, M = 256 (the pair)
constexpr uint32_t IDESC2 = (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)(256 >> 4) << 24);
DEVI void umma2(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC2), "r"(acc) : "memory");
}
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }
DEVI float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }

DEVI void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }

struct Bars {
  uint64_t bfull, afull, acce[2];                    // leader: weight bytes of both CTAs; 4 producer / 4 epilogue warps of each CTA
  uint64_t aempty, accf[2];                          // both: the leader's multicast commits
  uint64_t rawfull, xfull[4], xempty[4];             // local; x by 32-column block
  uint64_t mrawfull, mxfull, mfull, maccf;           // the tail tile: cond / x landed (local), its cn normalised (leader, both CTAs), its product done
  uint32_t tmem;
};

// Epilogue of the output-column blocks [c0, c1) of the tile for this thread's row (TMEM lane); each block is TMA-stored as soon as its 128
// threads (barrier ``bar``) have written it. ``lead``: the thread that issues the stores of these blocks and returns the staging to the next
// x load.
DEVI void epi_range(uint32_t su, Bars& B, const float* BSs, const CUtensorMap* mY, uint32_t tb, uint32_t rl, int m0, int buf, int c0, int c1,
                    float rx, float cx, int rna, int bar, bool lead, int lane) {
#pragma unroll 1
  for (int c = c0; c < c1; ++c) {
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
    if (c == c1 - 1) {
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_relaxed(&B.acce[buf], 0);   // the accumulators are read: the pair's next MMA may write them
    }
    fence_proxy_async();                               // y block c in the staging -> its TMA store (async proxy)
    named_bar_sync(bar, 128);
    if (lead) {
      tma_store_2d(mY, su + O_X + c * KB_A, c * BKE, m0);
      tma_store_commit();
      if (c > c0) { tma_store_wait_read1(); mbar_arrive(&B.xempty[c - 1]); }   // block c - 1 is read out: the next x may land there
      if (c == c1 - 1) { tma_store_wait_read0(); mbar_arrive(&B.xempty[c]); }
    }
  }
}

// maps: B0 Wb, B1 Ws [128][KC] fp32 (TF32 grid) (box 32 x 64), C cond [M][KC], X x [M][DX], Y y [M][DX] (box 32 x 128). WC wc (fp32),
// BS the gate bias (fp32). One pair tile = 256 rows; this CTA's rows: pair tile * 256 + 128 rank.
extern "C" __global__ void __launch_bounds__(320, 1)
adaln_fused2_sm100(const __grid_constant__ CUtensorMap mB0, const __grid_constant__ CUtensorMap mB1, const __grid_constant__ CUtensorMap mC,
                   const __grid_constant__ CUtensorMap mX, const __grid_constant__ CUtensorMap mY, const __grid_constant__ CUtensorMap mC16,
                   const __grid_constant__ CUtensorMap mX16, const __grid_constant__ CUtensorMap mY16, const float* __restrict__ WC,
                   const float* __restrict__ BS, int M, float eps_c, float eps_x, int rna, int mini_base, int mini) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  float* WCs = reinterpret_cast<float*>(sm + O_WC);
  float* BSs = reinterpret_cast<float*>(sm + O_BS);
  float2* XSs = reinterpret_cast<float2*>(sm + O_XS);   // x's (rstd, mean rstd) a row, published by warps 6-9 for the helper epilogue of warps 2-5
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  const int pair = (int)blockIdx.x >> 1, npairs = (int)gridDim.x >> 1;
  // ``mini``: one 256-row tile a pair (``npairs`` of them from row 0), and the rows from ``mini_base`` on cut into 32-row slices, pair p's the
  // p-th (16 rows a CTA, own buffers, accumulator buffer 1): the pair's whole work is then one pass, with no second tile to queue behind the first
  const int ptiles = mini ? npairs : (M + 255) / 256;
  const int mrow = mini_base + pair * 32 + crank * 16;
  const bool mactive = mini && mini_base + pair * 32 < M;

  if (threadIdx.x == 0) {
    mbar_init(&B.bfull, 1); mbar_init(&B.afull, 16); mbar_init(&B.aempty, 1);
    for (int s = 0; s < 2; ++s) { mbar_init(&B.acce[s], 16); mbar_init(&B.accf[s], 1); }
    mbar_init(&B.rawfull, 1);
    mbar_init(&B.mrawfull, 1); mbar_init(&B.mxfull, 1); mbar_init(&B.mfull, 2); mbar_init(&B.maccf, 1);
    for (int kb = 0; kb < 4; ++kb) { mbar_init(&B.xfull[kb], 1); mbar_init(&B.xempty[kb], 1); }
    fence_barrier_init();
    prefetch_map(&mB0); prefetch_map(&mB1); prefetch_map(&mC); prefetch_map(&mX); prefetch_map(&mY);
    if (mini) { prefetch_map(&mC16); prefetch_map(&mX16); prefetch_map(&mY16); }
    // the first tile's cond and x go out before the cluster handshake and the tensor-memory allocation: their completion is on this CTA's
    // own barriers, and the loads then run under the setup (~1 us)
    if ((int)(blockIdx.x >> 1) < ptiles) {
      const int m0 = (int)(blockIdx.x >> 1) * 256 + (int)cluster_rank() * 128;
      mbar_expect_tx(&B.rawfull, A_BYTES);
#pragma unroll
      for (int kb = 0; kb < NKB; ++kb) tma_load_2d(su + O_A + kb * KB_A, &mC, &B.rawfull, kb * BKE, m0);
#pragma unroll
      for (int kb = 0; kb < DX / BKE; ++kb) {
        mbar_expect_tx(&B.xfull[kb], KB_A);
        tma_load_2d(su + O_X + kb * KB_A, &mX, &B.xfull[kb], kb * BKE, m0);
      }
    }
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
        if (tc > 0) {
          mbar_wait(&B.aempty, (tc - 1) & 1);           // the pair's MMA of the last tile has read A
          mbar_expect_tx(&B.rawfull, A_BYTES);
#pragma unroll
          for (int kb = 0; kb < NKB; ++kb) tma_load_2d(su + O_A + kb * KB_A, &mC, &B.rawfull, kb * BKE, m0);
        }
        if (tc == 0) {                                   // the weights after the first cond tile (the LayerNorm comes first)
          if (leader) mbar_expect_tx(&B.bfull, 2 * NACC * BH_BYTES);
#pragma unroll
          for (int kb = 0; kb < NKB; ++kb) {
            tma_load_2d_2sm(su + O_B + kb * KB_BH, &mB0, &B.bfull, kb * BKE, crank * (BN / 2));
            tma_load_2d_2sm(su + O_B + BH_BYTES + kb * KB_BH, &mB1, &B.bfull, kb * BKE, crank * (BN / 2));
          }
        }
        if (tc > 0) {
#pragma unroll
          for (int kb = 0; kb < DX / BKE; ++kb) {
            mbar_wait(&B.xempty[kb], (tc - 1) & 1);     // the last tile's y block kb has been stored out of the staging
            mbar_expect_tx(&B.xfull[kb], KB_A);
            tma_load_2d(su + O_X + kb * KB_A, &mX, &B.xfull[kb], kb * BKE, m0);
          }
        }
      }
      if (mactive) {
        mbar_expect_tx(&B.mrawfull, NKB * MKB);
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) tma_load_2d(su + O_MC + kb * MKB, &mC16, &B.mrawfull, kb * BKE, mrow);
        mbar_expect_tx(&B.mxfull, NKB * MKB);
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) tma_load_2d(su + O_MX + kb * MKB, &mX16, &B.mxfull, kb * BKE, mrow);
      }
    }
    if (mactive) {
      // ---------------------------------------------------------------------------------------- the tail tile (whole warp; TMEM lanes 0-31)
      __syncwarp();
      const uint32_t lr = (uint32_t)(lane >> 1), hf = (uint32_t)(lane & 1);
      mbar_wait(&B.mrawfull, 0);
      {
        float ps[4] = {0.f, 0.f, 0.f, 0.f};
        uint4 raw[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const uint32_t j = hf * 16 + ((i + 4 * hf) & 15);
          raw[i] = lds128(su + O_MC + (j >> 3) * MKB + sw128(lr, j & 7));
          ps[i & 3] += (__uint_as_float(raw[i].x) + __uint_as_float(raw[i].y)) + (__uint_as_float(raw[i].z) + __uint_as_float(raw[i].w));
        }
        float sm = (ps[0] + ps[1]) + (ps[2] + ps[3]);
        sm += __shfl_xor_sync(0xffffffffu, sm, 1);
        const float mc = sm * (1.f / KC);
#pragma unroll
        for (int j = 0; j < 4; ++j) ps[j] = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const float a = __uint_as_float(raw[i].x) - mc, b = __uint_as_float(raw[i].y) - mc, d = __uint_as_float(raw[i].z) - mc,
                      e = __uint_as_float(raw[i].w) - mc;
          ps[i & 3] += (a * a + b * b) + (d * d + e * e);
        }
        float vs = (ps[0] + ps[1]) + (ps[2] + ps[3]);
        vs += __shfl_xor_sync(0xffffffffu, vs, 1);
        const float rc = rsqrtf(vs * (1.f / KC) + eps_c);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const uint32_t j = hf * 16 + ((i + 4 * hf) & 15);
          const float4 w = *reinterpret_cast<const float4*>(WCs + 4 * j);
          sts128(su + O_MC + (j >> 3) * MKB + sw128(lr, j & 7),
                 make_uint4(__float_as_uint(tf32r((__uint_as_float(raw[i].x) - mc) * rc * w.x)), __float_as_uint(tf32r((__uint_as_float(raw[i].y) - mc) * rc * w.y)),
                            __float_as_uint(tf32r((__uint_as_float(raw[i].z) - mc) * rc * w.z)), __float_as_uint(tf32r((__uint_as_float(raw[i].w) - mc) * rc * w.w))));
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive_remote_cta(&B.mfull, 0);
      }
      // x's statistics (a lane a row; lanes 16-31 repeat rows 0-15 and store nothing), then the product's epilogue in place over x
      const uint32_t rx_row = (uint32_t)(lane & 15);
      mbar_wait(&B.mxfull, 0);
      float ps[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c = 0; c < 32; ++c) {
        const uint4 u = lds128(su + O_MX + (c / 8) * MKB + sw128(rx_row, c % 8));
        ps[c & 3] += (__uint_as_float(u.x) + __uint_as_float(u.y)) + (__uint_as_float(u.z) + __uint_as_float(u.w));
      }
      const float mx = ((ps[0] + ps[1]) + (ps[2] + ps[3])) * (1.f / DX);
#pragma unroll
      for (int j = 0; j < 4; ++j) ps[j] = 0.f;
#pragma unroll
      for (int c = 0; c < 32; ++c) {
        const uint4 u = lds128(su + O_MX + (c / 8) * MKB + sw128(rx_row, c % 8));
        const float a = __uint_as_float(u.x) - mx, b = __uint_as_float(u.y) - mx, d = __uint_as_float(u.z) - mx, e = __uint_as_float(u.w) - mx;
        ps[c & 3] += (a * a + b * b) + (d * d + e * e);
      }
      const float rx = rsqrtf(((ps[0] + ps[1]) + (ps[2] + ps[3])) * (1.f / DX) + eps_x), cx = mx * rx;
      mbar_wait(&B.maccf, 0);
      tc_fence_after();
      const uint32_t tb = tmem + TCOLS;                  // accumulator buffer 1, TMEM lanes 0-31 (this warp's quadrant)
#pragma unroll 1
      for (int c = 0; c < BN / 32; ++c) {
        uint32_t acc[NACC][32];
        tmem_ld32(tb + c * 32, acc[0]);
        tmem_ld32(tb + BN + c * 32, acc[1]);
        tmem_wait_ld();
        if (lane < MR) {
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const uint32_t a = su + O_MX + c * MKB + sw128(rx_row, q);
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
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) tma_store_2d(&mY16, su + O_MX + kb * MKB, kb * BKE, mrow);
        tma_store_commit();
        tma_store_wait0();
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA (leader)
    if (leader && lane == 0) {
      mbar_wait(&B.bfull, 0);
      if (mactive) {                                     // the tail tile first (its LayerNorm is the shorter): the same product from the small buffers into accumulator buffer 1
        mbar_wait_cl(&B.mfull, 0);
        tc_fence_after();
        const uint32_t dbase = tmem + TCOLS;
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) {
          const uint64_t da = desc_k128(su + O_MC + kb * MKB);
#pragma unroll
          for (int j = 0; j < NACC; ++j) {
            const uint64_t db = desc_k128(su + O_B + j * BH_BYTES + kb * KB_BH);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              umma2(dbase + j * BN, da + (uint64_t)(kk * 2), db + (uint64_t)(kk * 2), (kb > 0 || kk > 0) ? 1u : 0u);
          }
        }
        tc_commit2_mc(&B.maccf, 3);
      }
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
  } else {
    // ------------------------------------------------------------------------------------------------ LN(cond), x statistics, epilogue
    // Warps 2-9 are the same 8 warps for the three jobs of a tile. LayerNorm of cond in place: two lanes a row (16 rows a warp, 64 columns a
    // lane, the two partial sums combined by one shuffle), written back as the TF32-grid cn. Then warps 6-9 take x's statistics (two passes
    // over the staged row) and the output columns 0..63, warps 2-5 columns 64..127 (``epi_range``); a TMEM lane a thread.
    const int wi = warp - 2;
    const bool grpA = warp >= 6;
    const int q4 = warp & 3;
    const uint32_t rl = (uint32_t)(q4 * 32 + lane);    // this thread's row as the epilogue (TMEM lane)
    const uint32_t lr = (uint32_t)(wi * 16 + (lane >> 1)), hf = (uint32_t)(lane & 1);   // as the LayerNorm: row, column half
    int tc = 0;
    for (int t = pair; t < ptiles; t += npairs, ++tc) {
      const int buf = tc & 1;
      const int m0 = t * 256 + crank * 128;
      mbar_wait(&B.rawfull, tc & 1);
      {
        // chunk i of this lane: 16-byte column group (hf * 16 + ((i + 4 hf) & 15)): the two lanes of a row start 4 groups apart, which keeps
        // the eight lanes of a shared-memory phase (4 rows x 2 halves) on 32 distinct banks
        float ps[4] = {0.f, 0.f, 0.f, 0.f};
        uint4 raw[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const uint32_t j = hf * 16 + ((i + 4 * hf) & 15);
          raw[i] = lds128(su + O_A + (j >> 3) * KB_A + sw128(lr, j & 7));
          ps[i & 3] += (__uint_as_float(raw[i].x) + __uint_as_float(raw[i].y)) + (__uint_as_float(raw[i].z) + __uint_as_float(raw[i].w));
        }
        float sm = (ps[0] + ps[1]) + (ps[2] + ps[3]);
        sm += __shfl_xor_sync(0xffffffffu, sm, 1);
        const float mc = sm * (1.f / KC);
#pragma unroll
        for (int j = 0; j < 4; ++j) ps[j] = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const float a = __uint_as_float(raw[i].x) - mc, b = __uint_as_float(raw[i].y) - mc, d = __uint_as_float(raw[i].z) - mc,
                      e = __uint_as_float(raw[i].w) - mc;
          ps[i & 3] += (a * a + b * b) + (d * d + e * e);
        }
        float vs = (ps[0] + ps[1]) + (ps[2] + ps[3]);
        vs += __shfl_xor_sync(0xffffffffu, vs, 1);
        const float rc = rsqrtf(vs * (1.f / KC) + eps_c);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const uint32_t j = hf * 16 + ((i + 4 * hf) & 15);
          const float4 w = *reinterpret_cast<const float4*>(WCs + 4 * j);
          sts128(su + O_A + (j >> 3) * KB_A + sw128(lr, j & 7),
                 make_uint4(__float_as_uint(tf32r((__uint_as_float(raw[i].x) - mc) * rc * w.x)), __float_as_uint(tf32r((__uint_as_float(raw[i].y) - mc) * rc * w.y)),
                            __float_as_uint(tf32r((__uint_as_float(raw[i].z) - mc) * rc * w.z)), __float_as_uint(tf32r((__uint_as_float(raw[i].w) - mc) * rc * w.w))));
        }
        fence_proxy_async();                             // generic-proxy stores of A -> the pair's MMA (async proxy)
        __syncwarp();
        if (lane == 0) mbar_arrive_remote_cta(&B.afull, 0);
      }
      if (grpA) {
#pragma unroll
        for (int kb = 0; kb < DX / BKE; ++kb) mbar_wait(&B.xfull[kb], tc & 1);
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
        XSs[rl] = make_float2(rx, cx);
        named_bar_arrive(4, 256);                        // the warps 2-5 read it (the same TMEM lanes)
        mbar_wait(&B.accf[buf], (tc >> 1) & 1);
        tc_fence_after();
        epi_range(su, B, BSs, &mY, tmem + ((uint32_t)(q4 * 32) << 16) + buf * TCOLS, rl, m0, buf, 0, 2, rx, cx, rna, 2, warp == 6 && lane == 0, lane);
      } else {
        named_bar_sync(4, 256);                          // warps 6-9 have published x's statistics
        const float2 xs = XSs[rl];
        mbar_wait(&B.accf[buf], (tc >> 1) & 1);
        tc_fence_after();
        epi_range(su, B, BSs, &mY, tmem + ((uint32_t)(q4 * 32) << 16) + buf * TCOLS, rl, m0, buf, 2, 4, xs.x, xs.y, rna, 3, warp == 2 && lane == 0, lane);
      }
    }
    if ((warp == 6 || warp == 2) && lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 1) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
