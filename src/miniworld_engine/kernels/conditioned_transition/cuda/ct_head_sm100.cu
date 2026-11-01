// ct_head_sm100.cu -- the B200 (sm_100a) head of a bf16 ConditionedTransition call without a backward: the module's adaptive LayerNorm
// and the tail's gate projection, which read the same conditioning rows, as ONE tcgen05 GEMM over the RAW cond rows:
//
//   xn = sigmoid(cn Ws^T + bs) * (x rstd_x - mean_x rstd_x) + cn Wb^T,      cn = LN(cond) * wc
//   sp = cond Wg^T + bg
//
// with LN(cond) folded into the weights: cn W^T = rstd_c (cond (W wc)^T) - rstd_c mean_c u,  u[n] = sum_k (W wc)[n, k]  (the host passes
// W wc rounded to bf16 and u summed from those rounded values, so the identity is exact up to the fp32 accumulation). All three
// accumulators -- acc 0 = cond (Wb wc)^T, acc 1 = cond (Ws wc)^T, acc 2 = cond Wg^T -- share the one A operand: cn never exists, and the
// AdaLN row pass, its GEMM and the cuBLAS addmm of sp become this kernel plus ``ct_head_stats`` (ct_rows.cu: the statistics of x and of
// cond per row), of which this launch is the programmatic dependent; only the epilogue waits for it.
//
// The structure is adaln_gemm_sm100.cu's (persistent, warp-specialised: warp 0 TMA producer, warp 1 MMA issuer and TMEM owner, warps 2-9
// the epilogue reading TMEM lanes 32 (w % 4) .., the two warps of a lane quarter splitting each 64-column group and TMA-storing 32-row slabs
// of xn and sp from swizzled staging), with three accumulators: double-buffered in tensor memory where 2 x 3 x BN columns fit (BN 64),
// single-buffered at BN 128 (the epilogue of a tile then precedes the next tile's main loop).
//
// Where xn and sp tiles fit one wave, ct_head_split_sm100.cu (the two outputs on different CTAs) is launched instead. A CTA-pair build
// (cta_group::2, halving each CTA's weight rows) measured slower at these row counts (latency-, not intake-bound). bf16 only (kind::f16). Any M (TMA zero-fills the tail rows and clips the stores), N a multiple of BN, K a multiple of 64.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef BN
#define BN 128
#endif

constexpr int NACC = 3;
constexpr int BKE = 64;                              // bf16 elements per 128-byte K-block row
constexpr int OCG = 64;                              // output columns per 128-byte staging row (bf16)
constexpr int NBUF = 2 * NACC * BN <= 512 ? 2 : 1;   // accumulator buffers in tensor memory
constexpr int NOUT = 2;                              // staged outputs: xn, sp
constexpr int STG_PAR = 1;                           // staging buffers per output (1: room for a deeper operand ring)
constexpr int NOWN = 4;                              // staging owners: the half-0 warp of each lane quarter
constexpr int A_BYTES = 128 * 128, B_BYTES = BN * 128, STAGE = A_BYTES + NACC * B_BYTES;
constexpr int SLAB = 32 * 128;                       // one warp's staging slab: 32 rows x 128 B
constexpr int NEPI = 8;
constexpr int STG_BYTES = NOWN * NOUT * STG_PAR * SLAB;
constexpr int BUDGET = 232448 - 1024;
constexpr int NST_FIT = (BUDGET - STG_BYTES) / STAGE;
constexpr int NST = NST_FIT > 8 ? 8 : NST_FIT;
constexpr int O_ST = 0, O_STG = NST * STAGE, O_BAR = O_STG + STG_BYTES;
constexpr int SMEM_BYTES = O_BAR + 1024;
constexpr int TCOLS = NACC * BN;
static_assert(NST >= 2, "ring depth");
static_assert(NBUF * TCOLS <= 512, "tensor memory");
static_assert(BN % OCG == 0 && BN <= 128, "tile width");
static_assert(STAGE % 1024 == 0 && SLAB % 1024 == 0, "128-B swizzle atoms");

constexpr uint32_t IDESC = idesc_bf16(128, BN);
DEVI void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) { umma_ss(d, a, b, IDESC, acc); }
DEVI float sigm(float a) { return sigmoid_kit(a); }

struct Bars {
  uint64_t full[NST], empty[NST], accf[2], acce[2];
  uint32_t tmem;
};

// 32 values of one row -> 32 columns (unit u: the 64-B half u) of its 128-B staging row
DEVI void stage32(uint32_t slab, uint32_t r, int u, const float (&v)[32]) {
#pragma unroll
  for (int q = 0; q < 4; ++q)
    sts128(slab + sw128(r, u * 4 + q), make_uint4(pack_bf16(v[8 * q], v[8 * q + 1]), pack_bf16(v[8 * q + 2], v[8 * q + 3]),
                                                  pack_bf16(v[8 * q + 4], v[8 * q + 5]), pack_bf16(v[8 * q + 6], v[8 * q + 7])));
}
DEVI void load32(const __nv_bfloat16* p, bool ok, float (&v)[32]) {
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint4 w = ok ? ldg128(p + 8 * q) : make_uint4(0, 0, 0, 0);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) { v[8 * q + 2 * e] = bf16lo(ww[e]); v[8 * q + 2 * e + 1] = bf16hi(ww[e]); }
  }
}

// maps: A cond [M][K] (box 64 x 128); B0 Wb wc, B1 Ws wc, B2 Wg [N][K] (box 64 x BN); O0 xn, O1 sp [M][N] (box 64 x 32).
// X [M][N] bf16; ST [M] float4 (rstd_x, mean_x rstd_x, rstd_c, mean_c rstd_c); BS / BG the AdaLN / CT gate biases, UB / US the
// row sums of the folded Wb / Ws [N] fp32.
extern "C" __global__ void __launch_bounds__(64 + 32 * NEPI, 1)
ct_head_sm100(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB0, const __grid_constant__ CUtensorMap mB1,
              const __grid_constant__ CUtensorMap mB2, const __grid_constant__ CUtensorMap mO0, const __grid_constant__ CUtensorMap mO1,
              const __nv_bfloat16* __restrict__ X, const float4* __restrict__ ST, const float* __restrict__ BS, const float* __restrict__ BG,
              const float* __restrict__ UB, const float* __restrict__ US, int M, int N, int K) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NT = N / BN, tiles = ((M + 127) / 128) * NT;

  if (threadIdx.x == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.accf[s], 1); mbar_init(&B.acce[s], NEPI); }
    fence_barrier_init();
    prefetch_map(&mA); prefetch_map(&mB0); prefetch_map(&mB1); prefetch_map(&mB2); prefetch_map(&mO0); prefetch_map(&mO1);
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ producer
    if (lane == 0) {
      int it = 0;
      const int nk = K / BKE;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        const int m0 = (t / NT) * 128, n0 = (t % NT) * BN;
        for (int k = 0; k < nk; ++k, ++it) {
          const int s = it % NST;
          mbar_wait(&B.empty[s], ((it / NST) & 1) ^ 1);
          mbar_expect_tx(&B.full[s], STAGE);
          const uint32_t base = su + O_ST + s * STAGE;
          tma_load_2d(base, &mA, &B.full[s], k * BKE, m0);
          tma_load_2d(base + A_BYTES, &mB0, &B.full[s], k * BKE, n0);
          tma_load_2d(base + A_BYTES + B_BYTES, &mB1, &B.full[s], k * BKE, n0);
          tma_load_2d(base + A_BYTES + 2 * B_BYTES, &mB2, &B.full[s], k * BKE, n0);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    if (lane == 0) {
      int it = 0, tc = 0;
      const int nk = K / BKE;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
        const int buf = NBUF == 2 ? (tc & 1) : 0;
        const uint32_t use = NBUF == 2 ? (uint32_t)(tc >> 1) : (uint32_t)tc;   // uses of this buffer so far
        mbar_wait(&B.acce[buf], (use & 1) ^ 1);
        tc_fence_after();
        const uint32_t dbase = tmem + buf * TCOLS;
        for (int k = 0; k < nk; ++k, ++it) {
          const int s = it % NST;
          mbar_wait(&B.full[s], (it / NST) & 1);
          tc_fence_after();
          const uint32_t base = su + O_ST + s * STAGE;
          const uint64_t da = desc_k128(base);
#pragma unroll
          for (int j = 0; j < NACC; ++j) {
            const uint64_t db = desc_k128(base + A_BYTES + j * B_BYTES);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              umma(dbase + j * BN, da + (uint64_t)(kk * 2), db + (uint64_t)(kk * 2), (k > 0 || kk > 0) ? 1u : 0u);
          }
          tc_commit(&B.empty[s]);
        }
        tc_commit(&B.accf[buf]);
      }
    }
  } else {
    // ------------------------------------------------------------------------------------------------ epilogue (warps 2-9)
    const int ew = warp - 2, q4 = warp & 3, half = ew >> 2;
    const uint32_t r = (uint32_t)lane;
    const uint32_t stg = su + O_STG + (uint32_t)(q4 * NOUT * STG_PAR) * SLAB;
    const bool storer = half == 0;
    pdl_wait();   // the row statistics come from ct_head_stats, the programmatic primary of this launch (only the epilogue reads them)
    int tc = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
      const int buf = NBUF == 2 ? (tc & 1) : 0;
      const uint32_t use = NBUF == 2 ? (uint32_t)(tc >> 1) : (uint32_t)tc;
      const int m0 = (t / NT) * 128, n0 = (t % NT) * BN;
      const int row = m0 + q4 * 32 + lane;
      const bool rok = row < M;
      const float4 st = rok ? ST[row] : make_float4(0.f, 0.f, 0.f, 0.f);
      mbar_wait(&B.accf[buf], use & 1);
      tc_fence_after();
      const uint32_t tb = tmem + ((uint32_t)(q4 * 32) << 16) + buf * TCOLS;
      for (int g = 0; g < BN / OCG; ++g) {
        const int c0 = n0 + g * OCG;
        if (storer && lane == 0) tma_store_wait_read0();   // the previous group's slabs have been read out
        named_bar_sync(1 + q4, 64);
        {
          const int col = g * OCG + half * 32;               // column within the tile
          const int n = n0 + col;
          float o[32], p[32];
          load32(X + (long)row * N + n, rok, o);           // x, issued before the tensor-memory loads
          uint32_t acc[NACC][32];
#pragma unroll
          for (int j = 0; j < NACC; ++j) tmem_ld32(tb + j * BN + col, acc[j]);
          tmem_wait_ld();
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            const float scale = st.z * __uint_as_float(acc[1][e]) - st.w * __ldg(US + n + e) + __ldg(BS + n + e);
            const float shift = st.z * __uint_as_float(acc[0][e]) - st.w * __ldg(UB + n + e);
            o[e] = sigm(scale) * (o[e] * st.x - st.y) + shift;
            p[e] = __uint_as_float(acc[2][e]) + __ldg(BG + n + e);
          }
          stage32(stg, r, half, o);
          stage32(stg + SLAB, r, half, p);
        }
        fence_proxy_async();
        named_bar_sync(1 + q4, 64);
        if (storer && lane == 0) {
          tma_store_2d(&mO0, stg, c0, m0 + q4 * 32);
          tma_store_2d(&mO1, stg + SLAB, c0, m0 + q4 * 32);
          tma_store_commit();
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.acce[buf]);
    }
    if (storer && lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
