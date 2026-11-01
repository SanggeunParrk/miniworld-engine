// ct_head_split_sm100.cu -- ct_head_sm100.cu's head (the AdaLN of x by cond and the gate projection sp = cond Wg^T + bg, LN(cond) folded
// into the AdaLN weights) with its two outputs on DIFFERENT CTAs, for the row counts where both fit one wave:
//
//   CTAs [0, XT)        xn tiles, 128 rows x 128 columns: acc 0 = cond (Wb wc)^T, acc 1 = cond (Ws wc)^T (two accumulators), the AdaLN
//                       epilogue (x and cond row statistics from ct_head_stats, the programmatic primary: only these CTAs wait for it)
//   CTAs [XT, XT + GT)  sp tiles, 128 rows x 256 columns: acc = cond Wg^T (one accumulator, N = 256), sp = acc + bg
//
// Every CTA streams the same bytes per K-block (128 cond rows + 256 weight rows) and holds 256 tensor-memory columns, so the grid is
// balanced; at a few row tiles the three-accumulator tile of ct_head_sm100 left SMs idle while each CTA brought 128 + 384 weight rows.
// One tile per CTA (grid = XT + GT), warp 0 TMA, warp 1 MMA + TMEM, warps 2-9 the epilogue (the two warps of a lane quarter split each
// 64-column group, TMA-storing 32-row slabs from swizzled staging). bf16 only. Any M; N a multiple of 256; K a multiple of 64.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int BKE = 64;                              // bf16 elements per 128-byte K-block row
constexpr int OCG = 64;                              // output columns per 128-byte staging row
constexpr int BNX = 128, BNG = 256;                  // output tile widths of the xn and sp roles
constexpr int A_BYTES = 128 * 128, B_BYTES = 256 * 128, STAGE = A_BYTES + B_BYTES;   // xn: [Wb wc | Ws wc] (2 x 128 rows), sp: Wg (256)
constexpr int SLAB = 32 * 128;
constexpr int NEPI = 8;
constexpr int STG_PAR = 2;                           // staging buffers per owner (one output a role)
constexpr int STG_BYTES = 4 * STG_PAR * SLAB;
constexpr int BUDGET = 232448 - 1024;
constexpr int NST_FIT = (BUDGET - STG_BYTES) / STAGE;
constexpr int NST = NST_FIT > 8 ? 8 : NST_FIT;
constexpr int O_ST = 0, O_STG = NST * STAGE, O_BAR = O_STG + STG_BYTES;
constexpr int SMEM_BYTES = O_BAR + 1024;
static_assert(NST >= 2, "ring depth");
static_assert(STAGE % 1024 == 0 && SLAB % 1024 == 0, "128-B swizzle atoms");

constexpr uint32_t IDESC_X = idesc_bf16(128, BNX), IDESC_G = idesc_bf16(128, BNG);
DEVI float sigm(float a) { return sigmoid_kit(a); }
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }

struct Bars {
  uint64_t full[NST], empty[NST], accf;
  uint32_t tmem;
};

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

// maps: A cond [M][K] (box 64 x 128); B0 Wb wc, B1 Ws wc [N][K] (box 64 x 128); BG Wg [N][K] (box 64 x 256); O0 xn, O1 sp [M][N] (box 64 x 32).
// X [M][N] bf16; ST [M] float4 (rstd_x, mean_x rstd_x, rstd_c, mean_c rstd_c); BS / BG the AdaLN / CT gate biases, UB / US the row sums of
// the folded Wb / Ws [N] fp32.
extern "C" __global__ void __launch_bounds__(64 + 32 * NEPI, 1)
ct_head_split_sm100(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB0, const __grid_constant__ CUtensorMap mB1,
                    const __grid_constant__ CUtensorMap mBG, const __grid_constant__ CUtensorMap mO0, const __grid_constant__ CUtensorMap mO1,
                    const __nv_bfloat16* __restrict__ X, const float4* __restrict__ ST, const float* __restrict__ BS, const float* __restrict__ BG,
                    const float* __restrict__ UB, const float* __restrict__ US, int M, int N, int K) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int MT = (M + 127) / 128, XT = MT * (N / BNX);
  const bool gate = (int)blockIdx.x >= XT;           // this CTA's role
  const int t = gate ? (int)blockIdx.x - XT : (int)blockIdx.x;
  const int NTR = gate ? N / BNG : N / BNX;
  const int m0 = (t / NTR) * 128, n0 = (t % NTR) * (gate ? BNG : BNX);

  if (threadIdx.x == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.accf, 1);
    fence_barrier_init();
    prefetch_map(&mA);
    if (gate) { prefetch_map(&mBG); prefetch_map(&mO1); } else { prefetch_map(&mB0); prefetch_map(&mB1); prefetch_map(&mO0); }
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  const int nk = K / BKE;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ producer
    if (lane == 0) {
      for (int k = 0; k < nk; ++k) {
        const int s = k % NST;
        mbar_wait(&B.empty[s], ((k / NST) & 1) ^ 1);
        mbar_expect_tx(&B.full[s], STAGE);
        const uint32_t base = su + O_ST + s * STAGE;
        tma_load_2d(base, &mA, &B.full[s], k * BKE, m0);
        if (gate) {
          tma_load_2d(base + A_BYTES, &mBG, &B.full[s], k * BKE, n0);
        } else {
          tma_load_2d(base + A_BYTES, &mB0, &B.full[s], k * BKE, n0);
          tma_load_2d(base + A_BYTES + B_BYTES / 2, &mB1, &B.full[s], k * BKE, n0);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    if (lane == 0) {
      for (int k = 0; k < nk; ++k) {
        const int s = k % NST;
        mbar_wait(&B.full[s], (k / NST) & 1);
        tc_fence_after();
        const uint32_t base = su + O_ST + s * STAGE;
        const uint64_t da = desc_k128(base);
        if (gate) {
          const uint64_t db = desc_k128(base + A_BYTES);
#pragma unroll
          for (int kk = 0; kk < 4; ++kk)
            umma_ss(tmem, da + (uint64_t)(kk * 2), db + (uint64_t)(kk * 2), IDESC_G, (k > 0 || kk > 0) ? 1u : 0u);
        } else {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const uint64_t db = desc_k128(base + A_BYTES + j * (B_BYTES / 2));
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              umma_ss(tmem + j * BNX, da + (uint64_t)(kk * 2), db + (uint64_t)(kk * 2), IDESC_X, (k > 0 || kk > 0) ? 1u : 0u);
          }
        }
        tc_commit(&B.empty[s]);
      }
      tc_commit(&B.accf);
    }
  } else {
    // ------------------------------------------------------------------------------------------------ epilogue (warps 2-9)
    const int ew = warp - 2, q4 = warp & 3, half = ew >> 2;
    const uint32_t r = (uint32_t)lane;
    const uint32_t stg = su + O_STG + (uint32_t)(q4 * STG_PAR) * SLAB;
    const bool storer = half == 0;
    const int row = m0 + q4 * 32 + lane;
    const bool rok = row < M;
    float4 st = make_float4(0.f, 0.f, 0.f, 0.f);
    if (!gate) {
      pdl_wait();   // the row statistics come from ct_head_stats, the programmatic primary of this launch (only the xn role reads them)
      if (rok) st = ST[row];
    }
    mbar_wait(&B.accf, 0);
    tc_fence_after();
    const uint32_t tb = tmem + ((uint32_t)(q4 * 32) << 16);
    const int ngrp = (gate ? BNG : BNX) / OCG;
    for (int g = 0; g < ngrp; ++g) {
      const int c0 = n0 + g * OCG;
      const int par = g & 1;
      if (storer && lane == 0) tma_store_wait_read1();
      named_bar_sync(1 + q4, 64);
      {
        const int col = g * OCG + half * 32;               // column within the tile
        const int n = n0 + col;
        float o[32];
        if (gate) {
          uint32_t acc[32];
          tmem_ld32(tb + col, acc);
          tmem_wait_ld();
#pragma unroll
          for (int e = 0; e < 32; ++e) o[e] = __uint_as_float(acc[e]) + __ldg(BG + n + e);
        } else {
          load32(X + (long)row * N + n, rok, o);
          uint32_t acc[2][32];
          tmem_ld32(tb + col, acc[0]);
          tmem_ld32(tb + BNX + col, acc[1]);
          tmem_wait_ld();
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            const float scale = st.z * __uint_as_float(acc[1][e]) - st.w * __ldg(US + n + e) + __ldg(BS + n + e);
            const float shift = st.z * __uint_as_float(acc[0][e]) - st.w * __ldg(UB + n + e);
            o[e] = sigm(scale) * (o[e] * st.x - st.y) + shift;
          }
        }
        stage32(stg + (uint32_t)par * SLAB, r, half, o);
      }
      fence_proxy_async();
      named_bar_sync(1 + q4, 64);
      if (storer && lane == 0) {
        tma_store_2d(gate ? &mO1 : &mO0, stg + (uint32_t)par * SLAB, c0, m0 + q4 * 32);
        tma_store_commit();
      }
    }
    if (storer && lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 256); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
