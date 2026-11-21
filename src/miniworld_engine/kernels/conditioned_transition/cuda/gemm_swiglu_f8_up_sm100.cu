// gemm_swiglu_f8_up_sm100.cu -- h = bf16(silu(a) * b * u), [a | b | u] = (sx x8) (W8 sw)^T, sm_100a: Boltz-2's up-gated token DiT transition on e4m3 (agent E,
// 2026-11-19; OPT-IN, token_dit_fp8_b200.py; gemm_swiglu_f8_sm100.cu is untouched).
//
// The 2-CTA structure of gemm_swiglu_f8_sm100.cu with 64-unit hidden chunks: W8 [3 H, D] is the H100 ``_bf16_wab64`` pack, chunk j = rows 192 j .. 192 j + 191 =
// [Wa 64 | Wb 64 | Wu 64]; one item = (tile pair, chunk j) = M 256, N 192, K D e4m3 products (the pair's CTA c holds B rows 96 c .. 96 c + 95 of the chunk:
// a 96-row TMA box, = Wa64 + Wb32 | Wb32 + Wu64). Accumulators [a 64 | b 64 | u 64] (192 columns) double-buffered in tensor memory; buffer s is
// read by epilogue warpgroup s (items q = s, s + 2, ..): the three unit-scale vectors are the pack's row scales (sw[192 j + c], + 64, + 128), the row scale
// sx[row]. h = bf16(silu(a) b u) with a = acc sx sw: one 64-unit tile [128 rows][64] per item staged for TMA stores.
// -DDIM=<D> (multiple of 128) -DHMUL=<H / D>.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DIM
#define DIM 768
#endif
#ifndef HMUL
#define HMUL 2
#endif
constexpr int D_ = DIM, H_ = HMUL * DIM, HC = 64, NCH = H_ / HC, ROWS = 128, NKB = D_ / 128;
static_assert(D_ % 128 == 0, "K blocks of 128 e4m3 elements");
constexpr int NST = 5;
constexpr int KBT = ROWS * 128;
constexpr int S_XN = 0, S_W = KBT, STAGE = KBT + 96 * 128;               // 28 KB: x [128][128 B] | B [96][128 B]
static_assert(STAGE % 1024 == 0, "stage alignment");
constexpr int O_ST = 0, O_OUT = NST * STAGE;
constexpr int O_BAR = O_OUT + 2 * KBT;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t IDESC = idesc_e4m3(256, 192);

#if defined(SIG_TANH)
#define SIGF(a) fmaf(0.5f, tanhf_approx(0.5f * (a)), 0.5f)
#else
#define SIGF(a) sigmoid_kit(a)
#endif

struct Bars {
  uint64_t full[NST];
  uint64_t empty[NST], acc_full[2];
  uint64_t acc_empty[2];
  uint64_t staged[2], stage_free[2];
  uint32_t tmem;
};

// maps: xn [M][D] u8 (box 128 x 64); w [3 H][D] u8 (box 128 x 96); h [M][H] bf16 (box 64 x 64)
extern "C" __global__ void __launch_bounds__(512, 1)
gemm_swiglu_f8_up_sm100(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mh,
                        const float* __restrict__ sx, const float* __restrict__ sw, int tiles) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  const int npairs = G >> 1, pair = cta >> 1, ntp = (tiles + 1) >> 1, ntot = ntp * NCH;
  const int nitem = (ntot > pair) ? (ntot - pair + npairs - 1) / npairs : 0;
  auto item_j = [&](int q) { return (pair + q * npairs) % NCH; };
  auto item_tile = [&](int q) { return 2 * ((pair + q * npairs) / NCH) + crank; };
  auto item_valid = [&](int q) { return item_tile(q) < tiles; };
  auto item_row = [&](int q) { const int t = item_tile(q); return (t < tiles ? t : tiles - 1) * ROWS; };

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.acc_full[s], 1); mbar_init(&B.acc_empty[s], 8); }   // 4 epilogue warps a CTA x 2 CTAs
    for (int g = 0; g < 2; ++g) { mbar_init(&B.staged[g], 4); mbar_init(&B.stage_free[g], 1); }
    fence_barrier_init();
    prefetch_map(&mxn); prefetch_map(&mw); prefetch_map(&mh);
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
      int st = 0;
      for (int q = 0; q < nitem; ++q) {
        const int j = item_j(q), row = item_row(q);
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int s = st % NST;
          if (st >= NST) mbar_wait(&B.empty[s], ((st / NST) - 1) & 1);
          if (leader) mbar_expect_tx(&B.full[s], 2 * STAGE);
          const uint32_t base = su + O_ST + s * STAGE;
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_load_2d_2sm(base + S_XN + h * 8192, &mxn, &B.full[s], kb * 128, row + h * 64);
          tma_load_2d_2sm(base + S_W, &mw, &B.full[s], kb * 128, j * 192 + 96 * crank);
        }
      }
    }
  } else if (warp == 1) {
    if (leader) {
      int st = 0;
      for (int q = 0; q < nitem; ++q) {
        const int s = q & 1, u = q >> 1;
        if (q >= 2) mbar_wait_cl(&B.acc_empty[s], (u - 1) & 1);
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int sg = st % NST;
          mbar_wait(&B.full[sg], (st / NST) & 1);
          tc_fence_after();
          const uint32_t base = su + O_ST + sg * STAGE;
          const uint64_t dxn = desc_k128(base + S_XN), dw = desc_k128(base + S_W);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma8_ss2(tmem + s * 256, dxn + (uint64_t)(ks * 2), dw + (uint64_t)(ks * 2), IDESC, (kb > 0 || ks > 0) ? 1u : 0u);
            tc_commit2_mc(&B.empty[sg], 3);
            if (kb == NKB - 1) tc_commit2_mc(&B.acc_full[s], 3);
          }
          __syncwarp();
        }
      }
    }
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    setmaxnreg_inc<152>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int half = warp >= 12 ? 1 : 0;                                   // epilogue group = TMEM buffer = parity of the item
    for (int q = half; q < nitem; q += 2) {
      const int s = half, u = q >> 1;
      mbar_wait(&B.acc_full[s], u & 1);
      tc_fence_after();
      const float sxr = __ldg(sx + item_row(q) + r);
      const float* swj = sw + (size_t)item_j(q) * 192;
      uint32_t hp[32];
#pragma unroll
      for (int st2 = 0; st2 < 2; ++st2) {
        uint32_t av[32], bv[32];
        tmem_ld32(trow + s * 256 + st2 * 32, av);
        tmem_ld32(trow + s * 256 + 64 + st2 * 32, bv);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float2 wa = __ldg(reinterpret_cast<const float2*>(swj + st2 * 32) + k);
          const float2 wb = __ldg(reinterpret_cast<const float2*>(swj + 64 + st2 * 32) + k);
          const float a0 = __uint_as_float(av[2 * k]) * (sxr * wa.x), a1 = __uint_as_float(av[2 * k + 1]) * (sxr * wa.y);
          const float b0 = __uint_as_float(bv[2 * k]) * (sxr * wb.x), b1 = __uint_as_float(bv[2 * k + 1]) * (sxr * wb.y);
          av[2 * k] = __float_as_uint(a0 * SIGF(a0) * b0);
          av[2 * k + 1] = __float_as_uint(a1 * SIGF(a1) * b1);
        }
        tmem_ld32(trow + s * 256 + 128 + st2 * 32, bv);                   // u (bv's registers are free)
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float2 wu = __ldg(reinterpret_cast<const float2*>(swj + 128 + st2 * 32) + k);
          hp[st2 * 16 + k] = pack_bf16(__uint_as_float(av[2 * k]) * (__uint_as_float(bv[2 * k]) * (sxr * wu.x)),
                                       __uint_as_float(av[2 * k + 1]) * (__uint_as_float(bv[2 * k + 1]) * (sxr * wu.y)));
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_relaxed(&B.acc_empty[s], 0);
      if (u >= 1) mbar_wait(&B.stage_free[half], (u - 1) & 1);
#pragma unroll
      for (int qq = 0; qq < 8; ++qq)
        sts128(su + O_OUT + half * KBT + sw128(r, qq), make_uint4(hp[4 * qq], hp[4 * qq + 1], hp[4 * qq + 2], hp[4 * qq + 3]));
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.staged[half]);
    }
  } else if (warp == 8 || warp == 9) {
    if (lane == 0) {
      const int t = warp - 8;
      for (int q = t; q < nitem; q += 2) {
        const int u = q >> 1, j = item_j(q);
        mbar_wait(&B.staged[t], u & 1);
        if (item_valid(q)) {
          const int row = item_row(q);
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_store_2d(&mh, su + O_OUT + t * KBT + h * 8192, j * HC, row + h * 64);
          tma_store_commit();
          tma_store_wait_read0();
        }
        mbar_arrive(&B.stage_free[t]);
      }
      tma_store_wait0();
    }
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}
