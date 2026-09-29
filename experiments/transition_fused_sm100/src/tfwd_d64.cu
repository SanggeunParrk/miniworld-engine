// tfwd_d64.cu — the Transition forward at D = 64, H = 4D = 256, bf16, as ONE fused sm_100a kernel with 2-CTA tcgen05.mma: tfwd2.cu's
// schedule (the two CTAs of a cluster run their own 128-row tiles in lockstep; the leader issues M = 256 products whose B operand is split
// by N across the pair) with what D = 64 changes:
//   * the weights are RESIDENT: this CTA's half of all four 64-unit chunks is 48 KB (Wa_j or Wb_j: 4 x 8 KB; Ws rows d = 32 crank ..
//     32 crank + 31: 4 x 4 KB), loaded once per launch -- no weight ring
//   * expand   [a|b]_j = xn [Wa_j; Wb_j]^T   M 256, N 128, K 64 (one 128-B-swizzled K-block)
//   * squeeze  acc    += h_j Ws_j^T          M 256, N  64 (32 per CTA), K 64, A = h_j from tensor memory
// Rounding points as tfwd2.cu: xn bf16, a / b fp32, h = bf16(silu(a) b) (kit sigmoid), acc fp32, out = bf16(x + acc).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int D_ = 64, H_ = 256, HS = 64, NCH = H_ / HS, ROWS = 128;
constexpr int WAB_CH = 8192, WS_CH = 4096;                     // this CTA's half of one chunk: [64 n][64 k]; Ws [32 d][64 k]
constexpr int O_WAB = 0, O_WS = NCH * WAB_CH;
constexpr int TILE = ROWS * D_ * 2;                            // 16 KB: one K-block of 128 rows
constexpr int O_X = O_WS + NCH * WS_CH;
constexpr int O_XN = O_X + 2 * TILE;
constexpr int O_GB = O_XN + 2 * TILE;
constexpr int O_BAR = O_GB + 512;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
static_assert(SMEM_BYTES == 115712, "bench_w.py SMEM[64]");
static_assert(O_X % 1024 == 0 && O_XN % 1024 == 0 && O_WS % 1024 == 0, "128-B swizzled operands need 1 KB alignment");
constexpr int KS = 4;                                          // K = 64 in K16 steps (expand: D; squeeze: the chunk's 64 hidden units)
constexpr uint32_t T_AB = 0, T_H = 256, T_OUT = 384;
// sigmoid of the SwiGLU: the kit form (ex2 + rcp on MUFU) by default; SIG_NR takes the reciprocal on the FMA pipe (Newton, error below
// rcp.approx's), SIG_POLY the exponential (degree-6 polynomial, error below ex2.approx's), SIG_TANH 0.5 tanh(a / 2) + 0.5 (one MUFU op;
// relaxed: tanh.approx relative error ~2^-11)
#if defined(SIG_NR)
#define SIGF(a) sigmoid_nr(a)
#elif defined(SIG_POLY)
#define SIGF(a) sigmoid_poly(a)
#elif defined(SIG_TANH)
#define SIGF(a) fmaf(0.5f, tanhf_approx(0.5f * (a)), 0.5f)
#else
#define SIGF(a) sigmoid_kit(a)
#endif
constexpr uint32_t IDESC_EX = idesc_bf16(256, 128), IDESC_SQ = idesc_bf16(256, 64);
#ifdef TRACE
// single-tile latency probe: g_t64 = clock64 stamps of CTA 0 (see the TR() sites), g_gt = globaltimer entry / exit of every CTA
__device__ unsigned long long g_t64[2][32], g_gt[512][2];
#define TR(k) do { if (blockIdx.x < 2) g_t64[blockIdx.x][k] = clock64(); } while (0)
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#else
#define TR(k) do { } while (0)
#endif

struct Bars {
  // leader-only: w_full (both CTAs' weight transactions), xn_full (2 LN arrivals), ab_empty / h_full / out_empty (4 warps per CTA)
  uint64_t w_full, xn_full[2], ab_empty[2], h_full[2], out_empty;
  // both CTAs (leader's multicast commits)
  uint64_t xn_empty[2], ex_done[2], sq_done[2], out_full;
  // local
  uint64_t x_full[2], x_empty[2];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(512, 1)
transition_fwd_d64_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mwa,
                         const __grid_constant__ CUtensorMap mwb, const __grid_constant__ CUtensorMap mws,
                         const __grid_constant__ CUtensorMap mout, const __grid_constant__ CUtensorMap mxn,
                         const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ rstd,
                         float* __restrict__ c1, int tiles, float eps, int save) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
#ifdef TRACE
  if (tid == 0) { g_gt[blockIdx.x][0] = gtimer(); TR(0); }
#endif
  auto count = [&](int k) { return (tiles > k) ? (tiles - k + G - 1) / G : 0; };
  const int n_valid = count(cta), n_local = count(cta & ~1);   // the pair walks the leader's (larger) count; a short CTA repeats
  const int nch = n_local * NCH;                                // its last tile as a dummy whose results are not stored
  auto tile_of = [&](int i) { return i < n_valid ? cta + i * G : (n_valid > 0 ? cta + (n_valid - 1) * G : 0); };

  if (tid == 0) {
    mbar_init(&B.w_full, 1);
    for (int s = 0; s < 2; ++s) {
#ifdef SWIGLU_SPLIT
      mbar_init(&B.xn_full[s], 2); mbar_init(&B.ab_empty[s], 16); mbar_init(&B.h_full[s], 16);   // both warpgroups on every chunk
#else
      mbar_init(&B.xn_full[s], 2); mbar_init(&B.ab_empty[s], 8); mbar_init(&B.h_full[s], 8);
#endif
      mbar_init(&B.xn_empty[s], 1); mbar_init(&B.ex_done[s], 1); mbar_init(&B.sq_done[s], 1);
      mbar_init(&B.x_full[s], 1); mbar_init(&B.x_empty[s], 1);
    }
    mbar_init(&B.out_empty, 8); mbar_init(&B.out_full, 1);
    fence_barrier_init();
    TR(27);
#ifndef OLD_SYNC
    if (n_local > 0) {                                         // tile 0's x now, not after the cluster sync (its barrier is local)
      fence_proxy_async();
      mbar_expect_tx(&B.x_full[0], TILE);
#pragma unroll
      for (int h = 0; h < 2; ++h) tma_load_2d(su + O_X + h * 8192, &mx, &B.x_full[0], 0, tile_of(0) * ROWS + h * 64);
    }
#endif
    prefetch_map(&mx); prefetch_map(&mwa); prefetch_map(&mwb); prefetch_map(&mws); prefetch_map(&mout); prefetch_map(&mxn);
  }
#ifdef OLD_SYNC
  if (tid < 64) { reinterpret_cast<float*>(sm + O_GB)[tid] = gamma[tid]; reinterpret_cast<float*>(sm + O_GB)[64 + tid] = beta[tid]; }
#endif
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  if (tid == 0) TR(25);
#ifdef OLD_SYNC
  cluster_sync();
#else
  cluster_sync_relaxed();
#endif
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (tid == 0) TR(1);

  if (warp < 4) setmaxnreg_dec<56>();
  if (tid == 0) TR(28);
  if (warp == 0) {
    // ------------------------------------------------------------------------------------------ x producer (own tiles)
    if (lane == 0) {
#ifdef OLD_SYNC
      for (int i = 0; i < n_local; ++i) {
#else
      for (int i = 1; i < n_local; ++i) {                     // tile 0 was issued at setup
#endif
        const int b = i & 1, row = tile_of(i) * ROWS;
        if (i >= 2) mbar_wait(&B.x_empty[b], ((i >> 1) - 1) & 1);
        mbar_expect_tx(&B.x_full[b], TILE);
        const uint32_t dst = su + O_X + b * TILE;
#pragma unroll
        for (int h = 0; h < 2; ++h) tma_load_2d(dst + h * 8192, &mx, &B.x_full[b], 0, row + h * 64);
        if (i == 0) TR(2);
      }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------ weights, once: this CTA's half of every
    // chunk; the transactions of both CTAs complete on the leader's barrier, which the leader armed with the pair's total
    if (lane == 0 && n_local > 0) {
      if (leader) mbar_expect_tx(&B.w_full, 2 * (NCH * (WAB_CH + WS_CH)));
      const CUtensorMap* mab = leader ? &mwa : &mwb;
#pragma unroll
      for (int j = 0; j < NCH; ++j) {
        tma_load_2d_2sm(su + O_WAB + j * WAB_CH, mab, &B.w_full, 0, j * HS);
        tma_load_2d_2sm(su + O_WS + j * WS_CH, &mws, &B.w_full, j * HS, crank * 32);
      }
      TR(3);
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------ MMA issue: leader only; warp 1 expands,
    // warp 2 squeezes (converged warps, one elected lane issues)
    if (leader && n_local > 0) {
      mbar_wait(&B.w_full, 0);
      if (warp == 1 && lane == 0) TR(6);
      if (warp == 1) {
        for (int c = 0; c < nch; ++c) {
          const int i = c >> 2, j = c & (NCH - 1), s = c & 1, u = c >> 1;
          if (j == 0) mbar_wait_cl(&B.xn_full[i & 1], (i >> 1) & 1);
          if (c == 0 && lane == 0) TR(7);
          if (c >= 2) mbar_wait_cl(&B.ab_empty[s], (u - 1) & 1);
          tc_fence_after();
          const uint64_t ad = desc_k128(su + O_XN + (i & 1) * TILE), bd = desc_k128(su + O_WAB + j * WAB_CH);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < KS; ++ks)
              umma_ss2(tmem + T_AB + s * 128, ad + (uint64_t)(ks * 2), bd + (uint64_t)(ks * 2), IDESC_EX, ks > 0 ? 1u : 0u);
            tc_commit2_mc(&B.ex_done[s], 3);
            if (j == NCH - 1) tc_commit2_mc(&B.xn_empty[i & 1], 3);
            if (c == NCH - 1) TR(8);
          }
          __syncwarp();
        }
      } else {
        for (int q = 0; q < nch; ++q) {
          const int i = q >> 2, j = q & (NCH - 1), s = q & 1, u = q >> 1;
          mbar_wait_cl(&B.h_full[s], u & 1);
          if (j == 0 && i >= 1) mbar_wait_cl(&B.out_empty, (i - 1) & 1);
          tc_fence_after();
          const uint64_t bd = desc_k128(su + O_WS + j * WS_CH);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < KS; ++ks)
              umma_ts2(tmem + T_OUT, tmem + T_H + s * 32 + ks * 8, bd + (uint64_t)(ks * 2), IDESC_SQ, (j > 0 || ks > 0) ? 1u : 0u);
            tc_commit2_mc(&B.sq_done[s], 3);
            if (j == NCH - 1) tc_commit2_mc(&B.out_full, 3);
            if (q == NCH - 1) TR(11);
          }
          __syncwarp();
        }
      }
    }
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    // ------------------------------------------------------------------------------------------ SwiGLU (as tfwd2.cu): two warpgroups
    // alternate chunks, each in four 16-column steps with the next step's TMEM loads in flight
    setmaxnreg_inc<152>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, trow = tmem + (lb << 16);
#ifdef SWIGLU_SPLIT
    // small-L variant: both warpgroups on every chunk, warpgroup g taking units 32 g .. 32 g + 31 (half the per-chunk latency)
    const int g = warp >= 12 ? 1 : 0;
    for (int c = 0; c < nch; ++c) {
      const int s = c & 1, u = c >> 1;
      mbar_wait(&B.ex_done[s], u & 1);
      tc_fence_after();
      uint32_t hp[16];
      {
        uint32_t a[32], b[32];
        tmem_ld32(trow + T_AB + s * 128 + g * 32, a);
        tmem_ld32(trow + T_AB + s * 128 + 64 + g * 32, b);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float a0 = __uint_as_float(a[2 * k]), a1 = __uint_as_float(a[2 * k + 1]);
          const float b0 = __uint_as_float(b[2 * k]), b1 = __uint_as_float(b[2 * k + 1]);
          hp[k] = pack_bf16(a0 * SIGF(a0) * b0, a1 * SIGF(a1) * b1);
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_relaxed(&B.ab_empty[s], 0);
      if (c >= 2) mbar_wait(&B.sq_done[s], (u - 1) & 1);
      tc_fence_after();
      tmem_st16(trow + T_H + s * 32 + g * 16, hp);
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_relaxed(&B.h_full[s], 0);
    }
#else
    for (int c = (warp >= 12 ? 1 : 0); c < nch; c += 2) {
      const int s = c & 1, u = c >> 1;
      mbar_wait(&B.ex_done[s], u & 1);
      if (tid == 128 && c < 4) TR(16 + c);
      tc_fence_after();
      uint32_t hp[32];
      {
        uint32_t a[2][16], b[2][16];
        tmem_ld16(trow + T_AB + s * 128, a[0]);
        tmem_ld16(trow + T_AB + s * 128 + 64, b[0]);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          if (q < 3) {
            tmem_ld16(trow + T_AB + s * 128 + (q + 1) * 16, a[(q + 1) & 1]);
            tmem_ld16(trow + T_AB + s * 128 + 64 + (q + 1) * 16, b[(q + 1) & 1]);
          }
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const float a0 = __uint_as_float(a[q & 1][2 * k]), a1 = __uint_as_float(a[q & 1][2 * k + 1]);
            const float b0 = __uint_as_float(b[q & 1][2 * k]), b1 = __uint_as_float(b[q & 1][2 * k + 1]);
            #ifdef SIG_MIX
            hp[q * 8 + k] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_nr(a1) * b1);   // half the reciprocals on the FMA pipe
#else
            hp[q * 8 + k] = pack_bf16(a0 * SIGF(a0) * b0, a1 * SIGF(a1) * b1);
#endif
          }
          if (q < 3) tmem_wait_ld();
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_relaxed(&B.ab_empty[s], 0);
      if (c >= 2) mbar_wait(&B.sq_done[s], (u - 1) & 1);
      tc_fence_after();
      uint32_t h0[16], h1[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) { h0[k] = hp[k]; h1[k] = hp[16 + k]; }
      tmem_st16(trow + T_H + s * 32, h0);
      tmem_st16(trow + T_H + s * 32 + 16, h1);
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_relaxed(&B.h_full[s], 0);
      if (tid == 128 && c < 4) TR(20 + c);
    }
#endif
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------------ LayerNorm + residual epilogue (own rows)
    setmaxnreg_inc<152>();
    const int t2 = tid - 256;
    if (t2 == 0) TR(29);
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t gb_u = su + O_GB;
#ifndef OLD_SYNC
    // gamma / beta by the LayerNorm warps themselves, under tile 0's x load (ln's first named barrier publishes them)
    if (t2 < 64) { reinterpret_cast<float*>(sm + O_GB)[t2] = gamma[t2]; reinterpret_cast<float*>(sm + O_GB)[64 + t2] = beta[t2]; }
    if (t2 == 0) TR(30);
#endif
    auto ln = [&](int i) {
      const int b = i & 1, grow = tile_of(i) * ROWS + (int)r;
      const bool real = i < n_valid;
      mbar_wait(&B.x_full[b], (i >> 1) & 1);
      if (i == 0 && t2 == 0) TR(4);
      if (i >= 2) mbar_wait(&B.xn_empty[b], ((i >> 1) - 1) & 1);
      if (t2 == 0) tma_store_wait_read0();
      named_bar_sync(1, 128);
      const uint32_t xb = su + O_X + b * TILE, xnb = su + O_XN + b * TILE;
      uint32_t v[32];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint4 t = lds128(xb + sw128(r, q));
        v[q * 4 + 0] = t.x; v[q * 4 + 1] = t.y; v[q * 4 + 2] = t.z; v[q * 4 + 3] = t.w;
      }
      float p[16];                                             // pairwise tree over 16 partials of four columns
#pragma unroll
      for (int l = 0; l < 16; ++l) p[l] = (bf16lo(v[2 * l]) + bf16hi(v[2 * l])) + (bf16lo(v[2 * l + 1]) + bf16hi(v[2 * l + 1]));
#pragma unroll
      for (int k = 8; k; k >>= 1)
#pragma unroll
        for (int l = 0; l < k; ++l) p[l] = p[l] + p[l + k];
      const float mean = p[0] * (1.f / D_);
#pragma unroll
      for (int l = 0; l < 16; ++l) {
        float acc = 0.f, d;
        d = bf16lo(v[2 * l]) - mean; acc += d * d;
        d = bf16hi(v[2 * l]) - mean; acc += d * d;
        d = bf16lo(v[2 * l + 1]) - mean; acc += d * d;
        d = bf16hi(v[2 * l + 1]) - mean; acc += d * d;
        p[l] = acc;
      }
#pragma unroll
      for (int k = 8; k; k >>= 1)
#pragma unroll
        for (int l = 0; l < k; ++l) p[l] = p[l] + p[l + k];
      const float rs = rsqrtf(p[0] * (1.f / D_) + eps);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int col = q * 8 + 2 * k;
          const uint32_t w = v[q * 4 + k];
          const float2 g2 = lds64f(gb_u + col * 4), b2 = lds64f(gb_u + 256 + col * 4);
          o[k] = pack_bf16((bf16lo(w) - mean) * rs * g2.x + b2.x, (bf16hi(w) - mean) * rs * g2.y + b2.y);
        }
        sts128(xnb + sw128(r, q), make_uint4(o[0], o[1], o[2], o[3]));
      }
      if (save && real) { rstd[grow] = rs; c1[grow] = mean * rs; }
      fence_proxy_async();
      named_bar_sync(1, 128);
      if (t2 == 0) {
        if (i == 0) TR(5);
#ifdef OLD_SYNC
        mbar_arrive_remote(&B.xn_full[b], 0);
#else
        mbar_arrive_remote_cta(&B.xn_full[b], 0);
#endif
        if (i == 0) TR(9);
        if (save && real) {
          const int row0 = tile_of(i) * ROWS;
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_store_2d(&mxn, xnb + h * 8192, 0, row0 + h * 64);
          tma_store_commit();
        }
      }
    };
    auto epi = [&](int i) {
      const int b = i & 1;
      mbar_wait(&B.out_full, i & 1);
      if (i == 0 && t2 == 0) TR(12);
      tc_fence_after();
      const uint32_t xb = su + O_X + b * TILE;
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        uint32_t acc[32];
        tmem_ld32(trow + T_OUT + cc * 32, acc);
        tmem_wait_ld();
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const uint32_t ad = xb + sw128(r, cc * 4 + qq);
          const uint4 xv = lds128(ad);
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
          uint32_t o[4];
#pragma unroll
          for (int k = 0; k < 4; ++k)
            o[k] = pack_bf16(bf16lo(xw[k]) + __uint_as_float(acc[qq * 8 + 2 * k]), bf16hi(xw[k]) + __uint_as_float(acc[qq * 8 + 2 * k + 1]));
          sts128(ad, make_uint4(o[0], o[1], o[2], o[3]));
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_relaxed(&B.out_empty, 0);
      fence_proxy_async();
      named_bar_sync(1, 128);
      if (t2 == 0) {
        const int row0 = tile_of(i) * ROWS;
        if (i < n_valid) {
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_store_2d(&mout, xb + h * 8192, 0, row0 + h * 64);
        }
        if (i == 0) TR(13);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.x_empty[b]);
      }
    };
    if (n_local > 0) ln(0);
    for (int i = 0; i < n_local; ++i) {
      if (i + 1 < n_local) ln(i + 1);
      epi(i);
    }
    if (t2 == 0) { tma_store_wait0(); TR(14); }
  }
  tc_fence_before();
  __syncthreads();
  if (tid == 64) TR(10);
#ifdef OLD_SYNC
  cluster_sync();
#else
  cluster_sync_relaxed();
#endif
  if (tid == 64) TR(24);
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
#ifdef TRACE
  if (tid == 64) { TR(15); g_gt[blockIdx.x][1] = gtimer(); }
#endif
}
