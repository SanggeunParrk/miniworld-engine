// tfwd2.cu — the Transition forward of tfwd.cu (same fusion, same arithmetic and rounding points) with 2-CTA tcgen05.mma: the two
// CTAs of a cluster run their own 128-row tiles in lockstep, and the leader (rank 0) issues M = 256 products whose B operand is split
// by N across the pair. Per chunk j:
//   expand   [a|b] = xn [Wa_j; Wb_j]^T   N = 128: the leader holds Wa_j (N rows 0..63), the peer Wb_j (64..127) -- 16 KB each
//   squeeze  acc  += h_j Ws_j^T          N = 128: the leader holds Ws_j rows d 0..63, the peer d 64..127      -- 8 KB each
// so each SM streams and reads half of the weights, and one MMA instruction covers both SMs. Each CTA keeps its own accumulators in
// its own tensor memory (same columns as tfwd.cu) and runs its own SwiGLU, LayerNorm and epilogue; their completions are counted on
// the leader's barriers (remote arrives), and the leader's commits arrive on both CTAs' barriers (multicast).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128;
constexpr int NW = 4;                                          // weight ring depth (both rings)
constexpr int WAB_SLOT = 16384, WS_SLOT = 8192;                // this CTA's half: [64 n][128 k] as 2 K-blocks of 8 KB; Ws [64 d][64 k]
constexpr int O_WAB = 0, O_WS = NW * WAB_SLOT;
constexpr int O_X = O_WS + NW * WS_SLOT, TILE = 32768;
constexpr int O_XN = O_X + 2 * TILE;
constexpr int O_GB = O_XN + 2 * TILE;
constexpr int O_BAR = O_GB + 1024;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
#ifdef FP8SIM
constexpr int KS_EX = 4, KS_SQ = 2;                            // timing model of fp8 operands: half the MMA instructions (numerics wrong)
#else
constexpr int KS_EX = 8, KS_SQ = 4;
#endif
constexpr uint32_t T_AB = 0, T_H = 256, T_OUT = 384;
constexpr uint32_t IDESC2 = idesc_bf16(256, 128);

#ifdef TRACE
__device__ unsigned long long g_trace2[2][4][1024];     // [cta 0/1][role: 0 expand, 1 squeeze, 2 SwiGLU warp 4, 3 SwiGLU warp 12][event]
#define TR2(role, idx) do { if (cta < 2 && (idx) < 1024) g_trace2[cta][role][(idx)] = clock64(); } while (0)
#else
#define TR2(role, idx) do { } while (0)
#endif
#ifdef TC_ARRIVE_RELEASE
#define ARRIVE_TC mbar_arrive_remote
#else
#define ARRIVE_TC mbar_arrive_remote_relaxed                    // TMEM-only handoffs: ordering via tcgen05.fence::before_thread_sync
#endif
struct Bars {
  // leader-only (both CTAs' completions are counted here): weight fulls (TMA transactions of both CTAs), xn_full (2 LN arrivals),
  // ab_empty / h_full / out_empty (4 warp arrivals from each CTA)
  uint64_t wab_full[NW], ws_full[NW], xn_full[2], ab_empty[2], h_full[2], out_empty;
  // both CTAs (leader's multicast commits)
  uint64_t wab_empty[NW], ws_empty[NW], xn_empty[2], ex_done[2], sq_done[2], out_full;
  // local
  uint64_t x_full[2], x_empty[2];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(512, 1)
transition_fwd2_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mwa,
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
  auto count = [&](int k) { return (tiles > k) ? (tiles - k + G - 1) / G : 0; };
  const int n_valid = count(cta), n_local = count(cta & ~1);   // the pair walks the leader's (larger) count; a short CTA repeats
  const int nch = n_local * NCH;                                // its last tile as a dummy whose results are not stored
  auto tile_of = [&](int i) { return i < n_valid ? cta + i * G : (n_valid > 0 ? cta + (n_valid - 1) * G : 0); };

  if (tid == 0) {
    for (int s = 0; s < NW; ++s) {
      mbar_init(&B.wab_full[s], 1); mbar_init(&B.ws_full[s], 1); mbar_init(&B.wab_empty[s], 1); mbar_init(&B.ws_empty[s], 1);
    }
    for (int s = 0; s < 2; ++s) {
      mbar_init(&B.xn_full[s], 2); mbar_init(&B.ab_empty[s], 8); mbar_init(&B.h_full[s], 8);
      mbar_init(&B.xn_empty[s], 1); mbar_init(&B.ex_done[s], 1); mbar_init(&B.sq_done[s], 1);
      mbar_init(&B.x_full[s], 1); mbar_init(&B.x_empty[s], 1);
    }
    mbar_init(&B.out_empty, 8); mbar_init(&B.out_full, 1);
    fence_barrier_init();
#ifndef OLD_SYNC
    if (n_local > 0) {                                         // tile 0's x now, not after the cluster sync (its barrier is local)
      fence_proxy_async();
      mbar_expect_tx(&B.x_full[0], TILE);
#pragma unroll
      for (int cb = 0; cb < 2; ++cb)
#pragma unroll
        for (int h = 0; h < 2; ++h) tma_load_2d(su + O_X + cb * 16384 + h * 8192, &mx, &B.x_full[0], cb * 64, tile_of(0) * ROWS + h * 64);
    }
#endif
    prefetch_map(&mx); prefetch_map(&mwa); prefetch_map(&mwb); prefetch_map(&mws); prefetch_map(&mout); prefetch_map(&mxn);
  }
  if (tid < 128) { reinterpret_cast<float*>(sm + O_GB)[tid] = gamma[tid]; reinterpret_cast<float*>(sm + O_GB)[128 + tid] = beta[tid]; }
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
        for (int cb = 0; cb < 2; ++cb)
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_load_2d(dst + cb * 16384 + h * 8192, &mx, &B.x_full[b], cb * 64, row + h * 64);
      }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------ weight producer: this CTA's half of
    // every chunk; the transactions complete on the leader's barriers, which the leader armed with the pair's total bytes
    if (lane == 0) {
      const CUtensorMap* mab = leader ? &mwa : &mwb;
      int ca = 0, cs = 0;
      while (ca < nch || cs < nch) {
        if (ca < nch && (ca < NW || mbar_test(&B.wab_empty[ca % NW], ((ca / NW) - 1) & 1))) {
          const int s = ca % NW, j = ca & (NCH - 1);
          if (leader) mbar_expect_tx(&B.wab_full[s], 2 * WAB_SLOT);
          const uint32_t slot = su + O_WAB + s * WAB_SLOT;
          tma_load_2d_2sm(slot, mab, &B.wab_full[s], 0, j * HS);
          tma_load_2d_2sm(slot + 8192, mab, &B.wab_full[s], 64, j * HS);
          ++ca;
        }
        if (cs < nch && (cs < NW || mbar_test(&B.ws_empty[cs % NW], ((cs / NW) - 1) & 1))) {
          const int s = cs % NW, j = cs & (NCH - 1);
          if (leader) mbar_expect_tx(&B.ws_full[s], 2 * WS_SLOT);
          tma_load_2d_2sm(su + O_WS + s * WS_SLOT, &mws, &B.ws_full[s], j * HS, crank * 64);
          ++cs;
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------ MMA issue: leader only; warp 1 expands,
    // warp 2 squeezes (converged warps, one elected lane issues)
    if (leader) {
      if (warp == 1) {
        for (int c = 0; c < nch; ++c) {
          const int i = c >> 3, j = c & (NCH - 1), s = c & 1, u = c >> 1, sw = c % NW;
          if (lane == 0) TR2(0, 4 * c);
          if (j == 0) mbar_wait_cl(&B.xn_full[i & 1], (i >> 1) & 1);
          mbar_wait(&B.wab_full[sw], (c / NW) & 1);
          if (lane == 0) TR2(0, 4 * c + 1);
          if (c >= 2) mbar_wait_cl(&B.ab_empty[s], (u - 1) & 1);
          if (lane == 0) TR2(0, 4 * c + 2);
          tc_fence_after();
          const uint64_t ad = desc_k128(su + O_XN + (i & 1) * TILE), bd = desc_k128(su + O_WAB + sw * WAB_SLOT);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < KS_EX; ++ks)
              umma_ss2(tmem + T_AB + s * 128, ad + (uint64_t)(((ks >> 2) * 16384 + (ks & 3) * 32) >> 4),
                       bd + (uint64_t)(((ks >> 2) * 8192 + (ks & 3) * 32) >> 4), IDESC2, ks > 0 ? 1u : 0u);
            tc_commit2_mc(&B.ex_done[s], 3);
            tc_commit2_mc(&B.wab_empty[sw], 3);
            if (j == NCH - 1) tc_commit2_mc(&B.xn_empty[i & 1], 3);
          }
          __syncwarp();
        }
      } else {
        for (int q = 0; q < nch; ++q) {
          const int i = q >> 3, j = q & (NCH - 1), s = q & 1, u = q >> 1, sw = q % NW;
          if (lane == 0) TR2(1, 4 * q);
          mbar_wait(&B.ws_full[sw], (q / NW) & 1);
          if (lane == 0) TR2(1, 4 * q + 1);
          mbar_wait_cl(&B.h_full[s], u & 1);
          if (lane == 0) TR2(1, 4 * q + 2);
          if (j == 0 && i >= 1) mbar_wait_cl(&B.out_empty, (i - 1) & 1);
          tc_fence_after();
          const uint64_t bd = desc_k128(su + O_WS + sw * WS_SLOT);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < KS_SQ; ++ks)
              umma_ts2(tmem + T_OUT, tmem + T_H + s * 32 + ks * 8, bd + (uint64_t)(ks * 2), IDESC2, (j > 0 || ks > 0) ? 1u : 0u);
            tc_commit2_mc(&B.sq_done[s], 3);
            tc_commit2_mc(&B.ws_empty[sw], 3);
            if (j == NCH - 1) tc_commit2_mc(&B.out_full, 3);
          }
          __syncwarp();
        }
      }
    }
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    // ------------------------------------------------------------------------------------------ SwiGLU (as tfwd.cu v6): two
    // warpgroups alternate chunks, each in four 16-column steps with the next step's TMEM loads in flight
    setmaxnreg_inc<152>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, trow = tmem + (lb << 16);
    for (int c = (warp >= 12 ? 1 : 0); c < nch; c += 2) {
      const int s = c & 1, u = c >> 1;
      const int rl = warp == 4 ? 2 : 3;
      if (lane == 0 && (warp == 4 || warp == 12)) TR2(rl, 4 * c);
      mbar_wait(&B.ex_done[s], u & 1);
      if (lane == 0 && (warp == 4 || warp == 12)) TR2(rl, 4 * c + 1);
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
#ifdef FWD_F2
            { const f2 H = mul2(mul2(mk2(a0, a1), mk2(sigmoid_kit(a0), sigmoid_kit(a1))), mk2(b0, b1)); hp[q * 8 + k] = pack_bf16(lo2(H), hi2(H)); }
#else
            hp[q * 8 + k] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_kit(a1) * b1);
#endif
          }
          if (q < 3) tmem_wait_ld();
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) ARRIVE_TC(&B.ab_empty[s], 0);
      if (lane == 0 && (warp == 4 || warp == 12)) TR2(rl, 4 * c + 2);
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
      if (lane == 0) ARRIVE_TC(&B.h_full[s], 0);
      if (lane == 0 && (warp == 4 || warp == 12)) TR2(rl, 4 * c + 3);
    }
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------------ LayerNorm + residual epilogue (own rows)
    setmaxnreg_inc<152>();
    const int t2 = tid - 256;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t gb_u = su + O_GB;
    auto ln = [&](int i) {
      const int b = i & 1, grow = tile_of(i) * ROWS + (int)r;
      const bool real = i < n_valid;
      mbar_wait(&B.x_full[b], (i >> 1) & 1);
      if (i >= 2) mbar_wait(&B.xn_empty[b], ((i >> 1) - 1) & 1);
      if (t2 == 0) tma_store_wait_read0();
      named_bar_sync(1, 128);
      const uint32_t xb = su + O_X + b * TILE, xnb = su + O_XN + b * TILE;
      uint32_t v[64];
#pragma unroll
      for (int cb = 0; cb < 2; ++cb)
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 t = lds128(xb + cb * 16384 + sw128(r, q));
          v[cb * 32 + q * 4 + 0] = t.x; v[cb * 32 + q * 4 + 1] = t.y; v[cb * 32 + q * 4 + 2] = t.z; v[cb * 32 + q * 4 + 3] = t.w;
        }
      float p[32];                                             // the sm_90a reduction tree (see tfwd.cu)
#pragma unroll
      for (int l = 0; l < 32; ++l) p[l] = (bf16lo(v[2 * l]) + bf16hi(v[2 * l])) + (bf16lo(v[2 * l + 1]) + bf16hi(v[2 * l + 1]));
#pragma unroll
      for (int k = 16; k; k >>= 1)
#pragma unroll
        for (int l = 0; l < k; ++l) p[l] = p[l] + p[l + k];
      const float mean = p[0] * (1.f / D_);
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        float acc = 0.f, d;
        d = bf16lo(v[2 * l]) - mean; acc += d * d;
        d = bf16hi(v[2 * l]) - mean; acc += d * d;
        d = bf16lo(v[2 * l + 1]) - mean; acc += d * d;
        d = bf16hi(v[2 * l + 1]) - mean; acc += d * d;
        p[l] = acc;
      }
#pragma unroll
      for (int k = 16; k; k >>= 1)
#pragma unroll
        for (int l = 0; l < k; ++l) p[l] = p[l] + p[l + k];
      const float rs = rsqrtf(p[0] * (1.f / D_) + eps);
#pragma unroll
      for (int cb = 0; cb < 2; ++cb)
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          uint32_t o[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int col = cb * 64 + q * 8 + 2 * k;
            const uint32_t w = v[cb * 32 + q * 4 + k];
            const float2 g2 = lds64f(gb_u + col * 4), b2 = lds64f(gb_u + 512 + col * 4);
            o[k] = pack_bf16((bf16lo(w) - mean) * rs * g2.x + b2.x, (bf16hi(w) - mean) * rs * g2.y + b2.y);
          }
          sts128(xnb + cb * 16384 + sw128(r, q), make_uint4(o[0], o[1], o[2], o[3]));
        }
      if (save && real) { rstd[grow] = rs; c1[grow] = mean * rs; }
      fence_proxy_async();
      named_bar_sync(1, 128);
      if (t2 == 0) {
#ifdef OLD_SYNC
        mbar_arrive_remote(&B.xn_full[b], 0);
#else
        mbar_arrive_remote_cta(&B.xn_full[b], 0);
#endif
        if (save && real) {
          const int row0 = tile_of(i) * ROWS;
#pragma unroll
          for (int cb = 0; cb < 2; ++cb)
#pragma unroll
            for (int h = 0; h < 2; ++h) tma_store_2d(&mxn, xnb + cb * 16384 + h * 8192, cb * 64, row0 + h * 64);
          tma_store_commit();
        }
      }
    };
    auto epi = [&](int i) {
      const int b = i & 1;
      mbar_wait(&B.out_full, i & 1);
      tc_fence_after();
      const uint32_t xb = su + O_X + b * TILE;
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        uint32_t acc[32];
        tmem_ld32(trow + T_OUT + cc * 32, acc);
        tmem_wait_ld();
        const int cb = cc >> 1;
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const int q = (cc & 1) * 4 + qq;
          const uint32_t ad = xb + cb * 16384 + sw128(r, q);
          const uint4 xv = lds128(ad);
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
          uint32_t o[4];
#pragma unroll
          for (int k = 0; k < 4; ++k)
#ifdef FWD_F2
            { const f2 Y = add2(mk2(bf16lo(xw[k]), bf16hi(xw[k])), mk2u(acc[qq * 8 + 2 * k], acc[qq * 8 + 2 * k + 1])); o[k] = pack_bf16(lo2(Y), hi2(Y)); }
#else
            o[k] = pack_bf16(bf16lo(xw[k]) + __uint_as_float(acc[qq * 8 + 2 * k]), bf16hi(xw[k]) + __uint_as_float(acc[qq * 8 + 2 * k + 1]));
#endif
          sts128(ad, make_uint4(o[0], o[1], o[2], o[3]));
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) ARRIVE_TC(&B.out_empty, 0);
      fence_proxy_async();
      named_bar_sync(1, 128);
      if (t2 == 0) {
        const int row0 = tile_of(i) * ROWS;
        if (i < n_valid) {
#pragma unroll
          for (int cb = 0; cb < 2; ++cb)
#pragma unroll
            for (int h = 0; h < 2; ++h) tma_store_2d(&mout, xb + cb * 16384 + h * 8192, cb * 64, row0 + h * 64);
        }
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
    if (t2 == 0) tma_store_wait0();
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
