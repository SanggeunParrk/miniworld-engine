// tfwd8.cu — the Transition forward of tfwd2.cu (same fusion, 2-CTA tcgen05.mma, same schedule) with RELAXED PRECISION: e4m3
// tensor-core operands (kind::f8f6f4, fp32 accumulate). xn is quantized with the bound-based scale s_x (sc[0]) and stored as e4m3 for
// the backward; the weights come pre-quantized (quant8.cu: wab_q = [Wa; Wb] with s_wab = sc[1], ws_q with s_ws = sc[2]); h is quantized
// with the delayed scale s_hf = sc[6]; the SwiGLU runs in fp32x2 with sigmoid(a) = 0.5 tanh(a / 2) + 0.5.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128;
constexpr int NW = 4;                                          // weight ring depth (both rings)
constexpr int WAB_SLOT = 8192, WS_SLOT = 4096;                 // this CTA's half: e4m3 [64 n][128 k] (128-B swizzle); Ws [64 d][64 k] (64-B)
constexpr int O_WAB = 0, O_WS = NW * WAB_SLOT;
constexpr int O_X = O_WS + NW * WS_SLOT, TILE = 32768, XQT = 16384;
constexpr int O_XN = O_X + 2 * TILE;
constexpr int O_GB = O_XN + 2 * XQT;
constexpr int O_BAR = O_GB + 1024;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t T_AB = 0, T_H = 256, T_OUT = 384;         // h (e4m3) x2: 16 columns each
constexpr uint32_t IDESC2 = idesc_e4m3(256, 128);

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
transition_fwd8_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mwab,
                      const __grid_constant__ CUtensorMap mws, const __grid_constant__ CUtensorMap mout, const __grid_constant__ CUtensorMap mxn,
                      const float* __restrict__ gamma, const float* __restrict__ beta, const float* __restrict__ sc, float* __restrict__ rstd,
                      float* __restrict__ c1, int tiles, float eps, int save) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
#ifdef PDL
  pdl_launch();                                                  // the backward's CTAs may take SMs as ours exit
#endif
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
    prefetch_map(&mx); prefetch_map(&mwab); prefetch_map(&mws); prefetch_map(&mout); prefetch_map(&mxn);
  }
  if (tid < 128) {                                               // gamma / s_x, beta / s_x: the LayerNorm lands in e4m3 units directly
    const float ix = 1.f / sc[0];
    reinterpret_cast<float*>(sm + O_GB)[tid] = gamma[tid] * ix; reinterpret_cast<float*>(sm + O_GB)[128 + tid] = beta[tid] * ix;
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    // ------------------------------------------------------------------------------------------ x producer (own tiles)
    if (lane == 0) {
      for (int i = 0; i < n_local; ++i) {
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
#ifdef PDL
      pdl_wait();                                                // the e4m3 weights come from the quantization kernel
#endif
      int ca = 0, cs = 0;
      while (ca < nch || cs < nch) {
        if (ca < nch && (ca < NW || mbar_test(&B.wab_empty[ca % NW], ((ca / NW) - 1) & 1))) {
          const int s = ca % NW, j = ca & (NCH - 1);
          if (leader) mbar_expect_tx(&B.wab_full[s], 2 * WAB_SLOT);
          const uint32_t slot = su + O_WAB + s * WAB_SLOT;
          tma_load_2d_2sm(slot, &mwab, &B.wab_full[s], 0, crank * H_ + j * HS);    // Wa_j (leader) / Wb_j (peer)
          ++ca;
        }
        if (cs < nch && (cs < NW || mbar_test(&B.ws_empty[cs % NW], ((cs / NW) - 1) & 1))) {
          const int s = cs % NW, j = cs & (NCH - 1);
          if (leader) mbar_expect_tx(&B.ws_full[s], 2 * WS_SLOT);
          tma_load_2d_2sm(su + O_WS + s * WS_SLOT, &mws, &B.ws_full[s], j * HS, crank * 64);  // Ws [d 64 of this CTA][k 64 of chunk j]
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
          const uint64_t ad = desc_k128(su + O_XN + (i & 1) * XQT), bd = desc_k128(su + O_WAB + sw * WAB_SLOT);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma8_ss2(tmem + T_AB + s * 128, ad + (uint64_t)(ks * 2), bd + (uint64_t)(ks * 2), IDESC2, ks > 0 ? 1u : 0u);
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
          const uint64_t bd = desc_sw64(su + O_WS + sw * WS_SLOT);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
              umma8_ts2(tmem + T_OUT, tmem + T_H + s * 16 + ks * 8, bd + (uint64_t)(ks * 2), IDESC2, (j > 0 || ks > 0) ? 1u : 0u);
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
    const float ca = sc[0] * sc[1];
    const f2 CA = mk2(ca, ca), CB = mk2(ca / sc[6], ca / sc[6]);  // a in true units, b folded with 1 / s_hf: h lands in e4m3 units
    for (int c = (warp >= 12 ? 1 : 0); c < nch; c += 2) {
      const int s = c & 1, u = c >> 1;
      const int rl = warp == 4 ? 2 : 3;
      if (lane == 0 && (warp == 4 || warp == 12)) TR2(rl, 4 * c);
      mbar_wait(&B.ex_done[s], u & 1);
      if (lane == 0 && (warp == 4 || warp == 12)) TR2(rl, 4 * c + 1);
      tc_fence_after();
      uint32_t hq[16];
      {
        uint32_t a[2][16], b[2][16];
        tmem_ld16(trow + T_AB + s * 128, a[0]);
        tmem_ld16(trow + T_AB + s * 128 + 64, b[0]);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
#ifndef FABL_TLD
          if (q < 3) {
            tmem_ld16(trow + T_AB + s * 128 + (q + 1) * 16, a[(q + 1) & 1]);
            tmem_ld16(trow + T_AB + s * 128 + 64 + (q + 1) * 16, b[(q + 1) & 1]);
          }
#endif
#pragma unroll
          for (int k = 0; k < 4; ++k) {                      // 4 units -> one word of e4m3 h
            f2 hv[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int x = 4 * k + 2 * e;
              const f2 A = mul2(mk2u(a[q & 1][x], a[q & 1][x + 1]), CA), Bv = mul2(mk2u(b[q & 1][x], b[q & 1][x + 1]), CB);
#ifdef FABL_SW
              hv[e] = mul2(A, Bv);                             // ablation: no sigmoid / silu
#else
              hv[e] = mul2(mul2(A, sigmoid2(A)), Bv);
#endif
            }
            hq[q * 4 + k] = e4m3x4(hv[0], hv[1]);
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
      tmem_st16(trow + T_H + s * 16, hq);
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
      const uint32_t xb = su + O_X + b * TILE, xnb = su + O_XN + b * XQT;
      uint32_t v[64];
#pragma unroll
      for (int cb = 0; cb < 2; ++cb)
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 t = lds128(xb + cb * 16384 + sw128(r, q));
          v[cb * 32 + q * 4 + 0] = t.x; v[cb * 32 + q * 4 + 1] = t.y; v[cb * 32 + q * 4 + 2] = t.z; v[cb * 32 + q * 4 + 3] = t.w;
        }
      // mean and variance in fp32x2 (relaxed-precision path: summation order differs from the sm_90a tree)
      f2 s2[8];
#pragma unroll
      for (int l = 0; l < 8; ++l) s2[l] = mk2(0.f, 0.f);
#pragma unroll
      for (int l = 0; l < 64; ++l) s2[l & 7] = add2(s2[l & 7], mk2(bf16lo(v[l]), bf16hi(v[l])));
#pragma unroll
      for (int k = 4; k; k >>= 1)
#pragma unroll
        for (int l = 0; l < k; ++l) s2[l] = add2(s2[l], s2[l + k]);
      const float mean = (lo2(s2[0]) + hi2(s2[0])) * (1.f / D_);
      const f2 NMN = mk2(-mean, -mean);
#pragma unroll
      for (int l = 0; l < 8; ++l) s2[l] = mk2(0.f, 0.f);
#pragma unroll
      for (int l = 0; l < 64; ++l) { const f2 d = add2(mk2(bf16lo(v[l]), bf16hi(v[l])), NMN); s2[l & 7] = fma2(d, d, s2[l & 7]); }
#pragma unroll
      for (int k = 4; k; k >>= 1)
#pragma unroll
        for (int l = 0; l < k; ++l) s2[l] = add2(s2[l], s2[l + k]);
      const float rs = rsqrtf((lo2(s2[0]) + hi2(s2[0])) * (1.f / D_) + eps);
      const f2 RS2 = mk2(rs, rs), NMR = mk2(-mean * rs, -mean * rs);
#pragma unroll
      for (int q = 0; q < 8; ++q) {                              // 16 columns -> one 16-byte chunk of e4m3 xn
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          f2 pr[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int col = q * 16 + 4 * k + 2 * e;
            const uint32_t w = v[col >> 1];
            const float2 g2 = lds64f(gb_u + col * 4), b2 = lds64f(gb_u + 512 + col * 4);
#ifdef FABL_LN
            pr[e] = mk2(bf16lo(w), bf16hi(w)); (void)g2; (void)b2;
#else
            const f2 A = mul2(mk2(g2.x, g2.y), RS2);           // x (rs g) + (b - mean rs g)
            pr[e] = fma2(mk2(bf16lo(w), bf16hi(w)), A, fma2(mk2(g2.x, g2.y), NMR, mk2(b2.x, b2.y)));
#endif
          }
          o[k] = e4m3x4(pr[0], pr[1]);
        }
        sts128(xnb + sw128(r, q), make_uint4(o[0], o[1], o[2], o[3]));
      }
      if (save && real) { rstd[grow] = rs; c1[grow] = mean * rs; }
      fence_proxy_async();
      named_bar_sync(1, 128);
      if (t2 == 0) {
        mbar_arrive_remote(&B.xn_full[b], 0);
        if (save && real) {
          const int row0 = tile_of(i) * ROWS;
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_store_2d(&mxn, xnb + h * 8192, 0, row0 + h * 64);
          tma_store_commit();
        }
      }
    };
    const float co = sc[6] * sc[2];                                 // acc (e4m3 h x e4m3 Ws) -> true units
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
          {
            const f2 yv = fma2(mk2u(acc[qq * 8 + 2 * k], acc[qq * 8 + 2 * k + 1]), mk2(co, co), mk2(bf16lo(xw[k]), bf16hi(xw[k])));
            o[k] = pack_bf16(lo2(yv), hi2(yv));
          }
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
  cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}
