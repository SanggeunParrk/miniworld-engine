// tfwd.cu — the Transition forward (LayerNorm + SwiGLU expand + squeeze + residual) at D = 128, H = 4D = 512, bf16, as ONE
// fused sm_100a kernel.  SPDX-License-Identifier: Apache-2.0
//
// The fusion is the sm_90a kernel's (experiments/transition_fused/src/transition_fwd.cu), unchanged:
//   xn = bf16(LN(x))                          saved with rstd and c1 = mean rstd when `save` (the backward's inputs)
//   per hidden chunk j (8 x 64 units):  [a|b] = xn [Wa_j; Wb_j]^T      one M128 N128 K128 product over the packed weight tile
//                                       h_j   = bf16(silu(a) b)        silu with the kit sigmoid rcp(1 + ex2(-a log2 e))
//                                       acc  += h_j Ws_j^T             M128 N128 K64, accumulated across the eight chunks
//   out = bf16(x + acc)
// a persistent CTA per 128-row tile and a two-slot TMA ring over the eight weight chunks.  What changes is how Blackwell runs it:
// the accumulators live in tensor memory, and the sm_90a register-source squeeze becomes an A-from-TMEM tcgen05.mma (h is written
// back to TMEM as bf16, never to shared or global memory).  Roles (384 threads):
//   warp 0      TMA: x tiles                warp 3      TMA: weight chunks
//   warp 1      tcgen05.mma issue           warp 2      TMEM allocation
//   warps 4-7   SwiGLU: [a|b] TMEM -> h TMEM, one row per thread
//   warps 8-11  LayerNorm of the next tile (x smem -> xn smem, + saves) and the residual epilogue (TMEM acc + x -> TMA store)
#include "sm100.cuh"
using namespace s100;

constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128;
constexpr int WAB_SLOT = 32768, WS_SLOT = 16384;               // [Wa_j; Wb_j] as 2 K-blocks of [128 n][64 k]; Ws_j [128 d][64 k]
constexpr int O_WAB = 0, O_WS = 2 * WAB_SLOT;                  // two separate rings: [Wa;Wb] is freed after the expand, Ws after the squeeze
constexpr int O_X = O_WS + 2 * WS_SLOT, TILE = 32768;          // x tile [128 rows][128 d] as 2 K-blocks of 16 KB, double-buffered
constexpr int O_XN = O_X + 2 * TILE;                           // xn tile, double-buffered
constexpr int O_GB = O_XN + 2 * TILE;                          // gamma, beta fp32 [2][128]
constexpr int O_BAR = O_GB + 1024;
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
// tensor memory columns
constexpr uint32_t T_AB = 0, T_H = 256, T_OUT = 384;           // [a|b] x2 (128 cols each), h x2 (32 packed cols each), acc (128)
constexpr uint32_t IDESC = idesc_bf16(128, 128);
#ifdef TRACE
// clock64 stamps for the first TRACE_CTAS CTAs: trace[cta][role][event]; role 0 MMA, 1 SwiGLU (warp 4), 2 LN/epi (warp 8)
constexpr int TR_N = 1024;
__device__ unsigned long long g_trace[4][4][TR_N];   // role 3: MMA wait breakdown
__device__ unsigned long long g_span[256][3];          // per CTA: globaltimer at entry, after setup, at exit
#define TR(role, idx) do { if (cta < 4 && (idx) < TR_N) g_trace[cta][role][(idx)] = clock64(); } while (0)
#else
#define TR(role, idx) do { } while (0)
#endif

struct Bars {
  uint64_t wab_full[2], ws_full[2], x_full[2], x_empty[2], xn_full[2], xn_empty[2], ex_done[2], ab_empty[2], h_full[2], sq_done[2];
  uint64_t out_full, out_empty;
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(512, 1)
transition_fwd_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mwa,
                     const __grid_constant__ CUtensorMap mwb, const __grid_constant__ CUtensorMap mws,
                     const __grid_constant__ CUtensorMap mout, const __grid_constant__ CUtensorMap mxn,
                     const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ rstd,
                     float* __restrict__ c1, int tiles, float eps, int save) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
#ifdef TRACE
  auto gtime = [] { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
  if (threadIdx.x == 0) g_span[blockIdx.x][0] = gtime();
#endif
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int n_local = (tiles > cta) ? (tiles - cta + G - 1) / G : 0;
  const int nch = n_local * NCH;

  if (tid == 0) {
    for (int s = 0; s < 2; ++s) {
      mbar_init(&B.wab_full[s], 1); mbar_init(&B.ws_full[s], 1);
      mbar_init(&B.x_full[s], 1); mbar_init(&B.x_empty[s], 1);
      mbar_init(&B.xn_full[s], 1); mbar_init(&B.xn_empty[s], 1); mbar_init(&B.ex_done[s], 1); mbar_init(&B.ab_empty[s], 4);
      mbar_init(&B.h_full[s], 4); mbar_init(&B.sq_done[s], 1);
    }
    mbar_init(&B.out_full, 1); mbar_init(&B.out_empty, 4);
    fence_barrier_init();
    prefetch_map(&mx); prefetch_map(&mwa); prefetch_map(&mwb); prefetch_map(&mws); prefetch_map(&mout); prefetch_map(&mxn);
  }
  if (tid < 128) { reinterpret_cast<float*>(sm + O_GB)[tid] = gamma[tid]; reinterpret_cast<float*>(sm + O_GB)[128 + tid] = beta[tid]; }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
#ifdef TRACE
  if (threadIdx.x == 0) { uint32_t sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid)); g_span[blockIdx.x][1] = (gtime() & ~0xFFull) | sid; }
#endif

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    // ------------------------------------------------------------------------------------------ x producer
    if (lane == 0) {
      for (int i = 0; i < n_local; ++i) {
        const int b = i & 1, row = (cta + i * G) * ROWS;
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
    // ------------------------------------------------------------------------------------------ weight producer: one thread keeps both
    // rings full, polling (non-blocking) each ring's release so neither stream waits behind the other
    if (lane == 0) {
      int ca = 0, cs = 0;                                // next chunk to load into the [Wa;Wb] ring / the Ws ring
      while (ca < nch || cs < nch) {
        if (ca < nch && (ca < 2 || mbar_test(&B.ex_done[ca & 1], ((ca >> 1) - 1) & 1))) {
          const int s = ca & 1, j = ca & (NCH - 1);
#ifdef ABL_NOW
          if (ca >= 2) { mbar_arrive(&B.wab_full[s]); ++ca; continue; }
#endif
          mbar_expect_tx(&B.wab_full[s], WAB_SLOT);
          const uint32_t slot = su + O_WAB + s * WAB_SLOT;
#pragma unroll
          for (int cb = 0; cb < 2; ++cb) {
            tma_load_2d(slot + cb * 16384, &mwa, &B.wab_full[s], cb * 64, j * HS);
            tma_load_2d(slot + cb * 16384 + 8192, &mwb, &B.wab_full[s], cb * 64, j * HS);
          }
          ++ca;
        }
        if (cs < nch && (cs < 2 || mbar_test(&B.sq_done[cs & 1], ((cs >> 1) - 1) & 1))) {
          const int s = cs & 1, j = cs & (NCH - 1);
#ifdef ABL_NOW
          if (cs >= 2) { mbar_arrive(&B.ws_full[s]); ++cs; continue; }
#endif
          mbar_expect_tx(&B.ws_full[s], WS_SLOT);
          const uint32_t slot = su + O_WS + s * WS_SLOT;
          tma_load_2d(slot, &mws, &B.ws_full[s], j * HS, 0);
          tma_load_2d(slot + 8192, &mws, &B.ws_full[s], j * HS, 64);
          ++cs;
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------ MMA issue: the whole warp runs the
    // schedule (so descriptors stay warp-uniform and live in uniform registers), one elected lane issues
    {
      auto squeeze = [&](int q) {
        const int i = q >> 3, j = q & (NCH - 1), s = q & 1, u = q >> 1;
        TR(3, 8 * q + 3);
        mbar_wait(&B.ws_full[s], u & 1);
        TR(3, 8 * q + 4);
        mbar_wait(&B.h_full[s], u & 1);
        TR(3, 8 * q + 5);
        if (j == 0 && i >= 1) mbar_wait(&B.out_empty, (i - 1) & 1);
        TR(0, 2 * q + 1);
        tc_fence_after();
        const uint64_t bd = desc_k128(su + O_WS + s * WS_SLOT);
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ts(tmem + T_OUT, tmem + T_H + s * 32 + ks * 8, bd + (uint64_t)(ks * 2), IDESC, (j > 0 || ks > 0) ? 1u : 0u);
          tc_commit(&B.sq_done[s]);                  // h buffer and Ws slot both free
          if (j == NCH - 1) tc_commit(&B.out_full);
        }
        __syncwarp();
        TR(3, 8 * q + 7);
      };
      auto expand = [&](int c) {
        const int i = c >> 3, j = c & (NCH - 1), s = c & 1, u = c >> 1;
        TR(3, 8 * c + 0);
        if (j == 0) mbar_wait(&B.xn_full[i & 1], (i >> 1) & 1);
        TR(3, 8 * c + 1);
        mbar_wait(&B.wab_full[s], u & 1);
        TR(3, 8 * c + 2);
        if (c >= 2) mbar_wait(&B.ab_empty[s], (u - 1) & 1);
        TR(0, 2 * c);
        tc_fence_after();
        const uint64_t ad = desc_k128(su + O_XN + (i & 1) * TILE), bdw = desc_k128(su + O_WAB + s * WAB_SLOT);
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            const uint64_t off = (uint64_t)(((ks >> 2) * 16384 + (ks & 3) * 32) >> 4);
            umma_ss(tmem + T_AB + s * 128, ad + off, bdw + off, IDESC, ks > 0 ? 1u : 0u);
          }
          tc_commit(&B.ex_done[s]);                  // [a|b] ready and the [Wa;Wb] slot free
          if (j == NCH - 1) tc_commit(&B.xn_empty[i & 1]);
        }
        __syncwarp();
        TR(3, 8 * c + 6);
      };
      // expand(c + 2) goes out as soon as the SwiGLU warpgroup of chunk c has READ its [a|b] buffer, ahead of squeeze(c), so the
      // next [a|b] of that warpgroup is computed while it is still busy with chunk c
      // two issuing warps feed the tensor pipe: warp 1 the expands, warp 2 the squeezes; each waits only on its own operands, so
      // one warp's barrier waits never leave the pipe without queued work from the other
      if (warp == 1) { for (int c = 0; c < nch; ++c) expand(c); }
      else { for (int c = 0; c < nch; ++c) squeeze(c); }
    }
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    // ------------------------------------------------------------------------------------------ SwiGLU, two warpgroups in ping-pong:
    // warps 4-7 take the even chunks ([a|b] / h buffer 0), warps 12-15 the odd ones (buffer 1)
    setmaxnreg_inc<152>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, trow = tmem + (lb << 16);
    for (int c = (warp >= 12 ? 1 : 0); c < nch; c += 2) {
      const int s = c & 1, u = c >> 1;
      if (lane == 0 && (warp == 4 || warp == 12)) TR(1, 4 * c);
      mbar_wait(&B.ex_done[s], u & 1);
      if (lane == 0 && (warp == 4 || warp == 12)) TR(1, 4 * c + 1);
      tc_fence_after();
      uint32_t hp[32];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        uint32_t a[32], b[32];
#ifndef ABL_NOSWLD
        tmem_ld32(trow + T_AB + s * 128 + half * 32, a);
        tmem_ld32(trow + T_AB + s * 128 + 64 + half * 32, b);
        tmem_wait_ld();
#else
        for (int k = 0; k < 32; ++k) { a[k] = __float_as_uint(0.01f * k + c); b[k] = a[k] ^ lane; }
#endif
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float a0 = __uint_as_float(a[2 * k]), a1 = __uint_as_float(a[2 * k + 1]);
          const float b0 = __uint_as_float(b[2 * k]), b1 = __uint_as_float(b[2 * k + 1]);
#ifndef ABL_NOMATH
#if defined(SIG_POLY_QUARTER)
          // opt-in: one exponential in four on the FMA pipe
          hp[half * 16 + k] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * ((k & 1) ? sigmoid_poly(a1) : sigmoid_kit(a1)) * b1);
#elif defined(SIG_POLY_HALF)
          // opt-in experiment: half of the exponentials on the FMA pipe (ex2_poly, more accurate than ex2.approx) -- measured slower
          hp[half * 16 + k] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_poly(a1) * b1);
#else
          hp[half * 16 + k] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_kit(a1) * b1);
#endif
#else
          hp[half * 16 + k] = pack_bf16(a0 * b0, a1 * b1);
#endif
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.ab_empty[s]);
      if (lane == 0 && (warp == 4 || warp == 12)) TR(1, 4 * c + 2);
      if (c >= 2) mbar_wait(&B.sq_done[s], (u - 1) & 1);
      tc_fence_after();
      uint32_t h0[16], h1[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) { h0[k] = hp[k]; h1[k] = hp[16 + k]; }
#ifndef ABL_NOSWLD
      tmem_st16(trow + T_H + s * 32, h0);
      tmem_st16(trow + T_H + s * 32 + 16, h1);
      tmem_wait_st();
#else
      if (h0[3] == 12345u && h1[5] == 7u) tmem_st16(trow + T_H + s * 32, h0);
#endif
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.h_full[s]);
      if (lane == 0 && (warp == 4 || warp == 12)) TR(1, 4 * c + 3);
    }
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------------ LayerNorm + residual epilogue
    setmaxnreg_inc<152>();
    const int t2 = tid - 256;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t gb_u = su + O_GB;
    auto ln = [&](int i) {
      const int b = i & 1, grow = (cta + i * G) * ROWS + (int)r;
      if (t2 == 0) TR(2, 8 * i);
      mbar_wait(&B.x_full[b], (i >> 1) & 1);
      if (i >= 2) mbar_wait(&B.xn_empty[b], ((i >> 1) - 1) & 1);
      if (t2 == 0) TR(2, 8 * i + 1);
      if (t2 == 0) tma_store_wait_read0();                     // the previous xn store out of this buffer has been read
      named_bar_sync(1, 128);
      const uint32_t xb = su + O_X + b * TILE, xnb = su + O_XN + b * TILE;
#ifdef ABL_NOLN
      if (true) { fence_proxy_async(); named_bar_sync(1, 128); if (t2 == 0) mbar_arrive(&B.xn_full[b]); return; }
#endif
      uint32_t v[64];
#pragma unroll
      for (int cb = 0; cb < 2; ++cb)
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 t = lds128(xb + cb * 16384 + sw128(r, q));
          v[cb * 32 + q * 4 + 0] = t.x; v[cb * 32 + q * 4 + 1] = t.y; v[cb * 32 + q * 4 + 2] = t.z; v[cb * 32 + q * 4 + 3] = t.w;
        }
      // the sm_90a kernel's reduction tree: "lane" l holds columns 4l .. 4l + 3, then xor-shuffles over 16, 8, 4, 2, 1
      float p[32];
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
      if (save) { rstd[grow] = rs; c1[grow] = mean * rs; }
      fence_proxy_async();
      named_bar_sync(1, 128);
      if (t2 == 0) {
        TR(2, 8 * i + 2);
        mbar_arrive(&B.xn_full[b]);
        if (save) {
          const int row0 = (cta + i * G) * ROWS;
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
      if (t2 == 0) TR(2, 8 * i + 3);
      mbar_wait(&B.out_full, i & 1);
      if (t2 == 0) TR(2, 8 * i + 4);
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
            o[k] = pack_bf16(bf16lo(xw[k]) + __uint_as_float(acc[qq * 8 + 2 * k]), bf16hi(xw[k]) + __uint_as_float(acc[qq * 8 + 2 * k + 1]));
          sts128(ad, make_uint4(o[0], o[1], o[2], o[3]));
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.out_empty);
      fence_proxy_async();
      named_bar_sync(1, 128);
      if (t2 == 0) {
        const int row0 = (cta + i * G) * ROWS;
#pragma unroll
        for (int cb = 0; cb < 2; ++cb)
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_store_2d(&mout, xb + cb * 16384 + h * 8192, cb * 64, row0 + h * 64);
        TR(2, 8 * i + 5);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.x_empty[b]);
        TR(2, 8 * i + 6);
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
#ifdef TRACE
  if (threadIdx.x == 0) g_span[blockIdx.x][2] = gtime();
#endif
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
