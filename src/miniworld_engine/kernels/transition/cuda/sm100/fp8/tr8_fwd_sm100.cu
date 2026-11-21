// tr8_fwd_sm100.cu -- the e4m3 fused Transition forward of the opt-in B200 FP8 route (fp8_sm100a.py; docs/b200/fp8.md "Fused e4m3"): the structure of
// tr_fwd_sm100.cu (2-CTA tcgen05 pair, 128-row tiles, 64-unit hidden chunks, SwiGLU from tensor memory on two warpgroups, h written back to tensor memory
// and consumed there by the squeeze as the TMEM A operand, LayerNorm of the next tile + residual epilogue on warps 8-11) with e4m3 operands:
//   LayerNorm  xn8 = e4m3(LN(x) sx) from the fp32 LayerNorm (sx = pow2 LayerNorm-bound scale: cannot saturate), x stays bf16 in smem for the residual
//   expand     [a|b] = xn8 [Wa8_j; Wb8_j]^T      N = 128 (leader: Wa8_j rows, peer: Wb8_j), K = D = 4 (D 128) / 8 (D 256) kind::f8f6f4 K32 steps
//   SwiGLU     h = silu(a ca) (b cb) -> e4m3 (cvt.rn.satfinite), 64 bytes a row and chunk = 16 TMEM columns, max |h| -> guard buffer
//   squeeze    acc += h8 Ws8^T over a PAIR of chunks (K = 128 = 128 B rows of Ws8: the 128-B-swizzle tensor map; 4 TS MMAs K32, A from TMEM), N = D
//   epilogue   out = x + acc co  (co = 1 / (sh s2))
// D = 128: h in its own TMEM columns (double buffered per pair), two xn8 buffers. D = 256 (DIM=256): the output accumulator takes 256 columns, so h is
// written IN PLACE over the first 16 columns of its own [a|b] buffer and expand(c + 2) waits for the squeeze of the pair that read it (as tfwd_d256.cu);
// one xn8 buffer (the LayerNorm of tile i + 1 waits for the last expand of tile i); the LayerNorm reads the row in three passes (no 128-register row).
// scales sc[] = [sx, ca = 1 / (sx s1), cb = sh ca, co = 1 / (sh s2)]. Inference only, rows a multiple of 128.
// Build switches (defaults in brackets; the measured A/B of every one is in docs/b200/fp8.md section 8.2): HS_ hidden units a chunk [64; 32 = four [a|b] buffers in flight, slower], NAB_, PAIRSQ [1 at D 128: one squeeze per pair
// of chunks; 0: per chunk], LNP [1: packed f32x2 LayerNorm], LNH [1 at D 256: LayerNorm + epilogue on warps 4-11 (halves of the columns), SwiGLU on warps 12-15], RESG [1 at D 256: residual re-read from L2], NXS_ / NW_ / NWS_ / XNB_ (ring depths),
// ABL (timing ablations: WRONG output), PROBE (clock64 stamps of one CTA into the peak buffer: `work/n14-f/probe.py`).
// SPDX-License-Identifier: Apache-2.0
#include <cuda_fp16.h>
#include "sm100.cuh"
using namespace s100;

template <int B, int S, int N, class F> DEVI float tree_sum(F& f) {
  if constexpr (N == 1) return f(B);
  else return tree_sum<B, 2 * S, N / 2>(f) + tree_sum<B + S, 2 * S, N / 2>(f);
}
template <class F> DEVI float tree32(F f) { return tree_sum<0, 1, 32>(f); }

#ifndef DIM
#define DIM 128
#endif
#ifndef HID
#define HID 512
#endif
#ifndef HS_
#define HS_ 64      // hidden units a chunk (64: [a|b] N = 128; 32: N = 64, more [a|b] buffers in flight)
#endif
#ifndef NAB_
#define NAB_ (256 / (2 * HS_))   // [a|b] buffers (256 tensor-memory columns)
#endif
constexpr int D_ = DIM, H_ = HID, HS = HS_, NAB = NAB_, NCH = H_ / HS, CPS = 128 / HS, NSL = NCH / CPS, ROWS = 128;
static_assert((D_ == 128 || D_ == 256) && (HS == 32 || HS == 64) && (NCH & (NCH - 1)) == 0 && NSL >= 1 && NAB * 2 * HS <= 256, "shape");
constexpr bool INPL = D_ > 128;                                // h in place over its [a|b] buffer (the output accumulator takes 256 columns)
#ifndef NXS_
#define NXS_ (INPL ? 1 : 4)
#endif
#ifndef NW_
#define NW_ (HS_ == 32 ? (DIM == 128 ? 8 : 6) : ((LNH && DIM == 256) ? 3 : 4))
#endif
#ifndef NWS_
#define NWS_ 2
#endif
#ifndef XNB_
#define XNB_ 2
#endif
#ifndef PAIRSQ
#define PAIRSQ (HS_ == 64 && DIM == 128)   // 1: one squeeze per PAIR of chunks (K = 128 on the whole Ws slot); 0: one per chunk
#endif
#ifndef LNH
#define LNH (DIM == 256)   // 1: LayerNorm + epilogue on warps 4-11 (two warpgroups, one half of the columns each, statistics exchanged in shared memory), every SwiGLU chunk on warps 12-15
#endif
#ifndef LNP
#define LNP 1      // 1: packed-pair (f32x2) LayerNorm, three passes
#endif
#ifndef ABL
#define ABL 0      // timing ablations (WRONG output): 1 no tanh, 2 no LayerNorm reductions, 4 no residual loads in the epilogue
#endif
#ifndef RESG
#define RESG (INPL ? 1 : 0)       // 1: the residual x is read back from global memory (L2) in the epilogue and y stored from registers: the x stage lives only until the LayerNorm has read it
#endif
constexpr int NXS = NXS_, NW = NW_, NWS = NWS_, XNB = XNB_;
constexpr int XB = D_ / 64;                                    // 128-B (64 bf16) column blocks of an x tile
constexpr int KB8 = D_ / 128;                                  // 128-B (128 e4m3) K-blocks of xn8 and of a weight slot
constexpr int TILE = ROWS * D_ * 2, TILE8 = ROWS * D_;
constexpr int WAB_SLOT = HS * D_, WS_SLOT = (D_ / 2) * 128, WAB_KB = HS * 128;    // this CTA's half: [64 n][D B] as KB8 blocks of 8 KB; Ws8 [D/2 d][128 B (a pair of chunks)]
constexpr int O_WAB = 0, O_WS = NW * WAB_SLOT;
constexpr int O_X = O_WS + NWS * WS_SLOT;
constexpr int O_XN = O_X + NXS * TILE;
constexpr int O_GB = O_XN + XNB * TILE8;
constexpr int O_BAR = O_GB + 2 * D_ * 4;
constexpr int O_RED = O_BAR + 512;                     // LNH: 2 x 256 fp32 (partial sums, partial squares)
constexpr int SMEM_BYTES = O_RED + (LNH ? 2048 : 0);
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
static_assert(!LNH || LNP, "LNH needs the packed LayerNorm");
constexpr uint32_t T_AB = 0, T_H = 256, T_OUT = INPL ? 256 : 384;   // (T_AB: NAB buffers of 2 HS columns)
constexpr uint32_t IDESC8 = idesc_e4m3(256, 2 * HS), IDESC8Q = idesc_e4m3(256, D_);

#define ARRIVE_TC mbar_arrive_remote_relaxed
#ifdef PROBE   // clock64 stamps of CTA 0 (tile TILE_P) into peak + 32 (u64 slots): the chunk loop timeline
#define STAMP(k) do { if (blockIdx.x == 0) reinterpret_cast<long long*>(peak)[32 + (k)] = clock64(); } while (0)
#else
#define STAMP(k) do { } while (0)
#endif
#ifndef TILE_P
#define TILE_P 3
#endif
#define CH_P(c) ((c) >= TILE_P * NCH && (c) < (TILE_P + 1) * NCH)
#define MAXS 16

struct Bars {
  uint64_t wab_full[NW], ws_full[NWS], xn_full[XNB], ab_empty[NAB], h_full[NAB], out_empty;               // leader-only (both CTAs' completions)
  uint64_t wab_empty[NW], ws_empty[NWS], xn_empty[XNB], ex_done[NAB], sq_done[NAB], out_full;             // both CTAs (leader's multicast commits)
  uint64_t x_full[NXS], x_empty[NXS];
  uint32_t tmem;
};

DEVI uint32_t e4m3x4f(float a, float b, float c, float d) { return e4m3x2(a, b) | (e4m3x2(c, d) << 16); }

extern "C" __global__ void __launch_bounds__(512, 1)
tr8_fwd_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mwa, const __grid_constant__ CUtensorMap mwb,
              const __grid_constant__ CUtensorMap mws, const __grid_constant__ CUtensorMap mout, const float* __restrict__ gamma,
              const float* __restrict__ beta, const float* __restrict__ sc, float* __restrict__ peak, int tiles, float eps,
              const __nv_bfloat16* __restrict__ xg, __nv_bfloat16* __restrict__ yg) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  auto count = [&](int k) { return (tiles > k) ? (tiles - k + G - 1) / G : 0; };
  const int n_valid = count(cta), n_local = count(cta & ~1);
  const int nch = n_local * NCH, nps = n_local * NSL;
  auto tile_of = [&](int i) { return i < n_valid ? cta + i * G : (n_valid > 0 ? cta + (n_valid - 1) * G : 0); };
  auto load_x = [&](int stage, int row) {
    mbar_expect_tx(&B.x_full[stage], TILE);
    const uint32_t dst = su + O_X + stage * TILE;
#pragma unroll
    for (int cb = 0; cb < XB; ++cb)
#pragma unroll
      for (int h = 0; h < 2; ++h) tma_load_2d(dst + cb * 16384 + h * 8192, &mx, &B.x_full[stage], cb * 64, row + h * 64);
  };

  if (tid == 0) {
    for (int s = 0; s < NW; ++s) { mbar_init(&B.wab_full[s], 1); mbar_init(&B.wab_empty[s], 1); }
    for (int s = 0; s < NWS; ++s) { mbar_init(&B.ws_full[s], 1); mbar_init(&B.ws_empty[s], 1); }
    for (int s = 0; s < XNB; ++s) { mbar_init(&B.xn_full[s], 2); mbar_init(&B.xn_empty[s], 1); }
    for (int s = 0; s < NAB; ++s) {
      mbar_init(&B.ab_empty[s], 8); mbar_init(&B.h_full[s], PAIRSQ ? 16 : 8);
      mbar_init(&B.ex_done[s], 1); mbar_init(&B.sq_done[s], 1);
    }
    for (int s = 0; s < NXS; ++s) { mbar_init(&B.x_full[s], 1); mbar_init(&B.x_empty[s], 1); }
    mbar_init(&B.out_empty, LNH ? 16 : 8); mbar_init(&B.out_full, 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  pdl_wait();
  pdl_launch();
  if (tid == 0) {
    if (n_local > 0) {
      fence_proxy_async();
      load_x(0, tile_of(0) * ROWS);
    }
    prefetch_map(&mx); prefetch_map(&mwa); prefetch_map(&mwb); prefetch_map(&mws); prefetch_map(&mout);
  }
  if (tid < D_) {
    const float sx = sc[0];
    reinterpret_cast<float*>(sm + O_GB)[tid] = gamma[tid] * sx;
    reinterpret_cast<float*>(sm + O_GB)[D_ + tid] = beta[tid] * sx;
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    if (lane == 0) {
      for (int i = 1; i < n_local; ++i) {
        const int b = i % NXS;
        if (i >= NXS) mbar_wait(&B.x_empty[b], ((i / NXS) - 1) & 1);
        load_x(b, tile_of(i) * ROWS);
      }
    }
  } else if (warp == 3) {
    if (lane == 0) {
      const CUtensorMap* mab = leader ? &mwa : &mwb;
      int ca = 0, cs = 0;
      while (ca < nch || cs < nps) {
        if (ca < nch && (ca < NW || mbar_test(&B.wab_empty[ca % NW], ((ca / NW) - 1) & 1))) {
          const int s = ca % NW, j = ca & (NCH - 1);
          if (leader) mbar_expect_tx(&B.wab_full[s], 2 * WAB_SLOT);
#pragma unroll
          for (int kb = 0; kb < KB8; ++kb) tma_load_2d_2sm(su + O_WAB + s * WAB_SLOT + kb * WAB_KB, mab, &B.wab_full[s], kb * 64, j * HS);
          ++ca;
        }
        if (cs < nps && (cs < NWS || mbar_test(&B.ws_empty[cs % NWS], ((cs / NWS) - 1) & 1))) {
          const int s = cs % NWS, j2 = cs & (NSL - 1);
          if (leader) mbar_expect_tx(&B.ws_full[s], 2 * WS_SLOT);
          tma_load_2d_2sm(su + O_WS + s * WS_SLOT, &mws, &B.ws_full[s], j2 * 64, crank * (D_ / 2));
          ++cs;
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    if (leader) {
      if (warp == 1) {
        for (int c = 0; c < nch; ++c) {
          const int i = c / NCH, j = c & (NCH - 1), s = c % NAB, sw = c % NW;
          if (j == 0) mbar_wait_cl(&B.xn_full[i % XNB], (i / XNB) & 1);
          mbar_wait(&B.wab_full[sw], (c / NW) & 1);
          if (c >= NAB) {
            if (INPL) {
              if (PAIRSQ) mbar_wait_cl(&B.sq_done[((c / NAB) - 1) & 1], (((c / NAB) - 1) >> 1) & 1);   // the squeeze (of the pair) that read h in this buffer
              else mbar_wait_cl(&B.sq_done[s], ((c / NAB) - 1) & 1);
            } else mbar_wait_cl(&B.ab_empty[s], ((c / NAB) - 1) & 1);
          }
          if (lane == 0 && CH_P(c) && (c - TILE_P * NCH) < MAXS) STAMP(48 + (c - TILE_P * NCH));
          tc_fence_after();
          const uint64_t ad = desc_k128(su + O_XN + (i % XNB) * TILE8), bd = desc_k128(su + O_WAB + sw * WAB_SLOT);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < D_ / 32; ++ks)
              umma8_ss2(tmem + T_AB + s * (2 * HS), ad + (uint64_t)(((ks >> 2) * (ROWS * 128) + (ks & 3) * 32) >> 4),
                        bd + (uint64_t)(((ks >> 2) * WAB_KB + (ks & 3) * 32) >> 4), IDESC8, ks > 0 ? 1u : 0u);
            tc_commit2_mc(&B.ex_done[s], 3);
            tc_commit2_mc(&B.wab_empty[sw], 3);
            if (j == NCH - 1) tc_commit2_mc(&B.xn_empty[i % XNB], 3);
          }
          __syncwarp();
        }
      } else {
#if PAIRSQ
        for (int q = 0; q < nps; ++q) {                              // one squeeze per pair of chunks (K = 128)
          const int i = q / NSL, j2 = q & (NSL - 1), pb = q & 1, sw = q % NWS;
          mbar_wait(&B.ws_full[sw], (q / NWS) & 1);
          mbar_wait_cl(&B.h_full[pb], (q >> 1) & 1);
          if (lane == 0 && i == TILE_P && j2 < MAXS) STAMP(64 + j2);
          if (j2 == 0 && i >= 1) mbar_wait_cl(&B.out_empty, (i - 1) & 1);
          tc_fence_after();
          const uint64_t bd = desc_k128(su + O_WS + sw * WS_SLOT);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma8_ts2(tmem + T_OUT, INPL ? tmem + T_AB + (ks >> 1) * 128 + (ks & 1) * 8 : tmem + T_H + pb * 32 + ks * 8,
                        bd + (uint64_t)(ks * 2), IDESC8Q, (j2 > 0 || ks > 0) ? 1u : 0u);
            tc_commit2_mc(&B.sq_done[pb], 3);
            tc_commit2_mc(&B.ws_empty[sw], 3);
            if (j2 == NSL - 1) tc_commit2_mc(&B.out_full, 3);
          }
          __syncwarp();
        }
#else
        for (int q = 0; q < nch; ++q) {                              // one squeeze per chunk, on its HS bytes of the (128-byte-row) Ws slot
          const int i = q / NCH, j = q & (NCH - 1), s = q % NAB, sl = q / CPS, sw = sl % NWS, sub = q % CPS;
          mbar_wait(&B.ws_full[sw], (sl / NWS) & 1);
          mbar_wait_cl(&B.h_full[s], (q / NAB) & 1);
          if (lane == 0 && i == TILE_P && j < MAXS) STAMP(64 + j);
          if (j == 0 && i >= 1) mbar_wait_cl(&B.out_empty, (i - 1) & 1);
          tc_fence_after();
          const uint64_t bd = desc_k128(su + O_WS + sw * WS_SLOT);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < HS / 32; ++ks)
              umma8_ts2(tmem + T_OUT, INPL ? tmem + T_AB + s * (2 * HS) + ks * 8 : tmem + T_H + s * (HS / 4) + ks * 8,
                        bd + (uint64_t)((sub * HS + ks * 32) >> 4), IDESC8Q, (j > 0 || ks > 0) ? 1u : 0u);
            tc_commit2_mc(&B.sq_done[s], 3);
            if (sub == CPS - 1) tc_commit2_mc(&B.ws_empty[sw], 3);
            if (j == NCH - 1) tc_commit2_mc(&B.out_full, 3);
          }
          __syncwarp();
        }
#endif
      }
    }
  } else {
    setmaxnreg_inc<152>();
    const uint32_t lb = (uint32_t)(warp & 3) * 32, trow = tmem + (lb << 16);
    float pk = 0.f;
    auto sw = [&](int c) {
      const int s = c % NAB, u = c / NAB;
      mbar_wait(&B.ex_done[s], u & 1);
      tc_fence_after();
      if (lane == 0 && (warp & 3) == 0 && CH_P(c) && (c - TILE_P * NCH) < MAXS) STAMP(0 + (c - TILE_P * NCH));
      uint32_t hp[HS / 4];
      const uint32_t ab0 = trow + T_AB + s * (2 * HS);
      {
        uint32_t a[2][16], b[2][16];
        tmem_ld16(ab0, a[0]);
        tmem_ld16(ab0 + HS, b[0]);
        tmem_wait_ld();
        const float ch = 0.5f * sc[1], cb = sc[2];
        const f2 CH = mk2(ch, ch), CB = mk2(cb, cb);
#pragma unroll
        for (int q = 0; q < HS / 16; ++q) {
          if (q < HS / 16 - 1) {
            tmem_ld16(ab0 + (q + 1) * 16, a[(q + 1) & 1]);
            tmem_ld16(ab0 + HS + (q + 1) * 16, b[(q + 1) & 1]);
          }
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const f2 hh = mul2(mk2(__uint_as_float(a[q & 1][2 * k]), __uint_as_float(a[q & 1][2 * k + 1])), CH);   // a ca / 2
            const f2 t = (ABL & 1) ? hh : mk2(tanhf_approx(lo2(hh)), tanhf_approx(hi2(hh)));
            const f2 H = mul2(fma2(hh, t, hh), mul2(mk2(__uint_as_float(b[q & 1][2 * k]), __uint_as_float(b[q & 1][2 * k + 1])), CB));
            pk = fmaxf(pk, fmaxf(fabsf(lo2(H)), fabsf(hi2(H))));
            const uint32_t w = e4m3x2(lo2(H), hi2(H));
            if (k & 1) hp[q * 4 + (k >> 1)] |= w << 16; else hp[q * 4 + (k >> 1)] = w;
          }
          if (q < HS / 16 - 1) tmem_wait_ld();
        }
      }
      if (lane == 0 && (warp & 3) == 0 && CH_P(c) && (c - TILE_P * NCH) < MAXS) STAMP(16 + (c - TILE_P * NCH));
      tc_fence_before();
      __syncwarp();
      if (!INPL) {
        if (lane == 0) ARRIVE_TC(&B.ab_empty[s], 0);
        if (PAIRSQ) { if (u >= 2) mbar_wait(&B.sq_done[u & 1], ((u >> 1) - 1) & 1); }
        else if (u >= 1) mbar_wait(&B.sq_done[s], (u - 1) & 1);
        tc_fence_after();
      }
      {
        const uint32_t dst = INPL ? ab0 : (PAIRSQ ? trow + T_H + (u & 1) * 32 + s * 16 : trow + T_H + s * (HS / 4));
#if HS_ == 64
        tmem_st16(dst, hp);
#else
        tmem_st8(dst, hp);
#endif
      }
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) ARRIVE_TC(&B.h_full[PAIRSQ ? (u & 1) : s], 0);
      if (lane == 0 && (warp & 3) == 0 && CH_P(c) && (c - TILE_P * NCH) < MAXS) STAMP(32 + (c - TILE_P * NCH));
    };
    if (LNH ? warp >= 12 : ((warp >= 4 && warp < 8) || warp >= 12)) {
      for (int c = (LNH ? 0 : (warp >= 12 ? 1 : 0)); c < nch; c += (LNH ? 1 : 2)) sw(c);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) pk = fmaxf(pk, __shfl_xor_sync(0xffffffffu, pk, o));
      if (lane == 0 && pk > 0.f) atomicMax(reinterpret_cast<int*>(peak), __float_as_int(pk));
    } else {
      const int t2 = LNH ? tid - 128 : tid - 256;
      const int half = LNH ? (warp >= 8 ? 1 : 0) : 0, NT = LNH ? 256 : 128, BARID = LNH ? 3 : 1;
      constexpr int NHALF = LNH ? 2 : 1;
      float* red = reinterpret_cast<float*>(sm + O_RED);
      const uint32_t r = lb + lane;
      const uint32_t gb_u = su + O_GB;
      const float co = sc[3];
      auto ln = [&](int i) {
        const int b = i % XNB;
        if (t2 == 0 && i == TILE_P) STAMP(80);
        mbar_wait(&B.x_full[i % NXS], (i / NXS) & 1);
        if (i >= XNB) mbar_wait(&B.xn_empty[b], ((i / XNB) - 1) & 1);
        if (t2 == 0) tma_store_wait_read0();
        named_bar_sync(BARID, NT);
        const uint32_t xb = su + O_X + (i % NXS) * TILE, xnb = su + O_XN + b * TILE8;
        if (LNP) {                                 // packed-pair LayerNorm: three passes over the shared row, all arithmetic on f32x2 pairs
          auto pr = [](uint32_t w) { return mk2u(w << 16, w & 0xffff0000u); };
          f2 sa = mk2(0.f, 0.f), sb = mk2(0.f, 0.f);
#pragma unroll
          for (int qq = 0; qq < D_ / 8 / NHALF; ++qq) {
            const int q = half * (D_ / 8 / NHALF) + qq;
            const uint4 t = lds128(xb + (q >> 3) * 16384 + sw128(r, q & 7));
            sa = add2(sa, pr(t.x)); sb = add2(sb, pr(t.y)); sa = add2(sa, pr(t.z)); sb = add2(sb, pr(t.w));
          }
          float psum = (lo2(sa) + hi2(sa)) + (lo2(sb) + hi2(sb));
          if (LNH) {
            red[half * 128 + r] = psum;
            named_bar_sync(BARID, NT);
            psum = red[r] + red[128 + r];
          }
          const float mean = psum * (1.f / D_);
          const f2 nm2 = mk2(-mean, -mean);
          f2 qa = mk2(0.f, 0.f), qb = mk2(0.f, 0.f);
#pragma unroll
          for (int qq = 0; qq < D_ / 8 / NHALF; ++qq) {
            const int q = half * (D_ / 8 / NHALF) + qq;
            const uint4 t = lds128(xb + (q >> 3) * 16384 + sw128(r, q & 7));
            f2 d = add2(pr(t.x), nm2); qa = fma2(d, d, qa);
            d = add2(pr(t.y), nm2); qb = fma2(d, d, qb);
            d = add2(pr(t.z), nm2); qa = fma2(d, d, qa);
            d = add2(pr(t.w), nm2); qb = fma2(d, d, qb);
          }
          float pq = (lo2(qa) + hi2(qa)) + (lo2(qb) + hi2(qb));
          if (LNH) {
            red[256 + half * 128 + r] = pq;
            named_bar_sync(BARID, NT);
            pq = red[256 + r] + red[256 + 128 + r];
          }
          const float rs = rsqrtf(pq * (1.f / D_) + eps);
          const f2 RS2 = mk2(rs, rs), NMR2 = mk2(-mean * rs, -mean * rs);
#pragma unroll
          for (int QQ = 0; QQ < D_ / 16 / NHALF; ++QQ) {      // xn8 chunk Q = columns 16 Q .. 16 Q + 15 = x chunks 2 Q, 2 Q + 1
            const int Q = half * (D_ / 16 / NHALF) + QQ;
            const uint4 ta = lds128(xb + ((2 * Q) >> 3) * 16384 + sw128(r, (2 * Q) & 7)), tb = lds128(xb + ((2 * Q + 1) >> 3) * 16384 + sw128(r, (2 * Q + 1) & 7));
            const uint32_t w[8] = {ta.x, ta.y, ta.z, ta.w, tb.x, tb.y, tb.z, tb.w};
            uint32_t o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const int col = 16 * Q + 4 * k;
              const float2 g01 = lds64f(gb_u + col * 4), g23 = lds64f(gb_u + (col + 2) * 4);
              const float2 b01 = lds64f(gb_u + D_ * 4 + col * 4), b23 = lds64f(gb_u + D_ * 4 + (col + 2) * 4);
              const f2 G0 = mk2(g01.x, g01.y), G1 = mk2(g23.x, g23.y);
              const f2 y0 = fma2(pr(w[2 * k]), mul2(G0, RS2), fma2(G0, NMR2, mk2(b01.x, b01.y)));
              const f2 y1 = fma2(pr(w[2 * k + 1]), mul2(G1, RS2), fma2(G1, NMR2, mk2(b23.x, b23.y)));
              o[k] = e4m3x2(lo2(y0), hi2(y0)) | (e4m3x2(lo2(y1), hi2(y1)) << 16);
            }
            sts128(xnb + (Q >> 3) * 16384 + sw128(r, Q & 7), make_uint4(o[0], o[1], o[2], o[3]));
          }
        } else
        if constexpr (D_ == 128) {
          uint32_t v[64];
#pragma unroll
          for (int cb = 0; cb < 2; ++cb)
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const uint4 t = lds128(xb + cb * 16384 + sw128(r, q));
              v[cb * 32 + q * 4 + 0] = t.x; v[cb * 32 + q * 4 + 1] = t.y; v[cb * 32 + q * 4 + 2] = t.z; v[cb * 32 + q * 4 + 3] = t.w;
            }
          const float mean = (ABL & 2) ? 0.f : tree32([&](int l) {
            return (bf16lo(v[2 * l]) + bf16hi(v[2 * l])) + (bf16lo(v[2 * l + 1]) + bf16hi(v[2 * l + 1]));
          }) * (1.f / D_);
          const float rs = (ABL & 2) ? 1.f : rsqrtf(tree32([&](int l) {
            float acc = 0.f, d;
            d = bf16lo(v[2 * l]) - mean; acc += d * d;
            d = bf16hi(v[2 * l]) - mean; acc += d * d;
            d = bf16lo(v[2 * l + 1]) - mean; acc += d * d;
            d = bf16hi(v[2 * l + 1]) - mean; acc += d * d;
            return acc;
          }) * (1.f / D_) + eps);
#pragma unroll
          for (int Q = 0; Q < 8; ++Q) {            // xn8 chunk Q = columns 16 Q .. 16 Q + 15 (words v[8 Q .. 8 Q + 7])
            uint32_t o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const int col = 16 * Q + 4 * k;
              const uint32_t w0 = v[8 * Q + 2 * k], w1 = v[8 * Q + 2 * k + 1];
              const float2 g01 = lds64f(gb_u + col * 4), g23 = lds64f(gb_u + (col + 2) * 4);
              const float2 b01 = lds64f(gb_u + D_ * 4 + col * 4), b23 = lds64f(gb_u + D_ * 4 + (col + 2) * 4);
              o[k] = e4m3x4f((bf16lo(w0) - mean) * rs * g01.x + b01.x, (bf16hi(w0) - mean) * rs * g01.y + b01.y,
                             (bf16lo(w1) - mean) * rs * g23.x + b23.x, (bf16hi(w1) - mean) * rs * g23.y + b23.y);
            }
            sts128(xnb + sw128(r, Q), make_uint4(o[0], o[1], o[2], o[3]));
          }
        } else {                                   // D = 256: three passes over the shared row (mean, variance, quantize), 32 x 16-byte chunks of 8 columns
          float s0 = 0.f, s1 = 0.f;
#pragma unroll
          for (int q = 0; q < 32; ++q) {
            const uint4 t = lds128(xb + (q >> 3) * 16384 + sw128(r, q & 7));
            s0 += (bf16lo(t.x) + bf16hi(t.x)) + (bf16lo(t.y) + bf16hi(t.y));
            s1 += (bf16lo(t.z) + bf16hi(t.z)) + (bf16lo(t.w) + bf16hi(t.w));
          }
          const float mean = (ABL & 2) ? 0.f : (s0 + s1) * (1.f / D_);
          float q0 = 0.f, q1 = 0.f;
#pragma unroll
          for (int q = 0; q < 32; ++q) {
            const uint4 t = lds128(xb + (q >> 3) * 16384 + sw128(r, q & 7));
            float d;
            d = bf16lo(t.x) - mean; q0 += d * d; d = bf16hi(t.x) - mean; q1 += d * d;
            d = bf16lo(t.y) - mean; q0 += d * d; d = bf16hi(t.y) - mean; q1 += d * d;
            d = bf16lo(t.z) - mean; q0 += d * d; d = bf16hi(t.z) - mean; q1 += d * d;
            d = bf16lo(t.w) - mean; q0 += d * d; d = bf16hi(t.w) - mean; q1 += d * d;
          }
          const float rs = (ABL & 2) ? 1.f : rsqrtf((q0 + q1) * (1.f / D_) + eps);
#pragma unroll
          for (int Q = 0; Q < 16; ++Q) {            // xn8 chunk Q = columns 16 Q .. 16 Q + 15 = x chunks 2 Q, 2 Q + 1
            const uint4 ta = lds128(xb + ((2 * Q) >> 3) * 16384 + sw128(r, (2 * Q) & 7)), tb = lds128(xb + ((2 * Q + 1) >> 3) * 16384 + sw128(r, (2 * Q + 1) & 7));
            const uint32_t w[8] = {ta.x, ta.y, ta.z, ta.w, tb.x, tb.y, tb.z, tb.w};
            uint32_t o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const int col = 16 * Q + 4 * k;
              const float2 g01 = lds64f(gb_u + col * 4), g23 = lds64f(gb_u + (col + 2) * 4);
              const float2 b01 = lds64f(gb_u + D_ * 4 + col * 4), b23 = lds64f(gb_u + D_ * 4 + (col + 2) * 4);
              o[k] = e4m3x4f((bf16lo(w[2 * k]) - mean) * rs * g01.x + b01.x, (bf16hi(w[2 * k]) - mean) * rs * g01.y + b01.y,
                             (bf16lo(w[2 * k + 1]) - mean) * rs * g23.x + b23.x, (bf16hi(w[2 * k + 1]) - mean) * rs * g23.y + b23.y);
            }
            sts128(xnb + (Q >> 3) * 16384 + sw128(r, Q & 7), make_uint4(o[0], o[1], o[2], o[3]));
          }
        }
        fence_proxy_async();
        named_bar_sync(BARID, NT);
        if (t2 == 0) {
          mbar_arrive_remote_cta(&B.xn_full[b], 0);
          if (i == TILE_P) STAMP(81);
          if (RESG) mbar_arrive(&B.x_empty[i % NXS]);
        }
      };
      auto epi = [&](int i) {
        const int b = i % NXS;
        if (t2 == 0 && i == TILE_P) STAMP(82);
        if (RESG) {                                // residual from global (L2), y stored straight from registers
          mbar_wait(&B.out_full, i & 1);
          tc_fence_after();
          const size_t row = (size_t)tile_of(i) * ROWS + r;
          const __nv_bfloat16* xr = xg + row * D_;
          __nv_bfloat16* yr = yg + row * D_;
          const int cc0 = half * (D_ / 32 / NHALF);
          uint4 xv[4];
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) xv[qq] = ldg128(xr + cc0 * 32 + qq * 8);
#pragma unroll
          for (int ccl = 0; ccl < D_ / 32 / NHALF; ++ccl) {
            const int cc = cc0 + ccl;
            uint32_t acc[32];
            tmem_ld32(trow + T_OUT + cc * 32, acc);
            uint4 xw_next[4];
            if (ccl + 1 < D_ / 32 / NHALF) {         // the next group's residual in flight while this one is combined and stored
#pragma unroll
              for (int qq = 0; qq < 4; ++qq) xw_next[qq] = ldg128(xr + (cc + 1) * 32 + qq * 8);
            }
            tmem_wait_ld();
#pragma unroll
            for (int qq = 0; qq < 4; ++qq) {
              const uint32_t xw[4] = {xv[qq].x, xv[qq].y, xv[qq].z, xv[qq].w};
              uint32_t o[4];
#pragma unroll
              for (int k = 0; k < 4; ++k)
                o[k] = pack_bf16(fmaf(__uint_as_float(acc[qq * 8 + 2 * k]), co, bf16lo(xw[k])),
                                 fmaf(__uint_as_float(acc[qq * 8 + 2 * k + 1]), co, bf16hi(xw[k])));
              if (i < n_valid) stg128(yr + cc * 32 + qq * 8, make_uint4(o[0], o[1], o[2], o[3]));
            }
            if (ccl + 1 < D_ / 32 / NHALF) {
#pragma unroll
              for (int qq = 0; qq < 4; ++qq) xv[qq] = xw_next[qq];
            }
          }
          tc_fence_before();
          __syncwarp();
          if (t2 == 0 && i == TILE_P) STAMP(83);
          if (lane == 0) ARRIVE_TC(&B.out_empty, 0);
          return;
        }
        mbar_wait(&B.out_full, i & 1);
        tc_fence_after();
        const uint32_t xb = su + O_X + b * TILE;
#pragma unroll
        for (int ccl = 0; ccl < D_ / 32 / NHALF; ++ccl) {
          const int cc = half * (D_ / 32 / NHALF) + ccl;
          uint32_t acc[32];
          tmem_ld32(trow + T_OUT + cc * 32, acc);
          tmem_wait_ld();
          const int cb = cc >> 1;
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            const int q = (cc & 1) * 4 + qq;
            const uint32_t ad = xb + cb * 16384 + sw128(r, q);
            const uint4 xv = (ABL & 4) ? make_uint4(0, 0, 0, 0) : lds128(ad);
            const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
            uint32_t o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
              o[k] = pack_bf16(fmaf(__uint_as_float(acc[qq * 8 + 2 * k]), co, bf16lo(xw[k])),
                               fmaf(__uint_as_float(acc[qq * 8 + 2 * k + 1]), co, bf16hi(xw[k])));
            sts128(ad, make_uint4(o[0], o[1], o[2], o[3]));
          }
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) ARRIVE_TC(&B.out_empty, 0);
        fence_proxy_async();
        named_bar_sync(BARID, NT);
        if (t2 == 0) {
          const int row0 = tile_of(i) * ROWS;
          if (i < n_valid) {
#pragma unroll
            for (int cb = 0; cb < XB; ++cb)
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
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
