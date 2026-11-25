// apb_pair_fwd_sm100.cuh -- AttentionPairBias pair -> bias on sm_100a (tcgen05), 2026-11-17. Built as a cubin by ../sm100/pair_fwd.cu.
//
//   bias[h, i, j] = Wf[h] . LN(pair[i, j])  (LN without affine: its weight is folded into Wf), ``neg`` on masked keys; bf16 head-major.
//
// The shared mma.sync pass (apb_rows.cu pair_bias_fwd_k) moves 4.1-4.4 TB/s of pair on this card, time ~ 1 / SMs, while a load-only
// ring moves 6.2-6.6 TB/s (L2 flushed; the card's rate, reached already on 74 SMs): an SM's rate is (bytes it has in flight) /
// (load latency + the time a landed tile waits in its stage), and that pass keeps 128-144 KB in flight with a tile's whole compute
// (40 mma.sync per warp, 24 of them the row statistics; the planes staged through 2048 scalar smem stores and loads) as the wait.
// Here a unit's whole product is two tcgen05 instructions per 16 pair columns, issued by one thread, and its stage reloads as soon
// as those have committed (a loader thread; 192 KB in flight):
//   X  X^T  (M = N = 128 rows of the tile, K = d_pair)   -> the diagonal is every row's sum of squares
//   X  B^T  (N = 32: Wf's heads, then a row of ones)      -> Wf . x per head and the row sum
// both exact in fp32 for the statistics (bf16 products), as the Gram mma of the shared pass. Wf . LN(x) = rstd (Wf . x - mean sum_c Wf).
// A tile = 128 consecutive keys of one query row; a unit = KC of its columns (the tile, or half of it at d_pair 512), brought by
// KC / 64 TMA boxes (64 columns, 128-B swizzle) into a ring of NS stages; a tile's units accumulate into one of two TMEM
// accumulators, so the epilogue of tile n (4 warps: a thread per key reads its row from TMEM, forms the statistics and stores its NH
// values -- a warp instruction writes 64 contiguous bytes of one head's plane) runs under the MMAs of tile n + 1.
// Wf: bf16 as given, or formed here from the parameters (``LPW`` given: Wf = bf16(bf16(to_bias.weight) ln_pair.weight ws), prep's
// arithmetic), so the pass needs no pack launch before it.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../sm100/sm100.cuh"

#ifndef PF_DBG
#define PF_DBG 0
#endif
#ifndef PF_NS128
#define PF_NS128 4
#endif
namespace pf100 {
using namespace s100;
using bf = __nv_bfloat16;

constexpr int TR = 128;                                                                     // keys per tile (the MMA's M)
template <int DPW> __host__ __device__ constexpr int kc() { return DPW == 512 ? 256 : DPW; }        // columns per unit
template <int DPW> __host__ __device__ constexpr int stages() { return DPW == 128 ? PF_NS128 : DPW == 384 ? 2 : 3; }
template <int DPW> __host__ __device__ constexpr int unit_bytes() { return TR * kc<DPW>() * 2; }
constexpr int NB32 = 32, ONES = 24, WFBOX = NB32 * 128, NSMAX = 6;                          // B rows (heads, then ones at row 24); a Wf box
template <int DPW> __host__ __device__ constexpr int smem_bytes() { return stages<DPW>() * unit_bytes<DPW>() + (DPW / 64) * WFBOX + 1024; }
struct Misc {
  uint64_t full[NSMAX], udone[NSMAX], done[2], afree[2];
  float sw[NB32];
  uint32_t tmem;
};

// WSRC: Wf bf16 [NH, DPW] (LPW null), or to_bias.weight [NH, DPW] in bf16 (wfp32 0) / fp32 (wfp32 1) with LPW = ln_pair.weight
template <int NH, int DPW>
__device__ __forceinline__ void pair_fwd_tc_body(const CUtensorMap& tmz, const void* __restrict__ WSRC,
    const float* __restrict__ LPW, float ws, int wfp32, const bool* __restrict__ MASK, bf* __restrict__ OUT, float2* __restrict__ ST,
    float* __restrict__ BU, int L, float eps, float neg, int NB) {
  constexpr int NS = stages<DPW>(), KC = kc<DPW>(), UB = unit_bytes<DPW>(), UPT = DPW / KC, NBOX = KC / 64, BOXB = TR * 128, ACC = TR + NB32;
  static_assert(NH <= ONES && 2 * ACC <= 512 && NS <= NSMAX, "heads / TMEM columns / stages");
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* wfs = smem + NS * UB;
  Misc& B = *reinterpret_cast<Misc*>(smem + NS * UB + (DPW / 64) * WFBOX);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint32_t su = smem_u32(smem);
  if (PF_DBG & 32) return;
  const int ntj = (L + TR - 1) / TR;
  const long ntiles = (long)NB * L * ntj;
  const long mine = blockIdx.x < ntiles ? (ntiles - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;   // this block's tiles: blockIdx + n grid
  auto issue = [&](long u) {                                                // unit u of this block into stage u % NS
    const long n = u / UPT;
    if (n >= mine) return;
    const long t = blockIdx.x + n * gridDim.x;
    const long row0 = (t / ntj) * L + (t % ntj) * TR;                       // a tile past L runs into the next rows (not stored); past the end: zeros
    const int s = (int)(u % NS), c0 = (int)(u % UPT) * KC;
    mbar_expect_tx(&B.full[s], (uint32_t)UB);
#pragma unroll
    for (int b = 0; b < NBOX; ++b) tma_load_2d(su + s * UB + b * BOXB, &tmz, &B.full[s], c0 + 64 * b, (int)row0);
  };
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.udone[s], 1); }
    for (int a = 0; a < 2; ++a) { mbar_init(&B.done[a], 1); mbar_init(&B.afree[a], 4); }
    fence_barrier_init();
  }
  if (warp == 0) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (PF_DBG & 64) { tc_fence_before(); __syncthreads(); if (warp == 0) { tc_fence_after(); tmem_dealloc(B.tmem, 512); } return; }
  // Wf's rows into the B boxes (row r, column c at box c / 64, 128-B swizzle), heads NH.. zero, row ONES ones; sum_c Wf per head
  // (every global read is issued before the first is used: after an L2 flush each row is a round trip to memory)
  constexpr int RPW = (NH + 5) / 6, CPL = DPW / 32;
  float raw[RPW][CPL], lp[CPL];
#pragma unroll
  for (int m = 0; m < CPL; ++m) lp[m] = LPW != nullptr ? LPW[lane + 32 * m] : 1.f;
#pragma unroll
  for (int k = 0; k < RPW; ++k)
#pragma unroll
    for (int m = 0; m < CPL; ++m) {
      const int r = warp + 6 * k;
      const long e = (long)r * DPW + lane + 32 * m;
      raw[k][m] = r >= NH ? 0.f : (LPW != nullptr && wfp32) ? static_cast<const float*>(WSRC)[e] : __bfloat162float(static_cast<const bf*>(WSRC)[e]);
    }
  if (threadIdx.x == 0)                                                     // after Wf's reads: they would queue behind 148 SMs' tiles
    for (int s = 0; s < NS; ++s) issue(s);
  auto wf_put = [&](int r, int c, bf vb) {
    *reinterpret_cast<bf*>(wfs + (c >> 6) * WFBOX + sw128((uint32_t)r, (uint32_t)((c & 63) >> 3)) + (c & 7) * 2) = vb;
  };
#pragma unroll
  for (int k = 0; k < RPW; ++k) {
    const int r = warp + 6 * k;
    float s = 0.f;
#pragma unroll
    for (int m = 0; m < CPL; ++m) {
      const float v = LPW != nullptr ? __bfloat162float(__float2bfloat16_rn(raw[k][m])) * lp[m] * ws : raw[k][m];
      const bf vb = __float2bfloat16_rn(v);
      s += __bfloat162float(vb);
      if (r < NH) wf_put(r, lane + 32 * m, vb);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0 && r < NH) B.sw[r] = s;
  }
  for (int r = NH + warp; r < NB32; r += 6)
    for (int c = lane; c < DPW; c += 32) wf_put(r, c, __float2bfloat16_rn(r == ONES ? 1.f : 0.f));
  fence_proxy_async();
  __syncthreads();
  const uint32_t tmem = B.tmem;
  if (PF_DBG & 128) { tc_fence_before(); __syncthreads(); if (warp == 0) { tc_fence_after(); tmem_dealloc(tmem, 512); } return; }
  if (warp == 4) {
    if (lane == 0) {                                                        // the MMA thread
      constexpr uint32_t I_G = idesc_bf16(TR, TR), I_W = idesc_bf16(TR, NB32);
      for (long u = 0; u < mine * UPT; ++u) {
        const long n = u / UPT;
        const int s = (int)(u % NS), a = (int)(n & 1), p = (int)(u % UPT);
        mbar_wait(&B.full[s], (uint32_t)((u / NS) & 1));
        if (p == 0 && n >= 2) mbar_wait(&B.afree[a], (uint32_t)(((n - 2) >> 1) & 1));   // the epilogue of tile n - 2 has read this accumulator
        tc_fence_after();
#pragma unroll
        for (int b = 0; b < NBOX; ++b) {
          const uint64_t dx = desc_k128(su + s * UB + b * BOXB), dw = desc_k128(su + NS * UB + (p * NBOX + b) * WFBOX);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            if (!(PF_DBG & 1)) umma_ss(tmem + a * ACC, dx + (uint64_t)(2 * ks), dx + (uint64_t)(2 * ks), I_G, (p | b | ks) ? 1u : 0u);
            umma_ss(tmem + a * ACC + TR, dx + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_W, (p | b | ks) ? 1u : 0u);
          }
        }
        tc_commit(&B.udone[s]);
        if (p == UPT - 1) tc_commit(&B.done[a]);
      }
    }
  } else if (warp == 5) {                                                   // the loader (a warp of its own: a thread spinning beside the
    if (lane == 0)                                                          // MMA thread in one warp would take turns with it)
      for (long u = 0; u < mine * UPT - NS; ++u) {
        mbar_wait(&B.udone[u % NS], (uint32_t)((u / NS) & 1));
        issue(u + NS);
      }
  } else {                                                                  // the epilogue: thread = key r of the tile
    float sw[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) sw[h] = B.sw[h];
    const int r = 32 * warp + lane;
    const uint32_t trow = tmem + ((uint32_t)(32 * warp) << 16);
    for (long n = 0; n < mine; ++n) {
      const long t = blockIdx.x + n * gridDim.x;
      const int a = (int)(n & 1);
      const int ig = (int)(t / ntj), j0 = (int)(t % ntj) * TR, bb = ig / L, i = ig - bb * L, j = j0 + r;
      mbar_wait(&B.done[a], (uint32_t)((n >> 1) & 1));
      tc_fence_after();
      uint32_t c[32], gd[32];
      if (PF_DBG & 8) { tc_fence_before(); if (lane == 0) mbar_arrive(&B.afree[a]); continue; }
      tmem_ld32(trow + a * ACC + TR, c);
      if (PF_DBG & 16) { for (int k = 0; k < 32; ++k) gd[k] = c[k]; } else
      tmem_ld32(trow + a * ACC + 32 * warp, gd);
      tc_fence_before();
      if (lane == 0) mbar_arrive(&B.afree[a]);
      uint32_t ssb = gd[0];
#pragma unroll
      for (int k = 1; k < 32; ++k) ssb = lane == k ? gd[k] : ssb;
      const float mean = __uint_as_float(c[ONES]) * (1.f / DPW);
      const float rstd = rsqrtf(fmaxf(__uint_as_float(ssb) * (1.f / DPW) - mean * mean, 0.f) + eps);
      if (j >= L) continue;
      const bool keep = MASK == nullptr || MASK[(long)bb * L + j];
      if (ST != nullptr) ST[(long)ig * L + j] = make_float2(mean, rstd);
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const float v = rstd * (__uint_as_float(c[h]) - mean * sw[h]);
        const long o = (((long)h * NB + bb) * L + i) * L + j;
        if (BU != nullptr) BU[o] = v;
        if (!(PF_DBG & 2) || h == 0) OUT[o] = __float2bfloat16_rn(keep ? v : neg);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

template <int NH, int DPW>
__global__ void __launch_bounds__(192, 1) pair_fwd_tc_k(const __grid_constant__ CUtensorMap tmz, const void* __restrict__ WSRC,
    const float* __restrict__ LPW, float ws, int wfp32, const bool* __restrict__ MASK, bf* __restrict__ OUT, float2* __restrict__ ST,
    float* __restrict__ BU, int L, float eps, float neg, int NB) {
  pair_fwd_tc_body<NH, DPW>(tmz, WSRC, LPW, ws, wfp32, MASK, OUT, ST, BU, L, eps, neg, NB);
}

}  // namespace pf100
