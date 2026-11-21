// apb_pair_fwd32_sm100.cuh -- AttentionPairBias fp32 pair -> bias on sm_100a (tcgen05 kind::tf32), inference, 2026-11-17 (part 6). Built
// as a cubin by ../sm100/pair_fwd32.cu.
//
//   bias[h, i, p] = Wf[h] . LN(pair[i, k(p)])  fp32 head-major, keys in the TF32 core's key_perm order (``perm``: position p of a group of 8
//                   holds key p - r + 4 (r % 2) + r / 2, r = p % 8), ``neg`` on masked keys -- what apb_rows.cu pair_bias32_fwd writes.
//
// The fp32 FMA pass (pair_bias32_fwd) moves ~4.2 TB/s of pair and is 80 % of the 8-head d_pair 384 inference step; the mma.sync TF32 pass
// (pair_bias32_tc) is slower at 8 heads. This is apb_pair_fwd_sm100.cuh's schedule on fp32 tiles: per 32 pair columns (one 128-B
// swizzled TMA box of 128 keys) two kind::tf32 MMAs issued by one thread --
//   X  X^T  (M = N = 128, K = 32)   -> the diagonal is every row's sum of squares
//   X  B^T  (N = 32: Wf's heads rounded to TF32 to nearest, then a row of ones) -> Wf . x per head and the row sum
// a unit = 64 columns (two boxes, 32 KB), five stages, a loader warp that reloads a stage as soon as its MMAs commit, two TMEM accumulators
// (the epilogue of tile n under the MMAs of tile n + 1). The MMAs read x truncated to TF32: the scale of that rounding (x (1 - e), e <
// 2^-10) leaves the mean, the sum of squares and Wf . x scaled alike, which LN's normalisation cancels to first order; the rest is a
// rounding per element in the sums (as the TF32 Linear of the reference). Wf . LN(x) = rstd (Wf . x - mean sum_c Wf).
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../sm100/sm100.cuh"

namespace pf32 {
using namespace s100;

constexpr int TR = 128, KC = 64, NBOX = KC / 32, BOXB = TR * 128, UB = TR * KC * 4, NB32 = 32, ONES = 24, WFBOX = NB32 * 128;
template <int DPW> __host__ __device__ constexpr int stages() { return DPW >= 384 ? 5 : 6; }
template <int DPW> __host__ __device__ constexpr int smem_bytes() { return stages<DPW>() * UB + (DPW / 32) * WFBOX + 1024; }
constexpr int NSMAX = 6;
struct Misc {
  uint64_t full[NSMAX], udone[NSMAX], done[2], afree[2];
  float sw[NB32];
  uint32_t tmem;
};
__device__ __forceinline__ float tf32_rna(float v) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v)); return __uint_as_float(r); }

// WF: Wf fp32 [NH, DPW] (exp2 units as packed); OUT fp32 [NH, L, L]
template <int NH, int DPW>
__device__ __forceinline__ void pair_fwd32_body(const CUtensorMap& tmz, const float* __restrict__ WF, const bool* __restrict__ MASK,
    float* __restrict__ OUT, int L, float eps, float neg, int perm) {
  constexpr int NS = stages<DPW>(), UPT = DPW / KC, ACC = TR + NB32;
  static_assert(NH <= ONES && 2 * ACC <= 512 && NS <= NSMAX && DPW % KC == 0, "heads / TMEM columns / stages");
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* wfs = smem + NS * UB;
  Misc& B = *reinterpret_cast<Misc*>(smem + NS * UB + (DPW / 32) * WFBOX);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint32_t su = smem_u32(smem);
  const int ntj = (L + TR - 1) / TR;
  const long ntiles = (long)L * ntj;
  const long mine = blockIdx.x < ntiles ? (ntiles - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
  auto issue = [&](long u) {
    const long n = u / UPT;
    if (n >= mine) return;
    const long t = blockIdx.x + n * gridDim.x;
    const long row0 = (t / ntj) * L + (t % ntj) * TR;
    const int s = (int)(u % NS), c0 = (int)(u % UPT) * KC;
    mbar_expect_tx(&B.full[s], (uint32_t)UB);
#pragma unroll
    for (int b = 0; b < NBOX; ++b) tma_load_2d(su + s * UB + b * BOXB, &tmz, &B.full[s], c0 + 32 * b, (int)row0);
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
  // Wf's rows (TF32, to nearest) into the B boxes (row r, column c at box c / 32, 128-B swizzle), heads NH.. zero, row ONES ones
  constexpr int RPW = (NH + 5) / 6, CPL = DPW / 32;
  float raw[RPW][CPL];
#pragma unroll
  for (int k = 0; k < RPW; ++k)
#pragma unroll
    for (int m = 0; m < CPL; ++m) {
      const int r = warp + 6 * k;
      raw[k][m] = r >= NH ? 0.f : WF[(long)r * DPW + lane + 32 * m];
    }
  if (threadIdx.x == 0)
    for (int s = 0; s < NS; ++s) issue(s);
  auto wf_put = [&](int r, int c, float v) {
    *reinterpret_cast<float*>(wfs + (c >> 5) * WFBOX + sw128((uint32_t)r, (uint32_t)((c & 31) >> 2)) + (c & 3) * 4) = v;
  };
#pragma unroll
  for (int k = 0; k < RPW; ++k) {
    const int r = warp + 6 * k;
    float s = 0.f;
#pragma unroll
    for (int m = 0; m < CPL; ++m) {
      const float v = tf32_rna(raw[k][m]);
      s += v;
      if (r < NH) wf_put(r, lane + 32 * m, v);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0 && r < NH) B.sw[r] = s;
  }
  for (int r = NH + warp; r < NB32; r += 6)
    for (int c = lane; c < DPW; c += 32) wf_put(r, c, r == ONES ? 1.f : 0.f);
  fence_proxy_async();
  __syncthreads();
  const uint32_t tmem = B.tmem;
  if (warp == 4) {
    if (lane == 0) {  // the MMA thread
      constexpr uint32_t I_G = idesc_tf32(TR, TR), I_W = idesc_tf32(TR, NB32);
      for (long u = 0; u < mine * UPT; ++u) {
        const long n = u / UPT;
        const int s = (int)(u % NS), a = (int)(n & 1), p = (int)(u % UPT);
        mbar_wait(&B.full[s], (uint32_t)((u / NS) & 1));
        if (p == 0 && n >= 2) mbar_wait(&B.afree[a], (uint32_t)(((n - 2) >> 1) & 1));
        tc_fence_after();
#pragma unroll
        for (int b = 0; b < NBOX; ++b) {
          const uint64_t dx = desc_k128(su + s * UB + b * BOXB), dw = desc_k128(su + NS * UB + (p * NBOX + b) * WFBOX);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {  // K = 8 fp32 = 32 B per step
            umma_ss_tf32(tmem + a * ACC, dx + (uint64_t)(2 * ks), dx + (uint64_t)(2 * ks), I_G, (p | b | ks) ? 1u : 0u);
            umma_ss_tf32(tmem + a * ACC + TR, dx + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_W, (p | b | ks) ? 1u : 0u);
          }
        }
        tc_commit(&B.udone[s]);
        if (p == UPT - 1) tc_commit(&B.done[a]);
      }
    }
  } else if (warp == 5) {  // the loader
    if (lane == 0)
      for (long u = 0; u < mine * UPT - NS; ++u) {
        mbar_wait(&B.udone[u % NS], (uint32_t)((u / NS) & 1));
        issue(u + NS);
      }
  } else {  // the epilogue: thread = key r of the tile
    float sw[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) sw[h] = B.sw[h];
    const int r = 32 * warp + lane;
    const uint32_t trow = tmem + ((uint32_t)(32 * warp) << 16);
    for (long n = 0; n < mine; ++n) {
      const long t = blockIdx.x + n * gridDim.x;
      const int a = (int)(n & 1);
      const int i = (int)(t / ntj), j0 = (int)(t % ntj) * TR, j = j0 + r;
      mbar_wait(&B.done[a], (uint32_t)((n >> 1) & 1));
      tc_fence_after();
      uint32_t c[32], gd[32];
      tmem_ld32(trow + a * ACC + TR, c);
      tmem_ld32(trow + a * ACC + 32 * warp, gd);
      tc_fence_before();
      if (lane == 0) mbar_arrive(&B.afree[a]);
      uint32_t ssb = gd[0];
#pragma unroll
      for (int k = 1; k < 32; ++k) ssb = lane == k ? gd[k] : ssb;
      const float mean = __uint_as_float(c[ONES]) * (1.f / DPW);
      const float rstd = rsqrtf(fmaxf(__uint_as_float(ssb) * (1.f / DPW) - mean * mean, 0.f) + eps);
      if (j >= L) continue;
      const bool keep = MASK == nullptr || MASK[j];
      const int pos = perm ? (j & ~7) + 2 * (j & 3) + ((j >> 2) & 1) : j;  // key j's position in key_perm order
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const float v = rstd * (__uint_as_float(c[h]) - mean * sw[h]);
        OUT[((long)h * L + i) * L + pos] = keep ? v : neg;
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

}  // namespace pf32
