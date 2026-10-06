// mod_bwd_tf32.cu — the fp32 backward of the SWA atom block's hoisted adaLN modulation mod = silu(c) Wmod^T on sm_100a, TF32 tensor cores
// (the fp32 counterpart of mod_bwd.cu's dc kernel):
//   dc = silu'(c) (g Wmod),  silu'(x) = s (1 + x (1 - s)), s = 1 / (1 + exp(-x))       g = d mod [R, 768] fp32, c [R, 128] fp32
// and SC = silu(c) [R, 128] fp32, the operand of dWmod = g^T silu(c), which the host runs as a cuBLAS TF32 GEMM (a weight gradient).
// Transposed (lane = channel): dc^T [128 ch][128 rows] = Wmod^T g^T, M = N = 128, K = 768 in 24 slices of 32, kind::tf32: A = the Wmod^T
// slice [128 ch][32] (Wmod^T rounded on the host), B = the g slice [128 rows][32] (K-major, 128-B swizzle), rounded to TF32 in place by the
// eight epilogue warps while they would otherwise wait (the MMA truncates what it is given); a 4-stage ring of (A | B) pairs. Epilogue: one channel per thread, warpgroup w on rows 64 w .., coalesced 512-B row loads of c and stores of dc / SC.
// One 128-row tile per CTA (R / 128 CTAs; R = B S).
// Shared memory: 4 stages x 32 KB | barriers (128.1 KB).  TMEM: dc^T at 0 (128 columns).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

namespace tf {
__host__ __device__ constexpr uint32_t idesc(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEVI void mma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI uint32_t rna(uint32_t x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(__uint_as_float(x))); return r & 0xffffe000u; }
}  // namespace tf

constexpr int C = 128, CH = 768, NK = CH / 32, NST = 4;
constexpr int SL = 128 * 128, STB = 2 * SL;                                // A | B per stage
constexpr int O_ST = 0, O_BAR = NST * STB, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t I_DC = tf::idesc(128, 128);

extern "C" __global__ void __launch_bounds__(512, 1)
swa_mod_bwd_dc_tf32_sm100(const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mwt, const float* __restrict__ Cc,
                          float* __restrict__ DC, float* __restrict__ SC) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t* sfree = full + NST;
  uint64_t* rdy = sfree + NST;
  uint64_t* done = rdy + NST;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(done + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r0 = blockIdx.x * 128;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&full[s], 1); mbar_init(&sfree[s], 1); mbar_init(&rdy[s], 1); }
    mbar_init(done, 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(tptr), 128); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tptr;

  if (warp == 0) {
    if (lane == 0)
      for (int kc = 0; kc < NK; ++kc) {
        const int s = kc % NST;
        const uint32_t st = su + O_ST + s * STB;
        if (kc >= NST) mbar_wait(&sfree[s], ((kc / NST) - 1) & 1);
        mbar_expect_tx(&full[s], STB);
        tma_load_2d(st, &mwt, &full[s], 32 * kc, 0);                       // Wmod^T [128 ch][32 k]
        tma_load_2d(st + SL, &mg, &full[s], 32 * kc, r0);                  // g [128 rows][32 k]
      }
  } else if (warp == 1) {
    for (int kc = 0; kc < NK; ++kc) {
      const int s = kc % NST;
      const uint32_t st = su + O_ST + s * STB;
      mbar_wait(&rdy[s], (kc / NST) & 1);                                  // g slice loaded and rounded
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < 4; ++k)
          tf::mma_ss(tmem, desc_k128(st) + (uint64_t)(k * 2), desc_k128(st + SL) + (uint64_t)(k * 2), I_DC, (kc > 0 || k > 0) ? 1u : 0u);
        tc_commit(&sfree[s]);
        if (kc == NK - 1) tc_commit(done);
      }
      __syncwarp();
    }
  } else if (warp >= 4) {
    const int w = (warp - 4) >> 2, lq = warp & 3, c = lq * 32 + lane, t = tid - 128;
    for (int kc = 0; kc < NK; ++kc) {                                      // round the g slice of stage kc in place: 16 B x 4 per thread
      const int s = kc % NST;
      const uint32_t gb = su + O_ST + s * STB + SL;
      mbar_wait(&full[s], (kc / NST) & 1);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t a = gb + (uint32_t)(16 * (t + 256 * e));
        const uint4 u = lds128(a);
        sts128(a, make_uint4(tf::rna(u.x), tf::rna(u.y), tf::rna(u.z), tf::rna(u.w)));
      }
      fence_proxy_async();
      named_bar_sync(1, 256);
      if (t == 0) mbar_arrive(&rdy[s]);
    }
    mbar_wait(done, 0);
    tc_fence_after();
#pragma unroll 1
    for (int g0 = 64 * w; g0 < 64 * w + 64; g0 += 16) {
      uint32_t v[16];
      tmem_ld16(tmem + ((uint32_t)(lq * 32) << 16) + (uint32_t)g0, v);
      float x[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) x[j] = __ldg(Cc + (size_t)(r0 + g0 + j) * C + c);
      tmem_wait_ld();
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float s = 1.f / (1.f + expf(-x[j]));
        const size_t o = (size_t)(r0 + g0 + j) * C + c;
        DC[o] = __uint_as_float(v[j]) * s * (1.f + x[j] * (1.f - s));
        SC[o] = x[j] * s;
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 128); }
}
