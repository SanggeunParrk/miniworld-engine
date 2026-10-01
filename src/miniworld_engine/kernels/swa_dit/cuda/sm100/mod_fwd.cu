// mod_fwd.cu — the SWA atom block's adaLN modulation on sm_100a, one kernel:  mod = silu(c) Wmod^T  (fp32 out [B S, 768])
// silu rounded to bf16 as torch does on a bf16 tensor (fp32 silu with the approximate exp / reciprocal of the gates, then rn), then a bf16
// MMA with fp32 accumulation: the
// products are exact, as in the TF32 GEMM of bf16-valued operands it replaces (only the summation order differs).
// One CTA per (128 rows, 128 output channels); 4 warps: thread r takes row r for the silu and the epilogue (TMEM lane r), the result
// leaves through a 128-B-swizzled fp32 staging tile and four TMA stores.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int C = 128;
constexpr int KB = 128 * 128;                                              // [128 rows][64] bf16 k-block
constexpr int O_C = 0, O_W = 2 * KB, O_O = 4 * KB, O_BAR = O_O + 4 * KB;   // c tile | Wmod block | fp32 staging (4 x [128][32])
constexpr int SMEM = O_BAR + 64;
constexpr uint32_t I_M = idesc_bf16(128, 128);

extern "C" __global__ void __launch_bounds__(128, 1)
swa_mod_fwd_sm100(const __grid_constant__ CUtensorMap mc, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mout) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t* done = full + 1;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(full + 2);
  const int tid = threadIdx.x, warp = tid >> 5;
  const int r0 = blockIdx.x * 128, n0 = blockIdx.y * 128;
  if (tid == 0) { mbar_init(full, 1); mbar_init(done, 1); fence_barrier_init(); }
  if (warp == 0) { tmem_alloc(smem_u32(tptr), 128); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tptr;
  pdl_launch();
  if (tid == 0) {
    mbar_expect_tx(full, 4 * KB);
    for (int kb = 0; kb < 2; ++kb) tma_load_2d(su + O_W + kb * KB, &mw, full, kb * 64, n0);    // weights: no dependence
    pdl_wait();                                                            // c comes from the previous kernels
    for (int kb = 0; kb < 2; ++kb) tma_load_2d(su + O_C + kb * KB, &mc, full, kb * 64, r0);
  }
  mbar_wait(full, 0);
  // silu in place (bf16), row tid
#pragma unroll 4
  for (int k = 0; k < 16; ++k) {
    const uint32_t a = su + O_C + (k >> 3) * KB + sw128((uint32_t)tid, k & 7);
    uint4 u = lds128(a);
    uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float x0 = bf16lo(w4[e]), x1 = bf16hi(w4[e]);
      w4[e] = pack_bf16(x0 * sigmoid_kit(x0), x1 * sigmoid_kit(x1));      // rcp.approx(1 + ex2.approx(-x log2 e)), as the gates
    }
    sts128(a, make_uint4(w4[0], w4[1], w4[2], w4[3]));
  }
  fence_proxy_async();
  __syncthreads();
  if (warp == 0) {
    tc_fence_after();
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < 8; ++ks)
        umma_ss(tmem, desc_k128(su + O_C + (ks >> 2) * KB) + (uint64_t)((ks & 3) * 2), desc_k128(su + O_W + (ks >> 2) * KB) + (uint64_t)((ks & 3) * 2), I_M,
                ks > 0 ? 1u : 0u);
      tc_commit(done);
    }
    __syncwarp();
  }
  mbar_wait(done, 0);
  tc_fence_after();
  // epilogue: row tid, 128 fp32 channels -> staging (4 blocks of [128 rows][32 fp32], 128-B swizzle)
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    uint32_t v[32];
    tmem_ld32(tmem + ((uint32_t)(warp * 32) << 16) + 32 * q, v);
    tmem_wait_ld();
    const uint32_t base = su + O_O + q * KB;
#pragma unroll
    for (int k = 0; k < 8; ++k) sts128(base + sw128((uint32_t)tid, k), make_uint4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]));
  }
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  if (tid == 0) {
    for (int q = 0; q < 4; ++q) tma_store_2d(&mout, su + O_O + q * KB, n0 + 32 * q, r0);
    tma_store_commit();
    tma_store_wait0();
  }
  __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(tmem, 128); }
}
