// mod_fwd_tf32.cu — the SWA atom block's adaLN modulation for the fp32 path on sm_100a, one kernel:  mod = silu(c) Wmod^T
// c [R, 128] fp32, Wmod [768, 128] fp32 -> fp32 out [R, 768], on TF32 tensor cores (tcgen05.mma kind::tf32, fp32 accumulation in TMEM).
// silu in fp32 (x / (1 + e^-x), accurate reciprocal) rounded to TF32 by cvt.rna; Wmod arrives already rounded to TF32 (the host's
// tf32_fwd._round_tf32, cached per weight version): both MMA operands are round-to-nearest TF32, not the tensor core's truncation of the
// low mantissa bits (a truncation biases every product toward zero, and that bias does not average out over K = 128).
// One CTA per (128 rows, 128 output channels), 4 warps: thread r takes row r for the silu and the epilogue (TMEM lane r).
//   shared memory  c (4 k-blocks of [128 rows][32 fp32], 128-B swizzle: 64 KB; silu in place, then the fp32 out staging) | Wmod block
//                  (4 k-blocks of [128 output channels][32 fp32]: 64 KB) | 2 barriers + the TMEM address = 128 KB + 64 B
//   TMEM           the [128 rows][128 channels] fp32 accumulator: 128 columns
//   MMA            16 x M128 N128 K8 (SS, both operands K-major, SW128: a 128-B row holds the 32 fp32 of one k-block, K = 8 is 32 B
//                  -> descriptor + 2 per step), issued by one elected lane of warp 0
// The result leaves through the c tile (free once the MMAs are done) as four 128-B-swizzled [128][32] fp32 boxes (TMA stores, clipped
// at R). Registers: one 16-B chunk of the row at a time for the silu, 32 fp32 per TMEM load in the epilogue (well under 128).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

// ------------------------------------------------------------------ kind::tf32 (local: sm100.cuh is the bf16 kernels' header)
// instruction descriptor: D fp32 (bit 4), A / B tf32 (format 2 at bits 7 / 10), a / b major (0 = K, 1 = MN), N >> 3, M >> 4
__host__ __device__ constexpr uint32_t idesc_tf32(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEVI void umma_ss_tf32(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
// fp32 -> the nearest TF32 (ties away from zero), kept in an fp32 container: the MMA then reads it exactly
DEVI uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI float sigm32(float x) { return __frcp_rn(1.f + __expf(-x)); }

constexpr int C = 128;
constexpr int KB = 128 * 128;                                              // one k-block: [128 rows][32 fp32], 128-B swizzle (16 KB)
constexpr int O_C = 0, O_W = 4 * KB, O_BAR = 8 * KB;                       // c (silu in place; then the out staging) | Wmod block
constexpr int SMEM = O_BAR + 64;
static_assert(SMEM <= 232448, "shared memory");
constexpr uint32_t I_M = idesc_tf32(128, 128);

extern "C" __global__ void __launch_bounds__(128, 1)
swa_mod_fwd_tf32_sm100(const __grid_constant__ CUtensorMap mc, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mout) {
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
    mbar_expect_tx(full, 8 * KB);
    for (int kb = 0; kb < 4; ++kb) tma_load_2d(su + O_W + kb * KB, &mw, full, 32 * kb, n0);    // weights: no dependence
    pdl_wait();                                                            // c comes from the previous kernels
    for (int kb = 0; kb < 4; ++kb) tma_load_2d(su + O_C + kb * KB, &mc, full, 32 * kb, r0);
  }
  mbar_wait(full, 0);
  // silu in place, row tid: 32 chunks of 4 fp32 (chunk k = k-block k / 8, 16-B chunk k % 8 of the swizzled row), rounded to TF32
#pragma unroll 4
  for (int k = 0; k < 32; ++k) {
    const uint32_t a = su + O_C + (k >> 3) * KB + sw128((uint32_t)tid, k & 7);
    const uint4 u = lds128(a);
    const float x0 = __uint_as_float(u.x), x1 = __uint_as_float(u.y), x2 = __uint_as_float(u.z), x3 = __uint_as_float(u.w);
    sts128(a, make_uint4(tf32r(x0 * sigm32(x0)), tf32r(x1 * sigm32(x1)), tf32r(x2 * sigm32(x2)), tf32r(x3 * sigm32(x3))));
  }
  fence_proxy_async();
  __syncthreads();
  if (warp == 0) {
    tc_fence_after();
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < 16; ++ks)
        umma_ss_tf32(tmem, desc_k128(su + O_C + (ks >> 2) * KB) + (uint64_t)((ks & 3) * 2),
                     desc_k128(su + O_W + (ks >> 2) * KB) + (uint64_t)((ks & 3) * 2), I_M, ks > 0 ? 1u : 0u);
      tc_commit(done);
    }
    __syncwarp();
  }
  mbar_wait(done, 0);                                                      // the MMAs have read c and Wmod: c becomes the staging
  tc_fence_after();
  // epilogue: row tid, 128 fp32 channels -> staging over c (4 blocks of [128 rows][32 fp32], 128-B swizzle)
#pragma unroll 1
  for (int q = 0; q < 4; ++q) {
    uint32_t v[32];
    tmem_ld32(tmem + ((uint32_t)(warp * 32) << 16) + 32 * q, v);
    tmem_wait_ld();
    const uint32_t base = su + O_C + q * KB;
#pragma unroll
    for (int k = 0; k < 8; ++k) sts128(base + sw128((uint32_t)tid, k), make_uint4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]));
  }
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  if (tid == 0) {
    for (int q = 0; q < 4; ++q) tma_store_2d(&mout, su + O_C + q * KB, n0 + 32 * q, r0);
    tma_store_commit();
    tma_store_wait0();
  }
  __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(tmem, 128); }
}
