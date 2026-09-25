// Unit test of kind::f8f6f4 (e4m3 x e4m3 -> f32) tcgen05.mma operand layouts: K-major / MN-major SW128 smem operands and the
// A-from-TMEM variant. One CTA, M = N = K = 128; the host supplies pre-swizzled 16 KB smem images.
#include "sm100.cuh"
using namespace s100;
__host__ __device__ constexpr uint32_t idesc_e4m3(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
DEVI void umma8_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p; }" :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI void umma8_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p; }" :: "r"(d), "r"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
// MN-major, 64-byte swizzle (layout type 4): 8-row K groups 512 B apart
DEVI uint64_t desc_mn64(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)1 << 46) | ((uint64_t)4 << 61);
}
// mode 0: A K-major, B K-major; 1: A MN-major, B MN-major; 2: A from TMEM (row-major bytes of A), B K-major; 3: A K-major, B MN-major
extern "C" __global__ void f8_test(const uint4* __restrict__ ai, const uint4* __restrict__ bi, const uint32_t* __restrict__ araw, float* __restrict__ out, int mode) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t bar; __shared__ uint32_t tm;
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (warp == 0) { tmem_alloc(smem_u32(&tm), 256); tmem_relinquish(); }
  if (tid == 0) { mbar_init(&bar, 1); fence_barrier_init(); }
  for (int i = tid; i < 1024; i += 128) { reinterpret_cast<uint4*>(sm)[i] = ai[i]; reinterpret_cast<uint4*>(sm + 16384)[i] = bi[i]; }
  fence_proxy_async();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t t = tm;
  if (mode == 2) {                                          // row m = lane (warp*32+lane), 128 fp8 = 32 columns at t + 128
    uint32_t v[16];
    for (int h = 0; h < 2; ++h) {
      for (int k = 0; k < 16; ++k) v[k] = araw[tid * 32 + h * 16 + k];
      tmem_st16(t + ((uint32_t)(warp * 32) << 16) + 128 + h * 16, v);
    }
    tmem_wait_st();
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) {
    if (elect_one()) {
      const uint32_t id = mode == 4 ? idesc_e4m3(128, 64, 0, 1) : idesc_e4m3(128, 128, mode == 1, mode == 1 || mode == 3);
      for (int ks = 0; ks < 4; ++ks) {
        const uint64_t ak = desc_k128(su) + (uint64_t)(ks * 2), bk = desc_k128(su + 16384) + (uint64_t)(ks * 2);
        const uint64_t am = desc_mn128(su, 16384) + (uint64_t)(ks * 256), bm = desc_mn128(su + 16384, 16384) + (uint64_t)(ks * 256);
        if (mode == 0) umma8_ss(t, ak, bk, id, ks > 0);
        else if (mode == 1) umma8_ss(t, am, bm, id, ks > 0);
        else if (mode == 2) umma8_ts(t, t + 128 + ks * 8, bk, id, ks > 0);
        else if (mode == 3) umma8_ss(t, ak, bm, id, ks > 0);
        else umma8_ss(t, ak, desc_mn64(su + 16384) + (uint64_t)(ks * 128), id, ks > 0);   // B: [K 128][N 64] SW64, 2048 B per K = 32
      }
      tc_commit(&bar);
    }
    __syncwarp();
  }
  mbar_wait(&bar, 0);
  tc_fence_after();
  for (int cc = 0; cc < 4; ++cc) {
    uint32_t v[32];
    tmem_ld32(t + ((uint32_t)(warp * 32) << 16) + cc * 32, v);
    tmem_wait_ld();
    for (int k = 0; k < 32; ++k) out[tid * 128 + cc * 32 + k] = __uint_as_float(v[k]);
  }
  tc_fence_before(); __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(t, 256); }
}
