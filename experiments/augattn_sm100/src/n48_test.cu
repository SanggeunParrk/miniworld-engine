// Unit test: bf16 tcgen05.mma with N = 48 and an MN-major B whose 128-B swizzle atom (64 elements) is only 48 wide, and a
// K-major A with K = 48 inside one 128-B atom (3 K-steps). D = A[128 x K] B[K x 48].
#include "sm100.cuh"
using namespace s100;
// mode 0: A K-major [128][K=48 of 64], B MN-major [K=64 rows][N=48 of 64] (4 K-steps of 16); mode 1: B K-major [N=48 rows][K=64]
extern "C" __global__ void n48_test(const uint4* __restrict__ ai, const uint4* __restrict__ bi, float* __restrict__ out, int mode, int kdim) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t bar; __shared__ uint32_t tm;
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, warp = tid >> 5;
  if (warp == 0) { tmem_alloc(smem_u32(&tm), 64); tmem_relinquish(); }
  if (tid == 0) { mbar_init(&bar, 1); fence_barrier_init(); }
  for (int i = tid; i < 1024; i += 128) { reinterpret_cast<uint4*>(sm)[i] = ai[i]; reinterpret_cast<uint4*>(sm + 16384)[i] = bi[i]; }
  fence_proxy_async();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t t = tm;
  if (warp == 0) {
    if (elect_one()) {
      const uint32_t id = idesc_bf16(128, 48, 0, mode == 0 ? 1 : 0);
      for (int ks = 0; ks < kdim / 16; ++ks) {
        const uint64_t a = desc_k128(su) + (uint64_t)(ks * 2);
        const uint64_t b = mode == 0 ? desc_mn128(su + 16384, 8192) + (uint64_t)(ks * 2048 >> 4) : desc_k128(su + 16384) + (uint64_t)(ks * 2);
        umma_ss(t, a, b, id, ks > 0);
      }
      tc_commit(&bar);
    }
    __syncwarp();
  }
  mbar_wait(&bar, 0);
  tc_fence_after();
  uint32_t v[16];
  for (int cc = 0; cc < 3; ++cc) {
    tmem_ld16(t + ((uint32_t)(warp * 32) << 16) + cc * 16, v);
    tmem_wait_ld();
    for (int k = 0; k < 16; ++k) out[tid * 48 + cc * 16 + k] = __uint_as_float(v[k]);
  }
  tc_fence_before(); __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(t, 64); }
}
