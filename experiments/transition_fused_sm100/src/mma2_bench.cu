// cta_group::2 tcgen05.mma throughput: 2-CTA clusters, the leader issues M256 x N x K16 chains on resident smem (A 128 rows per CTA,
// B N/2 rows per CTA). Compare with the 1-CTA M128 x N numbers of mma_bench.
#include "sm100.cuh"
using namespace s100;
template <int N, int TS>
__device__ void body(unsigned long long* out, int iters) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t bar; __shared__ uint32_t tm;
  const uint32_t su = smem_u32(sm);
  if (threadIdx.x < 32) { tmem_alloc2(smem_u32(&tm), 512); tmem_relinquish2(); }
  if (threadIdx.x == 0) { mbar_init(&bar, 1); fence_barrier_init(); }
  for (int i = threadIdx.x; i < 65536 / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = 0x3c003c00u;
  fence_proxy_async();
  tc_fence_before(); __syncthreads(); cluster_sync(); tc_fence_after();
  constexpr uint32_t ID = idesc_bf16(256, N);
  if (cluster_rank() == 0 && threadIdx.x < 32) {
    const unsigned long long t0 = clock64();
    for (int it = 0; it < iters; ++it) {
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * 16384 + (ks & 3) * 32;
          if (TS) umma_ts2(tm + 256, tm + ks * 8, desc_k128(su + 32768 + (ks & 3) * 32), ID, 1);
          else umma_ss2(tm + (it & 1) * 128, desc_k128(su + off), desc_k128(su + 32768 + (ks >> 2) * 8192 + (ks & 3) * 32), ID, ks > 0);
        }
      }
      __syncwarp();
    }
    if (elect_one()) tc_commit2_mc(&bar, 3);
    __syncwarp();
    mbar_wait(&bar, 0);
    if (threadIdx.x == 0) out[blockIdx.x] = clock64() - t0;
  } else if (threadIdx.x == 0) {
    mbar_wait(&bar, 0);
  }
  tc_fence_before(); __syncthreads(); cluster_sync();
  if (threadIdx.x < 32) { tc_fence_after(); tmem_dealloc2(tm, 512); }
}
extern "C" __global__ void mma2_ss128(unsigned long long* o, int n) { body<128, 0>(o, n); }
extern "C" __global__ void mma2_ss256(unsigned long long* o, int n) { body<256, 0>(o, n); }
extern "C" __global__ void mma2_ts128(unsigned long long* o, int n) { body<128, 1>(o, n); }
