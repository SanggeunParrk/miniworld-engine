// tcgen05.mma throughput probe: one CTA per SM, one thread issues NITER x (K=128 chain) products on resident smem operands.
#include "sm100.cuh"
using namespace s100;
template <int N, int TS>
__device__ void body(unsigned long long* out, int iters) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t bar; __shared__ uint32_t tm;
  const uint32_t su = smem_u32(sm);
  if (threadIdx.x < 32) { tmem_alloc(smem_u32(&tm), 512); tmem_relinquish(); }
  if (threadIdx.x == 0) { mbar_init(&bar, 1); fence_barrier_init(); }
  for (int i = threadIdx.x; i < 65536 / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = 0x3c003c00u;
  fence_proxy_async();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  constexpr uint32_t ID = idesc_bf16(128, N);
  if (threadIdx.x == 0) {
    unsigned long long t0 = clock64();
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t off = (ks >> 2) * 16384 + (ks & 3) * 32;
        if (TS) umma_ts(tm + 256, tm + ks * 8, desc_k128(su + 32768 + off), ID, 1);
        else umma_ss(tm + (it & 1) * 256, desc_k128(su + off), desc_k128(su + 32768 + off), ID, ks > 0);
      }
    }
    tc_commit(&bar);
    mbar_wait(&bar, 0);
    unsigned long long t1 = clock64();
    out[blockIdx.x] = t1 - t0;
  }
  tc_fence_before(); __syncthreads();
  if (threadIdx.x < 32) { tc_fence_after(); tmem_dealloc(tm, 512); }
}
extern "C" __global__ void mma_ss64(unsigned long long* o, int n) { body<64, 0>(o, n); }
extern "C" __global__ void mma_ss128(unsigned long long* o, int n) { body<128, 0>(o, n); }
extern "C" __global__ void mma_ss256(unsigned long long* o, int n) { body<256, 0>(o, n); }
extern "C" __global__ void mma_ts128(unsigned long long* o, int n) { body<128, 1>(o, n); }
extern "C" __global__ void mma_ts256(unsigned long long* o, int n) { body<256, 1>(o, n); }
