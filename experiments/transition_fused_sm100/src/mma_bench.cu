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
// the forward's issue pattern per chunk: expand (8 SS, N128) -> commit, squeeze (4 TS, N128, K64) -> commit x2
template <int MODE>
__device__ void pattern(unsigned long long* out, int iters) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t bar[4]; __shared__ uint32_t tm;
  const uint32_t su = smem_u32(sm);
  if (threadIdx.x < 32) { tmem_alloc(smem_u32(&tm), 512); tmem_relinquish(); }
  if (threadIdx.x == 0) { for (int i = 0; i < 4; ++i) mbar_init(&bar[i], 1); fence_barrier_init(); }
  for (int i = threadIdx.x; i < 65536 / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = 0x3c003c00u;
  fence_proxy_async();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  constexpr uint32_t ID = idesc_bf16(128, 128);
  if (threadIdx.x == 0) {
    unsigned long long t0 = clock64();
    for (int c = 0; c < iters; ++c) {
      const int s = c & 1;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t off = (ks >> 2) * 16384 + (ks & 3) * 32;
        umma_ss(tm + s * 128, desc_k128(su + off), desc_k128(su + 32768 + off), ID, ks > 0);
      }
      if (MODE >= 1) tc_commit(&bar[s]);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        if (MODE == 2) umma_ss(tm + 384, desc_k128(su + ks * 32), desc_k128(su + 32768 + ks * 32), ID, 1);
        else umma_ts(tm + 384, tm + 256 + s * 32 + ks * 8, desc_k128(su + 32768 + ks * 32), ID, 1);
      }
      if (MODE >= 1) { tc_commit(&bar[2 + s]); tc_commit(&bar[2 + s]); }
    }
    tc_commit(&bar[0]);
    unsigned long long t1 = clock64();
    while (!mbar_try_wait(&bar[0], (iters / 2) & 1) && !mbar_try_wait(&bar[0], ((iters / 2) + 1) & 1)) {}
    out[blockIdx.x] = t1 - t0;
  }
  __syncthreads();
  for (int i = 0; i < 100000; ++i) __nanosleep(100);   // let everything drain
  tc_fence_before(); __syncthreads();
  if (threadIdx.x < 32) { tc_fence_after(); tmem_dealloc(tm, 512); }
}
extern "C" __global__ void pat0(unsigned long long* o, int n) { pattern<0>(o, n); }
extern "C" __global__ void pat1(unsigned long long* o, int n) { pattern<1>(o, n); }
extern "C" __global__ void pat2(unsigned long long* o, int n) { pattern<2>(o, n); }
extern "C" __global__ void mma_ss16(unsigned long long* o, int n) { body<16, 0>(o, n); }
extern "C" __global__ void mma_ss32(unsigned long long* o, int n) { body<32, 0>(o, n); }
// same as ss128 but with 3 extra warps per SMSP spinning on FMA (issue-slot contention)
extern "C" __global__ void mma_ss128_busy(unsigned long long* o, int n) {
  if (threadIdx.x >= 128) {
    float a = threadIdx.x, b = 1.0001f;
    for (int i = 0; i < n * 40; ++i) { a = fmaf(a, b, 0.5f); b = fmaf(b, a, -0.25f); }
    if (a == 1234.5f) o[0] = 1;
    __syncthreads(); __syncthreads();
    return;
  }
  body<128, 0>(o, n);
}
// independent accumulators: 8 products into 8 different TMEM regions (N cols each), K=16 each
template <int N, int NW>
__device__ void indep(unsigned long long* out, int iters) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t bar; __shared__ uint32_t tm;
  const uint32_t su = smem_u32(sm);
  if (threadIdx.x < 32) { tmem_alloc(smem_u32(&tm), 512); tmem_relinquish(); }
  if (threadIdx.x == 0) { mbar_init(&bar, NW); fence_barrier_init(); }
  for (int i = threadIdx.x; i < 65536 / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = 0x3c003c00u;
  fence_proxy_async();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  constexpr uint32_t ID = idesc_bf16(128, N);
  const int w = threadIdx.x >> 5;
  if (w < NW) {
    const uint64_t a = desc_k128(su), b = desc_k128(su + 32768);
    unsigned long long t0 = clock64();
    for (int it = 0; it < iters; ++it) {
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < 8; ++k) umma_ss(tm + ((k * N + w * 8 * N) & 511), a + 2 * (k & 3), b + 2 * (k & 3), ID, 1);
      }
      __syncwarp();
    }
    if (elect_one()) tc_commit(&bar);
    __syncwarp();
    mbar_wait(&bar, 0);
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
  }
  tc_fence_before(); __syncthreads();
  if (threadIdx.x < 32) { tc_fence_after(); tmem_dealloc(tm, 512); }
}
extern "C" __global__ void ind32(unsigned long long* o, int n) { indep<32, 1>(o, n); }
extern "C" __global__ void ind64(unsigned long long* o, int n) { indep<64, 1>(o, n); }
extern "C" __global__ void ind128(unsigned long long* o, int n) { indep<128, 1>(o, n); }
extern "C" __global__ void ind32w2(unsigned long long* o, int n) { indep<32, 2>(o, n); }
extern "C" __global__ void ind64w2(unsigned long long* o, int n) { indep<64, 2>(o, n); }
// the backward's d_xn product: A from TMEM, B MN-major (N = 128 over two 64-wide atoms 16 KB apart), and the dh product (SS, N = 64, B MN-major)
template <int MODE>
__device__ void body_mn(unsigned long long* out, int iters) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t bar; __shared__ uint32_t tm;
  const uint32_t su = smem_u32(sm);
  if (threadIdx.x < 32) { tmem_alloc(smem_u32(&tm), 512); tmem_relinquish(); }
  if (threadIdx.x == 0) { mbar_init(&bar, 1); fence_barrier_init(); }
  for (int i = threadIdx.x; i < 65536 / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = 0x3c003c00u;
  fence_proxy_async();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (threadIdx.x < 32) {
    const unsigned long long t0 = clock64();
    for (int it = 0; it < iters; ++it) {
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          if (MODE == 0) umma_ts(tm + 384, tm + ks * 8, desc_mn128(su + 32768 + ks * 2048, 16384), idesc_bf16(128, 128, 0, 1), 1);
          if (MODE == 1) umma_ss(tm + 256, desc_k128(su + (ks >> 2) * 16384 + (ks & 3) * 32), desc_mn128(su + 32768 + ks * 2048, 16384), idesc_bf16(128, 64, 0, 1), ks > 0);
          if (MODE == 2) umma_ss(tm + 256, desc_k128(su + (ks >> 2) * 16384 + (ks & 3) * 32), desc_mn128(su + 32768 + ks * 2048, 16384), idesc_bf16(128, 128, 0, 1), ks > 0);
        }
      }
      __syncwarp();
    }
    if (elect_one()) tc_commit(&bar);
    __syncwarp();
    mbar_wait(&bar, 0);
    if (threadIdx.x == 0) out[blockIdx.x] = clock64() - t0;
  }
  tc_fence_before(); __syncthreads();
  if (threadIdx.x < 32) { tc_fence_after(); tmem_dealloc(tm, 512); }
}
extern "C" __global__ void mn_ts128(unsigned long long* o, int n) { body_mn<0>(o, n); }
extern "C" __global__ void mn_ss64(unsigned long long* o, int n) { body_mn<1>(o, n); }
extern "C" __global__ void mn_ss128(unsigned long long* o, int n) { body_mn<2>(o, n); }
