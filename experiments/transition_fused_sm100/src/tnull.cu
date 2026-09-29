// tnull.cu — launch-cost floor for the small-L study: the fused forward's prologue / epilogue only (barrier init, 512-column TMEM
// allocation on the pair, cluster syncs, dealloc) and, with -DLOADS, one 16 KB x TMA load per CTA. SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;
extern "C" __global__ void __launch_bounds__(512, 1)
transition_null_sm100(const __grid_constant__ CUtensorMap mx, int tiles) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t bar; __shared__ uint32_t tm;
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) { mbar_init(&bar, 1); fence_barrier_init(); }
  if (warp == 2) { tmem_alloc2(smem_u32(&tm), 512); tmem_relinquish2(); }
  tc_fence_before(); __syncthreads(); cluster_sync(); tc_fence_after();
#ifdef LOADS
  if (threadIdx.x == 0) {
    mbar_expect_tx(&bar, 16384);
    tma_load_2d(smem_u32(sm), &mx, &bar, 0, (blockIdx.x % tiles) * 128);
    tma_load_2d(smem_u32(sm) + 8192, &mx, &bar, 0, (blockIdx.x % tiles) * 128 + 64);
    mbar_wait(&bar, 0);
  }
#endif
  tc_fence_before(); __syncthreads(); cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tm, 512); }
}
