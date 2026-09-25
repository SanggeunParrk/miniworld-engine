// TMA multicast probe: bursts of NB tiles; "uni": each CTA loads NB x 32 KB itself; "mc": each CTA of a 2-CTA cluster loads its
// half (16 KB) of each tile with .multicast::cluster to both, so each SM still RECEIVES NB x 32 KB but ISSUES half.
#include "sm100.cuh"
using namespace s100;
constexpr int NB = 6;
DEVI uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEVI void cluster_sync() { asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory"); }
DEVI void tma_load_2d_mc(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
template <int MC>
__device__ void body(const CUtensorMap* m, unsigned long long* out, int reps, int rows) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t full[NB];
  const uint32_t su = smem_u32(sm);
  if (threadIdx.x == 0) { for (int s = 0; s < NB; ++s) mbar_init(&full[s], 1); fence_barrier_init(); }
  __syncthreads();
  if (MC) cluster_sync();
  const uint32_t rank = MC ? cluster_rank() : 0;
  unsigned long long tot = 0;
  for (int rep = 0; rep < reps; ++rep) {
    if (threadIdx.x == 0) {
      const unsigned long long t0 = clock64();
      for (int s = 0; s < NB; ++s) {
        mbar_expect_tx(&full[s], 32768);
        const int r0 = (((blockIdx.x >> MC) * 13 + rep * NB + s) * 128) % rows;
        if (MC) {
          for (int h = 0; h < 2; ++h) tma_load_2d_mc(su + s * 32768 + rank * 16384 + h * 8192, m, &full[s], rank * 64, r0 + h * 64, 3);
        } else {
          for (int cb = 0; cb < 2; ++cb)
            for (int h = 0; h < 2; ++h) tma_load_2d(su + s * 32768 + cb * 16384 + h * 8192, m, &full[s], cb * 64, r0 + h * 64);
        }
      }
      for (int s = 0; s < NB; ++s) mbar_wait(&full[s], rep & 1);
      tot += clock64() - t0;
    }
    __syncthreads();
    if (MC) cluster_sync();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = tot / reps;
}
extern "C" __global__ void uni(const __grid_constant__ CUtensorMap m, unsigned long long* o, int n, int rows) { body<0>(&m, o, n, rows); }
extern "C" __global__ void __cluster_dims__(2, 1, 1) mc(const __grid_constant__ CUtensorMap m, unsigned long long* o, int n, int rows) { body<1>(&m, o, n, rows); }
