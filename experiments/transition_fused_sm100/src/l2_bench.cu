// L2 -> smem TMA throughput: every CTA streams 32 KB tiles of a small (L2-resident) bf16 matrix through an S-stage ring.
#include "sm100.cuh"
using namespace s100;
template <int S>
__device__ void body(const CUtensorMap* m, unsigned long long* out, int iters, int rows) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ uint64_t full[8];
  const uint32_t su = smem_u32(sm);
  if (threadIdx.x == 0) { for (int s = 0; s < S; ++s) mbar_init(&full[s], 1); fence_barrier_init(); }
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned long long t0 = clock64();
    for (int it = 0; it < iters + S; ++it) {
      const int s = it % S;
      if (it >= S) mbar_wait(&full[s], ((it / S) - 1) & 1);
      if (it < iters) {
        mbar_expect_tx(&full[s], 32768);
        const int r0 = ((blockIdx.x * 7 + it) * 128) % rows;
        for (int cb = 0; cb < 2; ++cb)
          for (int h = 0; h < 2; ++h) tma_load_2d(su + s * 32768 + cb * 16384 + h * 8192, m, &full[s], cb * 64, r0 + h * 64);
      }
    }
    out[blockIdx.x] = clock64() - t0;
  }
}
extern "C" __global__ void l2s2(const __grid_constant__ CUtensorMap m, unsigned long long* o, int n, int rows) { body<2>(&m, o, n, rows); }
extern "C" __global__ void l2s4(const __grid_constant__ CUtensorMap m, unsigned long long* o, int n, int rows) { body<4>(&m, o, n, rows); }
extern "C" __global__ void l2s6(const __grid_constant__ CUtensorMap m, unsigned long long* o, int n, int rows) { body<6>(&m, o, n, rows); }
