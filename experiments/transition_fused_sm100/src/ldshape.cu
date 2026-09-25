// which (lane, column) of TMEM each thread receives from tcgen05.ld.16x256b.x1 / .16x128b.x1 (values encode lane * 1000 + column)
#include "sm100.cuh"
using namespace s100;
extern "C" __global__ void ldshape(int* out) {
  __shared__ uint32_t tm;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp == 0) { tmem_alloc(smem_u32(&tm), 32); tmem_relinquish(); }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t row = warp * 32 + lane;
  uint32_t v[16];
  for (int k = 0; k < 16; ++k) v[k] = row * 1000 + k;
  tmem_st16(tm + ((warp * 32) << 16), v);
  uint32_t w[16];
  for (int k = 0; k < 16; ++k) w[k] = row * 1000 + 16 + k;
  tmem_st16(tm + ((warp * 32) << 16) + 16, w);
  tmem_wait_st();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) {
    uint32_t r[4];
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(tm) : "memory");
    uint32_t q[2];
    asm volatile("tcgen05.ld.sync.aligned.16x128b.x1.b32 {%0,%1}, [%2];" : "=r"(q[0]), "=r"(q[1]) : "r"(tm) : "memory");
    tmem_wait_ld();
    for (int k = 0; k < 4; ++k) out[lane * 6 + k] = r[k];
    out[lane * 6 + 4] = q[0]; out[lane * 6 + 5] = q[1];
  }
  tc_fence_before(); __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(tm, 32); }
}
