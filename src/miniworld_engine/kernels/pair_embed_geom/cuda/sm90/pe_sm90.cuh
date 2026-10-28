// pe_sm90.cuh -- the device helpers the two sm_90a pair-embedding sources share (pair_embed_geom.cu, pair_embed_adaln.cu): the warp sum,
// cp.async with zero fill, the wgmma m64n64k16 call with its descriptor / fence / group helpers, and the 16-bit shared store.
#pragma once
#include <stdint.h>

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred) {
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  int sz = pred ? 16 : 0;                                   // src-size 0: zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ uint64_t wg_desc(unsigned smem_addr, unsigned lbo_bytes, unsigned sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((smem_addr & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo_bytes & 0x3FFFFu) >> 4) << 16;
  d |= (uint64_t)((sbo_bytes & 0x3FFFFu) >> 4) << 32;
  d |= (uint64_t)1 << 62;                                   // 128-byte swizzle
  return d;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_m64n64k16(float (&d)[32], uint64_t da, uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "l"(da), "l"(db));
}
__device__ __forceinline__ void wg_fence_acc(float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; i++) asm volatile("" : "+f"(d[i])::"memory");
}

__device__ __forceinline__ void st_u16(unsigned addr, unsigned v) { asm volatile("st.shared.u16 [%0], %1;\n" ::"r"(addr), "r"(v)); }
