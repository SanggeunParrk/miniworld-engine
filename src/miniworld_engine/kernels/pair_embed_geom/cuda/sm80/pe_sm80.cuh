// pe_sm80.cuh -- the A100 (sm_80) side of pe_sm90.cuh, for the sm_80 builds of the two pair-embedding sources (pair_embed_geom.cu,
// pair_embed_adaln.cu, compiled unchanged by pair_embed_geom_sm80.cu / pair_embed_adaln_sm80.cu). Included FIRST: pe_sm90.cuh is pulled in
// here with its five sm_90-only helpers renamed out of the way (never called, so never emitted), and its `#pragma once` makes the sources'
// own include of it a no-op; then the five names are defined for sm_80:
//   * wgmma_m64n64k16(acc, descA, descB): the same 64 x 64 x 16 product on the same shared-memory operands (A K-major, B MN-major, both
//     128-byte swizzled, read through the same descriptors), issued synchronously as mma.sync m16n8k16 per warp of the warpgroup: warp w
//     of the 4 owns rows 16w..16w+15, and its accumulator layout (32 floats: 8 n-tiles x the m16n8 C fragment) IS the wgmma D layout,
//     so the kernels' epilogues are unchanged. A by ldmatrix.x4, B by ldmatrix.x4.trans, at the swizzled addresses;
//   * wg_fence / wg_commit / wg_wait<N>: nothing to order (the product above is complete when it returns);
//   * fence_async: the operands are written and read through the generic proxy; the kernels' __syncthreads orders them.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#define wgmma_m64n64k16 pe_sm90_wgmma_m64n64k16
#define wg_fence pe_sm90_wg_fence
#define wg_commit pe_sm90_wg_commit
#define wg_wait pe_sm90_wg_wait
#define fence_async pe_sm90_fence_async
#include "../sm90/pe_sm90.cuh"
#undef wgmma_m64n64k16
#undef wg_fence
#undef wg_commit
#undef wg_wait
#undef fence_async

__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int N> __device__ __forceinline__ void wg_wait() {}
__device__ __forceinline__ void fence_async() {}

// the 128-byte swizzle of a 1024-byte-aligned atom: 16-byte chunk c of 128-byte row r sits at chunk c ^ (r & 7) (address bits 4-6 ^= bits 7-9)
__device__ __forceinline__ unsigned pe80_swz(unsigned a) { return a ^ (((a >> 7) & 7u) << 4); }

__device__ __forceinline__ void pe80_mma(float* c, const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void wgmma_m64n64k16(float (&d)[32], uint64_t da, uint64_t db) {
  const unsigned lane = threadIdx.x & 31u, w = (threadIdx.x >> 5) & 3u;
  const unsigned a0 = (unsigned)(da & 0x3FFFu) << 4, sboa = (unsigned)((da >> 32) & 0x3FFFu) << 4;
  const unsigned b0 = (unsigned)(db & 0x3FFFu) << 4, sbob = (unsigned)((db >> 32) & 0x3FFFu) << 4;
  // A (64 rows x 16 k, K-major): rows 16w + (lane & 15), k half lane >> 4 -> the four 8x8 matrices of the m16k16 A fragment
  const unsigned r = 16u * w + (lane & 15u);
  unsigned fa[4];
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(fa[0]), "=r"(fa[1]), "=r"(fa[2]), "=r"(fa[3])
               : "r"(pe80_swz(a0 + (r >> 3) * sboa + (r & 7u) * 128u + (lane >> 4) * 16u)));
  // B (16 k x 64 n, MN-major): k row (lane & 7) + 8 * ((lane >> 3) & 1), 16-byte n chunk 2j + (lane >> 4)
  const unsigned k = (lane & 7u) + (((lane >> 3) & 1u) << 3);
  const unsigned brow = b0 + (k >> 3) * sbob + (k & 7u) * 128u;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    unsigned q0, q1, q2, q3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(q0), "=r"(q1), "=r"(q2), "=r"(q3)
                 : "r"(pe80_swz(brow + (2u * j + (lane >> 4)) * 16u)));
    pe80_mma(d + 8 * j, fa, q0, q1);
    pe80_mma(d + 8 * j + 4, fa, q2, q3);
  }
}
