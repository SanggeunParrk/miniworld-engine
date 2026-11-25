// adaln_mma.cuh -- PTX helpers of the A100 (sm_80) AdaLN / ConditionedTransition tensor-core kernels at the atom width (128): cp.async, ldmatrix, mma.sync m16n8k16
// (bf16 -> fp32), bf16 packing, the quad (4-lane) sum, and the row / weight layouts the kernels share.
//
// Layout of a warp's 16-row tile (the "f1 order" of the SWA atom DiT kernels): lane = 4 g8 + q4 holds rows g8 and g8 + 8; of a row's 128 channels it holds the 32 channels
// 32 gq + 8 q4 + 0 .. 7 (gq = 0 .. 3): four 16-byte vectors of bf16 (a quad reads 64 contiguous bytes per group).  Each vector is the A fragment of two k steps of a product with
// K = 128 (word pairs (0, 1) | (2, 3) of the vector = the fragment registers of k step 2 gq, (4, 5) | (6, 7) those of 2 gq + 1: see ``a_pack``), as long as the weight's k order is
// the same (the weight rows are stored in natural channel order and a thread reads its own 8 channels of them: ``wbq``).  A product whose weight ROWS are stored in the order
// ``qf_channel`` has accumulators in the same layout again: the pair (c0, c1) of n tile 4 gq + j holds channels 32 gq + 8 q4 + 2 j + {0, 1} of row g8, (c2, c3) those of row g8 + 8.
// A row's statistics are therefore a sum over the thread's 32 channels and a quad sum (2 shuffles).
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#include "adaln_common.cuh"

namespace adl {

ADL_DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// cp.async (16 B, L2-only .cg); src_bytes 0 zero-fills the destination
ADL_DEVI void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes = 16) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
ADL_DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> ADL_DEVI void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// shared-memory vector loads of read-only data (not volatile: the compiler may schedule them ahead of the mma they feed)
ADL_DEVI uint4 lds128_ro(uint32_t a) {
  uint4 v; asm("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
ADL_DEVI uint4 lds128(uint32_t a) {   // volatile: ordered after the cp.async wait that published the data
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
ADL_DEVI uint4 ldg128(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
ADL_DEVI void stg128(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
ADL_DEVI float4 ldg_f4(const float* p) { return __ldg(reinterpret_cast<const float4*>(p)); }

// mma.sync m16n8k16 bf16 x bf16 -> fp32 (accumulate in place)
ADL_DEVI void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// bf16 packing: pack_bf16(lo, hi) = the two roundings in one register (lo in the low half)
ADL_DEVI uint32_t pack_bf16(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
ADL_DEVI float rn(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
ADL_DEVI float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
ADL_DEVI float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
// the 8 channels of a 16-byte vector as fp32
ADL_DEVI void unpack8(uint4 u, float (&v)[8]) {
  v[0] = bf16lo(u.x); v[1] = bf16hi(u.x); v[2] = bf16lo(u.y); v[3] = bf16hi(u.y);
  v[4] = bf16lo(u.z); v[5] = bf16hi(u.z); v[6] = bf16lo(u.w); v[7] = bf16hi(u.w);
}
ADL_DEVI uint4 pack8(const float (&v)[8]) {
  return make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7]));
}
// rn(a * b + c) per half with ONE rounding to bf16 (fma.rn.bf16x2: the framework's `x + g * y` on bf16 tensors)
ADL_DEVI uint32_t fma_bf16x2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}

ADL_DEVI float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}

// A fragments of a 16 x 128 tile from the thread's vectors v[hh][gq] (row g8 + 8 hh, channels 32 gq + 8 q4 + 0 .. 7): a[ks][0 .. 3] of k step ks = 0 .. 7
ADL_DEVI void a_from_vec(uint32_t (&a)[8][4], const uint4 (&v)[2][4]) {
#pragma unroll
  for (int hh = 0; hh < 2; ++hh)
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      a[2 * gq][hh] = v[hh][gq].x;      a[2 * gq][2 + hh] = v[hh][gq].y;
      a[2 * gq + 1][hh] = v[hh][gq].z;  a[2 * gq + 1][2 + hh] = v[hh][gq].w;
    }
}

constexpr int QF_ROWS = 128;                           // packed weight rows of one 128 x 128 matrix
// packed row s -> the output channel it holds: in a block of 64 rows the 8 n tiles are (og, j) = (nt >> 2, nt & 3) and column n of a tile holds channel 32 og + 8 (n >> 1) + 2 j + (n & 1)
ADL_DEVI int qf_channel(int s) { return (s & ~63) + 32 * ((s >> 5) & 1) + 8 * ((s >> 1) & 3) + 2 * ((s >> 3) & 3) + (s & 1); }
// byte offset of 16-byte chunk `chunk` of packed row `row` (rows of 256 B; the chunk index is XOR-ed with 4 on odd rows: the 2 rows x 4 chunks a quarter warp reads fall in 8 bank groups)
ADL_DEVI uint32_t qf_woff(uint32_t row, uint32_t chunk) { return row * 256u + ((chunk ^ ((row & 1u) << 2)) << 4); }

// a 128 x 128 bf16 weight matrix W [out][in] into shared memory in the f1 row order (cooperative, cp.async; commit and wait are the caller's)
ADL_DEVI void load_weight128(uint32_t dst, const bf* w, int tid, int nthr) {
  for (int i = tid; i < QF_ROWS * 16; i += nthr) {
    const int row = i >> 4;
    cp_async16(dst + qf_woff(row, i & 15), w + (size_t)qf_channel(row) * 128 + (i & 15) * 8);
  }
}

// rows of 128 B (8 granules of 16 B): granule g of row r at r * 128 + ((g ^ (r & 7)) << 4) -- ldmatrix / fragment reads of 8 consecutive rows are conflict-free
ADL_DEVI uint32_t sw128(uint32_t row, uint32_t granule) { return row * 128u + ((granule ^ (row & 7u)) << 4); }

}  // namespace adl
