// sm80_common.cuh -- PTX helpers of the A100 (sm_80) MSA pair-weighted-averaging kernels: cp.async (zero-fill), ldmatrix (+ .trans), mma.sync m16n8k16
// bf16 -> fp32, bf16 packing, the 128-byte-row XOR swizzle.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace pwa80 {

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---- cp.async (16 B, L2 only); src_bytes = 0 zero-fills the destination (the source address must still be valid)
DEVI void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes = 16) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> DEVI void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// ---- ldmatrix
DEVI void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEVI void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// ---- mma.sync m16n8k16 bf16 x bf16 -> fp32 (accumulate in place)
DEVI void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---- shared / global vector access
DEVI uint4 lds128(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
DEVI void sts128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEVI uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a)); return v; }
DEVI void sts32(uint32_t a, uint32_t v) { asm volatile("st.shared.u32 [%0], %1;\n" ::"r"(a), "r"(v) : "memory"); }
DEVI uint4 ldg128(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
DEVI void stg128(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEVI void stg32(void* p, uint32_t v) { asm volatile("st.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }
DEVI void stg64(void* p, float2 v) { asm volatile("st.global.v2.f32 [%0], {%1,%2};\n" ::"l"(p), "f"(v.x), "f"(v.y) : "memory"); }

// ---- bf16 packing: the low half is `lo` (the lower column), round-to-nearest-even
DEVI uint32_t pack_bf16(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
DEVI float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
DEVI float round_bf16f(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEVI uint32_t add_bf16x2(uint32_t a, uint32_t b) {   // bf16(a + b) per half (sm_80 has no add.bf16x2): a * 1 + b, one rounding
  uint32_t r; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(0x3f803f80u), "r"(b)); return r;
}

// ---- reductions over the lanes of a group (xor butterflies)
template <int W> DEVI float group_sum(float v) {
#pragma unroll
  for (int o = W / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---- sigmoid through one MUFU op (tanh.approx), the four lanes of an mma row
DEVI float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
DEVI float sigmoid_fast(float g) { return fmaf(tanh_approx(0.5f * g), 0.5f, 0.5f); }
DEVI float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEVI float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// The head-major tensors v, o, dO, dgp, dv ([8 ns, L, (S / ns) C]): the MSA rows are split in `ns` chunks of kp = S / ns rows, each chunk its own batch of the cuBLAS products
// (the weight-gradient product dw = dO v^T contracts over S C, a K the card runs far better as ns batches of K / ns).  Element (head h, row j, MSA row s, channel c) lives at
// (((h ns + s / kp) L + j) kp + s % kp) C + c; ns = 1 is the plain [8, L, S C] layout.  A tile of 128 (or 64) consecutive rows never straddles a chunk (kp is a multiple of 128
// when ns > 1).  hm_row: index of the tile's first row (in units of C elements).
DEVI long hm_row(int h, int j, int s0, int L, int kp, int ns) {
  const int sp = s0 / kp;
  return (((long)h * ns + sp) * L + j) * kp + (s0 - sp * kp);
}

// byte offset of 16-byte chunk `chunk` of row `row` in a tile of 128-byte rows (8 chunks), chunk ^ (row & 7): the eight rows of an ldmatrix
// (or a cp.async warp access) fall on eight distinct bank groups
DEVI uint32_t swz128(uint32_t row, uint32_t chunk) { return row * 128u + (((chunk ^ row) & 7u) << 4); }
// rows of 16 * NCH bytes, NCH a multiple of 8: the chunk index is XORed in its low three bits only
template <int NCH> DEVI uint32_t swzn(uint32_t row, uint32_t chunk) { return row * (16u * NCH) + ((chunk ^ (row & 7u)) << 4); }

}  // namespace pwa80
