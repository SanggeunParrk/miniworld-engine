// common.cuh -- helpers of the A100 (sm_80) MPNN edge kernels: PTX wrappers (cp.async, ldmatrix, mma.sync m16n8k16 bf16 -> fp32), bf16 packing, the exact-erf GELU and its
// derivative, a counter-based hash for the dropout draw, and the two index maps every kernel of the family shares (the shared-memory swizzle of a 256-byte weight row and the "f1"
// channel permutation).  Every kernel family of this repository owns its own copy of such a header.
//
// Geometry (m16n8k16, a warp = 16 rows, g = lane / 4, q = lane % 4): the accumulator of n tile j holds (row g, cols 8 j + 2 q, +1) and (row g + 8, the same columns); the A fragment
// of k block kk = (row g, k 16 kk + 2 q, +1), (row g + 8, same), (row g, k 16 kk + 8 + 2 q, +1), (row g + 8, same); the B fragment of (kk, n tile) = (k 16 kk + 2 q, +1; n = g) and
// (k 16 kk + 8 + 2 q, +1; n = g).
//
// The channel permutation ("f1 order").  A thread's accumulators over the 4 n tiles j = 4 a + s of a group a are the 8 CONSECUTIVE logical channels 32 a + 8 q + 0..7 of its row,
// and the same 8 channels are, as bf16 pairs, the A fragments (k blocks 2 a, 2 a + 1) of the next product: so every activation tensor is read and written as 16-byte vectors
// (thread q of a quad owns the vector at channel 32 a + 8 q of each of its two rows), the layers chain through registers, and the only thing that is permuted is the ROW order of
// the weights in shared memory (``perm``): packed row r = 32 a + 8 s + 2 q + e holds logical output channel 32 a + 8 q + 2 s + e.  The k order needs nothing: lane (g, q)
// reads the 16-byte chunk 4 kb + q of its weight row, i.e. logical k = 32 kb + 8 q .. + 7, which is exactly the thread's own vector of that group.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace me80 {

constexpr int D = 128;

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---- cp.async (16 B, L2-only .cg); src_bytes 0 zero-fills the destination
DEVI void cp_async16(uint32_t dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory"); }
DEVI void cp_async16z(uint32_t dst, const void* src, uint32_t src_bytes) {
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

// the first k step of an accumulator: C = 0 (ptxas turns the constant into RZ, so no register is initialised by a move)
DEVI void mma16816_z(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

// ---- shared / global vector access
// volatile on purpose: the weight fragments are loop-invariant, and a plain asm is hoisted out of the tile loop (the whole 96 KiB of fragments, spilled to local memory)
DEVI uint4 lds128(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
DEVI void sts128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEVI uint4 ldg128(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
DEVI uint4 ldg128_stream(const void* p) {      // read once (the saved activations of the backward): do not keep it in L1
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
DEVI void stg128(void* p, uint4 v) { asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
DEVI void stg128_stream(void* p, uint4 v) {     // written once, read much later: evict-first
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEVI void stg32(void* p, uint32_t v) { asm volatile("st.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }

// ---- bf16 packing
DEVI uint32_t pack_bf16(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
DEVI float round_bf16f(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// fp16 pairs: the saved GELU derivatives (10 mantissa bits: four times the precision of bf16 for the same two bytes)
DEVI uint32_t pack_f16(float lo, float hi) { uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI float f16lo(uint32_t v) { float r; asm("cvt.f32.f16 %0, %1;\n" : "=f"(r) : "h"((unsigned short)(v & 0xffffu))); return r; }
DEVI float f16hi(uint32_t v) { float r; asm("cvt.f32.f16 %0, %1;\n" : "=f"(r) : "h"((unsigned short)(v >> 16))); return r; }
constexpr uint32_t ONE2 = 0x3f803f80u;      // bf16x2 (1, 1)
// rn(a * b + c) per half with ONE rounding: fma.rn.bf16x2 (sm_80 has no add.bf16x2: a * 1 + c is the bf16 add)
DEVI uint32_t fma_bf16x2(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
DEVI uint32_t add_bf16x2(uint32_t a, uint32_t c) { return fma_bf16x2(a, ONE2, c); }

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

// ---- MUFU
DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
DEVI float rcpf(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
DEVI float rsqrtf_(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }

// ---- exact GELU (erf form) and its derivative: Phi(x) = 0.5 erfc(-x / sqrt 2) with the Abramowitz-Stegun 7.1.26 erfc (max abs error 1.5e-7 on erf, far below a bf16 ulp), one ex2 and one rcp
// per element; the derivative Phi(x) + x phi(x) shares the exponential: phi(x) = exp(-x^2 / 2) / sqrt(2 pi).
DEVI float gelu_cdf(float x, float& ex) {
  const float t = rcpf(fmaf(0.23164190f, fabsf(x), 1.0f));                      // 1 / (1 + p z), z = |x| / sqrt 2, p = 0.3275911
  ex = ex2f(-0.72134752044448170f * x * x);                                       // exp(-x^2 / 2) = 2^(-x^2 log2(e) / 2)
  // 0.5 erfc(|x| / sqrt 2) = t (a1 + t (a2 + t (a3 + t (a4 + t a5)))) exp(-x^2 / 2) / 2: the 0.5 is folded into the coefficients
  float poly = fmaf(0.5307027145f, t, -0.7265760135f);
  poly = fmaf(poly, t, 0.7107068705f);
  poly = fmaf(poly, t, -0.1422483680f);
  poly = fmaf(poly, t, 0.1274147960f);
  const float erfc_half = poly * t * ex;
  return x >= 0.f ? 1.0f - erfc_half : erfc_half;
}
DEVI float gelu(float x) { float e; return x * gelu_cdf(x, e); }
// GELU(x) and GELU'(x) = Phi(x) + x phi(x)
DEVI float gelu_with_grad(float x, float& dgelu) {
  float e;
  const float cdf = gelu_cdf(x, e);
  dgelu = fmaf(x * 0.39894228040143268f, e, cdf);
  return x * cdf;
}
DEVI float gelu_grad(float x) { float e; const float cdf = gelu_cdf(x, e); return fmaf(x * 0.39894228040143268f, e, cdf); }

// ---- the dropout draw: the murmur3 finalizer over a Weyl sequence keyed by the seed (a bijection of the 32-bit counter with full avalanche), one 32-bit word per PAIR of elements:
// the low half-word decides the even channel, the high half-word the odd one, an element is kept when its 16 bits are below keep_probability x 65536.  A pair is addressed by
//   counter = row * 64 + (4 a + s) * 4 + q       for the channels 32 a + 8 q + 2 s + (0, 1) of the edge tensor row (a, s, q of the f1 order above).
// About 8 instructions per element: the Triton path's Philox-7 draw costs four times that.
DEVI uint32_t fmix32(uint32_t x) { x ^= x >> 16; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16; return x; }
__host__ __device__ __forceinline__ uint32_t fmix32h(uint32_t x) { x ^= x >> 16; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16; return x; }
__host__ __device__ __forceinline__ uint32_t drop_key(unsigned long long seed) { return fmix32h((uint32_t)seed ^ fmix32h((uint32_t)(seed >> 32) + 0x9E3779B9u)); }
DEVI uint32_t drop_word(uint32_t counter, uint32_t key) { return fmix32(counter * 0x9E3779B1u + key); }
__host__ __device__ __forceinline__ uint32_t keep_threshold(float keep_probability) {
  const float t = keep_probability * 65536.0f;
  return (uint32_t)(t < 65535.0f ? t : 65535.0f);
}
//   counter = (row, 2 (4 a + q) + half),  output i = 2 (s % 2) + e,   for c = 32 a + 8 q + 2 s + e

// ---- shared-memory layout of a weight row (256 B, 16 chunks of 16 B): the chunk index is XOR-ed with 4 on odd rows, so the 2 rows x 4 chunks a quarter warp reads per B-fragment load
// fall in 8 different bank groups
DEVI uint32_t woff(uint32_t row, uint32_t chunk) { return row * 256u + ((chunk ^ ((row & 1u) << 2)) << 4); }
// packed row r -> the logical channel it holds (the f1 order): r = 32 a + 8 s + 2 q + e -> 32 a + 8 q + 2 s + e
DEVI int perm(int r) { return (r & ~31) + 8 * ((r >> 1) & 3) + 2 * ((r >> 3) & 3) + (r & 1); }
// tile rows of 256 B for the transposed ldmatrix operands of the weight-gradient kernel: chunk ^ (row & 7)
DEVI uint32_t sw256(uint32_t row, uint32_t chunk) { return row * 256u + ((chunk ^ (row & 7u)) << 4); }

constexpr int LAYER_BYTES = 128 * 256;       // one packed weight tile
constexpr int IMG_BYTES = 3 * LAYER_BYTES;   // the three layers of a chain
constexpr int TAB_FLOATS = 4 * 128;

}  // namespace me80
