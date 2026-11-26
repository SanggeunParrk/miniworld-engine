// sm80_common.cuh -- PTX helpers of the A100 (sm_80) TriMul forward kernels: cp.async (+ mbarrier completion), mbarrier, ldmatrix,
// mma.sync m16n8k16 bf16 -> fp32, and the small arithmetic statements shared by K1 and K3.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace a100 {

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---- cp.async (16 B, L2-only .cg); src_bytes 0 zero-fills the destination
DEVI void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes = 16) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> DEVI void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
// every cp.async this thread issued so far arrives on the mbarrier when it completes (the arrival is not counted up front: .noinc)
DEVI void cp_async_mbar_arrive(uint32_t bar) { asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(bar) : "memory"); }

// ---- mbarrier (sm_80: init / arrive / test_wait.parity)
DEVI void mbar_init(uint32_t bar, uint32_t count) { asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory"); }
DEVI void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(bar) : "memory");
}
DEVI bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
DEVI void mbar_wait(uint32_t bar, uint32_t parity) { while (!mbar_test(bar, parity)) { } }

// ---- named barriers (id 0 is __syncthreads)
DEVI void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }

// ---- ldmatrix
DEVI void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEVI void ldsm_x2(uint32_t (&r)[2], uint32_t addr) {   // lanes 0-15 provide the row addresses of the two matrices
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
DEVI void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// ---- mma.sync m16n8k16 bf16 x bf16 -> fp32 (accumulate in place)
DEVI void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---- shared / global vector access
DEVI uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }
DEVI void sts64(uint32_t a, uint2 v) { asm volatile("st.shared.v2.u32 [%0], {%1,%2};\n" ::"r"(a), "r"(v.x), "r"(v.y) : "memory"); }
DEVI uint4 lds128(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
DEVI uint4 lds128_ro(uint32_t a) {   // a shared-memory load of read-only data: not volatile, so the compiler may schedule it ahead of the mma it feeds
  uint4 v; asm("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
DEVI uint4 ldg128(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
DEVI void sts128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEVI uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a)); return v; }
DEVI void sts32(uint32_t a, uint32_t v) { asm volatile("st.shared.u32 [%0], %1;\n" ::"r"(a), "r"(v) : "memory"); }
DEVI void stg128(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEVI void stg32(void* p, uint32_t v) { asm volatile("st.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }

// ---- bf16 packing
DEVI uint32_t pack_bf16(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
DEVI float round_bf16f(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEVI float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
DEVI uint32_t add_bf16x2(uint32_t a, uint32_t b) {   // bf16(a + b) per half: the framework's bf16 residual add
  uint32_t r; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(0x3f803f80u), "r"(b)); return r;   // a * 1 + b, one rounding (sm_80 has no add.bf16x2)
}

// ---- arithmetic statement
DEVI float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
DEVI float sigmoid(float g) { return fmaf(tanh_approx(0.5f * g), 0.5f, 0.5f); }   // one MUFU op
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

// 16-byte-granule XOR swizzle inside a row of `row_bytes` (a multiple of 128): granule g -> g ^ (row & 7)
template <int ROW_BYTES>
DEVI uint32_t swz(uint32_t row, uint32_t byte_in_row) {
  return row * ROW_BYTES + ((((byte_in_row >> 4) ^ (row & 7u)) << 4) | (byte_in_row & 15u));
}

}  // namespace a100
