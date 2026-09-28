// SPDX-License-Identifier: Apache-2.0
// Device helpers of the D64 training backward (sm_90a): TMA, mbarrier, wgmma (both operands from shared memory),
// ldmatrix / stmatrix, 128-byte swizzled tiles.  All shared tiles are 1024-byte aligned [rows][128 B] (SW128: the
// 16-byte granule index is XORed with row % 8), the layout TMA writes with CU_TENSOR_MAP_SWIZZLE_128B.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__
typedef __nv_bfloat16 bf16;

namespace d64 {

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI uint32_t swz(uint32_t row, uint32_t cb) { return row * 128u + ((((cb >> 4) ^ (row & 7u)) << 4) | (cb & 15u)); }

// ---------------------------------------------------------------- ldmatrix / stmatrix / shared
DEVI void ldsm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
DEVI void ldsm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
DEVI void stsm4(uint32_t a, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
               :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}
DEVI void stsm4t(uint32_t a, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};\n"
               :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}
DEVI uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(a) : "memory"); return v; }
DEVI void sts32(uint32_t a, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(a), "r"(v) : "memory"); }
DEVI float4 lds128f(uint32_t a) {
  float4 v; asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory"); return v;
}
DEVI void sts128f(uint32_t a, float4 v) {
  asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};\n" :: "r"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
DEVI float lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
DEVI uint32_t pack(float a, float b) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(b), "f"(a)); return r; }
DEVI float tanh_approx(float x) { float t; asm("tanh.approx.f32 %0, %1;\n" : "=f"(t) : "f"(x)); return t; }
DEVI float sigm(float g) { return fmaf(0.5f, tanh_approx(0.5f * g), 0.5f); }   // one MUFU op
DEVI float quad(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
DEVI float colsum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  return v;
}

// ---------------------------------------------------------------- barriers
DEVI void mbar_init(uint32_t bar, uint32_t count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count) : "memory"); }
DEVI void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
DEVI void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
DEVI void mbar_expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(bytes) : "memory");
}
DEVI void mbar_wait(uint32_t bar, uint32_t phase) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(bar), "r"(phase) : "memory");
}
DEVI void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }

// ---------------------------------------------------------------- TMA (2D tiles)
DEVI void tma_load(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
DEVI void tma_store(const CUtensorMap* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
DEVI void tma_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
DEVI void tma_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
DEVI void tma_wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
DEVI void prefetch_map(const CUtensorMap* m) { asm volatile("prefetch.tensormap [%0];\n" :: "l"(m) : "memory"); }

// ---------------------------------------------------------------- wgmma
// Matrix descriptor, 128-byte swizzle.  K-major tile [rows][64 k]: start + 32 B per k16 step, SBO 1024.
// MN-major tile [k rows][64 mn]: start + 2048 B per k16 step, SBO 1024, LBO = bytes between 64-wide MN blocks.
DEVI uint64_t desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr >> 4) & 0x3fffu) | ((uint64_t)((lbo >> 4) & 0x3fffu) << 16) |
         ((uint64_t)((sbo >> 4) & 0x3fffu) << 32) | (1ull << 62);
}
DEVI uint64_t kdesc(uint32_t addr) { return desc(addr, 16, 1024); }
DEVI uint64_t mdesc(uint32_t addr, uint32_t lbo = 8192) { return desc(addr, lbo, 1024); }
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> DEVI void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
template <int N> DEVI void fence_acc(float (&a)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}

template <int TA, int TB>
DEVI void mma8(float (&d)[4], uint64_t a, uint64_t b, int acc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, %4, %5, p, 1, 1, %7, %8;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "l"(a), "l"(b), "r"(acc), "n"(TA), "n"(TB));
}
template <int TA, int TB>
DEVI void mma32(float (&d)[16], uint64_t a, uint64_t b, int acc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %18, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, %19, %20;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
               : "l"(a), "l"(b), "r"(acc), "n"(TA), "n"(TB));
}
template <int TA, int TB>
DEVI void mma64(float (&d)[32], uint64_t a, uint64_t b, int acc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
               " %32, %33, p, 1, 1, %35, %36;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                 "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                 "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "l"(a), "l"(b), "r"(acc), "n"(TA), "n"(TB));
}
template <int TA, int TB>
DEVI void mma128(float (&d)[64], uint64_t a, uint64_t b, int acc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %66, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
               "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
               " %64, %65, p, 1, 1, %67, %68;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                 "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                 "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
                 "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
                 "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
                 "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
                 "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
               : "l"(a), "l"(b), "r"(acc), "n"(TA), "n"(TB));
}

// A-fragment (16 x 16) address of a [rows][128 B] tile stored row = m, col = k (ldmatrix / stmatrix, non-transposed)
DEVI uint32_t afrag(uint32_t tile, int row0, int col0, int lane) {
  return tile + swz(row0 + (lane & 7) + 8 * ((lane >> 3) & 1), (col0 + 8 * (lane >> 4)) * 2);
}
// A-fragment address of a tile stored transposed: row = k, col = m (ldmatrix.trans / stmatrix.trans)
DEVI uint32_t afrag_t(uint32_t tile, int k0, int m0, int lane) {
  return tile + swz(k0 + (lane & 7) + 8 * (lane >> 4), (m0 + 8 * ((lane >> 3) & 1)) * 2);
}
// accumulator n-tiles (2t, 2t+1) of a C fragment -> the A fragment of k-step t (bf16)
template <int NT>
DEVI void c2a(uint32_t (&a)[4], const float (&c)[NT * 4], int t) {
  a[0] = pack(c[8 * t + 0], c[8 * t + 1]);
  a[1] = pack(c[8 * t + 2], c[8 * t + 3]);
  a[2] = pack(c[8 * t + 4], c[8 * t + 5]);
  a[3] = pack(c[8 * t + 6], c[8 * t + 7]);
}

}  // namespace d64

// Optional per-phase cycle counters (development builds with -DD64_PROF): each warp sums clock64 deltas per phase and
// lane 0 adds them to prof[phase]; prof[15] counts warps.
#ifdef D64_PROF
#define PF_DECL unsigned long long pf_[16] = {}; long long pt_ = clock64();
#define PF(i) do { const long long t_ = clock64(); pf_[i] += (unsigned long long)(t_ - pt_); pt_ = t_; } while (0)
#define PF_FLUSH(ptr) do { if ((ptr) && (threadIdx.x & 31) == 0) { for (int i_ = 0; i_ < 15; ++i_) atomicAdd((ptr) + i_, pf_[i_]); atomicAdd((ptr) + 15, 1ull); } } while (0)
#else
#define PF_DECL
#define PF(i) do { } while (0)
#define PF_FLUSH(ptr) do { } while (0)
#endif
