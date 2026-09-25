// sm100.cuh — the few sm_100a device primitives the Transition kernels use: mbarrier, TMA, tcgen05 (UMMA + TMEM).
// SPDX-License-Identifier: Apache-2.0
// Descriptor bit layouts follow cute/arch/mma_sm100_desc.hpp (SmemDescriptor / InstrDescriptor).  Everything else is plain PTX.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace s100 {

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ------------------------------------------------------------------ scalar helpers (same arithmetic as the sm_90a kernels)
DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI float rcpf(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI uint32_t pack_bf16(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
// math::sigmoid of the Anthropic kit, exactly as the sm_90a kernels evaluate it: rcp.approx(1 + ex2.approx(-a log2 e))
DEVI float sigmoid_kit(float a) { return rcpf(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a)))); }

DEVI uint4 lds128(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v; }
DEVI void sts128(uint32_t a, uint4 v) { asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
// volatile shared loads for small per-column parameter vectors: plain C++ reads of them get hoisted into 128+ registers
DEVI float2 lds64f(uint32_t a) { float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory"); return v; }
DEVI uint4 ldg128(const void* p) { uint4 v; asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
DEVI void stg128(void* p, uint4 v) { asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }

// ------------------------------------------------------------------ mbarrier
DEVI void mbar_init(uint64_t* b, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(n) : "memory"); }
DEVI void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEVI void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory"); }
DEVI void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
DEVI bool mbar_try_wait(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok != 0;
}
DEVI bool mbar_test(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok != 0;
}
DEVI void mbar_wait(uint64_t* b, uint32_t parity) { while (!mbar_try_wait(b, parity)) { } }

DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
template <int N> DEVI void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N)); }
template <int N> DEVI void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N)); }
DEVI bool elect_one() {
  uint32_t pred;
  asm volatile("{ .reg .pred p; .reg .b32 r; elect.sync r|p, 0xffffffff; selp.u32 %0, 1, 0, p; }" : "=r"(pred));
  return pred != 0;
}
DEVI void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }

// ------------------------------------------------------------------ TMA
DEVI void tma_load_2d(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
}
// cluster helpers: rank, full cluster barrier, TMA multicast to the CTAs in `mask`, tcgen05.commit arriving on the same barrier offset
// in every CTA of `mask`
DEVI uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEVI void cluster_sync() { asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory"); }
DEVI void tma_load_2d_mc(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
DEVI void tma_store_2d(const CUtensorMap* m, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" :: "l"(m), "r"(src), "r"(c0), "r"(c1) : "memory");
}
DEVI void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
DEVI void tma_store_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
DEVI void tma_store_wait0() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
DEVI void prefetch_map(const CUtensorMap* m) { asm volatile("prefetch.tensormap [%0];" :: "l"(m) : "memory"); }

// 128-byte swizzle of a [rows][64 bf16] tile (1024-B aligned): 16-byte chunk q of row r lives at chunk q ^ (r & 7)
DEVI uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }

// ------------------------------------------------------------------ UMMA descriptors
// K-major operand, 128-B swizzle, 8-row core groups 1024 B apart (SBO), version 1 (sm_100), layout type 2 (SWIZZLE_128B).
DEVI uint64_t desc_k128(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
}
// MN-major operand, 128-B swizzle: LBO = byte distance between 64-element MN blocks, SBO = distance between 8-row K groups (1024).
DEVI uint64_t desc_mn128(uint32_t saddr, uint32_t lbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
}
// kind::f16 instruction descriptor: bf16 A/B, fp32 D, a/b major (0 = K, 1 = MN), N, M
__host__ __device__ constexpr uint32_t idesc_bf16(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

// ------------------------------------------------------------------ tcgen05
DEVI void tmem_alloc(uint32_t smem_dst, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_dst), "r"(ncols) : "memory");
}
DEVI void tmem_relinquish() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory"); }
DEVI void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
DEVI void tc_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(smem_u32(bar)) : "memory");
}
DEVI void tc_commit_mc(uint64_t* bar, uint16_t mask) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
               :: "r"(smem_u32(bar)), "h"(mask) : "memory");
}
// D[tmem] (+)= A[smem] B[smem]
DEVI void umma_ss(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
// D[tmem] (+)= A[tmem] B[smem]
DEVI void umma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p; }"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
DEVI void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
DEVI void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// 32 lanes x 32 bits, 32 consecutive columns: thread t of the warp gets lane (base lane + t), columns col .. col + 31
DEVI void tmem_ld32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
                 "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]),
                 "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
                 "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr) : "memory");
}
DEVI void tmem_ld16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
                 "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEVI void tmem_st16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]),
                  "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]) : "memory");
}

}  // namespace s100
