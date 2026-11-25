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
DEVI float2 h2f2(uint32_t v) { float2 r; asm("{ .reg .f16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h; }" : "=f"(r.x), "=f"(r.y) : "r"(v)); return r; }
// math::sigmoid of the Anthropic kit, exactly as the sm_90a kernels evaluate it: rcp.approx(1 + ex2.approx(-a log2 e))
// 2^x on the FMA pipe (no MUFU): x = n + f with n = round(x), f in [-0.5, 0.5], 2^f by a degree-6 fit (max relative error 1.9e-9
// before rounding; measured 7.9e-8 in fp32 against ex2.approx's 1.4e-7), exponent added in the integer domain. x is clamped to [-125, 125], where 1 + 2^x and its
// reciprocal behave exactly as with ex2.approx.ftz for the sigmoid (both ends round to 1 or to a value below bf16 resolution).
DEVI float ex2_poly(float x) {
  x = fminf(fmaxf(x, -125.f), 125.f);
  const float j = __fadd_rn(x, 12582912.f);                  // 1.5 * 2^23: round(x) lands in the low mantissa bits
  const float f = __fsub_rn(x, __fsub_rn(j, 12582912.f));
  float p = fmaf(0.0001533770700916648f, f, 0.0013399861054494977f);
  p = fmaf(p, f, 0.009618518874049187f);
  p = fmaf(p, f, 0.05550329014658928f);
  p = fmaf(p, f, 0.24022646248340607f);
  p = fmaf(p, f, 0.6931471824645996f);
  p = fmaf(p, f, 1.0f);
  return __int_as_float(__float_as_int(p) + (__float_as_int(j) << 23));
}
// 1/d on the FMA pipe: bit-trick seed (relative error <= ~12 %) three Newton steps r <- r (2 - d r) (error squares each step) and a residual correction, for
// the sigmoid's denominator d = 1 + 2^x in [1, 2^125]
// min that keeps a NaN (fminf would return the other operand and hide it)
DEVI float fmin_nan(float a, float b) { float y; asm("min.NaN.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
// the bit-trick reciprocal seed is valid for 0 < d < 2^126 only: a sigmoid denominator 1 + 2^(-a log2 e) passes 2^126 at
// a <= -87.3 and is inf at a <= -88.7, where an unclamped seed turns the Newton steps into NaN / inf. Clamped at 2^125 the result
// is ~2^-125 (the true value is smaller still), a NaN stays NaN.
constexpr float RCP_SEED_MAX = 4.2535295865117308e37f;      // 2^125
DEVI float rcp_nr(float d) {
  d = fmin_nan(d, RCP_SEED_MAX);
  float r = __int_as_float(0x7EF311C3 - __float_as_int(d));
  r = r * fmaf(-d, r, 2.f);
  r = r * fmaf(-d, r, 2.f);
  r = r * fmaf(-d, r, 2.f);
  return fmaf(r, fmaf(-d, r, 1.f), r);                         // final residual correction: ~0.5 ulp
}
// the kit sigmoid with its reciprocal on the FMA pipe (ex2 stays on MUFU)
DEVI float sigmoid_nr(float a) { return rcp_nr(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a)))); }
// the kit sigmoid with the exponential on the FMA pipe instead of MUFU (same formula: rcp.approx(1 + 2^(-a log2 e)))
DEVI float sigmoid_poly(float a) { return rcpf(__fadd_rn(1.f, ex2_poly(__fmul_rn(-1.4426950408889634f, a)))); }
// relaxed-precision sigmoids (opt-in): 0.5 tanh(a / 2) + 0.5 with one MUFU op per element (f32) or per two elements (f16x2)
DEVI float tanhf_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI void sigmoid2_tanh(float a0, float a1, float& s0, float& s1) {
  s0 = fmaf(0.5f, tanhf_approx(0.5f * a0), 0.5f); s1 = fmaf(0.5f, tanhf_approx(0.5f * a1), 0.5f);
}
DEVI void sigmoid2_tanh_h2(float a0, float a1, float& s0, float& s1) {
  uint32_t x, y;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(x) : "f"(0.5f * a1), "f"(0.5f * a0));
  asm("tanh.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x));
  const float2 t = h2f2(y);
  s0 = fmaf(0.5f, t.x, 0.5f); s1 = fmaf(0.5f, t.y, 0.5f);
}
// packed f16x2 arithmetic for the relaxed-precision gate
DEVI uint32_t f2h2(float lo, float hi) { uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI uint32_t hmul2(uint32_t a, uint32_t b) { uint32_t d; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
DEVI uint32_t hadd2(uint32_t a, uint32_t b) { uint32_t d; asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
DEVI uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) { uint32_t d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
DEVI uint32_t htanh2(uint32_t a) { uint32_t d; asm("tanh.approx.f16x2 %0, %1;" : "=r"(d) : "r"(a)); return d; }
// SwiGLU backward in f16x2: s = sigmoid(a) = 0.5 tanh(a/2) + 0.5, l = a s, h = l b, dB = g l, dA = (g b)(s + l (1 - s))
DEVI void gate_h2(uint32_t g, uint32_t a, uint32_t b, uint32_t& h, uint32_t& da, uint32_t& db) {
  const uint32_t HALF = 0x38003800u;                          // (0.5, 0.5)
  const uint32_t s = hfma2(htanh2(hmul2(a, HALF)), HALF, HALF);
  const uint32_t l = hmul2(a, s);
  h = hmul2(l, b);
  db = hmul2(g, l);
  const uint32_t u = hadd2(hfma2(l ^ 0x80008000u, s, l), s);  // l (1 - s) + s
  da = hmul2(hmul2(g, b), u);
}
#if defined(SIG_TANH2)
#define SIGMOID2(a0, a1, s0, s1) sigmoid2_tanh_h2(a0, a1, s0, s1)
#elif defined(SIG_TANH)
#define SIGMOID2(a0, a1, s0, s1) sigmoid2_tanh(a0, a1, s0, s1)
#else
#define SIGMOID2(a0, a1, s0, s1) do { s0 = sigmoid_kit(a0); s1 = sigmoid_kit(a1); } while (0)
#endif
DEVI float sigmoid_kit(float a) { return rcpf(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a)))); }

DEVI uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
DEVI void sts32(uint32_t a, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
DEVI uint4 lds128(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v; }
DEVI void sts128(uint32_t a, uint4 v) { asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
// volatile shared loads for small per-column parameter vectors: plain C++ reads of them get hoisted into 128+ registers
DEVI float2 lds64f(uint32_t a) { float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory"); return v; }
// non-volatile shared loads without a memory clobber: the compiler may overlap them (use only for data no concurrent store touches)
DEVI float2 lds64f_nv(uint32_t a) { float2 v; asm("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a)); return v; }
DEVI uint4 lds128_nv(uint32_t a) { uint4 v; asm("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
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
DEVI void mbar_wait_spin(uint64_t* b, uint32_t parity);
DEVI void mbar_wait(uint64_t* b, uint32_t parity) {
#if defined(MBAR_SPIN)
  while (!mbar_test(b, parity)) { }
#elif defined(MBAR_SLEEP)
  while (!mbar_try_wait(b, parity)) { __nanosleep(MBAR_SLEEP); }   // back off: idle warps draw less power under the cap
#else
  while (!mbar_try_wait(b, parity)) { }
#endif
}

DEVI void mbar_wait_spin(uint64_t* b, uint32_t parity) { while (!mbar_test(b, parity)) { } }   // no suspend
DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
template <int N> DEVI void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N)); }
template <int N> DEVI void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N)); }
DEVI bool elect_one() {
  uint32_t pred;
  asm volatile("{ .reg .pred p; .reg .b32 r; elect.sync r|p, 0xffffffff; selp.u32 %0, 1, 0, p; }" : "=r"(pred));
  return pred != 0;
}
// programmatic dependent launch: wait for the previous grid's completion (and memory flush) / let the next grid be scheduled
DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEVI void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
DEVI void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
DEVI void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }

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
// 3-D tiles (coordinates innermost-first): a per-sample row dimension makes TMA zero-fill loads and clip stores past its end
DEVI void tma_load_3d(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
DEVI void tma_store_3d(const CUtensorMap* m, uint32_t src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
               :: "l"(m), "r"(src), "r"(c0), "r"(c1), "r"(c2) : "memory");
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
// smem (matrix descriptor, same format as an MMA operand) -> TMEM: 128 rows x 32 bytes -> 128 lanes x 8 columns
DEVI void tmem_cp_128x256b(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" :: "r"(taddr), "l"(sdesc) : "memory");
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

// ------------------------------------------------------------------ 2-CTA (cta_group::2) variants: a pair of CTAs in a cluster, the
// leader (rank 0) issues M = 256 products whose A rows and D rows are split 128 / 128 across the pair and whose B is split by N
namespace s100 {
DEVI void tmem_alloc2(uint32_t smem_dst, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_dst), "r"(ncols) : "memory");
}
DEVI void tmem_relinquish2() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory"); }
DEVI void tmem_dealloc2(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}
DEVI void umma_ss2(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p; }"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
DEVI void umma_ts2(uint32_t d_tmem, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p; }"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
// arrive on the barrier at the same offset in every CTA of `mask` once the leader's prior tcgen05 ops are complete
DEVI void tc_commit2_mc(uint64_t* bar, uint16_t mask) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
               :: "r"(smem_u32(bar)), "h"(mask) : "memory");
}
// TMA into this CTA's shared memory, completing the transaction on the LEADER's barrier (peer bit cleared)
DEVI void tma_load_2d_2sm(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar) & 0xFEFFFFFFu), "r"(c0), "r"(c1) : "memory");
}
// arrive (release, cluster scope) on the barrier at the same offset in CTA `rank` of the cluster
DEVI void mbar_arrive_remote(uint64_t* bar, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_u32(bar)), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
}
// relaxed variant: for signals whose data ordering is already carried by tcgen05.fence::before_thread_sync (TMEM reads/writes done)
DEVI void mbar_arrive_remote_relaxed(uint64_t* bar, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_u32(bar)), "r"(rank));
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
}
DEVI void mbar_wait_cl(uint64_t* b, uint32_t parity) {      // acquire at cluster scope (arrivals come from the peer CTA)
#ifdef WAIT_CL_AS_CTA
  mbar_wait(b, parity); return;
#endif
  uint32_t ok = 0;
  while (!ok)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
}
}  // namespace s100

// ------------------------------------------------------------------ fp8 (e4m3) operands, packed f32x2 arithmetic
namespace s100 {
// kind::f8f6f4 instruction descriptor: e4m3 A / B (format 0), fp32 D, a/b major (0 = K, 1 = MN), N, M
__host__ __device__ constexpr uint32_t idesc_e4m3(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
DEVI void umma8_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p; }" :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI void umma8_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p; }" :: "r"(d), "r"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI void umma8_ss2(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p; }" :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI void umma8_ts2(uint32_t d, uint32_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, p; }" :: "r"(d), "r"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
// 64-byte swizzle (layout type 4), 8-row groups 512 B apart: MN-major and K-major variants have the same encoding here
DEVI uint64_t desc_sw64(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)1 << 46) | ((uint64_t)4 << 61);
}
// 64-byte swizzle of a [rows][64 B] tile (512-B aligned): 16-byte chunk q of row r lives at chunk q ^ ((r >> 1) & 3)
DEVI uint32_t sw64(uint32_t r, uint32_t q) { return r * 64u + ((q ^ ((r >> 1) & 3u)) << 4); }
// ------------------------------------------------------------------ kind::tf32 (fp32 operands, fp32 accumulation)
// K-major operands use the usual layouts (desc_k128 / desc_sw64). An MN-major operand must sit in the 128-B swizzle with 32-B
// atomicity (TMA CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, UMMA layout type 1): 32-B chunk j of 128-B row r at chunk j ^ (r & 3), SBO =
// 512 (4-row groups), LBO = the distance between 32-element MN atoms. In the plain 128-B swizzle an MN-major tf32 operand
// multiplies to zeros (measured on B200), and a K-major one in the 32-B-atom layout faults.
__host__ __device__ constexpr uint32_t idesc_tf32(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEVI uint64_t desc_mn32b(uint32_t saddr, uint32_t lbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)(512 >> 4) << 32) |
         ((uint64_t)1 << 46) | ((uint64_t)1 << 61);
}
DEVI void umma_ss_tf32(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
DEVI void umma_ts_tf32(uint32_t d_tmem, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p; }"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
DEVI void tmem_st8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
}
// packed pairs of fp32 (FFMA2 / FMUL2 / FADD2 on sm_100)
struct f2 { uint64_t v; };
DEVI f2 mk2(float lo, float hi) { f2 r; asm("mov.b64 %0, {%1, %2};" : "=l"(r.v) : "f"(lo), "f"(hi)); return r; }
DEVI f2 mk2u(uint32_t lo, uint32_t hi) { f2 r; asm("mov.b64 %0, {%1, %2};" : "=l"(r.v) : "r"(lo), "r"(hi)); return r; }
DEVI float lo2(f2 a) { float l, h; asm("mov.b64 {%0, %1}, %2;" : "=f"(l), "=f"(h) : "l"(a.v)); return l; }
DEVI float hi2(f2 a) { float l, h; asm("mov.b64 {%0, %1}, %2;" : "=f"(l), "=f"(h) : "l"(a.v)); return h; }
DEVI f2 mul2(f2 a, f2 b) { f2 r; asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(r.v) : "l"(a.v), "l"(b.v)); return r; }
DEVI f2 add2(f2 a, f2 b) { f2 r; asm("add.rn.f32x2 %0, %1, %2;" : "=l"(r.v) : "l"(a.v), "l"(b.v)); return r; }
DEVI f2 fma2(f2 a, f2 b, f2 c) { f2 r; asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(r.v) : "l"(a.v), "l"(b.v), "l"(c.v)); return r; }
DEVI f2 neg2(f2 a) { f2 r; r.v = a.v ^ 0x8000000080000000ull; return r; }
// two fp32 -> two e4m3 (saturating) in the low 16 bits; four -> one word (byte k = element k)
DEVI uint32_t e4m3x2(float lo, float hi) { uint16_t q; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(q) : "f"(hi), "f"(lo)); return q; }
DEVI uint32_t e4m3x4(f2 a, f2 b) { return e4m3x2(lo2(a), hi2(a)) | (e4m3x2(lo2(b), hi2(b)) << 16); }
// sigmoid through one MUFU op per element: 0.5 tanh(a / 2) + 0.5 (relaxed precision; tanh.approx relative error ~2^-11)
DEVI f2 sigmoid2(f2 a) {
  const f2 h = mul2(a, mk2(0.5f, 0.5f));
  const f2 t = mk2(tanhf_approx(lo2(h)), tanhf_approx(hi2(h)));
  return fma2(t, mk2(0.5f, 0.5f), mk2(0.5f, 0.5f));
}
}  // namespace s100

// ------------------------------------------------------------------ L2 cache-policy hints for TMA
namespace s100 {
DEVI uint64_t pol_evict_last() { uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p; }
DEVI uint64_t pol_evict_first() { uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }
DEVI void tma_load_2d_h(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
DEVI void tma_load_2d_mc_h(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, uint16_t mask, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5, %6;"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask), "l"(pol) : "memory");
}
DEVI void tma_store_2d_h(const CUtensorMap* m, uint32_t src, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0, {%2, %3}], [%1], %4;" :: "l"(m), "r"(src), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
}  // namespace s100
