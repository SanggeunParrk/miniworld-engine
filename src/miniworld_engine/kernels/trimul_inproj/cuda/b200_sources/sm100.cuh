// sm100.cuh -- thin PTX wrappers for B200 (sm_100a): mbarrier, TMA (cp.async.bulk.tensor), tcgen05 (UMMA / TMEM).
// Encodings follow the PTX ISA and CUTLASS 4.8 cute/arch/mma_sm100_desc.hpp (SmemDescriptor, InstrDescriptor).
#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>

#define DEV __device__ __forceinline__

namespace sm100 {

DEV uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// ------------------------------------------------------------------ mbarrier
DEV void mbar_init(uint64_t* b, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count));
}
DEV void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEV void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
DEV void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
#ifdef MBAR_DEBUG
__shared__ volatile int g_dbg_s[8];
#define MBAR_DBG_PTR g_dbg_s
DEV void mbar_wait(uint64_t* b, uint32_t parity) {
  for (long long n = 0;; ++n) {
    uint32_t ok;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
    if (ok) return;
    if (n == 20000000) {
#ifdef MBAR_DBG_PTR
      const volatile int* d = MBAR_DBG_PTR;
      printf("MBAR HANG blk %d thr %d bar_off %u parity %u dbg %d %d %d %d %d\n", blockIdx.x, threadIdx.x, smem_u32(b), parity, d[0], d[1], d[2], d[3], d[4]);
#else
      printf("MBAR HANG blk %d thr %d bar_off %u parity %u\n", blockIdx.x, threadIdx.x, smem_u32(b), parity);
#endif
    }
    if (n == 400000000) asm volatile("trap;");
  }
}
#else
DEV void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=:\n\t"
#ifdef SPIN_WAIT
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
#else
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, 10000000;\n\t"
#endif
      "@!p bra WAIT_%=;\n\t}" ::"r"(smem_u32(b)),
      "r"(parity)
      : "memory");
}
#endif

// ------------------------------------------------------------------ TMA
DEV void prefetch_tmap(const CUtensorMap* m) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(m)) : "memory");
}
DEV void tma_load_2d(void* dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, uint64_t hint) {
#ifdef NO_HINT
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(m)), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
  return;
#endif
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(m)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(hint)
      : "memory");
}
DEV void tma_load_3d(void* dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, int c2, uint64_t hint) {
#ifdef NO_HINT
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(m)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2) : "memory");
  return;
#endif
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(m)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "l"(hint)
      : "memory");
}
DEV void tma_store_3d(const CUtensorMap* m, const void* src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];" ::"l"(
                   reinterpret_cast<uint64_t>(m)),
               "r"(smem_u32(src)), "r"(c0), "r"(c1), "r"(c2)
               : "memory");
}
DEV void tma_store_2d(const CUtensorMap* m, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" ::"l"(reinterpret_cast<uint64_t>(m)),
               "r"(smem_u32(src)), "r"(c0), "r"(c1)
               : "memory");
}
DEV void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N>
DEV void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory"); }
template <int N>
DEV void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory"); }
DEV void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// L2 cache-policy hints (createpolicy constants as used by CUTLASS TMA::CacheHintSm90)
constexpr uint64_t EVICT_NORMAL = 0x1000000000000000ull;
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ull;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000ull;

// ------------------------------------------------------------------ tcgen05 / TMEM
// Allocation is warp-collective; the TMEM base address is written to *dst (shared memory).
DEV void tmem_alloc(uint32_t* dst, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst)), "r"(ncols) : "memory");
}
DEV void tmem_relinquish() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory"); }
DEV void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
DEV void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEV void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// D[tmem] (+)= A[smem desc] * B[smem desc]^T, kind::f16 (bf16 in, fp32 accumulate). Issued by ONE thread.
DEV void umma_ss(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(tmem_d),
      "l"(da), "l"(db), "r"(idesc), "r"(accumulate)
      : "memory");
}
// D[tmem] (+)= A[tmem] * B[smem desc]^T, kind::f16. A: M lanes x K (bf16 pairs packed per 32-bit column).
DEV void umma_ts(uint32_t tmem_d, uint32_t tmem_a, uint64_t db, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}" ::"r"(tmem_d),
      "r"(tmem_a), "l"(db), "r"(idesc), "r"(accumulate)
      : "memory");
}
// Arrive on an mbarrier once every previously issued tcgen05.mma of this thread has completed.
DEV void umma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}

// 32 lanes x 32 columns of 32-bit: each thread of the warp gets its lane's 32 consecutive columns.
DEV void tmem_ld32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
}
DEV void tmem_ld16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr));
}
DEV void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

// Shared-memory matrix descriptor, K-major operand with 128-byte swizzle (rows of 128 B, 8-row groups of 1024 B).
// start/LBO/SBO in 16-byte units; version 1 (sm_100); layout type 2 = SWIZZLE_128B. The tile base must be 1024-B aligned.
DEV uint64_t desc_k_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)(1) << 16;               // LBO (ignored for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;       // SBO: 8 rows x 128 B
  d |= (uint64_t)1 << 46;                 // version
  d |= (uint64_t)2 << 61;                 // SWIZZLE_128B
  return d;
}
// Instruction descriptor: kind::f16, A/B = BF16 K-major, D = F32, shape M x N.
__host__ __device__ constexpr uint32_t idesc_bf16(int M, int N) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

DEV float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
DEV float bf16hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
DEV uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
DEV int lane_id() { return threadIdx.x & 31; }
DEV int warp_id() { return __shfl_sync(0xffffffff, threadIdx.x >> 5, 0); }
DEV bool elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t.reg .pred P;\n\t.reg .b32 R;\n\t"
      "elect.sync R|P, 0xffffffff;\n\t"
      "selp.b32 %0, 1, 0, P;\n\t}"
      : "=r"(pred));
  return pred != 0;
}

// MN-major operand, 128-byte swizzle: rows of 64 MN-elements (128 B) indexed by K, 8-row K groups 1024 B apart (SBO);
// LBO = byte distance between consecutive 64-element MN blocks. (Validated in transition_fused_sm100.)
DEV uint64_t desc_mn_sw128(uint32_t saddr, uint32_t lbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)(1024 >> 4) << 32) |
         ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
}
// Instruction descriptor with operand majorness: a_mn / b_mn = 1 selects MN-major A / B.
__host__ __device__ constexpr uint32_t idesc_bf16_mj(int M, int N, int a_mn, int b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEV void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
DEV void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
DEV void mbar_arrive_cnt(uint64_t* b, uint32_t c) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(c) : "memory");
}
DEV void sts128(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
DEV float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rcpf(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
template <int N> DEV void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N)); }
template <int N> DEV void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N)); }



// ------------------------------------------------------------------ clusters / distributed shared memory
DEV uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEV uint32_t cluster_idx() { uint32_t r; asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(r)); return r; }
DEV uint32_t cluster_num() { uint32_t r; asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(r)); return r; }
DEV void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
DEV void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
// shared::cta address of this CTA -> shared::cluster address of the same offset in CTA `rank`
DEV uint32_t mapa(uint32_t saddr, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(saddr), "r"(rank)); return r;
}
// 16-byte store into a peer CTA's shared memory, completing tx bytes on the peer's mbarrier (both cluster addresses)
DEV void st_async_v4(uint32_t raddr, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(raddr), "r"(a), "r"(b), "r"(c), "r"(d), "r"(rbar) : "memory");
}
DEV void mbar_arrive_remote(uint32_t rbar) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(rbar) : "memory");
}
DEV void mbar_expect_tx_only(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
// acquire at cluster scope for data delivered by st.async / remote arrivals
DEV void mbar_wait_cluster(uint64_t* b, uint32_t parity) {
  uint32_t ok = 0;
  while (!ok)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
}
// K-major operand with 32-byte swizzle (rows of 16 bf16 = 32 B, 8-row groups 256 B apart); layout type 6 = SWIZZLE_32B
DEV uint64_t desc_k_sw32(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)1 << 16) | ((uint64_t)(256 >> 4) << 32) | ((uint64_t)1 << 46) | ((uint64_t)6 << 61);
}
DEV void stsm_x4(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
// ------------------------------------------------------------------ optional role profiling (build with -DPROF=1)
// PROF_BEGIN in the kernel body after setup; PW(slot, stmt) accumulates the cycles of stmt; PROF_END(row) stores the per-thread
// counters + total into g_prof[blockIdx.x][row][0..7] (slot 7 = total). Read with prof_dump().
#ifdef PROF
__device__ unsigned long long g_prof[160][8][8];
#define PROF_BEGIN unsigned long long pacc_[7] = {0, 0, 0, 0, 0, 0, 0}; const long long pt0_ = clock64();
#define PW(slot, stmt) { const long long _t0 = clock64(); stmt; pacc_[slot] += clock64() - _t0; }
#define PROF_END(row) { for (int k_ = 0; k_ < 7; ++k_) g_prof[blockIdx.x][row][k_] = pacc_[k_]; g_prof[blockIdx.x][row][7] = clock64() - pt0_; }
#else
#define PROF_BEGIN
#define PW(slot, stmt) stmt;
#define PROF_END(row)
#endif
}  // namespace sm100
