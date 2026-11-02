// Blackwell (sm_100a) building blocks shared by the B200 OPM / PWA kernels.
//
// The H100 kernels feed wgmma from 128-byte-swizzled shared tiles and keep the accumulator in
// registers.  sm_100 has no wgmma: the tensor core is driven by ONE thread issuing tcgen05.mma, the
// accumulator lives in tensor memory (TMEM, 128 lanes x 512 32-bit columns per SM), and completion is
// signalled on an mbarrier through tcgen05.commit.  The shared-memory tile formats are the same
// canonical 128-byte-swizzle forms, so every staging layout of the H100 kernels carries over; only the
// descriptor encoding changed (3-bit layout field at [61,64), version bits [46,48) = 1).
//
// Accumulator access (M = 128, cta_group::1): D row m is TMEM lane m, column n is TMEM column n.  A
// warp may touch only its sub-partition's 32 lanes (warp_id % 4), so with tcgen05.ld.32x32b a thread
// holds ONE ROW and consecutive columns -- not the wgmma fragment layout.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cstdint>

namespace sm100 {

__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// ---------------------------------------------------------------- mbarrier
__device__ __forceinline__ void bar_init(uint64_t* b, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(sa(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(sa(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(sa(b)) : "memory");
}
__device__ __forceinline__ void wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}\n"
               :: "r"(sa(b)), "r"(parity) : "memory");
}
// The same wait with a suspend-time hint: the thread sleeps until the phase completes (or ~the hint, in ns)
// instead of re-issuing try_wait -- for warps that wait most of the time, so they do not take issue slots
// from the warps doing the work.
__device__ __forceinline__ void wait_sleep(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAITS_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n@!p bra WAITS_%=;\n}\n"
               :: "r"(sa(b)), "r"(parity), "r"(1000000u) : "memory");
}
// The same operations on 32-bit shared-window addresses: sa() of a generic pointer is an S2R SR_CgaCtaId (tens of clk) each time,
// which on a single-thread issue loop (MMA / TMA warps) sits on the critical path.  Compute a base once and add offsets.
__device__ __forceinline__ void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait(uint32_t b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAITA_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAITA_%=;\n}\n"
               :: "r"(b), "r"(parity) : "memory");
}
// One probe of a phase, no spin: independent probes issue back to back and their ~90 clk round trips overlap; spin (wait) only on the
// ones that were not ready.
__device__ __forceinline__ uint32_t probe(uint32_t b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(parity) : "memory");
  return ok;
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }

// ---------------------------------------------------------------- TMA
__device__ __forceinline__ void prefetch_map(const void* m) { asm volatile("prefetch.tensormap [%0];\n" :: "l"(m) : "memory"); }
__device__ __forceinline__ void load_2d(const void* map, void* dst, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void load_3d(const void* map, void* dst, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void load_4d(const void* map, void* dst, uint64_t* bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void load_2d(const void* map, uint32_t dst, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void store_2d(const void* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void store_2d(const void* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void store_3d(const void* map, const void* src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void store_4d(const void* map, const void* src, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory"); }

// ---------------------------------------------------------------- tcgen05 / TMEM
// Allocation is warp-wide; the column count must be a power of two >= 32.
__device__ __forceinline__ void tmem_alloc(uint32_t* slot, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(sa(slot)), "r"(ncols) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }

// D[tmem] (+)= A[smem] * B[smem]^T, kind::f16 (bf16 in, fp32 accumulate).  Issued by ONE thread.
__device__ __forceinline__ void mma_ss(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate) : "memory");
}
// A from TMEM (row m on lane m, two bf16 per 32-bit column), B from shared.
__device__ __forceinline__ void mma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(accumulate) : "memory");
}
// Predicated forms for a converged issue loop: every lane runs the loop, `leader` (one elected lane) issues -- no branch, so no
// BSSY / BSYNC / WARPSYNC per MMA group.
__device__ __forceinline__ void mma_ts_if(uint32_t leader, uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{\n.reg .pred p, q;\nsetp.ne.b32 p, %4, 0;\nsetp.ne.b32 q, %5, 0;\n"
               "@q tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(accumulate), "r"(leader) : "memory");
}
__device__ __forceinline__ void mma_ss_if(uint32_t leader, uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{\n.reg .pred p, q;\nsetp.ne.b32 p, %4, 0;\nsetp.ne.b32 q, %5, 0;\n"
               "@q tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate), "r"(leader) : "memory");
}
// shared -> TMEM copy of a 128-row x 256-bit block (the matrix descriptor's canonical layout -> lane = row, 8 columns), ordered with
// this thread's tcgen05.mma in the tensor pipe and tracked by tcgen05.commit
__device__ __forceinline__ void tmem_cp_if(uint32_t leader, uint32_t taddr, uint64_t s_desc) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.b32 q, %2, 0;\n@q tcgen05.cp.cta_group::1.128x256b [%0], %1;\n}\n"
               :: "r"(taddr), "l"(s_desc), "r"(leader) : "memory");
}
__device__ __forceinline__ void mma_commit_if(uint32_t leader, uint32_t bar) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.b32 q, %1, 0;\n@q tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}\n"
               :: "r"(bar), "r"(leader) : "memory");
}
// Every tcgen05.mma this thread issued before it arrives on `bar` once they have completed (their
// shared-memory operands are free and the accumulator is final).
__device__ __forceinline__ void mma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" :: "r"(sa(bar)) : "memory");
}

// Instruction descriptor, kind::f16: bf16 x bf16 -> fp32.  a_mn / b_mn: 1 = MN-major operand.
__host__ __device__ constexpr uint32_t idesc_bf16(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4)                       // D = f32
       | (1u << 7) | (1u << 10)          // A, B = bf16
       | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16)
       | ((uint32_t)(N >> 3) << 17)
       | ((uint32_t)(M >> 4) << 24);
}

// Shared-memory matrix descriptor.  swz: 2 = 128B, 4 = 64B, 6 = 32B, 0 = none.
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t swz) {
  uint64_t d = (uint64_t)((saddr >> 4) & 0x3FFFu);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFFu) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFFu) << 32;
  d |= (uint64_t)1 << 46;                // version 1 (sm_100)
  d |= (uint64_t)(swz & 7u) << 61;
  return d;
}
// K-major, 128-byte swizzle: rows of 64 bf16, 8-row atoms of 1 KiB.  Stepping k by 16 = +32 bytes.
__device__ __forceinline__ uint64_t desc_k128(const void* p) { return sdesc(sa(p), 16, 1024, 2); }
// MN-major, 128-byte swizzle: [k][64 mn] rows, 8-k atoms of 1 KiB; `mn_block_bytes` = stride between
// consecutive 64-wide mn blocks.  Stepping k by 16 = +2048 bytes.
__device__ __forceinline__ uint64_t desc_mn128(const void* p, uint32_t mn_block_bytes) { return sdesc(sa(p), mn_block_bytes, 1024, 2); }
// K-major, 64-byte swizzle: rows of 32 bf16, 8-row atoms of 512 B.
__device__ __forceinline__ uint64_t desc_k64(const void* p) { return sdesc(sa(p), 16, 512, 4); }

// TMEM -> registers.  32 lanes x 32 bits, `N` consecutive columns per thread.
#define SM100_LD_X(NREG, ...)                                                                                  \
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x" #NREG ".b32 " __VA_ARGS__ : "memory")
__device__ __forceinline__ void tmem_ld8(uint32_t taddr, float* v) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(taddr) : "memory");
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float* v) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float* v) {
  tmem_ld16(taddr, v);
  tmem_ld16(taddr + 16, v + 16);
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
// registers -> TMEM, 8 consecutive columns
__device__ __forceinline__ void tmem_st8(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
// the TMEM address of (lane = row, column): a warp's sub-partition owns lanes 32 * (warp % 4) ...
// one lane of the warp (elect.sync): the warp-converged way to issue a single-thread instruction
__device__ __forceinline__ bool elect_one() {
  uint32_t pred = 0;
  asm volatile("{\n.reg .pred P;\nelect.sync _|P, 0xffffffff;\nselp.u32 %0, 1, 0, P;\n}\n" : "=r"(pred));
  return pred != 0;
}
__device__ __forceinline__ uint32_t tmem_at(uint32_t base, int lane, int col) { return base + ((uint32_t)lane << 16) + (uint32_t)col; }

// packed fp32 pairs (sm_100: FADD2 / FMUL2 / FFMA2 issue two fp32 lanes per instruction)
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
  float2 d;
  asm("{\n.reg .b64 ra, rb, rd;\nmov.b64 ra, {%2, %3};\nmov.b64 rb, {%4, %5};\nadd.rn.f32x2 rd, ra, rb;\nmov.b64 {%0, %1}, rd;\n}\n"
      : "=f"(d.x), "=f"(d.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}
__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
  float2 d;
  asm("{\n.reg .b64 ra, rb, rd;\nmov.b64 ra, {%2, %3};\nmov.b64 rb, {%4, %5};\nmul.rn.f32x2 rd, ra, rb;\nmov.b64 {%0, %1}, rd;\n}\n"
      : "=f"(d.x), "=f"(d.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
  float2 d;
  asm("{\n.reg .b64 ra, rb, rc, rd;\nmov.b64 ra, {%2, %3};\nmov.b64 rb, {%4, %5};\nmov.b64 rc, {%6, %7};\nfma.rn.f32x2 rd, ra, rb, rc;\nmov.b64 {%0, %1}, rd;\n}\n"
      : "=f"(d.x), "=f"(d.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
  return d;
}
// bf16 pair (one 32-bit word) -> two fp32: the low half is the first element
__device__ __forceinline__ float2 bf2f(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
__device__ __forceinline__ float to_f(float x) { return x; }
template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<const uint32_t*>(&h);
}
// element offset of (row, col) inside a [rows][64] bf16 tile with the 128-byte XOR swizzle
__device__ __forceinline__ int sw128(int row, int col) { return row * 64 + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7); }


// ---------------------------------------------------------------- deterministic split reduction
// out[w] = sum_k part[k][w] in split order (two passes: chunks of splits, then the chunk sums), written as OutT.
// torch's dim-0 reduce of a tall, thin buffer and cuBLAS's GEMV both run far below bandwidth on these shapes.
template <typename OutT>
__global__ void __launch_bounds__(256) colsum_kernel(const float* __restrict__ P, OutT* __restrict__ OUT, float* __restrict__ TMP,
                                                     int splits, int chunk, int G, long width, int final_pass) {
  const long vcols = width >> 2;
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long v4 = t % vcols; const int g = (int)(t / vcols);
  if (g >= G) return;
  const int k0 = g * chunk, k1 = min(k0 + chunk, splits);
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
  for (int k = k0; k < k1; ++k) {
    const float4 q = __ldg(reinterpret_cast<const float4*>(P + (long)k * width) + v4);
    s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
  }
  if (!final_pass) { reinterpret_cast<float4*>(TMP + (long)g * width)[v4] = s; return; }
  OutT* o = OUT + v4 * 4;
  o[0] = from_f<OutT>(s.x); o[1] = from_f<OutT>(s.y); o[2] = from_f<OutT>(s.z); o[3] = from_f<OutT>(s.w);
}
// One launch: block = 32 float4 columns x 8 split groups; group g sums splits g, g + 8, ... (four loads in flight), then
// group 0 adds the 8 group sums in order.  Deterministic; the two-pass version paid two latency-bound launches.
template <typename OutT>
__global__ void __launch_bounds__(1024) colsum1_kernel(const float* __restrict__ P, OutT* __restrict__ OUT, int splits, long width) {
  extern __shared__ float4 red1[];                // [G][32]
  const int G = blockDim.x >> 5;                  // split groups: 8, or 32 for a narrow sum (few blocks, more loads in flight)
  const long v4 = (long)blockIdx.x * 32 + (threadIdx.x & 31);
  const int g = threadIdx.x >> 5;
  const long vcols = width >> 2;
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  if (v4 < vcols) {
#pragma unroll 4
    for (int k = g; k < splits; k += G) {
      const float4 q = __ldg(reinterpret_cast<const float4*>(P + (long)k * width) + v4);
      s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
    }
  }
  red1[g * 32 + (threadIdx.x & 31)] = s;
  __syncthreads();
  if (g == 0 && v4 < vcols) {
    float4 t = red1[threadIdx.x];
    for (int k = 1; k < G; ++k) { const float4 q = red1[k * 32 + threadIdx.x]; t.x += q.x; t.y += q.y; t.z += q.z; t.w += q.w; }
    OutT* o = OUT + v4 * 4;
    o[0] = from_f<OutT>(t.x); o[1] = from_f<OutT>(t.y); o[2] = from_f<OutT>(t.z); o[3] = from_f<OutT>(t.w);
  }
}
}  // namespace sm100

// ---------------------------------------------------------------- host: tensor maps
#ifndef SM100_NO_HOST
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
namespace sm100 {
// CUDA 13 drops the unversioned PFN name and cudaGetDriverEntryPoint: ask for the 12.0 ABI explicitly.
using encode_fn = PFN_cuTensorMapEncodeTiled_v12000;
inline encode_fn tma_encode() {
  static encode_fn fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult qr;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qr));
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<encode_fn>(p);
  }
  return fn;
}
// rank-R bf16 map; dims / box innermost first; strides in ELEMENTS for dims 1..R-1 (need not be monotonic)
template <int R>
inline CUtensorMap make_map(const void* base, const uint64_t (&dims)[R], const uint64_t (&strides_el)[R - 1],
                            const uint32_t (&box)[R], CUtensorMapSwizzle sw, const char* what,
                            CUtensorMapDataType dt = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, int esize = 2) {
  alignas(64) CUtensorMap m{};
  uint64_t gstride[R > 1 ? R - 1 : 1];
  for (int i = 0; i < R - 1; ++i) gstride[i] = strides_el[i] * (uint64_t)esize;
  uint32_t estride[R];
  for (int i = 0; i < R; ++i) estride[i] = 1;
  uint64_t gdim[R];
  uint32_t bdim[R];
  for (int i = 0; i < R; ++i) { gdim[i] = dims[i]; bdim[i] = box[i]; }
  // the driver call needs a current context, which a fresh thread (autograd's backward thread) may not have yet
  // (cudaSetDevice makes the primary context current without a legacy-stream operation: cudaFree(nullptr) here broke a CUDA-graph
  // capture whenever the capturing thread built its first map)
  static thread_local bool ctx_ok = false;
  if (!ctx_ok) { int dev = 0; C10_CUDA_CHECK(cudaGetDevice(&dev)); C10_CUDA_CHECK(cudaSetDevice(dev)); ctx_ok = true; }
  CUresult r = tma_encode()(&m, dt, R, const_cast<void*>(base), gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(", what, ") failed: ", (int)r);
  return m;
}

// sum a [splits, ...] fp32 buffer over dim 0 -> dtype (fp32 or bf16), deterministic
inline torch::Tensor colsum(const torch::Tensor& part, torch::ScalarType dtype = torch::kFloat32) {
  TORCH_CHECK(part.scalar_type() == torch::kFloat32 && part.is_contiguous(), "colsum: contiguous fp32 partials");
  const int splits = (int)part.size(0);
  const long width = part.numel() / splits;
  TORCH_CHECK(width % 4 == 0, "colsum: width must be a multiple of 4");
  auto out = torch::empty(part.sizes().slice(1), part.options().dtype(dtype));
  const long vcols = width / 4;
  if ((vcols + 31) / 32 >= 128 || splits <= 256) {   // wide, or few splits (<= 32 loads a thread): one launch; else split groups
    auto st1 = at::cuda::getCurrentCUDAStream();
    const int blocks = (int)((vcols + 31) / 32);
    const int G = blocks < 32 ? 32 : 8, thr = 32 * G, sm1 = G * 32 * 16;
    if (dtype == torch::kFloat32) colsum1_kernel<float><<<blocks, thr, sm1, st1>>>(part.data_ptr<float>(), out.data_ptr<float>(), splits, width);
    else colsum1_kernel<__nv_bfloat16><<<blocks, thr, sm1, st1>>>(part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), splits, width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
  long G = 16384 / std::max<long>(vcols, 1);
  G = std::max<long>(1, std::min<long>({G, 64, (long)splits / 4}));
  const int chunk = (int)((splits + G - 1) / G);
  G = (splits + chunk - 1) / chunk;
  auto st = at::cuda::getCurrentCUDAStream();
  auto launch = [&](const float* src, int s_, int c_, int g_, float* tmp, int fin) {
    const long threads = vcols * g_;
    const int blocks = (int)((threads + 255) / 256);
    if (dtype == torch::kFloat32) colsum_kernel<float><<<blocks, 256, 0, st>>>(src, out.data_ptr<float>(), tmp, s_, c_, g_, width, fin);
    else colsum_kernel<__nv_bfloat16><<<blocks, 256, 0, st>>>(src, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), tmp, s_, c_, g_, width, fin);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  if (G == 1) { launch(part.data_ptr<float>(), splits, splits, 1, nullptr, 1); return out; }
  auto tmp = torch::empty({G, width}, part.options());
  launch(part.data_ptr<float>(), splits, chunk, (int)G, tmp.data_ptr<float>(), 0);
  launch(tmp.data_ptr<float>(), (int)G, (int)G, 1, nullptr, 1);
  return out;
}
inline int num_sms(int dev) {
  static int sms[16] = {0};
  if (sms[dev] == 0) cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev);
  return sms[dev];
}
}  // namespace sm100
#endif
