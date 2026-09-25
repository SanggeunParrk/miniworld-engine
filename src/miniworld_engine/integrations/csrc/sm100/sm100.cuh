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
// Every tcgen05.mma this thread issued before it arrives on `bar` once they have completed (their
// shared-memory operands are free and the accumulator is final).
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
__device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<const uint32_t*>(&h);
}
// element offset of (row, col) inside a [rows][64] bf16 tile with the 128-byte XOR swizzle
__device__ __forceinline__ int sw128(int row, int col) { return row * 64 + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7); }

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
  CUresult r = tma_encode()(&m, dt, R, const_cast<void*>(base), gdim, gstride, bdim, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(", what, ") failed: ", (int)r);
  return m;
}
inline int num_sms(int dev) {
  static int sms[16] = {0};
  if (sms[dev] == 0) cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev);
  return sms[dev];
}
}  // namespace sm100
#endif
