// Hopper (sm_90a) building blocks for the H100 TriangleAttention kernels: mbarrier, TMA, wgmma (raw PTX, no CUTLASS / CuTe).
//
// The mbarrier / TMA wrappers are the same PTX the B200 kernels use (b200_sources/sm100.cuh); the tensor core differs: sm_90 issues
// wgmma per warpgroup (128 threads), the accumulator lives in registers, and A may come from registers (RS) or shared memory (SS).
//
// wgmma accumulator fragment, m64nN (fp32): thread T of the warpgroup (warp w = T / 32, g = lane / 4, t = lane % 4) holds, for each
// 8-column block i = 0 .. N/8 - 1:
//   d[4i + 0] = (row 16w + g,     col 8i + 2t)      d[4i + 1] = (row 16w + g,     col 8i + 2t + 1)
//   d[4i + 2] = (row 16w + g + 8, col 8i + 2t)      d[4i + 3] = (row 16w + g + 8, col 8i + 2t + 1)
// A fragment from registers, m64k16 (bf16): a[0] = (row g, k 2t..2t+1), a[1] = (row g + 8, k 2t..), a[2] = (row g, k 8 + 2t..),
// a[3] = (row g + 8, k 8 + 2t..) -- so the accumulator blocks 2j, 2j + 1 of an m64nN product, packed to bf16 pairs, ARE the A
// fragment of k16 block j (the FlashAttention-3 relabelling).
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cstdint>

namespace sm90 {

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
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void named_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }

// ---------------------------------------------------------------- TMA
__device__ __forceinline__ void prefetch_map(const void* m) { asm volatile("prefetch.tensormap [%0];\n" :: "l"(m) : "memory"); }
__device__ __forceinline__ void load_2d(const void* map, void* dst, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1) : "memory");
}
// plain bulk copy global -> shared (no tensor map), bytes % 16 == 0, both addresses 16-byte aligned
__device__ __forceinline__ void load_1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void store_2d(const void* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void store_2d(const void* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory"); }

// ---------------------------------------------------------------- warpgroup register budget
template <int N> __device__ __forceinline__ void reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N)); }
template <int N> __device__ __forceinline__ void reg_alloc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N)); }

// ---------------------------------------------------------------- wgmma
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
// Keep the compiler from moving accesses of wgmma operands across the (register-invisible) async boundary.
template <int N> __device__ __forceinline__ void fence_regs(float (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i]) :: "memory");
}
template <int N> __device__ __forceinline__ void fence_regs(uint32_t (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i]) :: "memory");
}

// Shared-memory matrix descriptor (sm_90): start >> 4 at [0, 14), leading byte offset >> 4 at [16, 30), stride byte offset >> 4
// at [32, 46), layout at [62, 64): 1 = 128B swizzle, 2 = 64B, 3 = 32B, 0 = none.
__device__ __forceinline__ uint64_t gdesc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t layout) {
  uint64_t d = (uint64_t)((saddr >> 4) & 0x3FFFu);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFFu) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFFu) << 32;
  d |= (uint64_t)(layout & 3u) << 62;
  return d;
}
constexpr uint32_t SW128 = 1, SW64 = 2, SW32 = 3;
// A descriptor moved by `bytes` (a multiple of 16) inside its own window: only the 14-bit start field (low word) changes, so a
// descriptor built once outside a loop costs one 32-bit add per use instead of being re-assembled.
__device__ __forceinline__ uint64_t desc_off(uint64_t d, uint32_t bytes) {
  uint64_t r;
  asm("{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, %1;\nadd.u32 lo, lo, %2;\nmov.b64 %0, {lo, hi};\n}\n" : "=l"(r) : "l"(d), "r"(bytes >> 4));
  return r;
}
// K-major, 64-byte swizzle: rows of 32 bf16 (64 B), 8-row atoms of 512 B; a k16 step = +32 bytes.
__device__ __forceinline__ uint64_t desc_k64(uint32_t saddr) { return gdesc(saddr, 16, 512, SW64); }
// K-major, 128-byte swizzle: rows of 64 bf16, 8-row atoms of 1 KiB; a k16 step = +32 bytes.
__device__ __forceinline__ uint64_t desc_k128(uint32_t saddr) { return gdesc(saddr, 16, 1024, SW128); }
// MN-major, 64-byte swizzle: [k][32 mn] rows of 64 B, 8-k atoms of 512 B; a k16 step = +1024 bytes; `mn_block_bytes` = stride between
// 32-wide mn blocks (unused when N = 32).
__device__ __forceinline__ uint64_t desc_mn64(uint32_t saddr, uint32_t mn_block_bytes) { return gdesc(saddr, mn_block_bytes, 512, SW64); }
// MN-major, 128-byte swizzle: [k][64 mn] rows of 128 B, 8-k atoms of 1 KiB; a k16 step = +2048 bytes.
__device__ __forceinline__ uint64_t desc_mn128(uint32_t saddr, uint32_t mn_block_bytes) { return gdesc(saddr, mn_block_bytes, 1024, SW128); }

#define SM90_D8(b) "+f"(d[b]), "+f"(d[b + 1]), "+f"(d[b + 2]), "+f"(d[b + 3]), "+f"(d[b + 4]), "+f"(d[b + 5]), "+f"(d[b + 6]), "+f"(d[b + 7])
#define SM90_D16(b) SM90_D8(b), SM90_D8(b + 8)
#define SM90_D32(b) SM90_D16(b), SM90_D16(b + 16)
#define SM90_D64(b) SM90_D32(b), SM90_D32(b + 32)
#define SM90_L16 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}"
#define SM90_L32 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, " \
                 "%24, %25, %26, %27, %28, %29, %30, %31}"
#define SM90_L64 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, " \
                 "%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, "   \
                 "%46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}"

// D (+)= A . B, A / B from shared (descriptors); TA / TB: 0 = K-major, 1 = MN-major.  scale_d = 0 overwrites D.
template <int TA, int TB>
__device__ __forceinline__ void mma_ss_n128(float (&d)[64], uint64_t a, uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " SM90_L64 ", %64, %65, p, 1, 1, %67, %68;\n}\n"
               : SM90_D64(0) : "l"(a), "l"(b), "r"(scale_d), "n"(TA), "n"(TB));
}
template <int TA, int TB>
__device__ __forceinline__ void mma_ss_n8(float (&d)[4], uint64_t a, uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, %7, %8;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(a), "l"(b), "r"(scale_d), "n"(TA), "n"(TB));
}
template <int TA, int TB>
__device__ __forceinline__ void mma_ss_n64(float (&d)[32], uint64_t a, uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " SM90_L32 ", %32, %33, p, 1, 1, %35, %36;\n}\n"
               : SM90_D32(0) : "l"(a), "l"(b), "r"(scale_d), "n"(TA), "n"(TB));
}
// D (+)= A . B, A from registers (bf16 pairs, the fragment above), B from shared; TB: 0 = K-major, 1 = MN-major.
template <int TB>
__device__ __forceinline__ void mma_rs_n128(float (&d)[64], const uint32_t (&a)[4], uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " SM90_L64 ", {%64, %65, %66, %67}, %68, p, 1, 1, %70;\n}\n"
               : SM90_D64(0) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(scale_d), "n"(TB));
}
template <int TB>
__device__ __forceinline__ void mma_rs_n64(float (&d)[32], const uint32_t (&a)[4], uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " SM90_L32 ", {%32, %33, %34, %35}, %36, p, 1, 1, %38;\n}\n"
               : SM90_D32(0) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(scale_d), "n"(TB));
}
template <int TB>
__device__ __forceinline__ void mma_rs_n32(float (&d)[16], const uint32_t (&a)[4], uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 " SM90_L16 ", {%16, %17, %18, %19}, %20, p, 1, 1, %22;\n}\n"
               : SM90_D16(0) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(scale_d), "n"(TB));
}

template <int TB>
__device__ __forceinline__ void mma_rs_n40(float (&d)[20], const uint32_t (&a)[4], uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %25, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
               "%15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, %26;\n}\n"
               : SM90_D16(0), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(scale_d), "n"(TB));
}
// D (+)= A . B, both from shared (descriptors): m64n32k16; TA / TB: 0 = K-major, 1 = MN-major.
template <int TA, int TB>
__device__ __forceinline__ void mma_ss_n32(float (&d)[16], uint64_t a, uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 " SM90_L16 ", %16, %17, p, 1, 1, %19, %20;\n}\n"
               : SM90_D16(0) : "l"(a), "l"(b), "r"(scale_d), "n"(TA), "n"(TB));
}
template <int TB>
__device__ __forceinline__ void mma_rs_n16(float (&d)[8], const uint32_t (&a)[4], uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, %14;\n}\n"
               : SM90_D8(0) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(scale_d), "n"(TB));
}
__device__ __forceinline__ void mma_rs_n8(float (&d)[4], const uint32_t (&a)[4], uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(scale_d));
}

// ---------------------------------------------------------------- mbarrier on 32-bit shared addresses (no generic -> shared
// conversion per call: sa() of a generic pointer costs an S2R each time, on the critical path of single-thread loops)
__device__ __forceinline__ void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait(uint32_t b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAITA_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAITA_%=;\n}\n"
               :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void load_2d(const void* map, uint32_t dst, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}

// ---------------------------------------------------------------- shared-memory fragment loads
__device__ __forceinline__ void ldsm_x4(uint32_t saddr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr) : "memory");
}
// four 8 x 8 bf16 matrices from the accumulator-pair layout (lane l: row l / 4, columns 2 (l % 4), +1 of each) to shared memory; lanes
// 8i .. 8i + 7 give the row addresses of matrix i
__device__ __forceinline__ void stsm_x4(uint32_t saddr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(saddr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t saddr, uint32_t (&r)[4]) {   // transposed 8 x 8 matrices
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr) : "memory");
}
// bulk reduce-add of a shared tile into global memory through a tensor map (fp32 add in L2)
__device__ __forceinline__ void reduce_add_2d(const void* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ float2 lds64f(uint32_t saddr) {
  float2 v;
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(saddr) : "memory");
  return v;
}
__device__ __forceinline__ float4 lds128(uint32_t saddr) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(saddr) : "memory");
  return v;
}

// 4-D tile transfers: the pair tensor of an ending node is read / written through a map (c, j, i, b) with strides (1, L C, C, L L C), so the
// kernels see the transposed pair [b, i, j] without a copy.  pair_* take the 2-D row of the starting layout and L; tr = 0 keeps 2-D.
__device__ __forceinline__ void load_4d(const void* map, uint32_t dst, uint32_t bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void store_4d(const void* map, uint32_t src, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void reduce_add_4d(const void* map, uint32_t src, int c0, int c1, int c2, int c3) {
  asm volatile("cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void pair_coord(int row, int L, int& i, int& j, int& b) {
  b = row / (L * L);
  const int rem = row - b * L * L;
  i = rem / L;
  j = rem - i * L;
}
__device__ __forceinline__ void load_pair(const void* map, uint32_t dst, uint32_t bar, int c0, int row, int L, int tr) {
  if (!tr) { load_2d(map, dst, bar, c0, row); return; }
  int i, j, b; pair_coord(row, L, i, j, b);
  load_4d(map, dst, bar, c0, j, i, b);
}
__device__ __forceinline__ void store_pair(const void* map, uint32_t src, int c0, int row, int L, int tr) {
  if (!tr) { store_2d(map, src, c0, row); return; }
  int i, j, b; pair_coord(row, L, i, j, b);
  store_4d(map, src, c0, j, i, b);
}
__device__ __forceinline__ void reduce_add_pair(const void* map, uint32_t src, int c0, int row, int L, int tr) {
  if (!tr) { reduce_add_2d(map, src, c0, row); return; }
  int i, j, b; pair_coord(row, L, i, j, b);
  reduce_add_4d(map, src, c0, j, i, b);
}

// ---------------------------------------------------------------- math
__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// 2^x on the FMA pipe (the MUFU is the softmax's bound): n = round(x) by the 1.5 * 2^23 add, 2^f (|f| <= 1/2) by a degree-3 minimax
// polynomial (relative error 7.5e-5, well under P's bf16 rounding), n added into the exponent bits.  x is clamped at -126.
__device__ __forceinline__ float ex2_poly(float x) {
  x = fmaxf(x, -126.f);
  const float t = x + 12582912.f;
  const float f = x - (t - 12582912.f);
  float p = fmaf(f, 0.05517097723f, 0.24260970271f);
  p = fmaf(f, p, 0.69326096627f);
  p = fmaf(f, p, 0.99992816244f);
  return __int_as_float(__float_as_int(p) + (__float_as_int(t) << 23));
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {        // a -> low half (the first element)
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a));
  return r;
}
// bf16 pair rounding on the integer pipes: round half up (differs from cvt.rn only on exact ties) + one PRMT.  cvt.rn.bf16x2 runs on
// the XU pipe -- the MUFU's pipe, 16 / clk / SM -- so in the attention loops (an exponential and a pack per score) it halves the
// exponentials' throughput.
__device__ __forceinline__ uint32_t pack2i(float a, float b) {
  return __byte_perm(__float_as_uint(a) + 0x8000u, __float_as_uint(b) + 0x8000u, 0x7632);
}
#ifndef TA_PACKI
#define TA_PACKI 0        // 1: integer-pipe packs in the attention loops -- measured 9-17 % slower (clock-locked)
#endif
#if TA_PACKI
#define PACKP(a, b) pack2i((a), (b))
#else
#define PACKP(a, b) pack2((a), (b))
#endif
__device__ __forceinline__ float2 bf2f(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
// A quad's 4 x 4 words of one accumulator row: lane t4 holds in[i] = the bf16 pair of columns 8i + 2 t4 (i = 0..3, a 32-column group);
// returns columns 8 t4 .. 8 t4 + 7, so each lane stores 16 contiguous bytes (full 32-byte sectors per warp store)
__device__ __forceinline__ uint4 quad_transpose(const uint32_t (&in)[4], int lane) {
  const int t4 = lane & 3;
  uint32_t outw[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int sidx = (t4 - k) & 3, didx = (t4 + k) & 3;
    uint32_t send = in[0];
#pragma unroll
    for (int i = 1; i < 4; ++i) send = sidx == i ? in[i] : send;
    const uint32_t recv = __shfl_sync(0xffffffffu, send, (lane & ~3) | didx);
#pragma unroll
    for (int i = 0; i < 4; ++i) if (didx == i) outw[i] = recv;
  }
  return make_uint4(outw[0], outw[1], outw[2], outw[3]);
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

}  // namespace sm90

// ---------------------------------------------------------------- host: tensor maps
#ifndef SM90_NO_HOST
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
namespace sm90 {
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
// rank-R map; dims / box innermost first; strides in ELEMENTS for dims 1..R-1
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
  static thread_local bool ctx_ok = false;
  if (!ctx_ok) { int dev = 0; C10_CUDA_CHECK(cudaGetDevice(&dev)); C10_CUDA_CHECK(cudaSetDevice(dev)); ctx_ok = true; }
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
}  // namespace sm90
#endif
