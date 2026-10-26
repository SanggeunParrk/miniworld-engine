// Device helpers of the Chai-1 OuterProductMean kernels (sm_90a): mbarrier, TMA, wgmma m64n128k16 (both operands from shared
// memory, 128-byte swizzle).  Same conventions as integrations/csrc/opm_epilogue.cu (namespaces wg / opm_pro_tma): a K-major tile
// is [rows][64 bf16] = 128 B per row with the 16-byte chunk index XORed by row % 8.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace chai {
__device__ __forceinline__ uint32_t sa(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mb_init(uint64_t* p, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(p)), "r"(n) : "memory"); }
__device__ __forceinline__ void mb_expect(uint64_t* p, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(p)), "r"(n) : "memory"); }
__device__ __forceinline__ void mb_arrive(uint64_t* p) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(p)) : "memory"); }
__device__ __forceinline__ void mb_wait(uint64_t* p, int phase) {
  asm volatile("{.reg .pred p; W_%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W_%=;}" :: "r"(sa(p)), "r"(phase) : "memory");
}
__device__ __forceinline__ void fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void tma_ld2(const void* map, void* dst, uint64_t* b, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
               :: "r"(sa(dst)), "l"(map), "r"(sa(b)), "r"(x), "r"(y) : "memory");
}
__device__ __forceinline__ void tma_ld4(const void* map, void* dst, uint64_t* b, int x, int y, int z, int w) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4,%5,%6}],[%2];"
               :: "r"(sa(dst)), "l"(map), "r"(sa(b)), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
__device__ __forceinline__ void tma_st2(const void* map, const void* src, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0,{%2,%3}],[%1];" :: "l"(map), "r"(sa(src)), "r"(x), "r"(y) : "memory");
}
__device__ __forceinline__ void tma_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void tma_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void tma_wait_all() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }

// K-major operand descriptor (128 B swizzle, rows of 64 bf16, 1024 B per 8-row group)
__device__ __forceinline__ uint64_t wg_desc(const void* p) {
  uint64_t d = uint64_t((sa(p) >> 4) & 0x3FFFu);
  d |= uint64_t(1) << 16;     // leading offset 1 (swizzled)
  d |= uint64_t(64) << 32;    // stride offset: 1024 B between 8-row groups
  d |= uint64_t(1) << 62;     // SWIZZLE_128B
  return d;
}
// no-swizzle K-major descriptor (core matrices of 8 rows x 16 B; lbo = bytes between core matrices along K, sbo = along M)
__device__ __forceinline__ uint64_t wg_desc_ns(const void* p, uint32_t lbo, uint32_t sbo) {
  return uint64_t((sa(p) >> 4) & 0x3FFFu) | (uint64_t((lbo >> 4) & 0x3FFFu) << 16) | (uint64_t((sbo >> 4) & 0x3FFFu) << 32);
}
__device__ __forceinline__ void tma_st4(const void* map, const void* src, int x, int y, int z, int w) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0,{%2,%3,%4,%5}],[%1];" :: "l"(map), "r"(sa(src)), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
__device__ __forceinline__ void tma_st3(const void* map, const void* src, int x, int y, int z) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0,{%2,%3,%4}],[%1];" :: "l"(map), "r"(sa(src)), "r"(x), "r"(y), "r"(z) : "memory");
}
// element offset of the 16-byte chunk kc (0..7) of row `mn` in a K-major tile of 64-element blocks (rows rows per block)
__device__ __forceinline__ int off_n(int rows, int blk, int mn, int kc) { return ((blk * rows + mn) * 8 + (kc ^ (mn & 7))) * 8; }

// D[64 x 128] (+)= A[64 x 16] B[128 x 16]^T, A and B K-major from shared memory
__device__ __forceinline__ void mma_m64n128k16(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %64, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31,\n"
      " %32, %33, %34, %35, %36, %37, %38, %39,\n"
      " %40, %41, %42, %43, %44, %45, %46, %47,\n"
      " %48, %49, %50, %51, %52, %53, %54, %55,\n"
      " %56, %57, %58, %59, %60, %61, %62, %63},\n"
      " %65, %66, p, 1, 1, 0, 0;\n}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15]),
        "+f"(d[16]),
        "+f"(d[17]),
        "+f"(d[18]),
        "+f"(d[19]),
        "+f"(d[20]),
        "+f"(d[21]),
        "+f"(d[22]),
        "+f"(d[23]),
        "+f"(d[24]),
        "+f"(d[25]),
        "+f"(d[26]),
        "+f"(d[27]),
        "+f"(d[28]),
        "+f"(d[29]),
        "+f"(d[30]),
        "+f"(d[31]),
        "+f"(d[32]),
        "+f"(d[33]),
        "+f"(d[34]),
        "+f"(d[35]),
        "+f"(d[36]),
        "+f"(d[37]),
        "+f"(d[38]),
        "+f"(d[39]),
        "+f"(d[40]),
        "+f"(d[41]),
        "+f"(d[42]),
        "+f"(d[43]),
        "+f"(d[44]),
        "+f"(d[45]),
        "+f"(d[46]),
        "+f"(d[47]),
        "+f"(d[48]),
        "+f"(d[49]),
        "+f"(d[50]),
        "+f"(d[51]),
        "+f"(d[52]),
        "+f"(d[53]),
        "+f"(d[54]),
        "+f"(d[55]),
        "+f"(d[56]),
        "+f"(d[57]),
        "+f"(d[58]),
        "+f"(d[59]),
        "+f"(d[60]),
        "+f"(d[61]),
        "+f"(d[62]),
        "+f"(d[63])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}
}  // namespace chai
