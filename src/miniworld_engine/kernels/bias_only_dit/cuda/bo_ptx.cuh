// bo_ptx.cuh -- the cp.async helpers of the bias-only DiT kernels (bo_psb*.cu, bo_hoist_*.cu, the pair-bias passes of
// bias_only_dit_train_rows.cu / bias_only_dit_f32_rows.cu): one definition each.
#pragma once
#include <cstdint>

namespace bop {
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
// 16 bytes global -> shared (L2 only), by shared-window address or by generic pointer
__device__ __forceinline__ void cp_async16(uint32_t s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void cp_async16(void* s, const void* g) { cp_async16(smem_u32(s), g); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
}  // namespace bop
