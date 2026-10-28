// The shared-memory / global PTX helpers (cvta, cp.async 16 B, ld/st.shared vectors, predicated st.global) the cute-based
// sm_90a Transition kernels share: transition_b2b_kernel.cu and transition_expand_gate_kernel.cu. One definition each; a
// file brings them into its own detail namespace with `using transition_smem::x;`.
#pragma once
#include <cstdint>

namespace transition_smem {

__device__ __forceinline__ uint32_t smem_addr_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 addr64; cvta.to.shared.u64 addr64, %1; cvt.u32.u64 %0, addr64; }\n"
        : "=r"(addr)
        : "l"(ptr)
    );
    return addr;
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src, bool pred) {
    const int bytes = pred ? 16 : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(smem_addr_u32(dst)), "l"(src), "r"(bytes)
    );
}

__device__ __forceinline__ uint4 smem_load_u128(const void* ptr) {
    uint4 value;
    asm volatile(
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "r"(smem_addr_u32(ptr))
        : "memory"
    );
    return value;
}

__device__ __forceinline__ void smem_store_u128(void* ptr, uint4 value) {
    asm volatile(
        "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
        :
        : "r"(smem_addr_u32(ptr)), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
        : "memory"
    );
}

__device__ __forceinline__ void smem_store_u32(void* ptr, uint32_t value) {
    asm volatile(
        "st.shared.u32 [%0], %1;\n"
        :
        : "r"(smem_addr_u32(ptr)), "r"(value)
        : "memory"
    );
}

__device__ __forceinline__ void global_store_u128(void* ptr, uint4 value, bool pred) {
    asm volatile(
        "{ .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  @p st.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
        "}\n"
        :
        : "l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w),
          "r"(static_cast<int>(pred))
        : "memory"
    );
}

}  // namespace transition_smem
