// Shared device helpers of the A100 (sm_80) norm kernels: LayerNorm / RMSNorm rows (norm_rows.cu), 3D RoPE and the fused QK-norm + RoPE (kernels/rope/cuda/sm80),
// the RMSNorm + adaLN modulation (kernels/rmsnorm/cuda/sm80).  bf16 <-> fp32 conversions, 16-byte chunks of either dtype, sub-warp group reductions.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace norms {

using bf = __nv_bfloat16;

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
// {a, b} -> one 32-bit word, a in the low half (round to nearest even)
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
  union { __nv_bfloat162 h; uint32_t u; } t;
  t.h = __floats2bfloat162_rn(a, b);
  return t.u;
}

// ------------------------------------------------------------------------------------------------ 16-byte chunks of x / y / dy / dx
// CE elements of T per 16-byte chunk: 8 for bf16, 4 for fp32.  Chunk loads / stores are plain loads and stores (the `.cs` evict-first hints cost 3 % on a streaming kernel in the sandbox).
// `load_raw` keeps the chunk packed (4 registers for 8 bf16: half the live state of the floats), `unpack` widens it where it is used.
template <typename T> struct Chunk;
template <> struct Chunk<bf> {
  static constexpr int CE = 8;
  __device__ __forceinline__ static uint4 load_raw(const bf* p) { return *reinterpret_cast<const uint4*>(p); }
  __device__ __forceinline__ static void unpack(const uint4& u, float (&f)[8]) {
    f[0] = bf_lo(u.x); f[1] = bf_hi(u.x); f[2] = bf_lo(u.y); f[3] = bf_hi(u.y);
    f[4] = bf_lo(u.z); f[5] = bf_hi(u.z); f[6] = bf_lo(u.w); f[7] = bf_hi(u.w);
  }
  __device__ __forceinline__ static void load(const bf* p, float (&f)[8]) { unpack(load_raw(p), f); }
  __device__ __forceinline__ static void store(bf* p, const float (&f)[8]) {
    uint4 u;
    u.x = pack_bf2(f[0], f[1]); u.y = pack_bf2(f[2], f[3]); u.z = pack_bf2(f[4], f[5]); u.w = pack_bf2(f[6], f[7]);
    *reinterpret_cast<uint4*>(p) = u;
  }
};
template <> struct Chunk<float> {
  static constexpr int CE = 4;
  __device__ __forceinline__ static uint4 load_raw(const float* p) { return *reinterpret_cast<const uint4*>(p); }
  __device__ __forceinline__ static void unpack(const uint4& u, float (&f)[4]) {
    f[0] = __uint_as_float(u.x); f[1] = __uint_as_float(u.y); f[2] = __uint_as_float(u.z); f[3] = __uint_as_float(u.w);
  }
  __device__ __forceinline__ static void load(const float* p, float (&f)[4]) { unpack(load_raw(p), f); }
  __device__ __forceinline__ static void store(float* p, const float (&f)[4]) {
    *reinterpret_cast<float4*>(p) = make_float4(f[0], f[1], f[2], f[3]);
  }
};

// scalar element <-> float
__device__ __forceinline__ float to_f(bf v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(float v) { return v; }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ bf from_f<bf>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }

// CE consecutive parameters (norm weight / bias), fp32 or bf16 in memory, as floats (16-byte aligned base).
template <int CE>
__device__ __forceinline__ void load_param(const void* p, bool is_bf16, float (&f)[CE]) {
  if (is_bf16) {
    if constexpr (CE == 8) {
      const uint4 u = *reinterpret_cast<const uint4*>(p);
      f[0] = bf_lo(u.x); f[1] = bf_hi(u.x); f[2] = bf_lo(u.y); f[3] = bf_hi(u.y);
      f[4] = bf_lo(u.z); f[5] = bf_hi(u.z); f[6] = bf_lo(u.w); f[7] = bf_hi(u.w);
    } else {
      const uint2 u = *reinterpret_cast<const uint2*>(p);
      f[0] = bf_lo(u.x); f[1] = bf_hi(u.x); f[2] = bf_lo(u.y); f[3] = bf_hi(u.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < CE / 4; ++k) {
      const float4 u = *(reinterpret_cast<const float4*>(p) + k);
      f[4 * k] = u.x; f[4 * k + 1] = u.y; f[4 * k + 2] = u.z; f[4 * k + 3] = u.w;
    }
  }
}

// one parameter element as float
__device__ __forceinline__ float param_at(const void* p, bool is_bf16, int i) {
  return is_bf16 ? __bfloat162float(reinterpret_cast<const bf*>(p)[i]) : reinterpret_cast<const float*>(p)[i];
}

// sum over the G lanes of an aligned lane group (G a power of two <= 32); every lane of the group gets the sum
template <int G> __device__ __forceinline__ float group_sum(float v) {
#pragma unroll
  for (int o = G >> 1; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

}  // namespace norms

// ------------------------------------------------------------------------------------------------ cp.async (sm_80): 16-byte global -> shared copies, bypassing L1
namespace norms {
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gsrc) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gsrc));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int NPENDING> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(NPENDING)); }
}  // namespace norms

// a hint to bring the line at `p` into L1 (no register is held): the weight / bias chunks of a wide row are read after the row's statistics, and without the hint every chunk's first
// touch is a serial L2 round trip
namespace norms {
__device__ __forceinline__ void prefetch_l1(const void* p) { asm volatile("prefetch.global.L1 [%0];\n" ::"l"(p)); }
}  // namespace norms
