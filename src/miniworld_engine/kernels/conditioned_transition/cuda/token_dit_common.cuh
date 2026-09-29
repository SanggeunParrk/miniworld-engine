// token_dit_common.cuh -- shared helpers of the token DiT row kernels: sigmoid, warp / block sums, 4-wide fp32 / bf16 vectors.
#pragma once
#include <cuda_bf16.h>

namespace tdr {



__device__ __forceinline__ float sigm(float v) { return 1.f / (1.f + __expf(-v)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <typename T> struct V4;
template <> struct V4<float> {
  static __device__ __forceinline__ float4 load(const float* p) { return *reinterpret_cast<const float4*>(p); }
  static __device__ __forceinline__ void store(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
};
template <> struct V4<__nv_bfloat16> {
  static __device__ __forceinline__ float4 load(const __nv_bfloat16* p) {
    uint2 u = *reinterpret_cast<const uint2*>(p);
    __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&u.x), b = *reinterpret_cast<__nv_bfloat162*>(&u.y);
    float2 fa = __bfloat1622float2(a), fb = __bfloat1622float2(b);
    return make_float4(fa.x, fa.y, fb.x, fb.y);
  }
  static __device__ __forceinline__ void store(__nv_bfloat16* p, float4 v) {
    __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    uint2 u;
    u.x = *reinterpret_cast<unsigned*>(&a);
    u.y = *reinterpret_cast<unsigned*>(&b);
    *reinterpret_cast<uint2*>(p) = u;
  }
};


// Sum over a block of NT threads (NT a multiple of 32); red holds NT / 32 floats. Ends with a barrier so red can be reused.
template <int NT>
__device__ __forceinline__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  if (l == 0) red[w] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; ++i) t += red[i];
  __syncthreads();
  return t;
}
}  // namespace tdr
