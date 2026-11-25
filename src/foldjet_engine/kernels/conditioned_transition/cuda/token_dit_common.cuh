// token_dit_common.cuh -- shared helpers of the token DiT row kernels: sigmoid, warp / block sums, 4-wide fp32 / bf16 vectors.
#pragma once
#include <cuda_bf16.h>
#include "../../adaln/cuda/row_math.cuh"   // the one definition of warp_sum / sigmoid shared with the AdaLN kernels

namespace tdr {

__device__ __forceinline__ float sigm(float v) { return rowmath::sigmoid(v); }
__device__ __forceinline__ float warp_sum(float v) { return rowmath::warp_sum(v); }

// 4-wide fp32 lane arithmetic
__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 mul4(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ __forceinline__ float4 zero4() { return make_float4(0.f, 0.f, 0.f, 0.f); }
__device__ __forceinline__ float sum4(float4 a) { return a.x + a.y + a.z + a.w; }
__device__ __forceinline__ float4 sig4(float4 a) { return make_float4(sigm(a.x), sigm(a.y), sigm(a.z), sigm(a.w)); }
__device__ __forceinline__ float4 dsig4(float4 v, float4 s) {   // v * s * (1 - s)
  return make_float4(v.x * s.x * (1.f - s.x), v.y * s.y * (1.f - s.y), v.z * s.z * (1.f - s.z), v.w * s.w * (1.f - s.w));
}
template <typename AT> __device__ __forceinline__ float4 rnd(float4 a) { return a; }   // the value a store of AT keeps
template <> __device__ __forceinline__ float4 rnd<__nv_bfloat16>(float4 a) {           // (sums use what was stored)
  return make_float4(__bfloat162float(__float2bfloat16_rn(a.x)), __bfloat162float(__float2bfloat16_rn(a.y)),
                     __bfloat162float(__float2bfloat16_rn(a.z)), __bfloat162float(__float2bfloat16_rn(a.w)));
}

// dst [N, M] = src [M, N]^T (either operand dtype): 32 x 32 tiles through shared memory, both sides coalesced. Launch: grid (ceil(N / 32), ceil(M / 32)), 256 threads.
// GUARD = false: M and N are multiples of 32 (the token DiT / pair kernels). GUARD = true: any M, N (bounds-checked; the DiT variant training rows).
template <typename T, bool GUARD = false>
__global__ void __launch_bounds__(256) transpose_k(const T* __restrict__ S_, T* __restrict__ Dst, long M, long N) {
  __shared__ T tile[32][33];
  const long r0 = (long)blockIdx.y * 32, c0 = (long)blockIdx.x * 32;
  const int tx = threadIdx.x % 32, ty = threadIdx.x / 32;
#pragma unroll
  for (int i = 0; i < 32; i += 8) {
    if (GUARD && !((r0 + ty + i) < M && (c0 + tx) < N)) continue;
    tile[ty + i][tx] = S_[(r0 + ty + i) * N + c0 + tx];
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 32; i += 8) {
    if (GUARD && !((c0 + ty + i) < N && (r0 + tx) < M)) continue;
    Dst[(c0 + ty + i) * M + r0 + tx] = tile[tx][ty + i];
  }
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
