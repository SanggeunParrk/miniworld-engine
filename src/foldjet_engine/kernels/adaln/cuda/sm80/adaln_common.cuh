// adaln_common.cuh -- helpers shared by the A100 (sm_80) AdaLN kernels: sigmoid, row sums, 4-wide fp32 / bf16 vectors, bf16 rounding.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#define ADL_DEVI __device__ __forceinline__

namespace adl {

using bf = __nv_bfloat16;

ADL_DEVI float sigm(float v) { return 1.f / (1.f + __expf(-v)); }   // exact at the tails: exp(-v) = inf -> 0
ADL_DEVI float sigmoidf(float v) { return __fdividef(1.f, 1.f + __expf(-v)); }   // ex2 + rcp: the IEEE division costs 10x more (the tensor-core kernels, where the pass is not the bottleneck either way)

ADL_DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// 4 consecutive elements of a row as fp32: a 16-B load of fp32, an 8-B load of bf16 (the rows are 8-B / 16-B aligned: the host checks it)
template <typename T> struct V4;
template <> struct V4<float> {
  static ADL_DEVI float4 load(const float* p) { return *reinterpret_cast<const float4*>(p); }
  static ADL_DEVI void store(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
};
template <> struct V4<bf> {
  static ADL_DEVI float4 load(const bf* p) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
  }
  static ADL_DEVI void store(bf* p, float4 v) {
    const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    uint2 u;
    u.x = *reinterpret_cast<const unsigned*>(&a);
    u.y = *reinterpret_cast<const unsigned*>(&b);
    *reinterpret_cast<uint2*>(p) = u;
  }
};

ADL_DEVI float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
ADL_DEVI float4 sub4(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
ADL_DEVI float4 mul4(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
ADL_DEVI float4 scale4(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
ADL_DEVI float4 sig4(float4 a) { return make_float4(sigm(a.x), sigm(a.y), sigm(a.z), sigm(a.w)); }
ADL_DEVI float sum4(float4 a) { return a.x + a.y + a.z + a.w; }
ADL_DEVI float4 zero4() { return make_float4(0.f, 0.f, 0.f, 0.f); }

// the value a store of T keeps (bf16: rounded to nearest even; fp32: unchanged)
template <typename T> ADL_DEVI float4 rnd4(float4 a) { return a; }
template <> ADL_DEVI float4 rnd4<bf>(float4 a) {
  return make_float4(__bfloat162float(__float2bfloat16_rn(a.x)), __bfloat162float(__float2bfloat16_rn(a.y)),
                     __bfloat162float(__float2bfloat16_rn(a.z)), __bfloat162float(__float2bfloat16_rn(a.w)));
}

// Sum over the NT threads of one row of a block (NT a multiple of 32; threadIdx.x is the position in the row, threadIdx.y the row);
// red is this row's NT / 32 floats. NT == 32 is one warp: no barrier. Every thread of the block must call it (barriers).
template <int NT>
ADL_DEVI float row_sum(float v, float* red) {
  v = warp_sum(v);
  if constexpr (NT == 32) {
    return v;
  } else {
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < NT / 32; ++i) t += red[i];
    __syncthreads();
    return t;
  }
}

}  // namespace adl
