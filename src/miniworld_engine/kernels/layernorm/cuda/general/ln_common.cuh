// General LayerNorm over the last dimension (any width, any row count): shared pieces of the forward and backward kernels.
//
// A row of N elements is held by TPR threads (TPR in 4..1024, TPR <= 32 is a sub-warp group of a warp); each thread keeps CHUNKS
// vectors of VEC elements of its row in registers -- column (c * TPR + lane) * VEC for chunk c, so a warp's loads are contiguous.
// VEC = 16 bytes / sizeof(T) when N is a multiple of it (and the rows are 16-byte aligned), otherwise 1 (scalar, any N). A CTA is
// max(TPR, 256) threads = RPC = CTA / TPR rows at a time, and grid-strides over the row groups. Statistics are fp32.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace mwln {

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }

template <typename T, int VEC> struct alignas(sizeof(T) * VEC) Pack { T v[VEC]; };

template <int TPR> constexpr int cta_threads() { return TPR >= 256 ? TPR : 256; }

// Multiprocessor count of the current device, looked up once per device (cudaDeviceGetAttribute is not free per launch).
inline int num_sms() {
  static int cached[64] = {0};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= 64) dev = 0;
  if (!cached[dev]) cudaDeviceGetAttribute(&cached[dev], cudaDevAttrMultiProcessorCount, dev);
  return cached[dev];
}

// Sum over the TPR threads of a row. `red` holds RPC * (TPR / 32) floats when TPR > 32; every thread of the CTA must call it.
template <int TPR>
__device__ __forceinline__ float row_sum(float s, float* red, int r_in, int lane) {
#pragma unroll
  for (int o = (TPR < 32 ? TPR : 32) / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if constexpr (TPR > 32) {
    constexpr int W = TPR / 32;
    if ((lane & 31) == 0) red[r_in * W + (lane >> 5)] = s;
    __syncthreads();
    s = 0.f;
#pragma unroll
    for (int i = 0; i < W; ++i) s += red[r_in * W + i];
    __syncthreads();
  }
  return s;
}

template <typename T, int VEC>
__device__ __forceinline__ void load_vec(const T* p, float* out) {
  if constexpr (VEC == 1) {
    out[0] = to_f<T>(*p);
  } else {
    Pack<T, VEC> pk = *reinterpret_cast<const Pack<T, VEC>*>(p);
#pragma unroll
    for (int i = 0; i < VEC; ++i) out[i] = to_f<T>(pk.v[i]);
  }
}

template <typename T, int VEC>
__device__ __forceinline__ void store_vec(T* p, const float* in) {
  if constexpr (VEC == 1) {
    *p = from_f<T>(in[0]);
  } else {
    Pack<T, VEC> pk;
#pragma unroll
    for (int i = 0; i < VEC; ++i) pk.v[i] = from_f<T>(in[i]);
    *reinterpret_cast<Pack<T, VEC>*>(p) = pk;
  }
}

// A thread's affine values for its columns (fixed across the rows it visits): 1 / 0 where `p` is null or past N. VEC fp32 values are
// read as 16-byte vectors (the host passes 16-byte-aligned weights).
template <int TPR, int CHUNKS, int VEC>
__device__ __forceinline__ void load_affine(const float* __restrict__ p, float fill, int lane, int N, float* out) {
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int col = (c * TPR + lane) * VEC;
    if (p && col < N) {
      if constexpr (VEC % 4 == 0) {
#pragma unroll
        for (int i = 0; i < VEC; i += 4) {
          const float4 t = *reinterpret_cast<const float4*>(p + col + i);
          out[c * VEC + i] = t.x;
          out[c * VEC + i + 1] = t.y;
          out[c * VEC + i + 2] = t.z;
          out[c * VEC + i + 3] = t.w;
        }
      } else {
#pragma unroll
        for (int i = 0; i < VEC; ++i) out[c * VEC + i] = p[col + i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < VEC; ++i) out[c * VEC + i] = fill;
    }
  }
}

}  // namespace mwln

// Instantiation lists, shared by the launchers. Vector path: TPR 4..1024 x CHUNKS {1, 2, 4, 8}; scalar path: TPR 32..1024.
#define MWLN_FOR_CHUNKS(M, ...) M(1, __VA_ARGS__) M(2, __VA_ARGS__) M(4, __VA_ARGS__) M(8, __VA_ARGS__)
