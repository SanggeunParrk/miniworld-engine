// glue.cu — the small passes around the bf16 attention core's backward (CUDA; they were Triton):
//
//   bias_transpose_bf16   bias [H, L(query), L(key)] -> [H, L(key), L(query)] (attn_dkv reads its key rows contiguously)
//   prep_do_bf16          dO [M, 768] fp32 -> bf16, and D = rowsum_head(dO O) as [A, 16, L] fp32 (O fp32)
// and the module's pair-fused core (``fused_pair.py``), any head layout (H heads x DH, DH a multiple of 4; rows W = H DH wide):
//   prep_do_f32 / _b16    dO [M, W] (fp32 / bf16) -> D = rowsum_head(dO O) as [A, H, L] fp32 and (dob != null) dO as bf16 rows
//   finish_f32 / _bf16    dQ = the NP key-chunk partials summed in chunk order (deterministic), and (bf16) dK / dV rounded to bf16
//   round3_tf32           q, k, v rounded to the TF32 grid (to nearest, as cuBLAS takes fp32 operands; the tensor core alone would
//                         truncate them) into [A' L, W] copies whose pad sample rows are zero
//   prep_do_f32r          prep_do_f32 that writes dO rounded to the TF32 grid instead of the bf16 copy
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cstdint>

// 32 x 32 tiles through shared memory (padded against bank conflicts); grid (L / 32, L / 32, H), block (32, 8)
extern "C" __global__ void __launch_bounds__(256) bias_transpose_bf16(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                                                                      int L) {
  __shared__ __nv_bfloat16 t[32][34];
  const size_t base = (size_t)blockIdx.z * L * L;
  const int i0 = blockIdx.y * 32, j0 = blockIdx.x * 32;
#pragma unroll
  for (int k = 0; k < 32; k += 8) t[threadIdx.y + k][threadIdx.x] = src[base + (size_t)(i0 + threadIdx.y + k) * L + j0 + threadIdx.x];
  __syncthreads();
#pragma unroll
  for (int k = 0; k < 32; k += 8) dst[base + (size_t)(j0 + threadIdx.y + k) * L + i0 + threadIdx.x] = t[threadIdx.x][threadIdx.y + k];
}

// one thread per (row, head): 12 float4 of dO and O; consecutive threads take consecutive heads of a row
extern "C" __global__ void __launch_bounds__(256) prep_do_bf16(const float* __restrict__ dO, const float* __restrict__ O,
                                                               __nv_bfloat16* __restrict__ dob, float* __restrict__ dd, int M, int L) {
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long)M * 16) return;
  const long r = t >> 4;
  const int h = (int)(t & 15);
  const float4* g = reinterpret_cast<const float4*>(dO + r * 768 + h * 48);
  const float4* o = reinterpret_cast<const float4*>(O + r * 768 + h * 48);
  __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(dob + r * 768 + h * 48);
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 12; ++k) {
    const float4 x = g[k], y = o[k];
    s += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
    b[2 * k] = __floats2bfloat162_rn(x.x, x.y);
    b[2 * k + 1] = __floats2bfloat162_rn(x.z, x.w);
  }
  dd[((r / L) * 16 + h) * L + r % L] = s;
}

// one thread per (row, head); M rows of W = H DH columns (row stride W), D = rowsum_head(dO O) into dd[(a H + h) L + l]
template <typename T> __device__ __forceinline__ float4 ld4t(const T* p);
template <> __device__ __forceinline__ float4 ld4t<float>(const float* p) { return *reinterpret_cast<const float4*>(p); }
template <> __device__ __forceinline__ float4 ld4t<__nv_bfloat16>(const __nv_bfloat16* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float tf32r(float x) {
  uint32_t u;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
  return __uint_as_float(u);
}
template <typename T, bool R32 = false>
__device__ __forceinline__ void prep_do_any(const T* __restrict__ dO, const float* __restrict__ O, __nv_bfloat16* __restrict__ dob,
                                            float* __restrict__ dd, int M, int L, int H, int DH, float* __restrict__ dor = nullptr) {
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long)M * H) return;
  const long r = t / H;
  const int h = (int)(t - r * H), W = H * DH;
  const T* g = dO + r * W + h * DH;
  const float* o = O + r * W + h * DH;
  float s = 0.f;
  for (int k = 0; k < DH; k += 4) {
    const float4 x = ld4t<T>(g + k), y = *reinterpret_cast<const float4*>(o + k);
    s += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
    if (R32)
      *reinterpret_cast<float4*>(dor + r * W + h * DH + k) = make_float4(tf32r(x.x), tf32r(x.y), tf32r(x.z), tf32r(x.w));
    else if (dob) {
      __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(dob + r * W + h * DH + k);
      b[0] = __floats2bfloat162_rn(x.x, x.y);
      b[1] = __floats2bfloat162_rn(x.z, x.w);
    }
  }
  dd[((r / L) * H + h) * L + r % L] = s;
}
extern "C" __global__ void __launch_bounds__(256) prep_do_f32(const float* __restrict__ dO, const float* __restrict__ O,
                                                              __nv_bfloat16* __restrict__ dob, float* __restrict__ dd, int M, int L, int H, int DH) {
  prep_do_any<float>(dO, O, dob, dd, M, L, H, DH);
}
extern "C" __global__ void __launch_bounds__(256) prep_do_b16(const __nv_bfloat16* __restrict__ dO, const float* __restrict__ O,
                                                              __nv_bfloat16* __restrict__ dob, float* __restrict__ dd, int M, int L, int H, int DH) {
  prep_do_any<__nv_bfloat16>(dO, O, dob, dd, M, L, H, DH);
}

// dq[i] = sum_{c < np} P[c stride + i] in c order (n4 float4 per output); grid-stride
extern "C" __global__ void __launch_bounds__(256) finish_f32(const float* __restrict__ P, int np, int stride, float* __restrict__ dq, int n4) {
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x) {
    float4 a = reinterpret_cast<const float4*>(P)[i];
    for (int c = 1; c < np; ++c) {
      const float4 b = reinterpret_cast<const float4*>(P + (long)c * stride)[i];
      a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    }
    reinterpret_cast<float4*>(dq)[i] = a;
  }
}
__device__ __forceinline__ void st4b(__nv_bfloat16* p, float4 a) {
  __nv_bfloat162 u = __floats2bfloat162_rn(a.x, a.y), v = __floats2bfloat162_rn(a.z, a.w);
  uint2 w;
  w.x = *reinterpret_cast<uint32_t*>(&u);
  w.y = *reinterpret_cast<uint32_t*>(&v);
  *reinterpret_cast<uint2*>(p) = w;
}
// blockIdx.y: 0 dq (partials summed in chunk order), 1 dk, 2 dv (rounded to bf16)
extern "C" __global__ void __launch_bounds__(256) finish_bf16(const float* __restrict__ P, int np, int stride, const float* __restrict__ DK,
                                                              const float* __restrict__ DV, __nv_bfloat16* __restrict__ dq,
                                                              __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dv, int n4) {
  const int which = blockIdx.y;
  const float* src = which == 0 ? P : which == 1 ? DK : DV;
  __nv_bfloat16* dst = which == 0 ? dq : which == 1 ? dk : dv;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x) {
    float4 a = reinterpret_cast<const float4*>(src)[i];
    if (which == 0)
      for (int c = 1; c < np; ++c) {
        const float4 b = reinterpret_cast<const float4*>(P + (long)c * stride)[i];
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
      }
    st4b(dst + 4 * i, a);
  }
}
extern "C" __global__ void __launch_bounds__(256) prep_do_f32r(const float* __restrict__ dO, const float* __restrict__ O,
                                                               float* __restrict__ dor, float* __restrict__ dd, int M, int L, int H, int DH) {
  prep_do_any<float, true>(dO, O, nullptr, dd, M, L, H, DH, dor);
}

// blockIdx.y picks the tensor (q, k, v); n4 float4 of real rows, n4p of the padded output (the rest zero)
extern "C" __global__ void __launch_bounds__(256) round3_tf32(const float* __restrict__ a, const float* __restrict__ b,
                                                              const float* __restrict__ c, float* __restrict__ ao, float* __restrict__ bo,
                                                              float* __restrict__ co, int n4, int n4p) {
  const int which = blockIdx.y;
  const float4* src = reinterpret_cast<const float4*>(which == 0 ? a : which == 1 ? b : c);
  float4* dst = reinterpret_cast<float4*>(which == 0 ? ao : which == 1 ? bo : co);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4p; i += gridDim.x * blockDim.x) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (i < n4) {
      v = src[i];
      v = make_float4(tf32r(v.x), tf32r(v.y), tf32r(v.z), tf32r(v.w));
    }
    dst[i] = v;
  }
}
