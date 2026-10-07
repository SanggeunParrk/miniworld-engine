// mod_bwd_tf32.cu — the element-wise parts of the hoisted modulation's fp32 backward on sm_100a (mod = silu(c) Wmod^T, c [R, C]):
//   swa_mod_silu_tf32_sm100   a = silu(c), rounded to TF32: the operand of dWmod = g^T a (cuBLAS TF32)
//   swa_mod_dsilu_tf32_sm100  dc = acc silu'(c),  silu'(c) = s (1 + c (1 - s)),  s = sigmoid(c),  acc = g Wmod (cuBLAS TF32)
// One warp per row (lane = four channels).
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>

constexpr int C = 128, NWARP = 8;

__device__ __forceinline__ uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r & 0xffffe000u; }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_mod_silu_tf32_sm100(const float* __restrict__ CIN, float* __restrict__ A, int R) {
  const int row = (int)blockIdx.x * NWARP + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= R) return;
  const float4 c = *reinterpret_cast<const float4*>(CIN + (size_t)row * C + 4 * lane);
  *reinterpret_cast<uint4*>(A + (size_t)row * C + 4 * lane) =
      make_uint4(tf32r(c.x * sigm(c.x)), tf32r(c.y * sigm(c.y)), tf32r(c.z * sigm(c.z)), tf32r(c.w * sigm(c.w)));
}

extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_mod_dsilu_tf32_sm100(const float* __restrict__ ACC, const float* __restrict__ CIN, float* __restrict__ DC, int R) {
  const int row = (int)blockIdx.x * NWARP + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= R) return;
  const size_t o = (size_t)row * C + 4 * lane;
  const float4 c = *reinterpret_cast<const float4*>(CIN + o), g = *reinterpret_cast<const float4*>(ACC + o);
  const float sx = sigm(c.x), sy = sigm(c.y), sz = sigm(c.z), sw = sigm(c.w);
  *reinterpret_cast<float4*>(DC + o) = make_float4(g.x * sx * (1.f + c.x * (1.f - sx)), g.y * sy * (1.f + c.y * (1.f - sy)),
                                                  g.z * sz * (1.f + c.z * (1.f - sz)), g.w * sw * (1.f + c.w * (1.f - sw)));
}
