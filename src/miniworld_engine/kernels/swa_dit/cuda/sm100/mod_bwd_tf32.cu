// mod_bwd_tf32.cu — the element-wise parts of the hoisted modulation's fp32 backward on sm_100a (mod = silu(c) Wmod^T, c [R, C]):
//   swa_mod_silu_tf32_sm100   a = silu(c), rounded to TF32: the operand of dWmod = g^T a (cuBLAS TF32)
//   swa_mod_dsilu_tf32_sm100  dc = acc silu'(c),  silu'(c) = s (1 + c (1 - s)),  s = sigmoid(c),  acc = g Wmod (cuBLAS TF32)
// One warp per row (lane = four channels).
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>

constexpr int C = 128, NWARP = 8;

__device__ __forceinline__ uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r & 0xffffe000u; }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
// acc = the sum of np partials ps floats apart, in partial order (round 3 part 12: g Wmod as a K-split batched GEMM; np = 1: acc itself)
__device__ __forceinline__ float4 acc_sum(const float* __restrict__ p, int np, int ps) {
  float4 g = *reinterpret_cast<const float4*>(p);
  for (int k = 1; k < np; ++k) {
    const float4 v = *reinterpret_cast<const float4*>(p + (size_t)k * ps);
    g.x += v.x; g.y += v.y; g.z += v.z; g.w += v.w;
  }
  return g;
}

extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_mod_silu_tf32_sm100(const float* __restrict__ CIN, float* __restrict__ A, int R) {
  const int row = (int)blockIdx.x * NWARP + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= R) return;
  const float4 c = *reinterpret_cast<const float4*>(CIN + (size_t)row * C + 4 * lane);
  *reinterpret_cast<uint4*>(A + (size_t)row * C + 4 * lane) =
      make_uint4(tf32r(c.x * sigm(c.x)), tf32r(c.y * sigm(c.y)), tf32r(c.z * sigm(c.z)), tf32r(c.w * sigm(c.w)));
}

extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_mod_dsilu_tf32_sm100(const float* __restrict__ ACC, const float* __restrict__ CIN, float* __restrict__ DC, int R, int np, int ps) {
  const int row = (int)blockIdx.x * NWARP + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= R) return;
  const size_t o = (size_t)row * C + 4 * lane;
  const float4 c = *reinterpret_cast<const float4*>(CIN + o), g = acc_sum(ACC + o, np, ps);
  const float sx = sigm(c.x), sy = sigm(c.y), sz = sigm(c.z), sw = sigm(c.w);
  *reinterpret_cast<float4*>(DC + o) = make_float4(g.x * sx * (1.f + c.x * (1.f - sx)), g.y * sy * (1.f + c.y * (1.f - sy)),
                                                  g.z * sz * (1.f + c.z * (1.f - sz)), g.w * sw * (1.f + c.w * (1.f - sw)));
}

// c bf16 (round 3 part 6, master training: c is read in place, dc leaves as bf16): the same equations on c's bf16 values
__device__ __forceinline__ float4 ldb4(const uint16_t* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
__device__ __forceinline__ uint32_t bf2(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }

extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_mod_silu_tf32_c16_sm100(const uint16_t* __restrict__ CIN, float* __restrict__ A, int R) {
  const int row = (int)blockIdx.x * NWARP + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= R) return;
  const float4 c = ldb4(CIN + (size_t)row * C + 4 * lane);
  *reinterpret_cast<uint4*>(A + (size_t)row * C + 4 * lane) =
      make_uint4(tf32r(c.x * sigm(c.x)), tf32r(c.y * sigm(c.y)), tf32r(c.z * sigm(c.z)), tf32r(c.w * sigm(c.w)));
}

extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_mod_dsilu_tf32_c16_sm100(const float* __restrict__ ACC, const uint16_t* __restrict__ CIN, uint16_t* __restrict__ DC, int R, int np,
                             int ps) {
  const int row = (int)blockIdx.x * NWARP + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= R) return;
  const size_t o = (size_t)row * C + 4 * lane;
  const float4 c = ldb4(CIN + o), g = acc_sum(ACC + o, np, ps);
  const float sx = sigm(c.x), sy = sigm(c.y), sz = sigm(c.z), sw = sigm(c.w);
  *reinterpret_cast<uint2*>(DC + o) = make_uint2(bf2(g.x * sx * (1.f + c.x * (1.f - sx)), g.y * sy * (1.f + c.y * (1.f - sy))),
                                                 bf2(g.z * sz * (1.f + c.z * (1.f - sz)), g.w * sw * (1.f + c.w * (1.f - sw))));
}

// B200 round 3 part 14: dc of one block of a stack whose blocks share c, summed across the blocks in the kernel (no autograd adds): out =
// (PREV + dc) [+ XADD], fp32 sums in the stack's backward order (the last block first: PREV null), rounded once at the end. c fp32 or bf16
// (c_bf16), out fp32 (the running sum) or bf16 (out_bf16: the stack's c gradient), XADD (null or the first block's x gradient, x_bf16) when
// x and c are one tensor (the InputFeatureEmbedder's single): the sum leaves as that tensor's whole gradient.
extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_mod_dsilu_acc_sm100(const float* __restrict__ ACC, const void* __restrict__ CIN, const float* __restrict__ PREV, void* __restrict__ OUT,
                        const void* __restrict__ XADD, int R, int np, int ps, int c_bf16, int out_bf16, int x_bf16) {
  const int row = (int)blockIdx.x * NWARP + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= R) return;
  const size_t o = (size_t)row * C + 4 * lane;
  const float4 c = c_bf16 ? ldb4(reinterpret_cast<const uint16_t*>(CIN) + o) : *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(CIN) + o);
  const float4 g = acc_sum(ACC + o, np, ps);
  const float sx = sigm(c.x), sy = sigm(c.y), sz = sigm(c.z), sw = sigm(c.w);
  float4 d = make_float4(g.x * sx * (1.f + c.x * (1.f - sx)), g.y * sy * (1.f + c.y * (1.f - sy)), g.z * sz * (1.f + c.z * (1.f - sz)),
                         g.w * sw * (1.f + c.w * (1.f - sw)));
  if (PREV) {
    const float4 p = *reinterpret_cast<const float4*>(PREV + o);
    d = make_float4(p.x + d.x, p.y + d.y, p.z + d.z, p.w + d.w);
  }
  if (XADD) {
    const float4 x = x_bf16 ? ldb4(reinterpret_cast<const uint16_t*>(XADD) + o) : *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(XADD) + o);
    d = make_float4(x.x + d.x, x.y + d.y, x.z + d.z, x.w + d.w);
  }
  if (out_bf16)
    *reinterpret_cast<uint2*>(reinterpret_cast<uint16_t*>(OUT) + o) = make_uint2(bf2(d.x, d.y), bf2(d.z, d.w));
  else
    *reinterpret_cast<float4*>(reinterpret_cast<float*>(OUT) + o) = d;
}
