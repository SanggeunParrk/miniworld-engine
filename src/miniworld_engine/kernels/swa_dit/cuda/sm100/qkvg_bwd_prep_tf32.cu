// qkvg_bwd_prep_tf32.cu — the head-wise first half of the fp32 SWA atom block's qkvg backward on sm_100a (the Triton _swa_qkvg_bwd_fp32_kernel's
// dP): from the attention backward's dQ / dK / dV (head-major [N, H, S, D], fp32 or bf16) and dG, back through RoPE and the per-head
// RMSNorm of the saved raw projections:
//   dpq = head_rms_bwd(pq, unrope(dQ)),  dpk = head_rms_bwd(pk, unrope(dK)),  dpv = dV,  dg = dG
//   head_rms_bwd(p, du) = r (du - yh mean_D(du yh)),  yh = p r,  r = 1 / sqrt(mean_D(p^2) + eps);  unrope = the transpose of the
//   RoPE rotation: dx1 = dy1 c + dy2 s, dx2 = dy2 c - dy1 s (halves d < 16 | d >= 16 of a head, c / s = cos / sin of the atom)
// -> dP [M, 512] = [dpq | dpk | dpv | dg], rounded to TF32 (cvt.rna): the A operand of dx = dP W and of the weight gradients dP^T x.
// One warp per row (lane = four channels, a head = eight lanes: the RoPE partner of channel d is d + 16, four lanes away; the head
// reductions by xor shuffles over the eight lanes), eight rows per block.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>

constexpr int C = 128, H = 4, D = 32, NWARP = 8;

__device__ __forceinline__ uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r & 0xffffe000u; }
__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ float4 ld4(const __nv_bfloat16* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
__device__ __forceinline__ float sum8(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}
__device__ __forceinline__ void st4(float* p, float a, float b, float c, float d) {
  *reinterpret_cast<uint4*>(p) = make_uint4(tf32r(a), tf32r(b), tf32r(c), tf32r(d));
}
// dp = r (du - yh mean(du yh)) for this lane's four channels of head h; (p4: raw projection, du4: gradient after unrope)
__device__ __forceinline__ float4 head_rms_bwd(const float4 p, const float4 du, float eps) {
  const float r = 1.f / sqrtf(sum8(p.x * p.x + p.y * p.y + p.z * p.z + p.w * p.w) * (1.f / D) + eps);
  const float4 yh = make_float4(p.x * r, p.y * r, p.z * r, p.w * r);
  const float t = sum8(du.x * yh.x + du.y * yh.y + du.z * yh.z + du.w * yh.w) * (1.f / D);
  return make_float4(r * (du.x - yh.x * t), r * (du.y - yh.y * t), r * (du.z - yh.z * t), r * (du.w - yh.w * t));
}
__device__ __forceinline__ float4 unrope(const float4 dy, const float4 c, const float4 s, bool lo) {
  const float4 o = make_float4(__shfl_xor_sync(0xffffffffu, dy.x, 4), __shfl_xor_sync(0xffffffffu, dy.y, 4),
                               __shfl_xor_sync(0xffffffffu, dy.z, 4), __shfl_xor_sync(0xffffffffu, dy.w, 4));
  const float k = lo ? 1.f : -1.f;                                         // lo: dx1 = dy1 c + dy2 s;  hi: dx2 = dy2 c - dy1 s
  return make_float4(dy.x * c.x + k * o.x * s.x, dy.y * c.y + k * o.y * s.y, dy.z * c.z + k * o.z * s.z, dy.w * c.w + k * o.w * s.w);
}

template <typename T>
__device__ __forceinline__ void run(const float* PQ, const float* PK, const T* DQH, const T* DKH, const T* DVH, const float* DG,
                                    const float* COS, const float* SIN, float* DP, int S, int Bn, int M, float eps) {
  const int row = (int)blockIdx.x * NWARP + (threadIdx.x >> 5);
  if (row >= M) return;
  const int lane = threadIdx.x & 31, h = lane >> 3, l8 = lane & 7;
  const int n = row / S, s = row - n * S, b = n % Bn, mrow = b * S + s;
  const bool lo = l8 < 4;
  const float4 c = ld4(COS + (size_t)mrow * (D / 2) + 4 * (l8 & 3)), sn = ld4(SIN + (size_t)mrow * (D / 2) + 4 * (l8 & 3));
  const size_t hm = (((size_t)n * H + h) * S + s) * D + 4 * l8;           // head-major element of this lane's channels
  const size_t rc = (size_t)row * C + 4 * lane;
  float* dp = DP + (size_t)row * (4 * C) + 4 * lane;
  const float4 pq = ld4(PQ + rc), dq = unrope(ld4(DQH + hm), c, sn, lo);
  const float4 oq = head_rms_bwd(pq, dq, eps);
  st4(dp, oq.x, oq.y, oq.z, oq.w);
  const float4 pk = ld4(PK + rc), dk = unrope(ld4(DKH + hm), c, sn, lo);
  const float4 ok = head_rms_bwd(pk, dk, eps);
  st4(dp + C, ok.x, ok.y, ok.z, ok.w);
  const float4 dv = ld4(DVH + hm);
  st4(dp + 2 * C, dv.x, dv.y, dv.z, dv.w);
  const float4 dg = ld4(DG + rc);
  st4(dp + 3 * C, dg.x, dg.y, dg.z, dg.w);
}

extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_qkvg_bwd_prep_tf32_sm100(const float* PQ, const float* PK, const float* DQH, const float* DKH, const float* DVH, const float* DG,
                             const float* COS, const float* SIN, float* DP, int S, int Bn, int M, float eps) {
  run<float>(PQ, PK, DQH, DKH, DVH, DG, COS, SIN, DP, S, Bn, M, eps);
}
extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_qkvg_bwd_prep_bf16h_tf32_sm100(const float* PQ, const float* PK, const __nv_bfloat16* DQH, const __nv_bfloat16* DKH,
                                   const __nv_bfloat16* DVH, const float* DG, const float* COS, const float* SIN, float* DP,
                                   int S, int Bn, int M, float eps) {
  run<__nv_bfloat16>(PQ, PK, DQH, DKH, DVH, DG, COS, SIN, DP, S, Bn, M, eps);
}
