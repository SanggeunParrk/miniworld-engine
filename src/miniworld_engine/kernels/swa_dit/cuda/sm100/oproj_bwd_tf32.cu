// oproj_bwd_tf32.cu — the fp32 SWA atom block's out-projection backward on sm_100a (the Triton _swa_oproj_bwd_fp32_kernel), in two element-wise
// kernels around the one GEMM  dgated = datt Wo  (cuBLAS TF32): the block output is  q1 = q + gate_a att,  att = gated Wo^T,
// gated = sigmoid(g) o  (g: the raw gate, o: the attention output).
//   prep:  datt = dq1 gate_a;  gated = sigmoid(g) o;  d gate_a = sum_aug dq1 att        -> DATT, GATED (TF32-rounded: dgated's and dWo's
//          operands), dmod column 2C (plain stores: one block per (batch element, atom), eight warps over the augments)
//   post:  so = sigmoid(g);  dO = dgated so;  dG = dgated o so (1 - so);  Dv[n, h, s] = sum_d dO o   (dO as the attention backward takes
//          it: TF32-rounded fp32 or bf16; Dv from the rounded dO). dG goes straight into dP [M, 4C] at column 3C (TF32-rounded: dP is the
//          operand of dx = dP W and of the weight gradients), next to the dpq | dpk | dpv the attention epilogues write
// One warp per 128-channel row in both (lane = four channels, a head = eight lanes).
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>

constexpr int C = 128, H = 4, D = 32, MODW = 6 * C, NWARP = 8;

__device__ __forceinline__ uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r & 0xffffe000u; }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float rtf(float x) { return __uint_as_float(tf32r(x)); }
__device__ __forceinline__ float sum8(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}
__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ void st4t(float* p, float a, float b, float c, float d) {
  *reinterpret_cast<uint4*>(p) = make_uint4(tf32r(a), tf32r(b), tf32r(c), tf32r(d));
}

extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_oproj_bwd_prep_tf32_sm100(const float* __restrict__ DQ1, const float* __restrict__ O, const float* __restrict__ G,
                              const float* __restrict__ ATT, const float* __restrict__ MOD, float* __restrict__ DATT,
                              float* __restrict__ GATED, float* __restrict__ DMOD, int S, int A, int Bn) {
  __shared__ float red[NWARP][C];
  const int bs = (int)blockIdx.x, b = bs / S, s = bs % S;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float4 ga = ld4(MOD + (size_t)bs * MODW + 2 * C + 4 * lane);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int a = warp; a < A; a += NWARP) {
    const size_t o = (((size_t)a * Bn + b) * S + s) * C + 4 * lane;
    const float4 dq = ld4(DQ1 + o), ov = ld4(O + o), g = ld4(G + o), at = ld4(ATT + o);
    st4t(DATT + o, dq.x * ga.x, dq.y * ga.y, dq.z * ga.z, dq.w * ga.w);
    st4t(GATED + o, sigm(g.x) * ov.x, sigm(g.y) * ov.y, sigm(g.z) * ov.z, sigm(g.w) * ov.w);
    acc.x += dq.x * at.x; acc.y += dq.y * at.y; acc.z += dq.z * at.z; acc.w += dq.w * at.w;
  }
  *reinterpret_cast<float4*>(&red[warp][4 * lane]) = acc;
  __syncthreads();
  if (threadIdx.x < C) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < NWARP; ++w) t += red[w][threadIdx.x];
    DMOD[(size_t)bs * MODW + 2 * C + threadIdx.x] = t;
  }
}

template <bool BF>
__device__ __forceinline__ void post(const float* DGATED, const float* O, const float* G, void* DOUT, float* DP, float* DV, int S, int M) {
  const int row = (int)blockIdx.x * NWARP + (threadIdx.x >> 5);
  if (row >= M) return;
  const int lane = threadIdx.x & 31, h = lane >> 3, l8 = lane & 7;
  const int n = row / S, s = row - n * S;
  const size_t rc = (size_t)row * C + 4 * lane;
  const float4 dg = ld4(DGATED + rc), ov = ld4(O + rc), g = ld4(G + rc);
  const float sx = sigm(g.x), sy = sigm(g.y), sz = sigm(g.z), sw = sigm(g.w);
  float dox, doy, doz, dow;
  if (BF) { dox = rbf(dg.x * sx); doy = rbf(dg.y * sy); doz = rbf(dg.z * sz); dow = rbf(dg.w * sw); }
  else    { dox = rtf(dg.x * sx); doy = rtf(dg.y * sy); doz = rtf(dg.z * sz); dow = rtf(dg.w * sw); }
  if (BF) {
    uint2 u;
    u.x = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(dox)) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(doy)) << 16);
    u.y = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(doz)) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(dow)) << 16);
    *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(DOUT) + rc) = u;
  } else {
    *reinterpret_cast<float4*>(reinterpret_cast<float*>(DOUT) + rc) = make_float4(dox, doy, doz, dow);
  }
  *reinterpret_cast<uint4*>(DP + (size_t)row * (4 * C) + 3 * C + 4 * lane) =
      make_uint4(tf32r(dg.x * ov.x * sx * (1.f - sx)), tf32r(dg.y * ov.y * sy * (1.f - sy)), tf32r(dg.z * ov.z * sz * (1.f - sz)),
                 tf32r(dg.w * ov.w * sw * (1.f - sw)));
  const float dv = sum8(dox * ov.x + doy * ov.y + doz * ov.z + dow * ov.w);
  if (l8 == 0) DV[((size_t)n * H + h) * S + s] = dv;
}

extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_oproj_bwd_post_tf32_sm100(const float* DGATED, const float* O, const float* G, float* DOUT, float* DP, float* DV, int S, int M) {
  post<false>(DGATED, O, G, DOUT, DP, DV, S, M);
}
extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_oproj_bwd_post_bf16o_tf32_sm100(const float* DGATED, const float* O, const float* G, __nv_bfloat16* DOUT, float* DP, float* DV, int S, int M) {
  post<true>(DGATED, O, G, DOUT, DP, DV, S, M);
}
