// mod_bwd_act.cu — the elementwise part of the SWA modulation backward on many rows (sm_100a; the GEMMs are cuBLAS, TF32 on fp32 g):
//   pre = g Wmod [R, 128] fp32 (cuBLAS),  c [R, 128] bf16  ->  dc = silu'(c) pre  (bf16; s = 1 / (1 + exp(-x)), dy s (1 + x (1 - s)) in
//   fp32, as mod_bwd.cu's dc epilogue)  and  S = rn_bf16(silu(c)) as fp32 [R, 128] (the forward's bf16 operand, mod_fwd.cu's approximate
//   sigmoid), the B operand of the dWmod GEMM g^T S (cuBLAS).
// One thread per 8 consecutive elements (one 16-B load of c).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

extern "C" __global__ void __launch_bounds__(256)
swa_mod_bwd_act_sm100(const float* __restrict__ PRE, const uint16_t* __restrict__ Cc, uint16_t* __restrict__ DC, float* __restrict__ SO,
                      int n8) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (long long)gridDim.x * blockDim.x) {
    const uint4 cu4 = __ldg(reinterpret_cast<const uint4*>(Cc) + i);
    const float4 p0 = __ldg(reinterpret_cast<const float4*>(PRE) + 2 * i), p1 = __ldg(reinterpret_cast<const float4*>(PRE) + 2 * i + 1);
    const float pv[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
    const uint32_t cw[4] = {cu4.x, cu4.y, cu4.z, cu4.w};
    uint32_t o[4];
    float s[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float dx[2];
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const float x = t ? bf16hi(cw[e]) : bf16lo(cw[e]), sg = 1.f / (1.f + expf(-x));
        dx[t] = pv[2 * e + t] * sg * (1.f + x * (1.f - sg));
        s[2 * e + t] = x * sigmoid_kit(x);
      }
      o[e] = pack_bf16(dx[0], dx[1]);
    }
    reinterpret_cast<uint4*>(DC)[i] = make_uint4(o[0], o[1], o[2], o[3]);
    float4* so = reinterpret_cast<float4*>(SO) + 2 * i;
    uint32_t r[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) r[e] = pack_bf16(s[2 * e], s[2 * e + 1]);
    so[0] = make_float4(bf16lo(r[0]), bf16hi(r[0]), bf16lo(r[1]), bf16hi(r[1]));
    so[1] = make_float4(bf16lo(r[2]), bf16hi(r[2]), bf16lo(r[3]), bf16hi(r[3]));
  }
}
