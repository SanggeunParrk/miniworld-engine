// tf8_ln.cu -- the LayerNorm + e4m3 quantization pass of the opt-in B200 FP8 Transition (route fp8_sm100a.py): xn8 = e4m3(LN(x) sc[0]) from the fp32
// LayerNorm (no bf16 rounding first); one warp per row, D a multiple of 8 up to 1024. sc[0] = sx = pow2 scale of the LayerNorm bound (cannot saturate).
// tf8_rowquant (microbenchmark): xq = e4m3(x / s_row), s_row = amax_row / 448 (one read + one write of the bf16 activations).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

DEVI uint32_t pack4(float a, float b, float c, float d) {
  uint32_t lo, hi;
  asm("{ .reg .b16 r; cvt.rn.satfinite.e4m3x2.f32 r, %1, %2; cvt.u32.u16 %0, r; }" : "=r"(lo) : "f"(b), "f"(a));
  asm("{ .reg .b16 r; cvt.rn.satfinite.e4m3x2.f32 r, %1, %2; cvt.u32.u16 %0, r; }" : "=r"(hi) : "f"(d), "f"(c));
  return lo | (hi << 16);
}

extern "C" __global__ void __launch_bounds__(256)
tf8_ln(const __nv_bfloat16* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta, uint8_t* __restrict__ xn,
       const float* __restrict__ sc, int M, int D, float eps) {
  pdl_wait();
  pdl_launch();
  const int lane = threadIdx.x & 31;
  const int nch = D >> 3;                                       // 8-element chunks per row
  const float sx = sc[0];
  for (int row = blockIdx.x * 8 + (threadIdx.x >> 5); row < M; row += gridDim.x * 8) {
    float v[4][8];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int ch = lane + 32 * i;
      if (ch < nch) {
        const uint4 w = ldg128(x + (size_t)row * D + ch * 8);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { v[i][2 * e] = bf16lo(ww[e]); v[i][2 * e + 1] = bf16hi(ww[e]); s += v[i][2 * e] + v[i][2 * e + 1]; }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / (float)D;
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (lane + 32 * i < nch)
#pragma unroll
        for (int e = 0; e < 8; ++e) { const float d = v[i][e] - mean; q += d * d; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rstd = rsqrtf(q / (float)D + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int ch = lane + 32 * i;
      if (ch < nch) {
        const float4 g0 = *reinterpret_cast<const float4*>(gamma + ch * 8), g1 = *reinterpret_cast<const float4*>(gamma + ch * 8 + 4);
        const float4 b0 = *reinterpret_cast<const float4*>(beta + ch * 8), b1 = *reinterpret_cast<const float4*>(beta + ch * 8 + 4);
        const float gg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w}, bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        float y[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) y[e] = ((v[i][e] - mean) * rstd * gg[e] + bb[e]) * sx;
        *reinterpret_cast<uint2*>(xn + (size_t)row * D + ch * 8) =
            make_uint2(pack4(y[0], y[1], y[2], y[3]), pack4(y[4], y[5], y[6], y[7]));
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(256)
tf8_rowquant(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ xq, float* __restrict__ srow, int M, int D) {
  pdl_wait();
  pdl_launch();
  const int lane = threadIdx.x & 31, nch = D >> 3;
  for (int row = blockIdx.x * 8 + (threadIdx.x >> 5); row < M; row += gridDim.x * 8) {
    float v[4][8];
    float am = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int ch = lane + 32 * i;
      if (ch < nch) {
        const uint4 w = ldg128(x + (size_t)row * D + ch * 8);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { v[i][2 * e] = bf16lo(ww[e]); v[i][2 * e + 1] = bf16hi(ww[e]); am = fmaxf(am, fmaxf(fabsf(v[i][2 * e]), fabsf(v[i][2 * e + 1]))); }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = 448.f / fmaxf(am, 1e-30f);
    if (lane == 0) srow[row] = am / 448.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int ch = lane + 32 * i;
      if (ch < nch)
        *reinterpret_cast<uint2*>(xq + (size_t)row * D + ch * 8) = make_uint2(pack4(v[i][0] * sc, v[i][1] * sc, v[i][2] * sc, v[i][3] * sc),
                                                                              pack4(v[i][4] * sc, v[i][5] * sc, v[i][6] * sc, v[i][7] * sc));
    }
  }
}
