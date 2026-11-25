// SPDX-License-Identifier: Apache-2.0
// LN_in forward for C_Z = 512 (bf16 in, bf16 out): one warp per pair row, fp32 two-pass centred statistics, y = bf16((x - mean) rstd gamma + beta).
// Feeds the front K1 (front_pre_ss.cu with MW_PROF_MODE bit 0 = operand already normalised) and is the saved x_n of the training forward.
#include <cuda_bf16.h>
#include <cstdint>
struct Params { const __nv_bfloat16* x; __nv_bfloat16* xn; const float* gamma; const float* beta; int M; float eps; };
__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t pk(float a, float b) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
extern "C" __global__ __launch_bounds__(256, 4)
void mw_ln_in_d512(__grid_constant__ const Params p) {
  const int lane = threadIdx.x & 31;
  const int warp0 = blockIdx.x * 8 + (threadIdx.x >> 5), nwarps = gridDim.x * 8;
  float g[16], b[16];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float4* gp = reinterpret_cast<const float4*>(p.gamma + h * 256 + lane * 8);
    const float4* bp = reinterpret_cast<const float4*>(p.beta + h * 256 + lane * 8);
    float4 g0 = __ldg(gp), g1 = __ldg(gp + 1), b0 = __ldg(bp), b1 = __ldg(bp + 1);
    g[h * 8 + 0] = g0.x; g[h * 8 + 1] = g0.y; g[h * 8 + 2] = g0.z; g[h * 8 + 3] = g0.w;
    g[h * 8 + 4] = g1.x; g[h * 8 + 5] = g1.y; g[h * 8 + 6] = g1.z; g[h * 8 + 7] = g1.w;
    b[h * 8 + 0] = b0.x; b[h * 8 + 1] = b0.y; b[h * 8 + 2] = b0.z; b[h * 8 + 3] = b0.w;
    b[h * 8 + 4] = b1.x; b[h * 8 + 5] = b1.y; b[h * 8 + 6] = b1.z; b[h * 8 + 7] = b1.w;
  }
  for (int row = warp0; row < p.M; row += nwarps) {
    const uint4* xr = reinterpret_cast<const uint4*>(p.x + (size_t)row * 512);
    const uint4 v0 = __ldcs(xr + lane), v1 = __ldcs(xr + 32 + lane);
    const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += lo(w[i]) + hi(w[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s * (1.f / 512.f);
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { const float a = lo(w[i]) - mean, c = hi(w[i]) - mean; q += a * a + c * c; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rstd = rsqrtf(q * (1.f / 512.f) + p.eps);
    uint4 o0, o1;
    uint32_t* po[8] = {&o0.x, &o0.y, &o0.z, &o0.w, &o1.x, &o1.y, &o1.z, &o1.w};
    // granule 0 = words 0..3 (channels lane*8 .. +7, gamma/beta index 0..7); granule 1 = words 4..7 (index 8..15)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int e = 2 * i;   // gamma/beta slot of the pair's low element (0..14)
      *po[i] = pk((lo(w[i]) - mean) * rstd * g[e] + b[e], (hi(w[i]) - mean) * rstd * g[e + 1] + b[e + 1]);
    }
    uint4* yr = reinterpret_cast<uint4*>(p.xn + (size_t)row * 512);
    yr[lane] = o0; yr[32 + lane] = o1;
  }
}
