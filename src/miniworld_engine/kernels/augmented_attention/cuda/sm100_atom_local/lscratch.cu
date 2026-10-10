#include "local.cuh"
DEVI float hsum(float v) {   // half-warp (16 lanes) sum
#pragma unroll
  for (int o = 8; o >= 1; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEVI float wsum(float v) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// V1: 16 B per lane (8 channels), 16 lanes per row, 2 rows per warp, one row per half-warp per iteration, UNR rows in flight per half-warp
template <int U>
__device__ __forceinline__ void body16(const __nv_bfloat16* c, const float* gamma, __nv_bfloat16* out, int M, float eps, int step) {
  const int hl = threadIdx.x & 15, half = (threadIdx.x >> 4);          // half 0..15 within 256 threads
  float g[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) g[i] = gamma[hl * 8 + i];
  for (int row0 = blockIdx.x * 16 + half; row0 < M; row0 += U * step) {
    uint4 u[U];
#pragma unroll
    for (int j = 0; j < U; ++j) { const int row = row0 + j * step; if (row < M) u[j] = *reinterpret_cast<const uint4*>(c + (size_t)row * 128 + hl * 8); }
#pragma unroll
    for (int j = 0; j < U; ++j) {
      const int row = row0 + j * step;
      float x[8];
      const uint32_t w[4] = {u[j].x, u[j].y, u[j].z, u[j].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) { x[2 * e] = __uint_as_float(w[e] << 16); x[2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u); }
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) s += x[i];
      const float mean = hsum(s) * (1.f / 128.f);
      float v = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) { x[i] -= mean; v += x[i] * x[i]; }
      const float rstd = 1.f / sqrtf(hsum(v) * (1.f / 128.f) + eps);
      uint4 o;
      o.x = pack_bf16(x[0] * rstd * g[0], x[1] * rstd * g[1]); o.y = pack_bf16(x[2] * rstd * g[2], x[3] * rstd * g[3]);
      o.z = pack_bf16(x[4] * rstd * g[4], x[5] * rstd * g[5]); o.w = pack_bf16(x[6] * rstd * g[6], x[7] * rstd * g[7]);
      if (row < M) *reinterpret_cast<uint4*>(out + (size_t)row * 128 + hl * 8) = o;
    }
  }
}
extern "C" __global__ void __launch_bounds__(256) v16_u1(const __nv_bfloat16* c, const float* g, __nv_bfloat16* o, int M, float eps) { body16<1>(c, g, o, M, eps, gridDim.x * 16); }
extern "C" __global__ void __launch_bounds__(256) v16_u2(const __nv_bfloat16* c, const float* g, __nv_bfloat16* o, int M, float eps) { body16<2>(c, g, o, M, eps, gridDim.x * 16); }
extern "C" __global__ void __launch_bounds__(256) v16_u4(const __nv_bfloat16* c, const float* g, __nv_bfloat16* o, int M, float eps) { body16<4>(c, g, o, M, eps, gridDim.x * 16); }
extern "C" __global__ void __launch_bounds__(256) v16_u8(const __nv_bfloat16* c, const float* g, __nv_bfloat16* o, int M, float eps) { body16<8>(c, g, o, M, eps, gridDim.x * 16); }
