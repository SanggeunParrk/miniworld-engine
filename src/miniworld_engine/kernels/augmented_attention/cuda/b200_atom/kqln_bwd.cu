// kqln_bwd.cu — the backward of RoseTTAFold3's flat q / k LayerNorm (pre_fwd.cu -DKQLN=1) on the attention backward's dq | dk, in place, one
// launch (B200 variant path, 2026-11-17; sm_100a, no tensor cores: 2 x 128 channels per row).
//
//   forward (per row, p = q, k):  y_p = rn(xhat_p w_p + b_p),  xhat_p = (x_p - mean) rstd over the 128 channels (fp32, x_p the raw projection)
//   backward:  d w_p += sum_rows dy_p xhat_p;  d b_p += sum_rows dy_p;  g = dy_p w_p;
//              dx_p = rn(rstd (g - mean(g) - xhat_p mean(g xhat_p)))   -> written over dy_p in dP
// dP [M][512] bf16 = [dq | dk | dv | dg] (blocks 0, 1 read and rewritten), QKR [M][256] bf16 = the raw [q | k] (pre_fwd's save),
// KQ [4][128] fp32 = [wq | bq | wk | bk] (EW = 4 bytes) or the weights' own bf16 rows (EW = 2), PARTS [grid][512] fp32 = this CTA's column
// sums [d wq | d bq | d wk | d bk] (one partial row per CTA: the host sums them in a fixed order -- deterministic).
// One warp per row (a lane: 4 channels of q and of k), 8 rows per CTA step, the CTAs stride over the rows.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cuda_bf16.h>

static __device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
static __device__ __forceinline__ uint16_t f2bf(float x) {
  const __nv_bfloat16 b = __float2bfloat16_rn(x);
  return *reinterpret_cast<const uint16_t*>(&b);
}
static __device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

extern "C" __global__ void __launch_bounds__(256) atom_kqln_bwd(uint16_t* __restrict__ dP, const uint16_t* __restrict__ QKR,
                                                                 const float* __restrict__ KQ, float* __restrict__ PARTS, int M, float eps) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  __shared__ float red[8][32][16];
  float w[2][4], acc[16];                                                  // acc: [p][dw 4 | db 4]
#pragma unroll
  for (int p = 0; p < 2; ++p)
#pragma unroll
    for (int j = 0; j < 4; ++j) w[p][j] = __ldg(KQ + 256 * p + 4 * lane + j);
#pragma unroll
  for (int e = 0; e < 16; ++e) acc[e] = 0.f;
  for (int r = 8 * (int)blockIdx.x + warp; r < M; r += 8 * (int)gridDim.x) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const uint2 xu = *reinterpret_cast<const uint2*>(QKR + (size_t)r * 256 + 128 * p + 4 * lane);
      uint2* dptr = reinterpret_cast<uint2*>(dP + (size_t)r * 512 + 128 * p + 4 * lane);
      const uint2 du = *dptr;
      float x[4] = {bf2f(xu.x & 0xffffu), bf2f(xu.x >> 16), bf2f(xu.y & 0xffffu), bf2f(xu.y >> 16)};
      const float d[4] = {bf2f(du.x & 0xffffu), bf2f(du.x >> 16), bf2f(du.y & 0xffffu), bf2f(du.y >> 16)};
      const float mean = wsum(x[0] + x[1] + x[2] + x[3]) * (1.f / 128.f);
      float var = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) { x[j] -= mean; var += x[j] * x[j]; }
      const float rstd = rsqrtf(wsum(var) * (1.f / 128.f) + eps);
      float g[4], a = 0.f, b = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        x[j] *= rstd;
        acc[8 * p + j] += d[j] * x[j];
        acc[8 * p + 4 + j] += d[j];
        g[j] = d[j] * w[p][j];
        a += g[j];
        b += g[j] * x[j];
      }
      a = wsum(a) * (1.f / 128.f);
      b = wsum(b) * (1.f / 128.f);
      uint16_t o[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j] = f2bf(rstd * (g[j] - a - x[j] * b));
      *dptr = make_uint2((uint32_t)o[0] | ((uint32_t)o[1] << 16), (uint32_t)o[2] | ((uint32_t)o[3] << 16));
    }
  }
#pragma unroll
  for (int e = 0; e < 16; ++e) red[warp][lane][e] = acc[e];
  __syncthreads();
  // the CTA's partial row: [d wq | d bq | d wk | d bk], each 128 channels (channel 4 lane + j), the eight warps summed in order
  for (int c = tid; c < 512; c += 256) {
    const int p = c >> 8, kind = (c >> 7) & 1, ch = c & 127;
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) v += red[k][ch >> 2][8 * p + 4 * kind + (ch & 3)];
    PARTS[(size_t)blockIdx.x * 512 + c] = v;
  }
}
