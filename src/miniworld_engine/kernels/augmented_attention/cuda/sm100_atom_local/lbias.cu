// lbias.cu — the per-window pair bias of the AF3 windowed atom attention and its backward.
//
//   bias[h, r] = sum_c w[h, c] * LN(z[r])[c]      w = Wb * gamma  [4, 16] (fp32), LN without affine, eps as given
//   z: the trunked atom pair [nwin * 32 * 128 rows, 16] bf16, bias fp32 [4, rows] (= [4, nwin, 32, 128])
//
// Backward from dbias [4, rows] (fp32): dz [rows, 16] (z's dtype) and per-block partial G[h, c] = sum_r dbias[h, r] LN(z[r])[c]
// (the caller sums the blocks: dWb = G * gamma, dgamma = sum_h G * Wb).
// SPDX-License-Identifier: Apache-2.0
#include "local.cuh"

DEVI void load_row(const __nv_bfloat16* z, size_t r, float* x) {
  const uint4 a = *reinterpret_cast<const uint4*>(z + r * 16), b = *reinterpret_cast<const uint4*>(z + r * 16 + 8);
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    x[2 * i] = __uint_as_float(w[i] << 16);
    x[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}
DEVI void normalise(float* x, float eps, float& rstd) {
  float mean = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) mean += x[i];
  mean *= 0.0625f;
  float var = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) { x[i] -= mean; var += x[i] * x[i]; }
  rstd = 1.f / sqrtf(var * 0.0625f + eps);
#pragma unroll
  for (int i = 0; i < 16; ++i) x[i] *= rstd;
}

extern "C" __global__ void __launch_bounds__(256)
local_bias_fwd(const __nv_bfloat16* __restrict__ z, const float* __restrict__ w, float* __restrict__ bias, int rows, float eps) {
  __shared__ float sw[64];
  if (threadIdx.x < 64) sw[threadIdx.x] = w[threadIdx.x];
  __syncthreads();
  for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += (long long)gridDim.x * blockDim.x) {
    float x[16], rstd;
    load_row(z, r, x);
    normalise(x, eps, rstd);
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      float b = 0.f;
#pragma unroll
      for (int c = 0; c < 16; ++c) b += sw[h * 16 + c] * x[c];
      bias[(size_t)h * rows + r] = b;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256)
local_bias_bwd(const __nv_bfloat16* __restrict__ z, const float* __restrict__ w, const float* __restrict__ dbias,
               __nv_bfloat16* __restrict__ dz, float* __restrict__ part, int rows, float eps) {
  __shared__ float sw[64], red[8][64];
  if (threadIdx.x < 64) sw[threadIdx.x] = w[threadIdx.x];
  __syncthreads();
  float G[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) G[i] = 0.f;
  for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += (long long)gridDim.x * blockDim.x) {
    float x[16], rstd;
    load_row(z, r, x);
    normalise(x, eps, rstd);                                           // x = zhat
    float db[4], dzh[16];
#pragma unroll
    for (int h = 0; h < 4; ++h) db[h] = dbias[(size_t)h * rows + r];
#pragma unroll
    for (int c = 0; c < 16; ++c) {
      dzh[c] = db[0] * sw[c] + db[1] * sw[16 + c] + db[2] * sw[32 + c] + db[3] * sw[48 + c];
#pragma unroll
      for (int h = 0; h < 4; ++h) G[h * 16 + c] += db[h] * x[c];
    }
    float m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int c = 0; c < 16; ++c) { m1 += dzh[c]; m2 += dzh[c] * x[c]; }
    m1 *= 0.0625f; m2 *= 0.0625f;
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      o[i] = pack_bf16(rstd * (dzh[2 * i] - m1 - x[2 * i] * m2), rstd * (dzh[2 * i + 1] - m1 - x[2 * i + 1] * m2));
    *reinterpret_cast<uint4*>(dz + (size_t)r * 16) = make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4*>(dz + (size_t)r * 16 + 8) = make_uint4(o[4], o[5], o[6], o[7]);
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    float v = G[i];
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[warp][i] = v;
  }
  __syncthreads();
  if (threadIdx.x < 64) {
    float s = 0.f;
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) s += red[wv][threadIdx.x];
    part[(size_t)blockIdx.x * 64 + threadIdx.x] = s;
  }
}
