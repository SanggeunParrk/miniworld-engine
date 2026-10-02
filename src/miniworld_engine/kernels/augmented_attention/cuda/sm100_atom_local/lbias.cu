// lbias.cu — the per-window pair bias of the AF3 windowed atom attention and its backward.
//
//   bias[h, r] = sum_c w[h, c] * LN(z[r])[c]      w = Wb * gamma  [4, 16] (fp32), LN without affine, eps as given
//   z: the trunked atom pair [nwin * 32 * 128 rows, 16] bf16, bias fp32 [4, rows] (= [4, nwin, 32, 128])
//
// Backward from dbias [4, rows] (fp32): dz [rows, 16] (z's dtype) and per-block partial G[h, c] = sum_r dbias[h, r] LN(z[r])[c]
// (local_bias_fin sums the blocks: dWb = G * gamma, dgamma = sum_h G * Wb). Wb and gamma are read as given (bf16 or fp32).
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

// w[h, c] = Wb[h, c] * gamma[c] in fp32 (bf16 or fp32 parameters)
DEVI float param_f(const void* p, int i, int is_bf16) {
  return is_bf16 ? __uint_as_float((uint32_t)reinterpret_cast<const unsigned short*>(p)[i] << 16) : reinterpret_cast<const float*>(p)[i];
}
DEVI void load_w(float* sw, const void* wb, const void* gamma, int is_bf16) {
  if (threadIdx.x < 64) sw[threadIdx.x] = param_f(wb, threadIdx.x, is_bf16) * param_f(gamma, threadIdx.x & 15, is_bf16);
}

extern "C" __global__ void __launch_bounds__(256)
local_bias_fwd(const __nv_bfloat16* __restrict__ z, const void* __restrict__ wb, const void* __restrict__ gamma, float* __restrict__ bias,
               int rows, float eps, int is_bf16) {
  __shared__ float sw[64];
  load_w(sw, wb, gamma, is_bf16);
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
local_bias_bwd(const __nv_bfloat16* __restrict__ z, const void* __restrict__ wb, const void* __restrict__ gamma, const float* __restrict__ dbias,
               __nv_bfloat16* __restrict__ dz, float* __restrict__ part, int rows, float eps, int is_bf16) {
  __shared__ float sw[64], red[8][64];
  load_w(sw, wb, gamma, is_bf16);
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
  // reduce-scatter of the 64 partial sums over the 32 lanes (62 shuffles instead of 320): after the five steps lane l holds the
  // sums of indices 2 l' and 2 l' + 1 with l' = the lane bits reversed into the index (bit4 -> 32, bit3 -> 16, ... bit0 -> 2)
  float v32[32];
  {
    const bool hi = (lane & 16) != 0;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
      const float keep = hi ? G[k + 32] : G[k], send = hi ? G[k] : G[k + 32];
      v32[k] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
    }
  }
  float v16[16];
  {
    const bool hi = (lane & 8) != 0;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const float keep = hi ? v32[k + 16] : v32[k], send = hi ? v32[k] : v32[k + 16];
      v16[k] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
    }
  }
  float v8[8];
  {
    const bool hi = (lane & 4) != 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float keep = hi ? v16[k + 8] : v16[k], send = hi ? v16[k] : v16[k + 8];
      v8[k] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
    }
  }
  float v4[4];
  {
    const bool hi = (lane & 2) != 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float keep = hi ? v8[k + 4] : v8[k], send = hi ? v8[k] : v8[k + 4];
      v4[k] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
    }
  }
  float v2[2];
  {
    const bool hi = (lane & 1) != 0;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const float keep = hi ? v4[k + 2] : v4[k], send = hi ? v4[k] : v4[k + 2];
      v2[k] = keep + __shfl_xor_sync(0xffffffffu, send, 1);
    }
  }
  {
    const int base = ((lane >> 4) & 1) * 32 + ((lane >> 3) & 1) * 16 + ((lane >> 2) & 1) * 8 + ((lane >> 1) & 1) * 4 + (lane & 1) * 2;
    red[warp][base] = v2[0];
    red[warp][base + 1] = v2[1];
  }
  __syncthreads();
  if (threadIdx.x < 64) {
    float s = 0.f;
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) s += red[wv][threadIdx.x];
    part[(size_t)blockIdx.x * 64 + threadIdx.x] = s;
  }
}

// the sum of the per-block partials G[h, c] -> dgamma[c] = sum_h G[h, c] Wb[h, c] (fp32 [16]) and dWb[h, c] = G[h, c] gamma[c] (fp32 [4, 16])
extern "C" __global__ void __launch_bounds__(64)
local_bias_fin(const float* __restrict__ part, int nb, const void* __restrict__ wb, const void* __restrict__ gamma,
               float* __restrict__ dgamma, float* __restrict__ dwb, int is_bf16) {
  __shared__ float G[64];
  const int t = threadIdx.x;
  float s = 0.f;
  for (int b = 0; b < nb; ++b) s += part[(size_t)b * 64 + t];
  G[t] = s;
  __syncthreads();
  dwb[t] = s * param_f(gamma, t & 15, is_bf16);
  if (t < 16) {
    float d = 0.f;
#pragma unroll
    for (int h = 0; h < 4; ++h) d += G[h * 16 + t] * param_f(wb, h * 16 + t, is_bf16);
    dgamma[t] = d;
  }
}
