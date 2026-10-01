// pair_bias.cu — the atom DiT's pair-bias producer and its backward, memory-bound CUDA (one element (i, j) per thread-iteration).
//
//   forward:  bias[h, i, j] = rstd (z[i, j, :] . W'[h] - mean sum_c W'[h, c]),  W' = gamma * Wb   (LayerNorm without offset, eps 1e-5)
//             written head-major [H, N, N] and transposed [H, N(key), N(query)] (staged through shared memory, so both stores coalesce)
//   backward: dz = rstd (dxh - mean(dxh) - xh mean(dxh xh)),  dxh = sum_h dbias[h] W'[h];   d(W')[h, c] = sum_ij dbias[h] xh[c]
//             (per-block partials, summed on the host side)
// Key mask and padding: the bias side is N x N (N a multiple of 128, the attention's length), z is NZ x NZ (NZ <= N atoms);
// kv [NZ] (bytes 0 / 1: the caller's bool mask as is; null = every key valid) marks the valid keys; keys >= NZ are padding
// (pair_bias_fwd_pad, the bounds-checked build). A masked key's
// column is written as MASKED (-1e4, bf16 -9984: 2^(-9984 log2 e + score - max) is 0 in fp32 for any row with a valid key, and
// small enough that scores and the LSE keep their bits, so the backward's recomputed P matches the forward's even in a row
// whose keys are all masked -- there the weights are the softmax of the scores, where the module's finfo.min fill gives
// uniform ones); padded query rows (i >= NZ) are 0 on valid keys. The backward reads only i, j < NZ and drops dbias on masked
// keys (the module's masked_fill passes no gradient there).
// Block tile: 32 rows (i) x 64 columns (j), 256 threads: thread (ty, tx) = (tid / 64, tid % 64) takes rows ty + 4 k, k < 8.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cuda_bf16.h>

constexpr int C = 16, H = 4, TI = 32, TJ = 64, NT = 256;
constexpr float MASKED = -1e4f;

__device__ __forceinline__ void load_row(const __nv_bfloat16* z, size_t e, float (&x)[C]) {
  const uint4* p = reinterpret_cast<const uint4*>(z + e * C);
  const uint4 a = __ldg(p), b = __ldg(p + 1);
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int k = 0; k < 8; ++k) { x[2 * k] = __uint_as_float(w[k] << 16); x[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u); }
}

// PAD: N > NZ (padded atoms: bounds checks on the z reads); without padding the loop is branch-free, the mask a select.
template <bool PAD>
__device__ __forceinline__ void pair_bias_fwd_body(const __nv_bfloat16* __restrict__ z, const float* __restrict__ wp,
                                                   __nv_bfloat16* __restrict__ bo, __nv_bfloat16* __restrict__ bt, int N, int NZ,
                                                   const unsigned char* __restrict__ kv, float eps) {
  __shared__ float w[H * C], ws[H];
  __shared__ __nv_bfloat16 st[H][TJ][TI + 2];
  const int tid = threadIdx.x, tx = tid & 63, ty = tid >> 6;
  if (tid < H * C) w[tid] = wp[tid];
  __syncthreads();
  if (tid < H) { float s = 0.f; for (int c = 0; c < C; ++c) s += w[tid * C + c]; ws[tid] = s; }
  __syncthreads();
  const int i0 = blockIdx.y * TI, j = blockIdx.x * TJ + tx;
  const bool jin = !PAD || j < NZ;
  const bool valid = jin && (kv == nullptr || kv[j]);
#pragma unroll 2
  for (int k = 0; k < TI / 4; ++k) {
    const int il = ty + 4 * k, i = i0 + il;
    float y[H];
    if (!PAD || (i < NZ && jin)) {
      float x[C];
      load_row(z, (size_t)i * NZ + j, x);
      float s = 0.f, s2 = 0.f;
#pragma unroll
      for (int c = 0; c < C; ++c) { s += x[c]; s2 += x[c] * x[c]; }
      const float mean = s * (1.f / C), rstd = rsqrtf(fmaxf(s2 * (1.f / C) - mean * mean, 0.f) + eps);
#pragma unroll
      for (int h = 0; h < H; ++h) {
        float d = 0.f;
#pragma unroll
        for (int c = 0; c < C; ++c) d = fmaf(x[c], w[h * C + c], d);
        y[h] = rstd * (d - mean * ws[h]);
      }
    } else {                                                               // a padded query row or key
#pragma unroll
      for (int h = 0; h < H; ++h) y[h] = 0.f;
    }
#pragma unroll
    for (int h = 0; h < H; ++h) {
      const __nv_bfloat16 v = __float2bfloat16_rn(valid ? y[h] : MASKED);
      bo[((size_t)h * N + i) * N + j] = v;
      st[h][tx][il] = v;
    }
  }
  if (bt == nullptr) return;
  __syncthreads();
  // transposed: rows j0 .. j0 + 63 of bias^T[h], 32 queries (64 B) each; a thread writes 4 bf16 pairs per head
  const int j0 = blockIdx.x * TJ;
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int idx = q * NT + tid, r = idx >> 4, pr = idx & 15;                                     // row r (0..63), pair pr (0..15)
      const __nv_bfloat162 v = __halves2bfloat162(st[h][r][2 * pr], st[h][r][2 * pr + 1]);
      *reinterpret_cast<__nv_bfloat162*>(bt + ((size_t)h * N + j0 + r) * N + i0 + 2 * pr) = v;
    }
}

extern "C" __global__ void __launch_bounds__(NT)
pair_bias_fwd(const __nv_bfloat16* __restrict__ z, const float* __restrict__ wp, __nv_bfloat16* __restrict__ bo,
              __nv_bfloat16* __restrict__ bt, int N, int NZ, const unsigned char* __restrict__ kv, float eps) {
  pair_bias_fwd_body<false>(z, wp, bo, bt, N, NZ, kv, eps);
}

extern "C" __global__ void __launch_bounds__(NT)
pair_bias_fwd_pad(const __nv_bfloat16* __restrict__ z, const float* __restrict__ wp, __nv_bfloat16* __restrict__ bo,
                  __nv_bfloat16* __restrict__ bt, int N, int NZ, const unsigned char* __restrict__ kv, float eps) {
  pair_bias_fwd_body<true>(z, wp, bo, bt, N, NZ, kv, eps);
}

extern "C" __global__ void __launch_bounds__(NT)
pair_bias_bwd(const __nv_bfloat16* __restrict__ z, const float* __restrict__ wp, const float* __restrict__ db,
              __nv_bfloat16* __restrict__ dz, float* __restrict__ pw, int N, int NZ, const unsigned char* __restrict__ kv, float eps) {
  __shared__ float w[H * C];
  __shared__ float red[NT / 32][H * C];
  const int tid = threadIdx.x, tx = tid & 63, ty = tid >> 6;
  if (tid < H * C) w[tid] = wp[tid];
  __syncthreads();
  const int i0 = blockIdx.y * TI, j = blockIdx.x * TJ + tx;
  float acc[H][C];
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int c = 0; c < C; ++c) acc[h][c] = 0.f;
  const bool valid = j < NZ && (kv == nullptr || kv[j]);
  for (int k = 0; k < TI / 4; ++k) {
    const int i = i0 + ty + 4 * k;
    if (i >= NZ || j >= NZ) continue;                                      // padding: no z there
    const size_t e = (size_t)i * NZ + j;
    float x[C];
    load_row(z, e, x);
    float s = 0.f, s2 = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) { s += x[c]; s2 += x[c] * x[c]; }
    const float mean = s * (1.f / C), rstd = rsqrtf(fmaxf(s2 * (1.f / C) - mean * mean, 0.f) + eps);
    float g[H];
#pragma unroll
    for (int h = 0; h < H; ++h) g[h] = valid ? __ldg(db + ((size_t)h * N + i) * N + j) : 0.f;
    float m1 = 0.f, m2 = 0.f, dxh[C];
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const float xh = (x[c] - mean) * rstd;
      float d = 0.f;
#pragma unroll
      for (int h = 0; h < H; ++h) { d = fmaf(g[h], w[h * C + c], d); acc[h][c] = fmaf(g[h], xh, acc[h][c]); }
      dxh[c] = d; m1 += d; m2 += d * xh; x[c] = xh;
    }
    m1 *= 1.f / C; m2 *= 1.f / C;
    uint32_t o[8];
#pragma unroll
    for (int k2 = 0; k2 < 8; ++k2) {
      const __nv_bfloat162 v = __floats2bfloat162_rn((dxh[2 * k2] - m1 - x[2 * k2] * m2) * rstd, (dxh[2 * k2 + 1] - m1 - x[2 * k2 + 1] * m2) * rstd);
      o[k2] = *reinterpret_cast<const uint32_t*>(&v);
    }
    uint4* p = reinterpret_cast<uint4*>(dz + e * C);
    p[0] = make_uint4(o[0], o[1], o[2], o[3]);
    p[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
  // block partial of d(W'): warp shuffle, then across the 8 warps
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int c = 0; c < C; ++c) {
      float v = acc[h][c];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if ((tid & 31) == 0) red[tid >> 5][h * C + c] = v;
    }
  __syncthreads();
  if (tid < H * C) {
    float v = 0.f;
#pragma unroll
    for (int r = 0; r < NT / 32; ++r) v += red[r][tid];
    pw[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * H * C + tid] = v;
  }
}
