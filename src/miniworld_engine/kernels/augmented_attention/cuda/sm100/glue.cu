// glue.cu — the small passes around the bf16 attention core's backward (CUDA; they were Triton):
//
//   bias_transpose_bf16   bias [H, L(query), L(key)] -> [H, L(key), L(query)] (attn_dkv reads its key rows contiguously)
//   prep_do_bf16          dO [M, 768] fp32 -> bf16, and D = rowsum_head(dO O) as [A, 16, L] fp32 (O fp32)
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cstdint>

// 32 x 32 tiles through shared memory (padded against bank conflicts); grid (L / 32, L / 32, H), block (32, 8)
extern "C" __global__ void __launch_bounds__(256) bias_transpose_bf16(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                                                                      int L) {
  __shared__ __nv_bfloat16 t[32][34];
  const size_t base = (size_t)blockIdx.z * L * L;
  const int i0 = blockIdx.y * 32, j0 = blockIdx.x * 32;
#pragma unroll
  for (int k = 0; k < 32; k += 8) t[threadIdx.y + k][threadIdx.x] = src[base + (size_t)(i0 + threadIdx.y + k) * L + j0 + threadIdx.x];
  __syncthreads();
#pragma unroll
  for (int k = 0; k < 32; k += 8) dst[base + (size_t)(j0 + threadIdx.y + k) * L + i0 + threadIdx.x] = t[threadIdx.x][threadIdx.y + k];
}

// bias_transpose64_bf16 (2026-11-17): the same transpose in 64 x 64 tiles, 16-byte loads and stores, a persistent grid (``grid`` blocks walk
// the tiles): the AttentionPairBias B200 training forward runs it beside the attention core, which needs the SMs the 32 x 32 grid
// (L / 32)^2 H blocks took. The tile sits in shared memory as 32-bit words, row stride 33 words (bank-conflict free both ways).
extern "C" __global__ void __launch_bounds__(256) bias_transpose64_bf16(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                                                                        int L, int H) {
  __shared__ uint32_t t[64][33];                                       // t[r][w] = elements (r, 2 w), (r, 2 w + 1) of the tile
  const int nt = L / 64, tiles = nt * nt * H, tid = threadIdx.x;
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int h = tile / (nt * nt), i0 = (tile / nt) % nt * 64, j0 = tile % nt * 64;
    const size_t base = (size_t)h * L * L;
    uint4 v[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {                                         // 64 rows x 8 chunks of 8: thread p = tid + 256 k
      const int p = tid + 256 * k, r = p >> 3, c = (p & 7) * 8;
      v[k] = *reinterpret_cast<const uint4*>(src + base + (size_t)(i0 + r) * L + j0 + c);
    }
    __syncthreads();                                                      // the previous tile's reads of t are done
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int p = tid + 256 * k, r = p >> 3, w = (p & 7) * 4;
      t[r][w] = v[k].x; t[r][w + 1] = v[k].y; t[r][w + 2] = v[k].z; t[r][w + 3] = v[k].w;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 2; ++k) {                                         // output row j (a key), 8 queries from i = c
      const int p = tid + 256 * k, j = p >> 3, c = (p & 7) * 8;
      const int w = j >> 1, hi = j & 1;
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t a = t[c + 2 * e][w], b = t[c + 2 * e + 1][w];
        o[e] = hi ? __byte_perm(a, b, 0x7632) : __byte_perm(a, b, 0x5410);
      }
      *reinterpret_cast<uint4*>(dst + base + (size_t)(j0 + j) * L + i0 + c) = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
}

// one thread per (row, head): 12 float4 of dO and O; consecutive threads take consecutive heads of a row
extern "C" __global__ void __launch_bounds__(256) prep_do_bf16(const float* __restrict__ dO, const float* __restrict__ O,
                                                               __nv_bfloat16* __restrict__ dob, float* __restrict__ dd, int M, int L) {
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long)M * 16) return;
  const long r = t >> 4;
  const int h = (int)(t & 15);
  const float4* g = reinterpret_cast<const float4*>(dO + r * 768 + h * 48);
  const float4* o = reinterpret_cast<const float4*>(O + r * 768 + h * 48);
  __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(dob + r * 768 + h * 48);
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 12; ++k) {
    const float4 x = g[k], y = o[k];
    s += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
    b[2 * k] = __floats2bfloat162_rn(x.x, x.y);
    b[2 * k + 1] = __floats2bfloat162_rn(x.z, x.w);
  }
  dd[((r / L) * 16 + h) * L + r % L] = s;
}
