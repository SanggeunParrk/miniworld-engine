// attn_dd.cu — the atom DiT attention backward's row term D[a, h, i] = sum_d dO[a, i, 32 h + d] O[a, i, 32 h + d] (fp32 [A, 4, N]) from
// bf16 dO and O [A N, 128]; one (row, head) per thread (four threads read a row's 256 B), fp32 products and sum.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

__device__ __forceinline__ float lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

extern "C" __global__ void __launch_bounds__(256) atom_dd(const uint4* __restrict__ dO, const uint4* __restrict__ O, float* __restrict__ D,
                                                         int N, int rows) {
  const int t = blockIdx.x * 256 + threadIdx.x;
  if (t >= 4 * rows) return;
  const int row = t >> 2, h = t & 3;                                      // row = a N + i
  const uint4* p = dO + (size_t)row * 16 + h * 4;
  const uint4* o = O + (size_t)row * 16 + h * 4;
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint4 x = __ldg(p + k), y = __ldg(o + k);
    s += lo(x.x) * lo(y.x) + hi(x.x) * hi(y.x) + lo(x.y) * lo(y.y) + hi(x.y) * hi(y.y)
       + lo(x.z) * lo(y.z) + hi(x.z) * hi(y.z) + lo(x.w) * lo(y.w) + hi(x.w) * hi(y.w);
  }
  const int a = row / N, i = row - a * N;
  D[((size_t)a * 4 + h) * N + i] = s;
}
