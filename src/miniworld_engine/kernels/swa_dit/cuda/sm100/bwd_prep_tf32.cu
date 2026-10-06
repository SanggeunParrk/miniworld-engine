// bwd_prep_tf32.cu — weight preparation for the fp32 SWA atom block's sm_100a backward (tf32_bwd.py): dst = rna_tf32(src) or its
// transpose, written at a row stride `ldd` (so [Wqkv; Wg]^T is two launches into one [128][512] buffer).
// The TF32 MMAs read their shared-memory operands as stored and drop the low 13 mantissa bits (truncation: a bias of -2^-11 on
// average per operand); cuBLAS TF32 rounds. Every weight operand of the backward kernels is therefore rounded to nearest here, once per
// weight version (cached by the host), and the kernels round the operands their threads produce (dffn, datt, h / da / db, dpq / dpk,
// P / dS, dO) the same way (cvt.rna.tf32.f32).
// 32 x 32 tiles through shared memory, block (32, 8), grid (ceil(cols / 32), ceil(rows / 32)).
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>

__device__ __forceinline__ float rna_tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return __uint_as_float(r & 0xffffe000u);
}

extern "C" __global__ void __launch_bounds__(256)
swa_tf32_prep(const float* __restrict__ src, float* __restrict__ dst, int rows, int cols, int ldd, int transpose) {
  __shared__ float t[32][33];
  const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;
  if (!transpose) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int r = r0 + ty + 8 * k, c = c0 + tx;
      if (r < rows && c < cols) dst[(size_t)r * ldd + c] = rna_tf32(src[(size_t)r * cols + c]);
    }
    return;
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {                                            // coalesced read of src rows r0 ..
    const int r = r0 + ty + 8 * k, c = c0 + tx;
    t[ty + 8 * k][tx] = (r < rows && c < cols) ? src[(size_t)r * cols + c] : 0.f;
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < 4; ++k) {                                            // coalesced write of dst rows c0 .. (src columns)
    const int c = c0 + ty + 8 * k, r = r0 + tx;
    if (c < cols && r < rows) dst[(size_t)c * ldd + r] = rna_tf32(t[tx][ty + 8 * k]);
  }
}
