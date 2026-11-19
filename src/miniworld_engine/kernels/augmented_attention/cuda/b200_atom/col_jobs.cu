// col_jobs.cu — the small gradients of the B200 variant block's backward in ONE launch (2026-11-17; sm_100a, CUDA cores): up to 16 jobs, each the
// column sums of a few fp32 partial rows into a fresh output of the parameter's dtype,
//   dst[c] = sum_{r < rows} src[r * pitch + c],  c < n     (the rows in order: deterministic; rows = 1: a copy / cast)
// It replaces a reduction per partial buffer and a slice + clone (+ cast) per bias / LayerNorm-weight gradient (~14 launches of a step).
// One CTA per job. The job table is a by-value parameter read with CONSTANT indices (a compare chain over the 16 slots): indexed at run
// time the compiler copies the whole table into every thread's local memory (512 bytes per thread; measured 10 -> 6 - 8 us at 1024 atoms).
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cuda_bf16.h>

struct Job {
  const float* src;
  void* dst;
  int32_t rows, pitch, n, bf16;
};
struct Jobs {
  Job j[16];
};

extern "C" __global__ void __launch_bounds__(256) atom_col_jobs(const Jobs J) {
  const float* src = nullptr;
  void* dst = nullptr;
  int rows = 0, pitch = 0, n = 0, bf16 = 0;
#pragma unroll
  for (int k = 0; k < 16; ++k)
    if ((int)blockIdx.x == k) { src = J.j[k].src; dst = J.j[k].dst; rows = J.j[k].rows; pitch = J.j[k].pitch; n = J.j[k].n; bf16 = J.j[k].bf16; }
  for (int c = threadIdx.x; c < n; c += 256) {
    float v = 0.f;
    for (int r0 = 0; r0 < rows; r0 += 16) {                                // 16 loads issued together, added in the rows' order
      float b[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) b[k] = r0 + k < rows ? src[(size_t)(r0 + k) * pitch + c] : 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) v += b[k];
    }
    if (bf16)
      reinterpret_cast<__nv_bfloat16*>(dst)[c] = __float2bfloat16_rn(v);
    else
      reinterpret_cast<float*>(dst)[c] = v;
  }
}
