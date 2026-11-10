// adaln_finish.cuh -- the closing pass of a backward, ONE launch for all of it: the column sums of the per-block partial rows (d sb, d w, d b: [blocks, n] fp32 -> a vector in the parameter's
// dtype) and the casts of the weight gradients (fp32 GEMM outputs, any strides -> a fresh contiguous matrix in the parameter's dtype).  Before it a backward closed with a dozen small torch
// kernels (a column reduce_kernel at ~30 us a call, copy kernels at 2-5 us): 70-130 us of a step that the weight-gradient GEMMs and the row passes do not need.  Fixed order everywhere
// (rows are dealt to sixteen warps round-robin, the sixteen partial sums added in order): bit-reproducible, no atomics.
#pragma once
#include "adaln_common.cuh"

namespace adl {

constexpr int FIN_JOBS = 16, FIN_THREADS = 512, FIN_WARPS = FIN_THREADS / 32, FIN_SUM_COLS = 32, FIN_CAST_ELEMS = FIN_THREADS * 4;

struct FinJob {
  const float* src;
  void* dst;
  long n;                      // sum: columns; cast: elements
  long rows;                   // sum: partial rows; cast: matrix rows (columns = n / rows)
  long sr, sc;                 // cast: element strides of the source's [rows, columns]
  long tile0;                  // the first block of this job
  int kind;                    // 0: column sums, 1: cast
  int bf16;                    // the destination is bf16 (else fp32)
  int vec;                     // cast: contiguous rows of a multiple of 4 columns, 16-byte aligned source: 4 elements per load
};
struct FinParams {
  FinJob job[FIN_JOBS];
  int njob;
};

ADL_DEVI void fin_store(const FinJob& jb, long i, float v) {
  if (jb.bf16) reinterpret_cast<bf*>(jb.dst)[i] = __float2bfloat16_rn(v);
  else reinterpret_cast<float*>(jb.dst)[i] = v;
}

__global__ void __launch_bounds__(FIN_THREADS) finish_kernel(const FinParams p) {
  __shared__ float red[FIN_WARPS][FIN_SUM_COLS];
  int j = 0;
  while (j + 1 < p.njob && (long)blockIdx.x >= p.job[j + 1].tile0) ++j;
  const FinJob jb = p.job[j];
  const long tile = (long)blockIdx.x - jb.tile0;
  const int tid = threadIdx.x;
  if (jb.kind == 0) {
    const int lane = tid & 31, w = tid >> 5;
    const long col = tile * FIN_SUM_COLS + lane;
    float s = 0.f;
    if (col < jb.n) {
#pragma unroll 8
      for (long r = w; r < jb.rows; r += FIN_WARPS) s += jb.src[r * jb.n + col];
    }
    red[w][lane] = s;
    __syncthreads();
    if (w == 0 && col < jb.n) {
      float t = red[0][lane];
#pragma unroll
      for (int i = 1; i < FIN_WARPS; ++i) t += red[i][lane];
      fin_store(jb, col, t);
    }
    return;
  }
  const long e0 = tile * FIN_CAST_ELEMS + (long)tid * 4, cols = jb.n / jb.rows;
  if (e0 >= jb.n) return;
  if (jb.vec) {
    const long r = e0 / cols, c = e0 - r * cols;
    const float4 v = *reinterpret_cast<const float4*>(jb.src + r * jb.sr + c);
    if (jb.bf16) {
      const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
      uint2 u;
      u.x = *reinterpret_cast<const unsigned*>(&a);
      u.y = *reinterpret_cast<const unsigned*>(&b);
      *reinterpret_cast<uint2*>(reinterpret_cast<bf*>(jb.dst) + e0) = u;
    } else {
      *reinterpret_cast<float4*>(reinterpret_cast<float*>(jb.dst) + e0) = v;
    }
    return;
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const long e = e0 + k;
    if (e < jb.n) {
      const long r = e / cols, c = e - r * cols;
      fin_store(jb, e, jb.src[r * jb.sr + c * jb.sc]);
    }
  }
}

}  // namespace adl
