// bo_rows.cuh -- the row machinery the bias-only DiT row kernels share (bias_only_dit_rows.cu, bias_only_dit_train_rows.cu bf16 and
// bias_only_dit_f32_rows.cu fp32): the fast-division sigmoid, the pairwise 4-sum, one-warp row statistics, and the D = 768 two-warp
// rows (ROWS2_BEGIN / row_sum2 / row_stats2 / block partials). ROWS_BEGIN expects the including file's WPB (rows per block).
#pragma once
#include "../../conditioned_transition/cuda/token_dit_common.cuh"

namespace bor {
using namespace tdr;

constexpr int DW = 768, RPB2 = 4;   // the two-warp rows: width 768, four rows per block at a time

__device__ __forceinline__ float sg(float v) { return __fdividef(1.f, 1.f + __expf(-v)); }
__device__ __forceinline__ float sum4p(float4 a) { return (a.x + a.y) + (a.z + a.w); }   // pairwise (tdr::sum4 is left to right)

// the warp's first row, the stride, and this lane's column of chunk j
#define ROWS_BEGIN                                                                                   \
  const int lane = threadIdx.x % 32;                                                                 \
  const long wid = (long)blockIdx.x * WPB + threadIdx.x / 32, nw = (long)gridDim.x * WPB;
#define COL(j) (((j) * 32 + lane) * 4)

// the row's mean and rstd from the N float4 chunks a lane holds (one warp per row)
template <int N>
__device__ __forceinline__ void row_stats(const float4 (&x)[N], float eps, float& mean, float& rstd) {
  constexpr float inv = 1.f / (N * 128);
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < N; ++j) s += sum4p(x[j]);
  mean = warp_sum(s) * inv;
  float q = 0.f;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    const float4 d = make_float4(x[j].x - mean, x[j].y - mean, x[j].z - mean, x[j].w - mean);
    q += sum4p(mul4(d, d));
  }
  rstd = rsqrtf(warp_sum(q) * inv + eps);
}

// ---- D-wide rows: two warps per row (64 threads; lane l of warp half hw owns chunks hw * 3 + j, j < 3, of the six), four rows per
// block at a time. A thread holds 12 of the row's 768 values per tensor, so the backward kernels fit their live state and column
// sums in registers without spilling (one warp per row needed ~200). The row's sums combine the two warps through shared memory
// under a named barrier of the pair (ids 1..4), double-buffered by call parity.
#define ROWS2_BEGIN                                                                                  \
  const int lane = threadIdx.x % 32, hw = (threadIdx.x / 32) & 1, slot = threadIdx.x / 64;           \
  const long sid = (long)blockIdx.x * RPB2 + slot, ns = (long)gridDim.x * RPB2;                     \
  __shared__ float2 xr_[2][RPB2][2];                                                                 \
  int par_ = 0;
#define COL2(j) ((((hw) * 3 + (j)) * 32 + lane) * 4)
#define ROWSUM2(a, b) row_sum2(make_float2((a), (b)), xr_, par_, slot, hw, lane)

__device__ __forceinline__ float2 row_sum2(float2 v, float2 (&xr)[2][RPB2][2], int& par, int slot, int hw, int lane) {
  v.x = warp_sum(v.x); v.y = warp_sum(v.y);
  if (lane == 0) xr[par][slot][hw] = v;
  asm volatile("bar.sync %0, 64;" :: "r"(1 + slot) : "memory");
  const float2 a = xr[par][slot][0], b = xr[par][slot][1];
  par ^= 1;
  return make_float2(a.x + b.x, a.y + b.y);
}

// LayerNorm statistics of a 768-row held as 12 values per thread of the slot's two warps, in ONE exchange: each thread's (mean,
// sum of squared deviations) over its values, merged pairwise (Chan et al.; equal counts at every level, so exact and stable)
// over the warp by shuffles and over the two warps through shared memory. Returns (mean, rstd).
__device__ __forceinline__ float2 row_stats2(const float4 (&x)[3], float eps, float2 (&xr)[2][RPB2][2], int& par, int slot, int hw,
                                             int lane) {
  float m = (sum4p(x[0]) + sum4p(x[1]) + sum4p(x[2])) * (1.f / 12);
  float q = 0.f;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const float4 d = make_float4(x[j].x - m, x[j].y - m, x[j].z - m, x[j].w - m);
    q += sum4p(mul4(d, d));
  }
  float n = 12.f;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float mo = __shfl_xor_sync(0xffffffffu, m, o), qo = __shfl_xor_sync(0xffffffffu, q, o), dl = mo - m;
    q = q + qo + dl * dl * (0.5f * n);                 // nA nB / (nA + nB) with nA = nB = n
    m = 0.5f * (m + mo);
    n *= 2.f;
  }
  if (lane == 0) xr[par][slot][hw] = make_float2(m, q);
  asm volatile("bar.sync %0, 64;" :: "r"(1 + slot) : "memory");
  const float2 a = xr[par][slot][0], b = xr[par][slot][1];
  par ^= 1;
  const float dl = b.x - a.x;
  return make_float2(0.5f * (a.x + b.x), rsqrtf((a.y + b.y + dl * dl * (0.5f * n)) * (1.f / DW) + eps));
}

// the attention residual x1 = x + sigmoid(g1) y, rebuilt wherever it is needed instead of kept in fp32 (one fused multiply-add,
// so every kernel gets the same bits and the saved LayerNorm statistics stay exact for it)
__device__ __forceinline__ float4 resid4(float4 x, float4 sg1, float4 y) {
  return make_float4(__fmaf_rn(sg1.x, y.x, x.x), __fmaf_rn(sg1.y, y.y, x.y), __fmaf_rn(sg1.z, y.z, x.z), __fmaf_rn(sg1.w, y.w, x.w));
}

// the block's four row slots add their column sums in shared memory; the block leaves one row of the partial buffer
__device__ __forceinline__ void block_partial2(const float4 (&acc)[3], float (&red)[RPB2][DW], float* part) {
  const int lane = threadIdx.x % 32, hw = (threadIdx.x / 32) & 1, slot = threadIdx.x / 64;
#pragma unroll
  for (int j = 0; j < 3; ++j) V4<float>::store(&red[slot][COL2(j)], acc[j]);
  __syncthreads();
  for (int c = threadIdx.x; c < DW; c += blockDim.x) part[(long)blockIdx.x * DW + c] = (red[0][c] + red[1][c]) + (red[2][c] + red[3][c]);
  __syncthreads();
}

__device__ __forceinline__ void block_partial_smem(float4 (&acc)[RPB2][DW / 4], float* part) {
  __syncthreads();
  for (int c = threadIdx.x; c < DW / 4; c += blockDim.x)
    V4<float>::store(part + (long)blockIdx.x * DW + 4 * c, add4(add4(acc[0][c], acc[1][c]), add4(acc[2][c], acc[3][c])));
}
}  // namespace bor
