// tr_bwd_fused_sm80.cuh -- the Transition backward as ONE launch (the H100 layout): every CTA first normalises its share of the rows
// (xn in bf16 with the forward's arithmetic, and (mean, rstd)), a grid barrier, then CTAs 0 .. NDX-1 run role DX (tr_bwd_dx_sm80.cuh:
// dx, dgamma, dbeta) and the rest role DW (tr_bwd_dw_sm80.cuh: 8 hidden slices x (grid - NDX) / 8 row replicas: dWa, dWb, dWs).
// The roles never communicate: both recompute a, b, dh (22 M D H in total).  The grid (one CTA per SM, all resident) is required by the
// barrier.
#pragma once
#include "tr_bwd_dx_sm80.cuh"
#include "tr_bwd_dw_sm80.cuh"

namespace a100 {

struct FusedParams {
  DXParams dx;
  DWParams dw;
  const float* beta;
  __nv_bfloat16* xn;           // [T][128] written by the prologue
  float2* stats;               // [T]
  unsigned int* barrier;       // zeroed before the launch
  int ndx;                     // CTAs of role DX
  float eps;
};

constexpr int FUSED_SMEM = CfgDW::SMEM > CfgDX::SMEM ? CfgDW::SMEM : CfgDX::SMEM;

__global__ void __launch_bounds__(256, 1) tr_bwd_fused_kernel(const FusedParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, b = blockIdx.x, grid = gridDim.x;
  const int T = p.dx.T;
  // ---- prologue: rows b, b + grid, ... one per warp iteration, 4 elements per lane
  {
    const float* gam = p.dx.gamma;
    const int col = lane * 4;
    const float g0 = gam[col], g1 = gam[col + 1], g2 = gam[col + 2], g3 = gam[col + 3];
    const float b0 = p.beta[col], b1 = p.beta[col + 1], b2 = p.beta[col + 2], b3 = p.beta[col + 3];
#pragma unroll 1
    for (int r = b * 8 + warp; r < T; r += grid * 8) {
      const uint2 v = *reinterpret_cast<const uint2*>(p.dx.x + (size_t)r * 128 + col);
      float xv[4] = {bf16lo(v.x), bf16hi(v.x), bf16lo(v.y), bf16hi(v.y)};
      const float mean = warp_sum(xv[0] + xv[1] + xv[2] + xv[3]) * (1.f / 128);
      float sq = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) { xv[e] -= mean; sq = fmaf(xv[e], xv[e], sq); }
      const float rstd = rsqrtf(warp_sum(sq) * (1.f / 128) + p.eps);
      *reinterpret_cast<uint2*>(p.xn + (size_t)r * 128 + col) =
          make_uint2(pack_bf16(fmaf(xv[0] * rstd, g0, b0), fmaf(xv[1] * rstd, g1, b1)), pack_bf16(fmaf(xv[2] * rstd, g2, b2), fmaf(xv[3] * rstd, g3, b3)));
      if (lane == 0) p.stats[r] = make_float2(mean, rstd);
    }
  }
  // ---- grid barrier (all CTAs resident)
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    atomicAdd(p.barrier, 1u);
    while (*reinterpret_cast<volatile unsigned int*>(p.barrier) < (unsigned)grid) __nanosleep(100);
    __threadfence();
  }
  __syncthreads();
  if (b < p.ndx) dx_role(p.dx, smem, b, p.ndx);
  else {
    const int k = b - p.ndx;
    dw_role(p.dw, smem, k & 7, k >> 3, (grid - p.ndx) >> 3);
  }
}

}  // namespace a100
