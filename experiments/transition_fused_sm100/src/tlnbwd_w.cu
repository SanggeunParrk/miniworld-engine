// tlnbwd_w.cu — the LayerNorm backward + residual of the wide-width Transition backward (D = 256 / 384 / 512; -DDIM=<D>), the tbwd.cu
// contract from the saved statistics:  xhat = (x - mean) rstd, w = gamma d_xn,  ca = mean(xhat w), cb = mean(w),
//   dx = bf16(bf16((w - xhat ca - cb) rstd) + dy);   dgamma, dbeta partials = sum over this block's rows of d_xn xhat, d_xn.
// 16 lanes per row (two rows per warp), D / 128 16-byte chunks per lane, all three inputs loaded up front; per-lane column partials of
// dgamma / dbeta kept across the block's rows and written once per block (fixed order: a second kernel sums the blocks, so a replay is
// bit-identical). SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef DIM
#define DIM 384
#endif
#ifndef LPR
#define LPR 16                                                 // lanes per row: 16 (two rows per warp) or 32
#endif
constexpr int D_ = DIM, NCHK = D_ / 8, PER = NCHK / LPR, WPB = 8, RPW = 32 / LPR, RPB = RPW * WPB;
static_assert(NCHK % LPR == 0, "whole chunks per lane");

__device__ __forceinline__ float lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ uint32_t pk(float a, float b) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ uint4 ldnc(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}

#ifndef LNB_MINB
#define LNB_MINB 2
#endif
// part: [gridDim.x][2 D] (dgamma | dbeta) per block
extern "C" __global__ void __launch_bounds__(WPB * 32, LNB_MINB)
transition_lnbwd_w(const __nv_bfloat16* __restrict__ dxn, const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ dy,
                   const float* __restrict__ rstd, const float* __restrict__ c1, const float* __restrict__ gamma,
                   __nv_bfloat16* __restrict__ dx, float* __restrict__ part, int M) {
  __shared__ float red[WPB][2 * D_];
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, sub = lane / LPR, l16 = lane % LPR;
  float ag[PER][8], ab[PER][8], g[PER][8];
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int c = l16 + LPR * k;
    const float4 g0 = __ldg(reinterpret_cast<const float4*>(gamma) + 2 * c), g1 = __ldg(reinterpret_cast<const float4*>(gamma) + 2 * c + 1);
    g[k][0] = g0.x; g[k][1] = g0.y; g[k][2] = g0.z; g[k][3] = g0.w; g[k][4] = g1.x; g[k][5] = g1.y; g[k][6] = g1.z; g[k][7] = g1.w;
#pragma unroll
    for (int e = 0; e < 8; ++e) { ag[k][e] = 0.f; ab[k][e] = 0.f; }
  }
  for (int row = blockIdx.x * RPB + wi * RPW + sub; row < M; row += gridDim.x * RPB) {
    const uint4* nr = reinterpret_cast<const uint4*>(dxn + (size_t)row * D_);
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)row * D_);
    const uint4* dr = reinterpret_cast<const uint4*>(dy + (size_t)row * D_);
    uint4 nv[PER], xv[PER], dv[PER];
#pragma unroll
    for (int k = 0; k < PER; ++k) { nv[k] = ldnc(nr + l16 + LPR * k); xv[k] = ldnc(xr + l16 + LPR * k); dv[k] = ldnc(dr + l16 + LPR * k); }
    const float rs = __ldg(rstd + row), mean = __ldg(c1 + row) / rs;
    float pa = 0.f, pb = 0.f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
      const uint32_t n4[4] = {nv[k].x, nv[k].y, nv[k].z, nv[k].w}, x4[4] = {xv[k].x, xv[k].y, xv[k].z, xv[k].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float n0 = lo(n4[e]), n1 = hi(n4[e]);
        const float x0 = (lo(x4[e]) - mean) * rs, x1 = (hi(x4[e]) - mean) * rs;
        const float w0 = g[k][2 * e] * n0, w1 = g[k][2 * e + 1] * n1;
        pa += x0 * w0 + x1 * w1; pb += w0 + w1;
        ag[k][2 * e] += n0 * x0; ag[k][2 * e + 1] += n1 * x1; ab[k][2 * e] += n0; ab[k][2 * e + 1] += n1;
      }
    }
#pragma unroll
    for (int o = LPR / 2; o; o >>= 1) { pa += __shfl_xor_sync(0xffffffffu, pa, o); pb += __shfl_xor_sync(0xffffffffu, pb, o); }
    const float ca = pa * (1.f / D_), cb = pb * (1.f / D_);
    uint4* outr = reinterpret_cast<uint4*>(dx + (size_t)row * D_);
#pragma unroll
    for (int k = 0; k < PER; ++k) {
      const uint32_t n4[4] = {nv[k].x, nv[k].y, nv[k].z, nv[k].w}, x4[4] = {xv[k].x, xv[k].y, xv[k].z, xv[k].w};
      const uint32_t d4[4] = {dv[k].x, dv[k].y, dv[k].z, dv[k].w};
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x0 = (lo(x4[e]) - mean) * rs, x1 = (hi(x4[e]) - mean) * rs;
        const float w0 = g[k][2 * e] * lo(n4[e]), w1 = g[k][2 * e + 1] * hi(n4[e]);
        const uint32_t t = pk((w0 - (x0 * ca + cb)) * rs, (w1 - (x1 * ca + cb)) * rs);
        o[e] = pk(lo(t) + lo(d4[e]), hi(t) + hi(d4[e]));
      }
      outr[l16 + LPR * k] = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
  // block partials: the two half-warps of each warp combined by shuffle, then the warps through shared memory, in a fixed order
  if (LPR == 16) {
#pragma unroll
    for (int k = 0; k < PER; ++k)
#pragma unroll
      for (int e = 0; e < 8; ++e) { ag[k][e] += __shfl_xor_sync(0xffffffffu, ag[k][e], 16); ab[k][e] += __shfl_xor_sync(0xffffffffu, ab[k][e], 16); }
  }
  if (sub == 0) {
#pragma unroll
    for (int k = 0; k < PER; ++k)
#pragma unroll
      for (int e = 0; e < 8; ++e) { const int col = (l16 + LPR * k) * 8 + e; red[wi][col] = ag[k][e]; red[wi][D_ + col] = ab[k][e]; }
  }
  __syncthreads();
  for (int col = threadIdx.x; col < 2 * D_; col += blockDim.x) {
    float v = 0.f;
#pragma unroll
    for (int j = 0; j < WPB; ++j) v += red[j][col];
    part[(size_t)blockIdx.x * 2 * D_ + col] = v;
  }
}

// dgamma | dbeta = sum over the blocks' partials (fixed order)
extern "C" __global__ void transition_lnbwd_w_reduce(const float* __restrict__ part, float* __restrict__ dgb, int nblk) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= 2 * D_) return;
  float v = 0.f;
  for (int b = 0; b < nblk; ++b) v += part[(size_t)b * 2 * D_ + col];
  dgb[col] = v;
}
