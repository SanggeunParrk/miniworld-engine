// tstat_w.cu (S90) -- the LayerNorm FOLD's pre-pass for the two-kernel Transition forward at D = 384 / 512 (inference, -DDIM=<D>): ONE launch with two kinds of blocks.
//   row blocks   (blockIdx.x < nrb): the statistics of every row of x: rstd and c1 = mean rstd (one warp half / full per row as tln_w.cu), x read once, NO xn written
//   prep blocks  (the rest): the gamma-folded weights Wp = bf16(W gamma) for [Wa ; Wb] (2 H rows) and, per weight row n, u_n = sum_k Wp_nk (of the rounded values) and
//                v_n = sum_k beta_k W_nk, so that  a_n = rstd (x . Wp_n) - c1 u_n + v_n  =  LN(x) . W_n  (the expand GEMM takes the raw x as its A operand, tswiglu_w.cu -DFOLD)
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef DIM
#define DIM 384
#endif
#ifndef LPR
#define LPR 16
#endif
constexpr int D_ = DIM, NCHK = D_ / 8, PER = NCHK / LPR, RPW = 32 / LPR;
static_assert(NCHK % LPR == 0, "whole chunks per lane");

__device__ __forceinline__ float lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ uint32_t pk(float a, float b) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }

extern "C" __global__ void __launch_bounds__(256)
transition_ln_stats_w(const __nv_bfloat16* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
                      const __nv_bfloat16* __restrict__ wa, const __nv_bfloat16* __restrict__ wb, __nv_bfloat16* __restrict__ wp,
                      float* __restrict__ u, float* __restrict__ v, float* __restrict__ rstd, float* __restrict__ c1, int M, int H, float eps, int nrb,
                      uint16_t* __restrict__ aug, int aug_on) {
  if (blockIdx.x < nrb) {
    const int lane = threadIdx.x % LPR;
    const int rpb = (blockDim.x >> 5) * RPW;
    for (int row = blockIdx.x * rpb + (threadIdx.x >> 5) * RPW + (threadIdx.x & 31) / LPR; row < M; row += nrb * rpb) {
      const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)row * D_);
      uint4 vv[PER];
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < PER; ++k) {
        vv[k] = __ldcs(xr + lane + LPR * k);
        s += ((lo(vv[k].x) + hi(vv[k].x)) + (lo(vv[k].y) + hi(vv[k].y))) + ((lo(vv[k].z) + hi(vv[k].z)) + (lo(vv[k].w) + hi(vv[k].w)));
      }
#pragma unroll
      for (int o = LPR / 2; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      const float mean = s * (1.f / D_);
      float q = 0.f;
#pragma unroll
      for (int k = 0; k < PER; ++k) {
        const uint32_t w[4] = {vv[k].x, vv[k].y, vv[k].z, vv[k].w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float d0 = lo(w[e]) - mean, d1 = hi(w[e]) - mean; q += d0 * d0 + d1 * d1; }
      }
#pragma unroll
      for (int o = LPR / 2; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
      const float rs = rsqrtf(q * (1.f / D_) + eps);
      if (lane == 0) { rstd[row] = rs; c1[row] = mean * rs; }
    }
  } else {
    const int pb = blockIdx.x - nrb, npb = gridDim.x - nrb, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int n = pb * 8 + warp; n < 2 * H; n += npb * 8) {
      const __nv_bfloat16* src = (n < H ? wa + (size_t)n * D_ : wb + (size_t)(n - H) * D_);
      uint4* dst = reinterpret_cast<uint4*>(wp + (size_t)n * D_);
      float su = 0.f, sv = 0.f;
      for (int c = lane; c < NCHK; c += 32) {
        const uint4 w = reinterpret_cast<const uint4*>(src)[c];
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        const float4 g0 = __ldg(reinterpret_cast<const float4*>(gamma) + 2 * c), g1 = __ldg(reinterpret_cast<const float4*>(gamma) + 2 * c + 1);
        const float4 b0 = __ldg(reinterpret_cast<const float4*>(beta) + 2 * c), b1 = __ldg(reinterpret_cast<const float4*>(beta) + 2 * c + 1);
        const float g[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w}, b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float w0 = lo(ww[e]), w1 = hi(ww[e]);
          o[e] = pk(w0 * g[2 * e], w1 * g[2 * e + 1]);
          su += lo(o[e]) + hi(o[e]);
          sv += w0 * b[2 * e] + w1 * b[2 * e + 1];
        }
        dst[c] = make_uint4(o[0], o[1], o[2], o[3]);
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) { su += __shfl_xor_sync(0xffffffffu, su, o); sv += __shfl_xor_sync(0xffffffffu, sv, o); }
      if (lane == 0) { u[n] = su; v[n] = sv; }
      if (aug_on && lane == 0) {
        // (S90, tfwd_d384.cu -DFOLD) the K16 correction operand of this weight row: [u_hi u_lo u_hi v_hi v_lo v_hi 0 0] (bf16 hi / lo pairs: -mean u and v / rstd are rank-1 terms of the expand's K loop),
        // laid out as the chunk image the kernel bulk-copies: image (j, q) = 64 rows [32 Wa | 32 Wb] of the CTA rank q's 32 units of chunk j (64 units), 16 bytes a row
        const int unit = n < H ? n : n - H, j = unit >> 6, w = unit & 63, q = w >> 5, i = (n < H ? 0 : 32) + (w & 31);
        auto hi_lo = [](float f, uint16_t& h, uint16_t& l) {
          const __nv_bfloat16 bh = __float2bfloat16(f);
          const __nv_bfloat16 bl = __float2bfloat16(f - __bfloat162float(bh));
          h = *reinterpret_cast<const uint16_t*>(&bh); l = *reinterpret_cast<const uint16_t*>(&bl);
        };
        uint16_t uh, ul, vh, vl;
        hi_lo(su, uh, ul); hi_lo(sv, vh, vl);
        const uint16_t e[8] = {uh, ul, uh, vh, vl, vh, 0, 0};
        uint4 pk8 = make_uint4(e[0] | ((uint32_t)e[1] << 16), e[2] | ((uint32_t)e[3] << 16), e[4] | ((uint32_t)e[5] << 16), 0u);
        *reinterpret_cast<uint4*>(aug + (size_t)((j * 2 + q) * 64 + i) * 8) = pk8;
      }
    }
  }
}
