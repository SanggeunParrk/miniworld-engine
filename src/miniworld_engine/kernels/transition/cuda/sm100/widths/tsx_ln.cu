// tsx_ln.cu -- the Transition forward at SMALL M, first of three programmatic-dependent kernels: the input LayerNorm
// (-DDIM=<D>), sm_100a, bf16 in / bf16 out with fp32 statistics. One warp per row (D / 128 eight-byte chunks per lane, coalesced),
// two-pass statistics from registers, xn = bf16((x rs - mean rs) gamma + beta) -- the arithmetic of the other builds' LayerNorm
// (tfwd_d384.cu: fp32x2, the same association). Writes xn always (the expand's A operand) and, with save, rstd and
// c1 = mean rstd for the backward. Normalising once per tile here instead of in every (tile, hidden chunk) CTA of the expand takes the
// LayerNorm (instruction bound: ~4k cycles per 128 x 384 tile) off the expand's critical path.
// With ncast > 0 the blocks past the rows also cast three fp32 master matrices (ncast elements each) to the kernels' bf16 copies
// (round to nearest even, as .to): the training forward's per-step weight cast rides in this launch instead of one of its own.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DIM
#define DIM 384
#endif
constexpr int D_ = DIM, PER = D_ / 128, WPB = 8;               // eight-byte chunks per lane, warps (rows) per block
static_assert(D_ % 128 == 0, "whole 4-column chunks per lane");

DEVI f2 unpack_bf16x2(uint32_t w) { return mk2u(w << 16, w & 0xffff0000u); }

extern "C" __global__ void __launch_bounds__(WPB * 32)
transition_sx_ln(const __nv_bfloat16* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
                 __nv_bfloat16* __restrict__ xn, float* __restrict__ rstd, float* __restrict__ c1, int M, float eps, int save,
                 const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2, __nv_bfloat16* __restrict__ o0,
                 __nv_bfloat16* __restrict__ o1, __nv_bfloat16* __restrict__ o2, int ncast) {
  const int lane = threadIdx.x & 31, nrb = (M + WPB - 1) / WPB, row = blockIdx.x * WPB + (threadIdx.x >> 5);
  pdl_wait();
  pdl_launch();
  if ((int)blockIdx.x >= nrb) {                                // weight casts: two 8-element groups per thread, loads first
    const int n8 = ncast / 8, i0 = ((blockIdx.x - nrb) * blockDim.x + threadIdx.x) * 2;
    float4 v[4];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int i = i0 + u, w = i / n8, j = i - w * n8;
      if (i < 3 * n8) {
        const float4* src = reinterpret_cast<const float4*>(w == 0 ? w0 : w == 1 ? w1 : w2) + 2 * j;
        v[2 * u] = __ldg(src); v[2 * u + 1] = __ldg(src + 1);
      }
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int i = i0 + u, w = i / n8, j = i - w * n8;
      if (i < 3 * n8) {
        const float4 a = v[2 * u], b = v[2 * u + 1];
        reinterpret_cast<uint4*>(w == 0 ? o0 : w == 1 ? o1 : o2)[j] =
            make_uint4(pack_bf16(a.x, a.y), pack_bf16(a.z, a.w), pack_bf16(b.x, b.y), pack_bf16(b.z, b.w));
      }
    }
    return;
  }
  if (row >= M) return;
  const uint2* xr = reinterpret_cast<const uint2*>(x + (size_t)row * D_);
  uint2 v[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) v[i] = xr[lane + 32 * i];
  f2 s2 = mk2(0.f, 0.f);
#pragma unroll
  for (int i = 0; i < PER; ++i) s2 = add2(s2, add2(unpack_bf16x2(v[i].x), unpack_bf16x2(v[i].y)));
  float s = lo2(s2) + hi2(s2);
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = s * (1.f / D_);
  const f2 nm2 = mk2(-mean, -mean);
  f2 q2 = mk2(0.f, 0.f);
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const f2 d0 = add2(unpack_bf16x2(v[i].x), nm2), d1 = add2(unpack_bf16x2(v[i].y), nm2);
    q2 = fma2(d0, d0, q2);
    q2 = fma2(d1, d1, q2);
  }
  float q = lo2(q2) + hi2(q2);
#pragma unroll
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
  const float rs = rsqrtf(q * (1.f / D_) + eps), cc = mean * rs;
  const f2 rs2 = mk2(rs, rs), ncc2 = mk2(-cc, -cc);
  uint2* outr = reinterpret_cast<uint2*>(xn + (size_t)row * D_);
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int col = (lane + 32 * i) * 4;
    const float4 g = __ldg(reinterpret_cast<const float4*>(gamma + col)), b = __ldg(reinterpret_cast<const float4*>(beta + col));
    const f2 g0 = mk2(g.x, g.y), g1 = mk2(g.z, g.w), b0 = mk2(b.x, b.y), b1 = mk2(b.z, b.w);
    const f2 y0 = fma2(unpack_bf16x2(v[i].x), mul2(rs2, g0), fma2(ncc2, g0, b0));   // (x rs - mean rs) g + b
    const f2 y1 = fma2(unpack_bf16x2(v[i].y), mul2(rs2, g1), fma2(ncc2, g1, b1));
    outr[lane + 32 * i] = make_uint2(pack_bf16(lo2(y0), hi2(y0)), pack_bf16(lo2(y1), hi2(y1)));
  }
  if (save && lane == 0) { rstd[row] = rs; c1[row] = cc; }
}
