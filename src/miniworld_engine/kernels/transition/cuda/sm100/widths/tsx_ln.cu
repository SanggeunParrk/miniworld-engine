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
#include "tsx_store.cuh"
using namespace s100;

#ifndef DIM
#define DIM 384
#endif
constexpr int D_ = DIM, PER = D_ / 128, WPB = 8;               // eight-byte chunks per lane, warps (rows) per block
static_assert(D_ % 128 == 0, "whole 4-column chunks per lane");
#ifndef F32
#define F32 0
#endif
#ifndef H16
#define H16 0
#endif
#ifndef LNPF
#define LNPF (H16 || !F32)                                          // gamma / beta (and sc) loaded before griddepcontrol.wait (part 16)
#endif

#if H16
// -DH16=1 (round 3 part 4): fp32 activations on the fp16 tensor cores (inference). x fp32 in; xn = fp16(LN(x) / s_x) out (sc[0] = 1 / s_x,
// fused_generic_sm100a._h16_forms): fp16 has TF32's 10-bit mantissa and rounds to nearest, at half the operand bytes of the TF32 chain.
// The statistics are the F32 build's (tg_ln_fwd's order and arithmetic).
constexpr int PER4 = D_ / 128;                                 // 16-byte chunks (4 fp32) per lane
extern "C" __global__ void __launch_bounds__(WPB * 32)
transition_sx_ln(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta, uint2* __restrict__ xn,
                 const float* __restrict__ sc, int M, float eps) {
  const int lane = threadIdx.x & 31, row = blockIdx.x * WPB + (threadIdx.x >> 5);
#if LNPF
  float4 gv[PER4], bv[PER4];                                   // (part 16) parameters: loaded before griddepcontrol.wait
#pragma unroll
  for (int i = 0; i < PER4; ++i) {
    gv[i] = __ldg(reinterpret_cast<const float4*>(gamma) + lane + 32 * i);
    bv[i] = __ldg(reinterpret_cast<const float4*>(beta) + lane + 32 * i);
  }
  const float s0 = __ldg(sc);
#endif
  pdl_wait();
  pdl_launch();
  if (row >= M) return;
  const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * D_);
  float4 v[PER4];
#pragma unroll
  for (int i = 0; i < PER4; ++i) v[i] = xr[lane + 32 * i];
#if !LNPF
  const float s0 = __ldg(sc);
#endif
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < PER4; ++i) { s += v[i].x; s += v[i].y; s += v[i].z; s += v[i].w; }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = s / (float)D_;
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < PER4; ++i) {
    const float a = v[i].x - mean, b = v[i].y - mean, c = v[i].z - mean, e = v[i].w - mean;
    q += a * a; q += b * b; q += c * c; q += e * e;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
  const float rs = rsqrtf(q / (float)D_ + eps);
  uint2* outr = xn + (size_t)row * (D_ / 4);
#pragma unroll
  for (int i = 0; i < PER4; ++i) {
#if LNPF
    const float4 g = gv[i], b = bv[i];
#else
    const int col = (lane + 32 * i) * 4;
    const float4 g = __ldg(reinterpret_cast<const float4*>(gamma + col)), b = __ldg(reinterpret_cast<const float4*>(beta + col));
#endif
    outr[lane + 32 * i] = make_uint2(f2h2(((v[i].x - mean) * rs * g.x + b.x) * s0, ((v[i].y - mean) * rs * g.y + b.y) * s0),
                                     f2h2(((v[i].z - mean) * rs * g.z + b.z) * s0, ((v[i].w - mean) * rs * g.w + b.w) * s0));
  }
}
#elif F32
// -DF32=1: fp32 activations (the TF32 chain). x fp32 in, xn fp32 out on the TF32 grid (cvt.rna: the expand's kind::tf32 products would
// truncate it); the cast blocks round three fp32 matrices to the TF32 grid instead (the training forward's per-step weight forms).
DEVI float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
constexpr int PER4 = D_ / 128;                                 // 16-byte chunks (4 fp32) per lane
extern "C" __global__ void __launch_bounds__(WPB * 32)
transition_sx_ln(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
                 float* __restrict__ xn, float* __restrict__ rstd, float* __restrict__ c1, int M, float eps, int save,
                 const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2, float* __restrict__ o0,
                 float* __restrict__ o1, float* __restrict__ o2, int ncast) {
  const int lane = threadIdx.x & 31, nrb = (M + WPB - 1) / WPB, row = blockIdx.x * WPB + (threadIdx.x >> 5);
  pdl_wait();
  pdl_launch();
  if ((int)blockIdx.x >= nrb) {                                // weight rounding: four 4-element groups per thread, loads first
    const int n4 = ncast / 4, i0 = ((blockIdx.x - nrb) * blockDim.x + threadIdx.x) * 4;
    float4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int i = i0 + u, w = i / n4, j = i - w * n4;
      if (i < 3 * n4) v[u] = __ldg(reinterpret_cast<const float4*>(w == 0 ? w0 : w == 1 ? w1 : w2) + j);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int i = i0 + u, w = i / n4, j = i - w * n4;
      if (i < 3 * n4)
        reinterpret_cast<float4*>(w == 0 ? o0 : w == 1 ? o1 : o2)[j] = make_float4(tf32r(v[u].x), tf32r(v[u].y), tf32r(v[u].z), tf32r(v[u].w));
    }
    return;
  }
  if (row >= M) return;
  const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * D_);
  float4 v[PER4];
#pragma unroll
  for (int i = 0; i < PER4; ++i) v[i] = xr[lane + 32 * i];
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < PER4; ++i) { s += v[i].x; s += v[i].y; s += v[i].z; s += v[i].w; }   // tg_ln_fwd's order and arithmetic: the same xn bits
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = s / (float)D_;
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < PER4; ++i) {
    const float a = v[i].x - mean, b = v[i].y - mean, c = v[i].z - mean, e = v[i].w - mean;
    q += a * a; q += b * b; q += c * c; q += e * e;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
  const float rs = rsqrtf(q / (float)D_ + eps), cc = mean * rs;
  float4* outr = reinterpret_cast<float4*>(xn + (size_t)row * D_);
#pragma unroll
  for (int i = 0; i < PER4; ++i) {
    const int col = (lane + 32 * i) * 4;
    const float4 g = __ldg(reinterpret_cast<const float4*>(gamma + col)), b = __ldg(reinterpret_cast<const float4*>(beta + col));
    outr[lane + 32 * i] = make_float4(tf32r((v[i].x - mean) * rs * g.x + b.x), tf32r((v[i].y - mean) * rs * g.y + b.y),
                                      tf32r((v[i].z - mean) * rs * g.z + b.z), tf32r((v[i].w - mean) * rs * g.w + b.w));
  }
  if (save && lane == 0) { rstd[row] = rs; c1[row] = save == 2 ? mean : cc; }   // save 2: the mean (the generic backward's statistics)
}
#else

DEVI f2 unpack_bf16x2(uint32_t w) { return mk2u(w << 16, w & 0xffff0000u); }

extern "C" __global__ void __launch_bounds__(WPB * 32)
transition_sx_ln(const __nv_bfloat16* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
                 __nv_bfloat16* __restrict__ xn, float* __restrict__ rstd, float* __restrict__ c1, int M, float eps, int save,
                 const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2, __nv_bfloat16* __restrict__ o0,
                 __nv_bfloat16* __restrict__ o1, __nv_bfloat16* __restrict__ o2, int ncast
                 TSX_STAMP_ARG) {
  if (threadIdx.x == 0) TSX_ST(0);
  const int lane = threadIdx.x & 31, nrb = (M + WPB - 1) / WPB, row = blockIdx.x * WPB + (threadIdx.x >> 5);
#if LNPF
  // (round 3 part 16) gamma / beta are parameters: their (cold, L2-flushed) loads go out before griddepcontrol.wait instead of after the
  // row statistics (the compiler kept all but the first chunk behind rsqrt: a second dependent DRAM round trip on the critical path)
  float4 gv[PER], bv[PER];
  if ((int)blockIdx.x < nrb) {
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      gv[i] = __ldg(reinterpret_cast<const float4*>(gamma) + lane + 32 * i);
      bv[i] = __ldg(reinterpret_cast<const float4*>(beta) + lane + 32 * i);
    }
  }
#endif
  if (threadIdx.x == 0) TSX_ST(1);
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
  if (lane == 0 && (threadIdx.x >> 5) == 0) TSX_ST(2);
  const float rs = rsqrtf(q * (1.f / D_) + eps), cc = mean * rs;
  const f2 rs2 = mk2(rs, rs), ncc2 = mk2(-cc, -cc);
  uint2* outr = reinterpret_cast<uint2*>(xn + (size_t)row * D_);
#pragma unroll
  for (int i = 0; i < PER; ++i) {
#if LNPF
    const float4 g = gv[i], b = bv[i];
#else
    const int col = (lane + 32 * i) * 4;
    const float4 g = __ldg(reinterpret_cast<const float4*>(gamma + col)), b = __ldg(reinterpret_cast<const float4*>(beta + col));
#endif
    const f2 g0 = mk2(g.x, g.y), g1 = mk2(g.z, g.w), b0 = mk2(b.x, b.y), b1 = mk2(b.z, b.w);
    const f2 y0 = fma2(unpack_bf16x2(v[i].x), mul2(rs2, g0), fma2(ncc2, g0, b0));   // (x rs - mean rs) g + b
    const f2 y1 = fma2(unpack_bf16x2(v[i].y), mul2(rs2, g1), fma2(ncc2, g1, b1));
    outr[lane + 32 * i] = make_uint2(pack_bf16(lo2(y0), hi2(y0)), pack_bf16(lo2(y1), hi2(y1)));
  }
  if (threadIdx.x == 0) TSX_ST(3);
  if (save == 3) {                                             // (part 19) no statistics: x copied into o0 (the cuBLAS squeeze's C = D)
    uint2* cr = reinterpret_cast<uint2*>(o0 + (size_t)row * D_);
#pragma unroll
    for (int i = 0; i < PER; ++i) cr[lane + 32 * i] = v[i];
  } else if (save && lane == 0) { rstd[row] = rs; c1[row] = save == 2 ? mean : cc; }   // save 2: the mean (the generic backward's statistics)
}
#endif
