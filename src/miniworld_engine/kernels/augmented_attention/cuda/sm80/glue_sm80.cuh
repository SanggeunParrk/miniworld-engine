// glue_sm80.cuh -- the row-wise elementwise passes of the A100 (sm_80) AugmentedAttentionPairBias path: the sigmoid gates and their backward.  Memory-bound CUDA
// (16-byte vectors, 8 elements a thread), bf16 or fp32 operands, fp32 arithmetic, one rounding.  Operands are [M][d] row-major with a row stride (a gate that is a column
// block of the projection output keeps that stride).
//
//   gate_rows       og = sigmoid(g) o                                       (the attention output gate)
//   gate_bwd        dob = dog sigmoid(g),  dg = dog o sigmoid(g) (1 - sigmoid(g))
//   res_gate        out = res + sigmoid(g2) y            (res optional)     (the output gate of the update and the residual)
//   res_gate_bwd    dy = dout sigmoid(g2),  dg2 = dout y sigmoid(g2) (1 - sigmoid(g2))
#pragma once
#include "sm80_common.cuh"

namespace aa80 {

// 8 consecutive elements of a row
template <typename T> struct V8;
template <> struct V8<__nv_bfloat16> {
  static DEVI void load(const __nv_bfloat16* p, float (&x)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    x[0] = bf16lo(u.x); x[1] = bf16hi(u.x); x[2] = bf16lo(u.y); x[3] = bf16hi(u.y);
    x[4] = bf16lo(u.z); x[5] = bf16hi(u.z); x[6] = bf16lo(u.w); x[7] = bf16hi(u.w);
  }
  static DEVI void store(__nv_bfloat16* p, const float (&x)[8]) {
    stg128(p, make_uint4(pack_bf16(x[0], x[1]), pack_bf16(x[2], x[3]), pack_bf16(x[4], x[5]), pack_bf16(x[6], x[7])));
  }
};
template <> struct V8<float> {
  static DEVI void load(const float* p, float (&x)[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w; x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
  }
  static DEVI void store(float* p, const float (&x)[8]) {
    reinterpret_cast<float4*>(p)[0] = make_float4(x[0], x[1], x[2], x[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(x[4], x[5], x[6], x[7]);
  }
};

constexpr int GLUE_NT = 256;

// the thread's vector: (row r, first column j) of chunk c over a [M][d] operand
#define GLUE_CHUNK(c, d8, r, j) const long long r = (c) / (d8); const int j = (int)((c) - r * (d8)) * 8;

template <typename T>
__global__ void __launch_bounds__(GLUE_NT) gate_rows_kernel(const T* __restrict__ o, const T* __restrict__ g, T* __restrict__ og, long long M, int d, long long ldo, long long ldg,
                                                             long long ldog) {
  const int d8 = d / 8;
  for (long long c = (long long)blockIdx.x * GLUE_NT + threadIdx.x; c < M * d8; c += (long long)gridDim.x * GLUE_NT) {
    GLUE_CHUNK(c, d8, r, j)
    float xo[8], xg[8], y[8];
    V8<T>::load(o + r * ldo + j, xo); V8<T>::load(g + r * ldg + j, xg);
#pragma unroll
    for (int e = 0; e < 8; ++e) y[e] = xo[e] * sigmoid(xg[e]);
    V8<T>::store(og + r * ldog + j, y);
  }
}

template <typename T>
__global__ void __launch_bounds__(GLUE_NT) gate_bwd_kernel(const T* __restrict__ dog, const T* __restrict__ o, const T* __restrict__ g, T* __restrict__ dob, T* __restrict__ dg,
                                                            long long M, int d, long long lddog, long long ldo, long long ldg, long long lddob, long long lddg) {
  const int d8 = d / 8;
  for (long long c = (long long)blockIdx.x * GLUE_NT + threadIdx.x; c < M * d8; c += (long long)gridDim.x * GLUE_NT) {
    GLUE_CHUNK(c, d8, r, j)
    float xd[8], xo[8], xg[8], a[8], b[8];
    V8<T>::load(dog + r * lddog + j, xd); V8<T>::load(o + r * ldo + j, xo); V8<T>::load(g + r * ldg + j, xg);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float s = sigmoid(xg[e]);
      a[e] = xd[e] * s;
      b[e] = xd[e] * xo[e] * s * (1.f - s);
    }
    V8<T>::store(dob + r * lddob + j, a); V8<T>::store(dg + r * lddg + j, b);
  }
}

template <typename T, bool RES>
__global__ void __launch_bounds__(GLUE_NT) res_gate_kernel(const T* __restrict__ y, const T* __restrict__ g2, const T* __restrict__ res, T* __restrict__ out, long long M, int d,
                                                            long long ldy, long long ldg, long long ldres, long long ldout) {
  const int d8 = d / 8;
  for (long long c = (long long)blockIdx.x * GLUE_NT + threadIdx.x; c < M * d8; c += (long long)gridDim.x * GLUE_NT) {
    GLUE_CHUNK(c, d8, r, j)
    float xy[8], xg[8], xr[8], z[8];
    V8<T>::load(y + r * ldy + j, xy); V8<T>::load(g2 + r * ldg + j, xg);
    if constexpr (RES) V8<T>::load(res + r * ldres + j, xr);
#pragma unroll
    for (int e = 0; e < 8; ++e) z[e] = RES ? fmaf(sigmoid(xg[e]), xy[e], xr[e]) : sigmoid(xg[e]) * xy[e];
    V8<T>::store(out + r * ldout + j, z);
  }
}

// bias_pack: the attention core's bias [H][Lp][Lp] bf16 in RAW units from a natural-unit bias [H][L][L] (bf16 / fp32): out = in x scale, the key columns of a masked key (``km``: one
// byte per key of each batch element b = plane / H, null = all valid) and the columns j >= L (padding) carry ``fill``, the query rows i >= L are 0 (finite).  L a multiple of 8.
// 8 output columns a thread; the planes are (b, h).
template <typename T>
__global__ void __launch_bounds__(GLUE_NT) bias_pack_kernel(const T* __restrict__ in, const unsigned char* __restrict__ km, __nv_bfloat16* __restrict__ out, int L, int Lp, int H, long long rows,
                                                             float scale, float fill) {
  const int c8 = Lp / 8;                                          // chunks per output row
  for (long long c = (long long)blockIdx.x * GLUE_NT + threadIdx.x; c < rows * c8; c += (long long)gridDim.x * GLUE_NT) {
    const long long row = c / c8;                                 // (h, i) of the plane (b folded into the head index by the caller)
    const int j = (int)(c - row * c8) * 8, i = (int)(row % Lp);
    float x[8];
    if (i >= L) {
#pragma unroll
      for (int e = 0; e < 8; ++e) x[e] = 0.f;
    } else if (j >= L) {
#pragma unroll
      for (int e = 0; e < 8; ++e) x[e] = fill;
    } else {
      const long long plane = row / Lp;                           // (b, h)
      V8<T>::load(in + (plane * L + i) * L + j, x);
#pragma unroll
      for (int e = 0; e < 8; ++e) x[e] = (km == nullptr || km[(plane / H) * L + j + e]) ? x[e] * scale : fill;
    }
    V8<__nv_bfloat16>::store(out + row * Lp + j, x);
  }
}

template <typename T>
__global__ void __launch_bounds__(GLUE_NT) res_gate_bwd_kernel(const T* __restrict__ dout, const T* __restrict__ y, const T* __restrict__ g2, T* __restrict__ dy, T* __restrict__ dg2,
                                                                long long M, int d, long long lddout, long long ldy, long long ldg, long long lddy, long long lddg) {
  const int d8 = d / 8;
  for (long long c = (long long)blockIdx.x * GLUE_NT + threadIdx.x; c < M * d8; c += (long long)gridDim.x * GLUE_NT) {
    GLUE_CHUNK(c, d8, r, j)
    float xd[8], xy[8], xg[8], a[8], b[8];
    V8<T>::load(dout + r * lddout + j, xd); V8<T>::load(y + r * ldy + j, xy); V8<T>::load(g2 + r * ldg + j, xg);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float s = sigmoid(xg[e]);
      a[e] = xd[e] * s;
      b[e] = xd[e] * xy[e] * s * (1.f - s);
    }
    V8<T>::store(dy + r * lddy + j, a); V8<T>::store(dg2 + r * lddg + j, b);
  }
}

}  // namespace aa80
