// wide_fin.cuh -- the end of the wide TriMul backward in one launch: the fp32 results of the weight-gradient GEMMs and the LayerNorm column sums become the
// parameters' gradients, in the parameters' dtypes and strides (weights: bf16 or fp32 per the path's element type; LayerNorm vectors: bf16 or fp32 each).
//   items [0, 4 Hs)             dW1 [4 Hs + D, D] row r (k1w packed order) -> W_l / W_lg / W_r / W_rg row (written through the output's strides: the bidirectional
//                               module stores them [in, out], and a gradient with the same strides is taken over by autograd, not copied)
//   items [4 Hs, 4 Hs + D)      dW1 row 4 Hs + n -> W_g row n
//   items [.. + D)              dWo[n, k] = g_o[k] (G[n, k] - r1[n]) + r0[n] b_o[k]      (G = dpr^T X^T, r0 / r1 the column sums of gate_bwd)
//   items [.. + 4)              d gamma_out, d beta_out ([Hs], from lnout_bwd) and d gamma_in, d beta_in ([D], from lnin_bwd)
#pragma once
#include "wide_elt.cuh"

namespace a100 {

template <class E>
struct WFinParams {
  using T = typename E::T;
  const float* dw1;            // [4 Hs + D, D]
  const float* G;              // [D, Hs]
  const float* r01;            // [2 D]: r0 | r1
  const float* gout;           // [2 Hs]: d gamma_out | d beta_out
  const float* gin;            // [2 D]: d gamma_in | d beta_in
  const float *go, *bo;        // [Hs] LayerNorm_out affine
  T* w[4];                     // outputs W_l, W_lg, W_r, W_rg
  int rs[4], ks[4];            // their element strides: output channel, input channel
  T *dwg, *dwo;                // [D, D], [D, Hs] row-major
  void* ln[4];                 // d gamma_out, d beta_out, d gamma_in, d beta_in
  int ln_bf16[4];              // per LayerNorm gradient: 1 = bf16, 0 = fp32
  int Hs, D;
};

template <class E>
__global__ void __launch_bounds__(256) wfin_kernel(const WFinParams<E> p) {
  const int item = (int)(blockIdx.x * 8 + (threadIdx.x >> 5)), lane = threadIdx.x & 31;
  const int Hs = p.Hs, D = p.D;
  if (item < 4 * Hs) {
    const int r = item, j = r >> 4, i = r & 15, pc = 8 * j + (i & 7), side = pc >= Hs ? 1 : 0, c = pc - side * Hs;
    const int which = (i < 8 ? 1 : 0) + 2 * side;
    typename E::T* dst = p.w[which] + (size_t)c * p.rs[which];
    const size_t ks = (size_t)p.ks[which];
    for (int k = lane; k < D; k += 32) dst[k * ks] = E::from_f(p.dw1[(size_t)r * D + k]);
  } else if (item < 4 * Hs + D) {
    const int n = item - 4 * Hs;
    for (int k = lane; k < D; k += 32) p.dwg[(size_t)n * D + k] = E::from_f(p.dw1[(size_t)(4 * Hs + n) * D + k]);
  } else if (item < 4 * Hs + 2 * D) {
    const int n = item - 4 * Hs - D;
    const float r0 = p.r01[n], r1 = p.r01[D + n];
    for (int k = lane; k < Hs; k += 32) p.dwo[(size_t)n * Hs + k] = E::from_f(fmaf(p.go[k], p.G[(size_t)n * Hs + k] - r1, r0 * p.bo[k]));
  } else if (item < 4 * Hs + 2 * D + 4) {
    const int v = item - 4 * Hs - 2 * D;                         // 0: d gamma_out, 1: d beta_out, 2: d gamma_in, 3: d beta_in
    const int len = v < 2 ? Hs : D;
    const float* src = (v < 2 ? p.gout : p.gin) + (v & 1) * len;
    for (int k = lane; k < len; k += 32) {
      if (p.ln_bf16[v]) reinterpret_cast<__nv_bfloat16*>(p.ln[v])[k] = __float2bfloat16_rn(src[k]);
      else reinterpret_cast<float*>(p.ln[v])[k] = src[k];
    }
  }
}

}  // namespace a100
