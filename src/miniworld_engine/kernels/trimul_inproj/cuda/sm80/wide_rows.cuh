// wide_rows.cuh -- the memory-bound pieces around the A100 "wide" TriMul GEMM kernels (any width D, either direction; bf16 or fp32 storage, see wide_elt.cuh):
//   ln_stats_rows   per-token (mean, rstd) over the D channels of a token-major [T, D] tensor (two-pass, fp32)
//   ln_stats_cm     per-token (mean, rstd) over the H channels of the channel-major contraction output [H, T] (the B200 wide_aux kernel: every
//                   read covers 512 contiguous bytes of a channel row; pivot-shifted partial sums merged with Chan's formula)
//   wpack_kernel    every folded weight layout of the forward in one launch (one warp per row):
//                     rows [0, 4 H)         W1p row r = 16 j + i -> plane channel 8 j + (i % 8), gate row for i < 8 (left = channel < H), else projection:
//                                           bf16 (TF32 for fp32) of 0.5 gamma_in W ; vs = row sum of the rounded values ; vb = 0.5 W beta_in
//                     rows [4 H, 4 H + D)   Wo' [D, Hc]  rounded 0.5 gamma_out Wo, so, eo
//                     rows [.. + D)         Wg' [D, D]   rounded 0.5 gamma_in Wg, sg, eg
//                   LayerNorm folds: LN(x) . W^T = r (x . W'^T) - r mu s + b, and the 0.5 makes sigmoid(g) p = p' (1 + tanh g') (exact in bf16)
#pragma once
#include "wide_elt.cuh"

namespace a100 {

// ---------------------------------------------------------------------------------------------------------------- ln_stats_rows
// D / VEC sixteen-byte granules per row; LPR lanes cooperate on a row (RPW rows per warp pass); every load of a pass is issued before the first
// reduction.  Statistics: mean = sum / D, var = sum (x - mean)^2 / D over the registers (two-pass), rstd = rsqrt(var + eps).
template <int D, class E>
__global__ void __launch_bounds__(256) ln_stats_rows_kernel(const typename E::T* __restrict__ z, float2* __restrict__ st, int T, float eps) {
  constexpr int V = E::VEC, G = D / V, LPR = G < 32 ? G : 32, RPW = 32 / LPR, NI = (G + LPR - 1) / LPR;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int sub = lane / LPR, ln = lane - sub * LPR;
  const int rows_per_pass = 8 * RPW;
  for (int base = blockIdx.x * rows_per_pass; base < T; base += gridDim.x * rows_per_pass) {
    const int row = base + warp * RPW + sub;
    const bool live = row < T;
    float x[NI][V];
#pragma unroll
    for (int it = 0; it < NI; ++it) {
      const int g = ln + LPR * it;
      const uint4 v = (live && g < G) ? __ldg(reinterpret_cast<const uint4*>(z) + (size_t)row * G + g) : make_uint4(0u, 0u, 0u, 0u);
      E::unpack(v, x[it]);
    }
    float s = 0.f;
#pragma unroll
    for (int it = 0; it < NI; ++it)
#pragma unroll
      for (int e = 0; e < V; ++e) s += x[it][e];
#pragma unroll
    for (int o = 1; o < LPR; o <<= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s * (1.f / D);
    float q = 0.f;
#pragma unroll
    for (int it = 0; it < NI; ++it) {
      const int g = ln + LPR * it;
      if (g < G) {
#pragma unroll
        for (int e = 0; e < V; ++e) { const float a = x[it][e] - mean; q = fmaf(a, a, q); }
      }
    }
#pragma unroll
    for (int o = 1; o < LPR; o <<= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    if (live && ln == 0) st[row] = make_float2(mean, rsqrtf(q * (1.f / D) + eps));
  }
}

// ------------------------------------------------------------------------------------------------------------------- ln_stats_cm
// x is channel-major [H, M]: block = 32 * TPL tokens (lane = TPL tokens), 8 warps split the channels; each thread keeps pivot-shifted sums (pivot =
// its first channel's value), the eight partitions are merged with Chan's formula.  TPL = E::VEC: one 16-byte load per channel row (256 / 128 tokens per
// block); TPL = 4 bytes' worth (2 bf16, 1 fp32): one 4-byte load, for the lengths that leave fewer than two blocks per SM.
template <int H, int TPL, class E>
__global__ void __launch_bounds__(256) ln_stats_cm_kernel(const typename E::T* __restrict__ x, float2* __restrict__ st, int M, float eps) {
  using T = typename E::T;
  constexpr int CW = H / 8, BT = 32 * TPL;
  static_assert(TPL * sizeof(T) == 16 || TPL * sizeof(T) == 4, "one 16-byte or one 4-byte load per channel row");
  __shared__ float pm[8][BT], pq[8][BT];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tok0 = blockIdx.x * BT + lane * TPL;
  const T* base = x + (size_t)(warp * CW) * M + tok0;
  auto load = [&](size_t off, float (&f)[TPL]) {
    if constexpr (TPL * sizeof(T) == 16) {
      E::unpack(*reinterpret_cast<const uint4*>(base + off), f);
    } else if constexpr (E::IS_F32) {
      f[0] = base[off];
    } else {
      const uint32_t w = *reinterpret_cast<const uint32_t*>(base + off);
      f[0] = bf16lo(w); f[1] = bf16hi(w);
    }
  };
  float piv[TPL], s[TPL], q[TPL];
  load(0, piv);
#pragma unroll
  for (int j = 0; j < TPL; ++j) { s[j] = 0.f; q[j] = 0.f; }
#pragma unroll 16
  for (int k = 1; k < CW; ++k) {
    float f[TPL];
    load((size_t)k * M, f);
#pragma unroll
    for (int j = 0; j < TPL; ++j) {
      const float a = f[j] - piv[j];
      s[j] += a; q[j] = fmaf(a, a, q[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < TPL; ++j) {          // partition mean / M2 over CW values (the pivot contributes d = 0)
    pm[warp][lane * TPL + j] = piv[j] + s[j] * (1.f / CW);
    pq[warp][lane * TPL + j] = fmaxf(q[j] - s[j] * s[j] * (1.f / CW), 0.f);
  }
  __syncthreads();
  for (int tk = threadIdx.x; tk < BT; tk += 256) {
    float N = 0.f, Mu = 0.f, Q = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      const float nb = (float)CW, mb = pm[w][tk], qb = pq[w][tk];
      const float nt = N + nb, d = mb - Mu;
      Mu += d * (nb / nt);
      Q += qb + d * d * (N * nb / nt);
      N = nt;
    }
    st[blockIdx.x * BT + tk] = make_float2(Mu, rsqrtf(Q * (1.f / H) + eps));
  }
}

// ------------------------------------------------------------------------------------------------------------------------ wpack_kernel
template <class E>
struct WPackParams {
  using T = typename E::T;
  const T *wl, *wlg, *wr, *wrg;                  // [Hs, D] each, through the strides below (a row-major Linear: rs = D, ks = 1)
  int rs[4], ks[4];                              // element strides of wl, wlg, wr, wrg: output channel, input channel
  const T *wg, *wo;                              // [D, D] and [D, Hc], row-major
  const float *gi, *bi, *go, *bo;                // LayerNorm affine: in [D], out [Hc]
  T *w1, *wo3, *wg3;                             // [4 Hs, D], [D, Hc], [D, D]
  float *vs, *vb;                                // [4 Hs] (front rows, packed order)
  float *so, *eo, *sg, *eg;                      // [D] each
  T* wdx;                                        // training only (else nullptr): [4 Hs + D, D] the UNSCALED weights, packed row order, then Wg
  float* spx;                                    // training only: [D]  sum_k Wo[n, k] gamma_out[k]  (exact: the backward's LayerNorm_out row-sum identity)
  int Hs, D, Hc;                                 // per-side channels, input width, hidden channels (= Hs: the contraction output has Hs channels)
};

DEVI float warp_sum_w(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// the stored folded weight: bf16 (rounded to nearest even) or TF32 (rounded to nearest) of a float
template <class E> DEVI float fold_round(float x) {
  if constexpr (E::IS_F32) return tf32_round(x);
  else return round_bf16f(x);
}

template <class E>
__global__ void __launch_bounds__(256) wpack_kernel(const WPackParams<E> p) {
  const int item = (int)(blockIdx.x * 8 + (threadIdx.x >> 5)), lane = threadIdx.x & 31;
  const int D = p.D;
  if (item < 4 * p.Hs) {
    const int r = item, j = r >> 4, i = r & 15, pc = 8 * j + (i & 7), side = pc >= p.Hs ? 1 : 0, c = pc - side * p.Hs;
    const int which = (i < 8 ? 1 : 0) + 2 * side;                              // wl, wlg, wr, wrg
    const auto* src = (which == 0 ? p.wl : which == 1 ? p.wlg : which == 2 ? p.wr : p.wrg) + (size_t)c * p.rs[which];
    const size_t ks = (size_t)p.ks[which];
    float s = 0.f, b = 0.f;
    for (int k = lane; k < D; k += 32) {
      const float w = E::to_f(src[k * ks]);
      const float h = fold_round<E>(0.5f * w * p.gi[k]);
      p.w1[(size_t)r * D + k] = E::from_f(h);
      if (p.wdx) p.wdx[(size_t)r * D + k] = E::from_f(w);
      s += h;
      b = fmaf(w, p.bi[k], b);
    }
    s = warp_sum_w(s); b = warp_sum_w(b);
    if (lane == 0) { p.vs[r] = s; p.vb[r] = 0.5f * b; }
  } else if (item < 4 * p.Hs + D) {
    const int n = item - 4 * p.Hs;
    float s = 0.f, b = 0.f, sx = 0.f;
    for (int k = lane; k < p.Hc; k += 32) {
      const float w = E::to_f(p.wo[(size_t)n * p.Hc + k]);
      const float h = fold_round<E>(0.5f * w * p.go[k]);
      p.wo3[(size_t)n * p.Hc + k] = E::from_f(h);
      s += h;
      b = fmaf(w, p.bo[k], b);
      sx = fmaf(w, p.go[k], sx);
    }
    s = warp_sum_w(s); b = warp_sum_w(b);
    if (p.spx) sx = warp_sum_w(sx);
    if (lane == 0) { p.so[n] = s; p.eo[n] = 0.5f * b; if (p.spx) p.spx[n] = sx; }
  } else if (item < 4 * p.Hs + 2 * D) {
    const int n = item - 4 * p.Hs - D;
    float s = 0.f, b = 0.f;
    for (int k = lane; k < D; k += 32) {
      const float w = E::to_f(p.wg[(size_t)n * D + k]);
      const float h = fold_round<E>(0.5f * w * p.gi[k]);
      p.wg3[(size_t)n * D + k] = E::from_f(h);
      if (p.wdx) p.wdx[(size_t)(4 * p.Hs + n) * D + k] = E::from_f(w);
      s += h;
      b = fmaf(w, p.bi[k], b);
    }
    s = warp_sum_w(s); b = warp_sum_w(b);
    if (lane == 0) { p.sg[n] = s; p.eg[n] = 0.5f * b; }
  }
}

}  // namespace a100
