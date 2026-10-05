// adaln_rows.cuh -- the row passes of AdaLN on A100 (sm_80), forward and backward, fp32 or bf16 rows, fp32 statistics.  The GEMMs between them are cuBLAS.
//
//   y = sigmoid(scale) LN(x) + bias,   [scale | bias] = (LN(cond) w) [Ws | Wb]^T + [sb | 0]       (LN without affine on x; w = the cond norm's weight)
//
//   cond_ln       aff = LN(cond) w                          [P, dc]   (+ (mean, rstd) of every cond row)
//   adaln_epi     y = sigmoid(S + sb) LN(x) + B             S | B = a row of the [P, 2 d] GEMM output (row r of x reads table row r % P: P < M is one
//                                                           conditioning shared by M / P samples), (+ (mean, rstd) of every x row)
//   adaln_bwd_x   D = [dscale | dy], dx = LN-backward(dy sigmoid) (+ dres), the column sums of dscale (d sb)
//   cond_ln_bwd   dcond = LN-backward(dcond_aff) (+ dextra), the column sums of dcond_aff cond_hat (d w)
//
// One WARP owns a row (no block barriers: the row sums are shuffles): lane l holds the 4-element vectors v = 0 .. D / 128 - 1 at columns 4 (32 v + l) .. +3 (a 16-B fp32 / 8-B bf16 load,
// 256 / 512 contiguous bytes a warp-wide load), so a row is D / 128 independent loads per tensor, all issued before the first sum.  Column sums leave a block as one row of a
// [blocks, width] fp32 partial buffer (the block's warps added in order through shared memory; plain stores: no atomics, a fixed order); ``finish`` (adaln_finish.cuh) adds the rows.
#pragma once
#include "adaln_common.cuh"

namespace adl {

constexpr int RW = 4;                                       // warps (rows) per block

ADL_DEVI int vcol(int v, int lane) { return (v * 32 + lane) * 4; }

// ------------------------------------------------------------------------------------------------------------------------------- forward
template <int D, typename T>
__global__ void __launch_bounds__(RW * 32) cond_ln_kernel(const T* __restrict__ C, long sc, const float* __restrict__ W, T* __restrict__ AFF, float2* __restrict__ ST, long P, float eps) {
  constexpr int NV = D / 128;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (row >= P) return;
  float4 c[NV];
  float s = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) { c[v] = V4<T>::load(C + row * sc + vcol(v, lane)); s += sum4(c[v]); }
  const float mean = warp_sum(s) / D;
  float q = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) { c[v] = make_float4(c[v].x - mean, c[v].y - mean, c[v].z - mean, c[v].w - mean); q += sum4(mul4(c[v], c[v])); }
  const float rstd = rsqrtf(warp_sum(q) / D + eps);
#pragma unroll
  for (int v = 0; v < NV; ++v) V4<T>::store(AFF + row * D + vcol(v, lane), mul4(scale4(c[v], rstd), V4<float>::load(W + vcol(v, lane))));
  if (ST != nullptr && lane == 0) ST[row] = make_float2(mean, rstd);
}

template <int D, typename XT>
__global__ void __launch_bounds__(RW * 32) adaln_epi_kernel(const XT* __restrict__ X, long sx, const XT* __restrict__ SBT, long ssb, const XT* __restrict__ SB, XT* __restrict__ Y,
                                                            float2* __restrict__ XST, long M, long P, float eps) {
  constexpr int NV = D / 128;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (row >= M) return;
  const XT* t = SBT + (row % P) * ssb;
  float4 x[NV], sc[NV], bi[NV];
  float s = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) {                           // every load of the row before the first sum
    x[v] = V4<XT>::load(X + row * sx + vcol(v, lane));
    sc[v] = V4<XT>::load(t + vcol(v, lane));
    bi[v] = V4<XT>::load(t + D + vcol(v, lane));
    s += sum4(x[v]);
  }
  const float mean = warp_sum(s) / D;
  float q = 0.f;
#pragma unroll
  for (int v = 0; v < NV; ++v) { x[v] = make_float4(x[v].x - mean, x[v].y - mean, x[v].z - mean, x[v].w - mean); q += sum4(mul4(x[v], x[v])); }
  const float rstd = rsqrtf(warp_sum(q) / D + eps);
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const float4 g = sig4(add4(sc[v], V4<XT>::load(SB + vcol(v, lane))));
    V4<XT>::store(Y + row * D + vcol(v, lane), add4(mul4(g, scale4(x[v], rstd)), bi[v]));
  }
  if (XST != nullptr && lane == 0) XST[row] = make_float2(mean, rstd);
}

// ------------------------------------------------------------------------------------------------------------------------------ backward
// The block's warps' column sums, added in warp order, one partial row of D fp32 (cs: [RW][D / 4] float4, this warp's vectors at v * 32 + lane).
template <int D>
ADL_DEVI void block_partial_row(float4 (*cs)[D / 4], const float4 (&acc)[D / 128], float* __restrict__ out) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int v = 0; v < D / 128; ++v) cs[warp][v * 32 + lane] = acc[v];
  __syncthreads();
  for (int i = threadIdx.x; i < D / 4; i += RW * 32) {
    float4 t = cs[0][i];
#pragma unroll
    for (int w = 1; w < RW; ++w) t = add4(t, cs[w][i]);
    V4<float>::store(out + 4 * i, t);
  }
}

// D (row r: dscale | dy, 2 d wide, AT: the GEMM operand dtype), dx, and the column sums of dscale.  xhat = (x - mean) rstd, g = sigmoid(S + sb),
// dscale = dy xhat g (1 - g), dxhat = dy g, dx = rstd (dxhat - mean(dxhat) - xhat mean(dxhat xhat)) + dres.  Rows are dealt to the warps of the grid round-robin;
// gridDim.x is the number of partial rows.
template <int D, typename XT>
__global__ void __launch_bounds__(RW * 32) adaln_bwd_x_kernel(const XT* __restrict__ DY, const XT* __restrict__ X, const float2* __restrict__ XST, const XT* __restrict__ SBT, long ssb,
                                                              const XT* __restrict__ SB, const XT* __restrict__ DRES, XT* __restrict__ DM, XT* __restrict__ DX, float* __restrict__ PSB,
                                                              long M, long P) {
  constexpr int NV = D / 128;
  __shared__ float4 cs[RW][D / 4];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float4 acc[NV];
#pragma unroll
  for (int v = 0; v < NV; ++v) acc[v] = zero4();
  for (long row = (long)blockIdx.x * RW + warp; row < M; row += (long)gridDim.x * RW) {
    float4 dy[NV], x[NV], g[NV];
    const float2 st = XST[row];
    const XT* t = SBT + (row % P) * ssb;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      dy[v] = V4<XT>::load(DY + row * D + vcol(v, lane));
      x[v] = V4<XT>::load(X + row * D + vcol(v, lane));
      g[v] = V4<XT>::load(t + vcol(v, lane));
    }
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      const float4 xh = make_float4((x[v].x - st.x) * st.y, (x[v].y - st.x) * st.y, (x[v].z - st.x) * st.y, (x[v].w - st.x) * st.y);
      g[v] = sig4(add4(g[v], V4<XT>::load(SB + vcol(v, lane))));
      const float4 dsc = make_float4(dy[v].x * xh.x * g[v].x * (1.f - g[v].x), dy[v].y * xh.y * g[v].y * (1.f - g[v].y), dy[v].z * xh.z * g[v].z * (1.f - g[v].z),
                                     dy[v].w * xh.w * g[v].w * (1.f - g[v].w));
      V4<XT>::store(DM + row * 2 * D + vcol(v, lane), dsc);
      V4<XT>::store(DM + row * 2 * D + D + vcol(v, lane), dy[v]);
      acc[v] = add4(acc[v], dsc);
      const float4 dxh = mul4(dy[v], g[v]);
      s1 += sum4(dxh);
      s2 += sum4(mul4(dxh, xh));
      x[v] = xh;                                       // x now holds xhat, g holds the gate, dy dy: dxh is recomputed below
    }
    const float m1 = warp_sum(s1) / D, m2 = warp_sum(s2) / D;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      const float4 dxh = mul4(dy[v], g[v]);
      float4 dx = make_float4(st.y * (dxh.x - m1 - x[v].x * m2), st.y * (dxh.y - m1 - x[v].y * m2), st.y * (dxh.z - m1 - x[v].z * m2), st.y * (dxh.w - m1 - x[v].w * m2));
      if (DRES != nullptr) dx = add4(dx, V4<XT>::load(DRES + row * D + vcol(v, lane)));
      V4<XT>::store(DX + row * D + vcol(v, lane), dx);
    }
  }
  block_partial_row<D>(cs, acc, PSB + (long)blockIdx.x * D);
}

// dcond = rstd (dchat - mean(dchat) - chat mean(dchat chat)) + dextra, dchat = dcond_aff w, chat = (cond - mean) rstd; the partial sums of dcond_aff chat.
template <int D, typename CT>
__global__ void __launch_bounds__(RW * 32) cond_ln_bwd_kernel(const float* __restrict__ DCA, const CT* __restrict__ C, const float2* __restrict__ CST, const float* __restrict__ W,
                                                              const CT* __restrict__ DEXTRA, CT* __restrict__ DC, float* __restrict__ PW, long M) {
  constexpr int NV = D / 128;
  __shared__ float4 cs[RW][D / 4];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float4 acc[NV];
#pragma unroll
  for (int v = 0; v < NV; ++v) acc[v] = zero4();
  for (long row = (long)blockIdx.x * RW + warp; row < M; row += (long)gridDim.x * RW) {
    float4 dca[NV], ch[NV];
    const float2 st = CST[row];
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      dca[v] = V4<float>::load(DCA + row * D + vcol(v, lane));
      ch[v] = V4<CT>::load(C + row * D + vcol(v, lane));
    }
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      ch[v] = make_float4((ch[v].x - st.x) * st.y, (ch[v].y - st.x) * st.y, (ch[v].z - st.x) * st.y, (ch[v].w - st.x) * st.y);
      acc[v] = add4(acc[v], mul4(dca[v], ch[v]));
      const float4 dh = mul4(dca[v], V4<float>::load(W + vcol(v, lane)));
      s1 += sum4(dh);
      s2 += sum4(mul4(dh, ch[v]));
    }
    const float m1 = warp_sum(s1) / D, m2 = warp_sum(s2) / D;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      const float4 dh = mul4(dca[v], V4<float>::load(W + vcol(v, lane)));
      float4 dc4 = make_float4(st.y * (dh.x - m1 - ch[v].x * m2), st.y * (dh.y - m1 - ch[v].y * m2), st.y * (dh.z - m1 - ch[v].z * m2), st.y * (dh.w - m1 - ch[v].w * m2));
      if (DEXTRA != nullptr) dc4 = add4(dc4, V4<CT>::load(DEXTRA + row * D + vcol(v, lane)));
      V4<CT>::store(DC + row * D + vcol(v, lane), dc4);
    }
  }
  block_partial_row<D>(cs, acc, PW + (long)blockIdx.x * D);
}

}  // namespace adl
