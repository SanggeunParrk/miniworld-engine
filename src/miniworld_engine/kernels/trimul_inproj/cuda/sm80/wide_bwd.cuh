// wide_bwd.cuh -- the memory-bound pieces of the A100 "wide" TriMul backward (any width D, hidden channels Hs; bf16 or fp32 storage, see wide_elt.cuh), between the
// cuBLAS GEMMs, the contraction backward and k1wb (the data flow of the B200 wide backward, wide_bwd.cu; plain CUDA):
//   gate_bwd    from dy, the saved p' / g' (0.5-scaled units) and ds:   s = 0.5 + 0.5 tanh g',  p = 2 p'
//                 dp = dy ds s,   dg = dy ds p s (1 - s)      ->  dpr = dp r_o  [T, D],  dg -> dcat[:, 4 Hs ..]  (row stride ldd)
//               per token S1 = dp . sp,  S2 = dp . (p - ep)   (the LN_out backward's two row sums, from the fold identities  sum_k g_o[k] dy_o[t, k] = dp . sp
//               and  sum_k g_o[k] dy_o[t, k] xhat[t, k] = dp . (p - ep), dy_o = dp Wo, sp = Wo g_o, ep = Wo b_o): no pass over the Hs channels
//               per column r0 = sum_t dp, r1 = sum_t dp r_o mu_o   (dWo = g_o (dpr^T X^T - r1) + b_o r0)
//   lnout_bwd   dor = dpr Wo = r_o dy_o [Hs, T] (channel-major, cuBLAS) and X [Hs, T]:  dX = g_o dor - r_o (S1 + xhat S2) / Hs ;  dg_o += dor (X - mu_o), db_o += dor / r_o
//   lnin_bwd    dxn [T, D] -> dx = dy + r_i (g_i dxn - mean(g_i dxn) - xhat mean(g_i dxn xhat)) ;  dg_i += dxn xhat,  db_i += dxn
//   ln_apply    xn = LN_in(z) with its affine (the A operand of the weight-gradient GEMM)
//   colsum      out[c] = sum_r part[r][c] in a fixed order (every cross-block reduction here is a row of partials + this: bit-identical replays)
#pragma once
#include "wide_elt.cuh"

namespace a100 {

// ------------------------------------------------------------------------------------------------------------------------------- gate_bwd
// Row structure of ln_stats_rows: D / VEC granules per row, LPR lanes per row, RPW rows per warp pass.  A block owns a static set of rows; its columns' r0 / r1
// partials are kept in registers, merged over the warp's sub-rows and then over the 8 warps in a fixed order, and written as one row of ``part`` ([grid][2 D]).
template <int D, class E>
__global__ void __launch_bounds__(256) gate_bwd_kernel(const typename E::T* __restrict__ dy, const typename E::T* __restrict__ ps,
                                                        const typename E::T* __restrict__ gs, const typename E::T* __restrict__ ds,
                                                        const float* __restrict__ spx, const float* __restrict__ epx, const float2* __restrict__ sto,
                                                        typename E::T* __restrict__ dpr, typename E::T* __restrict__ dgo, int ldg,
                                                        float2* __restrict__ s12, float* __restrict__ part, int T, int L) {
  constexpr int V = E::VEC, G = D / V, LPR = G < 32 ? G : 32, RPW = 32 / LPR, NI = (G + LPR - 1) / LPR;
  __shared__ float cs[8][2 * D];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int sub = lane / LPR, ln = lane - sub * LPR;
  float sp[NI][V], ep[NI][V], r0[NI][V], r1[NI][V];
#pragma unroll
  for (int it = 0; it < NI; ++it) {
    const int g = ln + LPR * it;
#pragma unroll
    for (int j = 0; j < V; ++j) {
      sp[it][j] = g < G ? spx[V * g + j] : 0.f;
      ep[it][j] = g < G ? epx[V * g + j] : 0.f;
      r0[it][j] = 0.f; r1[it][j] = 0.f;
    }
  }
  const int rows_per_pass = 8 * RPW;
  for (int base = blockIdx.x * rows_per_pass; base < T; base += gridDim.x * rows_per_pass) {
    const int row = base + warp * RPW + sub;
    const bool live = row < T;
    uint4 vy[NI], vp[NI], vg[NI], vd[NI];
#pragma unroll
    for (int it = 0; it < NI; ++it) {
      const int g = ln + LPR * it;
      const bool ok = live && g < G;
      const size_t e = (size_t)(live ? row : 0) * G + (g < G ? g : 0);
      vy[it] = ok ? __ldg(reinterpret_cast<const uint4*>(dy) + e) : make_uint4(0u, 0u, 0u, 0u);
      vp[it] = ok ? __ldg(reinterpret_cast<const uint4*>(ps) + e) : make_uint4(0u, 0u, 0u, 0u);
      vg[it] = ok ? __ldg(reinterpret_cast<const uint4*>(gs) + e) : make_uint4(0u, 0u, 0u, 0u);
      vd[it] = E::ones();
      if (ok && ds != nullptr) vd[it] = __ldg(reinterpret_cast<const uint4*>(ds) + (size_t)(row % L) * G + g);
    }
    const float2 so = live ? __ldg(sto + row) : make_float2(0.f, 0.f);
    const float rm = so.y * so.x;
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int it = 0; it < NI; ++it) {
      const int g = ln + LPR * it;
      if (g < G && live) {
        float fy[V], fp[V], fg[V], fd[V], odpr[V], odg[V];
        E::unpack(vy[it], fy); E::unpack(vp[it], fp); E::unpack(vg[it], fg); E::unpack(vd[it], fd);
#pragma unroll
        for (int j = 0; j < V; ++j) {
          const float s = fmaf(0.5f, tanh_approx(fg[j]), 0.5f);
          const float pv = 2.f * fp[j];
          const float dpv = fy[j] * fd[j] * s;
          odpr[j] = dpv * so.y;
          odg[j] = fy[j] * fd[j] * pv * s * (1.f - s);
          s1 = fmaf(dpv, sp[it][j], s1);
          s2 = fmaf(dpv, pv - ep[it][j], s2);
          r0[it][j] += dpv;
          r1[it][j] = fmaf(dpv, rm, r1[it][j]);
        }
        *reinterpret_cast<uint4*>(dpr + (size_t)row * D + V * g) = E::pack(odpr);
        *reinterpret_cast<uint4*>(dgo + (size_t)row * ldg + V * g) = E::pack(odg);
      }
    }
#pragma unroll
    for (int o = 1; o < LPR; o <<= 1) { s1 += __shfl_xor_sync(0xffffffffu, s1, o); s2 += __shfl_xor_sync(0xffffffffu, s2, o); }
    if (live && ln == 0) s12[row] = make_float2(s1, s2);
  }
  // merge the sub-rows (lanes with the same column group), then the warps
#pragma unroll
  for (int it = 0; it < NI; ++it)
#pragma unroll
    for (int j = 0; j < V; ++j)
#pragma unroll
      for (int o = LPR; o < 32; o <<= 1) {
        r0[it][j] += __shfl_xor_sync(0xffffffffu, r0[it][j], o);
        r1[it][j] += __shfl_xor_sync(0xffffffffu, r1[it][j], o);
      }
  if (sub == 0) {
#pragma unroll
    for (int it = 0; it < NI; ++it) {
      const int g = ln + LPR * it;
      if (g < G) {
#pragma unroll
        for (int j = 0; j < V; ++j) { cs[warp][V * g + j] = r0[it][j]; cs[warp][D + V * g + j] = r1[it][j]; }
      }
    }
  }
  __syncthreads();
  for (int c = threadIdx.x; c < 2 * D; c += 256) {
    float a = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) a += cs[w][c];
    part[(size_t)blockIdx.x * 2 * D + c] = a;
  }
}

// ------------------------------------------------------------------------------------------------------------------------------ lnout_bwd
// dor, x, dt are all channel-major [H, M]: a thread owns VEC consecutive tokens of a 32 VEC-token group (16-byte accesses, a warp covers 512 contiguous bytes of
// a channel row), keeps their per-token terms in registers and walks the channel rows of its warp.  Block (x, y): token group x, channels [64 y, 64 y + 64).
constexpr int LOB_CH = 64;
template <class E>
__global__ void __launch_bounds__(256) lnout_bwd_kernel(const typename E::T* __restrict__ dor, const typename E::T* __restrict__ x, const float2* __restrict__ sto,
                                                         const float2* __restrict__ s12, const float* __restrict__ go, typename E::T* __restrict__ dt,
                                                         float* __restrict__ part, int M, int H) {      // part [gridDim.x][2 H]
  constexpr int V = E::VEC;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int c0 = blockIdx.y * LOB_CH + warp * (LOB_CH / 8);
  const int tok = blockIdx.x * (32 * V) + lane * V;
  float mu[V], a1[V], a2r[V], irs[V];
  {
    const float4* so = reinterpret_cast<const float4*>(sto + tok);     // V tokens x (mean, rstd) = V / 2 float4
    const float4* sv = reinterpret_cast<const float4*>(s12 + tok);
    const float ih = 1.f / H;
#pragma unroll
    for (int k = 0; k < V / 2; ++k) {
      const float4 m = so[k], s = sv[k];
      const float mm[2] = {m.x, m.z}, rr[2] = {m.y, m.w}, ss1[2] = {s.x, s.z}, ss2[2] = {s.y, s.w};
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        mu[2 * k + j] = mm[j]; irs[2 * k + j] = 1.f / rr[j];
        a1[2 * k + j] = rr[j] * ss1[j] * ih;
        a2r[2 * k + j] = rr[j] * rr[j] * ss2[j] * ih;       // xhat S2 r / H = (x - mu) r^2 S2 / H
      }
    }
  }
  float sg_[LOB_CH / 8], sb_[LOB_CH / 8];
#pragma unroll
  for (int k = 0; k < LOB_CH / 8; ++k) {
    const int c = c0 + k;
    const size_t off = (size_t)c * M + tok;
    float dv[V], tv[V], o[V];
    E::unpack(*reinterpret_cast<const uint4*>(dor + off), dv);
    E::unpack(*reinterpret_cast<const uint4*>(x + off), tv);
    const float gc = go[c];
    float sg = 0.f, sb = 0.f;
#pragma unroll
    for (int j = 0; j < V; ++j) {
      const float tc = tv[j] - mu[j];
      o[j] = gc * dv[j] - a1[j] - tc * a2r[j];
      sg = fmaf(dv[j], tc, sg);
      sb = fmaf(dv[j], irs[j], sb);
    }
    *reinterpret_cast<uint4*>(dt + off) = E::pack(o);
    sg_[k] = sg; sb_[k] = sb;
  }
#pragma unroll
  for (int k = 0; k < LOB_CH / 8; ++k) {       // one warp sum per channel and block -> this block's row of the partials
    const float sg = warp_sum(sg_[k]), sb = warp_sum(sb_[k]);
    if (lane == 0) { part[(size_t)blockIdx.x * 2 * H + c0 + k] = sg; part[(size_t)blockIdx.x * 2 * H + H + c0 + k] = sb; }
  }
}

// out[c] = sum_r part[r][c]: block = 32 columns (lanes, coalesced) x 8 warps striding the rows, combined in a fixed order
__global__ void __launch_bounds__(256) colsum_kernel(const float* __restrict__ part, float* __restrict__ out, int R, int C) {
  __shared__ float red[8][32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, c = blockIdx.x * 32 + lane;
  float a = 0.f;
  if (c < C)
    for (int r = warp; r < R; r += 8) a += part[(size_t)r * C + c];
  red[warp][lane] = a;
  __syncthreads();
  if (warp == 0 && c < C) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w][lane];
    out[c] = s;
  }
}

// ------------------------------------------------------------------------------------------------------------------------------- lnin_bwd
// LPR lanes per row (RPW rows per warp pass), RR passes in flight: every load of the RR x RPW rows (dxn, z, dy) is issued before the first reduction.
template <int D, class E>
__global__ void __launch_bounds__(256) lnin_bwd_kernel(const typename E::T* __restrict__ dxn, const typename E::T* __restrict__ z,
                                                        const typename E::T* __restrict__ dy, const float2* __restrict__ sti, const float* __restrict__ gi,
                                                        typename E::T* __restrict__ dx, float* __restrict__ part, int T) {
  constexpr int V = E::VEC, G = D / V, LPR = G < 32 ? G : 32, RPW = 32 / LPR, NI = (G + LPR - 1) / LPR, RR = 2;
  __shared__ float cs[2][8][D];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int sub = lane / LPR, ln = lane - sub * LPR;
  float gg[NI][V], c0[NI][V], c1[NI][V];
#pragma unroll
  for (int it = 0; it < NI; ++it) {
    const int g = ln + LPR * it;
#pragma unroll
    for (int j = 0; j < V; ++j) { gg[it][j] = g < G ? gi[V * g + j] : 0.f; c0[it][j] = 0.f; c1[it][j] = 0.f; }
  }
  const int rows_per_pass = 8 * RPW;
  for (int base = blockIdx.x * rows_per_pass * RR; base < T; base += gridDim.x * rows_per_pass * RR) {
    uint4 vd[RR][NI], vx[RR][NI], vy[RR][NI];
    int rows[RR];
#pragma unroll
    for (int r = 0; r < RR; ++r) {
      rows[r] = base + r * rows_per_pass + warp * RPW + sub;
#pragma unroll
      for (int it = 0; it < NI; ++it) {
        const int g = ln + LPR * it;
        if (g < G && rows[r] < T) {
          const size_t e = (size_t)rows[r] * G + g;
          vd[r][it] = __ldg(reinterpret_cast<const uint4*>(dxn) + e);
          vx[r][it] = __ldg(reinterpret_cast<const uint4*>(z) + e);
          vy[r][it] = __ldg(reinterpret_cast<const uint4*>(dy) + e);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < RR; ++r) {
      const int row = rows[r];
      const bool live = row < T;
      const float2 st = live ? __ldg(sti + row) : make_float2(0.f, 1.f);
      float a = 0.f, b = 0.f;
#pragma unroll
      for (int it = 0; it < NI; ++it) {
        const int g = ln + LPR * it;
        if (g < G && live) {
          float fd[V], fx[V];
          E::unpack(vd[r][it], fd); E::unpack(vx[r][it], fx);
#pragma unroll
          for (int j = 0; j < V; ++j) {
            const float xh = (fx[j] - st.x) * st.y, gd = gg[it][j] * fd[j];
            a += gd; b = fmaf(gd, xh, b);
            c0[it][j] = fmaf(fd[j], xh, c0[it][j]);
            c1[it][j] += fd[j];
          }
        }
      }
#pragma unroll
      for (int o = 1; o < LPR; o <<= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); b += __shfl_xor_sync(0xffffffffu, b, o); }
      a *= 1.f / D; b *= 1.f / D;
#pragma unroll
      for (int it = 0; it < NI; ++it) {
        const int g = ln + LPR * it;
        if (g < G && live) {
          float fd[V], fx[V], fy[V], o[V];
          E::unpack(vd[r][it], fd); E::unpack(vx[r][it], fx); E::unpack(vy[r][it], fy);
#pragma unroll
          for (int j = 0; j < V; ++j) o[j] = fy[j] + st.y * (gg[it][j] * fd[j] - a - (fx[j] - st.x) * st.y * b);
          *reinterpret_cast<uint4*>(dx + (size_t)row * D + V * g) = E::pack(o);
        }
      }
    }
  }
#pragma unroll
  for (int it = 0; it < NI; ++it)
#pragma unroll
    for (int j = 0; j < V; ++j)
#pragma unroll
      for (int o = LPR; o < 32; o <<= 1) {
        c0[it][j] += __shfl_xor_sync(0xffffffffu, c0[it][j], o);
        c1[it][j] += __shfl_xor_sync(0xffffffffu, c1[it][j], o);
      }
  if (sub == 0) {
#pragma unroll
    for (int it = 0; it < NI; ++it) {
      const int g = ln + LPR * it;
      if (g < G) {
#pragma unroll
        for (int j = 0; j < V; ++j) { cs[0][warp][V * g + j] = c0[it][j]; cs[1][warp][V * g + j] = c1[it][j]; }
      }
    }
  }
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += 256) {
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) { a += cs[0][w][c]; b += cs[1][w][c]; }
    part[(size_t)blockIdx.x * 2 * D + c] = a;           // this block's row; colsum_kernel adds the rows in a fixed order
    part[(size_t)blockIdx.x * 2 * D + D + c] = b;
  }
}

// ------------------------------------------------------------------------------------------------------------------------------- ln_apply
template <int D, class E>
__global__ void __launch_bounds__(256) ln_apply_kernel(const typename E::T* __restrict__ z, const float2* __restrict__ sti, const float* __restrict__ g,
                                                        const float* __restrict__ b, typename E::T* __restrict__ xn, int T) {
  constexpr int V = E::VEC, G = D / V;
  __shared__ __align__(16) float gs[2 * D];
  for (int k = threadIdx.x; k < D; k += 256) { gs[k] = g[k]; gs[D + k] = b[k]; }
  __syncthreads();
  const size_t n = (size_t)T * G;
  for (size_t e = (size_t)blockIdx.x * 256 + threadIdx.x; e < n; e += (size_t)gridDim.x * 256) {
    const int row = (int)(e / G), c8 = (int)(e % G) * V;
    float f[V];
    E::unpack(reinterpret_cast<const uint4*>(z)[e], f);
    const float2 st = __ldg(sti + row);
#pragma unroll
    for (int j = 0; j < V; ++j) f[j] = fmaf((f[j] - st.x) * st.y, gs[c8 + j], gs[D + c8 + j]);
    reinterpret_cast<uint4*>(xn)[e] = E::pack(f);
  }
}

}  // namespace a100
