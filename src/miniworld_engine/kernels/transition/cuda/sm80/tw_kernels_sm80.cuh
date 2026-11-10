// tw_kernels_sm80.cuh -- the kernels of the wide-width A100 (sm_80) Transition: any D that is a multiple of 8 for the row kernels, a multiple of 64 for the GEMM
// kernels, any hidden width H that is a multiple of 64, any row count M (tails are zero-filled / predicated).  Everything bf16 in, fp32 statistics.
//
//   y = x + Ws (silu(Wa xn) * (Wb xn)),  xn = LN(x) gamma + beta                    (the FFN has neither the LayerNorm nor the residual)
//
//   ln_fwd_kernel      x -> xn (bf16), (mean, rstd) fp32           one 16-byte chunk per lane, the two passes over the row in registers
//   dual_swiglu_kernel xn, Wa, Wb -> h = rn(silu(a) b)             the dual-B tile GEMM (a rows | b rows of the same hidden units per warp), SwiGLU in the accumulators
//   gate_bwd_kernel    xn, Wa, Wb, dh -> dA, dB, h                  the same GEMM, the SwiGLU backward as the epilogue (dh is prefetched into shared memory)
//   ln_bwd_kernel      d_xn (fp32), x, (mean, rstd), gamma, dy -> dx = rstd (g - mean(g) - xhat mean(g xhat)) + dy, dgamma / dbeta partials per CTA
//   ln_finalize_kernel the partials -> dgamma, dbeta in a fixed order
// The squeeze GEMM (+ residual) and the weight / d_xn GEMMs are cuBLAS (transition_wide_sm80.cu).
// The fp32 Transition (F16Of configs, F32IO row kernels): the same kernels on fp16 operands with device scales, gate_f16_kernel (dh and a | b in one CTA), the
// fp16 forms (f32_forms_*), dy_absmax / dy_convert (the backward's dy in fp16) and split_res (the few-rows split-K squeeze's reduction).
#pragma once
#include <cuda_fp16.h>

#include <type_traits>

#include "tw_gemm_sm80.cuh"

#ifndef TW_ROUND_SWIGLU
#define TW_ROUND_SWIGLU 0
#endif

namespace a100 {

// ===================================================================================================================== LayerNorm forward
// F32IO (the fp32 Transition): x fp32, xn written as fp16 xn * sc[0] (the scale of the fp16 operand).
template <int D, bool F32IO = false>
__global__ void __launch_bounds__(256) ln_fwd_kernel(const void* __restrict__ x_, const float* __restrict__ gamma, const float* __restrict__ beta,
                                                     __nv_bfloat16* __restrict__ xn, float2* __restrict__ stats, int M, float eps,
                                                     const float* __restrict__ sc = nullptr) {
  const float s_x = F32IO ? __ldg(sc) : 1.f;
  constexpr int CH = D / 8, LPR = CH < 32 ? CH : 32, RPW = 32 / LPR, XPL = (CH + LPR - 1) / LPR;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, sub = lane / LPR, cl = lane % LPR;
  for (int base = (blockIdx.x * 8 + warp) * RPW; base < M; base += gridDim.x * 8 * RPW) {
    const int row = base + sub;
    const bool valid = row < M;
    float v[XPL][8];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < XPL; ++i) {
      const int c = cl + i * LPR;
      if constexpr (F32IO) {
        float4 f0 = make_float4(0.f, 0.f, 0.f, 0.f), f1 = f0;
        if (valid && c < CH) {
          const float* xr = reinterpret_cast<const float*>(x_) + (size_t)row * D + c * 8;
          f0 = *reinterpret_cast<const float4*>(xr);
          f1 = *reinterpret_cast<const float4*>(xr + 4);
        }
        const float f[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) { v[i][e] = f[e]; s += f[e]; }
      } else {
        uint4 u = make_uint4(0u, 0u, 0u, 0u);
        if (valid && c < CH) u = *reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(x_) + (size_t)row * D + c * 8);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { v[i][2 * e] = bf16lo(w[e]); v[i][2 * e + 1] = bf16hi(w[e]); s += v[i][2 * e] + v[i][2 * e + 1]; }
      }
    }
#pragma unroll
    for (int o = LPR / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s * (1.f / D);
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < XPL; ++i) {
      const int c = cl + i * LPR;
#pragma unroll
      for (int e = 0; e < 8; ++e) { v[i][e] -= mean; if (c < CH) q = fmaf(v[i][e], v[i][e], q); }
    }
#pragma unroll
    for (int o = LPR / 2; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rstd = rsqrtf(q * (1.f / D) + eps);
    if (valid && cl == 0 && stats != nullptr) stats[row] = make_float2(mean, rstd);
#pragma unroll
    for (int i = 0; i < XPL; ++i) {
      const int c = cl + i * LPR;
      if (valid && c < CH) {
        const float4 g0 = *reinterpret_cast<const float4*>(gamma + c * 8), g1 = *reinterpret_cast<const float4*>(gamma + c * 8 + 4);
        const float4 b0 = *reinterpret_cast<const float4*>(beta + c * 8), b1 = *reinterpret_cast<const float4*>(beta + c * 8 + 4);
        const float gg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w}, bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        uint32_t o[4];
        if constexpr (F32IO) {
          float y[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) y[e] = fmaf(v[i][e] * rstd, gg[e], bb[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) o[e] = pack_f16(y[2 * e] * s_x, y[2 * e + 1] * s_x);
        } else {
#pragma unroll
          for (int e = 0; e < 4; ++e)
            o[e] = pack_bf16(fmaf(v[i][2 * e] * rstd, gg[2 * e], bb[2 * e]), fmaf(v[i][2 * e + 1] * rstd, gg[2 * e + 1], bb[2 * e + 1]));
        }
        *reinterpret_cast<uint4*>(xn + (size_t)row * D + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
      }
    }
  }
}

// ===================================================================================================================== LayerNorm backward
// One row per warp (lanes spread over the chunks) or several rows per warp for D <= 128; dgamma / dbeta are summed per lane over the rows the lane's warp visits, then per CTA.

// F32IO (the fp32 Transition): x, dy and dx fp32.
template <int D, bool RES = true, bool F32IO = false>
__global__ void __launch_bounds__(128) ln_bwd_kernel(const float* __restrict__ dxn, const void* __restrict__ x_, const float2* __restrict__ stats,
                                                     const float* __restrict__ gamma, const void* __restrict__ dy_, void* __restrict__ dx_,
                                                     float* __restrict__ part, int M, const float* __restrict__ dscale = nullptr) {
  // F32IO: dscale (if given) multiplies dxn (the fp16 d_xn GEMM's output scale)
  const float ds = (F32IO && dscale != nullptr) ? __ldg(dscale) : 1.f;
  using T = typename std::conditional<F32IO, float, __nv_bfloat16>::type;
  const T* x = reinterpret_cast<const T*>(x_);
  const T* dy = reinterpret_cast<const T*>(dy_);
  T* dx = reinterpret_cast<T*>(dx_);
  constexpr int CH = D / 8, LPR = CH < 32 ? CH : 32, RPW = 32 / LPR, XPL = (CH + LPR - 1) / LPR;
  __shared__ float sred[4][2][D];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, sub = lane / LPR, cl = lane % LPR;
  float dg[XPL][8], db[XPL][8];
#pragma unroll
  for (int i = 0; i < XPL; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) dg[i][e] = db[i][e] = 0.f;
  for (int base = (blockIdx.x * 4 + warp) * RPW; base < M; base += gridDim.x * 4 * RPW) {
    const int row = base + sub;
    const bool valid = row < M;
    const float2 st = valid ? stats[row] : make_float2(0.f, 1.f);
    float xh[XPL][8], g[XPL][8], dn[XPL][8];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int i = 0; i < XPL; ++i) {
      const int c = cl + i * LPR;
      uint4 u = make_uint4(0u, 0u, 0u, 0u);
      float4 d0 = make_float4(0.f, 0.f, 0.f, 0.f), d1 = d0, g0 = d0, g1 = d0, x0 = d0, x1 = d0;
      if (valid && c < CH) {
        if constexpr (F32IO) {
          x0 = *reinterpret_cast<const float4*>(x + (size_t)row * D + c * 8);
          x1 = *reinterpret_cast<const float4*>(x + (size_t)row * D + c * 8 + 4);
        } else {
          u = *reinterpret_cast<const uint4*>(x + (size_t)row * D + c * 8);
        }
        d0 = *reinterpret_cast<const float4*>(dxn + (size_t)row * D + c * 8);
        d1 = *reinterpret_cast<const float4*>(dxn + (size_t)row * D + c * 8 + 4);
        if constexpr (F32IO) {
          d0.x *= ds; d0.y *= ds; d0.z *= ds; d0.w *= ds; d1.x *= ds; d1.y *= ds; d1.z *= ds; d1.w *= ds;
        }
        g0 = *reinterpret_cast<const float4*>(gamma + c * 8);
        g1 = *reinterpret_cast<const float4*>(gamma + c * 8 + 4);
      }
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};

      const float dd[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w}, gm[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
      const float xf[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int j = 2 * e + k;
          const float xv = F32IO ? xf[j] : (k ? bf16hi(w[e]) : bf16lo(w[e]));
          xh[i][j] = (xv - st.x) * st.y;
          dn[i][j] = dd[j];
          g[i][j] = gm[j] * dd[j];
          s1 = fmaf(g[i][j], xh[i][j], s1);
          s2 += g[i][j];
          dg[i][j] = fmaf(dd[j], xh[i][j], dg[i][j]);
          db[i][j] += dd[j];
        }
      }
    }
#pragma unroll
    for (int o = LPR / 2; o > 0; o >>= 1) { s1 += __shfl_xor_sync(0xffffffffu, s1, o); s2 += __shfl_xor_sync(0xffffffffu, s2, o); }
    const float c1 = s1 * (1.f / D), c2 = s2 * (1.f / D);
#pragma unroll
    for (int i = 0; i < XPL; ++i) {
      const int c = cl + i * LPR;
      if constexpr (F32IO) {
        if (valid && c < CH) {
          float yv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
          if (RES) {
            const float4 y0 = *reinterpret_cast<const float4*>(dy + (size_t)row * D + c * 8), y1 = *reinterpret_cast<const float4*>(dy + (size_t)row * D + c * 8 + 4);
            yv[0] = y0.x; yv[1] = y0.y; yv[2] = y0.z; yv[3] = y0.w; yv[4] = y1.x; yv[5] = y1.y; yv[6] = y1.z; yv[7] = y1.w;
          }
          float o[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) o[e] = fmaf(g[i][e] - fmaf(xh[i][e], c1, c2), st.y, yv[e]);
          *reinterpret_cast<float4*>(dx + (size_t)row * D + c * 8) = make_float4(o[0], o[1], o[2], o[3]);
          *reinterpret_cast<float4*>(dx + (size_t)row * D + c * 8 + 4) = make_float4(o[4], o[5], o[6], o[7]);
        }
      } else if (valid && c < CH) {
        const uint4 y = RES ? *reinterpret_cast<const uint4*>(dy + (size_t)row * D + c * 8) : make_uint4(0u, 0u, 0u, 0u);   // RES: dx carries the residual branch dy
        const uint32_t yw[4] = {y.x, y.y, y.z, y.w};
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float v0 = fmaf(g[i][2 * e] - fmaf(xh[i][2 * e], c1, c2), st.y, bf16lo(yw[e]));
          const float v1 = fmaf(g[i][2 * e + 1] - fmaf(xh[i][2 * e + 1], c1, c2), st.y, bf16hi(yw[e]));
          o[e] = pack_bf16(v0, v1);
        }
        *reinterpret_cast<uint4*>(dx + (size_t)row * D + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
      }
    }
  }
  // column sums: lanes sharing a chunk (RPW > 1) first, then the 8 warps of the CTA through shared memory in a fixed order
  if (RPW > 1) {
#pragma unroll
    for (int i = 0; i < XPL; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e)
#pragma unroll
        for (int o = LPR; o < 32; o <<= 1) { dg[i][e] += __shfl_xor_sync(0xffffffffu, dg[i][e], o); db[i][e] += __shfl_xor_sync(0xffffffffu, db[i][e], o); }
  }
  if (sub == 0) {
#pragma unroll
    for (int i = 0; i < XPL; ++i) {
      const int c = cl + i * LPR;
      if (c < CH) {
#pragma unroll
        for (int e = 0; e < 8; ++e) { sred[warp][0][c * 8 + e] = dg[i][e]; sred[warp][1][c * 8 + e] = db[i][e]; }
      }
    }
  }
  __syncthreads();
  for (int k = threadIdx.x; k < 2 * D; k += 128) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) s += sred[w][k / D][k % D];
    part[(size_t)blockIdx.x * 2 * D + k] = s;
  }
}

template <class TA>
DEVI void put(TA& o, float v);
template <> DEVI void put<float>(float& o, float v) { o = v; }
template <> DEVI void put<__nv_bfloat16>(__nv_bfloat16& o, float v) { o = __float2bfloat16_rn(v); }

// dgamma | dbeta [2][D] = the sums over the nblk partials of ln_bwd_kernel
// 32 columns per CTA, the partials split over 8 warps (warp g sums b = g, g + 8, ... in order; the 8 sums are added in warp order: a fixed order, no atomics)
// The fp32 Transition also hands it the weight gradients' scaling (scale2_rows, blocks from fin_blocks on): one launch for both.
struct Scale2 {
  float4* a = nullptr; long long na4 = 0; const float* sa = nullptr;
  float4* b = nullptr; long long nb4 = 0; const float* sb = nullptr;
  int fin_blocks = 0;   // 0: no scaling blocks
};
DEVI void scale2_rows(const Scale2& s, long long i0, long long step) {
  const float x = __ldg(s.sa), y = __ldg(s.sb);
  for (long long i = i0; i < s.na4 + s.nb4; i += step) {
    const bool first = i < s.na4;
    float4* p = first ? s.a + i : s.b + (i - s.na4);
    const float f = first ? x : y;
    float4 v = *p;
    v.x *= f; v.y *= f; v.z *= f; v.w *= f;
    *p = v;
  }
}
template <class TA>
__global__ void __launch_bounds__(256) ln_finalize_kernel(const float* __restrict__ part, int nblk, int D, TA* __restrict__ dgamma, TA* __restrict__ dbeta,
                                                          const Scale2 sj = Scale2()) {
  if (sj.fin_blocks > 0 && (int)blockIdx.x >= sj.fin_blocks) {
    scale2_rows(sj, (long long)(blockIdx.x - sj.fin_blocks) * blockDim.x + threadIdx.x, (long long)(gridDim.x - sj.fin_blocks) * blockDim.x);
    return;
  }
  __shared__ float red[8][33];
  const int c = threadIdx.x & 31, g = threadIdx.x >> 5, k = blockIdx.x * 32 + c;
  float s = 0.f;
  if (k < 2 * D)
    for (int b = g; b < nblk; b += 8) s += part[(size_t)b * 2 * D + k];
  red[g][c] = s;
  __syncthreads();
  if (g == 0 && k < 2 * D) {
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) t += red[i][c];
    if (k < D) put(dgamma[k], t);
    else put(dbeta[k - D], t);
  }
}

// y = rn(y + res) in place on 16-byte vectors (the residual of a cuBLAS squeeze: the GEMM's own bf16 rounding comes first, as in Triton's ``mm(...).to(dtype) + residual``)
__global__ void __launch_bounds__(256) add_res_kernel(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ res, int64_t nvec) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += (int64_t)gridDim.x * blockDim.x) {
    uint4 a = reinterpret_cast<const uint4*>(y)[i];
    const uint4 b = reinterpret_cast<const uint4*>(res)[i];
    uint32_t aw[4] = {a.x, a.y, a.z, a.w};
    const uint32_t bw[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) aw[e] = pack_bf16(bf16lo(aw[e]) + bf16lo(bw[e]), bf16hi(aw[e]) + bf16hi(bw[e]));
    reinterpret_cast<uint4*>(y)[i] = make_uint4(aw[0], aw[1], aw[2], aw[3]);
  }
}

// ===================================================================================================================== dual-B GEMM + SwiGLU
struct DualParams {
  const __nv_bfloat16* a;      // [M][K]  (xn, or x for the FFN)
  const __nv_bfloat16* wa;     // [H][K]
  const __nv_bfloat16* wb;     // [H][K]
  const __nv_bfloat16* dh;     // gate backward: [M][H]
  __nv_bfloat16* h;            // [M][H]  out
  __nv_bfloat16* dab;          // gate backward: [M][2H] out (dA | dB)
  __nv_bfloat16* ab;           // forward (training): [M][2H] out (a | b, rounded to bf16) or nullptr; gate_dh: [M][2H] in
  const __nv_bfloat16* wsT;    // gate_dh: Ws^T [H][K] (K = D)
  // the fp32 Transition (fp16 operands in the 16-bit slots above: a = fp16 xn * sc[0], wa / wb = fp16 W / s_w): sc = [1 / s_x, s_x s_w, 1 / s_h, s_h s_ws]
  const float* sc;             // the forward / gate_f16: [8] (f32_forms_apply_kernel)
  const __nv_bfloat16* a2;     // gate_f16: dy16 [M][K] (fp16 dy / s_dy); wsT: Ws16^T [H][K]
  const float* bs;             // gate_f16: the backward scales [8] (dy_convert_kernel)
  int M, K, H;
};

// Shared geometry of the two kernels: CTA tile BM rows x HN hidden units (B tile = HN a rows + HN b rows, per warp: its HN / WN a rows then the same b rows).
template <int BM, int HN, int WM, int WN, int ST, int BK = 32>
struct DualCfg : GCfg<BM, 2 * HN, BK, WM, WN, ST, false> {
  using G = GCfg<BM, 2 * HN, BK, WM, WN, ST, false>;
  static constexpr int HPW = HN / WN;                       // hidden units per warp
  static_assert(HPW % 8 == 0 && G::NT == 2 * (HPW / 8), "dual tiling");
  static constexpr int CPH = HN / 8;                        // 16-byte chunks per staged output row
  static constexpr int OUT_TILE = BM * HN * 2;              // one staged bf16 [BM][HN] tile
  static constexpr int GATE_WIN = (G::SMEM > 3 * OUT_TILE ? G::SMEM : 3 * OUT_TILE);   // gate backward: the window holding the pipeline stages, then the three staged outputs
  static constexpr int GATE_SMEM = GATE_WIN + OUT_TILE;     // ... plus the prefetched dh tile
};

template <class C>
DEVI void dual_mainloop(float (&acc)[C::MT][C::NT][4], uint32_t smem, const DualParams& p, int m0, int hn0, const Frag<C>& fr, int tid) {
  const int KT = p.K / C::BK;
  if (m0 + C::BM <= p.M) {                                  // a whole tile: the lean loop (per-thread source pointers computed once, no predicates)
    using L = LeanCopies<C>;
    const char* pa[L::NJA];
    const char* pb[L::NJB];
    const int ra = tid / C::CPA, ca = tid % C::CPA, rb = tid / C::CPB, cb = tid % C::CPB;
#pragma unroll
    for (int j = 0; j < L::NJA; ++j) pa[j] = reinterpret_cast<const char*>(p.a + (size_t)(m0 + ra + j * L::RPJA) * p.K) + ca * 16;
#pragma unroll
    for (int j = 0; j < L::NJB; ++j) {
      const int r = rb + j * L::RPJB, w = r % (2 * C::HPW), wn = r / (2 * C::HPW);
      pb[j] = reinterpret_cast<const char*>((w < C::HPW ? p.wa : p.wb) + (size_t)(hn0 + wn * C::HPW + (w % C::HPW)) * p.K) + cb * 16;
    }
    gemm_mainloop_lean<C>(acc, smem, KT, pa, pb, LeanFrag<C>(fr.wm, fr.wn, tid & 31), tid);
    return;
  }
  auto load_a = [&](uint32_t stage, int kt) {
    load_rows_k<C::BM, C::CPA, C::NTHR>(stage, kt * C::BK, [&](int r) { return m0 + r < p.M ? p.a + (size_t)(m0 + r) * p.K : nullptr; }, tid);
  };
  auto load_b = [&](uint32_t stage, int kt) {
    load_rows_k<C::BN, C::CPB, C::NTHR>(stage, kt * C::BK, [&](int r) {
      const int w = r % (2 * C::HPW), wn = r / (2 * C::HPW);
      const int hid = hn0 + wn * C::HPW + (w % C::HPW);
      return (w < C::HPW ? p.wa : p.wb) + (size_t)hid * p.K;
    }, tid);
  };
  gemm_mainloop<C>(acc, smem, KT, load_a, load_b, fr);
}

template <class C, int EPI = 0>
__global__ void __launch_bounds__(C::NTHR, C::MINB) dual_swiglu_kernel(const DualParams p) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  const uint32_t smem = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q = lane & 3;
  const int wm = warp / C::WN, wn = warp % C::WN;
  const int hn0 = blockIdx.x * C::HPW * C::WN, m0 = blockIdx.y * C::BM;
  Frag<C> fr(wm, wn, lane);
  float acc[C::MT][C::NT][4];
  zero_acc<C>(acc);
  dual_mainloop<C>(acc, smem, p, m0, hn0, fr, tid);

  // SwiGLU: a = acc[mt][j], b = acc[mt][NT / 2 + j]; h = rn(silu(a) b) staged as [BM][HN] bf16 (swizzled); p.ab != nullptr (training): a and b are staged too ([h | a | b])
  // C::F16 (the fp32 Transition): a = acc s_x s_w, b likewise, h = silu(a) b in full fp32 precision, staged as fp16 h / s_h
  const bool save_ab = !C::F16 && p.ab != nullptr;
  const uint32_t sa_ = smem + C::OUT_TILE, sb_ = smem + 2 * C::OUT_TILE;
  const float s_ab = C::F16 ? __ldg(p.sc + 1) : 1.f, s_h = C::F16 ? __ldg(p.sc + 2) : 1.f;
#pragma unroll
  for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
    for (int j = 0; j < C::NT / 2; ++j)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float hv[2];
        if (save_ab) {
          const uint32_t row = (uint32_t)(wm * C::MT * 16 + 16 * mt + 8 * hh + g8), chunk = (uint32_t)(wn * (C::HPW / 8) + j);
          const uint32_t off = swz_off<C::CPH>(row, chunk) + 4 * q;
          sts32(sa_ + off, pack_bf16(acc[mt][j][2 * hh], acc[mt][j][2 * hh + 1]));
          sts32(sb_ + off, pack_bf16(acc[mt][C::NT / 2 + j][2 * hh], acc[mt][C::NT / 2 + j][2 * hh + 1]));
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          float av = acc[mt][j][2 * hh + k], bv = acc[mt][C::NT / 2 + j][2 * hh + k];
          if constexpr (C::F16) {
            static_assert(EPI == 0, "the fp32 forward is the SwiGLU");
            av *= s_ab; bv *= s_ab;
            hv[k] = av * sigmoid_f32(av) * bv * s_h;
            continue;
          }
          if constexpr (TW_ROUND_SWIGLU && EPI == 0) {
            if (p.K == 256) { av = round_bf16f(av); bv = round_bf16f(bv); }
          }
          hv[k] = EPI == 0 ? av * sigmoid(av) * bv : av * sigmoid(bv);                 // EPI 0: silu(a) b (SwiGLU);  1: a sigmoid(b) (the triangle-multiplication gate)
          if constexpr (TW_ROUND_SWIGLU && EPI == 0) {
            if (p.K == 256) hv[k] = round_bf16f(av * sigmoid(av)) * bv;
          }
        }
        const uint32_t row = (uint32_t)(wm * C::MT * 16 + 16 * mt + 8 * hh + g8), chunk = (uint32_t)(wn * (C::HPW / 8) + j);
        sts32(smem + swz_off<C::CPH>(row, chunk) + 4 * q, C::F16 ? pack_f16(hv[0], hv[1]) : pack_bf16(hv[0], hv[1]));
      }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < (C::BM * C::CPH) / C::NTHR; ++j) {
    const int i = tid + j * C::NTHR, r = i / C::CPH, c = i % C::CPH;
    if (m0 + r < p.M) {
      const uint32_t off = swz_off<C::CPH>(r, c);
      stg128(p.h + (size_t)(m0 + r) * p.H + (size_t)hn0 + c * 8, lds128(smem + off));
      if (save_ab) {
        stg128(p.ab + (size_t)(m0 + r) * 2 * p.H + (size_t)hn0 + c * 8, lds128(sa_ + off));
        stg128(p.ab + (size_t)(m0 + r) * 2 * p.H + p.H + (size_t)hn0 + c * 8, lds128(sb_ + off));
      }
    }
  }
}

// ===================================================================================================================== gate backward
// dh arrives as a [BM][HN] bf16 tile prefetched into shared memory behind the pipeline window; per element (a, b, d = dh):
//   sig = sigmoid(a), silu = a sig, h = silu b, dB = d silu, dA = d b (sig + silu (1 - sig))      (h, dA, dB rounded to bf16)
template <class C, int EPI = 0>
__global__ void __launch_bounds__(C::NTHR, C::MINB) gate_bwd_kernel(const DualParams p) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  const uint32_t smem = smem_u32(smem_raw), sdh = smem + C::GATE_WIN;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q = lane & 3;
  const int wm = warp / C::WN, wn = warp % C::WN;
  const int hn0 = blockIdx.x * C::HPW * C::WN, m0 = blockIdx.y * C::BM;
  Frag<C> fr(wm, wn, lane);
  float acc[C::MT][C::NT][4];
  zero_acc<C>(acc);
  // the dh tile joins the first cp.async group (it is waited on by the mainloop's first barrier)
  load_rows_k<C::BM, C::CPH, C::NTHR>(sdh, 0, [&](int r) { return m0 + r < p.M ? p.dh + (size_t)(m0 + r) * p.H + hn0 : nullptr; }, tid);
  dual_mainloop<C>(acc, smem, p, m0, hn0, fr, tid);

  // the three outputs are staged in the (free) pipeline window: [h | dA | dB] x [BM][HN]
  const uint32_t sh = smem, sda = smem + C::OUT_TILE, sdb = smem + 2 * C::OUT_TILE;
#pragma unroll
  for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
    for (int j = 0; j < C::NT / 2; ++j)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint32_t row = (uint32_t)(wm * C::MT * 16 + 16 * mt + 8 * hh + g8), chunk = (uint32_t)(wn * (C::HPW / 8) + j);
        const uint32_t off = swz_off<C::CPH>(row, chunk) + 4 * q;
        const uint32_t dd = lds32(sdh + off);
        const float dv[2] = {bf16lo(dd), bf16hi(dd)};
        float hv[2], av2[2], bv2[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          float av = acc[mt][j][2 * hh + k], bv = acc[mt][C::NT / 2 + j][2 * hh + k];
          if constexpr (TW_ROUND_SWIGLU && EPI == 0) {
            if (p.K == 256) { av = round_bf16f(av); bv = round_bf16f(bv); }
          }
          if (EPI == 0) {
            const float sg = sigmoid(av), silu = av * sg;
            hv[k] = silu * bv;
            bv2[k] = dv[k] * silu;
            av2[k] = dv[k] * bv * fmaf(silu, 1.f - sg, sg);
            if constexpr (TW_ROUND_SWIGLU) {
              if (p.K == 256) {
                const float rounded = round_bf16f(silu);
                hv[k] = rounded * bv;
                bv2[k] = dv[k] * rounded;
                av2[k] = round_bf16f(dv[k] * bv) * fmaf(silu, 1.f - sg, sg);
              }
            }
          } else {                                                 // out = a sigmoid(b): d_a = d sigmoid(b), d_b = d a sigmoid(b) (1 - sigmoid(b))
            const float sg = sigmoid(bv);
            hv[k] = av * sg;
            av2[k] = dv[k] * sg;
            bv2[k] = dv[k] * av * sg * (1.f - sg);
          }
        }
        __syncwarp();
        sts32(sh + off, pack_bf16(hv[0], hv[1]));
        sts32(sda + off, pack_bf16(av2[0], av2[1]));
        sts32(sdb + off, pack_bf16(bv2[0], bv2[1]));
      }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < (C::BM * C::CPH) / C::NTHR; ++j) {
    const int i = tid + j * C::NTHR, r = i / C::CPH, c = i % C::CPH;
    if (m0 + r < p.M) {
      const uint32_t off = swz_off<C::CPH>(r, c);
      stg128(p.h + (size_t)(m0 + r) * p.H + (size_t)hn0 + c * 8, lds128(sh + off));
      stg128(p.dab + (size_t)(m0 + r) * 2 * p.H + (size_t)hn0 + c * 8, lds128(sda + off));
      stg128(p.dab + (size_t)(m0 + r) * 2 * p.H + p.H + (size_t)hn0 + c * 8, lds128(sdb + off));
    }
  }
}

// ===================================================================================================================== gate backward with the dh GEMM, from the saved a | b
// dh = dy Ws is the kernel's own GEMM (A = dy [M][D], B = Ws^T [H][D], one B row per hidden unit); the saved a, b tiles are prefetched into shared memory behind the pipeline
// window exactly like the dh tile of gate_bwd_kernel; the epilogue is the SwiGLU backward: dB = dh silu(a), dA = dh b (sig + silu (1 - sig)), both rounded to bf16 into dab [M][2H].
#ifndef GATE_DH_PIECES
#define GATE_DH_PIECES 8   // gate_dh: the saved a | b tiles are prefetched in this many pieces, one per k slice (0 pieces per slice beyond the count)
#endif

template <int BM, int HN, int WM, int WN, int ST, int BK = 32>
struct DhCfg : GCfg<BM, HN, BK, WM, WN, ST, false> {
  using G = GCfg<BM, HN, BK, WM, WN, ST, false>;
  static constexpr int CPH = HN / 8;
  static constexpr int OUT_TILE = BM * HN * 2;
  static constexpr int WIN = (G::SMEM > 2 * OUT_TILE ? G::SMEM : 2 * OUT_TILE);          // pipeline stages, then the staged dA | dB
  static constexpr int TOTAL_SMEM = WIN + 2 * OUT_TILE;                                  // ... plus the prefetched a | b tiles
};

template <class C>
__global__ void __launch_bounds__(C::NTHR, C::MINB) gate_dh_kernel(const DualParams p) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  const uint32_t smem = smem_u32(smem_raw), sa = smem + C::WIN, sb = sa + C::OUT_TILE;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q = lane & 3;
  const int wm = warp / C::WN, wn = warp % C::WN;
  const int hn0 = blockIdx.x * C::BN, m0 = blockIdx.y * C::BM;
  Frag<C> fr(wm, wn, lane);
  float acc[C::MT][C::NT][4];
  zero_acc<C>(acc);
  const int KT = p.K / C::BK;
  if (m0 + C::BM <= p.M) {
    using L = LeanCopies<C>;
    const char* pa[L::NJA];
    const char* pb[L::NJB];
    const int ra = tid / C::CPA, ca = tid % C::CPA, rb = tid / C::CPB, cb = tid % C::CPB;
#pragma unroll
    for (int j = 0; j < L::NJA; ++j) pa[j] = reinterpret_cast<const char*>(p.a + (size_t)(m0 + ra + j * L::RPJA) * p.K) + ca * 16;
#pragma unroll
    for (int j = 0; j < L::NJB; ++j) pb[j] = reinterpret_cast<const char*>(p.wsT + (size_t)(hn0 + rb + j * L::RPJB) * p.K) + cb * 16;
    // the saved a | b tiles stream in pieces behind the mainloop (one burst in front of slice 0 stalled the first MMAs and made the CTAs' load / MMA / store phases coincide)
    constexpr int NP = GATE_DH_PIECES, PC = 2 * C::BM * C::CPH / NP;
    static_assert((2 * C::BM * C::CPH) % (NP * C::NTHR) == 0, "a | b prefetch pieces");
    auto pre = [&](int kt) {
      for (int pc = kt; pc < NP; pc += KT) {
#pragma unroll
        for (int j = 0; j < PC / C::NTHR; ++j) {
          const int i = pc * PC + tid + j * C::NTHR, tile = i / (C::BM * C::CPH), rem = i % (C::BM * C::CPH), r = rem / C::CPH, c = rem % C::CPH;
          cp_async16((tile ? sb : sa) + swz_off<C::CPH>(r, c), p.ab + (size_t)(m0 + r) * 2 * p.H + (size_t)tile * p.H + hn0 + c * 8, 16u);
        }
      }
    };
    gemm_mainloop_lean<C>(acc, smem, KT, pa, pb, LeanFrag<C>(fr.wm, fr.wn, lane), tid, pre);
  } else {
    load_rows_k<C::BM, C::CPH, C::NTHR>(sa, 0, [&](int r) { return m0 + r < p.M ? p.ab + (size_t)(m0 + r) * 2 * p.H + hn0 : nullptr; }, tid);
    load_rows_k<C::BM, C::CPH, C::NTHR>(sb, 0, [&](int r) { return m0 + r < p.M ? p.ab + (size_t)(m0 + r) * 2 * p.H + p.H + hn0 : nullptr; }, tid);
    auto load_a = [&](uint32_t stage, int kt) {
      load_rows_k<C::BM, C::CPA, C::NTHR>(stage, kt * C::BK, [&](int r) { return m0 + r < p.M ? p.a + (size_t)(m0 + r) * p.K : nullptr; }, tid);
    };
    auto load_b = [&](uint32_t stage, int kt) {
      load_rows_k<C::BN, C::CPB, C::NTHR>(stage, kt * C::BK, [&](int r) { return p.wsT + (size_t)(hn0 + r) * p.K; }, tid);
    };
    gemm_mainloop<C>(acc, smem, KT, load_a, load_b, fr);
  }
  const uint32_t sda = smem, sdb = smem + C::OUT_TILE;
#pragma unroll
  for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < C::NT; ++nt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint32_t row = (uint32_t)(wm * C::MT * 16 + 16 * mt + 8 * hh + g8), chunk = (uint32_t)(wn * C::NT + nt);
        const uint32_t off = swz_off<C::CPH>(row, chunk) + 4 * q;
        const uint32_t aa = lds32(sa + off), bb = lds32(sb + off);
        const float av[2] = {bf16lo(aa), bf16hi(aa)}, bv[2] = {bf16lo(bb), bf16hi(bb)};
        float da[2], db[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const float dv = acc[mt][nt][2 * hh + k];
          const float sg = sigmoid(av[k]), silu = av[k] * sg;
          db[k] = dv * silu;
          da[k] = dv * bv[k] * fmaf(silu, 1.f - sg, sg);
        }
        sts32(sda + off, pack_bf16(da[0], da[1]));
        sts32(sdb + off, pack_bf16(db[0], db[1]));
      }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < (C::BM * C::CPH) / C::NTHR; ++j) {
    const int i = tid + j * C::NTHR, r = i / C::CPH, c = i % C::CPH;
    if (m0 + r < p.M) {
      const uint32_t off = swz_off<C::CPH>(r, c);
      stg128(p.dab + (size_t)(m0 + r) * 2 * p.H + (size_t)hn0 + c * 8, lds128(sda + off));
      stg128(p.dab + (size_t)(m0 + r) * 2 * p.H + p.H + (size_t)hn0 + c * 8, lds128(sdb + off));
    }
  }
}

// ===================================================================================================================== gate backward, fp16 (the fp32 Transition, many rows)
// Two fp16 GEMMs per [BM x HN] tile, one after the other on the same shared-memory ring: dh = dy16 Ws16 (A = dy16 [M][D], B = Ws16^T [H][D]; dh = acc s_dy s_ws) and
// a | b = xn16 [Wa16; Wb16]^T (the forward's dual GEMM, a = acc s_x s_w). The SwiGLU backward in fp32, the outputs fp16 for the fp16 cuBLAS weight / d_xn GEMMs:
//   h16 = fp16(silu(a) b / s_h)  [M][H],  dab16 = fp16([dA | dB] / s_dab)  [M][2H]
template <class C>
__global__ void __launch_bounds__(C::NTHR, C::MINB) gate_f16_kernel(const DualParams p) {
  static_assert(C::F16, "gate_f16: fp16 operands");
  using Dh = F16Of<GCfg<C::BM, C::HPW * C::WN, C::BK, C::WM, C::WN, C::ST, false>>;
  static_assert(Dh::MT == C::MT && 2 * Dh::NT == C::NT, "the dh tile is the a (or b) half of the dual tile");
  extern __shared__ __align__(128) uint8_t smem_raw[];
  const uint32_t smem = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q = lane & 3;
  const int wm = warp / C::WN, wn = warp % C::WN;
  const int hn0 = blockIdx.x * C::HPW * C::WN, m0 = blockIdx.y * C::BM;
  float dacc[Dh::MT][Dh::NT][4];
  zero_acc<Dh>(dacc);
  const int KT = p.K / C::BK;
  if (m0 + C::BM <= p.M) {
    using L = LeanCopies<Dh>;
    const char* pa[L::NJA];
    const char* pb[L::NJB];
    const int ra = tid / Dh::CPA, ca = tid % Dh::CPA, rb = tid / Dh::CPB, cb = tid % Dh::CPB;
#pragma unroll
    for (int j = 0; j < L::NJA; ++j) pa[j] = reinterpret_cast<const char*>(p.a2 + (size_t)(m0 + ra + j * L::RPJA) * p.K) + ca * 16;
#pragma unroll
    for (int j = 0; j < L::NJB; ++j) pb[j] = reinterpret_cast<const char*>(p.wsT + (size_t)(hn0 + rb + j * L::RPJB) * p.K) + cb * 16;
    gemm_mainloop_lean<Dh>(dacc, smem, KT, pa, pb, LeanFrag<Dh>(wm, wn, lane), tid);
  } else {
    auto load_a = [&](uint32_t stage, int kt) {
      load_rows_k<Dh::BM, Dh::CPA, Dh::NTHR>(stage, kt * Dh::BK, [&](int r) { return m0 + r < p.M ? p.a2 + (size_t)(m0 + r) * p.K : nullptr; }, tid);
    };
    auto load_b = [&](uint32_t stage, int kt) {
      load_rows_k<Dh::BN, Dh::CPB, Dh::NTHR>(stage, kt * Dh::BK, [&](int r) { return p.wsT + (size_t)(hn0 + r) * p.K; }, tid);
    };
    gemm_mainloop<Dh>(dacc, smem, KT, load_a, load_b, Frag<Dh>(wm, wn, lane));
  }
  Frag<C> fr(wm, wn, lane);
  float acc[C::MT][C::NT][4];
  zero_acc<C>(acc);
  dual_mainloop<C>(acc, smem, p, m0, hn0, fr, tid);
  const float s_ab = __ldg(p.sc + 1), s_h = __ldg(p.sc + 2), s_dh = __ldg(p.bs + 5), inv_dab = __ldg(p.bs + 4);
  uint32_t* h16 = reinterpret_cast<uint32_t*>(p.h);
  uint32_t* dab16 = reinterpret_cast<uint32_t*>(p.dab);
#pragma unroll
  for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int row = m0 + wm * C::MT * 16 + 16 * mt + 8 * hh + g8;
      if (row >= p.M) continue;
#pragma unroll
      for (int j = 0; j < C::NT / 2; ++j) {
        const int col = hn0 + (wn * (C::HPW / 8) + j) * 8 + 2 * q;
        float hv[2], dav[2], dbv[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const float av = acc[mt][j][2 * hh + k] * s_ab, bv = acc[mt][C::NT / 2 + j][2 * hh + k] * s_ab, d = dacc[mt][j][2 * hh + k] * s_dh;
          const float sg = sigmoid_f32(av), silu = av * sg;
          hv[k] = silu * bv * s_h;
          dbv[k] = d * silu * inv_dab;
          dav[k] = d * bv * fmaf(silu, 1.f - sg, sg) * inv_dab;
        }
        h16[((size_t)row * p.H + col) >> 1] = pack_f16(hv[0], hv[1]);
        dab16[((size_t)row * 2 * p.H + col) >> 1] = pack_f16(dav[0], dav[1]);
        dab16[((size_t)row * 2 * p.H + p.H + col) >> 1] = pack_f16(dbv[0], dbv[1]);
      }
    }
}

// ===================================================================================================================== GEMM + residual
#ifndef RES_PF_SLICE
#define RES_PF_SLICE 0   // the k slice behind which gemm_res_kernel prefetches its residual tile into L2
#endif
// a 128-byte line into L2 (no register, no shared memory; a hint: nothing waits on it)
DEVI void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
// out = rn(rn(A W^T) + res)  (the squeeze projection with the residual folded in; res == nullptr: out = rn(A W^T)).  A [M][K], W [N][K] (the weight of an nn.Linear), N % BN == 0.
struct ResParams {
  const __nv_bfloat16* a;      // [M][K]
  const __nv_bfloat16* w;      // [N][K]
  const __nv_bfloat16* res;    // [M][N] or nullptr
  __nv_bfloat16* out;          // [M][N]
  // the fp32 Transition (F16Of configs): a = fp16 h * sc[2], w = fp16 Ws / s_ws; out32 = acc sc[3] + res32 (fp32 [M][N])
  const float* sc;
  const float* res32;
  float* out32;
  int kt;                      // split K (F16Of configs, grid.z splits): K tiles per split, the fp32 partial of split z into out32 + z M N (no scale, no residual); 0: all
  int M, N, K;
};

template <class C>
__global__ void __launch_bounds__(C::NTHR, C::MINB) gemm_res_kernel(const ResParams p) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  const uint32_t smem = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q = lane & 3;
  const int wm = warp / C::WN, wn = warp % C::WN;
  const int n0 = blockIdx.x * C::BN, m0 = blockIdx.y * C::BM;
  Frag<C> fr(wm, wn, lane);
  float acc[C::MT][C::NT][4];
  zero_acc<C>(acc);
  const int KT = p.kt > 0 ? p.kt : p.K / C::BK;          // the K tiles of this CTA's split (blockIdx.z)
  const size_t koff = (size_t)blockIdx.z * KT * C::BK;
  auto load_a = [&](uint32_t stage, int kt) {
    load_rows_k<C::BM, C::CPA, C::NTHR>(stage, kt * C::BK, [&](int r) { return m0 + r < p.M ? p.a + (size_t)(m0 + r) * p.K + koff : nullptr; }, tid);
  };
  auto load_b = [&](uint32_t stage, int kt) {
    load_rows_k<C::BN, C::CPB, C::NTHR>(stage, kt * C::BK, [&](int r) { return p.w + (size_t)(n0 + r) * p.K + koff; }, tid);
  };
  if (m0 + C::BM <= p.M) {                                  // a whole tile: the lean loop
    using L = LeanCopies<C>;
    const char* pa[L::NJA];
    const char* pb[L::NJB];
    const int ra = tid / C::CPA, ca = tid % C::CPA, rb = tid / C::CPB, cb = tid % C::CPB;
#pragma unroll
    for (int j = 0; j < L::NJA; ++j) pa[j] = reinterpret_cast<const char*>(p.a + (size_t)(m0 + ra + j * L::RPJA) * p.K + koff) + ca * 16;
#pragma unroll
    for (int j = 0; j < L::NJB; ++j) pb[j] = reinterpret_cast<const char*>(p.w + (size_t)(n0 + rb + j * L::RPJB) * p.K + koff) + cb * 16;
    // the bf16 residual tile is pulled into L2 behind slice RES_PF_SLICE (prefetch.global.L2, no registers / shared memory): the epilogue's residual loads then hit L2
    // instead of exposing a DRAM round trip after the mainloop (measured at D 256 / H 512)
    auto pre = [&](int kt) {
      if constexpr (!C::F16) {
        if (kt == (KT > RES_PF_SLICE ? RES_PF_SLICE : 0) && p.res != nullptr) {
          constexpr int LPR = C::BN * 2 / 128;            // 128-byte lines per residual tile row
          static_assert(LPR >= 1 && LPR * 128 == C::BN * 2, "residual prefetch: BN a multiple of 64");
#pragma unroll
          for (int i = tid; i < C::BM * LPR; i += C::NTHR) {
            const int r = i / LPR, l = i % LPR;
            prefetch_l2(p.res + (size_t)(m0 + r) * p.N + n0 + l * 64);
          }
        }
      }
    };
    gemm_mainloop_lean<C>(acc, smem, KT, pa, pb, LeanFrag<C>(wm, wn, lane), tid, pre);
  } else {
    gemm_mainloop<C>(acc, smem, KT, load_a, load_b, fr);
  }
  if constexpr (C::F16) {
    // the fp32 Transition: out32 = acc s_h s_ws + res32, straight from the fragments (a quad covers 8 consecutive floats of a row: whole 32-byte sectors)
    const bool split = gridDim.z > 1;
    const float s_o = split ? 1.f : __ldg(p.sc + 3);
    float* out = p.out32 + (split ? (size_t)blockIdx.z * p.M * p.N : 0);
    const float* res = split ? nullptr : p.res32;
#pragma unroll
    for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = m0 + wm * C::MT * 16 + 16 * mt + 8 * hh + g8;
        if (row >= p.M) continue;
#pragma unroll
        for (int nt = 0; nt < C::NT; ++nt) {
          const size_t o = (size_t)row * p.N + n0 + (wn * C::NT + nt) * 8 + 2 * q;
          const float2 r = res != nullptr ? *reinterpret_cast<const float2*>(res + o) : make_float2(0.f, 0.f);
          *reinterpret_cast<float2*>(out + o) = make_float2(fmaf(acc[mt][nt][2 * hh], s_o, r.x), fmaf(acc[mt][nt][2 * hh + 1], s_o, r.y));
        }
      }
    return;
  }
  // rn(acc) staged as [BM][BN] bf16, then the residual is added on 16-byte vectors
  constexpr int CPO = C::BN / 8;
  static_assert(C::BM * C::BN * 2 <= C::SMEM, "epilogue staging fits the pipeline window");
#pragma unroll
  for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < C::NT; ++nt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint32_t row = (uint32_t)(wm * C::MT * 16 + 16 * mt + 8 * hh + g8), chunk = (uint32_t)(wn * C::NT + nt);
        sts32(smem + swz_off<CPO>(row, chunk) + 4 * q, pack_bf16(acc[mt][nt][2 * hh], acc[mt][nt][2 * hh + 1]));
      }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < (C::BM * CPO) / C::NTHR; ++j) {
    const int i = tid + j * C::NTHR, r = i / CPO, c = i % CPO;
    if (m0 + r < p.M) {
      uint4 y = lds128(smem + swz_off<CPO>(r, c));
      if (p.res != nullptr) {
        const uint4 s = *reinterpret_cast<const uint4*>(p.res + (size_t)(m0 + r) * p.N + n0 + c * 8);
        uint32_t yw[4] = {y.x, y.y, y.z, y.w};
        const uint32_t sw[4] = {s.x, s.y, s.z, s.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) yw[e] = pack_bf16(bf16lo(yw[e]) + bf16lo(sw[e]), bf16hi(yw[e]) + bf16hi(sw[e]));
        y = make_uint4(yw[0], yw[1], yw[2], yw[3]);
      }
      stg128(p.out + (size_t)(m0 + r) * p.N + n0 + c * 8, y);
    }
  }
}

// ===================================================================================================================== fp32 Transition: fp16 weight forms and scales
// The sm_90a f32io family's scheme (wide/f32io.cu f32_forms_*), runtime D / H (D % 32 == 0). stats (zeroed first): [0] max|gamma|, [1] max|beta|, [2] max(|Wa|, |Wb|),
// [3] max row L1 of Wa, Wb, [4] max|Ws| -- nonnegative floats, so atomicMax on their bit patterns is the float max (order-free: deterministic).
constexpr float F16_BOUND = 30000.f;   // fp16 range kept by the scales (fused_wide_sm90a._F16_BOUND)

__global__ void __launch_bounds__(256) f32_forms_stats_kernel(const float* __restrict__ gamma, const float* __restrict__ beta, const float* __restrict__ wa,
                                                             const float* __restrict__ wb, const float* __restrict__ ws, float* __restrict__ st, int D, int H) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, nwarps = (gridDim.x * blockDim.x) >> 5;
  float mg = 0.f, mb = 0.f, mw = 0.f, ml = 0.f, ms = 0.f;
  if (warp == 0)
    for (int c = lane; c < D; c += 32) { mg = fmaxf(mg, fabsf(gamma[c])); mb = fmaxf(mb, fabsf(beta[c])); }
  const int d4 = D / 4;
  for (int r = warp; r < 2 * H; r += nwarps) {           // rows of Wa then Wb: absmax and L1 (16-byte loads)
    const float4* p = reinterpret_cast<const float4*>(r < H ? wa + (size_t)r * D : wb + (size_t)(r - H) * D);
    float l1 = 0.f;
#pragma unroll 4
    for (int c = lane; c < d4; c += 32) {
      const float4 v = __ldg(p + c);
      const float a0 = fabsf(v.x), a1 = fabsf(v.y), a2 = fabsf(v.z), a3 = fabsf(v.w);
      l1 += (a0 + a1) + (a2 + a3);
      mw = fmaxf(mw, fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) l1 += __shfl_xor_sync(0xffffffffu, l1, o);
    ml = fmaxf(ml, l1);
  }
  const float4* w4 = reinterpret_cast<const float4*>(ws);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (long long)D * H / 4; i += (long long)gridDim.x * blockDim.x) {
    const float4 v = __ldg(w4 + i);
    ms = fmaxf(ms, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, o)); mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, o));
    mw = fmaxf(mw, __shfl_xor_sync(0xffffffffu, mw, o)); ms = fmaxf(ms, __shfl_xor_sync(0xffffffffu, ms, o));
  }
  __shared__ float red[5][8];                             // per block first: one atomic per block and statistic, not per warp
  const int w = threadIdx.x >> 5;
  if (lane == 0) { red[0][w] = mg; red[1][w] = mb; red[2][w] = mw; red[3][w] = ml; red[4][w] = ms; }
  __syncthreads();
  if (threadIdx.x < 5) {
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) m = fmaxf(m, red[threadIdx.x][i]);
    atomicMax(reinterpret_cast<int*>(st) + threadIdx.x, __float_as_int(m));
  }
}

DEVI uint2 pack_f16x4(float4 v, float s) { return make_uint2(pack_f16(v.x / s, v.y / s), pack_f16(v.z / s, v.w / s)); }

// sc [8] = [1 / s_x, s_x s_w, 1 / s_h, s_h s_ws, a_max, D max|Ws|, -, -]: the forward's scales and the backward's bounds (|a|, |b|, |silu(a)| <= a_max;
// |dh| <= max|dy| D max|Ws|). Wab16 = [Wa; Wb] / s_w (fp16 [2H][D], stacked: the backward's d_xn operand), Ws16 = Ws / s_ws (fp16 [D][H]) and, when wst16 is
// given, Ws16^T (fp16 [H][D], the backward's dh operand). Each scale is max(bound / F16_BOUND, 1): |xn| <= sqrt(D) max|gamma| + max|beta|,
// |a|, |b| <= |xn|max max-row-L1, |h| <= |a|max |b|max. Blocks [0, nflat) cast (16-byte loads), the others transpose 32 x 32 tiles of Ws.
__global__ void __launch_bounds__(256) f32_forms_apply_kernel(const float* __restrict__ wa, const float* __restrict__ wb, const float* __restrict__ ws,
                                                             const float* __restrict__ st, uint16_t* __restrict__ wab16, uint16_t* __restrict__ ws16,
                                                             uint16_t* __restrict__ wst16, float* __restrict__ sc, int D, int H, int nflat) {
  const float xn_max = sqrtf((float)D) * st[0] + st[1];
  const float s_x = fmaxf(xn_max / F16_BOUND, 1.f), s_w = fmaxf(st[2] / F16_BOUND, 1.f);
  const float a_max = xn_max * st[3];
  const float s_h = fmaxf(a_max * a_max / F16_BOUND, 1.f), s_ws = fmaxf(st[4] / F16_BOUND, 1.f);
  if ((int)blockIdx.x < nflat) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, step = (long long)nflat * blockDim.x, n4 = (long long)D * H / 4;
    if (tid == 0) {
      sc[0] = 1.f / s_x; sc[1] = s_x * s_w; sc[2] = 1.f / s_h; sc[3] = s_h * s_ws; sc[4] = a_max; sc[5] = (float)D * st[4]; sc[6] = 0.f; sc[7] = 0.f;
    }
    const float4 *a4 = reinterpret_cast<const float4*>(wa), *b4 = reinterpret_cast<const float4*>(wb), *s4 = reinterpret_cast<const float4*>(ws);
    uint2* o = reinterpret_cast<uint2*>(wab16);
    uint2* os = reinterpret_cast<uint2*>(ws16);
    for (long long i = tid; i < n4; i += step) {
      o[i] = pack_f16x4(__ldg(a4 + i), s_w);
      o[n4 + i] = pack_f16x4(__ldg(b4 + i), s_w);
      os[i] = pack_f16x4(__ldg(s4 + i), s_ws);
    }
    return;
  }
  // Ws [D][H] -> Ws16^T [H][D]: one 32 x 32 tile per block (8 rows per pass of the 256 threads)
  __shared__ float tile[32][33];
  const int t = blockIdx.x - nflat, tiles_h = H / 32, d0 = (t / tiles_h) * 32, h0 = (t % tiles_h) * 32;
  const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
#pragma unroll
  for (int r = ty; r < 32; r += 8) tile[r][tx] = __ldg(ws + (size_t)(d0 + r) * H + h0 + tx);
  __syncthreads();
#pragma unroll
  for (int r = ty; r < 32; r += 8) wst16[(size_t)(h0 + r) * D + d0 + tx] = __half_as_ushort(__float2half_rn(tile[tx][r] / s_ws));
}

// out = res + s sum_z part[z] (the split-K squeeze's reduction, the splits summed in order; s = sc[3])
__global__ void __launch_bounds__(256) split_res_kernel(const float4* __restrict__ part, int splits, long long n4, const float4* __restrict__ res,
                                                        const float* __restrict__ sc, float4* __restrict__ out) {
  const float s = __ldg(sc + 3);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) {
    float4 a = __ldg(part + i);
    for (int z = 1; z < splits; ++z) {
      const float4 b = __ldg(part + (size_t)z * n4 + i);
      a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    }
    const float4 r = __ldg(res + i);
    out[i] = make_float4(fmaf(a.x, s, r.x), fmaf(a.y, s, r.y), fmaf(a.z, s, r.z), fmaf(a.w, s, r.w));
  }
}

// ---------------------------------------------------------------------------------------------------------------- the backward's dy in fp16
// dy_absmax: max|dy| into *mx (zeroed first; atomicMax on the bit patterns: deterministic). dy_convert: dy16 = fp16(dy / s_dy), s_dy = max|dy| / F16_BOUND (any
// magnitude: tiny gradients are scaled up), and block 0 writes the backward's scale vector bs [8] (see the bindings): with s_dab = (bound of |dA|, |dB|) / F16_BOUND,
//   bs = [s_dy s_h (dWs), s_dab s_x ([dWa; dWb]), s_dab s_w (d_xn), 1 / s_dy, 1 / s_dab, s_dy s_ws (dh), -, -]
__global__ void __launch_bounds__(256) dy_absmax_kernel(const float4* __restrict__ dy, long long n4, int* __restrict__ mx) {
  float m = 0.f;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) {
    const float4 v = __ldg(dy + i);
    m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) atomicMax(mx, __float_as_int(m));
}

DEVI float pos_or_one(float v) { return v > 0.f && v < 3.0e38f ? v : 1.f; }

__global__ void __launch_bounds__(256) dy_convert_kernel(const float4* __restrict__ dy, long long n4, const int* __restrict__ mx, const float* __restrict__ sc,
                                                         uint2* __restrict__ dy16, float* __restrict__ bs) {
  const float dymax = __int_as_float(__ldg(mx));
  const float s_dy = pos_or_one(dymax / F16_BOUND);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const float s_x = 1.f / sc[0], s_w = sc[1] * sc[0], s_h = 1.f / sc[2], s_ws = sc[3] * sc[2];
    const float s_dab = pos_or_one(dymax * sc[5] * sc[4] * 1.1f / F16_BOUND);   // |dA| <= |dh| |b| max silu' (< 1.1), |dB| <= |dh| |silu(a)|
    bs[0] = s_dy * s_h; bs[1] = s_dab * s_x; bs[2] = s_dab * s_w; bs[3] = 1.f / s_dy; bs[4] = 1.f / s_dab; bs[5] = s_dy * s_ws; bs[6] = 0.f; bs[7] = 0.f;
  }
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) dy16[i] = pack_f16x4(__ldg(dy + i), s_dy);
}

}  // namespace a100
