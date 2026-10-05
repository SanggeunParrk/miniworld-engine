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
#pragma once
#include "tw_gemm_sm80.cuh"

namespace a100 {

// ===================================================================================================================== LayerNorm forward
template <int D>
__global__ void __launch_bounds__(256) ln_fwd_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
                                                     __nv_bfloat16* __restrict__ xn, float2* __restrict__ stats, int M, float eps) {
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
      uint4 u = make_uint4(0u, 0u, 0u, 0u);
      if (valid && c < CH) u = *reinterpret_cast<const uint4*>(x + (size_t)row * D + c * 8);
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) { v[i][2 * e] = bf16lo(w[e]); v[i][2 * e + 1] = bf16hi(w[e]); s += v[i][2 * e] + v[i][2 * e + 1]; }
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
#pragma unroll
        for (int e = 0; e < 4; ++e)
          o[e] = pack_bf16(fmaf(v[i][2 * e] * rstd, gg[2 * e], bb[2 * e]), fmaf(v[i][2 * e + 1] * rstd, gg[2 * e + 1], bb[2 * e + 1]));
        *reinterpret_cast<uint4*>(xn + (size_t)row * D + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
      }
    }
  }
}

// ===================================================================================================================== LayerNorm backward
// One row per warp (lanes spread over the chunks) or several rows per warp for D <= 128; dgamma / dbeta are summed per lane over the rows the lane's warp visits, then per CTA.
template <int D, bool RES = true>
__global__ void __launch_bounds__(128) ln_bwd_kernel(const float* __restrict__ dxn, const __nv_bfloat16* __restrict__ x, const float2* __restrict__ stats,
                                                     const float* __restrict__ gamma, const __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx,
                                                     float* __restrict__ part, int M) {
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
      float4 d0 = make_float4(0.f, 0.f, 0.f, 0.f), d1 = d0, g0 = d0, g1 = d0;
      if (valid && c < CH) {
        u = *reinterpret_cast<const uint4*>(x + (size_t)row * D + c * 8);
        d0 = *reinterpret_cast<const float4*>(dxn + (size_t)row * D + c * 8);
        d1 = *reinterpret_cast<const float4*>(dxn + (size_t)row * D + c * 8 + 4);
        g0 = *reinterpret_cast<const float4*>(gamma + c * 8);
        g1 = *reinterpret_cast<const float4*>(gamma + c * 8 + 4);
      }
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};
      const float dd[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w}, gm[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int j = 2 * e + k;
          const float xv = k ? bf16hi(w[e]) : bf16lo(w[e]);
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
      if (valid && c < CH) {
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
template <class TA>
__global__ void __launch_bounds__(256) ln_finalize_kernel(const float* __restrict__ part, int nblk, int D, TA* __restrict__ dgamma, TA* __restrict__ dbeta) {
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

  // SwiGLU: a = acc[mt][j], b = acc[mt][NT / 2 + j]; h = rn(silu(a) b) staged as [BM][HN] bf16 (swizzled)
#pragma unroll
  for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
    for (int j = 0; j < C::NT / 2; ++j)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float hv[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const float av = acc[mt][j][2 * hh + k], bv = acc[mt][C::NT / 2 + j][2 * hh + k];
          hv[k] = EPI == 0 ? av * sigmoid(av) * bv : av * sigmoid(bv);                 // EPI 0: silu(a) b (SwiGLU);  1: a sigmoid(b) (the triangle-multiplication gate)
        }
        const uint32_t row = (uint32_t)(wm * C::MT * 16 + 16 * mt + 8 * hh + g8), chunk = (uint32_t)(wn * (C::HPW / 8) + j);
        sts32(smem + swz_off<C::CPH>(row, chunk) + 4 * q, pack_bf16(hv[0], hv[1]));
      }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < (C::BM * C::CPH) / C::NTHR; ++j) {
    const int i = tid + j * C::NTHR, r = i / C::CPH, c = i % C::CPH;
    if (m0 + r < p.M) stg128(p.h + (size_t)(m0 + r) * p.H + (size_t)hn0 + c * 8, lds128(smem + swz_off<C::CPH>(r, c)));
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
          const float av = acc[mt][j][2 * hh + k], bv = acc[mt][C::NT / 2 + j][2 * hh + k];
          if (EPI == 0) {
            const float sg = sigmoid(av), silu = av * sg;
            hv[k] = silu * bv;
            bv2[k] = dv[k] * silu;
            av2[k] = dv[k] * bv * fmaf(silu, 1.f - sg, sg);
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

// ===================================================================================================================== GEMM + residual
// out = rn(rn(A W^T) + res)  (the squeeze projection with the residual folded in; res == nullptr: out = rn(A W^T)).  A [M][K], W [N][K] (the weight of an nn.Linear), N % BN == 0.
struct ResParams {
  const __nv_bfloat16* a;      // [M][K]
  const __nv_bfloat16* w;      // [N][K]
  const __nv_bfloat16* res;    // [M][N] or nullptr
  __nv_bfloat16* out;          // [M][N]
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
  auto load_a = [&](uint32_t stage, int kt) {
    load_rows_k<C::BM, C::CPA, C::NTHR>(stage, kt * C::BK, [&](int r) { return m0 + r < p.M ? p.a + (size_t)(m0 + r) * p.K : nullptr; }, tid);
  };
  auto load_b = [&](uint32_t stage, int kt) {
    load_rows_k<C::BN, C::CPB, C::NTHR>(stage, kt * C::BK, [&](int r) { return p.w + (size_t)(n0 + r) * p.K; }, tid);
  };
  if (m0 + C::BM <= p.M) {                                  // a whole tile: the lean loop
    using L = LeanCopies<C>;
    const char* pa[L::NJA];
    const char* pb[L::NJB];
    const int ra = tid / C::CPA, ca = tid % C::CPA, rb = tid / C::CPB, cb = tid % C::CPB;
#pragma unroll
    for (int j = 0; j < L::NJA; ++j) pa[j] = reinterpret_cast<const char*>(p.a + (size_t)(m0 + ra + j * L::RPJA) * p.K) + ca * 16;
#pragma unroll
    for (int j = 0; j < L::NJB; ++j) pb[j] = reinterpret_cast<const char*>(p.w + (size_t)(n0 + rb + j * L::RPJB) * p.K) + cb * 16;
    gemm_mainloop_lean<C>(acc, smem, p.K / C::BK, pa, pb, LeanFrag<C>(wm, wn, lane), tid);
  } else {
    gemm_mainloop<C>(acc, smem, p.K / C::BK, load_a, load_b, fr);
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

}  // namespace a100
