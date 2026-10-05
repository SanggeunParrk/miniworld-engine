// pwa_ln_v_sm80.cuh -- MSA pair-weighted averaging forward, the value projection: LayerNorm of the MSA rows and v = y Wv^T, written HEAD-MAJOR
//
//   y[s, j, :]            = bf16( LN(m[s, j, :]) )                          fp32 statistics, affine
//   v[h, j, s * C + c]    = bf16( y[s, j, :] . Wv[h * C + c, :] )           [8, L, S C]: for one (h, j) the S x C values are one contiguous run
//
// so that the contraction over the keys is a plain batched GEMM o[h] = w[h] v[h] (cuBLAS: [L x L] . [L x S C]).  A CTA owns 128 consecutive MSA rows
// of one token j (8 warps x 16 rows) and loops over tiles (persistent: Wv is staged once per CTA).  The rows arrive by cp.async (double-buffered, swizzled): the
// LayerNorm runs on the A fragments (ldmatrix, quad shuffles) and leaves y in registers as the A fragments of the projection (mma.sync, B = Wv's rows, K-major
// already), per head a [16 x C] accumulator; the bf16 results are staged as [128 rows][8 C] and stored as 16-byte vectors along the (s, c) runs.  Training keeps the
// LayerNorm statistics (mean, rstd) per token.
#pragma once
#include "sm80_common.cuh"

namespace pwa80 {

struct LnVParams {
  const __nv_bfloat16* m;      // [S][L][D]
  const float* lnw;            // [D]
  const float* lnb;            // [D]
  const __nv_bfloat16* wv;     // [8 C][D]
  __nv_bfloat16* v;            // [8][L][S C]
  float2* stats;               // [S][L] or nullptr
  int S, L, ntile_s, ntile, kp, ns;                  // kp, ns: the MSA rows in ns chunks of kp (see hm_row)
  float eps;
};

template <int D, int C> struct LnVCfg {
  static constexpr int NTHR = 256, NTOK = 128, NCH = D / 8, HC = 8 * C, NCO = HC / 8;   // 16 B chunks per m row, per staged output row
  static constexpr int MB = NTOK * D * 2, WB = HC * D * 2, OB = NTOK * HC * 2;
  static constexpr int SMEM = 2 * MB + WB + OB + 2 * D * 4;
  static_assert((D == 64 || D == 128) && (C == 8 || C == 16 || C == 32), "d_msa / d_hidden");
  static_assert(NCO % 8 == 0, "staged rows need a multiple of 8 chunks for the swizzle (HC >= 64)");
  static_assert(SMEM <= 166912, "sm_80 shared memory");
};

template <int D, int C>
DEVI void lv_load(const LnVParams& p, uint32_t dst, int tl, int tid) {
  using G = LnVCfg<D, C>;
  const int j = tl / p.ntile_s, s0 = (tl % p.ntile_s) * G::NTOK;
#pragma unroll
  for (int it = 0; it < G::NTOK * G::NCH / G::NTHR; ++it) {
    const int v = it * G::NTHR + tid, ch = v % G::NCH, t = v / G::NCH;
    const bool ok = s0 + t < p.S;
    cp_async16(dst + swzn<G::NCH>(t, ch), p.m + (ok ? ((long)(s0 + t) * p.L + j) * D + ch * 8 : 0), ok ? 16u : 0u);
  }
}

template <int D, int C, bool SAVE_STATS>
__global__ void __launch_bounds__(256, 1) pwa_ln_v_kernel(const LnVParams p) {
  using G = LnVCfg<D, C>;
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t sm0 = smem_u32(smem), sw = sm0 + 2 * G::MB, so = sw + G::WB;
  float* sgb = reinterpret_cast<float*>(smem + 2 * G::MB + G::WB + G::OB);                   // gamma [D], beta [D]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;

  for (int u = tid; u < G::HC * G::NCH; u += G::NTHR) {
    const int row = u / G::NCH, ch = u % G::NCH;
    cp_async16(sw + swzn<G::NCH>(row, ch), p.wv + (long)row * D + ch * 8);
  }
  for (int k = tid; k < D; k += G::NTHR) { sgb[k] = p.lnw[k]; sgb[D + k] = p.lnb[k]; }
  int tl = blockIdx.x, buf = 0;
  if (tl < p.ntile) lv_load<D, C>(p, sm0, tl, tid);
  cp_async_commit();

#pragma unroll 1
  for (; tl < p.ntile; tl += gridDim.x, buf ^= 1) {
    const int j = tl / p.ntile_s, s0 = (tl % p.ntile_s) * G::NTOK;
    const int nx = tl + gridDim.x;
    if (nx < p.ntile) lv_load<D, C>(p, sm0 + (buf ^ 1) * G::MB, nx, tid);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const uint32_t sm = sm0 + buf * G::MB;
    const int r0 = warp * 16 + gq, r1 = r0 + 8;

    // ---- LayerNorm of the warp's 16 rows on the A fragments -> ya
    uint32_t xa[D / 16][4];
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) ldsm_x4(xa[ks], sm + swzn<G::NCH>(warp * 16 + (lane & 15), 2 * ks + (lane >> 4)));
    float s0v = 0.f, s1v = 0.f;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      s0v += bf16lo(xa[ks][0]) + bf16hi(xa[ks][0]) + bf16lo(xa[ks][2]) + bf16hi(xa[ks][2]);
      s1v += bf16lo(xa[ks][1]) + bf16hi(xa[ks][1]) + bf16lo(xa[ks][3]) + bf16hi(xa[ks][3]);
    }
    const float mean0 = quad_sum(s0v) * (1.f / D), mean1 = quad_sum(s1v) * (1.f / D);
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      float d;
      d = bf16lo(xa[ks][0]) - mean0; v0 = fmaf(d, d, v0); d = bf16hi(xa[ks][0]) - mean0; v0 = fmaf(d, d, v0);
      d = bf16lo(xa[ks][2]) - mean0; v0 = fmaf(d, d, v0); d = bf16hi(xa[ks][2]) - mean0; v0 = fmaf(d, d, v0);
      d = bf16lo(xa[ks][1]) - mean1; v1 = fmaf(d, d, v1); d = bf16hi(xa[ks][1]) - mean1; v1 = fmaf(d, d, v1);
      d = bf16lo(xa[ks][3]) - mean1; v1 = fmaf(d, d, v1); d = bf16hi(xa[ks][3]) - mean1; v1 = fmaf(d, d, v1);
    }
    const float rs0 = rsqrtf(quad_sum(v0) * (1.f / D) + p.eps), rs1 = rsqrtf(quad_sum(v1) * (1.f / D) + p.eps);
    if (SAVE_STATS && q == 0) {
      if (s0 + r0 < p.S) p.stats[(long)(s0 + r0) * p.L + j] = make_float2(mean0, rs0);
      if (s0 + r1 < p.S) p.stats[(long)(s0 + r1) * p.L + j] = make_float2(mean1, rs1);
    }
    uint32_t ya[D / 16][4];
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      const int col = 16 * ks + 2 * q;
      const float2 g0 = *reinterpret_cast<const float2*>(sgb + col), g1 = *reinterpret_cast<const float2*>(sgb + col + 8);
      const float2 b0 = *reinterpret_cast<const float2*>(sgb + D + col), b1 = *reinterpret_cast<const float2*>(sgb + D + col + 8);
      ya[ks][0] = pack_bf16(fmaf((bf16lo(xa[ks][0]) - mean0) * rs0, g0.x, b0.x), fmaf((bf16hi(xa[ks][0]) - mean0) * rs0, g0.y, b0.y));
      ya[ks][1] = pack_bf16(fmaf((bf16lo(xa[ks][1]) - mean1) * rs1, g0.x, b0.x), fmaf((bf16hi(xa[ks][1]) - mean1) * rs1, g0.y, b0.y));
      ya[ks][2] = pack_bf16(fmaf((bf16lo(xa[ks][2]) - mean0) * rs0, g1.x, b1.x), fmaf((bf16hi(xa[ks][2]) - mean0) * rs0, g1.y, b1.y));
      ya[ks][3] = pack_bf16(fmaf((bf16lo(xa[ks][3]) - mean1) * rs1, g1.x, b1.x), fmaf((bf16hi(xa[ks][3]) - mean1) * rs1, g1.y, b1.y));
    }

    // ---- the projection, head by head: A = y (this warp's 16 rows), B = Wv rows h C .. h C + C - 1
#pragma unroll
    for (int h = 0; h < 8; ++h) {
      float acc[C / 8][4];
#pragma unroll
      for (int nt = 0; nt < C / 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks) {
        if (C >= 16) {
#pragma unroll
          for (int np = 0; np < C / 16; ++np) {
            uint32_t bf[4];
            ldsm_x4(bf, sw + swzn<G::NCH>(h * C + 16 * np + (lane & 7) + ((lane >> 4) << 3), 2 * ks + ((lane >> 3) & 1)));
            mma16816(acc[2 * np], ya[ks], bf[0], bf[1]);
            mma16816(acc[2 * np + 1], ya[ks], bf[2], bf[3]);
          }
        } else {                                       // C = 8: one n8 tile
          uint32_t bf[4];
          ldsm_x4(bf, sw + swzn<G::NCH>(h * C + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
          mma16816(acc[0], ya[ks], bf[0], bf[1]);
        }
      }
#pragma unroll
      for (int nt = 0; nt < C / 8; ++nt) {
        const int col = h * C + nt * 8 + 2 * q;
        sts32(so + swzn<G::NCO>(r0, col >> 3) + (col & 7) * 2, pack_bf16(acc[nt][0], acc[nt][1]));
        sts32(so + swzn<G::NCO>(r1, col >> 3) + (col & 7) * 2, pack_bf16(acc[nt][2], acc[nt][3]));
      }
    }
    __syncthreads();
    // ---- stores: for each head, the CTA's 128 rows x C values are one contiguous run of v[h][j]
#pragma unroll
    for (int h = 0; h < 8; ++h) {
#pragma unroll
      for (int it = 0; it < (G::NTOK * C / 8 + G::NTHR - 1) / G::NTHR; ++it) {
        const int u = it * G::NTHR + tid;
        if (u < G::NTOK * C / 8) {
          const int t = u / (C / 8), pc = u % (C / 8);
          if (s0 + t < p.S) stg128(p.v + (hm_row(h, j, s0, p.L, p.kp, p.ns) + t) * C + pc * 8, lds128(so + swzn<G::NCO>(t, h * (C / 8) + pc)));
        }
      }
    }
    __syncthreads();                                   // the output tile is rewritten by the next tile
  }
  cp_async_wait<0>();
}

}  // namespace pwa80
