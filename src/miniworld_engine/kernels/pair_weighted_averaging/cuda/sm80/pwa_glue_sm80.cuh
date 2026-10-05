// pwa_glue_sm80.cuh -- MSA pair-weighted averaging backward, first stage ("glue"): from the output gradient to the gradients of the gated contraction output
//
//   drb[s, i, :]   = bf16( dres[s, i, :] * keep[i, :] * dscale )                          the dropout backward (no dropout: dres itself)
//   du[s, i, hc]   = drb . Wo[:, hc]                                                       (fp32, per unit of 16 channels)
//   g[s, i, hc]    = sigmoid( y . Wg[hc, :] ),  y = bf16( LN(m) ) recomputed
//   do             = bf16( du g )                                  -> head-major [8, L, S C] (the cotangent of o: dv = w^T do, dw = do v^T are cuBLAS bmm)
//   dgp            = bf16( du o g (1 - g) )                         -> head-major (the cotangent of the gate's pre-activation)
//   dWo[d, hc]    += sum_tokens drb[t, d] bf16( g o )[t, hc]                                (fp32 partial per CTA)
//   dWg[hc, d]    += sum_tokens dgp[t, hc] y[t, d]
//
// A CTA owns one GROUP of channels (64, or all of them when the layer has fewer: one group = one CTA column of the grid) and loops over tiles of one query row
// i x NTOK consecutive MSA rows (persistent: the group's slices of Wo and Wg stay in shared memory).  Warps own 16 rows each; the per-token chain (LayerNorm on the
// A fragments, the du and gate products on mma.sync, the elementwise gate algebra on the accumulators, o read in the accumulator layout) runs per unit of 16 channels
// (C = 8, 16, 32 alike).  The two weight gradients contract over the tokens of the tile: g o and dgp go to shared memory, drb and y already live there (written
// in place over the dres / m tiles), and every warp accumulates its fixed blocks of dWo / dWg in registers across the CTA's tiles (ldmatrix.trans operands);
// the CTA's partials are written once at the end.
#pragma once
#include "sm80_common.cuh"

namespace pwa80 {

struct GlueParams {
  const __nv_bfloat16* m;      // [S][L][D]
  const __nv_bfloat16* dres;   // [S][L][D]
  const __nv_bfloat16* o;      // [8][L][S C]
  const float* lnw;
  const float* lnb;
  const __nv_bfloat16* wg;     // [HC][D]
  const __nv_bfloat16* wo;     // [D][HC]
  const __nv_bfloat16* keep;   // [L][D] or nullptr
  __nv_bfloat16* dO;           // [8][L][S C]
  __nv_bfloat16* dgp;          // [8][L][S C]
  float* part;                 // [gridDim.x][NG][2 * D * GCH]: dWo group slice [D][GCH], then dWg group slice [GCH][D]
  int S, L, ntile_s, ntile, kp, ns;                  // kp, ns: the MSA rows in ns chunks of kp (see hm_row)
  float eps, dscale;
};

template <int D, int C> struct GlueCfg {
  static constexpr int HC = 8 * C, GCH = HC < 64 ? HC : 64, NG = HC / GCH, GU = GCH / 16;
  static constexpr int NW = D == 64 ? 8 : 4, NTHR = NW * 32, NTOK = NW * 16, NKS = NTOK / 16;
  static constexpr int NCH = D / 8, NCG = GCH / 8;
  static constexpr int WOB = D * GCH * 2, WGB = GCH * D * 2, YB = NTOK * D * 2, GOB = NTOK * GCH * 2;
  static constexpr int SMEM = WOB + WGB + 2 * YB + 2 * GOB + 2 * D * 4;
  // warp grids of the two weight-gradient GEMMs: dWo [D x GCH] (m = d), dWg [GCH x D] (m = hc)
  static constexpr int MTO = D / 16, NTO = GCH / 8, WMO = MTO < 4 ? MTO : 4, WNO = NW / WMO, RMO = MTO / WMO, RNO = NTO / WNO;
  static constexpr int MTG = GCH / 16, NTG = D / 8, WMG = MTG < 4 ? MTG : 4, WNG = NW / WMG, RMG = MTG / WMG, RNG = NTG / WNG;
  static_assert((D == 64 && NW == 8) || (D == 128 && NW == 4), "d_msa");
  static_assert(RNO % 2 == 0 && RNG % 2 == 0 && WMO * WNO == NW && WMG * WNG == NW, "warp grids");
  static_assert(SMEM <= 166912, "sm_80 shared memory");
};

// the tile's m and dres rows, and the group's slice of o ([NTOK tokens][GCH channels] in 128-byte-swizzled rows; the unit loop turns it into g o in place)
template <int D, int C>
DEVI void gl_load(const GlueParams& p, uint32_t sym, uint32_t sdr, uint32_t sgo, int tl, int cg, int tid) {
  using G = GlueCfg<D, C>;
  const int i = tl / p.ntile_s, s0 = (tl % p.ntile_s) * G::NTOK;
#pragma unroll
  for (int it = 0; it < G::NTOK * G::NCH / G::NTHR; ++it) {
    const int v = it * G::NTHR + tid, ch = v % G::NCH, t = v / G::NCH;
    const bool ok = s0 + t < p.S;
    const long off = ok ? ((long)(s0 + t) * p.L + i) * D + ch * 8 : 0;
    cp_async16(sym + swzn<G::NCH>(t, ch), p.m + off, ok ? 16u : 0u);
    cp_async16(sdr + swzn<G::NCH>(t, ch), p.dres + off, ok ? 16u : 0u);
  }
#pragma unroll
  for (int it = 0; it < G::NTOK * G::NCG / G::NTHR; ++it) {
    const int v = it * G::NTHR + tid, x = v % G::NCG, t = v / G::NCG;
    const int cabs = cg * G::GCH + 8 * x, h = cabs / C, cc = cabs % C;
    const bool ok = s0 + t < p.S;
    cp_async16(sgo + swz128(t, x), p.o + (ok ? (hm_row(h, i, s0, p.L, p.kp, p.ns) + t) * C + cc : 0), ok ? 16u : 0u);
  }
}

template <int D, int C>
__global__ void __launch_bounds__(GlueCfg<D, C>::NTHR, 2) pwa_glue_kernel(const GlueParams p) {
  using G = GlueCfg<D, C>;
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t swo = smem_u32(smem), swg = swo + G::WOB, sym = swg + G::WGB, sdr = sym + G::YB, sgo = sdr + G::YB, sdg = sgo + G::GOB;
  float* sgb = reinterpret_cast<float*>(smem + G::WOB + G::WGB + 2 * G::YB + 2 * G::GOB);       // gamma [D], beta [D]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;
  const int cg = blockIdx.y;

  for (int u = tid; u < D * G::NCG; u += G::NTHR) {                        // Wo[:, group] -> [D][GCH]
    const int row = u / G::NCG, ch = u % G::NCG;
    cp_async16(swo + swzn<G::NCG>(row, ch), p.wo + (long)row * G::HC + cg * G::GCH + ch * 8);
  }
  for (int u = tid; u < G::GCH * G::NCH; u += G::NTHR) {                   // Wg[group, :] -> [GCH][D]
    const int row = u / G::NCH, ch = u % G::NCH;
    cp_async16(swg + swzn<G::NCH>(row, ch), p.wg + (long)(cg * G::GCH + row) * D + ch * 8);
  }
  for (int k = tid; k < D; k += G::NTHR) { sgb[k] = p.lnw[k]; sgb[D + k] = p.lnb[k]; }
  cp_async_commit();

  float awo[G::RMO][G::RNO][4], awg[G::RMG][G::RNG][4];
#pragma unroll
  for (int a = 0; a < G::RMO; ++a)
#pragma unroll
    for (int b = 0; b < G::RNO; ++b) awo[a][b][0] = awo[a][b][1] = awo[a][b][2] = awo[a][b][3] = 0.f;
#pragma unroll
  for (int a = 0; a < G::RMG; ++a)
#pragma unroll
    for (int b = 0; b < G::RNG; ++b) awg[a][b][0] = awg[a][b][1] = awg[a][b][2] = awg[a][b][3] = 0.f;

  const int r0 = warp * 16 + gq, r1 = r0 + 8;
#pragma unroll 1
  for (int tl = blockIdx.x; tl < p.ntile; tl += gridDim.x) {
    const int i = tl / p.ntile_s, s0 = (tl % p.ntile_s) * G::NTOK;
    gl_load<D, C>(p, sym, sdr, sgo, tl, cg, tid);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    // ---- LayerNorm of the warp's 16 rows -> ya (A fragments); y goes back over the m tile (the dWg operand)
    uint32_t xa[D / 16][4];
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) ldsm_x4(xa[ks], sym + swzn<G::NCH>(warp * 16 + (lane & 15), 2 * ks + (lane >> 4)));
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
    uint32_t ya[D / 16][4], da[D / 16][4];
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      const int col = 16 * ks + 2 * q;
      const float2 g0 = *reinterpret_cast<const float2*>(sgb + col), g1 = *reinterpret_cast<const float2*>(sgb + col + 8);
      const float2 b0 = *reinterpret_cast<const float2*>(sgb + D + col), b1 = *reinterpret_cast<const float2*>(sgb + D + col + 8);
      ya[ks][0] = pack_bf16(fmaf((bf16lo(xa[ks][0]) - mean0) * rs0, g0.x, b0.x), fmaf((bf16hi(xa[ks][0]) - mean0) * rs0, g0.y, b0.y));
      ya[ks][1] = pack_bf16(fmaf((bf16lo(xa[ks][1]) - mean1) * rs1, g0.x, b0.x), fmaf((bf16hi(xa[ks][1]) - mean1) * rs1, g0.y, b0.y));
      ya[ks][2] = pack_bf16(fmaf((bf16lo(xa[ks][2]) - mean0) * rs0, g1.x, b1.x), fmaf((bf16hi(xa[ks][2]) - mean0) * rs0, g1.y, b1.y));
      ya[ks][3] = pack_bf16(fmaf((bf16lo(xa[ks][3]) - mean1) * rs1, g1.x, b1.x), fmaf((bf16hi(xa[ks][3]) - mean1) * rs1, g1.y, b1.y));
      // y over the m tile (this warp's rows only: the fragment owns the (row, column) pairs it writes)
      sts32(sym + swzn<G::NCH>(r0, (16 * ks) / 8) + 4 * q, ya[ks][0]);
      sts32(sym + swzn<G::NCH>(r1, (16 * ks) / 8) + 4 * q, ya[ks][1]);
      sts32(sym + swzn<G::NCH>(r0, (16 * ks + 8) / 8) + 4 * q, ya[ks][2]);
      sts32(sym + swzn<G::NCH>(r1, (16 * ks + 8) / 8) + 4 * q, ya[ks][3]);
    }
    // ---- drb: the dropout backward on the A fragments of the dres rows, in place
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      uint32_t dr[4];
      ldsm_x4(dr, sdr + swzn<G::NCH>(warp * 16 + (lane & 15), 2 * ks + (lane >> 4)));
      if (p.keep != nullptr) {
        const int col = 16 * ks + 2 * q;
        const uint32_t k0 = __ldg(reinterpret_cast<const unsigned int*>(p.keep + (long)i * D + col)), k1 = __ldg(reinterpret_cast<const unsigned int*>(p.keep + (long)i * D + col + 8));
        dr[0] = pack_bf16(bf16lo(dr[0]) * bf16lo(k0) * p.dscale, bf16hi(dr[0]) * bf16hi(k0) * p.dscale);
        dr[1] = pack_bf16(bf16lo(dr[1]) * bf16lo(k0) * p.dscale, bf16hi(dr[1]) * bf16hi(k0) * p.dscale);
        dr[2] = pack_bf16(bf16lo(dr[2]) * bf16lo(k1) * p.dscale, bf16hi(dr[2]) * bf16hi(k1) * p.dscale);
        dr[3] = pack_bf16(bf16lo(dr[3]) * bf16lo(k1) * p.dscale, bf16hi(dr[3]) * bf16hi(k1) * p.dscale);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) da[ks][e] = dr[e];
      sts32(sdr + swzn<G::NCH>(r0, (16 * ks) / 8) + 4 * q, dr[0]);
      sts32(sdr + swzn<G::NCH>(r1, (16 * ks) / 8) + 4 * q, dr[1]);
      sts32(sdr + swzn<G::NCH>(r0, (16 * ks + 8) / 8) + 4 * q, dr[2]);
      sts32(sdr + swzn<G::NCH>(r1, (16 * ks + 8) / 8) + 4 * q, dr[3]);
    }

    // ---- units of 16 channels of this group
    const bool v0ok = s0 + r0 < p.S, v1ok = s0 + r1 < p.S;
#pragma unroll 2
    for (int u = 0; u < G::GU; ++u) {
      uint32_t ov[4];                                                   // o of the unit in the accumulator layout: (r0, nt 0) (r1, nt 0) (r0, nt 1) (r1, nt 1)
      ldsm_x4(ov, sgo + swz128(warp * 16 + (lane & 15), 2 * u + (lane >> 4)));
      long off[2][2];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int cabs = cg * G::GCH + 16 * u + 8 * nt + 2 * q, h = cabs / C, cc = cabs % C;
        const long base = hm_row(h, i, s0, p.L, p.kp, p.ns) * C + cc;
        off[nt][0] = base + (long)r0 * C;
        off[nt][1] = base + (long)r1 * C;
      }
      float du[2][4], ga[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) { du[nt][0] = du[nt][1] = du[nt][2] = du[nt][3] = 0.f; ga[nt][0] = ga[nt][1] = ga[nt][2] = ga[nt][3] = 0.f; }
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks) {
        uint32_t bw[4], bg[4];
        ldsm_x4_t(bw, swo + swzn<G::NCG>(16 * ks + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * u + (lane >> 4)));         // Wo[d][hc]: k = d rows, n = hc columns
        ldsm_x4(bg, swg + swzn<G::NCH>(16 * u + (lane & 7) + ((lane >> 4) << 3), 2 * ks + ((lane >> 3) & 1)));          // Wg[hc][d]: n = hc rows, k = d columns
        mma16816(du[0], da[ks], bw[0], bw[1]);
        mma16816(du[1], da[ks], bw[2], bw[3]);
        mma16816(ga[0], ya[ks], bg[0], bg[1]);
        mma16816(ga[1], ya[ks], bg[2], bg[3]);
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const uint32_t o2[2] = {ov[2 * nt], ov[2 * nt + 1]};
        uint32_t dop[2], dgpk[2], gop[2];
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {                                // h2 = 0: row r0 (c0, c1); h2 = 1: row r1 (c2, c3)
          const float g0 = sigmoid_fast(ga[nt][2 * h2]), g1 = sigmoid_fast(ga[nt][2 * h2 + 1]);
          const float o0 = bf16lo(o2[h2]), o1 = bf16hi(o2[h2]);
          const float d0 = du[nt][2 * h2], d1 = du[nt][2 * h2 + 1];
          dop[h2] = pack_bf16(d0 * g0, d1 * g1);
          dgpk[h2] = pack_bf16(d0 * o0 * g0 * (1.f - g0), d1 * o1 * g1 * (1.f - g1));
          gop[h2] = pack_bf16(g0 * o0, g1 * o1);
        }
        if (v0ok) { *reinterpret_cast<unsigned int*>(p.dO + off[nt][0]) = dop[0]; *reinterpret_cast<unsigned int*>(p.dgp + off[nt][0]) = dgpk[0]; }
        if (v1ok) { *reinterpret_cast<unsigned int*>(p.dO + off[nt][1]) = dop[1]; *reinterpret_cast<unsigned int*>(p.dgp + off[nt][1]) = dgpk[1]; }
        const int chunk = 2 * u + nt;
        sts32(sgo + swz128(r0, chunk) + 4 * q, gop[0]);
        sts32(sgo + swz128(r1, chunk) + 4 * q, gop[1]);
        sts32(sdg + swz128(r0, chunk) + 4 * q, dgpk[0]);
        sts32(sdg + swz128(r1, chunk) + 4 * q, dgpk[1]);
      }
    }
    __syncthreads();                                                    // drb, y, g o and dgp tiles are complete

    // ---- weight-gradient partials over the tile's tokens
    {
      const int wm = warp % G::WMO, wn = warp / G::WMO;
#pragma unroll
      for (int ks = 0; ks < G::NKS; ++ks) {
        uint32_t af[G::RMO][4];
#pragma unroll
        for (int rm = 0; rm < G::RMO; ++rm)
          ldsm_x4_t(af[rm], sdr + swzn<G::NCH>(16 * ks + (lane & 7) + ((lane >> 4) << 3), 2 * (wm * G::RMO + rm) + ((lane >> 3) & 1)));
#pragma unroll
        for (int np = 0; np < G::RNO / 2; ++np) {
          uint32_t bf[4];
          ldsm_x4_t(bf, sgo + swz128(16 * ks + (lane & 7) + (((lane >> 3) & 1) << 3), wn * G::RNO + 2 * np + (lane >> 4)));
#pragma unroll
          for (int rm = 0; rm < G::RMO; ++rm) {
            mma16816(awo[rm][2 * np], af[rm], bf[0], bf[1]);
            mma16816(awo[rm][2 * np + 1], af[rm], bf[2], bf[3]);
          }
        }
      }
    }
    {
      const int wm = warp % G::WMG, wn = warp / G::WMG;
#pragma unroll
      for (int ks = 0; ks < G::NKS; ++ks) {
        uint32_t af[G::RMG][4];
#pragma unroll
        for (int rm = 0; rm < G::RMG; ++rm)
          ldsm_x4_t(af[rm], sdg + swz128(16 * ks + (lane & 7) + ((lane >> 4) << 3), 2 * (wm * G::RMG + rm) + ((lane >> 3) & 1)));
#pragma unroll
        for (int np = 0; np < G::RNG / 2; ++np) {
          uint32_t bf[4];
          ldsm_x4_t(bf, sym + swzn<G::NCH>(16 * ks + (lane & 7) + (((lane >> 3) & 1) << 3), wn * G::RNG + 2 * np + (lane >> 4)));
#pragma unroll
          for (int rm = 0; rm < G::RMG; ++rm) {
            mma16816(awg[rm][2 * np], af[rm], bf[0], bf[1]);
            mma16816(awg[rm][2 * np + 1], af[rm], bf[2], bf[3]);
          }
        }
      }
    }
    __syncthreads();                                                    // the tiles are rewritten by the next iteration's loads
  }

  // ---- the CTA's partials: dWo slice [D][GCH], dWg slice [GCH][D]
  float* part = p.part + ((long)blockIdx.x * G::NG + cg) * (2 * D * G::GCH);
  {
    const int wm = warp % G::WMO, wn = warp / G::WMO;
#pragma unroll
    for (int rm = 0; rm < G::RMO; ++rm)
#pragma unroll
      for (int rn = 0; rn < G::RNO; ++rn) {
        const int row = 16 * (wm * G::RMO + rm) + gq, col = 8 * (wn * G::RNO + rn) + 2 * q;
        stg64(part + (long)row * G::GCH + col, make_float2(awo[rm][rn][0], awo[rm][rn][1]));
        stg64(part + (long)(row + 8) * G::GCH + col, make_float2(awo[rm][rn][2], awo[rm][rn][3]));
      }
  }
  {
    const int wm = warp % G::WMG, wn = warp / G::WMG;
    float* pg = part + D * G::GCH;
#pragma unroll
    for (int rm = 0; rm < G::RMG; ++rm)
#pragma unroll
      for (int rn = 0; rn < G::RNG; ++rn) {
        const int row = 16 * (wm * G::RMG + rm) + gq, col = 8 * (wn * G::RNG + rn) + 2 * q;
        stg64(pg + (long)row * D + col, make_float2(awg[rm][rn][0], awg[rm][rn][1]));
        stg64(pg + (long)(row + 8) * D + col, make_float2(awg[rm][rn][2], awg[rm][rn][3]));
      }
  }
}

}  // namespace pwa80
