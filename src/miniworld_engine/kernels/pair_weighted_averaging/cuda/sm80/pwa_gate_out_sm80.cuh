// pwa_gate_out_sm80.cuh -- MSA pair-weighted averaging forward, the back end: the sigmoid gate, the output projection, the module's row dropout and the residual
//
//   y[s, i, :]    = bf16( LN(m[s, i, :]) )                                       (recomputed from m: the same arithmetic as `ln_v`)
//   g[s, i, hc]   = sigmoid( y . Wg[hc, :] )                                     per unit of 16 channels
//   u             = bf16( g * o[h, i, s * C + c] )                               o: the cuBLAS product w v, head-major [8, L, S C]
//   upd[s, i, :]  = bf16( sum_hc u Wo[:, hc] );  upd = bf16( upd * keep[i, :] * dscale ) when the dropout is live
//   out[s, i, :]  = bf16( m[s, i, :] + upd )
//
// A CTA owns one query row i and 128 consecutive MSA rows (8 warps x 16 rows) and loops over tiles (persistent: Wg and Wo stay in shared memory).  The m tile
// arrives by cp.async (double-buffered, swizzled rows); the LayerNorm runs on the A fragments (ldmatrix, quad shuffles) and its output stays in registers as the A
// fragments of the gate GEMM.  The contraction output o is the stream that decides the speed (201 MB at L384 / S1024): it is read by a cp.async ring of 8-KiB stages
// ([128 tokens][32 channels] -- C = 32, 16, 8 alike: a stage is 32 consecutive channels, i.e. one head, two or four -- in 64-byte swizzled rows), several stages in
// flight, and ldmatrix hands the fragments over in the accumulator layout.  The head loop is a loop over units of 16 channels: the gate for the unit is two n8 tiles
// of mma (K = d_msa), u = g o is the A fragment of the unit's slice of the output projection (accumulator = A layout), which accumulates over the units in registers.
// The output replaces the m tile in shared memory and is stored row by row.
#pragma once
#include "sm80_common.cuh"

namespace pwa80 {

struct GateOutParams {
  const __nv_bfloat16* m;      // [S][L][D]
  const __nv_bfloat16* o;      // [8][L][S C]
  const float* lnw;            // [D]
  const float* lnb;            // [D]
  const __nv_bfloat16* wg;     // [8 C][D]
  const __nv_bfloat16* wo;     // [D][8 C]
  const __nv_bfloat16* keep;   // [L][D] (0 / 1) or nullptr
  __nv_bfloat16* out;          // [S][L][D]
  int S, L, ntile_s, ntile, kp, ns;                  // kp, ns: the MSA rows in ns chunks of kp (see hm_row)
  float eps, dscale;
};

template <int D, int C> struct GoCfg {
  static constexpr int NTHR = 256, NTOK = 128, NCH = D / 8, HC = 8 * C, NU = HC / 16, NCW = HC / 8, NS = HC / 32;   // NS: o stages (32 channels) per tile
  static constexpr int NR = D == 64 ? 4 : 3;                                          // o ring depth (stages in flight + 1)
  static constexpr int MB = NTOK * D * 2, WGB = HC * D * 2, WOB = D * HC * 2, STG = NTOK * 64;
  static constexpr int SMEM = WGB + WOB + 2 * MB + NR * STG + 2 * D * 4;
  static_assert((D == 64 || D == 128) && (C == 8 || C == 16 || C == 32), "d_msa / d_hidden");
  static_assert(SMEM <= 166912, "sm_80 shared memory");
};

// 64-byte rows (four 16-byte chunks): chunk ^ ((row >> 1) & 3) puts the eight rows of an ldmatrix / the eight lanes of a cp.async write phase on eight bank groups
DEVI uint32_t swz64(uint32_t row, uint32_t chunk) { return row * 64u + (((chunk ^ (row >> 1)) & 3u) << 4); }

template <int D, int C>
DEVI void go_load_m(const GateOutParams& p, uint32_t dst, int tl, int tid) {
  using G = GoCfg<D, C>;
  const int i = tl / p.ntile_s, s0 = (tl % p.ntile_s) * G::NTOK;
#pragma unroll
  for (int it = 0; it < G::NTOK * G::NCH / G::NTHR; ++it) {
    const int v = it * G::NTHR + tid, ch = v % G::NCH, t = v / G::NCH;
    const bool ok = s0 + t < p.S;
    cp_async16(dst + swzn<G::NCH>(t, ch), p.m + (ok ? ((long)(s0 + t) * p.L + i) * D + ch * 8 : 0), ok ? 16u : 0u);
  }
}

// one stage of o: tokens x channels [32 stg, 32 stg + 32) of tile `tl`
template <int D, int C>
DEVI void go_load_o(const GateOutParams& p, uint32_t dst, int tl, int stg, int tid) {
  using G = GoCfg<D, C>;
  const int i = tl / p.ntile_s, s0 = (tl % p.ntile_s) * G::NTOK;
#pragma unroll
  for (int it = 0; it < G::NTOK * 4 / G::NTHR; ++it) {
    const int v = it * G::NTHR + tid, x = v & 3, t = v >> 2;
    const int cabs = 32 * stg + 8 * x, h = cabs / C, cc = cabs % C;
    const bool ok = s0 + t < p.S;
    cp_async16(dst + swz64(t, x), p.o + (ok ? (hm_row(h, i, s0, p.L, p.kp, p.ns) + t) * C + cc : 0), ok ? 16u : 0u);
  }
}

template <int D, int C>
__global__ void __launch_bounds__(256, 1) pwa_gate_out_kernel(const GateOutParams p) {
  using G = GoCfg<D, C>;
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t swg = smem_u32(smem), swo = swg + G::WGB, sm0 = swo + G::WOB, sor = sm0 + 2 * G::MB;
  float* sgb = reinterpret_cast<float*>(smem + G::WGB + G::WOB + 2 * G::MB + G::NR * G::STG);       // gamma [D], beta [D]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;

  for (int u = tid; u < G::HC * G::NCH; u += G::NTHR) {
    const int row = u / G::NCH, ch = u % G::NCH;
    cp_async16(swg + swzn<G::NCH>(row, ch), p.wg + (long)row * D + ch * 8);
  }
  for (int u = tid; u < D * G::NCW; u += G::NTHR) {
    const int row = u / G::NCW, ch = u % G::NCW;
    cp_async16(swo + swzn<G::NCW>(row, ch), p.wo + (long)row * G::HC + ch * 8);
  }
  for (int k = tid; k < D; k += G::NTHR) { sgb[k] = p.lnw[k]; sgb[D + k] = p.lnb[k]; }
  cp_async_commit();

  // the stream of o stages: n = 0, 1, ... enumerates (tile, stage) of this CTA in order.  The m rows of tile k + 1 are requested at the first stage of tile k (a whole tile
  // ahead: the buffer they land in was last read by tile k - 1, which every warp has left at the barrier above).
  const int ntiles_mine = p.ntile > (int)blockIdx.x ? (p.ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int nstream = ntiles_mine * G::NS;
  auto issue = [&](int n) {
    if (n < nstream) {
      const int k = n / G::NS, st = n % G::NS, tl = blockIdx.x + k * gridDim.x;
      go_load_o<D, C>(p, sor + (n % G::NR) * G::STG, tl, st, tid);
    }
    cp_async_commit();
  };
  if (ntiles_mine > 0) go_load_m<D, C>(p, sm0, blockIdx.x, tid);                         // tile 0's rows ride with the first o stage's group
#pragma unroll
  for (int n = 0; n < G::NR - 1; ++n) issue(n);

  int n = 0;                                                                              // the stage being consumed
#pragma unroll 1
  for (int k = 0; k < ntiles_mine; ++k) {
    const int tl = blockIdx.x + k * gridDim.x;
    const int i = tl / p.ntile_s, s0 = (tl % p.ntile_s) * G::NTOK;
    const uint32_t sm = sm0 + (k & 1) * G::MB;
    const int r0 = warp * 16 + gq, r1 = r0 + 8;

    cp_async_wait<G::NR - 2>();                                                            // stage n (and the m tile of this tile) has landed
    __syncthreads();
    if (k + 1 < ntiles_mine) go_load_m<D, C>(p, sm0 + ((k + 1) & 1) * G::MB, blockIdx.x + (k + 1) * gridDim.x, tid);
    issue(n + G::NR - 1);

    // ---- LayerNorm of the warp's 16 rows on the A fragments -> ya (bf16 A fragments of the gate GEMM)
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

    // ---- units of 16 channels: gate GEMM, u = g o, the unit's slice of the output projection
    float oacc[D / 8][4];
#pragma unroll
    for (int nt = 0; nt < D / 8; ++nt) oacc[nt][0] = oacc[nt][1] = oacc[nt][2] = oacc[nt][3] = 0.f;
#pragma unroll
    for (int st = 0; st < G::NS; ++st) {
      if (st > 0) {
        cp_async_wait<G::NR - 2>();
        __syncthreads();
        issue(n + G::NR - 1);
      }
      const uint32_t so = sor + (n % G::NR) * G::STG;
#pragma unroll
      for (int ul = 0; ul < 2; ++ul) {
        const int u = 2 * st + ul;
        uint32_t ov[4];                                                                // o of the unit in the accumulator layout: (r0, nt 0) (r1, nt 0) (r0, nt 1) (r1, nt 1)
        ldsm_x4(ov, so + swz64(warp * 16 + (lane & 15), 2 * ul + (lane >> 4)));
        float ga[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) ga[nt][0] = ga[nt][1] = ga[nt][2] = ga[nt][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
          uint32_t bf[4];
          ldsm_x4(bf, swg + swzn<G::NCH>(16 * u + (lane & 7) + ((lane >> 4) << 3), 2 * ks + ((lane >> 3) & 1)));
          mma16816(ga[0], ya[ks], bf[0], bf[1]);
          mma16816(ga[1], ya[ks], bf[2], bf[3]);
        }
        uint32_t ua[4];
        ua[0] = pack_bf16(sigmoid_fast(ga[0][0]) * bf16lo(ov[0]), sigmoid_fast(ga[0][1]) * bf16hi(ov[0]));
        ua[1] = pack_bf16(sigmoid_fast(ga[0][2]) * bf16lo(ov[1]), sigmoid_fast(ga[0][3]) * bf16hi(ov[1]));
        ua[2] = pack_bf16(sigmoid_fast(ga[1][0]) * bf16lo(ov[2]), sigmoid_fast(ga[1][1]) * bf16hi(ov[2]));
        ua[3] = pack_bf16(sigmoid_fast(ga[1][2]) * bf16lo(ov[3]), sigmoid_fast(ga[1][3]) * bf16hi(ov[3]));
#pragma unroll
        for (int np = 0; np < D / 16; ++np) {
          uint32_t bf[4];
          ldsm_x4(bf, swo + swzn<G::NCW>(16 * np + (lane & 7) + ((lane >> 4) << 3), 2 * u + ((lane >> 3) & 1)));
          mma16816(oacc[2 * np], ua, bf[0], bf[1]);
          mma16816(oacc[2 * np + 1], ua, bf[2], bf[3]);
        }
      }
      ++n;
    }

    // ---- update -> (dropout) -> + residual, in place over the m tile
#pragma unroll
    for (int nt = 0; nt < D / 8; ++nt) {
      const int col = nt * 8 + 2 * q;
      uint32_t kp = 0x3f803f80u;                                                        // keep (1, 1) in bf16
      if (p.keep != nullptr) kp = __ldg(reinterpret_cast<const unsigned int*>(p.keep + (long)i * D + col));
#pragma unroll
      for (int h2 = 0; h2 < 2; ++h2) {
        const int row = h2 == 0 ? r0 : r1;
        const uint32_t addr = sm + swzn<G::NCH>(row, col >> 3) + (col & 7) * 2;
        const uint32_t xr = lds32(addr);
        float u0 = round_bf16f(oacc[nt][2 * h2]), u1 = round_bf16f(oacc[nt][2 * h2 + 1]);
        if (p.keep != nullptr) {
          u0 = round_bf16f(u0 * bf16lo(kp) * p.dscale);
          u1 = round_bf16f(u1 * bf16hi(kp) * p.dscale);
        }
        sts32(addr, pack_bf16(bf16lo(xr) + u0, bf16hi(xr) + u1));
      }
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < G::NTOK * G::NCH / G::NTHR; ++it) {
      const int v = it * G::NTHR + tid, ch = v % G::NCH, t = v / G::NCH;
      if (s0 + t < p.S) stg128(p.out + ((long)(s0 + t) * p.L + i) * D + ch * 8, lds128(sm + swzn<G::NCH>(t, ch)));
    }
  }
  cp_async_wait<0>();
}

}  // namespace pwa80
