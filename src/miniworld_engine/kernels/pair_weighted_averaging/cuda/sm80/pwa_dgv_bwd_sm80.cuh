// pwa_dgv_bwd_sm80.cuh -- MSA pair-weighted averaging backward, last MSA-side stage: from the gradients of the gate and of the value (dgp, dv: head-major
// [8, L, S C], the glue's and the cuBLAS bmm's outputs) to the MSA gradient and the value weight gradient
//
//   dy             = dgp . Wg + dv . Wv                                         (fp32, mma.sync, K = 8 C)
//   dm             = bf16( LN_backward(dy; m, gamma) + dres )                    (the residual's gradient joins here)
//   dgamma, dbeta += sum_tokens dy xhat, dy
//   dWv[hc, d]    += sum_tokens dv[t, hc] y[t, d]                                (y = bf16(LN(m)) recomputed; fp32 partial per CTA)
//
// A CTA owns tiles of one position j x NTOK = 128 consecutive MSA rows (8 warps x 16, persistent: Wg and Wv stay in shared memory).  Everything the tile reads
// besides m arrives by cp.async as a STREAM OF 16-KiB ITEMS ([128 tokens][64 channels], 128-byte swizzled rows) through a ring of NRING slots: per group of 64 channels
// the dgp chunk and the dv chunk, then the dres columns; an item's slot is refilled with item + NRING as soon as the tile has consumed it (a barrier after the last
// reader), so the loads run one to two groups ahead of the arithmetic and no global load sits on the mma chain.  m has its own buffer (y over m in place; the next tile's
// m is requested once the tile's last dWv product has read y).  Per group the dy products read their A fragments by ldmatrix straight in the accumulator layout (the A
// fragments of a K = 16 step) and multiply with the unit's rows of Wg / Wv (ldmatrix.trans); the dWv contraction over the tokens runs on the dv chunk and y.  The LayerNorm
// backward runs on dy's accumulator layout (a row = a quad of lanes), dm replaces dres in its slot, dgamma / dbeta are summed per warp into shared memory (exclusive rows).
#pragma once
#include "sm80_common.cuh"

namespace pwa80 {

struct DgvParams {
  const __nv_bfloat16* m;      // [S][L][D]
  const __nv_bfloat16* dres;   // [S][L][D]
  const __nv_bfloat16* dgp;    // [8][L][S C]
  const __nv_bfloat16* dv;     // [8][L][S C]
  const float* lnw;
  const float* lnb;
  const __nv_bfloat16* wg;     // [HC][D]
  const __nv_bfloat16* wv;     // [HC][D]
  __nv_bfloat16* dm;           // [S][L][D]
  float* part;                 // [gridDim.x][HC D + 2 D]: dWv [HC][D], dgamma [D], dbeta [D]
  int S, L, ntile_s, ntile, kp, ns;                  // kp, ns: the MSA rows in ns chunks of kp (see hm_row)
  float eps;
};

template <int D, int C> struct DgvCfg {
  static constexpr int HC = 8 * C, NG = HC / 64, NTHR = 256, NTOK = 128, NCH = D / 8;
  static constexpr int NRING = (D == 128 && C == 16) ? 3 : 4;                 // slots of the item ring (3: the weights leave no room for a fourth)
  static constexpr int NDRES = D / 64, NITEM = 2 * NG + NDRES;                 // items per tile: (dgp, dv) of every group, then the dres columns
  static constexpr int WB = HC * D * 2, YB = NTOK * D * 2, CB = NTOK * 128;    // weights, the m / y tile, one item
  static constexpr int SMEM = 2 * WB + YB + NRING * CB + 2 * D * 4 + 8 * 2 * D * 4;
  static constexpr int RN = D / 16;                      // n8 tiles per warp of the dWv group block: m = 4 x 16 (hc), n = 2 warps x D / 2
  static_assert((D == 64 || D == 128) && HC % 64 == 0, "layout");
  static_assert(NDRES <= NRING, "ring");
  static_assert(SMEM <= 166912, "sm_80 shared memory");
};

template <int D, int C>
DEVI void dg_load_m(const DgvParams& p, uint32_t sym, int tl, int tid) {
  using G = DgvCfg<D, C>;
  const int j = tl / p.ntile_s, s0 = (tl % p.ntile_s) * G::NTOK;
#pragma unroll
  for (int it = 0; it < G::NTOK * G::NCH / G::NTHR; ++it) {
    const int v = it * G::NTHR + tid, ch = v % G::NCH, t = v / G::NCH;
    const bool ok = s0 + t < p.S;
    cp_async16(sym + swzn<G::NCH>(t, ch), p.m + (ok ? ((long)(s0 + t) * p.L + j) * D + ch * 8 : 0), ok ? 16u : 0u);
  }
}

// stream item n (tile k = n / NITEM of this CTA, item i = n % NITEM of the tile) into its slot; items past the last tile load nothing (the caller still commits a group)
template <int D, int C>
DEVI void dg_load_item(const DgvParams& p, uint32_t slot, int n, int tid) {
  using G = DgvCfg<D, C>;
  const int k = n / G::NITEM, i = n % G::NITEM, tl = (int)blockIdx.x + k * (int)gridDim.x;
  if (tl >= p.ntile) return;
  const int j = tl / p.ntile_s, s0 = (tl % p.ntile_s) * G::NTOK;
#pragma unroll
  for (int it = 0; it < G::NTOK * 8 / G::NTHR; ++it) {
    const int v = it * G::NTHR + tid, x = v & 7, t = v >> 3;
    const bool ok = s0 + t < p.S;
    const __nv_bfloat16* src;
    long off;
    if (i < 2 * G::NG) {                                                    // dgp / dv: channels 64 g + 8 x .. of the head-major run of (h, j)
      const int cabs = 64 * (i >> 1) + 8 * x, h = cabs / C, cc = cabs % C;
      src = (i & 1) ? p.dv : p.dgp;
      off = (hm_row(h, j, s0, p.L, p.kp, p.ns) + t) * C + cc;
    } else {                                                                // dres: columns 64 c + 8 x ..
      src = p.dres;
      off = ((long)(s0 + t) * p.L + j) * D + 64 * (i - 2 * G::NG) + 8 * x;
    }
    cp_async16(slot + swz128(t, x), src + (ok ? off : 0), ok ? 16u : 0u);
  }
}

template <int D, int C>
__global__ void __launch_bounds__(256, 1) pwa_dgv_bwd_kernel(const DgvParams p) {
  using G = DgvCfg<D, C>;
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t swg = smem_u32(smem), swv = swg + G::WB, sym = swv + G::WB, sring = sym + G::YB;
  float* sgb = reinterpret_cast<float*>(smem + 2 * G::WB + G::YB + G::NRING * G::CB);        // gamma [D], beta [D]
  float* sred = sgb + 2 * D;                                                                   // [8 warps][dgamma D | dbeta D]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;

  for (int u = tid; u < G::HC * G::NCH; u += G::NTHR) {
    const int row = u / G::NCH, ch = u % G::NCH;
    cp_async16(swg + swzn<G::NCH>(row, ch), p.wg + (long)row * D + ch * 8);
    cp_async16(swv + swzn<G::NCH>(row, ch), p.wv + (long)row * D + ch * 8);
  }
  for (int k = tid; k < D; k += G::NTHR) { sgb[k] = p.lnw[k]; sgb[D + k] = p.lnb[k]; }
  for (int k = tid; k < 8 * 2 * D; k += G::NTHR) sred[k] = 0.f;
  if ((int)blockIdx.x < p.ntile) dg_load_m<D, C>(p, sym, blockIdx.x, tid);
  cp_async_commit();                                                                          // group: weights + the first tile's m
  auto issue = [&](int n) {                                                                    // stream item n into slot n % NRING (always one group)
    dg_load_item<D, C>(p, sring + (n % G::NRING) * G::CB, n, tid);
    cp_async_commit();
  };
#pragma unroll
  for (int n0 = 0; n0 < G::NRING; ++n0) issue(n0);

  float awv[G::NG][G::RN][4];
#pragma unroll
  for (int a = 0; a < G::NG; ++a)
#pragma unroll
    for (int b = 0; b < G::RN; ++b) awv[a][b][0] = awv[a][b][1] = awv[a][b][2] = awv[a][b][3] = 0.f;

  const int r0 = warp * 16 + gq, r1 = r0 + 8;
  int n = 0;                                                                                   // the next stream item to consume
#pragma unroll 1
  for (int tl = blockIdx.x, kt = 0; tl < p.ntile; tl += gridDim.x, ++kt) {
    const int j = tl / p.ntile_s, s0 = (tl % p.ntile_s) * G::NTOK;
    if (kt == 0) cp_async_wait<G::NRING>(); else cp_async_wait<G::NDRES>();                    // this tile's m (and, first time, the weights) has landed
    __syncthreads();

    // ---- LayerNorm statistics; y over the m tile (the dWv operand); xa stays in registers for xhat
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
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      const int col = 16 * ks + 2 * q;
      const float2 g0 = *reinterpret_cast<const float2*>(sgb + col), g1 = *reinterpret_cast<const float2*>(sgb + col + 8);
      const float2 b0 = *reinterpret_cast<const float2*>(sgb + D + col), b1 = *reinterpret_cast<const float2*>(sgb + D + col + 8);
      sts32(sym + swzn<G::NCH>(r0, 2 * ks) + 4 * q, pack_bf16(fmaf((bf16lo(xa[ks][0]) - mean0) * rs0, g0.x, b0.x), fmaf((bf16hi(xa[ks][0]) - mean0) * rs0, g0.y, b0.y)));
      sts32(sym + swzn<G::NCH>(r1, 2 * ks) + 4 * q, pack_bf16(fmaf((bf16lo(xa[ks][1]) - mean1) * rs1, g0.x, b0.x), fmaf((bf16hi(xa[ks][1]) - mean1) * rs1, g0.y, b0.y)));
      sts32(sym + swzn<G::NCH>(r0, 2 * ks + 1) + 4 * q, pack_bf16(fmaf((bf16lo(xa[ks][2]) - mean0) * rs0, g1.x, b1.x), fmaf((bf16hi(xa[ks][2]) - mean0) * rs0, g1.y, b1.y)));
      sts32(sym + swzn<G::NCH>(r1, 2 * ks + 1) + 4 * q, pack_bf16(fmaf((bf16lo(xa[ks][3]) - mean1) * rs1, g1.x, b1.x), fmaf((bf16hi(xa[ks][3]) - mean1) * rs1, g1.y, b1.y)));
    }

    // ---- dy = dgp Wg + dv Wv, per group of 64 channels (four units of 16); the dv chunk is also the dWv operand
    float dy[D / 8][4];
#pragma unroll
    for (int nt = 0; nt < D / 8; ++nt) dy[nt][0] = dy[nt][1] = dy[nt][2] = dy[nt][3] = 0.f;
#pragma unroll
    for (int gi = 0; gi < G::NG; ++gi) {
      cp_async_wait<G::NRING - 2>();
      __syncthreads();                                                  // items n (dgp) and n + 1 (dv) have landed; on the first group y is complete too
      const uint32_t sdgp = sring + (n % G::NRING) * G::CB, sdv = sring + ((n + 1) % G::NRING) * G::CB;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        uint32_t ag[4];                                                 // the unit's dgp: the A fragments of the K = 16 step (rows r0 / r1, channels 2 q, 2 q + 8)
        ldsm_x4(ag, sdgp + swz128(warp * 16 + (lane & 15), 2 * u + (lane >> 4)));
#pragma unroll
        for (int np = 0; np < D / 16; ++np) {
          uint32_t bf[4];
          ldsm_x4_t(bf, swg + swzn<G::NCH>(gi * 64 + 16 * u + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * np + (lane >> 4)));
          mma16816(dy[2 * np], ag, bf[0], bf[1]);
          mma16816(dy[2 * np + 1], ag, bf[2], bf[3]);
        }
      }
      if (G::NRING == 3) {                                              // three slots: the dgp slot is refilled as soon as every warp has read it
        __syncthreads();
        issue(n + G::NRING);
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        uint32_t av[4];
        ldsm_x4(av, sdv + swz128(warp * 16 + (lane & 15), 2 * u + (lane >> 4)));
#pragma unroll
        for (int np = 0; np < D / 16; ++np) {
          uint32_t bf[4];
          ldsm_x4_t(bf, swv + swzn<G::NCH>(gi * 64 + 16 * u + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * np + (lane >> 4)));
          mma16816(dy[2 * np], av, bf[0], bf[1]);
          mma16816(dy[2 * np + 1], av, bf[2], bf[3]);
        }
      }
      {                                                                 // dWv group block [64 hc][D]: m tile wm, n8 tiles wn * RN ..
        const int wm = warp & 3, wn = warp >> 2;
#pragma unroll
        for (int ks = 0; ks < G::NTOK / 16; ++ks) {
          uint32_t af[4];
          ldsm_x4_t(af, sdv + swz128(16 * ks + (lane & 7) + ((lane >> 4) << 3), 2 * wm + ((lane >> 3) & 1)));
#pragma unroll
          for (int np = 0; np < G::RN / 2; ++np) {
            uint32_t bf[4];
            ldsm_x4_t(bf, sym + swzn<G::NCH>(16 * ks + (lane & 7) + (((lane >> 3) & 1) << 3), wn * G::RN + 2 * np + (lane >> 4)));
            mma16816(awv[gi][2 * np], af, bf[0], bf[1]);
            mma16816(awv[gi][2 * np + 1], af, bf[2], bf[3]);
          }
        }
      }
      __syncthreads();                                                  // every warp is done with both chunks: refill the slots
      if (G::NRING == 4) { issue(n + G::NRING); issue(n + 1 + G::NRING); }
      else issue(n + 1 + G::NRING);
      n += 2;
    }

    // ---- the dres columns have landed (items n ..); the next tile's m reuses the y tile
    cp_async_wait<G::NRING - G::NDRES>();
    __syncthreads();
    if (tl + (int)gridDim.x < p.ntile) dg_load_m<D, C>(p, sym, tl + gridDim.x, tid);
    cp_async_commit();

    // ---- LayerNorm backward on dy's accumulator layout; dm = dx + dres in place over the dres columns
    {
      float s1A = 0.f, s2A = 0.f, s1B = 0.f, s2B = 0.f;
      // xhat at the accumulator positions (n8 tile nt = 2 ks + b: the A fragments a0 / a1 (b = 0) and a2 / a3 (b = 1) of ks hold the same (row, column) pairs)
      auto xh = [&](int nt, int e) {
        const uint32_t w = xa[nt >> 1][2 * (nt & 1) + (e >> 1)];
        const float xv = (e & 1) ? bf16hi(w) : bf16lo(w);
        return (e >> 1) ? (xv - mean1) * rs1 : (xv - mean0) * rs0;
      };
#pragma unroll
      for (int nt = 0; nt < D / 8; ++nt) {
        const float2 gm = *reinterpret_cast<const float2*>(sgb + nt * 8 + 2 * q);
        const float a0 = dy[nt][0] * gm.x, a1 = dy[nt][1] * gm.y, b0 = dy[nt][2] * gm.x, b1 = dy[nt][3] * gm.y;
        s1A += a0 + a1; s2A = fmaf(a0, xh(nt, 0), fmaf(a1, xh(nt, 1), s2A));
        s1B += b0 + b1; s2B = fmaf(b0, xh(nt, 2), fmaf(b1, xh(nt, 3), s2B));
      }
      s1A = quad_sum(s1A) * (1.f / D); s2A = quad_sum(s2A) * (1.f / D); s1B = quad_sum(s1B) * (1.f / D); s2B = quad_sum(s2B) * (1.f / D);
#pragma unroll
      for (int nt = 0; nt < D / 8; ++nt) {
        const int col = nt * 8 + 2 * q;
        const float2 gm = *reinterpret_cast<const float2*>(sgb + col);
        const uint32_t sd = sring + ((n + (nt >> 3)) % G::NRING) * G::CB;           // the dres item of columns 64 (nt >> 3) ..
        const uint32_t aa = sd + swz128(r0, nt & 7) + (col & 7) * 2, ab = sd + swz128(r1, nt & 7) + (col & 7) * 2;
        const uint32_t ra = lds32(aa), rb = lds32(ab);
        const float dxa0 = rs0 * (dy[nt][0] * gm.x - s1A - xh(nt, 0) * s2A), dxa1 = rs0 * (dy[nt][1] * gm.y - s1A - xh(nt, 1) * s2A);
        const float dxb0 = rs1 * (dy[nt][2] * gm.x - s1B - xh(nt, 2) * s2B), dxb1 = rs1 * (dy[nt][3] * gm.y - s1B - xh(nt, 3) * s2B);
        sts32(aa, pack_bf16(dxa0 + bf16lo(ra), dxa1 + bf16hi(ra)));
        sts32(ab, pack_bf16(dxb0 + bf16lo(rb), dxb1 + bf16hi(rb)));
        // dgamma = sum dy xhat, dbeta = sum dy: the eight row lanes of this column first, then the warp's exclusive shared row
        float gA = dy[nt][0] * xh(nt, 0) + dy[nt][2] * xh(nt, 2), gB = dy[nt][1] * xh(nt, 1) + dy[nt][3] * xh(nt, 3);
        float bA = dy[nt][0] + dy[nt][2], bB = dy[nt][1] + dy[nt][3];
        gA += __shfl_xor_sync(0xffffffffu, gA, 4); gA += __shfl_xor_sync(0xffffffffu, gA, 8); gA += __shfl_xor_sync(0xffffffffu, gA, 16);
        gB += __shfl_xor_sync(0xffffffffu, gB, 4); gB += __shfl_xor_sync(0xffffffffu, gB, 8); gB += __shfl_xor_sync(0xffffffffu, gB, 16);
        bA += __shfl_xor_sync(0xffffffffu, bA, 4); bA += __shfl_xor_sync(0xffffffffu, bA, 8); bA += __shfl_xor_sync(0xffffffffu, bA, 16);
        bB += __shfl_xor_sync(0xffffffffu, bB, 4); bB += __shfl_xor_sync(0xffffffffu, bB, 8); bB += __shfl_xor_sync(0xffffffffu, bB, 16);
        if (gq == 0) {
          float* wr = sred + warp * 2 * D;
          wr[col] += gA; wr[col + 1] += gB; wr[D + col] += bA; wr[D + col + 1] += bB;
        }
      }
    }
    __syncthreads();                                                    // dm complete in the dres slots
#pragma unroll
    for (int it = 0; it < G::NDRES * G::NTOK * 8 / G::NTHR; ++it) {
      const int v = it * G::NTHR + tid, c = v / (G::NTOK * 8), w = v % (G::NTOK * 8), x = w & 7, t = w >> 3;
      if (s0 + t < p.S) stg128(p.dm + ((long)(s0 + t) * p.L + j) * D + 64 * c + 8 * x, lds128(sring + ((n + c) % G::NRING) * G::CB + swz128(t, x)));
    }
    __syncthreads();                                                    // the dres slots are read out: refill them (the next tile's first items)
#pragma unroll
    for (int c = 0; c < G::NDRES; ++c) issue(n + c + G::NRING);
    n += G::NDRES;
  }
  cp_async_wait<0>();

  // ---- the CTA's partial row: dWv [HC][D] | dgamma | dbeta
  float* part = p.part + (long)blockIdx.x * (G::HC * D + 2 * D);
  {
    const int wm = warp & 3, wn = warp >> 2;
#pragma unroll
    for (int gi = 0; gi < G::NG; ++gi)
#pragma unroll
      for (int rn = 0; rn < G::RN; ++rn) {
        const int row = gi * 64 + 16 * wm + gq, col = 8 * (wn * G::RN + rn) + 2 * q;
        stg64(part + (long)row * D + col, make_float2(awv[gi][rn][0], awv[gi][rn][1]));
        stg64(part + (long)(row + 8) * D + col, make_float2(awv[gi][rn][2], awv[gi][rn][3]));
      }
  }
  __syncthreads();
  for (int col = tid; col < 2 * D; col += G::NTHR) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += sred[w * 2 * D + col];
    part[G::HC * D + col] = s;
  }
}

}  // namespace pwa80
