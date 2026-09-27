// b7_sm80.cuh -- TriMul backward, input side "source" kernel (B7src), A100 / sm_80.
//
// Per plane channel c (a | b planes) and token t, with x_n = LN_in(z) and the forward's gated projection a = sigmoid(g) p m:
//   dp = dA s m,  dg = dA p s (1 - s) m          (s = sigmoid(g), dA = gradient of the plane from the contraction backward)
//   dW_g += dg (x) x_n,  dW_p += dp (x) x_n
// Channel-stationary (the idea of the sm_90 B7 "source" CTAs): a CTA owns one 64-row weight block (32 plane channels: 8 gate rows | 8 proj rows
// per group, the K1 packing) for a split of the token tiles, so its dW block stays in registers for the whole launch.  It recomputes (g, p) of
// its block (MMA over x_n, normalised in shared memory from the forward statistics), forms (dg, dp), writes them token-major for the dx GEMM
// (columns 64 b .. 64 b + 63 of the dx operand = the K1 weight-row order) and accumulates dW^T = x_n^T . DGP.  The NSTEP block CTAs of a split
// walk the same tiles side by side, so a z tile comes from DRAM once and from L2 for the other blocks.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct B7Params {
  const __nv_bfloat16* z;      // [T][128]
  const float* stats;          // [T][4] (mu_i, r_i at 2, 3)
  const float* gamma; const float* beta;   // LN_in affine [128]
  const __nv_bfloat16* w;      // [NSTEP][16 granules][64 rows][8]  (unscaled K1 packing)
  const __nv_bfloat16* dab;    // [2 CH][T] plane gradients
  const uint8_t* mask;         // [L] token mask or nullptr
  __nv_bfloat16* dgp;          // [T][ldd]
  float* dw;                   // [S][NSTEP * 64][128] partial dW (row = K1 packed weight row)
  int T, L, num_tiles, nstep, splits, ldd;
};

struct B7Cfg {
  static constexpr int CZ = 128, BM = 128, NTHR = 128;
  static constexpr int SMEM_W = 64 * CZ * 2, SMEM_Z = BM * CZ * 2, SMEM_A = 32 * BM * 2, SMEM_D = BM * 64 * 2, SMEM_GB = 2 * CZ * 4;
  static constexpr int SMEM = SMEM_W + SMEM_Z + SMEM_A + SMEM_D + SMEM_GB + BM * 4 * 2;   // + per-row (mu, r) of the tile, mask factor
  static_assert(2 * SMEM <= 166912, "two CTAs per SM");
};

__global__ void __launch_bounds__(128, 2) b7src_kernel(const B7Params p) {
  using G = B7Cfg;
  constexpr int CZ = G::CZ, BM = G::BM;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q = lane & 3, mi = lane >> 3, r8 = lane & 7;
  uint8_t* sW = smem;
  uint8_t* sZ = sW + G::SMEM_W;
  uint8_t* sA = sZ + G::SMEM_Z;
  uint8_t* sD = sA + G::SMEM_A;
  float* sG = reinterpret_cast<float*>(sD + G::SMEM_D);
  float* sB = sG + CZ;
  float* sRow = sB + CZ;                         // [BM][2]: mu, r
  float* sM = sRow + 2 * BM;                     // [BM] mask factor
  const uint32_t sW_u = smem_u32(sW), sZ_u = smem_u32(sZ), sA_u = smem_u32(sA), sD_u = smem_u32(sD);
  const int b = blockIdx.x % p.nstep, s = blockIdx.x / p.nstep;
  const int n_iter = s < p.num_tiles ? (p.num_tiles - s + p.splits - 1) / p.splits : 0;

  for (int i = tid; i < CZ; i += 128) { sG[i] = p.gamma[i]; sB[i] = p.beta[i]; }
#pragma unroll
  for (int i = 0; i < 8; ++i) { const int c = tid + 128 * i; cp_async16(sW_u + c * 16, p.w + (size_t)b * 64 * CZ + c * 8); }   // granule-major block
  cp_async_commit();

  float dwa[2][8][4];                            // dW^T [32 c of this warp][64 rows]
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dwa[mt][n][e] = 0.f;
  const int oc0 = 32 * b;                        // this block's plane channels oc0 .. oc0 + 31

  for (int it = 0; it < n_iter; ++it) {
    const int tile = s + it * p.splits, t0 = tile * BM;
    // ---- loads: z tile [128 tok][128] (256 B rows, granule ^= row & 7), dAB tile [32 ch][128 tok] (256 B rows), per-row stats + mask
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int c = tid + 128 * i, row = c >> 4, g = c & 15;
      const bool ok = t0 + row < p.T;
      cp_async16(sZ_u + swz<256>(row, g * 16), p.z + (size_t)(ok ? t0 + row : 0) * CZ + g * 8, ok ? 16u : 0u);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = tid + 128 * i, row = c >> 4, g = c & 15;
      const bool ok = t0 + 8 * g < p.T;
      cp_async16(sA_u + swz<256>(row, g * 16), p.dab + (size_t)(oc0 + row) * p.T + (ok ? t0 + 8 * g : 0), ok ? 16u : 0u);
    }
    cp_async_commit();
    {
      const int t = t0 + tid;
      float mu = 0.f, r = 0.f, m = 0.f;
      if (t < p.T) {
        const float4 st = *reinterpret_cast<const float4*>(p.stats + (size_t)t * 4);
        mu = st.z; r = st.w; m = 1.f;
        if (p.mask != nullptr) { const int i = t / p.L, j = t - i * p.L; m = (p.mask[i] && p.mask[j]) ? 1.f : 0.f; }
      }
      sRow[2 * tid] = mu; sRow[2 * tid + 1] = r; sM[tid] = m;
    }
    cp_async_wait<0>();
    __syncthreads();
    // ---- x_n = (z - mu) r gamma + beta, in place (thread = token row, 16 granules)
    {
      const float mu = sRow[2 * tid], r = sRow[2 * tid + 1];
#pragma unroll 4
      for (int g = 0; g < 16; ++g) {
        const uint32_t a = sZ_u + swz<256>(tid, g * 16);
        uint4 v = lds128(a);
        uint32_t* w = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int c = 8 * g + 2 * e;
          w[e] = pack_bf16(fmaf((bf16lo(w[e]) - mu) * r, sG[c], sB[c]), fmaf((bf16hi(w[e]) - mu) * r, sG[c + 1], sB[c + 1]));
        }
        sts128(a, v);
      }
    }
    __syncthreads();
    // ---- (g, p) of this warp's 32 tokens x the 64 block rows
    float acc[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mt][n][e] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t a[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int row = 32 * warp + 16 * mt + r8 + ((lane >> 3) & 1) * 8;
        ldsm_x4(a[mt], sZ_u + swz<256>(row, (2 * ks + (lane >> 4)) * 16));
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {           // rows 16 np .. 16 np + 15 = gate(8 ch) | proj(8 ch)
        uint32_t bb[4];
        ldsm_x4(bb, sW_u + ((lane >> 3) & 1) * 1024 + (16 * np + ((lane >> 4) & 1) * 8 + r8) * 16 + ks * 2048);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(acc[mt][2 * np], a[mt], bb[0], bb[1]); mma16816(acc[mt][2 * np + 1], a[mt], bb[2], bb[3]); }
      }
    }
    // ---- dA of (token, channel) in the accumulator layout: ldmatrix.trans of the [ch][tok] tile = A fragment (tok, ch) = C layout of n8 tiles
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kq = 0; kq < 2; ++kq) {           // channels 16 kq .. 16 kq + 15 = groups np = 2 kq, 2 kq + 1
        uint32_t f[4];
        const int krow = 16 * kq + (mi >> 1) * 8 + r8, tg = (32 * warp + 16 * mt) / 8 + (mi & 1);
        ldsm_x4_t(f, sA_u + swz<256>(krow, tg * 16));
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {         // group np = 2 kq + hh: gate tile acc[mt][2 np], proj tile acc[mt][2 np + 1]
          const int np = 2 * kq + hh;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = 32 * warp + 16 * mt + g8 + 8 * h;
            const uint32_t dv = f[2 * hh + h];
            const float m = sM[row];
            const float da0 = bf16lo(dv) * m, da1 = bf16hi(dv) * m;
            const float g0 = acc[mt][2 * np][2 * h], g1 = acc[mt][2 * np][2 * h + 1];
            const float p0 = acc[mt][2 * np + 1][2 * h], p1 = acc[mt][2 * np + 1][2 * h + 1];
            const float s0 = 1.f / (1.f + __expf(-g0)), s1 = 1.f / (1.f + __expf(-g1));
            // staging [128 tok][64 rows] (128 B rows, granule ^= tok & 7): gate row 16 np + 2q (+1) <- dg, proj row 16 np + 8 + 2q <- dp
            const uint32_t base = sD_u + row * 128;
            const int gr = 2 * np, pr = 2 * np + 1;          // 16 B granules of the gate / proj rows
            sts32(base + (((gr ^ (row & 7))) << 4) + q * 4, pack_bf16(da0 * p0 * s0 * (1.f - s0), da1 * p1 * s1 * (1.f - s1)));
            sts32(base + (((pr ^ (row & 7))) << 4) + q * 4, pack_bf16(da0 * s0, da1 * s1));
          }
        }
      }
    __syncthreads();
    // ---- DGP tile -> dx operand columns 64 b .. (token rows of 128 B)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = tid + 128 * i, row = c >> 3, g = c & 7;
      if (t0 + row < p.T) stg128(p.dgp + (size_t)(t0 + row) * p.ldd + 64 * b + 8 * g, lds128(sD_u + row * 128 + ((g ^ (row & 7)) << 4)));
    }
    // ---- dW^T [128 c][64 rows] += x_n^T [c][tok] . DGP [tok][rows]   (warp: c 32 warp .. + 31)
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {             // k = tokens 16 ks .. 16 ks + 15
      uint32_t a[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {           // A (m = c, k = tok) from x_n [tok][c] by .trans: rows = tokens
        const int trow = 16 * ks + (mi >> 1) * 8 + r8, cg = (32 * warp + 16 * mt) / 8 + (mi & 1);
        ldsm_x4_t(a[mt], sZ_u + swz<256>(trow, cg * 16));
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {           // B (k = tok, n = rows) from DGP [tok][rows] by .trans
        uint32_t bb[4];
        const int trow = 16 * ks + (mi & 1) * 8 + r8, rg = 2 * np + (mi >> 1);
        ldsm_x4_t(bb, sD_u + trow * 128 + ((rg ^ (trow & 7)) << 4));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(dwa[mt][2 * np], a[mt], bb[0], bb[1]); mma16816(dwa[mt][2 * np + 1], a[mt], bb[2], bb[3]); }
      }
    }
    __syncthreads();                             // every buffer free for the next tile
  }
  // ---- partial dW (row-major [row][c]) of this split
  float* dst = p.dw + ((size_t)s * p.nstep + b) * 64 * CZ;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int c = 32 * warp + 16 * mt + g8 + 8 * h, rw = 8 * n + 2 * q;
        dst[(size_t)rw * CZ + c] = dwa[mt][n][2 * h];
        dst[(size_t)(rw + 1) * CZ + c] = dwa[mt][n][2 * h + 1];
      }
}

}  // namespace a100
