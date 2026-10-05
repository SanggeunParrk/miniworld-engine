// pwa_pair_bwd_sm80.cuh -- MSA pair-weighted averaging backward, the pair side: the softmax backward, the bias-projection backward and the LayerNorm backward
// of the pair, one CTA (8 warps) per query row i.
//
//   sdot[h]       = sum_j w[h, i, j] dw[h, i, j]
//   db[h, j]      = bf16( w (dw - sdot) ),  0 on masked keys                  (the module's bias gradient is bf16)
//   dzn[j, :]     = sum_h db[h, j] Wb[h, :]                                   (mma, K = 8 heads padded to 16; fp32)
//   dz[i, j, :]   = bf16( rstd (g - mean(g) - xhat mean(g xhat)) ),  g = dzn gamma, xhat = (z - mean) rstd           (z read once more; LN statistics recomputed)
//   M[h, d]      += sum_j db[h, j] xhat[j, d],  S[h] += sum_j db[h, j]        per row; dWb = gamma M + beta S, dgamma = sum_h Wb M, dbeta = sum_h Wb S
//
// The softmax backward (phase 0) runs one warp per head and leaves db as bf16 [L][8] in shared memory.  Phase 1: a warp takes 16 keys at a time; the tile (cp.async,
// swizzled rows) is read by ldmatrix three times (statistics, the row sums of g and g xhat, then dz): dzn is recomputed per 64-column chunk (one mma per n8 tile),
// never held for the whole row.  M accumulates on the tensor cores: xhat (bf16, in the A-fragment layout of the tile) is transposed in registers with movmatrix into the
// A^T fragments of M^T = xhat^T db, accumulated in registers across the warp's tiles; the eight warps' partials are summed in a fixed order through shared memory and written
// per row (the caller sums the L rows: no atomics, bit-reproducible).
#pragma once
#include "sm80_common.cuh"

namespace pwa80 {

struct PairBwdParams {
  const __nv_bfloat16* z;      // [L][L][DZ]
  const __nv_bfloat16* w;      // [8][L][L]
  const float* dw;             // [8 ns][L][L]: ns partial products per head, summed here (in order)
  const uint8_t* mask;         // [L] key mask or nullptr
  const float* lnw;
  const float* lnb;
  const __nv_bfloat16* wb;     // [8][DZ]
  __nv_bfloat16* dz;           // [L][L][DZ]
  float* pM;                   // [L][8][DZ]
  float* pS;                   // [L][8]
  int L, ns;                   // ns: the dw partials per head (the contraction over S C in ns batches)
  float eps;
};

template <int DZ> struct PairBwdCfg {
  static constexpr int NW = 8, NTHR = 256, GJ = 16, NK = DZ / 16, NCHK = DZ / 8;
  static constexpr int NBUF = DZ <= 256 ? 2 : 1;                       // tile buffers per warp (the next tile's rows arrive while this one is processed)
  static constexpr int TILE = GJ * DZ * 2, TILES = NW * NBUF * TILE;
  static constexpr int PREP = DZ * 4 * 4 + 2 * DZ * 4;                 // wbt [DZ][4] words, gamma / beta
  static constexpr int NCH64 = DZ / 64;                                // 64-column chunks (4 ks, 8 n8 tiles)
  static_assert(NW * DZ * 8 * 4 <= TILES, "the partials of M reuse the tile buffers");
};

DEVI uint32_t movT(uint32_t v) {                                        // the transposed 8x8 b16 matrix of a warp-wide fragment
  uint32_t r;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(r) : "r"(v));
  return r;
}

template <int DZ> DEVI void pb_load_tile(uint32_t dst, const __nv_bfloat16* src_row0, int lane) {
  using C = PairBwdCfg<DZ>;
#pragma unroll
  for (int qq = 0; qq < C::GJ * C::NCHK / 32; ++qq) {
    const int v = qq * 32 + lane, r = v / C::NCHK, ch = v % C::NCHK;
    cp_async16(dst + swzn<C::NCHK>(r, ch), src_row0 + (long)r * DZ + ch * 8);
  }
}

template <int DZ, int NMINB>
__global__ void __launch_bounds__(256, NMINB) pwa_pair_bwd_kernel(const PairBwdParams p) {
  using C = PairBwdCfg<DZ>;
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t sbase = smem_u32(smem);
  uint32_t* wbt = reinterpret_cast<uint32_t*>(smem + C::TILES);                       // [DZ][4]
  float* sg = reinterpret_cast<float*>(smem + C::TILES + DZ * 16);                    // gamma [DZ], beta [DZ]
  __nv_bfloat16* sdb = reinterpret_cast<__nv_bfloat16*>(smem + C::TILES + C::PREP);   // [L][8] db (bf16)
  const int L = p.L;
  const int i = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;

  for (int k = tid; k < DZ; k += C::NTHR) {
    sg[k] = p.lnw[k]; sg[DZ + k] = p.lnb[k];
#pragma unroll
    for (int qq = 0; qq < 4; ++qq)
      wbt[k * 4 + qq] = pack_bf16(__bfloat162float(p.wb[(2 * qq) * DZ + k]), __bfloat162float(p.wb[(2 * qq + 1) * DZ + k]));
  }
  // ---- phase 0: the softmax backward of head `warp` for this query row.  The row's first 32 PB keys are loaded into registers first (one latency for all of them), then
  // the warp's first tile is requested, then the arithmetic runs under both.
  const int ngroups = L / C::GJ;
  const uint32_t tb0 = sbase + warp * C::NBUF * C::TILE;
  {
    constexpr int PB = 12;
    const int h = warp;
    const __nv_bfloat16* wr = p.w + ((long)h * L + i) * L;
    auto dwsum = [&](int j) {                                           // the ns (<= 8) partial products of the head, summed in order; fully unrolled so that the loads overlap
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k)
        if (k < p.ns) a += p.dw[(((long)h * p.ns + k) * L + i) * L + j];
      return a;
    };
    auto load = [&](int j0, float (&wv)[PB], float (&dv)[PB], bool (&ok)[PB]) {
#pragma unroll
      for (int t = 0; t < PB; ++t) {
        const int j = j0 + t * 32 + lane;
        wv[t] = j < L ? __bfloat162float(wr[j]) : 0.f;
        dv[t] = j < L ? dwsum(j) : 0.f;
        ok[t] = p.mask == nullptr || (j < L && p.mask[j] != 0);
      }
    };
    float w0[PB], d0[PB];
    bool k0[PB];
    load(0, w0, d0, k0);
    if (warp < ngroups) pb_load_tile<DZ>(tb0, p.z + ((long)i * L + warp * C::GJ) * DZ, lane);
    cp_async_commit();
    float sd = 0.f;
#pragma unroll
    for (int t = 0; t < PB; ++t) sd = fmaf(w0[t], d0[t], sd);
    for (int j0 = 32 * PB; j0 < L; j0 += 32 * PB) {
      float wv[PB], dv[PB];
      bool ok[PB];
      load(j0, wv, dv, ok);
#pragma unroll
      for (int t = 0; t < PB; ++t) sd = fmaf(wv[t], dv[t], sd);
    }
    sd = warp_sum(sd);
    float ss = 0.f;
#pragma unroll
    for (int t = 0; t < PB; ++t) {
      const int j = t * 32 + lane;
      if (j < L) {
        const __nv_bfloat16 db = __float2bfloat16_rn(k0[t] ? w0[t] * (d0[t] - sd) : 0.f);
        sdb[j * 8 + h] = db;
        ss += __bfloat162float(db);
      }
    }
    for (int j0 = 32 * PB; j0 < L; j0 += 32 * PB) {
      float wv[PB], dv[PB];
      bool ok[PB];
      load(j0, wv, dv, ok);
#pragma unroll
      for (int t = 0; t < PB; ++t) {
        const int j = j0 + t * 32 + lane;
        if (j < L) {
          const __nv_bfloat16 db = __float2bfloat16_rn(ok[t] ? wv[t] * (dv[t] - sd) : 0.f);
          sdb[j * 8 + h] = db;
          ss += __bfloat162float(db);
        }
      }
    }
    ss = warp_sum(ss);
    if (lane == 0) p.pS[i * 8 + h] = ss;
  }
  __syncthreads();

  float cm[C::NK][4];                                                                 // M^T [d][h] for the warp's tiles: m16 tile mt = rows 16 mt..
#pragma unroll
  for (int mt = 0; mt < C::NK; ++mt) cm[mt][0] = cm[mt][1] = cm[mt][2] = cm[mt][3] = 0.f;

  int it = 0;
  for (int gi = warp; gi < ngroups; gi += C::NW, ++it) {
    const int buf = it % C::NBUF;
    const uint32_t tile = tb0 + buf * C::TILE;
    const int j0 = gi * C::GJ;
    cp_async_wait<0>();
    __syncwarp();
    if (C::NBUF == 2 && gi + C::NW < ngroups) pb_load_tile<DZ>(tb0 + (buf ^ 1) * C::TILE, p.z + ((long)i * L + (gi + C::NW) * C::GJ) * DZ, lane);
    if (C::NBUF == 2) cp_async_commit();

    // ---- statistics (two passes over the tile)
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < C::NK; ++ks) {
      uint32_t a[4];
      ldsm_x4(a, tile + swzn<C::NCHK>(lane & 15, 2 * ks + (lane >> 4)));
      s0 += bf16lo(a[0]) + bf16hi(a[0]) + bf16lo(a[2]) + bf16hi(a[2]);
      s1 += bf16lo(a[1]) + bf16hi(a[1]) + bf16lo(a[3]) + bf16hi(a[3]);
    }
    const float m0 = quad_sum(s0) * (1.f / DZ), m1 = quad_sum(s1) * (1.f / DZ);
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < C::NK; ++ks) {
      uint32_t a[4];
      ldsm_x4(a, tile + swzn<C::NCHK>(lane & 15, 2 * ks + (lane >> 4)));
      float d;
      d = bf16lo(a[0]) - m0; v0 = fmaf(d, d, v0); d = bf16hi(a[0]) - m0; v0 = fmaf(d, d, v0);
      d = bf16lo(a[2]) - m0; v0 = fmaf(d, d, v0); d = bf16hi(a[2]) - m0; v0 = fmaf(d, d, v0);
      d = bf16lo(a[1]) - m1; v1 = fmaf(d, d, v1); d = bf16hi(a[1]) - m1; v1 = fmaf(d, d, v1);
      d = bf16lo(a[3]) - m1; v1 = fmaf(d, d, v1); d = bf16hi(a[3]) - m1; v1 = fmaf(d, d, v1);
    }
    const float rs0 = rsqrtf(quad_sum(v0) * (1.f / DZ) + p.eps), rs1 = rsqrtf(quad_sum(v1) * (1.f / DZ) + p.eps);

    // db as the A fragment of the dzn product: rows = keys, k = heads 2 q, 2 q + 1 (a0: key g, a1: key g + 8; heads 8..15 are zero)
    uint32_t adb[4];
    adb[0] = *reinterpret_cast<const uint32_t*>(sdb + (j0 + g) * 8 + 2 * q);
    adb[1] = *reinterpret_cast<const uint32_t*>(sdb + (j0 + g + 8) * 8 + 2 * q);
    adb[2] = 0u; adb[3] = 0u;

    // ---- pass A: the row sums of g = dzn gamma and g xhat
    float gs0 = 0.f, gx0 = 0.f, gs1 = 0.f, gx1 = 0.f;
#pragma unroll
    for (int ch = 0; ch < C::NCH64; ++ch) {
      uint32_t a[4][4];
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) ldsm_x4(a[kk], tile + swzn<C::NCHK>(lane & 15, 2 * (4 * ch + kk) + (lane >> 4)));
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int d0 = ch * 64 + nt * 8;
        float dzn[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(dzn, adb, wbt[(d0 + g) * 4 + q], 0u);
        const float2 gm = *reinterpret_cast<const float2*>(sg + d0 + 2 * q);
        const uint32_t xa = a[nt >> 1][(nt & 1) * 2], xb = a[nt >> 1][(nt & 1) * 2 + 1];            // (row g | row g + 8), columns d0 + 2 q, +1
        const float ga0 = dzn[0] * gm.x, ga1 = dzn[1] * gm.y, gb0 = dzn[2] * gm.x, gb1 = dzn[3] * gm.y;
        gs0 += ga0 + ga1; gs1 += gb0 + gb1;
        gx0 = fmaf(ga0, (bf16lo(xa) - m0) * rs0, fmaf(ga1, (bf16hi(xa) - m0) * rs0, gx0));
        gx1 = fmaf(gb0, (bf16lo(xb) - m1) * rs1, fmaf(gb1, (bf16hi(xb) - m1) * rs1, gx1));
      }
    }
    gs0 = quad_sum(gs0) * (1.f / DZ); gx0 = quad_sum(gx0) * (1.f / DZ); gs1 = quad_sum(gs1) * (1.f / DZ); gx1 = quad_sum(gx1) * (1.f / DZ);

    // B fragments of db for M^T = xhat^T db: n = head g, k = keys (2 q, 2 q + 1) / (2 q + 8, 2 q + 9)
    const uint32_t bd0 = (uint32_t)__bfloat16_as_ushort(sdb[(j0 + 2 * q) * 8 + g]) | ((uint32_t)__bfloat16_as_ushort(sdb[(j0 + 2 * q + 1) * 8 + g]) << 16);
    const uint32_t bd1 = (uint32_t)__bfloat16_as_ushort(sdb[(j0 + 2 * q + 8) * 8 + g]) | ((uint32_t)__bfloat16_as_ushort(sdb[(j0 + 2 * q + 9) * 8 + g]) << 16);

    // ---- pass B: dz into the tile, M^T accumulation
#pragma unroll
    for (int ch = 0; ch < C::NCH64; ++ch) {
      uint32_t a[4][4];
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) ldsm_x4(a[kk], tile + swzn<C::NCHK>(lane & 15, 2 * (4 * ch + kk) + (lane >> 4)));
      uint32_t xw[8][2];                                                                           // xhat (bf16 pairs) of the n8 tiles: (row g, row g + 8)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int d0 = ch * 64 + nt * 8;
        float dzn[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(dzn, adb, wbt[(d0 + g) * 4 + q], 0u);
        const float2 gm = *reinterpret_cast<const float2*>(sg + d0 + 2 * q);
        const uint32_t xa = a[nt >> 1][(nt & 1) * 2], xb = a[nt >> 1][(nt & 1) * 2 + 1];
        const float xa0 = (bf16lo(xa) - m0) * rs0, xa1 = (bf16hi(xa) - m0) * rs0, xb0 = (bf16lo(xb) - m1) * rs1, xb1 = (bf16hi(xb) - m1) * rs1;
        const float da0 = rs0 * (dzn[0] * gm.x - gs0 - xa0 * gx0), da1 = rs0 * (dzn[1] * gm.y - gs0 - xa1 * gx0);
        const float db0 = rs1 * (dzn[2] * gm.x - gs1 - xb0 * gx1), db1 = rs1 * (dzn[3] * gm.y - gs1 - xb1 * gx1);
        sts32(tile + swzn<C::NCHK>(g, d0 / 8) + 4 * q, pack_bf16(da0, da1));
        sts32(tile + swzn<C::NCHK>(g + 8, d0 / 8) + 4 * q, pack_bf16(db0, db1));
        xw[nt][0] = pack_bf16(xa0, xa1);
        xw[nt][1] = pack_bf16(xb0, xb1);
      }
#pragma unroll
      for (int mp = 0; mp < 4; ++mp) {                                                             // m16 tiles (d columns 16 mp ..) of this chunk = n8 tiles (2 mp, 2 mp + 1)
        uint32_t at[4];
        at[0] = movT(xw[2 * mp][0]); at[1] = movT(xw[2 * mp + 1][0]); at[2] = movT(xw[2 * mp][1]); at[3] = movT(xw[2 * mp + 1][1]);
        mma16816(cm[ch * 4 + mp], at, bd0, bd1);
      }
    }
    __syncwarp();
    // ---- dz tile -> global (16 rows x DZ contiguous)
#pragma unroll
    for (int qq = 0; qq < C::GJ * C::NCHK / 32; ++qq) {
      const int v = qq * 32 + lane, r = v / C::NCHK, ch = v % C::NCHK;
      stg128(p.dz + ((long)i * L + j0 + r) * DZ + ch * 8, lds128(tile + swzn<C::NCHK>(r, ch)));
    }
    __syncwarp();
    if (C::NBUF == 1) {
      if (gi + C::NW < ngroups) pb_load_tile<DZ>(tile, p.z + ((long)i * L + (gi + C::NW) * C::GJ) * DZ, lane);
      cp_async_commit();
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  // ---- M^T partials: warp w writes [h][d] floats at sbase + w * 8 DZ * 4; the eight are summed in order
  {
    float* mp = reinterpret_cast<float*>(smem) + warp * 8 * DZ;
#pragma unroll
    for (int mt = 0; mt < C::NK; ++mt) {
      const int d = 16 * mt + g;
      mp[(2 * q) * DZ + d] = cm[mt][0]; mp[(2 * q + 1) * DZ + d] = cm[mt][1];
      mp[(2 * q) * DZ + d + 8] = cm[mt][2]; mp[(2 * q + 1) * DZ + d + 8] = cm[mt][3];
    }
  }
  __syncthreads();
  const float* red = reinterpret_cast<const float*>(smem);
  for (int e = tid; e < 8 * DZ; e += C::NTHR) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w * 8 * DZ + e];
    p.pM[(long)i * 8 * DZ + e] = s;
  }
}

}  // namespace pwa80
