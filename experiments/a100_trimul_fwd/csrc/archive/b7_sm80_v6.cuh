// b7_sm80.cuh -- TriMul backward, input side "source" kernel (B7src), A100 / sm_80.
//
// Per plane channel c (a | b planes) and token t, with x_n = LN_in(z) and the forward's gated projection a = sigmoid(g) p m:
//   dp = dA s m,  dg = dA p s (1 - s) m          (s = sigmoid(g), dA = gradient of the plane from the contraction backward)
//   dW_g += dg (x) x_n,  dW_p += dp (x) x_n
// Channel-stationary (the idea of the sm_90 B7 "source" CTAs): a CTA owns one 64-row weight block (32 plane channels: 8 gate rows | 8 proj rows
// per group, the K1 packing) for a split of the token tiles, so its dW block stays in registers for the whole launch.  It recomputes (g, p) of
// its block (MMA over x_n, written once by B1), forms (dg, dp), writes them token-major for the dx GEMM
// (columns 64 b .. 64 b + 63 of the dx operand = the K1 weight-row order) and accumulates dW^T = x_n^T . DGP.  The NSTEP block CTAs of a split
// walk the same tiles side by side, so a z tile comes from DRAM once and from L2 for the other blocks.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct B7Params {
  const __nv_bfloat16* xn;     // [T][128] x_n = LN_in(z)
  const __nv_bfloat16* w;      // [NSTEP][16 granules][64 rows][8]  (the K1 packing: 0.5 W, so the MMA gives g / 2, p / 2)
  const __nv_bfloat16* dab;    // [2 CH][T] plane gradients
  const uint8_t* mask;         // [L] token mask or nullptr
  __nv_bfloat16* dgp;          // [T][ldd]
  float* dw;                   // [S][NSTEP * 64][128] partial dW (row = K1 packed weight row)
  int T, L, num_tiles, nstep, splits, ldd;
};

struct B7Cfg {
  static constexpr int CZ = 128, BM = 128, NTHR = 128;
  static constexpr int SMEM_W = 64 * CZ * 2, SMEM_Z = BM * CZ * 2, SMEM_A = 32 * BM * 2, SMEM_D = BM * 64 * 2;
  static constexpr int MAXL = 4096;                                                     // token mask staged in smem up to this L
  static constexpr int SMEM = SMEM_W + SMEM_Z + SMEM_A + SMEM_D + BM * 4 + MAXL;         // + per-row mask factor + the [L] mask
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
  float* sM = reinterpret_cast<float*>(sD + G::SMEM_D);   // [BM] mask factor
  uint8_t* sMask = reinterpret_cast<uint8_t*>(sM + BM);    // [L] token mask (L <= MAXL)
  const bool smask = p.mask != nullptr && p.L <= G::MAXL;
  if (smask) for (int i = tid; i < p.L; i += 128) sMask[i] = p.mask[i];
  const uint32_t sW_u = smem_u32(sW), sZ_u = smem_u32(sZ), sA_u = smem_u32(sA), sD_u = smem_u32(sD);
  const int b = blockIdx.x % p.nstep, s = blockIdx.x / p.nstep;
  const int n_iter = s < p.num_tiles ? (p.num_tiles - s + p.splits - 1) / p.splits : 0;

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

  // ---- tile loads: x_n [128 tok][128] (256 B rows, granule ^= row & 7), dAB [32 ch][128 tok] (256 B rows), per-row mask.
  // Software-pipelined: the next tile's dAB + mask are issued once this tile's elementwise pass is done (under the dgp store and the dW MMA),
  // its x_n once the dW MMA has released the x_n tile.
  // thread: row r0 + 8 i (x_n, i < 16) / channel row r0 + 8 i (dAB, i < 4), granule gq; lane-constant swizzle (row & 7 = r0 & 7)
  const int r0 = tid >> 4, gq = tid & 15;
  const uint32_t zdst = sZ_u + r0 * 256 + ((gq ^ (r0 & 7)) << 4), adst = sA_u + r0 * 256 + ((gq ^ (r0 & 7)) << 4);
  auto load_z = [&](int t0) {
    const __nv_bfloat16* zsrc = p.xn + (size_t)(t0 + r0) * CZ + gq * 8;
    if (t0 + BM <= p.T) {
#pragma unroll
      for (int i = 0; i < 16; ++i) cp_async16(zdst + i * 2048, zsrc + i * 8 * CZ);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) { const bool ok = t0 + r0 + 8 * i < p.T; cp_async16(zdst + i * 2048, ok ? zsrc + i * 8 * CZ : p.xn, ok ? 16u : 0u); }
    }
  };
  auto load_a = [&](int t0) {
    const __nv_bfloat16* asrc = p.dab + (size_t)(oc0 + r0) * p.T + t0 + gq * 8;
    const bool ok = t0 + gq * 8 < p.T;
#pragma unroll
    for (int i = 0; i < 4; ++i) cp_async16(adst + i * 2048, ok ? asrc + (size_t)i * 8 * p.T : p.dab, ok ? 16u : 0u);
  };
  auto set_mask = [&](int t0) {
    const int t = t0 + tid;
    float m = 0.f;
    if (t < p.T) {
      m = 1.f;
      if (p.mask != nullptr) {
        const int i = t / p.L, j = t - i * p.L;
        m = (smask ? (sMask[i] && sMask[j]) : (p.mask[i] && p.mask[j])) ? 1.f : 0.f;
      }
    }
    sM[tid] = m;
  };
  __syncthreads();                               // sMask
  if (n_iter > 0) { load_z(s * BM); load_a(s * BM); cp_async_commit(); set_mask(s * BM); }

  for (int it = 0; it < n_iter; ++it) {
    const int tile = s + it * p.splits, t0 = tile * BM;
    const int tn = t0 + p.splits * BM;           // the next tile of this CTA
    const bool has_next = it + 1 < n_iter;
    cp_async_wait<0>();
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
            // g' = g / 2, p' = p / 2:  s = 1/2 + th/2 (th = tanh g'),  dp = dA m s,  dg = dA m p s (1 - s) = dA m p' (1 - th^2) / 2
            const float hm = 0.5f * sM[row];
            const float da0 = bf16lo(dv) * hm, da1 = bf16hi(dv) * hm;          // dA m / 2
            const float th0 = tanh_approx(acc[mt][2 * np][2 * h]), th1 = tanh_approx(acc[mt][2 * np][2 * h + 1]);
            const float p0 = acc[mt][2 * np + 1][2 * h], p1 = acc[mt][2 * np + 1][2 * h + 1];
            // staging [128 tok][64 rows] (128 B rows, granule ^= tok & 7): gate row 16 np + 2q (+1) <- dg, proj row 16 np + 8 + 2q <- dp
            const uint32_t base = sD_u + row * 128;
            const int gr = 2 * np, pr = 2 * np + 1;          // 16 B granules of the gate / proj rows
            sts32(base + (((gr ^ (row & 7))) << 4) + q * 4, pack_bf16(da0 * p0 * fmaf(-th0, th0, 1.f), da1 * p1 * fmaf(-th1, th1, 1.f)));
            sts32(base + (((pr ^ (row & 7))) << 4) + q * 4, pack_bf16(fmaf(da0, th0, da0), fmaf(da1, th1, da1)));
          }
        }
      }
    __syncthreads();                             // dAB and mask consumed: the next tile's arrive under the store and the dW MMA
    if (has_next) { load_a(tn); cp_async_commit(); set_mask(tn); }
    // ---- DGP tile -> dx operand columns 64 b .. (token rows of 128 B)
    {
      const int r0 = tid >> 3, g = tid & 7;                                   // rows r0 + 16 i; (r0 + 16 i) & 7 = r0 & 7
      const uint32_t src = sD_u + r0 * 128 + ((g ^ (r0 & 7)) << 4);
      __nv_bfloat16* dst = p.dgp + (size_t)(t0 + r0) * p.ldd + 64 * b + 8 * g;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (t0 + r0 + 16 * i < p.T) stg128(dst + (size_t)i * 16 * p.ldd, lds128(src + i * 2048));
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
    __syncthreads();                             // x_n and dgp tiles free
    if (has_next) { load_z(tn); cp_async_commit(); }
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

// ---- B7src v5 ("pair"): one CTA of 8 warps per SM = two 4-warp subgroups, each owning one 64-row weight block (blocks 2 pb, 2 pb + 1), sharing
// a double-buffered x_n tile: the next tile's x_n lands during the whole current tile (the v4 kernel waited for it at every tile top), and
// each x_n tile feeds two blocks.  Per subgroup as v4: dAB tile, mask factors, dgp staging, dW^T in registers.
struct B7PCfg {
  static constexpr int CZ = 128, BM = 128, NTHR = 256;
  static constexpr int SMEM_W = 64 * CZ * 2, SMEM_Z = BM * CZ * 2, SMEM_A = 32 * BM * 2, SMEM_D = BM * 64 * 2, SMEM_M = BM * 4;
  static constexpr int MAXL = 1024;
  static constexpr int OFF_W = 0, OFF_Z = 2 * SMEM_W, OFF_A = OFF_Z + 2 * SMEM_Z, OFF_D = OFF_A + 2 * SMEM_A, OFF_M = OFF_D + 2 * SMEM_D;
  static constexpr int OFF_MASK = OFF_M + 2 * SMEM_M;
  static constexpr int SMEM = OFF_MASK + MAXL;
  static_assert(SMEM <= 166912, "sm_80 dynamic shared memory");
};

__global__ void __launch_bounds__(256, 1) b7pair_kernel(const B7Params p) {
  using G = B7PCfg;
  constexpr int CZ = G::CZ, BM = G::BM;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, sub = tid >> 7, ltid = tid & 127, warp = ltid >> 5, lane = tid & 31;
  const int g8 = lane >> 2, q = lane & 3, mi = lane >> 3, r8 = lane & 7;
  const uint32_t s0 = smem_u32(smem);
  const uint32_t sW_u = s0 + G::OFF_W + sub * G::SMEM_W, sZ0 = s0 + G::OFF_Z;
  const uint32_t sA_u = s0 + G::OFF_A + sub * G::SMEM_A, sD_u = s0 + G::OFF_D + sub * G::SMEM_D;
  float* sM = reinterpret_cast<float*>(smem + G::OFF_M) + sub * BM;
  uint8_t* sMask = smem + G::OFF_MASK;
  const bool smask = p.mask != nullptr && p.L <= G::MAXL;
  if (smask) for (int i = tid; i < p.L; i += 256) sMask[i] = p.mask[i];
  const int npair = p.nstep / 2;
  const int b = 2 * (blockIdx.x % npair) + sub, s = blockIdx.x / npair;
  const int n_iter = s < p.num_tiles ? (p.num_tiles - s + p.splits - 1) / p.splits : 0;
  const int bar_sub = 1 + sub;
#pragma unroll
  for (int i = 0; i < 8; ++i) { const int c = ltid + 128 * i; cp_async16(sW_u + c * 16, p.w + (size_t)b * 64 * CZ + c * 8); }
  cp_async_commit();
  float dwa[2][8][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dwa[mt][n][e] = 0.f;
  const int oc0 = 32 * b;
  // x_n tile: the CTA's 256 threads, rows zr0 + 16 i (i < 8), granule zq (row & 7 = zr0 & 7); dAB: the subgroup's 128, as v4
  const int zr0 = tid >> 4, zq = tid & 15;
  const uint32_t zoff = zr0 * 256 + ((zq ^ (zr0 & 7)) << 4);
  auto load_z = [&](int t0, int buf) {
    const uint32_t zdst = sZ0 + buf * G::SMEM_Z + zoff;
    const __nv_bfloat16* zsrc = p.xn + (size_t)(t0 + zr0) * CZ + zq * 8;
    if (t0 + BM <= p.T) {
#pragma unroll
      for (int i = 0; i < 8; ++i) cp_async16(zdst + i * 4096, zsrc + i * 16 * CZ);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) { const bool ok = t0 + zr0 + 16 * i < p.T; cp_async16(zdst + i * 4096, ok ? zsrc + i * 16 * CZ : p.xn, ok ? 16u : 0u); }
    }
  };
  const int ar0 = ltid >> 4, aq = ltid & 15;
  const uint32_t adst = sA_u + ar0 * 256 + ((aq ^ (ar0 & 7)) << 4);
  auto load_a = [&](int t0) {
    const __nv_bfloat16* asrc = p.dab + (size_t)(oc0 + ar0) * p.T + t0 + aq * 8;
    const bool ok = t0 + aq * 8 < p.T;
#pragma unroll
    for (int i = 0; i < 4; ++i) cp_async16(adst + i * 2048, ok ? asrc + (size_t)i * 8 * p.T : p.dab, ok ? 16u : 0u);
  };
  auto set_mask = [&](int t0) {
    const int t = t0 + ltid;
    float m = 0.f;
    if (t < p.T) {
      m = 1.f;
      if (p.mask != nullptr) {
        int i = t0 / p.L, j = t0 - i * p.L + ltid;             // (t0 / L is warp-uniform; the row index advances at most BM / L times)
        while (j >= p.L) { j -= p.L; ++i; }
        m = (smask ? (sMask[i] && sMask[j]) : (p.mask[i] && p.mask[j])) ? 1.f : 0.f;
      }
    }
    sM[ltid] = m;
  };
  __syncthreads();                               // sMask
  if (n_iter > 0) { load_z(s * BM, 0); load_a(s * BM); cp_async_commit(); set_mask(s * BM); }

  // lane-constant staging terms: warp rows 32 warp + g8 (+ 16 mt + 8 h), q word; the swizzle g8 << 4 (row & 7 = g8)
  const uint32_t gsw = (uint32_t)g8 << 4;
  const uint32_t sD_base = sD_u + (32 * warp + g8) * 128 + q * 4;
  // dgp store: rows sr0 + 16 i, granule sg (row & 7 = sr0 & 7)
  const int sr0 = ltid >> 3, sg = ltid & 7;
  const uint32_t ssrc = sD_u + sr0 * 128 + ((sg ^ (sr0 & 7)) << 4);
  const size_t sstep = (size_t)16 * p.ldd;
  for (int it = 0; it < n_iter; ++it) {
    const int t0 = (s + it * p.splits) * BM, tn = t0 + p.splits * BM, cur = it & 1;
    const bool has_next = it + 1 < n_iter;
    const uint32_t sZ_u = sZ0 + cur * G::SMEM_Z;
    cp_async_wait<0>();
    __syncthreads();                             // x_n[cur], dAB, mask landed; every warp is past iteration it - 1 (its x_n buffer is free)
    if (has_next) { load_z(tn, cur ^ 1); cp_async_commit(); }
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
      for (int np = 0; np < 4; ++np) {
        uint32_t bb[4];
        ldsm_x4(bb, sW_u + ((lane >> 3) & 1) * 1024 + (16 * np + ((lane >> 4) & 1) * 8 + r8) * 16 + ks * 2048);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(acc[mt][2 * np], a[mt], bb[0], bb[1]); mma16816(acc[mt][2 * np + 1], a[mt], bb[2], bb[3]); }
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kq = 0; kq < 2; ++kq) {
        uint32_t f[4];
        const int krow = 16 * kq + (mi >> 1) * 8 + r8, tg = (32 * warp + 16 * mt) / 8 + (mi & 1);
        ldsm_x4_t(f, sA_u + swz<256>(krow, tg * 16));
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int np = 2 * kq + hh;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = 32 * warp + 16 * mt + g8 + 8 * h;
            const uint32_t dv = f[2 * hh + h];
            const float hm = 0.5f * sM[row];
            const float da0 = bf16lo(dv) * hm, da1 = bf16hi(dv) * hm;
            const float th0 = tanh_approx(acc[mt][2 * np][2 * h]), th1 = tanh_approx(acc[mt][2 * np][2 * h + 1]);
            const float p0 = acc[mt][2 * np + 1][2 * h], p1 = acc[mt][2 * np + 1][2 * h + 1];
            // staging row = token (row & 7 = g8): gate granule 2 np ^ g8, proj granule (2 np + 1) ^ g8 = gate ^ 1
            const uint32_t ga = sD_base + (16 * mt + 8 * h) * 128 + ((2 * np) << 4 ^ gsw);
            sts32(ga, pack_bf16(da0 * p0 * fmaf(-th0, th0, 1.f), da1 * p1 * fmaf(-th1, th1, 1.f)));
            sts32(ga ^ 16u, pack_bf16(fmaf(da0, th0, da0), fmaf(da1, th1, da1)));
          }
        }
      }
    bar_sync(bar_sub, 128);                      // subgroup: dAB / mask consumed, dgp tile staged
    if (has_next) { load_a(tn); cp_async_commit(); set_mask(tn); }
    {
      __nv_bfloat16* dst = p.dgp + (size_t)(t0 + sr0) * p.ldd + 64 * b + 8 * sg;
      if (t0 + BM <= p.T) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { stg128(dst, lds128(ssrc + i * 2048)); dst += sstep; }
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) { if (t0 + sr0 + 16 * i < p.T) stg128(dst, lds128(ssrc + i * 2048)); dst += sstep; }
      }
    }
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t a[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int trow = 16 * ks + (mi >> 1) * 8 + r8, cg = (32 * warp + 16 * mt) / 8 + (mi & 1);
        ldsm_x4_t(a[mt], sZ_u + swz<256>(trow, cg * 16));
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bb[4];
        const int trow = 16 * ks + (mi & 1) * 8 + r8, rg = 2 * np + (mi >> 1);
        ldsm_x4_t(bb, sD_u + trow * 128 + ((rg ^ (trow & 7)) << 4));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(dwa[mt][2 * np], a[mt], bb[0], bb[1]); mma16816(dwa[mt][2 * np + 1], a[mt], bb[2], bb[3]); }
      }
    }
    bar_sync(bar_sub, 128);                      // subgroup: dgp tile free for the next elementwise pass
  }
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
