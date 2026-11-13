// wide_k1.cuh -- the front of the A100 "wide" TriMul (any width D): gated projections of the raw input rows -> channel-major planes.
//
//   plane[pc, t] = bf16( sigmoid(g) p ) * m_i m_j        g, p = LN_in(z[t]) . W_g[pc], LN_in(z[t]) . W_p[pc]        t = i L + j
//
// The input LayerNorm is folded into the weights (wide_rows.cuh): with W' = bf16(0.5 gamma W), s = sum_k W'[k], b = 0.5 W beta and the token's
// (mean, rstd) = (mu, r) from ln_stats_rows,   0.5 (g or p) = r (z . W'^T) - r mu s + b,   so the A operand is the RAW input tile (a plain cp.async
// stream, no LayerNorm pass) and the normalisation is two FMAs per accumulator in the epilogue.  sigmoid(g) p = p' (1 + tanh g') (one MUFU, one FMA).
// A masked pair's value is exactly 0 (the product is rounded to bf16 first, then multiplied by the 0 / 1 pair mask, as the Triton front).
//
// CTA tile = 128 tokens x 128 packed weight rows (= 64 plane channels: rows 16 j + 0..7 gate | 16 j + 8..15 projection of channels 8 j .. 8 j + 7, so one
// thread holds both factors of an output); 4 warps with 64 x 64 warp tiles (WTile).  The tokens of every m16 tile are permuted (RP) so a thread's
// (acc[.][.][0], acc[.][.][2]) / ([1], [3]) are the packed bf16 words (token 2 g8, 2 g8 + 1) of channels 2 q / 2 q + 1: they are staged channel-major in shared
// memory (272-B rows: conflict-free) and leave as 16-B stores of 256 contiguous bytes per channel row.  The grid walks the n-tiles of one token tile
// consecutively, so the token tile's input rows are read from DRAM once and re-read from L2 by its 6-12 n-tile CTAs.
#pragma once
#include "wide_gemm.cuh"

namespace a100 {

struct K1wParams {
  const __nv_bfloat16* z;      // [T][D]
  const __nv_bfloat16* w1;     // [4 Hs][D] packed rows
  const float* vs;             // [4 Hs] row sums of w1 (of the bf16 values), packed order
  const float* vb;             // [4 Hs] 0.5 W beta
  const float2* st;            // [T] (mean, rstd) of the input rows (STATS = false)
  float2* stw;                 // STATS: written here by the ntile == 0 CTAs (mean, rstd of the input rows, two-pass, fp32)
  float eps;
  const uint8_t* mask;         // [L] token mask or nullptr
  __nv_bfloat16* planes;       // [4 Hs / 2][T]
  int T, L, D, ncol;           // ncol = 4 Hs packed rows
};

using K1wTile = WTile<128, 128, 2, 2, 4, false, true>;
constexpr int K1W_STG = 272;   // bytes per staged channel row: 256 + 16
constexpr int K1W_STATS_SMEM = K1wTile::SMEM + 128 * 8;     // + (mean, rstd) of the tile's 128 tokens

// STATS: the input row statistics come from the tile itself instead of ln_stats_rows (D <= 128: every k-tile of the A operand is still in its ring slot when the mainloop
// ends, and thread i owns token row i of the tile: two passes over its 4 granules per k-tile, exact in fp32 like ln_stats_rows), saving that kernel's read of z and its launch
// INLOOP (D > 128, with STATS): the ring no longer holds the whole K, so each thread accumulates the statistics of its own cp.async granules (granule cb of tile rows
// rb + 32 e, e < 4) as the stages go by -- a pivot-shifted sum and sum of squares of its 8 values per stage -- and the four threads of a row merge them at the end
// (exact algebra: sum (x - m)^2 = sum_t [S2_t - 2 (m - p_t) S1_t + n (m - p_t)^2]).  The thread reads only granules it loaded itself, after its own cp.async wait, so
// the hook needs no barrier of its own.
struct K1StatHook {
  uint32_t off;                                    // wkmaj(rb, cb): this thread's granule of row rb inside a stage (+ 2048 per 32 rows)
  mutable float s1[4], s2[4], piv[4];
  mutable bool first;
  DEVI K1StatHook() : off(wkmaj(threadIdx.x >> 2, threadIdx.x & 3)), first(true) {
#pragma unroll
    for (int e = 0; e < 4; ++e) { s1[e] = 0.f; s2[e] = 0.f; piv[e] = 0.f; }
  }
  DEVI void load(uint32_t, int) const {}
  DEVI void transform(uint32_t stage) const {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint4 v = lds128(stage + off + e * 2048);
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
      if (first) piv[e] = bf16lo(w[0]);
      float a = 0.f, b = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float d0 = bf16lo(w[i]) - piv[e], d1 = bf16hi(w[i]) - piv[e];
        a += d0 + d1;
        b = fmaf(d0, d0, fmaf(d1, d1, b));
      }
      s1[e] += a; s2[e] += b;
    }
    first = false;
  }
};

template <bool STATS, bool ROW_MASK = false, bool INLOOP = false>
__global__ void __launch_bounds__(128, 2) k1w_kernel(const K1wParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = smem_u32(smem);
  const int ntn = p.ncol >> 7;
  const int mtile = (int)blockIdx.x / ntn, ntile = (int)blockIdx.x - mtile * ntn;
  const int t0 = mtile * 128, j0 = ntile * 128;
  // With L divisible by 128, a CTA never crosses a pair row. Resolve its row
  // mask once, and replace the per-output token divisions with local offsets.
  // This specialization does not own LayerNorm statistics (STATS must be off).
  static_assert(!ROW_MASK || !STATS, "row skipping must not skip statistics writes");
  const int first_col = ROW_MASK ? t0 % p.L : 0;
  if constexpr (ROW_MASK) {
    if (p.mask != nullptr && __ldg(p.mask + t0 / p.L) == 0) {
      __nv_bfloat16* out = p.planes + (size_t)(ntile * 64) * p.T + t0;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int idx = threadIdx.x + 128 * e;
        stg128(out + (size_t)(idx >> 4) * p.T + 8 * (idx & 15), make_uint4(0u, 0u, 0u, 0u));
      }
      return;
    }
  }
  K1wTile tl;
  const int tid0 = threadIdx.x;
  if constexpr (INLOOP) {
    static_assert(STATS && !ROW_MASK, "in-loop statistics feed the STATS epilogue and skip no rows");
    K1StatHook hook;
    tl.run_fast(s0, p.D >> 5, GemmOps{p.z, (size_t)p.D, t0, p.T, p.w1, (size_t)p.D, j0, p.ncol}, hook);
    const float n = p.D * 0.25f, inv = 1.f / p.D;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float tot = hook.s1[e] + n * hook.piv[e];
      tot += __shfl_xor_sync(0xffffffffu, tot, 1);
      tot += __shfl_xor_sync(0xffffffffu, tot, 2);
      const float mean = tot * inv, dm = mean - hook.piv[e];
      float ss = fmaf(n * dm, dm, fmaf(-2.f * dm, hook.s1[e], hook.s2[e]));
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float rstd = rsqrtf(fmaxf(ss * inv, 0.f) + p.eps);
      const int row = (tid0 >> 2) + 32 * e;
      if ((tid0 & 3) == 0) {
        sts64(s0 + K1wTile::SMEM + row * 8, make_uint2(__float_as_uint(mean), __float_as_uint(rstd)));
        if (ntile == 0) p.stw[t0 + row] = make_float2(mean, rstd);
      }
    }
    __syncthreads();
  } else {
    tl.run_fast(s0, p.D >> 5, GemmOps{p.z, (size_t)p.D, t0, p.T, p.w1, (size_t)p.D, j0, p.ncol});
  }
  if constexpr (STATS && !INLOOP) {
    const int nk = p.D >> 5;
    float sum = 0.f;
#pragma unroll 1
    for (int kt = 0; kt < nk; ++kt)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 v = lds128(s0 + kt * K1wTile::STAGE + wkmaj(tid0, c));
        sum += (bf16lo(v.x) + bf16hi(v.x)) + (bf16lo(v.y) + bf16hi(v.y)) + ((bf16lo(v.z) + bf16hi(v.z)) + (bf16lo(v.w) + bf16hi(v.w)));
      }
    const float mean = sum * (1.f / p.D);
    float q2 = 0.f;
#pragma unroll 1
    for (int kt = 0; kt < nk; ++kt)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 v = lds128(s0 + kt * K1wTile::STAGE + wkmaj(tid0, c));
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float a = bf16lo(w[i]) - mean, b = bf16hi(w[i]) - mean; q2 = fmaf(a, a, fmaf(b, b, q2)); }
      }
    const float rstd = rsqrtf(q2 * (1.f / p.D) + p.eps);
    sts64(s0 + K1wTile::SMEM + tid0 * 8, make_uint2(__float_as_uint(mean), __float_as_uint(rstd)));
    if (ntile == 0) p.stw[t0 + tid0] = make_float2(mean, rstd);
    __syncthreads();
  }
  // ---- epilogue
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1, g8 = lane >> 2, q = lane & 3;
  constexpr int MT = K1wTile::MT, NP = K1wTile::NP;
  float rs[MT][2], mr[MT][2], mk[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int tok = t0 + 64 * wm + 16 * mt + 2 * g8 + rh;
      float2 s;
      if constexpr (STATS) { const uint2 u = lds64(s0 + K1wTile::SMEM + (64 * wm + 16 * mt + 2 * g8 + rh) * 8); s = make_float2(__uint_as_float(u.x), __uint_as_float(u.y)); }
      else s = __ldg(p.st + tok);
      rs[mt][rh] = s.y; mr[mt][rh] = s.x * s.y;
      float m = 1.f;
      if (p.mask != nullptr) {
        if constexpr (ROW_MASK) {
          const int j = first_col + 64 * wm + 16 * mt + 2 * g8 + rh;
          m = __ldg(p.mask + j) != 0 ? 1.f : 0.f;
        } else {
          const int i = tok / p.L, j = tok - i * p.L;
          m = (__ldg(p.mask + i) != 0 && __ldg(p.mask + j) != 0) ? 1.f : 0.f;
        }
      }
      mk[mt][rh] = m;
    }
#pragma unroll
  for (int np = 0; np < NP; ++np) {
    const int R = j0 + 64 * wn + 16 * np + 2 * q;
    const float2 sg = __ldg(reinterpret_cast<const float2*>(p.vs + R)), sp = __ldg(reinterpret_cast<const float2*>(p.vs + R + 8));
    const float2 bg = __ldg(reinterpret_cast<const float2*>(p.vb + R)), bp = __ldg(reinterpret_cast<const float2*>(p.vb + R + 8));
    const int ch = 32 * wn + 8 * np + 2 * q;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      float u[2][2];
#pragma unroll
      for (int rh = 0; rh < 2; ++rh)
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          const float ag = tl.acc[mt][2 * np][2 * rh + cc], ap = tl.acc[mt][2 * np + 1][2 * rh + cc];
          const float g = fmaf(rs[mt][rh], ag, fmaf(-mr[mt][rh], cc ? sg.y : sg.x, cc ? bg.y : bg.x));
          const float pp = fmaf(rs[mt][rh], ap, fmaf(-mr[mt][rh], cc ? sp.y : sp.x, cc ? bp.y : bp.x));
          u[rh][cc] = round_bf16f(fmaf(pp, tanh_approx(g), pp)) * mk[mt][rh];
        }
      const int tw = 32 * wm + 8 * mt + g8;                            // 4-byte word of the 128-token row: tokens (64 wm + 16 mt + 2 g8, + 1)
      sts32(s0 + ch * K1W_STG + tw * 4, pack_bf16(u[0][0], u[1][0]));
      sts32(s0 + (ch + 1) * K1W_STG + tw * 4, pack_bf16(u[0][1], u[1][1]));
    }
  }
  __syncthreads();
  __nv_bfloat16* out = p.planes + (size_t)(ntile * 64) * p.T + t0;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int idx = tid + 128 * e, r = idx >> 4, gk = idx & 15;
    stg128(out + (size_t)r * p.T + 8 * gk, lds128(s0 + r * K1W_STG + gk * 16));
  }
}

}  // namespace a100

namespace a100 {

// ---------------------------------------------------------------------------------------------------------------------------------- k1w64
// k1w for D = 64 (bf16, in-kernel statistics): the same planes, bit for bit (the same MMAs in the same k order, the same two-pass statistics of each
// token row by one thread, the same epilogue arithmetic), with the work of a token tile in ONE persistent CTA instead of 4 Hs / 128 short CTAs.
//
// At D = 64 the k1w GEMM is two k-tiles: a k1w CTA is a prologue (both operand tiles from L2 / DRAM), 64 HMMA per warp, the epilogue and its stores, and
// it re-reads its token tile and recomputes its statistics once per n-tile -- latency-bound at 2 CTAs / SM (12 % occupancy, ~0.6 TB/s).  Here every CTA
// keeps all 4 Hs packed weight rows resident in shared memory (loaded once), walks the token tiles blockIdx.x, + gridDim.x, ... with the next tile's
// input rows streaming into the other half of a double buffer under the current tile's work, holds the tile's A fragments (all of K) in registers
// across the n-tiles, and stores the plane words straight from the accumulators (4 channels x 32 contiguous bytes per warp store: whole sectors).
//
// 8 warps: 4 (32 tokens each) x 2 (64 packed rows = 32 channels of each 128-row n-tile).
template <int HS>
struct K1w64 {
  static constexpr int D = 64, NCOL = 4 * HS, NTN = NCOL / 128, NTHR = 256;
  // G column groups (hidden 128: two): a CTA keeps the NCOLG = 256 packed rows of its group resident and does that group's NTNG = 2 n-tiles of every token tile it
  // walks, so two CTAs fit an SM at either hidden width (16 warps; one 512-row CTA per SM ran at 12 % occupancy, latency-bound).  The groups of a token tile start
  // together and walk the same tiles: the second read of an input tile is an L2 hit.  The same MMAs in the same k order per plane word: bit-identical planes.
  static constexpr int G = HS / 64, NCOLG = NCOL / G, NTNG = NTN / G;
  static constexpr int W_KT = NCOLG * 64;                // bytes of one 32-wide k-tile of the packed weights (wkmaj rows)
  static constexpr int Z_KT = 128 * 64;                  // bytes of one k-tile of a 128-token input tile
  static constexpr int ZOFF = 2 * W_KT, ZBUF = 2 * Z_KT;  // two input buffers after the weights
  static constexpr int STOFF = ZOFF + 2 * ZBUF;           // (mean, rstd) of the current tile's 128 tokens
  static constexpr int SMEM = STOFF + 128 * 8;
};

template <int HS>
__global__ void __launch_bounds__(256, 2) k1w64_kernel(const K1wParams p) {
  using C = K1w64<HS>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = smem_u32(smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1, g8 = lane >> 2, q = lane & 3;
  const int ntiles = p.T >> 7;
  const int grp = (int)blockIdx.x % C::G, cta = (int)blockIdx.x / C::G, step = (int)gridDim.x / C::G;   // column group; the CTA's first token tile and its stride
  // input tile loader: 1024 granules (128 rows x 2 k-tiles x 4), 4 per thread
  auto issue_z = [&](int tile, int buf) {
    const __nv_bfloat16* src = p.z + (size_t)tile * 128 * C::D;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int g = tid + C::NTHR * e, row = g >> 3, kt = (g >> 2) & 1, c = g & 3;
      cp_async16(s0 + C::ZOFF + buf * C::ZBUF + kt * C::Z_KT + wkmaj(row, c), src + row * C::D + kt * 32 + c * 8);
    }
  };
  // ---- the packed weights (once) + the first tile
#pragma unroll
  for (int e = 0; e < C::NCOLG * 8 / C::NTHR; ++e) {
    const int g = tid + C::NTHR * e, row = g >> 3, kt = (g >> 2) & 1, c = g & 3;
    cp_async16(s0 + kt * C::W_KT + wkmaj(row, c), p.w1 + (size_t)(grp * C::NCOLG + row) * C::D + kt * 32 + c * 8);
  }
  int tile = cta;
  if (tile < ntiles) issue_z(tile, 0);
  cp_async_commit();
  // lane-constant fragment offsets: A (RP token permutation, as WTile) and B, per k16 step kk (k-tile kk >> 1)
  const int mi = lane >> 3, r8 = lane & 7;
  uint32_t aoff[4], boff[4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    aoff[kk] = (kk >> 1) * C::Z_KT + wkmaj(32 * wm + 2 * r8 + (mi & 1), 2 * (kk & 1) + (mi >> 1));        // + 1024 per m16 tile
    boff[kk] = (kk >> 1) * C::W_KT + wkmaj(64 * wn + r8 + (mi >> 1) * 8, 2 * (kk & 1) + (mi & 1));        // + 1024 per n16, + 8192 per n-tile
  }
#pragma unroll 1
  for (int buf = 0; tile < ntiles; tile += step, buf ^= 1) {
    const int t0 = tile * 128;
    // the next tile into the other buffer (its readers finished before the previous iteration's second barrier)
    if (tile + step < ntiles) issue_z(tile + step, buf ^ 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const uint32_t zb = s0 + C::ZOFF + buf * C::ZBUF;
    {   // the input rows' statistics, two passes: two threads per token row (one k-tile of 32 channels each, the halves merged by a shuffle), so all 8 warps share
        // the work (one thread per row left half the CTA waiting at the next barrier)
      const int row = tid >> 1, kt = tid & 1;
      float sum = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 v = lds128(zb + kt * C::Z_KT + wkmaj(row, c));
        sum += (bf16lo(v.x) + bf16hi(v.x)) + (bf16lo(v.y) + bf16hi(v.y)) + ((bf16lo(v.z) + bf16hi(v.z)) + (bf16lo(v.w) + bf16hi(v.w)));
      }
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      const float mean = sum * (1.f / C::D);
      float q2 = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 v = lds128(zb + kt * C::Z_KT + wkmaj(row, c));
          const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) { const float a = bf16lo(w[i]) - mean, b = bf16hi(w[i]) - mean; q2 = fmaf(a, a, fmaf(b, b, q2)); }
      }
      q2 += __shfl_xor_sync(0xffffffffu, q2, 1);
      if (kt == 0) {
        const float rstd = rsqrtf(q2 * (1.f / C::D) + p.eps);
        sts64(s0 + C::STOFF + row * 8, make_uint2(__float_as_uint(mean), __float_as_uint(rstd)));
        if (grp == 0) p.stw[t0 + row] = make_float2(mean, rstd);
      }
    }
    uint32_t af[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) ldsm_x4(af[mt][kk], zb + aoff[kk] + mt * 1024);
    __syncthreads();
    float rs[2][2], mr[2][2], mk[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const int lt = 32 * wm + 16 * mt + 2 * g8 + rh, tok = t0 + lt;
        const uint2 u = lds64(s0 + C::STOFF + lt * 8);
        const float mean = __uint_as_float(u.x), rstd = __uint_as_float(u.y);
        rs[mt][rh] = rstd; mr[mt][rh] = mean * rstd;
        float m = 1.f;
        if (p.mask != nullptr) {
          const int i = tok / p.L, j = tok - i * p.L;
          m = (__ldg(p.mask + i) != 0 && __ldg(p.mask + j) != 0) ? 1.f : 0.f;
        }
        mk[mt][rh] = m;
      }
#pragma unroll 1
    for (int nt = 0; nt < C::NTNG; ++nt) {
      const int gnt = grp * C::NTNG + nt;                 // the n-tile among all NTN
      float acc[2][8][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[mt][n][e] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t bf[4];
          ldsm_x4(bf, s0 + boff[kk] + nt * 8192 + np * 1024);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(acc[mt][2 * np], af[mt][kk], bf[0], bf[1]);
            mma16816(acc[mt][2 * np + 1], af[mt][kk], bf[2], bf[3]);
          }
        }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        const int R = gnt * 128 + 64 * wn + 16 * np + 2 * q;
        const float2 sg = __ldg(reinterpret_cast<const float2*>(p.vs + R)), sp = __ldg(reinterpret_cast<const float2*>(p.vs + R + 8));
        const float2 bg = __ldg(reinterpret_cast<const float2*>(p.vb + R)), bp = __ldg(reinterpret_cast<const float2*>(p.vb + R + 8));
        const int ch = gnt * 64 + 32 * wn + 8 * np + 2 * q;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          float u[2][2];
#pragma unroll
          for (int rh = 0; rh < 2; ++rh)
#pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              const float ag = acc[mt][2 * np][2 * rh + cc], ap = acc[mt][2 * np + 1][2 * rh + cc];
              const float g = fmaf(rs[mt][rh], ag, fmaf(-mr[mt][rh], cc ? sg.y : sg.x, cc ? bg.y : bg.x));
              const float pp = fmaf(rs[mt][rh], ap, fmaf(-mr[mt][rh], cc ? sp.y : sp.x, cc ? bp.y : bp.x));
              u[rh][cc] = round_bf16f(fmaf(pp, tanh_approx(g), pp)) * mk[mt][rh];
            }
          __nv_bfloat16* o = p.planes + (size_t)ch * p.T + t0 + 32 * wm + 16 * mt + 2 * g8;
          stg32(o, pack_bf16(u[0][0], u[1][0]));
          stg32(o + p.T, pack_bf16(u[0][1], u[1][1]));
        }
      }
    }
  }
  cp_async_wait<0>();
}

}  // namespace a100
