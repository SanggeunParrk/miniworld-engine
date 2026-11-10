// opm_prologue_bwd_sm80.cuh -- OuterProductMean backward, last stage: from the gradients of the two projections (dA, dB, the cuBLAS products of dO with
// the kept B / A operands) to the MSA gradient and the weight gradients.  Everything between dA | dB and dm is one pass over the tokens:
//
//   da, db = dA, dB rows (token (s, i): 32 + 32 contiguous bf16), zeroed where the mask is 0 (the forward's `* mask`)
//   dy     = da . Wl + db . Wr                                  (fp32; mma.sync, M = 16 tokens per warp, K = 64, N = d_msa)
//   dx     = rstd (dxh - mean(dxh) - xhat mean(dxh xhat)),  dxh = dy gamma                 -> dm (bf16), xhat from the saved (mean, rstd)
//   dW[c | e, :] += [da | db]^T y,   y = bf16(LN(m)) recomputed from the saved statistics (the forward's bits)
//   dgamma += dy xhat, dbeta += dy
//
// Persistent: a CTA walks tiles of 64 MSA rows x 2 tokens (128 tokens, 8 warps x 16), prefetching the next tile's dA | dB / m rows and (mean, rstd) pairs (8-byte cp.async copies
// into shared memory) by cp.async, and the mask byte of the next tile into a register of the first 128 threads, while the
// current one is processed (NBUF = 2: one CTA per SM; NBUF = 1: the next tile's rows are requested at the end of the tile and two CTAs share the SM), and keeps dW
// (a [64][d_msa] fp32 tile, mma accumulators) in registers; dgamma / dbeta are summed per warp into shared memory (exclusive rows); one fp32 partial row per CTA is
// written at the end (`reduce_rows` sums them in a fixed order: no atomics, bit-reproducible).
#pragma once
#include "sm80_common.cuh"

namespace opm80 {

struct PbParams {
  const __nv_bfloat16* dA;     // [rows][ldd]: row s, column i * 32 + c
  const __nv_bfloat16* dB;
  const __nv_bfloat16* m;      // [S][L][CM]
  const float2* stats;         // [S][L] (mean, rstd)
  const uint8_t* mask;         // [S][L] or nullptr
  const float* lnw;            // [CM]
  const float* lnb;            // [CM]
  const __nv_bfloat16* wl;     // [32][CM]
  const __nv_bfloat16* wr;     // [32][CM]
  __nv_bfloat16* dm;           // [S][L][CM]
  float* part;                 // [gridDim.x][64 * CM + 2 * CM]
  long ldd;
  int S, L, ntile_s, ntile;
};

template <int CM, int NBUF = 2> struct PbCfg {
  static constexpr int NTOK = 128, NTHR = 256, NCH = CM / 8;
  static constexpr int WB = 64 * CM * 2, DAB = NTOK * 128, XT = NTOK * CM * 2, YT = NTOK * CM * 2;
  static constexpr int STB = NTOK * 8;                                   // one tile's (mean, rstd) pairs
  static constexpr int SMEM = WB + NBUF * DAB + NBUF * XT + YT + 2 * CM * 4 + 8 * 2 * CM * 4 + NBUF * STB;
  static constexpr int PW = 64 * CM + 2 * CM;
  static_assert(CM == 64 || CM == 128, "d_msa");
  static_assert(NBUF == 1 || NBUF == 2, "buffers");
  static_assert(SMEM <= 166912, "sm_80 shared memory");
};

template <int CM, int NBUF>
DEVI void pb_load(const PbParams& p, uint32_t sdab, uint32_t sx, uint32_t sst, int tl, int tid) {
  using C = PbCfg<CM, NBUF>;
  const int s0 = (tl % p.ntile_s) * 64, i0 = (tl / p.ntile_s) * 2;
  if (tid < C::NTOK) {                                                  // the tile's (mean, rstd) pairs: 8-byte copies, zero for padded tokens
    const int t = tid, s = s0 + (t >> 1), i = i0 + (t & 1);
    const bool ok = (s < p.S) & (i < p.L);
    cp_async8(sst + t * 8, p.stats + (ok ? (long)s * p.L + i : 0), ok ? 8u : 0u);
  }
  // dA | dB: two streams of 128 tokens x 4 pieces (64 B per token)
#pragma unroll
  for (int it = 0; it < 2; ++it) {
    const int v = it * 256 + tid, ch4 = v & 3, il = (v >> 2) & 1, sl = v >> 3;
    const int s = s0 + sl, i = i0 + il;
    const bool ok = (s < p.S) & (i < p.L);
    const long off = ok ? (long)s * p.ldd + (long)i * 32 + ch4 * 8 : 0;
    const int t = 2 * sl + il;
    cp_async16(sdab + swz128(t, ch4), p.dA + off, ok ? 16u : 0u);
    cp_async16(sdab + swz128(t, 4 + ch4), p.dB + off, ok ? 16u : 0u);
  }
  // m rows
#pragma unroll
  for (int it = 0; it < C::NTOK * C::NCH / 256; ++it) {
    const int v = it * 256 + tid, ch = v % C::NCH, t = v / C::NCH;
    const int sl = t >> 1, il = t & 1;
    const int s = s0 + sl, i = i0 + il;
    const bool ok = (s < p.S) & (i < p.L);
    cp_async16(sx + swzn<C::NCH>(t, ch), p.m + (ok ? ((long)s * p.L + i) * CM + ch * 8 : 0), ok ? 16u : 0u);
  }
}

template <int CM, int NBUF, int NMINB>
__global__ void __launch_bounds__(256, NMINB) opm_prologue_bwd_kernel(const PbParams p) {
  using C = PbCfg<CM, NBUF>;
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ uint8_t smask[C::NTOK];
  const uint32_t sw = smem_u32(smem), sdab0 = sw + C::WB, sx0 = sdab0 + NBUF * C::DAB, sy = sx0 + NBUF * C::XT;
  float* sgb = reinterpret_cast<float*>(smem + C::WB + NBUF * C::DAB + NBUF * C::XT + C::YT);       // gamma [CM], beta [CM]
  float* sred = sgb + 2 * CM;                                                                         // [8 warps][dgamma CM | dbeta CM]
  const float2* sst_gen = reinterpret_cast<const float2*>(sred + 8 * 2 * CM);                         // [NBUF][128] (mean, rstd) of the tiles
  const uint32_t sst0 = smem_u32(sst_gen);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp & 1, wn = warp >> 1;       // dW tile: 2 (rows of [64]) x 4 (columns of CM)
  const int gq = lane >> 2, q = lane & 3;

  for (int u = tid; u < 64 * C::NCH; u += C::NTHR) {
    const int row = u / C::NCH, ch = u % C::NCH;
    cp_async16(sw + swzn<C::NCH>(row, ch), (row < 32 ? p.wl + row * CM : p.wr + (row - 32) * CM) + ch * 8);
  }
  for (int k = tid; k < CM; k += C::NTHR) { sgb[k] = p.lnw[k]; sgb[CM + k] = p.lnb[k]; }
  for (int k = tid; k < 8 * 2 * CM; k += C::NTHR) sred[k] = 0.f;
  int tl = blockIdx.x, buf = 0;
  if (tl < p.ntile) pb_load<CM, NBUF>(p, sdab0, sx0, sst0, tl, tid);
  cp_async_commit();
  // the tile's mask bytes are fetched one tile ahead into a register of the first NTOK threads (no global load in front of the barriers)
  auto fetch = [&](int tl2) -> uint32_t {                                // 0 for padded tokens, 1 without a mask
    if (tid >= C::NTOK || tl2 >= p.ntile) return 0u;
    const int s = (tl2 % p.ntile_s) * 64 + (tid >> 1), i = (tl2 / p.ntile_s) * 2 + (tid & 1);
    if ((s >= p.S) | (i >= p.L)) return 0u;
    return p.mask == nullptr ? 1u : (uint32_t)p.mask[(long)s * p.L + i];
  };
  uint32_t mk = fetch(tl);

  float aw[2][CM / 32][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < CM / 32; ++nt) aw[mt][nt][0] = aw[mt][nt][1] = aw[mt][nt][2] = aw[mt][nt][3] = 0.f;

  const int lr = lane / C::NCH, piece = lane % C::NCH;     // LayerNorm-recompute lane mapping (NCH lanes a row)

#pragma unroll 1
  for (; tl < p.ntile; tl += gridDim.x, buf = NBUF == 2 ? buf ^ 1 : 0) {
    const int nx = tl + gridDim.x;
    if (NBUF == 2) {
      if (nx < p.ntile) pb_load<CM, NBUF>(p, sdab0 + (buf ^ 1) * C::DAB, sx0 + (buf ^ 1) * C::XT, sst0 + (buf ^ 1) * C::STB, nx, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const uint32_t sdab = sdab0 + buf * C::DAB, sx = sx0 + buf * C::XT;
    const int s0 = (tl % p.ntile_s) * 64, i0 = (tl / p.ntile_s) * 2;

    const float2* sst = sst_gen + buf * C::NTOK;
    // ---- the mask: zero the rows of masked / padded tokens
    if (tid < C::NTOK) {
      const int t = tid;
      const bool ok = mk != 0;
      smask[t] = ok;
      if (!ok) {
#pragma unroll
        for (int ch = 0; ch < 8; ++ch) sts128(sdab + swz128(t, ch), make_uint4(0, 0, 0, 0));
      }
    }
    mk = fetch(tl + gridDim.x);                                          // consumed at the next iteration
    // ---- y = bf16(LN(m)) from the saved statistics (this warp's 16 tokens, NCH lanes a row)
    {
      float lg[8], lb[8];
      const float4 g0 = *reinterpret_cast<const float4*>(sgb + piece * 8), g1 = *reinterpret_cast<const float4*>(sgb + piece * 8 + 4);
      const float4 b0 = *reinterpret_cast<const float4*>(sgb + CM + piece * 8), b1 = *reinterpret_cast<const float4*>(sgb + CM + piece * 8 + 4);
      lg[0] = g0.x; lg[1] = g0.y; lg[2] = g0.z; lg[3] = g0.w; lg[4] = g1.x; lg[5] = g1.y; lg[6] = g1.z; lg[7] = g1.w;
      lb[0] = b0.x; lb[1] = b0.y; lb[2] = b0.z; lb[3] = b0.w; lb[4] = b1.x; lb[5] = b1.y; lb[6] = b1.z; lb[7] = b1.w;
#pragma unroll
      for (int it = 0; it < 16 / (32 / C::NCH); ++it) {
        const int t = warp * 16 + it * (32 / C::NCH) + lr;
        const int s = s0 + (t >> 1), i = i0 + (t & 1);
        const bool ok = (s < p.S) & (i < p.L);
        const uint4 raw = lds128(sx + swzn<C::NCH>(t, piece));
        const float2 st = ok ? sst[t] : make_float2(0.f, 1.f);
        const float x[8] = {bf16lo(raw.x), bf16hi(raw.x), bf16lo(raw.y), bf16hi(raw.y), bf16lo(raw.z), bf16hi(raw.z), bf16lo(raw.w), bf16hi(raw.w)};
        uint4 y = make_uint4(0, 0, 0, 0);
        if (ok) {
          y.x = pack_bf16(fmaf((x[0] - st.x) * st.y, lg[0], lb[0]), fmaf((x[1] - st.x) * st.y, lg[1], lb[1]));
          y.y = pack_bf16(fmaf((x[2] - st.x) * st.y, lg[2], lb[2]), fmaf((x[3] - st.x) * st.y, lg[3], lb[3]));
          y.z = pack_bf16(fmaf((x[4] - st.x) * st.y, lg[4], lb[4]), fmaf((x[5] - st.x) * st.y, lg[5], lb[5]));
          y.w = pack_bf16(fmaf((x[6] - st.x) * st.y, lg[6], lb[6]), fmaf((x[7] - st.x) * st.y, lg[7], lb[7]));
        }
        sts128(sy + swzn<C::NCH>(t, piece), y);
      }
    }
    __syncthreads();                     // the mask zeroing, y and (first tile) the weights are visible; the next tile's loads keep running

    // ---- dy = [da | db] [Wl; Wr]  (16 tokens per warp)
    float acc[CM / 8][4];
#pragma unroll
    for (int nt = 0; nt < CM / 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t af[4];
      ldsm_x4(af, sdab + swz128(warp * 16 + (lane & 15), 2 * ks + (lane >> 4)));
#pragma unroll
      for (int np = 0; np < CM / 16; ++np) {
        uint32_t bf[4];
        ldsm_x4_t(bf, sw + swzn<C::NCH>(16 * ks + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * np + (lane >> 4)));
        mma16816(acc[2 * np], af, bf[0], bf[1]);
        mma16816(acc[2 * np + 1], af, bf[2], bf[3]);
      }
    }
    // ---- LayerNorm backward on the accumulators (row r0 = 16 warp + gq, r1 = r0 + 8), dx -> the m tile in place
    {
      const int r0 = warp * 16 + gq, r1 = r0 + 8;
      const int sA = s0 + (r0 >> 1), iA = i0 + (r0 & 1), sB = s0 + (r1 >> 1), iB = i0 + (r1 & 1);
      const bool okA = (sA < p.S) & (iA < p.L), okB = (sB < p.S) & (iB < p.L);
      const float2 stA = okA ? sst[r0] : make_float2(0.f, 1.f);
      const float2 stB = okB ? sst[r1] : make_float2(0.f, 1.f);
      float s1A = 0.f, s2A = 0.f, s1B = 0.f, s2B = 0.f;
#pragma unroll
      for (int nt = 0; nt < CM / 8; ++nt) {
        const int col = nt * 8 + 2 * q;
        const float2 gm = *reinterpret_cast<const float2*>(sgb + col);
        const uint32_t xa = lds32(sx + swzn<C::NCH>(r0, col >> 3) + (col & 7) * 2), xb = lds32(sx + swzn<C::NCH>(r1, col >> 3) + (col & 7) * 2);
        const float h0 = (bf16lo(xa) - stA.x) * stA.y, h1 = (bf16hi(xa) - stA.x) * stA.y, h2 = (bf16lo(xb) - stB.x) * stB.y, h3 = (bf16hi(xb) - stB.x) * stB.y;
        const float da0 = acc[nt][0] * gm.x, da1 = acc[nt][1] * gm.y, db0 = acc[nt][2] * gm.x, db1 = acc[nt][3] * gm.y;
        s1A += da0 + da1; s2A = fmaf(da0, h0, fmaf(da1, h1, s2A));
        s1B += db0 + db1; s2B = fmaf(db0, h2, fmaf(db1, h3, s2B));
      }
      s1A += __shfl_xor_sync(0xffffffffu, s1A, 1); s1A += __shfl_xor_sync(0xffffffffu, s1A, 2);
      s2A += __shfl_xor_sync(0xffffffffu, s2A, 1); s2A += __shfl_xor_sync(0xffffffffu, s2A, 2);
      s1B += __shfl_xor_sync(0xffffffffu, s1B, 1); s1B += __shfl_xor_sync(0xffffffffu, s1B, 2);
      s2B += __shfl_xor_sync(0xffffffffu, s2B, 1); s2B += __shfl_xor_sync(0xffffffffu, s2B, 2);
      const float m1A = s1A * (1.f / CM), m2A = s2A * (1.f / CM), m1B = s1B * (1.f / CM), m2B = s2B * (1.f / CM);
      float* wr = sred + warp * 2 * CM;
#pragma unroll
      for (int nt = 0; nt < CM / 8; ++nt) {
        const int col = nt * 8 + 2 * q;
        const float2 gm = *reinterpret_cast<const float2*>(sgb + col);
        const uint32_t xa = lds32(sx + swzn<C::NCH>(r0, col >> 3) + (col & 7) * 2), xb = lds32(sx + swzn<C::NCH>(r1, col >> 3) + (col & 7) * 2);
        const float h0 = (bf16lo(xa) - stA.x) * stA.y, h1 = (bf16hi(xa) - stA.x) * stA.y, h2 = (bf16lo(xb) - stB.x) * stB.y, h3 = (bf16hi(xb) - stB.x) * stB.y;
        const float dxa0 = stA.y * (acc[nt][0] * gm.x - m1A - h0 * m2A), dxa1 = stA.y * (acc[nt][1] * gm.y - m1A - h1 * m2A);
        const float dxb0 = stB.y * (acc[nt][2] * gm.x - m1B - h2 * m2B), dxb1 = stB.y * (acc[nt][3] * gm.y - m1B - h3 * m2B);
        sts32(sx + swzn<C::NCH>(r0, col >> 3) + (col & 7) * 2, pack_bf16(dxa0, dxa1));
        sts32(sx + swzn<C::NCH>(r1, col >> 3) + (col & 7) * 2, pack_bf16(dxb0, dxb1));
        // dgamma = sum dy xhat, dbeta = sum dy over the warp's 16 rows: the eight row lanes of the column first, then the warp's exclusive shared row
        float gA = acc[nt][0] * h0 + acc[nt][2] * h2, gB = acc[nt][1] * h1 + acc[nt][3] * h3;
        float bA = acc[nt][0] + acc[nt][2], bB = acc[nt][1] + acc[nt][3];
        gA += __shfl_xor_sync(0xffffffffu, gA, 4); gA += __shfl_xor_sync(0xffffffffu, gA, 8); gA += __shfl_xor_sync(0xffffffffu, gA, 16);
        gB += __shfl_xor_sync(0xffffffffu, gB, 4); gB += __shfl_xor_sync(0xffffffffu, gB, 8); gB += __shfl_xor_sync(0xffffffffu, gB, 16);
        bA += __shfl_xor_sync(0xffffffffu, bA, 4); bA += __shfl_xor_sync(0xffffffffu, bA, 8); bA += __shfl_xor_sync(0xffffffffu, bA, 16);
        bB += __shfl_xor_sync(0xffffffffu, bB, 4); bB += __shfl_xor_sync(0xffffffffu, bB, 8); bB += __shfl_xor_sync(0xffffffffu, bB, 16);
        if (gq == 0) { wr[col] += gA; wr[col + 1] += gB; wr[CM + col] += bA; wr[CM + col + 1] += bB; }
      }
    }
    // ---- dW += [da | db]^T y   (M = 64 channels, N = d_msa, K = the 128 tokens)
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t af[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) ldsm_x4_t(af[mt], sdab + swz128(16 * ks + (lane & 7) + ((lane >> 4) << 3), 4 * wm + 2 * mt + ((lane >> 3) & 1)));
#pragma unroll
      for (int np = 0; np < CM / 64; ++np) {
        uint32_t bf[4];
        ldsm_x4_t(bf, sy + swzn<C::NCH>(16 * ks + (lane & 7) + (((lane >> 3) & 1) << 3), wn * (CM / 32) + 2 * np + (lane >> 4)));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma16816(aw[mt][2 * np], af[mt], bf[0], bf[1]);
          mma16816(aw[mt][2 * np + 1], af[mt], bf[2], bf[3]);
        }
      }
    }
    __syncthreads();
    // ---- dm tile (in the m tile) -> global
#pragma unroll
    for (int it = 0; it < C::NTOK * C::NCH / 256; ++it) {
      const int v = it * 256 + tid, ch = v % C::NCH, t = v / C::NCH;
      const int s = s0 + (t >> 1), i = i0 + (t & 1);
      if ((s < p.S) & (i < p.L)) stg128(p.dm + ((long)s * p.L + i) * CM + ch * 8, lds128(sx + swzn<C::NCH>(t, ch)));
    }
    __syncthreads();                      // buffer `buf` is rewritten by the next prefetch
    if (NBUF == 1) {                      // one buffer: the next tile's rows are requested only now
      const int nx1 = tl + gridDim.x;
      if (nx1 < p.ntile) pb_load<CM, NBUF>(p, sdab0, sx0, sst0, nx1, tid);
      cp_async_commit();
    }
  }
  cp_async_wait<0>();

  // ---- the CTA's partial row: dW | dgamma | dbeta
  float* part = p.part + (long)blockIdx.x * C::PW;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < CM / 32; ++nt) {
      const int r0 = wm * 32 + mt * 16 + gq, col = wn * (CM / 4) + nt * 8 + 2 * q;
      stg64(part + (long)r0 * CM + col, make_float2(aw[mt][nt][0], aw[mt][nt][1]));
      stg64(part + (long)(r0 + 8) * CM + col, make_float2(aw[mt][nt][2], aw[mt][nt][3]));
    }
  __syncthreads();
  for (int col = tid; col < 2 * CM; col += C::NTHR) {
    float a = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) a += sred[w * 2 * CM + col];
    part[64 * CM + col] = a;
  }
}

}  // namespace opm80
