// mod_sm80.cuh -- the adaLN modulation of the SWA atom DiT block on A100 / sm_80: mod = silu(c) Wmod^T [R][768] fp32 and its backward (dc, dWmod).  The twin of the sm_100a
// mod_fwd / mod_bwd of the B200 build; what the engine otherwise runs is ``silu(c).float() @ Wmod.float().T`` (an fp32 GEMM and three elementwise passes).
//
//   forward   a = rn(silu(c)) (bf16, as the framework's bf16 silu), mod = a Wmod^T accumulated in fp32.  Products of two bf16 numbers are exact in fp32, so this is that fp32 GEMM
//             up to the order of the sums.  ``a`` is also written out for the backward (dWmod reads it).
//   backward  g = d mod (fp32): dA = rn(g Wmod), dc = rn(dA sigmoid(c) (1 + c (1 - sigmoid(c)))) (the framework's silu backward on the bf16 gradient), dWmod = rn(g^T a).  g goes
//             through the tensor cores as two bf16 terms (hi = rn(g), lo = rn(g - hi): 16 significant bits where the fp32 GEMM's TF32 operand has 11) for dc; dWmod sums over every
//             row, so the rounding of g to bf16 (unbiased) averages out and its single term is enough (TERMS = 1; the 2-term build stays selectable).
//
// The kernels are bound by the DRAM traffic of the [R][768] fp32 tensor (3 KB per row against 0.25 KB of c): the forward writes it once, the backward reads it twice (dc, dW).
//   mod_fwd_kernel   row stationary: a CTA deals itself tiles of 32 NW rows; the tile's silu(c) rows go into registers as A fragments, Wmod streams through a cp.async ring in
//                    chunks of 64 output columns (shared by the warps, 2 row blocks per warp so every B fragment feeds 2 MMAs); the next tile's rows are requested a chunk
//                    loop ahead.
//   mod_dc_kernel    row tiles of 16 NW rows, the 768-long reduction streamed through a cp.async ring (g as fp32, Wmod rows), dA stays in registers until the epilogue.
//   mod_dw_kernel    a CTA owns a 128-row slab of dWmod (64 accumulator registers per thread) and a contiguous range of 32-row chunks; the partial sums of the P ranges go to ``part``
//                    and mod_dw_reduce_kernel adds them in a fixed order (deterministic).
#pragma once
#include "sm80_common.cuh"

namespace sw80 {

constexpr int MOD_K = 128;                           // d_cond = C
constexpr int MOD_N = 768;                           // 6 C

DEVI float silu_bwd_f(float dy, float x) {           // the framework's silu_backward: dy sigmoid (1 + x (1 - sigmoid))
  const float s = sigmoidf(x);
  return dy * s * (1.f + x * (1.f - s));
}
DEVI uint32_t silu_bwd_pair(uint32_t d, uint32_t x) { return pack_bf16(silu_bwd_f(bf16lo(d), bf16lo(x)), silu_bwd_f(bf16hi(d), bf16hi(x))); }
DEVI uint4 silu_pack8(const uint4 v) {               // rn(silu(c)) of 8 bf16
  return make_uint4(pack_bf16(silu_f(bf16lo(v.x)), silu_f(bf16hi(v.x))), pack_bf16(silu_f(bf16lo(v.y)), silu_f(bf16hi(v.y))),
                    pack_bf16(silu_f(bf16lo(v.z)), silu_f(bf16hi(v.z))), pack_bf16(silu_f(bf16lo(v.w)), silu_f(bf16hi(v.w))));
}
DEVI float4 ldg_f4(const float* p) { return __ldg(reinterpret_cast<const float4*>(p)); }
// two fp32 -> (hi, lo) bf16 pairs: hi = rn(x), lo = rn(x - hi)
DEVI void split2(float x0, float x1, uint32_t& hi, uint32_t& lo) {
  hi = pack_bf16(x0, x1);
  lo = pack_bf16(x0 - bf16lo(hi), x1 - bf16hi(hi));
}
// rows of the fp32 g tiles (``row_bytes`` = 128 / 256 B) are read as 8-byte pairs of 4 rows x 2 neighbouring granules per half warp: the granule index is XOR-ed with 2 (row & 3)
DEVI uint32_t swg(uint32_t row, uint32_t granule, uint32_t row_bytes) { return row * row_bytes + ((granule ^ ((row & 3u) << 1)) << 4); }

// ================================================================== forward
struct ModFwdParams {
  const __nv_bfloat16* c;                            // [R][128]
  const __nv_bfloat16* w;                            // [768][128]
  float* out;                                        // [R][768]
  __nv_bfloat16* a;                                  // [R][128] rn(silu(c)) for the backward, or nullptr
  int R;
};

template <int NW_>
struct ModFwdCfg {
  static constexpr int NW = NW_, NTHR = 32 * NW_, BM = 32 * NW_;           // rows per tile (a warp: 2 blocks of 16)
  static constexpr int CC = 64, NCH = MOD_N / CC, NSTAGE = 3;              // output columns per chunk, chunks per tile, depth of the Wmod ring
  static constexpr int A_BYTES = BM * 256, W_BYTES = CC * 256;
  static constexpr int SMEM = A_BYTES + NSTAGE * W_BYTES;
  static constexpr int MINB = 2;                                           // CTAs per SM
};

template <class G>
DEVI void mf_load_a(const ModFwdParams& p, uint32_t abuf, int tile, int tid) {       // the raw c rows of a tile, zero past R
#pragma unroll
  for (int i = tid; i < G::BM * 16; i += G::NTHR) {
    const int row = i >> 4, gc = i & 15, r = tile * G::BM + row;
    cp_async16(abuf + sw256(row, gc), p.c + (size_t)(r < p.R ? r : 0) * MOD_K + gc * 8, r < p.R ? 16u : 0u);
  }
}
template <class G>
DEVI void mf_load_w(const ModFwdParams& p, uint32_t stage, int j, int tid) {         // Wmod rows 64 j .. 64 j + 63 (the output columns of chunk j)
#pragma unroll
  for (int i = tid; i < G::CC * 16; i += G::NTHR) {
    const int row = i >> 4, gc = i & 15;
    cp_async16(stage + sw256(row, gc), p.w + (size_t)(j * G::CC + row) * MOD_K + gc * 8);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) mod_fwd_kernel(const ModFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw), abuf = sb, wring = sb + G::A_BYTES;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int ntile = (p.R + G::BM - 1) / G::BM;
  const int mine = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;     // tiles blockIdx.x, + gridDim.x, ...
  const int total = mine * G::NCH;                                         // chunks of all this CTA's tiles, one stream through the ring

  if (mine > 0) mf_load_a<G>(p, abuf, blockIdx.x, tid);
  cp_async_commit();
#pragma unroll
  for (int s = 0; s < G::NSTAGE - 1; ++s) {
    if (s < total) mf_load_w<G>(p, wring + s * G::W_BYTES, s % G::NCH, tid);
    cp_async_commit();
  }

  uint32_t af[2][8][4];                                                    // the warp's 32 rows of silu(c) as A fragments (2 row blocks x 8 k steps)
  for (int ci = 0; ci < total; ++ci) {
    cp_async_wait<G::NSTAGE - 2>();
    __syncthreads();                                                       // chunk ci (and, at a tile start, the tile's rows) are in shared memory; the stage of chunk ci - 1 is free
    const int j = ci % G::NCH, tl = ci / G::NCH, tile = blockIdx.x + tl * gridDim.x;
    if (j == 0) {
#pragma unroll
      for (int i = tid; i < G::BM * 16; i += G::NTHR) {                    // silu in place on the raw rows, and the copy the backward reads
        const int row = i >> 4, gc = i & 15, r = tile * G::BM + row;
        const uint32_t addr = abuf + sw256(row, gc);
        const uint4 v = silu_pack8(lds128(addr));
        sts128(addr, v);
        if (p.a != nullptr && r < p.R) stg128(p.a + (size_t)r * MOD_K + gc * 8, v);
      }
      __syncthreads();
#pragma unroll
      for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) ldsm_x4(af[rb][ks], abuf + sw256(32 * warp + 16 * rb + (lane & 7) + 8 * ((lane >> 3) & 1), 2 * ks + (lane >> 4)));
      __syncthreads();                                                     // every warp has its fragments: the buffer is free for the next tile's rows
    }
    {
      const int cn = ci + G::NSTAGE - 1;
      if (cn < total) mf_load_w<G>(p, wring + (cn % G::NSTAGE) * G::W_BYTES, cn % G::NCH, tid);
      cp_async_commit();
    }
    if (j == 1 && tl + 1 < mine) {                                         // the next tile's rows, a whole chunk loop ahead of their use
      mf_load_a<G>(p, abuf, tile + (int)gridDim.x, tid);
      cp_async_commit();
    }

    const uint32_t wst = wring + (ci % G::NSTAGE) * G::W_BYTES;
    float acc[2][8][4];
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) { acc[rb][nt][0] = acc[rb][nt][1] = acc[rb][nt][2] = acc[rb][nt][3] = 0.f; }
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
#pragma unroll
      for (int n2 = 0; n2 < 4; ++n2) {                                     // n tiles 2 n2, 2 n2 + 1 of the chunk
        uint32_t r[4];
        ldsm_x4(r, wst + sw256(8 * (2 * n2 + (lane >> 4)) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          mma16816(acc[rb][2 * n2], af[rb][ks], r[0], r[1]);
          mma16816(acc[rb][2 * n2 + 1], af[rb][ks], r[2], r[3]);
        }
      }
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
      const int r0 = tile * G::BM + 32 * warp + 16 * rb + g8, r1 = r0 + 8;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = j * G::CC + nt * 8 + 2 * q4;
        if (r0 < p.R) stg64f(p.out + (size_t)r0 * MOD_N + col, acc[rb][nt][0], acc[rb][nt][1]);
        if (r1 < p.R) stg64f(p.out + (size_t)r1 * MOD_N + col, acc[rb][nt][2], acc[rb][nt][3]);
      }
    }
  }
}

// Weight stationary variant (small R): a CTA owns one 128-column chunk of Wmod (its B fragments sit in registers for the whole kernel) and deals itself tiles of 32 rows; the tiles'
// raw rows stream through a 3-deep cp.async ring, silu runs in place (once per tile and CTA; the chunk-0 CTAs also write the copy the backward reads).  6 chunks x P CTAs.
constexpr int MW_TR = 32, MW_THR = 128, MW_NST = 3, MW_SMEM = MW_NST * MW_TR * 256, MW_CHUNKS = MOD_N / 128;

__global__ void __launch_bounds__(MW_THR, 3) mod_fwd_ws_kernel(const ModFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int col0 = blockIdx.y * 128 + warp * 32;                    // this warp's 32 output columns = 4 n tiles
  const int ntile = (p.R + MW_TR - 1) / MW_TR;
  const int mine = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;     // tiles blockIdx.x, + gridDim.x, ...

  // the B fragments (b0: k = 2 q4, +1; b1: k + 8 of column g8 of each n tile) of all 8 k steps
  uint32_t bw[4][8][2];
  {
    const __nv_bfloat16* wp = p.w + (size_t)(col0 + g8) * MOD_K + 2 * q4;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        bw[nt][ks][0] = __ldg(reinterpret_cast<const unsigned*>(wp + nt * 8 * MOD_K + ks * 16));
        bw[nt][ks][1] = __ldg(reinterpret_cast<const unsigned*>(wp + nt * 8 * MOD_K + ks * 16 + 8));
      }
  }

  auto issue = [&](int k) {                                         // the raw rows of this CTA's k-th tile into ring slot k % MW_NST
    const int tile = blockIdx.x + k * gridDim.x;
    const uint32_t slot = sb + (k % MW_NST) * (MW_TR * 256);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = tid + MW_THR * j, row = i >> 4, gc = i & 15, r = tile * MW_TR + row;
      cp_async16(slot + sw256(row, gc), p.c + (size_t)(r < p.R ? r : 0) * MOD_K + gc * 8, r < p.R ? 16u : 0u);
    }
  };
#pragma unroll
  for (int s = 0; s < MW_NST - 1; ++s) {
    if (s < mine) issue(s);
    cp_async_commit();
  }

  for (int it = 0; it < mine; ++it) {
    cp_async_wait<MW_NST - 2>();
    __syncthreads();                                                // tile it is in shared memory; the slot of tile it - 1 is free
    const int tile = blockIdx.x + it * gridDim.x;
    const uint32_t cur = sb + (it % MW_NST) * (MW_TR * 256);
#pragma unroll
    for (int j = 0; j < 4; ++j) {                                   // silu in place (and the copy for the backward)
      const int i = tid + MW_THR * j, row = i >> 4, gc = i & 15, r = tile * MW_TR + row;
      const uint32_t addr = cur + sw256(row, gc);
      const uint4 v = silu_pack8(lds128(addr));
      sts128(addr, v);
      if (p.a != nullptr && blockIdx.y == 0 && r < p.R) stg128(p.a + (size_t)r * MOD_K + gc * 8, v);
    }
    __syncthreads();
    {
      const int nx = it + MW_NST - 1;
      if (nx < mine) issue(nx);
      cp_async_commit();
    }

    float acc[2][4][4];
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) { acc[rb][nt][0] = acc[rb][nt][1] = acc[rb][nt][2] = acc[rb][nt][3] = 0.f; }
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        uint32_t a[4];                                              // rows 16 rb + (lane & 7) + 8 ((lane >> 3) & 1), k granule 2 ks + (lane >> 4): a0 .. a3 in the A-fragment order
        ldsm_x4(a, cur + sw256(16 * rb + (lane & 7) + 8 * ((lane >> 3) & 1), 2 * ks + (lane >> 4)));
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma16816(acc[rb][nt], a, bw[nt][ks][0], bw[nt][ks][1]);
      }
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
      const int r0 = tile * MW_TR + 16 * rb + g8, r1 = r0 + 8;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col = col0 + nt * 8 + 2 * q4;
        if (r0 < p.R) stg64f(p.out + (size_t)r0 * MOD_N + col, acc[rb][nt][0], acc[rb][nt][1]);
        if (r1 < p.R) stg64f(p.out + (size_t)r1 * MOD_N + col, acc[rb][nt][2], acc[rb][nt][3]);
      }
    }
  }
}

// ================================================================== backward: dc
struct ModDcParams {
  const float* g;                                    // [R][768] d mod
  const __nv_bfloat16* c;                            // [R][128]
  const __nv_bfloat16* w;                            // [768][128]
  __nv_bfloat16* dc;                                 // [R][128]
  int R;
};

// BM = 16 NW rows per tile (a warp: 16 rows x all 128 outputs); the reduction over the 768 modulation channels in chunks of KC through a ring of NSTAGE stages
template <int NW_, int TERMS_, int KC_ = 32, int NSTAGE_ = 4>
struct ModDcCfg {
  static constexpr int NW = NW_, NTHR = 32 * NW_, BM = 16 * NW_, TERMS = TERMS_, KC = KC_, NSTAGE = NSTAGE_;
  static constexpr int NCHUNK = MOD_N / KC, KSTEPS = KC / 16, GROW = KC * 4, GGRAN = GROW / 16;   // chunks per tile, k steps per chunk, bytes / granules of a g row of a stage
  static constexpr int G_BYTES = BM * GROW, W_BYTES = KC * 256, STAGE = G_BYTES + W_BYTES;
  static constexpr int SMEM = NSTAGE * STAGE + BM * 256;           // the ring + the dA tile of the epilogue
  static constexpr int MINB = (SMEM <= 83000) ? 2 : 1;             // CTAs per SM
};

template <class G>
DEVI void dc_load(const ModDcParams& p, uint32_t stage, int row0, int j, int tid) {
#pragma unroll
  for (int i = tid; i < G::BM * G::GGRAN; i += G::NTHR) {          // g: BM rows x KC floats
    const int row = i / G::GGRAN, gr = i % G::GGRAN, r = row0 + row;
    cp_async16(stage + swg(row, gr, G::GROW), p.g + (size_t)(r < p.R ? r : 0) * MOD_N + j * G::KC + gr * 4, r < p.R ? 16u : 0u);
  }
#pragma unroll
  for (int i = tid; i < G::KC * 16; i += G::NTHR) {                // Wmod rows j KC .. + KC - 1 (the reduction index), 128 k
    const int row = i >> 4, gr = i & 15;
    cp_async16(stage + G::G_BYTES + sw256(row, gr), p.w + (size_t)(j * G::KC + row) * MOD_K + gr * 8);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) mod_dc_kernel(const ModDcParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw), epi = sb + G::NSTAGE * G::STAGE;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int ntile = (p.R + G::BM - 1) / G::BM;
  const int mine = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;     // tiles blockIdx.x, + gridDim.x, ...
  const int total = mine * G::NCHUNK;                              // chunks of all this CTA's tiles, one stream through the ring
  const int rA = warp * 16 + g8, rB = rA + 8;

#pragma unroll
  for (int s = 0; s < G::NSTAGE - 1; ++s) {
    if (s < total) dc_load<G>(p, sb + s * G::STAGE, (blockIdx.x + (s / G::NCHUNK) * gridDim.x) * G::BM, s % G::NCHUNK, tid);
    cp_async_commit();
  }
  float acc[16][4];
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f; }

  for (int ci = 0; ci < total; ++ci) {
    cp_async_wait<G::NSTAGE - 2>();
    __syncthreads();                                               // chunk ci is in shared memory; the stage of chunk ci - 1 is free
    {
      const int cn = ci + G::NSTAGE - 1;
      if (cn < total) dc_load<G>(p, sb + (cn % G::NSTAGE) * G::STAGE, (blockIdx.x + (cn / G::NCHUNK) * gridDim.x) * G::BM, cn % G::NCHUNK, tid);
      cp_async_commit();
    }
    const uint32_t gs = sb + (ci % G::NSTAGE) * G::STAGE, ws = gs + G::G_BYTES;
#pragma unroll
    for (int s = 0; s < G::KSTEPS; ++s) {                          // k steps of 16 reduction channels
      uint32_t ahi[4], alo[4];
      const int cg = 4 * s + (q4 >> 1), off = (q4 & 1) * 8;        // floats 16 s + 2 q4, +1 of a row (and + 8, +9)
      const uint2 v0 = lds64(gs + swg(rA, cg, G::GROW) + off), v1 = lds64(gs + swg(rB, cg, G::GROW) + off);
      const uint2 v2 = lds64(gs + swg(rA, cg + 2, G::GROW) + off), v3 = lds64(gs + swg(rB, cg + 2, G::GROW) + off);
      split2(__uint_as_float(v0.x), __uint_as_float(v0.y), ahi[0], alo[0]);
      split2(__uint_as_float(v1.x), __uint_as_float(v1.y), ahi[1], alo[1]);
      split2(__uint_as_float(v2.x), __uint_as_float(v2.y), ahi[2], alo[2]);
      split2(__uint_as_float(v3.x), __uint_as_float(v3.y), ahi[3], alo[3]);
      uint32_t bq[8][4];                                           // the B fragments of the 16 n tiles (output channels 16 n2 .. + 15), then all hi products, then all lo ones
#pragma unroll
      for (int n2 = 0; n2 < 8; ++n2) ldsm_x4_t(bq[n2], ws + sw256(16 * s + (lane & 7) + 8 * ((lane >> 3) & 1), 2 * n2 + (lane >> 4)));
#pragma unroll
      for (int n2 = 0; n2 < 8; ++n2) {
        mma16816(acc[2 * n2], ahi, bq[n2][0], bq[n2][1]);
        mma16816(acc[2 * n2 + 1], ahi, bq[n2][2], bq[n2][3]);
      }
      if constexpr (G::TERMS == 2) {
#pragma unroll
        for (int n2 = 0; n2 < 8; ++n2) {
          mma16816(acc[2 * n2], alo, bq[n2][0], bq[n2][1]);
          mma16816(acc[2 * n2 + 1], alo, bq[n2][2], bq[n2][3]);
        }
      }
    }

    if (ci % G::NCHUNK == G::NCHUNK - 1) {                         // the tile's reduction is complete: dA -> bf16 -> silu backward -> dc
      const int tile = blockIdx.x + (ci / G::NCHUNK) * gridDim.x;
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        sts32(epi + sw256(rA, nt) + 4 * q4, pack_bf16(acc[nt][0], acc[nt][1]));
        sts32(epi + sw256(rB, nt) + 4 * q4, pack_bf16(acc[nt][2], acc[nt][3]));
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
      }
      __syncthreads();
#pragma unroll
      for (int i = tid; i < G::BM * 16; i += G::NTHR) {
        const int row = i >> 4, gc = i & 15, r = tile * G::BM + row;
        if (r < p.R) {
          const uint4 da = lds128(epi + sw256(row, gc));
          const uint4 cv = ldg128(p.c + (size_t)r * MOD_K + gc * 8);
          stg128(p.dc + (size_t)r * MOD_K + gc * 8, make_uint4(silu_bwd_pair(da.x, cv.x), silu_bwd_pair(da.y, cv.y), silu_bwd_pair(da.z, cv.z), silu_bwd_pair(da.w, cv.w)));
        }
      }
    }
  }
}

// ================================================================== backward: dWmod
constexpr int DW_THR = 256, DW_RC = 32, DW_TILE = DW_RC * 256;     // rows per chunk; bytes of a [32][128] bf16 tile
constexpr int DW_BUF = 3 * DW_TILE, DW_SMEM = 2 * DW_BUF;          // a buffer = g hi | g lo | silu(c)

struct ModDwParams {
  const float* g;                                    // [R][768]
  const __nv_bfloat16* a;                            // [R][128] rn(silu(c)), written by the forward
  float* part;                                       // [P][768][128] fp32 partial sums
  int R, cpc;                                        // rows; chunks of 32 rows per CTA
};

template <int TERMS>
__global__ void __launch_bounds__(DW_THR, 2) mod_dw_kernel(const ModDwParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int wm = warp >> 1, wn = warp & 1;                         // the warp's 32 modulation channels (2 m tiles) x 64 output channels k (8 n tiles) of the slab
  const int slab = blockIdx.y;
  const int nchunk = (p.R + DW_RC - 1) / DW_RC;
  const int c0 = blockIdx.x * p.cpc, c1 = min(c0 + p.cpc, nchunk);

  float acc[2][8][4];
#pragma unroll
  for (int mb = 0; mb < 2; ++mb)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) { acc[mb][nt][0] = acc[mb][nt][1] = acc[mb][nt][2] = acc[mb][nt][3] = 0.f; }

  float4 gv[4];
  uint4 cv[2];
  auto load = [&](int ch) {
    const int r0 = ch * DW_RC;
#pragma unroll
    for (int j = 0; j < 4; ++j) {                                  // 32 rows x 128 floats: row i >> 5, float4 i & 31
      const int i = tid + DW_THR * j, r = r0 + (i >> 5);
      gv[j] = r < p.R ? ldg_f4(p.g + (size_t)r * MOD_N + slab * 128 + 4 * (i & 31)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {                                  // 32 rows x 16 granules of silu(c)
      const int i = tid + DW_THR * j, r = r0 + (i >> 4);
      cv[j] = r < p.R ? ldg128(p.a + (size_t)r * MOD_K + (i & 15) * 8) : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  auto stash = [&](uint32_t buf) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = tid + DW_THR * j, row = i >> 5, c4 = i & 31;
      const uint32_t off = sw256(row, c4 >> 1) + (c4 & 1) * 8;
      if constexpr (TERMS == 2) {
        uint32_t h01, l01, h23, l23;
        split2(gv[j].x, gv[j].y, h01, l01);
        split2(gv[j].z, gv[j].w, h23, l23);
        sts64(buf + off, make_uint2(h01, h23));
        sts64(buf + DW_TILE + off, make_uint2(l01, l23));
      } else {
        sts64(buf + off, make_uint2(pack_bf16(gv[j].x, gv[j].y), pack_bf16(gv[j].z, gv[j].w)));
      }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int i = tid + DW_THR * j;
      sts128(buf + 2 * DW_TILE + sw256(i >> 4, i & 15), cv[j]);
    }
  };

  if (c0 < c1) { load(c0); stash(sb); }
  __syncthreads();
  for (int ch = c0, it = 0; ch < c1; ++ch, ++it) {
    const bool more = ch + 1 < c1;
    if (more) load(ch + 1);
    const uint32_t buf = sb + (it & 1) * DW_BUF, gh = buf, gl = buf + DW_TILE, at = buf + 2 * DW_TILE;
#pragma unroll
    for (int s = 0; s < 2; ++s) {                                  // two k steps of 16 rows (the reduction runs over the rows)
      uint32_t ah[2][4], al[2][4];
#pragma unroll
      for (int mb = 0; mb < 2; ++mb) {                             // A = g^T: matrices (rows 0-7 | 8-15) x (channels 0-7 | 8-15), transposed on load
        const uint32_t ad = sw256(16 * s + (lane & 7) + 8 * (lane >> 4), 4 * wm + 2 * mb + ((lane >> 3) & 1));
        ldsm_x4_t(ah[mb], gh + ad);
        if constexpr (TERMS == 2) ldsm_x4_t(al[mb], gl + ad);
      }
      uint32_t bq[4][4];                                           // B = silu(c): n tiles 2 n2, 2 n2 + 1 of the warp's 64 output channels
#pragma unroll
      for (int n2 = 0; n2 < 4; ++n2) ldsm_x4_t(bq[n2], at + sw256(16 * s + (lane & 7) + 8 * ((lane >> 3) & 1), 8 * wn + 2 * n2 + (lane >> 4)));
#pragma unroll
      for (int n2 = 0; n2 < 4; ++n2)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int mb = 0; mb < 2; ++mb) mma16816(acc[mb][2 * n2 + h], ah[mb], bq[n2][2 * h], bq[n2][2 * h + 1]);
      if constexpr (TERMS == 2) {
#pragma unroll
        for (int n2 = 0; n2 < 4; ++n2)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int mb = 0; mb < 2; ++mb) mma16816(acc[mb][2 * n2 + h], al[mb], bq[n2][2 * h], bq[n2][2 * h + 1]);
      }
    }
    if (more) stash(sb + ((it + 1) & 1) * DW_BUF);
    __syncthreads();
  }

  float* base = p.part + ((size_t)blockIdx.x * MOD_N + slab * 128) * MOD_K;
#pragma unroll
  for (int mb = 0; mb < 2; ++mb)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int n = wm * 32 + mb * 16 + g8, k = wn * 64 + nt * 8 + 2 * q4;
      stg64f(base + (size_t)n * MOD_K + k, acc[mb][nt][0], acc[mb][nt][1]);
      stg64f(base + (size_t)(n + 8) * MOD_K + k, acc[mb][nt][2], acc[mb][nt][3]);
    }
}

// dWmod = rn(sum over the P partial sums, in order)
__global__ void mod_dw_reduce_kernel(const float* __restrict__ part, __nv_bfloat16* __restrict__ dw, int parts) {
  constexpr int N4 = MOD_N * MOD_K / 4;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N4) return;
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int q = 0; q < parts; ++q) {
    const float4 v = __ldg(reinterpret_cast<const float4*>(part) + (size_t)q * N4 + i);
    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  *reinterpret_cast<uint2*>(dw + 4 * (size_t)i) = make_uint2(pack_bf16(s.x, s.y), pack_bf16(s.z, s.w));
}

}  // namespace sw80
