// k1_sm80.cuh -- TriMul K1 (input LayerNorm + gated projections + pair mask -> channel-major planes), A100 / sm_80.
//
//   ab[oc, t] = bf16( sigmoid(LN_in(z)[t] . Wg[oc]) * (LN_in(z)[t] . Wp[oc]) * m_i m_j ),  t = i L + j,  oc in [0, 2 CH)  (a | b planes)
//
// Ideas carried over from the sm_90 K1 (tmn_kernels.cuh): persistent CTAs, two CTAs per SM (the sm_90 64-token tile runs that way too), weight
// blocks streamed through a ring (L2-resident after the first tile) while the tile's A operand stays in registers for every block, gate | proj
// rows interleaved so one thread owns both factors of an output, and a transposing staging buffer so the channel-major plane store leaves in
// 128 B rows.  sm_80 substitutions: cp.async (+ cp.async.mbarrier.arrive) for TMA, mma.sync m16n8k16 with ldmatrix B fragments for wgmma, and
//   * two independent 4-warp CTAs per SM instead of producer / consumer warpgroups: one CTA's epilogue, LayerNorm and tile turnover run under the
//     other's MMAs (with mma.sync a warp's own MMAs and epilogue do not overlap, and a 9th warp would cap registers at 168);
//   * weight blocks stored granule-column-major ([16 B k-granule][64 rows]): the 8 rows of an ldmatrix are consecutive 16 B, conflict-free, and
//     every k-step's address is the previous one plus an immediate;
//   * the LayerNorm done once per element in shared memory (the warps would otherwise repeat it on their fragments);
//   * sigmoid(g) p = 0.5 p (1 + tanh(g / 2)) with the 0.5 folded into both weight rows on the host (exact in bf16): one MUFU and one FFMA per
//     output; the pair mask is an AND on the packed word.
//
// CTA = 4 warps (2 M-groups x 2 N-warps).  Tile = 128 consecutive tokens; warp (mg, nw) owns token rows [64 mg, 64 mg + 64) and, per weight block
// (64 rows = 32 output channels in 4 groups of 8 gate rows | 8 proj rows), the groups 2 nw and 2 nw + 1.  Planes have Np = L, so plane offset = t
// and one warp's 64 tokens of a channel are a single 128 B segment.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

// z tile rows (256 B): granule g of row r at g ^ ((r >> 1) & 7) -- the A-fragment ldmatrix reads rows {2 i + b} (token-pair order), which this
// maps to 8 distinct granules; the LayerNorm pass and the fill touch whole rows, where any in-row permutation is conflict-free
DEVI uint32_t swz_z(uint32_t row, uint32_t byte_in_row) {
  return row * 256 + ((((byte_in_row >> 4) ^ ((row >> 1) & 7u)) << 4) | (byte_in_row & 15u));
}

struct K1Params {
  const __nv_bfloat16* z;      // [T][128]
  const uint8_t* mask;         // [L] token mask or nullptr
  const __nv_bfloat16* w;      // packed [NSTEP][16 granules][64 rows][8], 0.5-scaled
  const float* gamma;          // [128]
  const float* beta;           // [128]
  __nv_bfloat16* ab;           // [2 CH][T]
  int T, L, num_tiles;
  float eps;
};

template <int CH_, int NST_ = 2, int MT_ = 4>
struct K1Cfg {
  static constexpr int CH = CH_, CZ = 128, MT = MT_, WROWS = 16 * MT_, BM = 2 * WROWS, NST = NST_;   // MT m16 tiles per warp, 2 M-groups
  static constexpr int NTHR = 128, MINB = 2;
  static constexpr int NSTEP = 4 * CH / 64;                // weight blocks per tile
  static constexpr int SLOT = 64 * CZ * 2;                 // 16 KB, [16 granules][64 rows][16 B]
  static constexpr int SMEM_Z = BM * CZ * 2;
  static constexpr int SMEM_W = NST * SLOT;
  static constexpr int STG_PITCH = 4 * (8 * MT + 4);      // [8 ch][WROWS tok] staging rows padded by 4 words: conflict-free, additive addresses
  static constexpr int STG_BUF = 8 * STG_PITCH;
  static constexpr int SMEM_STG = 4 * 2 * STG_BUF;         // per warp: one buffer per channel group of the step
  static constexpr int SMEM_GB = 2 * CZ * 4;
  static constexpr int SMEM_MASK = 2 * BM;                 // per-token pair mask x validity, double-buffered (current | next tile)
  static constexpr int NBAR = NST + 1;
  static constexpr int SMEM = SMEM_Z + SMEM_W + SMEM_STG + SMEM_GB + SMEM_MASK + NBAR * 8;
  static constexpr int LN_NP = BM / 16;                    // four-row LayerNorm passes per warp per tile
  static constexpr int LN_S0 = NSTEP / 2, LN_PER_STEP = (LN_NP + NSTEP - LN_S0 - 1) / (NSTEP - LN_S0);   // spread over the last half of the steps
  static constexpr int NGR = 16 * MT;                      // 16 B staging granules per channel group
  static_assert(BM <= NTHR, "mask table: one token per thread");
  static_assert(SMEM * MINB <= 166912 + 1024 - 2048, "sm_80 shared memory for two CTAs per SM");
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) k1_kernel(const K1Params p) {
  constexpr int CZ = G::CZ, NST = G::NST, NSTEP = G::NSTEP, BM = G::BM, NT = G::NTHR;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sZ = smem;
  uint8_t* sW = sZ + G::SMEM_Z;
  uint8_t* sStg = sW + G::SMEM_W;
  float* sG = reinterpret_cast<float*>(sStg + G::SMEM_STG);
  float* sB = sG + CZ;
  uint8_t* sMask = reinterpret_cast<uint8_t*>(sB + CZ);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sMask + G::SMEM_MASK);
  const uint32_t sZ_u = smem_u32(sZ), sW_u = smem_u32(sW);
  const uint32_t barW = smem_u32(bars), barZ = barW + 8 * NST;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n_iter = (p.num_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int total = n_iter * NSTEP;
  // LayerNorm affine in the pass's lane order: float4 slot k * 8 + lc holds (k < 4: gamma, else beta) of channels
  // (k & 2 ? 64 : 0) + 8 lc + 4 (k & 1) ..: eight lanes read eight consecutive 16 B
  for (int i = tid; i < 64; i += NT) {
    const int k = i >> 3, l = i & 7, c0 = ((k & 2) ? 64 : 0) + 8 * l + 4 * (k & 1);
    const float* src = (k & 4) ? p.beta : p.gamma;
    reinterpret_cast<float4*>(sG)[i] = make_float4(src[c0], src[c0 + 1], src[c0 + 2], src[c0 + 3]);
  }
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) mbar_init(barW + 8 * s, NT);
    mbar_init(barZ, NT);
  }
  __syncthreads();

  // ---- weight block blk -> slot s: every thread issues 8 of the 1024 granules (host packs each block granule-major: a straight 16 KB copy)
  const __nv_bfloat16* w_src = p.w + tid * 8;
  const uint32_t w_dst = sW_u + tid * 16;
  auto issue_w = [&](int blk, int s) {
    const __nv_bfloat16* src = w_src + (size_t)blk * 64 * CZ;
    const uint32_t dst = w_dst + s * G::SLOT;
#pragma unroll
    for (int i = 0; i < 8; ++i) cp_async16(dst + i * NT * 16, src + i * NT * 8);
    cp_async_mbar_arrive(barW + 8 * s);
  };
  auto load_z = [&](int tile) {                 // BM tokens x 256 B, BM / 8 granules per thread, zero-filled past T
    const int t0 = tile * BM;
#pragma unroll
    for (int i = 0; i < BM / 8; ++i) {
      const int c = tid + NT * i, row = c >> 4, gr = c & 15;
      const bool ok = t0 + row < p.T;
      cp_async16(sZ_u + swz_z(row, gr * 16), p.z + (size_t)(ok ? t0 + row : 0) * CZ + gr * 8, ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barZ);
  };
  if (n_iter > 0) load_z(blockIdx.x);
  for (int u = 0; u < NST - 1 && u < total; ++u) issue_w(u % NSTEP, u);

  const int mg = warp >> 1, nw = warp & 1;
  const int g8 = lane >> 2, q = lane & 3, odd = g8 & 1;
  const int lr = lane >> 3, lc = lane & 7;                  // LayerNorm pass: 8 lanes per row (granules lc, lc + 8), 4 rows per instruction
  const uint32_t stg_u = smem_u32(sStg) + warp * 2 * G::STG_BUF;
  // token order inside each m16 tile: fragment row r <-> token 2 (r % 8) + r / 8, so accumulator rows g8 and g8 + 8 are the ADJACENT tokens
  // 2 g8, 2 g8 + 1 and a thread's (c0, c2) / (c1, c3) are already the packed (tok, tok + 1) words of channels 2q / 2q + 1 (no shuffle)
  const uint32_t st_off = (2 * q) * G::STG_PITCH + g8 * 4;   // staging [8 ch][WROWS tok]: word (channel 2q, token pair 8 mt + g8)
  // B fragments of group gi (rows 16 gi .. 16 gi + 15 of the block): matrix mi = lane / 8 -> row (mi >> 1) * 8 + lane % 8, granule 2 ks + (mi & 1)
  const uint32_t b_off = ((lane >> 3) & 1) * 1024 + (32 * nw + ((lane >> 4) & 1) * 8 + (lane & 7)) * 16;

  // ---- LayerNorm, in place, of 4 rows (32 w + r0 + lr): 8 lanes per row, granules lc and lc + 8 (128 B per 8 lanes: conflict-free)
  // pair mask x validity of tile `tile`'s 128 tokens -> mask buffer mb (one token per thread)
  auto mask_table = [&](int tile, int mb) {
    if (tid >= BM) return;
    const int t = tile * BM + tid;
    bool m = t < p.T;
    if (m && p.mask != nullptr) { const int i = t / p.L, j = t - i * p.L; m = __ldg(p.mask + i) && __ldg(p.mask + j); }
    sMask[mb * BM + tid] = m ? 1 : 0;
  };
  // masked (or past-T) tokens are normalised to ZERO rows: their projections, hence both planes' values, are exactly 0 -- the epilogue carries
  // no mask arithmetic
  auto ln_pass = [&](int r0, int mb) {
    const int row = (BM / 4) * warp + r0 + lr;
    const bool keep = sMask[mb * BM + row] != 0;
    const uint32_t a0 = sZ_u + swz_z(row, lc * 16), a1 = sZ_u + swz_z(row, lc * 16 + 128);
    const uint4 v0 = lds128(a0), v1 = lds128(a1);
    const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float x[16];
#pragma unroll
    for (int e = 0; e < 8; ++e) { x[2 * e] = bf16lo(w[e]); x[2 * e + 1] = bf16hi(w[e]); }
    float sm = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) sm += x[e];
    sm += __shfl_xor_sync(0xffffffffu, sm, 1); sm += __shfl_xor_sync(0xffffffffu, sm, 2); sm += __shfl_xor_sync(0xffffffffu, sm, 4);
    const float mean = sm * (1.f / CZ);
    float sq = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) { x[e] -= mean; sq = fmaf(x[e], x[e], sq); }
    sq += __shfl_xor_sync(0xffffffffu, sq, 1); sq += __shfl_xor_sync(0xffffffffu, sq, 2); sq += __shfl_xor_sync(0xffffffffu, sq, 4);
    const float rstd = rsqrtf(sq * (1.f / CZ) + p.eps);
    uint32_t o[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {                // x[0..7] = channels 8 lc .., x[8..15] = channels 64 + 8 lc ..
      const float4 g = reinterpret_cast<const float4*>(sG)[e * 8 + lc], b = reinterpret_cast<const float4*>(sG)[(4 + e) * 8 + lc];
      o[2 * e] = pack_bf16(fmaf(x[4 * e] * rstd, g.x, b.x), fmaf(x[4 * e + 1] * rstd, g.y, b.y));
      o[2 * e + 1] = pack_bf16(fmaf(x[4 * e + 2] * rstd, g.z, b.z), fmaf(x[4 * e + 3] * rstd, g.w, b.w));
    }
    if (!keep) {
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = 0u;
    }
    sts128(a0, make_uint4(o[0], o[1], o[2], o[3]));
    sts128(a1, make_uint4(o[4], o[5], o[6], o[7]));
  };
  if (n_iter > 0) {                              // first tile: normalised up front; every later tile under the previous tile's steps
    mask_table(blockIdx.x, 0);
    bar_sync(1, NT);
    mbar_wait(barZ, 0);
#pragma unroll 1
    for (int r0 = 0; r0 < BM / 4; r0 += 4) ln_pass(r0, 0);
  }

  int u = 0, slot = 0;
  uint32_t ph = 0;
  for (int it = 0; it < n_iter; ++it) {
    const int tile = (int)blockIdx.x + it * (int)gridDim.x;
    const int t0 = tile * BM;
    const bool has_next = it + 1 < n_iter;
    bar_sync(1, NT);                             // the tile's rows are normalised (by every warp) and its mask is written
    // ---- A fragments: 64 rows x 128 channels (4 m16 tiles x 8 k16 steps)
    constexpr int MT = G::MT, WR = G::WROWS;
    uint32_t fa[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int row = WR * mg + 16 * mt + 2 * (lane & 7) + ((lane >> 3) & 1);
        ldsm_x4(fa[mt][ks], sZ_u + swz_z(row, (2 * ks + (lane >> 4)) * 16));
      }
    // ---- bf16-half masks of the packed words this lane stores: tokens (g8 & ~1) + {0, 1} of each (mt, h)
    bar_sync(1, NT);                             // every warp has its fragments and mask: the tile buffers take the next tile
    if (has_next) { load_z(tile + (int)gridDim.x); mask_table(tile + (int)gridDim.x, (it + 1) & 1); }   // read by the spread LN (after step barriers)
    __nv_bfloat16* gout = p.ab + (size_t)(16 * nw) * p.T + t0 + WR * mg;
    // ---- weight blocks, software-pipelined inside the warp: a step's two channel groups alternate with the other group's epilogue,
    //      [MMA(s, g0) | EPI(s - 1, g1)] [MMA(s, g1) | EPI(s, g0)], interleaved k-step by k-step so the HMMA stream and the epilogue's
    //      MUFU / shuffle / store chain are independent instructions of one block (the same 64 accumulator registers as unpipelined)
    float acc0[MT][2][4], acc1[MT][2][4];       // [mt][gate, proj][4] of group 0 / group 1
    auto zero = [&](float (&ac)[MT][2][4]) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) ac[mt][n][e] = 0.f;
    };
    auto mma_ks = [&](float (&ac)[MT][2][4], uint32_t wb, int ks) {
      uint32_t b[4];
      ldsm_x4(b, wb + ks * 2048);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) { mma16816(ac[mt][0], fa[mt][ks], b[0], b[1]); mma16816(ac[mt][1], fa[mt][ks], b[2], b[3]); }
    };
    // epilogue chunk mt: v = p' (1 + tanh g') of tokens (2 g8, 2 g8 + 1) x channels (2q, 2q + 1) -> two packed words -> staging
    auto epi_chunk = [&](float (&ac)[MT][2][4], int gi, int mt) {
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) v[e] = fmaf(ac[mt][1][e], tanh_approx(ac[mt][0][e]), ac[mt][1][e]);
      const uint32_t a = stg_u + gi * G::STG_BUF + st_off + mt * 32;
      sts32(a, pack_bf16(v[0], v[2]));                    // channel 2q
      sts32(a + G::STG_PITCH, pack_bf16(v[1], v[3]));     // channel 2q + 1
    };
    // one group's 8 k-steps of MMAs (optional) with a group's 4 epilogue chunks interleaved (one per two k-steps)
    auto group = [&](float (&acm)[MT][2][4], uint32_t wb, bool do_mma, float (&ace)[MT][2][4], int gi, bool do_epi) {
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        if (do_mma) mma_ks(acm, wb, ks);
        if (do_epi && (ks & 1) && (ks >> 1) < MT) epi_chunk(ace, gi, ks >> 1);
      }
    };
    auto epi_store = [&](int gi, int stp) {    // staged [8 ch][WROWS tok] -> 16 B granules, (2 MT) per channel row
      __syncwarp();
#pragma unroll
      for (int k = 0; k < (G::NGR + 31) / 32; ++k) {
        const int idx = lane + 32 * k, ch = idx / (2 * MT), gk = idx % (2 * MT);
        if (idx < G::NGR) {
          const uint4 v = lds128(stg_u + gi * G::STG_BUF + ch * G::STG_PITCH + gk * 16);
          if (t0 + WR * mg + 8 * gk < p.T) stg128(gout + (size_t)(32 * stp + 8 * gi + ch) * p.T + 8 * gk, v);
        }
      }
      __syncwarp();
    };
    auto next_block = [&]() -> uint32_t {      // CTA barrier (every warp retired the previous block: its slot takes block u + NST - 1), then wait block u
      bar_sync(1, NT);
      if (u + NST - 1 < total) issue_w((u + NST - 1) % NSTEP, (slot + NST - 1) % NST);
      mbar_wait(barW + 8 * slot, ph);
      return sW_u + slot * G::SLOT + b_off;
    };
    auto advance = [&]() { ++u; if (++slot == NST) { slot = 0; ph ^= 1u; } };
    uint32_t wb = next_block();
    zero(acc0);
    group(acc0, wb, true, acc1, 1, false);
#pragma unroll 1
    for (int step = 0; step < NSTEP; ++step) {
      zero(acc1);
      group(acc1, wb + 256, true, acc0, 0, true);
      epi_store(0, step);
      advance();
      if (step + 1 < NSTEP) {
        wb = next_block();
        zero(acc0);
        group(acc0, wb, true, acc1, 1, true);
      } else {
        group(acc0, wb, false, acc1, 1, true);
      }
      epi_store(1, step);
      if (has_next && step >= G::LN_S0) {       // the next tile's z (issued at this tile's start) -> LayerNorm, spread over the second half
        if (step == G::LN_S0) mbar_wait(barZ, (it + 1) & 1);
#pragma unroll
        for (int k = 0; k < G::LN_PER_STEP; ++k) {
          const int pass = (step - G::LN_S0) * G::LN_PER_STEP + k;
          if (pass < G::LN_NP) ln_pass(4 * pass, (it + 1) & 1);
        }
      }
    }
  }
  cp_async_wait<0>();
}

}  // namespace a100
