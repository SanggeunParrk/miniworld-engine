// k1_sm80.cuh -- TriMul K1 (input LayerNorm + gated projections + pair mask -> channel-major planes), A100 / sm_80.
//
//   ab[oc, t] = bf16( sigmoid(LN_in(z)[t] . Wg[oc]) * (LN_in(z)[t] . Wp[oc]) * m_i m_j ),  t = i L + j,  oc in [0, 2 CH)  (a | b planes)
//
// Ideas carried over from the sm_90 K1 (tmn_kernels.cuh): persistent CTAs, weight blocks streamed through an mbarrier ring (L2-resident after the
// first tile) while the consumers hold the tile's A operand in registers for every block, gate | proj rows interleaved so one thread owns both
// factors of an output, and a transposing staging buffer so the channel-major plane store leaves in 128 B rows.
// sm_80 substitutions: cp.async (+ cp.async.mbarrier.arrive) for TMA, mma.sync m16n8k16 with ldmatrix B fragments for wgmma, and
//   * no producer warp (a ninth warp would cap every thread at 168 registers: 16 K registers per SM sub-partition): the LAST warp to retire a
//     ring slot refills it (shared counter), so the warps never meet at a per-block CTA barrier and their MMA / epilogue phases drift apart;
//   * weight blocks stored granule-column-major ([16 B k-granule][64 rows]): the 8 rows of an ldmatrix are consecutive 16 B, conflict-free,
//     and every k-step's address is the previous one plus an immediate (no swizzle arithmetic in the hot loop);
//   * the LayerNorm done once per element in shared memory (eight warps would otherwise repeat it on their fragments);
//   * sigmoid(g) p = 0.5 p (1 + tanh(g / 2)) with the 0.5 folded into both weight rows on the host (exact in bf16): one MUFU and one FFMA per
//     output; the pair mask is an AND on the packed word.
//
// CTA = 8 warps (2 M-groups x 4 N-warps).  Tile = 128 consecutive tokens; warp (mg, nw) owns token rows [64 mg, 64 mg + 64) and, per weight block
// (64 rows = 32 output channels), the 16 rows [16 nw, 16 nw + 16): 8 gate rows then the same 8 channels' proj rows.  Planes have Np = L, so
// plane offset = t and one warp's 64 tokens of a channel are a single 128 B segment.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct K1Params {
  const __nv_bfloat16* z;      // [T][128]
  const uint8_t* mask;         // [L] token mask or nullptr
  const __nv_bfloat16* w;      // packed [NSTEP][64][128], 0.5-scaled
  const float* gamma;          // [128]
  const float* beta;           // [128]
  __nv_bfloat16* ab;           // [2 CH][T]
  int T, L, num_tiles;
  float eps;
};

template <int CH_, int NST_ = 6>
struct K1Cfg {
  static constexpr int CH = CH_, CZ = 128, BM = 128, NST = NST_;
  static constexpr int NTHR = 256;
  static constexpr int NSTEP = 4 * CH / 64;                // weight blocks per tile
  static constexpr int SLOT = 64 * CZ * 2;                 // 16 KB, [16 granules][64 rows][16 B]
  static constexpr int SMEM_Z = BM * CZ * 2;               // 32 KB
  static constexpr int SMEM_W = NST * SLOT;
  static constexpr int STG_PITCH = 144;                    // [8 ch][64 tok] staging rows padded to 36 words: conflict-free, additive addresses
  static constexpr int STG_BUF = 8 * STG_PITCH;
  static constexpr int SMEM_STG = 8 * 2 * STG_BUF;
  static constexpr int SMEM_GB = 2 * CZ * 4;
  static constexpr int SMEM_CNT = 64;                      // per-slot retire counters
  static constexpr int NBAR = NST + 1;
  static constexpr int SMEM = SMEM_Z + SMEM_W + SMEM_STG + SMEM_GB + SMEM_CNT + NBAR * 8;
  static_assert(SMEM <= 166912, "sm_80 dynamic shared memory");
  static_assert(NST <= 16, "counters");
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) k1_kernel(const K1Params p) {
  constexpr int CZ = G::CZ, NST = G::NST, NSTEP = G::NSTEP, BM = G::BM;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sZ = smem;
  uint8_t* sW = sZ + G::SMEM_Z;
  uint8_t* sStg = sW + G::SMEM_W;
  float* sG = reinterpret_cast<float*>(sStg + G::SMEM_STG);
  float* sB = sG + CZ;
  int* sCnt = reinterpret_cast<int*>(sB + CZ);
  uint64_t* bars = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sCnt) + G::SMEM_CNT);
  const uint32_t sZ_u = smem_u32(sZ), sW_u = smem_u32(sW);
  const uint32_t barW = smem_u32(bars), barZ = barW + 8 * NST;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n_iter = (p.num_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int total = n_iter * NSTEP;
  for (int i = tid; i < CZ; i += G::NTHR) { sG[i] = p.gamma[i]; sB[i] = p.beta[i]; }
  if (tid < NST) sCnt[tid] = 0;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) mbar_init(barW + 8 * s, 32);   // the refilling warp's 32 lanes
    mbar_init(barZ, G::NTHR);
  }
  __syncthreads();

  // ---- weight block blk -> slot s, issued by one whole warp (32 granules per lane)
  auto issue_w = [&](int blk, int s) {
    const __nv_bfloat16* src = p.w + (size_t)blk * 64 * CZ;
    const uint32_t dst = sW_u + s * G::SLOT;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int c = lane + 32 * i, row = c >> 4, gr = c & 15;       // global row-major [64][128]; smem [gr][row][16 B]
      cp_async16(dst + gr * 1024 + row * 16, src + row * CZ + gr * 8);
    }
    cp_async_mbar_arrive(barW + 8 * s);
  };
  if (warp == 0)
    for (int u = 0; u < NST && u < total; ++u) issue_w(u % NSTEP, u);

  const int mg = warp >> 2, nw = warp & 3;
  const int g8 = lane >> 2, q = lane & 3, odd = g8 & 1;
  auto load_z = [&](int tile) {                 // 128 tokens x 256 B = 2048 granules, 8 per thread, zero-filled past T
    const int t0 = tile * BM;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = tid + 256 * i, row = c >> 4, gr = c & 15;
      const bool ok = t0 + row < p.T;
      cp_async16(sZ_u + swz<256>(row, gr * 16), p.z + (size_t)(ok ? t0 + row : 0) * CZ + gr * 8, ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barZ);
  };
  if (n_iter > 0) load_z(blockIdx.x);
  // LayerNorm pass geometry: 8 lanes per row (16 channels each, two 16 B granules), 4 rows per warp instruction
  const int lr = lane >> 3, lc = lane & 7;
  float gl[16], bl[16];
#pragma unroll
  for (int e = 0; e < 16; ++e) { gl[e] = sG[16 * lc + e]; bl[e] = sB[16 * lc + e]; }
  // staging: this lane writes channel ch = 2q + odd, token pair (g8 >> 1) + 4 (2 mt + h); reads 16 B granules (ch, gk)
  const uint32_t stg_u = smem_u32(sStg) + warp * 2 * G::STG_BUF;
  const uint32_t st_off = (2 * q + odd) * G::STG_PITCH + (g8 >> 1) * 4;
  const uint32_t sel = odd ? 0x3276u : 0x5410u;             // even: (my v0, partner v0); odd: (partner v1, my v1)
  // B fragment address inside a slot: matrix mi = lane / 8 -> rows (mi >> 1) * 8 + lane % 8 of this warp's 16, granule 2 ks + (mi & 1)
  const uint32_t b_off = ((lane >> 3) & 1) * 1024 + (16 * nw + ((lane >> 4) & 1) * 8 + (lane & 7)) * 16;

  int u = 0, slot = 0;
  uint32_t ph = 0;
  for (int it = 0; it < n_iter; ++it) {
    const int tile = (int)blockIdx.x + it * (int)gridDim.x;
    const int t0 = tile * BM;
    mbar_wait(barZ, it & 1);
    // ---- LayerNorm in place (4 rows per iteration, 8 lanes per row)
#pragma unroll 1
    for (int r0 = 0; r0 < 16; r0 += 4) {
      const int row = 16 * warp + r0 + lr;
      const uint32_t a0 = sZ_u + swz<256>(row, lc * 32), a1 = sZ_u + swz<256>(row, lc * 32 + 16);
      const uint4 v0 = lds128(a0), v1 = lds128(a1);
      const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
      float x[16];
#pragma unroll
      for (int e = 0; e < 8; ++e) { x[2 * e] = bf16lo(w[e]); x[2 * e + 1] = bf16hi(w[e]); }
      float s = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) s += x[e];
      s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2); s += __shfl_xor_sync(0xffffffffu, s, 4);
      const float mean = s * (1.f / CZ);
      float sq = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) { x[e] -= mean; sq = fmaf(x[e], x[e], sq); }
      sq += __shfl_xor_sync(0xffffffffu, sq, 1); sq += __shfl_xor_sync(0xffffffffu, sq, 2); sq += __shfl_xor_sync(0xffffffffu, sq, 4);
      const float rstd = rsqrtf(sq * (1.f / CZ) + p.eps);
      uint32_t o[8];
#pragma unroll
      for (int e = 0; e < 8; ++e)
        o[e] = pack_bf16(fmaf(x[2 * e] * rstd, gl[2 * e], bl[2 * e]), fmaf(x[2 * e + 1] * rstd, gl[2 * e + 1], bl[2 * e + 1]));
      sts128(a0, make_uint4(o[0], o[1], o[2], o[3]));
      sts128(a1, make_uint4(o[4], o[5], o[6], o[7]));
    }
    bar_sync(1, 256);
    // ---- A fragments: 64 rows x 128 channels (4 m16 tiles x 8 k16 steps)
    uint32_t fa[4][8][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int row = 64 * mg + 16 * mt + (lane & 7) + ((lane >> 3) & 1) * 8;
        ldsm_x4(fa[mt][ks], sZ_u + swz<256>(row, (2 * ks + (lane >> 4)) * 16));
      }
    bar_sync(1, 256);                          // every warp has its fragments: the tile buffer takes the next tile
    if (it + 1 < n_iter) load_z(tile + (int)gridDim.x);
    // ---- pair mask x validity of the accumulator rows, as bf16-half masks of the packed words this lane stores
    uint32_t mbits[4][2];
    {
      const int i0 = t0 / p.L, j0 = t0 - i0 * p.L;
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t bits = 0;
#pragma unroll
          for (int e = 0; e < 2; ++e) {         // the word's two tokens: (g8 & ~1) + e
            const int off = 64 * mg + 16 * mt + 8 * h + (g8 & ~1) + e;
            bool m = t0 + off < p.T;
            if (m && p.mask != nullptr) {
              int j = j0 + off, i = i0;
              while (j >= p.L) { j -= p.L; ++i; }
              m = __ldg(p.mask + i) && __ldg(p.mask + j);
            }
            if (m) bits |= e ? 0xffff0000u : 0x0000ffffu;
          }
          mbits[mt][h] = bits;
        }
    }
    __nv_bfloat16* gout = p.ab + (size_t)(8 * nw + (lane >> 3)) * p.T + t0 + 64 * mg + 8 * (lane & 7);
    const bool st_ok = t0 + 64 * mg + 8 * (lane & 7) < p.T;
    // ---- weight blocks
#pragma unroll 1
    for (int step = 0; step < NSTEP; ++step) {
      mbar_wait(barW + 8 * slot, ph);
      float acc[4][2][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[mt][n][e] = 0.f;
      const uint32_t wb = sW_u + slot * G::SLOT + b_off;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b[4];
        ldsm_x4(b, wb + ks * 2048);
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) { mma16816(acc[mt][0], fa[mt][ks], b[0], b[1]); mma16816(acc[mt][1], fa[mt][ks], b[2], b[3]); }
      }
      // ---- retire the slot; the last of the 8 warps refills it with block u + NST
      __syncwarp();
      int old = 0;
      if (lane == 0) { __threadfence_block(); old = atomicAdd(sCnt + slot, 1); }
      old = __shfl_sync(0xffffffffu, old, 0);
      if ((old & 7) == 7 && u + NST < total) { __threadfence_block(); issue_w((u + NST) % NSTEP, slot); }
      // ---- epilogue: v = p' (1 + tanh g') -> packed (tok, tok + 1) per channel via shfl + prmt -> staging -> 128 B plane rows
      const uint32_t sb = stg_u + (step & 1) * G::STG_BUF;
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float p0 = acc[mt][1][2 * h], p1 = acc[mt][1][2 * h + 1];
          const float v0 = fmaf(p0, tanh_approx(acc[mt][0][2 * h]), p0);        // channel 2q, token g8 + 8h
          const float v1 = fmaf(p1, tanh_approx(acc[mt][0][2 * h + 1]), p1);    // channel 2q + 1
          const uint32_t mine = pack_bf16(v0, v1);
          const uint32_t other = __shfl_xor_sync(0xffffffffu, mine, 4);
          sts32(sb + st_off + (8 * mt + 4 * h) * 4, __byte_perm(mine, other, sel) & mbits[mt][h]);
        }
      __syncwarp();
#pragma unroll
      for (int k = 0; k < 2; ++k) {             // granule (ch = lane / 8 + 4 k, tokens 8 (lane % 8) ..)
        const uint4 v = lds128(sb + ((lane >> 3) + 4 * k) * G::STG_PITCH + (lane & 7) * 16);
        if (st_ok) stg128(gout + (size_t)(32 * step + 4 * k) * p.T, v);
      }
      ++u;
      if (++slot == NST) { slot = 0; ph ^= 1u; }
    }
  }
  cp_async_wait<0>();
}

}  // namespace a100
