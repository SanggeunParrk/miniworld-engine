// k1_sm80.cuh -- TriMul K1 (input LayerNorm + gated projections + pair mask -> channel-major planes), A100 / sm_80.
//
//   ab[oc, t] = bf16( sigmoid(LN_in(z)[t] . Wg[oc]) * (LN_in(z)[t] . Wp[oc]) * m_i m_j ),  t = i L + j,  oc in [0, 2 CH)  (a | b planes)
//
// Ideas carried over from the sm_90 K1 (tmn_kernels.cuh): persistent CTAs, a producer that streams the weight blocks through an mbarrier ring
// (L2-resident after the first tile) while the consumers hold the tile's A operand in registers for every weight block, gate | proj rows
// interleaved so one thread owns both factors of an output, and a transposing staging buffer so the channel-major plane store is 128 B rows.
// sm_80 substitutions: cp.async (+ cp.async.mbarrier.arrive) for TMA, mma.sync m16n8k16 with ldmatrix B fragments for wgmma, a dedicated
// producer WARP instead of a warpgroup, and the LayerNorm done once per element in shared memory (8 consumer warps would otherwise repeat it).
//
// CTA = 8 consumer warps (2 M-groups x 4 N-warps) + 1 producer warp.  Tile = 128 consecutive tokens; consumer warp (mg, nw) owns token rows
// [64 mg, 64 mg + 64) and, per weight block (64 rows = 32 output channels), the 16 rows [16 nw, 16 nw + 16): gate rows of 8 channels, then
// their 8 proj rows.  Planes have Np = L, so plane offset = t and a tile's 64 tokens of one channel are one 128 B segment.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct K1Params {
  const __nv_bfloat16* z;      // [T][128]
  const uint8_t* mask;         // [L] token mask or nullptr
  const __nv_bfloat16* w;      // packed [NSTEP][64][128]
  const float* gamma;          // [128]
  const float* beta;           // [128]
  __nv_bfloat16* ab;           // [2 CH][T]
  int T, L, num_tiles;
  float eps;
};

template <int CH_, int NST_ = 6>
struct K1Cfg {
  static constexpr int CH = CH_, CZ = 128, BM = 128, NST = NST_;
  static constexpr int NCW = 8, NTHR = 32 * NCW;
  static constexpr int NSTEP = 4 * CH / 64;                // weight blocks per tile
  static constexpr int SLOT = 64 * CZ * 2;                 // 16 KB
  static constexpr int SMEM_Z = BM * CZ * 2;               // 32 KB
  static constexpr int SMEM_W = NST * SLOT;
  static constexpr int SMEM_STG = NCW * 2 * 1024;          // per warp 2 x [8 ch][64 tok] bf16
  static constexpr int SMEM_GB = 2 * CZ * 4;
  static constexpr int NBAR = NST + 1;
  static constexpr int SMEM = SMEM_Z + SMEM_W + SMEM_STG + SMEM_GB + NBAR * 8;
  static_assert(SMEM <= 166912, "sm_80 dynamic shared memory");
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
  uint64_t* bars = reinterpret_cast<uint64_t*>(sB + CZ);
  const uint32_t sZ_u = smem_u32(sZ), sW_u = smem_u32(sW);
  const uint32_t barW_full = smem_u32(bars), barZ = barW_full + 8 * NST;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n_iter = (p.num_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  for (int i = tid; i < CZ; i += G::NTHR) { sG[i] = p.gamma[i]; sB[i] = p.beta[i]; }
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) mbar_init(barW_full + 8 * s, G::NTHR);
    mbar_init(barZ, G::NTHR);
  }
  __syncthreads();

  // ------------------------------------------------------------------ weight ring: every thread issues 4 of the block's 1024 granules; slot
  // (u - 1) % NST is refilled with block u + NST - 1 after the step-u CTA barrier (every warp has retired step u - 1)
  const int total = n_iter * NSTEP;
  auto issue_w = [&](int u) {
    const int s = u % NST;
    const __nv_bfloat16* src = p.w + (size_t)(u % NSTEP) * 64 * CZ;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = tid + 256 * i, row = c >> 4, g = c & 15;
      cp_async16(sW_u + s * G::SLOT + swz<256>(row, g * 16), src + row * CZ + g * 8);
    }
    cp_async_mbar_arrive(barW_full + 8 * s);
  };
  const int mg = warp >> 2, nw = warp & 3;
  const int g8 = lane >> 2, q = lane & 3;
  auto load_z = [&](int tile) {                 // 128 tokens x 256 B = 2048 granules, 8 per consumer thread, zero-filled past T
    const int t0 = tile * BM;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = tid + 256 * i, row = c >> 4, g = c & 15;
      const bool ok = t0 + row < p.T;
      cp_async16(sZ_u + swz<256>(row, g * 16), p.z + (size_t)(ok ? t0 + row : 0) * CZ + g * 8, ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barZ);
  };
  if (n_iter > 0) load_z(blockIdx.x);
  for (int u = 0; u < NST - 1 && u < total; ++u) issue_w(u);
  // this lane's LayerNorm columns (4 consecutive channels) for the smem pass
  float gl[4], bl[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) { gl[e] = sG[4 * lane + e]; bl[e] = sB[4 * lane + e]; }
  const uint32_t stg_u = smem_u32(sStg) + warp * 2048;

  for (int it = 0; it < n_iter; ++it) {
    const int tile = (int)blockIdx.x + it * (int)gridDim.x;
    const int t0 = tile * BM;
    mbar_wait(barZ, it & 1);
    // ---- LayerNorm in place: warp w normalises rows 16 w .. 16 w + 15; lane holds channels 4 lane .. 4 lane + 3
#pragma unroll 2
    for (int r = 0; r < 16; ++r) {
      const int row = 16 * warp + r;
      const uint32_t a = sZ_u + swz<256>(row, lane * 8);
      const uint2 v = lds64(a);
      float x[4] = {bf16lo(v.x), bf16hi(v.x), bf16lo(v.y), bf16hi(v.y)};
      const float mean = warp_sum(x[0] + x[1] + x[2] + x[3]) * (1.f / CZ);
      float sq = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) { x[e] -= mean; sq = fmaf(x[e], x[e], sq); }
      const float rstd = rsqrtf(warp_sum(sq) * (1.f / CZ) + p.eps);
      uint2 o;
      o.x = pack_bf16(fmaf(x[0] * rstd, gl[0], bl[0]), fmaf(x[1] * rstd, gl[1], bl[1]));
      o.y = pack_bf16(fmaf(x[2] * rstd, gl[2], bl[2]), fmaf(x[3] * rstd, gl[3], bl[3]));
      sts64(a, o);
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
    bar_sync(1, 256);                          // every consumer has its fragments: the tile buffer takes the next tile
    if (it + 1 < n_iter) load_z(tile + (int)gridDim.x);
    // ---- per-row validity x pair mask for this thread's 8 accumulator rows
    float mrow[4][2];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int t = t0 + 64 * mg + 16 * mt + g8 + 8 * h;
        float m = t < p.T ? 1.f : 0.f;
        if (p.mask != nullptr && t < p.T) {
          const int i = t / p.L, j = t - i * p.L;
          m = (__ldg(p.mask + i) && __ldg(p.mask + j)) ? 1.f : 0.f;
        }
        mrow[mt][h] = m;
      }
    // ---- weight blocks
#pragma unroll 1
    for (int step = 0; step < NSTEP; ++step) {
      const int u = it * NSTEP + step, s = u % NST;
      bar_sync(1, 256);
      if (u + NST - 1 < total) issue_w(u + NST - 1);
      mbar_wait(barW_full + 8 * s, (u / NST) & 1);
      float acc[4][2][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[mt][n][e] = 0.f;
      const uint32_t wb = sW_u + s * G::SLOT;
      const int brow = 16 * nw + ((lane >> 4) & 1) * 8 + (lane & 7);
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b[4];
        ldsm_x4(b, wb + swz<256>(brow, (2 * ks + ((lane >> 3) & 1)) * 16));
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) { mma16816(acc[mt][0], fa[mt][ks], b[0], b[1]); mma16816(acc[mt][1], fa[mt][ks], b[2], b[3]); }
      }
      // ---- epilogue: v = sigmoid(g) p m -> token pairs per channel (shfl with lane ^ 4) -> [8 ch][64 tok] staging -> 128 B plane rows
      const uint32_t sb = stg_u + (step & 1) * 1024;
      const int odd = g8 & 1;
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float v0 = sigmoid(acc[mt][0][2 * h]) * acc[mt][1][2 * h] * mrow[mt][h];          // channel 2q
          const float v1 = sigmoid(acc[mt][0][2 * h + 1]) * acc[mt][1][2 * h + 1] * mrow[mt][h];  // channel 2q + 1
          const float recv = __shfl_xor_sync(0xffffffffu, odd ? v0 : v1, 4);
          const uint32_t word = odd ? pack_bf16(recv, v1) : pack_bf16(v0, recv);
          const int ch = 2 * q + odd;
          const int wi = (8 * mt + 4 * h + (g8 >> 1)) ^ (ch << 2);
          sts32(sb + ch * 128 + wi * 4, word);
        }
      __syncwarp();
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int c = lane + 32 * k, ch = c >> 3, gk = c & 7;
        const uint4 v = lds128(sb + ch * 128 + ((gk ^ ch) << 4));
        const int tok = t0 + 64 * mg + 8 * gk;
        const int oc = 32 * step + 8 * nw + ch;
        if (tok < p.T) stg128(p.ab + (size_t)oc * p.T + tok, v);
      }
    }
  }
}

}  // namespace a100
