// contract_sm80.cuh -- TriMul contraction on the planes, A100 / sm_80: per channel c, bf16 x bf16 -> fp32 accumulate -> bf16
//
//   outgoing (NT):  X_c = A_c B_c^T     X[c, i, j] = sum_k a[c, i, k] b[c, j, k]     (operand rows = tokens, k contiguous)
//   incoming (TN):  X_c = A_c^T B_c     X[c, i, j] = sum_k a[c, k, i] b[c, k, j]     (operand rows = k, tokens contiguous)
//
// One launch serves every channel; a bidirectional TriMul's two halves (NT on channels [0, h), TN on [h, CH)) run in the same grid, so neither
// half has a tail of its own.  CTA tile 128 x 128, k-step 32, cp.async multistage ring; 4 warps with 64 x 64 warp tiles (ldmatrix A / B
// fragments, mma.sync m16n8k16), two CTAs per SM.  The grid walks the 9 / 36 tiles of one channel consecutively, so a channel's two planes
// are read from DRAM once and reused from L2 by its tiles.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct ContractParams {
  const __nv_bfloat16* a;      // [CH][L][L]
  const __nv_bfloat16* b;      // [CH][L][L]
  __nv_bfloat16* x;            // [CH][L][L]
  int L, CH, h;                // channels [0, h) NT, [h, CH) TN
};

template <int ST_ = 4>
struct ContractCfg {
  static constexpr int BM = 128, BN = 128, BK = 32, ST = ST_, NTHR = 128;
  static constexpr int TILE_A = BM * BK * 2, TILE_B = BN * BK * 2;          // 8 KB each
  static constexpr int STAGE = TILE_A + TILE_B;
  static constexpr int SMEM = ST * STAGE;
  static_assert(2 * SMEM <= 166912, "two CTAs per SM");
};

// K-major tile: 128 rows x 32 k (64 B rows), granule c (0..3) of row r at c ^ ((r >> 1) & 3)
DEVI uint32_t kmaj(uint32_t row, uint32_t c) { return row * 64 + ((c ^ ((row >> 1) & 3u)) << 4); }
// MN-major tile: 32 k rows x 128 tokens (256 B rows), granule c (0..15) of row r at c ^ (r & 7)
DEVI uint32_t mmaj(uint32_t row, uint32_t c) { return row * 256 + ((c ^ (row & 7u)) << 4); }

template <class G, bool TN>
DEVI void contract_tile(const ContractParams& p, int c, int ti, int tj, uint8_t* smem) {
  constexpr int BM = G::BM, BN = G::BN, BK = G::BK, ST = G::ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp >> 1, wn = warp & 1;
  const size_t plane = (size_t)p.L * p.L;
  const __nv_bfloat16* A = p.a + (size_t)c * plane;
  const __nv_bfloat16* B = p.b + (size_t)c * plane;
  const int i0 = ti * BM, j0 = tj * BN, nk = p.L / BK;
  const uint32_t s0 = smem_u32(smem);

  auto load_stage = [&](int kt, int st) {
    const uint32_t sa = s0 + st * G::STAGE, sb = sa + G::TILE_A;
    const int k0 = kt * BK;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int g = tid + 128 * e;                                   // 512 granules per operand tile
      if (!TN) {                                                      // rows = tokens (128), 4 granules of k each
        const int row = g >> 2, cc = g & 3;
        cp_async16(sa + kmaj(row, cc), A + (size_t)(i0 + row) * p.L + k0 + cc * 8);
        cp_async16(sb + kmaj(row, cc), B + (size_t)(j0 + row) * p.L + k0 + cc * 8);
      } else {                                                        // rows = k (32), 16 granules of tokens each
        const int row = g >> 4, cc = g & 15;
        cp_async16(sa + mmaj(row, cc), A + (size_t)(k0 + row) * p.L + i0 + cc * 8);
        cp_async16(sb + mmaj(row, cc), B + (size_t)(k0 + row) * p.L + j0 + cc * 8);
      }
    }
  };

  float acc[4][8][4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;

#pragma unroll
  for (int st = 0; st < ST - 1; ++st) {
    if (st < nk) load_stage(st, st);
    cp_async_commit();
  }
  const int mi = lane >> 3, r8 = lane & 7;
#pragma unroll 1
  for (int kt = 0; kt < nk; ++kt) {
    cp_async_wait<ST - 2>();
    __syncthreads();                                                 // stage kt landed for everyone; stage kt - 1 is free
    if (kt + ST - 1 < nk) load_stage(kt + ST - 1, (kt + ST - 1) % ST);
    cp_async_commit();
    const uint32_t sa = s0 + (kt % ST) * G::STAGE, sb = sa + G::TILE_A;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      uint32_t af[4][4], bf[4][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        const int m0 = 64 * wm + 16 * mt;
        if (!TN) ldsm_x4(af[mt], sa + kmaj(m0 + r8 + (mi & 1) * 8, 2 * kk + (mi >> 1)));
        else ldsm_x4_t(af[mt], sa + mmaj(16 * kk + (mi >> 1) * 8 + r8, (m0 >> 3) + (mi & 1)));
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {                                // pairs of n8 tiles
        const int n0 = 64 * wn + 16 * np;
        if (!TN) ldsm_x4(bf[np], sb + kmaj(n0 + r8 + (mi >> 1) * 8, 2 * kk + (mi & 1)));
        else ldsm_x4_t(bf[np], sb + mmaj(16 * kk + (mi & 1) * 8 + r8, (n0 >> 3) + (mi >> 1)));
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          mma16816(acc[mt][2 * np], af[mt], bf[np][0], bf[np][1]);
          mma16816(acc[mt][2 * np + 1], af[mt], bf[np][2], bf[np][3]);
        }
    }
  }
  cp_async_wait<0>();
  // epilogue: bf16 channel-pair words straight to the plane rows (row g8 / g8 + 8, columns 2q, 2q + 1 of each n8 tile)
  __nv_bfloat16* X = p.x + (size_t)c * plane;
  const int g8 = lane >> 2, q = lane & 3;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      __nv_bfloat16* row = X + (size_t)(i0 + 64 * wm + 16 * mt + g8 + 8 * h) * p.L + j0 + 64 * wn + 2 * q;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) stg32(row + 8 * nt, pack_bf16(acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1]));
    }
  __syncthreads();                                                   // the ring is reused by the next tile of this CTA
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, 2) contract_kernel(const ContractParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tpc = (p.L / G::BM) * (p.L / G::BN);                    // tiles per channel
  const int total = tpc * p.CH;
  for (int t = blockIdx.x; t < total; t += gridDim.x) {
    const int c = t / tpc, r = t - c * tpc, nj = p.L / G::BN;
    const int ti = r / nj, tj = r - ti * nj;
    if (c < p.h) contract_tile<G, false>(p, c, ti, tj, smem);
    else contract_tile<G, true>(p, c, ti, tj, smem);
  }
}

}  // namespace a100
