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
  static_assert(ST >= 3, "the ring needs a stage in flight past the one being read");
};

// K-major tile: 128 rows x 32 k (64 B rows), granule c (0..3) of row r at c ^ ((r >> 1) & 3)
DEVI uint32_t kmaj(uint32_t row, uint32_t c) { return row * 64 + ((c ^ ((row >> 1) & 3u)) << 4); }
// MN-major tile: 32 k rows x 128 tokens (256 B rows), granule c (0..15) of row r at c ^ (r & 7)
DEVI uint32_t mmaj(uint32_t row, uint32_t c) { return row * 256 + ((c ^ (row & 7u)) << 4); }

// C = P Q per channel, C [L][L] row-major.  AM: P is M-major (P[m][k] stored at [k][m]); BMN: Q is N-major (Q[k][n] stored at [k][n]).
//   NT = <false, false>, TN = <true, true>, NN = <false, true>
template <class G, bool AM, bool BMN>
DEVI void contract_tile_g(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Xo, int L,
                          int ti, int tj, uint8_t* smem) {
  constexpr int BM = G::BM, BN = G::BN, BK = G::BK, ST = G::ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp >> 1, wn = warp & 1;
  const int i0 = ti * BM, j0 = tj * BN, nk = L / BK;
  const uint32_t s0 = smem_u32(smem);

  auto load_stage = [&](int kt, int st) {
    const uint32_t sa = s0 + st * G::STAGE, sb = sa + G::TILE_A;
    const int k0 = kt * BK;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int g = tid + 128 * e;                                   // 512 granules per operand tile
      const int krow = g >> 2, kc = g & 3;                            // K-major: rows = tokens (128), 4 granules of k each
      const int mrow = g >> 4, mc = g & 15;                           // MN-major: rows = k (32), 16 granules of tokens each
      if (!AM) cp_async16(sa + kmaj(krow, kc), A + (size_t)(i0 + krow) * L + k0 + kc * 8);
      else cp_async16(sa + mmaj(mrow, mc), A + (size_t)(k0 + mrow) * L + i0 + mc * 8);
      if (!BMN) cp_async16(sb + kmaj(krow, kc), B + (size_t)(j0 + krow) * L + k0 + kc * 8);
      else cp_async16(sb + mmaj(mrow, mc), B + (size_t)(k0 + mrow) * L + j0 + mc * 8);
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
  constexpr int KK = BK / 16;
  // fragments of (stage slot, k16 step) -> registers; double-buffered so the next step's ldmatrix run under this step's MMAs
  uint32_t af[2][4][4], bf[2][4][4];
  // lane-constant fragment offsets inside a stage (the swizzle terms depend only on the lane and the compile-time mt / np / kk)
  //   K-major (NT): row = 64 wm + 16 mt + r8 + 8 (mi & 1) [A] / 64 wn + 16 np + r8 + 8 (mi >> 1) [B]; (row >> 1) & 3 is lane-constant
  //   MN-major (TN): k row = 16 kk + 8 (mi >> 1) + r8 [A] / 16 kk + 8 (mi & 1) + r8 [B]; row & 7 = r8
  uint32_t aoff[4][KK], boff[4][KK];
#pragma unroll
  for (int t = 0; t < 4; ++t)
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      aoff[t][kk] = !AM ? kmaj(64 * wm + 16 * t + r8 + (mi & 1) * 8, 2 * kk + (mi >> 1))
                        : mmaj(16 * kk + (mi >> 1) * 8 + r8, 8 * wm + 2 * t + (mi & 1));
      boff[t][kk] = G::TILE_A + (!BMN ? kmaj(64 * wn + 16 * t + r8 + (mi >> 1) * 8, 2 * kk + (mi & 1))
                                     : mmaj(16 * kk + (mi & 1) * 8 + r8, 8 * wn + 2 * t + (mi >> 1)));
    }
  auto load_frags = [&](int buf, int slot, int kk) {
    const uint32_t sa = s0 + slot * G::STAGE;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      if (!AM) ldsm_x4(af[buf][mt], sa + aoff[mt][kk]); else ldsm_x4_t(af[buf][mt], sa + aoff[mt][kk]);
    }
#pragma unroll
    for (int np = 0; np < 4; ++np) {
      if (!BMN) ldsm_x4(bf[buf][np], sa + boff[np][kk]); else ldsm_x4_t(bf[buf][np], sa + boff[np][kk]);
    }
  };
  auto mma_all = [&](int buf) {
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        mma16816(acc[mt][2 * np], af[buf][mt], bf[buf][np][0], bf[buf][np][1]);
        mma16816(acc[mt][2 * np + 1], af[buf][mt], bf[buf][np][2], bf[buf][np][3]);
      }
  };
  cp_async_wait<ST - 2>();
  __syncthreads();
  load_frags(0, 0, 0);
  // group s = stage s.  At the last k16 step of k-tile kt: stage kt + 1 must have landed (wait<ST-3>), every warp is past stage kt - 1
  // (barrier), so its slot takes stage kt + ST - 1
#pragma unroll 1
  for (int kt = 0; kt < nk; ++kt) {
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      if (kk == KK - 1) {
        cp_async_wait<ST - 3>();
        __syncthreads();
        if (kt + ST - 1 < nk) load_stage(kt + ST - 1, (kt + ST - 1) % ST);
        cp_async_commit();
        if (kt + 1 < nk) load_frags((kk + 1) & 1, (kt + 1) % ST, 0);
      } else {
        load_frags((kk + 1) & 1, kt % ST, kk + 1);
      }
      mma_all(kk & 1);
    }
  }
  cp_async_wait<0>();
  __syncthreads();                                                   // every warp is out of the ring: it stages the output tile
  // epilogue: bf16 words -> [128][128] staging tile (16 B granule g of row r at g ^ (r & 7): the 8 rows of a store land in 8 distinct slots)
  //           -> 16 B coalesced plane rows (256 B per 16 threads)
  const int g8 = lane >> 2, q = lane & 3;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = 64 * wm + 16 * mt + g8 + 8 * h;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int gc = (64 * wn + 8 * nt) >> 3;
        sts32(s0 + r * 256 + ((gc ^ (r & 7)) << 4) + q * 4, pack_bf16(acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1]));
      }
    }
  __syncthreads();
  __nv_bfloat16* X = Xo + (size_t)i0 * L + j0;
#pragma unroll
  for (int e = 0; e < 16; ++e) {
    const int gidx = tid + 128 * e, r = gidx >> 4, gc = gidx & 15;
    stg128(X + (size_t)r * L + gc * 8, lds128(s0 + r * 256 + ((gc ^ (r & 7)) << 4)));
  }
  __syncthreads();                                                   // the ring is reused by the next tile of this CTA
}

template <class G, bool TN>
DEVI void contract_tile(const ContractParams& p, int c, int ti, int tj, uint8_t* smem) {
  const size_t plane = (size_t)p.L * p.L;
  contract_tile_g<G, TN, TN>(p.a + (size_t)c * plane, p.b + (size_t)c * plane, p.x + (size_t)c * plane, p.L, ti, tj, smem);
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

// Several contractions in one grid (the backward's four products): segment s = nch channels of C = P Q with its own layout
struct ContractSeg { const __nv_bfloat16* a; const __nv_bfloat16* b; __nv_bfloat16* x; int nch, mode; };   // mode 0 NT, 1 TN, 2 NN
struct ContractMultiParams { ContractSeg seg[4]; int nseg, L; };

template <class G>
__global__ void __launch_bounds__(G::NTHR, 2) contract_multi_kernel(const ContractMultiParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int nj = p.L / G::BN, tpc = (p.L / G::BM) * nj;
  const size_t plane = (size_t)p.L * p.L;
  int total = 0;
  for (int s = 0; s < p.nseg; ++s) total += p.seg[s].nch * tpc;
  for (int t = blockIdx.x; t < total; t += gridDim.x) {
    int s = 0, r = t;
    while (r >= p.seg[s].nch * tpc) { r -= p.seg[s].nch * tpc; ++s; }
    const int c = r / tpc, rt = r - c * tpc, ti = rt / nj, tj = rt - ti * nj;
    const ContractSeg& g = p.seg[s];
    const __nv_bfloat16* A = g.a + (size_t)c * plane;
    const __nv_bfloat16* B = g.b + (size_t)c * plane;
    __nv_bfloat16* X = g.x + (size_t)c * plane;
    if (g.mode == 0) contract_tile_g<G, false, false>(A, B, X, p.L, ti, tj, smem);
    else if (g.mode == 1) contract_tile_g<G, true, true>(A, B, X, p.L, ti, tj, smem);
    else contract_tile_g<G, false, true>(A, B, X, p.L, ti, tj, smem);
  }
}

}  // namespace a100
