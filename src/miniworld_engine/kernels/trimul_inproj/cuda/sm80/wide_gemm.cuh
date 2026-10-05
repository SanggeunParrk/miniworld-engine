// wide_gemm.cuh -- the bf16 tensor-core GEMM tile of the A100 "wide" TriMul path (any width D, either direction) and of the gated-projection ops.
//
//   C[m, n] = sum_k A[m, k] B[n, k]      bf16 x bf16 -> fp32 accumulate, one CTA tile BM x BN, BK = 32, a cp.async ring of ST stages
//
// The mainloop is the contraction kernel's (contract_sm80.cuh): ldmatrix fragments double-buffered across the k16 steps, one barrier per k-tile,
// the next stage issued under the current MMAs.  What this adds over it: runtime leading dimensions, M / N bounds (zero-filled loads), a
// 2 x 2 / 4 x 2 / ... warp grid, and a token permutation inside every m16 tile (RP) so that accumulator rows g8 and g8 + 8 are the ADJACENT tokens
// 2 g8, 2 g8 + 1 (a thread's (c0, c2) / (c1, c3) are then the packed bf16 pairs of one channel's two neighbouring tokens: the transposing
// epilogue stores need no shuffle).
//
// Operand layouts (bf16):
//   A: AM = false  A[m][k] at A[m * lda + k]      (K-major rows: ldmatrix)          AM = true  A[m][k] at A[k * lda + m]  (M-major: ldmatrix.trans)
//   B: always K-major rows B[n][k] at B[n * ldb + k] (the weights, [out, in])
// Shared tiles: K-major 64-B rows (granule c of row r at c ^ ((r >> 1) & 3)), M-major rows of BM * 2 bytes (granule c at c ^ (r & 7)): the eight
// rows of an ldmatrix land in eight distinct 16-B bank groups.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

// K-major tile (64-B rows, 4 granules): a pair of rows is one 128-B unit whose eight granules are permuted by (row >> 1) & 7.  Both the eight
// consecutive rows of an ldmatrix and the eight rows 2 k + b of the token-permuted (RP) fragment read land in eight distinct 16-B bank groups.
DEVI uint32_t wkmaj(uint32_t row, uint32_t c) { return (row >> 1) * 128 + ((((row & 1u) * 4u + c) ^ ((row >> 1) & 7u)) << 4); }
template <int BM> DEVI uint32_t wmmaj(uint32_t row, uint32_t c) { return row * (BM * 2) + ((c ^ (row & 7u)) << 4); }

struct GemmOps {
  const __nv_bfloat16* A;  size_t lda;  int i0, Mlim;      // tile origin (rows of C) and the number of valid rows
  const __nv_bfloat16* B;  size_t ldb;  int j0, Nlim;      // tile origin (columns of C = rows of B) and the number of valid columns
};

// A hook adds work to the ring: ``load(stage_base, k-tile)`` issues extra cp.async (EXTRA bytes of shared memory per stage, after the A and B tiles) and
// ``transform(stage_base)`` runs on this thread's own granules once its loads of the stage have landed, before the barrier that publishes the stage (the gated
// projections turn a (gate, value) pair of tiles into the A tile there).  NoHook is the plain GEMM.
struct NoHook {
  DEVI void load(uint32_t, int) const {}
  DEVI void transform(uint32_t) const {}
};

template <int BM_, int BN_, int WM_, int WN_, int ST_, bool AM_, bool RP_ = false, int EXTRA_ = 0>
struct WTile {
  static constexpr int BM = BM_, BN = BN_, BK = 32, ST = ST_, WM = WM_, WN = WN_;
  static constexpr int NTHR = 32 * WM * WN;
  static constexpr int WTM = BM / WM, WTN = BN / WN;               // warp tile
  static constexpr int MT = WTM / 16, NP = WTN / 16;                // m16 tiles; pairs of n8 tiles (one ldmatrix.x4 of B each)
  static constexpr int TILE_A = BM * BK * 2, TILE_B = BN * BK * 2, EXTRA = EXTRA_, STAGE = TILE_A + TILE_B + EXTRA;
  static constexpr int SMEM = ST * STAGE;
  static constexpr bool AM = AM_, RP = RP_;
  static_assert(WTM % 16 == 0 && WTN % 16 == 0 && ST >= 3, "warp tile / ring depth");
  static_assert((4 * BM) % NTHR == 0 && (4 * BN) % NTHR == 0, "one 16-B granule per thread and load round");

  float acc[MT][2 * NP][4];

  DEVI void zero() {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < 2 * NP; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
  }

  DEVI void load_stage(uint32_t s0, int st, int kt, const GemmOps& o) const {
    const int tid = threadIdx.x;
    const uint32_t sa = s0 + st * STAGE, sb = sa + TILE_A;
    const int k0 = kt * BK;
#pragma unroll
    for (int e = 0; e < 4 * BM / NTHR; ++e) {
      const int g = tid + NTHR * e;
      if (!AM) {                                                       // rows = tokens (BM), 4 granules of k
        const int row = g >> 2, c = g & 3;
        const bool ok = o.i0 + row < o.Mlim;
        cp_async16(sa + wkmaj(row, c), o.A + (size_t)(ok ? o.i0 + row : 0) * o.lda + k0 + c * 8, ok ? 16u : 0u);
      } else {                                                         // rows = k (32), BM / 8 granules of tokens
        const int row = g / (BM / 8), c = g % (BM / 8);
        const bool ok = o.i0 + 8 * c < o.Mlim;
        cp_async16(sa + wmmaj<BM>(row, c), o.A + (size_t)(k0 + row) * o.lda + (ok ? o.i0 + 8 * c : 0), ok ? 16u : 0u);
      }
    }
#pragma unroll
    for (int e = 0; e < 4 * BN / NTHR; ++e) {
      const int g = tid + NTHR * e;
      const int row = g >> 2, c = g & 3;
      const bool ok = o.j0 + row < o.Nlim;
      cp_async16(sb + wkmaj(row, c), o.B + (size_t)(ok ? o.j0 + row : 0) * o.ldb + k0 + c * 8, ok ? 16u : 0u);
    }
  }

  // acc = A B^T over nk k-tiles of 32; ends with every cp.async retired and every warp past the ring (the caller may reuse the shared memory)
  template <class Hook = NoHook>
  DEVI void run(uint32_t s0, int nk, const GemmOps& o, const Hook& hook = Hook()) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WN, wn = warp % WN;
    zero();
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) {
      if (st < nk) { load_stage(s0, st, st, o); hook.load(s0 + st * STAGE, st); }
      cp_async_commit();
    }
    const int mi = lane >> 3, r8 = lane & 7;
    constexpr int KK = BK / 16;
    uint32_t af[2][MT][4], bf[2][NP][4];
    // lane-constant fragment offsets inside a stage
    uint32_t aoff[MT][KK], boff[NP][KK];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int kk = 0; kk < KK; ++kk) {
        if (!AM) {
          const uint32_t row = WTM * wm + 16 * t + (RP ? 2 * r8 + (mi & 1) : r8 + (mi & 1) * 8);
          aoff[t][kk] = wkmaj(row, 2 * kk + (mi >> 1));
        } else {
          aoff[t][kk] = wmmaj<BM>(16 * kk + (mi >> 1) * 8 + r8, (WTM / 8) * wm + 2 * t + (mi & 1));
        }
      }
#pragma unroll
    for (int t = 0; t < NP; ++t)
#pragma unroll
      for (int kk = 0; kk < KK; ++kk) boff[t][kk] = TILE_A + wkmaj(WTN * wn + 16 * t + r8 + (mi >> 1) * 8, 2 * kk + (mi & 1));
    auto load_frags = [&](int buf, int slot, int kk) {
      const uint32_t sa = s0 + slot * STAGE;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        if (!AM) ldsm_x4(af[buf][mt], sa + aoff[mt][kk]); else ldsm_x4_t(af[buf][mt], sa + aoff[mt][kk]);
      }
#pragma unroll
      for (int np = 0; np < NP; ++np) ldsm_x4(bf[buf][np], sa + boff[np][kk]);
    };
    auto mma_all = [&](int buf) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int np = 0; np < NP; ++np) {
          mma16816(acc[mt][2 * np], af[buf][mt], bf[buf][np][0], bf[buf][np][1]);
          mma16816(acc[mt][2 * np + 1], af[buf][mt], bf[buf][np][2], bf[buf][np][3]);
        }
    };
    cp_async_wait<ST - 2>();
    hook.transform(s0);                                  // stage 0
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
          if (kt + 1 < nk) hook.transform(s0 + ((kt + 1) % ST) * STAGE);
          __syncthreads();
          if (kt + ST - 1 < nk) { load_stage(s0, (kt + ST - 1) % ST, kt + ST - 1, o); hook.load(s0 + ((kt + ST - 1) % ST) * STAGE, kt + ST - 1); }
          cp_async_commit();
          if (kt + 1 < nk) load_frags((kk + 1) & 1, (kt + 1) % ST, 0);
        } else {
          load_frags((kk + 1) & 1, kt % ST, kk + 1);
        }
        mma_all(kk & 1);
      }
    }
    cp_async_wait<0>();
    __syncthreads();
  }

  // run() for tiles that need no bounds handling (Mlim a multiple of BM, Nlim of BN: the caller guarantees it) and no hook, with the addressing of the loop reduced to a few
  // per-thread bases plus immediates.  run()'s generic code (a runtime predicate, a clamped 64-bit row product and a swizzle per granule, a modulo per stage) executes ~160
  // integer instructions per warp and k-tile next to 64 HMMA; the in-order issue of those dependent chains is what starves the tensor pipe.  Here every smem offset of a
  // thread's cp.async granules and ldmatrix fragments is linear in the granule index e / the m16 or n16 tile t (the swizzles XOR with bits that e and t do not touch: one
  // row step of 16 is +1024 bytes of a K-major tile, one granule step is NTHR * 16 bytes), the global source of a k-tile is a base plus k-tile * stride, and the stage
  // addresses rotate without a modulo.  Same MMAs in the same order as run(): bit-identical results.
  template <class Hook = NoHook>
  DEVI void run_fast(uint32_t s0, int nk, const GemmOps& o, const Hook& hook = Hook()) {
    static_assert(NTHR % 64 == 0, "granule steps keep the swizzle's (row >> 1) & 7");
    constexpr int GA = 4 * BM / NTHR, GB = 4 * BN / NTHR, KK = BK / 16;
    constexpr int GPR = BM / 8, RPE = NTHR / GPR;                         // (M-major A) granules per k row, k rows per granule step
    constexpr uint32_t DSB = NTHR * 16, DSA = AM ? RPE * BM * 2 : NTHR * 16;   // smem offset of granule e + 1 over granule e (B, A)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WN, wn = warp % WN;
    zero();
    // ---- loader state: granule 0 of this thread; pA / pB advance by one k-tile per issued stage
    const int rb = tid >> 2, cb = tid & 3;
    const uint32_t dB0 = TILE_A + wkmaj(rb, cb);
    const char* pB = reinterpret_cast<const char*>(o.B) + ((size_t)(o.j0 + rb) * o.ldb + 8 * cb) * 2;
    const size_t eB = (size_t)(NTHR / 4) * o.ldb * 2;
    uint32_t dA0;
    const char* pA;
    size_t eA, kA;
    if constexpr (!AM) {
      dA0 = wkmaj(rb, cb);
      pA = reinterpret_cast<const char*>(o.A) + ((size_t)(o.i0 + rb) * o.lda + 8 * cb) * 2;
      eA = (size_t)(NTHR / 4) * o.lda * 2; kA = 64;
    } else {
      const int ra = tid / GPR, ca = tid % GPR;
      dA0 = wmmaj<BM>(ra, ca);
      pA = reinterpret_cast<const char*>(o.A) + ((size_t)ra * o.lda + o.i0 + 8 * ca) * 2;
      eA = (size_t)RPE * o.lda * 2; kA = (size_t)BK * o.lda * 2;
    }
    auto issue = [&](uint32_t sa) {
#pragma unroll
      for (int e = 0; e < GA; ++e) cp_async16(sa + dA0 + e * DSA, pA + (size_t)e * eA);
#pragma unroll
      for (int e = 0; e < GB; ++e) cp_async16(sa + dB0 + e * DSB, pB + (size_t)e * eB);
      pA += kA; pB += 64;
    };
    // ---- fragment offsets inside a stage (lane constants)
    const int mi = lane >> 3, r8 = lane & 7;
    uint32_t afk[KK], afm[MT], bo[KK];
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      afk[kk] = AM ? 0u : wkmaj(WTM * wm + (RP ? 2 * r8 + (mi & 1) : r8 + (mi & 1) * 8), 2 * kk + (mi >> 1));        // + t * 1024
      bo[kk] = TILE_A + wkmaj(WTN * wn + r8 + (mi >> 1) * 8, 2 * kk + (mi & 1));                                     // + np * 1024
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) afm[t] = AM ? wmmaj<BM>((mi >> 1) * 8 + r8, (WTM / 8) * wm + 2 * t + (mi & 1)) : 0u;   // + kk * 16 * BM * 2
    uint32_t af[2][MT][4], bf[2][NP][4];
    auto load_frags = [&](int buf, uint32_t sa, int kk) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        if constexpr (!AM) ldsm_x4(af[buf][mt], sa + afk[kk] + mt * 1024); else ldsm_x4_t(af[buf][mt], sa + afm[mt] + kk * (16 * BM * 2));
      }
#pragma unroll
      for (int np = 0; np < NP; ++np) ldsm_x4(bf[buf][np], sa + bo[kk] + np * 1024);
    };
    auto mma_all = [&](int buf) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int np = 0; np < NP; ++np) {
          mma16816(acc[mt][2 * np], af[buf][mt], bf[buf][np][0], bf[buf][np][1]);
          mma16816(acc[mt][2 * np + 1], af[buf][mt], bf[buf][np][2], bf[buf][np][3]);
        }
    };
    const uint32_t send = s0 + ST * STAGE;
    uint32_t rd = s0, wr = s0 + (ST - 1) * STAGE;                          // the stage being read / the stage the next load goes to
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) {
      if (st < nk) { issue(s0 + st * STAGE); hook.load(s0 + st * STAGE, st); }
      cp_async_commit();
    }
    cp_async_wait<ST - 2>();
    hook.transform(s0);                                                // stage 0
    __syncthreads();
    load_frags(0, rd, 0);
    // as run(): at the last k16 step of k-tile kt stage kt + 1 must have landed and every warp be past stage kt - 1, so its slot takes stage kt + ST - 1
#pragma unroll 1
    for (int kt = 0; kt < nk; ++kt) {
#pragma unroll
      for (int kk = 0; kk < KK; ++kk) {
        if (kk == KK - 1) {
          cp_async_wait<ST - 3>();
          const uint32_t nxt = rd + STAGE == send ? s0 : rd + STAGE;
          if (kt + 1 < nk) hook.transform(nxt);                          // before the barrier that publishes the stage
          __syncthreads();
          if (kt + ST - 1 < nk) { issue(wr); hook.load(wr, kt + ST - 1); }
          cp_async_commit();
          wr = wr + STAGE == send ? s0 : wr + STAGE;
          rd = rd + STAGE == send ? s0 : rd + STAGE;
          if (kt + 1 < nk) load_frags((kk + 1) & 1, rd, 0);
        } else {
          load_frags((kk + 1) & 1, rd, kk + 1);
        }
        mma_all(kk & 1);
      }
    }
    cp_async_wait<0>();
    __syncthreads();
  }
};

// ---- epilogue tiles: a [ROWS][COLS] bf16 tile is staged in shared memory (rows of COLS * 2 + 16 bytes: the words of one mma fragment row group land in 32
// distinct banks) and leaves as 16-byte stores covering whole rows -- the scattered 4-byte fragment stores of a row-major output ran at a fraction of the
// DRAM rate.  ``stage_word`` writes one fragment word (two adjacent columns of one row); ``tile_store16`` stores the tile (the caller brackets both with barriers).
DEVI void stage_word(uint32_t stg, int row, int col2, int cols, uint32_t w) { sts32(stg + row * (cols * 2 + 16) + col2 * 4, w); }
template <int ROWS, int COLS, int NTHR>
DEVI void tile_store16(uint32_t stg, __nv_bfloat16* dst, size_t ld, int rows_valid = ROWS) {
  constexpr int GR = COLS / 8;
#pragma unroll
  for (int e = 0; e < ROWS * GR / NTHR; ++e) {
    const int idx = (int)threadIdx.x + NTHR * e, r = idx / GR, gk = idx - r * GR;
    if (r < rows_valid) stg128(dst + (size_t)r * ld + 8 * gk, lds128(stg + r * (COLS * 2 + 16) + gk * 16));
  }
}

}  // namespace a100
