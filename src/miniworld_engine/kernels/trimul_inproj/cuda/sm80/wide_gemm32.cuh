// wide_gemm32.cuh -- the TF32 sibling of wide_gemm.cuh: the GEMM tile of the A100 "wide" TriMul for fp32 operands (mma.sync m16n8k8 tf32 -> fp32 accumulate).
//
// The data flow is the bf16 tile's: BK = 16 elements = the same 64-byte rows, a cp.async ring of ST stages, ldmatrix fragments (the 16-byte granule is 4 fp32; a k8 step
// is two granules, so the A / B fragment addresses are the bf16 ones), one barrier per k-tile.  The tensor core reads the raw fp32 words and ignores the low 13 mantissa
// bits (the hardware's TF32 truncation); the folded weights are pre-rounded to TF32 (wide_rows.cuh).
//
//   A: AM = false  A[m][k] at A[m * lda + k]   (K-major rows: ldmatrix)
//      AM = true   A[m][k] at A[k * lda + m]   (M-major, the channel-major contraction output): ldmatrix cannot transpose 32-bit words, so the fragments are four
//                  32-bit shared loads per m16 tile; the rows (k) are swizzled so that the 4 k-rows x 8 tokens of a fragment load hit 32 distinct banks
//   B: always K-major rows B[n][k] at B[n * ldb + k]
#pragma once
#include "wide_gemm.cuh"

namespace a100 {

DEVI void mma1688(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

struct GemmOps32 {
  const float* A;  size_t lda;  int i0, Mlim;
  const float* B;  size_t ldb;  int j0, Nlim;
};

template <int BM_, int BN_, int WM_, int WN_, int ST_, bool AM_>
struct WTile32 {
  static constexpr int BM = BM_, BN = BN_, BK = 16, ST = ST_, WM = WM_, WN = WN_;
  static constexpr int NTHR = 32 * WM * WN;
  static constexpr int WTM = BM / WM, WTN = BN / WN;
  static constexpr int MT = WTM / 16, NP = WTN / 16;
  static constexpr int TILE_A = BM * 64, TILE_B = BN * 64, STAGE = TILE_A + TILE_B;
  static constexpr int SMEM = ST * STAGE;
  static constexpr bool AM = AM_;
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

  DEVI void load_stage(uint32_t s0, int st, int kt, const GemmOps32& o) const {
    const int tid = threadIdx.x;
    const uint32_t sa = s0 + st * STAGE, sb = sa + TILE_A;
    const int k0 = kt * BK;
#pragma unroll
    for (int e = 0; e < 4 * BM / NTHR; ++e) {
      const int g = tid + NTHR * e;
      if (!AM) {
        const int row = g >> 2, c = g & 3;
        const bool ok = o.i0 + row < o.Mlim;
        cp_async16(sa + wkmaj(row, c), o.A + (size_t)(ok ? o.i0 + row : 0) * o.lda + k0 + c * 4, ok ? 16u : 0u);
      } else {                                                        // rows = k (16), BM / 4 granules of 4 tokens
        const int row = g / (BM / 4), c = g % (BM / 4);
        const bool ok = o.i0 + 4 * c < o.Mlim;
        cp_async16(sa + row * (BM * 4) + ((c ^ ((row & 3) << 1)) << 4), o.A + (size_t)(k0 + row) * o.lda + (ok ? o.i0 + 4 * c : 0), ok ? 16u : 0u);
      }
    }
#pragma unroll
    for (int e = 0; e < 4 * BN / NTHR; ++e) {
      const int g = tid + NTHR * e;
      const int row = g >> 2, c = g & 3;
      const bool ok = o.j0 + row < o.Nlim;
      cp_async16(sb + wkmaj(row, c), o.B + (size_t)(ok ? o.j0 + row : 0) * o.ldb + k0 + c * 4, ok ? 16u : 0u);
    }
  }

  DEVI void run(uint32_t s0, int nk, const GemmOps32& o) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WN, wn = warp % WN;
    zero();
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) {
      if (st < nk) load_stage(s0, st, st, o);
      cp_async_commit();
    }
    const int mi = lane >> 3, r8 = lane & 7, g = lane >> 2, t = lane & 3;
    constexpr int KK = BK / 8;
    uint32_t af[2][MT][4], bf[2][NP][4];
    uint32_t aoff[MT][KK], boff[NP][KK];             // K-major A: ldmatrix addresses;  AM: (low-token, high-token) column offsets below
    uint32_t amlo[MT], amhi[MT];
#pragma unroll
    for (int tt = 0; tt < MT; ++tt) {
      if (!AM) {
#pragma unroll
        for (int kk = 0; kk < KK; ++kk) aoff[tt][kk] = wkmaj(WTM * wm + 16 * tt + r8 + (mi & 1) * 8, 2 * kk + (mi >> 1));
      } else {
        const int mlo = WTM * wm + 16 * tt + g;
        amlo[tt] = ((((uint32_t)(mlo >> 2)) ^ (2u * t)) << 4) + (g & 3) * 4;
        amhi[tt] = ((((uint32_t)((mlo + 8) >> 2)) ^ (2u * t)) << 4) + (g & 3) * 4;
      }
    }
#pragma unroll
    for (int tt = 0; tt < NP; ++tt)
#pragma unroll
      for (int kk = 0; kk < KK; ++kk) boff[tt][kk] = TILE_A + wkmaj(WTN * wn + 16 * tt + r8 + (mi >> 1) * 8, 2 * kk + (mi & 1));
    auto load_frags = [&](int buf, int slot, int kk) {
      const uint32_t sa = s0 + slot * STAGE;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        if (!AM) {
          ldsm_x4(af[buf][mt], sa + aoff[mt][kk]);
        } else {
          const uint32_t rk = sa + (8 * kk + t) * (BM * 4), rk4 = rk + 4 * (BM * 4);
          af[buf][mt][0] = lds32(rk + amlo[mt]);
          af[buf][mt][1] = lds32(rk + amhi[mt]);
          af[buf][mt][2] = lds32(rk4 + amlo[mt]);
          af[buf][mt][3] = lds32(rk4 + amhi[mt]);
        }
      }
#pragma unroll
      for (int np = 0; np < NP; ++np) ldsm_x4(bf[buf][np], sa + boff[np][kk]);
    };
    auto mma_all = [&](int buf) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int np = 0; np < NP; ++np) {
          mma1688(acc[mt][2 * np], af[buf][mt], bf[buf][np][0], bf[buf][np][1]);
          mma1688(acc[mt][2 * np + 1], af[buf][mt], bf[buf][np][2], bf[buf][np][3]);
        }
    };
    cp_async_wait<ST - 2>();
    __syncthreads();
    load_frags(0, 0, 0);
#pragma unroll 1
    for (int kt = 0; kt < nk; ++kt) {
#pragma unroll
      for (int kk = 0; kk < KK; ++kk) {
        if (kk == KK - 1) {
          cp_async_wait<ST - 3>();
          __syncthreads();
          if (kt + ST - 1 < nk) load_stage(s0, (kt + ST - 1) % ST, kt + ST - 1, o);
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
};

// ---- fp32 epilogue tiles (see tile_store16 in wide_gemm.cuh): a [ROWS][COLS] fp32 tile is staged in shared memory and leaves as 16-byte row stores.
// ``stage_f1``: one word at (row, col), rows of COLS * 4 + 16 bytes (the channel-major planes: 32-bit stores of 8 tokens x 4 channels hit 32 banks);
// ``stage_f2``: two adjacent columns as one 64-bit store, rows of COLS * 4 + 32 bytes (a half-warp's 8 x 4 pairs hit 32 banks).
DEVI void stage_f1(uint32_t stg, int row, int col, int cols, float v) { sts32(stg + row * (cols * 4 + 16) + col * 4, __float_as_uint(v)); }
DEVI void stage_f2(uint32_t stg, int row, int col, int cols, float a, float b) {
  sts64(stg + row * (cols * 4 + 32) + col * 4, make_uint2(__float_as_uint(a), __float_as_uint(b)));
}
template <int ROWS, int COLS, int NTHR, int PAD>
DEVI void tile_store16f(uint32_t stg, float* dst, size_t ld, int rows_valid = ROWS) {
  constexpr int GR = COLS / 4;
#pragma unroll
  for (int e = 0; e < ROWS * GR / NTHR; ++e) {
    const int idx = (int)threadIdx.x + NTHR * e, r = idx / GR, gk = idx - r * GR;
    if (r < rows_valid) stg128(dst + (size_t)r * ld + 4 * gk, lds128(stg + r * (COLS * 4 + PAD) + gk * 16));
  }
}

}  // namespace a100
