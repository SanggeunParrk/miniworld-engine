// tr_bwd_g_sm80.cuh -- the PW role of the sm_80 Transition backward (tr_bwd_pw_sm80.cuh) generalised over (D, H) for D = 64 / 128: for one hidden slice and one row replica,
// recompute a, b, dh from xn / dy, run the SwiGLU backward, hand dA | dB to the X kernel as fragment-native 16 x 16 blocks and accumulate dWa, dWb, dWs^T in registers
// over the replica's rows (f32 partial sums).  The slice is SU = 8192 / D hidden units (64 at D = 128, 128 at D = 64), so a slice's weights (48 KB), its phase-1 work per
// 64-row stage (6 M D SU) and its phase-2 work (24 blocks of 16 x 64 f32 outputs, 3 per warp) are the same size at both widths:
//   phase 1: warp w computes rows 32 (w & 1) .. + 32 x hidden steps (w >> 1) + 4 pp (16 units each; SU / 64 passes pp): [a | b] and dh with A fragments streamed from the stage
//            by ldmatrix, the SwiGLU backward, and h | dA | dB as bf16 into a [3][64 rows][SU hidden] shared tile;
//   phase 2: warp w owns blocks 3 w .. 3 w + 2 of the slice's [dWs^T; dWa; dWb] output (matrix, hidden m16, 64-column half): A = the transposed h | dA | dB tile
//            (ldmatrix.trans), B = xn or dy rows (ldmatrix.trans); 96 f32 accumulators for the whole kernel.
// Conventions of the D = 128 kernel kept: Wa is pre-scaled by 1/2 in the packed weights (the dA tile holds 2 dA, the X kernel's Wa carries the 1/2 again, dWa is summed
// and scaled by 1/2 at the end); the LayerNorm is not recomputed (xn comes from the forward).
#pragma once
#include "tr_bwd_sm80.cuh"
#include "tr_bwd_pw_sm80.cuh"          // PWParams (and the D = 128 / H = 512 roles); everything else of the file is the generalised role

namespace a100 {

template <int D_, int H_>
struct PwCfgT {
  static constexpr int D = D_, H = H_, NTHR = 256, RS = 64, NSTAGE = 2;
  static constexpr int SU = 8192 / D;                              // hidden units per slice CTA
  static constexpr int NSL = H / SU;                               // slices
  static constexpr int NS = D / 16;                                // k16 steps of phase 1
  static constexpr int GPR = D / 8;                                // 16-byte granules per x / dy row
  static constexpr int RB = 2 * D;                                 // bytes per x / dy row
  static constexpr int GR1 = 2 * SU * 16, GR3 = SU * 16;           // k-granule strides of the resident W1s (0.5 Wa | Wb) and W3s (Ws^T)
  static constexpr int W1 = GPR * GR1, W3 = GPR * GR3, WRES = W1 + W3;   // 32 KB, 16 KB, 48 KB at both widths
  static constexpr int TILE = RS * RB, STAGE = 2 * TILE;           // x (-> xn) | dy, [64 rows][RB] swizzled
  static constexpr int HROW = 2 * SU, HT = RS * HROW;              // one of h | dA | dB: [64 rows][SU hidden] bf16
  static constexpr int OFF_ST = WRES, OFF_H = OFF_ST + NSTAGE * STAGE;
  static constexpr int SMEM = OFF_H + 3 * HT;
  static_assert(SMEM + 1024 <= 167936, "sm_80 shared memory");
  static_assert(D == 64 || D == 128, "widths");
  static_assert(H % SU == 0, "hidden slices");
};

template <int RB>
DEVI uint32_t swz_rb(uint32_t r, uint32_t G) { return r * RB + ((G ^ (r & 7u)) << 4); }   // 16-byte granule G of row r of a tile with RB-byte rows

// phase-2 block k (0 .. 23) of a slice -> (tile matrix 0 h / 1 dA / 2 dB, 64-column half, hidden m16)
template <int D>
DEVI void pw_block(int k, int& mat, int& half, int& hm) {
  if constexpr (D == 128) {
    // groups of 4 hidden m16: (h, 0) (h, 1) (dA, 0) (dB, 0) (dA, 1) (dB, 1): neighbouring dA / dB groups share their B = xn half
    const int gm[6] = {0, 0, 1, 2, 1, 2}, gd[6] = {0, 1, 0, 0, 1, 1};
    mat = gm[k >> 2]; half = gd[k >> 2]; hm = k & 3;
  } else {
    mat = k >> 3; half = 0; hm = k & 7;                            // groups of 8 hidden m16: h (B = dy), dA, dB (B = xn)
  }
}

// SPLIT: the first block that uses the second B operand (0: a single B)
template <class G, int SPLIT>
DEVI void pwg_phase2(float (&acc)[3][8][4], const uint32_t (&abase)[3], const int (&ahg)[3], uint32_t b0base, uint32_t b1base, int lane) {
  const int ar = (((lane >> 4) & 1) << 3) + (lane & 7), ag = (lane >> 3) & 1;       // A: row half = lane / 16, hidden half = lane / 8 % 2
  const int br = (((lane >> 3) & 1) << 3) + (lane & 7), bg = lane >> 4;            // B: row half = lane / 8 % 2, column half = lane / 16
#pragma unroll
  for (int kk = 0; kk < G::RS / 16; ++kk) {
    uint32_t a[3][4];
#pragma unroll
    for (int m = 0; m < 3; ++m) ldsm_x4_t(a[m], abase[m] + swz_rb<G::HROW>(16 * kk + ar, ahg[m] + ag));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t b0[4], b1[4];
      ldsm_x4_t(b0, b0base + swz_rb<G::RB>(16 * kk + br, 2 * j + bg));
      if (SPLIT) ldsm_x4_t(b1, b1base + swz_rb<G::RB>(16 * kk + br, 2 * j + bg));
#pragma unroll
      for (int m = 0; m < 3; ++m) {
        const uint32_t* b = (SPLIT && m >= SPLIT) ? b1 : b0;
        mma16816(acc[m][2 * j], a[m], b[0], b[1]);
        mma16816(acc[m][2 * j + 1], a[m], b[2], b[3]);
      }
    }
  }
}

template <class G>
DEVI void pwg_role(const PWParams& p, uint8_t* smem, int sl, int rr, int nrep) {
  constexpr int SU = G::SU, NS = G::NS, GPR = G::GPR, RB = G::RB, HROW = G::HROW, D = G::D, HK = G::H / 16;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q = lane & 3;
  const int nst = p.T / G::RS, n_mine = nst > rr ? (nst - rr + nrep - 1) / nrep : 0;
  const uint32_t s_u = smem_u32(smem);
  {                                                                // the slice's weights, resident
    const __nv_bfloat16* src = p.wdw + (size_t)sl * (G::WRES / 2);
    for (int c = tid; c < G::WRES / 16; c += G::NTHR) cp_async16_full(s_u + c * 16, src + c * 8);
  }
  auto load_stage = [&](int i) {
    const int row0 = (rr + i * nrep) * G::RS;
    const uint32_t buf = s_u + G::OFF_ST + (i % G::NSTAGE) * G::STAGE;
#pragma unroll
    for (int k = 0; k < D / 16; ++k) {
      const int c = tid + G::NTHR * k, m = c / (G::RS * GPR), r = (c / GPR) % G::RS, gr = c % GPR;
      cp_async16_full(buf + m * G::TILE + swz_rb<RB>(r, gr), (m ? p.dy : p.x) + (size_t)(row0 + r) * D + gr * 8);
    }
  };
  float acc[3][8][4];
#pragma unroll
  for (int m = 0; m < 3; ++m)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[m][j][e] = 0.f;
  const int rg = warp & 1, ps = warp >> 1;
  int bmat[3], bhm[3], bdh[3];
#pragma unroll
  for (int m = 0; m < 3; ++m) pw_block<D>(3 * warp + m, bmat[m], bdh[m], bhm[m]);
  const int split = D == 128 ? ((warp == 1 || warp == 5) ? 1 : warp == 2 ? 2 : 0) : (warp == 2 ? 2 : 0);
  const int lrow = ((lane >> 4) << 3) + (lane & 7), lgr = (lane >> 3) & 1;
  const int ar = rg * 32 + (((lane >> 3) & 1) << 3) + (lane & 7), ag = lane >> 4;
  const uint32_t sh = s_u + G::OFF_H;
  // block copy-out: warp -> matrix (dA, dB) = warp / 4, row block warp % 4, the slice's SU / 16 K blocks
  const int cm = warp >> 2, crb = warp & 3, car = (((lane >> 3) & 1) << 3) + (lane & 7), cag = lane >> 4;

  if (n_mine > 0) load_stage(0);
  cp_async_commit();
#pragma unroll 1
  for (int i = 0; i < n_mine; ++i) {
    cp_async_wait<0>();
    __syncthreads();                                            // stage i landed; stage i - 1 retired
    if (i + 1 < n_mine) load_stage(i + 1);
    cp_async_commit();
    const uint32_t buf = s_u + G::OFF_ST + (i % G::NSTAGE) * G::STAGE, sx = buf, sy = buf + G::TILE;
    const int row0 = (rr + i * nrep) * G::RS;
    // ---- phase 1: rows 32 rg .. of the stage x hidden steps ps + 4 pp of the slice
#pragma unroll 1
    for (int pp = 0; pp < SU / 64; ++pp) {
      const int st = ps + 4 * pp;
      const uint32_t w1_off = lgr * G::GR1 + lrow * 16 + st * 512, w3_off = G::W1 + lgr * G::GR3 + lrow * 16 + st * 256;
      float acc1[2][4][4], accd[2][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
#pragma unroll
          for (int n = 0; n < 4; ++n) acc1[mt][n][e] = 0.f;
          accd[mt][0][e] = accd[mt][1][e] = 0.f;
        }
      uint32_t fx[2][2][4], fy[2][2][4], ba[2][4], bb[2][4], bd[2][4];
      auto ld = [&](int s, int cb) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          ldsm_x4(fx[cb][mt], sx + swz_rb<RB>(ar + 16 * mt, 2 * s + ag));
          ldsm_x4(fy[cb][mt], sy + swz_rb<RB>(ar + 16 * mt, 2 * s + ag));
        }
        ldsm_x4(ba[cb], s_u + w1_off + s * 2 * G::GR1);
        ldsm_x4(bb[cb], s_u + w1_off + 256 + s * 2 * G::GR1);
        ldsm_x4(bd[cb], s_u + w3_off + s * 2 * G::GR3);
      };
      ld(0, 0);
#pragma unroll
      for (int s = 0; s < NS; ++s) {
        const int cb = s & 1;
        if (s < NS - 1) ld(s + 1, cb ^ 1);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma16816(acc1[mt][0], fx[cb][mt], ba[cb][0], ba[cb][1]);
          mma16816(acc1[mt][1], fx[cb][mt], ba[cb][2], ba[cb][3]);
          mma16816(acc1[mt][2], fx[cb][mt], bb[cb][0], bb[cb][1]);
          mma16816(acc1[mt][3], fx[cb][mt], bb[cb][2], bb[cb][3]);
          mma16816(accd[mt][0], fy[cb][mt], bd[cb][0], bd[cb][1]);
          mma16816(accd[mt][1], fy[cb][mt], bd[cb][2], bd[cb][3]);
        }
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int hr = 0; hr < 2; ++hr) {
            float vA[2], vB[2], vH[2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
              const int e = 2 * hr + k;
              const float ap = acc1[mt][n][e], bv = acc1[mt][2 + n][e], d = accd[mt][n][e];   // ap = a / 2 (Wa pre-scaled)
              const float t = tanh_approx(ap), sp = 1.f + t, silu = ap * sp;
              vH[k] = silu * bv;
              vB[k] = d * silu;
              vA[k] = d * bv * fmaf(-silu, t, silu + sp);                     // 2 dA = d b (2 sig + 2 silu (1 - sig))
            }
            // C element (row g8 + 8 hr, hidden 8 n + 2 q + k of the step) -> [mat][row][hidden] bf16
            const int row = 32 * rg + 16 * mt + 8 * hr + g8, hid = 16 * st + 8 * n + 2 * q;
            const uint32_t o = swz_rb<HROW>(row, hid >> 3) + (hid & 7) * 2;
            sts32(sh + 0 * G::HT + o, pack_bf16(vH[0], vH[1]));
            sts32(sh + 1 * G::HT + o, pack_bf16(vA[0], vA[1]));
            sts32(sh + 2 * G::HT + o, pack_bf16(vB[0], vB[1]));
          }
    }
    __syncthreads();
    {                                                           // dA | dB blocks for X
      uint4* const dst = p.ab + ((size_t)(row0 / 16 + crb) * HK + (SU / 16) * sl) * 96 + cm * 32 + lane;
#pragma unroll
      for (int j = 0; j < SU / 16; ++j) {
        uint32_t a[4];
        ldsm_x4(a, sh + (1 + cm) * G::HT + swz_rb<HROW>(16 * crb + car, 2 * j + cag));
        stg128(dst + j * 96, make_uint4(a[0], a[1], a[2], a[3]));
      }
    }
    {
      uint32_t abase[3];
      int ahg[3];
#pragma unroll
      for (int m = 0; m < 3; ++m) { abase[m] = sh + bmat[m] * G::HT; ahg[m] = 2 * bhm[m]; }
      // B column half: granule + 8 dh (bit 3, outside the XOR of swz_rb, so a plain 128 B offset)
      const uint32_t b0 = (bmat[0] == 0 ? sy : sx) + bdh[0] * 128, b1 = (bmat[2] == 0 ? sy : sx) + bdh[2] * 128;
      if (split == 0) pwg_phase2<G, 0>(acc, abase, ahg, b0, b1, lane);
      else if (split == 1) pwg_phase2<G, 1>(acc, abase, ahg, b0, b1, lane);
      else pwg_phase2<G, 2>(acc, abase, ahg, b0, b1, lane);
    }
  }
  cp_async_wait<0>();
  // partial sums: acc[m][j] = hidden SU sl + 16 bhm + g8 (+ 8) x columns 64 bdh + 8 j + 2 q + e of tile matrix bmat; part [rep][dWa | dWb | dWs^T][H][D]:
  // tile matrix 0 (h) -> 2, 1 (dA) -> 0, 2 (dB) -> 1
#pragma unroll
  for (int m = 0; m < 3; ++m) {
    const int pm = bmat[m] == 0 ? 2 : bmat[m] - 1;
    float* out = p.part + (((size_t)rr * 3 + pm) * G::H + SU * sl + 16 * bhm[m]) * D + 64 * bdh[m];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr)
        *reinterpret_cast<float2*>(out + (size_t)(8 * hr + g8) * D + 8 * j + 2 * q) = make_float2(acc[m][j][2 * hr], acc[m][j][2 * hr + 1]);
  }
}

template <class G>
__global__ void __launch_bounds__(256, 1) tr_bwd_pwg_kernel(const PWParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  pwg_role<G>(p, smem, blockIdx.x % G::NSL, blockIdx.x / G::NSL, gridDim.x / G::NSL);
}

}  // namespace a100
