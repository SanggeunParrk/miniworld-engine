// tr_bwd_dw_sm80.cuh -- the H100-style weight-gradient role on A100: recompute a, b, dh for one 64-unit hidden slice, then
// dWa_s += dA^T xn, dWb_s += dB^T xn, dWs^T_s += h^T dy -- no intermediate crosses global memory (12 M D H over all slices).
//
// CTA = 8 warps on slice blockIdx % 8 and row replica blockIdx / 8.  The slice's weights stay resident (48 KB); rows come in 64-row stages
// (xn | dy, double-buffered cp.async).
//   phase 1: warp w computes rows 32 (w & 1) .. + 32 x hidden 16 (w >> 1) .. + 16: [a | b] and dh with A fragments streamed from the
//            stage by ldmatrix, the SwiGLU backward, and h | dA | dB as bf16 into a [3][64 rows][64 hidden] shared tile;
//   phase 2: warp w owns 3 of the 24 (matrix, hidden m16, 64-column half) blocks of the slice's [dWa; dWb; dWs^T] (96 f32 accumulators
//            for the whole kernel): A = the transposed h | dA | dB tile (ldmatrix.trans), B = xn or dy rows (ldmatrix.trans).
#pragma once
#include "tr_bwd_w_sm80.cuh"

namespace a100 {

struct DWParams {
  const __nv_bfloat16* xn;     // [T][128] LN(x), bf16 (the forward's bits; the fused kernel writes it in its prologue)
  const __nv_bfloat16* dy;     // [T][128]
  const __nv_bfloat16* wdw;    // [8 slices][W1s: 16 k-granules x 128 rows (0.5 Wa | Wb of 4 16-unit steps) | W3s: 16 x 64 rows (Ws^T)] x 16 B
  const float* gamma;
  const float* beta;
  float* part;                 // [nrep][3][512][128] f32
  int T;
  float eps;
};

struct CfgDW {
  static constexpr int NTHR = 256, RS = 64, NSTAGE = 2;
  static constexpr int W1 = 32768, W3 = 16384, WRES = W1 + W3;
  static constexpr int TILE = RS * 256, STAGE = 2 * TILE;      // x (-> xn) | dy, [64 rows][256 B] swizzled
  static constexpr int HROW = 128, HT = RS * HROW;             // one of h | dA | dB: [64 rows][64 hidden] bf16, 128 B rows swizzled
  static constexpr int OFF_ST = WRES, OFF_H = OFF_ST + NSTAGE * STAGE, OFF_GB = OFF_H + 3 * HT;
  static constexpr int SMEM = OFF_GB + 2 * 128 * 4;
  static_assert(SMEM + 1024 <= 167936, "sm_80 shared memory");
};
DEVI uint32_t swz_h(uint32_t r, uint32_t G) { return r * 128 + ((G ^ (r & 7u)) << 4); }   // 8 granules per 128 B row

// phase 2: the 24 (matrix, 64-column half, hidden m16) blocks in group order g = (h, dh 0) (h, dh 1) (dA, 0) (dB, 0) (dA, 1) (dB, 1)
// (neighbouring dA / dB groups share their B = xn half), 3 consecutive blocks per warp.  Warps 1, 2, 5 span two B operands: SPLIT = the
// first block that uses the second one (0: a single B).  A comes out of ldmatrix.trans already in fragment order (lane groups address
// (rows lo, hid lo), (rows lo, hid hi), (rows hi, hid lo), (rows hi, hid hi)).
template <int SPLIT>
DEVI void dw_phase2(float (&acc)[3][8][4], const uint32_t (&abase)[3], const int (&ahg)[3], uint32_t b0base, uint32_t b1base, int lane) {
  using G = CfgDW;
  const int ar = (((lane >> 4) & 1) << 3) + (lane & 7), ag = (lane >> 3) & 1;       // A: row half = lane / 16, hidden half = lane / 8 % 2
  const int br = (((lane >> 3) & 1) << 3) + (lane & 7), bg = lane >> 4;            // B: row half = lane / 8 % 2, column half = lane / 16
#pragma unroll
  for (int kk = 0; kk < G::RS / 16; ++kk) {
    uint32_t a[3][4];
#pragma unroll
    for (int m = 0; m < 3; ++m) ldsm_x4_t(a[m], abase[m] + swz_h(16 * kk + ar, ahg[m] + ag));   // the swizzle XOR needs the whole granule
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t b0[4], b1[4];
      ldsm_x4_t(b0, b0base + swz_w(16 * kk + br, 2 * j + bg));
      if (SPLIT) ldsm_x4_t(b1, b1base + swz_w(16 * kk + br, 2 * j + bg));
#pragma unroll
      for (int m = 0; m < 3; ++m) {
        const uint32_t* b = (SPLIT && m >= SPLIT) ? b1 : b0;
        mma16816(acc[m][2 * j], a[m], b[0], b[1]);
        mma16816(acc[m][2 * j + 1], a[m], b[2], b[3]);
      }
    }
  }
}

__global__ void __launch_bounds__(256, 1) tr_bwd_dw_kernel(const DWParams p) {
  using G = CfgDW;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q = lane & 3;
  const int sl = blockIdx.x & 7, rr = blockIdx.x >> 3, nrep = gridDim.x >> 3;
  const int nst = p.T / G::RS, n_mine = nst > rr ? (nst - rr + nrep - 1) / nrep : 0;
  const uint32_t s_u = smem_u32(smem);
  float* sGB = reinterpret_cast<float*>(smem + G::OFF_GB);
  for (int k = tid; k < 128; k += G::NTHR) { sGB[k] = p.gamma[k]; sGB[128 + k] = p.beta[k]; }
  {                                                             // resident slice weights
    const __nv_bfloat16* src = p.wdw + (size_t)sl * (G::WRES / 2);
    for (int c = tid; c < G::WRES / 16; c += G::NTHR) cp_async16_full(s_u + c * 16, src + c * 8);
  }
  auto load_stage = [&](int i) {
    const int row0 = (rr + i * nrep) * G::RS;
    const uint32_t buf = s_u + G::OFF_ST + (i % G::NSTAGE) * G::STAGE;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int c = tid + G::NTHR * k, m = c >> 10, r = (c >> 4) & 63, gr = c & 15;
      cp_async16_full(buf + m * G::TILE + swz_w(r, gr), (m ? p.dy : p.xn) + (size_t)(row0 + r) * 128 + gr * 8);
    }
  };
  float acc[3][8][4];
#pragma unroll
  for (int m = 0; m < 3; ++m)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[m][j][e] = 0.f;
  const int rg = warp & 1, ps = warp >> 1;                      // phase 1: row group, 16-unit step of the slice
  // phase 2 blocks of this warp: block k = 3 warp + m -> group g = k / 4 (matrix gm[g], half gd[g]), hidden m16 k % 4
  const int gm[6] = {0, 0, 1, 2, 1, 2}, gd[6] = {0, 1, 0, 0, 1, 1};   // tile matrices: 0 h, 1 dA, 2 dB
  int bmat[3], bhm[3], bdh[3];
#pragma unroll
  for (int m = 0; m < 3; ++m) { const int k = 3 * warp + m; bmat[m] = gm[k >> 2]; bdh[m] = gd[k >> 2]; bhm[m] = k & 3; }
  const int split = (warp == 1 || warp == 5) ? 1 : warp == 2 ? 2 : 0;   // first block on the second B operand
  const int lrow = ((lane >> 4) << 3) + (lane & 7), lgr = (lane >> 3) & 1;
  const uint32_t w1_off = lgr * 2048 + lrow * 16 + ps * 512, w3_off = G::W1 + lgr * 1024 + lrow * 16 + ps * 256;
  // A-fragment ldmatrix (non-trans) from a [rows][256 B] tile: matrix mi -> rows (mi & 1) * 8 + lane % 8, granule 2 ks + (mi >> 1)
  const int ar = rg * 32 + (((lane >> 3) & 1) << 3) + (lane & 7), ag = lane >> 4;
  const uint32_t sh = s_u + G::OFF_H;

  if (n_mine > 0) load_stage(0);
  cp_async_commit();
#pragma unroll 1
  for (int i = 0; i < n_mine; ++i) {
    cp_async_wait<0>();
    __syncthreads();                                            // stage i (and the resident weights) landed; stage i - 1 retired
    if (i + 1 < n_mine) load_stage(i + 1);
    cp_async_commit();
    const uint32_t buf = s_u + G::OFF_ST + (i % G::NSTAGE) * G::STAGE, sx = buf, sy = buf + G::TILE;
    // ---- phase 1: rows 32 rg .. of the stage x hidden 16 ps .. of the slice
    {
      float acc1[2][4][4], accd[2][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
#pragma unroll
          for (int n = 0; n < 4; ++n) acc1[mt][n][e] = 0.f;
          accd[mt][0][e] = accd[mt][1][e] = 0.f;
        }
#ifndef DW_P1_PIPE
#define DW_P1_PIPE 1
#endif
      // fragments of k-step s + 1 are loaded under the MMAs of k-step s (DW_P1_PIPE)
      uint32_t fx[2][2][4], fy[2][2][4], ba[2][4], bb[2][4], bd[2][4];
      auto ld = [&](int s, int cb) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          ldsm_x4(fx[cb][mt], sx + swz_w(ar + 16 * mt, 2 * s + ag));
          ldsm_x4(fy[cb][mt], sy + swz_w(ar + 16 * mt, 2 * s + ag));
        }
        ldsm_x4(ba[cb], s_u + w1_off + s * 4096);
        ldsm_x4(bb[cb], s_u + w1_off + 256 + s * 4096);
        ldsm_x4(bd[cb], s_u + w3_off + s * 2048);
      };
      ld(0, 0);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int cb = DW_P1_PIPE ? (s & 1) : 0;
        if (DW_P1_PIPE) { if (s < 7) ld(s + 1, cb ^ 1); }
        else if (s > 0) ld(s, 0);
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
              const float ap = acc1[mt][n][e], bv = acc1[mt][2 + n][e], d = accd[mt][n][e];
              const float t = tanh_approx(ap), sp = 1.f + t, silu = ap * sp;
              vH[k] = silu * bv;
              vB[k] = d * silu;
              vA[k] = d * bv * (0.5f * sp) * fmaf(ap, 1.f - t, 1.f);
            }
            // C element (row g8 + 8 hr, hidden 8 n + 2 q + k) -> [mat][row][hidden] bf16
            const int row = 32 * rg + 16 * mt + 8 * hr + g8, hid = 16 * ps + 8 * n + 2 * q;
            const uint32_t o = swz_h(row, hid >> 3) + (hid & 7) * 2;
            sts32(sh + 0 * G::HT + o, pack_bf16(vH[0], vH[1]));
            sts32(sh + 1 * G::HT + o, pack_bf16(vA[0], vA[1]));
            sts32(sh + 2 * G::HT + o, pack_bf16(vB[0], vB[1]));
          }
    }
    __syncthreads();
    // ---- phase 2: dW blocks over the stage's 64 rows; matrices in the tile order (0 h -> dWs^T, 1 dA -> dWa, 2 dB -> dWb)
    {
      uint32_t abase[3];
      int ahg[3];
#pragma unroll
      for (int m = 0; m < 3; ++m) { abase[m] = sh + bmat[m] * G::HT; ahg[m] = 2 * bhm[m]; }
      // B column half: granule + 8 dh (bit 3, outside the XOR of swz_w, so it is a plain 128 B offset)
      const uint32_t b0 = (bmat[0] == 0 ? sy : sx) + bdh[0] * 128, b1 = (bmat[2] == 0 ? sy : sx) + bdh[2] * 128;
      if (split == 0) dw_phase2<0>(acc, abase, ahg, b0, b1, lane);
      else if (split == 1) dw_phase2<1>(acc, abase, ahg, b0, b1, lane);
      else dw_phase2<2>(acc, abase, ahg, b0, b1, lane);
    }
  }
  cp_async_wait<0>();
  // partial sums: acc[m][J] = hidden 16 bhm + g8 (+ 8) x columns 64 bdh + 8 J + 2 q + e of tile matrix bmat;
  // part layout [rep][dWa | dWb | dWs^T]: tile matrix 0 (h) -> 2, 1 (dA) -> 0, 2 (dB) -> 1
#pragma unroll
  for (int m = 0; m < 3; ++m) {
    const int pm = bmat[m] == 0 ? 2 : bmat[m] - 1;
    float* out = p.part + (((size_t)rr * 3 + pm) * 512 + 64 * sl + 16 * bhm[m]) * 128 + 64 * bdh[m];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr)
        *reinterpret_cast<float2*>(out + (size_t)(8 * hr + g8) * 128 + 8 * j + 2 * q) = make_float2(acc[m][j][2 * hr], acc[m][j][2 * hr + 1]);
  }
}

}  // namespace a100
