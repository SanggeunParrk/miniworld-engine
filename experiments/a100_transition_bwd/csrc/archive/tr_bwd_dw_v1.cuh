// tr_bwd_dw_sm80.cuh -- the H100-style weight-gradient role on A100: recompute a, b, dh for one 64-unit hidden slice, then
// dWa_s += dA^T xn, dWb_s += dB^T xn, dWs^T_s += h^T dy -- no intermediate crosses global memory (12 M D H over all slices).
//
// CTA = 8 warps on slice blockIdx % 8 and row replica blockIdx / 8.  The slice's weights stay resident (48 KB); rows come in 64-row stages
// (x | dy, double-buffered cp.async), x is normalised in shared memory (row statistics computed here).
//   phase 1: warp w computes rows 32 (w & 1) .. + 32 x hidden 16 (w >> 1) .. + 16: [a | b] and dh with A fragments streamed from the
//            stage by ldmatrix, the SwiGLU backward, and h | dA | dB as bf16 into a [3][64 rows][64 hidden] shared tile;
//   phase 2: warp w owns 3 of the 24 (matrix, hidden m16, 64-column half) blocks of the slice's [dWa; dWb; dWs^T] (96 f32 accumulators
//            for the whole kernel): A = the transposed h | dA | dB tile (ldmatrix.trans), B = xn or dy rows (ldmatrix.trans).
#pragma once
#include "tr_bwd_w_sm80.cuh"

namespace a100 {

struct DWParams {
  const __nv_bfloat16* x;      // [T][128]
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

// phase 2 of warp w < 6: matrix mat = w / 2 of the tile (0 h -> dWs^T, B = dy; 1 dA, 2 dB -> B = xn), column half dh = w % 2, all four
// hidden m16 tiles: per k-step 4 A + 4 B ldmatrix for 32 MMAs
DEVI void dw_phase2(float (&acc)[4][8][4], uint32_t sb, uint32_t sh, int mat, int dh, int lane) {
  using G = CfgDW;
  const int lr = (((lane >> 3) & 1) << 3) + (lane & 7), lg = lane >> 4;
#pragma unroll
  for (int kk = 0; kk < G::RS / 16; ++kk) {
    const int row = 16 * kk + lr;
    uint32_t a[4][4];
#pragma unroll
    for (int hm = 0; hm < 4; ++hm) {
      // A = tile^T (hidden m16 x rows k16) from [rows][hidden] via .trans: matrices (rows lo, hid lo) (rows hi, hid lo) (rows lo, hid hi)
      // (rows hi, hid hi) transposed are A words 0, 2, 1, 3
      uint32_t t[4];
      ldsm_x4_t(t, sh + mat * G::HT + swz_h(row, 2 * hm + lg));
      a[hm][0] = t[0]; a[hm][1] = t[2]; a[hm][2] = t[1]; a[hm][3] = t[3];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t b[4];
      ldsm_x4_t(b, sb + swz_w(row, 8 * dh + 2 * j + lg));
#pragma unroll
      for (int hm = 0; hm < 4; ++hm) {
        mma16816(acc[hm][2 * j], a[hm], b[0], b[1]);
        mma16816(acc[hm][2 * j + 1], a[hm], b[2], b[3]);
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
      cp_async16_full(buf + m * G::TILE + swz_w(r, gr), (m ? p.dy : p.x) + (size_t)(row0 + r) * 128 + gr * 8);
    }
  };
  // x -> xn in place, the forward's arithmetic: warp w normalises rows 8 w .. 8 w + 7, 8 lanes x 16 elements... (4 lanes per 64 B)
  auto normalise = [&](uint32_t buf) {
#pragma unroll 1
    for (int rw = 0; rw < 8; ++rw) {
      const int r = 8 * warp + rw;
      const int gr = lane >> 1, half = lane & 1;                // lane: granule lane / 2, 4 elements (8 B) half
      const uint32_t a = buf + swz_w(r, gr) + half * 8;
      const uint2 v = lds64(a);
      float xv[4] = {bf16lo(v.x), bf16hi(v.x), bf16lo(v.y), bf16hi(v.y)};
      const float mean = warp_sum(xv[0] + xv[1] + xv[2] + xv[3]) * (1.f / 128);
      float sq = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) { xv[e] -= mean; sq = fmaf(xv[e], xv[e], sq); }
      const float rstd = rsqrtf(warp_sum(sq) * (1.f / 128) + p.eps);
      const int col = gr * 8 + half * 4;
      sts64(a, make_uint2(pack_bf16(fmaf(xv[0] * rstd, sGB[col], sGB[128 + col]), fmaf(xv[1] * rstd, sGB[col + 1], sGB[129 + col])),
                          pack_bf16(fmaf(xv[2] * rstd, sGB[col + 2], sGB[130 + col]), fmaf(xv[3] * rstd, sGB[col + 3], sGB[131 + col]))));
    }
  };

  float acc[4][8][4];
#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[m][j][e] = 0.f;
  const int rg = warp & 1, ps = warp >> 1;                      // phase 1: row group, 16-unit step of the slice
  const int mat = warp >> 1, dh = warp & 1;                     // phase 2 (warps 0-5)
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
    normalise(buf);
    __syncthreads();
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
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t fx[2][4], fy[2][4], ba[4], bb[4], bd[4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          ldsm_x4(fx[mt], sx + swz_w(ar + 16 * mt, 2 * s + ag));
          ldsm_x4(fy[mt], sy + swz_w(ar + 16 * mt, 2 * s + ag));
        }
        // the stage holds plain columns; the resident weights are in the same plain k order (see pack)
        ldsm_x4(ba, s_u + w1_off + s * 4096);
        ldsm_x4(bb, s_u + w1_off + 256 + s * 4096);
        ldsm_x4(bd, s_u + w3_off + s * 2048);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          // ldmatrix fragment words: [0] rows 0-7 k lo, [1] rows 8-15 k lo, [2] rows 0-7 k hi, [3] rows 8-15 k hi = A order
          mma16816(acc1[mt][0], fx[mt], ba[0], ba[1]);
          mma16816(acc1[mt][1], fx[mt], ba[2], ba[3]);
          mma16816(acc1[mt][2], fx[mt], bb[0], bb[1]);
          mma16816(acc1[mt][3], fx[mt], bb[2], bb[3]);
          mma16816(accd[mt][0], fy[mt], bd[0], bd[1]);
          mma16816(accd[mt][1], fy[mt], bd[2], bd[3]);
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
    if (warp < 6) dw_phase2(acc, mat == 0 ? sy : sx, sh, mat, dh, lane);
  }
  cp_async_wait<0>();
  // partial sums: acc[hm][J] = hidden 16 hm + g8 (+ 8) x columns 64 dh + 8 J + 2 q + e of tile matrix mat;
  // part layout [rep][dWa | dWb | dWs^T]: tile matrix 0 (h) -> 2, 1 (dA) -> 0, 2 (dB) -> 1
  if (warp < 6) {
    const int pm = mat == 0 ? 2 : mat - 1;
#pragma unroll
    for (int hm = 0; hm < 4; ++hm) {
      float* out = p.part + (((size_t)rr * 3 + pm) * 512 + 64 * sl + 16 * hm) * 128 + 64 * dh;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int hr = 0; hr < 2; ++hr)
          *reinterpret_cast<float2*>(out + (size_t)(8 * hr + g8) * 128 + 8 * j + 2 * q) = make_float2(acc[hm][j][2 * hr], acc[hm][j][2 * hr + 1]);
    }
  }
}

}  // namespace a100
