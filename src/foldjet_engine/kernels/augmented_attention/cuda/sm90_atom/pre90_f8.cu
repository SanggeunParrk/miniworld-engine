// pre90_f8.cu — EXPERIMENTAL (opt-in, inference only, bf16): pre90.cu's forward with the q | gate | k | v projections on e4m3 wgmma
// (f8w90.cuh). Same math and outputs as pre90_fwd / pre90_split without SAVE (KQLN: -DKQLN=1, the flat q / k LayerNorm of RoseTTAFold3):
//
//   x1  = LN(x) * s1 + bi1                 q = x1 Wq^T + bq      sg = sigmoid(x1 Wg^T)
//   xkv = LN(x1 | x) * sigmoid(mkv[:, 0:128] + bs) + mkv[:, 128:256]  (par 0 / 1; par 2: xkv = x1)      k = xkv Wk^T     v = xkv Wv^T
//
// w8: [512, 128] e4m3 (the bf16 pack's rows, each 64-row unit scaled by a power of two, K permuted per k32 step), wsi [8]: the inverse unit
// scales. Both A operands are quantized once (per-row power-of-two scales), so the fp32 tile is dead before the products.
// SPDX-License-Identifier: Apache-2.0
#include "f8w90.cuh"
using namespace r90;

template <int H>
DEVI void adaln_half8(Tile& c, const T* p, int ld, int scol, int bcol, const float* sbias, float m0, float r0, float m1, float r1, int g, int t) {
  float s[8][4], b[8][4];
  ld_nt<8>(s, p, ld, scol + 64 * H, g, t);
  ld_nt<8>(b, p, ld, bcol + 64 * H, g, t);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int nt = 8 * H + i;
    float2 bb = make_float2(0.f, 0.f);
    if (sbias) bb = colv(sbias, nt, t);
    const bool sgm = sbias != nullptr;
    const float bx[4] = {bb.x, bb.y, bb.x, bb.y};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float sv = sgm ? cond_act(s[i][e] + bx[e]) : s[i][e];
      c.f[nt][e] = round_store((c.f[nt][e] - (e < 2 ? m0 : m1)) * (e < 2 ? r0 : r1) * sv + b[i][e]);
    }
  }
}

#ifndef KQLN
#define KQLN 0   // 1: RoseTTAFold3's affine LayerNorm of the flat q / k (over all 128 columns), after their projections (as pre90.cu)
#endif

// KQLN (same math as pre90.cu's kq_ln_unit): the two 64-column units of q (mat 0) or k (mat 2) are held in registers; at the second,
// LN(.) w + b of the whole row is stored for both halves (kq = [wq | bq | wk | bk] fp32 [4][128]). -> true: the unit is done. The grid splits
// the units over at most 4 CTAs per tile, so both units of q (and of k) sit in one CTA.
DEVI bool kq_ln_unit8(Tile& hold, float (*v)[4], int mat, int h, const float* kq, float kq_eps, T* dst, size_t row0, int g, int t) {
  if (mat != 0 && mat != 2) return false;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) hold.f[8 * h + nt][e] = v[nt][e];
  if (h == 0) return true;
  float m0, r0, m1, r1;
  ln_stats(hold, kq_eps, m0, r0, m1, r1);
  const float* w = kq + (mat == 2 ? 256 : 0);
  const float* b = w + 128;
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    float o[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const float2 ww = colv(w + 64 * hh, nt, t), bb = colv(b + 64 * hh, nt, t);
#pragma unroll
      for (int e = 0; e < 4; ++e)
        o[nt][e] = (hold.f[8 * hh + nt][e] - (e < 2 ? m0 : m1)) * (e < 2 ? r0 : r1) * ((e & 1) ? ww.y : ww.x) + ((e & 1) ? bb.y : bb.x);
    }
    st_nt<8, true>(o, dst + row0 * 128, 128, 64 * hh, g, t);
  }
  return true;
}

// the 64 output columns of unit u (0 .. 7: q lo / hi, g lo / hi, k lo / hi, v lo / hi); f0 / f1: the rows' dequantization factors
DEVI void unit_out8(const float* acc, int u, float f0, float f1, const float* bq, T* q, T* sg, T* k, T* v, size_t row0, int g, int t, Tile& hold,
                    const float* kq, float kq_eps) {
  const int mat = u >> 1, col = 64 * (u & 1);
  T* dst = mat == 0 ? q : mat == 1 ? sg : mat == 2 ? k : v;
  float o[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    o[nt][0] = acc[4 * nt + 0] * f0;
    o[nt][1] = acc[4 * nt + 1] * f0;
    o[nt][2] = acc[4 * nt + 2] * f1;
    o[nt][3] = acc[4 * nt + 3] * f1;
    if (mat == 0) {
      const float2 bb = colv(bq + col, nt, t);
      o[nt][0] += bb.x; o[nt][1] += bb.y; o[nt][2] += bb.x; o[nt][3] += bb.y;
    } else if (mat == 1) {
#pragma unroll
      for (int e = 0; e < 4; ++e) o[nt][e] = sigm(o[nt][e]);
    }
  }
  if (KQLN && kq_ln_unit8(hold, o, mat, u & 1, kq, kq_eps, dst, row0, g, t)) return;
  if (mat == 1) st_nt<8, false>(o, dst + row0 * 128, 128, col, g, t);
  else st_nt<8, true>(o, dst + row0 * 128, 128, col, g, t);
}

// grid (tiles, units split): gridDim.y = 1 runs the 8 units, 2 / 4 / 8 split them; a CTA runs its x1 units, then its xkv units (two loops,
// each operand statically indexed)
extern "C" __global__ __launch_bounds__(NTHR, MINB) void pre90_f8(const T* __restrict__ x, const T* __restrict__ mod, const T* __restrict__ mkv,
                                                                 const uint8_t* __restrict__ w8, const float* __restrict__ wsi,
                                                                 const float* __restrict__ bq, const float* __restrict__ bs, T* __restrict__ q,
                                                                 T* __restrict__ sg, T* __restrict__ k, T* __restrict__ v, int M, int nmod,
                                                                 float eps, int par, const float* __restrict__ kq, float kq_eps) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int UNIT = 64 * 128;  // e4m3 bytes of one 64-row unit
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const size_t row0 = (size_t)blockIdx.x * ROWS + warp * 16;
  const size_t mrow = row0 % (size_t)nmod;
  const int upc = 8 / (int)gridDim.y, u0 = (int)blockIdx.y * upc;
  auto issue = [&](int ul) { load_w8(base + (ul & 1) * UNIT, w8 + (size_t)(u0 + ul) * 64 * 128, 128, 64, 128, tid, NTHR); cp_commit(); };
  issue(0);
  if (upc > 1) issue(1); else cp_commit();
  // units [u0, uq) read x1, [uq, u0 + upc) read xkv (par 2: every unit reads x1)
  const int uq = par == 2 ? u0 + upc : max(u0, min(4, u0 + upc));
  uint32_t aq[KSTEPS8][4], akv[KSTEPS8][4];
  float iq0 = 1.f, iq1 = 1.f, ik0 = 1.f, ik1 = 1.f;
  Tile hold;  // KQLN: the q / k row of the two 64-column units
  (void)hold;
  {
    Tile c;
    ld_tile(c, x + row0 * 128, 128, 0, g, t);
    float m0, r0, m1, r1;
    norm_stats(c, eps, m0, r0, m1, r1);
    if (uq > u0 || par == 0) {  // x1 = LN(x) * s1 + bi1 (par 0 chains the K / V AdaLN on it)
      adaln_half8<0>(c, mod + mrow * 768, 768, 0, 128, nullptr, m0, r0, m1, r1, g, t);
      adaln_half8<1>(c, mod + mrow * 768, 768, 0, 128, nullptr, m0, r0, m1, r1, g, t);
    }
    float s0, s1;
    if (uq > u0) {
      row_scales<16>(c.f, s0, s1);
      quant8<KSTEPS8>(c.f, s0, s1, aq);
      iq0 = inv_pow2(s0);
      iq1 = inv_pow2(s1);
    }
    if (uq < u0 + upc) {  // xkv = LN(x1 | x) * sigmoid(mkv pre + bs) + mkv shift
      if (par == 1) ld_tile(c, x + row0 * 128, 128, 0, g, t);
      norm_stats(c, eps, m0, r0, m1, r1);
      adaln_half8<0>(c, mkv + mrow * 256, 256, 0, 128, bs, m0, r0, m1, r1, g, t);
      adaln_half8<1>(c, mkv + mrow * 256, 256, 0, 128, bs, m0, r0, m1, r1, g, t);
      row_scales<16>(c.f, s0, s1);
      quant8<KSTEPS8>(c.f, s0, s1, akv);
      ik0 = inv_pow2(s0);
      ik1 = inv_pow2(s1);
    }
  }
#pragma unroll 1
  for (int u = u0; u < uq; ++u) {
    const int ul = u - u0;
    cp_wait<1>();
    proxy_fence();
    __syncthreads();
    float acc[32];
    wg_gemm8q<64, KSTEPS8>(acc, aq, base + (ul & 1) * UNIT, 0);
    __syncthreads();
    if (ul + 2 < upc) issue(ul + 2); else cp_commit();
    const float ws = __ldg(wsi + u);
    unit_out8(acc, u, iq0 * ws, iq1 * ws, bq, q, sg, k, v, row0, g, t, hold, kq, kq_eps);
  }
#pragma unroll 1
  for (int u = uq; u < u0 + upc; ++u) {
    const int ul = u - u0;
    cp_wait<1>();
    proxy_fence();
    __syncthreads();
    float acc[32];
    wg_gemm8q<64, KSTEPS8>(acc, akv, base + (ul & 1) * UNIT, 0);
    __syncthreads();
    if (ul + 2 < upc) issue(ul + 2); else cp_commit();
    const float ws = __ldg(wsi + u);
    unit_out8(acc, u, ik0 * ws, ik1 * ws, bq, q, sg, k, v, row0, g, t, hold, kq, kq_eps);
  }
}
