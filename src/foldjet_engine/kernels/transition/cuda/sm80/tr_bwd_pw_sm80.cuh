// tr_bwd_pw_sm80.cuh -- two-kernel backward, kernel 1 (PW): the DW role (tr_bwd_dw_sm80.cuh) fed with x instead of xn, and handing
// dA | dB to the X kernel (tr_bwd_x_kernel) as fragment-native blocks.  Every product once: a, b, dh of the slice (P's 6 M D H, split
// over the 8 hidden slices) and the slice's dWa, dWb, dWs^T (6 M D H); h never leaves the SM.
//
// CTA = 8 warps on slice blockIdx % 8, row replica blockIdx / 8.  Per 64-row stage (x | dy):
//   LN: 4 threads per row (thread q of a quad owns the row's columns 32 i + 8 q + 0..7, the forward's assignment and summation order,
//       so mean / rstd / xn are P's bits); slice 0 writes (mean, rstd) for X; xn replaces x in place;
//   phase 1 and phase 2 as the DW role; between them each warp copies 4 of the stage's 32 (dA | dB, 16 x 16) blocks from the
//   [3][64 rows][64 hidden] tile to global memory (ldmatrix = the A fragment, one 512 B store).
#pragma once
#include "tr_bwd_dw_sm80.cuh"

namespace a100 {

struct PWParams {
  const __nv_bfloat16* x;      // [T][128]
  const __nv_bfloat16* dy;     // [T][128]
  const __nv_bfloat16* wdw;    // as DWParams
  const float* gamma;
  const float* beta;
  float2* stats;               // out: [T] (mean, rstd)
  uint4* ab;                   // out: [T / 16][32 K][dA | dB | (h, unused)][32 lanes] (the X kernel's layout)
  float* part;                 // [nrep][3][512][128] f32
  int T;
  float eps;
};

// PW_ADDR: ldmatrix addresses as base ^ (step << 5) + immediate.  A swizzled granule (G ^ (r & 7)) << 4 with G = 2 s + g0 (g0 < 2)
// splits into ((r & 6 | (g0 ^ r) & 1) << 4) ^ (s << 5): one XOR per step instead of the compiler's per-load re-derivation.
// Needs the shared window's base to have bits 4-7 clear (checked once).
#ifndef PW_ADDR
#define PW_ADDR 1
#endif
// PW_2DA: the dA tile holds 2 dA (one multiply less per element); the X kernel's Wa and the dWa sum carry the 0.5
#ifndef PW_2DA
#define PW_2DA 1
#endif
template <int SPLIT>
DEVI void pw_phase2(float (&acc)[3][8][4], const uint32_t (&A)[3], uint32_t B0, uint32_t B1) {
#pragma unroll
  for (int kk = 0; kk < CfgDW::RS / 16; ++kk) {
    uint32_t a[3][4];
#pragma unroll
    for (int m = 0; m < 3; ++m) ldsm_x4_t(a[m], A[m] + kk * 16 * 128);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t b0[4], b1[4];
      ldsm_x4_t(b0, (B0 ^ (j << 5)) + kk * 16 * 256);
      if (SPLIT) ldsm_x4_t(b1, (B1 ^ (j << 5)) + kk * 16 * 256);
#pragma unroll
      for (int m = 0; m < 3; ++m) {
        const uint32_t* b = (SPLIT && m >= SPLIT) ? b1 : b0;
        mma16816(acc[m][2 * j], a[m], b[0], b[1]);
        mma16816(acc[m][2 * j + 1], a[m], b[2], b[3]);
      }
    }
  }
}

DEVI void pw_role(const PWParams& p, uint8_t* smem, int sl, int rr, int nrep) {
  using G = CfgDW;
  constexpr int D = 128;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q = lane & 3;
  const int nst = p.T / G::RS, n_mine = nst > rr ? (nst - rr + nrep - 1) / nrep : 0;
  const uint32_t s_u = smem_u32(smem);
  float* sGB = reinterpret_cast<float*>(smem + G::OFF_GB);
  for (int k = tid; k < 128; k += G::NTHR) { sGB[k] = p.gamma[k]; sGB[128 + k] = p.beta[k]; }
  {
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
  const uint32_t sGBu = s_u + G::OFF_GB;
  // x -> xn in place: row tid / 4, quad thread q, granules 4 i + q (the forward's LayerNorm arithmetic)
  auto layernorm = [&](int i) {
    const int r = tid >> 2, row0 = (rr + i * nrep) * G::RS;
    const uint32_t sx = s_u + G::OFF_ST + (i % G::NSTAGE) * G::STAGE;
    float xv[32];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint4 v = lds128(sx + swz_w(r, 4 * k + q));
      const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) { xv[8 * k + 2 * e] = bf16lo(wv[e]); xv[8 * k + 2 * e + 1] = bf16hi(wv[e]); }
    }
    float sm = 0.f;
#pragma unroll
    for (int e = 0; e < 32; ++e) sm += xv[e];
    const float mean = quad_sum(sm) * (1.f / D);
    float sq = 0.f;
#pragma unroll
    for (int e = 0; e < 32; ++e) { xv[e] -= mean; sq = fmaf(xv[e], xv[e], sq); }
    const float rstd = rsqrtf(quad_sum(sq) * (1.f / D) + p.eps);
    if (sl == 0 && q == 0) p.stats[row0 + r] = make_float2(mean, rstd);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint32_t col = 32 * k + 8 * q;
      const uint4 g0 = lds128(sGBu + col * 4), g1 = lds128(sGBu + col * 4 + 16);
      const uint4 b0 = lds128(sGBu + 512 + col * 4), b1 = lds128(sGBu + 512 + col * 4 + 16);
      const uint32_t gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w}, bw[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        o[e] = pack_bf16(fmaf(xv[8 * k + 2 * e] * rstd, __uint_as_float(gw[2 * e]), __uint_as_float(bw[2 * e])),
                         fmaf(xv[8 * k + 2 * e + 1] * rstd, __uint_as_float(gw[2 * e + 1]), __uint_as_float(bw[2 * e + 1])));
      sts128(sx + swz_w(r, 4 * k + q), make_uint4(o[0], o[1], o[2], o[3]));
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
  const int gm[6] = {0, 0, 1, 2, 1, 2}, gd[6] = {0, 1, 0, 0, 1, 1};
  int bmat[3], bhm[3], bdh[3];
#pragma unroll
  for (int m = 0; m < 3; ++m) { const int k = 3 * warp + m; bmat[m] = gm[k >> 2]; bdh[m] = gd[k >> 2]; bhm[m] = k & 3; }
  const int split = (warp == 1 || warp == 5) ? 1 : warp == 2 ? 2 : 0;
  const int lrow = ((lane >> 4) << 3) + (lane & 7), lgr = (lane >> 3) & 1;
  const uint32_t w1_off = lgr * 2048 + lrow * 16 + ps * 512, w3_off = G::W1 + lgr * 1024 + lrow * 16 + ps * 256;
  const int ar = rg * 32 + (((lane >> 3) & 1) << 3) + (lane & 7), ag = lane >> 4;
  const uint32_t sh = s_u + G::OFF_H;
#if PW_ADDR
  if (s_u & 0xf0u) __trap();
  const uint32_t r7 = lane & 7;
  const uint32_t x0off = ar * 256 + ((((r7 & 6) | ((ag ^ r7) & 1))) << 4);          // phase 1 A: + (s << 5) by XOR, + 4096 mt
  const uint32_t e0 = sh + (32 * rg + g8) * 128 + 4 * q;                              // epilogue: + ((2 ps + n) ^ g8) << 4
  const uint32_t eo[2] = {e0 + ((((2 * ps) ^ g8)) << 4), e0 + ((((2 * ps + 1) ^ g8)) << 4)};
  const int ar2 = (((lane >> 4) & 1) << 3) + (lane & 7), ag2 = (lane >> 3) & 1;
  const int br2 = (((lane >> 3) & 1) << 3) + (lane & 7), bg2 = lane >> 4;
  uint32_t pA[3];
#pragma unroll
  for (int m = 0; m < 3; ++m) pA[m] = sh + bmat[m] * G::HT + ar2 * 128 + (((2 * bhm[m]) | ag2) ^ r7) * 16;
  const uint32_t boff = br2 * 256 + ((((r7 & 6) | ((bg2 ^ r7) & 1))) << 4);
#endif
  // block copy-out: warp's blocks 4 warp + j -> matrix (dA, dB) = warp / 4, row block (4 warp + j) / 4 % 4, k-block j
  const int cm = warp >> 2, crb = warp & 3, car = (((lane >> 3) & 1) << 3) + (lane & 7), cag = lane >> 4;

  if (n_mine > 0) load_stage(0);
  cp_async_commit();
#pragma unroll 1
  for (int i = 0; i < n_mine; ++i) {
    cp_async_wait<0>();
    __syncthreads();                                            // stage i landed; stage i - 1 retired
    if (i + 1 < n_mine) load_stage(i + 1);
    cp_async_commit();
#ifndef PW_XN
    layernorm(i);
    __syncthreads();
#endif
    const uint32_t buf = s_u + G::OFF_ST + (i % G::NSTAGE) * G::STAGE, sx = buf, sy = buf + G::TILE;
    const int row0 = (rr + i * nrep) * G::RS;
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
      uint32_t fx[2][2][4], fy[2][2][4], ba[2][4], bb[2][4], bd[2][4];
      auto ld = [&](int s, int cb) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#if PW_ADDR
          const uint32_t xa = ((sx + x0off) ^ (s << 5)) + mt * 4096;
          ldsm_x4(fx[cb][mt], xa);
          ldsm_x4(fy[cb][mt], xa + G::TILE);
#else
          ldsm_x4(fx[cb][mt], sx + swz_w(ar + 16 * mt, 2 * s + ag));
          ldsm_x4(fy[cb][mt], sy + swz_w(ar + 16 * mt, 2 * s + ag));
#endif
        }
        ldsm_x4(ba[cb], s_u + w1_off + s * 4096);
        ldsm_x4(bb[cb], s_u + w1_off + 256 + s * 4096);
        ldsm_x4(bd[cb], s_u + w3_off + s * 2048);
      };
      ld(0, 0);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int cb = s & 1;
        if (s < 7) ld(s + 1, cb ^ 1);
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
#if PW_2DA
              vA[k] = d * bv * fmaf(-silu, t, silu + sp);             // 2 dA = d b (2 sig + 2 silu (1 - sig))
#else
              vA[k] = d * bv * (0.5f * sp) * fmaf(ap, 1.f - t, 1.f);
#endif
            }
#if PW_ADDR
            const uint32_t o = eo[n] - sh + (16 * mt + 8 * hr) * 128;
#else
            const int row = 32 * rg + 16 * mt + 8 * hr + g8, hid = 16 * ps + 8 * n + 2 * q;
            const uint32_t o = swz_h(row, hid >> 3) + (hid & 7) * 2;
#endif
            sts32(sh + 0 * G::HT + o, pack_bf16(vH[0], vH[1]));
            sts32(sh + 1 * G::HT + o, pack_bf16(vA[0], vA[1]));
            sts32(sh + 2 * G::HT + o, pack_bf16(vB[0], vB[1]));
          }
    }
    __syncthreads();
    {                                                           // dA | dB blocks for X
      uint4* const dst = p.ab + ((size_t)(row0 / 16 + crb) * 32 + 4 * sl) * 96 + cm * 32 + lane;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t a[4];
        ldsm_x4(a, sh + (1 + cm) * G::HT + swz_h(16 * crb + car, 2 * j + cag));
        stg128(dst + j * 96, make_uint4(a[0], a[1], a[2], a[3]));
      }
    }
    {
      uint32_t abase[3];
      int ahg[3];
#pragma unroll
      for (int m = 0; m < 3; ++m) { abase[m] = sh + bmat[m] * G::HT; ahg[m] = 2 * bhm[m]; }
      const uint32_t b0 = (bmat[0] == 0 ? sy : sx) + bdh[0] * 128, b1 = (bmat[2] == 0 ? sy : sx) + bdh[2] * 128;
#if PW_ADDR
      (void)abase; (void)ahg;
      if (split == 0) pw_phase2<0>(acc, pA, b0 + boff, b1 + boff);
      else if (split == 1) pw_phase2<1>(acc, pA, b0 + boff, b1 + boff);
      else pw_phase2<2>(acc, pA, b0 + boff, b1 + boff);
#else
      if (split == 0) dw_phase2<0>(acc, abase, ahg, b0, b1, lane);
      else if (split == 1) dw_phase2<1>(acc, abase, ahg, b0, b1, lane);
      else dw_phase2<2>(acc, abase, ahg, b0, b1, lane);
#endif
    }
  }
  cp_async_wait<0>();
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

__global__ void __launch_bounds__(256, 1) tr_bwd_pw_kernel(const PWParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  pw_role(p, smem, blockIdx.x & 7, blockIdx.x >> 3, gridDim.x >> 3);
}

}  // namespace a100
