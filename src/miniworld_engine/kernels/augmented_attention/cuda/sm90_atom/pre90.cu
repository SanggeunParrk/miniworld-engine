// pre90.cu — AdaLN 1, the q | gate projections, the K / V AdaLN and the k | v projections of the AF3 atom block (sm_90a, wgmma).
//
//   x1  = LN(x) * s1 + bi1                         (s1 = mod[:, 0:128], bi1 = mod[:, 128:256])
//   q   = x1 Wq^T + bq        sg = sigmoid(x1 Wg^T)
//   xkv = LN(x1) * sigmoid(mkv[:, 0:128] + bs) + mkv[:, 128:256]
//   k   = xkv Wk^T           v  = xkv Wv^T
// q, k, v only feed MMAs (TF32: rounded to nearest on the way out). SAVE also writes x1 and xkv for the backward.
// PAR (AlphaFold 3's K / V norm): xkv = LN(x) * sigmoid(...) + ..., the second AdaLN of the raw x next to the first, not of x1 (its backward: pre_bwd90_par;
// the chained one is pre_bwd90). par 2: ONE AdaLN for q and k / v (Boltz-2, RF3, Chai-1): xkv = x1 (not written; its backward: pre_bwd90_sh).
// SPDX-License-Identifier: Apache-2.0
#include "rms90.cuh"
using namespace r90;

// c <- (c - mean) rstd * S + B for n-tiles 8 H .. 8 H + 7, S = sigmoid(p[:, scol ..] + sbias) (sbias != null) or the stored scale as is,
// B = p[:, bcol ..]; the loads of a half are issued together
template <int H>
DEVI void adaln_half(Tile& c, const T* p, int ld, int scol, int bcol, const float* sbias, float m0, float r0, float m1, float r1, int g, int t) {
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
#define KQLN 0   // 1: RoseTTAFold3's affine LayerNorm of the flat q / k (over all 128 columns), after their projections
#endif

// KQLN: the two 64-column units of q (mat 0) or k (mat 2) are held in registers; at the second, LN(.) w + b of the whole row is stored for
// both halves (kq = [wq | bq | wk | bk] fp32 [4][128]; the raw projections go to qkr [M, 256] for the backward). -> true: the unit is done.
DEVI bool kq_ln_unit(Tile& hold, float (*v)[4], int mat, int h, const float* kq, float kq_eps, T* dst, T* qkr, size_t row0, int g, int t) {
  if (mat != 0 && mat != 2) return false;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) hold.f[8 * h + nt][e] = v[nt][e];
  if (qkr) st_nt<8, false>(v, qkr + row0 * 256, 256, (mat == 2 ? 128 : 0) + 64 * h, g, t);
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

extern "C" __global__ __launch_bounds__(NTHR, 1) void pre90_fwd(const T* __restrict__ x, const T* __restrict__ mod, const T* __restrict__ mkv,
                                                               const T* __restrict__ w /*[Wq;Wg;Wk;Wv] 512 x 128*/, const float* __restrict__ bq,
                                                               const float* __restrict__ bs, T* __restrict__ q, T* __restrict__ sg,
                                                               T* __restrict__ k, T* __restrict__ v, T* __restrict__ x1s, T* __restrict__ xkvs,
                                                               int M, int nmod, float eps, int save, int par,
                                                               const float* __restrict__ kq, float kq_eps, T* __restrict__ qkr) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int UNIT = 64 * 128 * ES;
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const size_t row0 = (size_t)blockIdx.x * ROWS + warp * 16;
  const size_t mrow = row0 % (size_t)nmod;   // rows of the modulation tables (nmod = M: per sample; = N: one conditioning shared by the samples)
  auto issue = [&](int u) { load_w(base + (u & 1) * UNIT, w + (size_t)u * 64 * 128, 128, 64, 128, tid, NTHR); cp_commit(); };
  issue(0);
  issue(1);
  Tile c;
  Tile hold;   // KQLN: the q / k row of the two 64-column units
  (void)hold;
  ld_tile(c, x + row0 * 128, 128, 0, g, t);
  float m0, r0, m1, r1;
  norm_stats(c, eps, m0, r0, m1, r1);
  // x1 = LN(x) * s1 + bi1, the modulation read in halves of the row (registers)
  adaln_half<0>(c, mod + mrow * 768, 768, 0, 128, nullptr, m0, r0, m1, r1, g, t);
  adaln_half<1>(c, mod + mrow * 768, 768, 0, 128, nullptr, m0, r0, m1, r1, g, t);
#if TF32
  st_tile(c, x1s + row0 * 128, 128, 0, g, t);  // exact x1: the K / V AdaLN reads it back (and the backward saves it)
  rna(c);
#endif
  for (int u = 0; u < 8; ++u) {
    if (u == 4 && par == 2) {  // one AdaLN for q and k / v (Boltz-2 / RF3 / Chai-1): k, v read x1 (c holds it, as the MMAs read it)
#if !TF32
      if (save) st_tile(c, x1s + row0 * 128, 128, 0, g, t);
#endif
    } else if (u == 4) {  // c = x1: write it, then the K / V AdaLN of it
#if TF32
      ld_tile(c, x1s + row0 * 128, 128, 0, g, t);
#else
      if (save) st_tile(c, x1s + row0 * 128, 128, 0, g, t);
#endif
      if (par) ld_tile(c, x + row0 * 128, 128, 0, g, t);  // AlphaFold 3: the K / V AdaLN reads the raw x, not x1
      norm_stats(c, eps, m0, r0, m1, r1);
      adaln_half<0>(c, mkv + mrow * 256, 256, 0, 128, bs, m0, r0, m1, r1, g, t);
      adaln_half<1>(c, mkv + mrow * 256, 256, 0, 128, bs, m0, r0, m1, r1, g, t);
      if (save) st_tile(c, xkvs + row0 * 128, 128, 0, g, t);
      rna(c);
    }
    cp_wait<1>();
    proxy_fence();
    __syncthreads();
    float acc[32];
    wg_gemm<64, KSTEPS>(acc, c, base + (u & 1) * UNIT, 0);
    __syncthreads();
    if (u + 2 < 8) issue(u + 2); else cp_commit();
    const int mat = u >> 1, col = 64 * (u & 1);
    T* dst = mat == 0 ? q : mat == 1 ? sg : mat == 2 ? k : v;
    float v[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) v[nt][e] = acc[4 * nt + e];
      if (mat == 0) {
        const float2 bb = colv(bq + col, nt, t);
        v[nt][0] += bb.x; v[nt][1] += bb.y; v[nt][2] += bb.x; v[nt][3] += bb.y;
      } else if (mat == 1) {
#pragma unroll
        for (int e = 0; e < 4; ++e) v[nt][e] = sigm(v[nt][e]);
      }
    }
    if (KQLN && kq_ln_unit(hold, v, mat, u & 1, kq, kq_eps, dst, save ? qkr : nullptr, row0, g, t)) continue;
    if (mat == 1) st_nt<8, false>(v, dst + row0 * 128, 128, col, g, t);
    else st_nt<8, true>(v, dst + row0 * 128, 128, col, g, t);
  }
}

// Small row counts: grid.y splits the 8 projection units (q lo / hi, g lo / hi, k lo / hi, v lo / hi) over CTAs. A CTA computes AdaLN 1 (and the
// K / V AdaLN when its units are k or v) in registers and runs its units; the saved x1 / xkv are written by the CTA of the first unit that
// reads them. 2, 4 or 8 CTAs per tile: a CTA's units never mix x1 and xkv operands.
extern "C" __global__ __launch_bounds__(NTHR, 1) void pre90_split(const T* __restrict__ x, const T* __restrict__ mod, const T* __restrict__ mkv,
                                                                const T* __restrict__ w, const float* __restrict__ bq, const float* __restrict__ bs,
                                                                T* __restrict__ q, T* __restrict__ sg, T* __restrict__ k, T* __restrict__ v,
                                                                T* __restrict__ x1s, T* __restrict__ xkvs, int M, int nmod, float eps, int save, int par,
                                                               const float* __restrict__ kq, float kq_eps, T* __restrict__ qkr) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int UNIT = 64 * 128 * ES;
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const size_t row0 = (size_t)blockIdx.x * ROWS + warp * 16;
  const size_t mrow = row0 % (size_t)nmod;
  const int upc = 8 / (int)gridDim.y, u0 = (int)blockIdx.y * upc;
  auto issue = [&](int ul) { load_w(base + (ul & 1) * UNIT, w + (size_t)(u0 + ul) * 64 * 128, 128, 64, 128, tid, NTHR); cp_commit(); };
  issue(0);
  if (upc > 1) issue(1); else cp_commit();
  Tile c;
  Tile hold;   // KQLN: the q / k row of the two 64-column units
  (void)hold;
  ld_tile(c, x + row0 * 128, 128, 0, g, t);
  float m0, r0, m1, r1;
  norm_stats(c, eps, m0, r0, m1, r1);
  adaln_half<0>(c, mod + mrow * 768, 768, 0, 128, nullptr, m0, r0, m1, r1, g, t);
  adaln_half<1>(c, mod + mrow * 768, 768, 0, 128, nullptr, m0, r0, m1, r1, g, t);
  if (save && u0 == 0) st_tile(c, x1s + row0 * 128, 128, 0, g, t);
  if (u0 >= 4 && par != 2) {  // k, v: xkv = LN(x1) * sigmoid(mkv pre + bs) + mkv shift (PAR: LN(x); par 2: x1 itself)
    if (par) ld_tile(c, x + row0 * 128, 128, 0, g, t);
    norm_stats(c, eps, m0, r0, m1, r1);
    adaln_half<0>(c, mkv + mrow * 256, 256, 0, 128, bs, m0, r0, m1, r1, g, t);
    adaln_half<1>(c, mkv + mrow * 256, 256, 0, 128, bs, m0, r0, m1, r1, g, t);
    if (save && u0 == 4) st_tile(c, xkvs + row0 * 128, 128, 0, g, t);
  }
  rna(c);
#pragma unroll 1
  for (int ul = 0; ul < upc; ++ul) {
    const int u = u0 + ul;
    cp_wait<1>();
    proxy_fence();
    __syncthreads();
    float acc[32];
    wg_gemm<64, KSTEPS>(acc, c, base + (ul & 1) * UNIT, 0);
    __syncthreads();
    if (ul + 2 < upc) issue(ul + 2); else cp_commit();
    const int mat = u >> 1, col = 64 * (u & 1);
    T* dst = mat == 0 ? q : mat == 1 ? sg : mat == 2 ? k : v;
    float vv[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) vv[nt][e] = acc[4 * nt + e];
      if (mat == 0) {
        const float2 bb = colv(bq + col, nt, t);
        vv[nt][0] += bb.x; vv[nt][1] += bb.y; vv[nt][2] += bb.x; vv[nt][3] += bb.y;
      } else if (mat == 1) {
#pragma unroll
        for (int e = 0; e < 4; ++e) vv[nt][e] = sigm(vv[nt][e]);
      }
    }
    if (KQLN && kq_ln_unit(hold, vv, mat, u & 1, kq, kq_eps, dst, save ? qkr : nullptr, row0, g, t)) continue;
    if (mat == 1) st_nt<8, false>(vv, dst + row0 * 128, 128, col, g, t);
    else st_nt<8, true>(vv, dst + row0 * 128, 128, col, g, t);
  }
}
