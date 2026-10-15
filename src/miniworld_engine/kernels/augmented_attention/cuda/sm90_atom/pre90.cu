// pre90.cu — AdaLN 1, the q | gate projections, the K / V AdaLN and the k | v projections of the AF3 atom block (sm_90a, wgmma).
//
//   x1  = LN(x) * s1 + bi1                         (s1 = mod[:, 0:128], bi1 = mod[:, 128:256])
//   q   = x1 Wq^T + bq        sg = sigmoid(x1 Wg^T)
//   xkv = LN(x1) * sigmoid(mkv[:, 0:128] + bs) + mkv[:, 128:256]
//   k   = xkv Wk^T           v  = xkv Wv^T
// q, k, v only feed MMAs (TF32: rounded to nearest on the way out). SAVE also writes x1 and xkv for the backward.
// SPDX-License-Identifier: Apache-2.0
#include "row90.cuh"
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
      const float sv = sgm ? sigm(s[i][e] + bx[e]) : s[i][e];
      c.f[nt][e] = round_store((c.f[nt][e] - (e < 2 ? m0 : m1)) * (e < 2 ? r0 : r1) * sv + b[i][e]);
    }
  }
}

extern "C" __global__ __launch_bounds__(NTHR, 1) void pre90_fwd(const T* __restrict__ x, const T* __restrict__ mod, const T* __restrict__ mkv,
                                                               const T* __restrict__ w /*[Wq;Wg;Wk;Wv] 512 x 128*/, const float* __restrict__ bq,
                                                               const float* __restrict__ bs, T* __restrict__ q, T* __restrict__ sg,
                                                               T* __restrict__ k, T* __restrict__ v, T* __restrict__ x1s, T* __restrict__ xkvs,
                                                               int M, int nmod, float eps, int save) {
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
  ld_tile(c, x + row0 * 128, 128, 0, g, t);
  float m0, r0, m1, r1;
  ln_stats(c, eps, m0, r0, m1, r1);
  // x1 = LN(x) * s1 + bi1, the modulation read in halves of the row (registers)
  adaln_half<0>(c, mod + mrow * 768, 768, 0, 128, nullptr, m0, r0, m1, r1, g, t);
  adaln_half<1>(c, mod + mrow * 768, 768, 0, 128, nullptr, m0, r0, m1, r1, g, t);
#if TF32
  st_tile(c, x1s + row0 * 128, 128, 0, g, t);  // exact x1: the K / V AdaLN reads it back (and the backward saves it)
  rna(c);
#endif
  for (int u = 0; u < 8; ++u) {
    if (u == 4) {  // c = x1: write it, then the K / V AdaLN of it
#if TF32
      ld_tile(c, x1s + row0 * 128, 128, 0, g, t);
#else
      if (save) st_tile(c, x1s + row0 * 128, 128, 0, g, t);
#endif
      ln_stats(c, eps, m0, r0, m1, r1);
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
                                                                T* __restrict__ x1s, T* __restrict__ xkvs, int M, int nmod, float eps, int save) {
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
  ld_tile(c, x + row0 * 128, 128, 0, g, t);
  float m0, r0, m1, r1;
  ln_stats(c, eps, m0, r0, m1, r1);
  adaln_half<0>(c, mod + mrow * 768, 768, 0, 128, nullptr, m0, r0, m1, r1, g, t);
  adaln_half<1>(c, mod + mrow * 768, 768, 0, 128, nullptr, m0, r0, m1, r1, g, t);
  if (save && u0 == 0) st_tile(c, x1s + row0 * 128, 128, 0, g, t);
  if (u0 >= 4) {  // k, v: xkv = LN(x1) * sigmoid(mkv pre + bs) + mkv shift
    ln_stats(c, eps, m0, r0, m1, r1);
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
    if (mat == 1) st_nt<8, false>(vv, dst + row0 * 128, 128, col, g, t);
    else st_nt<8, true>(vv, dst + row0 * 128, 128, col, g, t);
  }
}
