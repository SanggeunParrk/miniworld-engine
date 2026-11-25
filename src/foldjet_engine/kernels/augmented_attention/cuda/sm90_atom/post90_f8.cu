// post90_f8.cu — EXPERIMENTAL (opt-in, inference only, bf16): post90.cu's forward with the out-projection, the SwiGLU [a | b] projection and
// the squeeze on e4m3 wgmma (f8w90.cuh). Same math and output as post90_fwd without SAVE / UP:
//
//   u   = (o * sg) Wo^T            a2  = a + so * u          x2 = LN(a2) * s2 + bi2     (PARS: LN(a))
//   h_j = silu(x2 Wa_j^T) * (x2 Wb_j^T)       t = sum_j h_j Wd_j^T          out = a2 + st * t
//
// Weights (e4m3, scaled by powers of two, K permuted per k32 step): wo8 [128, 128], wab8 [4][128, 128] (chunk j = [Wa_j; Wb_j]), wd8 [4][128, 64]
// (K = the chunk's 64 hidden); wsi [11] the inverse scales: Wo units 0, 1; [Wa_j | Wb_j] 2 + 2 j, 3 + 2 j; Wd 10. h_j is scaled per row and
// chunk; the squeeze accumulator is rescaled by the ratio of the chunk scales before each chunk's product (powers of two: exact).
// SPDX-License-Identifier: Apache-2.0
#include "f8w90.cuh"
using namespace r90;
#ifndef NOOUT
#define NOOUT 0
#endif
#ifndef PARS
#define PARS 0
#endif

extern "C" __global__ __launch_bounds__(NTHR, MINB) void post90_f8(const T* __restrict__ a, const T* __restrict__ o, const T* __restrict__ sg,
                                                               const T* __restrict__ mod, const uint8_t* __restrict__ wo8,
                                                               const uint8_t* __restrict__ wab8,
                                                               const uint8_t* __restrict__ wd8, const float* __restrict__ wsi,
                                                               T* __restrict__ out, int M, int nmod, float eps) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int HB8 = 64, NCH = 256 / HB8;            // hidden chunk (as the bf16 kernel)
  constexpr int ABB = 2 * HB8 * 128, BUF = ABB + 128 * 128;  // [Wa_j; Wb_j] 128 rows x 128 B, Wd_j 128 rows x 128 B (64 used)
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const size_t row0 = (size_t)blockIdx.x * ROWS + warp * 16;
  const size_t mrow = row0 % (size_t)nmod;
  auto issue = [&](int s) {
    const uint32_t d = base + (s & 1) * BUF;
    if (s < 2) {
      load_w8(d, wo8 + (size_t)s * 64 * 128, 128, 64, 128, tid, NTHR);
    } else {
      const int j = s - 2;
      load_w8(d, wab8 + (size_t)j * 128 * 128, 128, 128, 128, tid, NTHR);
      load_w8(d + ABB, wd8 + (size_t)j * 128 * HB8, HB8, 128, HB8, tid, NTHR);
    }
    cp_commit();
  };
  issue(0);
  issue(1);
  Tile c, r;
  float s0 = 1.f, s1 = 1.f;
  uint32_t aq[KSTEPS8][4];  // the quantized A operand of the current product (gated o, then x2)
  {  // gated = o * sg
    ld_tile(c, o + row0 * 128, 128, 0, g, t);
    ld_tile(r, sg + row0 * 128, 128, 0, g, t);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) c.f[nt][e] = round_store(c.f[nt][e] * r.f[nt][e]);
    row_scales<16>(c.f, s0, s1);
    quant8<KSTEPS8>(c.f, s0, s1, aq);
  }
  Tile a2;
  for (int s = 0; s < 2; ++s) {  // u = gated Wo^T, in two 64-column halves
    cp_wait<1>();
    proxy_fence();
    __syncthreads();
#if !NOOUT
    float acc[32];
    wg_gemm8q<64, KSTEPS8>(acc, aq, base + (s & 1) * BUF, 0);
#endif
    __syncthreads();
    issue(s + 2);
#if !NOOUT
    const float ws = __ldg(wsi + s), i0 = inv_pow2(s0) * ws, i1 = inv_pow2(s1) * ws;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      a2.f[8 * s + nt][0] = acc[4 * nt + 0] * i0;
      a2.f[8 * s + nt][1] = acc[4 * nt + 1] * i0;
      a2.f[8 * s + nt][2] = acc[4 * nt + 2] * i1;
      a2.f[8 * s + nt][3] = acc[4 * nt + 3] * i1;
    }
#endif
  }
#if NOOUT
  ld_tile(a2, o + row0 * 128, 128, 0, g, t);
#endif
  {  // a2 = a + so * u
    Tile aa, so;
    ld_tile(aa, a + row0 * 128, 128, 0, g, t);
    ld_tile(so, mod + mrow * 768, 768, 256, g, t);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float u = round_store(a2.f[nt][e]);
        a2.f[nt][e] = round_store(aa.f[nt][e] + round_store(so.f[nt][e] * u));
      }
  }
  {  // x2 = LN(a2) * s2 + bi2 (into c); PARS: LN(a)
    Tile s2, b2;
    ld_tile(s2, mod + mrow * 768, 768, 384, g, t);
    ld_tile(b2, mod + mrow * 768, 768, 512, g, t);
    float m0, r0, m1, r1;
#if PARS
    ld_tile(c, a + row0 * 128, 128, 0, g, t);
    norm_stats(c, eps, m0, r0, m1, r1);
    const Tile& xin = c;
#else
    norm_stats(a2, eps, m0, r0, m1, r1);
    const Tile& xin = a2;
#endif
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      c.f[nt][0] = round_store((xin.f[nt][0] - m0) * r0 * s2.f[nt][0] + b2.f[nt][0]);
      c.f[nt][1] = round_store((xin.f[nt][1] - m0) * r0 * s2.f[nt][1] + b2.f[nt][1]);
      c.f[nt][2] = round_store((xin.f[nt][2] - m1) * r1 * s2.f[nt][2] + b2.f[nt][2]);
      c.f[nt][3] = round_store((xin.f[nt][3] - m1) * r1 * s2.f[nt][3] + b2.f[nt][3]);
    }
    row_scales<16>(c.f, s0, s1);
    quant8<KSTEPS8>(c.f, s0, s1, aq);
  }
  const float x0i = inv_pow2(s0), x1i = inv_pow2(s1);
  float o4[64];  // the squeeze, fp32, dequantized per chunk
  const float wsd = __ldg(wsi + 10);
  for (int j = 0; j < NCH; ++j) {
    cp_wait<1>();
    proxy_fence();
    __syncthreads();
    const uint32_t wa = base + ((j + 2) & 1) * BUF;
    float acc[64];
    wg_gemm8q<128, KSTEPS8>(acc, aq, wa, 0);
    const float wsa = __ldg(wsi + 2 + 2 * j), wsb = __ldg(wsi + 3 + 2 * j);
    const float fa0 = x0i * wsa, fa1 = x1i * wsa, fb0 = x0i * wsb, fb1 = x1i * wsb;
    Tile h;
#pragma unroll
    for (int nt = 0; nt < HB8 / 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float av = round_store(acc[4 * nt + e] * (e < 2 ? fa0 : fa1));
        const float bv = round_store(acc[4 * (nt + HB8 / 8) + e] * (e < 2 ? fb0 : fb1));
        h.f[nt][e] = round_store(round_store(silu(av)) * bv);
      }
    float n0, n1;
    row_scales<HB8 / 8>(h.f, n0, n1);
    uint32_t ah[HB8 / 32][4];
    quant8<HB8 / 32>(h.f, n0, n1, ah);
    // the chunk's squeeze into the dead [a | b] accumulator, then added to the fp32 sum dequantized (the chunks' row scales differ)
    wg_gemm8q<128, HB8 / 32>(acc, ah, wa + ABB, 0);
    const float d0 = inv_pow2(n0) * wsd, d1 = inv_pow2(n1) * wsd;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      o4[4 * nt + 0] = (j > 0 ? o4[4 * nt + 0] : 0.f) + acc[4 * nt + 0] * d0;
      o4[4 * nt + 1] = (j > 0 ? o4[4 * nt + 1] : 0.f) + acc[4 * nt + 1] * d0;
      o4[4 * nt + 2] = (j > 0 ? o4[4 * nt + 2] : 0.f) + acc[4 * nt + 2] * d1;
      o4[4 * nt + 3] = (j > 0 ? o4[4 * nt + 3] : 0.f) + acc[4 * nt + 3] * d1;
    }
    __syncthreads();
    if (j + 4 < NCH + 2) issue(j + 4); else cp_commit();
  }
  {  // out = a2 + st * t
    Tile st;
    ld_tile(st, mod + mrow * 768, 768, 640, g, t);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float tv = round_store(o4[4 * nt + e]);
        a2.f[nt][e] = round_store(a2.f[nt][e] + round_store(st.f[nt][e] * tv));
      }
    st_tile(a2, out + row0 * 128, 128, 0, g, t);
  }
}
