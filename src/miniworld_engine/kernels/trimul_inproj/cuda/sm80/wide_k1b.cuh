// wide_k1b.cuh -- backward of the wide TriMul front: recomputes k1w's pre-activations (the same raw-input GEMM with the LayerNorm_in fold)
//   g' = 0.5 g,  p' = 0.5 p        (g, p = LN_in(z) . W_g[pc], LN_in(z) . W_p[pc])
// and turns the plane gradient da [2 Hs, T] (channel-major, from the contraction backward) into the pre-activation gradients
//   dg = da p s (1 - s),   dp = da s,   s = 0.5 + 0.5 tanh g'          (0 for a masked pair)
// written TOKEN-major into dpre [T, ldd] in the packed-row order of k1w's weights (columns 16 j + 0..7: dg of channels 8 j .., 16 j + 8..15: dp), so cuBLAS takes
// dW1 = dpre^T x_n and dx_n = dpre [W1 ; ..] directly (the output gate's gradient sits in dpre's last D columns, written by gate_bwd).
// Same tile as k1w (128 tokens x 128 packed rows, 4 warps, no token permutation: the stores are token-major); the 64 x 128 da tile of the CTA is prefetched
// (cp.async, 272-B shared rows) under the mainloop and read in the accumulators' fragment layout.
#pragma once
#include "wide_gemm.cuh"

namespace a100 {

struct K1wbParams {
  const __nv_bfloat16* z;      // [T][D]
  const __nv_bfloat16* w1;     // [4 Hs][D] packed rows (as k1w)
  const float* vs;             // [4 Hs]
  const float* vb;             // [4 Hs]
  const float2* st;            // [T] (mean, rstd) of the input rows
  const uint8_t* mask;         // [L] token mask or nullptr
  const __nv_bfloat16* da;     // [2 Hs][T] plane gradients
  __nv_bfloat16* dpre;         // [T][ldd]
  int T, L, D, ncol, ldd;
};

using K1wbTile = WTile<128, 128, 2, 2, 3, false, false>;
constexpr int K1WB_DA = 272;                                        // bytes per staged da row (256 + 16)
constexpr int K1WB_SMEM = K1wbTile::SMEM + 64 * K1WB_DA;

__global__ void __launch_bounds__(128, 2) k1wb_kernel(const K1wbParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = smem_u32(smem), sda = s0 + K1wbTile::SMEM;
  const int tid = threadIdx.x;
  const int ntn = p.ncol >> 7;
  const int mtile = (int)blockIdx.x / ntn, ntile = (int)blockIdx.x - mtile * ntn;
  const int t0 = mtile * 128, j0 = ntile * 128;
  // da tile: plane channels [64 ntile, 64 ntile + 64) x tokens [t0, t0 + 128)
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int idx = tid + 128 * e, r = idx >> 4, gk = idx & 15;
    cp_async16(sda + r * K1WB_DA + gk * 16, p.da + (size_t)(ntile * 64 + r) * p.T + t0 + 8 * gk);
  }
  cp_async_commit();
  K1wbTile tl;
  tl.run_fast(s0, p.D >> 5, GemmOps{p.z, (size_t)p.D, t0, p.T, p.w1, (size_t)p.D, j0, p.ncol});
  // ---- epilogue: the fragment words of dg / dp, then one staged tile of 128 tokens x 128 packed columns leaves as 16-byte row stores
  const int warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1, g8 = lane >> 2, q = lane & 3;
  constexpr int MT = K1wbTile::MT, NP = K1wbTile::NP;
  float rs[MT][2], mr[MT][2], mk[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int tok = t0 + 64 * wm + 16 * mt + g8 + 8 * rh;
      const float2 s = __ldg(p.st + tok);
      rs[mt][rh] = s.y; mr[mt][rh] = s.x * s.y;
      float m = 1.f;
      if (p.mask != nullptr) { const int i = tok / p.L, j = tok - i * p.L; m = (__ldg(p.mask + i) != 0 && __ldg(p.mask + j) != 0) ? 1.f : 0.f; }
      mk[mt][rh] = m;
    }
  uint32_t wdg[MT][NP][2], wdp[MT][NP][2];
#pragma unroll
  for (int np = 0; np < NP; ++np) {
    const int R = j0 + 64 * wn + 16 * np + 2 * q;
    const float2 sg = __ldg(reinterpret_cast<const float2*>(p.vs + R)), sp = __ldg(reinterpret_cast<const float2*>(p.vs + R + 8));
    const float2 bg = __ldg(reinterpret_cast<const float2*>(p.vb + R)), bp = __ldg(reinterpret_cast<const float2*>(p.vb + R + 8));
    const int ch = 32 * wn + 8 * np + 2 * q;                         // channel inside the CTA's 64
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const int trow = 64 * wm + 16 * mt + g8 + 8 * rh;           // token inside the tile
        float dg[2], dp[2];
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          const float ag = tl.acc[mt][2 * np][2 * rh + cc], ap = tl.acc[mt][2 * np + 1][2 * rh + cc];
          const float g = fmaf(rs[mt][rh], ag, fmaf(-mr[mt][rh], cc ? sg.y : sg.x, cc ? bg.y : bg.x));
          const float pp = fmaf(rs[mt][rh], ap, fmaf(-mr[mt][rh], cc ? sp.y : sp.x, cc ? bp.y : bp.x));
          const float s = fmaf(0.5f, tanh_approx(g), 0.5f);
          unsigned short bits;
          asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(bits) : "r"(sda + (ch + cc) * K1WB_DA + trow * 2));
          const float dav = __uint_as_float(((uint32_t)bits) << 16) * mk[mt][rh];
          dp[cc] = dav * s;
          dg[cc] = dav * (2.f * pp) * s * (1.f - s);
        }
        wdg[mt][np][rh] = pack_bf16(dg[0], dg[1]);
        wdp[mt][np][rh] = pack_bf16(dp[0], dp[1]);
      }
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int np = 0; np < NP; ++np)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const int trow = 64 * wm + 16 * mt + g8 + 8 * rh;
        stage_word(s0, trow, 32 * wn + 8 * np + q, 128, wdg[mt][np][rh]);
        stage_word(s0, trow, 32 * wn + 8 * np + 4 + q, 128, wdp[mt][np][rh]);
      }
  __syncthreads();
  tile_store16<128, 128, 128>(s0, p.dpre + (size_t)t0 * p.ldd + j0, p.ldd);
}

}  // namespace a100
