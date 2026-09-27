// pwa_out2.cuh -- PWA split path with the gate moved here, K_O2:
//   out[t, :] = msa[t, :] + sum_h (sigmoid(y[t, :] Wg_h'^T + bg_h) .* o[t, 32 h ..]) Wo_h^T,   o from pwa_ctr (-DPWA_C_NOGATE).
// The gate GEMM (64 -> 256 per token) was the compute-bound contraction kernel's epilogue; here it fills the idle tensor time of a
// memory-bound kernel (o 512 B + y 128 B + msa 128 B in, 128 B out per token). Persistent CTA of 8 warps over 64-token tiles, double-
// buffered through cp.async; Wg' (256 x 128 B), Wo (64 x 512 B) and bg resident in smem. Warp w owns m-tile (w & 3) for heads
// 4 (w >> 2) .. +3; the second head half's partial out goes through smem and is added by the first half, which stores.
// v1 (4 warps, each 16 tokens x 8 heads, private rings): one warp per SMSP left the ldsm -> mma -> sigmoid -> mma chain unhidden.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct PwaOut2Params {
  const __nv_bfloat16* msa;   // [T, 64]
  const __nv_bfloat16* o;     // [T, 256]
  const __nv_bfloat16* y;     // [T, 64] LN(msa) without affine
  const __nv_bfloat16* wg;    // [256, 64] gamma folded
  const float* bg;            // [256]
  const __nv_bfloat16* wo;    // [64, 256]
  __nv_bfloat16* out;         // [T, 64]
  int T;
};

constexpr int PWA_O2_NTHR = 256, PWA_O2_TT = 64;
constexpr int PWA_O2_WG = 0, PWA_O2_WO = 256 * 128, PWA_O2_BG = PWA_O2_WO + 64 * 512, PWA_O2_RED = PWA_O2_BG + 1024;
constexpr int PWA_O2_BUF = PWA_O2_RED + 4 * 16 * 64 * 4;                       // partial-out exchange: 4 m-tiles x 16 x 64 fp32
constexpr int PWA_O2_SLOT = PWA_O2_TT * 512 + PWA_O2_TT * 128;                 // o 32 KB + y 8 KB
constexpr int PWA_O2_SMEM = PWA_O2_BUF + 2 * PWA_O2_SLOT;

DEVI uint32_t swz128o2(int row, int gr) { return row * 128 + ((gr ^ (row & 7)) << 4); }
DEVI uint32_t swz512o2(int row, int gr) { return row * 512 + ((gr ^ (row & 7)) << 4); }

__global__ void __launch_bounds__(PWA_O2_NTHR, 1) pwa_out2_kernel(PwaOut2Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, l4 = lane >> 4, mt = warp & 3, hg = warp >> 2;
  for (int idx = tid; idx < 256 * 8; idx += PWA_O2_NTHR) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sb + PWA_O2_WG + swz128o2(row, gr), p.wg + row * 64 + gr * 8);
    const int ro = idx >> 5, go = idx & 31;
    cp_async16(sb + PWA_O2_WO + swz512o2(ro, go), p.wo + ro * 256 + go * 8);
  }
  float* sbg = reinterpret_cast<float*>(smem + PWA_O2_BG);
  float* red = reinterpret_cast<float*>(smem + PWA_O2_RED);
  for (int idx = tid; idx < 256; idx += PWA_O2_NTHR) sbg[idx] = p.bg[idx];

  const int ntiles = p.T / PWA_O2_TT;
  auto issue = [&](int tile, int slot) {
    const uint32_t so = sb + PWA_O2_BUF + slot * PWA_O2_SLOT, sy = so + PWA_O2_TT * 512;
    const __nv_bfloat16* src_o = p.o + (size_t)tile * PWA_O2_TT * 256;
    const __nv_bfloat16* src_y = p.y + (size_t)tile * PWA_O2_TT * 64;
#pragma unroll
    for (int k = 0; k < 8; ++k) {         // o: 64 rows x 32 granules; row = (tid >> 5) + 8 k, granule = lane
      const int row = warp + 8 * k;
      cp_async16(so + swz512o2(row, lane), src_o + row * 256 + lane * 8);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {         // y: 64 rows x 8 granules
      const int idx = tid + PWA_O2_NTHR * k, row = idx >> 3, gr = idx & 7;
      cp_async16(sy + swz128o2(row, gr), src_y + row * 64 + gr * 8);
    }
  };
  int tile = blockIdx.x, slot = 0;
  if (tile < ntiles) issue(tile, 0);
  cp_async_commit();                      // group 0: weights + first tile
  const uint32_t yrow = swz128o2(16 * mt + (lane & 15), l4);                                   // ^ (kc << 5)
  const uint32_t gb = PWA_O2_WG + swz128o2((lane & 7) + (l4 << 3), (lane >> 3) & 1);           // + (32 h + 16 np) 128, ^ (kc << 5)
  const uint32_t ob = PWA_O2_WO + swz512o2((lane & 7) + (l4 << 3), (lane >> 3) & 1);           // + 16 np 512, ^ ((4 h + 2 kc) << 4)
  for (; tile < ntiles; tile += gridDim.x, slot ^= 1) {
    if (tile + (int)gridDim.x < ntiles) issue(tile + gridDim.x, slot ^ 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const uint32_t so = sb + PWA_O2_BUF + slot * PWA_O2_SLOT, sy = so + PWA_O2_TT * 512;
    uint32_t yf[4][4];
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) ldsm_x4(yf[kc], sy + (yrow ^ (kc << 5)));
    float oacc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) oacc[nt][e] = 0.f;
#pragma unroll 2
    for (int hh4 = 0; hh4 < 4; ++hh4) {
      const int h = 4 * hg + hh4;
      float gacc[4][4];
#pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        const float2 b = *reinterpret_cast<const float2*>(sbg + 32 * h + 8 * dt + 2 * q);
        gacc[dt][0] = gacc[dt][2] = b.x;
        gacc[dt][1] = gacc[dt][3] = b.y;
      }
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t bf[4];
          ldsm_x4(bf, sb + ((gb + (32 * h + 16 * np) * 128) ^ (kc << 5)));
          mma16816(gacc[2 * np], yf[kc], bf[0], bf[1]);
          mma16816(gacc[2 * np + 1], yf[kc], bf[2], bf[3]);
        }
      uint32_t ua[2][4];
#pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        const uint32_t o0 = lds32(so + swz512o2(16 * mt + g, 4 * h + dt) + 4 * q);
        const uint32_t o1 = lds32(so + swz512o2(16 * mt + g + 8, 4 * h + dt) + 4 * q);
        ua[dt >> 1][(dt & 1) * 2] = pack_bf16(sigmoid(gacc[dt][0]) * bf16lo(o0), sigmoid(gacc[dt][1]) * bf16hi(o0));
        ua[dt >> 1][(dt & 1) * 2 + 1] = pack_bf16(sigmoid(gacc[dt][2]) * bf16lo(o1), sigmoid(gacc[dt][3]) * bf16hi(o1));
      }
#pragma unroll
      for (int kc = 0; kc < 2; ++kc)
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t bf[4];
          ldsm_x4(bf, sb + ((ob + 16 * np * 512) ^ ((4 * h + 2 * kc) << 4)));
          mma16816(oacc[2 * np], ua[kc], bf[0], bf[1]);
          mma16816(oacc[2 * np + 1], ua[kc], bf[2], bf[3]);
        }
    }
    // head half 1 -> smem, half 0 adds it, the residual, and stores. red[mt][row][c] fp32, row = g (+8), float2 at c = 8 nt + 2 q
    float* rm = red + mt * 16 * 64;
    if (hg == 1) {
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        *reinterpret_cast<float2*>(rm + g * 64 + ((8 * nt + 2 * q) ^ (g << 3))) = make_float2(oacc[nt][0], oacc[nt][1]);
        *reinterpret_cast<float2*>(rm + (g + 8) * 64 + ((8 * nt + 2 * q) ^ (g << 3))) = make_float2(oacc[nt][2], oacc[nt][3]);
      }
    }
    __syncthreads();
    if (hg == 0) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = g + 8 * hh;
        const size_t base = ((size_t)tile * PWA_O2_TT + 16 * mt + r) * 64;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int c = 8 * nt + 2 * q;
          const float2 pr = *reinterpret_cast<const float2*>(rm + r * 64 + (c ^ (g << 3)));
          const uint32_t x = __ldg(reinterpret_cast<const unsigned int*>(p.msa + base + c));
          stg32(p.out + base + c, pack_bf16(bf16lo(x) + oacc[nt][2 * hh] + pr.x, bf16hi(x) + oacc[nt][2 * hh + 1] + pr.y));
        }
      }
    }
    // the next iteration's __syncthreads (after its wait) orders this tile's reads of slot / red before they are overwritten
  }
  cp_async_wait<0>();
}

}  // namespace a100
