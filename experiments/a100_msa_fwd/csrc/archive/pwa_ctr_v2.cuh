// pwa_ctr.cuh -- PWA split path, K_C: per head h, u[s, i, 32 h + d] = sigmoid(y[s, i, :] Wg_h'^T + bg_h) .* o_h[s, i, d],
//   o_h[s, i, :] = sum_j w[h, i, j] v[h][j][s 32 + :],  y = LN(msa) without affine (folded into Wg', bg).
// The out-projection (sum over heads) is NOT in this kernel: u goes to memory [S*L][256] and one GEMM (+ residual) follows. That frees
// the 64 fp32 per token the fused kernel kept across heads, so the contraction gets a cutlass-shaped tile: CTA = one head x 128 i x
// 4 s (N = 128), 4 warps of 64 i x 64 (2 s x 32 d), 2 CTAs / SM. Per k16 step a warp issues 8 ldmatrix for 32 mma (fused kernel: 4 for 8).
// K = L in 32-j chunks through an NS-stage cp.async ring: w chunk [128 i][32 j] (64 B rows), v chunk [32 j][128 = 4 s x 32 d] (256 B rows,
// ldmatrix.trans). Epilogue: the y tile [4 s][128 i][64] (LN(msa), written once by pwa_value; v1 re-normalized msa here for every
// head: FADD / FMUL / FFMA 21% of instructions) is staged into the idle ring, and the gate GEMM runs per (s, m-tile pair) with Wg_h'
// fragments pre-packed per lane (8 x 16 B global loads).
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct PwaCtrParams {
  const __nv_bfloat16* y;     // [S*L, 64] LN(msa) without affine (from pwa_value)
  const __nv_bfloat16* w;     // [8][L][L]
  const __nv_bfloat16* v;     // [8][L][S*32]
  const uint4* wgf;           // [8 h][32 lanes][8 x 16 B]: Wg' B fragments, e = (dt 4 + kc) 2 + half
  const float* bg;            // [256]
  __nv_bfloat16* u;           // [S*L, 256]
  int S, L;
  float eps;
};

template <int NS_>
struct PwaCCfg {
  static constexpr int NS = NS_, NTHR = 128, TI = 128, TS = 4, JC = 32;
  static constexpr int STAGE_A = 128 * 64, STAGE = STAGE_A + 32 * 256;       // 8 KB + 8 KB
  static constexpr int YB = TS * 128 * 128;                                   // staged y tile, 64 KB
  static constexpr int SMEM = NS * STAGE > YB ? NS * STAGE : YB;
};

DEVI uint32_t swz64c(int row, int gr) { return row * 64 + ((gr ^ ((row >> 1) & 3)) << 4); }
DEVI uint32_t swz128c(int row, int gr) { return row * 128 + ((gr ^ (row & 7)) << 4); }
DEVI uint32_t swz256c(int row, int gr) { return row * 256 + ((gr ^ (row & 7)) << 4); }

template <class G>
__global__ void __launch_bounds__(G::NTHR, 2) pwa_ctr_kernel(PwaCtrParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, wm = warp & 1, wn = warp >> 1, l4 = lane >> 4;
  const int L = p.L, nI = L / G::TI, nJ = L / G::JC;
  // grid order: i-block fastest, then head, then s-block (the y tile and the v slice of an s-block stay L2-resident)
  const int ib = blockIdx.x % nI, h = (blockIdx.x / nI) & 7, sblk = blockIdx.x / (nI * 8);
  const int i0 = ib * G::TI, s0 = sblk * G::TS;
  const size_t ldv = (size_t)p.S * 32;

  // copy slots: A granule (tid & 3) of rows (tid >> 2) + 32 k; B granule (tid & 15) of rows (tid >> 4) + 8 k; k = 0..3
  const uint32_t a_dst = swz64c(tid >> 2, tid & 3), b_dst = G::STAGE_A + swz256c(tid >> 4, tid & 15);
  const __nv_bfloat16* asrc = p.w + ((size_t)h * L + i0 + (tid >> 2)) * L + (tid & 3) * 8;
  const __nv_bfloat16* bsrc = p.v + ((size_t)h * L + (tid >> 4)) * ldv + (size_t)s0 * 32 + (tid & 15) * 8;
  auto issue = [&](int jc, int stage) {
    const uint32_t st = sb + stage * G::STAGE;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      cp_async16(st + a_dst + k * 32 * 64, asrc + (size_t)jc * G::JC + (size_t)k * 32 * L);
      cp_async16(st + b_dst + k * 8 * 256, bsrc + (size_t)(jc * G::JC + 8 * k) * ldv);
    }
  };

  uint32_t a_off[4], b_off[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) a_off[mt] = swz64c(64 * wm + 16 * mt + (lane & 15), l4);
  const int r0 = (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
  for (int np = 0; np < 4; ++np) b_off[np] = G::STAGE_A + swz256c(r0, 8 * wn + 2 * np + l4);

  float acc[4][8][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

#pragma unroll
  for (int s = 0; s < G::NS - 1; ++s) { if (s < nJ) issue(s, s); cp_async_commit(); }
  int stage = 0, lstage = G::NS - 1;
  for (int jc = 0; jc < nJ; ++jc) {
    cp_async_wait<G::NS - 2>();
    __syncthreads();
    if (jc + G::NS - 1 < nJ) issue(jc + G::NS - 1, lstage);
    cp_async_commit();
    if (++lstage == G::NS) lstage = 0;
    const uint32_t st = sb + stage * G::STAGE;
    if (++stage == G::NS) stage = 0;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t af[4][4], bf[4][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) ldsm_x4(af[mt], st + (a_off[mt] ^ (kk << 5)));
#pragma unroll
      for (int np = 0; np < 4; ++np) ldsm_x4_t(bf[np], st + b_off[np] + kk * 16 * 256);
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma16816(acc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  // y tile -> smem (row r = 128 sl + il)
  for (int idx = tid; idx < G::TS * 128 * 8; idx += G::NTHR) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sb + swz128c(row, gr), p.y + ((size_t)(s0 + (row >> 7)) * L + i0 + (row & 127)) * 64 + gr * 8);
  }
  cp_async_commit();
  uint4 wg4[8];
  const uint4* wsrc4 = p.wgf + ((size_t)h * 32 + lane) * 8;
#pragma unroll
  for (int e = 0; e < 8; ++e) wg4[e] = __ldg(wsrc4 + e);
  float2 bgv[4];
#pragma unroll
  for (int dt = 0; dt < 4; ++dt) bgv[dt] = __ldg(reinterpret_cast<const float2*>(p.bg + 32 * h + 8 * dt + 2 * q));
  cp_async_wait<0>();
  __syncthreads();

  // gate + u: warp tokens are i = i0 + 64 wm + 16 mt + g (+8), s = s0 + 2 wn + sn; acc n-tile nt = 4 sn + dt (d = 8 dt + 2 q)
  const uint32_t* wg = reinterpret_cast<const uint32_t*>(wg4);
#pragma unroll
  for (int sn = 0; sn < 2; ++sn)
#pragma unroll
    for (int mp = 0; mp < 2; ++mp) {
      float gacc[2][4][4];
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
          gacc[m][dt][0] = gacc[m][dt][2] = bgv[dt].x;
          gacc[m][dt][1] = gacc[m][dt][3] = bgv[dt].y;
        }
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          uint32_t af[4];
          const int row = 128 * (2 * wn + sn) + 64 * wm + 16 * (2 * mp + m) + (lane & 15);
          ldsm_x4(af, sb + swz128c(row, 2 * kc + l4));
#pragma unroll
          for (int dt = 0; dt < 4; ++dt) mma16816(gacc[m][dt], af, wg[(dt * 4 + kc) * 2], wg[(dt * 4 + kc) * 2 + 1]);
        }
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int mt = 2 * mp + m;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const size_t tok = (size_t)(s0 + 2 * wn + sn) * L + i0 + 64 * wm + 16 * mt + g + 8 * hh;
          __nv_bfloat16* dst = p.u + tok * 256 + 32 * h + 2 * q;
#pragma unroll
          for (int dt = 0; dt < 4; ++dt) {
            const float* o = acc[mt][4 * sn + dt];
            const float* gg = gacc[m][dt];
            stg32(dst + 8 * dt, pack_bf16(sigmoid(gg[2 * hh]) * o[2 * hh], sigmoid(gg[2 * hh + 1]) * o[2 * hh + 1]));
          }
        }
      }
    }
}

}  // namespace a100
