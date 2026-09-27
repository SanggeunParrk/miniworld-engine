// tr_bwd_dx_sm80.cuh -- the H100-style input-gradient role on A100: per 32 rows of one warp, for all 512 hidden units,
//   [a | b] = xn [0.5 Wa; Wb]^T, dh = dy Ws, the SwiGLU backward, d_xn += dA Wa + dB Wb, then the LayerNorm backward + residual -> dx.
// Nothing crosses global memory but x, xn, dy, (mean, rstd) in and dx out (10 M D H).  xn and dy stay in registers as A fragments (the
// forward's k order, 32-bit loads so no load vector groups them differently), d_xn accumulates in f16 (64 registers: f32 would not fit
// beside them), with dA / dB formed as f16 A fragments and [Wa; Wb] kept as f16 in the ring slot.
// CTA = 8 warps x 32 rows (256-row tile); one ring of 32-hidden chunks: [W1 16 KB | W3 8 KB | WX (f16) 16 KB] = 40 KB, 3 slots.
#pragma once
#include "tr_bwd_px_sm80.cuh"

namespace a100 {

DEVI void ldsm_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

struct DXParams {
  const __nv_bfloat16* x;      // [T][128]
  const __nv_bfloat16* xn;     // [T][128]
  const __nv_bfloat16* dy;     // [T][128]
  const float2* stats;         // [T] (mean, rstd)
  const __nv_bfloat16* w;      // [16 chunks][W1 | W3 | WX16]
  const float* gamma;
  __nv_bfloat16* dx;
  float* dgb;                  // [grid][2][128]
  int T;
};

template <int NST_>
struct CfgDXT {
  static constexpr int D = 128, H = 512, NWARP = 8, NTHR = 256, BM = 256, CH = 32, NCHUNK = 16;
  static constexpr int SLOT = 40960, NST = NST_, AHEAD = 1, OFF_W3 = 16384, OFF_WX = 24576;
  static constexpr int SMEM_W = NST * SLOT, OFF_GAM = SMEM_W, OFF_DGB = OFF_GAM + 512, OFF_BAR = OFF_DGB + NWARP * 1024;
  static constexpr int BAR_BYTES = (2 * NST * 8 + 15) / 16 * 16, SMEM = OFF_BAR + BAR_BYTES;
  static_assert(SMEM + 1024 <= 167936, "sm_80 shared memory");
};
using CfgDX = CfgDXT<3>;

DEVI void dx_tile(const DXParams& p, const Ring<CfgDX>& w, uint32_t s_u, int r0, int it, int total) {
  using G = CfgDX;
  constexpr int D = 128;
  const int lane = w.lane, g8 = lane >> 2, q = lane & 3;
  uint32_t fa[2][8][4], fd[2][8][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {                              // each fragment quad loaded as a unit (see p_tile: no MOV per MMA)
    const uint32_t* xr0 = reinterpret_cast<const uint32_t*>(p.xn + (size_t)(r0 + 16 * mt + g8) * D + 8 * q);
    const uint32_t* dr0 = reinterpret_cast<const uint32_t*>(p.dy + (size_t)(r0 + 16 * mt + g8) * D + 8 * q);
#pragma unroll
    for (int s = 0; s < 8; ++s) {                               // word v = 2 s + e2 of the thread's row words 16 (s / 2) + 2 (s % 2) + e2
      const int v0 = 16 * (s >> 1) + 2 * (s & 1);
      fa[mt][s][0] = __ldg(xr0 + v0); fa[mt][s][1] = __ldg(xr0 + 8 * 64 + v0);
      fa[mt][s][2] = __ldg(xr0 + v0 + 1); fa[mt][s][3] = __ldg(xr0 + 8 * 64 + v0 + 1);
      fd[mt][s][0] = __ldg(dr0 + v0); fd[mt][s][1] = __ldg(dr0 + 8 * 64 + v0);
      fd[mt][s][2] = __ldg(dr0 + v0 + 1); fd[mt][s][3] = __ldg(dr0 + 8 * 64 + v0 + 1);
    }
  }
  uint32_t acc[2][16][2];                                       // d_xn, f16x2 [mt][n8 tile J][row g8 | g8 + 8]
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[mt][j][0] = acc[mt][j][1] = 0u;
  const int lrow = ((lane >> 4) << 3) + (lane & 7), lgr = (lane >> 3) & 1;
  // ldmatrix.x2 lane address (lanes 0-15): matrix lane / 8 = k-granule half, row lane % 8 of the 8-unit half
  const uint32_t w1x2 = lgr * 1024 + (lane & 7) * 16, w3x2 = G::OFF_W3 + lgr * 512 + (lane & 7) * 16;
  (void)lrow;
  const uint32_t bx_off = G::OFF_WX + (lane >> 4) * 512 + ((((lane >> 3) & 1) << 3) + (lane & 7)) * 16;
#pragma unroll 1
  for (int c = 0; c < 16; ++c) {
    const int u = 16 * it + c;
    bool pending;
    const uint32_t wb = w.begin(u, total, pending);
#pragma unroll
    for (int ps = 0; ps < 2; ++ps) {
      // the step's 16 hidden units as two 8-unit halves n: 24 accumulators live at a time instead of 48 (registers), B via ldmatrix.x2
      uint32_t hA[2][4], hB[2][4];                               // dA, dB as f16 A fragments: words (c0, c1) row g8 of half n -> 2 n
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        float acc1[2][2][4], accd[2][4];                         // [mt][a | b][4], [mt][4]
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int e = 0; e < 4; ++e) { acc1[mt][0][e] = acc1[mt][1][e] = accd[mt][e] = 0.f; }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          uint32_t ba[2], bb[2], bd[2];
          ldsm_x2(ba, wb + w1x2 + ps * 512 + n * 128 + s * 2048);
          ldsm_x2(bb, wb + w1x2 + ps * 512 + n * 128 + 256 + s * 2048);
          ldsm_x2(bd, wb + w3x2 + ps * 256 + n * 128 + s * 1024);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(acc1[mt][0], fa[mt][s], ba[0], ba[1]);
            mma16816(acc1[mt][1], fa[mt][s], bb[0], bb[1]);
            mma16816(accd[mt], fd[mt][s], bd[0], bd[1]);
          }
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int hp = 0; hp < 2; ++hp) {
            float vA[2], vB[2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
              const int e = 2 * hp + k;
              const float ap = acc1[mt][0][e], bv = acc1[mt][1][e], d = accd[mt][e];
              const float t = tanh_approx(ap), sp = 1.f + t, silu = ap * sp;       // sig = sp / 2, silu = ap sp
              vB[k] = d * silu;
              vA[k] = d * bv * fmaf(-silu, t, silu + sp);             // 2 dA = d b (sp + silu (1 - t)); WX holds 0.5 Wa
            }
            hA[mt][2 * n + hp] = pack_f16(vA[0], vA[1]);
            hB[mt][2 * n + hp] = pack_f16(vB[0], vB[1]);
          }
      }
#pragma unroll
      for (int abk = 0; abk < 2; ++abk)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          uint32_t bw[4];
          ldsm_x4_t(bw, wb + bx_off + abk * 8192 + j * 1024 + ps * 256);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816_h(acc[mt][2 * j], abk ? hB[mt] : hA[mt], bw[0], bw[1]);
            mma16816_h(acc[mt][2 * j + 1], abk ? hB[mt] : hA[mt], bw[2], bw[3]);
          }
        }
    }
    w.end(u, pending);
  }

  // ---- LayerNorm backward + residual in the accumulator layout (tile J = word J: columns 32 (J / 4) + 8 q + 2 (J % 4) + e)
  const uint32_t sGam = s_u + G::OFF_GAM, wslot = s_u + G::OFF_DGB + (threadIdx.x >> 5) * 1024;
  float dgs[32], dbs[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) dgs[k] = dbs[k] = 0.f;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int r = r0 + 16 * mt + 8 * hr + g8;
      const float2 st = p.stats[r];
      uint32_t xw[16], dw[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 xi = ldg_nc_na(p.x + (size_t)r * D + 32 * i + 8 * q), di = ldg_nc_na(p.dy + (size_t)r * D + 32 * i + 8 * q);
        xw[4 * i] = xi.x; xw[4 * i + 1] = xi.y; xw[4 * i + 2] = xi.z; xw[4 * i + 3] = xi.w;
        dw[4 * i] = di.x; dw[4 * i + 1] = di.y; dw[4 * i + 2] = di.z; dw[4 * i + 3] = di.w;   // (keeping fd alive to here spills)
      }
      float xh[32], gd[32], s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int J = 0; J < 16; ++J) {
        const uint32_t col = 32 * (J >> 2) + 8 * q + 2 * (J & 3);
        const uint2 gi = lds64(sGam + col * 4);
        const float2 dn2 = unpack_f16(acc[mt][J][hr]);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float xv = e ? bf16hi(xw[J]) : bf16lo(xw[J]), dn = e ? dn2.y : dn2.x;
          xh[2 * J + e] = (xv - st.x) * st.y;
          gd[2 * J + e] = __uint_as_float(e ? gi.y : gi.x) * dn;
          dgs[2 * J + e] = fmaf(dn, xh[2 * J + e], dgs[2 * J + e]);
          dbs[2 * J + e] += dn;
          s1 = fmaf(gd[2 * J + e], xh[2 * J + e], s1);
          s2 += gd[2 * J + e];
        }
      }
      const float ca = quad_sum(s1) * (1.f / D), cb = quad_sum(s2) * (1.f / D);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int J = 4 * i + k;
          o[k] = pack_bf16(fmaf(gd[2 * J] - fmaf(xh[2 * J], ca, cb), st.y, bf16lo(dw[J])),
                           fmaf(gd[2 * J + 1] - fmaf(xh[2 * J + 1], ca, cb), st.y, bf16hi(dw[J])));
        }
        stg128(p.dx + (size_t)r * D + 32 * i + 8 * q, make_uint4(o[0], o[1], o[2], o[3]));
      }
    }
#pragma unroll
  for (int k = 0; k < 32; ++k)
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      dgs[k] += __shfl_xor_sync(0xffffffffu, dgs[k], o);
      dbs[k] += __shfl_xor_sync(0xffffffffu, dbs[k], o);
    }
  if (g8 == 0) {
#pragma unroll
    for (int J = 0; J < 16; ++J)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const uint32_t col = 32 * (J >> 2) + 8 * q + 2 * (J & 3) + e;
        sts32(wslot + col * 4, __float_as_uint(__uint_as_float(lds32(wslot + col * 4)) + dgs[2 * J + e]));
        sts32(wslot + 512 + col * 4, __float_as_uint(__uint_as_float(lds32(wslot + 512 + col * 4)) + dbs[2 * J + e]));
      }
  }
}

// role DX on CTA b of grid (tiles b, b + grid, ...); dgb row b
DEVI void dx_role(const DXParams& p, uint8_t* smem, int b, int grid) {
  using G = CfgDX;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t s_u = smem_u32(smem);
  const int ntile = p.T / G::BM;
  const int n_items = ntile > b ? (ntile - b + grid - 1) / grid : 0, total = n_items * 16;
  Ring<G> w;
  w.sW = s_u; w.bars = s_u + G::OFF_BAR; w.src = p.w + tid * 8; w.tid = tid; w.lane = lane;
  if (tid == 0)
    for (int s = 0; s < G::NST; ++s) { mbar_init(w.full(s), G::NTHR); mbar_init(w.empty(s), G::NWARP); }
  for (int k = tid; k < 128; k += G::NTHR) reinterpret_cast<float*>(smem + G::OFF_GAM)[k] = p.gamma[k];
  for (int k = tid; k < G::NWARP * 256; k += G::NTHR) reinterpret_cast<float*>(smem + G::OFF_DGB)[k] = 0.f;
  __syncthreads();
  if (n_items > 0) {
    for (int c = 0; c < G::AHEAD && c < total; ++c) w.issue(c % 16, c);
#pragma unroll 1
    for (int it = 0; it < n_items; ++it) dx_tile(p, w, s_u, (b + it * grid) * G::BM + 32 * warp, it, total);
    cp_async_wait<0>();
  }
  __syncthreads();
  for (int k = tid; k < 256; k += G::NTHR) {
    float v = 0.f;
#pragma unroll
    for (int wi = 0; wi < G::NWARP; ++wi) v += reinterpret_cast<float*>(smem + G::OFF_DGB)[wi * 256 + k];
    p.dgb[(size_t)b * 256 + k] = v;
  }
}

__global__ void __launch_bounds__(256, 1) tr_bwd_dx_kernel(const DXParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  dx_role(p, smem, blockIdx.x, gridDim.x);
}

}  // namespace a100
