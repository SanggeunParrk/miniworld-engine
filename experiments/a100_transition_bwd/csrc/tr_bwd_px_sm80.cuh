// tr_bwd_px_sm80.cuh -- roles P and X of the Transition backward in ONE kernel (see tr_bwd_sm80.cuh for the math and the layouts).
//
// CTA = 8 warps on a 128-row tile: warps 0-3 are P (rows 32 w .. of the tile: LayerNorm, [a | b] and dh GEMMs, the SwiGLU backward),
// warps 4-7 are X (the same rows: d_xn = [dA | dB] [Wa; Wb] with f32 accumulation, then the LayerNorm backward + residual).  P warp w and
// X warp w + 4 sit on the same SM sub-partition and hand dA / dB over per 32-hidden chunk through a double-buffered shared-memory
// block (full / empty mbarriers), so X never reads them from memory; P still writes dA, dB, h to global memory for role W.
// One weight ring for both roles: a slot holds the chunk's [0.5 Wa | Wb] (P, GEMM1), Ws^T (P, dh) and [Wa | Wb] in the output order
// (X, ldmatrix.trans), 40 KB; every warp copies its eighth of chunk u + 1 into the slot of chunk u - 2 once all 8 warps retired it.
#pragma once
#include "tr_bwd_sm80.cuh"

namespace a100 {

struct CfgPX {
  static constexpr int D = 128, H = 512, NWARP = 8, NTHR = 256, BM = 128, CH = 32, NCHUNK = 16;
  static constexpr int SLOT = 40960, NST = 3, AHEAD = 1;      // [W1 16 KB | W3 8 KB | WX 16 KB]
  static constexpr int OFF_W3 = 16384, OFF_WX = 24576;
  static constexpr int SMEM_W = NST * SLOT;
  static constexpr int HBUF = 4096, SMEM_H = 4 * 2 * HBUF;     // [pair][buffer][step][m16][dA | dB][32 lanes][16 B]
  static constexpr int SMEM_ST = 2 * BM * 8;                   // (mean, rstd) of the tile's rows, double-buffered by tile parity
  static constexpr int SMEM_GB = 2 * D * 4, SMEM_GAM = D * 4, SMEM_DGB = 2 * D * 4;
  static constexpr int NBAR = 2 * NST + 2 * 8;                 // ring full / empty, handoff full / empty per (pair, buffer)
  static constexpr int BAR_BYTES = (NBAR * 8 + 15) / 16 * 16;
  static constexpr int OFF_H = SMEM_W, OFF_ST = OFF_H + SMEM_H, OFF_GB = OFF_ST + SMEM_ST, OFF_GAM = OFF_GB + SMEM_GB;
  static constexpr int OFF_DGB = OFF_GAM + SMEM_GAM, OFF_BAR = OFF_DGB + SMEM_DGB;
  static constexpr int SMEM = OFF_BAR + BAR_BYTES;
  static_assert(SMEM + 1024 <= 167936, "sm_80 shared memory");
};

// ================================================================================================================== P warp
DEVI void px_p_tile(const BwdParams& p, const Ring<CfgPX>& w, uint32_t s_u, uint32_t hb, int pair, int r0, int nr0, bool has_next,
                    int it, int total) {
  using G = CfgPX;
  constexpr int D = 128;
  const int lane = w.lane, g8 = lane >> 2, q = lane & 3;
  const uint32_t sGB = s_u + G::OFF_GB, sSt = s_u + G::OFF_ST + (it & 1) * G::BM * 8;
  uint32_t fa[2][8][4], fd[2][8][4];                            // xn and dy A fragments (see p_tile: dy in 32-bit loads, no xn store)
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    uint32_t xnw[2][16], dyw[2][16];
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int lr = 16 * mt + 8 * hr + g8, r = r0 + lr;
      uint4 xin[4];
      const uint32_t* dyr = reinterpret_cast<const uint32_t*>(p.dy + (size_t)r * D + 8 * q);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        xin[i] = ldg_nc_na(p.x + (size_t)r * D + 32 * i + 8 * q);
#pragma unroll
        for (int k = 0; k < 4; ++k) dyw[hr][4 * i + k] = __ldg(dyr + 16 * i + k);
      }
      float xv[32];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t wv[4] = {xin[i].x, xin[i].y, xin[i].z, xin[i].w};
#pragma unroll
        for (int k = 0; k < 4; ++k) { xv[8 * i + 2 * k] = bf16lo(wv[k]); xv[8 * i + 2 * k + 1] = bf16hi(wv[k]); }
      }
      float sm = 0.f;
#pragma unroll
      for (int e = 0; e < 32; ++e) sm += xv[e];
      const float mean = quad_sum(sm) * (1.f / D);
      float sq = 0.f;
#pragma unroll
      for (int e = 0; e < 32; ++e) { xv[e] -= mean; sq = fmaf(xv[e], xv[e], sq); }
      const float rstd = rsqrtf(quad_sum(sq) * (1.f / D) + p.eps);
      if (q == 0) {
        p.stats[r] = make_float2(mean, rstd);                   // role W
        sts64(sSt + (32 * pair + lr) * 8, make_uint2(__float_as_uint(mean), __float_as_uint(rstd)));   // this CTA's X warp
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint4 gi = lds128(sGB + (i * 4 + q) * 16), bi = lds128(sGB + 512 + (i * 4 + q) * 16);
        const float4 gv = *reinterpret_cast<const float4*>(&gi), bv = *reinterpret_cast<const float4*>(&bi);
        xnw[hr][2 * i] = pack_bf16(fmaf(xv[4 * i] * rstd, gv.x, bv.x), fmaf(xv[4 * i + 1] * rstd, gv.y, bv.y));
        xnw[hr][2 * i + 1] = pack_bf16(fmaf(xv[4 * i + 2] * rstd, gv.z, bv.z), fmaf(xv[4 * i + 3] * rstd, gv.w, bv.w));
      }
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      fa[mt][s][0] = xnw[0][2 * s]; fa[mt][s][1] = xnw[1][2 * s]; fa[mt][s][2] = xnw[0][2 * s + 1]; fa[mt][s][3] = xnw[1][2 * s + 1];
      fd[mt][s][0] = dyw[0][2 * s]; fd[mt][s][1] = dyw[1][2 * s]; fd[mt][s][2] = dyw[0][2 * s + 1]; fd[mt][s][3] = dyw[1][2 * s + 1];
    }
  }
  __syncwarp();
  if (has_next) { prefetch_rows(p.x, nr0, 32, p.T, lane); prefetch_rows(p.dy, nr0, 32, p.T, lane); }

  const int lrow = ((lane >> 4) << 3) + (lane & 7), lgr = (lane >> 3) & 1;
  const uint32_t w1_off = lgr * 1024 + lrow * 16, w3_off = G::OFF_W3 + lgr * 512 + lrow * 16;
  uint4* const ab = p.ab + (size_t)(r0 / 16) * 3072 + lane;
  const uint32_t hfull = s_u + G::OFF_BAR + 8 * (2 * G::NST + 2 * pair), hempty = hfull + 8 * 8;
#pragma unroll 1
  for (int c = 0; c < 16; ++c) {
    const int u = 16 * it + c, hbuf = u & 1;
    bool pending;
    const uint32_t wb = w.begin(u, total, pending);
    const uint32_t hdst = hb + hbuf * G::HBUF + lane * 16;
    if (u >= 2) mbar_wait_bo(hempty + 8 * hbuf, ((u >> 1) - 1) & 1);   // the X warp consumed this buffer's previous chunk
#pragma unroll
    for (int ps = 0; ps < 2; ++ps) {
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
        uint32_t ba[4], bb[4], bd[4];
        ldsm_x4(ba, wb + w1_off + ps * 512 + s * 2048);
        ldsm_x4(bb, wb + w1_off + ps * 512 + 256 + s * 2048);
        ldsm_x4(bd, wb + w3_off + ps * 256 + s * 1024);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma16816(acc1[mt][0], fa[mt][s], ba[0], ba[1]);
          mma16816(acc1[mt][1], fa[mt][s], ba[2], ba[3]);
          mma16816(acc1[mt][2], fa[mt][s], bb[0], bb[1]);
          mma16816(acc1[mt][3], fa[mt][s], bb[2], bb[3]);
          mma16816(accd[mt][0], fd[mt][s], bd[0], bd[1]);
          mma16816(accd[mt][1], fd[mt][s], bd[2], bd[3]);
        }
      }
      const int K = 2 * c + ps;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        uint32_t wA[4], wB[4], wH[4];
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int hp = 0; hp < 2; ++hp) {
            float vA[2], vB[2], vH[2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
              const int e = 2 * hp + k;
              const float ap = acc1[mt][n][e], bv = acc1[mt][2 + n][e], d = accd[mt][n][e];
              const float t = tanh_approx(ap), sp = 1.f + t, silu = ap * sp;
              vH[k] = silu * bv;
              vB[k] = d * silu;
              vA[k] = d * bv * (0.5f * sp) * fmaf(ap, 1.f - t, 1.f);
            }
            wA[2 * n + hp] = pack_bf16(vA[0], vA[1]);
            wB[2 * n + hp] = pack_bf16(vB[0], vB[1]);
            wH[2 * n + hp] = pack_bf16(vH[0], vH[1]);
          }
        uint4* const o = ab + mt * 3072 + K * 96;
        const uint4 vA4 = make_uint4(wA[0], wA[1], wA[2], wA[3]), vB4 = make_uint4(wB[0], wB[1], wB[2], wB[3]);
        stg128(o, vA4);
        stg128(o + 32, vB4);
        stg128(o + 64, make_uint4(wH[0], wH[1], wH[2], wH[3]));
        sts128(hdst + ((ps * 2 + mt) * 2) * 512, vA4);
        sts128(hdst + ((ps * 2 + mt) * 2 + 1) * 512, vB4);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(hfull + 8 * hbuf);
    w.end(u, pending);
  }
}

// ================================================================================================================== X warp
DEVI void px_x_tile(const BwdParams& p, const Ring<CfgPX>& w, uint32_t s_u, uint32_t hb, int pair, int r0, int it, int total) {
  using G = CfgPX;
  constexpr int D = 128;
  const int lane = w.lane, g8 = lane >> 2, q = lane & 3;
  float acc[2][16][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][j][e] = 0.f;
  const uint32_t b_off = G::OFF_WX + (lane >> 4) * 512 + ((((lane >> 3) & 1) << 3) + (lane & 7)) * 16;
  const uint32_t hfull = s_u + G::OFF_BAR + 8 * (2 * G::NST + 2 * pair), hempty = hfull + 8 * 8;
#pragma unroll 1
  for (int c = 0; c < 16; ++c) {
    const int u = 16 * it + c, hbuf = u & 1;
    bool pending;
    const uint32_t wb = w.begin(u, total, pending);
    mbar_wait_bo(hfull + 8 * hbuf, (u >> 1) & 1);
    const uint32_t hsrc = hb + hbuf * G::HBUF + lane * 16;
#pragma unroll
    for (int ps = 0; ps < 2; ++ps) {
      uint32_t aA[2][4], aB[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const uint4 fA = lds128(hsrc + ((ps * 2 + mt) * 2) * 512), fB = lds128(hsrc + ((ps * 2 + mt) * 2 + 1) * 512);
        aA[mt][0] = fA.x; aA[mt][1] = fA.y; aA[mt][2] = fA.z; aA[mt][3] = fA.w;
        aB[mt][0] = fB.x; aB[mt][1] = fB.y; aB[mt][2] = fB.z; aB[mt][3] = fB.w;
      }
#pragma unroll
      for (int abk = 0; abk < 2; ++abk)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          uint32_t bw[4];
          ldsm_x4_t(bw, wb + b_off + abk * 8192 + j * 1024 + ps * 256);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(acc[mt][2 * j], abk ? aB[mt] : aA[mt], bw[0], bw[1]);
            mma16816(acc[mt][2 * j + 1], abk ? aB[mt] : aA[mt], bw[2], bw[3]);
          }
        }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(hempty + 8 * hbuf);             // the P warp may refill this buffer
    w.end(u, pending);
  }

  // ---- LayerNorm backward + residual (as x_tile), statistics from the P warp through shared memory
  const uint32_t sGam = s_u + G::OFF_GAM, sDgb = s_u + G::OFF_DGB, sSt = s_u + G::OFF_ST + (it & 1) * G::BM * 8;
  float dgs[32], dbs[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) dgs[k] = dbs[k] = 0.f;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int lr = 16 * mt + 8 * hr + g8, r = r0 + lr;
      const uint2 stw = lds64(sSt + (32 * pair + lr) * 8);
      const float mean = __uint_as_float(stw.x), rstd = __uint_as_float(stw.y);
      uint32_t xw[16], dw[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 xi = ldg_nc_na(p.x + (size_t)r * D + 32 * i + 8 * q), di = ldg_nc_na(p.dy + (size_t)r * D + 32 * i + 8 * q);
        xw[4 * i] = xi.x; xw[4 * i + 1] = xi.y; xw[4 * i + 2] = xi.z; xw[4 * i + 3] = xi.w;
        dw[4 * i] = di.x; dw[4 * i + 1] = di.y; dw[4 * i + 2] = di.z; dw[4 * i + 3] = di.w;
      }
      float xh[32], gd[32], s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int J = 0; J < 16; ++J) {
        const uint32_t col = 32 * (J >> 2) + 8 * q + 2 * (J & 3);
        const uint2 gi = lds64(sGam + col * 4);
        const float g0 = __uint_as_float(gi.x), g1 = __uint_as_float(gi.y);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float xv = e ? bf16hi(xw[J]) : bf16lo(xw[J]);
          const float dn = acc[mt][J][2 * hr + e];
          xh[2 * J + e] = (xv - mean) * rstd;
          gd[2 * J + e] = (e ? g1 : g0) * dn;
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
          const float v0 = fmaf(gd[2 * J] - fmaf(xh[2 * J], ca, cb), rstd, bf16lo(dw[J]));
          const float v1 = fmaf(gd[2 * J + 1] - fmaf(xh[2 * J + 1], ca, cb), rstd, bf16hi(dw[J]));
          o[k] = pack_bf16(v0, v1);
        }
        stg128(p.dx + (size_t)r * D + 32 * i + 8 * q, make_uint4(o[0], o[1], o[2], o[3]));
      }
    }
#pragma unroll
  for (int k = 0; k < 32; ++k) {
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      dgs[k] += __shfl_xor_sync(0xffffffffu, dgs[k], o);
      dbs[k] += __shfl_xor_sync(0xffffffffu, dbs[k], o);
    }
  }
  if (g8 == 0) {
#pragma unroll
    for (int J = 0; J < 16; ++J)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const uint32_t col = 32 * (J >> 2) + 8 * q + 2 * (J & 3) + e;
        asm volatile("red.shared.add.f32 [%0], %1;\n" ::"r"(sDgb + col * 4), "f"(dgs[2 * J + e]) : "memory");
        asm volatile("red.shared.add.f32 [%0], %1;\n" ::"r"(sDgb + 512 + col * 4), "f"(dbs[2 * J + e]) : "memory");
      }
  }
}

__global__ void __launch_bounds__(256, 1) tr_bwd_px_kernel(const BwdParams p) {
  using G = CfgPX;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t s_u = smem_u32(smem);
  const int pair = warp & 3;
  const int ntile = p.T / G::BM, b = blockIdx.x, grid = gridDim.x;
  const int n_items = ntile > b ? (ntile - b + grid - 1) / grid : 0;
  const int total = n_items * 16;
  Ring<G> w;
  w.sW = s_u; w.bars = s_u + G::OFF_BAR; w.src = p.wp + tid * 8; w.tid = tid; w.lane = lane;
  if (tid == 0) {
    for (int s = 0; s < G::NST; ++s) { mbar_init(w.full(s), G::NTHR); mbar_init(w.empty(s), G::NWARP); }
    for (int k = 0; k < 16; ++k) mbar_init(s_u + G::OFF_BAR + 8 * (2 * G::NST + k), 1);
  }
  for (int k = tid; k < 64; k += G::NTHR) reinterpret_cast<float4*>(smem + G::OFF_GB)[k] = p.gb[k];
  for (int k = tid; k < 128; k += G::NTHR) reinterpret_cast<float*>(smem + G::OFF_GAM)[k] = p.gamma[k];
  for (int k = tid; k < 256; k += G::NTHR) reinterpret_cast<float*>(smem + G::OFF_DGB)[k] = 0.f;
  __syncthreads();
  if (n_items > 0) {
    for (int c = 0; c < G::AHEAD && c < total; ++c) w.issue(c % 16, c);
    const uint32_t hb = s_u + G::OFF_H + pair * 2 * G::HBUF;
#pragma unroll 1
    for (int it = 0; it < n_items; ++it) {
      const int tile = b + it * grid, r0 = tile * G::BM + 32 * pair;
      if (warp < 4) px_p_tile(p, w, s_u, hb, pair, r0, (tile + grid) * G::BM + 32 * pair, it + 1 < n_items, it, total);
      else px_x_tile(p, w, s_u, hb, pair, r0, it, total);
    }
    cp_async_wait<0>();
  }
  __syncthreads();
  for (int k = tid; k < 256; k += G::NTHR) p.dgb[(size_t)blockIdx.x * 256 + k] = reinterpret_cast<float*>(smem + G::OFF_DGB)[k];
}

}  // namespace a100
