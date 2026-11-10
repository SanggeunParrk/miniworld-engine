// wide_k32.cuh -- the fp32 (TF32 tensor-core) front, output side and front backward of the A100 "wide" TriMul: the math, the folds, the epilogues and the data flow
// of wide_k1.cuh / wide_k3.cuh / wide_k1b.cuh with fp32 storage, TF32 MMAs (wide_gemm32.cuh) and fp32 accumulate; nothing is rounded to a narrower type (the product
// of the gate is not rounded either: the fp32 reference has no bf16 rounding points).  The folded weights are TF32-rounded and their sums are taken over the rounded values.
//
//   k1w32   plane[pc, t] = sigmoid(g) p * m_i m_j      (fp32 planes [2 Hs, T] channel-major, staged 512-byte token rows)
//   k3w32   y = z + ds sigmoid(g) p,  p' / g' optionally stored (fp32, the 0.5-scaled units)
//   k1wb32  dpre = (dg, dp) in the packed-row order, token-major fp32 [T, ldd]; the plane gradient da is read straight from global (fp32: 8 tokens x 4 bytes = a 32-byte
//           sector per channel row, so no shared staging)
#pragma once
#include "wide_gemm32.cuh"

namespace a100 {

// ------------------------------------------------------------------------------------------------------------------------------------ k1w32
struct K1w32Params {
  const float* z;              // [T][D]
  const float* w1;             // [4 Hs][D] packed rows, TF32-rounded
  const float* vs;             // [4 Hs]
  const float* vb;             // [4 Hs]
  const float2* st;            // [T] (mean, rstd) of the input rows
  const uint8_t* mask;         // [L] token mask or nullptr
  float* planes;               // [4 Hs / 2][T]
  int T, L, D, ncol;
};

using K1w32Tile = WTile32<128, 128, 2, 2, 4, false>;

__global__ void __launch_bounds__(128, 2) k1w32_kernel(const K1w32Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = smem_u32(smem);
  const int ntn = p.ncol >> 7;
  const int mtile = (int)blockIdx.x / ntn, ntile = (int)blockIdx.x - mtile * ntn;
  const int t0 = mtile * 128, j0 = ntile * 128;
  K1w32Tile tl;
  tl.run(s0, p.D >> 4, GemmOps32{p.z, (size_t)p.D, t0, p.T, p.w1, (size_t)p.D, j0, p.ncol});
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1, g8 = lane >> 2, q = lane & 3;
  constexpr int MT = K1w32Tile::MT, NP = K1w32Tile::NP;
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
#pragma unroll
  for (int np = 0; np < NP; ++np) {
    const int R = j0 + 64 * wn + 16 * np + 2 * q;
    const float2 sg = __ldg(reinterpret_cast<const float2*>(p.vs + R)), sp = __ldg(reinterpret_cast<const float2*>(p.vs + R + 8));
    const float2 bg = __ldg(reinterpret_cast<const float2*>(p.vb + R)), bp = __ldg(reinterpret_cast<const float2*>(p.vb + R + 8));
    const int ch = 32 * wn + 8 * np + 2 * q;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh)
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          const float ag = tl.acc[mt][2 * np][2 * rh + cc], ap = tl.acc[mt][2 * np + 1][2 * rh + cc];
          const float g = fmaf(rs[mt][rh], ag, fmaf(-mr[mt][rh], cc ? sg.y : sg.x, cc ? bg.y : bg.x));
          const float pp = fmaf(rs[mt][rh], ap, fmaf(-mr[mt][rh], cc ? sp.y : sp.x, cc ? bp.y : bp.x));
          stage_f1(s0, ch + cc, 64 * wm + 16 * mt + g8 + 8 * rh, 128, fmaf(pp, tanh_approx(g), pp) * mk[mt][rh]);
        }
  }
  __syncthreads();
  tile_store16f<64, 128, 128, 16>(s0, p.planes + (size_t)(ntile * 64) * p.T + t0, p.T);
}

// ------------------------------------------------------------------------------------------------------------------------------------ k3w32
struct K3w32Params {
  const float* x;              // [Hc][T] contraction output, channel-major
  const float* z;              // [T][D]
  const float* wo;             // [D][Hc]
  const float* wg;             // [D][D]
  const float *so, *eo, *sg, *eg;
  const float2* sto;
  const float2* sti;
  const float* ds;             // [L][D] or nullptr
  float* out;                  // [T][D]
  float* ps;                   // training: p' [T][D] or nullptr
  float* gs;                   // training: g' [T][D] or nullptr
  int T, L, D, Hc;
};

template <int BN>
struct K3w32Cfg {
  using TP = WTile32<128, BN, 4, 2, 5, true>;
  using TG = WTile32<128, BN, 4, 2, 5, false>;
  static constexpr int NTHR = TP::NTHR, SMEM = TP::SMEM;
  static_assert(TP::SMEM == TG::SMEM, "one ring for both phases");
  static_assert(128 * (BN * 4 + 32) <= SMEM, "the staged output tile fits the ring");
};

template <int BN>
__global__ void __launch_bounds__(256, 1) k3w32_kernel(const K3w32Params p) {
  using Cfg = K3w32Cfg<BN>;
  using TP = typename Cfg::TP;
  using TG = typename Cfg::TG;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = smem_u32(smem);
  const int ntn = p.D / BN;
  const int mtile = (int)blockIdx.x / ntn, ntile = (int)blockIdx.x - mtile * ntn;
  const int t0 = mtile * 128, n0 = ntile * BN;
  TP tp;
  tp.run(s0, p.Hc >> 4, GemmOps32{p.x, (size_t)p.T, t0, p.T, p.wo, (size_t)p.Hc, n0, p.D});
  TG tg;
  tg.run(s0, p.D >> 4, GemmOps32{p.z, (size_t)p.D, t0, p.T, p.wg, (size_t)p.D, n0, p.D});
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1, g8 = lane >> 2, q = lane & 3;
  constexpr int MT = TP::MT, NT = 2 * TP::NP, WTN = TP::WTN;
  float ro[MT][2], mro[MT][2], ri[MT][2], mri[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int tok = t0 + 32 * wm + 16 * mt + g8 + 8 * rh;
      const float2 a = __ldg(p.sto + tok), b = __ldg(p.sti + tok);
      ro[mt][rh] = a.y; mro[mt][rh] = a.x * a.y;
      ri[mt][rh] = b.y; mri[mt][rh] = b.x * b.y;
    }
  // epilogue pass 1: o = p' (1 + tanh g') into the staged fp32 tile (no global load in this loop; training keeps the fragment pairs of p' and g'); pass 2 (as k3_residual_pass in
  // wide_k3.cuh): y = z + [ds] o with coalesced 16-byte loads of z / ds
  float2 wp[MT][NT][2], wg_[MT][NT][2];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int col = n0 + WTN * wn + 8 * nt + 2 * q;
    const float2 vso = __ldg(reinterpret_cast<const float2*>(p.so + col)), veo = __ldg(reinterpret_cast<const float2*>(p.eo + col));
    const float2 vsg = __ldg(reinterpret_cast<const float2*>(p.sg + col)), veg = __ldg(reinterpret_cast<const float2*>(p.eg + col));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const float p0 = fmaf(ro[mt][rh], tp.acc[mt][nt][2 * rh], fmaf(-mro[mt][rh], vso.x, veo.x));
        const float p1 = fmaf(ro[mt][rh], tp.acc[mt][nt][2 * rh + 1], fmaf(-mro[mt][rh], vso.y, veo.y));
        const float g0 = fmaf(ri[mt][rh], tg.acc[mt][nt][2 * rh], fmaf(-mri[mt][rh], vsg.x, veg.x));
        const float g1 = fmaf(ri[mt][rh], tg.acc[mt][nt][2 * rh + 1], fmaf(-mri[mt][rh], vsg.y, veg.y));
        if (p.ps != nullptr) { wp[mt][nt][rh] = make_float2(p0, p1); wg_[mt][nt][rh] = make_float2(g0, g1); }
        stage_f2(s0, 32 * wm + 16 * mt + g8 + 8 * rh, WTN * wn + 8 * nt + 2 * q, BN, fmaf(p0, tanh_approx(g0), p0), fmaf(p1, tanh_approx(g1), p1));
      }
  }
  __syncthreads();
  {
    constexpr int GR = BN / 4, PER = 128 * GR / 256;                // 16-byte granules (4 floats) per row, per thread
    static_assert(PER % 4 == 0, "batches of four granules");
#pragma unroll
    for (int b = 0; b < PER / 4; ++b) {
      float4 zv[4], dv[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int idx = tid + 256 * (4 * b + e), r = idx / GR, gk = idx % GR;
        const size_t off = (size_t)(t0 + r) * p.D + n0 + 4 * gk;
        zv[e] = __ldg(reinterpret_cast<const float4*>(p.z + off));
        dv[e] = p.ds != nullptr ? __ldg(reinterpret_cast<const float4*>(p.ds + (size_t)((t0 + r) % p.L) * p.D + n0 + 4 * gk)) : make_float4(1.f, 1.f, 1.f, 1.f);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int idx = tid + 256 * (4 * b + e), r = idx / GR, gk = idx % GR;
        const float4 ov = *reinterpret_cast<const float4*>(smem + r * (BN * 4 + 32) + gk * 16);
        float4 y;
        if (p.ds != nullptr) { y.x = zv[e].x + ov.x * dv[e].x; y.y = zv[e].y + ov.y * dv[e].y; y.z = zv[e].z + ov.z * dv[e].z; y.w = zv[e].w + ov.w * dv[e].w; }
        else { y.x = zv[e].x + ov.x; y.y = zv[e].y + ov.y; y.z = zv[e].z + ov.z; y.w = zv[e].w + ov.w; }
        *reinterpret_cast<float4*>(p.out + (size_t)(t0 + r) * p.D + n0 + 4 * gk) = y;
      }
    }
  }
  auto emit = [&](const float2 (&w)[MT][NT][2], float* dst) {
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) stage_f2(s0, 32 * wm + 16 * mt + g8 + 8 * rh, WTN * wn + 8 * nt + 2 * q, BN, w[mt][nt][rh].x, w[mt][nt][rh].y);
    __syncthreads();
    tile_store16f<128, BN, 256, 32>(s0, dst + (size_t)t0 * p.D + n0, p.D);
  };
  if (p.ps != nullptr) { emit(wp, p.ps); emit(wg_, p.gs); }
}

// ----------------------------------------------------------------------------------------------------------------------------------- k1wb32
struct K1wb32Params {
  const float* z;
  const float* w1;
  const float* vs;
  const float* vb;
  const float2* st;
  const uint8_t* mask;
  const float* da;             // [2 Hs][T] plane gradients, channel-major
  float* dpre;                 // [T][ldd]
  int T, L, D, ncol, ldd;
};

using K1wb32Tile = WTile32<128, 128, 2, 2, 5, false>;
static_assert(128 * (128 * 4 + 32) <= K1wb32Tile::SMEM, "the staged dpre tile fits the ring");

__global__ void __launch_bounds__(128, 2) k1wb32_kernel(const K1wb32Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = smem_u32(smem);
  const int tid = threadIdx.x;
  const int ntn = p.ncol >> 7;
  const int mtile = (int)blockIdx.x / ntn, ntile = (int)blockIdx.x - mtile * ntn;
  const int t0 = mtile * 128, j0 = ntile * 128;
  K1wb32Tile tl;
  tl.run(s0, p.D >> 4, GemmOps32{p.z, (size_t)p.D, t0, p.T, p.w1, (size_t)p.D, j0, p.ncol});
  const int warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1, g8 = lane >> 2, q = lane & 3;
  constexpr int MT = K1wb32Tile::MT, NP = K1wb32Tile::NP;
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
  float2 wdg[MT][NP][2], wdp[MT][NP][2];
#pragma unroll
  for (int np = 0; np < NP; ++np) {
    const int R = j0 + 64 * wn + 16 * np + 2 * q;
    const float2 sg = __ldg(reinterpret_cast<const float2*>(p.vs + R)), sp = __ldg(reinterpret_cast<const float2*>(p.vs + R + 8));
    const float2 bg = __ldg(reinterpret_cast<const float2*>(p.vb + R)), bp = __ldg(reinterpret_cast<const float2*>(p.vb + R + 8));
    const int pch = ntile * 64 + 32 * wn + 8 * np + 2 * q;           // plane channel of this thread's first channel
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const int tok = t0 + 64 * wm + 16 * mt + g8 + 8 * rh;
        float dg[2], dp[2];
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          const float ag = tl.acc[mt][2 * np][2 * rh + cc], ap = tl.acc[mt][2 * np + 1][2 * rh + cc];
          const float g = fmaf(rs[mt][rh], ag, fmaf(-mr[mt][rh], cc ? sg.y : sg.x, cc ? bg.y : bg.x));
          const float pp = fmaf(rs[mt][rh], ap, fmaf(-mr[mt][rh], cc ? sp.y : sp.x, cc ? bp.y : bp.x));
          const float s = fmaf(0.5f, tanh_approx(g), 0.5f);
          const float dav = __ldg(p.da + (size_t)(pch + cc) * p.T + tok) * mk[mt][rh];
          dp[cc] = dav * s;
          dg[cc] = dav * (2.f * pp) * s * (1.f - s);
        }
        wdg[mt][np][rh] = make_float2(dg[0], dg[1]);
        wdp[mt][np][rh] = make_float2(dp[0], dp[1]);
      }
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int np = 0; np < NP; ++np)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const int trow = 64 * wm + 16 * mt + g8 + 8 * rh;
        stage_f2(s0, trow, 64 * wn + 16 * np + 2 * q, 128, wdg[mt][np][rh].x, wdg[mt][np][rh].y);
        stage_f2(s0, trow, 64 * wn + 16 * np + 8 + 2 * q, 128, wdp[mt][np][rh].x, wdp[mt][np][rh].y);
      }
  __syncthreads();
  tile_store16f<128, 128, 128, 32>(s0, p.dpre + (size_t)t0 * p.ldd + j0, p.ldd);
}

}  // namespace a100
