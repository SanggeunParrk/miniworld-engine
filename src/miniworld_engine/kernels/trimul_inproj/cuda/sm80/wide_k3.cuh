// wide_k3.cuh -- the output side of the A100 "wide" TriMul (any width D, hidden channels Hc = D or 2 D): output projection of the LayerNormed contraction,
// input-LayerNorm gate, dropout row scale, residual -- two tensor-core GEMMs and the whole epilogue in one kernel.
//
//   p' = r_o (X^T . Wo'^T) - r_o mu_o s_o + e_o           Wo' = bf16(0.5 gamma_out Wo)   [D, Hc]   (X = contraction planes, channel-major [Hc, T])
//   g' = r_i (z   . Wg'^T) - r_i mu_i s_g + e_g           Wg' = bf16(0.5 gamma_in  Wg)   [D, D]
//   y  = bf16(z + bf16(bf16(p') (1 + tanh g') [ds[j]]))   (= z + sigmoid(g) p ds)
// Both LayerNorms are folded into the weights (wide_rows.cuh) and undone per token from the statistics of ln_stats_cm (over X) and ln_stats_rows
// (over z): the MMAs read the raw X / z operands.  The X operand is channel-major, so it is the M-major A operand (ldmatrix.trans); the z tile is the
// gate GEMM's A operand.
// Epilogue in two passes (the second is ``k3_residual_pass``): o = bf16(p' (1 + tanh g')) leaves the accumulators into a staged shared tile, then every thread
// takes 16-byte granules of it and adds the residual z (and the dropout scale) with coalesced 16-byte loads issued four granules ahead.  The first version loaded
// z / ds per accumulator pair, each load followed by its use: that serialised a global-memory latency per element and made the epilogue about half of the
// kernel's time (measured: the whole TriMul forward 3 % faster at D384 L768, the training forward 8 %).
// Training variant: p' and g' are also stored (bf16, the 0.5-scaled units) for the backward.
//
// CTA tile = 128 tokens x BN output channels, 8 warps (4 x 2), warp tile 32 x BN / 2 for both accumulators; one CTA per SM, a 4-stage ring.  (Tried and not kept: one
// accumulator set with a shared-memory stash of p' and two CTAs per SM, which ran exactly as fast once the epilogue was fixed.)
#pragma once
#include "wide_gemm.cuh"

namespace a100 {

// ---------------------------------------------------------------------------------------------------------------------------------------------------- residual pass
// The residual pass of the output stages: the staged tile at s0 holds o (bf16, rows of BN * 2 + 16 bytes); y = z + [ds] o, written in 16-byte granules.  The loads of z (and ds)
// are coalesced 16-byte loads issued four granules ahead of their use, so a thread never waits on a global load between two dependent instructions (the per-element 4-byte
// loads this replaces made the output stage's epilogue about half of its time).  With a row scale the product ``o ds`` is rounded to bf16 before the add, as before.
template <int BN, int NTHR>
DEVI void k3_residual_pass(uint32_t s0, const __nv_bfloat16* z, const __nv_bfloat16* ds, __nv_bfloat16* out, int t0, int n0, int D, int L, int tid) {
  constexpr int GR = BN / 8, PER = 128 * GR / NTHR;      // granules per row; granules per thread
  static_assert(PER % 4 == 0, "batches of four granules");
#pragma unroll
  for (int b = 0; b < PER / 4; ++b) {
    uint4 zv[4], dv[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int idx = tid + NTHR * (4 * b + e), r = idx / GR, gk = idx % GR;
      const size_t off = (size_t)(t0 + r) * D + n0 + 8 * gk;
      zv[e] = __ldg(reinterpret_cast<const uint4*>(z + off));
      dv[e] = ds != nullptr ? __ldg(reinterpret_cast<const uint4*>(ds + (size_t)((t0 + r) % L) * D + n0 + 8 * gk)) : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int idx = tid + NTHR * (4 * b + e), r = idx / GR, gk = idx % GR;
      const uint4 ov = lds128(s0 + r * (BN * 2 + 16) + gk * 16);
      const uint32_t ow[4] = {ov.x, ov.y, ov.z, ov.w}, zw[4] = {zv[e].x, zv[e].y, zv[e].z, zv[e].w}, dw[4] = {dv[e].x, dv[e].y, dv[e].z, dv[e].w};
      uint32_t yw[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t o = ds != nullptr ? pack_bf16(bf16lo(ow[i]) * bf16lo(dw[i]), bf16hi(ow[i]) * bf16hi(dw[i])) : ow[i];
        yw[i] = add_bf16x2(zw[i], o);
      }
      stg128(out + (size_t)(t0 + r) * D + n0 + 8 * gk, make_uint4(yw[0], yw[1], yw[2], yw[3]));
    }
  }
}


struct K3wParams {
  const __nv_bfloat16* x;      // [Hc][T] contraction output, channel-major
  const __nv_bfloat16* z;      // [T][D]
  const __nv_bfloat16* wo;     // [D][Hc]
  const __nv_bfloat16* wg;     // [D][D]
  const float *so, *eo, *sg, *eg;   // [D] fold vectors (0.5-scaled)
  const float2* sto;           // [T] (mean, rstd) of X over its Hc channels (XSTATS = false)
  float2* stw;                 // XSTATS: written here (mean, rstd of X over its Hc channels, two-pass fp32), every CTA owns its token tile (D <= BN)
  float eps;
  const float2* sti;           // [T] (mean, rstd) of z over its D channels
  const __nv_bfloat16* ds;     // [L][D] dropout row scale or nullptr
  __nv_bfloat16* out;          // [T][D]
  __nv_bfloat16* ps;           // training: p' [T][D] or nullptr
  __nv_bfloat16* gs;           // training: g' [T][D] or nullptr
  int T, L, D, Hc;
};

template <int BN>
struct K3wCfg {
  using TP = WTile<128, BN, 4, 2, 4, true, false>;
  using TG = WTile<128, BN, 4, 2, 4, false, false>;
  static constexpr int NTHR = TP::NTHR, SMEM = TP::SMEM, SMEM_X = TP::SMEM + 3072;     // + the statistics exchange: sums, squares, (mean, rstd) of the 128 tokens
  static_assert(TP::SMEM == TG::SMEM, "one ring for both phases");
};

// XSTATS (Hc <= 128: every k-tile of the X operand is still in its ring slot when the first GEMM ends): the LayerNorm_out statistics come from the tile instead of
// ln_stats_cm -- thread (token, half) sums its half of the channels from shared memory, two passes, the halves merged through shared memory (exact in fp32)
template <int BN, bool XSTATS>
__global__ void __launch_bounds__(256, BN == 64 ? 2 : 1) k3w_kernel(const K3wParams p) {
  using Cfg = K3wCfg<BN>;
  using TP = typename Cfg::TP;
  using TG = typename Cfg::TG;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = smem_u32(smem);
  const int ntn = p.D / BN;
  const int mtile = (int)blockIdx.x / ntn, ntile = (int)blockIdx.x - mtile * ntn;
  const int t0 = mtile * 128, n0 = ntile * BN;
  TP tp;
  tp.run_fast(s0, p.Hc >> 5, GemmOps{p.x, (size_t)p.T, t0, p.T, p.wo, (size_t)p.Hc, n0, p.D});
  if constexpr (XSTATS) {
    const uint32_t xs = s0 + TP::SMEM;                              // float sum[2][128], sq[2][128], float2 sts[128]
    const int nk1 = p.Hc >> 5, tk = threadIdx.x & 127, hf = threadIdx.x >> 7;
    auto elem = [&](int kt, int k) {
      unsigned short b;
      asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(b) : "r"(s0 + kt * TP::STAGE + wmmaj<TP::BM>(k, tk >> 3) + (tk & 7) * 2));
      return __uint_as_float(((uint32_t)b) << 16);
    };
    float sum = 0.f;
#pragma unroll 1
    for (int kt = 0; kt < nk1; ++kt)
#pragma unroll
      for (int kr = 0; kr < 16; ++kr) sum += elem(kt, 16 * hf + kr);
    sts32(xs + (hf * 128 + tk) * 4, __float_as_uint(sum));
    __syncthreads();
    const float mean = (__uint_as_float(lds32(xs + tk * 4)) + __uint_as_float(lds32(xs + (128 + tk) * 4))) * (1.f / p.Hc);
    float q2 = 0.f;
#pragma unroll 1
    for (int kt = 0; kt < nk1; ++kt)
#pragma unroll
      for (int kr = 0; kr < 16; ++kr) { const float a = elem(kt, 16 * hf + kr) - mean; q2 = fmaf(a, a, q2); }
    sts32(xs + 1024 + (hf * 128 + tk) * 4, __float_as_uint(q2));
    __syncthreads();
    if (hf == 0) {
      const float rstd = rsqrtf((__uint_as_float(lds32(xs + 1024 + tk * 4)) + __uint_as_float(lds32(xs + 1024 + (128 + tk) * 4))) * (1.f / p.Hc) + p.eps);
      sts64(xs + 2048 + tk * 8, make_uint2(__float_as_uint(mean), __float_as_uint(rstd)));
      p.stw[t0 + tk] = make_float2(mean, rstd);
    }
    __syncthreads();
  }
  TG tg;
  tg.run_fast(s0, p.D >> 5, GemmOps{p.z, (size_t)p.D, t0, p.T, p.wg, (size_t)p.D, n0, p.D});
  // ---- epilogue pass 1: o = bf16(p' (1 + tanh g')) into the staged tile (no global load in this loop; training keeps the fragment words of p' and g')
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1, g8 = lane >> 2, q = lane & 3;
  constexpr int MT = TP::MT, NT = 2 * TP::NP, WTN = TP::WTN;
  float ro[MT][2], mro[MT][2], ri[MT][2], mri[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int tok = t0 + 32 * wm + 16 * mt + g8 + 8 * rh;
      float2 a;
      if constexpr (XSTATS) { const uint2 u = lds64(s0 + TP::SMEM + 2048 + (32 * wm + 16 * mt + g8 + 8 * rh) * 8); a = make_float2(__uint_as_float(u.x), __uint_as_float(u.y)); }
      else a = __ldg(p.sto + tok);
      const float2 b = __ldg(p.sti + tok);
      ro[mt][rh] = a.y; mro[mt][rh] = a.x * a.y;
      ri[mt][rh] = b.y; mri[mt][rh] = b.x * b.y;
    }
  uint32_t wp[MT][NT][2], wg[MT][NT][2];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int col = n0 + WTN * wn + 8 * nt + 2 * q;
    const float2 vso = __ldg(reinterpret_cast<const float2*>(p.so + col)), veo = __ldg(reinterpret_cast<const float2*>(p.eo + col));
    const float2 vsg = __ldg(reinterpret_cast<const float2*>(p.sg + col)), veg = __ldg(reinterpret_cast<const float2*>(p.eg + col));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const float p0 = round_bf16f(fmaf(ro[mt][rh], tp.acc[mt][nt][2 * rh], fmaf(-mro[mt][rh], vso.x, veo.x)));
        const float p1 = round_bf16f(fmaf(ro[mt][rh], tp.acc[mt][nt][2 * rh + 1], fmaf(-mro[mt][rh], vso.y, veo.y)));
        const float g0 = fmaf(ri[mt][rh], tg.acc[mt][nt][2 * rh], fmaf(-mri[mt][rh], vsg.x, veg.x));
        const float g1 = fmaf(ri[mt][rh], tg.acc[mt][nt][2 * rh + 1], fmaf(-mri[mt][rh], vsg.y, veg.y));
        if (p.ps != nullptr) { wp[mt][nt][rh] = pack_bf16(p0, p1); wg[mt][nt][rh] = pack_bf16(g0, g1); }
        stage_word(s0, 32 * wm + 16 * mt + g8 + 8 * rh, (WTN / 2) * wn + 4 * nt + q, BN, pack_bf16(fmaf(p0, tanh_approx(g0), p0), fmaf(p1, tanh_approx(g1), p1)));
      }
  }
  __syncthreads();
  k3_residual_pass<BN, 256>(s0, p.z, p.ds, p.out, t0, n0, p.D, p.L, tid);
  auto emit = [&](const uint32_t (&w)[MT][NT][2], __nv_bfloat16* dst) {
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) stage_word(s0, 32 * wm + 16 * mt + g8 + 8 * rh, (WTN / 2) * wn + 4 * nt + q, BN, w[mt][nt][rh]);
    __syncthreads();
    tile_store16<128, BN, 256>(s0, dst + (size_t)t0 * p.D + n0, p.D);
  };
  if (p.ps != nullptr) { emit(wp, p.ps); emit(wg, p.gs); }
}

}  // namespace a100
