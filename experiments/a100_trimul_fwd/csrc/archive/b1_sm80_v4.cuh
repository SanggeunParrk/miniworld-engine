// b1_sm80.cuh -- TriMul backward, output side (B1), A100 / sm_80.
//
// Forward (per token t, recomputed here):  x_hat = LN(X[:, t]) (no affine), y_n = x_hat g_o + b_o, o = y_n . Wo^T,
//                                           g = LN_in(z[t]) . Wog^T, s = sigmoid(g), out = z + (s o) ds[j]
// Backward from dy:  dupd = dy ds[j];  d_o = dupd s;  d_g = dupd o s (1 - s)
//                    dx_hat = d_o . (Wo diag g_o) = d_o . Wo'          (so the LN affine never needs dividing out)
//                    dX = r (dx_hat - mean(dx_hat) - x_hat mean(dx_hat x_hat))
// Outputs: dX planes [CH][T]; A_o = d_o r_o (dW_o is then a long-K GEMM against the raw X plus rank-1 corrections, done by the host; dW_og
// = d_g^T x_n needs no fold); d_g into the dx operand buffer; per-token (mu_o, r_o, mu_i, r_i).
//
// Structure (as K3): NG independent groups of 4 warps sharing the resident Wo' | Wog'; a group walks 32-token tiles; warp nw owns output channels
// [32 nw, 32 nw + 32) for o / g and plane channels [CH/4 nw, CH/4 (nw + 1)) for dX.  LN statistics on the tensor cores (sum = x . 1, sum of
// squares = Gram diagonal), K split over the 4 warps and exchanged.  The same o / g folds as K3: o = r (X . Wo'^T) - r mu s_o + b_o.
#pragma once
#include <type_traits>
#include "sm80_common.cuh"

namespace a100 {

struct B1Params {
  const __nv_bfloat16* x;      // [CH][T]
  const __nv_bfloat16* z;      // [T][128]
  const __nv_bfloat16* dy;     // [T][128]
  const __nv_bfloat16* ds;     // [L][128] dropout scale or nullptr
  const __nv_bfloat16* wo;     // [128][CH] bf16(Wo diag gamma_out)
  const __nv_bfloat16* wg;     // [128][128] bf16(Wog diag gamma_in)
  const float* so; const float* bo; const float* sg; const float* bg;   // [128] fold vectors (unscaled)
  __nv_bfloat16* dx;           // [CH][T]
  __nv_bfloat16* ao;           // [T][128]  d_o r_o
  __nv_bfloat16* dg;           // d_g at dg + t * ldg (the dx operand buffer's last 128 columns)
  float* stats;                // [T][4]  mu_o, r_o, mu_i, r_i
  __nv_bfloat16* xn;           // [T][128] x_n = LN_in(z) (with affine), for the input-side kernel
  const float* gin; const float* bin;   // LN_in affine [128]
  float* red;                  // [gridDim.x * NG][2][128]: per-group sums of A_o mu_o, d_o (the LN_out rank-1 folds' vectors)
  int T, L, num_tiles, ldg;
  float eps;
};

template <int CH_, int NG_ = 2>
struct B1Cfg {
  static constexpr int CH = CH_, CZ = 128, BM = 32, NG = NG_, NTHR = 128 * NG;
  static constexpr int KSX = CH / 16, KSZ = CZ / 16, CQ = CH / 4;         // CQ: this warp's dX channels
  static constexpr int SMEM_WO = CZ * CH * 2, SMEM_WG = CZ * CZ * 2;
  static constexpr int SMEM_X = CH * BM * 2, SMEM_Z = BM * CZ * 2;          // per group; X doubles as the dX staging, z as the d_o exchange
  static constexpr int SMEM_ST = 3 * 4 * 32 * 2 * 4;                        // per group: [x | z | ln-bwd][nw][32 rows][2]
  static constexpr int SMEM_GRP = SMEM_X + SMEM_Z + SMEM_ST;
  static constexpr int SMEM_V = 6 * CZ * 4;                                 // s_o, b_o, s_g, b_g, gamma_in, beta_in
  static constexpr int SMEM = SMEM_WO + SMEM_WG + NG * SMEM_GRP + SMEM_V + NG * 2 * 8;
  static_assert(SMEM <= 166912, "sm_80 dynamic shared memory");
  static_assert(KSX % 4 == 0 && CQ % 16 == 0, "K split");
};

DEVI uint32_t b1_swz64(uint32_t row, uint32_t g) { return row * 64 + ((g ^ ((row >> 1) & 3u)) << 4); }

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) b1_kernel(const B1Params p) {
  constexpr int CH = G::CH, CZ = G::CZ, BM = G::BM, KSX = G::KSX, KSZ = G::KSZ, CQ = G::CQ;
  constexpr uint32_t ONES = 0x3f803f80u;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int grp = warp >> 2, nw = warp & 3, gtid = tid & 127, g8 = lane >> 2, q = lane & 3;
  uint8_t* sWo = smem;
  uint8_t* sWg = sWo + G::SMEM_WO;
  uint8_t* sGrp = sWg + G::SMEM_WG + grp * G::SMEM_GRP;
  float* sSt = reinterpret_cast<float*>(sGrp + G::SMEM_X + G::SMEM_Z);
  float* sV = reinterpret_cast<float*>(sWg + G::SMEM_WG + G::NG * G::SMEM_GRP);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sV + 6 * CZ) + grp * 2;
  const uint32_t sWo_u = smem_u32(sWo), sWg_u = smem_u32(sWg), sX_u = smem_u32(sGrp), sZ_u = sX_u + G::SMEM_X;
  const uint32_t barX = smem_u32(bars), barZ = barX + 8;
  const int bar_id = 1 + grp;

  const int gstride = G::NG * (int)gridDim.x, gfirst = G::NG * (int)blockIdx.x + grp;
  const int n_iter = gfirst < p.num_tiles ? (p.num_tiles - gfirst + gstride - 1) / gstride : 0;
  if (gtid == 0) { mbar_init(barX, 128); mbar_init(barZ, 128); }
  for (int i = tid; i < CZ; i += G::NTHR) { sV[i] = p.so[i]; sV[CZ + i] = p.bo[i]; sV[2 * CZ + i] = p.sg[i]; sV[3 * CZ + i] = p.bg[i]; sV[4 * CZ + i] = p.gin[i]; sV[5 * CZ + i] = p.bin[i]; }
  for (int c = tid; c < CZ * CH / 8; c += G::NTHR) {
    const int row = c / (CH / 8), g = c % (CH / 8);
    cp_async16(sWo_u + swz<CH * 2>(row, g * 16), p.wo + (size_t)row * CH + g * 8);
  }
  for (int c = tid; c < CZ * CZ / 8; c += G::NTHR) {
    const int row = c >> 4, g = c & 15;
    cp_async16(sWg_u + swz<256>(row, g * 16), p.wg + (size_t)row * CZ + g * 8);
  }
  cp_async_wait<0>();
  __syncthreads();
  auto load_x = [&](int tile) {
    const int t0 = tile * BM;
#pragma unroll
    for (int i = 0; i < CH * 4 / 128; ++i) {
      const int c = gtid + 128 * i, row = c >> 2, g = c & 3;
      const bool ok = t0 + 8 * g < p.T;
      cp_async16(sX_u + b1_swz64(row, g), p.x + (size_t)row * p.T + (ok ? t0 + 8 * g : 0), ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barX);
  };
  auto load_z = [&](int tile) {
    const int t0 = tile * BM;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = gtid + 128 * i, row = c >> 4, g = c & 15;
      const bool ok = t0 + row < p.T;
      cp_async16(sZ_u + swz<256>(row, g * 16), p.z + (size_t)(ok ? t0 + row : 0) * CZ + g * 8, ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barZ);
  };
  if (n_iter > 0) { load_x(gfirst); load_z(gfirst); }

  // ---- tensor-core row statistics over k-steps [KS0, KS0 + NK) (see K3), exchanged across the group's 4 warps
  auto partial = [&](const auto& f, auto ks0c, auto nkc, float (&ps)[2][2][2]) {
    constexpr int KS0 = decltype(ks0c)::value, NK = decltype(nkc)::value;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      float s[4] = {0.f, 0.f, 0.f, 0.f}, g0[4] = {0.f, 0.f, 0.f, 0.f}, g1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int k = 0; k < NK; ++k) {
        const uint32_t (&a)[4] = f[mt][KS0 + k];
        mma16816(s, a, ONES, ONES);
        mma16816(g0, a, a[0], a[2]);
        mma16816(g1, a, a[1], a[3]);
      }
      const bool odd = g8 & 1;
      ps[mt][0][0] = s[0]; ps[mt][1][0] = s[2];
      ps[mt][0][1] = odd ? g0[1] : g0[0];
      ps[mt][1][1] = odd ? g1[3] : g1[2];
    }
  };
  auto quarter = [&](const auto& f, auto nks, float (&ps)[2][2][2]) {
    constexpr int NQ = decltype(nks)::value / 4;
    switch (nw) {
      case 0: partial(f, std::integral_constant<int, 0>{}, std::integral_constant<int, NQ>{}, ps); break;
      case 1: partial(f, std::integral_constant<int, NQ>{}, std::integral_constant<int, NQ>{}, ps); break;
      case 2: partial(f, std::integral_constant<int, 2 * NQ>{}, std::integral_constant<int, NQ>{}, ps); break;
      default: partial(f, std::integral_constant<int, 3 * NQ>{}, std::integral_constant<int, NQ>{}, ps); break;
    }
  };
  // exchange 2-value partials (held by lane q == g8 / 2 when diag, or any lane) of the 4 warps; returns the group sums of this thread's 4 rows
  auto exchange = [&](float (&ps)[2][2][2], int which, bool diag_lane_only, float (&sum)[2][2][2]) {
    float* base = sSt + which * (4 * 64);
    if (diag_lane_only ? (q == (g8 >> 1)) : (q == 0)) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          *reinterpret_cast<float2*>(base + nw * 64 + 2 * (16 * mt + g8 + 8 * h)) = make_float2(ps[mt][h][0], ps[mt][h][1]);
    }
    bar_sync(bar_id, 128);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int rr = 16 * mt + g8 + 8 * h;
        float a = 0.f, b = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) { const float2 v = *reinterpret_cast<const float2*>(base + w * 64 + 2 * rr); a += v.x; b += v.y; }
        sum[mt][h][0] = a; sum[mt][h][1] = b;
      }
  };
  auto to_stats = [&](float (&sum)[2][2][2], float n_inv, float (&mu)[2][2], float (&rs)[2][2]) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float m = sum[mt][h][0] * n_inv;
        mu[mt][h] = m;
        rs[mt][h] = rsqrtf(fmaxf(sum[mt][h][1] * n_inv - m * m, 0.f) + p.eps);
      }
  };
  auto zero4 = [](float (&a)[4]) { a[0] = a[1] = a[2] = a[3] = 0.f; };
  const int mi = lane >> 3, r8 = lane & 7, kx = (mi >> 1) * 8 + r8;
  const int zr = r8 + ((lane >> 3) & 1) * 8;

  float red[2][4][2];                            // [A_o mu_o | d_o][n8][2]: this thread's 8 output channels, its rows
#pragma unroll
  for (int k = 0; k < 2; ++k)
#pragma unroll
    for (int n = 0; n < 4; ++n) red[k][n][0] = red[k][n][1] = 0.f;
  for (int it = 0; it < n_iter; ++it) {
    const int tile = gfirst + it * gstride;
    const int t0 = tile * BM;
    // ================= X: statistics and o (warp's 32 output channels) =================
    // dy (and the dropout scale) of this thread's (row, output) pairs: issued now, consumed at d_o, so the DRAM latency hides under the X / z work
    uint32_t dyr[2][2][4], dsr[2][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int t = t0 + 16 * mt + g8 + 8 * h;
        const bool ok = t < p.T;
        const unsigned int* dyp = reinterpret_cast<const unsigned int*>(p.dy + (size_t)t * CZ + 32 * nw + 2 * q);
        const bool use_ds = ok && p.ds != nullptr;
        const unsigned int* dsp = reinterpret_cast<const unsigned int*>(p.ds + (size_t)(use_ds ? t % p.L : 0) * CZ + 32 * nw + 2 * q);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          dyr[mt][h][n] = ok ? __ldg(dyp + 4 * n) : 0u;
          dsr[mt][h][n] = use_ds ? __ldg(dsp + 4 * n) : 0x3f803f80u;
        }
      }
    mbar_wait(barX, it & 1);
    float o_acc[2][4][4];                        // [mt][n8 of 32 outputs][4]
    float mux[2][2], rsx[2][2];
    {
      constexpr int NQ = KSX / 4;
      uint32_t fq[2][NQ][4];                     // this warp's K quarter, for the statistics
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int k = 0; k < NQ; ++k) ldsm_x4_t(fq[mt][k], sX_u + b1_swz64(16 * (nw * NQ + k) + kx, 2 * mt + (mi & 1)));
      float ps[2][2][2], sum[2][2][2];
      partial(fq, std::integral_constant<int, 0>{}, std::integral_constant<int, NQ>{}, ps);
      exchange(ps, 0, true, sum);
      to_stats(sum, 1.f / CH, mux, rsx);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int n = 0; n < 4; ++n) zero4(o_acc[mt][n]);
#pragma unroll
      for (int ks = 0; ks < KSX; ++ks) {         // A fragments streamed from the tile
        uint32_t fx[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) ldsm_x4_t(fx[mt], sX_u + b1_swz64(16 * ks + kx, 2 * mt + (mi & 1)));
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
          uint32_t b[4];
          ldsm_x4(b, sWo_u + swz<CH * 2>(32 * nw + 16 * nb + ((lane >> 4) & 1) * 8 + r8, (2 * ks + ((lane >> 3) & 1)) * 16));
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) { mma16816(o_acc[mt][2 * nb], fx[mt], b[0], b[1]); mma16816(o_acc[mt][2 * nb + 1], fx[mt], b[2], b[3]); }
        }
      }
    }
    // o = r acc - r mu s_o + b_o   (fp32, the forward's value before its bf16 rounding)
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int oc = 32 * nw + 8 * n + 2 * q;
      const float2 s2 = *reinterpret_cast<const float2*>(sV + oc), b2 = *reinterpret_cast<const float2*>(sV + CZ + oc);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float r = rsx[mt][h], rm = r * mux[mt][h];
          o_acc[mt][n][2 * h] = fmaf(r, o_acc[mt][n][2 * h], fmaf(-rm, s2.x, b2.x));
          o_acc[mt][n][2 * h + 1] = fmaf(r, o_acc[mt][n][2 * h + 1], fmaf(-rm, s2.y, b2.y));
        }
    }
    // ================= z: statistics and the gate; d_o, d_g =================
    mbar_wait(barZ, it & 1);
    float muz[2][2], rsz[2][2];
    float g_acc[2][4][4];
    {
      uint32_t fz[2][KSZ][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int ks = 0; ks < KSZ; ++ks) ldsm_x4(fz[mt][ks], sZ_u + swz<256>(16 * mt + zr, (2 * ks + (lane >> 4)) * 16));
      float ps[2][2][2], sum[2][2][2];
      quarter(fz, std::integral_constant<int, KSZ>{}, ps);
      exchange(ps, 1, true, sum);                // (its barrier also orders every warp's z reads before the d_o exchange reuses the z tile)
      to_stats(sum, 1.f / CZ, muz, rsz);
      // x_n of the tile: row 16 mt + g8 + 8 h, granule 4 nw + q (the 16 threads that know a row's statistics cover its 16 granules)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = 16 * mt + g8 + 8 * h, g = 4 * nw + q;
          if (t0 + row < p.T) {
            uint4 v = lds128(sZ_u + swz<256>(row, g * 16));
            uint32_t* w = reinterpret_cast<uint32_t*>(&v);
            const float m = muz[mt][h], r = rsz[mt][h];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int c = 8 * g + 2 * e;
              w[e] = pack_bf16(fmaf((bf16lo(w[e]) - m) * r, sV[4 * CZ + c], sV[5 * CZ + c]), fmaf((bf16hi(w[e]) - m) * r, sV[4 * CZ + c + 1], sV[5 * CZ + c + 1]));
            }
            stg128(p.xn + (size_t)(t0 + row) * CZ + 8 * g, v);
          }
        }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int n = 0; n < 4; ++n) zero4(g_acc[mt][n]);
#pragma unroll
      for (int nb = 0; nb < 2; ++nb)
#pragma unroll
        for (int ks = 0; ks < KSZ; ++ks) {
          uint32_t b[4];
          ldsm_x4(b, sWg_u + swz<256>(32 * nw + 16 * nb + ((lane >> 4) & 1) * 8 + r8, (2 * ks + ((lane >> 3) & 1)) * 16));
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) { mma16816(g_acc[mt][2 * nb], fz[mt][ks], b[0], b[1]); mma16816(g_acc[mt][2 * nb + 1], fz[mt][ks], b[2], b[3]); }
        }
    }
    // per-token statistics out (warp 0, one lane per row)
    if (nw == 0 && q == 0) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int t = t0 + 16 * mt + g8 + 8 * h;
          if (t < p.T) *reinterpret_cast<float4*>(p.stats + (size_t)t * 4) = make_float4(mux[mt][h], rsx[mt][h], muz[mt][h], rsz[mt][h]);
        }
    }
    bar_sync(bar_id, 128);                       // every warp is past its x_n reads of the z tile
    // d_o, d_g at this warp's 32 outputs; d_o -> the group's exchange tile (the z tile)
    const uint32_t sDo = sZ_u;                   // [32 tok][128] bf16, 256 B rows, granule ^= row & 7
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int oc = 32 * nw + 8 * n + 2 * q;
      const float2 s2 = *reinterpret_cast<const float2*>(sV + 2 * CZ + oc), b2 = *reinterpret_cast<const float2*>(sV + 3 * CZ + oc);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = 16 * mt + g8 + 8 * h, t = t0 + row;
          const bool ok = t < p.T;
          const float r = rsz[mt][h], rm = r * muz[mt][h];
          const float g0 = fmaf(r, g_acc[mt][n][2 * h], fmaf(-rm, s2.x, b2.x)), g1 = fmaf(r, g_acc[mt][n][2 * h + 1], fmaf(-rm, s2.y, b2.y));
          // s = 1/2 + th/2, s (1 - s) = (1 - th^2) / 4, th = tanh(g / 2)
          const float th0 = tanh_approx(0.5f * g0), th1 = tanh_approx(0.5f * g1);
          const float sg0 = fmaf(0.5f, th0, 0.5f), sg1 = fmaf(0.5f, th1, 0.5f);
          const uint32_t dyv = dyr[mt][h][n], dsv = dsr[mt][h][n];
          const float du0 = bf16lo(dyv) * bf16lo(dsv), du1 = bf16hi(dyv) * bf16hi(dsv);
          const float o0 = o_acc[mt][n][2 * h], o1 = o_acc[mt][n][2 * h + 1];
          const float do0 = du0 * sg0, do1 = du1 * sg1;
          const float dg0 = 0.25f * du0 * o0 * fmaf(-th0, th0, 1.f), dg1 = 0.25f * du1 * o1 * fmaf(-th1, th1, 1.f);
          sts32(sDo + row * 256 + ((((oc >> 3) ^ (row & 7))) << 4) + (oc & 7) * 2, pack_bf16(do0, do1));
          if (ok) {
            const uint32_t aw = pack_bf16(do0 * rsx[mt][h], do1 * rsx[mt][h]);
            stg32(p.ao + (size_t)t * CZ + oc, aw);
            stg32(p.dg + (size_t)t * p.ldg + oc, pack_bf16(dg0, dg1));
            // the fold pairs the stored (bf16) A_o with the statistics, exactly as the GEMM sees it
            red[0][n][0] = fmaf(bf16lo(aw), mux[mt][h], red[0][n][0]); red[0][n][1] = fmaf(bf16hi(aw), mux[mt][h], red[0][n][1]);
            red[1][n][0] += do0; red[1][n][1] += do1;
          }
        }
    }
    bar_sync(bar_id, 128);                       // the d_o tile is complete
    // ================= dx_hat = d_o . Wo' for this warp's CQ plane channels =================
    float dxh[2][CQ / 8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int n = 0; n < CQ / 8; ++n) zero4(dxh[mt][n]);
#pragma unroll
    for (int ks = 0; ks < KSZ; ++ks) {           // K = the 128 outputs
      uint32_t a[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int row = 16 * mt + zr, g = 2 * ks + (lane >> 4);
        ldsm_x4(a[mt], sDo + row * 256 + ((g ^ (row & 7)) << 4));
      }
#pragma unroll
      for (int np = 0; np < CQ / 16; ++np) {     // B (k = output o, n = plane channel) from Wo' [o][ch] by ldmatrix.trans
        uint32_t b[4];
        const int krow = 16 * ks + (mi & 1) * 8 + r8, nch = nw * CQ + 16 * np + (mi >> 1) * 8;
        ldsm_x4_t(b, sWo_u + swz<CH * 2>(krow, nch * 2));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(dxh[mt][2 * np], a[mt], b[0], b[1]); mma16816(dxh[mt][2 * np + 1], a[mt], b[2], b[3]); }
      }
    }
    // ================= LN_out backward: x_hat of the same elements (the warp's quarter of X, reloaded from the tile) =================
    float xh[2][CQ / 8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kq = 0; kq < CQ / 16; ++kq) {
        uint32_t f[4];
        ldsm_x4_t(f, sX_u + b1_swz64(nw * CQ + 16 * kq + kx, 2 * mt + (mi & 1)));
        // A-fragment regs: 0 (row g8, k 2q..) 1 (row g8+8, k 2q..) 2 (row g8, k 8 + 2q..) 3 (row g8+8, k 8 + 2q..) == C layout of n8 tiles 2kq, 2kq+1
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {         // n8 tile 2 kq + hh
          const uint32_t va = f[2 * hh], vb = f[2 * hh + 1];
          xh[mt][2 * kq + hh][0] = (bf16lo(va) - mux[mt][0]) * rsx[mt][0];
          xh[mt][2 * kq + hh][1] = (bf16hi(va) - mux[mt][0]) * rsx[mt][0];
          xh[mt][2 * kq + hh][2] = (bf16lo(vb) - mux[mt][1]) * rsx[mt][1];
          xh[mt][2 * kq + hh][3] = (bf16hi(vb) - mux[mt][1]) * rsx[mt][1];
        }
      }
    {
      float ps[2][2][2], sum[2][2][2];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float a = 0.f, b = 0.f;
#pragma unroll
          for (int n = 0; n < CQ / 8; ++n) {
            a += dxh[mt][n][2 * h] + dxh[mt][n][2 * h + 1];
            b = fmaf(dxh[mt][n][2 * h], xh[mt][n][2 * h], fmaf(dxh[mt][n][2 * h + 1], xh[mt][n][2 * h + 1], b));
          }
          ps[mt][h][0] = quad_sum(a); ps[mt][h][1] = quad_sum(b);
        }
      exchange(ps, 2, false, sum);               // (its barrier also orders every warp's X reads before the dX staging reuses the X tile)
      // dX = r (dx_hat - mean(dx_hat) - x_hat mean(dx_hat x_hat)) -> the X tile as staging [CH][32 tok] (same swizzle) -> plane rows
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int n = 0; n < CQ / 8; ++n)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float m1 = sum[mt][h][0] * (1.f / CH), m2 = sum[mt][h][1] * (1.f / CH), r = rsx[mt][h];
            const int row = 16 * mt + g8 + 8 * h;                       // token
            const int ch = nw * CQ + 8 * n + 2 * q;
            const float d0 = r * (dxh[mt][n][2 * h] - m1 - xh[mt][n][2 * h] * m2);
            const float d1 = r * (dxh[mt][n][2 * h + 1] - m1 - xh[mt][n][2 * h + 1] * m2);
            // staging element (ch, row): 2-byte stores into the [ch][32 tok] tile
            const uint32_t a0 = sX_u + b1_swz64(ch, row >> 3) + (row & 7) * 2, a1 = sX_u + b1_swz64(ch + 1, row >> 3) + (row & 7) * 2;
            asm volatile("st.shared.b16 [%0], %1;\n" ::"r"(a0), "h"(__bfloat16_as_ushort(__float2bfloat16_rn(d0))) : "memory");
            asm volatile("st.shared.b16 [%0], %1;\n" ::"r"(a1), "h"(__bfloat16_as_ushort(__float2bfloat16_rn(d1))) : "memory");
          }
    }
    bar_sync(bar_id, 128);                       // staging complete
#pragma unroll
    for (int i = 0; i < CH * 4 / 128; ++i) {     // [CH][32 tok] -> planes, 16 B granules
      const int c = gtid + 128 * i, ch = c >> 2, g = c & 3;
      if (t0 + 8 * g < p.T) stg128(p.dx + (size_t)ch * p.T + t0 + 8 * g, lds128(sX_u + b1_swz64(ch, g)));
    }
    bar_sync(bar_id, 128);                       // X and z tiles free
    if (it + 1 < n_iter) { load_x(tile + gstride); load_z(tile + gstride); }
  }
  cp_async_wait<0>();
  // fold vectors: reduce over the warp's 8 row lanes, one partial per (group, channel)
  float* out = p.red + ((size_t)blockIdx.x * G::NG + grp) * 2 * CZ;
#pragma unroll
  for (int k = 0; k < 2; ++k)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float v = red[k][n][j];
        v += __shfl_xor_sync(0xffffffffu, v, 4); v += __shfl_xor_sync(0xffffffffu, v, 8); v += __shfl_xor_sync(0xffffffffu, v, 16);
        if (g8 == 0) out[k * CZ + 32 * nw + 8 * n + 2 * q + j] = v;
      }
}

}  // namespace a100
