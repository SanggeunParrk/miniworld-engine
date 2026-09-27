// b1g_sm80.cuh -- TriMul backward, output side, A100 / sm_80: the sm_90 B1 algorithm (h100_sources/b1/b1_fused.cu) on sm_80 -- the forward
// saved the LayerNorm statistics and x_n, and the W_o weight gradient accumulates on chip for the whole launch.
//
// Per token (the forward's saved mu_o, r_o, x_n):  o = r (X . Wo'^T) - r mu s_o + b_o  (Wo' = Wo diag g_o),  g = x_n . Wog^T,  s = sigmoid(g)
//   dupd = dy ds[j];  d_o = dupd s;  d_g = dupd o s (1 - s)
//   A_o = d_o r_o  ->  acc' = A_o . Wo' = r dx_hat  ->  dX = acc' - mean(acc') - x_hat mean(acc' x_hat)
//   G += A_o^T X^T   (on chip; the host forms dW_o = (G - v_o 1^T) g_o + S_o b_o^T and the LN_out affine gradients from G, v_o, S_o)
// Structure: the two 4-warp groups of a CTA take two consecutive 32-token tiles in lockstep (per group as B1: warp nw owns outputs
// [32 nw, 32 nw + 32) and plane channels [CH/4 nw, CH/4 (nw + 1))); then all 8 warps accumulate G over the 64 tokens (warp tile
// 16 GMT outputs x 64 plane channels: GMT = 4 -> 128 registers at CH = 256, GMT = 2 -> 64 at CH = 128).
#pragma once
#include "sm80_common.cuh"
#include "b1_sm80.cuh"

namespace a100 {

struct B1GParams {
  const __nv_bfloat16* x;      // [CH][T]
  const __nv_bfloat16* xn;     // [T][128]
  const __nv_bfloat16* dy;     // [T][128]
  const __nv_bfloat16* ds;     // [L][128] or nullptr
  const __nv_bfloat16* wo;     // [128][CH] bf16(Wo diag gamma_out)
  const __nv_bfloat16* wg;     // [128][128] bf16(Wog)
  const float* so; const float* bo;   // [128] o fold vectors
  const float* stats;          // [T][4] mu_o, r_o, mu_i, r_i (forward)
  __nv_bfloat16* dx;           // [CH][T]
  __nv_bfloat16* dg;           // [T][ldg]
  float* red;                  // [grid][2][128]: sums of A_o mu_o, d_o
  float* gpart;                // [grid][128][CH]: G partials
  int T, L, num_tiles, ldg;
};

template <int CH_>
struct B1GCfg {
  static constexpr int CH = CH_, CZ = 128, BM = 32, NTHR = 256;
  static constexpr int KSX = CH / 16, KSZ = CZ / 16, CQ = CH / 4;
  static constexpr int GMT = CH == 256 ? 4 : 2, OB = 128 / (16 * GMT);      // G warp tile: 16 GMT outputs x 64 plane channels
  static constexpr int SMEM_WO = CZ * CH * 2, SMEM_WG = CZ * CZ * 2;
  static constexpr int SMEM_X = CH * BM * 2, SMEM_Z = BM * CZ * 2, SMEM_DY = BM * CZ * 2;   // the dy tile doubles as the LN exchange
  static constexpr int SMEM_GRP = SMEM_X + SMEM_Z + SMEM_DY;
  static constexpr int SMEM_V = 2 * CZ * 4;
  static constexpr int SMEM = SMEM_WO + SMEM_WG + 2 * SMEM_GRP + SMEM_V + 2 * 2 * 8;
  static_assert(SMEM <= 166912, "sm_80 dynamic shared memory");
  static_assert(OB * (CH / 64) == 8, "8 warps tile G");
};

template <class G>
__global__ void __launch_bounds__(256, 1) b1g_kernel(const B1GParams p) {
  constexpr int CH = G::CH, CZ = G::CZ, BM = G::BM, KSX = G::KSX, KSZ = G::KSZ, CQ = G::CQ, GMT = G::GMT;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int grp = warp >> 2, nw = warp & 3, gtid = tid & 127, g8 = lane >> 2, q = lane & 3;
  const int mi = lane >> 3, r8 = lane & 7, kx = (mi >> 1) * 8 + r8, zr = r8 + ((lane >> 3) & 1) * 8;
  uint8_t* sWo = smem;
  uint8_t* sWg = sWo + G::SMEM_WO;
  uint8_t* sGrp0 = sWg + G::SMEM_WG;
  uint8_t* sGrp = sGrp0 + grp * G::SMEM_GRP;
  float* sSt = reinterpret_cast<float*>(sGrp + G::SMEM_X + G::SMEM_Z);   // = the dy tile (consumed before the LN exchange)
  float* sV = reinterpret_cast<float*>(sGrp0 + 2 * G::SMEM_GRP);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sV + 2 * CZ) + grp * 2;
  const uint32_t sWo_u = smem_u32(sWo), sWg_u = smem_u32(sWg), sX_u = smem_u32(sGrp), sZ_u = sX_u + G::SMEM_X, sDy_u = sZ_u + G::SMEM_Z;
  const uint32_t barX = smem_u32(bars), barZ = barX + 8;
  const int bar_id = 1 + grp;
  // lockstep pairs: iteration it takes tiles 2 (blockIdx + it grid) + grp; a group past the end runs on zero-filled tiles (no stores)
  const int n_iter = 2 * (int)blockIdx.x < p.num_tiles ? (p.num_tiles - 2 * (int)blockIdx.x + 2 * (int)gridDim.x - 1) / (2 * (int)gridDim.x) : 0;
  if (gtid == 0) { mbar_init(barX, 128); mbar_init(barZ, 128); }
  for (int i = tid; i < CZ; i += 256) { sV[i] = p.so[i]; sV[CZ + i] = p.bo[i]; }
  for (int c = tid; c < CZ * CH / 8; c += 256) {
    const int row = c / (CH / 8), g = c % (CH / 8);
    cp_async16(sWo_u + g * 2048 + row * 16, p.wo + (size_t)row * CH + g * 8);   // [granule][128 rows][16 B]: immediate k-step offsets
  }
  for (int c = tid; c < CZ * CZ / 8; c += 256) {
    const int row = c >> 4, g = c & 15;
    cp_async16(sWg_u + g * 2048 + row * 16, p.wg + (size_t)row * CZ + g * 8);
  }
  cp_async_wait<0>();
  __syncthreads();
  auto load_x = [&](int t0) {
#pragma unroll
    for (int i = 0; i < CH * 4 / 128; ++i) {
      const int c = gtid + 128 * i, row = c >> 2, g = c & 3;
      const bool ok = t0 + 8 * g < p.T;
      cp_async16(sX_u + b1_swz64(row, g), p.x + (size_t)row * p.T + (ok ? t0 + 8 * g : 0), ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barX);
  };
  auto load_z = [&](int t0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = gtid + 128 * i, row = c >> 4, g = c & 15;
      const bool ok = t0 + row < p.T;
      cp_async16(sZ_u + swz<256>(row, g * 16), p.xn + (size_t)(ok ? t0 + row : 0) * CZ + g * 8, ok ? 16u : 0u);
      cp_async16(sDy_u + swz<256>(row, g * 16), p.dy + (size_t)(ok ? t0 + row : 0) * CZ + g * 8, ok ? 16u : 0u);   // (zero past T)
    }
    cp_async_mbar_arrive(barZ);
  };
  auto tile_t0 = [&](int it) { return (2 * ((int)blockIdx.x + it * (int)gridDim.x) + grp) * BM; };
  if (n_iter > 0) { load_x(tile_t0(0)); load_z(tile_t0(0)); }
  auto zero4 = [](float (&a)[4]) { a[0] = a[1] = a[2] = a[3] = 0.f; };
  // lane bases of the weight fragments: B of o / gate (row 32 nw + 8 (lane / 16 % 2) + lane % 8, granule lane / 8 % 2), B of acc' by .trans
  // (row = output 8 (mi % 2) + r8, granule = (nw CQ + 8 (mi / 2)) / 8)
  const uint32_t b_row = (32 * nw + ((lane >> 4) & 1) * 8 + r8) * 16 + ((lane >> 3) & 1) * 2048;
  const uint32_t wb_o = sWo_u + b_row, wb_g = sWg_u + b_row;
  const uint32_t wt_o = sWo_u + ((nw * CQ + (mi >> 1) * 8) >> 3) * 2048 + ((mi & 1) * 8 + r8) * 16;
  // G warp tile
  const int o0 = (warp % G::OB) * 16 * GMT, c0 = (warp / G::OB) * 64;
  float gacc[GMT][8][4];
#pragma unroll
  for (int mt = 0; mt < GMT; ++mt)
#pragma unroll
    for (int n = 0; n < 8; ++n) zero4(gacc[mt][n]);
  float red[2][4][2];
#pragma unroll
  for (int k = 0; k < 2; ++k)
#pragma unroll
    for (int n = 0; n < 4; ++n) red[k][n][0] = red[k][n][1] = 0.f;

  for (int it = 0; it < n_iter; ++it) {
    const int t0 = tile_t0(it);
    // the forward's LN_out statistics (dy arrives with the x_n tile)
    float mux[2][2], rsx[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int t = t0 + 16 * mt + g8 + 8 * h;
        const bool ok = t < p.T;
        const float2 v = ok ? __ldg(reinterpret_cast<const float2*>(p.stats + (size_t)t * 4)) : make_float2(0.f, 0.f);
        mux[mt][h] = v.x; rsx[mt][h] = v.y;
      }
    mbar_wait(barX, it & 1);
    mbar_wait(barZ, it & 1);
    // ---- per 16-output block nb: o (A streamed from the X tile), gate (A streamed from the x_n tile), then d_o / d_g -- half the
    //      accumulators live at a time (the G accumulators hold 64 / 128 registers for the whole launch)
    uint32_t awv[2][2][2][2];                    // A_o words [nb][mt][n8][h], written to the tile after every warp's x_n reads
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      float o_acc[2][2][4], g_acc[2][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int n = 0; n < 2; ++n) { zero4(o_acc[mt][n]); zero4(g_acc[mt][n]); }
#pragma unroll
      for (int ks = 0; ks < KSX; ++ks) {
        uint32_t fx[2][4], b[4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) ldsm_x4_t(fx[mt], sX_u + b1_swz64(16 * ks + kx, 2 * mt + (mi & 1)));
        ldsm_x4(b, wb_o + nb * 256 + ks * 4096);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(o_acc[mt][0], fx[mt], b[0], b[1]); mma16816(o_acc[mt][1], fx[mt], b[2], b[3]); }
      }
#pragma unroll
      for (int ks = 0; ks < KSZ; ++ks) {
        uint32_t fz[2][4], b[4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) ldsm_x4(fz[mt], sZ_u + swz<256>(16 * mt + zr, (2 * ks + (lane >> 4)) * 16));
        ldsm_x4(b, wb_g + nb * 256 + ks * 4096);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(g_acc[mt][0], fz[mt], b[0], b[1]); mma16816(g_acc[mt][1], fz[mt], b[2], b[3]); }
      }
#pragma unroll
      for (int nn = 0; nn < 2; ++nn) {
        const int n = 2 * nb + nn, oc = 32 * nw + 8 * n + 2 * q;
        const float2 s2 = *reinterpret_cast<const float2*>(sV + oc), b2 = *reinterpret_cast<const float2*>(sV + CZ + oc);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = 16 * mt + g8 + 8 * h, t = t0 + row;
            const float r = rsx[mt][h], rm = r * mux[mt][h];
            const float o0 = fmaf(r, o_acc[mt][nn][2 * h], fmaf(-rm, s2.x, b2.x)), o1 = fmaf(r, o_acc[mt][nn][2 * h + 1], fmaf(-rm, s2.y, b2.y));
            const float g0 = g_acc[mt][nn][2 * h], g1 = g_acc[mt][nn][2 * h + 1];
            const float th0 = tanh_approx(0.5f * g0), th1 = tanh_approx(0.5f * g1);
            const float sg0 = fmaf(0.5f, th0, 0.5f), sg1 = fmaf(0.5f, th1, 0.5f);
            const uint32_t dyv = lds32(sDy_u + row * 256 + ((((oc >> 3) ^ (row & 7))) << 4) + (oc & 7) * 2);
            const uint32_t dsv = (p.ds != nullptr && t < p.T) ? __ldg(reinterpret_cast<const unsigned int*>(p.ds + (size_t)(t % p.L) * CZ + oc)) : 0x3f803f80u;
            const float du0 = bf16lo(dyv) * bf16lo(dsv), du1 = bf16hi(dyv) * bf16hi(dsv);
            const float do0 = du0 * sg0, do1 = du1 * sg1;
            const float dg0 = 0.25f * du0 * o0 * fmaf(-th0, th0, 1.f), dg1 = 0.25f * du1 * o1 * fmaf(-th1, th1, 1.f);
            const uint32_t aw = pack_bf16(do0 * r, do1 * r);                 // A_o (0 on rows past T: dy = 0)
            awv[nb][mt][nn][h] = aw;
            if (t < p.T) stg32(p.dg + (size_t)t * p.ldg + oc, pack_bf16(dg0, dg1));
            red[0][n][0] = fmaf(bf16lo(aw), mux[mt][h], red[0][n][0]); red[0][n][1] = fmaf(bf16hi(aw), mux[mt][h], red[0][n][1]);
            red[1][n][0] += do0; red[1][n][1] += do1;
          }
      }
    }
    bar_sync(bar_id, 128);                       // every warp is past its x_n reads: the tile takes A_o
    const uint32_t sDo = sZ_u;                   // [32 tok][128] bf16 A_o, 256 B rows, granule ^= row & 7
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
#pragma unroll
      for (int nn = 0; nn < 2; ++nn) {
        const int oc = 32 * nw + 8 * (2 * nb + nn) + 2 * q;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = 16 * mt + g8 + 8 * h;
            sts32(sDo + row * 256 + ((((oc >> 3) ^ (row & 7))) << 4) + (oc & 7) * 2, awv[nb][mt][nn][h]);
          }
      }
    // ---- G += A_o^T X^T over both groups' tiles (64 tokens), all 8 warps
    __syncthreads();                             // both groups' A_o tiles are complete (and their X tiles landed)
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const uint32_t xs = smem_u32(sGrp0 + sub * G::SMEM_GRP), as = xs + G::SMEM_X;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t a[GMT][4];
#pragma unroll
        for (int mt = 0; mt < GMT; ++mt) {       // A (m = output, k = token) from A_o [tok][o] by .trans
          const int krow = 16 * ks + 8 * (mi >> 1) + r8, og = (o0 + 16 * mt) / 8 + (mi & 1);
          ldsm_x4_t(a[mt], as + krow * 256 + ((og ^ (krow & 7)) << 4));
        }
#pragma unroll
        for (int np = 0; np < 4; ++np) {         // B (k = token, n = plane channel) from X [ch][tok]
          uint32_t b[4];
          ldsm_x4(b, xs + b1_swz64(c0 + 16 * np + 8 * (mi >> 1) + r8, 2 * ks + (mi & 1)));
#pragma unroll
          for (int mt = 0; mt < GMT; ++mt) { mma16816(gacc[mt][2 * np], a[mt], b[0], b[1]); mma16816(gacc[mt][2 * np + 1], a[mt], b[2], b[3]); }
        }
      }
    }
    __syncthreads();                             // every G read of both groups' tiles done: the X tiles may take the dX staging
    // ---- LN_out backward per 16-row block mt (half the accumulators live): acc' = A_o . Wo' for this warp's CQ plane channels, row sums
    //      (x_hat reloaded from the X tile, not held), exchange, dX = acc' - mean(acc') - x_hat mean(acc' x_hat) -> the X tile as staging, in
    //      place (every element is written by the thread that just read it; warps own disjoint channels)
    auto xhat = [&](int mt, int kq, float (&xv)[2][4]) {        // n8 tiles 2 kq, 2 kq + 1 of this warp's channels, C layout
      uint32_t f[4];
      ldsm_x4_t(f, sX_u + b1_swz64(nw * CQ + 16 * kq + kx, 2 * mt + (mi & 1)));
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint32_t va = f[2 * hh], vb = f[2 * hh + 1];
        xv[hh][0] = (bf16lo(va) - mux[mt][0]) * rsx[mt][0];
        xv[hh][1] = (bf16hi(va) - mux[mt][0]) * rsx[mt][0];
        xv[hh][2] = (bf16lo(vb) - mux[mt][1]) * rsx[mt][1];
        xv[hh][3] = (bf16hi(vb) - mux[mt][1]) * rsx[mt][1];
      }
    };
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      float dxh[CQ / 8][4];
#pragma unroll
      for (int n = 0; n < CQ / 8; ++n) zero4(dxh[n]);
#pragma unroll
      for (int ks = 0; ks < KSZ; ++ks) {
        uint32_t a[4];
        {
          const int row = 16 * mt + zr, g = 2 * ks + (lane >> 4);
          ldsm_x4(a, sDo + row * 256 + ((g ^ (row & 7)) << 4));
        }
#pragma unroll
        for (int np = 0; np < CQ / 16; ++np) {
          uint32_t b[4];
          ldsm_x4_t(b, wt_o + ks * 256 + np * 4096);
          mma16816(dxh[2 * np], a, b[0], b[1]); mma16816(dxh[2 * np + 1], a, b[2], b[3]);
        }
      }
      float a2[2] = {0.f, 0.f}, bs[2] = {0.f, 0.f};
#pragma unroll
      for (int kq = 0; kq < CQ / 16; ++kq) {
        float xv[2][4];
        xhat(mt, kq, xv);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float d0 = dxh[2 * kq + hh][2 * h], d1 = dxh[2 * kq + hh][2 * h + 1];
            a2[h] += d0 + d1;
            bs[h] = fmaf(d0, xv[hh][2 * h], fmaf(d1, xv[hh][2 * h + 1], bs[h]));
          }
      }
      float* ex = sSt + mt * 256;                // [nw][16 rows][2]; the two blocks use disjoint halves
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float sa = quad_sum(a2[h]), sb = quad_sum(bs[h]);
        if (q == 0) *reinterpret_cast<float2*>(ex + nw * 32 + 2 * (g8 + 8 * h)) = make_float2(sa, sb);
      }
      bar_sync(bar_id, 128);
      float m1[2], m2[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) { const float2 v = *reinterpret_cast<const float2*>(ex + w * 32 + 2 * (g8 + 8 * h)); sa += v.x; sb += v.y; }
        m1[h] = sa * (1.f / CH); m2[h] = sb * (1.f / CH);
      }
#pragma unroll
      for (int kq = 0; kq < CQ / 16; ++kq) {
        float xv[2][4];
        xhat(mt, kq, xv);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int n = 2 * kq + hh, row = 16 * mt + g8 + 8 * h, ch = nw * CQ + 8 * n + 2 * q;
            const float d0 = dxh[n][2 * h] - m1[h] - xv[hh][2 * h] * m2[h];
            const float d1 = dxh[n][2 * h + 1] - m1[h] - xv[hh][2 * h + 1] * m2[h];
            const uint32_t a0 = sX_u + b1_swz64(ch, row >> 3) + (row & 7) * 2, a1 = sX_u + b1_swz64(ch + 1, row >> 3) + (row & 7) * 2;
            asm volatile("st.shared.b16 [%0], %1;\n" ::"r"(a0), "h"(__bfloat16_as_ushort(__float2bfloat16_rn(d0))) : "memory");
            asm volatile("st.shared.b16 [%0], %1;\n" ::"r"(a1), "h"(__bfloat16_as_ushort(__float2bfloat16_rn(d1))) : "memory");
          }
      }
    }
    bar_sync(bar_id, 128);
#pragma unroll
    for (int i = 0; i < CH * 4 / 128; ++i) {
      const int c = gtid + 128 * i, ch = c >> 2, g = c & 3;
      if (t0 + 8 * g < p.T) stg128(p.dx + (size_t)ch * p.T + t0 + 8 * g, lds128(sX_u + b1_swz64(ch, g)));
    }
    bar_sync(bar_id, 128);                       // X and A_o tiles free
    if (it + 1 < n_iter) { load_x(tile_t0(it + 1)); load_z(tile_t0(it + 1)); }
  }
  cp_async_wait<0>();
  // G partial of this CTA
  float* gp = p.gpart + (size_t)blockIdx.x * 128 * CH;
#pragma unroll
  for (int mt = 0; mt < GMT; ++mt)
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int o = o0 + 16 * mt + g8 + 8 * h, c = c0 + 8 * n + 2 * q;
        *reinterpret_cast<float2*>(gp + (size_t)o * CH + c) = make_float2(gacc[mt][n][2 * h], gacc[mt][n][2 * h + 1]);
      }
  // fold vectors: reduce over the warp's 8 row lanes; the two groups add into one partial
  __syncthreads();
  float* sred = reinterpret_cast<float*>(sGrp0);             // [2][128]
  for (int i = tid; i < 256; i += 256) sred[i] = 0.f;
  __syncthreads();
#pragma unroll
  for (int k = 0; k < 2; ++k)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float v = red[k][n][j];
        v += __shfl_xor_sync(0xffffffffu, v, 4); v += __shfl_xor_sync(0xffffffffu, v, 8); v += __shfl_xor_sync(0xffffffffu, v, 16);
        if (g8 == 0) atomicAdd(&sred[k * CZ + 32 * nw + 8 * n + 2 * q + j], v);
      }
  __syncthreads();
  for (int i = tid; i < 256; i += 256) p.red[(size_t)blockIdx.x * 256 + i] = sred[i];
}

}  // namespace a100
