// k3_sm80.cuh -- TriMul K3 (output LayerNorm + projection, input LayerNorm + gate, residual), A100 / sm_80.
//
//   out[t, :] = z[t, :] + bf16( sigmoid(LN_in(z)[t] . Wog^T) * bf16(LN_out(X[:, t]) . Wo^T) )      (X = contraction planes [CH][T])
//
// The LayerNorm affine is folded into the weights: with W' = bf16(W diag(gamma)), s = sum_k W'[:, k] and b = W beta,
//   LN(x) . W^T = r (x . W'^T) - r mu s + b      (r = 1 / sqrt(var + eps), mu = mean of x)
// so the MMAs read the raw bf16 X / z fragments and the normalisation is two FMAs per OUTPUT.  Because s sums the same rounded W', the mean term
// cancels exactly and the only extra rounding is W' itself (one bf16 rounding of each weight, where the reference rounds each normalised input).
// Both weight sets carry a factor 0.5 (exact): sigmoid(g) p = p' (1 + tanh g') with g' = g / 2, p' = p / 2 -- one MUFU and one FFMA per output.
//
// Carried over from the sm_90 K3: persistent CTAs with the whole W_o | W_og resident in shared memory (loaded once), the X tile read
// channel-major straight from the contraction planes (A operand via ldmatrix.trans) and released as soon as it sits in registers, the
// projection finished before the z tile is touched (one raw fragment set live at a time), residual added in bf16x2.
// sm_80 structure: the CTA's 8 warps are two independent groups (warps 0-3, 4-7) that share the resident weights but walk their own 32-token
// tile sequences with their own X / z buffers and barriers, so one group's statistics / epilogue run under the other group's MMAs (each SM
// sub-partition holds one warp of each group).  In a group, warp nw owns output channels [32 nw, 32 nw + 32) of the group's 32 tokens, and the
// row statistics are split over K between the group's 4 warps and exchanged through shared memory.
#pragma once
#include <type_traits>
#include "sm80_common.cuh"

namespace a100 {

struct K3Params {
  const __nv_bfloat16* x;      // [CH][T] contraction planes
  const __nv_bfloat16* z;      // [T][128]
  const __nv_bfloat16* wo;     // [128][CH]  bf16(0.5 W_o diag gamma_out)
  const __nv_bfloat16* wg;     // [128][128] bf16(0.5 W_og diag gamma_in)
  const float* so; const float* bo;   // [128]  (0.5-scaled)
  const float* sg; const float* bg;   // [128]  (0.5-scaled)
  __nv_bfloat16* out;          // [T][128]
  int T, num_tiles;            // tiles of 32 tokens
  float eps;
};

template <int CH_, int NG_ = 2>
struct K3Cfg {
  static constexpr int CH = CH_, CZ = 128, BM = 32, NG = NG_, NTHR = 128 * NG;
  static constexpr int KSX = CH / 16, KSZ = CZ / 16;
  static constexpr int SMEM_WO = CZ * CH * 2, SMEM_WG = CZ * CZ * 2;
  static constexpr int SMEM_X = CH * BM * 2, SMEM_Z = BM * CZ * 2;          // per group
  static constexpr int SMEM_GRP = SMEM_X + SMEM_Z;                            // X + z (both refilled under the group's next MMAs)
  static constexpr int SMEM_V = 4 * CZ * 4;                                   // so, bo, sg, bg
  static constexpr int SMEM_ST = NG * 2 * 4 * 32 * 2 * 4;                     // [group][x|z][nw][32 rows][sum, sumsq]
  static constexpr int SMEM = SMEM_WO + SMEM_WG + NG * SMEM_GRP + SMEM_V + SMEM_ST + NG * 2 * 8;
  static_assert(SMEM <= 166912, "sm_80 dynamic shared memory");
  static_assert(KSX % 4 == 0, "K split over 4 warps");
};

// 64-byte rows (32 tokens): granule g (0..3) of row r at g ^ ((r >> 1) & 3) -> the 8 rows of an ldmatrix land in 8 distinct 16 B bank slots
DEVI uint32_t swz64(uint32_t row, uint32_t g) { return row * 64 + ((g ^ ((row >> 1) & 3u)) << 4); }

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) k3_kernel(const K3Params p) {
  static_assert(G::NG <= 7, "named barriers");
  constexpr int CH = G::CH, CZ = G::CZ, BM = G::BM, KSX = G::KSX, KSZ = G::KSZ;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int grp = warp >> 2, nw = warp & 3, gtid = tid & 127, g8 = lane >> 2, q = lane & 3;
  uint8_t* sWo = smem;
  uint8_t* sWg = sWo + G::SMEM_WO;
  uint8_t* sGrp = sWg + G::SMEM_WG + grp * G::SMEM_GRP;
  float* sV = reinterpret_cast<float*>(sWg + G::SMEM_WG + G::NG * G::SMEM_GRP);
  float* sSt = sV + 4 * CZ + grp * (2 * 4 * 32 * 2);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sV + 4 * CZ + G::SMEM_ST / 4) + grp * 2;
  const uint32_t sWo_u = smem_u32(sWo), sWg_u = smem_u32(sWg), sX_u = smem_u32(sGrp), sZ_u = sX_u + G::SMEM_X;
  const uint32_t barX = smem_u32(bars), barZ = barX + 8;

  // group tile sequence: tiles grp + NG (blockIdx.x + gridDim.x i)
  const int gstride = G::NG * (int)gridDim.x, gfirst = G::NG * (int)blockIdx.x + grp;
  const int n_iter = gfirst < p.num_tiles ? (p.num_tiles - gfirst + gstride - 1) / gstride : 0;
  if (gtid == 0) { mbar_init(barX, 128); mbar_init(barZ, 128); }
  for (int i = tid; i < CZ; i += G::NTHR) { sV[i] = p.so[i]; sV[CZ + i] = p.bo[i]; sV[2 * CZ + i] = p.sg[i]; sV[3 * CZ + i] = p.bg[i]; }
  // resident weights, granule-column-major [16 B k-granule][128 rows][16 B]: an ldmatrix's 8 rows are consecutive 16 B (conflict-free) and
  // every k-step is an immediate offset.  (Completion rides on every thread's first X / z arrivals.)
  for (int c = tid; c < CZ * CH / 8; c += G::NTHR) {
    const int row = c & 127, g = c >> 7;
    cp_async16(sWo_u + g * 2048 + row * 16, p.wo + (size_t)row * CH + g * 8);
  }
  for (int c = tid; c < CZ * CZ / 8; c += G::NTHR) {
    const int row = c & 127, g = c >> 7;
    cp_async16(sWg_u + g * 2048 + row * 16, p.wg + (size_t)row * CZ + g * 8);
  }
  __syncthreads();
  const int bar_id = 1 + grp;                    // the group's named barrier (128 threads)
  auto load_x = [&](int tile) {                  // [CH][32 tok]: CH rows x 4 granules, CH / 32 per thread
    const int t0 = tile * BM;
#pragma unroll
    for (int i = 0; i < CH * 4 / 128; ++i) {
      const int c = gtid + 128 * i, row = c >> 2, g = c & 3;
      const bool ok = t0 + 8 * g < p.T;
      cp_async16(sX_u + swz64(row, g), p.x + (size_t)row * p.T + (ok ? t0 + 8 * g : 0), ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barX);
  };
  auto load_z = [&](int tile) {                  // [32 tok][128 ch]: 32 rows x 16 granules, 4 per thread
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

  // row sums and sums of squares over the k-steps [KS0, KS0 + NK) of fragment set f, on the tensor cores: sum = x . 1 (B = ones), and the
  // squares are the diagonal of the 16 x 16 Gram block x x^T -- whose B operands ARE the A fragment's registers (a0, a2: tokens 0-7; a1, a3:
  // tokens 8-15), so no extra loads, exact bf16 products, fp32 accumulation.  Lane q == g8 / 2 holds the diagonal of rows g8 and g8 + 8.
  auto partial = [&](const auto& f, auto ks0c, auto nkc, float (&ps)[2][2][2]) {
    constexpr int KS0 = decltype(ks0c)::value, NK = decltype(nkc)::value;
    constexpr uint32_t ONES = 0x3f803f80u;
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
      ps[mt][0][1] = odd ? g0[1] : g0[0];          // (row g8, col g8)     -- valid on lane q == g8 / 2
      ps[mt][1][1] = odd ? g1[3] : g1[2];          // (row g8 + 8, col g8)
    }
  };
  // exchange the group's 4 partials; this thread's (mu, r) of its 4 rows
  auto stats = [&](float (&ps)[2][2][2], int which, float n_inv, float (&mu)[2][2], float (&rs)[2][2]) {
    float* base = sSt + which * (4 * 64);        // [nw][32 rows][2]
    if (q == (g8 >> 1)) {
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
        float sm = 0.f, s2 = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) { const float2 v = *reinterpret_cast<const float2*>(base + w * 64 + 2 * rr); sm += v.x; s2 += v.y; }
        const float m = sm * n_inv;
        mu[mt][h] = m;
        rs[mt][h] = rsqrtf(fmaxf(s2 * n_inv - m * m, 0.f) + p.eps);
      }
  };
  auto quarter = [&](const auto& f, auto nks, float (&ps)[2][2][2]) {   // this warp's quarter of the k-steps, compile-time register indices
    constexpr int NQ = decltype(nks)::value / 4;
    switch (nw) {
      case 0: partial(f, std::integral_constant<int, 0>{}, std::integral_constant<int, NQ>{}, ps); break;
      case 1: partial(f, std::integral_constant<int, NQ>{}, std::integral_constant<int, NQ>{}, ps); break;
      case 2: partial(f, std::integral_constant<int, 2 * NQ>{}, std::integral_constant<int, NQ>{}, ps); break;
      default: partial(f, std::integral_constant<int, 3 * NQ>{}, std::integral_constant<int, NQ>{}, ps); break;
    }
  };

  // per-lane fragment addresses, constant over the tiles (single X / z buffers): weights (B, row 32 nw + 16 nb + 8 (lane / 16) + lane % 8,
  // granule 2 ks + lane / 8 % 2), X (A via .trans: k row 16 ks + 8 (mi / 2) + lane % 8, token granule 2 mt + mi % 2), z (A: row 16 mt + lane % 16)
  const uint32_t b_row = (32 * nw + ((lane >> 4) & 1) * 8 + (lane & 7)) * 16 + ((lane >> 3) & 1) * 2048;
  const uint32_t wo_b = sWo_u + b_row, wg_b = sWg_u + b_row;
  const int mi = lane >> 3, kx = (mi >> 1) * 8 + (lane & 7);
  const uint32_t x_a0 = sX_u + swz64(kx, mi & 1), x_a1 = sX_u + swz64(kx, 2 + (mi & 1));   // + 1024 ks
  const int zr = (lane & 7) + ((lane >> 3) & 1) * 8;
  uint32_t z_a[KSZ];
#pragma unroll
  for (int ks = 0; ks < KSZ; ++ks) z_a[ks] = sZ_u + swz<256>(zr, (2 * ks + (lane >> 4)) * 16);      // + 4096 mt
  for (int it = 0; it < n_iter; ++it) {
    const int tile = gfirst + it * gstride;
    const int t0 = tile * BM;
    // ---- X fragments (A = tokens x channels from the channel-major tile: ldmatrix.trans)
    mbar_wait(barX, it & 1);
    uint32_t fx[2][KSX][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int ks = 0; ks < KSX; ++ks) ldsm_x4_t(fx[mt][ks], (mt ? x_a1 : x_a0) + ks * 1024);
    float ps[2][2][2], mu[2][2], rs[2][2];
    quarter(fx, std::integral_constant<int, KSX>{}, ps);
    stats(ps, 0, 1.f / CH, mu, rs);              // (its barrier also orders every warp's X reads before the refill below)
    if (it + 1 < n_iter) load_x(tile + gstride);
    // ---- projection: 2 blocks of 16 output channels; p' = bf16(r acc - r mu s + b)
    uint32_t pst[2][2][2][2];                    // [nb][mt][n8][row half] packed bf16 channel pairs
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      float acc[2][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[mt][n][e] = 0.f;
#pragma unroll
      for (int ks = 0; ks < KSX; ++ks) {
        uint32_t b[4];
        ldsm_x4(b, wo_b + nb * 256 + ks * 4096);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(acc[mt][0], fx[mt][ks], b[0], b[1]); mma16816(acc[mt][1], fx[mt][ks], b[2], b[3]); }
      }
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int oc = 32 * nw + 16 * nb + 8 * n + 2 * q;
        const float2 s2 = *reinterpret_cast<const float2*>(sV + oc), b2 = *reinterpret_cast<const float2*>(sV + CZ + oc);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float r = rs[mt][h], rm = r * mu[mt][h];
            pst[nb][mt][n][h] = pack_bf16(fmaf(r, acc[mt][n][2 * h], fmaf(-rm, s2.x, b2.x)), fmaf(r, acc[mt][n][2 * h + 1], fmaf(-rm, s2.y, b2.y)));
          }
      }
    }
    // ---- z fragments (A = tokens x channels, row-major tile)
    mbar_wait(barZ, it & 1);
    const uint32_t zt = sZ_u;
    uint32_t fz[2][KSZ][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int ks = 0; ks < KSZ; ++ks) ldsm_x4(fz[mt][ks], z_a[ks] + mt * 4096);
    quarter(fz, std::integral_constant<int, KSZ>{}, ps);
    stats(ps, 1, 1.f / CZ, mu, rs);
    // ---- gate + output: out = z + bf16(p' (1 + tanh(r acc - r mu s + b)))
    const bool full = t0 + BM <= p.T;
    __nv_bfloat16* orow = p.out + (size_t)t0 * CZ;
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      float acc[2][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[mt][n][e] = 0.f;
#pragma unroll
      for (int ks = 0; ks < KSZ; ++ks) {
        uint32_t b[4];
        ldsm_x4(b, wg_b + nb * 256 + ks * 4096);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(acc[mt][0], fz[mt][ks], b[0], b[1]); mma16816(acc[mt][1], fz[mt][ks], b[2], b[3]); }
      }
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int oc = 32 * nw + 16 * nb + 8 * n + 2 * q;
        const float2 s2 = *reinterpret_cast<const float2*>(sV + 2 * CZ + oc), b2 = *reinterpret_cast<const float2*>(sV + 3 * CZ + oc);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = 16 * mt + g8 + 8 * h;
            const float r = rs[mt][h], rm = r * mu[mt][h];
            const uint32_t pv = pst[nb][mt][n][h];
            const float p0 = bf16lo(pv), p1 = bf16hi(pv);
            const float o0 = fmaf(p0, tanh_approx(fmaf(r, acc[mt][n][2 * h], fmaf(-rm, s2.x, b2.x))), p0);
            const float o1 = fmaf(p1, tanh_approx(fmaf(r, acc[mt][n][2 * h + 1], fmaf(-rm, s2.y, b2.y))), p1);
            const uint32_t zv = lds32(zt + swz<256>(row, oc * 2));
            if (full || t0 + row < p.T) stg32(orow + row * CZ + oc, add_bf16x2(zv, pack_bf16(o0, o1)));
          }
      }
    }
    bar_sync(bar_id, 128);                       // the group's z tile is free: it takes the next tile (landing under the next projection)
    if (it + 1 < n_iter) load_z(tile + gstride);
  }
  cp_async_wait<0>();
}

}  // namespace a100
