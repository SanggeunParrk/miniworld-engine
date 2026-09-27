// k3_sm80.cuh -- TriMul K3 (output LayerNorm + projection, input LayerNorm + gate, residual), A100 / sm_80.
//
//   out[t, :] = z[t, :] + bf16( sigmoid(LN_in(z)[t] . Wog^T) * bf16(LN_out(X[:, t]) . Wo^T) )      (X = contraction planes [CH][T])
//
// The LayerNorm affine is folded into the weights: with W' = bf16(W diag(gamma)), s = sum_k W'[:, k] and b = W beta,
//   LN(x) . W^T = r (x . W'^T) - r mu s + b      (r = 1 / sqrt(var + eps), mu = mean of x)
// so the MMAs read the raw bf16 X / z fragments and the normalisation is two FMAs per OUTPUT.  Because s sums the same rounded W', the mean term
// cancels exactly and the only extra rounding is W' itself (one bf16 rounding of each weight, where the reference rounds each normalised input).
// The statistics are split over K between the 4 warps that share a token group and exchanged through shared memory (no per-warp duplicate pass).
//
// Carried over from the sm_90 K3: persistent CTAs with the whole W_o | W_og resident in shared memory (loaded once per CTA), the X tile read
// channel-major straight from the contraction planes (A operand via ldmatrix.trans) and released as soon as it sits in registers, the projection
// finished before the z tile is touched (one raw fragment set live at a time), residual added in bf16x2.
// CTA = 8 warps: mg = warp & 1 -> tokens [32 mg, 32 mg + 32) of a 64-token tile; nw = warp >> 1 -> output channels [32 nw, 32 nw + 32).
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct K3Params {
  const __nv_bfloat16* x;      // [CH][T] contraction planes
  const __nv_bfloat16* z;      // [T][128]
  const __nv_bfloat16* wo;     // [128][CH]  bf16(W_o diag gamma_out)
  const __nv_bfloat16* wg;     // [128][128] bf16(W_og diag gamma_in)
  const float* so; const float* bo;   // [128]
  const float* sg; const float* bg;   // [128]
  __nv_bfloat16* out;          // [T][128]
  int T, num_tiles;
  float eps;
};

template <int CH_>
struct K3Cfg {
  static constexpr int CH = CH_, CZ = 128, BM = 64, NTHR = 256;
  static constexpr int KSX = CH / 16, KSZ = CZ / 16;
  static constexpr int SMEM_WO = CZ * CH * 2, SMEM_WG = CZ * CZ * 2, SMEM_X = CH * BM * 2, SMEM_Z = BM * CZ * 2;
  static constexpr int SMEM_V = 4 * CZ * 4;                // so, bo, sg, bg
  static constexpr int SMEM_ST = 2 * 2 * 4 * 32 * 2 * 4;   // [x|z][mg][nw][32 rows][sum, sumsq] fp32
  static constexpr int SMEM = SMEM_WO + SMEM_WG + SMEM_X + SMEM_Z + SMEM_V + SMEM_ST + 2 * 8;
  static_assert(SMEM <= 166912, "sm_80 dynamic shared memory");
  static_assert(KSX % 4 == 0, "K split over 4 warps");
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) k3_kernel(const K3Params p) {
  constexpr int CH = G::CH, CZ = G::CZ, BM = G::BM, KSX = G::KSX, KSZ = G::KSZ;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sWo = smem;
  uint8_t* sWg = sWo + G::SMEM_WO;
  uint8_t* sX = sWg + G::SMEM_WG;
  uint8_t* sZ = sX + G::SMEM_X;
  float* sV = reinterpret_cast<float*>(sZ + G::SMEM_Z);
  float* sSt = sV + 4 * CZ;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sSt + G::SMEM_ST / 4);
  const uint32_t sWo_u = smem_u32(sWo), sWg_u = smem_u32(sWg), sX_u = smem_u32(sX), sZ_u = smem_u32(sZ);
  const uint32_t barX = smem_u32(bars), barZ = barX + 8;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mg = warp & 1, nw = warp >> 1, g8 = lane >> 2, q = lane & 3;
  const int n_iter = (p.num_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  if (tid == 0) { mbar_init(barX, G::NTHR); mbar_init(barZ, G::NTHR); }
  for (int i = tid; i < CZ; i += G::NTHR) { sV[i] = p.so[i]; sV[CZ + i] = p.bo[i]; sV[2 * CZ + i] = p.sg[i]; sV[3 * CZ + i] = p.bg[i]; }
  // resident weights (their completion rides on the first X / z arrivals of every thread)
  for (int c = tid; c < CZ * CH / 8; c += G::NTHR) {
    const int row = c / (CH / 8), g = c % (CH / 8);
    cp_async16(sWo_u + swz<CH * 2>(row, g * 16), p.wo + (size_t)row * CH + g * 8);
  }
  for (int c = tid; c < CZ * CZ / 8; c += G::NTHR) {
    const int row = c >> 4, g = c & 15;
    cp_async16(sWg_u + swz<256>(row, g * 16), p.wg + (size_t)row * CZ + g * 8);
  }
  __syncthreads();
  auto load_x = [&](int tile) {                 // [CH][64 tok]: CH rows x 8 granules
    const int t0 = tile * BM;
#pragma unroll
    for (int i = 0; i < CH * 8 / G::NTHR; ++i) {
      const int c = tid + G::NTHR * i, row = c >> 3, g = c & 7;
      const bool ok = t0 + 8 * g < p.T;
      cp_async16(sX_u + swz<128>(row, g * 16), p.x + (size_t)row * p.T + (ok ? t0 + 8 * g : 0), ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barX);
  };
  auto load_z = [&](int tile) {                 // [64 tok][128 ch]: 64 rows x 16 granules
    const int t0 = tile * BM;
#pragma unroll
    for (int i = 0; i < BM * 16 / G::NTHR; ++i) {
      const int c = tid + G::NTHR * i, row = c >> 4, g = c & 15;
      const bool ok = t0 + row < p.T;
      cp_async16(sZ_u + swz<256>(row, g * 16), p.z + (size_t)(ok ? t0 + row : 0) * CZ + g * 8, ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barZ);
  };
  if (n_iter > 0) { load_x(blockIdx.x); load_z(blockIdx.x); }
  const int bar_id = 1 + mg;                    // the 4 warps of one token group

  // stats of this thread's 4 rows (rr = 16 mt + g8 + 8 h) from the partial sums of the group's 4 warps
  auto stats = [&](float (&ps)[2][2][2], int which, float n_inv, float (&mu)[2][2], float (&rs)[2][2]) {
    float* base = sSt + ((which * 2 + mg) * 4) * 64;   // [nw][32 rows][2]
    if (q == 0) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int rr = 16 * mt + g8 + 8 * h;
          base[nw * 64 + 2 * rr] = ps[mt][h][0];
          base[nw * 64 + 2 * rr + 1] = ps[mt][h][1];
        }
    }
    bar_sync(bar_id, 128);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int rr = 16 * mt + g8 + 8 * h;
        float s = 0.f, s2 = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) { s += base[w * 64 + 2 * rr]; s2 += base[w * 64 + 2 * rr + 1]; }
        const float m = s * n_inv;
        mu[mt][h] = m;
        rs[mt][h] = rsqrtf(fmaxf(s2 * n_inv - m * m, 0.f) + p.eps);
      }
  };

  for (int it = 0; it < n_iter; ++it) {
    const int tile = (int)blockIdx.x + it * (int)gridDim.x;
    const int t0 = tile * BM;
    // ---- X fragments (A = tokens x channels from the channel-major tile: ldmatrix.trans)
    mbar_wait(barX, it & 1);
    uint32_t fx[2][KSX][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int ks = 0; ks < KSX; ++ks) {
        const int mi = lane >> 3;
        const int k = 16 * ks + (mi >> 1) * 8 + (lane & 7);
        const int tok = 32 * mg + 16 * mt + (mi & 1) * 8;
        ldsm_x4_t(fx[mt][ks], sX_u + swz<128>(k, tok * 2));
      }
    __syncthreads();                            // X tile free: next tile's X streams in under this tile's work
    if (it + 1 < n_iter) load_x(tile + (int)gridDim.x);
    float ps[2][2][2], mu[2][2], rs[2][2];
    {
      constexpr int NK = KSX / 4;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) { ps[mt][h][0] = 0.f; ps[mt][h][1] = 0.f; }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int k = 0; k < KSX; ++k)
#pragma unroll
          for (int r = 0; r < 4; ++r) {           // regs 0, 2: row g8; 1, 3: row g8 + 8
            if (k / NK != nw) continue;             // warp-uniform: compile-time register indices, this warp's quarter of K
            const uint32_t v = fx[mt][k][r];
            const float a = bf16lo(v), b = bf16hi(v);
            ps[mt][r & 1][0] += a + b;
            ps[mt][r & 1][1] = fmaf(a, a, fmaf(b, b, ps[mt][r & 1][1]));
          }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) { ps[mt][h][0] = quad_sum(ps[mt][h][0]); ps[mt][h][1] = quad_sum(ps[mt][h][1]); }
    }
    stats(ps, 0, 1.f / CH, mu, rs);
    // ---- projection: 2 blocks of 16 output channels; o = bf16(r acc - r mu s + b)
    uint32_t pst[2][2][2][2];                   // [nb][mt][n8][row half] packed bf16 channel pairs
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      float acc[2][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[mt][n][e] = 0.f;
      const int brow = 32 * nw + 16 * nb + ((lane >> 4) & 1) * 8 + (lane & 7);
#pragma unroll
      for (int ks = 0; ks < KSX; ++ks) {
        uint32_t b[4];
        ldsm_x4(b, sWo_u + swz<CH * 2>(brow, (2 * ks + ((lane >> 3) & 1)) * 16));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(acc[mt][0], fx[mt][ks], b[0], b[1]); mma16816(acc[mt][1], fx[mt][ks], b[2], b[3]); }
      }
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int oc = 32 * nw + 16 * nb + 8 * n + 2 * q;
        const float s0 = sV[oc], s1 = sV[oc + 1], b0 = sV[CZ + oc], b1 = sV[CZ + oc + 1];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float r = rs[mt][h], rm = r * mu[mt][h];
            pst[nb][mt][n][h] = pack_bf16(fmaf(r, acc[mt][n][2 * h], fmaf(-rm, s0, b0)), fmaf(r, acc[mt][n][2 * h + 1], fmaf(-rm, s1, b1)));
          }
      }
    }
    // ---- z fragments (A = tokens x channels, row-major tile)
    mbar_wait(barZ, it & 1);
    uint32_t fz[2][KSZ][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int ks = 0; ks < KSZ; ++ks) {
        const int row = 32 * mg + 16 * mt + (lane & 7) + ((lane >> 3) & 1) * 8;
        ldsm_x4(fz[mt][ks], sZ_u + swz<256>(row, (2 * ks + (lane >> 4)) * 16));
      }
    {
      constexpr int NK = KSZ / 4;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) { ps[mt][h][0] = 0.f; ps[mt][h][1] = 0.f; }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int k = 0; k < KSZ; ++k)
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            if (k / NK != nw) continue;
            const uint32_t v = fz[mt][k][r];
            const float a = bf16lo(v), b = bf16hi(v);
            ps[mt][r & 1][0] += a + b;
            ps[mt][r & 1][1] = fmaf(a, a, fmaf(b, b, ps[mt][r & 1][1]));
          }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) { ps[mt][h][0] = quad_sum(ps[mt][h][0]); ps[mt][h][1] = quad_sum(ps[mt][h][1]); }
    }
    stats(ps, 1, 1.f / CZ, mu, rs);
    // ---- gate + output: out = z + bf16(sigmoid(r acc - r mu s + b) * proj)
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      float acc[2][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[mt][n][e] = 0.f;
      const int brow = 32 * nw + 16 * nb + ((lane >> 4) & 1) * 8 + (lane & 7);
#pragma unroll
      for (int ks = 0; ks < KSZ; ++ks) {
        uint32_t b[4];
        ldsm_x4(b, sWg_u + swz<256>(brow, (2 * ks + ((lane >> 3) & 1)) * 16));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(acc[mt][0], fz[mt][ks], b[0], b[1]); mma16816(acc[mt][1], fz[mt][ks], b[2], b[3]); }
      }
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int oc = 32 * nw + 16 * nb + 8 * n + 2 * q;
        const float s0 = sV[2 * CZ + oc], s1 = sV[2 * CZ + oc + 1], b0 = sV[3 * CZ + oc], b1 = sV[3 * CZ + oc + 1];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = 32 * mg + 16 * mt + g8 + 8 * h;
            const float r = rs[mt][h], rm = r * mu[mt][h];
            const uint32_t pv = pst[nb][mt][n][h];
            const float o0 = sigmoid(fmaf(r, acc[mt][n][2 * h], fmaf(-rm, s0, b0))) * bf16lo(pv);
            const float o1 = sigmoid(fmaf(r, acc[mt][n][2 * h + 1], fmaf(-rm, s1, b1))) * bf16hi(pv);
            const uint32_t zr = lds32(sZ_u + swz<256>(row, oc * 2));
            if (t0 + row < p.T) stg32(p.out + (size_t)(t0 + row) * CZ + oc, add_bf16x2(zr, pack_bf16(o0, o1)));
          }
      }
    }
    __syncthreads();                            // z tile free
    if (it + 1 < n_iter) load_z(tile + (int)gridDim.x);
  }
  cp_async_wait<0>();
}

}  // namespace a100
