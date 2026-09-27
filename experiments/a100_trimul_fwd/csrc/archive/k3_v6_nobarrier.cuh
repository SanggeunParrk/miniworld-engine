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
// The row statistics come from the tensor cores and ride inside the MMA loops that need them only afterwards (the fold normalises the OUTPUTS):
// sum = x . 1 (B = ones) and sum of squares = the diagonal of the 16 x 16 Gram block x x^T, whose B operands are the A fragment's own registers
// (a0, a2: tokens 0-7; a1, a3: tokens 8-15) -- exact bf16 products, fp32 accumulation, no extra loads.  Every warp computes its own rows over the
// full K, so the group's warps never exchange anything and never wait for each other.
//
// Carried over from the sm_90 K3: persistent CTAs with the whole W_o | W_og resident in shared memory (loaded once), the X tile read
// channel-major straight from the contraction planes (A operand via ldmatrix.trans) and released as soon as it sits in registers, the
// projection finished before the z tile is touched, residual added in bf16x2.
// sm_80 structure: the CTA's warps form NG independent groups of 4 (warps 4 g .. 4 g + 3) that share the resident weights but walk their own
// 32-token tile sequences with their own X / z buffers; each SM sub-partition holds one warp of every group.  In a group, warp nw owns output
// channels [32 nw, 32 nw + 32) of the group's 32 tokens.  A tile buffer is released by the 4 warps' mbarrier arrivals once they hold its data in
// registers; the group's warp 0 alone waits for that and issues the refill, so no warp ever blocks on another warp's progress through a tile.
#pragma once
#include <type_traits>
#include "sm80_common.cuh"
#ifndef K3_PROF
#define K3_PROF 0
#endif
#define K3P(i) do { if (K3_PROF) { const long long t_ = clock64(); pf[i] += (unsigned long long)(t_ - pt); pt = t_; } } while (0)

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
  unsigned long long* prof;    // K3_PROF builds: per-phase cycle sums [8] + tile count (else unused)
  float eps;
};

template <int CH_, int NG_ = 2>
struct K3Cfg {
  static constexpr int CH = CH_, CZ = 128, BM = 32, NG = NG_, NTHR = 128 * NG;
  static constexpr int KSX = CH / 16, KSZ = CZ / 16;
  static constexpr int SMEM_WO = CZ * CH * 2, SMEM_WG = CZ * CZ * 2;
  static constexpr int SMEM_X = CH * BM * 2, SMEM_Z = BM * CZ * 2;          // per group
  static constexpr int SMEM_GRP = SMEM_X + SMEM_Z;
  static constexpr int SMEM_V = 4 * CZ * 4;                                   // so, bo, sg, bg
  static constexpr int NBAR = 4;                                              // per group: X full, X free, z full, z free
  static constexpr int SMEM = SMEM_WO + SMEM_WG + NG * SMEM_GRP + SMEM_V + NG * NBAR * 8;
  static_assert(SMEM <= 166912, "sm_80 dynamic shared memory");
};

// 64-byte rows (32 tokens): granule g (0..3) of row r at g ^ ((r >> 1) & 3) -> the 8 rows of an ldmatrix land in 8 distinct 16 B bank slots
DEVI uint32_t swz64(uint32_t row, uint32_t g) { return row * 64 + ((g ^ ((row >> 1) & 3u)) << 4); }

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) k3_kernel(const K3Params p) {
  constexpr int CH = G::CH, CZ = G::CZ, BM = G::BM, KSX = G::KSX, KSZ = G::KSZ;
  constexpr uint32_t ONES = 0x3f803f80u;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int grp = warp >> 2, nw = warp & 3, g8 = lane >> 2, q = lane & 3;
  uint8_t* sWo = smem;
  uint8_t* sWg = sWo + G::SMEM_WO;
  uint8_t* sGrp = sWg + G::SMEM_WG + grp * G::SMEM_GRP;
  float* sV = reinterpret_cast<float*>(sWg + G::SMEM_WG + G::NG * G::SMEM_GRP);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sV + 4 * CZ) + grp * G::NBAR;
  const uint32_t sWo_u = smem_u32(sWo), sWg_u = smem_u32(sWg), sX_u = smem_u32(sGrp), sZ_u = sX_u + G::SMEM_X;
  const uint32_t barX = smem_u32(bars), barXfree = barX + 8, barZ = barX + 16, barZfree = barX + 24;
  const bool filler = nw == 0;                   // the group's warp 0 issues every X / z refill

  // group tile sequence: tiles grp + NG (blockIdx.x + gridDim.x i)
  const int gstride = G::NG * (int)gridDim.x, gfirst = G::NG * (int)blockIdx.x + grp;
  const int n_iter = gfirst < p.num_tiles ? (p.num_tiles - gfirst + gstride - 1) / gstride : 0;
  if ((tid & 127) == 0) { mbar_init(barX, 32); mbar_init(barXfree, 4); mbar_init(barZ, 32); mbar_init(barZfree, 4); }
  for (int i = tid; i < CZ; i += G::NTHR) { sV[i] = p.so[i]; sV[CZ + i] = p.bo[i]; sV[2 * CZ + i] = p.sg[i]; sV[3 * CZ + i] = p.bg[i]; }
  for (int c = tid; c < CZ * CH / 8; c += G::NTHR) {             // resident weights (row-major, 16 B granules swizzled by row)
    const int row = c / (CH / 8), g = c % (CH / 8);
    cp_async16(sWo_u + swz<CH * 2>(row, g * 16), p.wo + (size_t)row * CH + g * 8);
  }
  for (int c = tid; c < CZ * CZ / 8; c += G::NTHR) {
    const int row = c >> 4, g = c & 15;
    cp_async16(sWg_u + swz<256>(row, g * 16), p.wg + (size_t)row * CZ + g * 8);
  }
  cp_async_wait<0>();
  __syncthreads();
  auto load_x = [&](int tile) {                  // warp 0: [CH][32 tok] = CH x 4 granules, CH / 8 per lane
    const int t0 = tile * BM;
#pragma unroll 4
    for (int c = lane; c < CH * 4; c += 32) {
      const int row = c >> 2, g = c & 3;
      const bool ok = t0 + 8 * g < p.T;
      cp_async16(sX_u + swz64(row, g), p.x + (size_t)row * p.T + (ok ? t0 + 8 * g : 0), ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barX);
  };
  auto load_z = [&](int tile) {                  // warp 0: [32 tok][128 ch] = 32 x 16 granules, 16 per lane
    const int t0 = tile * BM;
#pragma unroll 4
    for (int c = lane; c < BM * 16; c += 32) {
      const int row = c >> 4, g = c & 15;
      const bool ok = t0 + row < p.T;
      cp_async16(sZ_u + swz<256>(row, g * 16), p.z + (size_t)(ok ? t0 + row : 0) * CZ + g * 8, ok ? 16u : 0u);
    }
    cp_async_mbar_arrive(barZ);
  };
  if (filler && n_iter > 0) { load_x(gfirst); load_z(gfirst); }
  // release a tile buffer: every read of this warp has returned (dependency on one register of each), then one arrival
  auto release = [&](uint32_t bar, uint32_t dep) {
    dep = __reduce_or_sync(0xffffffffu, dep);
    if (lane == 0 && dep != 0x9e3779b9u) mbar_arrive(bar);
  };
  // stats accumulators of one m16 tile at one k-step: s += x . 1, g0 += x x^T (tokens 0-7), g1 += x x^T (tokens 8-15)
  auto stat_ks = [&](float (&s)[4], float (&g0)[4], float (&g1)[4], const uint32_t (&a)[4]) {
    mma16816(s, a, ONES, ONES);
    mma16816(g0, a, a[0], a[2]);
    mma16816(g1, a, a[1], a[3]);
  };
  // (mu, r) of this thread's rows g8 (h = 0) and g8 + 8 (h = 1): sums are in every lane of the quad, the Gram diagonal only on lane q == g8 / 2
  auto finish = [&](const float (&s)[4], const float (&g0)[4], const float (&g1)[4], float n_inv, float (&mu)[2], float (&rs)[2]) {
    const bool odd = g8 & 1;
    const int src = (lane & ~3) | (g8 >> 1);
    const float qa = __shfl_sync(0xffffffffu, odd ? g0[1] : g0[0], src);
    const float qb = __shfl_sync(0xffffffffu, odd ? g1[3] : g1[2], src);
    const float ma = s[0] * n_inv, mb = s[2] * n_inv;
    mu[0] = ma; rs[0] = rsqrtf(fmaxf(qa * n_inv - ma * ma, 0.f) + p.eps);
    mu[1] = mb; rs[1] = rsqrtf(fmaxf(qb * n_inv - mb * mb, 0.f) + p.eps);
  };
  auto zero4 = [](float (&a)[4]) { a[0] = a[1] = a[2] = a[3] = 0.f; };

  // per-lane fragment address bases, constant over the tiles: X (A via .trans: k row 16 ks + 8 (mi / 2) + lane % 8, token granule 2 mt + mi % 2),
  // z (A: row 16 mt + lane % 16, granule 2 ks + lane / 16)
  const int mi = lane >> 3, kx = (mi >> 1) * 8 + (lane & 7);
  const uint32_t x_a0 = sX_u + swz64(kx, mi & 1), x_a1 = sX_u + swz64(kx, 2 + (mi & 1));   // + 1024 ks
  const int zr = (lane & 7) + ((lane >> 3) & 1) * 8;
  unsigned long long pf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  long long pt = K3_PROF ? clock64() : 0;

  for (int it = 0; it < n_iter; ++it) {
    const int tile = gfirst + it * gstride;
    const int t0 = tile * BM;
    const bool full = t0 + BM <= p.T;
    // ---- X fragments (A = tokens x channels from the channel-major tile), then the buffer goes back to warp 0
    mbar_wait(barX, it & 1);
    K3P(0);
    uint32_t fx[2][KSX][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int ks = 0; ks < KSX; ++ks) ldsm_x4_t(fx[mt][ks], (mt ? x_a1 : x_a0) + ks * 1024);
    release(barXfree, fx[1][KSX - 1][3] ^ fx[0][KSX - 1][3]);
    if (filler && it + 1 < n_iter) { mbar_wait(barXfree, it & 1); load_x(tile + gstride); }
    K3P(1);
    // ---- projection block 0 with the x statistics riding along, then block 1; p' = bf16(r acc - r mu s + b)
    float mux[2][2], rsx[2][2];
    uint32_t pst[2][2][2][2];                    // [nb][mt][n8][row half] packed bf16 channel pairs
    auto proj_epi = [&](const float (&acc)[2][2][4], int nb) {
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int oc = 32 * nw + 16 * nb + 8 * n + 2 * q;
        const float2 s2 = *reinterpret_cast<const float2*>(sV + oc), b2 = *reinterpret_cast<const float2*>(sV + CZ + oc);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float r = rsx[mt][h], rm = r * mux[mt][h];
            pst[nb][mt][n][h] = pack_bf16(fmaf(r, acc[mt][n][2 * h], fmaf(-rm, s2.x, b2.x)), fmaf(r, acc[mt][n][2 * h + 1], fmaf(-rm, s2.y, b2.y)));
          }
      }
    };
    {
      float acc[2][2][4], st[2][3][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) { zero4(acc[mt][0]); zero4(acc[mt][1]); zero4(st[mt][0]); zero4(st[mt][1]); zero4(st[mt][2]); }
#pragma unroll
      for (int ks = 0; ks < KSX; ++ks) {
        uint32_t b[4];
        ldsm_x4(b, sWo_u + swz<CH * 2>(32 * nw + ((lane >> 4) & 1) * 8 + (lane & 7), (2 * ks + ((lane >> 3) & 1)) * 16));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma16816(acc[mt][0], fx[mt][ks], b[0], b[1]); mma16816(acc[mt][1], fx[mt][ks], b[2], b[3]);
          stat_ks(st[mt][0], st[mt][1], st[mt][2], fx[mt][ks]);
        }
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) finish(st[mt][0], st[mt][1], st[mt][2], 1.f / CH, mux[mt], rsx[mt]);
      proj_epi(acc, 0);
    }
    {
      float acc[2][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) { zero4(acc[mt][0]); zero4(acc[mt][1]); }
#pragma unroll
      for (int ks = 0; ks < KSX; ++ks) {
        uint32_t b[4];
        ldsm_x4(b, sWo_u + swz<CH * 2>(32 * nw + 16 + ((lane >> 4) & 1) * 8 + (lane & 7), (2 * ks + ((lane >> 3) & 1)) * 16));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(acc[mt][0], fx[mt][ks], b[0], b[1]); mma16816(acc[mt][1], fx[mt][ks], b[2], b[3]); }
      }
      proj_epi(acc, 1);
    }
    K3P(2);
    // ---- z fragments + this lane's residual values, then the buffer goes back to warp 0 (the refill lands under the next tile's projection)
    mbar_wait(barZ, it & 1);
    K3P(3);
    uint32_t fz[2][KSZ][4], zres[2][2][2][2];    // zres [nb][mt][n8][row half]: z channel pairs at the accumulator positions
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int ks = 0; ks < KSZ; ++ks) ldsm_x4(fz[mt][ks], sZ_u + swz<256>(16 * mt + zr, (2 * ks + (lane >> 4)) * 16));
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
#pragma unroll
      for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int h = 0; h < 2; ++h) zres[nb][mt][n][h] = lds32(sZ_u + swz<256>(16 * mt + g8 + 8 * h, (32 * nw + 16 * nb + 8 * n + 2 * q) * 2));
    release(barZfree, fz[1][KSZ - 1][3] ^ fz[0][KSZ - 1][3] ^ zres[1][1][1][1] ^ zres[0][1][1][1]);
    if (filler && it + 1 < n_iter) { mbar_wait(barZfree, it & 1); load_z(tile + gstride); }
    K3P(4);
    // ---- gate block 0 with the z statistics riding along, then block 1; out = z + bf16(p' (1 + tanh(r acc - r mu s + b)))
    float muz[2][2], rsz[2][2];
    __nv_bfloat16* orow = p.out + (size_t)t0 * CZ;
    auto gate_epi = [&](const float (&acc)[2][2][4], int nb) {
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int oc = 32 * nw + 16 * nb + 8 * n + 2 * q;
        const float2 s2 = *reinterpret_cast<const float2*>(sV + 2 * CZ + oc), b2 = *reinterpret_cast<const float2*>(sV + 3 * CZ + oc);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = 16 * mt + g8 + 8 * h;
            const float r = rsz[mt][h], rm = r * muz[mt][h];
            const uint32_t pv = pst[nb][mt][n][h];
            const float p0 = bf16lo(pv), p1 = bf16hi(pv);
            const float o0 = fmaf(p0, tanh_approx(fmaf(r, acc[mt][n][2 * h], fmaf(-rm, s2.x, b2.x))), p0);
            const float o1 = fmaf(p1, tanh_approx(fmaf(r, acc[mt][n][2 * h + 1], fmaf(-rm, s2.y, b2.y))), p1);
            if (full || t0 + row < p.T) stg32(orow + row * CZ + oc, add_bf16x2(zres[nb][mt][n][h], pack_bf16(o0, o1)));
          }
      }
    };
    {
      float acc[2][2][4], st[2][3][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) { zero4(acc[mt][0]); zero4(acc[mt][1]); zero4(st[mt][0]); zero4(st[mt][1]); zero4(st[mt][2]); }
#pragma unroll
      for (int ks = 0; ks < KSZ; ++ks) {
        uint32_t b[4];
        ldsm_x4(b, sWg_u + swz<256>(32 * nw + ((lane >> 4) & 1) * 8 + (lane & 7), (2 * ks + ((lane >> 3) & 1)) * 16));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma16816(acc[mt][0], fz[mt][ks], b[0], b[1]); mma16816(acc[mt][1], fz[mt][ks], b[2], b[3]);
          stat_ks(st[mt][0], st[mt][1], st[mt][2], fz[mt][ks]);
        }
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) finish(st[mt][0], st[mt][1], st[mt][2], 1.f / CZ, muz[mt], rsz[mt]);
      gate_epi(acc, 0);
    }
    {
      float acc[2][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) { zero4(acc[mt][0]); zero4(acc[mt][1]); }
#pragma unroll
      for (int ks = 0; ks < KSZ; ++ks) {
        uint32_t b[4];
        ldsm_x4(b, sWg_u + swz<256>(32 * nw + 16 + ((lane >> 4) & 1) * 8 + (lane & 7), (2 * ks + ((lane >> 3) & 1)) * 16));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(acc[mt][0], fz[mt][ks], b[0], b[1]); mma16816(acc[mt][1], fz[mt][ks], b[2], b[3]); }
      }
      gate_epi(acc, 1);
    }
    K3P(5);
  }
  if (K3_PROF && lane == 0 && p.prof) { for (int i = 0; i < 8; ++i) atomicAdd(p.prof + i, pf[i]); atomicAdd(p.prof + 8, (unsigned long long)n_iter); }
  cp_async_wait<0>();
}

}  // namespace a100
