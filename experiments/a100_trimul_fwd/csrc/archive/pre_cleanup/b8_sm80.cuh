// b8_sm80.cuh -- TriMul backward, input side, A100 / sm_80: dx_n = [dgp | d_g] . [W_in ; W_og] fused with the LayerNorm_in backward
//
//   dxn[t, :] = sum_k dgp[t, k] W[k, :]                     (K = 4 CH + 128, fp32 accumulate, never leaves the SM)
//   zh = (z - mu_i) r_i ;  dxh = dxn * gamma
//   dz = r_i (dxh - mean(dxh) - zh mean(dxh zh)) + dy        (bf16)
//   per-CTA partials: dgamma += sum_t dxn zh, dbeta += sum_t dxn                                            ([grid][2][128] fp32)
//
// Main loop = the contraction's (128 x 128 x 32 tiles, cp.async ring, 4 warps of 64 x 64, two CTAs per SM): A = dgp rows (K-major), B = W
// rows (k-major rows of 128 output channels, read MN-major with ldmatrix.trans).  Epilogue: the drained ring stages the z and dy tiles
// (coalesced cp.async), the row sums are reduced across the quad and exchanged between the two column warps through smem, dz overwrites z
// in the staging tile and leaves as 16 B rows.
#pragma once
#include "sm80_common.cuh"
#include "contract_sm80.cuh"

namespace a100 {

struct B8Params {
  const __nv_bfloat16* dgp;   // [T][ldd]
  const __nv_bfloat16* w;     // [K][128]
  const __nv_bfloat16* z;     // [T][128]
  const __nv_bfloat16* dy;    // [T][128]
  const float* stats;         // [T][4]: mu_o, r_o, mu_i, r_i
  const float* gamma;         // [128]
  __nv_bfloat16* dz;          // [T][128]
  float* part;                // [grid][2][128]
  int T, K, ldd, num_tiles;
};

struct B8Cfg {
  static constexpr int BM = 128, BN = 128, BK = 32, ST = 4, NTHR = 128;
  static constexpr int TILE_A = BM * BK * 2, TILE_B = BN * BK * 2;
  static constexpr int STAGE = TILE_A + TILE_B;
  static constexpr int RING = ST * STAGE;                          // 64 KB: also the z | dy staging tiles of the epilogue
  static constexpr int XCH = RING;                                  // [2 column warps][128 rows][2] fp32 row sums
  static constexpr int ACC = XCH + 2 * 128 * 2 * 4;                 // [2][128] fp32 dgamma | dbeta of this CTA
  static constexpr int GAM = ACC + 2 * 128 * 4;                     // [128] fp32 gamma
  static constexpr int SMEM = GAM + 128 * 4;
  static_assert(2 * SMEM <= 166912, "two CTAs per SM");
};

__global__ void __launch_bounds__(128, 2) b8_kernel(const B8Params p) {
  using G = B8Cfg;
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int BK = G::BK, ST = G::ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp >> 1, wn = warp & 1;
  const int g8 = lane >> 2, q = lane & 3, mi = lane >> 3, r8 = lane & 7;
  const uint32_t s0 = smem_u32(smem);
  float* xch = reinterpret_cast<float*>(smem + G::XCH);
  float* cacc = reinterpret_cast<float*>(smem + G::ACC);
  float* sgam = reinterpret_cast<float*>(smem + G::GAM);
  for (int i = tid; i < 256; i += 128) cacc[i] = 0.f;
  sgam[tid] = p.gamma[tid];
  const int nk = p.K / BK;
  constexpr int KK = BK / 16;
  uint32_t aoff[4][KK], boff[4][KK];
#pragma unroll
  for (int t = 0; t < 4; ++t)
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      aoff[t][kk] = kmaj(64 * wm + 16 * t + r8 + (mi & 1) * 8, 2 * kk + (mi >> 1));
      boff[t][kk] = G::TILE_A + mmaj(16 * kk + (mi & 1) * 8 + r8, 8 * wn + 2 * t + (mi >> 1));
    }
  auto gam2 = [&](int nt) { return *reinterpret_cast<const float2*>(sgam + 64 * wn + 8 * nt + 2 * q); };

#pragma unroll 1
  for (int tile = blockIdx.x; tile < p.num_tiles; tile += gridDim.x) {
    const int t0 = tile * G::BM;
    const __nv_bfloat16* A = p.dgp + (size_t)t0 * p.ldd;
    auto load_stage = [&](int kt, int st) {
      const uint32_t sa = s0 + st * G::STAGE, sb = sa + G::TILE_A;
      const int k0 = kt * BK;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int g = tid + 128 * e;
        const int row = g >> 2, cc = g & 3;                             // A: 128 token rows x 4 granules of k (zero past T)
        const bool ok = t0 + row < p.T;
        cp_async16(sa + kmaj(row, cc), A + (size_t)(ok ? row : 0) * p.ldd + k0 + cc * 8, ok ? 16u : 0u);
        const int kr = g >> 4, c16 = g & 15;                            // B: 32 k rows x 16 granules of channels
        cp_async16(sb + mmaj(kr, c16), p.w + (size_t)(k0 + kr) * 128 + c16 * 8);
      }
    };
    float acc[4][8][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) {
      if (st < nk) load_stage(st, st);
      cp_async_commit();
    }
    uint32_t af[2][4][4], bf[2][4][4];
    auto load_frags = [&](int buf, int slot, int kk) {
      const uint32_t sa = s0 + slot * G::STAGE;
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) ldsm_x4(af[buf][mt], sa + aoff[mt][kk]);
#pragma unroll
      for (int np = 0; np < 4; ++np) ldsm_x4_t(bf[buf][np], sa + boff[np][kk]);
    };
    auto mma_all = [&](int buf) {
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          mma16816(acc[mt][2 * np], af[buf][mt], bf[buf][np][0], bf[buf][np][1]);
          mma16816(acc[mt][2 * np + 1], af[buf][mt], bf[buf][np][2], bf[buf][np][3]);
        }
    };
    cp_async_wait<ST - 2>();
    __syncthreads();
    load_frags(0, 0, 0);
#pragma unroll 1
    for (int kt = 0; kt < nk; ++kt) {
#pragma unroll
      for (int kk = 0; kk < KK; ++kk) {
        if (kk == KK - 1) {
          cp_async_wait<ST - 3>();
          __syncthreads();
          if (kt + ST - 1 < nk) load_stage(kt + ST - 1, (kt + ST - 1) % ST);
          cp_async_commit();
          if (kt + 1 < nk) load_frags((kk + 1) & 1, (kt + 1) % ST, 0);
        } else {
          load_frags((kk + 1) & 1, kt % ST, kk + 1);
        }
        mma_all(kk & 1);
      }
    }
    cp_async_wait<0>();
    __syncthreads();                                                   // ring drained: stage z (first 32 KB) and dy (second 32 KB)
    // staging tile [128 rows][256 B], 16 B granule g of row r at g ^ (r & 7)
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const int gidx = tid + 128 * e, r = gidx >> 4, gc = gidx & 15;
      const uint32_t off = r * 256 + ((gc ^ (r & 7)) << 4);
      const bool ok = t0 + r < p.T;
      cp_async16(s0 + off, p.z + (size_t)(ok ? t0 + r : 0) * 128 + gc * 8, ok ? 16u : 0u);
      cp_async16(s0 + 32768 + off, p.dy + (size_t)(ok ? t0 + r : 0) * 128 + gc * 8, ok ? 16u : 0u);
    }
    cp_async_commit();
    float mu[4][2], rr[4][2];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int t = t0 + 64 * wm + 16 * mt + g8 + 8 * h;
        const float2 s = t < p.T ? *reinterpret_cast<const float2*>(p.stats + (size_t)t * 4 + 2) : make_float2(0.f, 0.f);
        mu[mt][h] = s.x; rr[mt][h] = s.y;                             // rows past T: r = 0, zero A rows -> no contribution
      }
    cp_async_wait<0>();
    __syncthreads();
    auto zoff = [&](int r, int nt) -> uint32_t {
      const int gc = (64 * wn + 8 * nt) >> 3;
      return r * 256 + ((gc ^ (r & 7)) << 4) + q * 4;
    };
    // row sums of dxh and dxh * zh over this thread's 16 columns -> quad -> exchange between the two column warps
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = 64 * wm + 16 * mt + g8 + 8 * h;
        float s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const uint32_t zw = lds32(s0 + zoff(r, nt));
          const float z0 = (bf16lo(zw) - mu[mt][h]) * rr[mt][h], z1 = (bf16hi(zw) - mu[mt][h]) * rr[mt][h];
          const float2 gm = gam2(nt);
          const float h0 = acc[mt][nt][2 * h] * gm.x, h1 = acc[mt][nt][2 * h + 1] * gm.y;
          s1 += h0 + h1;
          s2 += h0 * z0 + h1 * z1;
        }
        s1 += __shfl_xor_sync(0xffffffffu, s1, 1); s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
        s2 += __shfl_xor_sync(0xffffffffu, s2, 1); s2 += __shfl_xor_sync(0xffffffffu, s2, 2);
        if (q == 0) { xch[(wn * 128 + r) * 2] = s1; xch[(wn * 128 + r) * 2 + 1] = s2; }
      }
    // column partials (this thread's 8 rows) -> reduce over the 8 row groups of the warp -> CTA accumulator
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      float cg[2] = {0.f, 0.f}, cb[2] = {0.f, 0.f};
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t zw = lds32(s0 + zoff(64 * wm + 16 * mt + g8 + 8 * h, nt));
          const float d0 = acc[mt][nt][2 * h], d1 = acc[mt][nt][2 * h + 1];
          cg[0] = fmaf(d0, (bf16lo(zw) - mu[mt][h]) * rr[mt][h], cg[0]); cg[1] = fmaf(d1, (bf16hi(zw) - mu[mt][h]) * rr[mt][h], cg[1]);
          cb[0] += d0; cb[1] += d1;
        }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float a = cg[j], b = cb[j];
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); b += __shfl_xor_sync(0xffffffffu, b, o); }
        if (g8 == 0) {
          atomicAdd(&cacc[64 * wn + 8 * nt + 2 * q + j], a);
          atomicAdd(&cacc[128 + 64 * wn + 8 * nt + 2 * q + j], b);
        }
      }
    }
    __syncthreads();
    constexpr float inv = 1.f / 128.f;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = 64 * wm + 16 * mt + g8 + 8 * h;
        const float m1 = (xch[r * 2] + xch[(128 + r) * 2]) * inv, m2 = (xch[r * 2 + 1] + xch[(128 + r) * 2 + 1]) * inv;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const uint32_t o = zoff(r, nt);
          const uint32_t zw = lds32(s0 + o), dw = lds32(s0 + 32768 + o);
          const float z0 = (bf16lo(zw) - mu[mt][h]) * rr[mt][h], z1 = (bf16hi(zw) - mu[mt][h]) * rr[mt][h];
          const float2 gm = gam2(nt);
          const float o0 = rr[mt][h] * (acc[mt][nt][2 * h] * gm.x - m1 - z0 * m2) + bf16lo(dw);
          const float o1 = rr[mt][h] * (acc[mt][nt][2 * h + 1] * gm.y - m1 - z1 * m2) + bf16hi(dw);
          sts32(s0 + o, pack_bf16(o0, o1));
        }
      }
    __syncthreads();
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const int gidx = tid + 128 * e, r = gidx >> 4, gc = gidx & 15;
      if (t0 + r < p.T) stg128(p.dz + (size_t)(t0 + r) * 128 + gc * 8, lds128(s0 + r * 256 + ((gc ^ (r & 7)) << 4)));
    }
    __syncthreads();                                                   // the ring is reused by the next tile
  }
  for (int i = tid; i < 256; i += 128) p.part[(size_t)blockIdx.x * 256 + i] = cacc[i];
}

}  // namespace a100
