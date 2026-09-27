// b7j_sm80.cuh -- TriMul backward, input side, A100 / sm_80: the sm_90 "B7 joint" algorithm (h100_sources/b7_768/joint.cu) on sm_80.
//
// One cooperative launch of G groups x (S source + C consumer) CTAs.  A group walks the token tiles g, g + G, ... (128 tokens each):
//   source rank r (one 64-row block of the packed input weights; S = 4 CH / 64):  (g', p') = x_n . (0.5 W_r)^T, dg / dp from dA (the
//     contraction backward) and the pair mask, dW_r^T += x_n^T . dgp in registers for the whole launch; the dgp tile [128 tok][64 rows] goes
//     into the group's L2 ring slot (column block r) and a release flag says so.
//   consumer rank c (tiles k = c, c + C, ...):  waits for the S flags of its slot, then dx_n = [ring slot | d_g] . [W_in ; W_og] (the B8 main
//     loop, A streamed from the ring and from B1's d_g), LN_in backward + residual -> dz, per-CTA dgamma / dbeta partials; a consume flag
//     frees the slot.
// The dgp derivatives never make a DRAM round trip (the sm_90 kernel's L2 ring: TMA bulk stores + cluster barriers there; plain 16 B stores
// with an L2 evict_last policy, cp.async.cg loads and gpu-scope release / acquire flags here).
#pragma once
#include "sm80_common.cuh"
#include "contract_sm80.cuh"


#ifndef B7J_PROF
#define B7J_PROF 0                               // cycle counters: [0] source total, [1] source slot waits, [2] consumer total, [3] consumer flag waits
#endif

namespace a100 {
__device__ unsigned long long b7j_prof[4];

struct B7JParams {
  const __nv_bfloat16* xn;     // [T][128]
  const __nv_bfloat16* w1;     // [S][16 granules][64 rows][8]  (the K1 packing, 0.5 W)
  const __nv_bfloat16* dab;    // [2 CH][T]
  const uint8_t* mask;         // [L] or nullptr
  const __nv_bfloat16* wdx;    // [K4 + 128][128]: W_in rows (K1 order, unscaled) ; W_og
  const __nv_bfloat16* dg;     // [T][128] d_g of the output gate (B1)
  const __nv_bfloat16* z;      // [T][128]
  const __nv_bfloat16* dy;     // [T][128]
  const float* stats;          // [T][4] (mu_i, r_i at 2, 3)
  const float* gamma;          // [128] LN_in gamma
  __nv_bfloat16* ring;         // [G][RINGS][128][K4]
  unsigned* prod;              // [G][RINGS][S]   sequence + 1 of the tile a source published into the slot
  unsigned* cons;              // [G][RINGS]      times the slot was consumed
  __nv_bfloat16* dz;           // [T][128]
  float* dwpart;               // [G][S][64][128]
  float* lnpart;               // [G * C][2][128]
  int T, L, num_tiles, S, C, G, RINGS, K4;
};

struct B7JCfg {
  static constexpr int CZ = 128, BM = 128, NTHR = 128;
  // source
  static constexpr int SMEM_W = 64 * CZ * 2, SMEM_Z = BM * CZ * 2, SMEM_A = 32 * BM * 2, SMEM_D = BM * 64 * 2, MAXL = 1024;
  static constexpr int SRC_SMEM = SMEM_W + SMEM_Z + SMEM_A + SMEM_D + BM * 4 + MAXL;
  // consumer (the B8 layout)
  static constexpr int BK = 32, ST = 4, TILE_A = BM * BK * 2, TILE_B = 128 * BK * 2, STAGE = TILE_A + TILE_B, RINGB = ST * STAGE;
  static constexpr int XCH = RINGB, ACC = XCH + 2 * 128 * 2 * 4, GAM = ACC + 2 * 128 * 4, CON_SMEM = GAM + 128 * 4;
  static constexpr int SMEM = SRC_SMEM > CON_SMEM ? SRC_SMEM : CON_SMEM;
  static_assert(2 * SMEM <= 166912, "two CTAs per SM");
};

DEVI uint64_t l2_evict_last() { uint64_t v; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(v)); return v; }
DEVI void stg128_hint(void* p, uint4 v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol) : "memory");
}
DEVI unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
DEVI void st_release(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }
DEVI void wait_geq(const unsigned* p, unsigned want) { while (ld_acquire(p) < want) __nanosleep(64); }

// ------------------------------------------------------------------------------------------------------------------------------ source
DEVI void b7j_source(const B7JParams& p, uint8_t* smem, int g, int b) {
  using G = B7JCfg;
  constexpr int CZ = G::CZ, BM = G::BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q = lane & 3, mi = lane >> 3, r8 = lane & 7;
  uint8_t* sW = smem;
  uint8_t* sZ = sW + G::SMEM_W;
  uint8_t* sA = sZ + G::SMEM_Z;
  uint8_t* sD = sA + G::SMEM_A;
  float* sM = reinterpret_cast<float*>(sD + G::SMEM_D);
  uint8_t* sMask = reinterpret_cast<uint8_t*>(sM + BM);
  const bool smask = p.mask != nullptr && p.L <= G::MAXL;
  if (smask) for (int i = tid; i < p.L; i += 128) sMask[i] = p.mask[i];
  const uint32_t sW_u = smem_u32(sW), sZ_u = smem_u32(sZ), sA_u = smem_u32(sA), sD_u = smem_u32(sD);
  const int n_iter = g < p.num_tiles ? (p.num_tiles - g + p.G - 1) / p.G : 0;
  const uint64_t pol = l2_evict_last();
#pragma unroll
  for (int i = 0; i < 8; ++i) { const int c = tid + 128 * i; cp_async16(sW_u + c * 16, p.w1 + (size_t)b * 64 * CZ + c * 8); }
  cp_async_commit();
  float dwa[2][8][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dwa[mt][n][e] = 0.f;
  const int oc0 = 32 * b;
  const int r0 = tid >> 4, gq = tid & 15;
  const uint32_t zdst = sZ_u + r0 * 256 + ((gq ^ (r0 & 7)) << 4), adst = sA_u + r0 * 256 + ((gq ^ (r0 & 7)) << 4);
  auto load_z = [&](int t0) {
    const __nv_bfloat16* zsrc = p.xn + (size_t)(t0 + r0) * CZ + gq * 8;
#pragma unroll
    for (int i = 0; i < 16; ++i) cp_async16(zdst + i * 2048, zsrc + i * 8 * CZ);
  };
  auto load_a = [&](int t0) {
    const __nv_bfloat16* asrc = p.dab + (size_t)(oc0 + r0) * p.T + t0 + gq * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) cp_async16(adst + i * 2048, asrc + (size_t)i * 8 * p.T);
  };
  auto set_mask = [&](int t0) {
    float m = 1.f;
    if (p.mask != nullptr) {
      int i = t0 / p.L, j = t0 - i * p.L + tid;
      while (j >= p.L) { j -= p.L; ++i; }
      m = (smask ? (sMask[i] && sMask[j]) : (p.mask[i] && p.mask[j])) ? 1.f : 0.f;
    }
    sM[tid] = m;
  };
  __syncthreads();                               // sMask
  if (n_iter > 0) { load_z(g * BM); load_a(g * BM); cp_async_commit(); set_mask(g * BM); }
  // staging / ring-store lane constants
  const uint32_t gsw = (uint32_t)g8 << 4;
  const uint32_t sD_base = sD_u + (32 * warp + g8) * 128 + q * 4;
  const int sr0 = tid >> 3, sg = tid & 7;
  const uint32_t ssrc = sD_u + sr0 * 128 + ((sg ^ (sr0 & 7)) << 4);
  const size_t sstep = (size_t)16 * p.K4;
  for (int it = 0; it < n_iter; ++it) {
    const int t0 = (g + it * p.G) * BM, tn = t0 + p.G * BM, slot = it % p.RINGS;
    const bool has_next = it + 1 < n_iter;
    cp_async_wait<0>();
    __syncthreads();
    float acc[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mt][n][e] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t a[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int row = 32 * warp + 16 * mt + r8 + ((lane >> 3) & 1) * 8;
        ldsm_x4(a[mt], sZ_u + swz<256>(row, (2 * ks + (lane >> 4)) * 16));
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bb[4];
        ldsm_x4(bb, sW_u + ((lane >> 3) & 1) * 1024 + (16 * np + ((lane >> 4) & 1) * 8 + r8) * 16 + ks * 2048);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(acc[mt][2 * np], a[mt], bb[0], bb[1]); mma16816(acc[mt][2 * np + 1], a[mt], bb[2], bb[3]); }
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kq = 0; kq < 2; ++kq) {
        uint32_t f[4];
        const int krow = 16 * kq + (mi >> 1) * 8 + r8, tg = (32 * warp + 16 * mt) / 8 + (mi & 1);
        ldsm_x4_t(f, sA_u + swz<256>(krow, tg * 16));
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int np = 2 * kq + hh;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = 32 * warp + 16 * mt + g8 + 8 * h;
            const uint32_t dv = f[2 * hh + h];
            const float hm = 0.5f * sM[row];
            const float da0 = bf16lo(dv) * hm, da1 = bf16hi(dv) * hm;
            const float th0 = tanh_approx(acc[mt][2 * np][2 * h]), th1 = tanh_approx(acc[mt][2 * np][2 * h + 1]);
            const float p0 = acc[mt][2 * np + 1][2 * h], p1 = acc[mt][2 * np + 1][2 * h + 1];
            const uint32_t ga = sD_base + (16 * mt + 8 * h) * 128 + ((2 * np) << 4 ^ gsw);
            sts32(ga, pack_bf16(da0 * p0 * fmaf(-th0, th0, 1.f), da1 * p1 * fmaf(-th1, th1, 1.f)));
            sts32(ga ^ 16u, pack_bf16(fmaf(da0, th0, da0), fmaf(da1, th1, da1)));
          }
        }
      }
    // the ring slot must have been consumed RINGS tiles ago
    if (tid == 0 && it >= p.RINGS) {
      const long long w0 = B7J_PROF ? clock64() : 0;
      wait_geq(p.cons + g * p.RINGS + slot, (unsigned)(it / p.RINGS));
      if (B7J_PROF) atomicAdd(&b7j_prof[1], (unsigned long long)(clock64() - w0));
    }
    __syncthreads();                             // dAB / mask consumed, dgp tile staged, slot free
    if (has_next) { load_a(tn); cp_async_commit(); set_mask(tn); }
    {
      __nv_bfloat16* dst = p.ring + ((size_t)(g * p.RINGS + slot) * BM + sr0) * p.K4 + 64 * b + 8 * sg;
#pragma unroll
      for (int i = 0; i < 8; ++i) { stg128_hint(dst, lds128(ssrc + i * 2048), pol); dst += sstep; }
    }
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t a[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int trow = 16 * ks + (mi >> 1) * 8 + r8, cg = (32 * warp + 16 * mt) / 8 + (mi & 1);
        ldsm_x4_t(a[mt], sZ_u + swz<256>(trow, cg * 16));
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bb[4];
        const int trow = 16 * ks + (mi & 1) * 8 + r8, rg = 2 * np + (mi >> 1);
        ldsm_x4_t(bb, sD_u + trow * 128 + ((rg ^ (trow & 7)) << 4));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(dwa[mt][2 * np], a[mt], bb[0], bb[1]); mma16816(dwa[mt][2 * np + 1], a[mt], bb[2], bb[3]); }
      }
    }
    __syncthreads();                             // x_n / dgp tiles free; every thread's ring stores issued
    if (tid == 0) { __threadfence(); st_release(p.prod + (g * p.RINGS + slot) * p.S + b, (unsigned)(it + 1)); }
    if (has_next) { load_z(tn); cp_async_commit(); }
  }
  float* dst = p.dwpart + ((size_t)g * p.S + b) * 64 * CZ;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int c = 32 * warp + 16 * mt + g8 + 8 * h, rw = 8 * n + 2 * q;
        dst[(size_t)rw * CZ + c] = dwa[mt][n][2 * h];
        dst[(size_t)(rw + 1) * CZ + c] = dwa[mt][n][2 * h + 1];
      }
}

// ---------------------------------------------------------------------------------------------------------------------------- consumer
DEVI void b7j_consumer(const B7JParams& p, uint8_t* smem, int g, int c) {
  using G = B7JCfg;
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
  const int K = p.K4 + 128, nk = K / BK, nk4 = p.K4 / BK;
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
  const int n_seq = g < p.num_tiles ? (p.num_tiles - g + p.G - 1) / p.G : 0;
#pragma unroll 1
  for (int k = c; k < n_seq; k += p.C) {
    const int t0 = (g + k * p.G) * G::BM, slot = k % p.RINGS;
    const __nv_bfloat16* R = p.ring + (size_t)(g * p.RINGS + slot) * G::BM * p.K4;
    // the S sources of this slot have published tile k
    if (warp == 0 && lane < p.S) {
      const long long w0 = B7J_PROF ? clock64() : 0;
      wait_geq(p.prod + (g * p.RINGS + slot) * p.S + lane, (unsigned)(k + 1));
      if (B7J_PROF && lane == 0) atomicAdd(&b7j_prof[3], (unsigned long long)(clock64() - w0));
    }
    __syncthreads();
    auto load_stage = [&](int kt, int st) {
      const uint32_t sa = s0 + st * G::STAGE, sb = sa + G::TILE_A;
      const int k0 = kt * BK;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int gg = tid + 128 * e;
        const int row = gg >> 2, cc = gg & 3;
        const __nv_bfloat16* src = kt < nk4 ? R + (size_t)row * p.K4 + k0 + cc * 8 : p.dg + (size_t)(t0 + row) * 128 + (k0 - p.K4) + cc * 8;
        cp_async16(sa + kmaj(row, cc), src);
        const int kr = gg >> 4, c16 = gg & 15;
        cp_async16(sb + mmaj(kr, c16), p.wdx + (size_t)(k0 + kr) * 128 + c16 * 8);
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
    auto load_frags = [&](int buf, int sl, int kk) {
      const uint32_t sa = s0 + sl * G::STAGE;
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
    __syncthreads();                                                   // every ring read landed: the slot is free
    if (tid == 0) st_release(p.cons + g * p.RINGS + slot, (unsigned)(k / p.RINGS + 1));
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const int gidx = tid + 128 * e, r = gidx >> 4, gc = gidx & 15;
      const uint32_t off = r * 256 + ((gc ^ (r & 7)) << 4);
      cp_async16(s0 + off, p.z + (size_t)(t0 + r) * 128 + gc * 8);
      cp_async16(s0 + 32768 + off, p.dy + (size_t)(t0 + r) * 128 + gc * 8);
    }
    cp_async_commit();
    float mu[4][2], rr[4][2];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float2 s = *reinterpret_cast<const float2*>(p.stats + (size_t)(t0 + 64 * wm + 16 * mt + g8 + 8 * h) * 4 + 2);
        mu[mt][h] = s.x; rr[mt][h] = s.y;
      }
    cp_async_wait<0>();
    __syncthreads();
    auto zoff = [&](int r, int nt) -> uint32_t {
      const int gc = (64 * wn + 8 * nt) >> 3;
      return r * 256 + ((gc ^ (r & 7)) << 4) + q * 4;
    };
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
        float a = cg[j], bsum = cb[j];
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); bsum += __shfl_xor_sync(0xffffffffu, bsum, o); }
        if (g8 == 0) {
          atomicAdd(&cacc[64 * wn + 8 * nt + 2 * q + j], a);
          atomicAdd(&cacc[128 + 64 * wn + 8 * nt + 2 * q + j], bsum);
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
      stg128(p.dz + (size_t)(t0 + r) * 128 + gc * 8, lds128(s0 + r * 256 + ((gc ^ (r & 7)) << 4)));
    }
    __syncthreads();
  }
  for (int i = tid; i < 256; i += 128) p.lnpart[(size_t)(g * p.C + c) * 256 + i] = cacc[i];
}

__global__ void __launch_bounds__(128, 2) b7j_kernel(const B7JParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int per = p.S + p.C, g = blockIdx.x / per, r = blockIdx.x % per;
  const long long t0 = B7J_PROF ? clock64() : 0;
  if (r < p.S) b7j_source(p, smem, g, r);
  else b7j_consumer(p, smem, g, r - p.S);
  if (B7J_PROF && threadIdx.x == 0) atomicAdd(&b7j_prof[r < p.S ? 0 : 2], (unsigned long long)(clock64() - t0));
}

}  // namespace a100
