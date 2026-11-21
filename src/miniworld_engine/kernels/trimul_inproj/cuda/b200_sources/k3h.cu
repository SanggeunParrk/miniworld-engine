// k3h.cu -- the TriMul output (k3g.cu inference: LN_out + output projection + gate + residual) on M = 64 tcgen05 tiles, for small L:
//   xo = bf16(LN_out(t)),  xn = bf16(LN_in(x)),  p = xo Wp^T,  g = xn Wg^T,  y = bf16(x + bf16(sigmoid(g) p))
// Why (2026-11-17, round 3 sixth pass): see k1h.cu -- at L128 / L256 k3g's time is one 128-token tile's dependency chain. Here a tile
// is 64 tokens (tcgen05.mma cta_group::1 M = 64; two CTAs per SM): LN_in on two threads per row, LN_out on 16 tokens per warp, the
// epilogue on 16 rows x half the channels per warp.
//
// Both A operands live in shared memory (SW128 K-major [64 tok][64 ch] blocks): xn written by LN_in, xo by LN_out with stmatrix from
// k3g's fragment layout (natural channel order: the plain W_p, not k3g's TMEM-permuted wpp). Accumulators (M = 64 data path: row
// 16 w + i in TMEM lane 32 w + i): p in columns [0, C), g in [C, 2 C); the epilogue reads them with tcgen05.ld.16x256b (fragment
// layout) and applies the residual / output in place in the x tile (4-byte shared accesses, conflict-free under SW128), TMA store.
//
// 384 threads: warp 0 TMA (weights, x), warp 1 MMA, warp 2 TMEM + TMA t (after griddepcontrol.wait), warps 4-7 LN_in, warps 8-11
// LN_out; then the epilogue on warps 8-11 (channels [0, C/2)) and 4-7 ([C/2, C)) when every CTA has one tile, else on 8-11 alone.
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;
#ifdef TM_TL
static __device__ unsigned long long g_tl3h[320][16];
#define TL3(slot) { g_tl3h[blockIdx.x][slot] = clock64(); }
#else
#define TL3(slot) {}
#endif
#ifdef PH_TL   // per-CTA globaltimer phase marks (ns): [entry, griddepcontrol.wait returned, end]; read with ph_k3h()
static __device__ unsigned long long g_ph[320][3];
#define PHM(slot) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_ph[blockIdx.x][slot] = t_; }
#else
#define PHM(slot) {}
#endif

namespace k3h {

constexpr int TOK = 64, XB = TOK * 64 * 2;   // one [64 tok][64 ch] SW128 block = 8 KB
constexpr int NTHREADS = 384;

template <int C, int H>
struct Geo {
  static constexpr int WGB = C * 128, WG = (C / 64) * WGB, WP = (H / 64) * WGB;   // weight K-block [C rows][64 k]
  static constexpr int XT = (C / 64) * XB, TT = H * TOK * 2, XO = (H / 64) * XB;
  static constexpr int O_WG = 0, O_WP = O_WG + WG, O_X = O_WP + WP, O_XN = O_X + 2 * XT, O_T = O_XN + XT, O_XO = O_T + TT,
                       O_GB = O_XO + XO, O_BAR = O_GB + 8 * H + 8 * C;
  static constexpr int SMEM = O_BAR + 256;
  static_assert(SMEM + 1024 <= 233472 / 2, "two CTAs per SM");
  static constexpr int TMC = 2 * C <= 128 ? 128 : 256;   // TMEM columns
};

struct Bars {
  uint64_t w_full, x_full[2], x_empty[2], t_full, t_empty, xn_ready, xo_ready, a_free, acc_full, acc_empty;
  uint32_t tmem;
};

DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float2 rbf2(float2 v) {
  const uint32_t w = pack_bf16(v.x, v.y);
  return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}
DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
DEV void sts32(uint32_t a, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;" ::"r"(a), "r"(v) : "memory"); }
DEV float2 lds64f(uint32_t a) { float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory"); return v; }
DEV float4 lds128f(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
DEV void ldsm_x4_trans(uint32_t addr, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
DEV void tmem_ld16x256_x4(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
                 "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}

template <int C, int H>
__global__ void __launch_bounds__(NTHREADS, 2)
    k3h_kernel(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mtri, const __grid_constant__ CUtensorMap mwp,
               const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap my, const float* __restrict__ g_in,
               const float* __restrict__ b_in, const float* __restrict__ g_out, const float* __restrict__ b_out, int tiles, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];   // no static shared memory: the dynamic window starts 1024-aligned
  using GG = Geo<C, H>;
  constexpr int XT = GG::XT, TT = GG::TT;
  constexpr uint32_t IDG = idesc_bf16(64, C);
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + GG::O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int n_local = tiles > cta ? (tiles - cta + G - 1) / G : 0;
  const bool lat = tiles <= G;   // one tile per CTA: the LN_in warps take half of the epilogue
  const int nx0 = min(n_local, 2);
  auto row0_of = [&](int i) { return (cta + i * G) * TOK; };
  auto xload = [&](int b, int i) {
    mbar_expect_tx(&B.x_full[b], XT);
    for (int kb = 0; kb < C / 64; ++kb) tma_load_2d(sm + GG::O_X + b * XT + kb * XB, &mx, &B.x_full[b], kb * 64, row0_of(i), EVICT_FIRST);
  };
  pdl_trigger();
  if (tid == 0) TL3(0)
  if (tid == 0) PHM(0)
  if (tid == 0) {
    mbar_init(&B.w_full, 1);
    for (int b = 0; b < 2; ++b) { mbar_init(&B.x_full[b], 1); mbar_init(&B.x_empty[b], 1); }
    mbar_init(&B.t_full, 1); mbar_init(&B.t_empty, 4);
    mbar_init(&B.xn_ready, 1); mbar_init(&B.xo_ready, 1); mbar_init(&B.a_free, 1);
    mbar_init(&B.acc_full, 1); mbar_init(&B.acc_empty, lat ? 8 : 4);   // one arrival per epilogue warp
    fence_mbar_init();
    prefetch_tmap(&mtri); prefetch_tmap(&my);
    // x and the weights are not written by the contraction this kernel follows: loaded before griddepcontrol.wait
    mbar_expect_tx(&B.w_full, GG::WG + GG::WP);
    for (int kb = 0; kb < C / 64; ++kb) tma_load_2d(sm + GG::O_WG + kb * GG::WGB, &mwg, &B.w_full, kb * 64, 0, EVICT_LAST);
    for (int kb = 0; kb < H / 64; ++kb) tma_load_2d(sm + GG::O_WP + kb * GG::WGB, &mwp, &B.w_full, kb * 64, 0, EVICT_LAST);
    for (int b = 0; b < nx0; ++b) xload(b, b);
  }
  {
    float* gb = reinterpret_cast<float*>(sm + GG::O_GB);   // g_out [H], b_out [H], g_in [C], b_in [C]
    for (int k = tid; k < H; k += NTHREADS) { gb[k] = g_out[k]; gb[H + k] = b_out[k]; }
    for (int k = tid; k < C; k += NTHREADS) { gb[2 * H + k] = g_in[k]; gb[2 * H + C + k] = b_in[k]; }
  }
  if (warp == 2) { tmem_alloc(&B.tmem, GG::TMC); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (tid == 0) TL3(1)
  constexpr uint32_t T_P = 0, T_G = C;
  const uint32_t O_GO = su + GG::O_GB, O_BO = O_GO + 4 * H, O_GI = O_GO + 8 * H, O_BI = O_GI + 4 * C;

  if (warp == 0 && lane == 0) {
    for (int i = nx0; i < n_local; ++i) {
      const int b = i & 1;
      mbar_wait(&B.x_empty[b], ((i >> 1) - 1) & 1);
      xload(b, i);
    }
  } else if (warp == 2 && lane == 0) {
    pdl_wait();   // t is the previous kernel's output
    PHM(1)
    for (int i = 0; i < n_local; ++i) {
      if (i > 0) mbar_wait(&B.t_empty, (i - 1) & 1);
      mbar_expect_tx(&B.t_full, TT);
      tma_load_2d(sm + GG::O_T, &mtri, &B.t_full, row0_of(i), 0, EVICT_FIRST);
    }
  } else if (warp == 1 && lane == 0) {
    mbar_wait(&B.w_full, 0);
    for (int i = 0; i < n_local; ++i) {
      if (i > 0) mbar_wait(&B.acc_empty, (i - 1) & 1);
      mbar_wait(&B.xn_ready, i & 1);
      tc_fence_after();
#pragma unroll
      for (int k = 0; k < C / 16; ++k)
        umma_ss(tmem + T_G, desc_k_sw128(su + GG::O_XN + (k >> 2) * XB + (k & 3) * 32), desc_k_sw128(su + GG::O_WG + (k >> 2) * GG::WGB + (k & 3) * 32),
                IDG, k != 0);
      mbar_wait(&B.xo_ready, i & 1);
      tc_fence_after();
      if (i == 0) TL3(4)
#pragma unroll
      for (int k = 0; k < H / 16; ++k)
        umma_ss(tmem + T_P, desc_k_sw128(su + GG::O_XO + (k >> 2) * XB + (k & 3) * 32), desc_k_sw128(su + GG::O_WP + (k >> 2) * GG::WGB + (k & 3) * 32),
                IDG, k != 0);
      umma_commit(&B.a_free);
      umma_commit(&B.acc_full);
    }
  }

  // the epilogue of tile i, run by the LN_out warps right after their LN_out of tile i (and by the LN_in warps when every CTA has one
  // tile): with several tiles per CTA, an epilogue after the whole LN_out loop would deadlock (tile i + 2's LN_out waits for tile
  // i + 1's products, which wait for tile i's epilogue to free the accumulators)
  auto epi_tile = [&](int i, int part, int nparts) {
    // fragment layout: lanes 32 q .. 32 q + 15 hold rows 16 q .. 16 q + 15; thread t: rows tr, tr + 8, columns 8 j + 2 (t % 4) + {0, 1}
    const int q = warp & 3, tr = lane >> 2, t4 = lane & 3;
    constexpr int CW = 32;   // channels per TMEM load (x4: 32 columns)
    const int nq = C / CW / nparts, cq0 = part * nq;
    {
      const int b = i & 1;
      mbar_wait(&B.acc_full, i & 1);
      mbar_wait(&B.x_full[b], (i >> 1) & 1);
      tc_fence_after();
      if (i == 0 && lane == 0 && q == 0) TL3(9 + part)
      const uint32_t xr = su + GG::O_X + b * XT;
      for (int cq = cq0; cq < cq0 + nq; ++cq) {
        uint32_t pv[16], gv[16];
        const uint32_t ta = tmem + ((uint32_t)(q * 32) << 16) + cq * CW;
        tmem_ld16x256_x4(ta + T_P, pv);
        tmem_ld16x256_x4(ta + T_G, gv);
        tmem_wait_ld();
        if (cq == cq0 + nq - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty); }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int ch = cq * CW + 8 * j + 2 * t4;
#pragma unroll
          for (int tb = 0; tb < 2; ++tb) {
            const int r = 16 * q + tr + 8 * tb;
            const uint32_t a = xr + (ch >> 6) * XB + sw128(r, (ch & 63) >> 3) + (ch & 7) * 2;
            const uint32_t xw = lds32(a);
            const float2 pp = make_float2(__uint_as_float(pv[4 * j + 2 * tb]), __uint_as_float(pv[4 * j + 2 * tb + 1]));
            const float2 gg = make_float2(__uint_as_float(gv[4 * j + 2 * tb]), __uint_as_float(gv[4 * j + 2 * tb + 1]));
            const float2 h = __fmul2_rn(gg, make_float2(0.5f, 0.5f));   // sigmoid(g) = 0.5 tanh(g / 2) + 0.5 (k3g K3_SIG 2)
            float2 th;
            asm("tanh.approx.f32 %0, %1;" : "=f"(th.x) : "f"(h.x));
            asm("tanh.approx.f32 %0, %1;" : "=f"(th.y) : "f"(h.y));
            const float2 sg = __ffma2_rn(th, make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
            const float2 y = __fadd2_rn(make_float2(bf16lo(xw), bf16hi(xw)), rbf2(__fmul2_rn(sg, pp)));
            sts32(a, pack_bf16(y.x, y.y));
          }
        }
      }
      fence_async_smem();
      named_bar_sync(3, 128 * nparts);
      if (warp == 8 && lane == 0) {
        for (int kb = 0; kb < C / 64; ++kb) tma_store_2d(&my, sm + GG::O_X + b * XT + kb * XB, kb * 64, row0_of(i));
        bulk_commit();
        bulk_wait_read<0>();
        mbar_arrive(&B.x_empty[b]);
      }
    }
  };

  if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------ LN_in: thread pair = row r, half h of the channels
    const int t = tid - 128, r = t >> 1, h = t & 1;
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    constexpr int NQ = C / 16;
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1;
      mbar_wait(&B.x_full[b], (i >> 1) & 1);
      if (t == 0 && i == 0) TL3(2)
      const uint32_t xt = su + GG::O_X + b * XT;
      auto adr = [&](uint32_t base, int k) { const int qq = h * NQ + k; return base + (qq >> 3) * XB + sw128(r, qq & 7); };
      uint4 v[NQ];
      float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll
      for (int k = 0; k < NQ; ++k) {
        v[k] = lds128(adr(xt, k));
        a0 = __fadd2_rn(a0, up(v[k].x)); a1 = __fadd2_rn(a1, up(v[k].y)); a2 = __fadd2_rn(a2, up(v[k].z)); a3 = __fadd2_rn(a3, up(v[k].w));
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      float s = __fadd_rn(a0.x, a0.y);
      s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
      const float mu = __fmul_rn(s, 1.f / C);
      const float2 nm = make_float2(-mu, -mu);
      a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll
      for (int k = 0; k < NQ; ++k) {
        float2 dd;
        dd = __fadd2_rn(up(v[k].x), nm); a0 = __ffma2_rn(dd, dd, a0);
        dd = __fadd2_rn(up(v[k].y), nm); a1 = __ffma2_rn(dd, dd, a1);
        dd = __fadd2_rn(up(v[k].z), nm); a2 = __ffma2_rn(dd, dd, a2);
        dd = __fadd2_rn(up(v[k].w), nm); a3 = __ffma2_rn(dd, dd, a3);
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      float ss = __fadd_rn(a0.x, a0.y);
      ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));
      const float rs = rsqrt_ftz(__fadd_rn(__fmul_rn(ss, 1.f / C), eps));
      const float2 rr = make_float2(rs, rs);
      if (i > 0) mbar_wait(&B.a_free, (i - 1) & 1);   // tile i - 1's products have read xn
#pragma unroll
      for (int k = 0; k < NQ; ++k) {
        const int col = (h * NQ + k) * 8;
        const float4 ga = lds128f(O_GI + col * 4), gb4 = lds128f(O_GI + col * 4 + 16);
        const float4 ba = lds128f(O_BI + col * 4), bb = lds128f(O_BI + col * 4 + 16);
        auto aff2 = [&](uint32_t w, float g0, float g1, float b0, float b1) {
          const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w), nm), rr), make_float2(g0, g1), make_float2(b0, b1));
          return pack_bf16(y.x, y.y);
        };
        sts128(adr(su + GG::O_XN, k), aff2(v[k].x, ga.x, ga.y, ba.x, ba.y), aff2(v[k].y, ga.z, ga.w, ba.z, ba.w),
               aff2(v[k].z, gb4.x, gb4.y, bb.x, bb.y), aff2(v[k].w, gb4.z, gb4.w, bb.z, bb.w));
      }
      fence_async_smem();
      named_bar_sync(1, 128);
      if (t == 0) { mbar_arrive(&B.xn_ready); if (i == 0) TL3(3) }
      if (lat) epi_tile(i, 1, 2);
    }
  } else if (warp >= 8 && warp < 12) {
    // ------------------------------------------------------------------ LN_out (k3g's one-pass pivot statistics): warp q = tokens 16 q ..
    const int q = warp & 3, t4 = lane & 3, tr = lane >> 2;
    const int tok0 = q * 16, tc = tok0 >> 3;
    const uint32_t base = su + GG::O_T;
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    auto ldj = [&](int j, uint32_t (&w)[4]) {
      const int mm = lane >> 3, ch = 16 * j + 8 * (mm & 1) + (lane & 7), qc = tc + (mm >> 1);
      ldsm_x4_trans(base + sw128(ch, qc), w[0], w[1], w[2], w[3]);
    };
    for (int i = 0; i < n_local; ++i) {
      mbar_wait(&B.t_full, i & 1);
      if (i == 0 && lane == 0 && q == 0) TL3(5)
      float mean[2], rsd[2];
      {
        uint32_t w0[4]; ldj(0, w0);
        const float kA = __shfl_sync(0xffffffffu, up(w0[0]).x, lane & ~3), kB = __shfl_sync(0xffffffffu, up(w0[2]).x, lane & ~3);
        const float2 nkA = make_float2(-kA, -kA), nkB = make_float2(-kB, -kB);
        float2 sA = make_float2(0.f, 0.f), sA2 = sA, sB = sA, sB2 = sA, qA = sA, qA2 = sA, qB = sA, qB2 = sA;
#pragma unroll 4
        for (int j = 0; j < H / 16; ++j) {
          uint32_t w[4]; ldj(j, w);
          float2 d;
          d = __fadd2_rn(up(w[0]), nkA); sA = __fadd2_rn(sA, d); qA = __ffma2_rn(d, d, qA);
          d = __fadd2_rn(up(w[1]), nkA); sA2 = __fadd2_rn(sA2, d); qA2 = __ffma2_rn(d, d, qA2);
          d = __fadd2_rn(up(w[2]), nkB); sB = __fadd2_rn(sB, d); qB = __ffma2_rn(d, d, qB);
          d = __fadd2_rn(up(w[3]), nkB); sB2 = __fadd2_rn(sB2, d); qB2 = __ffma2_rn(d, d, qB2);
        }
        float tA = __fadd_rn(__fadd_rn(sA.x, sA.y), __fadd_rn(sA2.x, sA2.y)), tB = __fadd_rn(__fadd_rn(sB.x, sB.y), __fadd_rn(sB2.x, sB2.y));
        float uA = __fadd_rn(__fadd_rn(qA.x, qA.y), __fadd_rn(qA2.x, qA2.y)), uB = __fadd_rn(__fadd_rn(qB.x, qB.y), __fadd_rn(qB2.x, qB2.y));
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
          tA = __fadd_rn(tA, __shfl_xor_sync(0xffffffffu, tA, o)); tB = __fadd_rn(tB, __shfl_xor_sync(0xffffffffu, tB, o));
          uA = __fadd_rn(uA, __shfl_xor_sync(0xffffffffu, uA, o)); uB = __fadd_rn(uB, __shfl_xor_sync(0xffffffffu, uB, o));
        }
        const float mA = __fmul_rn(tA, 1.f / H), mB = __fmul_rn(tB, 1.f / H);
        mean[0] = __fadd_rn(kA, mA); mean[1] = __fadd_rn(kB, mB);
        const float vA = fmaxf(__fmaf_rn(-mA, mA, __fmul_rn(uA, 1.f / H)), 0.f), vB = fmaxf(__fmaf_rn(-mB, mB, __fmul_rn(uB, 1.f / H)), 0.f);
        rsd[0] = rsqrt_ftz(__fadd_rn(vA, eps)); rsd[1] = rsqrt_ftz(__fadd_rn(vB, eps));
      }
      if (i > 0) mbar_wait(&B.a_free, (i - 1) & 1);   // tile i - 1's products have read xo
#pragma unroll 2
      for (int j = 0; j < H / 16; ++j) {
        uint32_t w[4], rr[4];
        ldj(j, w);
#pragma unroll
        for (int e = 0; e < 4; ++e) {   // e: channel half 8 (e & 1), token half 8 (e >> 1)
          const int ch = 16 * j + 8 * (e & 1) + 2 * t4, tb = e >> 1;
          const float2 gg = lds64f(O_GO + ch * 4), bb = lds64f(O_BO + ch * 4);
          const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w[e]), make_float2(-mean[tb], -mean[tb])), make_float2(rsd[tb], rsd[tb])), gg, bb);
          rr[e] = pack_bf16(y.x, y.y);
        }
        // stmatrix (rows = tokens, 8 channels each): matrix mm = lane / 8 -> token half mm >> 1, channel half mm & 1
        const int mm = lane >> 3, tok = tok0 + (lane & 7) + 8 * (mm >> 1), c16 = 2 * j + (mm & 1);
        stsm_x4(su + GG::O_XO + (c16 >> 3) * XB + sw128(tok, c16 & 7), rr[0], rr[1], rr[2], rr[3]);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.t_empty);
      fence_async_smem();
      named_bar_sync(2, 128);
      if (warp == 8 && lane == 0) { mbar_arrive(&B.xo_ready); if (i == 0) TL3(6) }
      epi_tile(i, 0, lat ? 2 : 1);
    }
    if (warp == 8 && lane == 0) { TL3(11) bulk_wait<0>(); TL3(12) }
  }
  tc_fence_before();
  __syncthreads();
  if (tid == 0) TL3(13)
  if (tid == 0) PHM(2)
  if (warp == 2) tmem_dealloc(tmem, GG::TMC);
}

int num_sms() {
  static int n = 0;
  if (!n) { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
  return n;
}

}  // namespace k3h

// x [M, C] bf16, tri [H, (B,) L, L] bf16 (channel-major), wp [C, H] / wg [C, C] bf16 row-major ([out, in], natural order), LN affines
// fp32, y [M, C]. Inference only (no dropout). grid 0: min(tiles, 2 SMs). pdl: launched as a programmatic dependent.
void k3h_forward(torch::Tensor x, torch::Tensor tri, torch::Tensor wp, torch::Tensor wg, torch::Tensor g_in, torch::Tensor b_in,
                 torch::Tensor g_out, torch::Tensor b_out, torch::Tensor y, double eps, int64_t grid, int64_t pdl) {
  using namespace k3h;
  const int64_t M = x.size(0), C = x.size(1), H = tri.size(0);
  TORCH_CHECK(tri.numel() == H * M && M % TOK == 0 && x.is_contiguous() && tri.is_contiguous() && y.is_contiguous());
  TORCH_CHECK(wp.size(0) == C && wp.size(1) == H && wp.is_contiguous() && wg.size(0) == C && wg.size(1) == C && wg.is_contiguous());
  auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto mx = tmap::make(x.data_ptr(), bf, {(uint64_t)C, (uint64_t)M}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mt = tmap::make(tri.data_ptr(), bf, {(uint64_t)M, (uint64_t)H}, {(uint64_t)M * 2}, {64, (uint32_t)H}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwp = tmap::make(wp.data_ptr(), bf, {(uint64_t)H, (uint64_t)C}, {(uint64_t)H * 2}, {64, (uint32_t)C}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwg = tmap::make(wg.data_ptr(), bf, {(uint64_t)C, (uint64_t)C}, {(uint64_t)C * 2}, {64, (uint32_t)C}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto my = tmap::make(y.data_ptr(), bf, {(uint64_t)C, (uint64_t)M}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  const int tiles = (int)(M / TOK);
  int g = grid > 0 ? (int)grid : 2 * num_sms();
  g = std::min(g, tiles);
  auto st = at::cuda::getCurrentCUDAStream();
  const float *gi = g_in.data_ptr<float>(), *bi = b_in.data_ptr<float>(), *go = g_out.data_ptr<float>(), *bo = b_out.data_ptr<float>();
#define K3H_CASE(CC, HH)                                                                                                          \
  if (C == CC && H == HH) {                                                                                                       \
    static bool attr = false;                                                                                                     \
    if (!attr) { cudaFuncSetAttribute(k3h_kernel<CC, HH>, cudaFuncAttributeMaxDynamicSharedMemorySize, Geo<CC, HH>::SMEM); attr = true; } \
    tmap::launch_pdl(k3h_kernel<CC, HH>, dim3(g), dim3(NTHREADS), Geo<CC, HH>::SMEM, st, pdl != 0, mx, mt, mwp, mwg, my, gi, bi, go, bo, \
                     tiles, (float)eps);                                                                                          \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                                               \
    return;                                                                                                                       \
  }
  K3H_CASE(64, 64) K3H_CASE(64, 128)
#undef K3H_CASE
  TORCH_CHECK(false, "k3h_forward: unsupported (C, H) = (", C, ", ", H, ")");
}

torch::Tensor ph_k3h() {
#ifdef PH_TL
  torch::Tensor h = torch::empty({320, 3}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_ph, sizeof(g_ph));
  return h;
#else
  return torch::empty({0});
#endif
}

torch::Tensor k3h_tl() {
#ifdef TM_TL
  torch::Tensor h = torch::empty({320, 16}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_tl3h, sizeof(unsigned long long) * 320 * 16);
  return h;
#else
  return torch::empty({0});
#endif
}

// ======================================================================================================================== k3hf
// k3hf (2026-11-18, round 3 pass 15): k3h with BOTH LayerNorms folded into the weights (k3w's form at M = 64):
//   p = rs_o (t^T Wp'^T - mu_o sp) + ep        Wp' = bf16(Wp g_o) [C, H], sp = rowsum(Wp'), ep = Wp b_o
//   g = rs_i (x Wg'^T  - mu_i sg) + eg         Wg' = bf16(Wg g_i) [C, C], sg = rowsum(Wg'), eg = Wg b_i
//   y = bf16(x + bf16(sigmoid(g) p))
// Why: in k3h the p MMA waits for LN_out (statistics pass + normalise pass + xo stores, ~1.0 us after t lands at L128) and each
// tile's LN_out + epilogue run serially on warps 8-11. Here the MMAs read raw x (K-major, as loaded) and raw t (the channel-major
// tile is the MN-major A operand as stored) the moment they land; warps 4-7 / 8-11 only compute the per-token statistics (into
// shared tables) and both groups run every tile's epilogue (half the channels each). Accumulators and t are double buffered
// (256 TMEM columns: 2 x [p | g]), so the next tile's MMAs overlap this tile's epilogue. Not the same bits as k3h (no bf16
// rounding of xn / xo; inside the accuracy standard, as k3w).
namespace k3hf {
#ifdef TM_TL
static __device__ unsigned long long g_tlf[480][16];
#define TLF(i_, k_) { if ((i_) == 10) g_tlf[blockIdx.x][(k_)] = clock64(); else if ((i_) == 11) g_tlf[blockIdx.x][8 + (k_)] = clock64(); }
#else
#define TLF(i_, k_) {}
#endif

constexpr int TOK = 64, XB = TOK * 64 * 2, NTHREADS = 384;

template <int C, int H>
struct Geo {
  static constexpr int WGB = C * 128, WG = (C / 64) * WGB, WP = (H / 64) * WGB;
  static constexpr int XT = (C / 64) * XB, TT = H * TOK * 2;
  static constexpr int O_WG = 0, O_WP = O_WG + WG, O_X = O_WP + WP, O_T = O_X + 2 * XT, O_VEC = O_T + 2 * TT,
                       O_STI = O_VEC + 16 * C, O_STO = O_STI + 2 * TOK * 8, O_BAR = O_STO + 2 * TOK * 8;
  static constexpr int SMEM = O_BAR + 256;
  static constexpr int CPS3 = (SMEM + 1024 <= 233472 / 3) ? 1 : 0;   // three CTAs per SM fit (one accumulator buffer: 128 TMEM columns a CTA)
  static_assert(4 * C <= 256, "two [p | g] accumulators in 256 TMEM columns");
};

struct Bars {
  uint64_t w_full, x_full[2], x_empty[2], t_full[2], t_empty[2], sti[2], sto[2], acc_full[2], acc_empty[2];
  uint32_t tmem;
};

template <int C, int H, int CPS = 2>
__global__ void __launch_bounds__(NTHREADS, CPS)
    k3hf_kernel(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mtri, const __grid_constant__ CUtensorMap mwp,
                const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap my, const float* __restrict__ vec, int tiles,
                float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  using GG = Geo<C, H>;
  constexpr int ACCB = CPS == 3 ? 1 : 2;   // [p | g] accumulator buffers: one with three CTAs an SM (128 TMEM columns each)
  constexpr int XT = GG::XT, TT = GG::TT;
  constexpr uint32_t IDG = idesc_bf16(64, C), IDP = idesc_bf16_mj(64, C, 1, 0);
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + GG::O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int n_local = tiles > cta ? (tiles - cta + G - 1) / G : 0;
  const int nx0 = min(n_local, 2);
  auto row0_of = [&](int i) { return (cta + i * G) * TOK; };
  auto xload = [&](int b, int i) {
    mbar_expect_tx(&B.x_full[b], XT);
    for (int kb = 0; kb < C / 64; ++kb) tma_load_2d(sm + GG::O_X + b * XT + kb * XB, &mx, &B.x_full[b], kb * 64, row0_of(i), EVICT_FIRST);
  };
  pdl_trigger();
  if (tid == 0) {
    mbar_init(&B.w_full, 1);
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.x_full[b], 1); mbar_init(&B.x_empty[b], 1); mbar_init(&B.t_full[b], 1); mbar_init(&B.t_empty[b], 5);
      mbar_init(&B.sti[b], 1); mbar_init(&B.sto[b], 1); mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 8);
    }
    fence_mbar_init();
    prefetch_tmap(&mtri); prefetch_tmap(&my);
    // x and the (folded) weights are not written by the contraction this kernel follows: loaded before griddepcontrol.wait
    mbar_expect_tx(&B.w_full, GG::WG + GG::WP);
    for (int kb = 0; kb < C / 64; ++kb) tma_load_2d(sm + GG::O_WG + kb * GG::WGB, &mwg, &B.w_full, kb * 64, 0, EVICT_LAST);
    for (int kb = 0; kb < H / 64; ++kb) tma_load_2d(sm + GG::O_WP + kb * GG::WGB, &mwp, &B.w_full, kb * 64, 0, EVICT_LAST);
    for (int b = 0; b < nx0; ++b) xload(b, b);
  }
  for (int k = tid; k < 4 * C; k += NTHREADS) reinterpret_cast<float*>(sm + GG::O_VEC)[k] = vec[k];   // sp, ep, sg, eg
  if (warp == 2) { tmem_alloc(&B.tmem, ACCB * 128); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  const uint32_t O_SP = su + GG::O_VEC, O_EP = O_SP + 4 * C, O_SG = O_EP + 4 * C, O_EG = O_SG + 4 * C;
  float* sti = reinterpret_cast<float*>(sm + GG::O_STI);   // [2][2][TOK]: mu_i, rs_i
  float* sto = reinterpret_cast<float*>(sm + GG::O_STO);   // [2][2][TOK]: mu_o, rs_o

  if (warp == 0 && lane == 0) {
    for (int i = nx0; i < n_local; ++i) {
      const int b = i & 1;
      mbar_wait(&B.x_empty[b], ((i >> 1) - 1) & 1);
      xload(b, i);
#ifdef K3HF_PFX   // measured level at D64 L512-1536 (27.78 / 93.79 vs 27.85 / 94.19 us): off
      if (i + 1 < n_local)   // L2 prefetch of the next tile's x boxes (its slot frees about one tile period later)
        for (int kb = 0; kb < C / 64; ++kb)
          asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"(reinterpret_cast<uint64_t>(&mx)), "r"(kb * 64), "r"(row0_of(i + 1)) : "memory");
#endif
    }
  } else if (warp == 2 && lane == 0) {
    pdl_wait();   // t is the previous kernel's output
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1;
      if (i >= 2) mbar_wait(&B.t_empty[b], ((i >> 1) - 1) & 1);
      mbar_expect_tx(&B.t_full[b], TT);
      tma_load_2d(sm + GG::O_T + b * TT, &mtri, &B.t_full[b], row0_of(i), 0, EVICT_FIRST);
    }
  } else if (warp == 1 && lane == 0) {
    mbar_wait(&B.w_full, 0);
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1;
      const uint32_t ph = (i >> 1) & 1, ab = ACCB == 2 ? b : 0, aph = ACCB == 2 ? ph : (i & 1), d = tmem + ab * (2 * C);
      if (i >= ACCB) mbar_wait(&B.acc_empty[ab], aph ^ 1);
      mbar_wait(&B.x_full[b], ph);
      tc_fence_after();
#pragma unroll
      for (int k = 0; k < C / 16; ++k)
        umma_ss(d + C, desc_k_sw128(su + GG::O_X + b * XT + (k >> 2) * XB + (k & 3) * 32),
                desc_k_sw128(su + GG::O_WG + (k >> 2) * GG::WGB + (k & 3) * 32), IDG, k != 0);
      mbar_wait(&B.t_full[b], ph); TLF(i, 6)
      tc_fence_after();
#pragma unroll
      for (int k = 0; k < H / 16; ++k)
        umma_ss(d, desc_mn_sw128(su + GG::O_T + b * TT + k * 2048, 8192), desc_k_sw128(su + GG::O_WP + (k >> 2) * GG::WGB + (k & 3) * 32),
                IDP, k != 0);
      umma_commit(&B.t_empty[b]);
      umma_commit(&B.acc_full[ab]); TLF(i, 7)
    }
  }

  // epilogue of tile i, half `part` of the channels (warps 8-11: part 0, warps 4-7: part 1), right after the group's statistics
  auto epi_tile = [&](int i, int part) {
    const int q = warp & 3, tr = lane >> 2, t4 = lane & 3, b = i & 1;
    const uint32_t ph = (i >> 1) & 1;
    constexpr int CW = 32, nq = C / CW / 2;
    const int cq0 = part * nq;
    mbar_wait(&B.sti[b], ph);
    mbar_wait(&B.sto[b], ph);
    if (warp == 8 && lane == 0) TLF(i, 5)
    float mo[2], ro[2], mi[2], ri[2];
#pragma unroll
    for (int tb = 0; tb < 2; ++tb) {
      const int r = 16 * q + tr + 8 * tb;
      mi[tb] = sti[b * 2 * TOK + r]; ri[tb] = sti[b * 2 * TOK + TOK + r];
      mo[tb] = sto[b * 2 * TOK + r]; ro[tb] = sto[b * 2 * TOK + TOK + r];
    }
    const int ab = ACCB == 2 ? b : 0; mbar_wait(&B.acc_full[ab], ACCB == 2 ? ph : (i & 1));
    if (warp == 8 && lane == 0) TLF(i, 3)
    mbar_wait(&B.x_full[b], ph);
    tc_fence_after();
    if (warp == 8 && lane == 0) TLF(i, 2)
    const uint32_t xr = su + GG::O_X + b * XT;
    for (int cq = cq0; cq < cq0 + nq; ++cq) {
      uint32_t pv[16], gv[16];
      const uint32_t ta = tmem + ab * (2 * C) + ((uint32_t)(q * 32) << 16) + cq * CW;
      k3h::tmem_ld16x256_x4(ta, pv);
      k3h::tmem_ld16x256_x4(ta + C, gv);
      tmem_wait_ld();
      if (cq == cq0 + nq - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty[ab]); }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int ch = cq * CW + 8 * j + 2 * t4;
        const float2 sp = k3h::lds64f(O_SP + ch * 4), ep = k3h::lds64f(O_EP + ch * 4), sg = k3h::lds64f(O_SG + ch * 4), eg = k3h::lds64f(O_EG + ch * 4);
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) {
          const int r = 16 * q + tr + 8 * tb;
          const uint32_t a = xr + (ch >> 6) * XB + k3h::sw128(r, (ch & 63) >> 3) + (ch & 7) * 2;
          const uint32_t xw = k3h::lds32(a);
          const float2 pa = make_float2(__uint_as_float(pv[4 * j + 2 * tb]), __uint_as_float(pv[4 * j + 2 * tb + 1]));
          const float2 ga = make_float2(__uint_as_float(gv[4 * j + 2 * tb]), __uint_as_float(gv[4 * j + 2 * tb + 1]));
          const float2 pp = __ffma2_rn(make_float2(ro[tb], ro[tb]), __ffma2_rn(make_float2(-mo[tb], -mo[tb]), sp, pa), ep);
          const float2 gg = __ffma2_rn(make_float2(ri[tb], ri[tb]), __ffma2_rn(make_float2(-mi[tb], -mi[tb]), sg, ga), eg);
          const float2 h = __fmul2_rn(gg, make_float2(0.5f, 0.5f));   // sigmoid(g) = 0.5 tanh(g / 2) + 0.5
          float2 th;
          asm("tanh.approx.f32 %0, %1;" : "=f"(th.x) : "f"(h.x));
          asm("tanh.approx.f32 %0, %1;" : "=f"(th.y) : "f"(h.y));
          const float2 sgm = __ffma2_rn(th, make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
          {   // y = bf16(x + bf16(sigmoid(g) p)) as one packed bf16 add of the two bf16 words (the fp32 sum of two bf16 values rounded once)
            const float2 pr = __fmul2_rn(sgm, pp);
            uint32_t yw;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(yw) : "r"(xw), "r"(pack_bf16(pr.x, pr.y)));
            k3h::sts32(a, yw);
          }
        }
      }
    }
    fence_async_smem();
    if (warp == 8 && lane == 0) TLF(i, 4)
    named_bar_sync(3, 256);
    if (warp == 8 && lane == 0) {
      for (int kb = 0; kb < C / 64; ++kb) tma_store_2d(&my, sm + GG::O_X + b * XT + kb * XB, kb * 64, row0_of(i));
      bulk_commit();
      bulk_wait_read<0>();
      mbar_arrive(&B.x_empty[b]);
    }
  };

  if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------ LN_in statistics: thread pair = row r, half h of the channels
    const int t = tid - 128, r = t >> 1, h = t & 1;
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    constexpr int NQ = C / 16;
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1;
      mbar_wait(&B.x_full[b], (i >> 1) & 1);
      const uint32_t xt = su + GG::O_X + b * XT;
      auto adr = [&](int k) { const int qq = h * NQ + k; return xt + (qq >> 3) * XB + k3h::sw128(r, qq & 7); };
      uint4 v[NQ];
      float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll
      for (int k = 0; k < NQ; ++k) {
        v[k] = k3h::lds128(adr(k));
        a0 = __fadd2_rn(a0, up(v[k].x)); a1 = __fadd2_rn(a1, up(v[k].y)); a2 = __fadd2_rn(a2, up(v[k].z)); a3 = __fadd2_rn(a3, up(v[k].w));
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      float s = __fadd_rn(a0.x, a0.y);
      s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
      const float mu = __fmul_rn(s, 1.f / C);
      const float2 nm = make_float2(-mu, -mu);
      a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll
      for (int k = 0; k < NQ; ++k) {
        float2 dd;
        dd = __fadd2_rn(up(v[k].x), nm); a0 = __ffma2_rn(dd, dd, a0);
        dd = __fadd2_rn(up(v[k].y), nm); a1 = __ffma2_rn(dd, dd, a1);
        dd = __fadd2_rn(up(v[k].z), nm); a2 = __ffma2_rn(dd, dd, a2);
        dd = __fadd2_rn(up(v[k].w), nm); a3 = __ffma2_rn(dd, dd, a3);
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      float ss = __fadd_rn(a0.x, a0.y);
      ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));
      if (h == 0) { sti[b * 2 * TOK + r] = mu; sti[b * 2 * TOK + TOK + r] = k3h::rsqrt_ftz(__fadd_rn(__fmul_rn(ss, 1.f / C), eps)); }
      named_bar_sync(1, 128);
      if (t == 0) mbar_arrive(&B.sti[b]);
      epi_tile(i, 1);
    }
  } else if (warp >= 8 && warp < 12) {
    // ------------------------------------------------------------------ LN_out statistics (k3h's one-pass pivot form): warp q = tokens 16 q ..
    const int q = warp & 3, t4 = lane & 3, tr = lane >> 2;
    const int tok0 = q * 16, tc = tok0 >> 3;
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1;
      const uint32_t base = su + GG::O_T + b * TT;
      auto ldj = [&](int j, uint32_t (&w)[4]) {
        const int mm = lane >> 3, ch = 16 * j + 8 * (mm & 1) + (lane & 7), qc = tc + (mm >> 1);
        k3h::ldsm_x4_trans(base + k3h::sw128(ch, qc), w[0], w[1], w[2], w[3]);
      };
      mbar_wait(&B.t_full[b], (i >> 1) & 1);
      if (warp == 8 && lane == 0) TLF(i, 0)
      uint32_t w0[4]; ldj(0, w0);
      const float kA = __shfl_sync(0xffffffffu, up(w0[0]).x, lane & ~3), kB = __shfl_sync(0xffffffffu, up(w0[2]).x, lane & ~3);
      const float2 nkA = make_float2(-kA, -kA), nkB = make_float2(-kB, -kB);
      float2 sA = make_float2(0.f, 0.f), sA2 = sA, sB = sA, sB2 = sA, qA = sA, qA2 = sA, qB = sA, qB2 = sA;
#pragma unroll 4
      for (int j = 0; j < H / 16; ++j) {
        uint32_t w[4]; ldj(j, w);
        float2 d;
        d = __fadd2_rn(up(w[0]), nkA); sA = __fadd2_rn(sA, d); qA = __ffma2_rn(d, d, qA);
        d = __fadd2_rn(up(w[1]), nkA); sA2 = __fadd2_rn(sA2, d); qA2 = __ffma2_rn(d, d, qA2);
        d = __fadd2_rn(up(w[2]), nkB); sB = __fadd2_rn(sB, d); qB = __ffma2_rn(d, d, qB);
        d = __fadd2_rn(up(w[3]), nkB); sB2 = __fadd2_rn(sB2, d); qB2 = __ffma2_rn(d, d, qB2);
      }
      fence_async_smem();   // generic reads of t before the TMA that refills the buffer
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.t_empty[b]);
      float tA = __fadd_rn(__fadd_rn(sA.x, sA.y), __fadd_rn(sA2.x, sA2.y)), tB = __fadd_rn(__fadd_rn(sB.x, sB.y), __fadd_rn(sB2.x, sB2.y));
      float uA = __fadd_rn(__fadd_rn(qA.x, qA.y), __fadd_rn(qA2.x, qA2.y)), uB = __fadd_rn(__fadd_rn(qB.x, qB.y), __fadd_rn(qB2.x, qB2.y));
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) {
        tA = __fadd_rn(tA, __shfl_xor_sync(0xffffffffu, tA, o)); tB = __fadd_rn(tB, __shfl_xor_sync(0xffffffffu, tB, o));
        uA = __fadd_rn(uA, __shfl_xor_sync(0xffffffffu, uA, o)); uB = __fadd_rn(uB, __shfl_xor_sync(0xffffffffu, uB, o));
      }
      const float mA = __fmul_rn(tA, 1.f / H), mB = __fmul_rn(tB, 1.f / H);
      const float vA = fmaxf(__fmaf_rn(-mA, mA, __fmul_rn(uA, 1.f / H)), 0.f), vB = fmaxf(__fmaf_rn(-mB, mB, __fmul_rn(uB, 1.f / H)), 0.f);
      // this thread's statistics are tokens tok0 + tr (A) and tok0 + tr + 8 (B) (the epilogue's fragment rows)
      if (t4 == 0) {
        sto[b * 2 * TOK + tok0 + tr] = __fadd_rn(kA, mA); sto[b * 2 * TOK + TOK + tok0 + tr] = k3h::rsqrt_ftz(__fadd_rn(vA, eps));
        sto[b * 2 * TOK + tok0 + tr + 8] = __fadd_rn(kB, mB); sto[b * 2 * TOK + TOK + tok0 + tr + 8] = k3h::rsqrt_ftz(__fadd_rn(vB, eps));
      }
      named_bar_sync(2, 128);
      if (warp == 8 && lane == 0) TLF(i, 1)
      if (warp == 8 && lane == 0) mbar_arrive(&B.sto[b]);
      epi_tile(i, 0);
    }
    if (warp == 8 && lane == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, ACCB * 128);
}

}  // namespace k3hf

// x [M, C] bf16, tri [H, (B,) L, L] bf16, wpq = bf16(Wp g_o) [C, H], wgq = bf16(Wg g_i) [C, C], vec [4, C] fp32 (sp, ep, sg, eg:
// ``wide_fold_prep``), y [M, C]. Inference only (no dropout). grid 0: min(tiles, 2 SMs). pdl: launched as a programmatic dependent.
void k3hf_forward(torch::Tensor x, torch::Tensor tri, torch::Tensor wpq, torch::Tensor wgq, torch::Tensor vec, torch::Tensor y,
                  double eps, int64_t grid, int64_t pdl) {
  using namespace k3hf;
  const int64_t M = x.size(0), C = x.size(1), H = tri.size(0);
  TORCH_CHECK(tri.numel() == H * M && M % TOK == 0 && x.is_contiguous() && tri.is_contiguous() && y.is_contiguous());
  TORCH_CHECK(wpq.size(0) == C && wpq.size(1) == H && wpq.is_contiguous() && wgq.size(0) == C && wgq.size(1) == C && wgq.is_contiguous());
  TORCH_CHECK(vec.numel() == 4 * C && vec.is_contiguous() && vec.scalar_type() == at::kFloat);
  auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto mx = tmap::make(x.data_ptr(), bf, {(uint64_t)C, (uint64_t)M}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mt = tmap::make(tri.data_ptr(), bf, {(uint64_t)M, (uint64_t)H}, {(uint64_t)M * 2}, {64, (uint32_t)H}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwp = tmap::make(wpq.data_ptr(), bf, {(uint64_t)H, (uint64_t)C}, {(uint64_t)H * 2}, {64, (uint32_t)C}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwg = tmap::make(wgq.data_ptr(), bf, {(uint64_t)C, (uint64_t)C}, {(uint64_t)C * 2}, {64, (uint32_t)C}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto my = tmap::make(y.data_ptr(), bf, {(uint64_t)C, (uint64_t)M}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  const int tiles = (int)(M / TOK);
  auto st = at::cuda::getCurrentCUDAStream();
  const float* vp = vec.data_ptr<float>();
  const char* cps_env = std::getenv("TRIMUL_K3HF_CPS");   // 0: two CTAs an SM (A/B)
  const bool use3 = !(cps_env != nullptr && cps_env[0] == '0');
#define K3HF_GO(CC, HH, CP)                                                                                                       \
  {                                                                                                                               \
    int g = grid > 0 ? (int)grid : (CP) * k3h::num_sms();                                                                         \
    g = std::min(g, tiles);                                                                                                       \
    static bool attr = false;                                                                                                     \
    if (!attr) { cudaFuncSetAttribute(k3hf_kernel<CC, HH, CP>, cudaFuncAttributeMaxDynamicSharedMemorySize, Geo<CC, HH>::SMEM); attr = true; } \
    tmap::launch_pdl(k3hf_kernel<CC, HH, CP>, dim3(g), dim3(NTHREADS), Geo<CC, HH>::SMEM, st, pdl != 0, mx, mt, mwp, mwg, my, vp, tiles, (float)eps); \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                                               \
    return;                                                                                                                       \
  }
  if (C == 64 && H == 64) { if (use3 && Geo<64, 64>::CPS3) K3HF_GO(64, 64, (Geo<64, 64>::CPS3 ? 3 : 2)) else K3HF_GO(64, 64, 2) }
  if (C == 64 && H == 128) { K3HF_GO(64, 128, 2) }
#undef K3HF_GO
  TORCH_CHECK(false, "k3hf_forward: unsupported (C, H) = (", C, ", ", H, ")");
}

torch::Tensor k3hf_tl() {
#ifdef TM_TL
  torch::Tensor h = torch::empty({480, 16}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), k3hf::g_tlf, sizeof(unsigned long long) * 480 * 16);
  return h;
#else
  return torch::empty({0});
#endif
}
