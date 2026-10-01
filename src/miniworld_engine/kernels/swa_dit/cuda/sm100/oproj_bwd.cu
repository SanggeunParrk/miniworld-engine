// oproj_bwd.cu — the SWA atom block's out-projection backward on sm_100a (replaces the Triton _oproj_bwd; same math and rounding points):
//   q1 = q + gate_a ((sigmoid(g) o) Wo^T):   so = sigmoid(g);  gated = rn(so o);  d gate_a += sum_aug dq1 att;  datt = rn(dq1 rn(gate_a));
//   dgated = datt Wo (fp32);  dO = rn(dgated so);  dG = rn(dgated o so (1 - so));  D[n, h, s] = sum_{d in head h} dO o
// and writes datt, gated (the dWo = datt^T gated operands on cuBLAS), dO, dG, D.
// Structure (as qkvg_bwd.cu): persistent CTAs, 64-row tiles of SP augments x AT atoms, a ring of 16-KB tile slots with four uses per tile
// (dq1 -> datt, att -> dG, g -> gated, o -> dO: every output is written in place over an input the same thread consumed, then TMA-stored).
// Wo^T lives in TMEM as the bf16 A operand; dgated^T = Wo^T datt^T (M = 128 channels, N = 64 rows) takes datt straight from its slot.
// One channel per thread (the TMEM lane) and 32 rows per warpgroup: the augment sums of d gate_a are per-thread registers (AT = 8), and the
// per-head D is a warp sum (warp q = head q) done as a 32-lane reduce-scatter.
// TMEM: Wo^T (bf16) at 0 (64 columns), dgated^T[2] at 64 / 128.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include <type_traits>
using namespace s100;

constexpr int C = 128, H = 4, NSMAX = 12, UPT = 4;
constexpr int SLOT = 16384, KBLK = 8192;
enum { U_DQ1 = 0, U_ATT, U_G, U_O };
constexpr uint32_t T_W = 0, T_DG = 64;
constexpr uint32_t I_DG = idesc_bf16(128, 64);

struct Bars {
  uint64_t full[NSMAX], rdy[NSMAX], empty[NSMAX], auxfull[2], auxempty[2], dgfull[2], dgfree[2];
  uint32_t tmem;
};
struct RingPos {
  int b = 0; uint32_t p = 0;
  DEVI void get(int u, int ns, int& s, uint32_t& ph) const { s = b + u; ph = p; if (s >= ns) { s -= ns; ph ^= 1u; } }
  DEVI void next(int ns) { b += UPT; if (b >= ns) { b -= ns; p ^= 1u; } }
};
DEVI void mbar_arrive_cnt(uint64_t* b, uint32_t n) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(n) : "memory"); }
DEVI float lds_bf16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a) : "memory"); return __uint_as_float((uint32_t)v << 16); }
DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEVI void sts_bf16(uint32_t a, float x) {
  const unsigned short v = __bfloat16_as_ushort(__float2bfloat16_rn(x));
  asm volatile("st.shared.u16 [%0], %1;" :: "r"(a), "h"(v) : "memory");
}
DEVI void red1(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }
DEVI float sigm(float x) { return rcpf(1.f + ex2f(-1.4426950408889634f * x)); }
// v[j] held by every lane -> lane l returns the sum over the warp of v[l]
DEVI float rscatter32(float (&v)[32], int lane) {
#pragma unroll
  for (int i = 0; i < 16; ++i) { const bool up = lane & 16; const float k = up ? v[i + 16] : v[i], s = up ? v[i] : v[i + 16]; v[i] = k + __shfl_xor_sync(~0u, s, 16); }
#pragma unroll
  for (int i = 0; i < 8; ++i) { const bool up = lane & 8; const float k = up ? v[i + 8] : v[i], s = up ? v[i] : v[i + 8]; v[i] = k + __shfl_xor_sync(~0u, s, 8); }
#pragma unroll
  for (int i = 0; i < 4; ++i) { const bool up = lane & 4; const float k = up ? v[i + 4] : v[i], s = up ? v[i] : v[i + 4]; v[i] = k + __shfl_xor_sync(~0u, s, 4); }
#pragma unroll
  for (int i = 0; i < 2; ++i) { const bool up = lane & 2; const float k = up ? v[i + 2] : v[i], s = up ? v[i] : v[i + 2]; v[i] = k + __shfl_xor_sync(~0u, s, 2); }
  { const bool up = lane & 1; const float k = up ? v[1] : v[0], s = up ? v[0] : v[1]; v[0] = k + __shfl_xor_sync(~0u, s, 1); }
  return v[0];
}

extern "C" __global__ void __launch_bounds__(384, 1)
swa_oproj_bwd_sm100(const __grid_constant__ CUtensorMap mdq1, const __grid_constant__ CUtensorMap matt, const __grid_constant__ CUtensorMap mg,
                    const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mmod, const __grid_constant__ CUtensorMap mdatt,
                    const __grid_constant__ CUtensorMap mdg, const __grid_constant__ CUtensorMap mgated, const __grid_constant__ CUtensorMap mdo,
                    int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, int NS,
                    const __nv_bfloat16* __restrict__ WOT, float* __restrict__ DMOD, float* __restrict__ DV) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int AUXST = (AT * 512 + 1023) / 1024 * 1024;
  const int O_AUX = NS * SLOT, O_BAR = O_AUX + 2 * AUXST;
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto coords = [&](int T, int& b, int& a0, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    const int ab = t % nab, r = t / nab, ag = r % nag;
    b = r / nag; a0 = ag * SP; s0 = ab * AT;
  };

  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mbar_init(&B.full[i], 1); mbar_init(&B.rdy[i], 2); mbar_init(&B.empty[i], 2); }
    for (int i = 0; i < 2; ++i) { mbar_init(&B.auxfull[i], 1); mbar_init(&B.auxempty[i], 2); mbar_init(&B.dgfull[i], 1); mbar_init(&B.dgfree[i], 2); }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (warp >= 4 && warp < 8) {                                           // Wo^T -> TMEM (A operand): lane c gets row c of Wo^T
    const int lb = (warp & 3) * 32, c = lb + lane;
    const __nv_bfloat16* src = WOT + (size_t)c * C;
#pragma unroll 1
    for (int k = 0; k < 4; ++k) {
      uint32_t v[16];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const uint4 u = ldg128(src + 32 * k + 8 * e); v[4 * e] = u.x; v[4 * e + 1] = u.y; v[4 * e + 2] = u.z; v[4 * e + 3] = u.w; }
      tmem_st16(tmem + ((uint32_t)lb << 16) + T_W + 16 * k, v);
    }
    tmem_wait_st();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      const uint32_t tx = (uint32_t)(SP * AT * 256), auxtx = (uint32_t)(AT * 512);
      RingPos rp; int gtot = 0;
      for (int T = 0; T < ntT; ++T, rp.next(NS)) {
        int b, a0, s0; coords(T, b, a0, s0);
        const int xa = T & 1;
        if (T >= 2) mbar_wait(&B.auxempty[xa], ((T >> 1) - 1) & 1);
        mbar_expect_tx(&B.auxfull[xa], auxtx);
        tma_load_3d(su + O_AUX + xa * AUXST, &mmod, &B.auxfull[xa], 0, b * S + s0, 8);   // gate_a: blocks 8..11
        for (int u = 0; u < UPT; ++u, ++gtot) {
          int sl; uint32_t ph; rp.get(u, NS, sl, ph);
          if (gtot >= NS) mbar_wait(&B.empty[sl], ph ^ 1);
          const uint32_t d = su + sl * SLOT;
          const CUtensorMap* m = u == U_DQ1 ? &mdq1 : u == U_ATT ? &matt : u == U_G ? &mg : &mo;
          mbar_expect_tx(&B.full[sl], tx);
          for (int kb = 0; kb < 2; ++kb) tma_load_4d(d + kb * KBLK, m, &B.full[sl], kb * 64, s0, b, a0);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer: dgated^T = Wo^T datt^T
    RingPos rp;
    for (int T = 0; T < ntT; ++T, rp.next(NS)) {
      const int xb = T & 1;
      if (T >= 2) mbar_wait(&B.dgfree[xb], ((T >> 1) - 1) & 1);
      int sl; uint32_t ph; rp.get(U_DQ1, NS, sl, ph);
      mbar_wait(&B.rdy[sl], ph);
      tc_fence_after();
      const uint32_t sa = su + sl * SLOT;
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
          umma_ts(tmem + T_DG + xb * 64, tmem + T_W + ks * 8, desc_k128(sa + (ks >> 2) * KBLK) + (uint64_t)((ks & 3) * 2), I_DG, ks > 0 ? 1u : 0u);
        tc_commit(&B.empty[sl]);
        tc_commit(&B.dgfull[xb]);
      }
      __syncwarp();
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA stores: datt, dG, gated, dO
    if (lane == 0) {
      RingPos rp;
      for (int T = 0; T < ntT; ++T, rp.next(NS)) {
        int b, a0, s0; coords(T, b, a0, s0);
        for (int u = 0; u < UPT; ++u) {
          int sl; uint32_t ph; rp.get(u, NS, sl, ph);
          mbar_wait(&B.rdy[sl], ph);
          const CUtensorMap* m = u == U_DQ1 ? &mdatt : u == U_ATT ? &mdg : u == U_G ? &mgated : &mdo;
          for (int kb = 0; kb < 2; ++kb) tma_store_4d(m, su + sl * SLOT + kb * KBLK, kb * 64, s0, b, a0);
          tma_store_commit();
          tma_store_wait_read0();
          mbar_arrive_cnt(&B.empty[sl], u == U_DQ1 ? 1 : 2);
        }
      }
      tma_store_wait0();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ channel threads: channel c, rows 32 wg ..
    const int wg = (warp - 4) >> 2, qw = warp & 3;
    const uint32_t lb = (uint32_t)qw * 32;
    const int c = (int)lb + lane;
    const uint32_t cofs = (uint32_t)(c >> 6) * KBLK + (uint32_t)(c & 7) * 2, cch = (uint32_t)((c & 63) >> 3);
    uint32_t xo[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) xo[k] = (cch ^ (uint32_t)k) << 4;
    RingPos rp;
    auto tile = [&](int T, auto at8c) {
      constexpr bool AT8 = decltype(at8c)::value;
      int b, a0, s0; coords(T, b, a0, s0);
      const int xb = T & 1;
      int sD, sA, sG, sO; uint32_t pD, pA, pG, pO;
      rp.get(U_DQ1, NS, sD, pD); rp.get(U_ATT, NS, sA, pA); rp.get(U_G, NS, sG, pG); rp.get(U_O, NS, sO, pO);
      const uint32_t rb = (uint32_t)(32 * wg) * 128u;
      const uint32_t aD = su + sD * SLOT + cofs + rb, aA = su + sA * SLOT + cofs + rb, aG = su + sG * SLOT + cofs + rb, aO = su + sO * SLOT + cofs + rb;
      const uint32_t ax = O_AUX + (T & 1) * AUXST;
      auto gat = [&](int at) {                                             // gate_a[b, s0 + at][c]: block c / 32 of the aux tile
        const int row = (c >> 5) * AT + at;
        return *reinterpret_cast<const float*>(sm + ax + row * 128 + ((((c & 31) >> 2) ^ (row & 7)) << 4) + (c & 3) * 4);
      };
      auto rowinfo = [&](int j, int& at, bool& ok, long long& grow) {     // row 32 wg + j of the tile
        const int row = 32 * wg + j;
        const int sp = AT8 ? row >> 3 : row / AT;
        at = AT8 ? (j & 7) : row - sp * AT;
        ok = row < SP * AT && a0 + sp < A && s0 + at < S;
        grow = ((long long)(a0 + sp) * Bn + b) * S + s0 + at;
      };
      // ---- phase A: d gate_a sums and datt = rn(dq1 rn(gate_a)) in place over dq1
      mbar_wait(&B.auxfull[T & 1], (T >> 1) & 1);
      mbar_wait(&B.full[sD], pD);
      mbar_wait(&B.full[sA], pA);
      float gsum[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) gsum[k] = 0.f;
      if constexpr (AT8) {
        float ga[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) ga[k] = rnb(gat(k));
        uint32_t sok = 0, aok = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) sok |= (s0 + k < S) ? 1u << k : 0u;
#pragma unroll
        for (int k = 0; k < 4; ++k) aok |= (a0 + 4 * wg + k < A) ? 1u << k : 0u;
        float d1_[32], at__[32];                                           // all shared-memory loads first
#pragma unroll
        for (int j = 0; j < 32; ++j) { d1_[j] = lds_bf16(aD + j * 128 + xo[j & 7]); at__[j] = lds_bf16(aA + j * 128 + xo[j & 7]); }
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          const int k = j & 7;
          const float d1 = d1_[j], at_ = at__[j];
          const bool ok = ((aok >> (j >> 3)) & (sok >> k) & 1u) != 0;
          gsum[k] += ok ? d1 * at_ : 0.f;
          sts_bf16(aD + j * 128 + xo[k], d1 * ga[k]);
        }
      } else {
#pragma unroll 1
        for (int j = 0; j < 32; ++j) {
          int at; bool ok; long long g; rowinfo(j, at, ok, g);
          const uint32_t xr = (cch ^ (uint32_t)(j & 7)) << 4;
          const float d1 = lds_bf16(aD + j * 128 + xr), at_ = lds_bf16(aA + j * 128 + xr);
          if (ok) red1(DMOD + ((size_t)b * S + s0 + at) * (6 * C) + 2 * C + c, d1 * at_);
          sts_bf16(aD + j * 128 + xr, d1 * rnb(gat(at)));
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) mbar_arrive(&B.rdy[sD]);
      if constexpr (AT8) {
#pragma unroll
        for (int k = 0; k < 8; ++k)
          if (s0 + k < S) red1(DMOD + ((size_t)b * S + s0 + k) * (6 * C) + 2 * C + c, gsum[k]);
      }
      // ---- phase C: gated, dO, dG in place (over g, o, att); D per (row, head) by warp reduce-scatter
      mbar_wait(&B.full[sG], pG);
      mbar_wait(&B.full[sO], pO);
      mbar_wait(&B.dgfull[xb], (T >> 1) & 1);
      tc_fence_after();
      uint32_t dv[32];
      tmem_ld32(tmem + (lb << 16) + T_DG + xb * 64 + 32 * wg, dv);
      float g_[32], o_[32];                                                // all shared-memory loads first (under the TMEM load)
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const uint32_t xr = AT8 ? xo[j & 7] : ((cch ^ (uint32_t)(j & 7)) << 4);
        g_[j] = lds_bf16(aG + j * 128 + xr); o_[j] = lds_bf16(aO + j * 128 + xr);
      }
      tmem_wait_ld();
      tc_fence_before();
      float v[32];
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const uint32_t xr = AT8 ? xo[j & 7] : ((cch ^ (uint32_t)(j & 7)) << 4);
        const float g = g_[j], o = o_[j], dg = __uint_as_float(dv[j]);
        const float so = sigm(g);
        const float dob = rnb(dg * so);
        sts_bf16(aG + j * 128 + xr, so * o);
        sts_bf16(aO + j * 128 + xr, dob);
        sts_bf16(aA + j * 128 + xr, dg * o * so * (1.f - so));
        v[j] = dob * o;
      }
      const float dsum = rscatter32(v, lane);                              // row 32 wg + lane, head qw
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) {
        mbar_arrive(&B.rdy[sA]); mbar_arrive(&B.rdy[sG]); mbar_arrive(&B.rdy[sO]);
        mbar_arrive(&B.dgfree[xb]); mbar_arrive(&B.auxempty[T & 1]);
      }
      {
        int at; bool ok; long long g; rowinfo(lane, at, ok, g);
        if (ok) {
          const long long n = g / S, s = g - n * S;
          DV[(n * H + qw) * S + s] = dsum;
        }
      }
    };
    for (int T = 0; T < ntT; ++T, rp.next(NS)) {
      if (AT == 8) tile(T, std::true_type{}); else tile(T, std::false_type{});
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 256); }
}
