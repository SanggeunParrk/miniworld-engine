// qkvg_bwd.cu — the SWA atom block's first-stage backward on sm_100a (replaces the Triton _qkvg_bwd; same math and rounding points):
//   dpq = rn(headRMS_bwd(pq, unrope(dQ)));  dpk likewise;  dP = [dpq | dpk | dV | dG]  (bf16, [M, 512], for dWqkv / dWg on cuBLAS)
//   dx = dP [Wq; Wk; Wv; Wg]  (fp32);  xh = q rstd;  dxh = dx (1 + scale_a);  dq = rn(dq1 + rstd (dxh - xh mean(dxh xh)))
//   d shift_a += sum_aug dx,  d scale_a += sum_aug dx xh   (pre-summed over the tile's augments, then red.add)
// Structure: persistent CTAs, tiles of SP augments x AT atoms (64 rows: SP = min(A, 8)); small tiles keep the ring deep (12 slots). The weights never touch shared memory: W^T [128 in][512 out]
// is loaded once into TMEM as the bf16 A operand, and dx^T = W^T dP^T (M = 128 channels, N = 64 rows, K = 512) takes dP straight
// from the input tiles in shared memory (B operand, K-major): dV (head-major, SW64) and dG as loaded, dpq / dpk written in place over
// pq / pk by the row threads. So all of shared memory is a ring of 16-KB tile slots, eight uses per tile in the order
//   dV, dG, pq(->dpq), dQ, pk(->dpk), dK, q, dq1(->dq);
// dV / dG are copied to dP by TMA stores from their slots (no thread touches them). Row threads: one (row, head) per thread in the
// q / k phases (warpgroup w: heads 2w, 2w+1) and rows 32 w .. 32 w + 31 in the final phase, where a thread owns one channel (its TMEM lane): the
// per-row sums (q^2, dx (1 + scale) q) go through a warp reduce-scatter plus a 4-warp smem exchange, and the augment sums of the
// modulation gradient are per-thread register accumulators (AT == 8) or per-row red.add (other AT).
// TMEM: W^T (bf16) at 0 (256 columns), dx^T[2] at 256 / 320.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include <type_traits>
using namespace s100;

constexpr int C = 128, D = 32, NSMAX = 12, UPT = 8;
constexpr int SLOT = 16384, HB = 4096, KBLK = 8192;                  // [64 rows][128 ch] tile: 2 x [64][64] SW128 / 4 heads x [64][32] SW64
enum { U_DV = 0, U_DG, U_PQ, U_DQH, U_PK, U_DKH, U_Q, U_DQ1 };
constexpr uint32_t T_W = 0, T_DX = 256;
constexpr uint32_t I_DX = idesc_bf16(128, 64);

struct Bars {
  uint64_t full[NSMAX], rdy[NSMAX], empty[NSMAX], auxfull[2], auxempty[2], dxfull[2], dxfree[2];
  uint32_t tmem;
};

DEVI void mbar_arrive_cnt(uint64_t* b, uint32_t n) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(n) : "memory"); }
DEVI void tma_load_5d(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, int c2, int c3, int c4) {
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory");
}
DEVI float lds_bf16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a) : "memory"); return __uint_as_float((uint32_t)v << 16); }
DEVI void sts_bf16(uint32_t a, float x) {
  const unsigned short v = __bfloat16_as_ushort(__float2bfloat16_rn(x));
  asm volatile("st.shared.u16 [%0], %1;" :: "r"(a), "h"(v) : "memory");
}
#ifndef TRACE
#define TRACE 0
#endif
#define TR(T, e) do { if (TRACE && blockIdx.x == 0 && (T) < 24) TRb[(T) * 48 + (e)] = clock64(); } while (0)
// mbarrier wait that suspends the thread in hardware until the phase completes (no spinning warps stealing issue slots)
DEVI void mbar_wait_h(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok) : "r"(smem_u32(b)), "r"(parity), "r"(10000000u) : "memory");
  } while (!ok);
}
// position of a tile's first slot use in the ring (slot, parity), advanced per tile without divisions (UPT <= NS)
struct RingPos {
  int b = 0; uint32_t p = 0;
  DEVI void get(int u, int ns, int& s, uint32_t& ph) const { s = b + u; ph = p; if (s >= ns) { s -= ns; ph ^= 1u; } }
  DEVI void next(int ns) { b += UPT; if (b >= ns) { b -= ns; p ^= 1u; } }
};
DEVI void red1(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }

// v[2 j] / v[2 j + 1] = two per-row quantities of row j (32 rows) held by every lane -> lane l returns their sums over the warp for row l
DEVI void rscatter64(float (&v)[64], int lane, float& o0, float& o1) {
#pragma unroll
  for (int i = 0; i < 32; ++i) { const bool up = lane & 16; const float k = up ? v[i + 32] : v[i], s = up ? v[i] : v[i + 32]; v[i] = k + __shfl_xor_sync(~0u, s, 16); }
#pragma unroll
  for (int i = 0; i < 16; ++i) { const bool up = lane & 8; const float k = up ? v[i + 16] : v[i], s = up ? v[i] : v[i + 16]; v[i] = k + __shfl_xor_sync(~0u, s, 8); }
#pragma unroll
  for (int i = 0; i < 8; ++i) { const bool up = lane & 4; const float k = up ? v[i + 8] : v[i], s = up ? v[i] : v[i + 8]; v[i] = k + __shfl_xor_sync(~0u, s, 4); }
#pragma unroll
  for (int i = 0; i < 4; ++i) { const bool up = lane & 2; const float k = up ? v[i + 4] : v[i], s = up ? v[i] : v[i + 4]; v[i] = k + __shfl_xor_sync(~0u, s, 2); }
#pragma unroll
  for (int i = 0; i < 2; ++i) { const bool up = lane & 1; const float k = up ? v[i + 2] : v[i], s = up ? v[i] : v[i + 2]; v[i] = k + __shfl_xor_sync(~0u, s, 1); }
  o0 = v[0]; o1 = v[1];
}

extern "C" __global__ void __launch_bounds__(384, 1)
swa_qkvg_bwd_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mdq1, const __grid_constant__ CUtensorMap mpq,
                   const __grid_constant__ CUtensorMap mpk, const __grid_constant__ CUtensorMap mdg, const __grid_constant__ CUtensorMap mdqh,
                   const __grid_constant__ CUtensorMap mdkh, const __grid_constant__ CUtensorMap mdvh, const __grid_constant__ CUtensorMap mmod,
                   const __grid_constant__ CUtensorMap mcos, const __grid_constant__ CUtensorMap msin, const __grid_constant__ CUtensorMap mdp128,
                   const __grid_constant__ CUtensorMap mdp64, const __grid_constant__ CUtensorMap mdqo,
                   int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, int NS, float eps, float qk_eps,
                   const __nv_bfloat16* __restrict__ WT, float* __restrict__ DMOD, unsigned long long* __restrict__ TRb) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int O_CS = AT * 512, O_SN = O_CS + (AT * 64 + 127) / 128 * 128;
  const int AUXST = (O_SN + (AT * 64 + 127) / 128 * 128 + 1023) / 1024 * 1024;
  const int O_AUX = NS * SLOT, O_RED = O_AUX + 2 * AUXST, O_RS = O_RED + 2 * 4 * 32 * 8, O_BAR = O_RS + 2 * 32 * 8;
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
    for (int i = 0; i < 2; ++i) { mbar_init(&B.auxfull[i], 1); mbar_init(&B.auxempty[i], 2); mbar_init(&B.dxfull[i], 1); mbar_init(&B.dxfree[i], 2); }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (warp >= 4) {                                                       // W^T -> TMEM (A operand): warpgroup w writes outs 256 w ..
    const int wg = (warp - 4) >> 2, lb = (warp & 3) * 32, c = lb + lane;
    const __nv_bfloat16* src = WT + (size_t)c * 512 + 256 * wg;
#pragma unroll 1
    for (int k = 0; k < 8; ++k) {
      uint32_t v[16];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const uint4 u = ldg128(src + 32 * k + 8 * e); v[4 * e] = u.x; v[4 * e + 1] = u.y; v[4 * e + 2] = u.z; v[4 * e + 3] = u.w; }
      tmem_st16(tmem + ((uint32_t)lb << 16) + T_W + 128 * wg + 16 * k, v);
    }
    tmem_wait_st();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (tid == 0 && TRACE && blockIdx.x == 0) TRb[24 * 48] = clock64();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      const uint32_t tx = (uint32_t)(SP * AT * 256), auxtx = (uint32_t)(AT * 512 + 2 * AT * 64);
      RingPos rp; int gtot = 0;
      for (int T = 0; T < ntT; ++T, rp.next(NS)) {
        int b, a0, s0; coords(T, b, a0, s0);
        const int xa = T & 1;
        if (T >= 2) mbar_wait_h(&B.auxempty[xa], ((T >> 1) - 1) & 1);
        const uint32_t ax = su + O_AUX + xa * AUXST;
        mbar_expect_tx(&B.auxfull[xa], auxtx);
        tma_load_3d(ax, &mmod, &B.auxfull[xa], 0, b * S + s0, 4);             // scale_a: blocks 4..7
        tma_load_2d(ax + O_CS, &mcos, &B.auxfull[xa], 0, b * S + s0);
        tma_load_2d(ax + O_SN, &msin, &B.auxfull[xa], 0, b * S + s0);
        for (int u = 0; u < UPT; ++u, ++gtot) {
          int sl; uint32_t ph; rp.get(u, NS, sl, ph);
          if (gtot >= NS) mbar_wait_h(&B.empty[sl], ph ^ 1);
          TR(T, u);
          const uint32_t d = su + sl * SLOT;
          mbar_expect_tx(&B.full[sl], tx);
          if (u == U_DV || u == U_DQH || u == U_DKH) {
            const CUtensorMap* m = u == U_DV ? &mdvh : u == U_DQH ? &mdqh : &mdkh;
            for (int h = 0; h < 4; ++h) tma_load_5d(d + h * HB, m, &B.full[sl], 0, s0, h, b, a0);
          } else {
            const CUtensorMap* m = u == U_DG ? &mdg : u == U_PQ ? &mpq : u == U_PK ? &mpk : u == U_Q ? &mq : &mdq1;
            for (int kb = 0; kb < 2; ++kb) tma_load_4d(d + kb * KBLK, m, &B.full[sl], kb * 64, s0, b, a0);
          }
          if (u != U_PQ && u != U_PK && u != U_DQ1) mbar_arrive_cnt(&B.rdy[sl], 2);   // keeps the rdy phases in step with the ring
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer: dx^T = W^T dP^T
    RingPos rp;
    for (int T = 0; T < ntT; ++T, rp.next(NS)) {
      const int xb = T & 1;
      if (T >= 2) mbar_wait_h(&B.dxfree[xb], ((T >> 1) - 1) & 1);
      if (lane == 0) TR(T, 22);
      const uint32_t dacc = tmem + T_DX + xb * 64;
      const int order[4] = {U_DV, U_DG, U_PQ, U_PK};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int u = order[i]; int sl; uint32_t ph; rp.get(u, NS, sl, ph);
        if (u == U_DV || u == U_DG) mbar_wait_h(&B.full[sl], ph); else mbar_wait_h(&B.rdy[sl], ph);
        if (lane == 0) TR(T, 18 + i);
        tc_fence_after();
        const int kp = u == U_PQ ? 0 : u == U_PK ? 1 : u == U_DV ? 2 : 3;
        const uint32_t sa = su + sl * SLOT;
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            const uint64_t bd = u == U_DV ? desc_sw64(sa + (ks >> 1) * HB) + (uint64_t)((ks & 1) * 2)
                                          : desc_k128(sa + (ks >> 2) * KBLK) + (uint64_t)((ks & 3) * 2);
            umma_ts(dacc, tmem + T_W + kp * 64 + ks * 8, bd, I_DX, (i > 0 || ks > 0) ? 1u : 0u);
          }
          tc_commit(&B.empty[sl]);
          if (i == 3) tc_commit(&B.dxfull[xb]);
        }
        __syncwarp();
      }
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA stores: dP (dV, dG, dpq, dpk) and dq
    if (lane == 0) {
      RingPos rp;
      for (int T = 0; T < ntT; ++T, rp.next(NS)) {
        int b, a0, s0; coords(T, b, a0, s0);
        const int order[5] = {U_DV, U_DG, U_PQ, U_PK, U_DQ1};
        for (int i = 0; i < 5; ++i) {
          const int u = order[i]; int sl; uint32_t ph; rp.get(u, NS, sl, ph);
          if (u == U_DV || u == U_DG) mbar_wait_h(&B.full[sl], ph); else mbar_wait_h(&B.rdy[sl], ph);
          TR(T, 23 + 2 * i);
          const uint32_t sa = su + sl * SLOT;
          if (u == U_DV) {
            for (int h = 0; h < 4; ++h) tma_store_4d(&mdp64, sa + h * HB, 256 + 32 * h, s0, b, a0);
          } else {
            const CUtensorMap* m = u == U_DQ1 ? &mdqo : &mdp128;
            const int c0 = u == U_DG ? 384 : u == U_PQ ? 0 : u == U_PK ? 128 : 0;
            for (int kb = 0; kb < 2; ++kb) tma_store_4d(m, sa + kb * KBLK, c0 + 64 * kb, s0, b, a0);
          }
          tma_store_commit();
          tma_store_wait_read0();
          TR(T, 24 + 2 * i);
          mbar_arrive_cnt(&B.empty[sl], u == U_DQ1 ? 2 : 1);
        }
      }
      tma_store_wait0();
    }
  } else if (warp == 3) {
    RingPos rp;
    if (TRACE && lane == 0 && blockIdx.x == 0)                           // observer: completion time of every load
      for (int T = 0; T < ntT && T < 24; ++T, rp.next(NS))
        for (int u = 0; u < UPT; ++u) { int sl; uint32_t ph; rp.get(u, NS, sl, ph); mbar_wait_h(&B.full[sl], ph); TR(T, 33 + u); }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ row threads
    const int wg = (warp - 4) >> 2, qw = warp & 3;
    const uint32_t lb = (uint32_t)qw * 32;
    const int c = (int)lb + lane;                                          // final phase: channel c (TMEM lane)
    const uint32_t r = (uint32_t)((qw & 1) * 32 + lane);                   // q / k phases: row r, head hq
    const int hq = 2 * wg + (qw >> 1);
    const bool rowok = (int)r < SP * AT;
    const int atr = rowok ? (int)r % AT : 0;                               // the row's atom (q / k phases)
    float2* red = reinterpret_cast<float2*>(sm + O_RED);                   // [2 wg][4 warps][32 rows] (sum q^2, sum dx (1 + scale) q)
    float2* rs = reinterpret_cast<float2*>(sm + O_RS);                     // [2 wg][32 rows] (rstd, s2)

    const bool tl = wg == 0 && qw == 0 && lane == 0;
    RingPos rrp;
    auto qk_phase = [&](int T, int uP, int uD) {
      const int e0 = uP == U_PQ ? 8 : 11;
      if (tl) TR(T, e0);
      int slP, slD; uint32_t phP, phD; rrp.get(uP, NS, slP, phP); rrp.get(uD, NS, slD, phD);
      const uint32_t ax = O_AUX + (T & 1) * AUXST;
      float cs[16], sn[16];
      {
        const float4* c4 = reinterpret_cast<const float4*>(sm + ax + O_CS + atr * 64);
        const float4* s4 = reinterpret_cast<const float4*>(sm + ax + O_SN + atr * 64);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float4 a = c4[k], b2 = s4[k];
          cs[4 * k] = a.x; cs[4 * k + 1] = a.y; cs[4 * k + 2] = a.z; cs[4 * k + 3] = a.w;
          sn[4 * k] = b2.x; sn[4 * k + 1] = b2.y; sn[4 * k + 2] = b2.z; sn[4 * k + 3] = b2.w;
        }
      }
      mbar_wait_h(&B.full[slP], phP);
      mbar_wait_h(&B.full[slD], phD);
      if (tl) TR(T, e0 + 1);
      const uint32_t pb = su + slP * SLOT + (hq >> 1) * KBLK, db = su + slD * SLOT + hq * HB;
      const int c0 = 4 * (hq & 1);
      uint32_t pw[16], dw[16];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint4 u = lds128(pb + sw128(r, c0 + k)); pw[4 * k] = u.x; pw[4 * k + 1] = u.y; pw[4 * k + 2] = u.z; pw[4 * k + 3] = u.w;
        const uint4 v = lds128(db + sw64(r, k)); dw[4 * k] = v.x; dw[4 * k + 1] = v.y; dw[4 * k + 2] = v.z; dw[4 * k + 3] = v.w;
      }
      float p[32], gv[32];
#pragma unroll
      for (int k = 0; k < 16; ++k) { p[2 * k] = bf16lo(pw[k]); p[2 * k + 1] = bf16hi(pw[k]); gv[2 * k] = bf16lo(dw[k]); gv[2 * k + 1] = bf16hi(dw[k]); }
#pragma unroll
      for (int d = 0; d < 16; ++d) {                                       // transpose of RoPE: dx1 = dy1 c + dy2 s; dx2 = dy2 c - dy1 s
        const float y1 = gv[d], y2 = gv[d + 16];
        gv[d] = y1 * cs[d] + y2 * sn[d];
        gv[d + 16] = y2 * cs[d] - y1 * sn[d];
      }
      float s4[4] = {0.f, 0.f, 0.f, 0.f};                                  // 4 partial sums: short dependency chains
#pragma unroll
      for (int d = 0; d < 32; ++d) s4[d & 3] = fmaf(p[d], p[d], s4[d & 3]);
      const float rr = 1.f / sqrtf(((s4[0] + s4[1]) + (s4[2] + s4[3])) * (1.f / D) + qk_eps);
#pragma unroll
      for (int k = 0; k < 4; ++k) s4[k] = 0.f;
#pragma unroll
      for (int d = 0; d < 32; ++d) { p[d] *= rr; s4[d & 3] = fmaf(gv[d], p[d], s4[d & 3]); }
      const float m = ((s4[0] + s4[1]) + (s4[2] + s4[3])) * (1.f / D);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        uint4 o;
        o.x = pack_bf16(rr * (gv[8 * k] - p[8 * k] * m), rr * (gv[8 * k + 1] - p[8 * k + 1] * m));
        o.y = pack_bf16(rr * (gv[8 * k + 2] - p[8 * k + 2] * m), rr * (gv[8 * k + 3] - p[8 * k + 3] * m));
        o.z = pack_bf16(rr * (gv[8 * k + 4] - p[8 * k + 4] * m), rr * (gv[8 * k + 5] - p[8 * k + 5] * m));
        o.w = pack_bf16(rr * (gv[8 * k + 6] - p[8 * k + 6] * m), rr * (gv[8 * k + 7] - p[8 * k + 7] * m));
        sts128(pb + sw128(r, c0 + k), o);                                 // dpq in place over pq (the same bytes this thread read)
      }
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) { mbar_arrive(&B.rdy[slP]); mbar_arrive(&B.empty[slD]); }
      if (tl) TR(T, e0 + 2);
    };

    const uint32_t cofs = (uint32_t)(c >> 6) * KBLK + (uint32_t)(c & 7) * 2, cch = (uint32_t)((c & 63) >> 3);
    uint32_t xo[8];                                                          // swizzled 16-B chunk of channel c in row r: xo[r & 7]
#pragma unroll
    for (int k = 0; k < 8; ++k) xo[k] = (cch ^ (uint32_t)k) << 4;
    // final phase: one channel c per thread (its TMEM lane), rows 32 wg .. 32 wg + 31 (TMEM columns of dx^T)
    auto fin = [&](int T, auto at8c) {
      constexpr bool AT8 = decltype(at8c)::value;
      int b, a0, s0; coords(T, b, a0, s0);
      const int xb = T & 1; int slQ, slD1; uint32_t phQ, phD1; rrp.get(U_Q, NS, slQ, phQ); rrp.get(U_DQ1, NS, slD1, phD1);
      const uint32_t ax = O_AUX + (T & 1) * AUXST;
      if (tl) TR(T, 14);
      mbar_wait_h(&B.full[slQ], phQ);
      mbar_wait_h(&B.full[slD1], phD1);
      if (tl) TR(T, 15);
      mbar_wait_h(&B.dxfull[xb], (T >> 1) & 1);
      if (tl) TR(T, 16);
      tc_fence_after();
      const uint32_t tdx = tmem + (lb << 16) + T_DX + xb * 64 + 32 * wg;
      const uint32_t qa = su + slQ * SLOT + cofs + (uint32_t)(32 * wg) * 128u, d1a = su + slD1 * SLOT + cofs + (uint32_t)(32 * wg) * 128u;
      auto scl = [&](int at) {                                             // scale_a[b, s0 + at][c]: block c / 32 of the aux tile
        const int row = (c >> 5) * AT + at;
        return *reinterpret_cast<const float*>(sm + ax + row * 128 + ((((c & 31) >> 2) ^ (row & 7)) << 4) + (c & 3) * 4);
      };
      auto exchange = [&]() {                                              // red -> rs: per-row totals over the 4 warps
        named_bar_sync(1 + wg, 128);
        if (qw == 0) {
          float ssq = 0.f, tt = 0.f;
#pragma unroll
          for (int k = 0; k < 4; ++k) { const float2 x = red[(wg * 4 + k) * 32 + lane]; ssq += x.x; tt += x.y; }
          const float rstd = rsqrtf(ssq * (1.f / C) + eps);
          rs[wg * 32 + lane] = make_float2(rstd, rstd * tt * (1.f / C));
        }
        named_bar_sync(1 + wg, 128);
      };
      auto release = [&]() {
        fence_proxy_async();
        tc_fence_before();
        named_bar_sync(1 + wg, 128);
        if (qw == 0 && lane == 0) {
          mbar_arrive(&B.rdy[slD1]); mbar_arrive(&B.empty[slQ]); mbar_arrive(&B.dxfree[xb]); mbar_arrive(&B.auxempty[T & 1]);
        }
        if (tl) TR(T, 17);
      };
      if constexpr (AT8) {
        // rows 32 wg + j: augment a0 + 4 wg + j / 8, atom s0 + j % 8
        float sc[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) sc[k] = 1.f + scl(k);
        uint32_t aok = 0, sok = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) sok |= (s0 + k < S) ? 1u << k : 0u;
#pragma unroll
        for (int k = 0; k < 4; ++k) aok |= (a0 + 4 * wg + k < A) ? 1u << k : 0u;
        // ---- pass 1: per-row sums of q^2 and dx (1 + scale) q over this warp's 32 channels -> smem exchange over the 4 warps
        {
          uint32_t dv[32];
          tmem_ld32(tdx, dv);
          float v[64];
#pragma unroll
          for (int j = 0; j < 32; ++j) { const float qv = lds_bf16(qa + j * 128 + xo[j & 7]); v[2 * j] = qv * qv; v[2 * j + 1] = qv * sc[j & 7]; }
          tmem_wait_ld();
#pragma unroll
          for (int j = 0; j < 32; ++j) v[2 * j + 1] *= __uint_as_float(dv[j]);
          float o0, o1;
          rscatter64(v, lane, o0, o1);
          red[(wg * 4 + qw) * 32 + lane] = make_float2(o0, o1);
        }
        exchange();
        // ---- pass 2: dq in place over dq1; modulation-gradient sums in registers
        float ash[8], asc[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) { ash[k] = 0.f; asc[k] = 0.f; }
        {
          uint32_t dv[32];
          tmem_ld32(tdx, dv);
          const float4* rp = reinterpret_cast<const float4*>(rs + wg * 32);
          float qa_[32], d1a_[32];                                         // all shared-memory loads first, then the math and the stores
#pragma unroll
          for (int j = 0; j < 32; ++j) { qa_[j] = lds_bf16(qa + j * 128 + xo[j & 7]); d1a_[j] = lds_bf16(d1a + j * 128 + xo[j & 7]); }
          tmem_wait_ld();
#pragma unroll
          for (int j2 = 0; j2 < 16; ++j2) {
            const float4 r2 = rp[j2];                                     // (rstd, s2) of rows 2 j2, 2 j2 + 1
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int j = 2 * j2 + e, k = j & 7;
              const float rstd = e ? r2.z : r2.x, s2 = e ? r2.w : r2.y;
              const float qv = qa_[j], d1 = d1a_[j];
              const float dx = __uint_as_float(dv[j]), xh = qv * rstd, dxh = dx * sc[k];
              sts_bf16(d1a + j * 128 + xo[k], d1 + rstd * (dxh - xh * s2));
              const bool ok = ((aok >> (j >> 3)) & (sok >> k) & 1u) != 0;
              ash[k] += ok ? dx : 0.f; asc[k] += ok ? dx * xh : 0.f;
            }
          }
        }
        release();
#pragma unroll
        for (int k = 0; k < 8; ++k)
          if ((sok >> k) & 1u) { float* dm = DMOD + ((size_t)b * S + s0 + k) * (6 * C) + c; red1(dm, ash[k]); red1(dm + C, asc[k]); }
        if (tl) TR(T, 41);
      } else {
        // generic AT: one row at a time (rare training shapes, A < 8): warp sums by butterfly, per-row red.add of the modulation gradient
#pragma unroll 1
        for (int j = 0; j < 32; ++j) {
          const int row = 32 * wg + j;
          uint32_t dv;
          asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(dv) : "r"(tdx + j) : "memory");
          tmem_wait_ld();
          const float qv = lds_bf16(qa + j * 128 + ((cch ^ (uint32_t)(row & 7)) << 4));
          float x0 = qv * qv, x1 = __uint_as_float(dv) * (1.f + scl(row % AT)) * qv;
#pragma unroll
          for (int m = 16; m > 0; m >>= 1) { x0 += __shfl_xor_sync(~0u, x0, m); x1 += __shfl_xor_sync(~0u, x1, m); }
          if (lane == 0) red[(wg * 4 + qw) * 32 + j] = make_float2(x0, x1);
        }
        exchange();
#pragma unroll 1
        for (int j = 0; j < 32; ++j) {
          const int row = 32 * wg + j, at = row % AT, sp = row / AT;
          uint32_t dv;
          asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(dv) : "r"(tdx + j) : "memory");
          tmem_wait_ld();
          const float2 rr = rs[wg * 32 + j];
          const uint32_t xr = (cch ^ (uint32_t)(row & 7)) << 4, qe = qa + j * 128 + xr, de = d1a + j * 128 + xr;
          const float qv = lds_bf16(qe), d1 = lds_bf16(de);
          const float dx = __uint_as_float(dv), xh = qv * rr.x, dxh = dx * (1.f + scl(at));
          sts_bf16(de, d1 + rr.x * (dxh - xh * rr.y));
          if (row < SP * AT && a0 + sp < A && s0 + at < S) {
            float* dm = DMOD + ((size_t)b * S + s0 + at) * (6 * C) + c;
            red1(dm, dx); red1(dm + C, dx * xh);
          }
        }
        release();
      }
    };

    for (int T = 0; T < ntT; ++T) {
      mbar_wait_h(&B.auxfull[T & 1], (T >> 1) & 1);
      qk_phase(T, U_PQ, U_DQH);
      qk_phase(T, U_PK, U_DKH);
      if (AT == 8) fin(T, std::true_type{}); else fin(T, std::false_type{});
      rrp.next(NS);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
