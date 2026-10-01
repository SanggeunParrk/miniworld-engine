// ffn_bwd_dy.cu — second half of the SWA atom block's FFN backward on sm_100a (the Triton _ffn_bwd's dq1 and modulation gradient):
//   dy = DAB Wu (fp32, DAB = [da | db] from ffn_bwd_gate.cu);  xh = q1 rstd;  dxh = dy (1 + scale_f)
//   dq1 = rn(dq2 + rstd (dxh - xh mean(dxh xh)));  d shift_f += sum_aug dy,  d scale_f += sum_aug dy xh,  d gate_f += sum_aug dq2 ffn
// Structure of qkvg_bwd.cu: Wu^T [128 ch][512 hidden] lives in TMEM as the bf16 A operand, dy^T = Wu^T DAB^T (M = 128 channels, N = 64 rows,
// K = 512) takes DAB straight from its tiles (K-major B); 64-row tiles of SP augments x AT atoms; a ring of 8-KB slots (a [64 rows][64]
// block each), 14 uses per tile: DAB blocks 0-7, q1 x 2, ffn x 2, dq2 x 2 (-> dq1 in place, TMA-stored). Final phase: one channel per
// thread, rows 32 w .. per warpgroup; per-row sums by reduce-scatter + 4-warp exchange; augment sums in registers (AT = 8).
// TMEM: Wu^T (bf16) at 0 (256 columns), dy^T[2] at 256 / 320.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include <type_traits>
using namespace s100;

constexpr int C = 128, NSMAX = 28, UPT = 14;
constexpr int SLOT = 8192;                                                 // [64 rows][64] bf16, SW128
enum { U_DAB = 0, U_Q1 = 8, U_FFN = 10, U_DQ2 = 12 };
constexpr uint32_t T_W = 0, T_DY = 256;
constexpr uint32_t I_DY = idesc_bf16(128, 64);

struct Bars {
  uint64_t full[NSMAX], rdy[NSMAX], empty[NSMAX], auxfull[2], auxempty[2], dxfull[2], dxfree[2];
  uint32_t tmem;
};
struct RingPos {
  int b = 0; uint32_t p = 0;
  DEVI void get(int u, int ns, int& s, uint32_t& ph) const { s = b + u; ph = p; while (s >= ns) { s -= ns; ph ^= 1u; } }
  DEVI void next(int ns) { b += UPT; while (b >= ns) { b -= ns; p ^= 1u; } }
};
DEVI void mbar_arrive_cnt(uint64_t* b, uint32_t n) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(n) : "memory"); }
DEVI float lds_bf16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a) : "memory"); return __uint_as_float((uint32_t)v << 16); }
DEVI void sts_bf16(uint32_t a, float x) {
  const unsigned short v = __bfloat16_as_ushort(__float2bfloat16_rn(x));
  asm volatile("st.shared.u16 [%0], %1;" :: "r"(a), "h"(v) : "memory");
}
DEVI void red1(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }
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
swa_ffn_bwd_dy_sm100(const __grid_constant__ CUtensorMap mdab, const __grid_constant__ CUtensorMap mq1, const __grid_constant__ CUtensorMap mffn,
                     const __grid_constant__ CUtensorMap mdq2, const __grid_constant__ CUtensorMap mmod, const __grid_constant__ CUtensorMap mdq1,
                     int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, int NS, float eps,
                     const __nv_bfloat16* __restrict__ WUT, float* __restrict__ DMOD) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int AUXST = (AT * 512 + 1023) / 1024 * 1024;
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
  if (warp >= 4) {                                                        // Wu^T -> TMEM: warpgroup w writes hidden 256 w ..
    const int wg = (warp - 4) >> 2, lb = (warp & 3) * 32, c = lb + lane;
    const __nv_bfloat16* src = WUT + (size_t)c * 512 + 256 * wg;
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

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      const uint32_t tx = (uint32_t)(SP * AT * 128), auxtx = (uint32_t)(AT * 512);
      RingPos rp; int gtot = 0;
      for (int T = 0; T < ntT; ++T, rp.next(NS)) {
        int b, a0, s0; coords(T, b, a0, s0);
        const int xa = T & 1;
        if (T >= 2) mbar_wait(&B.auxempty[xa], ((T >> 1) - 1) & 1);
        mbar_expect_tx(&B.auxfull[xa], auxtx);
        tma_load_3d(su + O_AUX + xa * AUXST, &mmod, &B.auxfull[xa], 0, b * S + s0, 16);   // scale_f: blocks 16..19
        for (int u = 0; u < UPT; ++u, ++gtot) {
          int sl; uint32_t ph; rp.get(u, NS, sl, ph);
          if (gtot >= NS) mbar_wait(&B.empty[sl], ph ^ 1);
          mbar_expect_tx(&B.full[sl], tx);
          const CUtensorMap* m = u < U_Q1 ? &mdab : u < U_FFN ? &mq1 : u < U_DQ2 ? &mffn : &mdq2;
          const int c0 = u < U_Q1 ? 64 * u : 64 * ((u - U_Q1) & 1);
          tma_load_4d(su + sl * SLOT, m, &B.full[sl], c0, s0, b, a0);
          if (u < U_DQ2) mbar_arrive_cnt(&B.rdy[sl], 2);                  // only dq2 (-> dq1) is thread-written
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer: dy^T = Wu^T DAB^T
    RingPos rp;
    for (int T = 0; T < ntT; ++T, rp.next(NS)) {
      const int xb = T & 1;
      if (T >= 2) mbar_wait(&B.dxfree[xb], ((T >> 1) - 1) & 1);
      for (int kb = 0; kb < 8; ++kb) {
        int sl; uint32_t ph; rp.get(U_DAB + kb, NS, sl, ph);
        mbar_wait(&B.full[sl], ph);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ts(tmem + T_DY + xb * 64, tmem + T_W + kb * 32 + ks * 8, desc_k128(su + sl * SLOT) + (uint64_t)(ks * 2), I_DY,
                    (kb > 0 || ks > 0) ? 1u : 0u);
          tc_commit(&B.empty[sl]); tc_commit(&B.empty[sl]);              // the MMA is the only reader (count 2)
          if (kb == 7) tc_commit(&B.dxfull[xb]);
        }
        __syncwarp();
      }
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA stores: dq1
    if (lane == 0) {
      RingPos rp;
      for (int T = 0; T < ntT; ++T, rp.next(NS)) {
        int b, a0, s0; coords(T, b, a0, s0);
        int s0l, s1l; uint32_t p0, p1; rp.get(U_DQ2, NS, s0l, p0); rp.get(U_DQ2 + 1, NS, s1l, p1);
        mbar_wait(&B.rdy[s0l], p0); mbar_wait(&B.rdy[s1l], p1);
        tma_store_4d(&mdq1, su + s0l * SLOT, 0, s0, b, a0);
        tma_store_4d(&mdq1, su + s1l * SLOT, 64, s0, b, a0);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive_cnt(&B.empty[s0l], 2); mbar_arrive_cnt(&B.empty[s1l], 2);
      }
      tma_store_wait0();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ channel threads: channel c, rows 32 wg ..
    const int wg = (warp - 4) >> 2, qw = warp & 3;
    const uint32_t lb = (uint32_t)qw * 32;
    const int c = (int)lb + lane;
    float2* red = reinterpret_cast<float2*>(sm + O_RED);                   // [2 wg][4 warps][32 rows]
    float2* rs = reinterpret_cast<float2*>(sm + O_RS);                     // [2 wg][32 rows] (rstd, s2)
    const uint32_t cofs = (uint32_t)(c & 7) * 2 + (uint32_t)(32 * wg) * 128u, cch = (uint32_t)((c & 63) >> 3), cb = (uint32_t)(c >> 6);
    uint32_t xo[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) xo[k] = (cch ^ (uint32_t)k) << 4;
    RingPos rp;
    auto tile = [&](int T, auto at8c) {
      constexpr bool AT8 = decltype(at8c)::value;
      int b, a0, s0; coords(T, b, a0, s0);
      const int xb = T & 1;
      int sQ, sF, sD; uint32_t pQ, pF, pD;
      rp.get(U_Q1 + cb, NS, sQ, pQ); rp.get(U_FFN + cb, NS, sF, pF); rp.get(U_DQ2 + cb, NS, sD, pD);
      const uint32_t qa = su + sQ * SLOT + cofs, fa = su + sF * SLOT + cofs, da = su + sD * SLOT + cofs;
      const uint32_t ax = O_AUX + (T & 1) * AUXST;
      auto scl = [&](int at) {                                             // scale_f[b, s0 + at][c]
        const int row = (c >> 5) * AT + at;
        return *reinterpret_cast<const float*>(sm + ax + row * 128 + ((((c & 31) >> 2) ^ (row & 7)) << 4) + (c & 3) * 4);
      };
      mbar_wait(&B.auxfull[T & 1], (T >> 1) & 1);
      mbar_wait(&B.full[sQ], pQ);
      mbar_wait(&B.dxfull[xb], (T >> 1) & 1);
      tc_fence_after();
      const uint32_t tdy = tmem + (lb << 16) + T_DY + xb * 64 + 32 * wg;
      // ---- pass 1: per-row sums of q1^2 and dy (1 + scale) q1 over this warp's 32 channels -> exchange over the 4 warps
      {
        uint32_t dv[32];
        tmem_ld32(tdy, dv);
        float v[64];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          const int row = 32 * wg + j;
          const float qv = lds_bf16(qa + j * 128 + (AT8 ? xo[j & 7] : ((cch ^ (uint32_t)(j & 7)) << 4)));
          v[2 * j] = qv * qv; v[2 * j + 1] = qv * (1.f + scl(AT8 ? (j & 7) : row % AT));
        }
        tmem_wait_ld();
#pragma unroll
        for (int j = 0; j < 32; ++j) v[2 * j + 1] *= __uint_as_float(dv[j]);
        float o0, o1;
        rscatter64(v, lane, o0, o1);
        red[(wg * 4 + qw) * 32 + lane] = make_float2(o0, o1);
      }
      named_bar_sync(1 + wg, 128);
      if (qw == 0) {
        float ssq = 0.f, tt = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float2 x = red[(wg * 4 + k) * 32 + lane]; ssq += x.x; tt += x.y; }
        const float rstd = rsqrtf(ssq * (1.f / C) + eps);
        rs[wg * 32 + lane] = make_float2(rstd, rstd * tt * (1.f / C));
      }
      named_bar_sync(1 + wg, 128);
      // ---- pass 2: dq1 in place over dq2; shift / scale / gate sums
      mbar_wait(&B.full[sF], pF);
      mbar_wait(&B.full[sD], pD);
      float ash[8], asc[8], agt[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) { ash[k] = 0.f; asc[k] = 0.f; agt[k] = 0.f; }
      uint32_t aok = 0, sok = 0;
#pragma unroll
      for (int k = 0; k < 8; ++k) sok |= (s0 + k < S) ? 1u << k : 0u;
#pragma unroll
      for (int k = 0; k < 4; ++k) aok |= (a0 + 4 * wg + k < A) ? 1u << k : 0u;
      float scv[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) scv[k] = AT8 ? 1.f + scl(k) : 0.f;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {                                     // 16 rows at a time: all shared-memory loads first
        uint32_t dv[16];
        tmem_ld16(tdy + 16 * hf, dv);
        float q_[16], d2_[16], f_[16];
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
          const int j = 16 * hf + jj;
          const uint32_t xr = AT8 ? xo[j & 7] : ((cch ^ (uint32_t)(j & 7)) << 4);
          q_[jj] = lds_bf16(qa + j * 128 + xr); d2_[jj] = lds_bf16(da + j * 128 + xr); f_[jj] = lds_bf16(fa + j * 128 + xr);
        }
        tmem_wait_ld();
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
          const int j = 16 * hf + jj, row = 32 * wg + j;
          const uint32_t xr = AT8 ? xo[j & 7] : ((cch ^ (uint32_t)(j & 7)) << 4);
          const float2 rr = rs[wg * 32 + j];
          const float qv = q_[jj], d2 = d2_[jj], fv = f_[jj];
          const int at = AT8 ? (j & 7) : row % AT;
          const float dy = __uint_as_float(dv[jj]), xh = qv * rr.x, dxh = dy * (AT8 ? scv[j & 7] : 1.f + scl(at));
          sts_bf16(da + j * 128 + xr, d2 + rr.x * (dxh - xh * rr.y));
          if constexpr (AT8) {
            const bool ok = ((aok >> (j >> 3)) & (sok >> (j & 7)) & 1u) != 0;
            ash[j & 7] += ok ? dy : 0.f; asc[j & 7] += ok ? dy * xh : 0.f; agt[j & 7] += ok ? d2 * fv : 0.f;
          } else {
            const int sp = row / AT;
            if (row < SP * AT && a0 + sp < A && s0 + at < S) {
              float* dm = DMOD + ((size_t)b * S + s0 + at) * (6 * C) + 3 * C + c;
              red1(dm, dy); red1(dm + C, dy * xh); red1(dm + 2 * C, d2 * fv);
            }
          }
        }
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) {
        int s2l; uint32_t p2;
        for (int k = 0; k < 2; ++k) {                                      // both channel halves of q1 / ffn / dq2
          rp.get(U_DQ2 + k, NS, s2l, p2); mbar_arrive(&B.rdy[s2l]);
          rp.get(U_Q1 + k, NS, s2l, p2); mbar_arrive(&B.empty[s2l]);
          rp.get(U_FFN + k, NS, s2l, p2); mbar_arrive(&B.empty[s2l]);
        }
        mbar_arrive(&B.dxfree[xb]); mbar_arrive(&B.auxempty[T & 1]);
      }
      if constexpr (AT8) {
#pragma unroll
        for (int k = 0; k < 8; ++k)
          if ((sok >> k) & 1u) {
            float* dm = DMOD + ((size_t)b * S + s0 + k) * (6 * C) + 3 * C + c;
            red1(dm, ash[k]); red1(dm + C, asc[k]); red1(dm + 2 * C, agt[k]);
          }
      }
    };
    for (int T = 0; T < ntT; ++T, rp.next(NS)) {
      if (AT == 8) tile(T, std::true_type{}); else tile(T, std::false_type{});
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
