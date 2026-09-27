// qkvg_fwd.cu — the SWA atom block's first forward stage on sm_100a (the B200 port of team-gm swa_cuda/swa_qkvg_fwd.cu, same math and
// rounding points as the Triton _qkvg_fwd):
//   x = rn(RMS(q) (1 + scale_a) + shift_a);  p_{q,k,v,g} = x W_{q,k,v,g}^T;
//   Q = rn(rope(rn(headRMS(rn(p_q)))))  (same for K);  V = rn(p_v);  G = rn(p_g)
//   Q / K / V written head-major [N, H, S, D]; G row-major [M, C]; optionally x, rn(p_q), rn(p_k) saved for the backward.
// Structure: persistent CTAs; the four projections' weights (128 KB) stay resident in shared memory; a tile is SP augments x AT atoms of
// one batch element (SP * AT <= 128 rows, so the tile needs only AT modulation rows); x goes to TMEM as the bf16 A operand and each
// projection is one M128 N128 K128 TS MMA into one of two TMEM accumulators, so the epilogue of projection p overlaps the MMA of p + 1.
// One query row per thread, two warpgroups (warps 4-7, 8-11) splitting the columns -- TMEM lets two warpgroups use the same lanes in
// different columns: warpgroup w writes x channels 64 w .. 64 w + 63 of the next tile (both compute the row's RMS) and runs the epilogue
// of heads 2 w, 2 w + 1 of every projection; the per-head RMS and the RoPE pairs (d, d + 16) live in one thread's registers.
// TMEM: x[b] at b * 64 (bf16, 64 cols), acc[b] at 128 + b * 128.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int C = 128, H = 4, D = 32, ATMAX = 25;
constexpr int KB = 128 * 128;                                              // one k-block: [128 rows][64 bf16] (16 KB)
constexpr int O_W = 0, O_Q = O_W + 8 * KB;                                  // W: projection p, k-block kb at (2 p + kb) KB
constexpr int QST = 2 * KB;                                                // q stage: 2 k-blocks
constexpr int CSA = 2048;                                                  // one cos or sin table: [ATMAX][16] fp32, padded to 2 KB
constexpr int O_CS = O_Q + 2 * QST;                                         // cos | sin per stage (TMA destinations: 128-B aligned)
constexpr int CSST = 2 * CSA;
constexpr int O_MOD = (O_CS + 2 * CSST + 1023) / 1024 * 1024;               // shift | scale: [8 blocks][AT][32] fp32, 128-B swizzle
                                                                           // (block k = channels 32 (k % 4).. of shift (k < 4) / scale)
constexpr int O_BAR = O_MOD + ATMAX * 256 * 4;
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_X = 0, T_ACC = 128;
constexpr uint32_t I_P = idesc_bf16(128, 128);

struct Bars {
  uint64_t wbar, qfull[2], qempty[2], modfull, modempty, xfull[2], xfree[2], afull[2], afree[2];
  uint32_t tmem;
};
struct Args {
  int S, A, B, SP, AT, nab, nag, ntile;
  float eps, qk_eps;
  int save;
};

#ifndef NOSTORE
#define NOSTORE 0                        // timing diagnostics: 1 drops the global stores of the epilogue
#endif
DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

extern "C" __global__ void __launch_bounds__(384, 1)
swa_qkvg_fwd_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mmod,
                   const __grid_constant__ CUtensorMap mcos, const __grid_constant__ CUtensorMap msin,
                   int S_, int A_, int B_, int SP_, int AT_, int nab_, int nag_, int ntile_, float eps_, float qk_eps_, int save_,
                   __nv_bfloat16* __restrict__ Qo, __nv_bfloat16* __restrict__ Ko, __nv_bfloat16* __restrict__ Vo, __nv_bfloat16* __restrict__ Go,
                   __nv_bfloat16* __restrict__ Xs, __nv_bfloat16* __restrict__ PQs, __nv_bfloat16* __restrict__ PKs) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const Args args{S_, A_, B_, SP_, AT_, nab_, nag_, ntile_, eps_, qk_eps_, save_};
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int S = args.S, SP = args.SP, AT = args.AT;
  const int ntT = (int)blockIdx.x < args.ntile ? (args.ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto coords = [&](int T, int& b, int& a0, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    const int ab = t % args.nab, r = t / args.nab, ag = r % args.nag;
    b = r / args.nag; a0 = ag * SP; s0 = ab * AT;
  };

  if (tid == 0) {
    mbar_init(&B.wbar, 1); mbar_init(&B.modfull, 1); mbar_init(&B.modempty, 8);   // 8 = both warpgroups
    for (int i = 0; i < 2; ++i) {
      mbar_init(&B.qfull[i], 1); mbar_init(&B.qempty[i], 8);
      mbar_init(&B.xfull[i], 8); mbar_init(&B.xfree[i], 1);
      mbar_init(&B.afull[i], 1); mbar_init(&B.afree[i], 8);
    }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      mbar_expect_tx(&B.wbar, 8 * KB);
      for (int p = 0; p < 4; ++p)
        for (int kb = 0; kb < 2; ++kb) tma_load_2d(su + O_W + (2 * p + kb) * KB, &mw, &B.wbar, kb * 64, p * 128);
      const uint32_t qbytes = (uint32_t)(2 * SP * AT * 128), csbytes = (uint32_t)(2 * AT * 16 * 4), mbytes = (uint32_t)(AT * 256 * 4);
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        const int st = T & 1;
        if (T >= 2) mbar_wait(&B.qempty[st], ((T >> 1) - 1) & 1);
        mbar_expect_tx(&B.qfull[st], qbytes + csbytes);
        for (int kb = 0; kb < 2; ++kb) tma_load_4d(su + O_Q + st * QST + kb * KB, &mq, &B.qfull[st], kb * 64, s0, b, a0);
        tma_load_2d(su + O_CS + st * CSST, &mcos, &B.qfull[st], 0, b * S + s0);
        tma_load_2d(su + O_CS + st * CSST + CSA, &msin, &B.qfull[st], 0, b * S + s0);
        if (T >= 1) mbar_wait(&B.modempty, (T - 1) & 1);
        mbar_expect_tx(&B.modfull, mbytes);
        tma_load_3d(su + O_MOD, &mmod, &B.modfull, 0, b * S + s0, 0);
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    mbar_wait(&B.wbar, 0);
    for (int T = 0; T < ntT; ++T) {
      const int xb = T & 1;
      for (int p = 0; p < 4; ++p) {
        const int g = T * 4 + p, ab = g & 1;
        if (p == 0) mbar_wait(&B.xfull[xb], (T >> 1) & 1);
        if (g >= 2) mbar_wait(&B.afree[ab], ((g >> 1) - 1) & 1);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            const uint64_t dw = desc_k128(su + O_W + (2 * p + (ks >> 2)) * KB) + (uint64_t)((ks & 3) * 2);
            umma_ts(tmem + T_ACC + ab * 128, tmem + T_X + xb * 64 + ks * 8, dw, I_P, ks > 0 ? 1u : 0u);
          }
          tc_commit(&B.afull[ab]);
          if (p == 3) tc_commit(&B.xfree[xb]);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ row threads: warpgroup wg = column half
    const int wg = (warp - 4) >> 2, role = wg + 1;                         // role 1 / 2: heads 0-1 / 2-3, x channels 0-63 / 64-127
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int sp = (int)r / AT, at = (int)r % AT;
    const bool rowok = (int)r < SP * AT;
    const size_t NS = (size_t)args.A * args.B;                               // samples n
    auto pro = [&](int T) {                                                // x of tile T -> TMEM x[T & 1] (and the x save)
      int b, a0, s0; coords(T, b, a0, s0);
      const int st = T & 1, xb = T & 1;
      mbar_wait(&B.qfull[st], (T >> 1) & 1);
      mbar_wait(&B.modfull, T & 1);
      const uint32_t qs = su + O_Q + st * QST;
      float ss = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) {                                       // pass 1: sum of squares over 16 chunks of 8 bf16
        const uint4 u = lds128(qs + (k >> 3) * KB + sw128(r, k & 7));
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float a = bf16lo(w4[e]), b2 = bf16hi(w4[e]); ss = fmaf(a, a, fmaf(b2, b2, ss)); }
      }
      const float rstd = rsqrtf(ss * (1.f / C) + args.eps);
      const int al = rowok ? at : 0;
      // float4 of channels 4 j .. 4 j + 3 of shift (kind 0) / scale (kind 1): row (kind * 4 + j / 8) * AT + al, 16-B chunk j % 8, swizzled
      auto modv = [&](int kind, int j) {
        const int row = (kind * 4 + (j >> 3)) * AT + al;
        return *reinterpret_cast<const float4*>(sm + O_MOD + row * 128 + (((j & 7) ^ (row & 7)) << 4));
      };
      uint32_t xp[32];
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {                                     // pass 2: this warpgroup's 64 channels of x, 8 per chunk
        const int k = 8 * wg + kk;
        const uint4 u = lds128(qs + (k >> 3) * KB + sw128(r, k & 7));
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
        const float4 sh0 = modv(0, 2 * k), sh1 = modv(0, 2 * k + 1), sc0 = modv(1, 2 * k), sc1 = modv(1, 2 * k + 1);
        const float shv[8] = {sh0.x, sh0.y, sh0.z, sh0.w, sh1.x, sh1.y, sh1.z, sh1.w};
        const float scv[8] = {sc0.x, sc0.y, sc0.z, sc0.w, sc1.x, sc1.y, sc1.z, sc1.w};
#pragma unroll
        for (int e = 0; e < 4; ++e)
          xp[4 * kk + e] = pack_bf16(bf16lo(w4[e]) * rstd * (1.f + scv[2 * e]) + shv[2 * e], bf16hi(w4[e]) * rstd * (1.f + scv[2 * e + 1]) + shv[2 * e + 1]);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.modempty);
      if (T >= 2) mbar_wait(&B.xfree[xb], ((T >> 1) - 1) & 1);           // the MMAs of tile T - 2 have read x[xb]
      tc_fence_after();
#pragma unroll
      for (int k = 0; k < 2; ++k) tmem_st16(trow + T_X + xb * 64 + 32 * wg + 16 * k, *reinterpret_cast<uint32_t(*)[16]>(xp + 16 * k));
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.xfull[xb]);
      const int n = (a0 + sp) * args.B + b, s = s0 + at;
      if (args.save && rowok && (size_t)n < NS && s < S) {
        uint4* dst = reinterpret_cast<uint4*>(Xs + ((size_t)n * S + s) * C + 64 * wg);
#pragma unroll
        for (int k = 0; k < 8; ++k) dst[k] = make_uint4(xp[4 * k], xp[4 * k + 1], xp[4 * k + 2], xp[4 * k + 3]);
      }
    };
    auto epi = [&](int T) {
      int b, a0, s0; coords(T, b, a0, s0);
      const int st = T & 1;
      const int n = (a0 + sp) * args.B + b, s = s0 + at;
      const bool ok = rowok && (size_t)n < NS && s < S;
      float cs[16], sn[16];                                                // this row's RoPE tables, once per tile
      mbar_wait(&B.qfull[st], (T >> 1) & 1);                               // (the stage's cos / sin have landed)
      {
        const float4* c4 = reinterpret_cast<const float4*>(sm + O_CS + st * CSST) + (size_t)(rowok ? at : 0) * 4;
        const float4* s4 = c4 + CSA / 16;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float4 a = c4[k], b2 = s4[k];
          cs[4 * k] = a.x; cs[4 * k + 1] = a.y; cs[4 * k + 2] = a.z; cs[4 * k + 3] = a.w;
          sn[4 * k] = b2.x; sn[4 * k + 1] = b2.y; sn[4 * k + 2] = b2.z; sn[4 * k + 3] = b2.w;
        }
      }
      for (int p = 0; p < 4; ++p) {
        const int g = T * 4 + p, ab = g & 1;
        mbar_wait(&B.afull[ab], (g >> 1) & 1);
        tc_fence_after();
        for (int h = 2 * (role - 1); h < 2 * role; ++h) {                  // this warpgroup's two heads (32 columns each)
          uint32_t v[32];
          tmem_ld32(trow + T_ACC + ab * 128 + h * 32, v);
          tmem_wait_ld();
          if (h == 2 * role - 1) {
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(&B.afree[ab]);
          }
          float y[32];
#pragma unroll
          for (int d = 0; d < 32; ++d) y[d] = rnb(__uint_as_float(v[d]));
          uint32_t o[16];
          if (p < 2) {
            if (args.save && ok) {                                         // rn(p_q) / rn(p_k) for the backward
              uint32_t pw[16];
#pragma unroll
              for (int d = 0; d < 16; ++d) pw[d] = pack_bf16(y[2 * d], y[2 * d + 1]);
              uint4* dst = reinterpret_cast<uint4*>((p == 0 ? PQs : PKs) + ((size_t)n * S + s) * C + h * D);
#pragma unroll
              for (int k = 0; k < 4; ++k) dst[k] = make_uint4(pw[4 * k], pw[4 * k + 1], pw[4 * k + 2], pw[4 * k + 3]);
            }
            float ss = 0.f;
#pragma unroll
            for (int d = 0; d < 32; ++d) ss = fmaf(y[d], y[d], ss);
            const float rr = rsqrtf(ss * (1.f / D) + args.qk_eps);
#pragma unroll
            for (int d = 0; d < 32; ++d) y[d] = rnb(y[d] * rr);
            float z[32];
#pragma unroll
            for (int d = 0; d < 16; ++d) {
              const float c = cs[d], si = sn[d];
              z[d] = y[d] * c - y[d + 16] * si;
              z[d + 16] = y[d + 16] * c + y[d] * si;
            }
#pragma unroll
            for (int d = 0; d < 16; ++d) o[d] = pack_bf16(z[2 * d], z[2 * d + 1]);
          } else {
#pragma unroll
            for (int d = 0; d < 16; ++d) o[d] = pack_bf16(y[2 * d], y[2 * d + 1]);
          }
          if (ok && !(NOSTORE && v[0] != 0x7fc00001u)) {
            __nv_bfloat16* base = p == 3 ? Go + ((size_t)n * S + s) * C + h * D
                                         : (p == 0 ? Qo : p == 1 ? Ko : Vo) + (((size_t)n * H + h) * S + s) * D;
            uint4* dst = reinterpret_cast<uint4*>(base);
#pragma unroll
            for (int k = 0; k < 4; ++k) dst[k] = make_uint4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.qempty[st]);                          // q tile, cos / sin of stage st consumed
    };
    if (ntT > 0) pro(0);
    for (int T = 0; T < ntT; ++T) {
      if (T + 1 < ntT) pro(T + 1);
      epi(T);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
