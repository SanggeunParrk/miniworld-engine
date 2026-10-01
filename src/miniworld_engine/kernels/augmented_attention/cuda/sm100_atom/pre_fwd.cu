// pre_fwd.cu — the atom DiT block's attention input stage on sm_100a (per row r of M = A N; the bf16 module's rounding points):
//   xn = rn(LayerNorm(a))  (no affine, eps 1e-5);  x1 = rn(rn(s1 xn) + bi1)   (AdaLN; s1 = rn(sigmoid(sc1)) / bi1 from cond_fwd's mod)
//   Q = rn(x1 Wq^T + bq);  K = rn(x1 Wk^T);  V = rn(x1 Wv^T);  SG = rn(sigmoid(rn(x1 Wg^T)))   all row-major bf16 [M, 128]; optionally
//   x1 saved. The gate is only ever used through its sigmoid (and autograd's sigmoid backward reads that output), so it leaves as one.
// Structure (the SWA qkvg_fwd2.cu frame): 32-row tiles, W = [Wq; Wk; Wv; Wg] (4 x 128 x 128) in TMEM as the bf16 A operand (TMA into the
// idle stages, tcgen05.cp by the MMA warp), p^T = W_p x1^T (M = 128 output channels, N = 32 rows) into a double-buffered set of four
// accumulators, so tile T + 1's projections run under tile T's epilogue. Separate rings (inputs x NI, x1 x 3, outputs x 2) so the loads
// run up to NI tiles ahead.
// Threads: the AdaLN eight threads per row (16 channels each, 16-B loads, two-pass mean / variance by 3-step shuffles); the epilogue
// the accumulators in the mma-fragment layout (tcgen05.ld 16x256b; warpgroup w: rows 16 w ..), rows paired into bf16x2 and
// stored transposed by stmatrix.trans into row-major 128-B-swizzled staging tiles; TMA stores.
// TMEM: W_p at 64 p (256 columns); accumulator set b at 256 + 128 b, projection p at + 32 p.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DIAG
#define DIAG 0
#endif
#ifndef NI
#define NI 4
#endif
constexpr int C = 128;
constexpr int KBLK = 4096;                                                 // [32 rows][64] bf16, SW128
constexpr int T_ = 2 * KBLK;                                               // [32][128] row-major tile (8 KB)
// rings: inputs (a | mod s1 | bi1) x NI, freed as soon as the AdaLN has read them; x1 x 3 (the MMA's B operand, the save's source);
// outputs (Q | K | V | G) x 2, freed when the TMA stores have read them
constexpr int IST = 3 * T_, O_IN = 0, O_X = NI * IST, O_O = O_X + 3 * T_, OST = 4 * T_;
constexpr int O_BAR = O_O + 2 * OST;
constexpr int SMEM = O_BAR + 256;
static_assert(SMEM <= 232448, "shared memory");
static_assert(O_BAR >= 4 * 32768, "the weight units stage through the rings");
constexpr uint32_t T_W = 0, T_ACC = 256;
constexpr uint32_t I_32 = idesc_bf16(128, 32);

struct Bars {
  uint64_t wfull, wfree, infull[NI], infree[NI], xfull[3], ofull[2], ofree[2], accfull[2], accfree[2];
  uint32_t tmem;
};
DEVI float sigm(float x) { return __fdividef(1.f, 1.f + __expf(-x)); }
DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEVI uint32_t bmul2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t badd2(uint32_t a, uint32_t b) { uint32_t r; asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
// 16 lanes x 32 columns, the mma-accumulator fragment: r[4m + 0, 1] = lane t / 4, columns 8 m + 2 (t % 4) + {0, 1}; r[4m + 2, 3] = lane 8 + t / 4
DEVI void tmem_ld16x256b4(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]),
                 "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr));
}
DEVI void stsm4t(uint32_t a, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}
DEVI void tmem_ld16x256b2(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}

extern "C" __global__ void __launch_bounds__(384, 1)
atom_pre_fwd_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mmod,
                   const __grid_constant__ CUtensorMap mQ, const __grid_constant__ CUtensorMap mK, const __grid_constant__ CUtensorMap mV,
                   const __grid_constant__ CUtensorMap mG, const __grid_constant__ CUtensorMap mX, const float* __restrict__ BQ, int ntile,
                   float eps, int save) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto row0 = [&](int T) { return 32 * ((int)blockIdx.x + T * (int)gridDim.x); };

  if (tid == 0) {
    mbar_init(&B.wfull, 1); mbar_init(&B.wfree, 1);
    for (int i = 0; i < NI; ++i) { mbar_init(&B.infull[i], 1); mbar_init(&B.infree[i], 2); }
    for (int i = 0; i < 3; ++i) mbar_init(&B.xfull[i], 2);
    for (int i = 0; i < 2; ++i) { mbar_init(&B.ofull[i], 2); mbar_init(&B.ofree[i], 1); mbar_init(&B.accfull[i], 1); mbar_init(&B.accfree[i], 2); }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      mbar_expect_tx(&B.wfull, 4 * 32768);                                // W_p: K-major SW64 [128][32] atoms, 32 KB per projection
      for (int p = 0; p < 4; ++p)
        for (int ka = 0; ka < 4; ++ka) tma_load_2d(su + p * 32768 + ka * 8192, &mw, &B.wfull, 32 * ka, 128 * p);
      mbar_wait(&B.wfree, 0);
      pdl_wait();                                                          // a / mod come from the previous kernels
      for (int T = 0; T < ntT; ++T) {
        const int s = T % NI, r0 = row0(T);
        if (T >= NI) mbar_wait(&B.infree[s], ((T / NI) - 1) & 1);
        const uint32_t st = su + O_IN + s * IST;
#if DIAG == 2
        mbar_arrive(&B.infull[s]); (void)st; (void)r0;
#else
        mbar_expect_tx(&B.infull[s], 6 * KBLK);
        for (int kb = 0; kb < 2; ++kb) tma_load_2d(st + kb * KBLK, &ma, &B.infull[s], 64 * kb, r0);
        for (int kb = 0; kb < 4; ++kb) tma_load_2d(st + T_ + kb * KBLK, &mmod, &B.infull[s], 64 * kb, r0);   // s1 | bi1
#endif
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    mbar_wait(&B.wfull, 0);
    tc_fence_after();
    if (elect_one()) {
      for (int p = 0; p < 4; ++p)
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + T_W + 64 * p + ks * 8, desc_sw64(su + p * 32768 + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
      tc_commit(&B.wfree);
    }
    __syncwarp();
    for (int T = 0; T < ntT; ++T) {
      const int ab = T & 1, xb = T % 3;
      const uint32_t xa = su + O_X + xb * T_;
      if (T >= 2) mbar_wait(&B.accfree[ab], ((T >> 1) - 1) & 1);
      mbar_wait(&B.xfull[xb], (T / 3) & 1);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int p = 0; p < (DIAG == 3 ? 0 : 4); ++p)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ts(tmem + T_ACC + 128 * ab + 32 * p, tmem + T_W + 64 * p + ks * 8, desc_k128(xa + (ks >> 2) * KBLK) + (uint64_t)((ks & 3) * 2), I_32,
                    ks > 0 ? 1u : 0u);
        tc_commit(&B.accfull[ab]);
      }
      __syncwarp();
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA stores
    if (lane == 0) {
      pdl_wait();
      const CUtensorMap* mo[4] = {&mQ, &mK, &mV, &mG};
      for (int T = 0; T < ntT; ++T) {
        const int ob = T & 1, r0 = row0(T);
        mbar_wait(&B.ofull[ob], (T >> 1) & 1);
#if DIAG == 1
        mbar_arrive(&B.ofree[ob]); continue;
#endif
        for (int p = 0; p < 4; ++p)
          for (int kb = 0; kb < 2; ++kb) tma_store_2d(mo[p], su + O_O + ob * OST + p * T_ + kb * KBLK, 64 * kb, r0);
        if (save)
          for (int kb = 0; kb < 2; ++kb) tma_store_2d(&mX, su + O_X + (T % 3) * T_ + kb * KBLK, 64 * kb, r0);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.ofree[ob]);
      }
      tma_store_wait0();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ compute threads
    const int ct = tid - 128, wg = ct >> 7, qw = warp & 3;
    const uint32_t lb = (uint32_t)qw * 32;
    // AdaLN: row ar = ct / 8 (rows 16 wg ..), 16 channels: 16-B chunk ak of both 64-channel blocks
    const int ar = ct >> 3, ak = ct & 7;
    const uint32_t aoff = sw128((uint32_t)ar, (uint32_t)ak);
    float bq[4];                                                           // the query bias of channel lb + 16 L + 8 h + lane / 4
#pragma unroll
    for (int i = 0; i < 4; ++i) bq[i] = __ldg(BQ + lb + 8 * i + (lane >> 2));
    const uint32_t fj = (uint32_t)(16 * wg + 8 * ((lane >> 3) & 1) + (lane & 7)), fh = (uint32_t)(lane >> 4);
    auto xphase = [&](int T) {
      const int s = T % NI, xb = T % 3;
      const uint32_t st = su + O_IN + s * IST, xs = su + O_X + xb * T_;
      if (T >= 3) mbar_wait(&B.ofree[(T - 3) & 1], ((T - 3) >> 1) & 1);    // tile T - 3's x1 save has left this buffer
      mbar_wait(&B.infull[s], (T / NI) & 1);
#if DIAG == 5
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) { mbar_arrive(&B.xfull[xb]); mbar_arrive(&B.infree[s]); }
      return;
#endif
      uint4 u[2], sc[2], bi[2];
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        u[kb] = lds128(st + kb * KBLK + aoff);
        sc[kb] = lds128(st + T_ + kb * KBLK + aoff);
        bi[kb] = lds128(st + T_ + (2 + kb) * KBLK + aoff);
      }
      float x[16];
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        const uint32_t w4[4] = {u[kb].x, u[kb].y, u[kb].z, u[kb].w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { x[8 * kb + 2 * e] = bf16lo(w4[e]); x[8 * kb + 2 * e + 1] = bf16hi(w4[e]); }
      }
      float sum = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) sum += x[e];
      sum += __shfl_xor_sync(~0u, sum, 1); sum += __shfl_xor_sync(~0u, sum, 2); sum += __shfl_xor_sync(~0u, sum, 4);
      const float mean = sum * (1.f / C);
      float var = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) { const float d = x[e] - mean; var += d * d; }
      var += __shfl_xor_sync(~0u, var, 1); var += __shfl_xor_sync(~0u, var, 2); var += __shfl_xor_sync(~0u, var, 4);
      const float rstd = rsqrtf(var * (1.f / C) + eps);
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        const uint32_t s4[4] = {sc[kb].x, sc[kb].y, sc[kb].z, sc[kb].w}, b4[4] = {bi[kb].x, bi[kb].y, bi[kb].z, bi[kb].w};
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {                                      // bf16x2 multiply / add: the module's single roundings
          const uint32_t xn = pack_bf16((x[8 * kb + 2 * e] - mean) * rstd, (x[8 * kb + 2 * e + 1] - mean) * rstd);
          o[e] = badd2(bmul2(s4[e], xn), b4[e]);
        }
        sts128(xs + kb * KBLK + aoff, make_uint4(o[0], o[1], o[2], o[3]));
      }
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) { mbar_arrive(&B.xfull[xb]); mbar_arrive(&B.infree[s]); }
    };
    auto epi = [&](int T) {
      // warp qw: channels lb .. lb + 31 of all four projections, rows 16 wg .. 16 wg + 15 (the warpgroups split the rows, so the gate's
      // sigmoids are shared). Two 16-lane halves in the mma-fragment layout, rows paired into bf16x2, 8 x 8 blocks stored transposed
      // (stmatrix.trans) into the row-major staging: lane i addresses row fj of the block (channel group fh) it names
      const int ab = T & 1, ob = T & 1;
      mbar_wait(&B.accfull[ab], (T >> 1) & 1);
      if (T >= 2) mbar_wait(&B.ofree[ob], ((T >> 1) - 1) & 1);
      tc_fence_after();
#pragma unroll
      for (int p = 0; p < 4; ++p) {
#pragma unroll
        for (int L = 0; L < 2; ++L) {
          uint32_t v[8], r[4];
          tmem_ld16x256b2(tmem + ((lb + 16 * L) << 16) + T_ACC + 128 * ab + 32 * p + 16 * wg, v);
          tmem_wait_ld();
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) {
            const int k = 4 * (mi & 1) + 2 * (mi >> 1);
            const float b = p == 0 ? bq[2 * L + (mi >> 1)] : 0.f;
            const uint32_t y2 = pack_bf16(__uint_as_float(v[k]) + b, __uint_as_float(v[k + 1]) + b);
            r[mi] = p == 3 ? sig_rn2(y2) : y2;                            // the gate leaves as rn(sigmoid(rn(x1 Wg^T)))
          }
          const uint32_t ch = lb + 16 * L + 8 * fh;
          stsm4t(su + O_O + ob * OST + p * T_ + (ch >> 6) * KBLK + fj * 128u + ((((ch & 63u) >> 3) ^ (fj & 7u)) << 4), r);
        }
      }
      tc_fence_before();
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) { mbar_arrive(&B.accfree[ab]); mbar_arrive(&B.ofull[ob]); }
    };
    if (ntT > 0) xphase(0);
    for (int T = 0; T < ntT; ++T) {
      if (T + 1 < ntT) xphase(T + 1);                                     // tile T + 1's projections run under this tile's epilogue
      epi(T);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
