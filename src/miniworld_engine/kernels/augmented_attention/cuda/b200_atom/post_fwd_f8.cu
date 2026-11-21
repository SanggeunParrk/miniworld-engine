#define F8 1   // the e4m3 build (agent E, 2026-11-19): post_fwd.cu's sm_100a half with the F8 sections on; post_fwd.cu itself is untouched
// post_fwd.cu — the atom DiT block's attention output + ConditionedTransition on sm_100a (per row of M = A N; the bf16 module's rounding
// points; so / s2 / bi2 / st from cond_fwd's mod and sg = rn(sigmoid(g)) from pre_fwd, the gates already as rn(sigmoid)):
//   gated = rn(sg o);  u = rn(gated Wo^T);  a2 = rn(a + rn(so u))
//   x2 = rn(rn(s2 rn(LayerNorm(a2))) + bi2);  h = rn(rn(silu(rn(x2 Wa^T))) rn(x2 Wb^T));  t = rn(h Ws^T)
//   a3 = rn(a2 + rn(st t))                          -> a3 [M, 128] bf16; optionally u, a2, x2, t saved for the backward.
// Structure: transposed 16-row tiles (the MMA N is the row count): Wa | Wb (512 x 128) and Ws (128 x 256) live in TMEM as bf16 A operands
// (384 columns), Wo in shared memory (A of u^T = Wo gated^T). The two compute warpgroups take alternate tiles, each with its own TMA
// producer warp, MMA-issuing warp, 3-stage input ring and 64 accumulator columns (u^T; then a^T | b^T of both 128-hidden chunks; then
// t^T), so one warpgroup's math runs while the other's MMAs do.
// Threads: row-wise stages (gated; a2, LayerNorm, x2) eight threads per row, 16 channels each (16-B shared loads, bf16x2 math);
// accumulator stages in the mma-fragment layout (tcgen05.ld 16x256b: lane = channel, column pairs = row pairs), moved between that layout
// and the row-major 128-B-swizzled tiles by ldmatrix / stmatrix .trans. Every activation feeding an MMA is written into a row-major tile
// (the K-major B operand), in place over an input it replaces; the leader thread issues the TMA stores.
// Stage (28 KB): a -> a2 | o -> u | sg -> gated -> x2 | mod: so, s2 (-> h), bi2 (-> a3), st; the t save goes over h's first half.
// TMEM: Wa (hidden 0-127, 128-255) 0 / 64, Wb 128 / 192, Ws 256 (128 columns); warpgroup w's accumulators at 384 + 64 w:
// u^T / a0^T / t^T 0, b0^T 16, a1^T 32, b1^T 48.
// mrows: the rows of the conditioning tables (mod, mkv): M, or N when the samples share one conditioning (table row = r % mrows)
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;
#ifndef TRACE
#define TRACE 0  // development only: %globaltimer phase stamps (g_tr), read by the probe
#endif
#if TRACE
__device__ unsigned long long g_tr[296][32];
#define TRS(slot) do { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_tr[blockIdx.x][slot] = t_; } while (0)
#else
#define TRS(slot) do { } while (0)
#endif

#ifndef RMS
#define RMS 0                              // ESMFold2: RMSNorm statistics (the K / V AdaLN's scale raw, identity-centred by its bias)
#endif
#ifndef RAWSC
#define RAWSC 0                            // ESMFold2 inference (with RMS): the AdaLN as its reference module computes it, ((1 + s) x_n + b) in fp32
#endif                                     // and ONE rounding; the table holds the raw s (bias 0), not rn(1 + s) (which costs s its low bits)

#ifndef DIAG
#define DIAG 0                             // 1: no compute math (handshakes only), 2: no MMAs
#endif
#ifndef PARS
#define PARS 0                             // RoseTTAFold3 / Chai-1 (B200 variant path, 2026-11-17): the parallel schedule, the transition's AdaLN
#endif                                     // normalises the block input a (a3 = a2 + st t with x2 = AdaLN2(LN(a)))
#ifndef UP
#define UP 0                               // Boltz-2 (B200 variant path, 2026-11-17): the up-gated SwiGLU h = silu(x2 Wa^T) (x2 Wb^T) (x2 Wu^T); Wu [256, 128]
#endif                                     // in shared memory (mwup), the hidden in two 128-wide chunks (a | b | u accumulators per chunk)
#ifndef ADA1
#define ADA1 0                             // B200 variant path, round 3b (2026-11-17): x2 = rn(s2 LN(.) + bi2) with ONE rounding (the bf16-autocast
#endif                                     // reference keeps the LayerNorm and the AdaLN's products in fp32)
#ifndef RES1
#define RES1 0                             // ... and the residual updates as fp32 multiply-adds with one rounding each: a2 = rn(a + so u),
#endif                                     // a3 = rn(a2 + st t) (t the unrounded accumulator) -- two bf16 roundings of the output stream less
#ifndef RES3
#define RES3 0                             // ESMFold2 bf16 inference (2026-11-17): only the transition's residual update a3 = rn(a2 + st t) with one rounding
#endif                                     // (t unrounded; the attention's a2 keeps the bf16 product + add: e / tol 1.12 -> 1.03 worst)
#ifndef BOX3
#define BOX3 0                             // round 3 part 10: 3-D tensor maps -- a tile's inputs in 4 boxes (was 14), the weights in 7 (was 26)
#endif
#ifndef NSW
#define NSW (UP ? 2 : 3)                   // input stages per warpgroup
#endif
#ifndef F8ABL
#define F8ABL 0
#endif
#ifndef SQABL
#define SQABL 0
#endif
#ifndef F8
#define F8 0                               // FP8 (agent E, 2026-11-19; opt-in, inference only, ``save`` = 0, the BOX3 non-UP builds): the out projection
#endif                                     // (gated o, Wo) and the SwiGLU expansion (x2, [Wa; Wb]) on tcgen05 kind::f8f6f4: the row threads quantise the MMA's
                                           // B operand (gated, x2) to e4m3 with a dynamic per-row scale (kept in shared memory), the weights are e4m3 with a
                                           // static per-output-channel scale (SWO [128], SWU [512], K axis permuted as pre_fwd.cu), the epilogues multiply
                                           // the fp32 accumulator by sx[row] sw[channel]; h and the squeeze stay bf16
static_assert(!F8 || BOX3, "F8: the BOX3 builds (UP: Wu e4m3 in shared memory, SWUP [256] its channel scales)");
#if BOX3
DEVI void tma_store_3d(const CUtensorMap* m, uint32_t src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];" :: "l"(m), "r"(src), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
#endif
constexpr int R = 16;
constexpr int KB = R * 128;                                                // [16 rows][64] bf16, SW128 (2 KB)
constexpr int T_ = 2 * KB;                                                 // [16][128] row-major tile (4 KB)
constexpr int O_A = 0, O_O = T_, O_G = 2 * T_, O_M = 3 * T_, STG = 7 * T_;  // mod blocks: os 0-1, sc2 2-3, bi2 4-5, ts 6-7
constexpr int O_WO = 2 * NSW * STG, O_WU = O_WO + 32768, O_BAR = O_WU + (UP ? 65536 : 0);
constexpr int O_SX = O_BAR + 512;                                          // F8: row scales [warpgroup][stage][gated | x2][16] fp32
constexpr int SMEM = O_BAR + 512 + (F8 ? 2 * NSW * 2 * 16 * 4 : 0);
static_assert(SMEM <= 232448, "shared memory");
static_assert(O_BAR >= 6 * 32768, "the six weight units stage through the rings and the Wo slot");
constexpr uint32_t T_WU = 0, T_WD = 256, T_ACC = 384;
constexpr uint32_t I_16 = idesc_bf16(128, 16);
constexpr uint32_t I8_16 = idesc_e4m3(128, 16);
DEVI uint32_t cvt_e4m3x2(float x0, float x1) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(x1), "f"(x0));
  return r;
}
// the 16 channels (bf16x2 words w[2][4], channel 8 kb + 2 e + {0, 1}) of a thread -> 16 e4m3 bytes with the row's scale (the eight threads of a row
// share the maximum); sc = max / 448 is the dequantisation factor (1 for an all-zero row)
DEVI uint4 quant_row16(const uint32_t (&w)[2][4], float& sc) {
#if F8ABL
  sc = 1.f;
  return make_uint4(w[0][0] ^ (w[0][1] << 8), w[0][2] ^ (w[0][3] << 8), w[1][0] ^ (w[1][1] << 8), w[1][2] ^ (w[1][3] << 8));   // timing ablation: no amax, no cast
#endif
  float y[16], mx = 0.f;
#pragma unroll
  for (int kb = 0; kb < 2; ++kb)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      y[8 * kb + 2 * e] = __uint_as_float(w[kb][e] << 16);
      y[8 * kb + 2 * e + 1] = __uint_as_float(w[kb][e] & 0xffff0000u);
    }
#pragma unroll
  for (int e = 0; e < 16; ++e) mx = fmaxf(mx, fabsf(y[e]));
  mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, 1)); mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, 2)); mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, 4));
  const float inv = mx > 0.f ? 448.f / mx : 1.f;
  sc = mx > 0.f ? mx * (1.f / 448.f) : 1.f;
  uint32_t o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = cvt_e4m3x2(y[4 * i] * inv, y[4 * i + 1] * inv) | (cvt_e4m3x2(y[4 * i + 2] * inv, y[4 * i + 3] * inv) << 16);
  return make_uint4(o[0], o[1], o[2], o[3]);
}

struct Bars {
  uint64_t wfull, wfree, wofull, infull[2][NSW], infree[2][NSW], stfull[2][NSW], gfull[2], ufull[2], yfull[2], abfull[2], hfull[2], tfull[2], ab2full[2], h0full[2];
  uint32_t tmem;
};
DEVI uint32_t bmul2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t badd2(uint32_t a, uint32_t b) { uint32_t r; asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI float sigm(float x) { return __fdividef(1.f, 1.f + __expf(-x)); }
DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// 16 lanes x 16 columns, the mma-accumulator fragment: r[4m + 0, 1] = lane t / 4, columns 8 m + 2 (t % 4) + {0, 1}; r[4m + 2, 3] = lane 8 + t / 4
DEVI void tmem_ld16x256b2(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
DEVI void stsm4t(uint32_t a, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}
DEVI void ldsm4t(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
atom_post_fwd_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mg,
                    const __grid_constant__ CUtensorMap mmod, const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwu,
                    const __grid_constant__ CUtensorMap mwd, const __grid_constant__ CUtensorMap mout, const __grid_constant__ CUtensorMap mu,
                    const __grid_constant__ CUtensorMap ma2, const __grid_constant__ CUtensorMap mx2, const __grid_constant__ CUtensorMap mt,
                    int ntile, float eps, int save, int mrows
#if UP
                    , const __grid_constant__ CUtensorMap mwup
#endif
#if F8
                    , const float* __restrict__ SWO, const float* __restrict__ SWU
#if UP
                    , const float* __restrict__ SWUP
#endif
#endif
                    ) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto row0 = [&](int T) { return R * ((int)blockIdx.x + T * (int)gridDim.x); };
  auto stage = [&](int w, int i) { return su + (uint32_t)((w * NSW + i % NSW) * STG); };

  if (tid == 0) {
    mbar_init(&B.wfull, 1); mbar_init(&B.wfree, 1); mbar_init(&B.wofull, 1);
    for (int w = 0; w < 2; ++w) {
      for (int s = 0; s < NSW; ++s) { mbar_init(&B.infull[w][s], 1); mbar_init(&B.infree[w][s], 1); mbar_init(&B.stfull[w][s], 1); }
      mbar_init(&B.gfull[w], 1); mbar_init(&B.ufull[w], 1); mbar_init(&B.yfull[w], 1); mbar_init(&B.abfull[w], 1); mbar_init(&B.hfull[w], 1);
      mbar_init(&B.tfull[w], 1); mbar_init(&B.ab2full[w], 1); mbar_init(&B.h0full[w], 1);
    }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (tid == 0) TRS(0);
  pdl_launch();

  if (warp == 0 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA producers (warp 0: warpgroup 0, 2: 1)
    const int w = warp >> 1;
    if (lane == 0) {
      if (w == 0) {
        // Wa | Wb (units 0-3: rows 128 u ..) and Ws (units 4-5: k 128 (u - 4) ..) as K-major SW64 [128][32] atoms into the idle rings
        mbar_expect_tx(&B.wfull, F8 ? 4 * 16384 + 2 * 32768 : 6 * 32768);   // (F8: Wa | Wb units e4m3, 16 KB each)
#if BOX3
        for (int u = 0; u < 6; ++u)                                        // (BOX3: one [4][128][32] box a unit)
          tma_load_3d(su + u * 32768, u < 4 ? &mwu : &mwd, &B.wfull, 0, u < 4 ? 128 * u : 0, u < 4 ? 0 : 4 * (u - 4));
#else
        for (int i = 0; i < 24; ++i) {
          const int u = i >> 2, ka = i & 3;
          tma_load_2d(su + u * 32768 + ka * 8192, u < 4 ? &mwu : &mwd, &B.wfull, u < 4 ? 32 * ka : 128 * (u - 4) + 32 * ka, u < 4 ? 128 * u : 0);
        }
#endif
      }
      mbar_wait(&B.wfree, 0);
      if (w == 0) {
        mbar_expect_tx(&B.wofull, UP ? (F8 ? 3 * 16384 : 3 * 32768) : F8 ? 16384 : 32768);
#if F8
        tma_load_2d(su + O_WO, &mwo, &B.wofull, 0, 0);                    // Wo e4m3: one SW128 box [128 rows][128 B]
#elif BOX3
        tma_load_3d(su + O_WO, &mwo, &B.wofull, 0, 0, 0);
#else
        for (int kb = 0; kb < 2; ++kb) tma_load_2d(su + O_WO + kb * 16384, &mwo, &B.wofull, 64 * kb, 0);
#endif
#if UP
#if F8
        for (int hu = 0; hu < 2; ++hu) tma_load_2d(su + O_WU + hu * 16384, &mwup, &B.wofull, 0, 128 * hu);   // Wu e4m3: [128 rows][128 B] a unit
#else
        for (int hu = 0; hu < 2; ++hu)                                     // Wu: hidden 128 hu .., as Wo (K-major SW128, two 64-wide K blocks)
          for (int kb = 0; kb < 2; ++kb) tma_load_2d(su + O_WU + hu * 32768 + kb * 16384, &mwup, &B.wofull, 64 * kb, 128 * hu);
#endif
#endif
      }
      pdl_wait();                                                          // a / o / g / mod come from the previous kernels
      if (w == 0) TRS(1);
      for (int i = 0; 2 * i + w < ntT; ++i) {
        const int s = i % NSW, r0 = row0(2 * i + w);
        if (i >= NSW) mbar_wait(&B.infree[w][s], ((i / NSW) - 1) & 1);
        const uint32_t st = stage(w, i);
        uint64_t* bar = &B.infull[w][s];
        mbar_expect_tx(bar, 14 * KB);
#if BOX3
        tma_load_3d(st + O_A, &ma, bar, 0, r0, 0);                         // (BOX3: [2][16][64] a / o / g, [8][16][64] of mod: 4 boxes a tile)
        tma_load_3d(st + O_O, &mo, bar, 0, r0, 0);
        tma_load_3d(st + O_G, &mg, bar, 0, r0, 0);
        tma_load_3d(st + O_M, &mmod, bar, 0, r0 % mrows, 4);
#else
        for (int kb = 0; kb < 2; ++kb) {
          tma_load_2d(st + O_A + kb * KB, &ma, bar, 64 * kb, r0);
          tma_load_2d(st + O_O + kb * KB, &mo, bar, 64 * kb, r0);
          tma_load_2d(st + O_G + kb * KB, &mg, bar, 64 * kb, r0);
        }
        for (int kb = 0; kb < 8; ++kb) tma_load_2d(st + O_M + kb * KB, &mmod, bar, 256 + 64 * kb, r0 % mrows);   // os | sc2 | bi2 | ts
#endif
      }
    } else if (lane == 1) {
      // the TMA stores of this warpgroup's tiles: issued (and waited for) off the compute threads, which go on to the next tile's row work
      const int w = warp >> 1;
      pdl_wait();
      for (int i = 0; 2 * i + w < ntT; ++i) {
        const int s = i % NSW, r0 = row0(2 * i + w);
        const uint32_t st = stage(w, i);
        mbar_wait(&B.stfull[w][s], (i / NSW) & 1);
#if BOX3
        tma_store_3d(&mout, st + O_M + 4 * KB, 0, r0, 0);
        if (save) {
          tma_store_3d(&mu, st + O_O, 0, r0, 0);
          tma_store_3d(&ma2, st + O_A, 0, r0, 0);
          tma_store_3d(&mx2, st + O_G, 0, r0, 0);
          tma_store_3d(&mt, st + O_M, 0, r0, 0);
        }
#else
        for (int kb = 0; kb < 2; ++kb) {
          tma_store_2d(&mout, st + O_M + (4 + kb) * KB, 64 * kb, r0);
          if (save) {
            tma_store_2d(&mu, st + O_O + kb * KB, 64 * kb, r0);
            tma_store_2d(&ma2, st + O_A + kb * KB, 64 * kb, r0);
            tma_store_2d(&mx2, st + O_G + kb * KB, 64 * kb, r0);
            tma_store_2d(&mt, st + O_M + kb * KB, 64 * kb, r0);
          }
        }
#endif
        tma_store_commit();
        tma_store_wait_read0();                                            // (a short wait: the stores only have to read the stage)
        mbar_arrive(&B.infree[w][s]);                                      // this tile's stage is free for tile i + NSW
      }
      tma_store_wait0();
      if (w == 0) TRS(10);
    }
  } else if (warp == 1 || warp == 3) {
    // ------------------------------------------------------------------------------------------------ MMA issuers (warp 1: warpgroup 0, 3: 1)
    const int w = warp >> 1;
    if (w == 0) {
      mbar_wait(&B.wfull, 0);
      tc_fence_after();
      if (elect_one()) {
        for (int u = 0; u < 6; ++u) {
#if F8
          if (u < 4) {                                                     // e4m3 units: 4 copies of K = 32
            const uint32_t col = T_WU + 32 * u;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) tmem_cp_128x256b(tmem + col + ks * 8, desc_sw64(su + u * 32768 + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
            continue;
          }
#endif
          const uint32_t col = u < 4 ? T_WU + 64 * u : T_WD + 64 * (u - 4);
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + col + ks * 8, desc_sw64(su + u * 32768 + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
        }
        tc_commit(&B.wfree);
      }
      __syncwarp();
    }
    mbar_wait(&B.wfree, 0);
    mbar_wait(&B.wofull, 0);
    const uint32_t ta = tmem + T_ACC + 64 * w;
    auto bdesc = [&](uint32_t base, int ks) { return desc_k128(base + (ks >> 2) * KB) + (uint64_t)((ks & 3) * 2); };
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const uint32_t st = stage(w, i);
      mbar_wait(&B.gfull[w], i & 1);                                       // gated written over g
      tc_fence_after();
      if (elect_one()) {
#if F8
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) umma8_ss(ta, desc_k128(su + O_WO) + (uint64_t)(ks * 2), desc_k128(st + O_G) + (uint64_t)(ks * 2), I8_16, ks > 0 ? 1u : 0u);
#else
#pragma unroll
        for (int ks = 0; ks < (DIAG == 2 ? 0 : 8); ++ks)
          umma_ss(ta, desc_k128(su + O_WO + (ks >> 2) * 16384) + (uint64_t)((ks & 3) * 2), bdesc(st + O_G, ks), I_16, ks > 0 ? 1u : 0u);
#endif
        tc_commit(&B.ufull[w]);
      }
      __syncwarp();
      mbar_wait(&B.yfull[w], i & 1);                                       // x2 written over gated
      tc_fence_after();
#if UP
      // the up-gated hidden in two 128-wide chunks hc: a_hc -> 0 (Wa unit hc), b_hc -> 16 (unit 2 + hc), u_hc -> 32 (Wu in shared memory);
      // chunk 1 overwrites chunk 0's accumulators once the compute warps have read them (h0full)
      for (int hc = 0; hc < 2; ++hc) {
        if (hc) { mbar_wait(&B.h0full[w], i & 1); tc_fence_after(); }
        if (elect_one()) {
#pragma unroll
#if F8
          for (int ks = 0; ks < 4; ++ks) {
            umma8_ts(ta, tmem + T_WU + 32 * hc + ks * 8, desc_k128(st + O_G) + (uint64_t)(ks * 2), I8_16, ks > 0 ? 1u : 0u);
            umma8_ts(ta + 16, tmem + T_WU + 32 * (2 + hc) + ks * 8, desc_k128(st + O_G) + (uint64_t)(ks * 2), I8_16, ks > 0 ? 1u : 0u);
            umma8_ss(ta + 32, desc_k128(su + O_WU + hc * 16384) + (uint64_t)(ks * 2), desc_k128(st + O_G) + (uint64_t)(ks * 2), I8_16, ks > 0 ? 1u : 0u);
          }
#else
          for (int ks = 0; ks < (DIAG == 2 ? 0 : 8); ++ks) {
            umma_ts(ta, tmem + T_WU + 64 * hc + ks * 8, bdesc(st + O_G, ks), I_16, ks > 0 ? 1u : 0u);
            umma_ts(ta + 16, tmem + T_WU + 64 * (2 + hc) + ks * 8, bdesc(st + O_G, ks), I_16, ks > 0 ? 1u : 0u);
            umma_ss(ta + 32, desc_k128(su + O_WU + hc * 32768 + (ks >> 2) * 16384) + (uint64_t)((ks & 3) * 2), bdesc(st + O_G, ks), I_16,
                    ks > 0 ? 1u : 0u);
          }
#endif
          tc_commit(hc ? &B.ab2full[w] : &B.abfull[w]);
        }
        __syncwarp();
      }
#else
      if (elect_one()) {
#if F8
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
#pragma unroll
          for (int q = 0; q < 4; ++q)
            umma8_ts(ta + 16 * q, tmem + T_WU + 32 * ((q & 1) * 2 + (q >> 1)) + ks * 8, desc_k128(st + O_G) + (uint64_t)(ks * 2), I8_16, ks > 0 ? 1u : 0u);
#else
#pragma unroll
        for (int ks = 0; ks < (DIAG == 2 ? 0 : 8); ++ks)
#pragma unroll
          for (int q = 0; q < 4; ++q)                                      // a0 -> 0 (Wa unit 0), b0 -> 16 (unit 2), a1 -> 32 (unit 1), b1 -> 48 (unit 3)
            umma_ts(ta + 16 * q, tmem + T_WU + 64 * ((q & 1) * 2 + (q >> 1)) + ks * 8, bdesc(st + O_G, ks), I_16, ks > 0 ? 1u : 0u);
#endif
        tc_commit(&B.abfull[w]);
      }
      __syncwarp();
#endif
      mbar_wait(&B.hfull[w], i & 1);                                       // h written over os | sc2
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
#if SQABL                                                                 // timing ablation of an e4m3 squeeze: K = 256 e4m3 steps on the h bytes, garbage values, free quantisation
        for (int ks = 0; ks < 8; ++ks) umma8_ts(ta, tmem + T_WD + ks * 8, desc_k128(st + O_M + (ks >> 2) * KB) + (uint64_t)((ks & 3) * 2), I8_16, ks > 0 ? 1u : 0u);
#else
        for (int ks = 0; ks < (DIAG == 2 ? 0 : 16); ++ks) umma_ts(ta, tmem + T_WD + ks * 8, bdesc(st + O_M, ks), I_16, ks > 0 ? 1u : 0u);
#endif
        tc_commit(&B.tfull[w]);
      }
      __syncwarp();
    }
  } else {
    // ------------------------------------------------------------------------------------------------ compute warpgroups
    const int w = (warp - 4) >> 2, qw = warp & 3, ct = tid & 127;
    const uint32_t lb = (uint32_t)qw * 32;
    const bool leader = qw == 0 && lane == 0;
    const uint32_t ta = tmem + T_ACC + 64 * w;
    // row-wise stages: row ar, 16-B chunk ak of both 64-channel blocks
    const int ar = ct >> 3, ak = ct & 7;
    const uint32_t aoff = sw128((uint32_t)ar, (uint32_t)ak);
    // fragment stages: lane i addresses row j = 8 (mi & 1) + (i & 7) of 8 x 8 block mi = i / 8 (channel group 8 (mi >> 1) of a 16-lane half)
    const uint32_t fj = (uint32_t)(8 * ((lane >> 3) & 1) + (lane & 7)), fh = (uint32_t)(lane >> 4);
    auto faddr = [&](uint32_t base, uint32_t ch) {                          // row fj, channels ch .. ch + 7 of a row-major tile
      return base + (ch >> 6) * KB + fj * 128u + ((((ch & 63u) >> 3) ^ (fj & 7u)) << 4);
    };
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const int s = i % NSW, r0 = row0(2 * i + w);
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][s], (i / NSW) & 1);
      if (leader && w == 0 && i == 0) TRS(2);
      // ---- P1: gated = rn(sg o) over sg
#if F8
      float* const sxr = reinterpret_cast<float*>(sm + O_SX) + ((w * NSW + s) * 2) * 16;   // [gated | x2][16 rows]
      {
        uint32_t gw[2][4];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 g = lds128(st + O_G + kb * KB + aoff), o = lds128(st + O_O + kb * KB + aoff);
          gw[kb][0] = bmul2(g.x, o.x); gw[kb][1] = bmul2(g.y, o.y); gw[kb][2] = bmul2(g.z, o.z); gw[kb][3] = bmul2(g.w, o.w);
        }
        float sc;
        const uint4 q = quant_row16(gw, sc);
        sts128(st + O_G + aoff, q);                                        // gated as e4m3 over g (the first 2 KB)
        if (ak == 0) sxr[ar] = sc;
      }
#else
#pragma unroll
      for (int kb = 0; kb < (DIAG == 1 ? 0 : 2); ++kb) {
        const uint4 g = lds128(st + O_G + kb * KB + aoff), o = lds128(st + O_O + kb * KB + aoff);
        sts128(st + O_G + kb * KB + aoff, make_uint4(bmul2(g.x, o.x), bmul2(g.y, o.y), bmul2(g.z, o.z), bmul2(g.w, o.w)));
      }
#endif
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + w, 128);
      if (leader) mbar_arrive(&B.gfull[w]);
      if (leader && w == 0 && i == 0) TRS(3);
      // ---- P2: u = rn(u^T) -> row-major over o; then row-wise a2, LayerNorm, x2
      mbar_wait(&B.ufull[w], i & 1);
      if (leader && w == 0 && i == 0) TRS(4);
      tc_fence_after();
      if (DIAG != 1) {
        uint32_t v[2][8];                                                  // both 16-channel halves in flight, one wait
#pragma unroll
        for (int L = 0; L < 2; ++L) tmem_ld16x256b2(ta + ((lb + 16 * L) << 16), v[L]);
        tmem_wait_ld();
#pragma unroll
        for (int L = 0; L < 2; ++L) {
          uint32_t r[4];
#if F8
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) {                                 // u = acc sx[row] sw[channel]
            const int k = 4 * (mi & 1) + 2 * (mi >> 1);
            const float sw = __ldg(SWO + lb + 16 * L + (lane >> 2) + 8 * (mi >> 1));
            const float2 sx = *reinterpret_cast<const float2*>(sxr + 2 * (lane & 3) + 8 * (mi & 1));
            r[mi] = pack_bf16(__uint_as_float(v[L][k]) * (sx.x * sw), __uint_as_float(v[L][k + 1]) * (sx.y * sw));
          }
#else
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) r[mi] = pack_bf16(__uint_as_float(v[L][4 * (mi & 1) + 2 * (mi >> 1)]), __uint_as_float(v[L][4 * (mi & 1) + 2 * (mi >> 1) + 1]));
#endif
          stsm4t(faddr(st + O_O, lb + 16 * L + 8 * fh), r);
        }
      }
      tc_fence_before();
      named_bar_sync(1 + w, 128);
      if (DIAG != 1) {
        uint32_t a2w[8];
        float x[16];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 a = lds128(st + O_A + kb * KB + aoff), u = lds128(st + O_O + kb * KB + aoff), os = lds128(st + O_M + kb * KB + aoff);
#if RES1
          {
            const uint32_t aw[4] = {a.x, a.y, a.z, a.w}, uw[4] = {u.x, u.y, u.z, u.w}, ow[4] = {os.x, os.y, os.z, os.w};
#pragma unroll
            for (int e = 0; e < 4; ++e)
              a2w[4 * kb + e] = pack_bf16(bf16lo(aw[e]) + bf16lo(ow[e]) * bf16lo(uw[e]), bf16hi(aw[e]) + bf16hi(ow[e]) * bf16hi(uw[e]));
          }
#else
          a2w[4 * kb + 0] = badd2(a.x, bmul2(os.x, u.x));
          a2w[4 * kb + 1] = badd2(a.y, bmul2(os.y, u.y));
          a2w[4 * kb + 2] = badd2(a.z, bmul2(os.z, u.z));
          a2w[4 * kb + 3] = badd2(a.w, bmul2(os.w, u.w));
#endif
#if PARS                                                                   // the transition normalises the block input a
          const uint32_t a4[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) { x[8 * kb + 2 * e] = bf16lo(a4[e]); x[8 * kb + 2 * e + 1] = bf16hi(a4[e]); }
#endif
        }
#if !PARS
#pragma unroll
        for (int e = 0; e < 8; ++e) { x[2 * e] = bf16lo(a2w[e]); x[2 * e + 1] = bf16hi(a2w[e]); }
#endif
        float sum = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) sum += x[e];
        sum += __shfl_xor_sync(~0u, sum, 1); sum += __shfl_xor_sync(~0u, sum, 2); sum += __shfl_xor_sync(~0u, sum, 4);
        const float mean = RMS ? 0.f : sum * (1.f / 128.f);                       // RMS (ESMFold2): x rsqrt(mean(x^2) + eps)
        float var = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) { const float d = x[e] - mean; var += d * d; }
        var += __shfl_xor_sync(~0u, var, 1); var += __shfl_xor_sync(~0u, var, 2); var += __shfl_xor_sync(~0u, var, 4);
        const float rstd = rsqrtf(var * (1.f / 128.f) + eps);
#if F8
        uint32_t x2w[2][4];
#endif
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 sc = lds128(st + O_M + (2 + kb) * KB + aoff), bi = lds128(st + O_M + (4 + kb) * KB + aoff);
          const uint32_t s4[4] = {sc.x, sc.y, sc.z, sc.w}, b4[4] = {bi.x, bi.y, bi.z, bi.w};
          uint32_t o[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 4 * kb + e;
#if RAWSC
            o[e] = pack_bf16((x[2 * k] - mean) * rstd * (1.f + bf16lo(s4[e])) + bf16lo(b4[e]),
                             (x[2 * k + 1] - mean) * rstd * (1.f + bf16hi(s4[e])) + bf16hi(b4[e]));
#elif ADA1
            o[e] = pack_bf16(bf16lo(s4[e]) * ((x[2 * k] - mean) * rstd) + bf16lo(b4[e]), bf16hi(s4[e]) * ((x[2 * k + 1] - mean) * rstd) + bf16hi(b4[e]));
#else
            const uint32_t xn = pack_bf16((x[2 * k] - mean) * rstd, (x[2 * k + 1] - mean) * rstd);
            o[e] = badd2(bmul2(s4[e], xn), b4[e]);
#endif
          }
#if F8
          x2w[kb][0] = o[0]; x2w[kb][1] = o[1]; x2w[kb][2] = o[2]; x2w[kb][3] = o[3];
#else
          sts128(st + O_G + kb * KB + aoff, make_uint4(o[0], o[1], o[2], o[3]));
#endif
          sts128(st + O_A + kb * KB + aoff, make_uint4(a2w[4 * kb], a2w[4 * kb + 1], a2w[4 * kb + 2], a2w[4 * kb + 3]));
        }
#if F8
        {
          float sc;
          const uint4 q = quant_row16(x2w, sc);                            // x2 as e4m3 over gated
          sts128(st + O_G + aoff, q);
          if (ak == 0) sxr[16 + ar] = sc;
        }
#endif
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (leader) mbar_arrive(&B.yfull[w]);
      if (leader && w == 0 && i == 0) TRS(5);
      // ---- P3: h = rn(rn(silu(rn(a))) rn(b)) for hidden 128 hc + lb + 16 L + .., row-major over os | sc2
#if UP
      // (UP: h = rn(rn(rn(silu(rn(a))) rn(b)) rn(u)), one 128-wide chunk at a time: a | b | u accumulators at 0 | 16 | 32)
      for (int hc = 0; hc < 2; ++hc) {
        mbar_wait(hc ? &B.ab2full[w] : &B.abfull[w], i & 1);
        tc_fence_after();
        uint32_t va[2][8], vb[2][8], vu[2][8];
#pragma unroll
        for (int L = 0; L < 2; ++L) {
          tmem_ld16x256b2(ta + ((lb + 16 * L) << 16), va[L]);
          tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 16, vb[L]);
          tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 32, vu[L]);
        }
        tmem_wait_ld();
        if (hc == 0) {                                                      // the accumulators are read: chunk 1's MMAs may overwrite them
          tc_fence_before();
          named_bar_sync(1 + w, 128);
          if (leader) mbar_arrive(&B.h0full[w]);
        }
#pragma unroll
        for (int L = 0; L < 2; ++L) {
          uint32_t r[4];
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) {
            const int k = 4 * (mi & 1) + 2 * (mi >> 1);
#if F8
            const int hid = 128 * hc + lb + 16 * L + (lane >> 2) + 8 * (mi >> 1);
            const float2 sx = *reinterpret_cast<const float2*>(sxr + 16 + 2 * (lane & 3) + 8 * (mi & 1));
            const float sa = __ldg(SWU + hid), sb = __ldg(SWU + 256 + hid), su_ = __ldg(SWUP + hid);
            r[mi] = bmul2(bmul2(silu_rn2(pack_bf16(__uint_as_float(va[L][k]) * (sx.x * sa), __uint_as_float(va[L][k + 1]) * (sx.y * sa))),
                                pack_bf16(__uint_as_float(vb[L][k]) * (sx.x * sb), __uint_as_float(vb[L][k + 1]) * (sx.y * sb))),
                          pack_bf16(__uint_as_float(vu[L][k]) * (sx.x * su_), __uint_as_float(vu[L][k + 1]) * (sx.y * su_)));
#else
            r[mi] = bmul2(bmul2(silu_rn2(pack_bf16(__uint_as_float(va[L][k]), __uint_as_float(va[L][k + 1]))),
                                pack_bf16(__uint_as_float(vb[L][k]), __uint_as_float(vb[L][k + 1]))),
                          pack_bf16(__uint_as_float(vu[L][k]), __uint_as_float(vu[L][k + 1])));
#endif
          }
          stsm4t(faddr(st + O_M, 128 * hc + lb + 16 * L + 8 * fh), r);
        }
      }
      if (false) {
#else
      mbar_wait(&B.abfull[w], i & 1);
      if (leader && w == 0 && i == 0) TRS(6);
      tc_fence_after();
      if (DIAG != 1) {
#endif
        uint32_t va[2][2][8], vb[2][2][8];                                 // all eight accumulator reads in flight, one wait
#pragma unroll
        for (int hc = 0; hc < 2; ++hc)
#pragma unroll
          for (int L = 0; L < 2; ++L) {
            tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 32 * hc, va[hc][L]);
            tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 32 * hc + 16, vb[hc][L]);
          }
        tmem_wait_ld();
#pragma unroll
        for (int hc = 0; hc < 2; ++hc)
#pragma unroll
          for (int L = 0; L < 2; ++L) {
            uint32_t r[4];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
              const int k = 4 * (mi & 1) + 2 * (mi >> 1);
#if F8
              const int hid = 128 * hc + lb + 16 * L + (lane >> 2) + 8 * (mi >> 1);   // hidden channel: a = Wa row hid, b = Wb row 256 + hid
              const float2 sx = *reinterpret_cast<const float2*>(sxr + 16 + 2 * (lane & 3) + 8 * (mi & 1));
              const float sa = __ldg(SWU + hid), sb = __ldg(SWU + 256 + hid);
              r[mi] = bmul2(silu_rn2(pack_bf16(__uint_as_float(va[hc][L][k]) * (sx.x * sa), __uint_as_float(va[hc][L][k + 1]) * (sx.y * sa))),
                            pack_bf16(__uint_as_float(vb[hc][L][k]) * (sx.x * sb), __uint_as_float(vb[hc][L][k + 1]) * (sx.y * sb)));
#else
              r[mi] = bmul2(silu_rn2(pack_bf16(__uint_as_float(va[hc][L][k]), __uint_as_float(va[hc][L][k + 1]))),
                            pack_bf16(__uint_as_float(vb[hc][L][k]), __uint_as_float(vb[hc][L][k + 1])));
#endif
            }
            stsm4t(faddr(st + O_M, 128 * hc + lb + 16 * L + 8 * fh), r);
          }
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + w, 128);
      if (leader) mbar_arrive(&B.hfull[w]);
      if (leader && w == 0 && i == 0) TRS(7);
      // ---- P4: a3 = rn(a2 + rn(rn(sigmoid(ts)) rn(t))) in the fragment layout (ts, a2 by ldmatrix.trans) -> row-major over bi2
      mbar_wait(&B.tfull[w], i & 1);
      if (leader && w == 0 && i == 0) TRS(8);
      tc_fence_after();
      if (DIAG != 1) {
        uint32_t v[2][8], ts[2][4], a2[2][4];                               // both halves' reads in flight, one wait
#pragma unroll
        for (int L = 0; L < 2; ++L) {
          const uint32_t ch = lb + 16 * L + 8 * fh;
          tmem_ld16x256b2(ta + ((lb + 16 * L) << 16), v[L]);
          ldsm4t(faddr(st + O_M + 6 * KB, ch), ts[L]);
          ldsm4t(faddr(st + O_A, ch), a2[L]);
        }
        tmem_wait_ld();
#pragma unroll
        for (int L = 0; L < 2; ++L) {
          const uint32_t ch = lb + 16 * L + 8 * fh;
          uint32_t tv[4], r[4];
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) {
            const int k = 4 * (mi & 1) + 2 * (mi >> 1);
            tv[mi] = pack_bf16(__uint_as_float(v[L][k]), __uint_as_float(v[L][k + 1]));
#if RES1 || RES3
            r[mi] = pack_bf16(bf16lo(a2[L][mi]) + bf16lo(ts[L][mi]) * __uint_as_float(v[L][k]), bf16hi(a2[L][mi]) + bf16hi(ts[L][mi]) * __uint_as_float(v[L][k + 1]));
#else
            r[mi] = badd2(a2[L][mi], bmul2(ts[L][mi], tv[mi]));
#endif
          }
          stsm4t(faddr(st + O_M + 4 * KB, ch), r);
          if (save) stsm4t(faddr(st + O_M, ch), tv);                        // t over h's first half (M3 has read it)
        }
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + w, 128);
      if (leader) mbar_arrive(&B.stfull[w][s]);                            // the store thread of this warpgroup takes it from here
      if (leader && w == 0 && i == 0) TRS(9);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
  if (tid == 0) TRS(19);
}

