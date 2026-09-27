// swiglu_gemm.cu -- h = bf16( silu(xn Wa^T) * (xn Wb^T) ), the Transition's expand GEMM with the SwiGLU in its epilogue,
// for the widths where the whole Transition does not fit one kernel (D >= 384: a 64-row warpgroup's output accumulator alone
// would be 192-256 registers).  sm_90a, bf16 operands, fp32 accumulation.  SPDX-License-Identifier: Apache-2.0
//
//   xn [M][K] bf16 (K = D), W1p [2H][K] bf16 packed in 128-row blocks [Wa 128 | Wb 128] so that one 256-wide N tile holds a
//   hidden block's a AND b columns, h [M][H] bf16.  Tile 128 x 256 (two consumer warpgroups x m64n256), K in 64-wide slabs
//   through an NSTAGE-deep TMA ring fed by a producer warpgroup (setmaxnreg 40 / 232), persistent grid, epilogue:
//   SwiGLU from the fp32 accumulator -> stmatrix into a per-warpgroup staging tile -> TMA store.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma256.cuh"
TMN_DEVI void named_bar_arrive(int id, int n) { __syncwarp(); asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }

#ifndef NCTA
#define NCTA 132
#endif
#ifndef STAGGER
#define STAGGER 0         // slabs warpgroup 0 issues before warpgroup 1 starts (one-time de-phasing of the two epilogues)
#endif
#ifndef SIG_TANH
#define SIG_TANH 0
#endif
#ifndef EPI_ABLATE
#define EPI_ABLATE 0
#endif
#ifndef NSTAGE
#define NSTAGE 4
#endif
constexpr int TBM = 128, TBN = 256, TBK = 64, THB = 128;
constexpr int F_A = 16384, F_B = 32768, F_STAGE = F_A + F_B;          // A slab [128][64], B slab [256][64]
constexpr int F_STG = NSTAGE * F_STAGE, F_STGW = 16384;               // per consumer warpgroup: h tile [64 rows][128 cols]
constexpr int F_BAR = F_STG + 2 * F_STGW;
constexpr int SMEM_BYTES = F_BAR + 128;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");

TMN_DEVI float sigmoid_kit(float a) {
#if SIG_TANH
  float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a)); return fmaf(0.5f, t, 0.5f);   // one MUFU
#else
  return rcpf(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a))));                          // two MUFU
#endif
}
TMN_DEVI uint64_t dsc_(uint32_t addr, uint32_t lbo, uint32_t sbo) { return smem_desc(addr, lbo, sbo, 1); }
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
swiglu_gemm(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB,
            const __grid_constant__ CUtensorMap mH, int M, int K, int H) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* empty = full + NSTAGE;
  const int NT = H / THB, tiles = (M / TBM) * NT, KB = K / TBK;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    uint32_t it = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      const int m0 = (t / NT) * TBM, r0 = (t % NT) * TBN;
      for (int kb = 0; kb < KB; ++kb, ++it) {
        const int s = (int)(it % NSTAGE);
        if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
        uint8_t* st = sm + s * F_STAGE;
        mbar_arrive_expect_tx(full + s, F_STAGE);
        tma_load_2d(st, &mA, full + s, kb * TBK, m0);
        tma_load_2d(st + 8192, &mA, full + s, kb * TBK, m0 + 64);
        tma_load_2d(st + F_A, &mB, full + s, kb * TBK, r0);
      }
    }
    return;
  }
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  uint8_t* stg = sm + F_STG + cw * F_STGW;
  uint32_t it = 0;
  constexpr int BAR_GO = 3;
  if (STAGGER > 0 && cw == 1) named_bar_sync(BAR_GO, 256);   // warpgroup 1 starts once warpgroup 0 has issued STAGGER slabs
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int m0 = (t / NT) * TBM, h0 = (t % NT) * THB;
    float acc[128];
#pragma unroll
    for (int e = 0; e < 128; ++e) acc[e] = 0.f;
    int prev = -1;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t a0 = su + s * F_STAGE + cw * 8192, b0 = su + s * F_STAGE + F_A;
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma256_ss<0>(acc, dsc_(a0 + ks * 32, 16, 1024), dsc_(b0 + ks * 32, 16, 1024));
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
      if (STAGGER > 0 && cw == 0 && it == (uint32_t)(STAGGER - 1)) named_bar_arrive(BAR_GO, 256);
    }
    wgmma_wait<0>(); fence_regs(acc);
    if (wtid == 0) mbar_arrive(empty + prev);
    // ---------------------------------------------------------------- epilogue: h = silu(a) b, a = columns 0..127, b = 128..255
    if (wtid == 0) tma_store_wait_read<0>();               // the previous tile's store has read the staging tile
    named_bar_sync(1 + cw, 128);
    {
      const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < 8; ++gp) {
        uint32_t hp[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {                     // q: (rb, g) = (q & 1, 2 gp + (q >> 1))
          const int g = 2 * gp + (q >> 1), rb = q & 1, ia = 4 * g + 2 * rb, ib = 4 * (g + 16) + 2 * rb;
          const float x0 = acc[ia], x1 = acc[ia + 1];
#if EPI_ABLATE
          hp[q] = pack_bf16(x0 + acc[ib], x1 + acc[ib + 1]);   // TIMING ONLY: every accumulator still consumed, no transcendental
#else
          hp[q] = pack_bf16(x0 * sigmoid_kit(x0) * acc[ib], x1 * sigmoid_kit(x1) * acc[ib + 1]);
#endif
        }
        const int col = 8 * (2 * gp + (mi >> 1));
        stsm_x4(smem_u32(stg) + (col >> 6) * 8192 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2)), hp[0], hp[1], hp[2], hp[3]);
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
      tma_store_2d(&mH, stg, h0, m0 + 64 * cw);
      tma_store_2d(&mH, stg + 8192, h0 + 64, m0 + 64 * cw);
      tma_store_commit();
    }
  }
  if (wtid == 0) tma_store_wait_all();
}
