// squeeze_gemm.cu -- out = bf16( h Ws^T + x ), the Transition's squeeze GEMM with the residual in its epilogue (wide widths).
// SPDX-License-Identifier: Apache-2.0
//   h [M][K] bf16 (K = 4D), Ws [D][K] bf16 row-major (already K-major for the B operand), x / out [M][D] bf16.
//   Tile 128 x SQ_BN (two consumer warpgroups x m64nBN; SQ_BN divides D), K in 64-wide slabs through an NSTAGE ring fed by a
//   producer warpgroup, persistent grid.  The residual x tile rides the same producer: loaded into the output staging tile
//   at the start of each tile, added in the fragment layout (ldmatrix), stored back in place (stmatrix), TMA-stored.
#ifndef WAIT0
#define WAIT0 0
#endif
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma_n.cuh"

#ifndef SQ_BN
#define SQ_BN 256
#endif
#ifndef NSTAGE
#define NSTAGE (SQ_BN == 256 ? 3 : 4)
#endif
#define MMA_N_(n) mma_n##n
#define MMA_N(n) MMA_N_(n)
constexpr int TBM = 128, TBK = 64, NACC = SQ_BN / 2, NQ = SQ_BN / 64;      // NQ: 64-column quarters of a SQ_BN tile
constexpr int F_A = 16384, F_B = SQ_BN * 128, F_STAGE = F_A + F_B;      // A slab [128][64], B slab [SQ_BN][64]
constexpr int F_STG = NSTAGE * F_STAGE, F_STGW = 64 * SQ_BN * 2;        // per consumer warpgroup: x / out tile [64 rows][SQ_BN]
constexpr int F_BAR = F_STG + 2 * F_STGW;
constexpr int SMEM_BYTES = F_BAR + 128;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");

TMN_DEVI uint64_t dsc_(uint32_t addr, uint32_t lbo, uint32_t sbo) { return smem_desc(addr, lbo, sbo, 1); }
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
squeeze_gemm(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB,
             const __grid_constant__ CUtensorMap mX, const __grid_constant__ CUtensorMap mO, int M, int K, int D) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* empty = full + NSTAGE;
  uint64_t* xfull = empty + NSTAGE;                        // [2]: the residual tile of each consumer warpgroup has landed
  uint64_t* xfree = xfull + 2;                             // [2]: that warpgroup's staging tile may be refilled
  const int NT = D / SQ_BN, tiles = (M / TBM) * NT, KB = K / TBK;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    for (int c = 0; c < 2; ++c) { mbar_init(xfull + c, 1); mbar_init(xfree + c, 1); }
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    uint32_t it = 0, tl = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tl) {
      const int m0 = (t / NT) * TBM, n0 = (t % NT) * SQ_BN;
      for (int c = 0; c < 2; ++c) {                         // residual tiles: into each warpgroup's staging once it is free
        if (tl > 0) mbar_wait(xfree + c, (tl - 1) & 1u);
        uint8_t* xs = sm + F_STG + c * F_STGW;
        mbar_arrive_expect_tx(xfull + c, F_STGW);
        for (int q = 0; q < NQ; ++q) tma_load_2d(xs + q * 8192, &mX, xfull + c, n0 + 64 * q, m0 + 64 * c);
      }
      for (int kb = 0; kb < KB; ++kb, ++it) {
        const int s = (int)(it % NSTAGE);
        if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
        uint8_t* st = sm + s * F_STAGE;
        mbar_arrive_expect_tx(full + s, F_STAGE);
        tma_load_2d(st, &mA, full + s, kb * TBK, m0);
        tma_load_2d(st + 8192, &mA, full + s, kb * TBK, m0 + 64);
        for (int r = 0; r < SQ_BN; r += 256)                  // TMA boxes are at most 256 rows
          tma_load_2d(st + F_A + r * 128, &mB, full + s, kb * TBK, n0 + r);
      }
    }
    return;
  }
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  uint8_t* stg = sm + F_STG + cw * F_STGW;
  uint32_t it = 0, tl = 0;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tl) {
    const int m0 = (t / NT) * TBM, n0 = (t % NT) * SQ_BN;
    float acc[NACC];
#pragma unroll
    for (int e = 0; e < NACC; ++e) acc[e] = 0.f;
    int prev = -1;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t a0 = su + s * F_STAGE + cw * 8192, b0 = su + s * F_STAGE + F_A;
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) MMA_N(SQ_BN)(acc, dsc_(a0 + ks * 32, 16, 1024), dsc_(b0 + ks * 32, 16, 1024));
      wgmma_commit();
#if WAIT0                                                 // release this slab as soon as its MMAs retire
      wgmma_wait<0>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + s);
#else
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
#endif
    }
    wgmma_wait<0>(); fence_regs(acc);
#if !WAIT0
    if (wtid == 0) mbar_arrive(empty + prev);
#endif
    // ---------------------------------------------------------------- epilogue: out = bf16(x + acc), in place over the staged x
    mbar_wait(xfull + cw, tl & 1u);
    {
      const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < SQ_BN / 16; ++gp) {
        const int col = 8 * (2 * gp + (mi >> 1));
        const uint32_t ad = smem_u32(stg) + (col >> 6) * 8192 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2));
        uint32_t xr[4];
        ldsm_x4(xr, ad);
        const int g0 = 2 * gp, g1 = 2 * gp + 1;
        stsm_x4(ad, pack_bf16(bf16lo(xr[0]) + acc[4 * g0 + 0], bf16hi(xr[0]) + acc[4 * g0 + 1]),
                    pack_bf16(bf16lo(xr[1]) + acc[4 * g0 + 2], bf16hi(xr[1]) + acc[4 * g0 + 3]),
                    pack_bf16(bf16lo(xr[2]) + acc[4 * g1 + 0], bf16hi(xr[2]) + acc[4 * g1 + 1]),
                    pack_bf16(bf16lo(xr[3]) + acc[4 * g1 + 2], bf16hi(xr[3]) + acc[4 * g1 + 3]));
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
      for (int q = 0; q < NQ; ++q) tma_store_2d(&mO, stg + q * 8192, n0 + 64 * q, m0 + 64 * cw);
      tma_store_commit();
      tma_store_wait_read<0>();                            // the store has read the staging tile: the producer may refill it
      mbar_arrive(xfree + cw);
    }
  }
  if (wtid == 0) tma_store_wait_all();
}
