// swiglu_gemm_pp.cu -- swiglu_gemm as a PING-PONG kernel: each consumer warpgroup owns whole 64 x 256 tiles and the two take
// alternate tiles, so one warpgroup's epilogue (SwiGLU, stmatrix, TMA store) runs while the other's mainloop keeps the tensor
// pipe busy.  The cooperative version ran both epilogues at once with the tensor pipe idle (NCU: 81 % tensor-active, 36 % of
// stalls on the epilogue barriers).  Weight slabs are re-read per 64-row tile instead of per 128, which the measured L2
// headroom (64 % of peak) allows.  SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma256.cuh"

#ifndef NCTA
#define NCTA 132
#endif
#ifndef NSTAGE
#define NSTAGE 4
#endif
#ifndef ORDERED
#define ORDERED 1          // 1: a warpgroup issues its tile's mainloop only after the other finished issuing its previous one
#endif
constexpr int TBM = 64, TBN = 256, TBK = 64, THB = 128;
constexpr int F_A = 8192, F_B = 32768, F_STAGE = F_A + F_B;          // A slab [64][64], B slab [256][64]
constexpr int F_STG = NSTAGE * F_STAGE, F_STGW = 16384;               // per consumer warpgroup: h tile [64 rows][128 cols]
constexpr int F_BAR = F_STG + 2 * F_STGW;
constexpr int SMEM_BYTES = F_BAR + 128;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");

TMN_DEVI float sigmoid_(float a) { float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a)); return fmaf(0.5f, t, 0.5f); }
TMN_DEVI uint64_t dsc_(uint32_t addr, uint32_t lbo, uint32_t sbo) { return smem_desc(addr, lbo, sbo, 1); }
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}
TMN_DEVI void named_bar_arrive(int id, int n) { __syncwarp(); asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }

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
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 1); }
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    uint32_t it = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {   // local tiles in order; consumer warpgroup (local index & 1)
      const int m0 = (t / NT) * TBM, r0 = (t % NT) * TBN;
      for (int kb = 0; kb < KB; ++kb, ++it) {
        const int s = (int)(it % NSTAGE);
        if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
        uint8_t* st = sm + s * F_STAGE;
        mbar_arrive_expect_tx(full + s, F_STAGE);
        tma_load_2d(st, &mA, full + s, kb * TBK, m0);
        tma_load_2d(st + F_A, &mB, full + s, kb * TBK, r0);
      }
    }
    return;
  }
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  uint8_t* stg = sm + F_STG + cw * F_STGW;
  constexpr int BAR_TURN = 3;                             // 3 + cw: "warpgroup cw may issue its next mainloop"
  if (ORDERED && cw == 1) named_bar_arrive(BAR_TURN + 0, 256);
  int lt = 0;                                             // local tile index of this CTA
  for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++lt) {
    if ((lt & 1) != cw) continue;                         // the other warpgroup's tile
    const int m0 = (t / NT) * TBM, h0 = (t % NT) * THB;
    uint32_t it = (uint32_t)lt * KB;                      // this tile's first slot in the producer's sequence
    float acc[128];
#pragma unroll
    for (int e = 0; e < 128; ++e) acc[e] = 0.f;
    if (ORDERED) named_bar_sync(BAR_TURN + cw, 256);
    int prev = -1;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t a0 = su + s * F_STAGE, b0 = a0 + F_A;
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma256_ss<0>(acc, dsc_(a0 + ks * 32, 16, 1024), dsc_(b0 + ks * 32, 16, 1024));
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    if (ORDERED) named_bar_arrive(BAR_TURN + (cw ^ 1), 256);   // every MMA of this tile is issued: the other may go
    wgmma_wait<0>(); fence_regs(acc);
    if (wtid == 0) mbar_arrive(empty + prev);
    // ---------------------------------------------------------------- epilogue: h = silu(a) b, a = columns 0..127, b = 128..255
    if (wtid == 0) tma_store_wait_read<0>();
    named_bar_sync(1 + cw, 128);
    {
      const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < 8; ++gp) {
        uint32_t hp[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int g = 2 * gp + (q >> 1), rb = q & 1, ia = 4 * g + 2 * rb, ib = 4 * (g + 16) + 2 * rb;
          const float x0 = acc[ia], x1 = acc[ia + 1];
          hp[q] = pack_bf16(x0 * sigmoid_(x0) * acc[ib], x1 * sigmoid_(x1) * acc[ib + 1]);
        }
        const int col = 8 * (2 * gp + (mi >> 1));
        stsm_x4(smem_u32(stg) + (col >> 6) * 8192 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2)), hp[0], hp[1], hp[2], hp[3]);
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
      tma_store_2d(&mH, stg, h0, m0);
      tma_store_2d(&mH, stg + 8192, h0 + 64, m0);
      tma_store_commit();
    }
  }
  // balance the turn barriers: the last hand-over of the warpgroup that finished second is consumed here
  if (ORDERED) {
    const int nl = (tiles > (int)blockIdx.x) ? (tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
    const int mine = (nl + (cw == 0 ? 1 : 0)) / 2;        // tiles this warpgroup ran
    const int theirs = nl - mine;
    // each tile run by a warpgroup = one sync on its own turn barrier + one arrive on the other's; the first turn of
    // warpgroup 0 was pre-arrived by warpgroup 1.  Arrivals on my barrier: (theirs) + (cw == 0 ? 1 : 0); syncs: mine.
    if (theirs + (cw == 0 ? 1 : 0) > mine) named_bar_sync(BAR_TURN + cw, 256);
  }
  if (wtid == 0) tma_store_wait_all();
}
