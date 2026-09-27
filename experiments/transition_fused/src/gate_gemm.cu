// gate_gemm.cu -- the Transition backward's gate stage for the wide widths, as ONE kernel:
//   ab = xn [Wa;Wb]^T  and  dh = bf16(dy Ws)      (two GEMMs over K = D, sharing the output tile position)
//   s = sigmoid(a), l = a s:   h = bf16(l b),  dA = bf16((dh b)(s + l (1 - s))),  dB = bf16(dh l)
// SPDX-License-Identifier: Apache-2.0
// The engine runs dh as its own cuBLAS GEMM (M x H through HBM and back) and the gate as a Triton kernel at ~20-30 % of tensor
// peak -- together ~45 % of its backward at D = 256.  Tile 128 rows x 64 hidden (both GEMMs, two consumer warpgroups of 64
// rows), K in 64-wide slabs {xn, dy, W1p (packed [Wa 64 | Wb 64]), Ws^T} through a 3-stage TMA ring from a producer
// warpgroup (setmaxnreg 40 / 232), gate in the fp32 accumulator, h / dA / dB staged with stmatrix and TMA-stored into
// h [M][H] and dAB [M][2H] (dA columns 0..H-1, dB H..2H-1: the layout the dW and d_xn GEMMs take).
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma_n.cuh"

#ifndef NSTAGE
#define NSTAGE 3
#endif
#ifndef EPI_ABLATE
#define EPI_ABLATE 0
#endif
#ifndef NO_STORE
#define NO_STORE 0
#endif
#ifndef STAGGER
#define STAGGER 0                                         // warpgroup 1 starts STAGGER slabs behind warpgroup 0
#endif
#ifndef SIG_TANH
#define SIG_TANH 1
#endif
#ifndef TBK
#define TBK 64                                            // K slab: 64 (128-B swizzle) or 32 (64-B swizzle, twice the stages)
#endif
constexpr int TBM = 128, THB = 64;
constexpr int RB = TBK * 2, HALF = 64 * RB, SW_LAYOUT = TBK == 64 ? 1 : 2, SBO = 8 * RB;   // row bytes, 64-row block
constexpr int F_XN = 0, F_DY = 2 * HALF, F_W1 = 4 * HALF, F_WS = 6 * HALF, F_STAGE = 7 * HALF;   // xn, dy [128][TBK]; W1p [128][TBK]; Ws^T [64][TBK]
constexpr int F_STG = NSTAGE * F_STAGE, F_STGW = 24576;                 // per consumer warpgroup: h | dA | dB, each [64 rows][64 cols]
constexpr int F_BAR = F_STG + 2 * F_STGW;
constexpr int SMEM_BYTES = F_BAR + 128;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");

TMN_DEVI float sigmoid_(float a) {
#if SIG_TANH
  float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a)); return fmaf(0.5f, t, 0.5f);
#else
  return rcpf(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a))));
#endif
}
TMN_DEVI void named_bar_arrive(int id, int n) { __syncwarp(); asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
TMN_DEVI uint64_t dsc_(uint32_t addr, uint32_t lbo, uint32_t sbo) { return smem_desc(addr, lbo, sbo, SW_LAYOUT); }
TMN_DEVI void mma_n64(float (&d)[32], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 0;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b));
}
#ifndef ARRIVE_CTA
#define ARRIVE_CTA 0
#endif
#ifndef MCA
#define MCA 0                                             // 1: 2-CTA cluster on adjacent hidden tiles of one row tile; rank 0
#endif                                                    //    multicasts the xn slab into both CTAs, rank 1 the dy slab
TMN_DEVI uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
TMN_DEVI void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
TMN_DEVI void mbar_arrive_peer(uint64_t* bar, uint32_t peer) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_u32(bar)), "r"(peer));
#if ARRIVE_CTA                                            // CUTLASS ClusterBarrier::arrive(cta_id): default .release.cta, no MEMBAR
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
#else
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
#endif
}
TMN_DEVI void tma_load_2d_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;\n"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
gate_gemm(const __grid_constant__ CUtensorMap mXN, const __grid_constant__ CUtensorMap mDY, const __grid_constant__ CUtensorMap mW1,
          const __grid_constant__ CUtensorMap mWS, const __grid_constant__ CUtensorMap mH, const __grid_constant__ CUtensorMap mDAB,
          int M, int K, int H) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* empty = full + NSTAGE;
  const int KB = K / TBK;
#if MCA
  const uint32_t rank = cluster_rank(), peer = rank ^ 1u;
  const int NT = H / THB / 2, tiles = (M / TBM) * NT, t0 = blockIdx.x >> 1, tstep = gridDim.x >> 1;
#define TILE_N(t) (2 * ((t) % NT) + (int)rank)
#else
  const uint32_t peer = 0;
  const int NT = H / THB, tiles = (M / TBM) * NT, t0 = blockIdx.x, tstep = gridDim.x;
#define TILE_N(t) ((t) % NT)
#endif
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2 * (1 + MCA)); }
    fence_barrier_init();
  }
  __syncthreads();
#if MCA
  cluster_sync_all();
#endif
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid == 0) {
    uint32_t it = 0;
    for (int t = t0; t < tiles; t += tstep) {
      const int m0 = (t / NT) * TBM, n = TILE_N(t);
      for (int kb = 0; kb < KB; ++kb, ++it) {
        const int s = (int)(it % NSTAGE);
        if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
        uint8_t* st = sm + s * F_STAGE;
        mbar_arrive_expect_tx(full + s, F_STAGE);
        for (int h = 0; h < 2; ++h) {
#if MCA
          if (rank == 0) tma_load_2d_mc(st + F_XN + h * HALF, &mXN, full + s, kb * TBK, m0 + 64 * h, (uint16_t)0x3);
          else           tma_load_2d_mc(st + F_DY + h * HALF, &mDY, full + s, kb * TBK, m0 + 64 * h, (uint16_t)0x3);
#else
          tma_load_2d(st + F_XN + h * HALF, &mXN, full + s, kb * TBK, m0 + 64 * h);
          tma_load_2d(st + F_DY + h * HALF, &mDY, full + s, kb * TBK, m0 + 64 * h);
#endif
        }
        tma_load_2d(st + F_W1, &mW1, full + s, kb * TBK, n * 128);
        tma_load_2d(st + F_WS, &mWS, full + s, kb * TBK, n * THB);
      }
    }
    }
  } else {
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  uint8_t* stg = sm + F_STG + cw * F_STGW;
  uint32_t it = 0;
  if (STAGGER > 0 && cw == 1) named_bar_sync(3, 256);
  for (int t = t0; t < tiles; t += tstep) {
    const int m0 = (t / NT) * TBM, n = TILE_N(t);
    float AB[64], DH[32];
#pragma unroll
    for (int e = 0; e < 64; ++e) AB[e] = 0.f;
#pragma unroll
    for (int e = 0; e < 32; ++e) DH[e] = 0.f;
    int prev = -1;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t st = su + s * F_STAGE;
      fence_regs(AB); fence_regs(DH); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < TBK / 16; ++ks) {
        mma_n128(AB, dsc_(st + F_XN + cw * HALF + ks * 32, 16, SBO), dsc_(st + F_W1 + ks * 32, 16, SBO));
        mma_n64(DH, dsc_(st + F_DY + cw * HALF + ks * 32, 16, SBO), dsc_(st + F_WS + ks * 32, 16, SBO));
      }
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(AB); fence_regs(DH); if (wtid == 0) { mbar_arrive(empty + prev); if (MCA) mbar_arrive_peer(empty + prev, peer); } }
      prev = s;
      if (STAGGER > 0 && cw == 0 && it == (uint32_t)(STAGGER - 1)) named_bar_arrive(3, 256);
    }
    wgmma_wait<0>(); fence_regs(AB); fence_regs(DH);
    if (wtid == 0) { mbar_arrive(empty + prev); if (MCA) mbar_arrive_peer(empty + prev, peer); }
    // ---------------------------------------------------------------- gate: C group g of DH / of AB (a: g, b: g + 8)
    if (wtid == 0) tma_store_wait_read<0>();
    named_bar_sync(1 + cw, 128);
    {
      const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < 4; ++gp) {
        uint32_t hp[4], dap[4], dbp[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int g = 2 * gp + (q >> 1), rb = q & 1, ia = 4 * g + 2 * rb, ib = 4 * (g + 8) + 2 * rb, id = 4 * g + 2 * rb;
          const uint32_t dhp = pack_bf16(DH[id], DH[id + 1]);                    // dh rounded to bf16 first (the contract)
          const float g0 = bf16lo(dhp), g1 = bf16hi(dhp);
          const float a0 = AB[ia], a1 = AB[ia + 1], b0 = AB[ib], b1 = AB[ib + 1];
#if EPI_ABLATE        // TIMING ONLY: every accumulator consumed and every store kept, no gate math
          hp[q] = pack_bf16(a0 + b0, a1 + b1); dap[q] = pack_bf16(g0, g1); dbp[q] = pack_bf16(b0, b1);
#else
          const float s0 = sigmoid_(a0), s1 = sigmoid_(a1), l0 = a0 * s0, l1 = a1 * s1;
          hp[q] = pack_bf16(l0 * b0, l1 * b1);
          dap[q] = pack_bf16((g0 * b0) * (s0 + l0 * (1.f - s0)), (g1 * b1) * (s1 + l1 * (1.f - s1)));
          dbp[q] = pack_bf16(g0 * l0, g1 * l1);
#endif
        }
        const int col = 8 * (2 * gp + (mi >> 1));
        const uint32_t off = swz128((uint32_t)mrow, (uint32_t)(col * 2));
        stsm_x4(smem_u32(stg) + off, hp[0], hp[1], hp[2], hp[3]);
        stsm_x4(smem_u32(stg) + 8192 + off, dap[0], dap[1], dap[2], dap[3]);
        stsm_x4(smem_u32(stg) + 16384 + off, dbp[0], dbp[1], dbp[2], dbp[3]);
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0 && (!NO_STORE || m0 < 0)) {           // NO_STORE: TIMING ONLY (staging still written, TMA store skipped)
      const int r = m0 + 64 * cw;
      tma_store_2d(&mH, stg, n * THB, r);
      tma_store_2d(&mDAB, stg + 8192, n * THB, r);
      tma_store_2d(&mDAB, stg + 16384, H + n * THB, r);
      tma_store_commit();
    }
  }
  if (wtid == 0) tma_store_wait_all();
  }
#if MCA
  __syncwarp();
  cluster_sync_all();
#endif
}
