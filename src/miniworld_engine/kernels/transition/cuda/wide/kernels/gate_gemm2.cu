// gate_gemm2.cu -- gate_gemm with a 128-row x 128-hidden tile: each consumer warpgroup holds a|b as one m64n256 chain and dh
// as an m64n128 chain (192 fp32 accumulator registers, setmaxnreg 24 / 240).  Operand intensity 77 FLOP per shared-memory byte
// against the 64-hidden tile's 56, which is what bounds the latter's mainloop at ~52 % of tensor peak even with no epilogue.
// SPDX-License-Identifier: Apache-2.0
// Same contract as gate_gemm.cu: h [M][H], dAB [M][2H]; W1p packed in 128-blocks [Wa 128 | Wb 128]; Ws^T [H][D].
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma_n.cuh"

#ifndef WAIT0
#define WAIT0 0
#endif
#ifndef NSTAGE
#define NSTAGE 3
#endif
#ifndef TBK
#define TBK 32
#endif
#ifndef REG_C
#define REG_C 240
#endif
constexpr int TBM = 128, THB = 128;
constexpr int RB = TBK * 2, B64 = 64 * RB, SW_LAYOUT = TBK == 64 ? 1 : 2, SBO = 8 * RB;
#if MW_GATE_SAVED
constexpr int F_XN = 0, F_DY = 0, F_W1 = 0, F_WS = 2 * B64, F_STAGE = 4 * B64;
#define GATE_FENCE_AB(x) ((void)0)
#elif MW_EXPAND_ONLY
constexpr int F_XN = 0, F_DY = 0, F_W1 = 2 * B64, F_WS = 0, F_STAGE = 6 * B64;
#define GATE_FENCE_AB(x) fence_regs(x)
#else
constexpr int F_XN = 0, F_DY = 2 * B64, F_W1 = 4 * B64, F_WS = 8 * B64, F_STAGE = 10 * B64;   // xn, dy [128][TBK]; W1p [256][TBK]; Ws^T [128][TBK]
#define GATE_FENCE_AB(x) fence_regs(x)
#endif
#if MW_EXPAND_ONLY
#define GATE_FENCE_DH(x) ((void)0)
#else
#define GATE_FENCE_DH(x) fence_regs(x)
#endif
#ifndef STG_HALF
#define STG_HALF 0                                        // 1: stage and store the epilogue in two 64-column passes (half the staging smem)
#endif
constexpr int NPASS = STG_HALF ? 2 : 1, CPP = 2 / NPASS;                // passes, 64-column blocks per pass
constexpr int F_STG = NSTAGE * F_STAGE, F_STGW = 3 * CPP * 8192;       // per consumer WG: {h, dA, dB} x CPP blocks of [64][64]
#if MW_GATE_SAVED_TMA && MW_GATE_SAVED
// Two saved A/B tiles overlap the DH weight ring. Each tile owns both
// consumer warpgroups' [64,128] operands until both have read the epilogue.
constexpr int AB_TILE = 65536, F_AB = F_STG + 2 * F_STGW;
constexpr int F_BAR = F_AB + 2 * AB_TILE;
#else
constexpr int F_BAR = F_STG + 2 * F_STGW;
#endif
constexpr int SMEM_BYTES = F_BAR + 128;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");

TMN_DEVI float sigmoid_(float a) { float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a)); return fmaf(0.5f, t, 0.5f); }
TMN_DEVI uint64_t dsc_(uint32_t addr) { return smem_desc(addr, 16, SBO, SW_LAYOUT); }
#ifndef NO_H
#define NO_H 0
#endif
#ifndef MCW
#define MCW 0                                             // 1: 2-CTA cluster on adjacent 128-row blocks of one hidden tile; each rank
#endif                                                    //    loads half of every weight slab (W1 box rank, Ws rows 64 rank) multicast
TMN_DEVI uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
TMN_DEVI void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
TMN_DEVI void mbar_arrive_peer(uint64_t* bar, uint32_t peer) {    // CUTLASS ClusterBarrier::arrive(cta_id): default .release.cta
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_u32(bar)), "r"(peer));
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
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
#if MCW
__cluster_dims__(2, 1, 1)
#endif
gate_gemm(const __grid_constant__ CUtensorMap mXN, const __grid_constant__ CUtensorMap mDY, const __grid_constant__ CUtensorMap mW1,
          const __grid_constant__ CUtensorMap mWS, const __grid_constant__ CUtensorMap mH, const __grid_constant__ CUtensorMap mDAB,
          int M, int K, int H
#if MW_GATE_SAVED
          ,const __nv_bfloat16* saved_a,const __nv_bfloat16* saved_b
#endif
#if MW_EXPAND_ONLY
          ,const __grid_constant__ CUtensorMap mA,const __grid_constant__ CUtensorMap mB
#endif
          ) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* empty = full + NSTAGE;
#if MW_GATE_SAVED_TMA && MW_GATE_SAVED
  uint64_t* ab_full = empty + NSTAGE;
  uint64_t* ab_empty = ab_full + 2;
  static_assert((2*NSTAGE+4)*sizeof(uint64_t)<=128,"saved-operand barriers");
#endif
#if MCW
  uint64_t* ready = empty + NSTAGE;
  static_assert(3 * NSTAGE * sizeof(uint64_t) <= 128, "cluster pipeline barrier area");
#endif
  const int NT = H / THB, KB = K / TBK;
#if MCW
  const uint32_t rank = cluster_rank(), peer = rank ^ 1u;
  const int tiles = (M / TBM / 2) * NT, t0 = blockIdx.x >> 1, tstep = gridDim.x >> 1;
#define TILE_M0(t) ((2 * ((t) / NT) + (int)rank) * TBM)
#else
  const uint32_t peer = 0;
  const int tiles = (M / TBM) * NT, t0 = blockIdx.x, tstep = gridDim.x;
#define TILE_M0(t) (((t) / NT) * TBM)
#endif
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) {
      mbar_init(full + s, 1); mbar_init(empty + s, 2 * (1 + MCW));
#if MCW
      mbar_init(ready + s, 2);
#endif
    }
#if MW_GATE_SAVED_TMA && MW_GATE_SAVED
    for(int s=0;s<2;++s){mbar_init(ab_full+s,1);mbar_init(ab_empty+s,2);}
#endif
    fence_barrier_init();
  }
  __syncthreads();
#if MCW
  cluster_sync_all();
#endif
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<24>();
    if (tid == 0) {
      uint32_t it = 0;
      int item=0;
      for (int t = t0; t < tiles; t += tstep,++item) {
        const int m0 = TILE_M0(t), n = t % NT;
#if MW_GATE_SAVED_TMA && MW_GATE_SAVED
        const int as=item&1;
        if(item>=2)mbar_wait(ab_empty+as,((item/2)-1)&1);
        mbar_arrive_expect_tx(ab_full+as,AB_TILE);
        for(int cw=0;cw<2;++cw)for(int cb=0;cb<2;++cb){
          uint8_t* dst=sm+F_AB+as*AB_TILE+cw*16384+cb*8192;
          tma_load_2d(dst,&mXN,ab_full+as,n*THB+cb*64,m0+cw*64);
          tma_load_2d(dst+32768,&mW1,ab_full+as,n*THB+cb*64,m0+cw*64);
        }
#endif
        for (int kb = 0; kb < KB; ++kb, ++it) {
          const int s = (int)(it % NSTAGE);
          if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
          uint8_t* st = sm + s * F_STAGE;
          mbar_arrive_expect_tx(full + s, F_STAGE);
#if MCW
          // Both destinations must install this transaction expectation before
          // either producer sends multicast weights to the peer's barrier.
          mbar_arrive(ready + s); mbar_arrive_peer(ready + s, peer);
          mbar_wait(ready + s, (it / NSTAGE) & 1u);
#endif
          for (int h = 0; h < 2; ++h) {
#if !MW_GATE_SAVED
            tma_load_2d(st + F_XN + h * B64, &mXN, full + s, kb * TBK, m0 + 64 * h);
#endif
#if !MW_EXPAND_ONLY
            tma_load_2d(st + F_DY + h * B64, &mDY, full + s, kb * TBK, m0 + 64 * h);
#endif
          }
#if MCW
          tma_load_2d_mc(st + F_W1 + rank * 2 * B64, &mW1, full + s, kb * TBK, n * 256 + 128 * (int)rank, (uint16_t)0x3);
          tma_load_2d_mc(st + F_WS + rank * B64, &mWS, full + s, kb * TBK, n * THB + 64 * (int)rank, (uint16_t)0x3);
#else
#if !MW_GATE_SAVED
          tma_load_2d(st + F_W1, &mW1, full + s, kb * TBK, n * 256);
          tma_load_2d(st + F_W1 + 2 * B64, &mW1, full + s, kb * TBK, n * 256 + 128);
#endif
#if !MW_EXPAND_ONLY
          tma_load_2d(st + F_WS, &mWS, full + s, kb * TBK, n * THB);
#endif
#endif
        }
      }
    }
  } else {
  setmaxnreg_inc<REG_C>();                                // ------------------------------------------------ consumers
  const int cw = wg - 1;
  uint8_t* stg = sm + F_STG + cw * F_STGW;
  uint32_t it = 0;
  int item=0;
  for (int t = t0; t < tiles; t += tstep,++item) {
    const int m0 = TILE_M0(t), n = t % NT;
#if !MW_EXPAND_ONLY
    float DH[64];
#endif
#if !MW_GATE_SAVED
    float AB[128];
#pragma unroll
    for (int e = 0; e < 128; ++e) AB[e] = 0.f;
#endif
#if !MW_EXPAND_ONLY
#pragma unroll
    for (int e = 0; e < 64; ++e) DH[e] = 0.f;
#endif
    int prev = -1;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t st = su + s * F_STAGE;
      GATE_FENCE_AB(AB); GATE_FENCE_DH(DH); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < TBK / 16; ++ks) {
#if !MW_GATE_SAVED
        mma_n256(AB, dsc_(st + F_XN + cw * B64 + ks * 32), dsc_(st + F_W1 + ks * 32));
#endif
#if !MW_EXPAND_ONLY
        mma_n128(DH, dsc_(st + F_DY + cw * B64 + ks * 32), dsc_(st + F_WS + ks * 32));
#endif
      }
      wgmma_commit();
#if WAIT0
      wgmma_wait<0>(); GATE_FENCE_AB(AB); GATE_FENCE_DH(DH); if (wtid == 0) { mbar_arrive(empty + s); if (MCW) mbar_arrive_peer(empty + s, peer); }
    }
#else
      if (prev >= 0) { wgmma_wait<1>(); GATE_FENCE_AB(AB); GATE_FENCE_DH(DH); if (wtid == 0) { mbar_arrive(empty + prev); if (MCW) mbar_arrive_peer(empty + prev, peer); } }
      prev = s;
    }
    wgmma_wait<0>(); GATE_FENCE_AB(AB); GATE_FENCE_DH(DH);
    if (wtid == 0) { mbar_arrive(empty + prev); if (MCW) mbar_arrive_peer(empty + prev, peer); }
#endif
#if MW_GATE_SAVED_TMA && MW_GATE_SAVED
    mbar_wait(ab_full+(item&1),(item/2)&1);
#endif
#pragma unroll
    for (int ps = 0; ps < NPASS; ++ps) {
    if (wtid == 0) tma_store_wait_read<0>();
    named_bar_sync(1 + cw, 128);
    {
      const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gq = 0; gq < 8 / NPASS; ++gq) {
        const int gp = ps * (8 / NPASS) + gq;
        uint32_t hp[4], dap[4], dbp[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int g = 2 * gp + (q >> 1), rb = q & 1, ia = 4 * g + 2 * rb, ib = 4 * (g + 16) + 2 * rb, id = 4 * g + 2 * rb;
#if !MW_EXPAND_ONLY
          const uint32_t dhp = pack_bf16(DH[id], DH[id + 1]);
          const float g0 = bf16lo(dhp), g1 = bf16hi(dhp);
#endif
#if MW_GATE_SAVED
#if MW_GATE_SAVED_TMA
          const int ar=16*warp+(lane>>2)+8*rb,ac=8*g+2*(lane&3);
          const uint32_t off=F_AB+(item&1)*AB_TILE+cw*16384+(ac/64)*8192+swz128(ar,(ac%64)*2);
          const uint32_t ap=*reinterpret_cast<const uint32_t*>(sm+off),bp=*reinterpret_cast<const uint32_t*>(sm+off+32768);
#else
          const size_t off=(size_t)(m0+64*cw+16*warp+(lane>>2)+8*rb)*H+n*THB+8*g+2*(lane&3);
          const uint32_t ap=*reinterpret_cast<const uint32_t*>(saved_a+off),bp=*reinterpret_cast<const uint32_t*>(saved_b+off);
#endif
          const float a0=bf16lo(ap),a1=bf16hi(ap),b0=bf16lo(bp),b1=bf16hi(bp);
#else
          const float a0 = AB[ia], a1 = AB[ia + 1], b0 = AB[ib], b1 = AB[ib + 1];
#endif
          const float s0 = sigmoid_(a0), s1 = sigmoid_(a1), l0 = a0 * s0, l1 = a1 * s1;
          hp[q] = pack_bf16(l0 * b0, l1 * b1);
#if MW_EXPAND_ONLY
          dap[q]=pack_bf16(a0,a1);dbp[q]=pack_bf16(b0,b1);
#else
          dap[q] = pack_bf16((g0 * b0) * (s0 + l0 * (1.f - s0)), (g1 * b1) * (s1 + l1 * (1.f - s1)));
          dbp[q] = pack_bf16(g0 * l0, g1 * l1);
#endif
        }
        const int col = 8 * (2 * gp + (mi >> 1)) - 64 * ps * (NPASS - 1);
        const uint32_t off = (uint32_t)(col >> 6) * 8192 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2));
        stsm_x4(smem_u32(stg) + off, hp[0], hp[1], hp[2], hp[3]);
        stsm_x4(smem_u32(stg) + CPP * 8192 + off, dap[0], dap[1], dap[2], dap[3]);
        stsm_x4(smem_u32(stg) + 2 * CPP * 8192 + off, dbp[0], dbp[1], dbp[2], dbp[3]);
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
      const int r = m0 + 64 * cw;
      for (int cb = 0; cb < CPP; ++cb) {
        const int c = n * THB + 64 * (ps * CPP + cb);
        if (!NO_H) tma_store_2d(&mH, stg + cb * 8192, c, r);   // NO_H: h is the forward's saved activation
#if MW_EXPAND_ONLY
        tma_store_2d(&mA, stg + (CPP + cb) * 8192, c, r);
        tma_store_2d(&mB, stg + (2 * CPP + cb) * 8192, c, r);
#else
        tma_store_2d(&mDAB, stg + (CPP + cb) * 8192, c, r);
        tma_store_2d(&mDAB, stg + (2 * CPP + cb) * 8192, H + c, r);
#endif
      }
      tma_store_commit();
    }
    }
#if MW_GATE_SAVED_TMA && MW_GATE_SAVED
    // The preceding epilogue barrier joins every shared A/B reader.
    if(wtid==0)mbar_arrive(ab_empty+(item&1));
#endif
  }
  if (wtid == 0) tma_store_wait_all();
  }
#if MCW
  __syncwarp();
  cluster_sync_all();
#endif
}
