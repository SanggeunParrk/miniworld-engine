// gate_pp.cu -- gate_gemm2's math (dh = dy Ws, ab = xn [Wa;Wb]^T, SwiGLU backward -> h, [dA|dB]) in a PING-PONG layout: each
// consumer warpgroup owns whole 64-row x 128-hidden tiles (alternate tiles of the CTA's sequence), and an order barrier hands the
// tensor pipe from one warpgroup's mainloop to the other's, so one warpgroup's epilogue (gate math + 3 staged TMA stores) runs
// under the other's mainloop.  In gate_gemm2 both warpgroups share each slab, so their epilogues coincide and the pipe idles.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma_n.cuh"

#ifndef NSTAGE
#define NSTAGE 4
#endif
#ifndef TBK
#define TBK 32
#endif
#ifndef ORDER
#define ORDER 1                                           // 0: no order barrier (the warpgroups run free)
#endif
constexpr int TBM = 64, THB = 128;
constexpr int RB = TBK * 2, B64 = 64 * RB, SW_LAYOUT = TBK == 64 ? 1 : 2, SBO = 8 * RB;
constexpr int F_XN = 0, F_DY = B64, F_W1 = 2 * B64, F_WS = 6 * B64, F_STAGE = 8 * B64;   // xn, dy [64][TBK]; W1p [256][TBK]; Ws^T [128][TBK]
constexpr int F_STG = NSTAGE * F_STAGE, F_STGW = 6 * 8192;             // per WG: {h, dA, dB} x {cols 0-63, 64-127}
constexpr int F_BAR = F_STG + 2 * F_STGW;
constexpr int SMEM_BYTES = F_BAR + 128;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");

TMN_DEVI float sigmoid_(float a) { float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a)); return fmaf(0.5f, t, 0.5f); }
TMN_DEVI uint64_t dsc_(uint32_t addr) { return smem_desc(addr, 16, SBO, SW_LAYOUT); }
TMN_DEVI void named_bar_arrive(int id, int n) { __syncwarp(); asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
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
  const int NT = H / THB, tiles = (M / TBM) * NT, KB = K / TBK;
  const int ntl = tiles > (int)blockIdx.x ? (tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;   // this CTA's tiles
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 1); }
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<24>();
    if (tid == 0) {
      uint32_t it = 0;
      for (int k = 0; k < ntl; ++k) {
        const int t = (int)blockIdx.x + k * (int)gridDim.x, m0 = (t / NT) * TBM, n = t % NT;
        for (int kb = 0; kb < KB; ++kb, ++it) {
          const int s = (int)(it % NSTAGE);
          if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
          uint8_t* st = sm + s * F_STAGE;
          mbar_arrive_expect_tx(full + s, F_STAGE);
          tma_load_2d(st + F_XN, &mXN, full + s, kb * TBK, m0);
          tma_load_2d(st + F_DY, &mDY, full + s, kb * TBK, m0);
          tma_load_2d(st + F_W1, &mW1, full + s, kb * TBK, n * 256);
          tma_load_2d(st + F_W1 + 2 * B64, &mW1, full + s, kb * TBK, n * 256 + 128);
          tma_load_2d(st + F_WS, &mWS, full + s, kb * TBK, n * THB);
        }
      }
    }
  } else {
  setmaxnreg_inc<240>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  uint8_t* stg = sm + F_STG + cw * F_STGW;
  constexpr int BAR_T0 = 3, BAR_T1 = 4;                   // "warpgroup w may start its mainloop"
  for (int k = cw; k < ntl; k += 2) {
    const int t = (int)blockIdx.x + k * (int)gridDim.x, m0 = (t / NT) * TBM, n = t % NT;
    uint32_t it = (uint32_t)(k * KB);                     // ring position of this tile's first slab
    if (ORDER && k > 0) named_bar_sync(cw ? BAR_T1 : BAR_T0, 256);
    float AB[128], DH[64];
#pragma unroll
    for (int e = 0; e < 128; ++e) AB[e] = 0.f;
#pragma unroll
    for (int e = 0; e < 64; ++e) DH[e] = 0.f;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t st = su + s * F_STAGE;
      fence_regs(AB); fence_regs(DH); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < TBK / 16; ++ks) {
        mma_n256(AB, dsc_(st + F_XN + ks * 32), dsc_(st + F_W1 + ks * 32));
        mma_n128(DH, dsc_(st + F_DY + ks * 32), dsc_(st + F_WS + ks * 32));
      }
      wgmma_commit();
      wgmma_wait<0>(); fence_regs(AB); fence_regs(DH);
      if (wtid == 0) mbar_arrive(empty + s);
    }
    if (ORDER && k + 1 < ntl) named_bar_arrive(cw ? BAR_T0 : BAR_T1, 256);   // hand the pipe to the other warpgroup
    if (wtid == 0) tma_store_wait_read<0>();
    named_bar_sync(1 + cw, 128);
    {
      const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < 8; ++gp) {
        uint32_t hp[4], dap[4], dbp[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int g = 2 * gp + (q >> 1), rb = q & 1, ia = 4 * g + 2 * rb, ib = 4 * (g + 16) + 2 * rb, id = 4 * g + 2 * rb;
          const uint32_t dhp = pack_bf16(DH[id], DH[id + 1]);
          const float g0 = bf16lo(dhp), g1 = bf16hi(dhp);
          const float a0 = AB[ia], a1 = AB[ia + 1], b0 = AB[ib], b1 = AB[ib + 1];
          const float s0 = sigmoid_(a0), s1 = sigmoid_(a1), l0 = a0 * s0, l1 = a1 * s1;
          hp[q] = pack_bf16(l0 * b0, l1 * b1);
          dap[q] = pack_bf16((g0 * b0) * (s0 + l0 * (1.f - s0)), (g1 * b1) * (s1 + l1 * (1.f - s1)));
          dbp[q] = pack_bf16(g0 * l0, g1 * l1);
        }
        const int col = 8 * (2 * gp + (mi >> 1));
        const uint32_t off = (uint32_t)(col >> 6) * 8192 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2));
        stsm_x4(smem_u32(stg) + off, hp[0], hp[1], hp[2], hp[3]);
        stsm_x4(smem_u32(stg) + 16384 + off, dap[0], dap[1], dap[2], dap[3]);
        stsm_x4(smem_u32(stg) + 32768 + off, dbp[0], dbp[1], dbp[2], dbp[3]);
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
      const int c = n * THB;
      for (int cb = 0; cb < 2; ++cb) {
        tma_store_2d(&mH, stg + cb * 8192, c + 64 * cb, m0);
        tma_store_2d(&mDAB, stg + 16384 + cb * 8192, c + 64 * cb, m0);
        tma_store_2d(&mDAB, stg + 32768 + cb * 8192, H + c + 64 * cb, m0);
      }
      tma_store_commit();
    }
  }
  if (wtid == 0) tma_store_wait_all();
  }
}
