// gate_sw.cu -- gate_gemm (128 rows x 64 hidden, both warpgroups share each weight slab) with the epilogue SOFTWARE-PIPELINED
// across tiles: two accumulator sets (2 x 96 fp32), and while tile t+1's slabs are issued to one set the gate epilogue of tile
// t runs on the other, a quarter per slab.  In gate_gemm/gate_gemm2 the two warpgroups hit their (heavy: gate math + three
// staged outputs) epilogue together and the tensor pipe idles for it (~26 % of the kernel by ablation); ping-pong fixes that
// but halves the weight-slab reuse and lost 27-48 %.  D is compiled in (-DDW) so every accumulator index is static.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma_n.cuh"

#ifndef DW
#define DW 256
#endif
#ifndef NSTAGE
#define NSTAGE 3
#endif
#ifndef TBK
#define TBK 64                                            // K slab: 64 (128-B swizzle) or 32 (64-B swizzle, twice the stages)
#endif
constexpr int TBM = 128, THB = 64, KB = DW / TBK;
static_assert(KB >= 4, "the epilogue is spread over >= 4 slabs");
constexpr int RB = TBK * 2, B64 = 64 * RB, SW_LAYOUT = TBK == 64 ? 1 : 2, SBO = 8 * RB;
constexpr int F_XN = 0, F_DY = 2 * B64, F_W1 = 4 * B64, F_WS = 6 * B64, F_STAGE = 7 * B64;   // xn, dy [128][TBK]; W1p [128][TBK]; Ws^T [64][TBK]
constexpr int F_STG = NSTAGE * F_STAGE, F_STGW = 24576;                 // per consumer WG: h | dA | dB, [64][64] each
constexpr int F_BAR = F_STG + 2 * F_STGW;
constexpr int SMEM_BYTES = F_BAR + 128;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");

TMN_DEVI float sigmoid_(float a) { float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a)); return fmaf(0.5f, t, 0.5f); }
TMN_DEVI uint64_t dsc_(uint32_t addr) { return smem_desc(addr, 16, SBO, SW_LAYOUT); }
TMN_DEVI void mma_n64(float (&d)[32], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 0;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b));
}
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}

struct Ctx { uint32_t su; uint64_t* full; uint64_t* empty; int cw, wtid, warp, lane; uint32_t it; };

// one gate group pair gp (0..3) of a 64 x 64 warpgroup tile: 4 x (h, dA, dB) stmatrix.x4 into the staging tile
template <int GP> TMN_DEVI void epi_chunk(const float (&AB)[64], const float (&DH)[32], uint8_t* stg, const Ctx& c) {
  const int mi = c.lane >> 3, mrow = 16 * c.warp + 8 * (mi & 1) + (c.lane & 7);
  uint32_t hp[4], dap[4], dbp[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int g = 2 * GP + (q >> 1), rb = q & 1, ia = 4 * g + 2 * rb, ib = 4 * (g + 8) + 2 * rb, id = 4 * g + 2 * rb;
    const uint32_t dhp = pack_bf16(DH[id], DH[id + 1]);
    const float g0 = bf16lo(dhp), g1 = bf16hi(dhp);
    const float a0 = AB[ia], a1 = AB[ia + 1], b0 = AB[ib], b1 = AB[ib + 1];
    const float s0 = sigmoid_(a0), s1 = sigmoid_(a1), l0 = a0 * s0, l1 = a1 * s1;
    hp[q] = pack_bf16(l0 * b0, l1 * b1);
    dap[q] = pack_bf16((g0 * b0) * (s0 + l0 * (1.f - s0)), (g1 * b1) * (s1 + l1 * (1.f - s1)));
    dbp[q] = pack_bf16(g0 * l0, g1 * l1);
  }
  const int col = 8 * (2 * GP + (mi >> 1));
  const uint32_t off = swz128((uint32_t)mrow, (uint32_t)(col * 2));
  stsm_x4(smem_u32(stg) + off, hp[0], hp[1], hp[2], hp[3]);
  stsm_x4(smem_u32(stg) + 8192 + off, dap[0], dap[1], dap[2], dap[3]);
  stsm_x4(smem_u32(stg) + 16384 + off, dbp[0], dbp[1], dbp[2], dbp[3]);
}
template <int C> TMN_DEVI void epi_chunks_after_slab(int kb_unused, const float (&AB)[64], const float (&DH)[32], uint8_t* stg, const Ctx& c) {}

// issue all slabs of one tile into (A, D); when EPI, run the previous tile's epilogue chunks on (PA, PD) between slabs
template <bool EPI> TMN_DEVI void mainloop(float (&A)[64], float (&D)[32], const float (&PA)[64], const float (&PD)[32], uint8_t* stg, Ctx& c) {
#pragma unroll
  for (int e = 0; e < 64; ++e) A[e] = 0.f;
#pragma unroll
  for (int e = 0; e < 32; ++e) D[e] = 0.f;
#pragma unroll
  for (int kb = 0; kb < KB; ++kb, ++c.it) {
    const int s = (int)(c.it % NSTAGE);
    mbar_wait(c.full + s, (c.it / NSTAGE) & 1u);
    const uint32_t st = c.su + s * F_STAGE;
    fence_regs(A); fence_regs(D); wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < TBK / 16; ++ks) {
      mma_n128(A, dsc_(st + F_XN + c.cw * B64 + ks * 32), dsc_(st + F_W1 + ks * 32));
      mma_n64(D, dsc_(st + F_DY + c.cw * B64 + ks * 32), dsc_(st + F_WS + ks * 32));
    }
    wgmma_commit();
    if constexpr (EPI) {                                  // chunks gp with gp * KB / 4 == kb run under this slab's MMAs
      if (kb == 0) epi_chunk<0>(PA, PD, stg, c);
      if (kb == KB / 4) epi_chunk<1>(PA, PD, stg, c);
      if (kb == KB / 2) epi_chunk<2>(PA, PD, stg, c);
      if (kb == 3 * KB / 4) epi_chunk<3>(PA, PD, stg, c);
    }
    wgmma_wait<0>(); fence_regs(A); fence_regs(D);
    if (c.wtid == 0) mbar_arrive(c.empty + s);
  }
}
TMN_DEVI void epi_begin(const Ctx& c) {                   // the staging tile's previous TMA store has read it
  if (c.wtid == 0) tma_store_wait_read<0>();
  named_bar_sync(1 + c.cw, 128);
}
TMN_DEVI void epi_all(const float (&A)[64], const float (&D)[32], uint8_t* stg, const Ctx& c) {
  epi_chunk<0>(A, D, stg, c); epi_chunk<1>(A, D, stg, c); epi_chunk<2>(A, D, stg, c); epi_chunk<3>(A, D, stg, c);
}
TMN_DEVI void epi_store(const CUtensorMap* mH, const CUtensorMap* mDAB, uint8_t* stg, int m0, int n, int H, const Ctx& c) {
  fence_proxy_async();
  named_bar_sync(1 + c.cw, 128);
  if (c.wtid == 0) {
    const int r = m0 + 64 * c.cw;
    tma_store_2d(mH, stg, n * THB, r);
    tma_store_2d(mDAB, stg + 8192, n * THB, r);
    tma_store_2d(mDAB, stg + 16384, H + n * THB, r);
    tma_store_commit();
  }
}

extern "C" __global__ void __launch_bounds__(384, 1)
gate_gemm(const __grid_constant__ CUtensorMap mXN, const __grid_constant__ CUtensorMap mDY, const __grid_constant__ CUtensorMap mW1,
          const __grid_constant__ CUtensorMap mWS, const __grid_constant__ CUtensorMap mH, const __grid_constant__ CUtensorMap mDAB,
          int M, int K, int H) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const int tid = threadIdx.x, wg = tid >> 7;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* empty = full + NSTAGE;
  const int NT = H / THB, tiles = (M / TBM) * NT;
  const int ntl = tiles > (int)blockIdx.x ? (tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid == 0) {
      uint32_t it = 0;
      for (int k = 0; k < ntl; ++k) {
        const int t = (int)blockIdx.x + k * (int)gridDim.x, m0 = (t / NT) * TBM, n = t % NT;
        for (int kb = 0; kb < KB; ++kb, ++it) {
          const int s = (int)(it % NSTAGE);
          if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
          uint8_t* st = sm + s * F_STAGE;
          mbar_arrive_expect_tx(full + s, F_STAGE);
          for (int h = 0; h < 2; ++h) {
            tma_load_2d(st + F_XN + h * B64, &mXN, full + s, kb * TBK, m0 + 64 * h);
            tma_load_2d(st + F_DY + h * B64, &mDY, full + s, kb * TBK, m0 + 64 * h);
          }
          tma_load_2d(st + F_W1, &mW1, full + s, kb * TBK, n * 128);
          tma_load_2d(st + F_WS, &mWS, full + s, kb * TBK, n * THB);
        }
      }
    }
  } else {
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  Ctx c{smem_u32(sm), full, empty, wg - 1, tid & 127, (tid & 127) >> 5, tid & 31, 0u};
  uint8_t* stg = sm + F_STG + c.cw * F_STGW;
  float AX[64], DX[32], AY[64], DY[32];
  auto tile_of = [&](int k, int& m0, int& n) { const int t = (int)blockIdx.x + k * (int)gridDim.x; m0 = (t / NT) * TBM; n = t % NT; };
  if (ntl > 0) mainloop<false>(AX, DX, AY, DY, stg, c);
  for (int k = 0; k < ntl; k += 2) {
    int m0, n;
    tile_of(k, m0, n);                                    // X holds tile k
    epi_begin(c);
    if (k + 1 < ntl) mainloop<true>(AY, DY, AX, DX, stg, c); else epi_all(AX, DX, stg, c);
    epi_store(&mH, &mDAB, stg, m0, n, H, c);
    if (k + 1 >= ntl) break;
    tile_of(k + 1, m0, n);                                // Y holds tile k + 1
    epi_begin(c);
    if (k + 2 < ntl) mainloop<true>(AX, DX, AY, DY, stg, c); else epi_all(AY, DY, stg, c);
    epi_store(&mH, &mDAB, stg, m0, n, H, c);
  }
  if (c.wtid == 0) tma_store_wait_all();
  }
}
