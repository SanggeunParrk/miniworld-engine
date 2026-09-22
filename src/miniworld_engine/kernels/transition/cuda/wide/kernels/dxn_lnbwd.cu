// dxn_lnbwd.cu -- the Transition backward's d_xn GEMM with the LayerNorm backward as its epilogue (wide widths, D >= 256):
//   d_xn = dAB [Wa; Wb]            (M x 2H) (2H x D), fp32 accumulator, never written
//   x_hat = x rstd - c1, w = gamma d_xn, dx = (w - x_hat mean(x_hat w) - mean(w)) rstd + dy       (dy: the residual branch)
//   dgamma = sum_rows d_xn x_hat, dbeta = sum_rows d_xn  -> one fp32 partial row per CTA, summed on the host.
// SPDX-License-Identifier: Apache-2.0
// K = 2H = 8D, so a 128 x 256 tile's mainloop is ~18 us and the whole LayerNorm backward rides in its epilogue for free; the
// engine runs it as cuBLAS (d_xn to HBM in bf16) + a separate LN-bwd kernel + the dx += dy add.  Producer warpgroup + two
// consumer warpgroups.  COLS = 1: the consumers take 64 rows each of a 128-row tile, all D columns (D = 256: m64n256).
// COLS = 2: both take the same 64 rows, D/2 columns each (D = 384: n192, 512: n256), and exchange the per-row sums in smem.
// x is prefetched into registers at the tile start (latency hidden by the mainloop); dy is read in the epilogue.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma_n.cuh"

#ifndef DW
#define DW 256
#endif
#ifndef COLS
#define COLS 1
#endif
#ifndef TBK
#define TBK 64
#endif
#ifndef WAIT0
#define WAIT0 0
#endif
#ifndef NSTAGE
#define NSTAGE 3
#endif
constexpr int D = DW, K2 = 8 * DW, NW = DW / COLS, ROWB = 128 / COLS, NG = NW / 8;
constexpr int RB = TBK * 2, SW_LAYOUT = TBK == 64 ? 1 : 2, SBO = 8 * RB;
constexpr int BR = COLS == 2 ? NW : D, NBOX = D / BR;                  // B (W_ab^T [D][2H]) TMA boxes of BR rows
static_assert(BR <= 256 && NW <= 256 && NW % 64 == 0, "tile");
constexpr int F_A = 0, F_B = ROWB * RB, F_STAGE = F_B + D * RB;
#ifndef ABL_NOEPI
#define ABL_NOEPI 0
#endif
#ifndef STGDX
#define STGDX 0                                           // 1 (COLS = 2): dx staged with stmatrix and TMA-stored (not st.global)
#endif
#ifndef ABL_NODX
#define ABL_NODX 0
#endif
#ifndef XB
#define XB 0                                              // 1: x read from global in double-buffered batches of 8 groups (16 regs each)
#endif
#ifndef XSM
#define XSM 0                                             // 1: the x tile TMA-loaded into shared memory once per tile (instead of
#endif                                                    //    a register prefetch, which spills at 128 x 256)
constexpr int F_X = NSTAGE * F_STAGE, F_GAM = F_X + (XSM ? ROWB * D * 2 : 0);
constexpr int F_DG = F_GAM + 4 * D, F_DB = F_DG + 8 * 4 * D, F_XS = F_DB + 8 * 4 * D;
constexpr int F_DXS = F_XS + (COLS == 2 ? 2 * 2 * 64 * 4 * 4 : 0);    // xs[parity][wg][64 rows][ca0 cb0 ...] (float4 per row)
static_assert(!STGDX || COLS == 2, "dx staging is sized for the 64-row tile");
constexpr int F_BAR = F_DXS + (STGDX ? 2 * 64 * NW * 2 : 0);          // STGDX: per warpgroup its [64 rows][NW] dx tile
constexpr int SMEM_BYTES = F_BAR + 128;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");

TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}
TMN_DEVI uint64_t dsc_(uint32_t addr) { return smem_desc(addr, 16, SBO, SW_LAYOUT); }
template <int N> TMN_DEVI void mma_nw(float (&d)[N], uint64_t a, uint64_t b) {
  if constexpr (N == 128) mma_n256(d, a, b); else if constexpr (N == 96) mma_n192(d, a, b); else mma_n128(d, a, b);
}

extern "C" __global__ void __launch_bounds__(384, 1)
dxn_lnbwd(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB, const __grid_constant__ CUtensorMap mX, const __grid_constant__ CUtensorMap mDX,
          const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx,
          const float* __restrict__ gamma, const float* __restrict__ rstd, const float* __restrict__ c1,
          float* __restrict__ pdg, float* __restrict__ pdb, int M) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* empty = full + NSTAGE;
  uint64_t* xfull = empty + NSTAGE;                        // XSM: this tile's x has landed
  uint64_t* xfree = xfull + 1;                             // XSM: both warpgroups finished reading the previous tile's x
  float* gs = reinterpret_cast<float*>(sm + F_GAM);
  float* dgs = reinterpret_cast<float*>(sm + F_DG);
  float* dbs = reinterpret_cast<float*>(sm + F_DB);
  float4* xs = reinterpret_cast<float4*>(sm + F_XS);
  const int tiles = M / ROWB;
  constexpr int KB = K2 / TBK;
  for (int i = tid; i < D; i += 384) gs[i] = gamma[i];
  for (int i = tid; i < 8 * D; i += 384) { dgs[i] = 0.f; dbs[i] = 0.f; }
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    mbar_init(xfull, 1); mbar_init(xfree, 2);
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid == 0) {
      uint32_t it = 0, tl = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tl) {
        const int m0 = t * ROWB;
        for (int kb = 0; kb < KB; ++kb, ++it) {
          if (XSM && kb == KB / 2) {                        // mid-mainloop: the previous tile's epilogue is long done
            if (tl > 0) mbar_wait(xfree, (tl - 1) & 1u);
            mbar_arrive_expect_tx(xfull, ROWB * D * 2);
            for (int qd = 0; qd < D / 64; ++qd)
              for (int h = 0; h < ROWB / 64; ++h) tma_load_2d(sm + F_X + qd * ROWB * 128 + h * 8192, &mX, xfull, 64 * qd, m0 + 64 * h);
          }
          const int s = (int)(it % NSTAGE);
          if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
          uint8_t* st = sm + s * F_STAGE;
          mbar_arrive_expect_tx(full + s, F_STAGE);
          for (int h = 0; h < ROWB / 64; ++h) tma_load_2d(st + F_A + h * 64 * RB, &mA, full + s, kb * TBK, m0 + 64 * h);
          for (int b = 0; b < NBOX; ++b) tma_load_2d(st + F_B + b * BR * RB, &mB, full + s, kb * TBK, b * BR);
        }
      }
    }
  } else {
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1, slot = cw * 4 + warp, q = lane & 3;
  const int rbase = COLS == 1 ? 64 * cw : 0, cbase = COLS == 1 ? 0 : NW * cw;
  const uint32_t aoff = (uint32_t)(F_A + (COLS == 1 ? cw * 64 * RB : 0)), boff = (uint32_t)(F_B + (COLS == 1 ? 0 : cw * NW * RB));
  uint32_t it = 0, par = 0;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x, par ^= 1u) {
    const int lr = 16 * warp + (lane >> 2), r0 = t * ROWB + rbase + lr, r1 = r0 + 8;
#if XSM
    const int rl0 = rbase + lr;
#define XR0(g) (*reinterpret_cast<const uint32_t*>(sm + F_X + ((cbase + 8 * (g)) >> 6) * ROWB * 128 + swz128((uint32_t)rl0, (uint32_t)(((cbase + 8 * (g)) & 63) * 2 + 4 * q))))
#define XR1(g) (*reinterpret_cast<const uint32_t*>(sm + F_X + ((cbase + 8 * (g)) >> 6) * ROWB * 128 + swz128((uint32_t)(rl0 + 8), (uint32_t)(((cbase + 8 * (g)) & 63) * 2 + 4 * q))))
#elif XB
    uint32_t xq0[2][8], xq1[2][8];
#define XLOAD(bi, g0_) do { _Pragma("unroll") for (int jj = 0; jj < 8; ++jj) { const int c_ = cbase + 8 * ((g0_) + jj) + 2 * q; \
      xq0[bi][jj] = ldg32(x + (size_t)r0 * D + c_); xq1[bi][jj] = ldg32(x + (size_t)r1 * D + c_); } } while (0)
    XLOAD(0, 0);                                          // batch 0 in flight under the whole mainloop
#define XR0(g) xq0[((g) >> 3) & 1][(g) & 7]
#define XR1(g) xq1[((g) >> 3) & 1][(g) & 7]
#else
    uint32_t xr0[NG], xr1[NG];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      const int c = cbase + 8 * g + 2 * q;
      xr0[g] = ldg32(x + (size_t)r0 * D + c); xr1[g] = ldg32(x + (size_t)r1 * D + c);
    }
#define XR0(g) xr0[g]
#define XR1(g) xr1[g]
#endif
    const float rs0 = rstd[r0], rs1 = rstd[r1], cc0 = c1[r0], cc1 = c1[r1];
    float acc[NW / 2];
#pragma unroll
    for (int e = 0; e < NW / 2; ++e) acc[e] = 0.f;
    int prev = -1;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t st = su + s * F_STAGE;
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < TBK / 16; ++ks) mma_nw(acc, dsc_(st + aoff + ks * 32), dsc_(st + boff + ks * 32));
      wgmma_commit();
#if WAIT0                                                 // release this slab as soon as its MMAs retire (no one-slab lag)
      wgmma_wait<0>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + s);
    }
#else
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    wgmma_wait<0>(); fence_regs(acc);
    if (wtid == 0) mbar_arrive(empty + prev);
#endif
#if ABL_NOEPI                                             // TIMING ONLY: mainloop alone -- every accumulator consumed, one guarded store
    {
      float sacc = 0.f;
#pragma unroll
      for (int e = 0; e < NW / 2; ++e) sacc += acc[e];
      if (sacc == 1234.5f) stg32(dx, 0u);
    }
    continue;
#endif
    if (XSM) mbar_wait(xfull, par);
    // ---------------------------------------------------------------- pass 1: row sums, dgamma / dbeta partials
    float ca0 = 0.f, cb0 = 0.f, ca1 = 0.f, cb1 = 0.f;
#pragma unroll
    for (int g = 0; g < NG; ++g) {
#if XB
      if ((g & 7) == 0 && g + 8 < NG) XLOAD(((g >> 3) + 1) & 1, g + 8);
#endif
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int c = cbase + 8 * g + 2 * q + e;
        const float gm = gs[c], a0 = acc[4 * g + e], a1 = acc[4 * g + 2 + e];
        const float xh0 = (e ? bf16hi(XR0(g)) : bf16lo(XR0(g))) * rs0 - cc0, xh1 = (e ? bf16hi(XR1(g)) : bf16lo(XR1(g))) * rs1 - cc1;
        const float w0 = gm * a0, w1 = gm * a1;
        ca0 = fmaf(xh0, w0, ca0); cb0 += w0; ca1 = fmaf(xh1, w1, ca1); cb1 += w1;
        float pg = fmaf(a0, xh0, a1 * xh1), pb = a0 + a1;
        pg += __shfl_xor_sync(0xffffffffu, pg, 4); pb += __shfl_xor_sync(0xffffffffu, pb, 4);
        pg += __shfl_xor_sync(0xffffffffu, pg, 8); pb += __shfl_xor_sync(0xffffffffu, pb, 8);
        pg += __shfl_xor_sync(0xffffffffu, pg, 16); pb += __shfl_xor_sync(0xffffffffu, pb, 16);
        if (lane < 4) { dgs[slot * D + c] += pg; dbs[slot * D + c] += pb; }
      }
    }
#if XB
    XLOAD(0, 0);                                          // pass 2's first batch, under the row-sum reductions
#endif
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      ca0 += __shfl_xor_sync(0xffffffffu, ca0, o); cb0 += __shfl_xor_sync(0xffffffffu, cb0, o);
      ca1 += __shfl_xor_sync(0xffffffffu, ca1, o); cb1 += __shfl_xor_sync(0xffffffffu, cb1, o);
    }
    if constexpr (COLS == 2) {
      float4* mine = xs + (par * 2 + cw) * 64;
      const float4* peer = xs + (par * 2 + (cw ^ 1)) * 64;
      if (q == 0) mine[lr] = make_float4(ca0, cb0, ca1, cb1);
      named_bar_sync(5, 256);
      const float4 p = peer[lr];
      ca0 += p.x; cb0 += p.y; ca1 += p.z; cb1 += p.w;
    }
    constexpr float inv = 1.f / D;
    ca0 *= inv; cb0 *= inv; ca1 *= inv; cb1 *= inv;
    // ---------------------------------------------------------------- pass 2: dx = (w - x_hat ca - cb) rstd + dy
#if STGDX
    uint8_t* dxs = sm + F_DXS + cw * 64 * NW * 2;
    if (wtid == 0) tma_store_wait_read<0>();             // the previous tile's dx store has read the staging tile
    named_bar_sync(1 + cw, 128);
    uint32_t pend0 = 0, pend1 = 0;
    const int smi = lane >> 3, smrow = 16 * warp + 8 * (smi & 1) + (lane & 7);
#endif
#pragma unroll
    for (int g0 = 0; g0 < NG; g0 += 8) {
#if XB
      if (g0 + 8 < NG) XLOAD(((g0 >> 3) + 1) & 1, g0 + 8);
#endif
      uint32_t d0[8], d1[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int c = cbase + 8 * (g0 + j) + 2 * q;
        d0[j] = ldg32(dy + (size_t)r0 * D + c); d1[j] = ldg32(dy + (size_t)r1 * D + c);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int g = g0 + j, c = cbase + 8 * g + 2 * q;
        float o0[2], o1[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float gm = gs[c + e];
          const float xh0 = (e ? bf16hi(XR0(g)) : bf16lo(XR0(g))) * rs0 - cc0, xh1 = (e ? bf16hi(XR1(g)) : bf16lo(XR1(g))) * rs1 - cc1;
          o0[e] = fmaf(gm * acc[4 * g + e] - fmaf(xh0, ca0, cb0), rs0, e ? bf16hi(d0[j]) : bf16lo(d0[j]));
          o1[e] = fmaf(gm * acc[4 * g + 2 + e] - fmaf(xh1, ca1, cb1), rs1, e ? bf16hi(d1[j]) : bf16lo(d1[j]));
        }
#if ABL_NODX                                              // TIMING ONLY: all math kept, the dx stores (almost) never execute
        if (o0[0] == 1234.5f && o1[1] == -1234.5f) { stg32(dx + (size_t)r0 * D + c, pack_bf16(o0[0], o0[1])); stg32(dx + (size_t)r1 * D + c, pack_bf16(o1[0], o1[1])); }
#elif STGDX
        if ((j & 1) == 0) { pend0 = pack_bf16(o0[0], o0[1]); pend1 = pack_bf16(o1[0], o1[1]); }
        else {                                            // groups g - 1, g: the four m8n8 matrices of one stmatrix.x4
          const int col = 8 * (2 * (g >> 1) + (smi >> 1));
          stsm_x4(smem_u32(dxs) + (col >> 6) * 8192 + swz128((uint32_t)smrow, (uint32_t)((col & 63) * 2)),
                  pend0, pend1, pack_bf16(o0[0], o0[1]), pack_bf16(o1[0], o1[1]));
        }
#else
        stg32(dx + (size_t)r0 * D + c, pack_bf16(o0[0], o0[1]));
        stg32(dx + (size_t)r1 * D + c, pack_bf16(o1[0], o1[1]));
#endif
      }
    }
#if STGDX
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
      for (int cb = 0; cb < NW / 64; ++cb) tma_store_2d(&mDX, dxs + cb * 8192, cbase + 64 * cb, t * ROWB);
      tma_store_commit();
    }
#endif
    if (XSM) {                                            // this warpgroup's reads of x are done: the producer may refill it
      named_bar_sync(1 + cw, 128);
      if (wtid == 0) mbar_arrive(xfree);
    }
  }
#undef XR0
#undef XR1
#if XB
#undef XLOAD
#endif
  if (STGDX && wtid == 0) tma_store_wait_all();
  named_bar_sync(6, 256);
  for (int c = tid - 128; c < D; c += 256) {
    float sg = 0.f, sb = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) { sg += dgs[w * D + c]; sb += dbs[w * D + c]; }
    pdg[(size_t)blockIdx.x * D + c] = sg; pdb[(size_t)blockIdx.x * D + c] = sb;
  }
  }
}
