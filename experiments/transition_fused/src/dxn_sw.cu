// dxn_sw.cu -- dxn_lnbwd (d_xn GEMM + LayerNorm backward + residual, D = 256) with the epilogue SOFTWARE-PIPELINED across
// tiles.  SPDX-License-Identifier: Apache-2.0
//
// Mainloop-only ablation of dxn_lnbwd: 240-257 us of its ~401 (base clocks, L384) -- the epilogue (row sums, dgamma / dbeta
// partials, dx) is 37 % of the kernel, exposed with the tensor pipe idle, and it is LATENCY-bound (x / dy loads).  Here each
// consumer warpgroup keeps two 64-register accumulator sets: while tile k's slabs are issued into one, tile k-1's epilogue runs
// on the other in ten steps, one per two slabs, with its x / dy operands loaded one step ahead into double buffers.
//   steps 0-3  pass 1, four groups each: row sums ca / cb, dgamma / dbeta partials
//   step  4    row sums reduced over the quad and exchanged with the other warpgroup (it holds the other 128 columns)
//   steps 5-8  pass 2, four groups each: dx = (gamma d_xn - x_hat ca - cb) rstd + dy -> stmatrix into the staging tile
//   step  9    TMA store of the dx tile
// Tile 64 rows x 256 columns, both warpgroups on the same 64 rows, 128 columns each; K = 2H = 2048 in 64-wide slabs.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma_n.cuh"

#ifndef NSTAGE
#define NSTAGE 4
#endif
constexpr int D = 256, K2 = 2048, NW = 128, ROWB = 64, NG = NW / 8, TBK = 64, KB = K2 / TBK;
constexpr int F_A = 0, F_B = ROWB * 128, F_STAGE = F_B + D * 128;          // A: dAB [64 rows][64]; B: W_ab^T [256][64]
constexpr int F_GAM = NSTAGE * F_STAGE, F_DG = F_GAM + 4 * D, F_DB = F_DG + 8 * 4 * D, F_XS = F_DB + 8 * 4 * D;
constexpr int F_DXS = F_XS + 2 * 2 * 64 * 16, F_BAR = F_DXS + 2 * 64 * NW * 2;
constexpr int SMEM_BYTES = F_BAR + 128;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");
static_assert(KB >= 20, "ten epilogue steps, one per two slabs");

TMN_DEVI uint64_t dsc_(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}

struct Ep {                                               // one tile's epilogue state
  int t, par, r0, r1;                                   // par: parity of the CTA-local tile index (t steps by the grid)
  float rs0, rs1, cc0, cc1, ca0, cb0, ca1, cb1;
};
struct Cx {
  uint32_t su; uint64_t* full; uint64_t* empty; uint8_t* sm;
  const __nv_bfloat16* x; const __nv_bfloat16* dy; const CUtensorMap* mdx;
  float* gs; float* dgs; float* dbs; float4* xs; uint8_t* dxs;
  int cw, wtid, warp, lane, q, slot, cbase, lr, smi, smrow; uint32_t it;
  uint32_t xq0[2][4], xq1[2][4], dq0[2][4], dq1[2][4];   // one step's operands: 4 groups x 2 rows, double-buffered
};

// load the x (and, for pass 2, dy) values of groups 4 c .. 4 c + 3 into buffer B
template <int B, bool DY> TMN_DEVI void load_step(Cx& c, const Ep& e, int ch) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = c.cbase + 8 * (4 * ch + j) + 2 * c.q;
    c.xq0[B][j] = ldg32(c.x + (size_t)e.r0 * D + col); c.xq1[B][j] = ldg32(c.x + (size_t)e.r1 * D + col);
    if constexpr (DY) { c.dq0[B][j] = ldg32(c.dy + (size_t)e.r0 * D + col); c.dq1[B][j] = ldg32(c.dy + (size_t)e.r1 * D + col); }
  }
}
template <int CH, int B> TMN_DEVI void pass1(Cx& c, Ep& e, const float (&P)[64]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int g = 4 * CH + j;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int col = c.cbase + 8 * g + 2 * c.q + h;
      const float gm = c.gs[col], a0 = P[4 * g + h], a1 = P[4 * g + 2 + h];
      const float xh0 = (h ? bf16hi(c.xq0[B][j]) : bf16lo(c.xq0[B][j])) * e.rs0 - e.cc0;
      const float xh1 = (h ? bf16hi(c.xq1[B][j]) : bf16lo(c.xq1[B][j])) * e.rs1 - e.cc1;
      const float w0 = gm * a0, w1 = gm * a1;
      e.ca0 = fmaf(xh0, w0, e.ca0); e.cb0 += w0; e.ca1 = fmaf(xh1, w1, e.ca1); e.cb1 += w1;
      float pg = fmaf(a0, xh0, a1 * xh1), pb = a0 + a1;
      pg += __shfl_xor_sync(0xffffffffu, pg, 4); pb += __shfl_xor_sync(0xffffffffu, pb, 4);
      pg += __shfl_xor_sync(0xffffffffu, pg, 8); pb += __shfl_xor_sync(0xffffffffu, pb, 8);
      pg += __shfl_xor_sync(0xffffffffu, pg, 16); pb += __shfl_xor_sync(0xffffffffu, pb, 16);
      if (c.lane < 4) { c.dgs[c.slot * D + col] += pg; c.dbs[c.slot * D + col] += pb; }
    }
  }
}
TMN_DEVI void rowsums(Cx& c, Ep& e) {
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    e.ca0 += __shfl_xor_sync(0xffffffffu, e.ca0, o); e.cb0 += __shfl_xor_sync(0xffffffffu, e.cb0, o);
    e.ca1 += __shfl_xor_sync(0xffffffffu, e.ca1, o); e.cb1 += __shfl_xor_sync(0xffffffffu, e.cb1, o);
  }
  const int par = e.par;                                  // parity buffer: the peer may still read the previous tile's
  float4* mine = c.xs + (par * 2 + c.cw) * 64;
  const float4* peer = c.xs + (par * 2 + (c.cw ^ 1)) * 64;
  if (c.q == 0) mine[c.lr] = make_float4(e.ca0, e.cb0, e.ca1, e.cb1);
  named_bar_sync(5, 256);
  const float4 p = peer[c.lr];
  constexpr float inv = 1.f / D;
  e.ca0 = (e.ca0 + p.x) * inv; e.cb0 = (e.cb0 + p.y) * inv; e.ca1 = (e.ca1 + p.z) * inv; e.cb1 = (e.cb1 + p.w) * inv;
}
template <int CH, int B> TMN_DEVI void pass2(Cx& c, const Ep& e, const float (&P)[64]) {
  uint32_t v0[4], v1[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int g = 4 * CH + j;
    float o0[2], o1[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int col = c.cbase + 8 * g + 2 * c.q + h;
      const float gm = c.gs[col];
      const float xh0 = (h ? bf16hi(c.xq0[B][j]) : bf16lo(c.xq0[B][j])) * e.rs0 - e.cc0;
      const float xh1 = (h ? bf16hi(c.xq1[B][j]) : bf16lo(c.xq1[B][j])) * e.rs1 - e.cc1;
      o0[h] = fmaf(gm * P[4 * g + h] - fmaf(xh0, e.ca0, e.cb0), e.rs0, h ? bf16hi(c.dq0[B][j]) : bf16lo(c.dq0[B][j]));
      o1[h] = fmaf(gm * P[4 * g + 2 + h] - fmaf(xh1, e.ca1, e.cb1), e.rs1, h ? bf16hi(c.dq1[B][j]) : bf16lo(c.dq1[B][j]));
    }
    v0[j] = pack_bf16(o0[0], o0[1]); v1[j] = pack_bf16(o1[0], o1[1]);
  }
#pragma unroll
  for (int pr = 0; pr < 2; ++pr) {                        // groups (4 CH + 2 pr, + 1): one stmatrix.x4
    const int gp = 2 * CH + pr, col = 8 * (2 * gp + (c.smi >> 1));
    stsm_x4(smem_u32(c.dxs) + (col >> 6) * 8192 + swz128((uint32_t)c.smrow, (uint32_t)((col & 63) * 2)),
            v0[2 * pr], v1[2 * pr], v0[2 * pr + 1], v1[2 * pr + 1]);
  }
}
TMN_DEVI void staging_free(const Cx& c) {                 // the previous tile's dx store has read the staging tile
  if (c.wtid == 0) tma_store_wait_read<0>();
  named_bar_sync(1 + c.cw, 128);
}
TMN_DEVI void store_dx(const Cx& c, const Ep& e) {
  fence_proxy_async();
  named_bar_sync(1 + c.cw, 128);
  if (c.wtid == 0) {
    for (int cb = 0; cb < NW / 64; ++cb) tma_store_2d(c.mdx, c.dxs + cb * 8192, c.cbase + 64 * cb, e.t * ROWB);
    tma_store_commit();
  }
}
// epilogue step S (0..9) of the tile held in P
template <int S> TMN_DEVI void epi_step(Cx& c, Ep& e, const float (&P)[64]) {
  if constexpr (S == 0) { load_step<1, false>(c, e, 1); pass1<0, 0>(c, e, P); }
  else if constexpr (S == 1) { load_step<0, false>(c, e, 2); pass1<1, 1>(c, e, P); }
  else if constexpr (S == 2) { load_step<1, false>(c, e, 3); pass1<2, 0>(c, e, P); }
  else if constexpr (S == 3) { load_step<0, true>(c, e, 0); pass1<3, 1>(c, e, P); }
  else if constexpr (S == 4) { rowsums(c, e); staging_free(c); load_step<1, true>(c, e, 1); }
  else if constexpr (S == 5) { pass2<0, 0>(c, e, P); load_step<0, true>(c, e, 2); }
  else if constexpr (S == 6) { pass2<1, 1>(c, e, P); load_step<1, true>(c, e, 3); }
  else if constexpr (S == 7) { pass2<2, 0>(c, e, P); }
  else if constexpr (S == 8) { pass2<3, 1>(c, e, P); }
  else { store_dx(c, e); }
}
TMN_DEVI void epi_all(Cx& c, Ep& e, const float (&P)[64]) {
  epi_step<0>(c, e, P); epi_step<1>(c, e, P); epi_step<2>(c, e, P); epi_step<3>(c, e, P); epi_step<4>(c, e, P);
  epi_step<5>(c, e, P); epi_step<6>(c, e, P); epi_step<7>(c, e, P); epi_step<8>(c, e, P); epi_step<9>(c, e, P);
}
TMN_DEVI void epi_begin(Cx& c, Ep& e, int t, int k) {     // tile t (the CTA's k-th) per-row constants
  e.t = t; e.par = k & 1; e.r0 = t * ROWB + c.lr; e.r1 = e.r0 + 8;
  e.ca0 = e.cb0 = e.ca1 = e.cb1 = 0.f;
}
TMN_DEVI void epi_stats(const Cx& c, Ep& e, const float* rstd, const float* c1) {
  e.rs0 = rstd[e.r0]; e.rs1 = rstd[e.r1]; e.cc0 = c1[e.r0]; e.cc1 = c1[e.r1];
}
// all KB slabs of one tile into A; when EPI the previous tile's ten epilogue steps run on P, one per two slabs
template <bool EPI> TMN_DEVI void mainloop(float (&A)[64], const float (&P)[64], Cx& c, Ep& pe) {
#pragma unroll
  for (int e = 0; e < 64; ++e) A[e] = 0.f;
  const uint32_t aoff = F_A, boff = F_B + c.cw * NW * 128;
#pragma unroll
  for (int kb = 0; kb < KB; ++kb, ++c.it) {
    const int s = (int)(c.it % NSTAGE);
    mbar_wait(c.full + s, (c.it / NSTAGE) & 1u);
    const uint32_t st = c.su + s * F_STAGE;
    fence_regs(A); wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < TBK / 16; ++ks) mma_n128(A, dsc_(st + aoff + ks * 32), dsc_(st + boff + ks * 32));
    wgmma_commit();
    if constexpr (EPI) {
      if (kb == 1) epi_step<0>(c, pe, P);
      if (kb == 3) epi_step<1>(c, pe, P);
      if (kb == 5) epi_step<2>(c, pe, P);
      if (kb == 7) epi_step<3>(c, pe, P);
      if (kb == 9) epi_step<4>(c, pe, P);
      if (kb == 11) epi_step<5>(c, pe, P);
      if (kb == 13) epi_step<6>(c, pe, P);
      if (kb == 15) epi_step<7>(c, pe, P);
      if (kb == 17) epi_step<8>(c, pe, P);
      if (kb == 19) epi_step<9>(c, pe, P);
    }
    wgmma_wait<0>(); fence_regs(A);
    if (c.wtid == 0) mbar_arrive(c.empty + s);
  }
}

extern "C" __global__ void __launch_bounds__(384, 1)
dxn_lnbwd(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB, const __grid_constant__ CUtensorMap mX,
          const __grid_constant__ CUtensorMap mDX,
          const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx,
          const float* __restrict__ gamma, const float* __restrict__ rstd, const float* __restrict__ c1,
          float* __restrict__ pdg, float* __restrict__ pdb, int M) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* empty = full + NSTAGE;
  float* gs = reinterpret_cast<float*>(sm + F_GAM);
  float* dgs = reinterpret_cast<float*>(sm + F_DG);
  float* dbs = reinterpret_cast<float*>(sm + F_DB);
  const int tiles = M / ROWB;
  for (int i = tid; i < D; i += 384) gs[i] = gamma[i];
  for (int i = tid; i < 8 * D; i += 384) { dgs[i] = 0.f; dbs[i] = 0.f; }
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid == 0) {
      uint32_t it = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        const int m0 = t * ROWB;
        for (int kb = 0; kb < KB; ++kb, ++it) {
          const int s = (int)(it % NSTAGE);
          if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
          uint8_t* st = sm + s * F_STAGE;
          mbar_arrive_expect_tx(full + s, F_STAGE);
          tma_load_2d(st + F_A, &mA, full + s, kb * TBK, m0);
          for (int b = 0; b < D / NW; ++b) tma_load_2d(st + F_B + b * NW * 128, &mB, full + s, kb * TBK, b * NW);
        }
      }
    }
  } else {
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  Cx c;
  c.su = su; c.full = full; c.empty = empty; c.sm = sm; c.x = x; c.dy = dy; c.mdx = &mDX;
  c.gs = gs; c.dgs = dgs; c.dbs = dbs; c.xs = reinterpret_cast<float4*>(sm + F_XS);
  c.cw = wg - 1; c.wtid = tid & 127; c.warp = c.wtid >> 5; c.lane = tid & 31; c.q = c.lane & 3; c.slot = c.cw * 4 + c.warp;
  c.cbase = NW * c.cw; c.lr = 16 * c.warp + (c.lane >> 2); c.smi = c.lane >> 3;
  c.smrow = 16 * c.warp + 8 * (c.smi & 1) + (c.lane & 7); c.it = 0;
  c.dxs = sm + F_DXS + c.cw * 64 * NW * 2;
  const int ntl = tiles > (int)blockIdx.x ? (tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  float AX[64], AY[64];
  Ep ex, ey;
  auto tile_of = [&](int k) { return (int)blockIdx.x + k * (int)gridDim.x; };
  if (ntl > 0) {
    epi_begin(c, ex, tile_of(0), 0); epi_stats(c, ex, rstd, c1); load_step<0, false>(c, ex, 0);
    mainloop<false>(AX, AY, c, ey);
    for (int k = 0; k < ntl; k += 2) {                    // X holds tile k
      if (k + 1 < ntl) {
        epi_begin(c, ey, tile_of(k + 1), k + 1); epi_stats(c, ey, rstd, c1);
        mainloop<true>(AY, AX, c, ex);                    // tile k + 1 -> Y, epilogue of tile k (X) under it
        load_step<0, false>(c, ey, 0);                    // step 0 operands of tile k + 1
      } else { epi_all(c, ex, AX); break; }
      if (k + 2 < ntl) {                                  // Y holds tile k + 1
        epi_begin(c, ex, tile_of(k + 2), k + 2); epi_stats(c, ex, rstd, c1);
        mainloop<true>(AX, AY, c, ey);
        load_step<0, false>(c, ex, 0);
      } else { epi_all(c, ey, AY); break; }
    }
  }
  if (c.wtid == 0) tma_store_wait_all();
  named_bar_sync(6, 256);
  for (int cc = tid - 128; cc < D; cc += 256) {
    float sg = 0.f, sb = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) { sg += dgs[w * D + cc]; sb += dbs[w * D + cc]; }
    pdg[(size_t)blockIdx.x * D + cc] = sg; pdb[(size_t)blockIdx.x * D + cc] = sb;
  }
  }
}
