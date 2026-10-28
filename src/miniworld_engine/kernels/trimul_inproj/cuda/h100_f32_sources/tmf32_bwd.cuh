// tmf32_bwd.cuh -- backward pieces of the fp32 / TF32 bidirectional TriMul (see tmf32.cuh for the forward and the layouts).
// SPDX-License-Identifier: Apache-2.0
//
//   gate_bwd_kernel     dy, gate logit, projection -> d_proj, d_glogit (TF32-rounded), and the two per-token LN_out sums s1, s2
//   lnout_bwd_kernel    d_on = d_proj Wo  as a wgmma GEMM with the CHANNELS as the M dimension (so the accumulators are [channel][token] and the
//                       LayerNorm backward + the c-major write of d_tri are element-wise on them; dgamma / dbeta_out accumulate in registers)
//   front_bwd_kernel    the front GEMM again (g, p recomputed, never saved) with the backward in its epilogue -> d_pre [4][256][T] (the gradient of g and p of both sides)
//   ln_in_bwd_kernel    dxn, x, stats, dy -> dx (+ the residual), per-block dgamma / dbeta_in partials
//
// LN_out backward without a cross-channel reduction in the GEMM: with on_c = w_c xh_c + b_c and proj = on Wo^T,
//   s1[t] = sum_c w_c d_on[t,c]          = d_proj[t,:] . (Wo w)
//   s2[t] = sum_c w_c d_on[t,c] xh[t,c]  = d_proj[t,:] . (proj[t,:] - Wo b)
// so both are 128-term dot products over data the gate backward already holds (proj is the forward's saved projection).
#pragma once
#include "tmf32.cuh"

namespace tmf {

// u = Wo w, v = Wo b (two [128] vectors), and woT = rna(Wo^T) [256][128].
__global__ void wo_prep_kernel(const float* __restrict__ wo, const float* __restrict__ w, const float* __restrict__ b,
                               float* __restrict__ woT, float* __restrict__ uv) {
  const int k = blockIdx.x;                           // output row of Wo: 128 blocks, 256 threads = the input channels
  const int c = threadIdx.x;
  const float v = wo[k * 256 + c];
  woT[c * 128 + k] = rna(v);
  float a = v * w[c], bb = v * b[c];
  __shared__ float ra[8], rb[8];
#pragma unroll
  for (int o = 16; o; o >>= 1) { a += __shfl_xor_sync(~0u, a, o); bb += __shfl_xor_sync(~0u, bb, o); }
  if ((c & 31) == 0) { ra[c >> 5] = a; rb[c >> 5] = bb; }
  __syncthreads();
  if (c == 0) {
    float sa = 0.f, sb = 0.f;
    for (int i = 0; i < 8; ++i) { sa += ra[i]; sb += rb[i]; }
    uv[k] = sa; uv[128 + k] = sb;
  }
}

// ------------------------------------------------------------------------------------------------------------------------------
// y = x + ds * sigmoid(g) * proj  =>  d_proj = dy ds s,  d_glogit = dy ds proj s (1 - s).  One warp per token row.
__global__ void __launch_bounds__(256) gate_bwd_kernel(const float* __restrict__ dy, const float* __restrict__ gs, const float* __restrict__ ps,
                                                       const float* __restrict__ ds, const float* __restrict__ uv, float* __restrict__ dproj,
                                                       float* __restrict__ dgl, float2* __restrict__ s12, int64_t T, int L) {
  const int64_t row = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= T) return;
  const int j = (int)(row % L);
  const float4 d = reinterpret_cast<const float4*>(dy + row * 128)[lane];
  const float4 g = reinterpret_cast<const float4*>(gs + row * 128)[lane];
  const float4 p = reinterpret_cast<const float4*>(ps + row * 128)[lane];
  float4 sc = make_float4(1.f, 1.f, 1.f, 1.f);
  if (ds != nullptr) sc = reinterpret_cast<const float4*>(ds + (size_t)j * 128)[lane];
  const float4 u = reinterpret_cast<const float4*>(uv)[lane], v = reinterpret_cast<const float4*>(uv + 128)[lane];
  const float dd[4] = {d.x * sc.x, d.y * sc.y, d.z * sc.z, d.w * sc.w};
  const float gg[4] = {g.x, g.y, g.z, g.w}, pp[4] = {p.x, p.y, p.z, p.w}, uu[4] = {u.x, u.y, u.z, u.w}, vv[4] = {v.x, v.y, v.z, v.w};
  float dp[4], dg[4], s1 = 0.f, s2 = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float s = sigm(gg[i]);
    dp[i] = dd[i] * s;
    dg[i] = dd[i] * pp[i] * s * (1.f - s);
    s1 += dp[i] * uu[i];
    s2 += dp[i] * (pp[i] - vv[i]);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { s1 += __shfl_xor_sync(~0u, s1, o); s2 += __shfl_xor_sync(~0u, s2, o); }
  reinterpret_cast<float4*>(dproj + row * 128)[lane] = make_float4(rna(dp[0]), rna(dp[1]), rna(dp[2]), rna(dp[3]));
  reinterpret_cast<float4*>(dgl + row * 128)[lane] = make_float4(rna(dg[0]), rna(dg[1]), rna(dg[2]), rna(dg[3]));
  if (lane == 0) s12[row] = make_float2(s1, s2);
}

// ------------------------------------------------------------------------------------------------------------------------------
// d_tri[c][t] = rstd_t (w_c d_on[c][t] - s1_t / 256 - xh[c][t] s2_t / 256),   d_on^T [256][128 tokens] = Wo^T [256][128] x d_proj^T.
// A = woT (resident in shared memory, 4 slabs of [256][32]), B = d_proj tiles [128 tokens][32] (4-stage ring), two consumer warpgroups of 128
// channels (two m64 blocks) each.  dgamma / dbeta partials: pdg / pdb [gridDim][256].
constexpr int LB_NSTAGE = 4, LB_BSTAGE = 128 * 128;
constexpr int LB_SMEM = 4 * 32768 + LB_NSTAGE * LB_BSTAGE + (2 * LB_NSTAGE + 1) * 8 + 1024;

__global__ void __launch_bounds__(384, 1)
lnout_bwd_kernel(const __grid_constant__ CUtensorMap mAw, const __grid_constant__ CUtensorMap mBd, int64_t T,
                 const float* __restrict__ tri, const float* __restrict__ mean, const float* __restrict__ rstd,
                 const float2* __restrict__ s12, const float* __restrict__ gamma, float* __restrict__ dtri, float* __restrict__ pdg,
                 float* __restrict__ pdb) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + 4 * 32768 + LB_NSTAGE * LB_BSTAGE);
  uint64_t* empty = full + LB_NSTAGE;
  uint64_t* abar = empty + LB_NSTAGE;
  const int64_t mtiles = (T + BM - 1) / BM;
  if (tid == 0) {
    for (int s = 0; s < LB_NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    mbar_init(abar, 1);
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    mbar_arrive_expect_tx(abar, 4 * 32768);
    for (int kc = 0; kc < 4; ++kc) tma_load_2d(sm + kc * 32768, &mAw, abar, kc * BK, 0);
    uint32_t it = 0;
    for (int64_t t = blockIdx.x; t < mtiles; t += gridDim.x) {
      const int m0 = (int)t * BM;
      for (int kb = 0; kb < 4; ++kb, ++it) {
        const int s = (int)(it % LB_NSTAGE);
        if (it >= LB_NSTAGE) mbar_wait(empty + s, ((it / LB_NSTAGE) - 1) & 1u);
        mbar_arrive_expect_tx(full + s, LB_BSTAGE);
        tma_load_2d(sm + 4 * 32768 + s * LB_BSTAGE, &mBd, full + s, kb * BK, m0);
      }
    }
    return;
  }
  setmaxnreg_inc<232>();
  const int cw = wg - 1;
  mbar_wait(abar, 0);
  const int q = lane & 3, rg = lane >> 2;
  float gam[2][2], dgs[2][2], dbs[2][2];
#pragma unroll
  for (int mb = 0; mb < 2; ++mb)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      gam[mb][hf] = gamma[128 * cw + 64 * mb + 16 * warp + rg + 8 * hf];
      dgs[mb][hf] = 0.f; dbs[mb][hf] = 0.f;
    }
  uint32_t it = 0;
  for (int64_t t = blockIdx.x; t < mtiles; t += gridDim.x) {
    const int64_t t0 = t * BM;
    float acc0[64], acc1[64];
#pragma unroll
    for (int e = 0; e < 64; ++e) { acc0[e] = 0.f; acc1[e] = 0.f; }
    int prev = -1;
    for (int kb = 0; kb < 4; ++kb, ++it) {
      const int s = (int)(it % LB_NSTAGE);
      mbar_wait(full + s, (it / LB_NSTAGE) & 1u);
      const uint32_t bst = su + 4 * 32768 + s * LB_BSTAGE;
      const uint32_t a0 = su + kb * 32768 + (128 * cw) * 128, a1 = a0 + 64 * 128;
      fence_regs(acc0); fence_regs(acc1); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        mma_tf32<128>(acc0, dsc(a0 + ks * 32), dsc(bst + ks * 32));
        mma_tf32<128>(acc1, dsc(a1 + ks * 32), dsc(bst + ks * 32));
      }
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc0); fence_regs(acc1); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    wgmma_wait<0>(); fence_regs(acc0); fence_regs(acc1);
    if (wtid == 0) mbar_arrive(empty + prev);
#pragma unroll
    for (int g = 0; g < 16; ++g) {
      const int64_t tb = t0 + 8 * g + 2 * q;
      if (tb >= T) continue;
      const float2 mu = *reinterpret_cast<const float2*>(mean + tb), rs = *reinterpret_cast<const float2*>(rstd + tb);
      const float4 ss = *reinterpret_cast<const float4*>(s12 + tb);
      const float k0 = ss.x * (1.f / 256.f), k1 = ss.z * (1.f / 256.f);      // s1 / 256 of the two tokens
      const float h0 = ss.y * (1.f / 256.f), h1 = ss.w * (1.f / 256.f);      // s2 / 256
#pragma unroll
      for (int mb = 0; mb < 2; ++mb) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int c = 128 * cw + 64 * mb + 16 * warp + rg + 8 * hf;
          const float v0 = (mb == 0 ? acc0 : acc1)[4 * g + 2 * hf], v1 = (mb == 0 ? acc0 : acc1)[4 * g + 2 * hf + 1];
          const float2 tv = *reinterpret_cast<const float2*>(tri + (size_t)c * T + tb);
          const float x0 = (tv.x - mu.x) * rs.x, x1 = (tv.y - mu.y) * rs.y;
          const float gm = gam[mb][hf];
          float2 o;
          o.x = rs.x * (gm * v0 - k0 - x0 * h0);
          o.y = rs.y * (gm * v1 - k1 - x1 * h1);
          *reinterpret_cast<float2*>(dtri + (size_t)c * T + tb) = o;
          dgs[mb][hf] += v0 * x0 + v1 * x1;
          dbs[mb][hf] += v0 + v1;
        }
      }
    }
  }
#pragma unroll
  for (int mb = 0; mb < 2; ++mb)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      float a = dgs[mb][hf], b = dbs[mb][hf];
      a += __shfl_xor_sync(~0u, a, 1); a += __shfl_xor_sync(~0u, a, 2);
      b += __shfl_xor_sync(~0u, b, 1); b += __shfl_xor_sync(~0u, b, 2);
      if (q == 0) {
        const int c = 128 * cw + 64 * mb + 16 * warp + rg + 8 * hf;
        pdg[(size_t)blockIdx.x * 256 + c] = a;
        pdb[(size_t)blockIdx.x * 256 + c] = b;
      }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------
// LN_in backward + residual: dx = dy + rstd (dxh - mean(dxh) - xh mean(dxh xh)),  dxh = w dxn,  xh = (x - mean) rstd.  One warp per row; each block
// walks a fixed set of rows (grid-stride over the block's 8 warps) and writes one dgamma / dbeta partial [gridDim][128] (fixed-order sums:
// bitwise repeatable).
__global__ void __launch_bounds__(256) ln_in_bwd_kernel(const float* __restrict__ dxn, const float* __restrict__ x, const float* __restrict__ mean,
                                                        const float* __restrict__ rstd, const float* __restrict__ dy, const float* __restrict__ w,
                                                        float* __restrict__ dx, float* __restrict__ pdg, float* __restrict__ pdb, int64_t T) {
  __shared__ float4 sg[8][32], sb[8][32];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float4 ww = reinterpret_cast<const float4*>(w)[lane];
  float4 ag = make_float4(0.f, 0.f, 0.f, 0.f), ab = ag;
  for (int64_t row = (int64_t)blockIdx.x * 8 + warp; row < T; row += (int64_t)gridDim.x * 8) {
    const float4 dn = reinterpret_cast<const float4*>(dxn + row * 128)[lane];
    const float4 xv = reinterpret_cast<const float4*>(x + row * 128)[lane];
    const float4 dyv = reinterpret_cast<const float4*>(dy + row * 128)[lane];
    const float mu = mean[row], rs = rstd[row];
    const float xh[4] = {(xv.x - mu) * rs, (xv.y - mu) * rs, (xv.z - mu) * rs, (xv.w - mu) * rs};
    const float dh[4] = {dn.x * ww.x, dn.y * ww.y, dn.z * ww.z, dn.w * ww.w};
    float a = dh[0] + dh[1] + dh[2] + dh[3];
    float b = dh[0] * xh[0] + dh[1] * xh[1] + dh[2] * xh[2] + dh[3] * xh[3];
#pragma unroll
    for (int o = 16; o; o >>= 1) { a += __shfl_xor_sync(~0u, a, o); b += __shfl_xor_sync(~0u, b, o); }
    a *= (1.f / 128.f); b *= (1.f / 128.f);
    float4 o4;
    o4.x = dyv.x + rs * (dh[0] - a - xh[0] * b);
    o4.y = dyv.y + rs * (dh[1] - a - xh[1] * b);
    o4.z = dyv.z + rs * (dh[2] - a - xh[2] * b);
    o4.w = dyv.w + rs * (dh[3] - a - xh[3] * b);
    reinterpret_cast<float4*>(dx + row * 128)[lane] = o4;
    ag.x += dn.x * xh[0]; ag.y += dn.y * xh[1]; ag.z += dn.z * xh[2]; ag.w += dn.w * xh[3];
    ab.x += dn.x; ab.y += dn.y; ab.z += dn.z; ab.w += dn.w;
  }
  sg[warp][lane] = ag; sb[warp][lane] = ab;
  __syncthreads();
  if (warp == 0) {
    float4 tg = sg[0][lane], tb = sb[0][lane];
    for (int i = 1; i < 8; ++i) {
      const float4 g = sg[i][lane], b = sb[i][lane];
      tg.x += g.x; tg.y += g.y; tg.z += g.z; tg.w += g.w;
      tb.x += b.x; tb.y += b.y; tb.z += b.z; tb.w += b.w;
    }
    reinterpret_cast<float4*>(pdg + (size_t)blockIdx.x * 128)[lane] = tg;
    reinterpret_cast<float4*>(pdb + (size_t)blockIdx.x * 128)[lane] = tb;
  }
}

// ------------------------------------------------------------------------------------------------------------------------------
// front_bwd: the front GEMM again (g, p of one (m-tile, side, half) unit) with the backward in its epilogue instead of saving g and p:
//   d_g = d_out p s (1 - s) pm,  d_p = d_out s pm,  s = sigmoid(g),  d_out = d_planes[side*256 + ch][t]
// written to dpre[side*2 + {g, p}][ch][t].  Recomputing the 128-term dot products costs less than writing g and p in the forward and
// reading them here (the GEMM is compute-bound at ~200 TFLOP/s, the pre-activations are 2.4 GB at L = 768).
__global__ void __launch_bounds__(384, 1)
front_bwd_kernel(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB, int64_t T, int L,
                 const uint8_t* __restrict__ tmask, const float* __restrict__ dpl, float* __restrict__ dpre) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + F_NSTAGE * F_STAGE);
  uint64_t* empty = full + F_NSTAGE;
  const int64_t mtiles = (T + BM - 1) / BM, tiles = mtiles * F_NT;
  if (tid == 0) {
    for (int s = 0; s < F_NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    uint32_t it = 0;
    for (int64_t t = blockIdx.x; t < tiles; t += gridDim.x) {
      const int m0 = (int)(t / F_NT) * BM, nt = (int)(t % F_NT);
      const int grow = (nt >> 1) * 512 + (nt & 1) * 128;
      for (int kb = 0; kb < F_KB; ++kb, ++it) {
        const int s = (int)(it % F_NSTAGE);
        if (it >= F_NSTAGE) mbar_wait(empty + s, ((it / F_NSTAGE) - 1) & 1u);
        uint8_t* st = sm + s * F_STAGE;
        mbar_arrive_expect_tx(full + s, F_STAGE);
        tma_load_2d(st, &mA, full + s, kb * BK, m0);
        tma_load_2d(st + F_A, &mB, full + s, kb * BK, grow);
        tma_load_2d(st + F_A + 128 * 128, &mB, full + s, kb * BK, grow + 256);
      }
    }
    return;
  }
  setmaxnreg_inc<232>();
  const int cw = wg - 1;
  uint32_t it = 0;
  for (int64_t t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int m0 = (int)(t / F_NT) * BM, nt = (int)(t % F_NT);
    const int side = nt >> 1, half = nt & 1;
    float acc[F_BN / 2];
#pragma unroll
    for (int e = 0; e < F_BN / 2; ++e) acc[e] = 0.f;
    int prev = -1;
    for (int kb = 0; kb < F_KB; ++kb, ++it) {
      const int s = (int)(it % F_NSTAGE);
      mbar_wait(full + s, (it / F_NSTAGE) & 1u);
      const uint32_t a0 = su + s * F_STAGE + cw * 8192, b0 = su + s * F_STAGE + F_A;
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma_tf32<F_BN>(acc, dsc(a0 + ks * 32), dsc(b0 + ks * 32));
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    // The d_planes of this thread's two rows are requested while the last MMAs are still running.
    const int rl = 16 * warp + (lane >> 2), c0 = 2 * (lane & 3);
    const float* dp0 = dpl + (size_t)(side * 256 + half * 128 + c0) * T;
    float* og = dpre + (size_t)((side * 2) * 256 + half * 128 + c0) * T;
    float* op = dpre + (size_t)((side * 2 + 1) * 256 + half * 128 + c0) * T;
    const int64_t rr0 = (int64_t)m0 + 64 * cw + rl, rr1 = rr0 + 8;
    float dl0[32], dl1[32];
#pragma unroll
    for (int g = 0; g < F_BN / 16; ++g) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        dl0[2 * g + i] = rr0 < T ? dp0[(size_t)(8 * g + i) * T + rr0] : 0.f;
        dl1[2 * g + i] = rr1 < T ? dp0[(size_t)(8 * g + i) * T + rr1] : 0.f;
      }
    }
    wgmma_wait<0>(); fence_regs(acc);
    if (wtid == 0) mbar_arrive(empty + prev);
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int64_t r = hf == 0 ? rr0 : rr1;
      if (r >= T) continue;
      float pm = 1.f;
      if (tmask != nullptr) {
        const int i = (int)(r / L), j = (int)(r - (int64_t)i * L);
        pm = (tmask[i] != 0 && tmask[j] != 0) ? 1.f : 0.f;
      }
#pragma unroll
      for (int g = 0; g < F_BN / 16; ++g) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const float a = acc[4 * g + 2 * hf + i], b = acc[4 * (g + F_BN / 16) + 2 * hf + i];
          const float s = sigm(a), d = (hf == 0 ? dl0[2 * g + i] : dl1[2 * g + i]) * pm;
          og[(size_t)(8 * g + i) * T + r] = d * b * s * (1.f - s);
          op[(size_t)(8 * g + i) * T + r] = d * s;
        }
      }
    }
  }
}

}  // namespace tmf
