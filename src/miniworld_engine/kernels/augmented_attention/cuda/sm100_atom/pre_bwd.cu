// pre_bwd.cu — the atom DiT block's attention-input backward on sm_100a (the bf16 module's autograd rounding points):
//   dx1 = the bf16 sum of rn(dq Wq), rn(dk Wk), rn(dv Wv), rn(dg Wg)          (dP = [dq | dk | dv | dg] [M, 512])
//   x1 = rn(rn(s1 xn) + bi1), xn = rn(LN(a)):  dbi1 = dx1;  ds1 = rn(dx1 xn);  dxn = rn(dx1 s1);  dsc1 = rn(rn(ds1 rn(1 - s1)) s1)
//   d single = rn(DA + rn(LN_bwd(dxn)))                                     (DA: post_bwd's residual path)
// -> d single [M, 128]; dmod blocks 0 (dsc1), 1 (dbi1) of the backward layout; d bsc1 into DBIAS[0:128], d bq = sum_rows dq into DBQ.
// Structure (post_fwd.cu's frame): transposed 16-row tiles; W^T = [Wq; Wk; Wv; Wg]^T (128 x 512) in TMEM as the bf16 A operand (256
// columns); the two compute warpgroups take alternate tiles, each with its own TMA producer warp, MMA warp, NSW-stage ring and four
// accumulators dx_p^T = W_p^T dP_p^T (M = 128 channels, N = 16 rows, K = 128).
// Stage (32 KB): dP [16][512] | a | s1 -> dsc1 | DA -> d single | dx1.
// TMEM: W^T 0 (256 columns, projection p at 64 p); warpgroup w's accumulators at 256 + 64 w + 16 p.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef NSW
#define NSW 3                              // input stages per warpgroup
#endif
#ifndef ORD
#define ORD 0                              // the order of the four bf16 additions (autograd's)
#endif
constexpr int R = 16;
constexpr int KB = R * 128;                                                // [16 rows][64] bf16, SW128 (2 KB)
constexpr int T_ = 2 * KB;                                                 // [16][128] row-major tile (4 KB)
constexpr int O_DP = 0, O_A = 4 * T_, O_S1 = 5 * T_, O_DA = 6 * T_, O_X = 7 * T_, STG = 8 * T_;
constexpr int O_BAR = (2 * NSW * STG > 4 * 32768 ? 2 * NSW * STG : 4 * 32768), O_RED = O_BAR + 512;
constexpr int SMEM = O_RED + 8 * 256 * 4;
static_assert(SMEM <= 232448, "shared memory");
constexpr uint32_t T_W = 0, T_ACC = 256;
constexpr uint32_t I_16 = idesc_bf16(128, 16);

struct Bars {
  uint64_t wfull, wfree, infull[2][NSW], infree[2][NSW], dxfull[2], dxfree[2];
  uint32_t tmem;
};
DEVI uint32_t bmul2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t badd2(uint32_t a, uint32_t b) { uint32_t r; asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t bsub2(uint32_t a, uint32_t b) { uint32_t r; asm("sub.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t sig_bwd2(uint32_t g, uint32_t y) { return bmul2(bmul2(g, bsub2(0x3F803F80u, y)), y); }   // torch's bf16 sigmoid backward
DEVI void tmem_ld16x256b2(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
DEVI void stsm4t(uint32_t a, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
atom_pre_bwd_sm100(const __grid_constant__ CUtensorMap mdp, const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mmod,
                   const __grid_constant__ CUtensorMap mda, const __grid_constant__ CUtensorMap mwt, const __grid_constant__ CUtensorMap mds,
                   const __grid_constant__ CUtensorMap mdmod, float* __restrict__ DBIAS, float* __restrict__ DBQ, int ntile, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto row0 = [&](int T) { return R * ((int)blockIdx.x + T * (int)gridDim.x); };
  auto stage = [&](int w, int i) { return su + (uint32_t)((w * NSW + i % NSW) * STG); };

  if (tid == 0) {
    mbar_init(&B.wfull, 1); mbar_init(&B.wfree, 1);
    for (int w = 0; w < 2; ++w) {
      for (int s = 0; s < NSW; ++s) { mbar_init(&B.infull[w][s], 1); mbar_init(&B.infree[w][s], 1); }
      mbar_init(&B.dxfull[w], 1); mbar_init(&B.dxfree[w], 4);
    }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA producers (warp 0: warpgroup 0, 2: 1)
    const int w = warp >> 1;
    if (lane == 0) {
      if (w == 0) {
        mbar_expect_tx(&B.wfull, 4 * 32768);                              // W^T units: projection p = columns 128 p ..
        for (int i = 0; i < 16; ++i) tma_load_2d(su + (i >> 2) * 32768 + (i & 3) * 8192, &mwt, &B.wfull, 32 * i, 0);
      }
      mbar_wait(&B.wfree, 0);
      pdl_wait();                                                          // dP / DA come from the previous kernels
      for (int i = 0; 2 * i + w < ntT; ++i) {
        const int s = i % NSW, r0 = row0(2 * i + w);
        if (i >= NSW) mbar_wait(&B.infree[w][s], ((i / NSW) - 1) & 1);
        const uint32_t st = stage(w, i);
        uint64_t* bar = &B.infull[w][s];
        mbar_expect_tx(bar, 14 * KB);
        for (int kb = 0; kb < 8; ++kb) tma_load_2d(st + O_DP + kb * KB, &mdp, bar, 64 * kb, r0);
        for (int kb = 0; kb < 2; ++kb) {
          tma_load_2d(st + O_A + kb * KB, &ma, bar, 64 * kb, r0);
          tma_load_2d(st + O_S1 + kb * KB, &mmod, bar, 64 * kb, r0);        // s1: the forward mod's block 0
          tma_load_2d(st + O_DA + kb * KB, &mda, bar, 64 * kb, r0);
        }
      }
    }
  } else if (warp == 1 || warp == 3) {
    // ------------------------------------------------------------------------------------------------ MMA issuers (warp 1: warpgroup 0, 3: 1)
    const int w = warp >> 1;
    if (w == 0) {
      mbar_wait(&B.wfull, 0);
      tc_fence_after();
      if (elect_one()) {
        for (int u = 0; u < 4; ++u)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + T_W + 64 * u + ks * 8, desc_sw64(su + u * 32768 + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
        tc_commit(&B.wfree);
      }
      __syncwarp();
    }
    mbar_wait(&B.wfree, 0);
    const uint32_t ta = tmem + T_ACC + 64 * w;
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][i % NSW], (i / NSW) & 1);
      if (i > 0) mbar_wait(&B.dxfree[w], (i - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int p = 0; p < 4; ++p)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ts(ta + 16 * p, tmem + T_W + 64 * p + ks * 8, desc_k128(st + O_DP + (2 * p + (ks >> 2)) * KB) + (uint64_t)((ks & 3) * 2), I_16,
                    ks > 0 ? 1u : 0u);
        tc_commit(&B.dxfull[w]);
      }
      __syncwarp();
    }
  } else {
    // ------------------------------------------------------------------------------------------------ compute warpgroups
    const int w = (warp - 4) >> 2, qw = warp & 3, ct = tid & 127;
    const uint32_t lb = (uint32_t)qw * 32;
    const bool leader = qw == 0 && lane == 0;
    const uint32_t ta = tmem + T_ACC + 64 * w;
    const int ar = ct >> 3, ak = ct & 7;
    const uint32_t aoff = sw128((uint32_t)ar, (uint32_t)ak);
    const uint32_t fj = (uint32_t)(8 * ((lane >> 3) & 1) + (lane & 7)), fh = (uint32_t)(lane >> 4);
    auto faddr = [&](uint32_t base, uint32_t ch) { return base + (ch >> 6) * KB + fj * 128u + ((((ch & 63u) >> 3) ^ (fj & 7u)) << 4); };
    float bs1[16], bq[16];                                                 // d bsc1, d bq over this thread's rows (channels 8 ak + e, 64 + 8 ak + e)
#pragma unroll
    for (int e = 0; e < 16; ++e) { bs1[e] = 0.f; bq[e] = 0.f; }
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const int r0 = row0(2 * i + w);
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][i % NSW], (i / NSW) & 1);
      // ---- P1: dx1 = the bf16 sum of the four projections' input gradients -> row-major
      mbar_wait(&B.dxfull[w], i & 1);
      tc_fence_after();
#pragma unroll
      for (int L = 0; L < 2; ++L) {
        uint32_t v[4][8], r[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 16 * p, v[p]);
        tmem_wait_ld();
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
          const int k = 4 * (mi & 1) + 2 * (mi >> 1);
          uint32_t g[4];
#pragma unroll
          for (int p = 0; p < 4; ++p) g[p] = pack_bf16(__uint_as_float(v[p][k]), __uint_as_float(v[p][k + 1]));
          // autograd's accumulation order of x1's four gradients (q 0, k 1, v 2, g 3)
          r[mi] = ORD == 0 ? badd2(badd2(badd2(g[3], g[2]), g[1]), g[0]) : ORD == 1 ? badd2(badd2(badd2(g[0], g[1]), g[2]), g[3])
                : ORD == 2 ? badd2(badd2(badd2(g[2], g[1]), g[0]), g[3]) : badd2(badd2(badd2(g[3], g[0]), g[1]), g[2]);
        }
        stsm4t(faddr(st + O_X, lb + 16 * L + 8 * fh), r);
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dxfree[w]);
      named_bar_sync(1 + w, 128);
      // ---- P2: row stage
      {
        float x[16];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 a = lds128(st + O_A + kb * KB + aoff), dq = lds128(st + O_DP + kb * KB + aoff);
          const uint32_t w4[4] = {a.x, a.y, a.z, a.w}, q4[4] = {dq.x, dq.y, dq.z, dq.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            x[8 * kb + 2 * e] = bf16lo(w4[e]); x[8 * kb + 2 * e + 1] = bf16hi(w4[e]);
            bq[8 * kb + 2 * e] += bf16lo(q4[e]); bq[8 * kb + 2 * e + 1] += bf16hi(q4[e]);
          }
        }
        float sum = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) sum += x[e];
        sum += __shfl_xor_sync(~0u, sum, 1); sum += __shfl_xor_sync(~0u, sum, 2); sum += __shfl_xor_sync(~0u, sum, 4);
        const float mean = sum * (1.f / 128.f);
        float var = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) { x[e] -= mean; var += x[e] * x[e]; }
        var += __shfl_xor_sync(~0u, var, 1); var += __shfl_xor_sync(~0u, var, 2); var += __shfl_xor_sync(~0u, var, 4);
        const float rstd = rsqrtf(var * (1.f / 128.f) + eps);
        float dxn[16], s1 = 0.f, s2 = 0.f;
        uint32_t dsc[8];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 dxu = lds128(st + O_X + kb * KB + aoff), gu = lds128(st + O_S1 + kb * KB + aoff);
          const uint32_t d4[4] = {dxu.x, dxu.y, dxu.z, dxu.w}, g4[4] = {gu.x, gu.y, gu.z, gu.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 8 * kb + 2 * e;
            x[k] *= rstd; x[k + 1] *= rstd;
            dsc[4 * kb + e] = sig_bwd2(bmul2(d4[e], pack_bf16(x[k], x[k + 1])), g4[e]);
            const uint32_t dn = bmul2(d4[e], g4[e]);
            dxn[k] = bf16lo(dn); dxn[k + 1] = bf16hi(dn);
            s1 += dxn[k] + dxn[k + 1];
            s2 += dxn[k] * x[k] + dxn[k + 1] * x[k + 1];
            bs1[k] += bf16lo(dsc[4 * kb + e]); bs1[k + 1] += bf16hi(dsc[4 * kb + e]);
          }
        }
        s1 += __shfl_xor_sync(~0u, s1, 1); s1 += __shfl_xor_sync(~0u, s1, 2); s1 += __shfl_xor_sync(~0u, s1, 4);
        s2 += __shfl_xor_sync(~0u, s2, 1); s2 += __shfl_xor_sync(~0u, s2, 2); s2 += __shfl_xor_sync(~0u, s2, 4);
        const float m1 = s1 * (1.f / 128.f), m2 = s2 * (1.f / 128.f);
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 dau = lds128(st + O_DA + kb * KB + aoff);
          const uint32_t y4[4] = {dau.x, dau.y, dau.z, dau.w};
          uint32_t ds[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 8 * kb + 2 * e;
            ds[e] = badd2(y4[e], pack_bf16(rstd * (dxn[k] - m1 - x[k] * m2), rstd * (dxn[k + 1] - m1 - x[k + 1] * m2)));
          }
          sts128(st + O_DA + kb * KB + aoff, make_uint4(ds[0], ds[1], ds[2], ds[3]));
          sts128(st + O_S1 + kb * KB + aoff, make_uint4(dsc[4 * kb], dsc[4 * kb + 1], dsc[4 * kb + 2], dsc[4 * kb + 3]));
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (leader) {
        for (int kb = 0; kb < 2; ++kb) {
          tma_store_2d(&mds, st + O_DA + kb * KB, 64 * kb, r0);
          tma_store_2d(&mdmod, st + O_S1 + kb * KB, 64 * kb, r0);          // dsc1 -> block 0
          tma_store_2d(&mdmod, st + O_X + kb * KB, 128 + 64 * kb, r0);     // dbi1 = dx1 -> block 1
        }
        tma_store_commit();
        tma_store_wait_read0();                                            // (a short wait: the stores only have to read the stage)
        mbar_arrive(&B.infree[w][i % NSW]);                                // this tile's stage is free for tile i + NSW
      }
    }
    if (leader) tma_store_wait0();
    float* red = reinterpret_cast<float*>(sm + O_RED);                      // [8 warps][256]
    const int cw = warp - 4;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      float v1 = bs1[e], vq = bq[e];
      v1 += __shfl_xor_sync(~0u, v1, 8); v1 += __shfl_xor_sync(~0u, v1, 16);
      vq += __shfl_xor_sync(~0u, vq, 8); vq += __shfl_xor_sync(~0u, vq, 16);
      const int c = e < 8 ? 8 * ak + e : 64 + 8 * ak + (e - 8);
      if ((lane >> 3) == 0) { red[cw * 256 + c] = v1; red[cw * 256 + 128 + c] = vq; }
    }
    named_bar_sync(3, 256);
    {
      const int c = tid - 128;
      float v = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) v += red[k * 256 + c];
      atomicAdd(c < 128 ? DBIAS + c : DBQ + (c - 128), v);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
