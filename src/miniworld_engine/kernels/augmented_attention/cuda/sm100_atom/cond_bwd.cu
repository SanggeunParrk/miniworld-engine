// cond_bwd.cu — the atom DiT block's conditioning backward on sm_100a (dmod in the backward layout [dsc1 | dbi1 | dsc2 | dbi2 | dos | dts]):
//   dcn_k = rn(dsc_k Wsc_k + dbi_k Wbi_k)  (k = 1, 2; one fp32 accumulation);  dco = rn(dos Wos);  dct = rn(dts Wts)
//   cn_k = rn(g_k xhat):  dx_k = rn(LN_bwd(dcn_k g_k)),  d g_k += sum_rows dcn_k xhat
//   dc = the bf16 sum of dct, dx_2, dco, dx_1 (autograd's order);  cn_1, cn_2 written for the dWmod GEMMs
// Structure (post_fwd.cu's frame): transposed 16-row tiles; Wmod^T (128 x 768, the six blocks in the forward order) in TMEM as bf16 A
// operands (384 columns); the two compute warpgroups take alternate tiles, each with its own TMA producer warp, MMA warp, 2-stage ring
// and four accumulators (M = 128 channels, N = 16 rows): dcn1^T, dcn2^T, dco^T, dct^T.
// Stage (44 KB): dmod [16][768] | c -> dc | dcn1 | dcn2 | dco -> cn1 | dct -> cn2.
// TMEM: Wmod^T block j at 64 j; warpgroup w's accumulators at 384 + 64 w + 16 q.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef NSW
#define NSW 2
#endif
#ifndef ORD
#define ORD 0                              // the order of dc's four bf16 additions
#endif
constexpr int R = 16;
constexpr int KB = R * 128;                                                // [16 rows][64] bf16, SW128 (2 KB)
constexpr int T_ = 2 * KB;                                                 // [16][128] row-major tile (4 KB)
constexpr int O_DM = 0, O_C = 6 * T_, O_N1 = 7 * T_, O_N2 = 8 * T_, O_CO = 9 * T_, O_CT = 10 * T_, STG = 11 * T_;
constexpr int O_BAR = (2 * NSW * STG > 6 * 32768 ? 2 * NSW * STG : 6 * 32768), O_RED = O_BAR + 512;
constexpr int SMEM = O_RED + 8 * 256 * 4;
static_assert(SMEM <= 232448, "shared memory");
constexpr uint32_t T_W = 0, T_ACC = 384;
constexpr uint32_t I_16 = idesc_bf16(128, 16);

struct Bars {
  uint64_t wfull, wfree, infull[2][NSW], infree[2][NSW], dfull[2], dfree[2];
  uint32_t tmem;
};
DEVI uint32_t badd2(uint32_t a, uint32_t b) { uint32_t r; asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI void tmem_ld16x256b2(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
DEVI void stsm4t(uint32_t a, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
atom_cond_bwd_sm100(const __grid_constant__ CUtensorMap mdmod, const __grid_constant__ CUtensorMap mc, const __grid_constant__ CUtensorMap mwt,
                    const __grid_constant__ CUtensorMap mdc, const __grid_constant__ CUtensorMap mcn1, const __grid_constant__ CUtensorMap mcn2,
                    const float* __restrict__ G1, const float* __restrict__ G2, float* __restrict__ DG, int ntile, float eps) {
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
      mbar_init(&B.dfull[w], 1); mbar_init(&B.dfree[w], 4);
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
        mbar_expect_tx(&B.wfull, 6 * 32768);                              // Wmod^T units: forward block j = columns 128 j ..
        for (int i = 0; i < 24; ++i) tma_load_2d(su + (i >> 2) * 32768 + (i & 3) * 8192, &mwt, &B.wfull, 32 * i, 0);
      }
      mbar_wait(&B.wfree, 0);
      pdl_wait();                                                          // dmod comes from the previous kernels
      for (int i = 0; 2 * i + w < ntT; ++i) {
        const int s = i % NSW, r0 = row0(2 * i + w);
        if (i >= NSW) mbar_wait(&B.infree[w][s], ((i / NSW) - 1) & 1);
        const uint32_t st = stage(w, i);
        uint64_t* bar = &B.infull[w][s];
        mbar_expect_tx(bar, 14 * KB);
        for (int kb = 0; kb < 12; ++kb) tma_load_2d(st + O_DM + kb * KB, &mdmod, bar, 64 * kb, r0);
        for (int kb = 0; kb < 2; ++kb) tma_load_2d(st + O_C + kb * KB, &mc, bar, 64 * kb, r0);
      }
    }
  } else if (warp == 1 || warp == 3) {
    // ------------------------------------------------------------------------------------------------ MMA issuers (warp 1: warpgroup 0, 3: 1)
    const int w = warp >> 1;
    if (w == 0) {
      mbar_wait(&B.wfull, 0);
      tc_fence_after();
      if (elect_one()) {
        for (int u = 0; u < 6; ++u)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + T_W + 64 * u + ks * 8, desc_sw64(su + u * 32768 + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
        tc_commit(&B.wfree);
      }
      __syncwarp();
    }
    mbar_wait(&B.wfree, 0);
    const uint32_t ta = tmem + T_ACC + 64 * w;
    // accumulator q: (weight block, dmod block) pairs -- q 0: (0, 0) + (1, 1); q 1: (3, 2) + (4, 3); q 2: (2, 4); q 3: (5, 5)
    auto mma = [&](uint32_t st, int q, int wb, int db, bool first) {
#pragma unroll
      for (int ks = 0; ks < 8; ++ks)
        umma_ts(ta + 16 * q, tmem + T_W + 64 * wb + ks * 8, desc_k128(st + O_DM + (2 * db + (ks >> 2)) * KB) + (uint64_t)((ks & 3) * 2), I_16,
                (first && ks == 0) ? 0u : 1u);
    };
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][i % NSW], (i / NSW) & 1);
      if (i > 0) mbar_wait(&B.dfree[w], (i - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
        mma(st, 0, 0, 0, true); mma(st, 0, 1, 1, false);
        mma(st, 1, 3, 2, true); mma(st, 1, 4, 3, false);
        mma(st, 2, 2, 4, true);
        mma(st, 3, 5, 5, true);
        tc_commit(&B.dfull[w]);
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
    float g1[16], g2[16], dg1[16], dg2[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const int c = e < 8 ? 8 * ak + e : 64 + 8 * ak + (e - 8);
      g1[e] = __ldg(G1 + c); g2[e] = __ldg(G2 + c); dg1[e] = 0.f; dg2[e] = 0.f;
    }
    const uint32_t stq[4] = {O_N1, O_N2, O_CO, O_CT};
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const int r0 = row0(2 * i + w);
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][i % NSW], (i / NSW) & 1);
      // ---- P1: the four accumulators -> rn -> row-major staging
      mbar_wait(&B.dfull[w], i & 1);
      tc_fence_after();
#pragma unroll
      for (int L = 0; L < 2; ++L) {
        uint32_t v[4][8];
#pragma unroll
        for (int q = 0; q < 4; ++q) tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 16 * q, v[q]);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          uint32_t r[4];
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) {
            const int k = 4 * (mi & 1) + 2 * (mi >> 1);
            r[mi] = pack_bf16(__uint_as_float(v[q][k]), __uint_as_float(v[q][k + 1]));
          }
          stsm4t(faddr(st + stq[q], lb + 16 * L + 8 * fh), r);
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dfree[w]);
      named_bar_sync(1 + w, 128);
      // ---- P2: row stage: LayerNorm recompute, the two LayerNorm backwards, dc; cn1 / cn2 over dco / dct
      {
        float x[16];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 cu = lds128(st + O_C + kb * KB + aoff);
          const uint32_t w4[4] = {cu.x, cu.y, cu.z, cu.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) { x[8 * kb + 2 * e] = bf16lo(w4[e]); x[8 * kb + 2 * e + 1] = bf16hi(w4[e]); }
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
#pragma unroll
        for (int e = 0; e < 16; ++e) x[e] *= rstd;                          // xhat
        float d1[16], d2[16], a1 = 0.f, b1 = 0.f, a2 = 0.f, b2 = 0.f;
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 n1 = lds128(st + O_N1 + kb * KB + aoff), n2 = lds128(st + O_N2 + kb * KB + aoff);
          const uint32_t p4[4] = {n1.x, n1.y, n1.z, n1.w}, q4[4] = {n2.x, n2.y, n2.z, n2.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 8 * kb + 2 * e;
            const float u0 = bf16lo(p4[e]), u1 = bf16hi(p4[e]), v0 = bf16lo(q4[e]), v1 = bf16hi(q4[e]);
            dg1[k] += u0 * x[k]; dg1[k + 1] += u1 * x[k + 1]; dg2[k] += v0 * x[k]; dg2[k + 1] += v1 * x[k + 1];
            d1[k] = u0 * g1[k]; d1[k + 1] = u1 * g1[k + 1]; d2[k] = v0 * g2[k]; d2[k + 1] = v1 * g2[k + 1];
            a1 += d1[k] + d1[k + 1]; b1 += d1[k] * x[k] + d1[k + 1] * x[k + 1];
            a2 += d2[k] + d2[k + 1]; b2 += d2[k] * x[k] + d2[k + 1] * x[k + 1];
          }
        }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
          a1 += __shfl_xor_sync(~0u, a1, o); b1 += __shfl_xor_sync(~0u, b1, o); a2 += __shfl_xor_sync(~0u, a2, o); b2 += __shfl_xor_sync(~0u, b2, o);
        }
        a1 *= 1.f / 128.f; b1 *= 1.f / 128.f; a2 *= 1.f / 128.f; b2 *= 1.f / 128.f;
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 co = lds128(st + O_CO + kb * KB + aoff), cq = lds128(st + O_CT + kb * KB + aoff);
          const uint32_t o4[4] = {co.x, co.y, co.z, co.w}, t4[4] = {cq.x, cq.y, cq.z, cq.w};
          uint32_t dc[4], c1[4], c2[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 8 * kb + 2 * e;
            const uint32_t x1 = pack_bf16(rstd * (d1[k] - a1 - x[k] * b1), rstd * (d1[k + 1] - a1 - x[k + 1] * b1));
            const uint32_t x2 = pack_bf16(rstd * (d2[k] - a2 - x[k] * b2), rstd * (d2[k + 1] - a2 - x[k + 1] * b2));
            dc[e] = ORD == 0 ? badd2(badd2(badd2(t4[e], x2), o4[e]), x1) : ORD == 1 ? badd2(badd2(badd2(x1, o4[e]), x2), t4[e])
                  : ORD == 2 ? badd2(badd2(badd2(t4[e], o4[e]), x2), x1) : badd2(badd2(badd2(x2, t4[e]), x1), o4[e]);
            c1[e] = pack_bf16(g1[k] * x[k], g1[k + 1] * x[k + 1]);
            c2[e] = pack_bf16(g2[k] * x[k], g2[k + 1] * x[k + 1]);
          }
          sts128(st + O_C + kb * KB + aoff, make_uint4(dc[0], dc[1], dc[2], dc[3]));
          sts128(st + O_CO + kb * KB + aoff, make_uint4(c1[0], c1[1], c1[2], c1[3]));
          sts128(st + O_CT + kb * KB + aoff, make_uint4(c2[0], c2[1], c2[2], c2[3]));
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (leader) {
        for (int kb = 0; kb < 2; ++kb) {
          tma_store_2d(&mdc, st + O_C + kb * KB, 64 * kb, r0);
          tma_store_2d(&mcn1, st + O_CO + kb * KB, 64 * kb, r0);
          tma_store_2d(&mcn2, st + O_CT + kb * KB, 64 * kb, r0);
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
      float v1 = dg1[e], v2 = dg2[e];
      v1 += __shfl_xor_sync(~0u, v1, 8); v1 += __shfl_xor_sync(~0u, v1, 16);
      v2 += __shfl_xor_sync(~0u, v2, 8); v2 += __shfl_xor_sync(~0u, v2, 16);
      const int c = e < 8 ? 8 * ak + e : 64 + 8 * ak + (e - 8);
      if ((lane >> 3) == 0) { red[cw * 256 + c] = v1; red[cw * 256 + 128 + c] = v2; }
    }
    named_bar_sync(3, 256);
    {
      const int c = tid - 128;
      float v = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) v += red[k * 256 + c];
      atomicAdd(DG + c, v);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
