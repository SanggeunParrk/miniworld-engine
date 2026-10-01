// post_bwd.cu — second part of the atom DiT block's post-attention backward on sm_100a (the bf16 module's autograd rounding points):
//   dx2 = rn(rn(da Wa) + rn(db Wb))                                        (DAB = [da | db] from tr_bwd_gate.cu)
//   x2 = rn(rn(s2 xn) + bi2), xn = rn(LN(a2)):  dbi2 = dx2;  ds2 = rn(dx2 xn);  dxn = rn(dx2 s2);  dsc2 = rn(rn(ds2 rn(1 - s2)) s2)
//   da2 = rn(dy + rn(LN_bwd(dxn)))                                         (a3 = a2 + y: the residual)
//   a2 = a + rn(so u):  da = da2 (-> DA, pre_bwd adds the AdaLN1 path);  dso = rn(da2 u);  du = rn(da2 so);  dos = sig_bwd(dso, so)
//   u = rn(gated Wo^T):  dgated = rn(du Wo);  gated = rn(sg o):  dsg = rn(dgated o);  dO = rn(dgated sg);  dg = sig_bwd(dsg, sg)
//   D[a, h, n] = sum over head h's 32 channels of dO o (fp32, the attention backward's row term)
// -> DA [M, 128]; dmod blocks 2 (dsc2), 3 (dbi2), 4 (dos) of the backward layout [dsc1 | dbi1 | dsc2 | dbi2 | dos | dts]; DU, GATED
//    (dWo = DU^T GATED); DO; dg into dP's block 3 (dP = [dq | dk | dv | dg]); D; d bsc2 / d bos (per-thread sums, one red.add at the end).
// Structure (post_fwd.cu's frame): transposed 16-row tiles; Wu^T = [Wa; Wb]^T (128 x 512) and Wo^T (128 x 128) in TMEM as bf16 A
// operands (320 columns); the two compute warpgroups take alternate tiles, each with its own TMA producer warp, MMA warp, 2-stage ring
// and 48 accumulator columns: dxa^T | dxb^T | dgated^T (M = 128 channels, N = 16 rows).
// Threads: accumulator stages in the mma-fragment layout (tcgen05.ld 16x256b; ldmatrix / stmatrix .trans against row-major tiles), the
// row stage (LayerNorm recompute and backward, gates) eight threads per row, 16 channels each; the leader thread issues the TMA stores.
// Stage (48 KB): dab [16][512] | dy -> gated | a2 -> da | s2 -> dsc2 | so -> dos | u -> du | sg -> dg | o -> dO | dx2.
// TMEM: Wu^T 0 (256 columns: hidden 0-255 = Wa at 0, Wb at 128), Wo^T 256; warpgroup w's accumulators at 384 + 64 w: dxa 0, dxb 16, dgated 32.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef NSW
#define NSW 2                              // input stages per warpgroup
#endif
constexpr int R = 16;
constexpr int KB = R * 128;                                                // [16 rows][64] bf16, SW128 (2 KB)
constexpr int T_ = 2 * KB;                                                 // [16][128] row-major tile (4 KB)
constexpr int O_DAB = 0, O_DY = 4 * T_, O_A2 = 5 * T_, O_S2 = 6 * T_, O_SO = 7 * T_, O_U = 8 * T_, O_SG = 9 * T_, O_O = 10 * T_, O_X = 11 * T_,
              STG = 12 * T_;
constexpr int O_BAR = (2 * NSW * STG > 5 * 32768 ? 2 * NSW * STG : 5 * 32768), O_RED = O_BAR + 512;
constexpr int SMEM = O_RED + 8 * 256 * 4;
static_assert(SMEM <= 232448, "shared memory");
constexpr uint32_t T_WU = 0, T_WO = 256, T_ACC = 384;
constexpr uint32_t I_16 = idesc_bf16(128, 16);

struct Bars {
  uint64_t wfull, wfree, infull[2][NSW], infree[2][NSW], dxfull[2], dufull[2], dgfull[2];
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
DEVI void ldsm4t(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

extern "C" __global__ void __launch_bounds__(384, 1)
atom_post_bwd_sm100(const __grid_constant__ CUtensorMap mdab, const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap ma2,
                    const __grid_constant__ CUtensorMap mmod, const __grid_constant__ CUtensorMap mu, const __grid_constant__ CUtensorMap msg,
                    const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mwut, const __grid_constant__ CUtensorMap mwot,
                    const __grid_constant__ CUtensorMap mda, const __grid_constant__ CUtensorMap mdmod, const __grid_constant__ CUtensorMap mdu,
                    const __grid_constant__ CUtensorMap mgated, const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mdp,
                    float* __restrict__ Dd, float* __restrict__ DBIAS, int N, int ntile, float eps) {
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
      mbar_init(&B.dxfull[w], 1); mbar_init(&B.dufull[w], 1); mbar_init(&B.dgfull[w], 1);
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
        // Wu^T (units 0-3: hidden 128 u ..) and Wo^T (unit 4) as K-major SW64 [128][32] atoms
        mbar_expect_tx(&B.wfull, 5 * 32768);
        for (int i = 0; i < 20; ++i) {
          const int u = i >> 2, ka = i & 3;
          tma_load_2d(su + u * 32768 + ka * 8192, u < 4 ? &mwut : &mwot, &B.wfull, u < 4 ? 128 * u + 32 * ka : 32 * ka, 0);
        }
      }
      mbar_wait(&B.wfree, 0);
      pdl_wait();                                                          // DAB / dy come from the previous kernels
      for (int i = 0; 2 * i + w < ntT; ++i) {
        const int s = i % NSW, r0 = row0(2 * i + w);
        if (i >= NSW) mbar_wait(&B.infree[w][s], ((i / NSW) - 1) & 1);
        const uint32_t st = stage(w, i);
        uint64_t* bar = &B.infull[w][s];
        mbar_expect_tx(bar, 22 * KB);
        for (int kb = 0; kb < 8; ++kb) tma_load_2d(st + O_DAB + kb * KB, &mdab, bar, 64 * kb, r0);
        for (int kb = 0; kb < 2; ++kb) {
          tma_load_2d(st + O_DY + kb * KB, &mdy, bar, 64 * kb, r0);
          tma_load_2d(st + O_A2 + kb * KB, &ma2, bar, 64 * kb, r0);
          tma_load_2d(st + O_S2 + kb * KB, &mmod, bar, 384 + 64 * kb, r0);  // the forward mod: s2 block 3, so block 2
          tma_load_2d(st + O_SO + kb * KB, &mmod, bar, 256 + 64 * kb, r0);
          tma_load_2d(st + O_U + kb * KB, &mu, bar, 64 * kb, r0);
          tma_load_2d(st + O_SG + kb * KB, &msg, bar, 64 * kb, r0);
          tma_load_2d(st + O_O + kb * KB, &mo, bar, 64 * kb, r0);
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
        for (int u = 0; u < 5; ++u)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + 64 * u + ks * 8, desc_sw64(su + u * 32768 + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
        tc_commit(&B.wfree);
      }
      __syncwarp();
    }
    mbar_wait(&B.wfree, 0);
    const uint32_t ta = tmem + T_ACC + 64 * w;
    auto bdesc = [&](uint32_t base, int ks) { return desc_k128(base + (ks >> 2) * KB) + (uint64_t)((ks & 3) * 2); };
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][i % NSW], (i / NSW) & 1);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {                                  // dxa^T = Wa^T da^T, dxb^T = Wb^T db^T (K = 256 each)
          umma_ts(ta, tmem + T_WU + ks * 8, bdesc(st + O_DAB, ks), I_16, ks > 0 ? 1u : 0u);
          umma_ts(ta + 16, tmem + T_WU + 128 + ks * 8, bdesc(st + O_DAB, 16 + ks), I_16, ks > 0 ? 1u : 0u);
        }
        tc_commit(&B.dxfull[w]);
      }
      __syncwarp();
      mbar_wait(&B.dufull[w], i & 1);                                      // du written over u
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) umma_ts(ta + 32, tmem + T_WO + ks * 8, bdesc(st + O_U, ks), I_16, ks > 0 ? 1u : 0u);
        tc_commit(&B.dgfull[w]);
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
    float bs2[16], bso[16];                                                // d bsc2, d bos over this thread's rows (channels 8 ak + e, 64 + 8 ak + e)
#pragma unroll
    for (int e = 0; e < 16; ++e) { bs2[e] = 0.f; bso[e] = 0.f; }
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const int r0 = row0(2 * i + w);
      const uint32_t st = stage(w, i);
      // ---- P1: dx2 = rn(rn(dxa) + rn(dxb)) -> row-major
      mbar_wait(&B.infull[w][i % NSW], (i / NSW) & 1);                     // (the stage's tiles are visible to these threads)
      mbar_wait(&B.dxfull[w], i & 1);
      tc_fence_after();
#pragma unroll
      for (int L = 0; L < 2; ++L) {
        uint32_t va[8], vb[8], r[4];
        tmem_ld16x256b2(ta + ((lb + 16 * L) << 16), va);
        tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 16, vb);
        tmem_wait_ld();
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
          const int k = 4 * (mi & 1) + 2 * (mi >> 1);
          r[mi] = badd2(pack_bf16(__uint_as_float(va[k]), __uint_as_float(va[k + 1])), pack_bf16(__uint_as_float(vb[k]), __uint_as_float(vb[k + 1])));
        }
        stsm4t(faddr(st + O_X, lb + 16 * L + 8 * fh), r);
      }
      tc_fence_before();
      named_bar_sync(1 + w, 128);
      // ---- P2: row stage
      {
        uint4 a2u[2], s2u[2], dxu[2];
        float x[16];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          a2u[kb] = lds128(st + O_A2 + kb * KB + aoff); s2u[kb] = lds128(st + O_S2 + kb * KB + aoff); dxu[kb] = lds128(st + O_X + kb * KB + aoff);
          const uint32_t w4[4] = {a2u[kb].x, a2u[kb].y, a2u[kb].z, a2u[kb].w};
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
        // xhat (fp32) in x; ds2 = rn(dx2 xn), dxn = rn(dx2 s2), dsc2 = sig_bwd(ds2, s2)
        float dxn[16], s1 = 0.f, s2 = 0.f;
        uint32_t dsc[8];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint32_t d4[4] = {dxu[kb].x, dxu[kb].y, dxu[kb].z, dxu[kb].w}, g4[4] = {s2u[kb].x, s2u[kb].y, s2u[kb].z, s2u[kb].w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 8 * kb + 2 * e;
            x[k] *= rstd; x[k + 1] *= rstd;
            const uint32_t xn = pack_bf16(x[k], x[k + 1]);
            dsc[4 * kb + e] = sig_bwd2(bmul2(d4[e], xn), g4[e]);
            const uint32_t dn = bmul2(d4[e], g4[e]);
            dxn[k] = bf16lo(dn); dxn[k + 1] = bf16hi(dn);
            s1 += dxn[k] + dxn[k + 1];
            s2 += dxn[k] * x[k] + dxn[k + 1] * x[k + 1];
          }
        }
        s1 += __shfl_xor_sync(~0u, s1, 1); s1 += __shfl_xor_sync(~0u, s1, 2); s1 += __shfl_xor_sync(~0u, s1, 4);
        s2 += __shfl_xor_sync(~0u, s2, 1); s2 += __shfl_xor_sync(~0u, s2, 2); s2 += __shfl_xor_sync(~0u, s2, 4);
        const float m1 = s1 * (1.f / 128.f), m2 = s2 * (1.f / 128.f);
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 dy = lds128(st + O_DY + kb * KB + aoff), so = lds128(st + O_SO + kb * KB + aoff), u = lds128(st + O_U + kb * KB + aoff);
          const uint32_t y4[4] = {dy.x, dy.y, dy.z, dy.w}, o4[4] = {so.x, so.y, so.z, so.w}, u4[4] = {u.x, u.y, u.z, u.w};
          uint32_t da[4], dos[4], du[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 8 * kb + 2 * e;
            const uint32_t dl = pack_bf16(rstd * (dxn[k] - m1 - x[k] * m2), rstd * (dxn[k + 1] - m1 - x[k + 1] * m2));
            da[e] = badd2(y4[e], dl);
            dos[e] = sig_bwd2(bmul2(da[e], u4[e]), o4[e]);
            du[e] = bmul2(da[e], o4[e]);
            bs2[k] += bf16lo(dsc[4 * kb + e]); bs2[k + 1] += bf16hi(dsc[4 * kb + e]);
            bso[k] += bf16lo(dos[e]); bso[k + 1] += bf16hi(dos[e]);
          }
          sts128(st + O_A2 + kb * KB + aoff, make_uint4(da[0], da[1], da[2], da[3]));
          sts128(st + O_S2 + kb * KB + aoff, make_uint4(dsc[4 * kb], dsc[4 * kb + 1], dsc[4 * kb + 2], dsc[4 * kb + 3]));
          sts128(st + O_SO + kb * KB + aoff, make_uint4(dos[0], dos[1], dos[2], dos[3]));
          sts128(st + O_U + kb * KB + aoff, make_uint4(du[0], du[1], du[2], du[3]));
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (leader) mbar_arrive(&B.dufull[w]);
      // ---- P3: dgated -> dsg, dO, dg, gated, D (fragment layout; warp qw = head qw)
      mbar_wait(&B.dgfull[w], i & 1);
      tc_fence_after();
      float dsum[4] = {0.f, 0.f, 0.f, 0.f};                                // rows 2 (lane % 4) + {0, 1} + 8 m: [2 m + q]
#pragma unroll
      for (int L = 0; L < 2; ++L) {
        const uint32_t ch = lb + 16 * L + 8 * fh;
        uint32_t v[8], sg[4], o[4], dO[4], dg[4], ga[4];
        tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 32, v);
        ldsm4t(faddr(st + O_SG, ch), sg);
        ldsm4t(faddr(st + O_O, ch), o);
        tmem_wait_ld();
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
          const int k = 4 * (mi & 1) + 2 * (mi >> 1);
          const uint32_t dgt = pack_bf16(__uint_as_float(v[k]), __uint_as_float(v[k + 1]));
          dO[mi] = bmul2(dgt, sg[mi]);
          dg[mi] = sig_bwd2(bmul2(dgt, o[mi]), sg[mi]);
          ga[mi] = bmul2(sg[mi], o[mi]);
          dsum[2 * (mi & 1)] += bf16lo(dO[mi]) * bf16lo(o[mi]);
          dsum[2 * (mi & 1) + 1] += bf16hi(dO[mi]) * bf16hi(o[mi]);
        }
        stsm4t(faddr(st + O_O, ch), dO);
        stsm4t(faddr(st + O_SG, ch), dg);
        stsm4t(faddr(st + O_DY, ch), ga);
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) { dsum[q] += __shfl_xor_sync(~0u, dsum[q], 4); dsum[q] += __shfl_xor_sync(~0u, dsum[q], 8); dsum[q] += __shfl_xor_sync(~0u, dsum[q], 16); }
      if (lane < 4) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int r = r0 + 8 * (q >> 1) + 2 * lane + (q & 1), an = r / N, n = r - an * N;
          Dd[((size_t)an * 4 + qw) * N + n] = dsum[q];
        }
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + w, 128);
      if (leader) {
        for (int kb = 0; kb < 2; ++kb) {
          tma_store_2d(&mda, st + O_A2 + kb * KB, 64 * kb, r0);
          tma_store_2d(&mdmod, st + O_S2 + kb * KB, 256 + 64 * kb, r0);    // dsc2 -> block 2
          tma_store_2d(&mdmod, st + O_X + kb * KB, 384 + 64 * kb, r0);     // dbi2 = dx2 -> block 3
          tma_store_2d(&mdmod, st + O_SO + kb * KB, 512 + 64 * kb, r0);    // dos -> block 4
          tma_store_2d(&mdu, st + O_U + kb * KB, 64 * kb, r0);
          tma_store_2d(&mgated, st + O_DY + kb * KB, 64 * kb, r0);
          tma_store_2d(&mdo, st + O_O + kb * KB, 64 * kb, r0);
          tma_store_2d(&mdp, st + O_SG + kb * KB, 384 + 64 * kb, r0);      // dg -> dP block 3
        }
        tma_store_commit();
        tma_store_wait_read0();                                            // (a short wait: the stores only have to read the stage)
        mbar_arrive(&B.infree[w][i % NSW]);                                // this tile's stage is free for tile i + NSW
      }
    }
    if (leader) tma_store_wait0();
    // d bsc2 (dmod block 2), d bos (block 4): the 16 threads sharing a chunk column -> shared sums -> one red.add per channel
    float* red = reinterpret_cast<float*>(sm + O_RED);                      // [8 warps][256]
    const int cw = warp - 4;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      float v2 = bs2[e], vo = bso[e];
      v2 += __shfl_xor_sync(~0u, v2, 8); v2 += __shfl_xor_sync(~0u, v2, 16);
      vo += __shfl_xor_sync(~0u, vo, 8); vo += __shfl_xor_sync(~0u, vo, 16);
      const int c = e < 8 ? 8 * ak + e : 64 + 8 * ak + (e - 8);
      if ((lane >> 3) == 0) { red[cw * 256 + c] = v2; red[cw * 256 + 128 + c] = vo; }
    }
    named_bar_sync(3, 256);
    {
      const int c = tid - 128;                                             // 0 .. 255
      float v = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) v += red[k * 256 + c];
      atomicAdd(DBIAS + (c < 128 ? 256 + c : 512 + (c - 128)), v);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
