// post_fwd.cu — the atom DiT block's attention output + ConditionedTransition on sm_100a (per row of M = A N; the bf16 module's rounding
// points; so / s2 / bi2 / st from cond_fwd's mod and sg = rn(sigmoid(g)) from pre_fwd, the gates already as rn(sigmoid)):
//   gated = rn(sg o);  u = rn(gated Wo^T);  a2 = rn(a + rn(so u))
//   x2 = rn(rn(s2 rn(LayerNorm(a2))) + bi2);  h = rn(rn(silu(rn(x2 Wa^T))) rn(x2 Wb^T));  t = rn(h Ws^T)
//   a3 = rn(a2 + rn(st t))                          -> a3 [M, 128] bf16; optionally u, a2, x2, t saved for the backward.
// Structure: transposed 16-row tiles (the MMA N is the row count): Wa | Wb (512 x 128) and Ws (128 x 256) live in TMEM as bf16 A operands
// (384 columns), Wo in shared memory (A of u^T = Wo gated^T). The two compute warpgroups take alternate tiles, each with its own TMA
// producer warp, MMA-issuing warp, 3-stage input ring and 64 accumulator columns (u^T; then a^T | b^T of both 128-hidden chunks; then
// t^T), so one warpgroup's math runs while the other's MMAs do.
// Threads: row-wise stages (gated; a2, LayerNorm, x2) eight threads per row, 16 channels each (16-B shared loads, bf16x2 math);
// accumulator stages in the mma-fragment layout (tcgen05.ld 16x256b: lane = channel, column pairs = row pairs), moved between that layout
// and the row-major 128-B-swizzled tiles by ldmatrix / stmatrix .trans. Every activation feeding an MMA is written into a row-major tile
// (the K-major B operand), in place over an input it replaces; the leader thread issues the TMA stores.
// Stage (28 KB): a -> a2 | o -> u | sg -> gated -> x2 | mod: so, s2 (-> h), bi2 (-> a3), st; the t save goes over h's first half.
// TMEM: Wa (hidden 0-127, 128-255) 0 / 64, Wb 128 / 192, Ws 256 (128 columns); warpgroup w's accumulators at 384 + 64 w:
// u^T / a0^T / t^T 0, b0^T 16, a1^T 32, b1^T 48.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DIAG
#define DIAG 0                             // 1: no compute math (handshakes only), 2: no MMAs
#endif
#ifndef NSW
#define NSW 3                              // input stages per warpgroup
#endif
constexpr int R = 16;
constexpr int KB = R * 128;                                                // [16 rows][64] bf16, SW128 (2 KB)
constexpr int T_ = 2 * KB;                                                 // [16][128] row-major tile (4 KB)
constexpr int O_A = 0, O_O = T_, O_G = 2 * T_, O_M = 3 * T_, STG = 7 * T_;  // mod blocks: os 0-1, sc2 2-3, bi2 4-5, ts 6-7
constexpr int O_WO = 2 * NSW * STG, O_BAR = O_WO + 32768;
constexpr int SMEM = O_BAR + 512;
static_assert(SMEM <= 232448, "shared memory");
static_assert(O_BAR >= 6 * 32768, "the six weight units stage through the rings and the Wo slot");
constexpr uint32_t T_WU = 0, T_WD = 256, T_ACC = 384;
constexpr uint32_t I_16 = idesc_bf16(128, 16);

struct Bars {
  uint64_t wfull, wfree, wofull, infull[2][NSW], infree[2][NSW], gfull[2], ufull[2], yfull[2], abfull[2], hfull[2], tfull[2];
  uint32_t tmem;
};
DEVI uint32_t bmul2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t badd2(uint32_t a, uint32_t b) { uint32_t r; asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI float sigm(float x) { return __fdividef(1.f, 1.f + __expf(-x)); }
DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// 16 lanes x 16 columns, the mma-accumulator fragment: r[4m + 0, 1] = lane t / 4, columns 8 m + 2 (t % 4) + {0, 1}; r[4m + 2, 3] = lane 8 + t / 4
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

extern "C" __global__ void __launch_bounds__(384, 1)
atom_post_fwd_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mg,
                    const __grid_constant__ CUtensorMap mmod, const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwu,
                    const __grid_constant__ CUtensorMap mwd, const __grid_constant__ CUtensorMap mout, const __grid_constant__ CUtensorMap mu,
                    const __grid_constant__ CUtensorMap ma2, const __grid_constant__ CUtensorMap mx2, const __grid_constant__ CUtensorMap mt,
                    int ntile, float eps, int save) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto row0 = [&](int T) { return R * ((int)blockIdx.x + T * (int)gridDim.x); };
  auto stage = [&](int w, int i) { return su + (uint32_t)((w * NSW + i % NSW) * STG); };

  if (tid == 0) {
    mbar_init(&B.wfull, 1); mbar_init(&B.wfree, 1); mbar_init(&B.wofull, 1);
    for (int w = 0; w < 2; ++w) {
      for (int s = 0; s < NSW; ++s) { mbar_init(&B.infull[w][s], 1); mbar_init(&B.infree[w][s], 1); }
      mbar_init(&B.gfull[w], 1); mbar_init(&B.ufull[w], 1); mbar_init(&B.yfull[w], 1); mbar_init(&B.abfull[w], 1); mbar_init(&B.hfull[w], 1);
      mbar_init(&B.tfull[w], 1);
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
        // Wa | Wb (units 0-3: rows 128 u ..) and Ws (units 4-5: k 128 (u - 4) ..) as K-major SW64 [128][32] atoms into the idle rings
        mbar_expect_tx(&B.wfull, 6 * 32768);
        for (int i = 0; i < 24; ++i) {
          const int u = i >> 2, ka = i & 3;
          tma_load_2d(su + u * 32768 + ka * 8192, u < 4 ? &mwu : &mwd, &B.wfull, u < 4 ? 32 * ka : 128 * (u - 4) + 32 * ka, u < 4 ? 128 * u : 0);
        }
      }
      mbar_wait(&B.wfree, 0);
      if (w == 0) {
        mbar_expect_tx(&B.wofull, 32768);
        for (int kb = 0; kb < 2; ++kb) tma_load_2d(su + O_WO + kb * 16384, &mwo, &B.wofull, 64 * kb, 0);
      }
      pdl_wait();                                                          // a / o / g / mod come from the previous kernels
      for (int i = 0; 2 * i + w < ntT; ++i) {
        const int s = i % NSW, r0 = row0(2 * i + w);
        if (i >= NSW) mbar_wait(&B.infree[w][s], ((i / NSW) - 1) & 1);
        const uint32_t st = stage(w, i);
        uint64_t* bar = &B.infull[w][s];
        mbar_expect_tx(bar, 14 * KB);
        for (int kb = 0; kb < 2; ++kb) {
          tma_load_2d(st + O_A + kb * KB, &ma, bar, 64 * kb, r0);
          tma_load_2d(st + O_O + kb * KB, &mo, bar, 64 * kb, r0);
          tma_load_2d(st + O_G + kb * KB, &mg, bar, 64 * kb, r0);
        }
        for (int kb = 0; kb < 8; ++kb) tma_load_2d(st + O_M + kb * KB, &mmod, bar, 256 + 64 * kb, r0);   // os | sc2 | bi2 | ts
      }
    }
  } else if (warp == 1 || warp == 3) {
    // ------------------------------------------------------------------------------------------------ MMA issuers (warp 1: warpgroup 0, 3: 1)
    const int w = warp >> 1;
    if (w == 0) {
      mbar_wait(&B.wfull, 0);
      tc_fence_after();
      if (elect_one()) {
        for (int u = 0; u < 6; ++u) {
          const uint32_t col = u < 4 ? T_WU + 64 * u : T_WD + 64 * (u - 4);
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) tmem_cp_128x256b(tmem + col + ks * 8, desc_sw64(su + u * 32768 + (ks >> 1) * 8192) + (uint64_t)((ks & 1) * 2));
        }
        tc_commit(&B.wfree);
      }
      __syncwarp();
    }
    mbar_wait(&B.wfree, 0);
    mbar_wait(&B.wofull, 0);
    const uint32_t ta = tmem + T_ACC + 64 * w;
    auto bdesc = [&](uint32_t base, int ks) { return desc_k128(base + (ks >> 2) * KB) + (uint64_t)((ks & 3) * 2); };
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const uint32_t st = stage(w, i);
      mbar_wait(&B.gfull[w], i & 1);                                       // gated written over g
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < (DIAG == 2 ? 0 : 8); ++ks)
          umma_ss(ta, desc_k128(su + O_WO + (ks >> 2) * 16384) + (uint64_t)((ks & 3) * 2), bdesc(st + O_G, ks), I_16, ks > 0 ? 1u : 0u);
        tc_commit(&B.ufull[w]);
      }
      __syncwarp();
      mbar_wait(&B.yfull[w], i & 1);                                       // x2 written over gated
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < (DIAG == 2 ? 0 : 8); ++ks)
#pragma unroll
          for (int q = 0; q < 4; ++q)                                      // a0 -> 0 (Wa unit 0), b0 -> 16 (unit 2), a1 -> 32 (unit 1), b1 -> 48 (unit 3)
            umma_ts(ta + 16 * q, tmem + T_WU + 64 * ((q & 1) * 2 + (q >> 1)) + ks * 8, bdesc(st + O_G, ks), I_16, ks > 0 ? 1u : 0u);
        tc_commit(&B.abfull[w]);
      }
      __syncwarp();
      mbar_wait(&B.hfull[w], i & 1);                                       // h written over os | sc2
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < (DIAG == 2 ? 0 : 16); ++ks) umma_ts(ta, tmem + T_WD + ks * 8, bdesc(st + O_M, ks), I_16, ks > 0 ? 1u : 0u);
        tc_commit(&B.tfull[w]);
      }
      __syncwarp();
    }
  } else {
    // ------------------------------------------------------------------------------------------------ compute warpgroups
    const int w = (warp - 4) >> 2, qw = warp & 3, ct = tid & 127;
    const uint32_t lb = (uint32_t)qw * 32;
    const bool leader = qw == 0 && lane == 0;
    const uint32_t ta = tmem + T_ACC + 64 * w;
    // row-wise stages: row ar, 16-B chunk ak of both 64-channel blocks
    const int ar = ct >> 3, ak = ct & 7;
    const uint32_t aoff = sw128((uint32_t)ar, (uint32_t)ak);
    // fragment stages: lane i addresses row j = 8 (mi & 1) + (i & 7) of 8 x 8 block mi = i / 8 (channel group 8 (mi >> 1) of a 16-lane half)
    const uint32_t fj = (uint32_t)(8 * ((lane >> 3) & 1) + (lane & 7)), fh = (uint32_t)(lane >> 4);
    auto faddr = [&](uint32_t base, uint32_t ch) {                          // row fj, channels ch .. ch + 7 of a row-major tile
      return base + (ch >> 6) * KB + fj * 128u + ((((ch & 63u) >> 3) ^ (fj & 7u)) << 4);
    };
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const int s = i % NSW, r0 = row0(2 * i + w);
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][s], (i / NSW) & 1);
      // ---- P1: gated = rn(sg o) over sg
#pragma unroll
      for (int kb = 0; kb < (DIAG == 1 ? 0 : 2); ++kb) {
        const uint4 g = lds128(st + O_G + kb * KB + aoff), o = lds128(st + O_O + kb * KB + aoff);
        sts128(st + O_G + kb * KB + aoff, make_uint4(bmul2(g.x, o.x), bmul2(g.y, o.y), bmul2(g.z, o.z), bmul2(g.w, o.w)));
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + w, 128);
      if (leader) mbar_arrive(&B.gfull[w]);
      // ---- P2: u = rn(u^T) -> row-major over o; then row-wise a2, LayerNorm, x2
      mbar_wait(&B.ufull[w], i & 1);
      tc_fence_after();
#pragma unroll
      for (int L = 0; L < (DIAG == 1 ? 0 : 2); ++L) {
        uint32_t v[8], r[4];
        tmem_ld16x256b2(ta + ((lb + 16 * L) << 16), v);
        tmem_wait_ld();
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) r[mi] = pack_bf16(__uint_as_float(v[4 * (mi & 1) + 2 * (mi >> 1)]), __uint_as_float(v[4 * (mi & 1) + 2 * (mi >> 1) + 1]));
        stsm4t(faddr(st + O_O, lb + 16 * L + 8 * fh), r);
      }
      tc_fence_before();
      named_bar_sync(1 + w, 128);
      if (DIAG != 1) {
        uint32_t a2w[8];
        float x[16];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 a = lds128(st + O_A + kb * KB + aoff), u = lds128(st + O_O + kb * KB + aoff), os = lds128(st + O_M + kb * KB + aoff);
          a2w[4 * kb + 0] = badd2(a.x, bmul2(os.x, u.x));
          a2w[4 * kb + 1] = badd2(a.y, bmul2(os.y, u.y));
          a2w[4 * kb + 2] = badd2(a.z, bmul2(os.z, u.z));
          a2w[4 * kb + 3] = badd2(a.w, bmul2(os.w, u.w));
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) { x[2 * e] = bf16lo(a2w[e]); x[2 * e + 1] = bf16hi(a2w[e]); }
        float sum = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) sum += x[e];
        sum += __shfl_xor_sync(~0u, sum, 1); sum += __shfl_xor_sync(~0u, sum, 2); sum += __shfl_xor_sync(~0u, sum, 4);
        const float mean = sum * (1.f / 128.f);
        float var = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) { const float d = x[e] - mean; var += d * d; }
        var += __shfl_xor_sync(~0u, var, 1); var += __shfl_xor_sync(~0u, var, 2); var += __shfl_xor_sync(~0u, var, 4);
        const float rstd = rsqrtf(var * (1.f / 128.f) + eps);
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 sc = lds128(st + O_M + (2 + kb) * KB + aoff), bi = lds128(st + O_M + (4 + kb) * KB + aoff);
          const uint32_t s4[4] = {sc.x, sc.y, sc.z, sc.w}, b4[4] = {bi.x, bi.y, bi.z, bi.w};
          uint32_t o[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = 4 * kb + e;
            const uint32_t xn = pack_bf16((x[2 * k] - mean) * rstd, (x[2 * k + 1] - mean) * rstd);
            o[e] = badd2(bmul2(s4[e], xn), b4[e]);
          }
          sts128(st + O_G + kb * KB + aoff, make_uint4(o[0], o[1], o[2], o[3]));
          sts128(st + O_A + kb * KB + aoff, make_uint4(a2w[4 * kb], a2w[4 * kb + 1], a2w[4 * kb + 2], a2w[4 * kb + 3]));
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (leader) mbar_arrive(&B.yfull[w]);
      // ---- P3: h = rn(rn(silu(rn(a))) rn(b)) for hidden 128 hc + lb + 16 L + .., row-major over os | sc2
      mbar_wait(&B.abfull[w], i & 1);
      tc_fence_after();
#pragma unroll
      for (int hc = 0; hc < (DIAG == 1 ? 0 : 2); ++hc)
#pragma unroll
        for (int L = 0; L < 2; ++L) {
          uint32_t va[8], vb[8], r[4];
          tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 32 * hc, va);
          tmem_ld16x256b2(ta + ((lb + 16 * L) << 16) + 32 * hc + 16, vb);
          tmem_wait_ld();
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) {
            const int k = 4 * (mi & 1) + 2 * (mi >> 1);
            r[mi] = bmul2(silu_rn2(pack_bf16(__uint_as_float(va[k]), __uint_as_float(va[k + 1]))),
                          pack_bf16(__uint_as_float(vb[k]), __uint_as_float(vb[k + 1])));
          }
          stsm4t(faddr(st + O_M, 128 * hc + lb + 16 * L + 8 * fh), r);
        }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + w, 128);
      if (leader) mbar_arrive(&B.hfull[w]);
      // ---- P4: a3 = rn(a2 + rn(rn(sigmoid(ts)) rn(t))) in the fragment layout (ts, a2 by ldmatrix.trans) -> row-major over bi2
      mbar_wait(&B.tfull[w], i & 1);
      tc_fence_after();
#pragma unroll
      for (int L = 0; L < (DIAG == 1 ? 0 : 2); ++L) {
        const uint32_t ch = lb + 16 * L + 8 * fh;
        uint32_t v[8], tv[4], ts[4], a2[4], r[4];
        tmem_ld16x256b2(ta + ((lb + 16 * L) << 16), v);
        ldsm4t(faddr(st + O_M + 6 * KB, ch), ts);
        ldsm4t(faddr(st + O_A, ch), a2);
        tmem_wait_ld();
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
          const int k = 4 * (mi & 1) + 2 * (mi >> 1);
          tv[mi] = pack_bf16(__uint_as_float(v[k]), __uint_as_float(v[k + 1]));
          r[mi] = badd2(a2[mi], bmul2(ts[mi], tv[mi]));
        }
        stsm4t(faddr(st + O_M + 4 * KB, ch), r);
        if (save) stsm4t(faddr(st + O_M, ch), tv);                          // t over h's first half (M3 has read it)
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1 + w, 128);
      if (leader) {
        for (int kb = 0; kb < 2; ++kb) {
          tma_store_2d(&mout, st + O_M + (4 + kb) * KB, 64 * kb, r0);
          if (save) {
            tma_store_2d(&mu, st + O_O + kb * KB, 64 * kb, r0);
            tma_store_2d(&ma2, st + O_A + kb * KB, 64 * kb, r0);
            tma_store_2d(&mx2, st + O_G + kb * KB, 64 * kb, r0);
            tma_store_2d(&mt, st + O_M + kb * KB, 64 * kb, r0);
          }
        }
        tma_store_commit();
        tma_store_wait_read0();                                            // (a short wait: the stores only have to read the stage)
        mbar_arrive(&B.infree[w][i % NSW]);                                // this tile's stage is free for tile i + NSW
      }
    }
    if (leader) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
