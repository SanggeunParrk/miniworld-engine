// tr_bwd_gate.cu — first part of the atom DiT block's ConditionedTransition backward on sm_100a (the bf16 module's autograd rounding
// points; forward y = rn(st t), t = rn(h Ws^T), h = rn(rn(silu(a)) b), a | b = rn(x2 Wa^T) | rn(x2 Wb^T)):
//   dt = rn(dy st);  dst = rn(dy t);  dts = rn(rn(dst rn(1 - st)) st)                       (st = rn(sigmoid(ts)) from cond_fwd's mod)
//   a, b = recomputed from x2;  dh = rn(dt Ws);  sa = sigmoid(a);  sl = rn(a sa);  h = rn(sl b)
//   ds = rn(dh b);  db = rn(dh sl);  da = rn(ds sa (1 + a (1 - sa)))
//   -> DT [M, 128], HH [M, 256] (the dWs = DT^T HH operands), DAB = [da | db] [M, 512] (dWu = DAB^T x2, and tr_bwd_dx's input),
//      dts into dmod's block 5, and d bts = sum_rows dts (per-thread register sums, one red.add per CTA and channel at the end).
// Structure (post_fwd.cu's frame): transposed 16-row tiles; Wa | Wb (512 x 128) and Ws^T (256 x 128) in TMEM as bf16 A operands (384
// columns); the two compute warpgroups take alternate tiles, each with its own TMA producer warp, MMA warp, NSW-stage ring and 48
// accumulator columns, reused by the two 128-hidden chunks: a^T | b^T | dh^T (M = 128 hidden, N = 16 rows).
// Threads: dt / dts eight threads per row (16 channels, 16-B shared loads, bf16x2 math); the SwiGLU backward in the mma-fragment layout
// (tcgen05.ld 16x256b), h / da / db stored transposed by stmatrix.trans into row-major tiles; the leader thread issues the TMA stores.
// Stage (40 KB): dy -> dt | t -> dts | st | x2 | h [16][256] | dab [16][512].
// TMEM: Wa (hidden 0-127, 128-255) 0 / 64, Wb 128 / 192, Ws^T 256 / 320; warpgroup w's accumulators at 384 + 64 w: a^T 0, b^T 16, dh^T 32.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DIAG
#define DIAG 0                             // 1: no compute math (handshakes only), 2: no MMAs
#endif
#ifndef NSW
#define NSW 2                              // input stages per warpgroup
#endif
constexpr int R = 16;
constexpr int KB = R * 128;                                                // [16 rows][64] bf16, SW128 (2 KB)
constexpr int T_ = 2 * KB;                                                 // [16][128] row-major tile (4 KB)
constexpr int O_DY = 0, O_T = T_, O_ST = 2 * T_, O_X2 = 3 * T_, O_H = 4 * T_, O_DAB = 6 * T_, STG = 10 * T_;
constexpr int O_BAR = (2 * NSW * STG > 6 * 32768 ? 2 * NSW * STG : 6 * 32768), O_RED = O_BAR + 512;
constexpr int SMEM = O_RED + 8 * 128 * 4;
static_assert(SMEM <= 232448, "shared memory");
constexpr uint32_t T_WU = 0, T_WS = 256, T_ACC = 384;
constexpr uint32_t I_16 = idesc_bf16(128, 16);

struct Bars {
  uint64_t wfull, wfree, infull[2][NSW], infree[2][NSW], afull[2], abfull[2], abfree[2];
  uint32_t tmem;
};
DEVI uint32_t bmul2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t bsub2(uint32_t a, uint32_t b) { uint32_t r; asm("sub.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
// torch's bf16 sigmoid backward, every step rounded: rn(rn(g rn(1 - y)) y)
DEVI uint32_t sig_bwd2(uint32_t g, uint32_t y) { return bmul2(bmul2(g, bsub2(0x3F803F80u, y)), y); }
DEVI void tmem_ld16x256b2(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
DEVI void stsm4t(uint32_t a, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
atom_tr_bwd_gate_sm100(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mt, const __grid_constant__ CUtensorMap mmod,
                       const __grid_constant__ CUtensorMap mx2, const __grid_constant__ CUtensorMap mwu, const __grid_constant__ CUtensorMap mwst,
                       const __grid_constant__ CUtensorMap mdt, const __grid_constant__ CUtensorMap mdmod, const __grid_constant__ CUtensorMap mh,
                       const __grid_constant__ CUtensorMap mdab, float* __restrict__ DBTS, int ntile) {
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
      mbar_init(&B.afull[w], 1); mbar_init(&B.abfull[w], 1); mbar_init(&B.abfree[w], 4);
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
        // Wa | Wb (units 0-3: rows 128 u ..) and Ws^T (units 4-5: hidden 128 (u - 4) ..) as K-major SW64 [128][32] atoms
        mbar_expect_tx(&B.wfull, 6 * 32768);
        for (int i = 0; i < 24; ++i) {
          const int u = i >> 2, ka = i & 3;
          tma_load_2d(su + u * 32768 + ka * 8192, u < 4 ? &mwu : &mwst, &B.wfull, 32 * ka, u < 4 ? 128 * u : 128 * (u - 4));
        }
      }
      mbar_wait(&B.wfree, 0);
      pdl_wait();                                                          // dy comes from the previous kernels
      for (int i = 0; 2 * i + w < ntT; ++i) {
        const int s = i % NSW, r0 = row0(2 * i + w);
        if (i >= NSW) mbar_wait(&B.infree[w][s], ((i / NSW) - 1) & 1);
        const uint32_t st = stage(w, i);
        uint64_t* bar = &B.infull[w][s];
        mbar_expect_tx(bar, 8 * KB);
        for (int kb = 0; kb < 2; ++kb) {
          tma_load_2d(st + O_DY + kb * KB, &mdy, bar, 64 * kb, r0);
          tma_load_2d(st + O_T + kb * KB, &mt, bar, 64 * kb, r0);
          tma_load_2d(st + O_ST + kb * KB, &mmod, bar, 640 + 64 * kb, r0);   // st: the forward mod's block 5
          tma_load_2d(st + O_X2 + kb * KB, &mx2, bar, 64 * kb, r0);
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
        for (int u = 0; u < 6; ++u)
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
      mbar_wait(&B.afull[w], i & 1);                                       // dt written over dy; x2 landed
      for (int hc = 0; hc < 2; ++hc) {
        const int k = 2 * i + hc;                                          // accumulator use
        if (k > 0) mbar_wait(&B.abfree[w], (k - 1) & 1);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < (DIAG == 2 ? 0 : 8); ++ks) {
            umma_ts(ta, tmem + T_WU + 64 * hc + ks * 8, bdesc(st + O_X2, ks), I_16, ks > 0 ? 1u : 0u);
            umma_ts(ta + 16, tmem + T_WU + 64 * (2 + hc) + ks * 8, bdesc(st + O_X2, ks), I_16, ks > 0 ? 1u : 0u);
            umma_ts(ta + 32, tmem + T_WS + 64 * hc + ks * 8, bdesc(st + O_DY, ks), I_16, ks > 0 ? 1u : 0u);
          }
          tc_commit(&B.abfull[w]);
        }
        __syncwarp();
      }
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
    float bsum[16];                                                        // d bts over this thread's rows: channels 8 ak + e, 64 + 8 ak + e
#pragma unroll
    for (int e = 0; e < 16; ++e) bsum[e] = 0.f;
    for (int i = 0; 2 * i + w < ntT; ++i) {
      const int s = i % NSW, r0 = row0(2 * i + w);
      const uint32_t st = stage(w, i);
      mbar_wait(&B.infull[w][s], (i / NSW) & 1);
      // ---- A: dt = rn(dy st) over dy, dts = rn(rn(dy t) (1 - st) st) over t
#pragma unroll
      for (int kb = 0; kb < (DIAG == 1 ? 0 : 2); ++kb) {
        const uint4 dy = lds128(st + O_DY + kb * KB + aoff), t = lds128(st + O_T + kb * KB + aoff), sg = lds128(st + O_ST + kb * KB + aoff);
        const uint32_t d4[4] = {dy.x, dy.y, dy.z, dy.w}, t4[4] = {t.x, t.y, t.z, t.w}, s4[4] = {sg.x, sg.y, sg.z, sg.w};
        uint32_t dt[4], ds[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          dt[e] = bmul2(d4[e], s4[e]);
          const uint32_t dst = bmul2(d4[e], t4[e]);
          ds[e] = sig_bwd2(dst, s4[e]);
          bsum[8 * kb + 2 * e] += bf16lo(ds[e]);
          bsum[8 * kb + 2 * e + 1] += bf16hi(ds[e]);
        }
        sts128(st + O_DY + kb * KB + aoff, make_uint4(dt[0], dt[1], dt[2], dt[3]));
        sts128(st + O_T + kb * KB + aoff, make_uint4(ds[0], ds[1], ds[2], ds[3]));
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (leader) mbar_arrive(&B.afull[w]);
      // ---- B: the SwiGLU backward for hidden 128 hc + lb + 16 L + .., rows in pairs
#pragma unroll 1
      for (int hc = 0; hc < 2; ++hc) {
        mbar_wait(&B.abfull[w], (2 * i + hc) & 1);
        tc_fence_after();
        uint32_t va[2][8], vb[2][8], vd[2][8];
#pragma unroll
        for (int L = 0; L < 2; ++L) {
          const uint32_t tl = ta + ((lb + 16 * L) << 16);
          tmem_ld16x256b2(tl, va[L]); tmem_ld16x256b2(tl + 16, vb[L]); tmem_ld16x256b2(tl + 32, vd[L]);
        }
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.abfree[w]);
#pragma unroll
        for (int L = 0; L < (DIAG == 1 ? 0 : 2); ++L) {
          uint32_t rh[4], ra[4], rb[4];
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) {                                 // rows 2 (lane % 4) + {0, 1} (+ 8): a bf16 pair each
            const int k = 4 * (mi & 1) + 2 * (mi >> 1);
            const uint32_t A2 = pack_bf16(__uint_as_float(va[L][k]), __uint_as_float(va[L][k + 1]));
            const uint32_t B2 = pack_bf16(__uint_as_float(vb[L][k]), __uint_as_float(vb[L][k + 1]));
            const uint32_t D2 = pack_bf16(__uint_as_float(vd[L][k]), __uint_as_float(vd[L][k + 1]));
            const f2 a = mk2u(A2 << 16, A2 & 0xFFFF0000u);
            const f2 e = mul2(a, mk2(-1.4426950408889634f, -1.4426950408889634f));
            const f2 d = add2(mk2(ex2f(lo2(e)), ex2f(hi2(e))), mk2(1.f, 1.f));
            const f2 sa = mk2(rcpf(lo2(d)), rcpf(hi2(d)));                  // sigmoid(a)
            const f2 sl = mul2(a, sa);
            const uint32_t SL2 = pack_bf16(lo2(sl), hi2(sl));
            const uint32_t DS2 = bmul2(D2, B2);                            // ds = rn(dh b)
            rh[mi] = bmul2(SL2, B2);                                       // h = rn(sl b)
            rb[mi] = bmul2(D2, SL2);                                       // db = rn(dh sl)
            const f2 ds = mk2u(DS2 << 16, DS2 & 0xFFFF0000u);
            const f2 w = fma2(a, add2(neg2(sa), mk2(1.f, 1.f)), mk2(1.f, 1.f));   // 1 + a (1 - sa)
            const f2 da = mul2(mul2(ds, sa), w);                           // da = rn(ds sa (1 + a (1 - sa)))
            ra[mi] = pack_bf16(lo2(da), hi2(da));
          }
          const uint32_t hid = 128 * hc + lb + 16 * L + 8 * fh;
          stsm4t(faddr(st + O_H, hid), rh);
          stsm4t(faddr(st + O_DAB, hid), ra);
          stsm4t(faddr(st + O_DAB, 256 + hid), rb);
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (leader) {
        for (int kb = 0; kb < 2; ++kb) {
          tma_store_2d(&mdt, st + O_DY + kb * KB, 64 * kb, r0);
          tma_store_2d(&mdmod, st + O_T + kb * KB, 640 + 64 * kb, r0);      // dts -> dmod block 5
        }
        for (int kb = 0; kb < 4; ++kb) tma_store_2d(&mh, st + O_H + kb * KB, 64 * kb, r0);
        for (int kb = 0; kb < 8; ++kb) tma_store_2d(&mdab, st + O_DAB + kb * KB, 64 * kb, r0);
        tma_store_commit();
        tma_store_wait_read0();                                            // (a short wait: the stores only have to read the stage)
        mbar_arrive(&B.infree[w][i % NSW]);                                // this tile's stage is free for tile i + NSW
      }
    }
    if (leader) tma_store_wait0();
    // d bts: the 16 threads of each chunk column (8 per row x 16 rows share ak) -> shared sums -> one red.add per channel
    float* red = reinterpret_cast<float*>(sm + O_RED);                      // [8 warps][128 channels]
    const int cw = warp - 4;
    for (int c = lane; c < 128; c += 32) red[cw * 128 + c] = 0.f;
    __syncwarp();
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      float v = bsum[e];
      v += __shfl_xor_sync(~0u, v, 8); v += __shfl_xor_sync(~0u, v, 16);   // the warp's four rows
      if ((lane >> 3) == 0) red[cw * 128 + (e < 8 ? 8 * ak + e : 64 + 8 * ak + (e - 8))] = v;
    }
    named_bar_sync(3, 256);
    if (ct < 128 && w == 0) {
      float v = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) v += red[k * 128 + ct];
      atomicAdd(DBTS + ct, v);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
