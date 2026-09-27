// ffn_fwd.cu — the SWA atom block's second forward stage on sm_100a (the B200 port of team-gm swa_cuda/swa_ffn_fwd.cu; the math and
// rounding points of the Triton _oproj_ffn_fwd):
//   gated = rn(sigmoid(g) o);  att = rn(gated Wo^T);  q1 = rn(q + rn(gate_a) att);
//   y = rn(RMS(q1) (1 + scale_f) + shift_f);  a|b = y Wu^T;  h = rn(silu(a) b);  ffn = rn(h Wd^T);  out = rn(q1 + rn(gate_f) ffn)
// optionally saving q1, att, y, ffn for the backward.
// Structure: persistent CTAs, tiles of SP augments x AT atoms (as qkvg_fwd.cu); the activations feeding the GEMMs (gated, y, h) are built by
// the row threads and written to TMEM as bf16 A operands of TS MMAs (the tcgen05 counterpart of the sm_90a kernel's register-A wgmma);
// weights stream from L2 through a ring of 16-KB k-block slots (4 or 6, whatever fits next to the tile) in the order Wo, [Wa_j ; Wb_j], Wd_j
// (j = 4 chunks of 64 hidden units). Two warpgroups split the channels (and each chunk's hidden units); q1's row RMS is the only exchange
// (one float per row through smem). The a|b accumulator is double-buffered, so chunk j + 1's up-projection runs while the threads form
// chunk j's h. g / o and q / mod load on separate barriers (the next tile's gated can start before this tile's output is out). q1
// overwrites the q tile in place and the output leaves through a TMA store.
// TMEM: A (gated -> y, bf16) at 0, a|b[2] accumulators at 64 / 192, h[2] (bf16) at 320, att -> ffn accumulator at 384.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int C = 128, NHID = 256, HC = 64, NCH = NHID / HC, RSMAX = 6;
constexpr int KB = 128 * 128;                                              // [128 rows][64 bf16] k-block (16 KB) = one ring slot
constexpr int O_Q = 0, O_G = O_Q + 2 * KB, O_O = O_G + 2 * KB, O_R = O_O + 2 * KB;   // then the ring (RS slots), mod, ss, bars (runtime)
constexpr uint32_t T_A = 0, T_AB = 64, T_H = 320, T_F = 384;
constexpr uint32_t I_N128 = idesc_bf16(128, 128);
constexpr int KPT = 2 + 3 * NCH;                                           // ring k-blocks per tile: Wo (2), per chunk Wab_j (2) + Wd_j (1)

struct Bars {
  uint64_t gofull, gofree, qfull, qfree, wfull[RSMAX], wempty[RSMAX], afull, attfull, yfull, abfull[2], abfree[2], hfull[2], hfree[2], ffull, ffree, yfree;
  uint32_t tmem;
};

DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEVI float sigm(float x) { return rcpf(1.f + ex2f(-1.4426950408889634f * x)); }   // as the sm_90a kernel: rcp.approx(1 + 2^(-x log2 e))

extern "C" __global__ void __launch_bounds__(384, 1)
swa_ffn_fwd_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mo,
                  const __grid_constant__ CUtensorMap mmod, const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwab,
                  const __grid_constant__ CUtensorMap mwd, const __grid_constant__ CUtensorMap mout,
                  int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, float eps, int save, int RS,
                  __nv_bfloat16* __restrict__ Q1s, __nv_bfloat16* __restrict__ Atts, __nv_bfloat16* __restrict__ Ys, __nv_bfloat16* __restrict__ Ffs) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int O_MOD = O_R + RS * KB, O_SS = O_MOD + 16 * AT * 128, O_BAR = (O_SS + 2 * 128 * 4 + 7) / 8 * 8;
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto coords = [&](int T, int& b, int& a0, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    const int ab = t % nab, r = t / nab, ag = r % nag;
    b = r / nag; a0 = ag * SP; s0 = ab * AT;
  };

  if (tid == 0) {
    mbar_init(&B.gofull, 1); mbar_init(&B.gofree, 8); mbar_init(&B.qfull, 1); mbar_init(&B.qfree, 1);
    for (int i = 0; i < RS; ++i) { mbar_init(&B.wfull[i], 1); mbar_init(&B.wempty[i], 1); }
    mbar_init(&B.afull, 8); mbar_init(&B.attfull, 1); mbar_init(&B.yfull, 8);
    for (int i = 0; i < 2; ++i) { mbar_init(&B.abfull[i], 1); mbar_init(&B.abfree[i], 8); mbar_init(&B.hfull[i], 8); mbar_init(&B.hfree[i], 1); }
    mbar_init(&B.ffull, 1); mbar_init(&B.ffree, 8); mbar_init(&B.yfree, 1);
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: activations
    if (lane == 0) {
      const uint32_t tb = (uint32_t)(SP * AT * 128);                        // bytes of one k-block box
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        if (T >= 1) mbar_wait(&B.gofree, (T - 1) & 1);                     // g / o of tile T - 1 consumed (its gated is in TMEM)
        mbar_expect_tx(&B.gofull, 4 * tb);
        for (int kb = 0; kb < 2; ++kb) {
          tma_load_4d(su + O_G + kb * KB, &mg, &B.gofull, kb * 64, s0, b, a0);
          tma_load_4d(su + O_O + kb * KB, &mo, &B.gofull, kb * 64, s0, b, a0);
        }
        if (T >= 1) mbar_wait(&B.qfree, (T - 1) & 1);                      // tile T - 1's output has left the q tile
        mbar_expect_tx(&B.qfull, 2 * tb + (uint32_t)(16 * AT * 128));
        for (int kb = 0; kb < 2; ++kb) tma_load_4d(su + O_Q + kb * KB, &mq, &B.qfull, kb * 64, s0, b, a0);
        tma_load_3d(su + O_MOD, &mmod, &B.qfull, 0, b * S + s0, 8);        // blocks 8..23: gate_a | shift_f | scale_f | gate_f
      }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ TMA producer: weight ring
    if (lane == 0) {
      for (int T = 0, g = 0; T < ntT; ++T)
        for (int kk = 0; kk < KPT; ++kk, ++g) {                            // k-block kk of the tile's weight sequence
          const int sl = g % RS;
          if (g >= RS) mbar_wait(&B.wempty[sl], ((g / RS) - 1) & 1);
          const uint32_t d = su + O_R + sl * KB;
          mbar_expect_tx(&B.wfull[sl], KB);
          // sequence: Wo (2 k-blocks), Wab_0 (2), then per j: Wab_{j+1} (2, j < 3), Wd_j (1) -- the MMA issue order
          if (kk < 2) tma_load_2d(d, &mwo, &B.wfull[sl], kk * 64, 0);
          else if (kk < 4) tma_load_2d(d, &mwab, &B.wfull[sl], (kk - 2) * 64, 0);
          else {
            const int j = (kk - 4) / 3, e = (kk - 4) % 3;
            if (j < NCH - 1 && e < 2) tma_load_2d(d, &mwab, &B.wfull[sl], e * 64, (j + 1) * 128);
            else tma_load_2d(d, &mwd, &B.wfull[sl], j * HC, 0);
          }
        }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    auto wslot = [&](int g) { const int sl = g % RS; mbar_wait(&B.wfull[sl], (g / RS) & 1); return sl; };
    auto mma_k128 = [&](uint32_t d, uint32_t a, int& g, uint64_t* done, uint64_t* done2) {   // K = 128 over ring k-blocks g, g + 1
      for (int kb = 0; kb < 2; ++kb, ++g) {
        const int sl = wslot(g);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ts(d, a + kb * 32 + ks * 8, desc_k128(su + O_R + sl * KB) + (uint64_t)(ks * 2), I_N128, (kb > 0 || ks > 0) ? 1u : 0u);
          tc_commit(&B.wempty[sl]);
          if (kb == 1) { tc_commit(done); if (done2) tc_commit(done2); }
        }
        __syncwarp();
      }
    };
    for (int T = 0, g = 0; T < ntT; ++T) {
      mbar_wait(&B.afull, T & 1);                                          // gated in TMEM
      if (T >= 1) mbar_wait(&B.ffree, (T - 1) & 1);                        // the previous tile's ffn read out of T_F
      mma_k128(tmem + T_F, tmem + T_A, g, &B.attfull, nullptr);            // att = gated Wo^T  (in the ffn accumulator's columns)
      mbar_wait(&B.yfull, T & 1);                                          // y in TMEM, att read out
      // issue order (= the ring's weight order): ab_0, then per chunk j: ab_{j+1} (into the other a|b buffer), ffn += h_j Wd_j^T --
      // chunk j + 1's up-projection runs while the threads form h_j
      auto ab_mma = [&](int j) {
        const int ci = T * NCH + j, ab = ci & 1;
        if (ci >= 2) mbar_wait(&B.abfree[ab], ((ci >> 1) - 1) & 1);        // chunk ci - 2's a|b read out
        mma_k128(tmem + T_AB + ab * 128, tmem + T_A, g, &B.abfull[ab], j == NCH - 1 ? &B.yfree : nullptr);
      };
      ab_mma(0);
      for (int j = 0; j < NCH; ++j) {
        if (j + 1 < NCH) ab_mma(j + 1);
        const int ci = T * NCH + j, hb = ci & 1;
        mbar_wait(&B.hfull[hb], (ci >> 1) & 1);
        const int sl = wslot(g);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ts(tmem + T_F, tmem + T_H + hb * 32 + ks * 8, desc_k128(su + O_R + sl * KB) + (uint64_t)(ks * 2), I_N128, (j > 0 || ks > 0) ? 1u : 0u);
          tc_commit(&B.hfree[hb]);
          tc_commit(&B.wempty[sl]);
          if (j == NCH - 1) tc_commit(&B.ffull);
        }
        __syncwarp();
        ++g;
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ row threads: warpgroup w = channel half
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int sp = (int)r / AT, at = (int)r % AT;
    const bool rowok = (int)r < SP * AT;
    const int al = rowok ? at : 0;
    const size_t NS = (size_t)A * Bn;
    float* ssx = reinterpret_cast<float*>(sm + O_SS);
    auto modv = [&](int kind, int j) {                                     // float4 of channels 4 j .. of kind (0 gate_a, 1 shift_f, 2 scale_f, 3 gate_f)
      const int row = (kind * 4 + (j >> 3)) * AT + al;
      return *reinterpret_cast<const float4*>(sm + O_MOD + row * 128 + (((j & 7) ^ (row & 7)) << 4));
    };
    for (int T = 0; T < ntT; ++T) {
      int b, a0, s0; coords(T, b, a0, s0);
      const int n = (a0 + sp) * Bn + b, s = s0 + at;
      const bool ok = rowok && (size_t)n < NS && s < S;
      const size_t row = ((size_t)n * S + s) * C;
      mbar_wait(&B.gofull, T & 1);
      // ---- P1: gated = rn(sigmoid(g) o) for this warpgroup's 64 channels -> TMEM A
      if (T >= 1) mbar_wait(&B.yfree, (T - 1) & 1);                        // the previous tile's a|b MMAs have read y
      {
        uint32_t ga[32];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int ch = 8 * w + k;
          const uint4 gu = lds128(su + O_G + (ch >> 3) * KB + sw128(r, ch & 7)), ou = lds128(su + O_O + (ch >> 3) * KB + sw128(r, ch & 7));
          const uint32_t g4[4] = {gu.x, gu.y, gu.z, gu.w}, o4[4] = {ou.x, ou.y, ou.z, ou.w};
#pragma unroll
          for (int e = 0; e < 4; ++e)
            ga[4 * k + e] = pack_bf16(sigm(bf16lo(g4[e])) * bf16lo(o4[e]), sigm(bf16hi(g4[e])) * bf16hi(o4[e]));
        }
        tc_fence_after();
#pragma unroll
        for (int k = 0; k < 2; ++k) tmem_st16(trow + T_A + 32 * w + 16 * k, *reinterpret_cast<uint32_t(*)[16]>(ga + 16 * k));
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) { mbar_arrive(&B.afull); mbar_arrive(&B.gofree); }
      }
      // ---- P2: q1 = rn(q + rn(gate_a) rn(att)) (in place in the q tile), row RMS (exchanged), y -> TMEM A
      mbar_wait(&B.qfull, T & 1);
      mbar_wait(&B.attfull, T & 1);
      tc_fence_after();
      float q1v[64];
      float ssp = 0.f;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        uint32_t v[32];
        tmem_ld32(trow + T_F + 64 * w + 32 * hh, v);
        tmem_wait_ld();
        uint32_t tpk[16];
#pragma unroll
        for (int k = 0; k < 4; ++k) {                                      // 8 channels per smem chunk
          const int ch = 8 * w + 4 * hh + k;
          const uint32_t qa = su + O_Q + (ch >> 3) * KB + sw128(r, ch & 7);
          const uint4 qu = lds128(qa);
          const uint32_t q4[4] = {qu.x, qu.y, qu.z, qu.w};
          const float4 g0 = modv(0, 2 * ch), g1 = modv(0, 2 * ch + 1);
          const float gav[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
          uint32_t o4[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float t0 = rnb(__uint_as_float(v[8 * k + 2 * e])), t1 = rnb(__uint_as_float(v[8 * k + 2 * e + 1]));
            tpk[4 * k + e] = pack_bf16(t0, t1);
            const float x0 = rnb(bf16lo(q4[e]) + rnb(gav[2 * e]) * t0), x1 = rnb(bf16hi(q4[e]) + rnb(gav[2 * e + 1]) * t1);
            q1v[32 * hh + 8 * k + 2 * e] = x0; q1v[32 * hh + 8 * k + 2 * e + 1] = x1;
            ssp = fmaf(x0, x0, fmaf(x1, x1, ssp));
            o4[e] = pack_bf16(x0, x1);
          }
          sts128(qa, make_uint4(o4[0], o4[1], o4[2], o4[3]));
        }
        if (save && ok) {
          uint4* da = reinterpret_cast<uint4*>(Atts + row + 64 * w + 32 * hh);
#pragma unroll
          for (int k = 0; k < 4; ++k) da[k] = make_uint4(tpk[4 * k], tpk[4 * k + 1], tpk[4 * k + 2], tpk[4 * k + 3]);
        }
      }
      ssx[w * 128 + r] = ssp;
      named_bar_sync(1, 256);                                              // both halves' sums of squares (and att reads) done
      const float rstd = rsqrtf((ssx[r] + ssx[128 + r]) * (1.f / C) + eps);
      {
        uint32_t yp[32];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const int j4 = 16 * w + k;                                       // channels 4 j4 ..
          const float4 sh = modv(1, j4), sc = modv(2, j4);
          const float* x = q1v + 4 * k;
          yp[2 * k] = pack_bf16(x[0] * rstd * (1.f + sc.x) + sh.x, x[1] * rstd * (1.f + sc.y) + sh.y);
          yp[2 * k + 1] = pack_bf16(x[2] * rstd * (1.f + sc.z) + sh.z, x[3] * rstd * (1.f + sc.w) + sh.w);
        }
        tc_fence_after();
#pragma unroll
        for (int k = 0; k < 2; ++k) tmem_st16(trow + T_A + 32 * w + 16 * k, *reinterpret_cast<uint32_t(*)[16]>(yp + 16 * k));
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.yfull);
        if (save && ok) {
          uint4* dy = reinterpret_cast<uint4*>(Ys + row + 64 * w);
#pragma unroll
          for (int k = 0; k < 8; ++k) dy[k] = make_uint4(yp[4 * k], yp[4 * k + 1], yp[4 * k + 2], yp[4 * k + 3]);
          uint4* dq = reinterpret_cast<uint4*>(Q1s + row + 64 * w);
#pragma unroll
          for (int k = 0; k < 8; ++k)
            dq[k] = make_uint4(pack_bf16(q1v[8 * k], q1v[8 * k + 1]), pack_bf16(q1v[8 * k + 2], q1v[8 * k + 3]),
                               pack_bf16(q1v[8 * k + 4], q1v[8 * k + 5]), pack_bf16(q1v[8 * k + 6], q1v[8 * k + 7]));
        }
      }
      // ---- P3: per chunk j, h = rn(silu(a) b) for this warpgroup's 32 hidden units of the chunk -> TMEM h[j & 1]
      for (int j = 0; j < NCH; ++j) {
        const int hi = T * NCH + j, hb = hi & 1;
        mbar_wait(&B.abfull[hb], (hi >> 1) & 1);
        tc_fence_after();
        uint32_t av[32], bv[32];
        tmem_ld32(trow + T_AB + hb * 128 + 32 * w, av);
        tmem_ld32(trow + T_AB + hb * 128 + 64 + 32 * w, bv);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.abfree[hb]);
        uint32_t hp[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
          hp[k] = pack_bf16(a0 * sigm(a0) * __uint_as_float(bv[2 * k]), a1 * sigm(a1) * __uint_as_float(bv[2 * k + 1]));
        }
        if (hi >= 2) mbar_wait(&B.hfree[hb], ((hi >> 1) - 1) & 1);
        tc_fence_after();
        tmem_st16(trow + T_H + hb * 32 + 16 * w, hp);
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.hfull[hb]);
      }
      // ---- P4: out = rn(q1 + rn(gate_f) rn(ffn)) into the q tile, TMA store
      mbar_wait(&B.ffull, T & 1);
      tc_fence_after();
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        uint32_t v[32];
        tmem_ld32(trow + T_F + 64 * w + 32 * hh, v);
        tmem_wait_ld();
        if (hh == 1) {
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.ffree);
        }
        uint32_t fpk[16];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int ch = 8 * w + 4 * hh + k;
          const uint32_t qa = su + O_Q + (ch >> 3) * KB + sw128(r, ch & 7);
          const float4 g0 = modv(3, 2 * ch), g1 = modv(3, 2 * ch + 1);
          const float gfv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
          uint32_t o4[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float f0 = rnb(__uint_as_float(v[8 * k + 2 * e])), f1 = rnb(__uint_as_float(v[8 * k + 2 * e + 1]));
            fpk[4 * k + e] = pack_bf16(f0, f1);
            o4[e] = pack_bf16(q1v[32 * hh + 8 * k + 2 * e] + rnb(gfv[2 * e]) * f0, q1v[32 * hh + 8 * k + 2 * e + 1] + rnb(gfv[2 * e + 1]) * f1);
          }
          sts128(qa, make_uint4(o4[0], o4[1], o4[2], o4[3]));
        }
        if (save && ok) {
          uint4* df = reinterpret_cast<uint4*>(Ffs + row + 64 * w + 32 * hh);
#pragma unroll
          for (int k = 0; k < 4; ++k) df[k] = make_uint4(fpk[4 * k], fpk[4 * k + 1], fpk[4 * k + 2], fpk[4 * k + 3]);
        }
      }
      fence_proxy_async();
      named_bar_sync(1, 256);
      if (w == 0 && r == 0) {
        for (int kb = 0; kb < 2; ++kb) tma_store_4d(&mout, su + O_Q + kb * KB, kb * 64, s0, b, a0);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.qfree);
      }
    }
    if (w == 0 && r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
