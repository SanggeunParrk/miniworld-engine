// attn_dq.cu — the atom DiT's attention backward, dQ pass, sm_100a:  dQ[a] = sum_j dS[a] K / sqrt 32, dS = P (dP - D).
//
// dbias has its own pass (attn_dbias.cu), so dQ can be resident: a CTA owns (sample, head, 128 queries) and walks every 128-key chunk
// (warpgroup w takes keys w * 64 .. w * 64 + 63 of each chunk); dQ accumulates in TMEM across the whole key range and leaves once per
// item through a TMA store -- no partials, no reductions, no zero fill. Per chunk:
//   S = q K^T, dP = dO V^T (TMEM)    P = 2^(S log2 e / sqrt 32 + bias log2 e - LSE)    dS = P (dP - D) -> TMEM (bf16)    dQ += dS K
// Structure from the token-DiT dqb kernel: two MMA issuers (one per warpgroup; dQ(g) is issued first, S(g + 1) whenever its operands are
// in), every dQ MMA accumulates (the epilogue zeroes the buffer after reading it), incremental indices.
// TMEM: S[w] at w * 128, dP[w] at w * 128 + 64, dS[w] at 256 + w * 32 (bf16), dQ[b] at 320 + b * 32 (two buffers, alternate items).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef ST
#define ST 2                             // K | V | bias stages (64 KB)
#endif
#ifndef QS
#define QS 2                             // q | dO item slots (32 KB)
#endif
constexpr int BN = 64, KC = 128, DH = 32, QM = 128, DM = 128, NH = 4;
constexpr int T128 = 128 * 128;
constexpr int STB = 4 * T128;                                              // K | V | bias [128 q][64 keys] x 2
constexpr int O_Q = 0, O_ST = QS * 2 * T128, O_X = O_ST + ST * STB;
constexpr int XW = 128 * 16 * 4;                                           // dQ staging per warpgroup: [128 rows][16 fp32] (SW64)
constexpr int O_BAR = O_X + 2 * XW, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_S = 0, T_DS = 256, T_DQ = 320;
constexpr uint32_t I_S = idesc_bf16(128, BN), I_DQ = idesc_bf16(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f, RSQD = 0.17677669529663687f;

struct Bars {
  uint64_t qfull[QS], qempty[QS], full[ST], empty[ST], s_full[2], s_free[2], ds_full[2], ds_free[2], dq_full[2], dq_free[2];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(384, 1)
atom_dq_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
              const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap mdq,
              const float* __restrict__ LSE, const float* __restrict__ DD, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = L / QM, nch = L / KC;
  const int items = A * NH * mt;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int ng = my_items * nch;
  auto item_of = [&](int li, int& a, int& m0, int& head) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    a = wi % A; const int r = wi / A;                                      // neighbouring CTAs: other samples of the same bias rows
    m0 = (r % mt) * QM; head = r / mt;
  };

  if (tid == 0) {
    for (int s = 0; s < QS; ++s) { mbar_init(&B.qfull[s], 1); mbar_init(&B.qempty[s], 2); }
    for (int s = 0; s < ST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 2); }
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.s_full[b], 1); mbar_init(&B.s_free[b], 4);
      mbar_init(&B.ds_full[b], 4); mbar_init(&B.ds_free[b], 1);
      mbar_init(&B.dq_full[b], 2); mbar_init(&B.dq_free[b], 8);
    }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (warp >= 4 && warp < 8) {                                             // the dQ buffers start at zero; every dQ MMA accumulates
    const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t trow = tmem + ((uint32_t)(warp & 3) * 32 << 16);
    for (int c = 0; c < 4; ++c) tmem_st16(trow + T_DQ + c * 16, z);
    tmem_wait_st();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        int a, m0, head; item_of(li, a, m0, head);
        const int qs = li % QS, qcol = head * DH;
        if (li >= QS) mbar_wait(&B.qempty[qs], ((li / QS) - 1) & 1);
        mbar_expect_tx(&B.qfull[qs], 2 * 128 * DH * 2);
        tma_load_2d(su + O_Q + qs * 2 * T128, &mq, &B.qfull[qs], qcol, a * L + m0);
        tma_load_2d(su + O_Q + qs * 2 * T128 + T128, &mdo, &B.qfull[qs], qcol, a * L + m0);
        for (int c = 0; c < nch; ++c, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.full[s], 2 * 128 * DH * 2 + 2 * T128);
          tma_load_2d(st, &mk, &B.full[s], qcol, a * L + c * KC);
          tma_load_2d(st + T128, &mv, &B.full[s], qcol, a * L + c * KC);
          for (int w = 0; w < 2; ++w) tma_load_2d(st + (2 + w) * T128, &mb, &B.full[s], c * KC + w * BN, head * L + m0);
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ MMA issuers: warp 1 + w serves warpgroup w
    const int w = warp - 1;
    auto sdp_ready = [&](int g, int li, int c) {
      return (c > 0 || mbar_test(&B.qfull[li % QS], (li / QS) & 1)) && mbar_test(&B.full[g % ST], (g / ST) & 1) &&
             (g < 1 || mbar_test(&B.s_free[w], (g - 1) & 1));
    };
    auto sdp = [&](int g, int li, int c) {                                 // S and dP of step g (item li, key chunk c)
      const int qs = li % QS;
      if (c == 0) mbar_wait(&B.qfull[qs], (li / QS) & 1);
      mbar_wait(&B.full[g % ST], (g / ST) & 1);
      if (g >= 1) mbar_wait(&B.s_free[w], (g - 1) & 1);
      tc_fence_after();
      const uint32_t st = su + O_ST + (g % ST) * STB, pq = su + O_Q + qs * 2 * T128;
      const uint64_t dq = desc_k128(pq), ddo = desc_k128(pq + T128);
      const uint64_t dk = desc_k128(st + w * BN * 128), dv = desc_k128(st + T128 + w * BN * 128);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks) umma_ss(tmem + T_S + w * 128, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks) umma_ss(tmem + T_S + w * 128 + 64, ddo + (uint64_t)(ks * 2), dv + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w]);
        if (c == nch - 1) tc_commit(&B.qempty[qs]);
      }
      __syncwarp();
    };
    if (ng > 0) sdp(0, 0, 0);
    for (int g = 0, li = 0, c = 0; g < ng; ++g) {
      const int b = li & 1;
      int li2 = li, c2 = c + 1;
      if (c2 == nch) { c2 = 0; ++li2; }
      bool issued = g + 1 >= ng;
      while (!__shfl_sync(0xffffffffu, (int)mbar_test(&B.ds_full[w], g & 1), 0)) {   // dQ(g) first; S(g + 1) when its operands are in
        if (!issued && __shfl_sync(0xffffffffu, (int)sdp_ready(g + 1, li2, c2), 0)) { sdp(g + 1, li2, c2); issued = true; }
        else __nanosleep(20);
      }
      if (c == 0 && li >= 2) mbar_wait(&B.dq_free[b], ((li >> 1) - 1) & 1);   // item li - 2's dQ has been read out (and zeroed)
      tc_fence_after();
      const uint64_t dk = desc_mn128(su + O_ST + (g % ST) * STB + w * BN * 128, 8192);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ts(tmem + T_DQ + b * 32, tmem + T_DS + w * 32 + ks * 8, dk + (uint64_t)(ks * 2048 >> 4), I_DQ, 1u);
        tc_commit(&B.ds_free[w]);
        tc_commit(&B.empty[g % ST]);
        if (c == nch - 1) tc_commit(&B.dq_full[b]);
      }
      __syncwarp();
      if (!issued) sdp(g + 1, li2, c2);
      li = li2; c = c2;
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ dS warpgroups
    setmaxnreg_inc<224>();
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const f2 CQL = mk2(RSQD * LOG2E, RSQD * LOG2E), L2E = mk2(LOG2E, LOG2E);
    int li = 0, c = 0, a = 0, m0 = 0, head = 0;
    if (ng > 0) item_of(0, a, m0, head);
    float lse = 0.f, dd = 0.f, lse_n = 0.f, dd_n = 0.f;
    if (ng > 0) { const size_t ri = ((size_t)a * NH + head) * L + m0 + r; lse_n = LSE[ri]; dd_n = DD[ri]; }
    for (int g = 0; g < ng; ++g) {
      if (c == 0) {                                                        // item start: take the prefetched LSE / D, prefetch the next item's
        lse = lse_n; dd = dd_n;
        if (li + 1 < my_items) {
          int a2, m2, h2; item_of(li + 1, a2, m2, h2);
          const size_t ri = ((size_t)a2 * NH + h2) * L + m2 + r; lse_n = LSE[ri]; dd_n = DD[ri];
        }
      }
      mbar_wait(&B.s_full[w], g & 1);
      tc_fence_after();
      const f2 NL = mk2(-lse, -lse), ND = mk2(-dd, -dd);
      const uint32_t sb = su + O_ST + (g % ST) * STB + (2 + w) * T128;
      uint32_t pk[32];
      uint32_t sa[16], da[16], sq[16], dq[16];
      tmem_ld16(trow + T_S + w * 128, sa);
      tmem_ld16(trow + T_S + w * 128 + 64, da);
      tmem_wait_ld();
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        uint32_t (&sv)[16] = (qq & 1) ? sq : sa;
        uint32_t (&dv)[16] = (qq & 1) ? dq : da;
        uint32_t (&sn)[16] = (qq & 1) ? sa : sq;
        uint32_t (&dn)[16] = (qq & 1) ? da : dq;
        if (qq < 3) {
          tmem_ld16(trow + T_S + w * 128 + (qq + 1) * 16, sn);
          tmem_ld16(trow + T_S + w * 128 + 64 + (qq + 1) * 16, dn);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint4 bw = lds128(sb + sw128(r, qq * 2 + h));
          const uint32_t bb[4] = {bw.x, bw.y, bw.z, bw.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int j = h * 4 + e, jj = qq * 8 + j;
            const f2 x = fma2(mk2u(sv[2 * j], sv[2 * j + 1]), CQL, fma2(mk2(bf16lo(bb[e]), bf16hi(bb[e])), L2E, NL));
            const f2 p = mk2(ex2f(lo2(x)), ex2f(hi2(x)));
            const f2 ds = mul2(p, add2(mk2u(dv[2 * j], dv[2 * j + 1]), ND));
            pk[jj] = pack_bf16(lo2(ds), hi2(ds));
          }
        }
        if (qq < 3) tmem_wait_ld();
        if (qq == 2) {
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.s_free[w]);
        }
      }
      if (g >= 1) mbar_wait(&B.ds_free[w], (g - 1) & 1);                   // dQ(g - 1) has consumed the previous dS
      tc_fence_after();
      tmem_st16(trow + T_DS + w * 32, *reinterpret_cast<uint32_t(*)[16]>(pk));
      tmem_st16(trow + T_DS + w * 32 + 16, *reinterpret_cast<uint32_t(*)[16]>(pk + 16));
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.ds_full[w]);
      if (c == nch - 1) {
        // ---- the item's dQ: warpgroup w writes columns w * 16 .. w * 16 + 15 (x 1 / sqrt 32) through a TMA store
        const int b = li & 1;
        mbar_wait(&B.dq_full[b], (li >> 1) & 1);
        tc_fence_after();
        uint32_t v[16];
        tmem_ld16(trow + T_DQ + b * 32 + w * 16, v);
        tmem_wait_ld();
        const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        tmem_st16(trow + T_DQ + b * 32 + w * 16, z);
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.dq_free[b]);
        const uint32_t xs = su + O_X + w * XW;
        if (r == 0) tma_store_wait_read0();
        named_bar_sync(1 + w, 128);
#pragma unroll
        for (int q = 0; q < 4; ++q)
          sts128(xs + sw64(r, q), make_uint4(__float_as_uint(__uint_as_float(v[4 * q]) * RSQD), __float_as_uint(__uint_as_float(v[4 * q + 1]) * RSQD),
                                             __float_as_uint(__uint_as_float(v[4 * q + 2]) * RSQD), __float_as_uint(__uint_as_float(v[4 * q + 3]) * RSQD)));
        fence_proxy_async();
        named_bar_sync(1 + w, 128);
        if (r == 0) { tma_store_2d(&mdq, xs, head * DH + w * 16, a * L + m0); tma_store_commit(); }
      }
      if (++c == nch) { c = 0; if (++li < my_items) item_of(li, a, m0, head); }
    }
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
