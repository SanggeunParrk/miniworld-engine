// attn_dqb.cu — the augmented pair-bias attention core's backward dQ + dbias pass, sm_100a (the B200 port of the sm_90a attn_dqb.cu).
//
// Same fusion as the H100 kernel: per key block, recompute P from the row LSE, form dS = P (dP - D), accumulate dQ += dS K / sqrt 48
// and dbias += dS summed over the samples. What the B200 changes is WHERE the sample sum happens. On H100 a CTA held three samples
// of one query tile and pushed every key block's summed dS to L2 with atomics. Here TMEM holds S, dP, dS and dQ, which leaves the
// registers free to hold the dbias tile itself: a CTA owns (head, 128-query tile, 128-key chunk) and walks ALL A samples, so dbias
// is summed entirely on chip and written once with plain stores (no zero fill, no atomics). The price moves to dQ, which is now
// partial per key chunk and added into a zeroed fp32 dQ with v4 reductions (L / 128 contributions per element instead of A / 3).
//
// Per sample a (warpgroup w owns keys w * 64 .. w * 64 + 63 of the chunk, one query row per thread):
//   S = q K_w^T, dP = dO V_w^T  (TMEM, fp32)      P = 2^(S log2 e / sqrt 48 + bias log2 e - LSE)      dS = P (dP - D)
//   dbias_w += dS (registers)   dS -> TMEM (bf16)  dQ += dS K_w  (TS MMA, both warpgroups into one accumulator)
// TMEM: S[w] at w * 128, dP[w] at w * 128 + 64, dS[w] at 256 + w * 32 (bf16), dQ[b] at 320 + b * 48 (NDQ buffers, round robin).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef TRED
#define TRED 1                           // dQ partials leave through TMA bulk reduce-adds (smem-staged tiles) instead of per-thread red.v4
#endif
#ifndef STA
#define STA 2                            // q | dO | V ring (48 KB slots): released as soon as both S / dP MMAs of the step are done
#endif
#ifndef STK
#define STK 3                            // K ring (16 KB slots): released after the step's dQ MMAs
#endif
#ifndef BSL
#define BSL 1                            // bias item slots (1 frees 32 KB for a third stage)
#endif
#ifndef NDQ
#define NDQ 3                            // dQ accumulator buffers (the epilogue of step g runs NDQ - 1 steps later)
#endif
#ifndef ROT
#define ROT 1                            // start each key chunk's sample walk at a different sample (spreads the dQ reductions)
#endif
#ifndef ROTK
#define ROTK 1                           // sample offset per key chunk (0: A / chunks, spread evenly); small keeps the working set in L2
#endif
constexpr int BN = 64, KC = 128, DH = 48, QM = 128, DM = 768;
constexpr int T128 = 128 * 128;                                            // one [128 rows][128 B] tile
constexpr int SA = 3 * T128;                                               // q | dO | V = 48 KB
constexpr int O_A = 0, O_K = STA * SA, O_B = O_K + STK * T128, O_X = O_B + BSL * 2 * T128;   // bias: BSL item slots x ([128 q][64 keys] x 2)
constexpr int XA = 128 * 32 * 4, XB = 128 * 16 * 4;                        // dQ staging: cols 0-31 (128-B swizzle), 32-47 (64-B swizzle)
constexpr int O_BAR = O_X + (TRED ? 2 * (XA + XB) : 0);                    // double-buffered per warpgroup
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_S = 0, T_DS = 256, T_DQ = 320;                      // dQ[b] at 320 + 48 b
static_assert(T_DQ + NDQ * 48 <= 512, "TMEM");
constexpr uint32_t I_S = idesc_bf16(128, BN), I_DQ = idesc_bf16(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f, RSQD = 0.14433756729740643f;

#ifdef TRACE
__device__ unsigned long long g_tr[12][256];   // CTA 0: 0 prod issued, 1 S(g,0) issued, 2 wg0 s_full seen, 3 wg0 S/dP loaded, 4 wg0 ds_free seen, 5 wg0 dS done, 6 wg0 epi(g-1) done, 7 dQ(g,1) issued
#define TR(ev, i) do { if (blockIdx.x == 0 && (i) < 256) g_tr[ev][i] = clock64(); } while (0)
#else
#define TR(ev, i) do { } while (0)
#endif
struct Bars {
  uint64_t fullA[STA], emptyA[STA], fullK[STK], emptyK[STK], bfull[2], bempty[2], s_full[2], s_free[2], ds_full[2], ds_free[2], dq_full[NDQ], dq_free[NDQ];
  uint32_t tmem;
};
#ifndef NORED
#define NORED 0                          // timing diagnostics: 1 skips the dQ reductions
#endif
DEVI void tma_reduce_add_2d(const CUtensorMap* m, uint32_t src, int c0, int c1) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%0, {%2, %3}], [%1];" :: "l"(m), "r"(src), "r"(c0), "r"(c1) : "memory");
}
DEVI void tma_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }
DEVI void red4(float* p, float a, float b, float c, float d) {
  if (NORED) { if (a == 1234.5f && b == -1.f) *p = c + d; return; }
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
augattn_dqb_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                  const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap mqa,
                  const __grid_constant__ CUtensorMap mqb, const float* __restrict__ LSE,
                  const float* __restrict__ DD, float* __restrict__ DQ, float* __restrict__ DB, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = L / QM, nch = L / KC;
  const int items = 16 * mt * nch;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int ng = my_items * A;                                             // (item, sample) steps of this CTA
  auto item_of = [&](int li, int& c, int& m0, int& head) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    c = wi % nch; const int r = wi / nch;
    m0 = (r % mt) * QM; head = r / mt;
  };
  auto samp = [&](int c, int i) { return ROT ? (i + c * (ROTK ? ROTK : A / nch)) % A : i; };

  if (tid == 0) {
    for (int s = 0; s < STA; ++s) { mbar_init(&B.fullA[s], 1); mbar_init(&B.emptyA[s], 2); }
    for (int s = 0; s < STK; ++s) { mbar_init(&B.fullK[s], 1); mbar_init(&B.emptyK[s], 2); }
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.bfull[b], 1); mbar_init(&B.bempty[b], 8);
      mbar_init(&B.s_full[b], 1); mbar_init(&B.s_free[b], 4);
      mbar_init(&B.ds_full[b], 4); mbar_init(&B.ds_free[b], 1);
    }
    for (int b = 0; b < NDQ; ++b) { mbar_init(&B.dq_full[b], 2); mbar_init(&B.dq_free[b], 8); }
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
    for (int c = 0; c < NDQ * 3; ++c) tmem_st16(trow + T_DQ + c * 16, z);
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
        int c, m0, head; item_of(li, c, m0, head);
        const int bs = li % BSL, qcol = head * DH;
        if (li >= BSL) mbar_wait(&B.bempty[bs], ((li / BSL) - 1) & 1);
        mbar_expect_tx(&B.bfull[bs], 2 * T128);
        for (int w = 0; w < 2; ++w) tma_load_2d(su + O_B + (bs * 2 + w) * T128, &mb, &B.bfull[bs], c * KC + w * BN, head * L + m0);
        for (int i = 0; i < A; ++i, ++g) {
          const int sa = g % STA, sk = g % STK, a = samp(c, i);
          if (g >= STA) mbar_wait(&B.emptyA[sa], ((g / STA) - 1) & 1);
          const uint32_t pa = su + O_A + sa * SA;
          mbar_expect_tx(&B.fullA[sa], 3 * 128 * DH * 2);
          tma_load_2d(pa, &mq, &B.fullA[sa], qcol, a * L + m0);
          tma_load_2d(pa + T128, &mdo, &B.fullA[sa], qcol, a * L + m0);
          tma_load_2d(pa + 2 * T128, &mv, &B.fullA[sa], qcol, a * L + c * KC);
          if (g >= STK) mbar_wait(&B.emptyK[sk], ((g / STK) - 1) & 1);
          mbar_expect_tx(&B.fullK[sk], 128 * DH * 2);
          tma_load_2d(su + O_K + sk * T128, &mk, &B.fullK[sk], qcol, a * L + c * KC);
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ MMA issuers: warp 1 + w serves
    // warpgroup w. dQ is shared: every MMA accumulates onto it and the epilogue zeroes it after the read-out, so the issuers need no
    // ordering between each other.
    const int w = warp - 1;
    auto sdp_ready = [&](int g) { return mbar_test(&B.fullA[g % STA], (g / STA) & 1) && mbar_test(&B.fullK[g % STK], (g / STK) & 1) &&
                                         (g < 1 || mbar_test(&B.s_free[w], (g - 1) & 1)); };
    auto sdp = [&](int g) {                                                // S and dP of step g
      mbar_wait(&B.fullA[g % STA], (g / STA) & 1);
      mbar_wait(&B.fullK[g % STK], (g / STK) & 1);
      if (g >= 1) mbar_wait(&B.s_free[w], (g - 1) & 1);
      tc_fence_after();
      const uint32_t pa = su + O_A + (g % STA) * SA;
      const uint64_t dq = desc_k128(pa), ddo = desc_k128(pa + T128);
      const uint64_t dk = desc_k128(su + O_K + (g % STK) * T128 + w * BN * 128), dv = desc_k128(pa + 2 * T128 + w * BN * 128);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 3; ++ks) umma_ss(tmem + T_S + w * 128, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < 3; ++ks) umma_ss(tmem + T_S + w * 128 + 64, ddo + (uint64_t)(ks * 2), dv + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w]);
        tc_commit(&B.emptyA[g % STA]);
      }
      __syncwarp();
      if (w == 0 && lane == 0) TR(1, g);
    };
    if (ng > 0) sdp(0);
    for (int g = 0; g < ng; ++g) {
      const int b = g % NDQ;
      // S(g + 1) as soon as its operands are in and warpgroup w holds S(g) in registers, but never ahead of a ready dQ(g)
      bool issued = g + 1 >= ng;
      while (!__shfl_sync(0xffffffffu, (int)mbar_test(&B.ds_full[w], g & 1), 0)) {   // warp-uniform decisions (lane 0's view)
        if (!issued && __shfl_sync(0xffffffffu, (int)sdp_ready(g + 1), 0)) { sdp(g + 1); issued = true; }
        else __nanosleep(20);
      }
      if (g >= NDQ) mbar_wait(&B.dq_free[b], ((g / NDQ) - 1) & 1);         // step g - NDQ's dQ has been read out (and zeroed)
      tc_fence_after();
      const uint64_t dk = desc_mn128(su + O_K + (g % STK) * T128 + w * BN * 128, 8192);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ts(tmem + T_DQ + b * 48, tmem + T_DS + w * 32 + ks * 8, dk + (uint64_t)(ks * 2048 >> 4), I_DQ, 1u);
        tc_commit(&B.ds_free[w]);
        tc_commit(&B.emptyK[g % STK]);
        tc_commit(&B.dq_full[b]);
      }
      __syncwarp();
      if (w == 1 && lane == 0) TR(7, g);
      if (!issued) sdp(g + 1);
    }
#ifdef TRACE
  } else if (warp == 3) {                                                  // observer: row 0 <- s_full[0] landed (overrides prod issue)
    if (lane == 0 && blockIdx.x == 0)
      for (int g = 0; g < ng && g < 256; ++g) { mbar_wait_spin(&B.s_full[0], g & 1); g_tr[0][g] = clock64(); }
#endif
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ dS warpgroups
    setmaxnreg_inc<224>();
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const f2 CQL = mk2(RSQD * LOG2E, RSQD * LOG2E), L2E = mk2(LOG2E, LOG2E);
    auto epi = [&](int g, int arow, int head) {                            // dQ of step g (rows arow .. arow + 127 of DQ, head): TMEM -> reductions
      const int b = g % NDQ;
      mbar_wait(&B.dq_full[b], (g / NDQ) & 1);
      if (w == 0 && r == 0) TR(8, g);
      tc_fence_after();
      const int nc = w == 0 ? 32 : 16;                                     // warpgroup 0: columns 0-31, warpgroup 1: 32-47
      uint32_t v[32];
      if (w == 0) tmem_ld32(trow + T_DQ + b * 48, v);
      else tmem_ld16(trow + T_DQ + b * 48 + 32, *reinterpret_cast<uint32_t(*)[16]>(v));
      tmem_wait_ld();
      {
        const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        tmem_st16(trow + T_DQ + b * 48 + w * 32, z);
        if (w == 0) tmem_st16(trow + T_DQ + b * 48 + 16, z);
        tmem_wait_st();
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dq_free[b]);
      if (w == 0 && r == 0) TR(9, g);
      if (TRED) {
        const bool issuer = r == 0;
        const int xb = g & 1;
        const uint32_t xs = su + O_X + (w == 0 ? xb * XA : 2 * XA + xb * XB);
        if (issuer) tma_wait_read1();                                      // the reduce of step g - 2 has read this buffer
        named_bar_sync(1 + w, 128);
        if (w == 0 && r == 0) TR(10, g);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          if (q * 4 >= nc) break;
          const uint4 u = make_uint4(__float_as_uint(__uint_as_float(v[4 * q]) * RSQD), __float_as_uint(__uint_as_float(v[4 * q + 1]) * RSQD),
                                     __float_as_uint(__uint_as_float(v[4 * q + 2]) * RSQD), __float_as_uint(__uint_as_float(v[4 * q + 3]) * RSQD));
          sts128(xs + (w == 0 ? sw128(r, q) : sw64(r, q)), u);
        }
        fence_proxy_async();
        named_bar_sync(1 + w, 128);
        if (w == 0 && r == 0) TR(11, g);
        if (issuer) {
          tma_reduce_add_2d(w == 0 ? &mqa : &mqb, xs, head * DH + w * 32, arow);
          tma_store_commit();
        }
      } else {
        float* drow = DQ + ((size_t)arow + r) * DM + head * DH + w * 32;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          if (k * 4 >= nc) break;
          red4(drow + 4 * k, __uint_as_float(v[4 * k]) * RSQD, __uint_as_float(v[4 * k + 1]) * RSQD, __uint_as_float(v[4 * k + 2]) * RSQD,
               __uint_as_float(v[4 * k + 3]) * RSQD);
        }
      }
    };
    f2 db[BN / 2];
    // positions advance incrementally: the runtime divisions of item_of run once per item, not per step
    auto advance = [&](int& pli, int& pi, int& pc, int& pm0, int& ph, int& pa) {
      if (++pi == A) { pi = 0; if (++pli < my_items) { item_of(pli, pc, pm0, ph); pa = samp(pc, 0); } }
      else pa = (pa + 1 == A) ? 0 : pa + 1;
    };
    int li = 0, i = 0, c = 0, m0 = 0, head = 0, a = 0;
    if (ng > 0) { item_of(0, c, m0, head); a = samp(c, 0); }
    int nli = li, ni = i, nc_ = c, nm0 = m0, nh = head, na = a;          // the next step's position (LSE / D prefetch)
    float lse_n = 0.f, dd_n = 0.f;
    if (ng > 0) { const size_t ri = ((size_t)a * 16 + head) * L + m0 + r; lse_n = LSE[ri]; dd_n = DD[ri]; }
    int e1r = 0, e1h = 0, e2r = 0, e2h = 0;                                // (DQ row, head) of steps g - 1 and g - 2
    static_assert(NDQ == 3, "the epilogue lag below is NDQ - 1 = 2 steps");
    for (int g = 0; g < ng; ++g) {
      const int bs = li % BSL;
      const float lse = lse_n, dd = dd_n;                                  // loaded one step ahead
      advance(nli, ni, nc_, nm0, nh, na);
      if (g + 1 < ng) { const size_t ri = ((size_t)na * 16 + nh) * L + nm0 + r; lse_n = LSE[ri]; dd_n = DD[ri]; }
      if (i == 0) {
#pragma unroll
        for (int j = 0; j < BN / 2; ++j) db[j] = mk2(0.f, 0.f);
        mbar_wait(&B.bfull[bs], (li / BSL) & 1);
      }
      mbar_wait(&B.s_full[w], g & 1);
      if (w == 0 && r == 0) TR(2, g);
      tc_fence_after();
      const f2 NL = mk2(-lse, -lse), ND = mk2(-dd, -dd);
      const uint32_t sb = su + O_B + (bs * 2 + w) * T128;
      // quarters of 16 keys, software-pipelined: the next quarter's TMEM loads are in flight while this one is computed
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
            db[jj] = add2(db[jj], ds);
            pk[jj] = pack_bf16(lo2(ds), hi2(ds));
          }
        }
        if (qq < 3) tmem_wait_ld();
        if (qq == 2) {                                                     // S / dP fully in registers: the next S MMA may overwrite them
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.s_free[w]);
          if (w == 0 && r == 0) TR(3, g);
        }
      }
      if (g >= 1) mbar_wait(&B.ds_free[w], (g - 1) & 1);                   // dQ(g - 1) has consumed the previous dS
      if (w == 0 && r == 0) TR(4, g);
      tc_fence_after();
      tmem_st16(trow + T_DS + w * 32, *reinterpret_cast<uint32_t(*)[16]>(pk));
      tmem_st16(trow + T_DS + w * 32 + 16, *reinterpret_cast<uint32_t(*)[16]>(pk + 16));
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.ds_full[w]);
      if (w == 0 && r == 0) TR(5, g);
      if (g >= 2) epi(g - 2, e2r, e2h);
      if (w == 0 && r == 0) TR(6, g);
      if (i == A - 1) {
        // ---- the item's dbias: this thread's 64 keys of query row m0 + r, natural units, plain stores
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.bempty[bs]);
        float* brow = DB + ((size_t)head * L + m0 + r) * L + c * KC + w * BN;
#pragma unroll
        for (int k = 0; k < BN / 4; ++k)
          *reinterpret_cast<float4*>(brow + 4 * k) = make_float4(lo2(db[2 * k]), hi2(db[2 * k]), lo2(db[2 * k + 1]), hi2(db[2 * k + 1]));
      }
      e2r = e1r; e2h = e1h; e1r = a * L + m0; e1h = head;
      advance(li, i, c, m0, head, a);
    }
    if (ng >= 2) epi(ng - 2, e2r, e2h);
    if (ng >= 1) epi(ng - 1, e1r, e1h);
    if (TRED && r == 0) tma_store_wait0();                                 // the last reduces have completed
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
