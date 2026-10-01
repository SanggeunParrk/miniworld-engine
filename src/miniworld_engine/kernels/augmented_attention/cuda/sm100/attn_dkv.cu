// attn_dkv.cu — the augmented pair-bias attention core's backward dK / dV pass, sm_100a (the B200 port of the sm_90a attn_dkv.cu,
// built there with DBIAS = 0 next to attn_dqb).
//
// Transposed as on H100: a CTA owns (sample, head, 128 keys) and streams the query blocks; one key row per thread:
//   S^T = K q^T, dP^T = V dO^T (TMEM, fp32)     P^T = 2^(S^T log2 e / sqrt 48 + bias^T log2 e - LSE)     dS^T = P^T (dP^T - D)
//   dV += P^T dO, dK += dS^T q                  (TS MMAs: P^T / dS^T bf16 in TMEM, dO / q as loaded = MN-major B)
// The two dS warpgroups take alternate 64-query blocks of the same key tile and feed one dK / dV accumulator pair. The bias arrives
// TRANSPOSED ([H, L(key), L(query)], a one-off copy in the op's prep) so each thread reads its key row contiguously; LSE and D of the
// block's queries are bulk-copied next to it and read as broadcasts.
// TMEM: S^T[w] at w * 128, dP^T[w] at w * 128 + 64, P^T[w] at 256 + w * 64 (bf16), dS^T[w] at 288 + w * 64, dK at 384, dV at 432.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef PP
#define PP 1                             // FA4-style ping-pong: the two dS warpgroups take turns on the exponentials (named bars 3, 4)
#endif
#ifndef TST
#define TST 1                            // dK / dV leave through smem-staged TMA bulk stores (coalesced) instead of per-row st.v4
#endif
#ifndef ST
#define ST (TST ? 3 : 4)
#endif
#ifndef NHEAD
#define NHEAD 16                         // heads (AttentionPairBias: 8, rows NHEAD * 48 wide)
#endif
#ifndef DHP
#define DHP 48                           // head width in memory and in the MMAs: 48 (token DiT 16 x 48), 64, 32, 16; 16 x 24: 32 (the
#endif                                   // projection pads each head's 24 channels with 8 zero ones, so q / k / v / dO pads are exactly 0)
#ifndef RSQDV
#define RSQDV 0.14433756729740643f       // 1 / sqrt(real head dim): 1 / sqrt 48; 1 / sqrt 24 = 0.2041241452319315f; 1 / sqrt 16 = 0.25f; 1 / sqrt 32, 1 / sqrt 64 = 0.125f
#endif
static_assert(DHP == 64 || DHP == 48 || DHP == 32 || DHP == 16, "head width 64, 48, 32 or 16");
constexpr int BQ = 64, KM = 128, DH = DHP, DM = NHEAD * DH;
constexpr int TQ = BQ * 128, TKV = KM * 128, TBT = KM * BQ * 2;           // q / dO block 8 KB, K / V tile 16 KB, bias^T tile 16 KB
constexpr int STB = 2 * TQ + TBT + 1024;                                  // q | dO | bias^T | LSE, D (512 B each) = 33 KB
constexpr int O_KV = 0, O_ST = 2 * 2 * TKV, O_BAR = O_ST + ST * STB;       // K / V: 2 item slots x (K | V)
constexpr int XA = 128 * 32 * 4, XB = 128 * 16 * 4;                        // output staging per warpgroup: cols 0-31 (SW128), 32-47 (SW64)
constexpr int O_X = O_BAR + 1024, SMEM_BYTES = O_X + (TST ? 2 * (XA + XB) : 0);    // swizzled staging needs 1 KB alignment
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_S = 0, T_P = 256, T_DK = 384, T_DV = T_DK + DH;     // dV after dK (432 for DH 48)
constexpr uint32_t I_S = idesc_bf16(128, BQ), I_O = idesc_bf16(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f, RSQD = RSQDV;

#ifdef TRACE
__device__ unsigned long long g_tr[8][256];
#define TR(ev, i) do { if (blockIdx.x == 0 && (i) < 256) g_tr[ev][i] = clock64(); } while (0)
#else
#define TR(ev, i) do { } while (0)
#endif
#ifndef PMOD
#define PMOD 4                           // of every PMOD exponential pairs, PCNT run on the FMA pipe (polynomial), the rest on MUFU
#endif
#ifndef PCNT
#define PCNT 0
#endif
// 2^x for a pair on the FMA pipe (as attn_fwd2.cu): degree-3 fit of 2^f, f = x - round(x), exponent added to the bit pattern
DEVI f2 ex2_poly2(f2 x) {
  const float x0 = fminf(fmaxf(lo2(x), -126.f), 126.f), x1 = fminf(fmaxf(hi2(x), -126.f), 126.f);
  const f2 xc = mk2(x0, x1), C = mk2(12582912.f, 12582912.f);
  const f2 j = add2(xc, C);
  const f2 f = add2(xc, neg2(add2(j, neg2(C))));
  f2 p = fma2(mk2(0.0555041086648216f, 0.0555041086648216f), f, mk2(0.2402264923172785f, 0.2402264923172785f));
  p = fma2(p, f, mk2(0.6931471805599453f, 0.6931471805599453f));
  p = fma2(p, f, mk2(1.0f, 1.0f));
  return mk2(__int_as_float(__float_as_int(lo2(p)) + (__float_as_int(lo2(j)) << 23)), __int_as_float(__float_as_int(hi2(p)) + (__float_as_int(hi2(j)) << 23)));
}
struct Bars {
  uint64_t full[ST], empty[ST], kvfull[2], kvempty[2], s_full[2], s_free[2], ds_full[2], ds_free[2], acc_full, acc_free;
  uint32_t tmem;
};
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1)
augattn_dkv_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                  const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mbt, const __grid_constant__ CUtensorMap mka,
                  const __grid_constant__ CUtensorMap mkb, const __grid_constant__ CUtensorMap mva, const __grid_constant__ CUtensorMap mvb,
                  const float* __restrict__ LSE,
                  const float* __restrict__ DD, float* __restrict__ DK, float* __restrict__ DV, float* __restrict__ DQZ, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kt = L / KM, nb = L / BQ;                                      // nb is even: every item starts on warpgroup 0
  const int items = A * NHEAD * kt;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int nblk = my_items * nb;
  auto item_of = [&](int li, int& a, int& k0, int& head) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    a = wi % A; const int r = wi / A;                                      // neighbouring CTAs: other samples of the same (head, key tile)
    k0 = (r % kt) * KM; head = r / kt;
  };

  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.kvfull[b], 1); mbar_init(&B.kvempty[b], 2);
      mbar_init(&B.s_full[b], 1); mbar_init(&B.s_free[b], 4);
      mbar_init(&B.ds_full[b], 4); mbar_init(&B.ds_free[b], 1);
    }
    mbar_init(&B.acc_full, 2); mbar_init(&B.acc_free, 8);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (warp >= 4 && warp < 8) {                                             // dK / dV start at zero; every MMA accumulates
    const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t trow = tmem + ((uint32_t)(warp & 3) * 32 << 16);
#pragma unroll
    for (int c = 0; c < 2 * DH / 16; ++c) tmem_st16(trow + T_DK + c * 16, z);   // dK and dV: 2 DH columns
    tmem_wait_st();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      int G = 0;
      for (int li = 0; li < my_items; ++li) {
        int a, k0, head; item_of(li, a, k0, head);
        const int ks = li & 1, qcol = head * DH;
        if (li >= 2) mbar_wait(&B.kvempty[ks], ((li >> 1) - 1) & 1);
        mbar_expect_tx(&B.kvfull[ks], 2 * KM * DH * 2);
        tma_load_2d(su + O_KV + ks * 2 * TKV, &mk, &B.kvfull[ks], qcol, a * L + k0);
        tma_load_2d(su + O_KV + ks * 2 * TKV + TKV, &mv, &B.kvfull[ks], qcol, a * L + k0);
        const size_t li0 = ((size_t)a * NHEAD + head) * L;
        for (int n = 0; n < nb; ++n, ++G) {
          const int s = G % ST;
          if (G >= ST) mbar_wait(&B.empty[s], ((G / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.full[s], 2 * BQ * DH * 2 + TBT + 2 * BQ * 4);
          TR(0, G);
          tma_load_2d(st, &mq, &B.full[s], qcol, a * L + n * BQ);
          tma_load_2d(st + TQ, &mdo, &B.full[s], qcol, a * L + n * BQ);
          tma_load_2d(st + 2 * TQ, &mbt, &B.full[s], n * BQ, head * L + k0);
          bulk_g2s(st + 2 * TQ + TBT, LSE + li0 + n * BQ, BQ * 4, &B.full[s]);
          bulk_g2s(st + 2 * TQ + TBT + 512, DD + li0 + n * BQ, BQ * 4, &B.full[s]);
          TR(0, G);
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ MMA issuers: warp 1 + w serves
    // warpgroup w (blocks G = w, w + 2, ...). dK / dV are shared: every MMA accumulates onto them, the epilogue zeroes them after the
    // read-out, so the two issuers need no ordering between each other.
    const int w = warp - 1;
    auto sdp_ready = [&](int G, int li, int n) {
      return (n > 1 || mbar_test(&B.kvfull[li & 1], (li >> 1) & 1)) && mbar_test(&B.full[G % ST], (G / ST) & 1) &&
             (G < 2 || mbar_test(&B.s_free[w], ((G >> 1) - 1) & 1));
    };
    auto sdp = [&](int G, int li, int n) {                                 // S^T and dP^T of block G (item li, block n of the item)
      const int s = G % ST, ks = li & 1;
      if (n <= 1) mbar_wait(&B.kvfull[ks], (li >> 1) & 1);
      mbar_wait(&B.full[s], (G / ST) & 1);
      if (G >= 2) mbar_wait(&B.s_free[w], ((G >> 1) - 1) & 1);
      tc_fence_after();
      const uint32_t st = su + O_ST + s * STB, kv = su + O_KV + ks * 2 * TKV;
      const uint64_t dk = desc_k128(kv), dv = desc_k128(kv + TKV), dq = desc_k128(st), ddo = desc_k128(st + TQ);
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < DH / 16; ++k) umma_ss(tmem + T_S + w * 128, dk + (uint64_t)(k * 2), dq + (uint64_t)(k * 2), I_S, k > 0 ? 1u : 0u);
#pragma unroll
        for (int k = 0; k < DH / 16; ++k) umma_ss(tmem + T_S + w * 128 + 64, dv + (uint64_t)(k * 2), ddo + (uint64_t)(k * 2), I_S, k > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w]);
      }
      __syncwarp();
      if (lane == 0 && w == 0) TR(1, G);
    };
    if (w < nblk) sdp(w, 0, w);
    for (int G = w, li = 0, n = w; G < nblk; G += 2) {                     // (li, n) advance incrementally (nb is even)
      const int s = G % ST;
      int li2 = li, n2 = n + 2;
      if (n2 >= nb) { n2 -= nb; ++li2; }
      // S^T(G + 2) as soon as its operands are in and warpgroup w holds S^T(G) in registers, but never ahead of a ready dV / dK(G)
      bool issued = G + 2 >= nblk;
      while (!__shfl_sync(0xffffffffu, (int)mbar_test(&B.ds_full[w], (G >> 1) & 1), 0)) {
        if (!issued && __shfl_sync(0xffffffffu, (int)sdp_ready(G + 2, li2, n2), 0)) { sdp(G + 2, li2, n2); issued = true; }
        else __nanosleep(20);
      }
      if (n <= 1 && li >= 1) mbar_wait(&B.acc_free, (li - 1) & 1);          // the previous item's dK / dV have been read out (and zeroed)
      tc_fence_after();
      const uint32_t st = su + O_ST + s * STB;
      const uint64_t dq = desc_mn128(st, 8192), ddo = desc_mn128(st + TQ, 8192);
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < 4; ++k) umma_ts(tmem + T_DV, tmem + T_P + w * 64 + k * 8, ddo + (uint64_t)(k * 2048 >> 4), I_O, 1u);
#pragma unroll
        for (int k = 0; k < 4; ++k) umma_ts(tmem + T_DK, tmem + T_P + w * 64 + 32 + k * 8, dq + (uint64_t)(k * 2048 >> 4), I_O, 1u);
        tc_commit(&B.ds_free[w]);
        tc_commit(&B.empty[s]);
        if (n >= nb - 2) { tc_commit(&B.acc_full); tc_commit(&B.kvempty[li & 1]); }
      }
      __syncwarp();
      if (lane == 0 && w == 0) TR(2, G);
      if (!issued) sdp(G + 2, li2, n2);
      li = li2; n = n2;
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ P^T / dS^T warpgroups
    setmaxnreg_inc<224>();
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const f2 CQL = mk2(RSQD * LOG2E, RSQD * LOG2E), L2E = mk2(LOG2E, LOG2E);
    if (PP && w == 1) named_bar_arrive(3, 256);                          // warpgroup 0 takes the first exp turn
    for (int G = w, li = 0, n = w; G < nblk; G += 2, n += 2) {
      if (n >= nb) { n -= nb; ++li; }
      const int s = G % ST;
      const uint32_t st = su + O_ST + s * STB, sb = st + 2 * TQ, sl = sb + TBT;
      if (w == 0 && r == 0) TR(5, G);
      mbar_wait(&B.full[s], (G / ST) & 1);                                 // bias^T, LSE, D of this block
      if (w == 0 && r == 0) TR(6, G);
      mbar_wait(&B.s_full[w], (G >> 1) & 1);
      if (w == 0 && r == 0) TR(3, G);
      tc_fence_after();
      uint32_t pp0[16], pd0[16];
      if (PP) named_bar_sync(3 + w, 256);                                  // my exp turn
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {                                     // 32 queries per half
        uint32_t sv[32], dv[32];
        tmem_ld32(trow + T_S + w * 128 + cc * 32, sv);
        tmem_ld32(trow + T_S + w * 128 + 64 + cc * 32, dv);
        tmem_wait_ld();
        if (cc == 1) {
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.s_free[w]);
          if (G >= 2) mbar_wait(&B.ds_free[w], ((G >> 1) - 1) & 1);        // dV / dK of this warpgroup's previous block are issued and done
          tc_fence_after();
        }
        uint32_t pp[16], pd[16];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 bw = lds128(sb + sw128(r, cc * 4 + q));
          const uint32_t bb[4] = {bw.x, bw.y, bw.z, bw.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int j = q * 4 + e, col = cc * 32 + 2 * j;
            const float2 lse = lds64f(sl + col * 4), dd = lds64f(sl + 512 + col * 4);
            const f2 x = fma2(mk2u(sv[2 * j], sv[2 * j + 1]), CQL, fma2(mk2(bf16lo(bb[e]), bf16hi(bb[e])), L2E, mk2(-lse.x, -lse.y)));
            const f2 p = (((j) % PMOD) < PCNT) ? ex2_poly2(x) : mk2(ex2f(lo2(x)), ex2f(hi2(x)));
            const f2 ds = mul2(p, add2(mk2u(dv[2 * j], dv[2 * j + 1]), mk2(-dd.x, -dd.y)));
            pp[j] = pack_bf16(lo2(p), hi2(p));
            pd[j] = pack_bf16(lo2(ds), hi2(ds));
          }
        }
        if (cc == 0) {
#pragma unroll
          for (int k = 0; k < 16; ++k) { pp0[k] = pp[k]; pd0[k] = pd[k]; }
        } else {
          tmem_st16(trow + T_P + w * 64, pp0);
          tmem_st16(trow + T_P + w * 64 + 16, pp);
          tmem_st16(trow + T_P + w * 64 + 32, pd0);
          tmem_st16(trow + T_P + w * 64 + 48, pd);
        }
      }
      if (PP) named_bar_arrive(4 - w, 256);                                // hand the MUFU over
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.ds_full[w]);
      if (w == 0 && r == 0) TR(4, G);
      if (n >= nb - 2) {                                                   // this warpgroup's last block of the item
        // ---- the item's epilogue: warpgroup 0 writes dK (scaled by 1 / sqrt 48), warpgroup 1 dV, one key row per thread
        int a, k0, head; item_of(li, a, k0, head);
        mbar_wait(&B.acc_full, li & 1);
        tc_fence_after();
        const float sc = w == 0 ? RSQD : 1.f;
        float* orow = (w == 0 ? DK : DV) + ((size_t)a * L + k0 + r) * DM + head * DH;
        const uint32_t tc = w == 0 ? T_DK : T_DV;
        uint32_t v[32], u[DH == 64 ? 32 : 16];
        if (DH == 16) tmem_ld16(trow + tc, *reinterpret_cast<uint32_t(*)[16]>(v));
        else tmem_ld32(trow + tc, v);
        if (DH == 64) tmem_ld32(trow + tc + 32, *reinterpret_cast<uint32_t(*)[32]>(u));
        else if (DH > 32) tmem_ld16(trow + tc + 32, *reinterpret_cast<uint32_t(*)[16]>(u));
        tmem_wait_ld();
        {
          const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
          for (int c = 0; c < DH / 16; ++c) tmem_st16(trow + tc + 16 * c, z);
          tmem_wait_st();
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_free);
        if (w == 0 && r == 0) TR(7, G);
        if (DQZ) {                                                         // zero this tile of dQ for attn_dqb's reductions (it runs next):
          float4* zq = reinterpret_cast<float4*>(DQZ + ((size_t)a * L + k0) * DM + head * DH);   // 128 rows x DH / 4 float4, lanes along rows
#pragma unroll
          for (int k = 0; k < DH / 8; ++k) {
            const int idx = w * 16 * DH + k * 128 + (int)r, row = idx / (DH / 4), ch = idx % (DH / 4);
            zq[(size_t)row * (DM / 4) + ch] = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        auto sv = [&](int k) {                                             // channels k .. k + 3, scaled
          const uint32_t* src = k < 32 ? v + k : u + (k - 32);
          return make_uint4(__float_as_uint(__uint_as_float(src[0]) * sc), __float_as_uint(__uint_as_float(src[1]) * sc),
                            __float_as_uint(__uint_as_float(src[2]) * sc), __float_as_uint(__uint_as_float(src[3]) * sc));
        };
        if (TST && DH == 64) {                                             // 64 columns: two 32-column halves through xa, in turn
          const uint32_t xa = su + O_X + w * (XA + XB);
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            if (r == 0) tma_store_wait_read0();
            named_bar_sync(1 + w, 128);
#pragma unroll
            for (int q = 0; q < 8; ++q) sts128(xa + sw128(r, q), sv(32 * hf + 4 * q));
            fence_proxy_async();
            named_bar_sync(1 + w, 128);
            if (r == 0) {
              tma_store_2d(w == 0 ? &mka : &mva, xa, head * DH + 32 * hf, a * L + k0);
              tma_store_commit();
            }
          }
        } else if (TST) {
          const uint32_t xa = su + O_X + w * (XA + XB), xb = xa + XA;
          if (r == 0) tma_store_wait_read0();                              // the previous item's store has left the staging buffer
          named_bar_sync(1 + w, 128);
#pragma unroll
          for (int q = 0; q < (DH == 16 ? 4 : 8); ++q) {                  // DH 16: the 64-B box alone
            if (DH == 16) sts128(xb + sw64(r, q), sv(4 * q)); else sts128(xa + sw128(r, q), sv(4 * q));
          }
#pragma unroll
          for (int q = 0; q < (DH == 48 ? 4 : 0); ++q) sts128(xb + sw64(r, q), sv(32 + 4 * q));
          fence_proxy_async();
          named_bar_sync(1 + w, 128);
          if (r == 0) {
            if (DH == 16) tma_store_2d(w == 0 ? &mkb : &mvb, xb, head * DH, a * L + k0);
            else tma_store_2d(w == 0 ? &mka : &mva, xa, head * DH, a * L + k0);
            if (DH == 48) tma_store_2d(w == 0 ? &mkb : &mvb, xb, head * DH + 32, a * L + k0);
            tma_store_commit();
          }
        } else {
#pragma unroll
        for (int k = 0; k < (DH == 16 ? 4 : 8); ++k)
          *reinterpret_cast<float4*>(orow + 4 * k) = make_float4(__uint_as_float(v[4 * k]) * sc, __uint_as_float(v[4 * k + 1]) * sc,
                                                                 __uint_as_float(v[4 * k + 2]) * sc, __uint_as_float(v[4 * k + 3]) * sc);
#pragma unroll
        for (int k = 0; k < (DH > 32 ? (DH - 32) / 4 : 0); ++k)
          *reinterpret_cast<float4*>(orow + 32 + 4 * k) = make_float4(__uint_as_float(u[4 * k]) * sc, __uint_as_float(u[4 * k + 1]) * sc,
                                                                      __uint_as_float(u[4 * k + 2]) * sc, __uint_as_float(u[4 * k + 3]) * sc);
        }
      }
    }
    if (TST && r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
