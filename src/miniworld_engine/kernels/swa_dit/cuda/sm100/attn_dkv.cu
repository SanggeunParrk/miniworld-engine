// attn_dkv.cu — the SWA atom block's window-attention backward dK / dV pass on sm_100a (replaces the Triton _attn_bwd_dkv; same math
// and rounding points): keys j < seqused[n], queries i with |i - j| <= 64 and i < seqused[n]
//   S^T = K q^T, dP^T = V dO^T (TMEM, fp32);  P^T = exp(S^T scale - LSE);  dS^T = rn(P^T (dP^T - D))
//   dV = rn(sum_blocks rn(P^T) dO);  dK = rn(scale sum_blocks dS^T q)
// Transposed: persistent CTAs walk (sample, 128-key tile, head pair) items, one key row per thread, the two compute warpgroups take the
// two heads (one MMA warp each); the window's <= 4 64-query blocks [j0 - 64, j0 + 192) stream through a TMA ring together with their
// LSE / D. Per block: S^T / dP^T (SS MMAs) -> the row threads read both and release them (the next block's S^T / dP^T overlap the
// math) -> P^T, dS^T to TMEM (bf16) -> dV += P^T dO, dK += dS^T q (TS MMAs, dO / q as the MN-major B). dK / dV leave through TMA stores
// (head-major [N, H, S, 32]). All tiles are dense 64-B rows, 64-B swizzled. A warp skips 8-column chunks outside its rows' windows.
// TMEM: S^T[w] at w * 128, dP^T[w] at w * 128 + 64, P^T[w] at 256 + w * 64, dS^T[w] at 288 + w * 64 (bf16), dK[w] at 384 + w * 64,
// dV[w] at 416 + w * 64.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef ST
#define ST 6
#endif
#ifndef QR
#define QR 2
#endif
constexpr int BQ = 64, DH = 32, KM = 128, H = 4, HW = 64;
constexpr int TKV = KM * 64, TQ = BQ * 64;                                 // K / V tile 8 KB, q / dO block 4 KB
constexpr int IST = 4 * TKV;                                               // K0 | K1 | V0 | V1 = 32 KB
constexpr int BLS = 4 * TQ + 1024;                                         // q0 | q1 | dO0 | dO1 | LSE0 | LSE1 | D0 | D1
constexpr int O_IT = 0, O_BL = QR * IST, O_X = O_BL + ST * BLS;            // then dK / dV staging (2 heads x 2 x [128][64 B], SW64), bars
constexpr int XS = 128 * 64;
constexpr int O_BAR = O_X + 4 * XS, O_ITAB = O_BAR + 512, NIT = 128, SMEM_BYTES = O_ITAB + NIT * 16;   // + item table
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(BLS % 512 == 0, "SW64 stage alignment");
constexpr uint32_t T_S = 0, T_P = 256, T_ACC = 384;
constexpr uint32_t I_S = idesc_bf16(128, BQ), I_KV = idesc_bf16(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f;

struct Bars {
  uint64_t it_full[QR], it_empty[QR], bl_full[ST], bl_empty[ST], s_full[2], s_free[2], p_full[2], p_free[2], acc_full[2], acc_free[2];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(384, 1)
swa_attn_dkv_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                   const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mlse, const __grid_constant__ CUtensorMap mdv,
                   const __grid_constant__ CUtensorMap mdk_o, const __grid_constant__ CUtensorMap mdv_o, const int* __restrict__ SEQU,
                   int S, int N, float scale) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = S / KM;
  const int items = N * mt * 2;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  int4* itab = reinterpret_cast<int4*>(sm + O_ITAB);                      // (n, 2 j0 + hp, 65536 lo + hi, sq) of item li < NIT
  auto item_raw = [&](int li, int& n, int& j0, int& hp, int& lo, int& hi, int& sq) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    hp = wi & 1; const int r = wi >> 1;
    j0 = (r % mt) * KM; n = r / mt;
    sq = __ldg(SEQU + n);
    lo = max(j0 - HW, 0) / BQ;
    const int qend = min(sq, j0 + KM + HW);
    hi = (qend + BQ - 1) / BQ;
    if (hi <= lo) hi = lo + 1;                                            // an item with no valid query still runs one (masked) block
  };
  auto item_of = [&](int li, int& n, int& j0, int& hp, int& lo, int& hi, int& sq) {
    if (li >= NIT) { item_raw(li, n, j0, hp, lo, hi, sq); return; }
    const int4 e = itab[li];
    n = e.x; j0 = e.y >> 1; hp = e.y & 1; lo = e.z >> 16; hi = e.z & 0xffff; sq = e.w;
  };
  if (tid < my_items && tid < NIT) {
    int n, j0, hp, lo, hi, sq; item_raw(tid, n, j0, hp, lo, hi, sq);
    itab[tid] = make_int4(n, 2 * j0 + hp, 65536 * lo + hi, sq);
  }

  if (tid == 0) {
    for (int s = 0; s < QR; ++s) { mbar_init(&B.it_full[s], 1); mbar_init(&B.it_empty[s], 2); }
    for (int s = 0; s < ST; ++s) { mbar_init(&B.bl_full[s], 1); mbar_init(&B.bl_empty[s], 2); }
    for (int w = 0; w < 2; ++w) {
      mbar_init(&B.s_full[w], 1); mbar_init(&B.s_free[w], 4); mbar_init(&B.p_full[w], 4); mbar_init(&B.p_free[w], 1);
      mbar_init(&B.acc_full[w], 1); mbar_init(&B.acc_free[w], 4);
    }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        int n, j0, hp, lo, hi, sq; item_of(li, n, j0, hp, lo, hi, sq);
        const int is = li % QR;
        if (li >= QR) mbar_wait(&B.it_empty[is], ((li / QR) - 1) & 1);
        mbar_expect_tx(&B.it_full[is], 4 * KM * DH * 2);
        for (int w = 0; w < 2; ++w) {
          const int row = (n * H + 2 * hp + w) * S + j0;
          tma_load_2d(su + O_IT + is * IST + w * TKV, &mk, &B.it_full[is], 0, row);
          tma_load_2d(su + O_IT + is * IST + (2 + w) * TKV, &mv, &B.it_full[is], 0, row);
        }
        for (int qb = lo; qb < hi; ++qb, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.bl_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_BL + s * BLS;
          mbar_expect_tx(&B.bl_full[s], 4 * BQ * DH * 2 + 4 * BQ * 4);
          for (int w = 0; w < 2; ++w) {
            const int h = 2 * hp + w, row = (n * H + h) * S + qb * BQ;
            tma_load_2d(st + w * TQ, &mq, &B.bl_full[s], 0, row);
            tma_load_2d(st + (2 + w) * TQ, &mdo, &B.bl_full[s], h * DH, n * S + qb * BQ);
            tma_load_2d(st + 4 * TQ + w * 256, &mlse, &B.bl_full[s], 0, row / BQ);
            tma_load_2d(st + 4 * TQ + 512 + w * 256, &mdv, &B.bl_full[s], 0, row / BQ);
          }
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ MMA issuers (warp 1 + w: head 2 hp + w)
    const int w = warp - 1;
    const uint32_t tS = tmem + T_S + w * 128, tDP = tS + 64;
    auto issue_s = [&](int G, int li, int qb, int lo, int hi) {
      const int s = G % ST, is = li % QR;
      if (qb == lo) mbar_wait(&B.it_full[is], (li / QR) & 1);
      mbar_wait(&B.bl_full[s], (G / ST) & 1);
      if (G >= 1) mbar_wait(&B.s_free[w], (G - 1) & 1);                   // the row threads have read S^T / dP^T of G - 1
      tc_fence_after();
      const uint64_t dk = desc_sw64(su + O_IT + is * IST + w * TKV), dv = desc_sw64(su + O_IT + is * IST + (2 + w) * TKV);
      const uint64_t dq = desc_sw64(su + O_BL + s * BLS + w * TQ), ddo = desc_sw64(su + O_BL + s * BLS + (2 + w) * TQ);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks) umma_ss(tS, dk + (uint64_t)(ks * 2), dq + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks) umma_ss(tDP, dv + (uint64_t)(ks * 2), ddo + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w]);
        if (qb == hi - 1) tc_commit(&B.it_empty[is]);
      }
      __syncwarp();
    };
    int nli = 0, nqb = 0, nlo = 0, nhi = 0;
    auto adv = [&]() {
      if (++nqb == nhi) { if (++nli < my_items) { int n, j0, hp, sq; item_of(nli, n, j0, hp, nlo, nhi, sq); nqb = nlo; } }
    };
    if (my_items > 0) {
      int n, j0, hp, sq; item_of(0, n, j0, hp, nlo, nhi, sq); nqb = nlo;
      issue_s(0, 0, nqb, nlo, nhi);
      adv();
    }
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      int n, j0, hp, lo, hi, sq; item_of(li, n, j0, hp, lo, hi, sq);
      for (int qb = lo; qb < hi; ++qb, ++G) {
        const int s = G % ST;
        if (nli < my_items) { issue_s(G + 1, nli, nqb, nlo, nhi); adv(); }   // S^T / dP^T(G + 1) first
        mbar_wait(&B.p_full[w], G & 1);
        if (qb == lo && li >= 1) mbar_wait(&B.acc_free[w], (li - 1) & 1);  // the previous item's dK / dV have been read out
        tc_fence_after();
        const uint64_t ddo = desc_sw64(su + O_BL + s * BLS + (2 + w) * TQ), dq = desc_sw64(su + O_BL + s * BLS + w * TQ);   // MN-major
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ts(tmem + T_ACC + w * 64 + 32, tmem + T_P + w * 64 + ks * 8, ddo + (uint64_t)(ks * 1024 >> 4), I_KV, (qb > lo || ks > 0) ? 1u : 0u);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ts(tmem + T_ACC + w * 64, tmem + T_P + w * 64 + 32 + ks * 8, dq + (uint64_t)(ks * 1024 >> 4), I_KV, (qb > lo || ks > 0) ? 1u : 0u);
          tc_commit(&B.p_free[w]);
          tc_commit(&B.bl_empty[s]);
          if (qb == hi - 1) tc_commit(&B.acc_full[w]);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ P^T / dS^T of head 2 hp + w, one key row per thread
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const float SC = scale * LOG2E;
    int G = 0;
    // dK / dV of item pli -> SW64 staging -> TMA stores; run inside the NEXT item's first block (after its P / dS math, before they go to
    // TMEM), so the last dV / dK MMA of an item is not waited for
    int pend = -1, pn = 0, pj0 = 0, ph = 0;
    auto epilogue = [&](int pli, int n, int j0, int h) {
      mbar_wait(&B.acc_full[w], pli & 1);
      tc_fence_after();
      uint32_t kv[32], vv[32];
      tmem_ld32(trow + T_ACC + w * 64, kv);
      tmem_ld32(trow + T_ACC + w * 64 + 32, vv);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.acc_free[w]);
      const uint32_t xk = su + O_X + (2 * w) * XS, xv = xk + XS;
      if (r == 0) tma_store_wait_read0();
      named_bar_sync(1 + w, 128);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint4 a, b;
        a.x = pack_bf16(__uint_as_float(kv[8 * q]) * scale, __uint_as_float(kv[8 * q + 1]) * scale);
        a.y = pack_bf16(__uint_as_float(kv[8 * q + 2]) * scale, __uint_as_float(kv[8 * q + 3]) * scale);
        a.z = pack_bf16(__uint_as_float(kv[8 * q + 4]) * scale, __uint_as_float(kv[8 * q + 5]) * scale);
        a.w = pack_bf16(__uint_as_float(kv[8 * q + 6]) * scale, __uint_as_float(kv[8 * q + 7]) * scale);
        b.x = pack_bf16(__uint_as_float(vv[8 * q]), __uint_as_float(vv[8 * q + 1]));
        b.y = pack_bf16(__uint_as_float(vv[8 * q + 2]), __uint_as_float(vv[8 * q + 3]));
        b.z = pack_bf16(__uint_as_float(vv[8 * q + 4]), __uint_as_float(vv[8 * q + 5]));
        b.w = pack_bf16(__uint_as_float(vv[8 * q + 6]), __uint_as_float(vv[8 * q + 7]));
        sts128(xk + sw64(r, q), a);
        sts128(xv + sw64(r, q), b);
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (r == 0) {
        tma_store_2d(&mdk_o, xk, 0, (n * H + h) * S + j0);
        tma_store_2d(&mdv_o, xv, 0, (n * H + h) * S + j0);
        tma_store_commit();
      }
    };
    for (int li = 0; li < my_items; ++li) {
      int n, j0, hp, lo, hi, sq; item_of(li, n, j0, hp, lo, hi, sq);
      const int j = j0 + (int)r, h = 2 * hp + w;
      const bool kok = j < sq;
      for (int qb = lo; qb < hi; ++qb, ++G) {
        const int s = G % ST;
        mbar_wait(&B.s_full[w], G & 1);
        tc_fence_after();
        const int i0 = qb * BQ;
        const int clo = max(j - HW - i0, 0), chi = kok ? min(min(j + HW, sq - 1) - i0, BQ - 1) : -1;
        const int wlo = __reduce_min_sync(~0u, chi >= clo ? clo : BQ), whi = __reduce_max_sync(~0u, chi >= clo ? chi : -1);
        const float* lse = reinterpret_cast<const float*>(sm + O_BL + s * BLS + 4 * TQ + w * 256);
        const float* dvv = reinterpret_cast<const float*>(sm + O_BL + s * BLS + 4 * TQ + 512 + w * 256);
        uint32_t pp[32], ds[32];
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          uint32_t sv[32], pv[32];
          tmem_ld32(trow + T_S + w * 128 + 32 * hf, sv);
          tmem_ld32(trow + T_S + w * 128 + 64 + 32 * hf, pv);
          tmem_wait_ld();
          if (hf == 1) {
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(&B.s_free[w]);
          }
#pragma unroll
          for (int cq = 0; cq < 4; ++cq) {
            const int ch = 4 * hf + cq;
            if (ch * 8 + 7 < wlo || ch * 8 > whi) {
#pragma unroll
              for (int k = 0; k < 4; ++k) { pp[4 * ch + k] = 0u; ds[4 * ch + k] = 0u; }
              continue;
            }
            const float4 l0 = *reinterpret_cast<const float4*>(lse + 8 * ch), l1 = *reinterpret_cast<const float4*>(lse + 8 * ch + 4);
            const float4 d0 = *reinterpret_cast<const float4*>(dvv + 8 * ch), d1 = *reinterpret_cast<const float4*>(dvv + 8 * ch + 4);
            const float lv[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w}, dd[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              float p2[2], d2[2];
#pragma unroll
              for (int e = 0; e < 2; ++e) {
                const int cl = 8 * cq + 2 * k + e, c = 32 * hf + cl, x = 2 * k + e;
                const float p = ex2f((c >= clo && c <= chi) ? __uint_as_float(sv[cl]) * SC - lv[x] * LOG2E : -INFINITY);
                p2[e] = p;
                d2[e] = p * (__uint_as_float(pv[cl]) - dd[x]);
              }
              pp[4 * ch + k] = pack_bf16(p2[0], p2[1]);
              ds[4 * ch + k] = pack_bf16(d2[0], d2[1]);
            }
          }
        }
        if (qb == lo && pend >= 0) { epilogue(pend, pn, pj0, ph); pend = -1; }
        if (G >= 1) mbar_wait(&B.p_free[w], (G - 1) & 1);                  // dV / dK MMAs of G - 1 have read P^T / dS^T
        tc_fence_after();
        tmem_st16(trow + T_P + w * 64, *reinterpret_cast<uint32_t(*)[16]>(pp));
        tmem_st16(trow + T_P + w * 64 + 16, *reinterpret_cast<uint32_t(*)[16]>(pp + 16));
        tmem_st16(trow + T_P + w * 64 + 32, *reinterpret_cast<uint32_t(*)[16]>(ds));
        tmem_st16(trow + T_P + w * 64 + 48, *reinterpret_cast<uint32_t(*)[16]>(ds + 16));
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.p_full[w]);
      }
      pend = li; pn = n; pj0 = j0; ph = h;
    }
    if (pend >= 0) epilogue(pend, pn, pj0, ph);
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
