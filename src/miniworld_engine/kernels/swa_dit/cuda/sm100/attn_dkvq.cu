// attn_dkvq.cu — attn_dkv.cu with dQ fused in (the whole attention backward in one pass; the dq kernel's recomputation of S / dP and
// its exponentials go away). dS^T is written once to shared memory ([128 keys][64 queries] bf16, 128-B rows, SW128) and serves as the
// K-major A of dK += dS^T q (M = 128 keys) and as the MN-major A of dQ_b = dS K (M = 64 queries, N = 32, K = 128 keys; K from the item
// stage as MN-major B). dQ_b takes the 32 TMEM columns dS^T had (M = 64: row 16 g + i at lane 32 g + i); the row threads read it out
// after handing the next block's P / dS to the MMA warp.
// A 64-query block b gets dQ from exactly two key tiles (m, m + 1 with b in {2m + 1, 2m + 2}; one at the sequence ends). Each CTA
// walks a contiguous run of (sample, head pair, key tile) items, so the two usually meet in the same CTA: the first tile's dQ_b waits
// in shared memory (fp32 carry), the second adds its own (carry + dQ_b, fp32) and writes dq = rn(scale sum) (bf16, from registers). Where a
// run ends / starts inside a sequence the two partial blocks are in neighbouring CTAs: each writes its fp32 partial to a global
// slot and bumps a counter; the second to arrive adds the two (a + b = b + a: deterministic), writes dq and resets the counter.
// Block stages 6 -> 3 for the shared memory.
// attn_dkv.cu — the SWA atom block's window-attention backward dK / dV pass on sm_100a (replaces the Triton _attn_bwd_dkv; same math
// and rounding points): keys j < seqused[n], queries i with |i - j| <= 64 and i < seqused[n]
//   S^T = K q^T, dP^T = V dO^T (TMEM, fp32);  P^T = exp(S^T scale - LSE);  dS^T = rn(P^T (dP^T - D))
//   dV = rn(sum_blocks rn(P^T) dO);  dK = rn(scale sum_blocks dS^T q)
// Transposed: persistent CTAs walk (sample, 128-key tile, head pair) items, one key row per thread, the two compute warpgroups take the
// two heads (one MMA warp each); the window's <= 4 64-query blocks [j0 - 64, j0 + 192) stream through a TMA ring together with their
// LSE / D. Per block: S^T / dP^T (SS MMAs) -> the row threads read both and release them (the next block's S^T / dP^T overlap the
// math) -> P^T, dS^T to TMEM (bf16) -> dV += P^T dO, dK += dS^T q (TS MMAs, dO / q as the MN-major B). dK / dV leave through TMA stores
// (head-major [N, H, S, 32]). All tiles are dense 64-B rows, 64-B swizzled. A warp skips 8-column chunks outside its rows' windows.
// TMEM: S^T[w] at w * 128, dP^T[w] at w * 128 + 64, P^T[w] at 256 + w * 64 (bf16), dQ_b[w] at 288 + w * 64 (M = 64), dK[w] at 384 + w * 64,
// dV[w] at 416 + w * 64.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;
template <int IMM> DEVI void tmem_ld16x2_16(uint32_t taddr, uint32_t (&r)[16]) {   // threads 0-15: lanes + t, columns from taddr; 16-31: + IMM
  asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16], %17;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
                 "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr), "n"(IMM));
}

#ifndef ST
#define ST 3
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
constexpr int O_DS = O_X + 4 * XS, O_CAR = O_DS + 2 * 16384;               // dS^T (2 heads), fp32 carries (2 x 2 slots x [64][32] SW128)
constexpr int O_BAR = O_CAR + 4 * 8192,
                 O_ITAB = O_BAR + 512, NIT = 128, SMEM_BYTES = O_ITAB + NIT * 16;   // + item table
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(BLS % 512 == 0, "SW64 stage alignment");
static_assert(O_DS % 1024 == 0 && O_CAR % 1024 == 0, "swizzle alignment");
constexpr uint32_t T_S = 0, T_P = 256, T_ACC = 384;
constexpr uint32_t I_S = idesc_bf16(128, BQ), I_KV = idesc_bf16(128, DH, 0, 1), I_DQ = idesc_bf16(64, DH, 1, 1);
constexpr float LOG2E = 1.4426950408889634f;

struct Bars {
  uint64_t it_full[QR], it_empty[QR], bl_full[ST], bl_empty[ST], s_full[2], s_free[2], p_full[2], p_free[2], acc_full[2], acc_free[2], dq_free[2];
  uint32_t tmem;
  int qflag[2];
};

extern "C" __global__ void __launch_bounds__(384, 1)
swa_attn_dkvq_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                   const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mlse, const __grid_constant__ CUtensorMap mdv,
                   const __grid_constant__ CUtensorMap mdk_o, const __grid_constant__ CUtensorMap mdv_o,
                   const int* __restrict__ SEQU, float* __restrict__ BND, int* __restrict__ CNT, uint16_t* __restrict__ DQ,
                   int S, int N, float scale) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = S / KM;
  const int items = N * mt * 2;
  // contiguous runs of items wi = (2 n + hp) mt + m (key tile m of sample n, head pair hp)
  const int per = items / (int)gridDim.x, rem = items % (int)gridDim.x;
  const int my_items = per + ((int)blockIdx.x < rem ? 1 : 0), first = (int)blockIdx.x * per + min((int)blockIdx.x, rem);
  int4* itab = reinterpret_cast<int4*>(sm + O_ITAB);                      // (n, 2 j0 + hp, 65536 lo + hi, sq) of item li < NIT
  auto item_raw = [&](int li, int& n, int& j0, int& hp, int& lo, int& hi, int& sq) {
    const int wi = first + li, sqi = wi / mt, m = wi % mt;
    hp = sqi & 1; n = sqi >> 1; j0 = m * KM;
    sq = __ldg(SEQU + n);
    lo = max(2 * m - 1, 0); hi = min(2 * m + 3, S / BQ);                  // all query blocks of the window (the dQ pairing counts on it)
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
      mbar_init(&B.acc_full[w], 1); mbar_init(&B.acc_free[w], 4); mbar_init(&B.dq_free[w], 4);
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
        const uint32_t sds = su + O_DS + w * 16384;
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ts(tmem + T_ACC + w * 64 + 32, tmem + T_P + w * 64 + ks * 8, ddo + (uint64_t)(ks * 1024 >> 4), I_KV, (qb > lo || ks > 0) ? 1u : 0u);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)                                   // dS^T from shared memory, K-major (16 queries = 32 B a step)
            umma_ss(tmem + T_ACC + w * 64, desc_k128(sds) + (uint64_t)(ks * 2), dq + (uint64_t)(ks * 1024 >> 4), I_KV, (qb > lo || ks > 0) ? 1u : 0u);
        }
        __syncwarp();
        if (G >= 1) mbar_wait(&B.dq_free[w], (G - 1) & 1);                // dQ_b of G - 1 has been read out
        tc_fence_after();
        if (elect_one()) {
          const uint64_t dk = desc_sw64(su + O_IT + (li % QR) * IST + w * TKV);                  // K as MN-major B (16 keys = 1 KB a step)
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ss(tmem + T_P + w * 64 + 32, desc_mn128(sds, 0) + (uint64_t)(ks * 128), dk + (uint64_t)(ks * 64), I_DQ, ks > 0 ? 1u : 0u);
          tc_commit(&B.p_free[w]);                                         // P^T, dS^T read and dQ_b complete
          tc_commit(&B.bl_empty[s]);
          if (qb == hi - 1) { tc_commit(&B.acc_full[w]); tc_commit(&B.it_empty[li % QR]); }
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
    // dQ_b of the previous block: TMEM (M = 64 layout; this thread: query row qr = 16 (warp % 4) + lane % 16, columns 16 (lane / 16) ..)
    //   Q_FINAL: the only tile of the block -> dq;  Q_COUT: first of two, the next item here adds -> carry slot;  Q_CIN: carry + dQ_b -> dq;
    //   Q_BS / Q_BE: first / last item of this CTA's run, the other tile in CTA - 1 / + 1 -> global slot + counter
    enum { Q_FINAL = 0, Q_CIN = 1, Q_COUT = 2, Q_BS = 3, Q_BE = 4 };
    const uint32_t sds = su + O_DS + w * 16384, car = su + O_CAR + w * 16384;
    const uint32_t qr = 16 * (warp & 3) + (lane & 15), qh = lane >> 4;
    int q_row = 0, q_mode = 0, q_slot = 0;
    auto bnd = [&](int cta, int side, int slot) { return BND + ((((size_t)cta * 2 + side) * 2 + w) * 2 + slot) * (BQ * DH) + qr * DH + 16 * qh; };
    auto dq_out = [&](int row0, int mode, int slot) {
      tc_fence_after();
      uint32_t qv[16];
      tmem_ld16x2_16<16>(tmem + (lb << 16) + T_P + w * 64 + 32, qv);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dq_free[w]);
      const uint32_t ca = car + slot * 8192;
      if (mode == Q_COUT) {                                                // thread-private carry (read back by this thread)
#pragma unroll
        for (int c = 0; c < 4; ++c) sts128(ca + sw128(qr, 4 * qh + c), make_uint4(qv[4 * c], qv[4 * c + 1], qv[4 * c + 2], qv[4 * c + 3]));
        return;
      }
      // (sums kept in qv as fp32 bits, in place: fewer live registers)
      auto addq = [&](int k, float x) { qv[k] = __float_as_uint(x + __uint_as_float(qv[k])); };
      if (mode == Q_CIN) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint4 u = lds128(ca + sw128(qr, 4 * qh + c));
          addq(4 * c, __uint_as_float(u.x)); addq(4 * c + 1, __uint_as_float(u.y)); addq(4 * c + 2, __uint_as_float(u.z)); addq(4 * c + 3, __uint_as_float(u.w));
        }
      }
      if (mode == Q_BS || mode == Q_BE) {
        const int side = mode == Q_BE, cs = side ? (int)blockIdx.x + 1 : (int)blockIdx.x;   // the counter belongs to the run-start CTA
        uint4* mine = reinterpret_cast<uint4*>(bnd(blockIdx.x, side, slot));
#pragma unroll
        for (int c = 0; c < 4; ++c) mine[c] = make_uint4(qv[4 * c], qv[4 * c + 1], qv[4 * c + 2], qv[4 * c + 3]);
        __threadfence();
        named_bar_sync(1 + w, 128);
        int* cnt = CNT + (cs * 2 + w) * 2 + slot;
        if (r == 64) B.qflag[w] = atomicAdd(cnt, 1);
        named_bar_sync(1 + w, 128);
        if (B.qflag[w] == 0) return;                                       // first to arrive: the neighbour finishes the block
        __threadfence();
        const float4* oth = reinterpret_cast<const float4*>(bnd(side ? (int)blockIdx.x + 1 : (int)blockIdx.x - 1, 1 - side, slot));
#pragma unroll
        for (int c = 0; c < 4; ++c) {                                      // own + other (a + b = b + a)
          const float4 o = __ldcg(oth + c);
          addq(4 * c, o.x); addq(4 * c + 1, o.y); addq(4 * c + 2, o.z); addq(4 * c + 3, o.w);
        }
        if (r == 64) *cnt = 0;                                             // ready for the next launch
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) qv[k] = pack_bf16(__uint_as_float(qv[2 * k]) * scale, __uint_as_float(qv[2 * k + 1]) * scale);
      uint4* dst = reinterpret_cast<uint4*>(DQ + (size_t)(row0 + (int)qr) * DH + 16 * qh);    // a warp: 16 whole 64-B rows, contiguous
      dst[0] = make_uint4(qv[0], qv[1], qv[2], qv[3]); dst[1] = make_uint4(qv[4], qv[5], qv[6], qv[7]);
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
        if (G >= 1) mbar_wait(&B.p_free[w], (G - 1) & 1);                  // the MMAs of G - 1 are done (P^T / dS^T read, dQ_b complete)
        tc_fence_after();
        tmem_st16(trow + T_P + w * 64, *reinterpret_cast<uint32_t(*)[16]>(pp));
        tmem_st16(trow + T_P + w * 64 + 16, *reinterpret_cast<uint32_t(*)[16]>(pp + 16));
#pragma unroll
        for (int c = 0; c < 8; ++c) sts128(sds + sw128(r, c), make_uint4(ds[4 * c], ds[4 * c + 1], ds[4 * c + 2], ds[4 * c + 3]));
        fence_proxy_async();
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.p_full[w]);
        if (G >= 1) dq_out(q_row, q_mode, q_slot);                         // dQ_b of G - 1
        {
          const int m = j0 / KM;
          q_row = (n * H + h) * S + i0;
          if (qb <= 2 * m) { q_slot = qb - (2 * m - 1); q_mode = m == 0 ? Q_FINAL : (li > 0 ? Q_CIN : Q_BS); }
          else { q_slot = qb - (2 * m + 1); q_mode = m + 1 >= mt ? Q_FINAL : (li + 1 < my_items ? Q_COUT : Q_BE); }
        }
      }
      pend = li; pn = n; pj0 = j0; ph = h;
    }
    if (pend >= 0) epilogue(pend, pn, pj0, ph);
    if (G >= 1) { mbar_wait(&B.p_free[w], (G - 1) & 1); dq_out(q_row, q_mode, q_slot); }
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
