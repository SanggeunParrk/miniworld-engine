// attn_dq.cu — the SWA atom block's window-attention backward dQ pass on sm_100a (replaces the Triton _attn_bwd_dq; same math and
// rounding points): keys j with |i - j| <= 64 and j < seqused[n], queries i < seqused[n]
//   S = q K^T, dP = dO V^T (TMEM, fp32);  P = exp(S scale - LSE);  dS = rn(P (dP - D));  dQ = rn(scale sum_blocks dS K)
// Structure of attn_fwd.cu: persistent CTAs walk (sample, 128-query tile, head pair) items; the two compute warpgroups take the two heads
// (one MMA warp each); the window's <= 4 64-key blocks [i0 - 64, i0 + 192) per item come through a TMA ring. Per block: S / dP (SS MMAs)
// -> the row threads read both into registers and release them at once (so S / dP of the next block overlap the math), dS -> TMEM
// (bf16, double-buffered) -> dQ += dS K (TS MMA, K as the MN-major B). dQ leaves through a TMA store (head-major [N, H, S, 32]).
// q / dO / K / V tiles are dense 64-B rows, 64-B swizzled (SW64 K-major for S / dP, SW64 MN-major K for dQ); dO is row-major [M, C]
// (head h = columns 32 h ..). A warp skips the exponentials of 8-column chunks outside its rows' windows.
// TMEM: S[w] at w * 128, dP[w] at w * 128 + 64, dS[w][b] at 256 + w * 64 + b * 32 (bf16), dQ[w] at 384 + w * 32.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;
#ifndef TRACE
#define TRACE 0
#endif
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define EV(k) do { if (TRACE && blockIdx.x == 0 && threadIdx.x == 128 && li < 3) TRb[16 * li + (k)] = gtime(); } while (0)

#ifndef ST
#define ST 6
#endif
#ifndef SETMAXNREG
#define SETMAXNREG 0                    // the kernel compiles to <= 168 registers anyway; dec<56> would cut live control-warp state
#endif
#ifndef NOSKIP
#define NOSKIP 0
#endif
#ifndef QR
#define QR 2
#endif
constexpr int BN = 64, DH = 32, QM = 128, H = 4, HW = 64;
constexpr int TQ = QM * 64, TK = BN * 64;
constexpr int QST = 4 * TQ;                                                // q0 | q1 | dO0 | dO1 = 32 KB
constexpr int STB = 4 * TK;                                                // K0 | K1 | V0 | V1 = 16 KB
constexpr int O_Q = 0, O_ST = QR * QST, O_X = O_ST + ST * STB;             // then dQ staging (2 x [128][64 B], SW64), bars
constexpr int XS = 128 * 64;
constexpr int O_BAR = O_X + 2 * XS, O_IT = O_BAR + 512, NIT = 128, SMEM_BYTES = O_IT + NIT * 16;   // + item table (int4 per item)
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_S = 0, T_DS = 256, T_DQ = 384;                        // dQ[w][b] at 384 + 64 w + 32 b
constexpr uint32_t I_S = idesc_bf16(128, BN), I_DQ = idesc_bf16(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f;

struct Bars {
  uint64_t q_full[QR], q_empty[QR], kv_full[ST], kv_empty[ST], s_full[2], s_free[2], ds_full[2][2], ds_free[2][2], dq_full[2][2], dq_free[2][2];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(384, 1)
swa_attn_dq_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                  const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mdq, const int* __restrict__ SEQU,
                  const float* __restrict__ LSE, const float* __restrict__ DV, int S, int N, float scale, unsigned long long* __restrict__ TRb) {
  if (TRACE && blockIdx.x == 0 && threadIdx.x == 0) TRb[60] = gtime();
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = (S + QM - 1) / QM;
  const int items = N * mt * 2;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  int4* itab = reinterpret_cast<int4*>(sm + O_IT);                      // (n, 2 i0 + hp, 65536 lo + hi, sq) of item li < NIT
  auto item_raw = [&](int li, int& n, int& i0, int& hp, int& lo, int& hi, int& sq) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    hp = wi & 1; const int r = wi >> 1;
    i0 = (r % mt) * QM; n = r / mt;
    sq = __ldg(SEQU + n);
    lo = max(i0 - HW, 0) / BN;
    const int kend = min(sq, i0 + QM + HW);
    hi = (kend + BN - 1) / BN;
    if (hi <= lo) hi = lo + 1;                                            // an item with no valid key still runs one (masked) block
  };
  auto item_of = [&](int li, int& n, int& i0, int& hp, int& lo, int& hi, int& sq) {
    if (li >= NIT) { item_raw(li, n, i0, hp, lo, hi, sq); return; }
    const int4 e = itab[li];
    n = e.x; i0 = e.y >> 1; hp = e.y & 1; lo = e.z >> 16; hi = e.z & 0xffff; sq = e.w;
  };
  if (tid < my_items && tid < NIT) {
    int n, i0, hp, lo, hi, sq; item_raw(tid, n, i0, hp, lo, hi, sq);
    itab[tid] = make_int4(n, 2 * i0 + hp, 65536 * lo + hi, sq);
  }

  if (tid == 0) {
    for (int s = 0; s < QR; ++s) { mbar_init(&B.q_full[s], 1); mbar_init(&B.q_empty[s], 2); }
    for (int s = 0; s < ST; ++s) { mbar_init(&B.kv_full[s], 1); mbar_init(&B.kv_empty[s], 2); }
    for (int w = 0; w < 2; ++w) {
      mbar_init(&B.s_full[w], 1); mbar_init(&B.s_free[w], 4); mbar_init(&B.ds_full[w][0], 4); mbar_init(&B.ds_full[w][1], 4);
      mbar_init(&B.ds_free[w][0], 1); mbar_init(&B.ds_free[w][1], 1); for (int b = 0; b < 2; ++b) { mbar_init(&B.dq_full[w][b], 1); mbar_init(&B.dq_free[w][b], 4); }
    }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

#if SETMAXNREG
  if (warp < 4) setmaxnreg_dec<56>();
#endif
  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        int n, i0, hp, lo, hi, sq; item_of(li, n, i0, hp, lo, hi, sq);
        const int qs = li % QR;
        if (li >= QR) mbar_wait(&B.q_empty[qs], ((li / QR) - 1) & 1);
        mbar_expect_tx(&B.q_full[qs], 4 * QM * DH * 2);
        for (int w = 0; w < 2; ++w) {
          tma_load_2d(su + O_Q + qs * QST + w * TQ, &mq, &B.q_full[qs], 0, (n * H + 2 * hp + w) * S + i0);
          tma_load_2d(su + O_Q + qs * QST + (2 + w) * TQ, &mdo, &B.q_full[qs], (2 * hp + w) * DH, n * S + i0);
        }
        for (int kb = lo; kb < hi; ++kb, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.kv_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.kv_full[s], 4 * BN * DH * 2);
          for (int w = 0; w < 2; ++w) {
            const int row = (n * H + 2 * hp + w) * S + kb * BN;
            tma_load_2d(st + w * TK, &mk, &B.kv_full[s], 0, row);
            tma_load_2d(st + 2 * TK + w * TK, &mv, &B.kv_full[s], 0, row);
          }
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ MMA issuers (warp 1 + w: head 2 hp + w)
    const int w = warp - 1;
    const uint32_t tS = tmem + T_S + w * 128, tDP = tS + 64;
    // S / dP of block G (global block counter G, item li, key block kb)
    auto issue_s = [&](int G, int li, int kb, int lo, int hi) {
      const int s = G % ST, qs = li % QR;
      if (kb == lo) mbar_wait(&B.q_full[qs], (li / QR) & 1);
      mbar_wait(&B.kv_full[s], (G / ST) & 1);
      if (G >= 1) mbar_wait(&B.s_free[w], (G - 1) & 1);                   // the row threads have read S / dP of G - 1
      tc_fence_after();
      const uint64_t dq = desc_sw64(su + O_Q + qs * QST + w * TQ), ddo = desc_sw64(su + O_Q + qs * QST + (2 + w) * TQ);
      const uint64_t dk = desc_sw64(su + O_ST + s * STB + w * TK), dv = desc_sw64(su + O_ST + s * STB + (2 + w) * TK);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks) umma_ss(tS, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks) umma_ss(tDP, ddo + (uint64_t)(ks * 2), dv + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w]);
        if (kb == hi - 1) tc_commit(&B.q_empty[qs]);
      }
      __syncwarp();
    };
    int G = 0;
    int nli = 0, nkb = 0, nlo = 0, nhi = 0;                                // the block after the current one (for the S / dP look-ahead)
    auto adv = [&](int& li, int& kb, int& lo, int& hi) {
      if (++kb == hi) { if (++li < my_items) { int n, i0, hp, sq; item_of(li, n, i0, hp, lo, hi, sq); kb = lo; } }
    };
    if (my_items > 0) {
      int n, i0, hp, sq; item_of(0, n, i0, hp, nlo, nhi, sq); nkb = nlo;
      issue_s(0, 0, nkb, nlo, nhi);
      adv(nli, nkb, nlo, nhi);
    }
    for (int li = 0; li < my_items; ++li) {
      int n, i0, hp, lo, hi, sq; item_of(li, n, i0, hp, lo, hi, sq);
      const int qb = li & 1;
      for (int kb = lo; kb < hi; ++kb, ++G) {
        const int s = G % ST;
        const bool last = kb == hi - 1;
        if (nli < my_items) { issue_s(G + 1, nli, nkb, nlo, nhi); adv(nli, nkb, nlo, nhi); }   // S / dP(G + 1) first (dQ is double-buffered)
        mbar_wait(&B.ds_full[w][G & 1], (G >> 1) & 1);          // per buffer: the row threads may run one block ahead
        if (kb == lo && li >= 2) mbar_wait(&B.dq_free[w][qb], ((li >> 1) - 1) & 1);   // item li - 2's dQ has been read out
        tc_fence_after();
        const uint64_t dk = desc_sw64(su + O_ST + s * STB + w * TK);   // MN-major: 8-key groups 512 B apart
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ts(tmem + T_DQ + w * 64 + qb * 32, tmem + T_DS + w * 64 + (G & 1) * 32 + ks * 8, dk + (uint64_t)(ks * 1024 >> 4), I_DQ,
                    (kb > lo || ks > 0) ? 1u : 0u);
          tc_commit(&B.ds_free[w][G & 1]);
          tc_commit(&B.kv_empty[s]);
          if (last) tc_commit(&B.dq_full[w][qb]);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ dS of head 2 hp + w, one query row per thread
#if SETMAXNREG
    setmaxnreg_inc<224>();
#endif
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const float SC = scale * LOG2E;
    int G = 0;
    auto rowvals = [&](int li, float& l2, float& d) {                      // LSE (log2 units) and D of this thread's row in item li
      int n, i0, hp, lo, hi, sq; item_of(li, n, i0, hp, lo, hi, sq);
      const int i = i0 + (int)r;
      const size_t rowi = ((size_t)n * H + 2 * hp + w) * S + i;
      l2 = i < sq ? __ldg(LSE + rowi) * LOG2E : 0.f; d = i < sq ? __ldg(DV + rowi) : 0.f;
    };
    float nl2 = 0.f, ndv = 0.f;
    if (my_items > 0) rowvals(0, nl2, ndv);
    // dQ of item pli = rn(scale acc) -> SW64 staging -> TMA store (head-major rows (n H + h) S + i0 ..); run after the NEXT item's first
    // block (dQ is double-buffered), so the last dQ MMA of an item is never waited for
    int pend = -1, pn = 0, pi0 = 0, ph = 0;
    auto epilogue = [&](int pli, int n, int i0, int h) {
      const int qb = pli & 1;
      { const int li = pli; EV(10); }
      mbar_wait(&B.dq_full[w][qb], (pli >> 1) & 1);
      { const int li = pli; EV(11); }
      tc_fence_after();
      uint32_t ov[32];
      tmem_ld32(trow + T_DQ + w * 64 + qb * 32, ov);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dq_free[w][qb]);
      const uint32_t xs = su + O_X + w * XS;
      if (r == 0) tma_store_wait_read0();
      named_bar_sync(1 + w, 128);
#pragma unroll
      for (int q = 0; q < 4; ++q)
        sts128(xs + sw64(r, q), make_uint4(pack_bf16(__uint_as_float(ov[8 * q]) * scale, __uint_as_float(ov[8 * q + 1]) * scale),
                                           pack_bf16(__uint_as_float(ov[8 * q + 2]) * scale, __uint_as_float(ov[8 * q + 3]) * scale),
                                           pack_bf16(__uint_as_float(ov[8 * q + 4]) * scale, __uint_as_float(ov[8 * q + 5]) * scale),
                                           pack_bf16(__uint_as_float(ov[8 * q + 6]) * scale, __uint_as_float(ov[8 * q + 7]) * scale)));
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (r == 0) { tma_store_2d(&mdq, xs, 0, (n * H + h) * S + i0); tma_store_commit(); }
      { const int li = pli; EV(12); }
    };
    for (int li = 0; li < my_items; ++li) {
      int n, i0, hp, lo, hi, sq; item_of(li, n, i0, hp, lo, hi, sq);
      const int i = i0 + (int)r, h = 2 * hp + w;
      const bool qok = i < sq;
      const float lse2 = nl2, dvv = ndv;
      if (li + 1 < my_items) rowvals(li + 1, nl2, ndv);                    // one item ahead
      for (int kb = lo; kb < hi; ++kb, ++G) {
        EV(2 * (kb - lo));
        mbar_wait(&B.s_full[w], G & 1);
        EV(2 * (kb - lo) + 1);
        tc_fence_after();
        const int j0 = kb * BN;
        const int clo = max(i - HW - j0, 0), chi = qok ? min(min(i + HW, sq - 1) - j0, BN - 1) : -1;
        const int wlo = __reduce_min_sync(~0u, chi >= clo ? clo : BN), whi = __reduce_max_sync(~0u, chi >= clo ? chi : -1);
        // two 32-column halves: S and dP of a half are loaded and waited for together (at most 64 loaded registers in flight)
        uint32_t ds[32];
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
          for (int cq = 0; cq < 4; ++cq) {                                 // 8-column chunks; a chunk outside the warp's windows is 0
            const int ch = 4 * hf + cq;
            if (!NOSKIP && (ch * 8 + 7 < wlo || ch * 8 > whi)) {
#pragma unroll
              for (int k = 0; k < 4; ++k) ds[4 * ch + k] = 0u;
              continue;
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              float d2[2];
#pragma unroll
              for (int e = 0; e < 2; ++e) {
                const int cl = 8 * cq + 2 * k + e, c = 32 * hf + cl;
                const float p = ex2f((c >= clo && c <= chi) ? __uint_as_float(sv[cl]) * SC - lse2 : -INFINITY);
                d2[e] = p * (__uint_as_float(pv[cl]) - dvv);
              }
              ds[4 * ch + k] = pack_bf16(d2[0], d2[1]);
            }
          }
        }
        if (G >= 2) mbar_wait(&B.ds_free[w][G & 1], ((G >> 1) - 1) & 1);  // dQ MMA of G - 2 has read this buffer
        tc_fence_after();
        tmem_st16(trow + T_DS + w * 64 + (G & 1) * 32, *reinterpret_cast<uint32_t(*)[16]>(ds));
        tmem_st16(trow + T_DS + w * 64 + (G & 1) * 32 + 16, *reinterpret_cast<uint32_t(*)[16]>(ds + 16));
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.ds_full[w][G & 1]);
        if (kb == lo && pend >= 0) { epilogue(pend, pn, pi0, ph); pend = -1; }
      }
      pend = li; pn = n; pi0 = i0; ph = h;
    }
    if (pend >= 0) epilogue(pend, pn, pi0, ph);
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
