// attn_dq_tf32.cu — the fp32 SWA atom block's window-attention backward dQ pass on sm_100a, TF32 tensor cores, fp32 operands throughout
// (the counterpart of attn_dq.cu): keys j with |i - j| <= 64 and j < seqused[n], queries i < seqused[n]
//   S = q K^T, dP = dO V^T (TMEM, fp32);  P = exp(S scale - LSE);  dS = P (dP - D);  dQ = scale sum_blocks dS K
// Same operands as attn_dkv_tf32.cu (Q / K / V head-major [N, H, S, 32] fp32, dO row-major [N S, 128], LSE / D [N, H, S]).
// Why a separate pass: dQ = dS K needs dS with the queries as rows. In the key-major dK / dV pass dS^T sits in TMEM with the keys as
// lanes, and turning it into an A operand with the queries as M would need an MN-major A (unmeasured for kind::tf32) or a transposing
// copy through shared memory (64 queries x 128 keys of fp32 per block). Recomputing S / dP here, query-major, keeps every operand in a
// layout that is measured on B200 (K-major SS for S / dP, TS with an MN-major B for dQ) and needs no cross-CTA pairing: each query
// tile's dQ is complete inside one item (deterministic, no atomics, no fp32 dQ buffer).
// Items (sample, 128-query tile, head): the window's <= 8 32-key blocks [i0 - 64, i0 + 192) stream through a ring (K, V K-major and K
// MN-major, 12 KB a block). The two compute warpgroups take alternate blocks (one query row per thread); one MMA warp issues every MMA in
// block order, so the shared dQ accumulator sums in a fixed order. Per block: S / dP (SS, M = 128 queries, N = 32 keys, K = 32;
// double-buffered per warpgroup) -> dS (rounded to TF32) into TMEM -> dQ += dS K (TS, M = 128, N = 32, K = 32 keys, K MN-major).
// Epilogue: warpgroup w writes columns 16 w .. 16 w + 15 of dQ (scaled, fp32), 64 B per thread straight from registers.
// Shared memory: q | dO item slots 2 x 32 KB | 12 block stages x 12 KB | barriers  (208.5 KB).
// TMEM: warpgroup w at 160 w: S[b] at + 64 b, dP[b] at + 64 b + 32, dS at + 128;  dQ at 320.
// Registers: <= 128 / thread (launch bound 512, launched with 384 threads).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

// ------------------------------------------------------------------ kind::tf32 (as augmented_attention/cuda/sm100/sm100.cuh, measured on B200)
namespace tf {
__host__ __device__ constexpr uint32_t idesc(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEVI uint64_t desc_mn32b(uint32_t saddr, uint32_t lbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)(512 >> 4) << 32) |
         ((uint64_t)1 << 46) | ((uint64_t)1 << 61);
}
DEVI void mma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI void mma_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p; }"
               :: "r"(d), "r"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI uint32_t rna(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r & 0xffffe000u; }
}  // namespace tf

#ifndef ST
#define ST 12                                                              // block stages
#endif
constexpr int BK = 32, DH = 32, QM = 128, H = 4, HW = 64, QR = 2;
constexpr int TQI = QM * 128, IST = 2 * TQI;                               // q (or dO) of the item: [128][32] fp32 = 16 KB; q | dO
constexpr int TK = BK * 128, BLS = 3 * TK;                                 // a [32 keys][32] tile: 4 KB; K | V | K MN-major
constexpr int O_IT = 0, O_BL = QR * IST, O_BAR = O_BL + ST * BLS, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_DQ = 320;
constexpr uint32_t I_S = tf::idesc(128, BK), I_DQ = tf::idesc(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f;

struct Bars {
  uint64_t it_full[QR], it_empty[QR], bl_full[ST], bl_empty[ST], s_full[2][2], s_free[2][2], d_full[2], d_free[2], q_full, q_free;
  uint32_t tmem;
};
DEVI void st4(float* p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(p), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

extern "C" __global__ void __launch_bounds__(512, 1)
swa_attn_dq_tf32_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mdo,
                       const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                       const __grid_constant__ CUtensorMap mkm, const float* __restrict__ LSE, const float* __restrict__ DD,
                       const int* __restrict__ SEQU, float* __restrict__ DQO, int S, int N, float scale) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = S / QM;
  const int items = N * mt * H;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  // item li of this CTA: wi = blockIdx + li grid = ((n mt + m) H + h); its key blocks [lo, hi) (one masked block when none is valid)
  auto item_of = [&](int li, int& n, int& i0, int& h, int& lo, int& hi, int& sq) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    h = wi % H; const int r = wi / H;
    i0 = (r % mt) * QM; n = r / mt;
    sq = __ldg(SEQU + n);
    lo = max(i0 - HW, 0) / BK;
    hi = (min(sq, i0 + QM + HW) + BK - 1) / BK;
    if (hi <= lo) hi = lo + 1;
  };

  if (tid == 0) {
    for (int s = 0; s < QR; ++s) { mbar_init(&B.it_full[s], 1); mbar_init(&B.it_empty[s], 1); }
    for (int s = 0; s < ST; ++s) { mbar_init(&B.bl_full[s], 1); mbar_init(&B.bl_empty[s], 1); }
    for (int w = 0; w < 2; ++w) {
      for (int b = 0; b < 2; ++b) { mbar_init(&B.s_full[w][b], 1); mbar_init(&B.s_free[w][b], 4); }
      mbar_init(&B.d_full[w], 4); mbar_init(&B.d_free[w], 1);
    }
    mbar_init(&B.q_full, 1); mbar_init(&B.q_free, 8);
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
        int n, i0, h, lo, hi, sq; item_of(li, n, i0, h, lo, hi, sq);
        const int is = li % QR;
        if (li >= QR) mbar_wait(&B.it_empty[is], ((li / QR) - 1) & 1);
        mbar_expect_tx(&B.it_full[is], IST);
        tma_load_2d(su + O_IT + is * IST, &mq, &B.it_full[is], 0, (n * H + h) * S + i0);
        tma_load_2d(su + O_IT + is * IST + TQI, &mdo, &B.it_full[is], h * DH, n * S + i0);
        for (int kb = lo; kb < hi; ++kb, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.bl_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_BL + s * BLS;
          const int krow = (n * H + h) * S + kb * BK;
          mbar_expect_tx(&B.bl_full[s], BLS);
          tma_load_2d(st, &mk, &B.bl_full[s], 0, krow);
          tma_load_2d(st + TK, &mv, &B.bl_full[s], 0, krow);
          tma_load_2d(st + 2 * TK, &mkm, &B.bl_full[s], 0, krow);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer (block G -> warpgroup G & 1)
    auto issue_s = [&](int G, int li, int kb, int lo, int hi) {
      const int s = G % ST, is = li % QR, w = G & 1, gw = G >> 1, bs = gw & 1;
      if (kb == lo) mbar_wait(&B.it_full[is], (li / QR) & 1);
      mbar_wait(&B.bl_full[s], (G / ST) & 1);
      if (gw >= 2) mbar_wait(&B.s_free[w][bs], ((gw >> 1) - 1) & 1);
      tc_fence_after();
      const uint32_t it = su + O_IT + is * IST, st = su + O_BL + s * BLS, d = tmem + 160 * w + 64 * bs;
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < 4; ++k) tf::mma_ss(d, desc_k128(it) + (uint64_t)(k * 2), desc_k128(st) + (uint64_t)(k * 2), I_S, k > 0 ? 1u : 0u);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          tf::mma_ss(d + 32, desc_k128(it + TQI) + (uint64_t)(k * 2), desc_k128(st + TK) + (uint64_t)(k * 2), I_S, k > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w][bs]);
        if (kb == hi - 1) tc_commit(&B.it_empty[is]);                      // the item's last read of q / dO
      }
      __syncwarp();
    };
    int nli = 0, nkb = 0, nlo = 0, nhi = 0;
    auto adv = [&]() {
      if (++nkb == nhi) { if (++nli < my_items) { int n, i0, h, sq; item_of(nli, n, i0, h, nlo, nhi, sq); nkb = nlo; } }
    };
    if (my_items > 0) {
      int n, i0, h, sq; item_of(0, n, i0, h, nlo, nhi, sq); nkb = nlo;
      issue_s(0, 0, nkb, nlo, nhi);
      adv();
    }
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      int n, i0, h, lo, hi, sq; item_of(li, n, i0, h, lo, hi, sq);
      for (int kb = lo; kb < hi; ++kb, ++G) {
        if (nli < my_items) { issue_s(G + 1, nli, nkb, nlo, nhi); adv(); }
        const int s = G % ST, w = G & 1, gw = G >> 1;
        mbar_wait(&B.d_full[w], gw & 1);                                   // dS of block G is in TMEM
        if (kb == lo && li >= 1) mbar_wait(&B.q_free, (li - 1) & 1);       // the previous item's dQ has been read out
        tc_fence_after();
        const uint32_t st = su + O_BL + s * BLS;
        if (elect_one()) {
#pragma unroll
          for (int k = 0; k < 4; ++k)                                      // K = 32 keys: 8 per step (1 KB of the MN-major K tile)
            tf::mma_ts(tmem + T_DQ, tmem + 160 * w + 128 + 8 * k, tf::desc_mn32b(st + 2 * TK, TK) + (uint64_t)((k * 1024) >> 4), I_DQ,
                       (kb > lo || k > 0) ? 1u : 0u);
          tc_commit(&B.d_free[w]);
          tc_commit(&B.bl_empty[s]);
          if (kb == hi - 1) tc_commit(&B.q_full);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ dS, one query row per thread
    const int w = (warp - 4) >> 2, lq = warp & 3, r = lq * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16);
    const float SC = scale * LOG2E;
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      int n, i0, h, lo, hi, sq; item_of(li, n, i0, h, lo, hi, sq);
      const int i = i0 + r;
      const bool qok = i < sq;
      const size_t rowi = (size_t)(n * H + h) * S + i;
      const float l2 = qok ? __ldg(LSE + rowi) * LOG2E : 0.f, dd = qok ? __ldg(DD + rowi) : 0.f;
      for (int kb = lo; kb < hi; ++kb, ++G) {
        if ((G & 1) != w) continue;
        const int gw = G >> 1, bs = gw & 1;
        mbar_wait(&B.s_full[w][bs], (gw >> 1) & 1);
        tc_fence_after();
        uint32_t sv[32], dv[32];
        tmem_ld32(trow + 160 * w + 64 * bs, sv);
        tmem_ld32(trow + 160 * w + 64 * bs + 32, dv);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.s_free[w][bs]);
        const int j0 = kb * BK;
        const int clo = max(i - HW - j0, 0), chi = qok ? min(min(i + HW, sq - 1) - j0, BK - 1) : -1;
#pragma unroll
        for (int c = 0; c < 32; ++c) {
          const bool ok = c >= clo && c <= chi;
          const float p = ok ? ex2f(__uint_as_float(sv[c]) * SC - l2) : 0.f;
          dv[c] = tf::rna(ok ? p * (__uint_as_float(dv[c]) - dd) : 0.f);
        }
        if (gw >= 1) mbar_wait(&B.d_free[w], (gw - 1) & 1);                // the dQ MMA of this warpgroup's previous block is done
        tc_fence_after();
        const uint32_t td = trow + 160 * w + 128;
        tmem_st16(td, *reinterpret_cast<uint32_t(*)[16]>(dv));
        tmem_st16(td + 16, *reinterpret_cast<uint32_t(*)[16]>(dv + 16));
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.d_full[w]);
      }
      // ---- the item's dQ: warpgroup w writes columns 16 w .. 16 w + 15 of query row i (scaled, fp32)
      mbar_wait(&B.q_full, li & 1);
      tc_fence_after();
      uint32_t v[16];
      tmem_ld16(trow + T_DQ + 16 * w, v);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.q_free);
      float* out = DQO + rowi * DH + 16 * w;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        st4(out + 4 * q, __float_as_uint(__uint_as_float(v[4 * q]) * scale), __float_as_uint(__uint_as_float(v[4 * q + 1]) * scale),
            __float_as_uint(__uint_as_float(v[4 * q + 2]) * scale), __float_as_uint(__uint_as_float(v[4 * q + 3]) * scale));
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
