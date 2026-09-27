// attn_fwd.cu — the SWA atom block's window attention forward on sm_100a (the kernel the H100 fused path runs in Triton as _attn_fwd;
// same math: keys j with |i - j| <= 64 and j < seqused[n], fp32 softmax in natural units, P rounded to bf16 for PV, O = acc / l in bf16,
// padding query rows (i >= seqused) 0, LSE = m + log l (natural log, 0 on rows without a valid key)).
// Structure from the token-DiT B200 forward (augattn_sm100/attn_fwd2.cu): persistent CTAs walk (sample, 128-query tile, head pair) items;
// the two softmax warpgroups take the two heads of the pair (one MMA warp each); 64-key blocks -- only the window's <= 4 blocks
// [i0 - 64, i0 + 192) per item -- through a TMA ring; S double-buffered in TMEM, P in TMEM, O accumulated in TMEM; lazy running max
// (log2 units); O leaves through a TMA store. Q / K / V are head-major [N, H, S, 32] (64-B rows in 128-B swizzled smem rows).
// TMEM: S[w][b] at w * 128 + b * 64, P[w] at 256 + w * 32 (bf16), O[w] at 320 + w * 64 (32 cols).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef ST
#define ST 4
#endif
#ifndef QR
#define QR 2
#endif
constexpr int BN = 64, DH = 32, QM = 128, C = 128, H = 4, HW = 64;
constexpr int TQ = QM * 128, TK = BN * 128;
constexpr int STB = 4 * TK;                                                // K0 | K1 | V0 | V1 = 32 KB
constexpr int O_Q = 0, O_ST = QR * 2 * TQ, O_X = O_ST + ST * STB;          // then O staging (2 x [128][64 B], SW64), bars
constexpr int XS = 128 * 64;
constexpr int O_BAR = O_X + 2 * XS, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_S = 0, T_P = 256, T_O = 320;
constexpr uint32_t I_QK = idesc_bf16(128, BN), I_PV = idesc_bf16(128, DH, 0, 1);
#ifndef LAZY
#define LAZY 0.f                         // running-max slack (log2 units); 0 = exact running max, the Triton kernel's rounding class for P
#endif
constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;

struct Bars {
  uint64_t q_full[QR], q_empty[QR], kv_full[ST], kv_empty[ST], s_full[2][2], p_full[2], p_free[2], o_free[2];
  uint32_t tmem;
};
DEVI float max3f(float a, float b, float c) { float d; asm("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }

extern "C" __global__ void __launch_bounds__(384, 1)
swa_attn_fwd_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                   const __grid_constant__ CUtensorMap mo, const int* __restrict__ SEQU, float* __restrict__ LSE, int S, int N, float scale) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = S / QM;
  const int items = N * mt * 2;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  // item li -> sample n, query tile start i0, head pair hp; its key blocks [lo, hi) (every role derives the same range)
  auto item_of = [&](int li, int& n, int& i0, int& hp, int& lo, int& hi, int& sq) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    hp = wi & 1; const int r = wi >> 1;
    i0 = (r % mt) * QM; n = r / mt;
    sq = __ldg(SEQU + n);
    lo = max(i0 - HW, 0) / BN;
    const int kend = min(sq, i0 + QM + HW);
    hi = (kend + BN - 1) / BN;
    if (hi <= lo) hi = lo + 1;                                            // an item with no valid key still runs one (masked) block
  };

  if (tid == 0) {
    for (int s = 0; s < QR; ++s) { mbar_init(&B.q_full[s], 1); mbar_init(&B.q_empty[s], 2); }
    for (int s = 0; s < ST; ++s) { mbar_init(&B.kv_full[s], 1); mbar_init(&B.kv_empty[s], 2); }
    for (int w = 0; w < 2; ++w) {
      mbar_init(&B.s_full[w][0], 1); mbar_init(&B.s_full[w][1], 1);
      mbar_init(&B.p_full[w], 4); mbar_init(&B.p_free[w], 1); mbar_init(&B.o_free[w], 4);
    }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        int n, i0, hp, lo, hi, sq; item_of(li, n, i0, hp, lo, hi, sq);
        const int qs = li % QR;
        if (li >= QR) mbar_wait(&B.q_empty[qs], ((li / QR) - 1) & 1);
        mbar_expect_tx(&B.q_full[qs], 2 * QM * DH * 2);
        for (int w = 0; w < 2; ++w) tma_load_2d(su + O_Q + qs * 2 * TQ + w * TQ, &mq, &B.q_full[qs], 0, (n * H + 2 * hp + w) * S + i0);
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
    // block sequence of this CTA: (li, kb) with a global counter G; QK(G + 2) follows PV(G)
    int ql = 0, qkb = 0, qlo, qhi, qn, qi0, qhp, qsq;                        // the next QK to issue
    item_of(0, qn, qi0, qhp, qlo, qhi, qsq); qkb = qlo;
    int qG = 0;
    auto issue_qk = [&]() {
      const int s = qG % ST, qs = ql % QR;
      if (qkb == qlo) mbar_wait(&B.q_full[qs], (ql / QR) & 1);
      mbar_wait(&B.kv_full[s], (qG / ST) & 1);
      tc_fence_after();
      const uint64_t dq = desc_k128(su + O_Q + qs * 2 * TQ + w * TQ), dk = desc_k128(su + O_ST + s * STB + w * TK);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks)
          umma_ss(tmem + T_S + w * 128 + (qG & 1) * 64, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_QK, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w][qG & 1]);
        if (qkb == qhi - 1) tc_commit(&B.q_empty[qs]);
      }
      __syncwarp();
      ++qG;
      if (++qkb == qhi) { if (++ql < my_items) { item_of(ql, qn, qi0, qhp, qlo, qhi, qsq); qkb = qlo; } }
    };
    if (my_items > 0) {
      issue_qk();
      if (ql < my_items) issue_qk();
    }
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      int n, i0, hp, lo, hi, sq; item_of(li, n, i0, hp, lo, hi, sq);
      for (int kb = lo; kb < hi; ++kb, ++G) {
        const int s = G % ST;
        mbar_wait(&B.p_full[w], G & 1);
        if (kb == lo && li >= 1) mbar_wait(&B.o_free[w], (li - 1) & 1);    // the previous item's O has been read out
        tc_fence_after();
        const uint64_t dv = desc_mn128(su + O_ST + s * STB + 2 * TK + w * TK, 8192);
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ts(tmem + T_O + w * 64, tmem + T_P + w * 32 + ks * 8, dv + (uint64_t)(ks * 2048 >> 4), I_PV, (kb > lo || ks > 0) ? 1u : 0u);
          tc_commit(&B.p_free[w]);
          tc_commit(&B.kv_empty[s]);
        }
        __syncwarp();
        if (ql < my_items) issue_qk();
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ softmax of head 2 hp + w, one query row per thread
    setmaxnreg_inc<224>();
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const float SC = scale * LOG2E;
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      int n, i0, hp, lo, hi, sq; item_of(li, n, i0, hp, lo, hi, sq);
      const int i = i0 + (int)r;
      float m_i = -INFINITY, l_i = 0.f;
      for (int kb = lo; kb < hi; ++kb, ++G) {
        mbar_wait(&B.s_full[w][G & 1], (G >> 1) & 1);
        tc_fence_after();
        // this row's valid keys in block-local columns: [clo, chi] (i - 64 <= j <= i + 64, j < seqused)
        const int j0 = kb * BN;
        const int clo = max(i - HW - j0, 0), chi = min(min(i + HW, sq - 1) - j0, BN - 1);
        // a warp whose 32 rows have no valid key in this block writes P = 0 and leaves the max / sum alone
        const bool wskip = __all_sync(0xffffffffu, clo > chi);
        if (wskip) {
          if (G >= 1) tc_fence_before();
          if (G >= 1) mbar_wait(&B.p_free[w], (G - 1) & 1);
          tc_fence_after();
          const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
          tmem_st16(trow + T_P + w * 32, z);
          tmem_st16(trow + T_P + w * 32 + 16, z);
          tmem_wait_st();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.p_full[w]);
          continue;
        }
        float t[BN];
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          uint32_t v[32];
          tmem_ld32(trow + T_S + w * 128 + (G & 1) * 64 + cc * 32, v);
          tmem_wait_ld();
#pragma unroll
          for (int k = 0; k < 32; ++k) {
            const int c = cc * 32 + k;
            t[c] = (c >= clo && c <= chi) ? (i < sq ? __uint_as_float(v[k]) * SC : 0.f) : -INFINITY;   // (padding query rows score 0,
          }                                                                                                  //  as Triton's zero q)
        }
        if (G >= 1) tc_fence_before();
        float mxp[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < BN / 2; ++j) mxp[j & 3] = max3f(mxp[j & 3], t[2 * j], t[2 * j + 1]);
        const float mx = max3f(mxp[0], mxp[1], fmaxf(mxp[2], mxp[3]));
        const float m_new = mx > m_i + LAZY ? mx : m_i;
        if (G >= 1) mbar_wait(&B.p_free[w], (G - 1) & 1);                  // PV(G - 1) done: P free, O final for G - 1
        tc_fence_after();
        if (__any_sync(0xffffffffu, m_new != m_i)) {
          const float alpha = (m_i == -INFINITY) ? 0.f : ex2f(m_i - m_new);
          l_i *= alpha;
          if (kb > lo) {
            uint32_t ov[32];
            tmem_ld32(trow + T_O + w * 64, ov);
            tmem_wait_ld();
#pragma unroll
            for (int k = 0; k < 32; ++k) ov[k] = __float_as_uint(__uint_as_float(ov[k]) * alpha);
            tmem_st16(trow + T_O + w * 64, *reinterpret_cast<uint32_t(*)[16]>(ov));
            tmem_st16(trow + T_O + w * 64 + 16, *reinterpret_cast<uint32_t(*)[16]>(ov + 16));
          }
          m_i = m_new;
        }
        const float NM = (m_i == -INFINITY) ? 0.f : -m_i;
        float ssum = 0.f;
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          uint32_t pk[16];
#pragma unroll
          for (int k = 0; k < 16; ++k) {
            const float p0 = ex2f(t[cc * 32 + 2 * k] + NM), p1 = ex2f(t[cc * 32 + 2 * k + 1] + NM);
            ssum += p0 + p1;
            pk[k] = pack_bf16(p0, p1);
          }
          tmem_st16(trow + T_P + w * 32 + cc * 16, pk);
        }
        l_i += ssum;
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.p_full[w]);
      }
      // ---- epilogue: O = acc / l (bf16; 0 on padding rows / empty rows), LSE = m ln 2 + log l
      mbar_wait(&B.p_free[w], (G - 1) & 1);
      tc_fence_after();
      uint32_t ov[32];
      tmem_ld32(trow + T_O + w * 64, ov);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.o_free[w]);
      const bool qok = i < sq && l_i > 0.f;
      const float inv = qok ? 1.f / l_i : 0.f;
      const uint32_t xs = su + O_X + w * XS;
      if (r == 0) tma_store_wait_read0();
      named_bar_sync(1 + w, 128);
#pragma unroll
      for (int q = 0; q < 4; ++q)
        sts128(xs + sw64(r, q), make_uint4(pack_bf16(__uint_as_float(ov[8 * q]) * inv, __uint_as_float(ov[8 * q + 1]) * inv),
                                           pack_bf16(__uint_as_float(ov[8 * q + 2]) * inv, __uint_as_float(ov[8 * q + 3]) * inv),
                                           pack_bf16(__uint_as_float(ov[8 * q + 4]) * inv, __uint_as_float(ov[8 * q + 5]) * inv),
                                           pack_bf16(__uint_as_float(ov[8 * q + 6]) * inv, __uint_as_float(ov[8 * q + 7]) * inv)));
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      const int h = 2 * hp + w;
      if (r == 0) { tma_store_2d(&mo, xs, h * DH, n * S + i0); tma_store_commit(); }
      LSE[((size_t)n * H + h) * S + i] = l_i > 0.f ? (m_i + __log2f(l_i)) * LN2 : 0.f;   // (as the Triton kernel: padding rows keep theirs)
    }
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
