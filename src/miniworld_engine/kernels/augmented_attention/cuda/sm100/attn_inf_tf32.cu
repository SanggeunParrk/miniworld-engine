// attn_inf_tf32.cu — the token DiT's INFERENCE attention core for the fp32 path, sm_100a, TF32 tensor cores (the fp32 twin of
// attn_inf.cu, same persistent schedule and warp roles):
//
//   o[a] = sigmoid(g[a]) * softmax(q[a] k[a]^T + bias) v[a]      written as fp32 OVER q (the tdit step contract)
//
// q | k | g are column views of the fp32 q|k|g GEMM output [S L, 2304]; v arrives TRANSPOSED, v^T [768, S L] (a second GEMM),
// so PV reads it K-major: an MN-major tf32 operand (v with keys as rows) needs the 32-B-atom swizzle (sm100.cuh), whose 128-B
// rows would cost the 48-channel tile a third more smem than fits next to the g / o staging. The bias is the hoisted fp32
// head-major bias.
// Logits arrive in exp2 units (sm_scale log2 e folded into Wq / bq, log2 e into the bias). Products are tcgen05 kind::tf32
// with fp32 accumulation; the softmax, P and the running sums are fp32 (P is rounded to tf32 by the MMA).
//
// What differs from the bf16 core, and why:
//   * a 48-wide fp32 head row is 192 B, more than a 128-B swizzle atom: every q / k / g / o tile is two TMA boxes, columns
//     0-31 (128-B swizzle) and 32-47 (64-B swizzle); QK runs K = 8 steps over both atoms. The v^T tile is one box, 48
//     channel rows x 32 keys (128 B, 128-B swizzle): PV is one N = 48 MMA per 8 keys.
//   * 32-key blocks (not 64), so three stages of k, v and an fp32 bias tile fit next to the q ring and the g / o staging.
//   * P is fp32 in TMEM (32 columns per buffer, double-buffered); TMEM: S[w][b] at w * 64 + b * 32, P[w][b] at
//     128 + w * 64 + b * 32, O[w] at 256 + w * 64 (48 columns).
// L is any multiple of 8, as in attn_inf.cu: 3-D maps (columns, row of the sample, sample | head; v^T: key, sample, channel) make
// TMA zero-fill the tails and clip the stores, and the last key block's keys past L are masked to -inf.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef PP
#define PP 1                             // the two softmax warpgroups take turns on the exponentials (named bars 3, 4)
#endif
#ifndef LAZY
#define LAZY 8.0f                        // log2 units
#endif
#ifndef ST
#define ST 3
#endif
constexpr int BN = 32, DH = 48, QM = 128;
// q tile of one sample: A (cols 0-31, 128 x 128 B, SW128) + B (cols 32-47, 128 x 64 B, SW64)
constexpr int QA = QM * 128, QB = QM * 64, TQ = QA + QB;                   // 24 KB
// stage: k A / B and v^T of both samples, then the bias tile (128 queries x 32 keys fp32, SW128)
constexpr int KA = BN * 128, KB = BN * 64, TV = DH * BN * 4;
constexpr int S_KA = 0, S_KB = 2 * KA, S_VT = 2 * KA + 2 * KB, S_BIAS = S_VT + 2 * TV;
constexpr int TBIAS = QM * BN * 4, STB = S_BIAS + TBIAS;                  // 40 KB
constexpr int O_Q = 0, O_ST = 2 * TQ, O_BAR = O_ST + ST * STB;
constexpr int XG = TQ;                                                     // per warpgroup: g in, gated o out (A + B boxes)
constexpr int O_X = O_BAR + 1024, SMEM_BYTES = O_X + 2 * XG;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert((O_ST % 1024) == 0 && (STB % 1024) == 0 && (O_X % 1024) == 0 && (S_VT % 1024) == 0 && (TV % 1024) == 0,
              "1 KB alignment of the 128-B-swizzled tiles");
constexpr uint32_t T_S = 0, T_P = 128, T_O = 256;
constexpr uint32_t I_QK = idesc_tf32(128, BN), I_PV = idesc_tf32(128, DH);

// p_full is per P buffer (G & 1): the softmax can hand over P(G) and P(G + 1) before the MMA warp tests P(G) -- QK(G + 1) is
// issued before that wait -- and one barrier completing two phases in between would leave the wait on a parity that never
// comes back (a hang that only showed up under concurrent load). Two barriers: completing one twice needs P(G + 2), which
// needs S(G + 2), issued only after the MMA warp has passed P(G).
struct Bars {
  uint64_t q_full, q_empty, kv_full[ST], kv_empty[ST], s_full[2][2], p_full[2][2], p_free[2][2], o_free[2], g_full[2];
  uint32_t tmem;
};
DEVI float max3f(float a, float b, float c) { float d; asm("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }

extern "C" __global__ void __launch_bounds__(384, 1)
augattn_inf_tf32_sm100(const __grid_constant__ CUtensorMap mqa, const __grid_constant__ CUtensorMap mqb,
                       const __grid_constant__ CUtensorMap mka, const __grid_constant__ CUtensorMap mkb,
                       const __grid_constant__ CUtensorMap mvt, const __grid_constant__ CUtensorMap mb,
                       const __grid_constant__ CUtensorMap mga, const __grid_constant__ CUtensorMap mgb, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int npair = (A + 1) >> 1, mt = (L + QM - 1) / QM, nb = (L + BN - 1) / BN;
  const int items = npair * mt * 16;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int nblk = my_items * nb;
  auto item_of = [&](int li, int& a0, int& m0, int& head) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    const int pair = wi % npair, cid = wi / npair;
    a0 = 2 * pair; m0 = (cid % mt) * QM; head = cid / mt;
  };

  if (tid == 0) {
    mbar_init(&B.q_full, 1); mbar_init(&B.q_empty, 2);
    for (int s = 0; s < ST; ++s) { mbar_init(&B.kv_full[s], 1); mbar_init(&B.kv_empty[s], 2); }
    for (int w = 0; w < 2; ++w) {
      mbar_init(&B.s_full[w][0], 1); mbar_init(&B.s_full[w][1], 1);
      mbar_init(&B.p_full[w][0], 4); mbar_init(&B.p_full[w][1], 4); mbar_init(&B.p_free[w][0], 1); mbar_init(&B.p_free[w][1], 1); mbar_init(&B.o_free[w], 4); mbar_init(&B.g_full[w], 1);
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
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        int a0, m0, head; item_of(li, a0, m0, head);
        const int qcol = head * DH;
        if (li >= 1) mbar_wait(&B.q_empty, (li - 1) & 1);
        mbar_expect_tx(&B.q_full, 2 * TQ);
        for (int w = 0; w < 2; ++w) {
          const int a = min(a0 + w, A - 1);
          tma_load_3d(su + O_Q + w * TQ, &mqa, &B.q_full, qcol, m0, a);
          tma_load_3d(su + O_Q + w * TQ + QA, &mqb, &B.q_full, qcol + 32, m0, a);
        }
        for (int n = 0; n < nb; ++n, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.kv_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.kv_full[s], STB);
          for (int w = 0; w < 2; ++w) {
            const int a = min(a0 + w, A - 1);
            tma_load_3d(st + S_KA + w * KA, &mka, &B.kv_full[s], qcol, n * BN, a);
            tma_load_3d(st + S_KB + w * KB, &mkb, &B.kv_full[s], qcol + 32, n * BN, a);
            tma_load_3d(st + S_VT + w * TV, &mvt, &B.kv_full[s], n * BN, a, qcol);
          }
          tma_load_3d(st + S_BIAS, &mb, &B.kv_full[s], n * BN, m0, head);
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // one MMA warp per softmax warpgroup (warp 1: sample a0, warp 2: a0 + 1); QK(G + 2) follows PV(G): S and P double-buffered
    const int w = warp - 1;
    auto qk = [&](int G, int li, int n) {
      const int s = G % ST;
      if (n == 0) mbar_wait(&B.q_full, li & 1);
      mbar_wait(&B.kv_full[s], (G / ST) & 1);
      tc_fence_after();
      const uint32_t qa = su + O_Q + w * TQ, st = su + O_ST + s * STB;
      const uint64_t dqa = desc_k128(qa), dqb = desc_sw64(qa + QA);
      const uint64_t dka = desc_k128(st + S_KA + w * KA), dkb = desc_sw64(st + S_KB + w * KB);
      const uint32_t d = tmem + T_S + w * 64 + (G & 1) * 32;
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) umma_ss_tf32(d, dqa + (uint64_t)(ks * 2), dka + (uint64_t)(ks * 2), I_QK, ks > 0 ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) umma_ss_tf32(d, dqb + (uint64_t)(ks * 2), dkb + (uint64_t)(ks * 2), I_QK, 1u);
        tc_commit(&B.s_full[w][G & 1]);
        if (n == nb - 1) tc_commit(&B.q_empty);
      }
      __syncwarp();
    };
    auto step2 = [&](int li, int n, int& li2, int& n2) { li2 = li; n2 = n + 2; while (n2 >= nb) { n2 -= nb; ++li2; } };
    for (int G = 0; G < 2 && G < nblk; ++G) qk(G, G / nb, G % nb);
    for (int G = 0, li = 0, n = 0; G < nblk; ++G) {
      const int s = G % ST;
      mbar_wait(&B.p_full[w][G & 1], (G >> 1) & 1);
      if (n == 0 && li >= 1) mbar_wait(&B.o_free[w], (li - 1) & 1);
      tc_fence_after();
      const uint32_t st = su + O_ST + s * STB;
      // v^T is the K-major B operand: 48 channel rows of 32 keys; 8 keys (32 B) per K step
      const uint64_t dv = desc_k128(st + S_VT + w * TV);
      const uint32_t pa = tmem + T_P + w * 64 + (G & 1) * 32, oa = tmem + T_O + w * 64;
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < BN / 8; ++ks) {
          const uint32_t acc = (n > 0 || ks > 0) ? 1u : 0u;
          umma_ts_tf32(oa, pa + ks * 8, dv + (uint64_t)(ks * 2), I_PV, acc);
        }
        tc_commit(&B.p_free[w][G & 1]);
        tc_commit(&B.kv_empty[s]);
      }
      __syncwarp();
      if (G + 2 < nblk) { int li2, n2; step2(li, n, li2, n2); qk(G + 2, li2, n2); }
      if (++n == nb) { n = 0; ++li; }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------ softmax of sample a0 + w, one query row per thread
    setmaxnreg_inc<224>();
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    if (PP && w == 1) named_bar_arrive(3, 256);
    const uint32_t xa = su + O_X + w * XG, xb = xa + QA;
    float m_i = -INFINITY, l_i = 0.f;
    for (int G = 0, li = 0, n = 0; G < nblk; ++G, ++n) {
      if (n == nb) { n = 0; ++li; }
      const int s = G % ST;
      if (n == 0) {
        m_i = -INFINITY; l_i = 0.f;
        if (r == 0) {                                                      // this item's g tile into the (drained) staging buffer
          int a0, m0, head; item_of(li, a0, m0, head);
          tma_store_wait_read0();
          mbar_expect_tx(&B.g_full[w], XG);
          const int a = min(a0 + w, A - 1);
          tma_load_3d(xa, &mga, &B.g_full[w], head * DH, m0, a);
          tma_load_3d(xb, &mgb, &B.g_full[w], head * DH + 32, m0, a);
        }
      }
      mbar_wait(&B.kv_full[s], (G / ST) & 1);                              // the bias tile of this block
      mbar_wait(&B.s_full[w][G & 1], (G >> 1) & 1);
      tc_fence_after();
      float t[BN];
      {
        uint32_t v[32];
        tmem_ld32(trow + T_S + w * 64 + (G & 1) * 32, v);
        tmem_wait_ld();
        const uint32_t sb = su + O_ST + s * STB + S_BIAS;
#pragma unroll
        for (int q = 0; q < 8; ++q) {                                      // 4 keys of fp32 bias per 16-byte chunk
          const uint4 bw = lds128(sb + sw128(r, q));
          t[4 * q + 0] = __uint_as_float(v[4 * q + 0]) + __uint_as_float(bw.x);
          t[4 * q + 1] = __uint_as_float(v[4 * q + 1]) + __uint_as_float(bw.y);
          t[4 * q + 2] = __uint_as_float(v[4 * q + 2]) + __uint_as_float(bw.z);
          t[4 * q + 3] = __uint_as_float(v[4 * q + 3]) + __uint_as_float(bw.w);
        }
      }
      if (G >= 1) { tc_fence_before(); }
      if (const int kv = L - n * BN; kv < BN) {                            // the last block's keys past L (kv is a multiple of 8)
#pragma unroll
        for (int j = 0; j < BN; ++j) if (j >= kv) t[j] = -INFINITY;
      }
      float mxp[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
      for (int j = 0; j < BN / 2; ++j) mxp[j & 3] = max3f(mxp[j & 3], t[2 * j], t[2 * j + 1]);
      const float mx = max3f(mxp[0], mxp[1], fmaxf(mxp[2], mxp[3]));
      const float m_new = mx > m_i + LAZY ? mx : m_i;
      const bool resc = __any_sync(0xffffffffu, m_new != m_i);
      // P buffer G & 1 is free once PV(G - 2) is done; O may be rescaled only once PV(G - 1) is done (rare: lazy max)
      if (resc && G >= 1) mbar_wait(&B.p_free[w][(G - 1) & 1], ((G - 1) >> 1) & 1);
      if (G >= 2) mbar_wait(&B.p_free[w][G & 1], ((G - 2) >> 1) & 1);
      tc_fence_after();
      if (resc) {
        const float alpha = ex2f(m_i - m_new);
        l_i *= alpha;
        if (n >= 1) {
          uint32_t ov[16];
#pragma unroll
          for (int cc = 0; cc < 3; ++cc) {
            tmem_ld16(trow + T_O + w * 64 + cc * 16, ov);
            tmem_wait_ld();
#pragma unroll
            for (int k = 0; k < 16; ++k) ov[k] = __float_as_uint(__uint_as_float(ov[k]) * alpha);
            tmem_st16(trow + T_O + w * 64 + cc * 16, ov);
          }
        }
        m_i = m_new;
      }
      float ssp[4] = {0.f, 0.f, 0.f, 0.f};
      if (PP) named_bar_sync(3 + w, 256);                                  // my exp turn
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {                                     // 32 keys -> 32 fp32 P columns
        uint32_t pk[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float p = ex2f(t[cc * 16 + k] - m_i);
          ssp[k & 3] += p;
          pk[k] = __float_as_uint(p);
        }
        tmem_st16(trow + T_P + w * 64 + (G & 1) * 32 + cc * 16, pk);
      }
      if (PP) named_bar_arrive(4 - w, 256);
      l_i += (ssp[0] + ssp[1]) + (ssp[2] + ssp[3]);
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.p_full[w][G & 1]);
      if (n == nb - 1) {
        // ---- epilogue of the item: o = sigmoid(g) acc / l, fp32, over q
        int a0, m0, head; item_of(li, a0, m0, head);
        mbar_wait(&B.p_free[w][G & 1], (G >> 1) & 1);
        tc_fence_after();
        const float inv = 1.f / l_i;
        uint32_t ov[48];
#pragma unroll
        for (int cc = 0; cc < 3; ++cc) tmem_ld16(trow + T_O + w * 64 + cc * 16, *reinterpret_cast<uint32_t(*)[16]>(ov + 16 * cc));
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.o_free[w]);
        mbar_wait(&B.g_full[w], li & 1);
#pragma unroll
        for (int q = 0; q < 12; ++q) {                                     // 4 channels per 16-byte chunk: 8 chunks in A, 4 in B
          const uint32_t ad = q < 8 ? xa + sw128(r, q) : xb + sw64(r, q - 8);
          const uint4 gw = lds128(ad);
          const float o0 = __uint_as_float(ov[4 * q + 0]) * inv * sigmoid_kit(__uint_as_float(gw.x));
          const float o1 = __uint_as_float(ov[4 * q + 1]) * inv * sigmoid_kit(__uint_as_float(gw.y));
          const float o2 = __uint_as_float(ov[4 * q + 2]) * inv * sigmoid_kit(__uint_as_float(gw.z));
          const float o3 = __uint_as_float(ov[4 * q + 3]) * inv * sigmoid_kit(__uint_as_float(gw.w));
          sts128(ad, make_uint4(__float_as_uint(o0), __float_as_uint(o1), __float_as_uint(o2), __float_as_uint(o3)));
        }
        fence_proxy_async();
        named_bar_sync(1 + w, 128);
        if (r == 0 && a0 + w < A) {
          tma_store_3d(&mqa, xa, head * DH, m0, a0 + w);
          tma_store_3d(&mqb, xb, head * DH + 32, m0, a0 + w);
          tma_store_commit();
        }
      }
    }
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
