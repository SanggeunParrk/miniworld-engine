// attn_fwd_tf32.cu — the augmented pair-bias attention core's TRAINING forward for the fp32 path, sm_100a, TF32 tensor cores (the
// fp32 twin of attn_fwd2.cu; the pipeline of attn_inf_tf32.cu):
//
//   S[a] = q[a] k[a]^T / sqrt(48) + bias      O[a] = softmax(S[a]) v[a]      (fp32 O, row LSE in log2 units)
//
// q, k, v [A L, 768] fp32 (natural units), bias [16, L, L] fp32 (natural units, -inf on masked keys). Products are tcgen05
// kind::tf32 with fp32 accumulation; the softmax, P (rounded to tf32 by the MMA) and the sums are fp32. Persistent CTAs walk
// (sample pair, 128-query tile, head) items as attn_fwd2 does; warpgroup w holds sample a0 + w.
//
//   * q / k tiles are K-major: a 48-wide fp32 row is two TMA boxes, columns 0-31 (128-B swizzle) and 32-47 (64-B swizzle).
//   * v is the MN-major B operand of PV (keys as rows): two 32-column boxes (columns 32-63 of the second: the extra 16 are the
//     next head's, zero past the last) in the 128-B swizzle with 32-B atoms that a tf32 MN-major operand needs (sm100.cuh).
//   * 32-key blocks, 3 stages (k | v of both samples + an fp32 bias tile = 44 KB).
//   * O leaves through one 16 KB staging buffer per warpgroup: columns 0-31 (128-B swizzle), then 32-47 (64-B swizzle) in the
//     same buffer once the first store has read it.
// TMEM: S[w][b] at w * 64 + b * 32, P[w][b] at 128 + w * 64 + b * 32 (fp32), O[w] at 256 + w * 64 (48 columns).
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
constexpr int QA = QM * 128, QB = QM * 64, TQ = QA + QB;                   // q of one sample: 24 KB
constexpr int KA = BN * 128, KB = BN * 64, TV = 2 * BN * 128;              // k A / B, v (two 32-column MN atoms)
constexpr int S_KA = 0, S_KB = 2 * KA, S_V = 2 * KA + 2 * KB, S_BIAS = S_V + 2 * TV;
constexpr int TBIAS = QM * BN * 4, STB = S_BIAS + TBIAS;                  // 44 KB
constexpr int O_Q = 0, O_ST = 2 * TQ, O_BAR = O_ST + ST * STB;
constexpr int O_X = O_BAR + 1024, SMEM_BYTES = O_X + 2 * QA;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert((O_ST % 1024) == 0 && (STB % 1024) == 0 && (S_V % 1024) == 0 && (S_BIAS % 1024) == 0 && (O_X % 1024) == 0,
              "1 KB alignment of the 128-B-swizzled tiles");
constexpr uint32_t T_S = 0, T_P = 128, T_O = 256;
constexpr uint32_t I_QK = idesc_tf32(128, BN), I_PV = idesc_tf32(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f, RSQD = 0.14433756729740643f;

struct Bars {
  uint64_t q_full, q_empty, kv_full[ST], kv_empty[ST], s_full[2][2], p_full[2], p_free[2][2], o_free[2];
  uint32_t tmem;
};
DEVI float max3f(float a, float b, float c) { float d; asm("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }

extern "C" __global__ void __launch_bounds__(384, 1)
augattn_fwd_tf32_sm100(const __grid_constant__ CUtensorMap mqa, const __grid_constant__ CUtensorMap mqb,
                       const __grid_constant__ CUtensorMap mka, const __grid_constant__ CUtensorMap mkb,
                       const __grid_constant__ CUtensorMap mv, const __grid_constant__ CUtensorMap mb,
                       const __grid_constant__ CUtensorMap moa, const __grid_constant__ CUtensorMap mob,
                       float* __restrict__ LSE, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int npair = (A + 1) >> 1, mt = L / QM, nb = L / BN;
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
      mbar_init(&B.p_full[w], 4); mbar_init(&B.p_free[w][0], 1); mbar_init(&B.p_free[w][1], 1); mbar_init(&B.o_free[w], 4);
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
          const int row = min(a0 + w, A - 1) * L + m0;
          tma_load_2d(su + O_Q + w * TQ, &mqa, &B.q_full, qcol, row);
          tma_load_2d(su + O_Q + w * TQ + QA, &mqb, &B.q_full, qcol + 32, row);
        }
        for (int n = 0; n < nb; ++n, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.kv_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.kv_full[s], STB);
          for (int w = 0; w < 2; ++w) {
            const int row = min(a0 + w, A - 1) * L + n * BN;
            tma_load_2d(st + S_KA + w * KA, &mka, &B.kv_full[s], qcol, row);
            tma_load_2d(st + S_KB + w * KB, &mkb, &B.kv_full[s], qcol + 32, row);
            tma_load_2d(st + S_V + w * TV, &mv, &B.kv_full[s], qcol, row);
            tma_load_2d(st + S_V + w * TV + BN * 128, &mv, &B.kv_full[s], qcol + 32, row);
          }
          tma_load_2d(st + S_BIAS, &mb, &B.kv_full[s], n * BN, head * L + m0);
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
      mbar_wait(&B.p_full[w], G & 1);
      if (n == 0 && li >= 1) mbar_wait(&B.o_free[w], (li - 1) & 1);
      tc_fence_after();
      // v: MN-major B, keys as rows (8 keys = 1 KB per K step), two 32-column atoms 4 KB apart
      const uint64_t dv = desc_mn32b(su + O_ST + s * STB + S_V + w * TV, BN * 128);
      const uint32_t pa = tmem + T_P + w * 64 + (G & 1) * 32, oa = tmem + T_O + w * 64;
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < BN / 8; ++ks)
          umma_ts_tf32(oa, pa + ks * 8, dv + (uint64_t)((ks * 1024) >> 4), I_PV, (n > 0 || ks > 0) ? 1u : 0u);
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
    const uint32_t xs = su + O_X + w * QA;
    float m_i = -INFINITY, l_i = 0.f;                                      // exp2 units
    for (int G = 0, li = 0, n = 0; G < nblk; ++G, ++n) {
      if (n == nb) { n = 0; ++li; }
      const int s = G % ST;
      if (n == 0) { m_i = -INFINITY; l_i = 0.f; }
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
          t[4 * q + 0] = fmaf(__uint_as_float(v[4 * q + 0]), RSQD * LOG2E, __uint_as_float(bw.x) * LOG2E);
          t[4 * q + 1] = fmaf(__uint_as_float(v[4 * q + 1]), RSQD * LOG2E, __uint_as_float(bw.y) * LOG2E);
          t[4 * q + 2] = fmaf(__uint_as_float(v[4 * q + 2]), RSQD * LOG2E, __uint_as_float(bw.z) * LOG2E);
          t[4 * q + 3] = fmaf(__uint_as_float(v[4 * q + 3]), RSQD * LOG2E, __uint_as_float(bw.w) * LOG2E);
        }
      }
      if (G >= 1) { tc_fence_before(); }
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
      if (lane == 0) mbar_arrive(&B.p_full[w]);
      if (n == nb - 1) {
        // ---- epilogue of the item: O = acc / l (fp32), LSE = m + log2 l
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
        const bool live = a0 + w < A;
        auto put = [&](int q) {
          return make_uint4(__float_as_uint(__uint_as_float(ov[4 * q]) * inv), __float_as_uint(__uint_as_float(ov[4 * q + 1]) * inv),
                            __float_as_uint(__uint_as_float(ov[4 * q + 2]) * inv), __float_as_uint(__uint_as_float(ov[4 * q + 3]) * inv));
        };
        if (r == 0) tma_store_wait_read0();                                // the previous item's second store has left the buffer
        named_bar_sync(1 + w, 128);
#pragma unroll
        for (int q = 0; q < 8; ++q) sts128(xs + sw128(r, q), put(q));     // columns 0-31
        fence_proxy_async();
        named_bar_sync(1 + w, 128);
        if (r == 0) {
          if (live) { tma_store_2d(&moa, xs, head * DH, (a0 + w) * L + m0); tma_store_commit(); }
          tma_store_wait_read0();
        }
        named_bar_sync(1 + w, 128);
#pragma unroll
        for (int q = 0; q < 4; ++q) sts128(xs + sw64(r, q), put(8 + q));  // columns 32-47, same buffer
        fence_proxy_async();
        named_bar_sync(1 + w, 128);
        if (r == 0 && live) { tma_store_2d(&mob, xs, head * DH + 32, (a0 + w) * L + m0); tma_store_commit(); }
        if (live) LSE[((size_t)(a0 + w) * 16 + head) * L + m0 + r] = m_i + __log2f(l_i);
      }
    }
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
