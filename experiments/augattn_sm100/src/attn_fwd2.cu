// attn_fwd2.cu — the augmented pair-bias attention core's TRAINING forward, sm_100a, PERSISTENT (v2 of attn_fwd.cu):
//
//   S[a] = q[a] k[a]^T / sqrt(48) + bias      O[a] = softmax(S[a]) v[a]      (fp32 O, row LSE in log2 units)
//
// Same arithmetic contract as attn_fwd.cu (bf16 operands, fp32 sums, the bias read as given, natural-unit logits with the log2 e
// scale folded into the exponent's FFMA, lazy running max, P rounded to bf16 for the tensor core). What changes:
//   * persistent CTAs walk (sample pair, query tile, head) work items; the TMA warp streams the next item's q and key blocks while
//     the current item is computed (no per-CTA prologue on the critical path)
//   * 64-key blocks, 3 stages (K0 | K1 | V0 | V1 | bias = 48 KB), a 2-slot q ring
//   * two S buffers per softmax warpgroup: QK of block n + 2 is issued as soon as block n's P is handed over
//   * the softmax in packed fp32x2 (FFMA2 / FADD2) with 3-input max
// TMEM: S[w][b] at w * 128 + b * 64 (64 cols), P[w] at 256 + w * 32 (bf16, 32 cols), O[w] at 320 + w * 64 (48 cols).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef LAZY
#define LAZY 5.545177444479562f          // natural units (8 in log2 units)
#endif
#ifndef ST
#define ST 3
#endif
#ifndef QR
#define QR 2                             // q ring slots
#endif
constexpr int BN = 64, DH = 48, QM = 128, DM = 768;
constexpr int TQ = QM * 128, TK = BN * 128, TB = QM * BN * 2;
constexpr int STB = 4 * TK + TB;                                           // 48 KB
constexpr int O_Q = 0, O_ST = QR * 2 * TQ, O_BAR = O_ST + ST * STB;        // q ring: QR slots x (q0 | q1)
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_S = 0, T_P = 256, T_O = 320;
constexpr uint32_t I_QK = idesc_bf16(128, BN), I_PV = idesc_bf16(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f;

#ifdef TRACE
__device__ unsigned long long g_tr[8][256];   // CTA 0: 0 prod issued, 1 mma QK(w0) issued, 2 sm0 s_full seen, 3 sm0 max done, 4 sm0 p_free seen, 5 sm0 P done, 6 PV(w0) issued, 7 sm1 P done
#define TR(ev, i) do { if (blockIdx.x == 0 && (i) < 256) g_tr[ev][i] = clock64(); } while (0)
#else
#define TR(ev, i) do { } while (0)
#endif
struct Bars {
  uint64_t q_full[QR], q_empty[QR], kv_full[ST], kv_empty[ST], s_full[2][2], p_full[2], p_free[2], o_free[2];
  uint32_t tmem;
};
#ifndef PMOD
#define PMOD 4                           // of every PMOD exponential pairs, PCNT run on the FMA pipe (polynomial), the rest on MUFU
#endif
#ifndef PCNT
#define PCNT 0
#endif
// 2^x for a pair on the FMA pipe: x = r + f (r = round(x), |f| <= 1/2), 2^f by a degree-3 fit (max rel. error 8.8e-5: below the
// bf16 rounding of P), the exponent added to the bit pattern; x clamped to [-126, 126]
DEVI f2 ex2_poly2(f2 x) {
  const float x0 = fminf(fmaxf(lo2(x), -126.f), 126.f), x1 = fminf(fmaxf(hi2(x), -126.f), 126.f);
  const f2 xc = mk2(x0, x1), C = mk2(12582912.f, 12582912.f);           // 1.5 * 2^23: round-to-nearest into the low mantissa bits
  const f2 j = add2(xc, C);
  const f2 f = add2(xc, neg2(add2(j, neg2(C))));
  f2 p = fma2(mk2(0.0555041086648216f, 0.0555041086648216f), f, mk2(0.2402264923172785f, 0.2402264923172785f));
  p = fma2(p, f, mk2(0.6931471805599453f, 0.6931471805599453f));
  p = fma2(p, f, mk2(1.0f, 1.0f));
  return mk2(__int_as_float(__float_as_int(lo2(p)) + (__float_as_int(lo2(j)) << 23)), __int_as_float(__float_as_int(hi2(p)) + (__float_as_int(hi2(j)) << 23)));
}
DEVI float max3f(float a, float b, float c) { float d; asm("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }

extern "C" __global__ void __launch_bounds__(384, 1)
augattn_fwd2_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                   const __grid_constant__ CUtensorMap mb, float* __restrict__ O, float* __restrict__ LSE, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int npair = A >> 1, mt = L / QM, nb = L / BN;
  const int items = npair * mt * 16;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int nblk = my_items * nb;                                          // this CTA's key blocks, all items
  auto item_of = [&](int li, int& a0, int& m0, int& head) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    const int pair = wi % npair, cid = wi / npair;
    a0 = 2 * pair; m0 = (cid % mt) * QM; head = cid / mt;
  };

  if (tid == 0) {
    for (int s = 0; s < QR; ++s) { mbar_init(&B.q_full[s], 1); mbar_init(&B.q_empty[s], 2); }   // released by both MMA warps
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
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        int a0, m0, head; item_of(li, a0, m0, head);
        const int qs = li % QR, qcol = head * DH;
        if (li >= QR) mbar_wait(&B.q_empty[qs], ((li / QR) - 1) & 1);
        mbar_expect_tx(&B.q_full[qs], 2 * QM * DH * 2);
        for (int w = 0; w < 2; ++w) tma_load_2d(su + O_Q + qs * 2 * TQ + w * TQ, &mq, &B.q_full[qs], qcol, (a0 + w) * L + m0);
        for (int n = 0; n < nb; ++n, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.kv_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.kv_full[s], 4 * BN * DH * 2 + TB);
          for (int w = 0; w < 2; ++w) {
            tma_load_2d(st + w * TK, &mk, &B.kv_full[s], qcol, (a0 + w) * L + n * BN);
            tma_load_2d(st + 2 * TK + w * TK, &mv, &B.kv_full[s], qcol, (a0 + w) * L + n * BN);
          }
          tma_load_2d(st + 4 * TK, &mb, &B.kv_full[s], n * BN, head * L + m0);
          TR(0, g);
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // one MMA warp per softmax warpgroup (warp 1: sample a0, warp 2: a0 + 1); QK(G + 2) follows PV(G): S double-buffered, P single
    const int w = warp - 1;
    auto qk = [&](int G) {
      const int li = G / nb, n = G % nb, s = G % ST, qs = li % QR;
      if (n == 0) mbar_wait(&B.q_full[qs], (li / QR) & 1);
      mbar_wait(&B.kv_full[s], (G / ST) & 1);
      tc_fence_after();
      const uint64_t dq = desc_k128(su + O_Q + qs * 2 * TQ + w * TQ), dk = desc_k128(su + O_ST + s * STB + w * TK);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 3; ++ks) umma_ss(tmem + T_S + w * 128 + (G & 1) * 64, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_QK, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w][G & 1]);
        if (n == nb - 1) tc_commit(&B.q_empty[qs]);
      }
      __syncwarp();
      if (w == 0 && lane == 0) TR(1, G);
    };
    for (int G = 0; G < 2 && G < nblk; ++G) qk(G);
    for (int G = 0; G < nblk; ++G) {
      const int li = G / nb, n = G % nb, s = G % ST;
      mbar_wait(&B.p_full[w], G & 1);
      if (n == 0 && li >= 1) mbar_wait(&B.o_free[w], (li - 1) & 1);        // the previous item's O has been read out
      tc_fence_after();
      const uint64_t dv = desc_mn128(su + O_ST + s * STB + 2 * TK + w * TK, 8192);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ts(tmem + T_O + w * 64, tmem + T_P + w * 32 + ks * 8, dv + (uint64_t)(ks * 2048 >> 4), I_PV, (n > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.p_free[w]);
        tc_commit(&B.kv_empty[s]);
      }
      __syncwarp();
      if (w == 0 && lane == 0) TR(6, G);
      if (G + 2 < nblk) qk(G + 2);
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------ softmax of sample a0 + w, one query row per thread
    setmaxnreg_inc<224>();
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const f2 CQ = mk2(0.14433756729740643f, 0.14433756729740643f), L2E = mk2(LOG2E, LOG2E);
    float m_i = -INFINITY, l_i = 0.f;
    for (int G = 0; G < nblk; ++G) {
      const int li = G / nb, n = G % nb, s = G % ST;
      if (n == 0) { m_i = -INFINITY; l_i = 0.f; }
      mbar_wait(&B.kv_full[s], (G / ST) & 1);                              // the bias tile of this block
      mbar_wait(&B.s_full[w][G & 1], (G >> 1) & 1);
      if (w == 0 && r == 0) TR(2, G);
      tc_fence_after();
      f2 t[BN / 2];
      const uint32_t sb = su + O_ST + s * STB + 4 * TK;
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        uint32_t v[32];
        tmem_ld32(trow + T_S + w * 128 + (G & 1) * 64 + cc * 32, v);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 4; ++q) {                                      // 8 keys of bias per 16-byte chunk
          const uint4 bw = lds128(sb + sw128(r, cc * 4 + q));
          const uint32_t bb[4] = {bw.x, bw.y, bw.z, bw.w};
#pragma unroll
          for (int e = 0; e < 4; ++e)
            t[cc * 16 + q * 4 + e] = fma2(mk2u(v[q * 8 + 2 * e], v[q * 8 + 2 * e + 1]), CQ, mk2(bf16lo(bb[e]), bf16hi(bb[e])));
        }
      }
      if (G >= 1) { tc_fence_before(); }
      float mxp[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};         // four independent chains
#pragma unroll
      for (int j = 0; j < BN / 2; ++j) mxp[j & 3] = max3f(mxp[j & 3], lo2(t[j]), hi2(t[j]));
      const float mx = max3f(mxp[0], mxp[1], fmaxf(mxp[2], mxp[3]));
      const float m_new = mx > m_i + LAZY ? mx : m_i;
      if (w == 0 && r == 0) TR(3, G);
      if (G >= 1) mbar_wait(&B.p_free[w], (G - 1) & 1);                    // PV(G - 1) done: O final for G - 1, P free
      if (w == 0 && r == 0) TR(4, G);
      tc_fence_after();
      if (__any_sync(0xffffffffu, m_new != m_i)) {
        const float alpha = ex2f((m_i - m_new) * LOG2E);
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
      const f2 NM = mk2(-m_i * LOG2E, -m_i * LOG2E);
      f2 ssp[4] = {mk2(0.f, 0.f), mk2(0.f, 0.f), mk2(0.f, 0.f), mk2(0.f, 0.f)};
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {                                     // 32 keys -> 16 packed P columns
        uint32_t pk[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const f2 e = fma2(t[cc * 16 + k], L2E, NM);
          float p0, p1;
          if ((k % PMOD) < PCNT) { const f2 pp = ex2_poly2(e); p0 = lo2(pp); p1 = hi2(pp); }
          else { p0 = ex2f(lo2(e)); p1 = ex2f(hi2(e)); }
          ssp[k & 3] = add2(ssp[k & 3], mk2(p0, p1));
          pk[k] = pack_bf16(p0, p1);
        }
        tmem_st16(trow + T_P + w * 32 + cc * 16, pk);
      }
      const f2 ss = add2(add2(ssp[0], ssp[1]), add2(ssp[2], ssp[3]));
      l_i += lo2(ss) + hi2(ss);
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.p_full[w]);
      if (w == 0 && r == 0) TR(5, G);
      if (w == 1 && r == 0) TR(7, G);
      if (n == nb - 1) {
        // ---- epilogue of the item: O = acc / l (fp32), LSE = m log2 e + log2 l
        int a0, m0, head; item_of(li, a0, m0, head);
        mbar_wait(&B.p_free[w], G & 1);
        tc_fence_after();
        const float inv = 1.f / l_i;
        const int row = m0 + (int)r, a = a0 + w;
        float* orow = O + ((size_t)a * L + row) * DM + head * DH;
#pragma unroll
        for (int cc = 0; cc < 3; ++cc) {
          uint32_t ov[16];
          tmem_ld16(trow + T_O + w * 64 + cc * 16, ov);
          tmem_wait_ld();
#pragma unroll
          for (int k = 0; k < 4; ++k)
            *reinterpret_cast<float4*>(orow + cc * 16 + 4 * k) = make_float4(__uint_as_float(ov[4 * k]) * inv, __uint_as_float(ov[4 * k + 1]) * inv,
                                                                              __uint_as_float(ov[4 * k + 2]) * inv, __uint_as_float(ov[4 * k + 3]) * inv);
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.o_free[w]);
        LSE[((size_t)a * 16 + head) * L + row] = m_i * LOG2E + __log2f(l_i);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
