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
// -DQPAIR=1 (AttentionPairBias, one sample, -DNHEAD heads, O rows NHEAD * 48 wide): a work item is (head, pair of 128-query
// tiles m0, m0 + 128), warpgroup w takes tile m0 + 128 w; both read the same k / v blocks, each its own bias tile; the partner
// past the last tile recomputes the last one and does not store. A is 1.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef PP
#define PP 1                             // FA4-style ping-pong: the two softmax warpgroups take turns on the exponentials (MUFU), so one
#endif                                   // warpgroup's exp phase runs under the other's S load / bias / max / P store (named bars 3, 4)
#ifndef LAZY
#define LAZY 5.545177444479562f          // natural units (8 in log2 units)
#endif
#ifndef TST
#define TST 1                            // O leaves through smem-staged TMA bulk stores (coalesced) instead of per-row st.v4
#endif
#ifndef ST
#define ST 3
#endif
#ifndef QR
#define QR 1                             // q ring slots
#endif
#ifndef DHP
#define DHP 48                           // head width in memory and in the MMAs: 48 (token DiT 16 x 48), 64, 32, 16; 16 x 24: 32 (the
#endif                                   // projection pads each head's 24 channels with 8 zero ones, so q / k / v / dO pads are exactly 0)
#ifndef RSQDV
#define RSQDV 0.14433756729740643f       // 1 / sqrt(real head dim): 1 / sqrt 48; 1 / sqrt 24 = 0.2041241452319315f; 1 / sqrt 16 = 0.25f; 1 / sqrt 32, 1 / sqrt 64 = 0.125f
#endif
static_assert(DHP == 64 || DHP == 48 || DHP == 32 || DHP == 16, "head width 64, 48, 32 or 16");
constexpr int BN = 64, DH = DHP, QM = 128, DM = 768;
constexpr int TQ = QM * 128, TK = BN * 128, TB = QM * BN * 2;
#ifndef QPAIR
#define QPAIR 0
#endif
#ifndef NHEAD
#define NHEAD 16
#endif
// stage: k0 | k1 | v0 | v1 | bias (sample pairs) or k | v | bias0 | bias1 (query-tile pairs), 48 KB either way
constexpr int STB = QPAIR ? 2 * TK + 2 * TB : 4 * TK + TB;
__host__ __device__ constexpr int K_OFF(int w) { return QPAIR ? 0 : w * TK; }
__host__ __device__ constexpr int V_OFF(int w) { return QPAIR ? TK : 2 * TK + w * TK; }
__host__ __device__ constexpr int B_OFF(int w) { return QPAIR ? 2 * TK + w * TB : 4 * TK; }
constexpr int O_Q = 0, O_ST = QR * 2 * TQ, O_BAR = O_ST + ST * STB;        // q ring: QR slots x (q0 | q1)
constexpr int XA = 128 * 32 * 4, XB = 128 * 16 * 4;                        // O staging per warpgroup: cols 0-31 (SW128), 32-47 (SW64)
constexpr int O_X = O_BAR + 1024, SMEM_BYTES = O_X + (TST ? 2 * (XA + XB) : 0);
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
                   const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap moa, const __grid_constant__ CUtensorMap mob,
                   float* __restrict__ O, float* __restrict__ LSE, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int npair = A >> 1, mt = L / QM, nb = L / BN;
  const int items = QPAIR ? ((mt + 1) >> 1) * NHEAD : npair * mt * NHEAD;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int nblk = my_items * nb;                                          // this CTA's key blocks, all items
  auto item_of = [&](int li, int& a0, int& m0, int& head) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    if (QPAIR) { a0 = 0; head = wi % NHEAD; m0 = (wi / NHEAD) * 2 * QM; return; }   // m0: the pair's first tile
    const int pair = wi % npair, cid = wi / npair;
    a0 = 2 * pair; m0 = (cid % mt) * QM; head = cid / mt;
  };
  // warpgroup w's first row in the [A L] rows (its tile, clamped for the missing partner / its sample), and whether it stores
  auto row_of = [&](int a0, int m0, int w) { return QPAIR ? min(m0 + w * QM, (mt - 1) * QM) : (a0 + w) * L + m0; };
  auto live = [&](int m0, int w) { return !QPAIR || m0 + w * QM < L; };

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
        for (int w = 0; w < 2; ++w) tma_load_2d(su + O_Q + qs * 2 * TQ + w * TQ, &mq, &B.q_full[qs], qcol, row_of(a0, m0, w));
        for (int n = 0; n < nb; ++n, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.kv_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.kv_full[s], QPAIR ? 2 * BN * DH * 2 + 2 * TB : 4 * BN * DH * 2 + TB);
          for (int w = 0; w < 2; ++w) {
            if (QPAIR && w == 1) break;                                   // one k / v for both query tiles
            tma_load_2d(st + K_OFF(w), &mk, &B.kv_full[s], qcol, (QPAIR ? 0 : (a0 + w) * L) + n * BN);
            tma_load_2d(st + V_OFF(w), &mv, &B.kv_full[s], qcol, (QPAIR ? 0 : (a0 + w) * L) + n * BN);
          }
          for (int w = 0; w < (QPAIR ? 2 : 1); ++w)
            tma_load_2d(st + B_OFF(w), &mb, &B.kv_full[s], n * BN, head * L + (QPAIR ? row_of(a0, m0, w) : m0));
          TR(0, g);
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // one MMA warp per softmax warpgroup (warp 1: sample a0, warp 2: a0 + 1); QK(G + 2) follows PV(G): S double-buffered, P single
    const int w = warp - 1;
    auto qk = [&](int G, int li, int n) {                                  // QK of block G (item li, key block n)
      const int s = G % ST, qs = li % QR;
      if (n == 0) mbar_wait(&B.q_full[qs], (li / QR) & 1);
      mbar_wait(&B.kv_full[s], (G / ST) & 1);
      tc_fence_after();
      const uint64_t dq = desc_k128(su + O_Q + qs * 2 * TQ + w * TQ), dk = desc_k128(su + O_ST + s * STB + K_OFF(w));
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks) umma_ss(tmem + T_S + w * 128 + (G & 1) * 64, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_QK, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w][G & 1]);
        if (n == nb - 1) tc_commit(&B.q_empty[qs]);
      }
      __syncwarp();
      if (w == 0 && lane == 0) TR(1, G);
    };
    auto step2 = [&](int li, int n, int& li2, int& n2) { li2 = li; n2 = n + 2; while (n2 >= nb) { n2 -= nb; ++li2; } };
    for (int G = 0; G < 2 && G < nblk; ++G) qk(G, G / nb, G % nb);
    for (int G = 0, li = 0, n = 0; G < nblk; ++G) {                        // (li, n) advance incrementally
      const int s = G % ST;
      mbar_wait(&B.p_full[w], G & 1);
      if (n == 0 && li >= 1) mbar_wait(&B.o_free[w], (li - 1) & 1);        // the previous item's O has been read out
      tc_fence_after();
      const uint64_t dv = desc_mn128(su + O_ST + s * STB + V_OFF(w), 8192);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ts(tmem + T_O + w * 64, tmem + T_P + w * 32 + ks * 8, dv + (uint64_t)(ks * 2048 >> 4), I_PV, (n > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.p_free[w]);
        tc_commit(&B.kv_empty[s]);
      }
      __syncwarp();
      if (w == 0 && lane == 0) TR(6, G);
      if (G + 2 < nblk) { int li2, n2; step2(li, n, li2, n2); qk(G + 2, li2, n2); }
      if (++n == nb) { n = 0; ++li; }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------ softmax of sample a0 + w, one query row per thread
    setmaxnreg_inc<224>();
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    if (PP && w == 1) named_bar_arrive(3, 256);                          // warpgroup 0 takes the first exp turn
    const f2 CQ = mk2(RSQDV, RSQDV), L2E = mk2(LOG2E, LOG2E);
    float m_i = -INFINITY, l_i = 0.f;
    for (int G = 0, li = 0, n = 0; G < nblk; ++G, ++n) {
      if (n == nb) { n = 0; ++li; }
      const int s = G % ST;
      if (n == 0) { m_i = -INFINITY; l_i = 0.f; }
      mbar_wait(&B.kv_full[s], (G / ST) & 1);                              // the bias tile of this block
      mbar_wait(&B.s_full[w][G & 1], (G >> 1) & 1);
      if (w == 0 && r == 0) TR(2, G);
      tc_fence_after();
      f2 t[BN / 2];
      const uint32_t sb = su + O_ST + s * STB + B_OFF(w);
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
          for (int cc = 0; cc < DH / 16; ++cc) {
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
      if (PP) named_bar_sync(3 + w, 256);                                  // my exp turn: the other warpgroup's exps are done
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
      if (PP) named_bar_arrive(4 - w, 256);                                // hand the MUFU to the other warpgroup
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
        const int a = QPAIR ? 0 : a0 + w, rbase = row_of(a0, m0, w) - a * L, row = rbase + (int)r;   // row within sample a
        float* orow = O + ((size_t)a * L + row) * (NHEAD * DH) + head * DH;
        uint32_t ov[DH];
#pragma unroll
        for (int cc = 0; cc < DH / 16; ++cc) tmem_ld16(trow + T_O + w * 64 + cc * 16, *reinterpret_cast<uint32_t(*)[16]>(ov + 16 * cc));
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.o_free[w]);
        if (TST && DH == 64) {                                             // 64 columns: two 32-column halves through xa, in turn
          const uint32_t xa = su + O_X + w * (XA + XB);
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            if (r == 0) tma_store_wait_read0();                            // the previous store has left the staging buffer
            named_bar_sync(1 + w, 128);
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const int k = 32 * hf + 4 * q;
              sts128(xa + sw128(r, q), make_uint4(__float_as_uint(__uint_as_float(ov[k]) * inv), __float_as_uint(__uint_as_float(ov[k + 1]) * inv),
                                                  __float_as_uint(__uint_as_float(ov[k + 2]) * inv), __float_as_uint(__uint_as_float(ov[k + 3]) * inv)));
            }
            fence_proxy_async();
            named_bar_sync(1 + w, 128);
            if (r == 0 && live(m0, w)) {
              tma_store_2d(&moa, xa, head * DH + 32 * hf, a * L + rbase);
              tma_store_commit();
            }
          }
        } else if (TST) {
          const uint32_t xa = su + O_X + w * (XA + XB), xb = xa + XA;
          if (r == 0) tma_store_wait_read0();                              // the previous item's store has left the staging buffer
          named_bar_sync(1 + w, 128);
#pragma unroll
          for (int q = 0; q < DH / 4; ++q) {
            const uint4 u = make_uint4(__float_as_uint(__uint_as_float(ov[4 * q]) * inv), __float_as_uint(__uint_as_float(ov[4 * q + 1]) * inv),
                                       __float_as_uint(__uint_as_float(ov[4 * q + 2]) * inv), __float_as_uint(__uint_as_float(ov[4 * q + 3]) * inv));
            if (DH == 16) sts128(xb + sw64(r, q), u);                       // 16 columns: the 64-B box alone
            else if (q < 8) sts128(xa + sw128(r, q), u); else sts128(xb + sw64(r, q - 8), u);
          }
          fence_proxy_async();
          named_bar_sync(1 + w, 128);
          if (r == 0 && live(m0, w)) {
            if (DH == 16) tma_store_2d(&mob, xb, head * DH, a * L + rbase);
            else tma_store_2d(&moa, xa, head * DH, a * L + rbase);
            if (DH > 32) tma_store_2d(&mob, xb, head * DH + 32, a * L + rbase);
            tma_store_commit();
          }
        } else if (live(m0, w)) {
#pragma unroll
          for (int k = 0; k < DH / 4; ++k)
            *reinterpret_cast<float4*>(orow + 4 * k) = make_float4(__uint_as_float(ov[4 * k]) * inv, __uint_as_float(ov[4 * k + 1]) * inv,
                                                                   __uint_as_float(ov[4 * k + 2]) * inv, __uint_as_float(ov[4 * k + 3]) * inv);
        }
        if (live(m0, w)) LSE[((size_t)a * NHEAD + head) * L + row] = m_i * LOG2E + __log2f(l_i);
      }
    }
    if (TST && r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
