// attn_fwd.cu — the atom DiT's pair-bias attention forward, sm_100a: the token-DiT B200 forward (augattn_sm100/attn_fwd2.cu) at atom
// shapes, 4 heads x 32 over N atoms, A samples sharing one bias [4, N, N] (from pair_bias.cu).
//
//   S[a] = q[a] k[a]^T / sqrt(32) + bias      O[a] = softmax(S[a]) v[a]      (bf16 O [A N, 128], row LSE [A, 4, N] in log2 units)
//
// Persistent CTAs walk (sample pair, 128-query tile, head) items; two softmax warpgroups (one per sample of the pair) share the bias tile;
// S, P and O in TMEM; 64-key blocks; fp32x2 softmax with a lazy running max. The two warpgroups' P phases take turns (ALT): while one
// exponentiates, the other loads its next S and bias and takes the row maximum (in lockstep both idled the MUFU for ~half a block).
// q / K / V tiles are dense 64-B rows (SW64), the bias [128 queries][64 keys] SW128. An odd A computes a duplicate of the last sample in
// the spare warpgroup and does not store it.
// TMEM: S[w][b] at w * 128 + b * 64 (64 cols), P[w] at 256 + w * 32 (bf16), O[w] at 320 + w * 64 (32 cols).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

DEVI void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
#ifndef LAZY
#define LAZY 5.545177444479562f          // natural units (8 in log2 units)
#endif
#ifndef ST
#define ST 5
#endif
#ifndef ALT
#define ALT 1                            // the two warpgroups' exponential (P) phases take turns (named barriers 3 / 4)
#endif
#ifndef QR
#define QR 1                             // q ring slots
#endif
constexpr int BN = 64, DH = 32, QM = 128, DM = 128, NH = 4;
constexpr int TQ = QM * 64, TK = BN * 64, TB = QM * BN * 2;                // q / K / V: dense 64-B rows (SW64); bias [128 q][64 keys] SW128
constexpr int STB = 4 * TK + TB;                                           // 32 KB
constexpr int O_Q = 0, O_ST = QR * 2 * TQ, O_BAR = O_ST + ST * STB;        // q ring: QR slots x (q0 | q1)
constexpr int XA = 128 * 64;                                               // O staging per warpgroup: [128 rows][32 bf16] (SW64)
constexpr int O_X = O_BAR + 1024, SMEM_BYTES = O_X + 2 * XA;
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
                   const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap moa,
                   float* __restrict__ LSE, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int npair = (A + 1) >> 1, mt = L / QM, nb = L / BN;
  const int items = npair * mt * NH;
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
        for (int w = 0; w < 2; ++w) tma_load_2d(su + O_Q + qs * 2 * TQ + w * TQ, &mq, &B.q_full[qs], qcol, min(a0 + w, A - 1) * L + m0);
        for (int n = 0; n < nb; ++n, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.kv_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.kv_full[s], 4 * BN * DH * 2 + TB);
          for (int w = 0; w < 2; ++w) {
            tma_load_2d(st + w * TK, &mk, &B.kv_full[s], qcol, min(a0 + w, A - 1) * L + n * BN);
            tma_load_2d(st + 2 * TK + w * TK, &mv, &B.kv_full[s], qcol, min(a0 + w, A - 1) * L + n * BN);
          }
          tma_load_2d(st + 4 * TK, &mb, &B.kv_full[s], n * BN, head * L + m0);
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
      const uint64_t dq = desc_sw64(su + O_Q + qs * 2 * TQ + w * TQ), dk = desc_sw64(su + O_ST + s * STB + w * TK);
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
      const uint64_t dv = desc_sw64(su + O_ST + s * STB + 2 * TK + w * TK);                // MN-major: 16 keys = 1 KB a step
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ts(tmem + T_O + w * 64, tmem + T_P + w * 32 + ks * 8, dv + (uint64_t)(ks * 64), I_PV, (n > 0 || ks > 0) ? 1u : 0u);
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
    const f2 CQ = mk2(0.17677669529663687f, 0.17677669529663687f), L2E = mk2(LOG2E, LOG2E);   // 1 / sqrt(32)
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
      if (ALT && (w == 1 || G > 0)) named_bar_sync(w == 0 ? 4 : 3, 256);   // the other warpgroup's P of this / the previous block is done
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
      if (ALT && (w == 0 || G + 1 < nblk)) named_bar_arrive(w == 0 ? 3 : 4, 256);
      if (w == 0 && r == 0) TR(5, G);
      if (w == 1 && r == 0) TR(7, G);
      if (n == nb - 1) {
        // ---- epilogue of the item: O = rn(acc / l) (bf16), LSE = m log2 e + log2 l
        int a0, m0, head; item_of(li, a0, m0, head);
        mbar_wait(&B.p_free[w], G & 1);
        tc_fence_after();
        const float inv = 1.f / l_i;
        const int row = m0 + (int)r, a = a0 + w;
        uint32_t ov[32];
        tmem_ld32(trow + T_O + w * 64, ov);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.o_free[w]);
        if (a < A) {                                                       // (warpgroup-uniform) the spare sample of an odd A is not stored
          const uint32_t xa = su + O_X + w * XA;                           // O (bf16) leaves through a staging tile and a TMA store
          if (r == 0) tma_store_wait_read0();                              // the previous item's store has left the staging buffer
          named_bar_sync(1 + w, 128);
#pragma unroll
          for (int q = 0; q < 4; ++q)
            sts128(xa + sw64(r, q), make_uint4(pack_bf16(__uint_as_float(ov[8 * q]) * inv, __uint_as_float(ov[8 * q + 1]) * inv),
                                               pack_bf16(__uint_as_float(ov[8 * q + 2]) * inv, __uint_as_float(ov[8 * q + 3]) * inv),
                                               pack_bf16(__uint_as_float(ov[8 * q + 4]) * inv, __uint_as_float(ov[8 * q + 5]) * inv),
                                               pack_bf16(__uint_as_float(ov[8 * q + 6]) * inv, __uint_as_float(ov[8 * q + 7]) * inv)));
          fence_proxy_async();
          named_bar_sync(1 + w, 128);
          if (r == 0) { tma_store_2d(&moa, xa, head * DH, a * L + m0); tma_store_commit(); }
        }
        if (a < A) LSE[((size_t)a * NH + head) * L + row] = m_i * LOG2E + __log2f(l_i);
      }
    }
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
