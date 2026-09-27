// attn_inf2.cu — the token DiT's INFERENCE attention core, sm_100a, v2: the KEYS of one (sample, 128 queries, head) item are split
// over the two softmax warpgroups (warpgroup w takes the 64-key blocks n = w, w + 2, ...), each keeping its own online softmax
// (m, l, O), and the two partial results are merged at the end through shared memory:
//
//   o = sigmoid(g) * softmax(q k^T + bias) v      written as bf16 OVER q (the tdit step contract)
//
// Against attn_inf.cu (two samples per item, one per warpgroup): an item is one sample, so there are A / 2 * 2 more items for the
// same work and half the key blocks per warpgroup -- the right trade at the step's small S (5), where attn_inf ran about one
// item per CTA and was latency-bound; odd S needs no dummy partner. The bias tile is no longer shared by two samples on chip
// (sample is the fastest item index, so the S CTAs reading one bias tile run together and share it through L2).
// Logits arrive in exp2 units (sm_scale log2 e folded into Wq / bq, log2 e into the hoisted bias). No LSE.
// TMEM: S[w][b] at w * 128 + b * 64 (64 cols), P[w] at 256 + w * 32 (bf16, 32 cols), O[w] at 320 + w * 64 (48 cols).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef LAZY
#define LAZY 8.0f                        // log2 units
#endif
#ifndef ST
#define ST 5
#endif
DEVI void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
constexpr int BN = 64, DH = 48, QM = 128, DM = 768;
constexpr int TQ = QM * 128, TK = BN * 128, TB = QM * BN * 2;
constexpr int STB = 2 * TK + TB;                                           // K | V | bias = 32 KB
constexpr int O_Q = 0, O_ST = TQ, O_BAR = O_ST + ST * STB;
constexpr int XG = 128 * 128;                                              // warpgroup 0: g tile in, gated o out (bf16, SW128)
constexpr int NXC = 50;                                                    // partial exchange per row: m, l, O[48] (fp32)
constexpr int O_X = O_BAR + 1024, O_XC = O_X + XG, SMEM_BYTES = O_XC + NXC * 128 * 4;   // one exchange buffer (bar 3 handshake)
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_S = 0, T_P = 256, T_O = 320;
constexpr uint32_t I_QK = idesc_bf16(128, BN), I_PV = idesc_bf16(128, DH, 0, 1);

struct Bars {
  uint64_t q_full, q_empty, kv_full[ST], kv_empty[ST], s_full[2][2], p_full[2], p_free[2], o_free[2], g_full;
  uint32_t tmem;
};
DEVI float max3f(float a, float b, float c) { float d; asm("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }

extern "C" __global__ void __launch_bounds__(384, 1)
augattn_inf2_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                   const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mo,
                   int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = L / QM, nb = L / BN, nbh = nb >> 1;                        // nb is even (L % 128 == 0)
  const int items = A * mt * 16;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int nown = my_items * nbh;                                         // key blocks per warpgroup, all items
  auto item_of = [&](int li, int& a, int& m0, int& head) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    a = wi % A; const int cid = wi / A;                                    // sample fastest: the S readers of a bias tile run together
    m0 = (cid % mt) * QM; head = cid / mt;
  };

  if (tid == 0) {
    mbar_init(&B.q_full, 1); mbar_init(&B.q_empty, 2);                     // q is read by both MMA warps
    for (int s = 0; s < ST; ++s) { mbar_init(&B.kv_full[s], 1); mbar_init(&B.kv_empty[s], 1); }   // a stage has one owner
    for (int w = 0; w < 2; ++w) {
      mbar_init(&B.s_full[w][0], 1); mbar_init(&B.s_full[w][1], 1);
      mbar_init(&B.p_full[w], 4); mbar_init(&B.p_free[w], 1); mbar_init(&B.o_free[w], 4);
    }
    mbar_init(&B.g_full, 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    // ------------------------------------------------------------------------------------ TMA producer: q per item, K | V | bias per block
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        int a, m0, head; item_of(li, a, m0, head);
        const int qcol = head * DH;
        if (li >= 1) mbar_wait(&B.q_empty, (li - 1) & 1);
        mbar_expect_tx(&B.q_full, QM * DH * 2);
        tma_load_2d(su + O_Q, &mq, &B.q_full, qcol, a * L + m0);
        for (int n = 0; n < nb; ++n, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.kv_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.kv_full[s], 2 * BN * DH * 2 + TB);
          tma_load_2d(st, &mk, &B.kv_full[s], qcol, a * L + n * BN);
          tma_load_2d(st + TK, &mv, &B.kv_full[s], qcol, a * L + n * BN);
          tma_load_2d(st + 2 * TK, &mb, &B.kv_full[s], n * BN, head * L + m0);
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------ MMA warp w serves warpgroup w's key blocks
    // own block j -> item li = j / nbh, block n = 2 (j % nbh) + w, global block G = li nb + n (the producer's stage order)
    const int w = warp - 1;
    auto qk = [&](int j) {
      const int li = j / nbh, jj = j - li * nbh, G = li * nb + 2 * jj + w, s = G % ST;
      if (jj == 0) mbar_wait(&B.q_full, li & 1);
      mbar_wait(&B.kv_full[s], (G / ST) & 1);
      tc_fence_after();
      const uint64_t dq = desc_k128(su + O_Q), dk = desc_k128(su + O_ST + s * STB);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 3; ++ks) umma_ss(tmem + T_S + w * 128 + (j & 1) * 64, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_QK, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w][j & 1]);
        if (jj == nbh - 1) tc_commit(&B.q_empty);
      }
      __syncwarp();
    };
    for (int j = 0; j < 2 && j < nown; ++j) qk(j);
    for (int j = 0, li = 0, jj = 0; j < nown; ++j) {
      const int G = li * nb + 2 * jj + w, s = G % ST;
      mbar_wait(&B.p_full[w], j & 1);
      if (jj == 0 && li >= 1) mbar_wait(&B.o_free[w], (li - 1) & 1);      // the previous item's O has been read out
      tc_fence_after();
      const uint64_t dv = desc_mn128(su + O_ST + s * STB + TK, 8192);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ts(tmem + T_O + w * 64, tmem + T_P + w * 32 + ks * 8, dv + (uint64_t)(ks * 2048 >> 4), I_PV, (jj > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.p_free[w]);
        tc_commit(&B.kv_empty[s]);
      }
      __syncwarp();
      if (j + 2 < nown) qk(j + 2);
      if (++jj == nbh) { jj = 0; ++li; }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------ softmax of warpgroup w's key blocks, one query row per thread
    setmaxnreg_inc<224>();
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t xg = su + O_X;
    float m_i = -INFINITY, l_i = 0.f;
    for (int j = 0, li = 0, jj = 0; j < nown; ++j, ++jj) {
      if (jj == nbh) { jj = 0; ++li; }
      const int G = li * nb + 2 * jj + w, s = G % ST;
      if (jj == 0) {
        m_i = -INFINITY; l_i = 0.f;
        if (w == 0 && r == 0) {                                            // this item's g tile into the (drained) staging buffer
          int a, m0, head; item_of(li, a, m0, head);
          tma_store_wait_read0();
          mbar_expect_tx(&B.g_full, QM * DH * 2);
          tma_load_2d(xg, &mg, &B.g_full, head * DH, a * L + m0);
        }
      }
      mbar_wait(&B.kv_full[s], (G / ST) & 1);                              // the bias tile of this block
      mbar_wait(&B.s_full[w][j & 1], (j >> 1) & 1);
      tc_fence_after();
      f2 t[BN / 2];
      const uint32_t sb = su + O_ST + s * STB + 2 * TK;
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        uint32_t v[32];
        tmem_ld32(trow + T_S + w * 128 + (j & 1) * 64 + cc * 32, v);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 bw = lds128(sb + sw128(r, cc * 4 + q));
          const uint32_t bb[4] = {bw.x, bw.y, bw.z, bw.w};
#pragma unroll
          for (int e = 0; e < 4; ++e)
            t[cc * 16 + q * 4 + e] = add2(mk2u(v[q * 8 + 2 * e], v[q * 8 + 2 * e + 1]), mk2(bf16lo(bb[e]), bf16hi(bb[e])));
        }
      }
      if (j >= 1) tc_fence_before();
      float mxp[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
      for (int k = 0; k < BN / 2; ++k) mxp[k & 3] = max3f(mxp[k & 3], lo2(t[k]), hi2(t[k]));
      const float mx = max3f(mxp[0], mxp[1], fmaxf(mxp[2], mxp[3]));
      const float m_new = mx > m_i + LAZY ? mx : m_i;
      if (j >= 1) mbar_wait(&B.p_free[w], (j - 1) & 1);                    // PV(j - 1) done: O final for it, P free
      tc_fence_after();
      if (__any_sync(0xffffffffu, m_new != m_i)) {
        const float alpha = ex2f(m_i - m_new);
        l_i *= alpha;
        if (jj >= 1) {
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
      const f2 NM = mk2(-m_i, -m_i);
      f2 ssp[4] = {mk2(0.f, 0.f), mk2(0.f, 0.f), mk2(0.f, 0.f), mk2(0.f, 0.f)};
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        uint32_t pk[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const f2 e = add2(t[cc * 16 + k], NM);
          const float p0 = ex2f(lo2(e)), p1 = ex2f(hi2(e));
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
      if (jj == nbh - 1) {
        // ---- the item's epilogue: merge the two warpgroups' partials, gate, bf16 over q
        mbar_wait(&B.p_free[w], j & 1);
        tc_fence_after();
        uint32_t ov[48];
#pragma unroll
        for (int cc = 0; cc < 3; ++cc) tmem_ld16(trow + T_O + w * 64 + cc * 16, *reinterpret_cast<uint32_t(*)[16]>(ov + 16 * cc));
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.o_free[w]);
        float* xc = reinterpret_cast<float*>(sm + O_XC);                   // [c][row]: warpgroup 1 -> warpgroup 0
        if (w == 1) {
          if (li >= 1) named_bar_sync(3, 256);                             // warpgroup 0 has read the previous item's partial
          xc[r] = m_i; xc[128 + r] = l_i;
#pragma unroll
          for (int c = 0; c < 48; ++c) xc[(2 + c) * 128 + r] = __uint_as_float(ov[c]);
          named_bar_sync(1, 256);
        } else {
          named_bar_sync(1, 256);
          const float m1 = xc[r], l1 = xc[128 + r];
          const float mm = fmaxf(m_i, m1), a0 = ex2f(m_i - mm), a1 = ex2f(m1 - mm);
          const float inv = 1.f / (l_i * a0 + l1 * a1);
          const float c0 = a0 * inv, c1 = a1 * inv;
          int a, m0, head; item_of(li, a, m0, head);
          mbar_wait(&B.g_full, li & 1);
#pragma unroll
          for (int q = 0; q < 6; ++q) {                                    // 8 channels per 16-byte chunk of this thread's row
            const uint32_t ad = xg + sw128(r, q);
            const uint4 gw = lds128(ad);
            const uint32_t gg[4] = {gw.x, gw.y, gw.z, gw.w};
            uint32_t o4[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int c = 8 * q + 2 * e;
              const float o0 = __uint_as_float(ov[c]) * c0 + xc[(2 + c) * 128 + r] * c1;
              const float o1 = __uint_as_float(ov[c + 1]) * c0 + xc[(3 + c) * 128 + r] * c1;
              o4[e] = pack_bf16(o0 * sigmoid_kit(bf16lo(gg[e])), o1 * sigmoid_kit(bf16hi(gg[e])));
            }
            sts128(ad, make_uint4(o4[0], o4[1], o4[2], o4[3]));
          }
          if (li + 1 < my_items) named_bar_arrive(3, 256);                // the exchange buffer is free again
          fence_proxy_async();
          named_bar_sync(2, 128);
          if (r == 0) {
            tma_store_2d(&mo, xg, head * DH, a * L + m0);
            tma_store_commit();
          }
        }
      }
    }
    if (w == 0 && r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
