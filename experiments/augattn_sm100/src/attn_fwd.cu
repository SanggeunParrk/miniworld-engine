// attn_fwd.cu — the token DiT's augmented pair-bias attention core, TRAINING forward, sm_100a (tcgen05 + TMEM + TMA):
//
//   S[a] = q[a] k[a]^T / sqrt(48) + bias          bias [H, L, L] per head, shared by all A samples
//   O[a] = softmax(S[a]) v[a]                     fp32 out, plus the row log-sum-exp (log2 units) for the backward
//
// Numerics as the sm_90a kernel (token_dit_train/attn_fwd.cu): bf16 operands, fp32 accumulation, the bias pre-scaled by log2 e
// and rounded to bf16 (host prep), logits in log2 units so every exponential is one ex2, a lazy running max (moves only when a block
// exceeds it by LAZY), P rounded to bf16 for the tensor core, the row sum of the unrounded p in fp32. q enters unscaled: the
// softmax warps form t = s * (log2 e / sqrt 48) + bias2 with one FFMA.
//
// One CTA = one (head, 128-row query tile) for TWO samples: warpgroup 1 (warps 4-7) runs sample a0's softmax, warpgroup 2 (warps
// 8-11) sample a0 + 1's, and the bias tile of a key block lands once in shared memory for both. Warp 0 issues TMA, warp 1 the MMAs:
//   S_w = Q_w K_w^T   (M128 N128 K48, SS, into TMEM)          O_w += P_w V_w   (M128 N48 K128, TS: P bf16 from TMEM)
// TMEM: S0 [0,128) S1 [128,256) P0 [256,320) P1 [320,384) O0 [384,432) O1 [448,496).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef LAZY
#define LAZY 5.545177444479562f          // running-max slack in natural units (= 8 in log2 units: p < 2^8)
#endif
#ifndef QWID
#define QWID 48                          // TMA box width of q / k / v rows (48 = the head; 64 pulls 16 columns of the next head)
#endif
constexpr int BN = 128, DH = 48, QM = 128, DM = 768, ST = 2;
constexpr int TQ = QM * 128, TK = BN * 128, TB = QM * BN * 2;             // bytes: q tile, k / v tile (128-B pitch), bias tile
constexpr int O_Q = 0, O_ST = 2 * TQ, STB = 4 * TK + TB;                  // stage: K0 | K1 | V0 | V1 | bias (2 K-blocks of 64 keys)
constexpr int O_BAR = O_ST + ST * STB;
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_S = 0, T_P = 256, T_O = 384;
constexpr uint32_t I_QK = idesc_bf16(128, BN), I_PV = idesc_bf16(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f;

#ifdef TRACE
// CTA TRC_CTA: [event][block]: 0 producer stage issued, 1 MMA saw kv_full, 2 MMA issued QK(w0), 3 softmax w0 saw s_full, 4 softmax w0 P done,
// 5 MMA issued PV(w0), 6 softmax w0 epilogue start, 7 CTA start (block 0), 8 q_full seen by MMA (block 0)
__device__ unsigned long long g_tr[9][64];
#ifndef TRC_CTA
#define TRC_CTA 0
#endif
#define TR(ev, i) do { if (blockIdx.x == TRC_CTA && (i) < 64) g_tr[ev][i] = clock64(); } while (0)
#else
#define TR(ev, i) do { } while (0)
#endif
struct Bars { uint64_t q_full, kv_full[ST], kv_empty[ST], s_full[2], p_full[2], p_free[2]; uint32_t tmem; };

extern "C" __global__ void __launch_bounds__(384, 1)
augattn_fwd_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                  const __grid_constant__ CUtensorMap mb, float* __restrict__ O, float* __restrict__ LSE, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int npair = A >> 1, mt = L / QM;
  const int pair = blockIdx.x % npair, cid = blockIdx.x / npair, mtile = cid % mt, head = cid / mt;
  const int a0 = 2 * pair, m0 = mtile * QM, qcol = head * DH, nb = L / BN;

  if (tid == 0) TR(7, 0);
  if (tid == 0) {
    mbar_init(&B.q_full, 1);
    for (int s = 0; s < ST; ++s) { mbar_init(&B.kv_full[s], 1); mbar_init(&B.kv_empty[s], 1); }
    for (int w = 0; w < 2; ++w) { mbar_init(&B.s_full[w], 1); mbar_init(&B.p_full[w], 4); mbar_init(&B.p_free[w], 1); }
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
      mbar_expect_tx(&B.q_full, 2 * QM * QWID * 2);
      for (int w = 0; w < 2; ++w) tma_load_2d(su + O_Q + w * TQ, &mq, &B.q_full, qcol, (a0 + w) * L + m0);
      for (int n = 0; n < nb; ++n) {
        const int s = n % ST;
        if (n >= ST) mbar_wait(&B.kv_empty[s], ((n / ST) - 1) & 1);
        const uint32_t st = su + O_ST + s * STB;
        mbar_expect_tx(&B.kv_full[s], 4 * BN * QWID * 2 + TB);
        for (int w = 0; w < 2; ++w) {
          tma_load_2d(st + w * TK, &mk, &B.kv_full[s], qcol, (a0 + w) * L + n * BN);
          tma_load_2d(st + 2 * TK + w * TK, &mv, &B.kv_full[s], qcol, (a0 + w) * L + n * BN);
        }
        for (int kb = 0; kb < 2; ++kb) tma_load_2d(st + 4 * TK + kb * (TB / 2), &mb, &B.kv_full[s], n * BN + kb * 64, head * L + m0);
        TR(0, n);
      }
    }
  } else if (warp == 1) {
    mbar_wait(&B.q_full, 0);
    if (lane == 0) TR(8, 0);
    auto qk = [&](int n, int w) {
      const uint32_t st = su + O_ST + (n % ST) * STB;
      const uint64_t dq = desc_k128(su + O_Q + w * TQ), dk = desc_k128(st + w * TK);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 3; ++ks) umma_ss(tmem + T_S + w * 128, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_QK, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w]);
      }
      __syncwarp();
      if (w == 0 && lane == 0) TR(2, n);
    };
    mbar_wait(&B.kv_full[0], 0);
    tc_fence_after();
    qk(0, 0); qk(0, 1);
    for (int n = 0; n < nb; ++n) {
      const int s = n % ST;
      for (int w = 0; w < 2; ++w) {
        mbar_wait(&B.p_full[w], n & 1);
        tc_fence_after();
        const uint64_t dv = desc_mn128(su + O_ST + s * STB + 2 * TK + w * TK, 8192);
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ts(tmem + T_O + w * 64, tmem + T_P + w * 64 + ks * 8, dv + (uint64_t)(ks * 2048 >> 4), I_PV, (n > 0 || ks > 0) ? 1u : 0u);
          tc_commit(&B.p_free[w]);
          if (w == 1) tc_commit(&B.kv_empty[s]);
        }
        __syncwarp();
        if (w == 0 && lane == 0) TR(5, n);
        if (n + 1 < nb) {
          if (w == 0) { mbar_wait(&B.kv_full[(n + 1) % ST], ((n + 1) / ST) & 1); tc_fence_after(); if (lane == 0) TR(1, n + 1); }
          qk(n + 1, w);
        }
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------ softmax of sample a0 + w, one query row per thread
    setmaxnreg_inc<224>();
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const float cq = 0.14433756729740643f;                                // 1 / sqrt 48: t = s / sqrt 48 + bias (natural units)
    float m_i = -INFINITY, l_i = 0.f;
    for (int n = 0; n < nb; ++n) {
      const int s = n % ST;
      mbar_wait(&B.kv_full[s], (n / ST) & 1);                              // the bias tile of this block
      mbar_wait(&B.s_full[w], n & 1);
      if (w == 0 && r == 0) TR(3, n);
      tc_fence_after();
      float t[BN];
      const uint32_t sb = su + O_ST + s * STB + 4 * TK;
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        uint32_t v[32];
        tmem_ld32(trow + T_S + w * 128 + cc * 32, v);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 4; ++q) {                                      // 8 keys of bias per 16-byte chunk
          const int key0 = cc * 32 + q * 8, kb = key0 >> 6;
          const uint4 bw = lds128(sb + kb * (TB / 2) + sw128(r, (key0 & 63) >> 3));
          const uint32_t bb[4] = {bw.x, bw.y, bw.z, bw.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            t[key0 + 2 * e] = fmaf(__uint_as_float(v[q * 8 + 2 * e]), cq, bf16lo(bb[e]));
            t[key0 + 2 * e + 1] = fmaf(__uint_as_float(v[q * 8 + 2 * e + 1]), cq, bf16hi(bb[e]));
          }
        }
      }
#ifdef ABL_SOFT
      // ablation: no softmax arithmetic (P = raw bits), keeps every load, barrier and MMA
      {
        if (n >= 1) mbar_wait(&B.p_free[w], (n - 1) & 1);
        tc_fence_after();
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) { uint32_t pk[16];
#pragma unroll
          for (int k = 0; k < 16; ++k) pk[k] = __float_as_uint(t[cc * 32 + 2 * k]) ^ __float_as_uint(t[cc * 32 + 2 * k + 1]);
          tmem_st16(trow + T_P + w * 64 + cc * 16, pk); }
        m_i = 0.f; l_i = 1.f;
        tmem_wait_st(); tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.p_full[w]);
        if (w == 0 && r == 0) TR(4, n);
        continue;
      }
#endif
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < BN; ++j) mx = fmaxf(mx, t[j]);
      const float m_new = mx > m_i + LAZY ? mx : m_i;          // natural units; p = 2^(t log2 e - m log2 e)
      if (n >= 1) mbar_wait(&B.p_free[w], (n - 1) & 1);                    // PV(n - 1) done: O final for n - 1, P free
      tc_fence_after();
      if (__any_sync(0xffffffffu, m_new != m_i)) {
        const float alpha = ex2f((m_i - m_new) * LOG2E);                     // m_i = -inf on the first block: alpha = 0
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
      float ssum = 0.f;
      const float nm2 = -m_i * LOG2E;
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {                                     // 32 keys -> 16 packed P columns
        uint32_t pk[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float p0 = ex2f(fmaf(t[cc * 32 + 2 * k], LOG2E, nm2)), p1 = ex2f(fmaf(t[cc * 32 + 2 * k + 1], LOG2E, nm2));
          ssum += p0 + p1;
          pk[k] = pack_bf16(p0, p1);
        }
        tmem_st16(trow + T_P + w * 64 + cc * 16, pk);
      }
      l_i += ssum;
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.p_full[w]);
      if (w == 0 && r == 0) TR(4, n);
    }
    if (w == 0 && r == 0) TR(6, 0);
    // ---- epilogue: O = acc / l (fp32), LSE = m + log2 l
    mbar_wait(&B.p_free[w], (nb - 1) & 1);
    tc_fence_after();
    const float inv = 1.f / l_i;
    const int row = m0 + (int)r, a = a0 + w;
    float* orow = O + ((size_t)a * L + row) * DM + qcol;
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
    LSE[((size_t)a * 16 + head) * L + row] = m_i * LOG2E + __log2f(l_i);   // log2 units
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
