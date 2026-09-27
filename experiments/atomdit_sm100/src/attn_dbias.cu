// attn_dbias.cu — the atom DiT's attention backward, dbias pass, sm_100a:  dbias[h, i, j] = sum_a dS[a, h, i, j]  (fp32, [4, N, N]).
//
// At atom lengths the token-DiT split (dQ + dbias in one pass) would push either ~2 GB of dbias atomics (H100 design) or ~2 GB of dQ
// partials (the token B200 design: dQ partial per 128-key chunk, N / 128 = 24 of them at N = 3072). So dbias gets its own pass, the
// token dqb kernel minus everything dQ: a CTA owns (head, 128 queries, 128-key chunk) and walks all A samples; per sample
//   S = q K^T, dP = dO V^T (TMEM)      P = 2^(S log2 e / sqrt 32 + bias log2 e - LSE)      dbias_w += P (dP - D)   (registers)
// and writes its 128 x 128 dbias tile once. Warpgroup w owns keys w * 64 .. w * 64 + 63 of the chunk; one query row per thread.
// TMEM: S[w] at w * 128, dP[w] at w * 128 + 64.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef ST
#define ST 3                             // q | dO | K | V stages (64 KB)
#endif
#ifndef ROTK
#define ROTK 1                           // sample offset per key chunk: neighbouring CTAs work on nearby samples (L2 working set)
#endif
constexpr int BN = 64, KC = 128, DH = 32, QM = 128, DM = 128, NH = 4;
constexpr int T128 = 128 * 128;                                            // one [128 rows][128 B] tile (32 bf16 used per row)
constexpr int STB = 4 * T128;
constexpr int O_ST = 0, O_B = ST * STB, O_BAR = O_B + 2 * T128;            // bias tile of the item: [128 q][64 keys] x 2
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t I_S = idesc_bf16(128, BN);
constexpr float LOG2E = 1.4426950408889634f, RSQD = 0.17677669529663687f;

struct Bars {
  uint64_t full[ST], empty[ST], bfull, bempty, s_full[2], s_free[2];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(384, 1)
atom_dbias_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                 const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mb, const float* __restrict__ LSE,
                 const float* __restrict__ DD, float* __restrict__ DB, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = L / QM, nch = L / KC;
  const int items = NH * mt * nch;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int ng = my_items * A;
  auto item_of = [&](int li, int& c, int& m0, int& head) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    c = wi % nch; const int r = wi / nch;
    m0 = (r % mt) * QM; head = r / mt;
  };
  auto samp0 = [&](int c) { return (c * ROTK) % A; };

  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 2); }
    mbar_init(&B.bfull, 1); mbar_init(&B.bempty, 8);
    for (int b = 0; b < 2; ++b) { mbar_init(&B.s_full[b], 1); mbar_init(&B.s_free[b], 4); }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 256); tmem_relinquish(); }
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
        int c, m0, head; item_of(li, c, m0, head);
        const int qcol = head * DH;
        if (li >= 1) mbar_wait(&B.bempty, (li - 1) & 1);
        mbar_expect_tx(&B.bfull, 2 * T128);
        for (int w = 0; w < 2; ++w) tma_load_2d(su + O_B + w * T128, &mb, &B.bfull, c * KC + w * BN, head * L + m0);
        for (int i = 0, a = samp0(c); i < A; ++i, ++g, a = (a + 1 == A) ? 0 : a + 1) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.full[s], 4 * 128 * DH * 2);
          tma_load_2d(st, &mq, &B.full[s], qcol, a * L + m0);
          tma_load_2d(st + T128, &mdo, &B.full[s], qcol, a * L + m0);
          tma_load_2d(st + 2 * T128, &mk, &B.full[s], qcol, a * L + c * KC);
          tma_load_2d(st + 3 * T128, &mv, &B.full[s], qcol, a * L + c * KC);
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ MMA issuers: warp 1 + w serves warpgroup w
    const int w = warp - 1;
    for (int g = 0; g < ng; ++g) {
      const int s = g % ST;
      mbar_wait(&B.full[s], (g / ST) & 1);
      if (g >= 1) mbar_wait(&B.s_free[w], (g - 1) & 1);
      tc_fence_after();
      const uint32_t st = su + O_ST + s * STB;
      const uint64_t dq = desc_k128(st), ddo = desc_k128(st + T128);
      const uint64_t dk = desc_k128(st + 2 * T128 + w * BN * 128), dv = desc_k128(st + 3 * T128 + w * BN * 128);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks) umma_ss(tmem + w * 128, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks) umma_ss(tmem + w * 128 + 64, ddo + (uint64_t)(ks * 2), dv + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w]);
        tc_commit(&B.empty[s]);
      }
      __syncwarp();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ dS warpgroups
    setmaxnreg_inc<224>();
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const f2 CQL = mk2(RSQD * LOG2E, RSQD * LOG2E), L2E = mk2(LOG2E, LOG2E);
    f2 db[BN / 2];
    // positions advance incrementally (the runtime divisions of item_of run once per item)
    int li = 0, i = 0, c = 0, m0 = 0, head = 0, a = 0;
    if (ng > 0) { item_of(0, c, m0, head); a = samp0(c); }
    auto advance = [&](int& pli, int& pi, int& pc, int& pm0, int& ph, int& pa) {
      if (++pi == A) { pi = 0; if (++pli < my_items) { item_of(pli, pc, pm0, ph); pa = samp0(pc); } }
      else pa = (pa + 1 == A) ? 0 : pa + 1;
    };
    int nli = li, ni = i, nc = c, nm0 = m0, nh = head, na = a;
    float lse_n = 0.f, dd_n = 0.f;
    if (ng > 0) { const size_t ri = ((size_t)a * NH + head) * L + m0 + r; lse_n = LSE[ri]; dd_n = DD[ri]; }
    for (int g = 0; g < ng; ++g) {
      const float lse = lse_n, dd = dd_n;                                  // loaded one step ahead
      advance(nli, ni, nc, nm0, nh, na);
      if (g + 1 < ng) { const size_t ri = ((size_t)na * NH + nh) * L + nm0 + r; lse_n = LSE[ri]; dd_n = DD[ri]; }
      if (i == 0) {
#pragma unroll
        for (int j = 0; j < BN / 2; ++j) db[j] = mk2(0.f, 0.f);
        mbar_wait(&B.bfull, li & 1);
      }
      mbar_wait(&B.s_full[w], g & 1);
      tc_fence_after();
      const f2 NL = mk2(-lse, -lse), ND = mk2(-dd, -dd);
      const uint32_t sb = su + O_B + w * T128;
      uint32_t sa[16], da[16], sq[16], dq[16];
      tmem_ld16(trow + w * 128, sa);
      tmem_ld16(trow + w * 128 + 64, da);
      tmem_wait_ld();
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {                                     // quarters of 16 keys; the next quarter's loads in flight
        uint32_t (&sv)[16] = (qq & 1) ? sq : sa;
        uint32_t (&dv)[16] = (qq & 1) ? dq : da;
        uint32_t (&sn)[16] = (qq & 1) ? sa : sq;
        uint32_t (&dn)[16] = (qq & 1) ? da : dq;
        if (qq < 3) {
          tmem_ld16(trow + w * 128 + (qq + 1) * 16, sn);
          tmem_ld16(trow + w * 128 + 64 + (qq + 1) * 16, dn);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint4 bw = lds128(sb + sw128(r, qq * 2 + h));
          const uint32_t bb[4] = {bw.x, bw.y, bw.z, bw.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int j = h * 4 + e, jj = qq * 8 + j;
            const f2 x = fma2(mk2u(sv[2 * j], sv[2 * j + 1]), CQL, fma2(mk2(bf16lo(bb[e]), bf16hi(bb[e])), L2E, NL));
            const f2 p = mk2(ex2f(lo2(x)), ex2f(hi2(x)));
            db[jj] = fma2(p, add2(mk2u(dv[2 * j], dv[2 * j + 1]), ND), db[jj]);
          }
        }
        if (qq < 3) tmem_wait_ld();
        if (qq == 2) {                                                     // S / dP fully in registers: the next S MMA may overwrite them
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.s_free[w]);
        }
      }
      if (i == A - 1) {
        // ---- the item's dbias tile: this thread's 64 keys of query row m0 + r, natural units, fp32
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.bempty);
        float* brow = DB + ((size_t)head * L + m0 + r) * L + c * KC + w * BN;
#pragma unroll
        for (int k = 0; k < BN / 4; ++k)
          *reinterpret_cast<float4*>(brow + 4 * k) = make_float4(lo2(db[2 * k]), hi2(db[2 * k]), lo2(db[2 * k + 1]), hi2(db[2 * k + 1]));
      }
      advance(li, i, c, m0, head, a);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 256); }
}
