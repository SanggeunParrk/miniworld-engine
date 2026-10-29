// lattn_esm.cu — ESMFold2's atom attention on sm_100a (``LocalDiTBlock(**ESMFOLD2)``, inference; tcgen05 / TMEM / TMA): query i sees the
// keys j with |i - j| <= 64 among the real atoms (the 192-key slid subset of its window always contains them), with the pair bias of its
// subset slot j - start_w. ``lesm.cu`` re-indexes that bias by position once per pair (``bias2`` [4, nwin, 32, 160], log2 units, -inf where
// j is not a key of i), so this kernel walks positions only:
//
//   S = q k^T / sqrt(32) + bias2[h][w][i][j - (32 w - 64)]        O = softmax(S) v        (q, k RMS-normalised and rotated: lesm.cu)
//
// lattn_fwd.cu's frame with the wider band. A CTA owns 128 queries (four windows) of one head and a range of samples; its keys are the 256
// positions [q0 - 64, q0 + 192) (window w%4 = qd sees the 160 of them from 32 qd: five 32-key cells, the outer two triangular). TMEM lane =
// query: S = Q K^T [128 x 256] in two buffers (cols 0..255 / 256..511), P (packed bf16) over the first half of its buffer, O = P V [128 x 32]
// in cols 128..159 of the same buffer (TMEM is full: the next sample's S into that buffer is issued once O is drained, i.e. at the
// tready of the following sample). Warp w < 16: lane quadrant qd = w & 3 (one window), qt = w >> 2: cells 1..3 (qt < 3), cells 0 and 4
// (qt = 3, 64 values); the bias from shared memory ([key][query], -inf folded in); the quadrant's four warps exchange row maxima and sums
// through shared memory; out-of-band P cells are zeroed. Warp 16 issues the TMA loads (2 stages) and the MMAs. O leaves through shared memory
// and per-warp TMA stores; the LSE (log2 units) is written when a pointer is given.
// Budgets: smem 2 x 40 KB stages + 84.5 KB bias + 16 KB O staging + 8 KB reductions; TMEM 512 columns; 544 threads.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, SPAN = 160, KOFF = 64, QN = 128, KN = 256, NST = 2, BSTR = 33;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_K = QN * 64, T_V = T_K + KN * 64;
constexpr int STAGE = T_V + KN * 64;                                 // 40 KB
static_assert(STAGE % 1024 == 0, "stage");
constexpr int O_B = NST * STAGE, O_O = O_B + 4 * SPAN * BSTR * 4, O_R = O_O + 16 * 1024, O_BAR = O_R + 2 * 2 * 16 * 32 * 4;
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES == 191232, "keep sm100_atom_local.KERNELS['esm'] in step");
constexpr uint32_t TX = QN * 64 + 2 * KN * 64;

DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

extern "C" __global__ void __launch_bounds__(544, 1)
local_attn_esm(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
               const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias, float* __restrict__ LSE, int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *sready = bars + NST, *tready = bars + NST + 2, *oready = bars + NST + 3, *empty = bars + NST + 4;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  float* sbias = reinterpret_cast<float*>(sm + O_B);                 // [quadrant 4][key 160][query BSTR]
  float* red = reinterpret_cast<float*>(sm + O_R);                   // [kind 2][parity 2][warp 16][lane 32]: row max / row sum partials
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int q0 = 128 * jc, kb = q0 - KOFF;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 16;                                       // TMA + MMA warp (no TMEM lanes)

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) mbar_init(&full[i], 1);
    mbar_init(&sready[0], 1); mbar_init(&sready[1], 1); mbar_init(tready, 16); mbar_init(oready, 1);
    for (int i = 0; i < NST; ++i) mbar_init(&empty[i], 1);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int wq = 4 * jc + qd;
  if (!ctl) {                                                        // this warp's cells of the window's bias (keys x queries)
    const float* src = gbias + (((size_t)h * nwin + (wq < nwin ? wq : 0)) * WQ) * SPAN;
    for (int cc = 0; cc < (qt < 3 ? 1 : 2); ++cc) {
      const int cell = qt < 3 ? qt + 1 : 4 * cc, c = 32 * cell + lane;
#pragma unroll 8
      for (int qq = 0; qq < 32; ++qq) sbias[(qd * SPAN + c) * BSTR + qq] = wq < nwin ? src[qq * SPAN + c] : -INFINITY;
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;
  const uint32_t id_s = idesc_bf16(128, KN, 0, 0), id_o = idesc_bf16(128, 32, 0, 1);

  auto load = [&](int it) {
    const int s = it % NST, a = a_lo + it;
    const uint32_t st = su + s * STAGE;
    mbar_expect_tx(&full[s], TX);
    tma_load_3d(st + T_Q, &tm_q, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_K, &tm_k, &full[s], h * DH, kb, a);
    tma_load_3d(st + T_V, &tm_v, &full[s], h * DH, kb, a);
  };
  auto mma_s = [&](int it) {                                         // S of sample it into buffer it & 1
    const int s = it % NST;
    const uint32_t st = su + s * STAGE;
    mbar_wait(&full[s], (it / NST) & 1);
    tc_fence_after();
    const uint64_t dq = desc_sw64(st + T_Q), dk = desc_sw64(st + T_K);
    if (elect_one()) {
      umma_ss(tm + KN * (it & 1), dq, dk, id_s, 0);
      umma_ss(tm + KN * (it & 1), dq + 2, dk + 2, id_s, 1);
      tc_commit(&sready[it & 1]);
    }
    __syncwarp();
  };
  auto mma_o = [&](int it) {                                         // O = P V into cols 128..159 of buffer it & 1 (A = its packed P)
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t dv = desc_sw64(st + T_V);
    const uint32_t tb = tm + KN * (it & 1);
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < KN / 16; ++ks) umma_ts(tb + 128, tb + 8 * ks, dv + (uint64_t)(ks * 1024 >> 4), id_o, ks);
      tc_commit(oready);
      tc_commit(&empty[it % NST]);
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0) for (int i = 0; i < NST && i < nit; ++i) load(i);
    __syncwarp();
    if (nit > 0) mma_s(0);
    for (int it = 0; it < nit; ++it) {
      mbar_wait(tready, it & 1);                                     // P(it) written, O(it - 1) drained (buffer (it + 1) & 1 is free)
      tc_fence_after();
      if (it + 1 < nit) mma_s(it + 1);
      mma_o(it);                                                     // commits oready and empty[stage of it]
      if (it + NST < nit) {
        mbar_wait(&empty[it % NST], (it / NST) & 1);
        if (lane == 0) load(it + NST);
        __syncwarp();
      }
    }
  } else {

  const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
  const float* sbq = sbias + qd * SPAN * BSTR + lane;
  const int qrow = q0 + 32 * qd + lane;
  auto drain = [&](int itd, float mm) {                              // O / l (qt 0, 1) and LSE (qt 2) of sample itd (buffer itd & 1)
    const int par = itd & 1, a = a_lo + itd;
    const float* rsum = red + ((1 * 2 + par) * 16) * 32;
    const float lt = rsum[qd * 32 + lane] + rsum[(qd + 4) * 32 + lane] + rsum[(qd + 8) * 32 + lane] + rsum[(qd + 12) * 32 + lane];
    if (qt < 2) {
      uint32_t v[16];
      tmem_ld16(tl + KN * par + 128 + 16 * qt, v);
      tmem_wait_ld();
      const float il = lt > 0.f ? 1.f / lt : 0.f;
      uint8_t* ob = sm + O_O + warp * 1024;
      if (lane == 0) tma_store_wait_read0();
      __syncwarp();
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        uint4 o;
        o.x = pack_bf16(__uint_as_float(v[8 * g + 0]) * il, __uint_as_float(v[8 * g + 1]) * il);
        o.y = pack_bf16(__uint_as_float(v[8 * g + 2]) * il, __uint_as_float(v[8 * g + 3]) * il);
        o.z = pack_bf16(__uint_as_float(v[8 * g + 4]) * il, __uint_as_float(v[8 * g + 5]) * il);
        o.w = pack_bf16(__uint_as_float(v[8 * g + 6]) * il, __uint_as_float(v[8 * g + 7]) * il);
        *reinterpret_cast<uint4*>(ob + lane * 32 + 16 * g) = o;
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) {
        tma_store_3d(&tm_o, su + O_O + warp * 1024, h * DH + 16 * qt, q0 + 32 * qd, a);
        tma_store_commit();
      }
    } else if (qt == 2 && qrow < N && LSE != nullptr) {
      LSE[((size_t)a * NH + h) * N + qrow] = lt > 0.f ? mm + __log2f(lt) : 1e30f;
    }
  };
  // the score of key j of this warp's cells (cells qt + 1, or 0 and 4): S * scale * log2 e + bias
  auto scores = [&](uint32_t col, int cell, float (&sv)[32]) {
    uint32_t r0[16], r1[16];
    tmem_ld16(col, r0);
    tmem_ld16(col + 16, r1);
    tmem_wait_ld();
    const float* sb = sbq + 32 * cell * BSTR;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      sv[j] = fmaf(__uint_as_float(r0[j]), QSCALE * LOG2E, sb[j * BSTR]);
      sv[16 + j] = fmaf(__uint_as_float(r1[j]), QSCALE * LOG2E, sb[(16 + j) * BSTR]);
    }
  };
  const int cell_a = qt < 3 ? qt + 1 : 0;
  float mm_prev = 0.f;
  for (int it = 0; it < nit; ++it) {
    const int b = it & 1, par = it & 1;
    const uint32_t tb = tl + KN * b;
    mbar_wait(&sready[b], (it >> 1) & 1);
    tc_fence_after();
    float sa[32];
    scores(tb + 32 * qd + 32 * cell_a, cell_a, sa);
    float m = sa[0];
#pragma unroll
    for (int j = 1; j < 32; ++j) m = fmaxf(m, sa[j]);
    if (qt == 3) {                                                   // cell 4: its maximum now, its values again after the exchange (P and
      float sz[32];                                                  // the zeros never reach columns >= 128, O(it) is issued after tready)
      scores(tb + 32 * qd + 128, 4, sz);
#pragma unroll
      for (int j = 0; j < 32; ++j) m = fmaxf(m, sz[j]);
    }
    float* rmax = red + ((0 * 2 + par) * 16) * 32;
    float* rsum = red + ((1 * 2 + par) * 16) * 32;
    rmax[warp * 32 + lane] = m;
    tc_fence_before();
    named_bar_sync(1 + qd, 128);
    tc_fence_after();
    m = fmaxf(fmaxf(rmax[qd * 32 + lane], rmax[(qd + 4) * 32 + lane]), fmaxf(rmax[(qd + 8) * 32 + lane], rmax[(qd + 12) * 32 + lane]));
    const float mm = m == -INFINITY ? 0.f : m;
    float l = 0.f;
    {
      uint32_t pw[16];
#pragma unroll
      for (int j = 0; j < 32; j += 2) {
        const float p0 = ex2f(sa[j] - mm), p1 = ex2f(sa[j + 1] - mm);
        l += p0 + p1;
        pw[j >> 1] = pack_bf16(p0, p1);
      }
      tmem_st16(tb + 16 * qd + 16 * cell_a, pw);
    }
    if (qt == 3) {
      float sz[32];
      scores(tb + 32 * qd + 128, 4, sz);
      uint32_t pw[16];
#pragma unroll
      for (int j = 0; j < 32; j += 2) {
        const float p0 = ex2f(sz[j] - mm), p1 = ex2f(sz[j + 1] - mm);
        l += p0 + p1;
        pw[j >> 1] = pack_bf16(p0, p1);
      }
      tmem_st16(tb + 16 * qd + 64, pw);
    }
    rsum[warp * 32 + lane] = l;
    if (qt < 3) tmem_st16z(tb + (qt < qd ? 16 * qt : 16 * qt + 80));  // the 48 packed columns outside this window's band
    if (it >= 1) {                                                   // the previous sample's O (its MMAs ran during this softmax)
      mbar_wait(oready, (it - 1) & 1);
      tc_fence_after();
      drain(it - 1, mm_prev);
    }
    mm_prev = mm;
    tmem_wait_st();
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(tready);
  }
  if (nit > 0) {
    mbar_wait(oready, (nit - 1) & 1);
    tc_fence_after();
    drain(nit - 1, mm_prev);
  }
  }
  if (lane == 0) tma_store_wait0();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
