// lattn_dq.cu — dQ of the AF3 windowed atom attention on tcgen05 (sm_100a), the query-centric mirror of lattn_dkv.cu.
//
//   P = exp2(S - LSE)    dP = dO V^T    dS = P (dP - D)    dQ = dS K / sqrt(32)     (bf16 into dq[row, h * 32 + d], row stride ldd)
//
// A CTA owns 128 queries [q0, q0 + 128) (windows 4 c .. 4 c + 3), one head and a range of samples (SP CTAs split the samples); they see
// the 224 keys [q0 - 48, q0 + 176). TMEM lane = query:  S = Q K^T [128 x 224] cols 0..223,  dP = dO V^T cols 224..447,  dQ cols 448..479.
// Warp w (lane quadrant w & 3 = one window, band quarter w >> 2) reads its 32 x 32 band cells, computes dS = exp2(S c + bias - LSE) (dP - D)
// and -- after the quadrant's warps have read -- writes dS as packed bf16 over S (cols 0..111, out-of-band cells zero): the TMEM A operand
// of dQ += dS K (B = the K tile, MN-major). Warps with band quarter < 2 drain 32 x 16 of dQ each through shared memory with a TMA store.
// Warp 16 issues the TMA loads (3 stages, refilled when a sample's MMAs commit) and the MMAs; the window's bias (log2 domain, key mask folded in) stays in shared memory.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, WK = 128, KOFF = 48, QN = 128, KN = 224, NST = 3;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_DO = QN * 64, T_K = 2 * QN * 64, T_V = T_K + KN * 64, T_L = T_V + KN * 64, T_D = T_L + QN * 4;
constexpr int STAGE = 46080;
static_assert(T_D + QN * 4 <= STAGE && STAGE % 1024 == 0, "stage");
constexpr int BSTR = 33;                                            // padded bias rows: conflict-free both ways
constexpr int O_B = NST * STAGE, O_O = O_B + 16 * 32 * BSTR * 4, O_BAR = O_O + 16 * 1024;
constexpr int SMEM_BYTES = O_BAR + 256;
constexpr int C_S = 0, C_DP = 224, C_DQ = 448, C_PS = 0;
constexpr uint32_t TX = 2 * QN * 64 + 2 * KN * 64 + 2 * QN * 4;

DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

extern "C" __global__ void __launch_bounds__(544, 1)
local_attn_dq(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                 const __grid_constant__ CUtensorMap tm_do, const __grid_constant__ CUtensorMap tm_l, const __grid_constant__ CUtensorMap tm_d,
                 const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias, const uint8_t* __restrict__ kmask,
                 int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *sready = bars + NST, *tready = bars + NST + 1, *dready = bars + NST + 2, *empty = bars + NST + 3;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  float* sbias = reinterpret_cast<float*>(sm + O_B);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int q0 = 128 * jc, kb = q0 - KOFF;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 16;                                       // TMA + MMA warp (no TMEM lanes)

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) mbar_init(&full[i], 1);
    mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(dready, 1);
    for (int i = 0; i < NST; ++i) mbar_init(&empty[i], 1);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int wq = 4 * jc + qd;                                        // this quadrant's window; lane = query qq of it
  if (!ctl) {                                                                  // lane = key offset jj (coalesced rows of the bias), loop over the queries
    const int key = 32 * wq - KOFF + 32 * qt + lane;
    const bool kok = key >= 0 && key < N && (kmask == nullptr || kmask[key]);
    const float* src = gbias + (((size_t)h * nwin + (wq < nwin ? wq : 0)) * WQ) * WK + 32 * qt + lane;
#pragma unroll 8
    for (int qq = 0; qq < 32; ++qq) {
      float v = 0.f;
      if (wq < nwin) v = kok ? src[qq * WK] * LOG2E : -INFINITY;
      sbias[(warp * 32 + lane) * BSTR + qq] = v;
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;
  const uint32_t id_s = idesc_bf16(128, KN, 0, 0), id_d = idesc_bf16(128, 32, 0, 1);

  auto load = [&](int it) {
    const int s = it % NST, a = a_lo + it;
    const uint32_t st = su + s * STAGE;
    mbar_expect_tx(&full[s], TX);
    tma_load_3d(st + T_Q, &tm_q, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_DO, &tm_do, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_K, &tm_k, &full[s], h * DH, kb, a);
    tma_load_3d(st + T_V, &tm_v, &full[s], h * DH, kb, a);
    tma_load_3d(st + T_L, &tm_l, &full[s], q0, h, a);
    tma_load_3d(st + T_D, &tm_d, &full[s], q0, h, a);
  };
  auto mma_s = [&](int it) {
    const int s = it % NST;
    const uint32_t st = su + s * STAGE;
    mbar_wait(&full[s], (it / NST) & 1);
    tc_fence_after();
    const uint64_t dq = desc_sw64(st + T_Q), dk = desc_sw64(st + T_K), ddo = desc_sw64(st + T_DO), dv = desc_sw64(st + T_V);
    if (elect_one()) {
      umma_ss(tm + C_S, dq, dk, id_s, 0);
      umma_ss(tm + C_DP, ddo, dv, id_s, 0);
      umma_ss(tm + C_S, dq + 2, dk + 2, id_s, 1);
      umma_ss(tm + C_DP, ddo + 2, dv + 2, id_s, 1);
      tc_commit(sready);
    }
    __syncwarp();
  };
  auto mma_d = [&](int it) {                                         // dQ = dS K (A = packed bf16 dS in TMEM, B = K tile MN-major)
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t dk = desc_sw64(st + T_K);
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < KN / 16; ++ks) umma_ts(tm + C_DQ, tm + C_PS + 8 * ks, dk + (uint64_t)(ks * 1024 >> 4), id_d, ks);
      tc_commit(dready);
      tc_commit(&empty[it % NST]);
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0) for (int i = 0; i < NST && i < nit; ++i) load(i);
    __syncwarp();
    if (nit > 0) mma_s(0);
    for (int it = 0; it < nit; ++it) {
      mbar_wait(tready, it & 1);
      tc_fence_after();
      mma_d(it);                                                     // commits dready and empty[stage of it]
      if (it + 1 < nit) mma_s(it + 1);
      if (it + NST < nit) {
        mbar_wait(&empty[it % NST], (it / NST) & 1);
        if (lane == 0) load(it + NST);
        __syncwarp();
      }
    }
  } else {

  auto drain = [&](int itd) {                                        // warps with qt < 2: 32 queries x 16 columns of dQ
    if (qt >= 2) return;
    uint32_t v[16];
    tmem_ld16(tm + ((uint32_t)(32 * qd) << 16) + C_DQ + 16 * qt, v);
    tmem_wait_ld();
    uint8_t* ob = sm + O_O + warp * 1024;
    if (lane == 0) tma_store_wait_read0();
    __syncwarp();
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      uint4 o;
      o.x = pack_bf16(__uint_as_float(v[8 * g + 0]) * QSCALE, __uint_as_float(v[8 * g + 1]) * QSCALE);
      o.y = pack_bf16(__uint_as_float(v[8 * g + 2]) * QSCALE, __uint_as_float(v[8 * g + 3]) * QSCALE);
      o.z = pack_bf16(__uint_as_float(v[8 * g + 4]) * QSCALE, __uint_as_float(v[8 * g + 5]) * QSCALE);
      o.w = pack_bf16(__uint_as_float(v[8 * g + 6]) * QSCALE, __uint_as_float(v[8 * g + 7]) * QSCALE);
      *reinterpret_cast<uint4*>(ob + lane * 32 + 16 * g) = o;
    }
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) {
      tma_store_3d(&tm_o, su + O_O + warp * 1024, h * DH + 16 * qt, q0 + 32 * qd, a_lo + itd);
      tma_store_commit();
    }
  };

  const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
  const int x0 = 32 * qd + 32 * qt;                                  // chunk-relative key column of the band quarter
  const float* sb = sbias + warp * 32 * BSTR + lane;
  for (int it = 0; it < nit; ++it) {
    const int s = it % NST;
    mbar_wait(sready, it & 1);
    tc_fence_after();
    const float lse = reinterpret_cast<const float*>(sm + s * STAGE + T_L)[32 * qd + lane];
    const float dd = reinterpret_cast<const float*>(sm + s * STAGE + T_D)[32 * qd + lane];
    uint32_t dw[16];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      uint32_t sv[16], dv[16];
      tmem_ld16(tl + C_S + x0 + 16 * c, sv);
      tmem_ld16(tl + C_DP + x0 + 16 * c, dv);
      tmem_wait_ld();
#pragma unroll
      for (int jj = 0; jj < 16; jj += 2) {
        const int b = 16 * c + jj;
        const float p0 = ex2f(fmaf(__uint_as_float(sv[jj]), QSCALE * LOG2E, sb[b * BSTR]) - lse);
        const float p1 = ex2f(fmaf(__uint_as_float(sv[jj + 1]), QSCALE * LOG2E, sb[(b + 1) * BSTR]) - lse);
        dw[b >> 1] = pack_bf16(p0 * (__uint_as_float(dv[jj]) - dd), p1 * (__uint_as_float(dv[jj + 1]) - dd));
      }
    }
    tc_fence_before();
    named_bar_sync(1 + qd, 128);
    tc_fence_after();
    tmem_st16(tl + C_PS + x0 / 2, dw);
    if (qt < 3) tmem_st16z(tl + C_PS + (qt < qd ? 16 * qt : 16 * qt + 64));
    tmem_wait_st();
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(tready);
    mbar_wait(dready, it & 1);
    tc_fence_after();
    drain(it);
  }
  }
  if (lane == 0) tma_store_wait0();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
