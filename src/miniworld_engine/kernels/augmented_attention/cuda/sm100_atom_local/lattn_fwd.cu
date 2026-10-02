// lattn_fwd.cu — AF3 windowed atom attention forward (32 queries x 128 keys per window) on tcgen05 (sm_100a).
//
//   S = q k^T / sqrt(32) + bias[h][w]      O = softmax(S) v      (bf16 O [A N, 128], row LSE [A, 4, N] in log2 units)
//
// A CTA owns 128 queries [q0, q0 + 128) (windows 4 c .. 4 c + 3), one head and a range of samples (SP CTAs split the samples); they see the
// 224 keys [q0 - 48, q0 + 176). TMEM lane = query: S = Q K^T [128 x 224] in two buffers (cols 0..223 / 224..447: the next sample's S runs
// during this one's softmax), O = P V [128 x 32] cols 448..479. Warp w < 16 (lane quadrant w & 3 = one window, band quarter w >> 2) reads
// its 32 band cells (bias from shared memory, log2 domain, key mask folded in as -inf); the quadrant's four warps exchange row maxima and
// sums through shared memory; P (packed bf16, out-of-band cells zero) goes over the S buffer as the TMEM A operand of O += P V (B = the V
// tile, MN-major). The previous sample's O / l is drained (band quarters 0, 1: shared memory + TMA store; quarter 2: LSE) while this
// sample's P is formed. Warp 16 issues the TMA loads (3 stages, refilled when a sample's MMAs commit) and the MMAs.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, WK = 128, KOFF = 48, QN = 128, KN = 224, NST = 3, BSTR = 33;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_K = QN * 64, T_V = T_K + KN * 64;
constexpr int STAGE = 36864;
static_assert(T_V + KN * 64 <= STAGE && STAGE % 1024 == 0, "stage");
constexpr int O_B = NST * STAGE, O_O = O_B + 16 * 32 * BSTR * 4, O_R = O_O + 16 * 1024, O_BAR = O_R + 2 * 2 * 16 * 32 * 4;
constexpr int SMEM_BYTES = O_BAR + 256;
constexpr int C_O = 448;
constexpr uint32_t TX = QN * 64 + 2 * KN * 64;

DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

extern "C" __global__ void __launch_bounds__(544, 1)
local_attn_fwd(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                  const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias, const uint8_t* __restrict__ kmask,
                  float* __restrict__ LSE, int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *sready = bars + NST, *tready = bars + NST + 2, *oready = bars + NST + 3, *empty = bars + NST + 4;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  float* sbias = reinterpret_cast<float*>(sm + O_B);
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
  if (!ctl) {
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
      umma_ss(tm + 224 * (it & 1), dq, dk, id_s, 0);
      umma_ss(tm + 224 * (it & 1), dq + 2, dk + 2, id_s, 1);
      tc_commit(&sready[it & 1]);
    }
    __syncwarp();
  };
  auto mma_o = [&](int it) {                                         // O = P V (A = packed bf16 P in buffer it & 1, B = V tile MN-major)
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t dv = desc_sw64(st + T_V);
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < KN / 16; ++ks) umma_ts(tm + C_O, tm + 224 * (it & 1) + 8 * ks, dv + (uint64_t)(ks * 1024 >> 4), id_o, ks);
      tc_commit(oready);
      tc_commit(&empty[it % NST]);
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0) for (int i = 0; i < NST && i < nit; ++i) load(i);
    __syncwarp();
    if (nit > 0) mma_s(0);
    if (nit > 1) mma_s(1);
    for (int it = 0; it < nit; ++it) {
      mbar_wait(tready, it & 1);
      tc_fence_after();
      mma_o(it);                                                     // commits oready and empty[stage of it]
      if (it + 2 < nit) mma_s(it + 2);
      if (it + NST < nit) {
        mbar_wait(&empty[it % NST], (it / NST) & 1);
        if (lane == 0) load(it + NST);
        __syncwarp();
      }
    }
  } else {

  const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
  const int x0 = 32 * qd + 32 * qt;
  const float* sb = sbias + warp * 32 * BSTR + lane;
  const int qrow = q0 + 32 * qd + lane;
  auto drain = [&](int itd, float mm) {                              // O / l (band quarters 0, 1) and LSE (quarter 2) of sample itd
    const int par = itd & 1, a = a_lo + itd;
    const float* rsum = red + ((1 * 2 + par) * 16) * 32;
    const float lt = rsum[qd * 32 + lane] + rsum[(qd + 4) * 32 + lane] + rsum[(qd + 8) * 32 + lane] + rsum[(qd + 12) * 32 + lane];
    if (qt < 2) {
      uint32_t v[16];
      tmem_ld16(tl + C_O + 16 * qt, v);
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
    } else if (qt == 2 && qrow < N) {
      LSE[((size_t)a * NH + h) * N + qrow] = lt > 0.f ? mm + __log2f(lt) : 1e30f;
    }
  };
  float mm_prev = 0.f;
  for (int it = 0; it < nit; ++it) {
    const int b = it & 1, par = it & 1;
    mbar_wait(&sready[b], (it >> 1) & 1);
    tc_fence_after();
    float sv[32];
    {
      uint32_t r0[16], r1[16];
      tmem_ld16(tl + 224 * b + x0, r0);
      tmem_ld16(tl + 224 * b + x0 + 16, r1);
      tmem_wait_ld();
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        sv[j] = fmaf(__uint_as_float(r0[j]), QSCALE * LOG2E, sb[j * BSTR]);
        sv[16 + j] = fmaf(__uint_as_float(r1[j]), QSCALE * LOG2E, sb[(16 + j) * BSTR]);
      }
    }
    float m = sv[0];
#pragma unroll
    for (int j = 1; j < 32; ++j) m = fmaxf(m, sv[j]);
    float* rmax = red + ((0 * 2 + par) * 16) * 32;
    float* rsum = red + ((1 * 2 + par) * 16) * 32;
    rmax[warp * 32 + lane] = m;
    tc_fence_before();
    named_bar_sync(1 + qd, 128);
    tc_fence_after();
    m = fmaxf(fmaxf(rmax[qd * 32 + lane], rmax[(qd + 4) * 32 + lane]), fmaxf(rmax[(qd + 8) * 32 + lane], rmax[(qd + 12) * 32 + lane]));
    const float mm = m == -INFINITY ? 0.f : m;
    uint32_t pw[16];
    float l = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j += 2) {
      const float p0 = ex2f(sv[j] - mm), p1 = ex2f(sv[j + 1] - mm);
      l += p0 + p1;
      pw[j >> 1] = pack_bf16(p0, p1);
    }
    rsum[warp * 32 + lane] = l;
    tmem_st16(tl + 224 * b + x0 / 2, pw);
    if (qt < 3) tmem_st16z(tl + 224 * b + (qt < qd ? 16 * qt : 16 * qt + 64));
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
