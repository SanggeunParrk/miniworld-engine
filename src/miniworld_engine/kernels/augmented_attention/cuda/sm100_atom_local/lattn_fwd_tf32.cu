// lattn_fwd_tf32.cu — AF3 windowed atom attention forward for the fp32 path (32 queries x 128 keys per window), tcgen05 kind::tf32
// (sm_100a): the fp32 twin of lattn_fwd.cu.
//
//   S = q k^T / sqrt(32) + bias[h][w]      O = softmax(S) v      (fp32 O [A N, 128] natural units, row LSE [A, 4, N] in log2 units)
//
// q, k, v are fp32 [A, N, 128] views (any row stride: column blocks of the block's [A N, 512] q | k | v | g projection). Products are
// tcgen05 kind::tf32 with fp32 accumulation; the softmax, P and the row sums are fp32 (P is rounded to tf32 by the PV MMA).
//
// Geometry (as lattn_fwd.cu): a CTA owns 128 queries [q0, q0 + 128) (windows 4 c .. 4 c + 3), one head and a range of samples (SP CTAs
// split the samples); they see the 224 keys [q0 - 48, q0 + 176). TMEM lane = query. Warp w < 16 = (lane quadrant w & 3 = one window,
// band quarter w >> 2) reads its 32 x 32 band cells; the quadrant's four warps exchange row maxima and sums through shared memory.
// What the fp32 operands change:
//   * a 32-wide fp32 head row is 128 B: the q / k tiles are K-major in the 128-B swizzle (QK^T: four K = 8 steps of 32 B); v is the
//     MN-major B operand of PV, which a tf32 MMA takes only in the 128-B swizzle with 32-B atoms (TMA SWIZZLE_128B_ATOM_32B, UMMA
//     layout type 1, sm100.cuh): one 32-column atom, 8 keys (1 KB) per K step.
//   * P is fp32 and as wide as S, so it goes IN PLACE over S: every warp writes exactly the 32 band cells it read (no barrier between
//     the quadrant's readers and writers), and warps of band quarter < 3 zero one of the quadrant's three out-of-band 32-column blocks.
//   * the bias lives in registers (32 per thread: its query row x its warp's band quarter; log2 domain, -inf on invalid keys): the
//     bf16 kernel's 66 KB shared copy does not fit next to two fp32 stages.
//   * two stages (72 KB each) instead of three: sample it + 2 loads once sample it's PV MMAs have committed (its S buffer is free by then).
// Budgets. smem: 2 x 72 KB stages (q 16 KB | k 28 KB | v 28 KB) + 8 KB max / sum exchange + 8 x 2 KB O staging + barriers = 172288 B.
// TMEM 512 columns: S / P buffers 0..223 and 224..447 (the next sample's QK^T runs during this softmax), O 448..479.
// Threads: 17 warps (544; <= 120 registers): 16 softmax warps, one query row per thread; warp 16 issues the TMA loads (lane 0) and the
// MMAs (whole warp, elect_one). O leaves through per-warp 2 KB stagings (32 rows x 16 fp32, 64-B swizzle) and TMA stores (band
// quarters 0, 1); band quarter 2 writes the LSE.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, WK = 128, KOFF = 48, QN = 128, KN = 224, NST = 2;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_K = QN * 128, T_V = T_K + KN * 128, STAGE = T_V + KN * 128;     // 16 KB | 28 KB | 28 KB
constexpr int O_R = NST * STAGE, O_O = O_R + 2 * 2 * 16 * 32 * 4, O_BAR = O_O + 8 * 2048, SMEM_BYTES = O_BAR + 256;
static_assert(STAGE % 1024 == 0 && T_K % 1024 == 0 && T_V % 1024 == 0 && O_O % 1024 == 0, "1 KB alignment of the swizzled tiles");
static_assert(SMEM_BYTES == 172288, "keep sm100_atom_local.KERNELS_TF32 in step");
constexpr int C_O = 2 * KN;
constexpr uint32_t TX = STAGE;
constexpr uint32_t I_S = idesc_tf32(128, KN), I_O = idesc_tf32(128, DH, 0, 1);

DEVI uint32_t rna_tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }   // kind::tf32 truncates
DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

extern "C" __global__ void __launch_bounds__(544, 1)
local_attn_fwd_tf32(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                    const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias, const uint8_t* __restrict__ kmask,
                    float* __restrict__ LSE, int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *empty = bars + NST, *sready = bars + 2 * NST, *tready = bars + 2 * NST + 2, *oready = bars + 2 * NST + 3;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  float* red = reinterpret_cast<float*>(sm + O_R);                   // [kind 2][parity 2][warp 16][lane 32]: row max / row sum partials
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int q0 = 128 * jc, kb = q0 - KOFF;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 16;                                       // TMA + MMA warp (no TMEM lanes)

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 1); mbar_init(&sready[i], 1); }
    mbar_init(tready, 16); mbar_init(oready, 1);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int wq = 4 * jc + qd;                                        // this quadrant's window; lane = query of it
  float bl[32];                                                      // bias row (log2 units) of this thread's 32 band cells
  if (!ctl) {
    const int key0 = 32 * wq - KOFF + 32 * qt;
    const float4* src = reinterpret_cast<const float4*>(gbias + (((size_t)h * nwin + (wq < nwin ? wq : 0)) * WQ + lane) * WK + 32 * qt);
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const float4 b4 = wq < nwin ? __ldg(src + g) : make_float4(0.f, 0.f, 0.f, 0.f);
      const float bb[4] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = key0 + 4 * g + e;
        const bool ok = key >= 0 && key < N && (kmask == nullptr || kmask[key]);
        bl[4 * g + e] = ok ? bb[e] * LOG2E : -INFINITY;
      }
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;

  auto load = [&](int it) {                                          // lane 0 of the control warp; stage it % 2
    const int s = it % NST, a = a_lo + it;
    const uint32_t st = su + s * STAGE;
    mbar_expect_tx(&full[s], TX);
    tma_load_3d(st + T_Q, &tm_q, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_K, &tm_k, &full[s], h * DH, kb, a);
    tma_load_3d(st + T_V, &tm_v, &full[s], h * DH, kb, a);
  };
  auto mma_s = [&](int it) {                                         // S of sample it into TMEM buffer it & 1 (= its stage)
    const int s = it % NST;
    const uint32_t st = su + s * STAGE;
    mbar_wait(&full[s], (it / NST) & 1);
    tc_fence_after();
    const uint64_t dq = desc_k128(st + T_Q), dk = desc_k128(st + T_K);
    if (elect_one()) {
#pragma unroll
      for (int k = 0; k < 4; ++k) umma_ss_tf32(tm + KN * s, dq + (uint64_t)(2 * k), dk + (uint64_t)(2 * k), I_S, k > 0 ? 1u : 0u);
      tc_commit(&sready[s]);
    }
    __syncwarp();
  };
  auto mma_o = [&](int it) {                                         // O = P V (A = fp32 P over S in TMEM, B = the v tile MN-major)
    const int s = it % NST;
    const uint64_t dv = desc_mn32b(su + s * STAGE + T_V, KN * 128);
    if (elect_one()) {
#pragma unroll 4
      for (int k = 0; k < KN / 8; ++k)
        umma_ts_tf32(tm + C_O, tm + KN * s + 8 * k, dv + (uint64_t)((k * 1024) >> 4), I_O, k > 0 ? 1u : 0u);
      tc_commit(oready);
      tc_commit(&empty[s]);
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0) for (int i = 0; i < NST && i < nit; ++i) load(i);
    __syncwarp();
    if (nit > 0) mma_s(0);
    if (nit > 1) mma_s(1);
    for (int it = 0; it < nit; ++it) {
      mbar_wait(tready, it & 1);                                     // P(it) written, O(it - 1) drained
      tc_fence_after();
      mma_o(it);                                                     // commits oready and empty[stage of it]
      if (it + NST < nit) {
        mbar_wait(&empty[it % NST], (it / NST) & 1);                 // PV(it) done: the stage and the S buffer of it are free
        if (lane == 0) load(it + NST);
        __syncwarp();
        mma_s(it + NST);
      }
    }
  } else {
    const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
    const int x0 = 32 * qd + 32 * qt;                                // chunk-relative key column of the band quarter
    const int z0 = 32 * (qt < qd ? qt : qt + 4);                     // the out-of-band block this warp zeroes (qt < 3)
    const int qrow = q0 + 32 * qd + lane;
    auto drain = [&](int itd, float mm) {                            // O / l (band quarters 0, 1) and LSE (quarter 2) of sample itd
      const int par = itd & 1, a = a_lo + itd;
      const float* rs = red + ((1 * 2 + par) * 16) * 32;
      const float lt = rs[qd * 32 + lane] + rs[(qd + 4) * 32 + lane] + rs[(qd + 8) * 32 + lane] + rs[(qd + 12) * 32 + lane];
      if (qt < 2) {
        uint32_t v[16];
        tmem_ld16(tl + C_O + 16 * qt, v);
        tmem_wait_ld();
        const float il = lt > 0.f ? 1.f / lt : 0.f;
        const uint32_t ob = su + O_O + warp * 2048;                  // [32 rows][64 B], 64-B swizzle
        if (lane == 0) tma_store_wait_read0();
        __syncwarp();
#pragma unroll
        for (int g = 0; g < 4; ++g)
          sts128(ob + sw64(lane, g), make_uint4(__float_as_uint(__uint_as_float(v[4 * g]) * il), __float_as_uint(__uint_as_float(v[4 * g + 1]) * il),
                                                __float_as_uint(__uint_as_float(v[4 * g + 2]) * il), __float_as_uint(__uint_as_float(v[4 * g + 3]) * il)));
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) {
          tma_store_3d(&tm_o, ob, h * DH + 16 * qt, q0 + 32 * qd, a);
          tma_store_commit();
        }
      } else if (qt == 2 && qrow < N) {
        LSE[((size_t)a * NH + h) * N + qrow] = lt > 0.f ? mm + log2f(lt) : 1e30f;
      }
    };
    float mm_prev = 0.f;
    for (int it = 0; it < nit; ++it) {
      const int b = it & 1, par = it & 1;
      mbar_wait(&sready[b], (it >> 1) & 1);
      tc_fence_after();
      float sv[32];
      {
        uint32_t r[32];
        tmem_ld32(tl + KN * b + x0, r);
        tmem_wait_ld();
#pragma unroll
        for (int j = 0; j < 32; ++j) sv[j] = fmaf(__uint_as_float(r[j]), QSCALE * LOG2E, bl[j]);
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
      float l = 0.f;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {                               // P (fp32) over the S cells this warp read
        uint32_t pk[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float p = ex2f(sv[16 * hf + k] - mm);
          l += p;
          pk[k] = rna_tf32(p);                                       // the PV operand, rounded (the row sum stays exact)
        }
        tmem_st16(tl + KN * b + x0 + 16 * hf, pk);
      }
      rsum[warp * 32 + lane] = l;
      if (qt < 3) { tmem_st16z(tl + KN * b + z0); tmem_st16z(tl + KN * b + z0 + 16); }
      if (it >= 1) {                                                 // the previous sample's O (its MMAs ran during this softmax)
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
