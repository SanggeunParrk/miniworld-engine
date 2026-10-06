// lattn_dq_tf32.cu — dQ of the AF3 windowed atom attention for the fp32 path, tcgen05 kind::tf32 (sm_100a): the fp32 twin of lattn_dq.cu.
//
//   P = exp2(S - LSE)    dP = dO V^T    dS = P (dP - D)    dQ = dS K / sqrt(32)     (fp32 into dq[row, qcol + h * 32 + d], row stride of dP)
//
// A CTA owns 128 queries [q0, q0 + 128) (windows 4 c .. 4 c + 3), one head and a range of samples (SP CTAs split the samples); they see
// the 224 keys [q0 - 48, q0 + 176). TMEM lane = query:  S = Q K^T [128 x 224] cols 0..223,  dP = dO V^T cols 224..447,  dQ cols 448..479.
// Warp w < 16 (lane quadrant w & 3 = one window, band quarter w >> 2) reads its 32 x 32 band cells of S and dP, computes
// dS = exp2(S c + bias - LSE) (dP - D) and writes it as fp32 IN PLACE over the S cells it read (no barrier: every warp owns its cells),
// zeroing one out-of-band 32-column block per band quarter < 3: the TMEM A operand of dQ += dS K. K is read twice: K-major for S (the
// 128-B swizzle) and MN-major for dQ (the 128-B swizzle with 32-B atoms, the only MN-major layout a tf32 MMA takes; sm100.cuh).
// Budgets. smem: 2 stages of q 16 KB | dO 16 KB | k 28 KB | v 28 KB | LSE 512 B | D 512 B (89 KB, released once the step's S / dP MMAs are
// done and the threads have read LSE / D), one MN-major k slot (28 KB, released by the dQ MMAs: S(it + 1) runs before dQ(it + 1) needs
// it), 8 x 2 KB dQ staging, barriers: 227584 B. TMEM: 480 of 512 columns. The bias stays in registers (32 per thread, log2 domain, key
// mask folded in as -inf).
// Threads: 17 warps (544; <= 120 registers): 16 dS warps (one query row per thread), warp 16 issues the TMA loads (lane 0) and the
// MMAs (whole warp, elect_one). Warps of band quarter < 2 drain 32 x 16 of dQ each (64-B swizzled staging + TMA store).
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, WK = 128, KOFF = 48, QN = 128, KN = 224, NST = 2;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_DO = QN * 128, T_K = 2 * QN * 128, T_V = T_K + KN * 128, T_L = T_V + KN * 128, T_D = T_L + QN * 4;
constexpr int STAGE = T_D + QN * 4;                                   // 89 KB
constexpr int O_KM = NST * STAGE, O_O = O_KM + KN * 128, O_BAR = O_O + 8 * 2048, SMEM_BYTES = O_BAR + 256;
static_assert(STAGE % 1024 == 0 && T_K % 1024 == 0 && T_V % 1024 == 0 && O_KM % 1024 == 0 && O_O % 1024 == 0, "alignment");
static_assert(SMEM_BYTES == 227584 && SMEM_BYTES <= 232448, "keep sm100_atom_local.KERNELS_TF32 in step");
constexpr int C_S = 0, C_DP = KN, C_DQ = 2 * KN;
constexpr uint32_t TX1 = STAGE, TX2 = KN * 128;
constexpr uint32_t I_S = idesc_tf32(128, KN), I_D = idesc_tf32(128, DH, 0, 1);

DEVI uint32_t rna_tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }   // kind::tf32 truncates
DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}
DEVI float lds32f(uint32_t a) { float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory"); return v; }

extern "C" __global__ void __launch_bounds__(544, 1)
local_attn_dq_tf32(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                   const __grid_constant__ CUtensorMap tm_km, const __grid_constant__ CUtensorMap tm_do, const __grid_constant__ CUtensorMap tm_l,
                   const __grid_constant__ CUtensorMap tm_d, const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias,
                   const uint8_t* __restrict__ kmask, int qcol, int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full1 = bars, *full2 = bars + 2, *sready = bars + 3, *tready = bars + 4, *dready = bars + 5, *empty2 = bars + 6;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int q0 = 128 * jc, kb = q0 - KOFF;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 16;                                       // TMA + MMA warp (no TMEM lanes)

  if (tid == 0) {
    mbar_init(&full1[0], 1); mbar_init(&full1[1], 1); mbar_init(full2, 1);
    mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(dready, 1); mbar_init(empty2, 1);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int wq = 4 * jc + qd;                                        // this quadrant's window; lane = query of it
  float bl[32];
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

  auto load1 = [&](int it) {                                         // q, dO, k, v, LSE, D of sample it into stage it % 2
    const int s = it % NST, a = a_lo + it;
    const uint32_t st = su + s * STAGE;
    mbar_expect_tx(&full1[s], TX1);
    tma_load_3d(st + T_Q, &tm_q, &full1[s], h * DH, q0, a);
    tma_load_3d(st + T_DO, &tm_do, &full1[s], h * DH, q0, a);
    tma_load_3d(st + T_K, &tm_k, &full1[s], h * DH, kb, a);
    tma_load_3d(st + T_V, &tm_v, &full1[s], h * DH, kb, a);
    tma_load_3d(st + T_L, &tm_l, &full1[s], q0, h, a);
    tma_load_3d(st + T_D, &tm_d, &full1[s], q0, h, a);
  };
  auto load2 = [&](int it) {                                         // the MN-major k of sample it
    mbar_expect_tx(full2, TX2);
    tma_load_3d(su + O_KM, &tm_km, full2, h * DH, kb, a_lo + it);
  };
  auto mma_s = [&](int it) {                                         // S and dP of sample it
    const int s = it % NST;
    const uint32_t st = su + s * STAGE;
    mbar_wait(&full1[s], (it / NST) & 1);
    tc_fence_after();
    const uint64_t dq = desc_k128(st + T_Q), dk = desc_k128(st + T_K), ddo = desc_k128(st + T_DO), dv = desc_k128(st + T_V);
    if (elect_one()) {
#pragma unroll
      for (int k = 0; k < 4; ++k) umma_ss_tf32(tm + C_S, dq + (uint64_t)(2 * k), dk + (uint64_t)(2 * k), I_S, k > 0 ? 1u : 0u);
#pragma unroll
      for (int k = 0; k < 4; ++k) umma_ss_tf32(tm + C_DP, ddo + (uint64_t)(2 * k), dv + (uint64_t)(2 * k), I_S, k > 0 ? 1u : 0u);
      tc_commit(sready);
    }
    __syncwarp();
  };
  auto mma_d = [&](int it) {                                         // dQ = dS K (A = fp32 dS over S in TMEM, B = the MN-major k)
    mbar_wait(full2, it & 1);
    tc_fence_after();
    const uint64_t dk = desc_mn32b(su + O_KM, KN * 128);
    if (elect_one()) {
#pragma unroll 4
      for (int k = 0; k < KN / 8; ++k)
        umma_ts_tf32(tm + C_DQ, tm + C_S + 8 * k, dk + (uint64_t)((k * 1024) >> 4), I_D, k > 0 ? 1u : 0u);
      tc_commit(dready);
      tc_commit(empty2);
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0) {
      for (int i = 0; i < NST && i < nit; ++i) load1(i);
      if (nit > 0) load2(0);
    }
    __syncwarp();
    if (nit > 0) mma_s(0);
    for (int it = 0; it < nit; ++it) {
      mbar_wait(tready, it & 1);                                     // dS(it) in TMEM, dQ(it - 1) drained, LSE / D of stage it read
      tc_fence_after();
      if (it + NST < nit) {                                          // the S / dP MMAs of it are done (the threads waited for them)
        if (lane == 0) load1(it + NST);
        __syncwarp();
      }
      mma_d(it);                                                     // commits dready and empty2
      if (it + 1 < nit) {
        mma_s(it + 1);                                               // after dQ(it) in the MMA order: it reads dS(it) before S(it + 1) lands
        mbar_wait(empty2, it & 1);
        if (lane == 0) load2(it + 1);
        __syncwarp();
      }
    }
  } else {
    const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
    const int x0 = 32 * qd + 32 * qt;                                // chunk-relative key column of the band quarter
    const int z0 = 32 * (qt < qd ? qt : qt + 4);                     // the out-of-band block this warp zeroes (qt < 3)
    auto drain = [&](int itd) {                                      // warps with qt < 2: 32 queries x 16 columns of dQ
      if (qt >= 2) return;
      uint32_t v[16];
      tmem_ld16(tl + C_DQ + 16 * qt, v);
      tmem_wait_ld();
      const uint32_t ob = su + O_O + warp * 2048;                    // [32 rows][64 B], 64-B swizzle
      if (lane == 0) tma_store_wait_read0();
      __syncwarp();
#pragma unroll
      for (int g = 0; g < 4; ++g)
        sts128(ob + sw64(lane, g), make_uint4(__float_as_uint(__uint_as_float(v[4 * g]) * QSCALE), __float_as_uint(__uint_as_float(v[4 * g + 1]) * QSCALE),
                                              __float_as_uint(__uint_as_float(v[4 * g + 2]) * QSCALE), __float_as_uint(__uint_as_float(v[4 * g + 3]) * QSCALE)));
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) {
        tma_store_3d(&tm_o, ob, qcol + h * DH + 16 * qt, q0 + 32 * qd, a_lo + itd);
        tma_store_commit();
      }
    };
    for (int it = 0; it < nit; ++it) {
      const int s = it % NST;
      mbar_wait(sready, it & 1);
      tc_fence_after();
      const float lse = lds32f(su + s * STAGE + T_L + 4 * (32 * qd + lane));
      const float dd = lds32f(su + s * STAGE + T_D + 4 * (32 * qd + lane));
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        uint32_t sv[16], dv[16];
        tmem_ld16(tl + C_S + x0 + 16 * hf, sv);
        tmem_ld16(tl + C_DP + x0 + 16 * hf, dv);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float p = ex2f(fmaf(__uint_as_float(sv[k]), QSCALE * LOG2E, bl[16 * hf + k]) - lse);
          sv[k] = rna_tf32(p * (__uint_as_float(dv[k]) - dd));      // the dQ operand, rounded
        }
        tmem_st16(tl + C_S + x0 + 16 * hf, sv);
      }
      if (qt < 3) { tmem_st16z(tl + C_S + z0); tmem_st16z(tl + C_S + z0 + 16); }
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
