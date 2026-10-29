// lattn_esm_dq.cu — dQ and the bias gradient of ESMFold2's atom attention on sm_100a (``LocalDiTBlock(**ESMFOLD2)`` training; tcgen05 / TMEM /
// TMA), the query-centric half of lattn_esm.cu's backward (lattn_esm_dkv.cu is the key-centric half):
//
//   P = exp2(S c + bias2 - LSE)   dP = dO V^T   dS = P (dP - D)   dQ = dS K / sqrt(32)   dbias[h][w][i][c] = sum over the samples of dS
//
// (S = q k^T, c = log2(e) / sqrt(32); bias2 and LSE in log2 units as lattn_esm.cu has them, dbias in natural units, by position like bias2).
// lattn_esm.cu's frame: a CTA owns 128 queries [q0, q0 + 128) of one head and a range of samples (SP CTAs split the samples), and the 256 key
// positions [q0 - 64, q0 + 192). TMEM lane = query: S = Q K^T cols 0..255, dP = dO V^T cols 256..511. Warp w < 16 (lane quadrant qd = w & 3 =
// one window, qt = w >> 2: cell qt + 1, or the two triangular cells 0 and 4 for qt = 3) computes dS over its cells, adds it into its dbias
// registers (summed over the samples; cell 0 is zero left of the diagonal and cell 4 right of it, so the pair shares 32 registers + 1), and --
// after the quadrant's four warps have read S -- writes dS as packed bf16 over S cols 0..127 (out-of-band cells zero): the TMEM A operand of
// dQ = dS K into cols 128..159. Warps with qt < 2 drain dQ (32 queries x 16 columns each, TMA stores into dq[row, h * 32 + d], row stride ldd).
// Warp 15 also issues the TMA loads (2 stages) and the MMAs: after dQ(it), dP(it + 1) and the first half of S(it + 1) at once, the second half (whose
// columns hold dQ) once dQ(it) is drained. dbias: SP = 1 writes [4, nwin, 32, 160] directly, SP > 1 one such plane per part (summed by the caller).
// Budgets: smem 2 x 49 KB stages + 82.5 KB bias + 8 KB dQ staging; TMEM 512 columns; 512 threads (<= 128 registers).
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, SPAN = 160, KOFF = 64, QN = 128, KN = 256, NST = 2, BSTR = 33;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_DO = QN * 64, T_K = 2 * QN * 64, T_V = T_K + KN * 64, T_L = T_V + KN * 64, T_D = T_L + QN * 4;
constexpr int STAGE = 50176;
static_assert(T_D + QN * 4 <= STAGE && STAGE % 1024 == 0, "stage");
constexpr int O_B = NST * STAGE, O_O = O_B + 4 * SPAN * BSTR * 4, O_BAR = O_O + 8 * 1024;
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES == 193280, "keep sm100_atom_local.KERNELS_ESM_TRAIN['esm_dq'] in step");
constexpr int C_S = 0, C_DQ = 128, C_DP = 256;
constexpr uint32_t TX = 2 * QN * 64 + 2 * KN * 64 + 2 * QN * 4;

DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

DEVI void tmem_ld8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}

extern "C" __global__ void __launch_bounds__(512, 1)
local_attn_esm_dq(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                  const __grid_constant__ CUtensorMap tm_do, const __grid_constant__ CUtensorMap tm_l, const __grid_constant__ CUtensorMap tm_d,
                  const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias, float* __restrict__ dbias, int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *empty = bars + NST, *sready = bars + 2 * NST, *tready = sready + 1, *dready = sready + 2, *dfree = sready + 3;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  float* sbias = reinterpret_cast<float*>(sm + O_B);                 // [quadrant 4][key 160][query BSTR] (log2 units, -inf off the band)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int q0 = 128 * jc, kb = q0 - KOFF;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 15;                                       // also issues the TMA loads and the MMAs

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 1); }
    mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(dready, 1); mbar_init(dfree, 8);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int wq = 4 * jc + qd;
  {                                                                  // this warp's cells of the window's bias (keys x queries)
    const float* src = gbias + (((size_t)h * nwin + wq) * WQ) * SPAN;
    for (int cc = 0; cc < (qt < 3 ? 1 : 2); ++cc) {
      const int cell = qt < 3 ? qt + 1 : 4 * cc, c = 32 * cell + lane;
#pragma unroll 8
      for (int qq = 0; qq < 32; ++qq) sbias[(qd * SPAN + c) * BSTR + qq] = src[qq * SPAN + c];
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;
  const uint32_t id_h = idesc_bf16(128, 128, 0, 0), id_p = idesc_bf16(128, KN, 0, 0), id_d = idesc_bf16(128, 32, 0, 1);

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
  auto mma_dp = [&](int it) {                                      // dP = dO V^T [128 x 256] into cols 256..511
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t ddo = desc_sw64(st + T_DO), dv = desc_sw64(st + T_V);
    if (elect_one()) { umma_ss(tm + C_DP, ddo, dv, id_p, 0); umma_ss(tm + C_DP, ddo + 2, dv + 2, id_p, 1); }
    __syncwarp();
  };
  auto mma_s = [&](int it, int half) {                             // S columns 128 half .. 128 half + 127 (keys 128 half ..)
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t dq = desc_sw64(st + T_Q), dk = desc_sw64(st + T_K + half * 128 * 64);
    if (elect_one()) { umma_ss(tm + C_S + 128 * half, dq, dk, id_h, 0); umma_ss(tm + C_S + 128 * half, dq + 2, dk + 2, id_h, 1); }
    __syncwarp();
  };
  auto mma_d = [&](int it) {                                       // dQ = dS K (A = packed bf16 dS over S, B = the K tile MN-major)
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t dk = desc_sw64(st + T_K);
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < KN / 16; ++ks) umma_ts(tm + C_DQ, tm + C_S + 8 * ks, dk + (uint64_t)(ks * 1024 >> 4), id_d, ks);
      tc_commit(dready);
      tc_commit(&empty[it % NST]);
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0) for (int i = 0; i < NST && i < nit; ++i) load(i);
    __syncwarp();
    if (nit > 0) {
      mbar_wait(&full[0], 0);
      tc_fence_after();
      mma_s(0, 0); mma_s(0, 1); mma_dp(0);
      if (elect_one()) tc_commit(sready);
      __syncwarp();
    }
  }
  {

  const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
  const float* sbq = sbias + qd * SPAN * BSTR + lane;
  const bool tri = qt == 3;
  const int cell_a = tri ? 0 : qt + 1, l2 = lane >> 1;
  float acc[32], ext = 0.f;                                          // dbias of this row's cells, summed over the samples
#pragma unroll
  for (int j = 0; j < 32; ++j) acc[j] = 0.f;
  auto drain = [&](int itd) {                                        // warps with qt < 2: 32 queries x 16 columns of dQ
    uint32_t v[16];
    tmem_ld16(tl + C_DQ + 16 * qt, v);
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
  for (int it = 0; it < nit; ++it) {
    const int s = it % NST;
    mbar_wait(sready, it & 1);
    tc_fence_after();
    const float lse = reinterpret_cast<const float*>(sm + s * STAGE + T_L)[32 * qd + lane];
    const float dd = reinterpret_cast<const float*>(sm + s * STAGE + T_D)[32 * qd + lane];
    uint32_t w[16], ov = 0u;                                          // packed dS of cell_a (the triangle pair: merged, see below)
    // second == false: cell `cell` into w / acc. second == true (cell 4 after cell 0): its words left of the diagonal word replace cell 0's
    // (zero there), the diagonal word goes to ov; its values add into acc except the diagonal one, which goes to ext.
    auto cell_ds = [&](int cell, bool second) {
      const float* sb = sbq + 32 * cell * BSTR;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {                               // 16 columns at a time
        uint32_t sv[16], dv[16];
        tmem_ld16(tl + C_S + 32 * qd + 32 * cell + 16 * hf, sv);
        tmem_ld16(tl + C_DP + 32 * qd + 32 * cell + 16 * hf, dv);
        tmem_wait_ld();
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
          const int b = 16 * hf + j;
          const float p0 = ex2f(fmaf(__uint_as_float(sv[j]), QSCALE * LOG2E, sb[b * BSTR]) - lse);
          const float p1 = ex2f(fmaf(__uint_as_float(sv[j + 1]), QSCALE * LOG2E, sb[(b + 1) * BSTR]) - lse);
          const float t0 = p0 * (__uint_as_float(dv[j]) - dd), t1 = p1 * (__uint_as_float(dv[j + 1]) - dd);
          const uint32_t pw = pack_bf16(t0, t1);
          if (!second) {
            acc[b] += t0; acc[b + 1] += t1;
            w[b >> 1] = pw;
          } else {
            acc[b] += b == lane ? 0.f : t0; acc[b + 1] += b + 1 == lane ? 0.f : t1;
            ext += b == lane ? t0 : (b + 1 == lane ? t1 : 0.f);
            if ((b >> 1) < l2) w[b >> 1] = pw;
            if ((b >> 1) == l2) ov = pw;
          }
        }
      }
    };
    cell_ds(cell_a, false);
    if (tri) cell_ds(4, true);
    tc_fence_before();
    named_bar_sync(1 + qd, 128);                                     // the quadrant's warps have read S
    tc_fence_after();
    if (!tri) {
      tmem_st16(tl + C_S + 16 * qd + 16 * cell_a, w);
      tmem_st16z(tl + C_S + (qt < qd ? 16 * qt : 16 * qt + 80));     // the 48 packed columns outside this window's band
    } else {
      uint32_t w4[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) w4[j] = j < l2 ? w[j] : (j == l2 ? ov : 0u);
      tmem_st16(tl + C_S + 16 * qd + 64, w4);
#pragma unroll
      for (int j = 0; j < 16; ++j) w[j] = j < l2 ? 0u : w[j];
      tmem_st16(tl + C_S + 16 * qd, w);
    }
    tmem_wait_st();
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(tready);
    if (ctl) {
      mbar_wait(tready, it & 1);                                     // dS(it) written, S / dP(it) read by all
      tc_fence_after();
      mma_d(it);
      if (it + 1 < nit) {
        mbar_wait(&full[(it + 1) % NST], ((it + 1) / NST) & 1);
        tc_fence_after();
        mma_dp(it + 1);
        mma_s(it + 1, 0);                                            // over dS(it): runs after dQ(it) (MMAs execute in order)
      }
    }
    mbar_wait(dready, it & 1);
    tc_fence_after();
    if (qt < 2) {
      drain(it);
      tc_fence_before(); __syncwarp();
      if (lane == 0) mbar_arrive(dfree);
    }
    if (ctl) {
      if (it + 1 < nit) {
        mbar_wait(dfree, it & 1);                                    // dQ(it) drained from cols 128..159
        tc_fence_after();
        mma_s(it + 1, 1);
        if (elect_one()) tc_commit(sready);
        __syncwarp();
      }
      if (it + NST < nit) {
        mbar_wait(&empty[it % NST], (it / NST) & 1);
        if (lane == 0) load(it + NST);
        __syncwarp();
      }
    }
  }
  // the bias gradient of this row's cells (natural units)
  float* dst = dbias + (SP > 1 ? (size_t)part * NH * nwin * WQ * SPAN : 0) + (((size_t)h * nwin + wq) * WQ + lane) * SPAN;
  if (!tri) {
#pragma unroll
    for (int j = 0; j < 32; j += 4) *reinterpret_cast<float4*>(dst + 32 * cell_a + j) = make_float4(acc[j], acc[j + 1], acc[j + 2], acc[j + 3]);
  } else {
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      float v0[4], v4[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v0[e] = j + e >= lane ? acc[j + e] : 0.f;
        v4[e] = j + e < lane ? acc[j + e] : (j + e == lane ? ext : 0.f);
      }
      *reinterpret_cast<float4*>(dst + j) = make_float4(v0[0], v0[1], v0[2], v0[3]);
      *reinterpret_cast<float4*>(dst + 128 + j) = make_float4(v4[0], v4[1], v4[2], v4[3]);
    }
  }
  }
  if (lane == 0) tma_store_wait0();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
