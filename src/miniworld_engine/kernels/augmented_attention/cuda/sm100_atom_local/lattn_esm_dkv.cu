// lattn_esm_dkv.cu — dK and dV of ESMFold2's atom attention on sm_100a (``LocalDiTBlock(**ESMFOLD2)`` training; tcgen05 / TMEM / TMA), the
// key-centric half of lattn_esm.cu's backward (the bias gradient is lattn_esm_dq.cu's):
//
//   P^T = exp2(S^T c + bias2 - LSE)   dS^T = P^T (dP^T - D)   dV = P^T dO   dK = dS^T Q / sqrt(32)
//
// A CTA owns 128 keys [k0, k0 + 128) of one head and a range of samples; they are seen by the 256 queries [k0 - 64, k0 + 192) (key j by the
// queries within +/- 64 of it). TMEM lane = key: S^T = K Q^T cols 0..255, dP^T = V dO^T cols 256..511. Warp w < 16 (lane quadrant kq = w & 3:
// keys k0 + 32 kq .., which see the five query cells kq .. kq + 4; qt = w >> 2: cell qt + 1, or the triangular cells 0 and 4 for qt = 3) reads
// its cells, computes P^T and dS^T (the bias of query q, key j is bias2[h][q / 32][q % 32][j - 32 (q / 32) + 64], held per warp in shared
// memory), and -- after the quadrant's warps have read -- writes both as packed bf16, P^T over S^T cols 0..127, dS^T over dP^T cols 256..383
// (out-of-band cells zero): the TMEM A operands of dV = P^T dO into cols 128..159 and dK = dS^T Q into cols 384..415. Every warp drains 32 keys
// x 16 columns of dK (qt < 2) or dV (qt >= 2) with a TMA store into dkv[row, 128 + h * 32 + d] / [.., 256 + ..] (row stride ldd). Warp 15
// also issues the TMA loads (2 stages) and the MMAs: after dV / dK(it), the first halves of S^T / dP^T(it + 1) at once, the second halves (whose
// columns hold dV / dK) once those are drained.
// Budgets: smem 2 x 50 KB stages + 80 KB bias + 16 KB staging; TMEM 512 columns; 512 threads (<= 128 registers).
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, SPAN = 160, KOFF = 64, KR = 128, QR = 256, NST = 2;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_DO = QR * 64, T_K = 2 * QR * 64, T_V = T_K + KR * 64, T_L = T_V + KR * 64, T_D = T_L + QR * 4;
constexpr int STAGE = 51200;
static_assert(T_D + QR * 4 <= STAGE && STAGE % 1024 == 0, "stage");
constexpr int O_B = NST * STAGE, O_O = O_B + 4 * 5 * 1024 * 4, O_BAR = O_O + 16 * 1024;
constexpr int SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES == 200960, "keep sm100_atom_local.KERNELS_ESM_TRAIN['esm_dkv'] in step");
constexpr int C_S = 0, C_DV = 128, C_DP = 256, C_DK = 384, C_PP = 0, C_PS = 256;
constexpr uint32_t TX = 2 * QR * 64 + 2 * KR * 64 + 2 * QR * 4;

DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

DEVI void tmem_ld8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}

extern "C" __global__ void __launch_bounds__(512, 1)
local_attn_esm_dkv(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                   const __grid_constant__ CUtensorMap tm_do, const __grid_constant__ CUtensorMap tm_l, const __grid_constant__ CUtensorMap tm_d,
                   const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias, int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *empty = bars + NST, *sready = bars + 2 * NST, *tready = sready + 1, *dready = sready + 2, *dfree = sready + 3;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  float* sbias = reinterpret_cast<float*>(sm + O_B);                 // [quadrant 4][cell 5][query 32][key 32] (log2 units)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int k0 = 128 * jc, qb = k0 - KOFF;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 15;                                       // also issues the TMA loads and the MMAs

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 1); }
    mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(dready, 1); mbar_init(dfree, 16);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int kq = warp & 3, qt = warp >> 2;
  {                                                                  // cell c of quadrant kq: queries of window 4 jc - 2 + kq + c, key column lane + 128 - 32 c
    for (int cc = 0; cc < (qt < 3 ? 1 : 2); ++cc) {
      const int c = qt < 3 ? qt + 1 : 4 * cc, w = 4 * jc - 2 + kq + c;
      const bool ok = w >= 0 && w < nwin;
      const float* src = gbias + (((size_t)h * nwin + (ok ? w : 0)) * WQ) * SPAN + lane + 128 - 32 * c;
      float* dstc = sbias + (kq * 5 + c) * 1024 + lane;
#pragma unroll 8
      for (int jj = 0; jj < 32; ++jj) dstc[jj * 32] = ok ? src[jj * SPAN] : -INFINITY;
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;
  const uint32_t id_h = idesc_bf16(128, 128, 0, 0), id_d = idesc_bf16(128, 32, 0, 1);

  auto load = [&](int it) {
    const int s = it % NST, a = a_lo + it;
    const uint32_t st = su + s * STAGE;
    mbar_expect_tx(&full[s], TX);
    tma_load_3d(st + T_Q, &tm_q, &full[s], h * DH, qb, a);
    tma_load_3d(st + T_DO, &tm_do, &full[s], h * DH, qb, a);
    tma_load_3d(st + T_K, &tm_k, &full[s], h * DH, k0, a);
    tma_load_3d(st + T_V, &tm_v, &full[s], h * DH, k0, a);
    tma_load_3d(st + T_L, &tm_l, &full[s], qb, h, a);
    tma_load_3d(st + T_D, &tm_d, &full[s], qb, h, a);
  };
  auto mma_st = [&](int it, int half) {                            // S^T, dP^T columns 128 half .. (queries qb + 128 half ..)
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t dk = desc_sw64(st + T_K), dv = desc_sw64(st + T_V);
    const uint64_t dq = desc_sw64(st + T_Q + half * 128 * 64), ddo = desc_sw64(st + T_DO + half * 128 * 64);
    if (elect_one()) {
      umma_ss(tm + C_S + 128 * half, dk, dq, id_h, 0);
      umma_ss(tm + C_DP + 128 * half, dv, ddo, id_h, 0);
      umma_ss(tm + C_S + 128 * half, dk + 2, dq + 2, id_h, 1);
      umma_ss(tm + C_DP + 128 * half, dv + 2, ddo + 2, id_h, 1);
    }
    __syncwarp();
  };
  auto mma_d = [&](int it) {                                       // dV = P^T dO, dK = dS^T Q (A = packed bf16 in TMEM, B MN-major)
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t ddo = desc_sw64(st + T_DO), dq = desc_sw64(st + T_Q);
    if (elect_one()) {
#pragma unroll
      for (int ks = 0; ks < QR / 16; ++ks) {
        const uint64_t boff = (uint64_t)(ks * 1024 >> 4);
        umma_ts(tm + C_DV, tm + C_PP + 8 * ks, ddo + boff, id_d, ks);
        umma_ts(tm + C_DK, tm + C_PS + 8 * ks, dq + boff, id_d, ks);
      }
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
      mma_st(0, 0); mma_st(0, 1);
      if (elect_one()) tc_commit(sready);
      __syncwarp();
    }
  }
  {

  const uint32_t tl = tm + ((uint32_t)(32 * kq) << 16);
  const bool tri = qt == 3;
  const int cell_a = tri ? 0 : qt + 1, l2 = lane >> 1;
  auto drain = [&](int itd) {                                        // per warp: 32 keys x 16 columns of dK (qt 0, 1) / dV (2, 3)
    uint32_t v[16];
    tmem_ld16(tl + (qt < 2 ? C_DK : C_DV) + 16 * (qt & 1), v);
    tmem_wait_ld();
    const float sc = qt < 2 ? QSCALE : 1.f;
    uint8_t* ob = sm + O_O + warp * 1024;
    if (lane == 0) tma_store_wait_read0();
    __syncwarp();
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      uint4 o;
      o.x = pack_bf16(__uint_as_float(v[8 * g + 0]) * sc, __uint_as_float(v[8 * g + 1]) * sc);
      o.y = pack_bf16(__uint_as_float(v[8 * g + 2]) * sc, __uint_as_float(v[8 * g + 3]) * sc);
      o.z = pack_bf16(__uint_as_float(v[8 * g + 4]) * sc, __uint_as_float(v[8 * g + 5]) * sc);
      o.w = pack_bf16(__uint_as_float(v[8 * g + 6]) * sc, __uint_as_float(v[8 * g + 7]) * sc);
      *reinterpret_cast<uint4*>(ob + lane * 32 + 16 * g) = o;
    }
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) {
      tma_store_3d(&tm_o, su + O_O + warp * 1024, (qt < 2 ? 128 : 256) + h * DH + 16 * (qt & 1), k0 + 32 * kq, a_lo + itd);
      tma_store_commit();
    }
  };
  for (int it = 0; it < nit; ++it) {
    const int s = it % NST;
    const float* sl = reinterpret_cast<const float*>(sm + s * STAGE + T_L);
    const float* sd = reinterpret_cast<const float*>(sm + s * STAGE + T_D);
    mbar_wait(sready, it & 1);
    tc_fence_after();
    uint32_t pw[16], dw[16], pov = 0u, dov = 0u;                      // packed P^T / dS^T of cell_a (the triangle pair merged: see lattn_esm_dq.cu)
    auto cell_pd = [&](int c, bool second) {
      const float* sb = sbias + (kq * 5 + c) * 1024 + lane;
      const int x0 = 32 * kq + 32 * c;                               // query column of the cell
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {                               // 16 columns at a time
        uint32_t sv[16], dv[16];
        tmem_ld16(tl + C_S + x0 + 16 * hf, sv);
        tmem_ld16(tl + C_DP + x0 + 16 * hf, dv);
        tmem_wait_ld();
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
          const int b = 16 * hf + j, x = x0 + b;
          const float2 l2v = *reinterpret_cast<const float2*>(sl + x), d2 = *reinterpret_cast<const float2*>(sd + x);
          const float p0 = ex2f(fmaf(__uint_as_float(sv[j]), QSCALE * LOG2E, sb[b * 32]) - l2v.x);
          const float p1 = ex2f(fmaf(__uint_as_float(sv[j + 1]), QSCALE * LOG2E, sb[(b + 1) * 32]) - l2v.y);
          const float t0 = p0 * (__uint_as_float(dv[j]) - d2.x), t1 = p1 * (__uint_as_float(dv[j + 1]) - d2.y);
          const uint32_t pp = pack_bf16(p0, p1), dp = pack_bf16(t0, t1);
          if (!second) { pw[b >> 1] = pp; dw[b >> 1] = dp; }
          else {
            if ((b >> 1) < l2) { pw[b >> 1] = pp; dw[b >> 1] = dp; }
            if ((b >> 1) == l2) { pov = pp; dov = dp; }
          }
        }
      }
    };
    cell_pd(cell_a, false);
    if (tri) cell_pd(4, true);
    tc_fence_before();
    named_bar_sync(1 + kq, 128);                                     // the quadrant's warps have read S^T / dP^T
    tc_fence_after();
    if (!tri) {
      tmem_st16(tl + C_PP + 16 * kq + 16 * cell_a, pw);
      tmem_st16(tl + C_PS + 16 * kq + 16 * cell_a, dw);
      const int z = qt < kq ? 16 * qt : 16 * qt + 80;               // out-of-band packed columns: [0, 16 kq) U [16 kq + 80, 128)
      tmem_st16z(tl + C_PP + z);
      tmem_st16z(tl + C_PS + z);
    } else {
      uint32_t t4[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) t4[j] = j < l2 ? pw[j] : (j == l2 ? pov : 0u);
      tmem_st16(tl + C_PP + 16 * kq + 64, t4);
#pragma unroll
      for (int j = 0; j < 16; ++j) t4[j] = j < l2 ? dw[j] : (j == l2 ? dov : 0u);
      tmem_st16(tl + C_PS + 16 * kq + 64, t4);
#pragma unroll
      for (int j = 0; j < 16; ++j) { pw[j] = j < l2 ? 0u : pw[j]; dw[j] = j < l2 ? 0u : dw[j]; }
      tmem_st16(tl + C_PP + 16 * kq, pw);
      tmem_st16(tl + C_PS + 16 * kq, dw);
    }
    tmem_wait_st();
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(tready);
    if (ctl) {
      mbar_wait(tready, it & 1);                                     // P^T / dS^T(it) written, S^T / dP^T(it) read by all
      tc_fence_after();
      mma_d(it);
      if (it + 1 < nit) {
        mbar_wait(&full[(it + 1) % NST], ((it + 1) / NST) & 1);
        tc_fence_after();
        mma_st(it + 1, 0);                                           // over P^T / dS^T(it): after dV / dK(it) (MMAs execute in order)
      }
    }
    mbar_wait(dready, it & 1);
    tc_fence_after();
    drain(it);
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(dfree);
    if (ctl) {
      if (it + 1 < nit) {
        mbar_wait(dfree, it & 1);                                    // dV / dK(it) drained
        tc_fence_after();
        mma_st(it + 1, 1);
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
  }
  if (lane == 0) tma_store_wait0();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
