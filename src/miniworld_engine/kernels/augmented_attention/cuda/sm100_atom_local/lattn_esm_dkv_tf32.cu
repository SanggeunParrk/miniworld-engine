// lattn_esm_dkv_tf32.cu — dK and dV of ESMFold2's atom attention for the fp32 path, tcgen05 kind::tf32 (sm_100a): the fp32 twin of
// lattn_esm_dkv.cu.
//
//   P^T = exp2(S^T c + bias2 - LSE)   dS^T = P^T (dP^T - D)   dV = P^T dO   dK = dS^T Q / sqrt(32)        (fp32 into dkv[row, 128 / 256 + ..])
//
// A CTA owns 128 keys [k0, k0 + 128) of one head and a range of samples, seen by the 256 queries [k0 - 64, k0 + 192). TMEM lane = key. In fp32
// P^T / dS^T are as wide as S^T / dP^T and dV / dK need their own columns, so the queries go in two halves of 128 through one buffer: S^T_h
// cols 0..127, dP^T_h cols 128..255, P^T_h / dS^T_h in place over them (every warp writes exactly the cells it read), dV += P^T_h dO_h into
// cols 256..287, dK += dS^T_h Q_h into 288..319; the second half's S^T / dP^T are issued right after the first half's dV / dK (the MMAs execute
// in order). q and dO are read K-major (S^T, dP^T) and MN-major (dK, dV: the 128-B swizzle with 32-B atoms) -- one stage of 162 KB, the next
// sample's loads issued once this sample's MMAs are done. Warp w (kq = w & 3, qt = w >> 2: cell qt + 1, or the triangular cells 0 and 4 for
// qt = 3; a warp works in the half its cell lies in) holds the bias of its cells in registers. Every warp drains 32 keys x 16 columns of dK
// (qt < 2) or dV (qt >= 2), stored by the threads. Warp 15 also issues the TMA loads and the MMAs.
// Budgets: smem 162 KB; TMEM 512 columns; 512 threads.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, SPAN = 160, KOFF = 64, KR = 128, QR = 256;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_DO = QR * 128, T_QM = 2 * QR * 128, T_DOM = 3 * QR * 128, T_K = 4 * QR * 128, T_V = T_K + KR * 128, T_L = T_V + KR * 128,
              T_D = T_L + QR * 4, STAGE = T_D + QR * 4;
static_assert(STAGE % 1024 == 0 && T_K % 1024 == 0, "stage");
constexpr int O_BAR = STAGE, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES == 166144, "keep sm100_atom_local.KERNELS_ESM32 in step");
constexpr int C_S = 0, C_DP = 128, C_DV = 256, C_DK = 288;
constexpr uint32_t TX = STAGE;
constexpr uint32_t I_H = idesc_tf32(128, 128), I_D = idesc_tf32(128, DH, 0, 1);

DEVI uint32_t rna_tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

DEVI void tmem_ld8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}

extern "C" __global__ void __launch_bounds__(512, 1)
local_attn_esm_dkv_tf32(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_qm, const __grid_constant__ CUtensorMap tm_k,
                        const __grid_constant__ CUtensorMap tm_v, const __grid_constant__ CUtensorMap tm_do, const __grid_constant__ CUtensorMap tm_dom,
                        const __grid_constant__ CUtensorMap tm_l, const __grid_constant__ CUtensorMap tm_d, const float* __restrict__ gbias,
                        float* __restrict__ dkv, int ldd, int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *empty = bars + 1, *sready = bars + 2, *tready = bars + 3, *dready = bars + 4, *dfree = bars + 5;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int k0 = 128 * jc, qb = k0 - KOFF;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 15;

  if (tid == 0) {
    mbar_init(full, 1); mbar_init(empty, 1); mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(dready, 1); mbar_init(dfree, 16);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int kq = warp & 3, qt = warp >> 2;
  const bool tri = qt == 3;
  const int cell_a = tri ? 0 : qt + 1;
  float bl[32], bx = -INFINITY;                                      // bias(query jj of the cell, this key): the triangle pair merged
  {
    auto src = [&](int c) {                                          // cell c: window 4 jc - 2 + kq + c, key column lane + 128 - 32 c
      const int w = 4 * jc - 2 + kq + c;
      return (w >= 0 && w < nwin) ? gbias + (((size_t)h * nwin + w) * WQ) * SPAN + lane + 128 - 32 * c : nullptr;
    };
    const float* sa = src(cell_a);
    const float* s4 = tri ? src(4) : nullptr;
#pragma unroll
    for (int jj = 0; jj < 32; ++jj) {
      const float va = sa ? __ldg(sa + jj * SPAN) : -INFINITY;
      if (!tri) bl[jj] = va;
      else {
        const float v4 = s4 ? __ldg(s4 + jj * SPAN) : -INFINITY;
        bl[jj] = jj >= lane ? va : v4;
        if (jj == lane) bx = v4;
      }
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;

  auto load = [&](int it) {
    const int a = a_lo + it;
    mbar_expect_tx(full, TX);
    tma_load_3d(su + T_Q, &tm_q, full, h * DH, qb, a);
    tma_load_3d(su + T_DO, &tm_do, full, h * DH, qb, a);
    tma_load_3d(su + T_QM, &tm_qm, full, h * DH, qb, a);
    tma_load_3d(su + T_DOM, &tm_dom, full, h * DH, qb, a);
    tma_load_3d(su + T_K, &tm_k, full, h * DH, k0, a);
    tma_load_3d(su + T_V, &tm_v, full, h * DH, k0, a);
    tma_load_3d(su + T_L, &tm_l, full, qb, h, a);
    tma_load_3d(su + T_D, &tm_d, full, qb, h, a);
  };
  auto mma_st = [&](int hh) {                                        // S^T_h = K Q_h^T, dP^T_h = V dO_h^T (queries 128 hh ..)
    const uint64_t dk = desc_k128(su + T_K), dv = desc_k128(su + T_V);
    const uint64_t dq = desc_k128(su + T_Q + hh * 128 * 128), ddo = desc_k128(su + T_DO + hh * 128 * 128);
    if (elect_one()) {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        umma_ss_tf32(tm + C_S, dk + (uint64_t)(2 * k), dq + (uint64_t)(2 * k), I_H, k > 0 ? 1u : 0u);
        umma_ss_tf32(tm + C_DP, dv + (uint64_t)(2 * k), ddo + (uint64_t)(2 * k), I_H, k > 0 ? 1u : 0u);
      }
      tc_commit(sready);
    }
    __syncwarp();
  };
  auto mma_d = [&](int hh) {                                         // dV (+)= P^T_h dO_h, dK (+)= dS^T_h Q_h (B MN-major, rows 128 hh ..)
    const uint64_t ddo = desc_mn32b(su + T_DOM + hh * 128 * 128, QR * 128), dq = desc_mn32b(su + T_QM + hh * 128 * 128, QR * 128);
    if (elect_one()) {
#pragma unroll 4
      for (int k = 0; k < 16; ++k) {
        const uint64_t boff = (uint64_t)((k * 1024) >> 4);
        const uint32_t acc = (hh > 0 || k > 0) ? 1u : 0u;
        umma_ts_tf32(tm + C_DV, tm + C_S + 8 * k, ddo + boff, I_D, acc);
        umma_ts_tf32(tm + C_DK, tm + C_DP + 8 * k, dq + boff, I_D, acc);
      }
    }
    __syncwarp();
  };
  if (ctl && nit > 0) {
    if (lane == 0) load(0);
    __syncwarp();
    mbar_wait(full, 0);
    tc_fence_after();
    mma_st(0);
  }

  const uint32_t tl = tm + ((uint32_t)(32 * kq) << 16);
  const int za = qt < kq ? qt : qt + 5;                              // the out-of-band query cell this warp zeroes (qt < 3)
  for (int it = 0; it < nit; ++it) {
    const float* sl = reinterpret_cast<const float*>(sm + T_L);
    const float* sd = reinterpret_cast<const float*>(sm + T_D);
#pragma unroll 1
    for (int hh = 0; hh < 2; ++hh) {
      mbar_wait(sready, (2 * it + hh) & 1);
      tc_fence_after();
      auto cell_pd = [&](int c, bool second) {
        const int ca = kq + c;                                       // the query cell, 0 .. 7 (half ca >> 2)
        if ((ca >> 2) != hh) return;
        const int x0 = 32 * (ca & 3);                                // column of the cell in the half
#pragma unroll
        for (int hf = 0; hf < 4; ++hf) {                             // 8 columns at a time (128 registers)
          uint32_t sv[8], dv[8];
          tmem_ld8(tl + C_S + x0 + 8 * hf, sv);
          tmem_ld8(tl + C_DP + x0 + 8 * hf, dv);
          tmem_wait_ld();
#pragma unroll
          for (int j = 0; j < 8; j += 2) {
            const int b = 8 * hf + j, x = 32 * ca + b;
            const float2 l2 = *reinterpret_cast<const float2*>(sl + x), d2 = *reinterpret_cast<const float2*>(sd + x);
            float bb[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int q = b + e;
              bb[e] = !tri ? bl[q] : (!second ? (q >= lane ? bl[q] : -INFINITY) : (q < lane ? bl[q] : (q == lane ? bx : -INFINITY)));
            }
            const float p0 = ex2f(fmaf(__uint_as_float(sv[j]), QSCALE * LOG2E, bb[0]) - l2.x);
            const float p1 = ex2f(fmaf(__uint_as_float(sv[j + 1]), QSCALE * LOG2E, bb[1]) - l2.y);
            const float t0 = p0 * (__uint_as_float(dv[j]) - d2.x), t1 = p1 * (__uint_as_float(dv[j + 1]) - d2.y);
            sv[j] = rna_tf32(p0); sv[j + 1] = rna_tf32(p1);
            dv[j] = rna_tf32(t0); dv[j + 1] = rna_tf32(t1);
          }
          tmem_st8(tl + C_S + x0 + 8 * hf, sv);
          tmem_st8(tl + C_DP + x0 + 8 * hf, dv);
        }
      };
      cell_pd(cell_a, false);
      if (tri) cell_pd(4, true);
      else if ((za >> 2) == hh) {
        const int x0 = 32 * (za & 3);
        tmem_st16z(tl + C_S + x0); tmem_st16z(tl + C_S + x0 + 16);
        tmem_st16z(tl + C_DP + x0); tmem_st16z(tl + C_DP + x0 + 16);
      }
      tmem_wait_st();
      tc_fence_before(); __syncwarp();
      if (lane == 0) mbar_arrive(tready);
      if (ctl) {
        mbar_wait(tready, (2 * it + hh) & 1);                        // P^T_h / dS^T_h written by all
        tc_fence_after();
        if (hh == 0 && it > 0) { mbar_wait(dfree, (it - 1) & 1); tc_fence_after(); }   // dV / dK(it - 1) drained
        mma_d(hh);
        if (hh == 0) mma_st(1);                                      // over P^T_0 / dS^T_0: after their dV / dK
        else {
          if (elect_one()) { tc_commit(dready); tc_commit(empty); }
          __syncwarp();
          if (it + 1 < nit) {
            mbar_wait(empty, it & 1);                                // this sample's MMAs are done: the stage is free
            if (lane == 0) load(it + 1);
            __syncwarp();
            mbar_wait(full, (it + 1) & 1);
            tc_fence_after();
            mma_st(0);
          }
        }
      }
    }
    mbar_wait(dready, it & 1);
    tc_fence_after();
    {                                                                // 32 keys x 16 columns of dK (qt 0, 1) / dV (2, 3), stored by the threads
      uint32_t v[16];
      tmem_ld16(tl + (qt < 2 ? C_DK : C_DV) + 16 * (qt & 1), v);
      tmem_wait_ld();
      const float sc = qt < 2 ? QSCALE : 1.f;
      float* dst = dkv + ((size_t)(a_lo + it) * N + k0 + 32 * kq + lane) * ldd + (qt < 2 ? 128 : 256) + h * DH + 16 * (qt & 1);
#pragma unroll
      for (int g = 0; g < 4; ++g)
        *reinterpret_cast<float4*>(dst + 4 * g) = make_float4(__uint_as_float(v[4 * g]) * sc, __uint_as_float(v[4 * g + 1]) * sc,
                                                              __uint_as_float(v[4 * g + 2]) * sc, __uint_as_float(v[4 * g + 3]) * sc);
      tc_fence_before(); __syncwarp();
      if (lane == 0) mbar_arrive(dfree);
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
