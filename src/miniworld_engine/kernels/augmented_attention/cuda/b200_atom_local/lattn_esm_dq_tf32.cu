// lattn_esm_dq_tf32.cu — dQ and the bias gradient of ESMFold2's atom attention for the fp32 path, tcgen05 kind::tf32 (sm_100a): the fp32 twin
// of lattn_esm_dq.cu with lattn_dq_tf32.cu's fp32 conventions.
//
//   P = exp2(S c + bias2 - LSE)   dP = dO V^T   dS = P (dP - D)   dQ = dS K / sqrt(32)   dbias[h][w][i][c] = sum over the samples of dS
//
// q, k, v, dO fp32 [A, N, 128] views (tf32-rounded: the forward's q / k, v, and dO from the gate backward); dQ fp32 into dq[row, h * 32 + d]
// (row stride ldd) through per-warp TMA stores, staged in the MN-major k slot once dQ has read it. A CTA owns 128 queries of one head and a range of samples, the 256 key positions [q0 - 64, q0 + 192).
// TMEM lane = query: S cols 0..255, dP cols 256..511. dS is fp32 and goes in place over S: every warp writes exactly the cells it read (no
// barrier between readers and writers); warps qt < 3 zero one out-of-band block. dQ = dS K (B = K a second time, MN-major: one slot, the 128-B
// swizzle with 32-B atoms) into cols 256..287 once dP is read; the next sample's S is issued right after it, its dP once dQ is drained.
// Warp w (qd = w & 3, qt = w >> 2: cell qt + 1, or the triangular cells 0 and 4 for qt = 3) holds the bias and the dbias sums of its cells in
// registers, each the triangle pair merged in 32 + 1 (cell 0 is -inf left of the diagonal, cell 4 right of it), the bias loaded once; LSE / D
// are read before the sample's MMAs are waited for. Warp 15 also issues the TMA loads (2 stages: q | dO | k | v, 96 KB) and the MMAs.
// Budgets: smem 2 x 96 KB + 32 KB (MN-major k) = 224 KB; TMEM 512 columns; 512 threads.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, SPAN = 160, KOFF = 64, QN = 128, KN = 256, NST = 2;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_DO = QN * 128, T_K = 2 * QN * 128, T_V = T_K + KN * 128;
constexpr int STAGE = T_V + KN * 128;                               // 96 KB (LSE / D: one global load per thread)
static_assert(STAGE % 1024 == 0, "stage");
constexpr int O_KM = NST * STAGE, O_BAR = O_KM + KN * 128, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES == 229632, "keep sm100_atom_local.KERNELS_ESM32 in step");
constexpr int C_S = 0, C_DP = 256, C_DQ = 256;
constexpr uint32_t TX = STAGE;
constexpr uint32_t I_S = idesc_tf32(128, KN), I_D = idesc_tf32(128, DH, 0, 1);

DEVI uint32_t rna_tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

DEVI void tmem_ld4(uint32_t taddr, uint32_t (&r)[4]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(taddr));
}
DEVI void tmem_st4(uint32_t taddr, const uint32_t (&r)[4]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}
DEVI void tmem_ld8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}

extern "C" __global__ void __launch_bounds__(512, 1)
local_attn_esm_dq_tf32(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                       const __grid_constant__ CUtensorMap tm_do, const float* __restrict__ lse_g, const float* __restrict__ dd_g,
                       const __grid_constant__ CUtensorMap tm_km, const float* __restrict__ gbias, const __grid_constant__ CUtensorMap tm_dq,
                       float* __restrict__ dbias, int N, int A, int nwin, int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  // barrier addresses rematerialised from the shared base at each use (kept as live 64-bit pointers they cost 16 registers)
#define BARS reinterpret_cast<uint64_t*>(sm + O_BAR)
#define full (BARS)
#define empty (BARS + NST)
#define sready (BARS + 2 * NST)
#define tready (BARS + 2 * NST + 1)
#define dready (BARS + 2 * NST + 2)
#define dfree (BARS + 2 * NST + 3)
#define kfull (BARS + 2 * NST + 4)
#define sfree (BARS + 2 * NST + 5)
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int q0 = 128 * jc, kb = q0 - KOFF;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 15;

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 1); }
    mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(dready, 1); mbar_init(dfree, 8); mbar_init(kfull, 1); mbar_init(sfree, 8);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int wq = 4 * jc + qd;
  const bool tri = qt == 3;
  const int cell_a = tri ? 0 : qt + 1;
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;

  auto load = [&](int it) {
    const int s = it % NST, a = a_lo + it;
    const uint32_t st = su + s * STAGE;
    mbar_expect_tx(&full[s], TX);
    tma_load_3d(st + T_Q, &tm_q, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_DO, &tm_do, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_K, &tm_k, &full[s], h * DH, kb, a);
    tma_load_3d(st + T_V, &tm_v, &full[s], h * DH, kb, a);
  };
  auto load_km = [&](int it) {
    mbar_expect_tx(kfull, KN * 128);
    tma_load_3d(su + O_KM, &tm_km, kfull, h * DH, kb, a_lo + it);
  };
  auto mma_s = [&](int it) {                                         // S = Q K^T into cols 0..255
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t dq_ = desc_k128(st + T_Q), dk = desc_k128(st + T_K);
    if (elect_one()) {
#pragma unroll 1
      for (int k = 0; k < 4; ++k) umma_ss_tf32(tm + C_S, dq_ + (uint64_t)(2 * k), dk + (uint64_t)(2 * k), I_S, k > 0 ? 1u : 0u);
    }
    __syncwarp();
  };
  auto mma_dp = [&](int it) {                                        // dP = dO V^T into cols 256..511
    const uint32_t st = su + (it % NST) * STAGE;
    const uint64_t ddo = desc_k128(st + T_DO), dv = desc_k128(st + T_V);
    if (elect_one()) {
#pragma unroll 1
      for (int k = 0; k < 4; ++k) umma_ss_tf32(tm + C_DP, ddo + (uint64_t)(2 * k), dv + (uint64_t)(2 * k), I_S, k > 0 ? 1u : 0u);
      tc_commit(sready);
    }
    __syncwarp();
  };
  auto mma_d = [&](int it) {                                         // dQ = dS K (A = fp32 dS over S, B = the MN-major k slot)
    const uint64_t dk = desc_mn32b(su + O_KM, KN * 128);
    if (elect_one()) {
#pragma unroll 1
      for (int k = 0; k < KN / 8; ++k) umma_ts_tf32(tm + C_DQ, tm + C_S + 8 * k, dk + (uint64_t)((k * 1024) >> 4), I_D, k > 0 ? 1u : 0u);
      tc_commit(dready);
    }
    __syncwarp();
  };
  if (ctl && nit > 0) {
    if (lane == 0) { for (int i = 0; i < NST && i < nit; ++i) load(i); load_km(0); }
    __syncwarp();
    mbar_wait(&full[0], 0);
    tc_fence_after();
    mma_s(0); mma_dp(0);
  }

  const int z0 = 32 * (qt < qd ? qt : qt + 5);
  float acc[32], ext = 0.f;
#pragma unroll
  for (int j = 0; j < 32; ++j) acc[j] = 0.f;
  float bl[32], bx = -INFINITY;                                      // this row's bias over its cells (log2 units), the triangle pair merged:
  {                                                                  // cell 0 is -inf left of the diagonal, cell 4 right of it
    const float* brow = gbias + (((size_t)h * nwin + wq) * WQ + lane) * SPAN;
    if (!tri) {
#pragma unroll
      for (int j = 0; j < 32; j += 4) {
        const float4 b4 = __ldg(reinterpret_cast<const float4*>(brow + 32 * cell_a + j));
        bl[j] = b4.x; bl[j + 1] = b4.y; bl[j + 2] = b4.z; bl[j + 3] = b4.w;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 32; ++j) bl[j] = __ldg(brow + (j >= lane ? j : 128 + j));
      bx = __ldg(brow + 128 + lane);
    }
  }
  for (int it = 0; it < nit; ++it) {
    const size_t lrow = ((size_t)(a_lo + it) * NH + h) * N + q0 + 32 * qd + lane;
    const float lse = __ldg(lse_g + lrow), dd = __ldg(dd_g + lrow);   // issued before the wait: their latency overlaps the MMAs
    mbar_wait(sready, it & 1);
    tc_fence_after();
    const uint32_t tl = *reinterpret_cast<volatile uint32_t*>(tptr) + ((uint32_t)(32 * qd) << 16);
    auto cell_ds = [&](int cell, bool second) {                      // dS (rounded to tf32) over this cell's S columns, dbias sums
#pragma unroll
      for (int hf = 0; hf < 8; ++hf) {                               // 4 columns at a time (128 registers)
        uint32_t sv[4], dv[4];
        const uint32_t col = tl + C_S + 32 * qd + 32 * cell + 4 * hf;
        tmem_ld4(col, sv);
        tmem_ld4(tl + C_DP + 32 * qd + 32 * cell + 4 * hf, dv);
        tmem_wait_ld();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int b = 4 * hf + j;
          const float bias = !tri ? bl[b] : (!second ? (b >= lane ? bl[b] : -INFINITY) : (b < lane ? bl[b] : (b == lane ? bx : -INFINITY)));
          const float p = ex2f(fmaf(__uint_as_float(sv[j]), QSCALE * LOG2E, bias) - lse);
          const float t = p * (__uint_as_float(dv[j]) - dd);
          if (!second) acc[b] += t;
          else { acc[b] += b == lane ? 0.f : t; ext += b == lane ? t : 0.f; }
          sv[j] = rna_tf32(t);
        }
        tmem_st4(col, sv);
      }
    };
    cell_ds(cell_a, false);
    if (tri) cell_ds(4, true);
    else { tmem_st16z(tl + C_S + z0); tmem_st16z(tl + C_S + z0 + 16); }
    tmem_wait_st();
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(tready);
    if (ctl) {
      mbar_wait(tready, it & 1);                                     // dS(it) written, dP(it) read by all
      mbar_wait(kfull, it & 1);
      tc_fence_after();
      mma_d(it);
      if (it + NST < nit) {                                          // stage of it free: S / dP(it) consumed, LSE / D read (tready)
        if (lane == 0) load(it + NST);
        __syncwarp();
      }
      if (it + 1 < nit) {
        mbar_wait(&full[(it + 1) % NST], ((it + 1) / NST) & 1);
        tc_fence_after();
        mma_s(it + 1);                                               // over dS(it): after dQ(it)
      }
    }
    mbar_wait(dready, it & 1);
    tc_fence_after();
    if (qt < 2) {                                                    // dQ: 32 queries x 16 columns per warp, staged in the MN-major k slot
      const uint32_t ob = su + O_KM + warp * 2048;                   // (dQ(it), its reader, is done) -> per-warp TMA stores; [32][64 B], 64-B swizzle
#pragma unroll
      for (int x = 0; x < 2; ++x) {                                  // 8 columns at a time (the bias and dbias registers stay live)
        uint32_t v[8];
        tmem_ld8(tl + C_DQ + 16 * qt + 8 * x, v);
        tmem_wait_ld();
#pragma unroll
        for (int g = 0; g < 2; ++g)
          sts128(ob + sw64(lane, 2 * x + g), make_uint4(__float_as_uint(__uint_as_float(v[4 * g]) * QSCALE), __float_as_uint(__uint_as_float(v[4 * g + 1]) * QSCALE),
                                                        __float_as_uint(__uint_as_float(v[4 * g + 2]) * QSCALE), __float_as_uint(__uint_as_float(v[4 * g + 3]) * QSCALE)));
      }
      tc_fence_before(); __syncwarp();
      if (lane == 0) mbar_arrive(dfree);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) {
        tma_store_3d(&tm_dq, ob, h * DH + 16 * qt, q0 + 32 * qd, a_lo + it);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(sfree);
      }
      __syncwarp();
    }
    if (ctl) {
      if (it + 1 < nit) {
        mbar_wait(dfree, it & 1);                                    // dQ(it) drained from cols 256..287
        tc_fence_after();
        mma_dp(it + 1);
        mbar_wait(sfree, it & 1);                                    // the staging read by the stores (dQ(it) done before): the k slot is free
        if (lane == 0) load_km(it + 1);
        __syncwarp();
      }
    }
  }
  if (lane == 0) tma_store_wait0();
  uint32_t t2;                                                       // the thread index re-read: the row offset is not kept live from the bias
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t2));                   // load (it was spilled)
  const int wq2 = 4 * jc + ((t2 >> 5) & 3);
  float* dstb = dbias + (SP > 1 ? (size_t)part * NH * nwin * WQ * SPAN : 0) + (((size_t)h * nwin + wq2) * WQ + (t2 & 31)) * SPAN;
  if (!tri) {
#pragma unroll
    for (int j = 0; j < 32; j += 4) *reinterpret_cast<float4*>(dstb + 32 * cell_a + j) = make_float4(acc[j], acc[j + 1], acc[j + 2], acc[j + 3]);
  } else {
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      float v0[4], v4[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v0[e] = j + e >= lane ? acc[j + e] : 0.f;
        v4[e] = j + e < lane ? acc[j + e] : (j + e == lane ? ext : 0.f);
      }
      *reinterpret_cast<float4*>(dstb + j) = make_float4(v0[0], v0[1], v0[2], v0[3]);
      *reinterpret_cast<float4*>(dstb + 128 + j) = make_float4(v4[0], v4[1], v4[2], v4[3]);
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
#undef full
#undef empty
#undef sready
#undef tready
#undef dready
#undef dfree
#undef kfull
#undef sfree
#undef BARS
