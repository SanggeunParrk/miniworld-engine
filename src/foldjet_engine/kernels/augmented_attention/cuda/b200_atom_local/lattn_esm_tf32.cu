// lattn_esm_tf32.cu — ESMFold2's atom attention for the fp32 path (``LocalDiTBlock(**ESMFOLD2)``), tcgen05 kind::tf32 (sm_100a): the fp32
// twin of lattn_esm.cu, with lattn_fwd_tf32.cu's fp32 conventions.
//
//   S = q k^T / sqrt(32) + bias2[h][w][i][j - (32 w - 64)]      O = softmax(S) v      (fp32 O, row LSE [A, 4, N] in log2 units when given)
//
// q, k, v fp32 [A, N, 128] views (any row stride; q / k normalised, rotated and rounded to tf32 by lesm.cu's local_esm_qkrope_f32, v by the
// projection's epilogue). Geometry (lattn_esm.cu): a CTA owns 128 queries [q0, q0 + 128) of one head and a range of samples, the 256 key
// positions [q0 - 64, q0 + 192); window w % 4 = qd sees the 160 of them from 32 qd (five 32-key cells, the outer two triangular).
// TMEM lane = query. P is fp32 and as wide as S, so it goes in place over S and fills its 256 columns: one S buffer (cols 0..255) and O
// (cols 256..287); the next sample's QK^T is issued right after this sample's PV (the MMAs execute in order), so only its latency shows.
// Warp w (lane quadrant qd = w & 3, qt = w >> 2: cell qt + 1, or the triangular cells 0 and 4 for qt = 3) holds the bias of its cells in
// registers (the triangle pair shares 32 + 1: cell 0 is -inf left of the diagonal, cell 4 right of it), reads its cells, exchanges row maxima
// and sums with the quadrant's warps through shared memory and writes P (rounded to tf32, RNA: kind::tf32 truncates) over exactly the
// cells it read; warps qt < 3 zero one of the quadrant's three out-of-band 32-column blocks. Warps qt < 2 drain O (32 rows x 16 fp32 each,
// TMA stores), qt = 2 writes the LSE. Warp 15 also issues the TMA loads (2 stages: q 16 KB | k 32 KB | v 32 KB, v MN-major in the 128-B
// swizzle with 32-B atoms) and the MMAs. Budgets: smem 2 x 80 KB + 8 KB exchange + 16 KB O staging; TMEM 512 columns; 512 threads.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int NH = 4, DH = 32, WQ = 32, SPAN = 160, KOFF = 64, QN = 128, KN = 256, NST = 2;
constexpr float LOG2E = 1.4426950408889634f, QSCALE = 0.17677669529663687f;
constexpr int T_Q = 0, T_K = QN * 128, T_V = T_K + KN * 128, STAGE = T_V + KN * 128;     // 16 KB | 32 KB | 32 KB
constexpr int O_R = NST * STAGE, O_O = O_R + 2 * 2 * 16 * 32 * 4, O_BAR = O_O + 8 * 2048, SMEM_BYTES = O_BAR + 256;
static_assert(STAGE % 1024 == 0 && T_K % 1024 == 0 && T_V % 1024 == 0 && O_O % 1024 == 0, "1 KB alignment of the swizzled tiles");
static_assert(SMEM_BYTES == 188672, "keep sm100_atom_local.KERNELS_ESM32 in step");
constexpr int C_S = 0, C_O = KN;
constexpr uint32_t TX = STAGE;
constexpr uint32_t I_S = idesc_tf32(128, KN), I_O = idesc_tf32(128, DH, 0, 1);

DEVI uint32_t rna_tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI void tmem_st16z(uint32_t taddr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" :: "r"(taddr), "r"(0u) : "memory");
}

extern "C" __global__ void __launch_bounds__(512, 1)
local_attn_esm_tf32(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                    const __grid_constant__ CUtensorMap tm_o, const float* __restrict__ gbias, float* __restrict__ LSE, int N, int A, int nwin,
                    int SP) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + O_BAR);
  uint64_t *full = bars, *empty = bars + NST, *sready = bars + 2 * NST, *tready = sready + 1, *oready = sready + 2;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(sm + O_BAR + 128);
  float* red = reinterpret_cast<float*>(sm + O_R);                   // [kind 2][parity 2][warp 16][lane 32]: row max / row sum partials
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x / SP, part = blockIdx.x - pair * SP;
  const int jc = pair >> 2, h = pair & 3;
  const int q0 = 128 * jc, kb = q0 - KOFF;
  const int a_lo = (int)((long)A * part / SP), a_hi = (int)((long)A * (part + 1) / SP);
  const int nit = a_hi - a_lo;
  const bool ctl = warp == 15;                                       // also issues the TMA loads and the MMAs

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 1); }
    mbar_init(sready, 1); mbar_init(tready, 16); mbar_init(oready, 1);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(smem_u32(tptr), 512);
  const int qd = warp & 3, qt = warp >> 2;
  const int wq = 4 * jc + qd;                                        // this quadrant's window; lane = query of it
  const bool tri = qt == 3;
  const int cell_a = tri ? 0 : qt + 1;
  float bl[32], bx = -INFINITY;                                      // the bias of this row's cells (log2 units); bx: cell 4's diagonal
  {
    const float4* src = reinterpret_cast<const float4*>(gbias + (((size_t)h * nwin + wq) * WQ + lane) * SPAN);
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const float4 a4 = __ldg(src + 8 * cell_a + g);
      const float aa[4] = {a4.x, a4.y, a4.z, a4.w};
      if (!tri) {
#pragma unroll
        for (int e = 0; e < 4; ++e) bl[4 * g + e] = aa[e];
      } else {
        const float4 c4 = __ldg(src + 32 + g);
        const float cc[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int j = 4 * g + e;
          bl[j] = j >= lane ? aa[e] : cc[e];
          if (j == lane) bx = cc[e];
        }
      }
    }
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = *tptr;

  auto load = [&](int it) {
    const int s = it % NST, a = a_lo + it;
    const uint32_t st = su + s * STAGE;
    mbar_expect_tx(&full[s], TX);
    tma_load_3d(st + T_Q, &tm_q, &full[s], h * DH, q0, a);
    tma_load_3d(st + T_K, &tm_k, &full[s], h * DH, kb, a);
    tma_load_3d(st + T_V, &tm_v, &full[s], h * DH, kb, a);
  };
  auto mma_s = [&](int it) {                                         // S of sample it (over P of it - 1: after its PV)
    const int s = it % NST;
    const uint32_t st = su + s * STAGE;
    mbar_wait(&full[s], (it / NST) & 1);
    tc_fence_after();
    const uint64_t dq = desc_k128(st + T_Q), dk = desc_k128(st + T_K);
    if (elect_one()) {
#pragma unroll
      for (int k = 0; k < 4; ++k) umma_ss_tf32(tm + C_S, dq + (uint64_t)(2 * k), dk + (uint64_t)(2 * k), I_S, k > 0 ? 1u : 0u);
      tc_commit(sready);
    }
    __syncwarp();
  };
  auto mma_o = [&](int it) {                                         // O = P V (A = fp32 P over S, B = the v tile MN-major)
    const int s = it % NST;
    const uint64_t dv = desc_mn32b(su + s * STAGE + T_V, KN * 128);
    if (elect_one()) {
#pragma unroll 4
      for (int k = 0; k < KN / 8; ++k) umma_ts_tf32(tm + C_O, tm + C_S + 8 * k, dv + (uint64_t)((k * 1024) >> 4), I_O, k > 0 ? 1u : 0u);
      tc_commit(oready);
      tc_commit(&empty[s]);
    }
    __syncwarp();
  };
  if (ctl) {
    if (lane == 0) for (int i = 0; i < NST && i < nit; ++i) load(i);
    __syncwarp();
    if (nit > 0) mma_s(0);
  }

  const uint32_t tl = tm + ((uint32_t)(32 * qd) << 16);
  const int z0 = 32 * (qt < qd ? qt : qt + 5);                       // the out-of-band block this warp zeroes (qt < 3)
  const int qrow = q0 + 32 * qd + lane;
  auto drain = [&](int itd, float mm) {                              // O / l (qt 0, 1) and LSE (qt 2) of sample itd
    const int par = itd & 1, a = a_lo + itd;
    const float* rs = red + ((1 * 2 + par) * 16) * 32;
    const float lt = rs[qd * 32 + lane] + rs[(qd + 4) * 32 + lane] + rs[(qd + 8) * 32 + lane] + rs[(qd + 12) * 32 + lane];
    if (qt < 2) {
      uint32_t v[16];
      tmem_ld16(tl + C_O + 16 * qt, v);
      tmem_wait_ld();
      const float il = lt > 0.f ? 1.f / lt : 0.f;
      const uint32_t ob = su + O_O + warp * 2048;                    // [32 rows][64 B], 64-B swizzle
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
    } else if (qt == 2 && LSE != nullptr && qrow < N) {
      LSE[((size_t)a * NH + h) * N + qrow] = lt > 0.f ? mm + log2f(lt) : 1e30f;
    }
  };
  // the score of column j of cell `cell` (cell 4, the second of the triangle pair: its bias from bl left of the diagonal, bx on it)
  auto score = [&](uint32_t raw, int j, int cell) {
    const float b = !tri ? bl[j] : (cell == 0 ? (j >= lane ? bl[j] : -INFINITY) : (j < lane ? bl[j] : (j == lane ? bx : -INFINITY)));
    return fmaf(__uint_as_float(raw), QSCALE * LOG2E, b);
  };
  auto scores = [&](int cell, float (&sv)[32]) {
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t r[16];
      tmem_ld16(tl + C_S + 32 * qd + 32 * cell + 16 * hf, r);
      tmem_wait_ld();
#pragma unroll
      for (int j = 0; j < 16; ++j) sv[16 * hf + j] = score(r[j], 16 * hf + j, cell);
    }
  };
  auto write_p = [&](int cell, const float (&sv)[32], float mm, float& l) {
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t pk[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const float p = ex2f(sv[16 * hf + k] - mm);
        l += p;
        pk[k] = rna_tf32(p);
      }
      tmem_st16(tl + C_S + 32 * qd + 32 * cell + 16 * hf, pk);
    }
  };
  auto cell4 = [&](float mm, float& m, float& l, bool write) {     // cell 4 of the triangle pair from TMEM, 16 columns at a time
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t r[16];
      tmem_ld16(tl + C_S + 32 * qd + 128 + 16 * hf, r);
      tmem_wait_ld();
      if (!write) {
#pragma unroll
        for (int j = 0; j < 16; ++j) m = fmaxf(m, score(r[j], 16 * hf + j, 4));
      } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const float p = ex2f(score(r[j], 16 * hf + j, 4) - mm);
          l += p;
          r[j] = rna_tf32(p);
        }
        tmem_st16(tl + C_S + 32 * qd + 128 + 16 * hf, r);
      }
    }
  };
  float mm_prev = 0.f;
  for (int it = 0; it < nit; ++it) {
    const int par = it & 1;
    mbar_wait(sready, it & 1);
    tc_fence_after();
    float sv[32];
    scores(cell_a, sv);
    float m = sv[0], l = 0.f;
#pragma unroll
    for (int j = 1; j < 32; ++j) m = fmaxf(m, sv[j]);
    if (tri) cell4(0.f, m, l, false);                                // cell 4: its maximum now, its values again after the exchange
    float* rmax = red + ((0 * 2 + par) * 16) * 32;
    float* rsum = red + ((1 * 2 + par) * 16) * 32;
    rmax[warp * 32 + lane] = m;
    named_bar_sync(1 + qd, 128);
    m = fmaxf(fmaxf(rmax[qd * 32 + lane], rmax[(qd + 4) * 32 + lane]), fmaxf(rmax[(qd + 8) * 32 + lane], rmax[(qd + 12) * 32 + lane]));
    const float mm = m == -INFINITY ? 0.f : m;
    write_p(cell_a, sv, mm, l);                                      // P over exactly the cells this warp read
    if (tri) {
      cell4(mm, m, l, true);
    } else {
      tmem_st16z(tl + C_S + z0); tmem_st16z(tl + C_S + z0 + 16);
    }
    rsum[warp * 32 + lane] = l;
    if (it >= 1) {                                                   // the previous sample's O (its MMAs ran before this sample's S)
      mbar_wait(oready, (it - 1) & 1);
      tc_fence_after();
      drain(it - 1, mm_prev);
    }
    mm_prev = mm;
    tmem_wait_st();
    tc_fence_before(); __syncwarp();
    if (lane == 0) mbar_arrive(tready);
    if (ctl) {
      mbar_wait(tready, it & 1);                                     // P(it) written by all, O(it - 1) drained
      tc_fence_after();
      mma_o(it);
      if (it + 1 < nit) mma_s(it + 1);
      if (it + NST < nit) {
        mbar_wait(&empty[it % NST], (it / NST) & 1);                 // PV(it) done: its stage is free
        if (lane == 0) load(it + NST);
        __syncwarp();
      }
    }
  }
  if (nit > 0) {
    mbar_wait(oready, (nit - 1) & 1);
    tc_fence_after();
    drain(nit - 1, mm_prev);
  }
  if (lane == 0) tma_store_wait0();
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) tmem_dealloc(tm, 512);
}
