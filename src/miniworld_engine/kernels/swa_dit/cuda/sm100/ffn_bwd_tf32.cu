// ffn_bwd_tf32.cu — the fp32 SWA atom block's FFN backward on sm_100a, TF32 tensor cores, in one kernel (the fp32 counterpart of
// ffn_bwd_gate.cu + ffn_bwd_dy.cu; the outputs of the Triton _swa_ffn_bwd_fp32_kernel, every one fp32):
//   dffn = dq2 gate_f;  a | b = y Wu^T (recomputed);  dh = dffn Wd;  sa = sigmoid(a);  h = a sa b;
//   da = dh b sa (1 + a (1 - sa));  db = dh a sa;  dy = [da | db] Wu;  xh = q1 rstd;  dxh = dy (1 + scale_f);
//   dq1 = dq2 + rstd (dxh - xh mean(dxh xh));  d shift_f += sum_aug dy,  d scale_f += sum_aug dy xh,  d gate_f += sum_aug dq2 ffn
// -> DFFN [M, 128], HH [M, 256], DAB = [da | db] [M, 512] (the cuBLAS dWd / dWu operands), dq1 [M, 128]; dq2 is the block's output
// gradient. DFFN / HH / DAB leave rounded to TF32 (cvt.rna): they are this kernel's MMA operands and the TF32 weight gradients' (which
// round them the same way); y is rounded on its way into TMEM; the weights arrive rounded (bwd_prep_tf32.cu). dq1 is plain fp32.
//
// Why this shape. In fp32 the weights no longer fit on chip (Wu 256 KB, Wd^T 128 KB, Wu^T 256 KB; the bf16 kernels kept Wu | Wd^T or
// Wu^T resident in TMEM as A operands), so they stream from L2 through a ring of 16-KB slots, and a tile is 128 rows (M = 128) to amortise
// them: 640 KB of weights per 128 rows. Two layouts, each where it pays:
//   * row layout (lane = row) for the gate: a, b, dh [128 rows][32 hidden] per 32-hidden slice j = 0..7 (M = 128, N = 32, K = 128):
//       a = y Wu_a[j]^T, b = y Wu_b[j]^T   TS: y in TMEM (A, 128 columns),  B = Wu rows 32 j .. as loaded ([32 hid][128 ch], K-major)
//       dh = dffn Wd[:, j]                  SS: dffn tile in smem (A),       B = Wd^T rows 32 j ..
//     so the gate is one row per thread and h / da / db go to row-major staging slices [128 rows][32 hid] (a thread writes its own
//     128-B row, 8 x 16 B: four wavefronts a warp, the minimum), which are both the TMA-store sources (HH, DAB) and
//   * the transposed dy^T [128 ch][128 rows] += Wu^T[:, j] da_j^T + Wu^T[:, 256 + j] db_j^T (SS, M = N = 128, K = 32): A = the Wu^T slice
//     [128 ch][32 hid] from the ring, B = the staging slices as stored (K-major). Lane = channel, so the RMS-adaLN backward runs one channel
//     per thread with coalesced 512-B row loads of q1 / dq2 (as ffn_bwd_dy.cu), the augment sums of the modulation gradient in registers.
// Warp roles: 0 TMA producer (y slices, then the weight slices in MMA order), 1 MMA issuer, 2 TMA stores (HH, DAB) + TMEM allocation,
// 4-7 warpgroup 0: y -> TMEM (row per thread), dffn (channel per thread: coalesced dq2 / ffn loads, DFFN stores, d gate_f), the gate
// (row per thread); 8-11 warpgroup 1: the RMS-adaLN backward of the previous tile (channel per thread), overlapping warpgroup 0.
// MMA order per tile: abdh(0), abdh(1), dy(0), abdh(2), dy(1), ..., abdh(7), dy(6), dy(7); the ring holds the uses in exactly that order:
// Y0..Y3 (warpgroup 0 consumes them), then {Wu_a, Wu_b, Wd^T}(j) and {Wu^T_a, Wu^T_b}(j).
// Shared memory: dffn 64 KB | ring 4 x 16 KB | staging 2 x (da | db | h) 96 KB | exchange 640 B | barriers  (225 KB).
// TMEM: y 0-127 | a / b / dh [2] at 128 + 96 buf (+0 / +32 / +64) | dy^T 320-447  (512 allocated).
// Registers: <= 128 / thread (launch bound 512, launched with 384 threads).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include <type_traits>
using namespace s100;

// ------------------------------------------------------------------ kind::tf32 (as augmented_attention/cuda/sm100/sm100.cuh, measured on B200)
namespace tf {
__host__ __device__ constexpr uint32_t idesc(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEVI void mma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI void mma_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p; }"
               :: "r"(d), "r"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI uint32_t rna(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r & 0xffffe000u; }
DEVI float rnaf(float x) { return __uint_as_float(rna(x)); }
}  // namespace tf

constexpr int C = 128, NH = 256, MODW = 6 * C, NW = 4, UPT = 44;           // ring slots; ring uses per tile (4 Y + 8 x 3 W + 8 x 2 W^T)
constexpr int SL = 128 * 128;                                              // a [128][32] fp32 slice (16 KB)
constexpr int STG = 3 * SL;                                                // staging: da | db | h
constexpr int O_DF = 0, O_RING = 4 * SL, O_ST = O_RING + NW * SL, O_EX = O_ST + 2 * STG, O_BAR = O_EX + 640, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_Y = 0, T_ACC = 128, T_DY = 320;
constexpr uint32_t I_AB = tf::idesc(128, 32), I_DY = tf::idesc(128, 128);

struct Bars {
  uint64_t full[NW], empty[NW], inready, infree, accfull[2], accfree[2], stfull[2], stfree[2], dyfull, dyfree;
  uint32_t tmem;
};
DEVI void red1(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }
DEVI float pick8(const float (&a)[8], int k) {                           // a[k] for a runtime k without indexing registers
  float v = a[0];
#pragma unroll
  for (int e = 1; e < 8; ++e) v = k == e ? a[e] : v;
  return v;
}
DEVI float ldf(const float* p) { return __ldg(p); }
DEVI float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
DEVI float rscatter32(float (&v)[32], int lane) {                          // v[j] in every lane -> lane l: sum over the warp of v[l]
#pragma unroll
  for (int i = 0; i < 16; ++i) { const bool up = lane & 16; const float k = up ? v[i + 16] : v[i], s = up ? v[i] : v[i + 16]; v[i] = k + __shfl_xor_sync(~0u, s, 16); }
#pragma unroll
  for (int i = 0; i < 8; ++i) { const bool up = lane & 8; const float k = up ? v[i + 8] : v[i], s = up ? v[i] : v[i + 8]; v[i] = k + __shfl_xor_sync(~0u, s, 8); }
#pragma unroll
  for (int i = 0; i < 4; ++i) { const bool up = lane & 4; const float k = up ? v[i + 4] : v[i], s = up ? v[i] : v[i + 4]; v[i] = k + __shfl_xor_sync(~0u, s, 4); }
#pragma unroll
  for (int i = 0; i < 2; ++i) { const bool up = lane & 2; const float k = up ? v[i + 2] : v[i], s = up ? v[i] : v[i + 2]; v[i] = k + __shfl_xor_sync(~0u, s, 2); }
  { const bool up = lane & 1; const float k = up ? v[1] : v[0], s = up ? v[0] : v[1]; v[0] = k + __shfl_xor_sync(~0u, s, 1); }
  return v[0];
}

// ------------------------------------------------------------------ the RMS-adaLN backward of one tile (shared with qkvg_bwd_tf32.cu)
// One channel c per thread of a warpgroup (warp qw = channels 32 qw .., its TMEM lanes); the tile rows are the TMEM columns of acc^T:
//   rstd = 1 / sqrt(mean_c xin^2 + eps);  xh = xin rstd;  dxh = acc (1 + scale);  out = dres + rstd (dxh - xh mean_c(dxh xh))
//   dmod[mrow][dcol + c] += sum_aug acc,  dmod[mrow][dcol + C + c] += sum_aug acc xh
// 16 rows at a time. Pass 1: xin^2 and xin (1 + scale) acc over the warp's 32 channels by a reduce-scatter (lane 2 j + e: quantity e of
// row j), the four warps' partials meet in `xch`, warp 0 forms (rstd, rstd mean) per row in `rs`; pass 2 writes out (coalesced rows).
// Augment sums in registers for AT = 8 (rows 8 sp + at), per-row red.add otherwise.
struct RmsArgs {
  const float* XIN; const float* DRES; float* OUT; const float* MOD; float* DMOD;
  int scol, dcol, b, a0, s0, S, A, Bn, SP, AT; float eps;
};
template <bool AT8>
DEVI void rms_bwd(uint32_t tacc, const RmsArgs& p, float* xch, float2* rs, int bar, int qw, int lane) {
  const int c = qw * 32 + lane, R = p.SP * p.AT;
  constexpr int NA = AT8 ? 8 : 1;                                          // register arrays the generic path does not need
  constexpr int NS = AT8 ? 1 : 16;
  float sc8[NA], ash[NA], asc[NA];
#pragma unroll
  for (int k = 0; k < NA; ++k) {
    sc8[k] = (AT8 && p.s0 + k < p.S) ? 1.f + ldf(p.MOD + ((size_t)p.b * p.S + p.s0 + k) * MODW + p.scol + c) : 1.f;
    ash[k] = 0.f; asc[k] = 0.f;
  }
#pragma unroll 1
  for (int r0 = 0; r0 < R; r0 += 16) {
    uint32_t dv[16];
    tmem_ld16(tacc + (uint32_t)r0, dv);
    float xv[16], svg[NS];
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
      const int j = r0 + jj, sp = AT8 ? j >> 3 : j / p.AT, at = AT8 ? j & 7 : j - sp * p.AT;
      const bool ok = j < R && p.a0 + sp < p.A && p.s0 + at < p.S;
      xv[jj] = ok ? ldf(p.XIN + ((((size_t)(p.a0 + sp) * p.Bn + p.b) * p.S + p.s0 + at) * C + c)) : 0.f;
      if constexpr (!AT8) svg[jj] = ok ? 1.f + ldf(p.MOD + ((size_t)p.b * p.S + p.s0 + at) * MODW + p.scol + c) : 1.f;
    }
    auto scale = [&](int jj) -> float {                                    // 1 + scale of row r0 + jj (r0 is a multiple of 16)
      if constexpr (AT8) return sc8[jj & 7]; else return svg[jj];
    };
    tmem_wait_ld();
    {
      float v[32];
#pragma unroll
      for (int jj = 0; jj < 16; ++jj) { v[2 * jj] = xv[jj] * xv[jj]; v[2 * jj + 1] = xv[jj] * scale(jj) * __uint_as_float(dv[jj]); }
      xch[qw * 32 + lane] = rscatter32(v, lane);
    }
    named_bar_sync(bar, 128);
    if (qw == 0) {
      const float tot = (xch[lane] + xch[32 + lane]) + (xch[64 + lane] + xch[96 + lane]);
      const float oth = __shfl_xor_sync(~0u, tot, 1);
      if (!(lane & 1)) {
        const float rstd = 1.f / sqrtf(tot * (1.f / C) + p.eps);
        rs[lane >> 1] = make_float2(rstd, rstd * oth * (1.f / C));
      }
    }
    named_bar_sync(bar, 128);
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
      const int j = r0 + jj, sp = AT8 ? j >> 3 : j / p.AT, at = AT8 ? j & 7 : j - sp * p.AT;
      const bool ok = j < R && p.a0 + sp < p.A && p.s0 + at < p.S;
      const float2 rr = rs[jj];
      const float dy = __uint_as_float(dv[jj]), xh = xv[jj] * rr.x, dxh = dy * scale(jj);
      if (ok) {
        const size_t gr = (((size_t)(p.a0 + sp) * p.Bn + p.b) * p.S + p.s0 + at) * C + c;
        p.OUT[gr] = ldf(p.DRES + gr) + rr.x * (dxh - xh * rr.y);
      }
      if constexpr (AT8) {
        ash[jj & 7] += ok ? dy : 0.f; asc[jj & 7] += ok ? dy * xh : 0.f;
      } else if (ok) {
        float* dm = p.DMOD + ((size_t)p.b * p.S + p.s0 + at) * MODW + p.dcol + c;
        red1(dm, dy); red1(dm + C, dy * xh);
      }
    }
  }
  if constexpr (AT8) {
    const int rot = (int)blockIdx.x & 7;                                   // CTAs start their flushes at different atoms
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int k = (kk + rot) & 7;
      if (p.s0 + k < p.S) {
        float* dm = p.DMOD + ((size_t)p.b * p.S + p.s0 + k) * MODW + p.dcol + c;
        red1(dm, pick8(ash, k)); red1(dm + C, pick8(asc, k));
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(512, 1)
swa_ffn_bwd_tf32_sm100(const __grid_constant__ CUtensorMap my, const __grid_constant__ CUtensorMap mwu,
                       const __grid_constant__ CUtensorMap mwdt, const __grid_constant__ CUtensorMap mwut,
                       const __grid_constant__ CUtensorMap mhh, const __grid_constant__ CUtensorMap mdab,
                       const float* __restrict__ DQ2, const float* __restrict__ Q1, const float* __restrict__ FFN,
                       const float* __restrict__ MOD, float* __restrict__ DFFN, float* __restrict__ DQ1, float* __restrict__ DMOD,
                       int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int R = SP * AT;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto coords = [&](int T, int& b, int& a0, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    const int ab = t % nab, r = t / nab, ag = r % nag;
    b = r / nag; a0 = ag * SP; s0 = ab * AT;
  };

  if (tid == 0) {
    for (int i = 0; i < NW; ++i) { mbar_init(&B.full[i], 1); mbar_init(&B.empty[i], 1); }
    mbar_init(&B.inready, 1); mbar_init(&B.infree, 1); mbar_init(&B.dyfull, 1); mbar_init(&B.dyfree, 4);
    for (int i = 0; i < 2; ++i) { mbar_init(&B.accfull[i], 1); mbar_init(&B.accfree[i], 4); mbar_init(&B.stfull[i], 1); mbar_init(&B.stfree[i], 2); }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: the ring, in MMA order
    if (lane == 0) {
      int g = 0;
      auto slot = [&](uint32_t bytes) {
        const int s = g % NW;
        if (g >= NW) mbar_wait(&B.empty[s], ((g / NW) - 1) & 1);
        mbar_expect_tx(&B.full[s], bytes);
        ++g;
        return s;
      };
      auto load_w = [&](int j) {                                           // Wu_a, Wu_b, Wd^T rows of hidden slice j: 4 boxes [32][32] each
        for (int p = 0; p < 3; ++p) {
          const int s = slot(SL);
          const CUtensorMap* m = p < 2 ? &mwu : &mwdt;
          const int row = p == 1 ? NH + 32 * j : 32 * j;
          for (int kb = 0; kb < 4; ++kb) tma_load_2d(su + O_RING + s * SL + kb * 4096, m, &B.full[s], 32 * kb, row);
        }
      };
      auto load_wt = [&](int j) {                                          // Wu^T columns 32 j .. and 256 + 32 j ..: one [128][32] box each
        for (int p = 0; p < 2; ++p) {
          const int s = slot(SL);
          tma_load_2d(su + O_RING + s * SL, &mwut, &B.full[s], p * NH + 32 * j, 0);
        }
      };
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        for (int k = 0; k < 4; ++k) {                                      // y channels 32 k ..: [R rows][32]
          const int s = slot((uint32_t)(R * 128));
          tma_load_4d(su + O_RING + s * SL, &my, &B.full[s], 32 * k, s0, b, a0);
        }
        load_w(0);
        for (int i = 0; i < 8; ++i) {
          if (i + 1 < 8) load_w(i + 1);
          load_wt(i);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    int g = 0;
    auto abdh = [&](int T, int j) {                                        // a, b, dh of hidden slice j -> accumulator buffer (8 T + j) & 1
      const int gi = 8 * T + j, bb = gi & 1;
      if (gi >= 2) mbar_wait(&B.accfree[bb], ((gi >> 1) - 1) & 1);         // the gate has read slice gi - 2
      const uint32_t d = tmem + T_ACC + 96 * bb;
      for (int p = 0; p < 3; ++p, ++g) {
        const int s = g % NW;
        mbar_wait(&B.full[s], (g / NW) & 1);
        tc_fence_after();
        const uint32_t ws = su + O_RING + s * SL;
        if (elect_one()) {
#pragma unroll
          for (int k = 0; k < 16; ++k) {                                   // K = 128 channels: 4 boxes x 4 steps of 8
            const uint64_t bd = desc_k128(ws + (k >> 2) * 4096) + (uint64_t)((k & 3) * 2);
            if (p < 2) tf::mma_ts(d + 32 * p, tmem + T_Y + 8 * k, bd, I_AB, k > 0 ? 1u : 0u);
            else tf::mma_ss(d + 64, desc_k128(su + O_DF + (k >> 2) * SL) + (uint64_t)((k & 3) * 2), bd, I_AB, k > 0 ? 1u : 0u);
          }
          tc_commit(&B.empty[s]);
          if (p == 2) {
            tc_commit(&B.accfull[bb]);
            if (j == 7) tc_commit(&B.infree);                              // the tile's last reads of y (TMEM) and dffn (smem)
          }
        }
        __syncwarp();
      }
    };
    auto dy = [&](int T, int i) {                                          // dy^T += Wu^T_a[i] da_i^T + Wu^T_b[i] db_i^T
      const int si = 8 * T + i, bb = si & 1;
      mbar_wait(&B.stfull[bb], (si >> 1) & 1);
      if (i == 0 && T >= 1) mbar_wait(&B.dyfree, (T - 1) & 1);             // warpgroup 1 has read dy^T of tile T - 1
      const uint32_t sg = su + O_ST + bb * STG;
      for (int p = 0; p < 2; ++p, ++g) {
        const int s = g % NW;
        mbar_wait(&B.full[s], (g / NW) & 1);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int k = 0; k < 4; ++k)
            tf::mma_ss(tmem + T_DY, desc_k128(su + O_RING + s * SL) + (uint64_t)(k * 2), desc_k128(sg + p * SL) + (uint64_t)(k * 2), I_DY,
                       (i > 0 || p > 0 || k > 0) ? 1u : 0u);
          tc_commit(&B.empty[s]);
          if (p == 1) {
            tc_commit(&B.stfree[bb]);
            if (i == 7) tc_commit(&B.dyfull);
          }
        }
        __syncwarp();
      }
    };
    for (int T = 0; T < ntT; ++T) {
      g += 4;                                                              // the Y uses: warpgroup 0's
      mbar_wait(&B.inready, T & 1);
      tc_fence_after();
      abdh(T, 0);
      for (int i = 0; i < 8; ++i) {
        if (i + 1 < 8) abdh(T, i + 1);
        dy(T, i);
      }
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA stores: DAB (da, db), HH
    if (lane == 0) {
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        for (int i = 0; i < 8; ++i) {
          const int si = 8 * T + i, bb = si & 1;
          mbar_wait(&B.stfull[bb], (si >> 1) & 1);
          const uint32_t sg = su + O_ST + bb * STG;
          tma_store_4d(&mdab, sg, 32 * i, s0, b, a0);
          tma_store_4d(&mdab, sg + SL, NH + 32 * i, s0, b, a0);
          tma_store_4d(&mhh, sg + 2 * SL, 32 * i, s0, b, a0);
          tma_store_commit();
          tma_store_wait_read0();
          mbar_arrive(&B.stfree[bb]);
        }
      }
      tma_store_wait0();
    }
  } else if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------------------------------------ warpgroup 0: y, dffn, the gate
    const int lq = warp & 3, r = lq * 32 + lane;                           // row r (TMEM lane) / channel r in the dffn pass
    const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16);
    const uint32_t cofs = (uint32_t)(r >> 5) * SL + (uint32_t)(r & 3) * 4, cq = (uint32_t)((r & 31) >> 2);
    auto dffn_pass = [&](int b, int a0, int s0, auto at8c) {
      // dffn = dq2 gate_f, channel r per thread, the rows in turn: into the A tile (rounded) and to DFFN; d gate_f = sum_aug dq2 ffn
      constexpr bool AT8 = decltype(at8c)::value;
      const int c = r;
      const uint32_t tb = su + O_DF + cofs;
      if constexpr (AT8) {
        float gf[8], gs[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) { gf[k] = s0 + k < S ? ldf(MOD + ((size_t)b * S + s0 + k) * MODW + 5 * C + c) : 0.f; gs[k] = 0.f; }
#pragma unroll 1
        for (int sp = 0; sp < 16; ++sp) {
          const bool aok = a0 + sp < A;
          const size_t g0 = (((size_t)(a0 + sp) * Bn + b) * S + s0) * C + c;
          float d2[8], ff[8];
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const bool ok = aok && s0 + k < S;
            d2[k] = ok ? ldf(DQ2 + g0 + (size_t)k * C) : 0.f; ff[k] = ok ? ldf(FFN + g0 + (size_t)k * C) : 0.f;
          }
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const int j = 8 * sp + k;
            const uint32_t v = tf::rna(d2[k] * gf[k]);
            sts32(tb + (uint32_t)j * 128u + ((cq ^ (uint32_t)k) << 4), v);
            if (aok && s0 + k < S) DFFN[g0 + (size_t)k * C] = __uint_as_float(v);
            gs[k] += d2[k] * ff[k];
          }
        }
        const int rot = (int)blockIdx.x & 7;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          const int k = (kk + rot) & 7;
          if (s0 + k < S) red1(DMOD + ((size_t)b * S + s0 + k) * MODW + 5 * C + c, pick8(gs, k));
        }
      } else {
#pragma unroll 4
        for (int j = 0; j < R; ++j) {
          const int sp = j / AT, at = j - sp * AT;
          const bool ok = a0 + sp < A && s0 + at < S;
          const size_t gr = (((size_t)(a0 + sp) * Bn + b) * S + s0 + at) * C + c;
          float d2 = 0.f, ff = 0.f, gf = 0.f;
          if (ok) { d2 = ldf(DQ2 + gr); ff = ldf(FFN + gr); gf = ldf(MOD + ((size_t)b * S + s0 + at) * MODW + 5 * C + c); }
          const uint32_t v = tf::rna(d2 * gf);
          sts32(tb + (uint32_t)j * 128u + ((cq ^ (uint32_t)(j & 7)) << 4), v);
          if (ok) { DFFN[gr] = __uint_as_float(v); red1(DMOD + ((size_t)b * S + s0 + at) * MODW + 5 * C + c, d2 * ff); }
        }
      }
    };
    for (int T = 0; T < ntT; ++T) {
      int b, a0, s0; coords(T, b, a0, s0);
      if (T >= 1) mbar_wait(&B.infree, (T - 1) & 1);                       // tile T - 1's MMAs are done with y (TMEM) and dffn (smem)
      tc_fence_after();
      // ---- y -> TMEM (A of a / b), rounded: row r, channels 32 k .. from ring use 44 T + k
      for (int k = 0; k < 4; ++k) {
        const int g = UPT * T + k, s = g % NW;
        mbar_wait(&B.full[s], (g / NW) & 1);
        const uint32_t ys = su + O_RING + s * SL;
        uint32_t v[16];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint4 u = lds128(ys + sw128((uint32_t)r, (uint32_t)(4 * h + q)));
            v[4 * q] = tf::rna(__uint_as_float(u.x)); v[4 * q + 1] = tf::rna(__uint_as_float(u.y));
            v[4 * q + 2] = tf::rna(__uint_as_float(u.z)); v[4 * q + 3] = tf::rna(__uint_as_float(u.w));
          }
          tmem_st16(trow + T_Y + 32 * k + 16 * h, v);
        }
        named_bar_sync(1, 128);                                            // every row of the slot has been read
        if (r == 0) mbar_arrive(&B.empty[s]);
      }
      if (AT == 8) dffn_pass(b, a0, s0, std::true_type{}); else dffn_pass(b, a0, s0, std::false_type{});
      tmem_wait_st();
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1, 128);
      if (r == 0) mbar_arrive(&B.inready);
      // ---- the gate, row r: hidden slice i (32 units) per step
#pragma unroll 1
      for (int i = 0; i < 8; ++i) {
        const int gi = 8 * T + i, bb = gi & 1;
        mbar_wait(&B.accfull[bb], (gi >> 1) & 1);
        if (gi >= 2) mbar_wait(&B.stfree[bb], ((gi >> 1) - 1) & 1);        // staging bb: dy MMA and stores of slice gi - 2 are done
        tc_fence_after();
        const uint32_t ta = trow + T_ACC + 96 * bb, sg = su + O_ST + bb * STG;
#pragma unroll
        for (int h = 0; h < 2; ++h) {                                      // hidden 16 h .. 16 h + 15
          uint32_t av[16], bv[16], dv[16];
          tmem_ld16(ta + 16 * h, av); tmem_ld16(ta + 32 + 16 * h, bv); tmem_ld16(ta + 64 + 16 * h, dv);
          tmem_wait_ld();
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            uint32_t oa[4], ob[4], oh[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float a = __uint_as_float(av[4 * q + e]), bq = __uint_as_float(bv[4 * q + e]), dh = __uint_as_float(dv[4 * q + e]);
              const float sa = sigm(a);
              oa[e] = tf::rna(dh * bq * sa * (1.f + a * (1.f - sa)));
              ob[e] = tf::rna(dh * a * sa);
              oh[e] = tf::rna(a * sa * bq);
            }
            const uint32_t o = sw128((uint32_t)r, (uint32_t)(4 * h + q));
            sts128(sg + o, make_uint4(oa[0], oa[1], oa[2], oa[3]));
            sts128(sg + SL + o, make_uint4(ob[0], ob[1], ob[2], ob[3]));
            sts128(sg + 2 * SL + o, make_uint4(oh[0], oh[1], oh[2], oh[3]));
          }
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.accfree[bb]);                        // this warp's a / b / dh reads are done
        fence_proxy_async();
        named_bar_sync(1, 128);
        if (r == 0) mbar_arrive(&B.stfull[bb]);
      }
    }
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------------------ warpgroup 1: RMS-adaLN backward (channel c)
    const int qw = warp & 3;
    float* xch = reinterpret_cast<float*>(sm + O_EX);
    float2* rs = reinterpret_cast<float2*>(sm + O_EX + 512);
    for (int T = 0; T < ntT; ++T) {
      RmsArgs p{Q1, DQ2, DQ1, MOD, DMOD, 4 * C, 3 * C, 0, 0, 0, S, A, Bn, SP, AT, eps};
      coords(T, p.b, p.a0, p.s0);
      mbar_wait(&B.dyfull, T & 1);
      tc_fence_after();
      const uint32_t tacc = tmem + ((uint32_t)(qw * 32) << 16) + T_DY;
      if (AT == 8) rms_bwd<true>(tacc, p, xch, rs, 2, qw, lane); else rms_bwd<false>(tacc, p, xch, rs, 2, qw, lane);
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dyfree);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
