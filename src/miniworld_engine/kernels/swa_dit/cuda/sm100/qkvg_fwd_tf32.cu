// qkvg_fwd_tf32.cu — the SWA atom block's first forward stage for the fp32 path on sm_100a, TF32 tensor cores (tcgen05.mma kind::tf32,
// fp32 accumulation). Same equations as the Triton _swa_qkvg_fwd_fp32_kernel, every elementwise step in fp32:
//   x = RMS(q) (1 + scale_a) + shift_a;  p_{q,k,v,g} = x W_{q,k,v,g}^T;  Q = rope(headRMS(p_q)), K = rope(headRMS(p_k)), V = p_v, G = p_g
//   Q / K / V written head-major [N, H, S, D] fp32, rounded to TF32 (they are the window attention's MMA operands: the attention kernel
//   then reads exactly what was saved); G row-major [M, C] fp32 (unrounded, it only feeds the sigmoid gate); with save, x (the TF32
//   MMA operand, rounded), p_q and p_k (the raw fp32 accumulators, pre head RMS) row-major for the backward.
// Rounding: x by cvt.rna in the kernel, W = [Wq; Wk; Wv; Wg] by the host (tf32_fwd._round_tf32, cached per weight version) -- the MMA
// sees round-to-nearest TF32 operands instead of truncating them.
//
// Why not the bf16 design (qkvg_fwd: the four 128 x 128 weights resident in shared memory, 128 KB; qkvg_fwd2: resident in TMEM as the
// A operand): in fp32 the weights are 256 KB, more than shared memory, and as the TMEM A operand they would fill all 512 columns. So x
// is the A operand (in TMEM: a TF32 A operand takes one column per element) and the weights stream from L2 (256 KB, resident there)
// through a ring of 8-KB [64 output rows][32 inputs] slots as the B operand, each slot feeding four M128 N64 K8 MMAs.
//
// Items: (tile, projection group). A tile is SP = min(A, 16) augments x AT = 128 / SP atoms of one batch element (SP AT <= 128 rows,
// the bf16 kernels' tiling: the AT modulation rows are shared by the SP augments and stay in L1), loaded by one 4-D TMA box per
// 32-channel k-block. NG = 1 / 2 / 4 projection groups split an item's four projections (4 / NG each) for small problems (the host
// picks NG so that ntile NG fills the GPU: A = 1, S = 1024 has 8 tiles); every item computes its tile's x (the whole row is the K of
// every projection) and only group 0 writes the x save.
// Warps: 0 TMA producer of the q k-blocks (lane 0); 1 TMEM allocator + MMA issuer (whole warp waits, elect_one() issues); 3 TMA
// producer of the weight ring (lane 0); 4-11 row threads, two warpgroups: thread (warpgroup wg, warp w % 4, lane) = tile row
// r = 32 (w % 4) + lane = TMEM lane r. Both warpgroups take the row's RMS over all 128 channels; warpgroup wg writes x channels
// 64 wg .. 64 wg + 63 and runs the epilogue of heads 2 wg, 2 wg + 1 (the per-head RMS and the RoPE pairs (d, d + 16) in one thread's
// registers). The modulation (shift_a / scale_a) and cos / sin are read straight from global memory (L1-resident rows).
//   shared memory  q ring NA = 8 k-blocks of [128 rows][32 fp32] SW128 (16 KB each; two tiles: tile T + 1 loads under tile T) = 128 KB
//                  | weight ring NW = 12 slots of [64 rows][32 fp32] SW128 (8 KB) = 96 KB | barriers 512 B   -> 229888 B (1 CTA / SM)
//   TMEM (512)     x[b] at 128 b (b = item parity; 128 fp32 columns), acc[b] at 256 + 128 b (one projection, 128 fp32 columns):
//                  item T + 1's x is written while item T's projections run, projection p + 1 accumulates while p's epilogue reads
//   registers      x: one 16-channel unit at a time (q, shift, scale as float4s); epilogue: one head's 32 accumulators + 4 cos / sin
//                  float4s at a time -- designed for <= 128 / thread (launch bound 384 threads x 1 CTA leaves the compiler up to 168)
// Outputs leave by plain 16-B global stores, 128 contiguous bytes per (row, head) and thread (as qkvg_fwd.cu).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

// ------------------------------------------------------------------ kind::tf32 (local: sm100.cuh is the bf16 kernels' header)
__host__ __device__ constexpr uint32_t idesc_tf32(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
// D[tmem] (+)= A[tmem] B[smem], TF32 operands (A one 32-bit element per TMEM column)
DEVI void umma_ts_tf32(uint32_t d_tmem, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p; }"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
DEVI uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI float4 u2f4(uint4 u) { return make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
DEVI float4 ldg4(const float* p) { return u2f4(ldg128(p)); }
DEVI void stg32w(float* p, const uint32_t (&v)[32]) {                       // 128 contiguous bytes
#pragma unroll
  for (int k = 0; k < 8; ++k) stg128(p + 4 * k, make_uint4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]));
}

constexpr int C = 128, H = 4, D = 32, MODW = 6 * C;
constexpr int KB = 128 * 128;                                              // activation k-block: [128 rows][32 fp32], SW128 (16 KB)
constexpr int WB = 64 * 128;                                               // weight slot: [64 output rows][32 fp32], SW128 (8 KB)
constexpr int NA = 8, NW = 12;                                             // q ring (two tiles), weight ring
constexpr int O_A = 0, O_W = NA * KB, O_BAR = O_W + NW * WB;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert((O_W % 1024) == 0 && (O_BAR % 1024) == 0, "1 KB alignment of the 128-B-swizzled tiles");
constexpr uint32_t T_X = 0, T_ACC = 256;
constexpr uint32_t I_P = idesc_tf32(128, 64);

struct Bars {
  uint64_t afull[NA], aempty[NA], wfull[NW], wempty[NW], xfull[2], xfree[2], accfull[2], accfree[2];
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

extern "C" __global__ void __launch_bounds__(384, 1)
swa_qkvg_fwd_tf32_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mw,
                        int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, int NG, float eps, float qk_eps, int save,
                        const float* __restrict__ MOD, const float* __restrict__ COS, const float* __restrict__ SIN,
                        float* __restrict__ Qo, float* __restrict__ Ko, float* __restrict__ Vo, float* __restrict__ Go,
                        float* __restrict__ Xs, float* __restrict__ PQs, float* __restrict__ PKs) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nitem = ntile * NG, PG = 4 / NG;                               // items, projections per item
  const int ntT = (int)blockIdx.x < nitem ? (nitem - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto item = [&](int T, int& b, int& a0, int& s0, int& grp) {             // this CTA's item T -> (batch element, augment, atom, group)
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    grp = t % NG;
    const int tile = t / NG, ab = tile % nab, r = tile / nab, ag = r % nag;
    b = r / nag; a0 = ag * SP; s0 = ab * AT;
  };

  if (tid == 0) {
    for (int i = 0; i < NA; ++i) { mbar_init(&B.afull[i], 1); mbar_init(&B.aempty[i], 8); }      // 8 = every row warp
    for (int i = 0; i < NW; ++i) { mbar_init(&B.wfull[i], 1); mbar_init(&B.wempty[i], 1); }
    for (int i = 0; i < 2; ++i) {
      mbar_init(&B.xfull[i], 8); mbar_init(&B.xfree[i], 1); mbar_init(&B.accfull[i], 1); mbar_init(&B.accfree[i], 8);
    }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();                                                            // PDL: the next kernel may launch now

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: q k-blocks
    if (lane == 0) {
      pdl_wait();                                                          // q comes from the previous kernel
      const uint32_t abytes = (uint32_t)(SP * AT * 128);                   // a full box (out-of-range atoms / augments zero-filled)
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0, grp; item(T, b, a0, s0, grp);
        for (int kb = 0; kb < 4; ++kb) {
          const int i = 4 * T + kb, slot = i % NA;
          if (i >= NA) mbar_wait(&B.aempty[slot], ((i / NA) - 1) & 1);
          mbar_expect_tx(&B.afull[slot], abytes);
          tma_load_4d(su + O_A + slot * KB, &mq, &B.afull[slot], 32 * kb, s0, b, a0);
        }
      }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ TMA producer: weight ring
    // per item, per projection p of its group: half hf (output rows 128 p + 64 hf ..), k-blocks 0-3 -- the MMA warp's order
    if (lane == 0) {
      int j = 0;
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0, grp; item(T, b, a0, s0, grp);
        for (int pp = 0; pp < PG; ++pp)
          for (int hf = 0; hf < 2; ++hf)
            for (int kb = 0; kb < 4; ++kb, ++j) {
              const int slot = j % NW;
              if (j >= NW) mbar_wait(&B.wempty[slot], ((j / NW) - 1) & 1);
              mbar_expect_tx(&B.wfull[slot], WB);
              tma_load_2d(su + O_W + slot * WB, &mw, &B.wfull[slot], 32 * kb, 128 * (grp * PG + pp) + 64 * hf);
            }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    // projection p of item T: acc[ab] (ab = parity of the item's running projection count) = x[T & 1] W_p^T, as two N = 64 halves
    int j = 0;
    for (int T = 0; T < ntT; ++T) {
      const int xb = T & 1;
      for (int pp = 0; pp < PG; ++pp) {
        const int g2 = T * PG + pp, ab = g2 & 1;
        if (pp == 0) mbar_wait(&B.xfull[xb], (T >> 1) & 1);
        if (g2 >= 2) mbar_wait(&B.accfree[ab], ((g2 >> 1) - 1) & 1);      // the epilogue has read projection g2 - 2 out of acc[ab]
        for (int hf = 0; hf < 2; ++hf)
          for (int kb = 0; kb < 4; ++kb, ++j) {
            const int slot = j % NW;
            mbar_wait(&B.wfull[slot], (j / NW) & 1);
            tc_fence_after();
            if (elect_one()) {
              const uint64_t dw = desc_k128(su + O_W + slot * WB);
#pragma unroll
              for (int ks = 0; ks < 4; ++ks)
                umma_ts_tf32(tmem + T_ACC + 128 * ab + 64 * hf, tmem + T_X + 128 * xb + (4 * kb + ks) * 8, dw + (uint64_t)(2 * ks), I_P,
                             (kb | ks) ? 1u : 0u);
              tc_commit(&B.wempty[slot]);
              if (hf == 1 && kb == 3) {
                tc_commit(&B.accfull[ab]);
                if (pp == PG - 1) tc_commit(&B.xfree[xb]);                 // every projection of item T has read x[xb]
              }
            }
            __syncwarp();
          }
      }
    }
  } else if (warp >= 4) {
    pdl_wait();                                                            // the modulation / RoPE tables come from earlier kernels
    // ------------------------------------------------------------------------------------------------ row threads
    const int wg = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int sp = (int)r / AT, at = (int)r % AT;
    const bool rowok = (int)r < SP * AT;
    auto slot_of = [&](int T, int kb) { return su + O_A + ((4 * T + kb) % NA) * KB; };
    auto pro = [&](int T) {                                                // x of item T -> TMEM x[T & 1] (and the x save)
      int b, a0, s0, grp; item(T, b, a0, s0, grp);
      const int xb = T & 1;
      const int n = (a0 + sp) * Bn + b, s = s0 + at;
      const bool ok = rowok && a0 + sp < A && s < S;
      const float* mp = MOD + (size_t)(b * S + min(s, S - 1)) * MODW;      // (a row past S: a clamped modulation row, nothing stored)
      for (int kb = 0; kb < 4; ++kb) mbar_wait(&B.afull[(4 * T + kb) % NA], ((4 * T + kb) / NA) & 1);
      float ss0 = 0.f, ss1 = 0.f;
#pragma unroll 4
      for (int k = 0; k < 32; ++k) {                                       // pass 1: the row's sum of squares (all 128 channels)
        const float4 v = u2f4(lds128(slot_of(T, k >> 3) + sw128(r, k & 7)));
        ss0 = fmaf(v.x, v.x, fmaf(v.y, v.y, ss0));
        ss1 = fmaf(v.z, v.z, fmaf(v.w, v.w, ss1));
      }
      const float rstd = rsqrtf((ss0 + ss1) * (1.f / C) + eps);
      if (T >= 2) mbar_wait(&B.xfree[xb], ((T >> 1) - 1) & 1);            // the MMAs of item T - 2 have read x[xb]
      tc_fence_after();
      float* xsave = Xs + ((size_t)n * S + s) * C;
      const bool sv = save && ok && grp == 0;
#pragma unroll 1
      for (int u = 0; u < 4; ++u) {                                        // pass 2: this warpgroup's 64 channels, 16 per unit
        const int c0 = 64 * wg + 16 * u, kb = c0 >> 5, q0 = (c0 & 31) >> 2;
        uint32_t xr[16];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float4 v = u2f4(lds128(slot_of(T, kb) + sw128(r, q0 + e)));
          const float4 sh = ldg4(mp + c0 + 4 * e), sc = ldg4(mp + C + c0 + 4 * e);
          xr[4 * e + 0] = tf32r(fmaf(v.x * rstd, 1.f + sc.x, sh.x));
          xr[4 * e + 1] = tf32r(fmaf(v.y * rstd, 1.f + sc.y, sh.y));
          xr[4 * e + 2] = tf32r(fmaf(v.z * rstd, 1.f + sc.z, sh.z));
          xr[4 * e + 3] = tf32r(fmaf(v.w * rstd, 1.f + sc.w, sh.w));
        }
        tmem_st16(trow + T_X + 128 * xb + c0, xr);
        if (sv) {
#pragma unroll
          for (int k = 0; k < 4; ++k) stg128(xsave + c0 + 4 * k, make_uint4(xr[4 * k], xr[4 * k + 1], xr[4 * k + 2], xr[4 * k + 3]));
        }
      }
      __syncwarp();
      if (lane == 0)
        for (int kb = 0; kb < 4; ++kb) mbar_arrive(&B.aempty[(4 * T + kb) % NA]);    // both passes have read the q k-blocks
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.xfull[xb]);
    };
    auto epi = [&](int T) {
      int b, a0, s0, grp; item(T, b, a0, s0, grp);
      const int n = (a0 + sp) * Bn + b, s = s0 + at;
      const bool ok = rowok && a0 + sp < A && s < S;
      const size_t mrow = (size_t)b * S + min(s, S - 1);
      const float* csp = COS + mrow * (D / 2);
      const float* snp = SIN + mrow * (D / 2);
#pragma unroll 1
      for (int pp = 0; pp < PG; ++pp) {
        const int p = grp * PG + pp, g2 = T * PG + pp, ab = g2 & 1;
        mbar_wait(&B.accfull[ab], (g2 >> 1) & 1);
        tc_fence_after();
#pragma unroll 1
        for (int hh = 0; hh < 2; ++hh) {                                   // this warpgroup's two heads (32 columns each)
          const int h = 2 * wg + hh;
          uint32_t v[32];
          tmem_ld32(trow + T_ACC + 128 * ab + 32 * h, v);
          tmem_wait_ld();
          if (hh == 1) {                                                   // acc[ab] read out: projection g2 + 2 may accumulate
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(&B.accfree[ab]);
          }
          if (!ok) continue;
          if (p == 3) {                                                    // G: raw fp32, row-major
            stg32w(Go + ((size_t)n * S + s) * C + 32 * h, v);
            continue;
          }
          float* hm = (p == 0 ? Qo : p == 1 ? Ko : Vo) + (((size_t)n * H + h) * S + s) * D;
          if (p == 2) {                                                    // V: head-major, rounded to TF32 (the PV operand)
#pragma unroll
            for (int d = 0; d < 32; ++d) v[d] = tf32r(__uint_as_float(v[d]));
            stg32w(hm, v);
            continue;
          }
          if (save) stg32w((p == 0 ? PQs : PKs) + ((size_t)n * S + s) * C + 32 * h, v);   // raw p_q / p_k for the backward
          float ss0 = 0.f, ss1 = 0.f;
#pragma unroll
          for (int d = 0; d < 32; d += 2) {
            ss0 = fmaf(__uint_as_float(v[d]), __uint_as_float(v[d]), ss0);
            ss1 = fmaf(__uint_as_float(v[d + 1]), __uint_as_float(v[d + 1]), ss1);
          }
          const float rr = rsqrtf((ss0 + ss1) * (1.f / D) + qk_eps);
#pragma unroll
          for (int k = 0; k < 4; ++k) {                                    // RoPE on the pairs (d, d + 16), d = 4 k .. 4 k + 3
            const float4 c4 = ldg4(csp + 4 * k), s4 = ldg4(snp + 4 * k);
            const float cv[4] = {c4.x, c4.y, c4.z, c4.w}, sv[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int d = 4 * k + e;
              const float y1 = __uint_as_float(v[d]) * rr, y2 = __uint_as_float(v[d + 16]) * rr;
              v[d] = tf32r(y1 * cv[e] - y2 * sv[e]);
              v[d + 16] = tf32r(y2 * cv[e] + y1 * sv[e]);
            }
          }
          stg32w(hm, v);
        }
      }
    };
    if (ntT > 0) pro(0);
    for (int T = 0; T < ntT; ++T) {
      if (T + 1 < ntT) pro(T + 1);                                         // item T + 1's x is written under item T's projections
      epi(T);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
