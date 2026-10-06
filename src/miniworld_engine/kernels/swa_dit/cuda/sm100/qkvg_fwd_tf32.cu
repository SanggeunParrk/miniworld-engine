// qkvg_fwd_tf32.cu — the SWA atom block's first forward stage for the fp32 path on sm_100a, TF32 tensor cores (tcgen05.mma kind::tf32,
// fp32 accumulation). Same equations as the Triton _swa_qkvg_fwd_fp32_kernel, every elementwise step in fp32:
//   x = RMS(q) (1 + scale_a) + shift_a;  p_{q,k,v,g} = x W_{q,k,v,g}^T;  Q = rope(headRMS(p_q)), K = rope(headRMS(p_k)), V = p_v, G = p_g
//   Q / K / V written head-major [N, H, S, D] fp32, rounded to TF32 (they are the window attention's MMA operands: the attention kernel
//   then reads exactly what was saved); G row-major [M, C] fp32 (unrounded, it only feeds the sigmoid gate); with save, x (the TF32
//   MMA operand, rounded), p_q and p_k (the raw fp32 accumulators, pre head RMS) row-major for the backward.
// Rounding: x by cvt.rna in the kernel, W = [Wq; Wk; Wv; Wg] by wprep_tf32.cu (cached per weight version) -- the MMA
// sees round-to-nearest TF32 operands instead of truncating them.
//
// Why not the bf16 design (qkvg_fwd: the four 128 x 128 weights resident in shared memory, 128 KB; qkvg_fwd2: resident in TMEM as the
// A operand): in fp32 the weights are 256 KB, more than shared memory, and as the TMEM A operand they would fill all 512 columns. So x
// is the A operand (in TMEM: a TF32 A operand takes one column per element) and the weights stream from L2 (256 KB, resident there)
// through a ring of 8-KB [64 output rows][32 inputs] slots as the B operand, each slot feeding four M128 N64 K8 MMAs.
//
// Items: (tile, projection group). A tile is SP = min(A, 16) augments x AT = 128 / SP atoms of one batch element (SP AT <= 128 rows,
// the bf16 kernels' tiling: the AT modulation / RoPE rows are shared by the SP augments), loaded by one 4-D TMA box per 32-channel
// k-block. NG = 1 / 2 / 4 projection groups split an item's four projections (4 / NG each) for small problems (the host
// picks NG so that ntile NG fills the GPU: A = 1, S = 1024 has 8 tiles); every item computes its tile's x (the whole row is the K of
// every projection) and only group 0 writes the x save.
// Warps: 0 TMA producer of the activation ring (lane 0); 1 TMEM allocator + MMA issuer (whole warp waits, elect_one() issues); 2 TMA
// producer of the cos / sin tile (lane 0); 3 TMA producer of the weight ring (lane 0); 4-11 row threads, two warpgroups: thread
// (warpgroup wg, warp w % 4, lane) = tile row r = 32 (w % 4) + lane = TMEM lane r. Both warpgroups take the row's RMS over all 128
// channels; warpgroup wg writes x channels 64 wg .. 64 wg + 63 and runs the epilogue of heads 2 wg, 2 wg + 1 (the per-head RMS and
// the RoPE pairs (d, d + 16) in one thread's registers).
// Every per-row table arrives by TMA (round 3: the modulation and cos / sin used to be 16-B global loads, one row per lane -- 32
// sectors per warp load through an L1 of ~24 KB -- and were the row threads' main stall in Nsight Compute):
//   * the ring carries, per item, q0 q1 q2 q3 (4-D boxes [32 channels, AT atoms, 1, SP augments]) then sh0 sc0 sh1 sc1 sh2 sc2 sh3 sc3
//     (2-D boxes [32 channels, AT rows] of mod [B S, 6C], columns 32 kb of shift_a / 128 + 32 kb of scale_a; row at serves every
//     augment of atom at). Pass 1 (the RMS) parks this warpgroup's raw q channels in its TMEM x buffer and releases the q k-blocks at
//     once, so the 8-slot ring can bring the 8 modulation k-blocks in behind them; pass 2 reads q back from TMEM, applies the
//     modulation and overwrites it with x;
//   * cos / sin of the tile: one [AT rows][16 fp32] box each (64-B swizzle), single-buffered, loaded by warp 2 while the previous
//     item's epilogue runs (it is free once that epilogue has applied its last RoPE).
//   shared memory  ring NA = 8 k-blocks of [128 rows][32 fp32] SW128 (16 KB each) = 128 KB | weight ring NW = 10 slots of [64 rows]
//                  [32 fp32] SW128 (8 KB) = 80 KB | cos / sin [2][128 rows][16 fp32] SW64 = 16 KB | barriers 512 B -> 229888 B
//   TMEM (512)     x[b] at 128 b (b = item parity; 128 fp32 columns), acc[b] at 256 + 128 b (one projection, 128 fp32 columns):
//                  item T + 1's x is written while item T's projections run, projection p + 1 accumulates while p's epilogue reads
//   registers      x: one 16-channel unit at a time (q from TMEM, shift, scale as float4s); epilogue: one head's 32 accumulators + 4
//                  cos / sin float4s at a time -- designed for <= 128 / thread (launch bound 384 threads x 1 CTA leaves the compiler up to 168)
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

constexpr int C = 128, H = 4, D = 32;
constexpr int KB = 128 * 128;                                              // activation k-block: [128 rows][32 fp32], SW128 (16 KB)
constexpr int WB = 64 * 128;                                               // weight slot: [64 output rows][32 fp32], SW128 (8 KB)
constexpr int NA = 8, NW = 10;                                             // activation ring, weight ring
constexpr int NQK = 12;                                                    // ring k-blocks per item: q0-3, then sh / sc per k-block
constexpr int O_A = 0, O_W = NA * KB, O_CS = O_W + NW * WB, O_BAR = O_CS + 2 * 128 * 64;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert((O_W % 1024) == 0 && (O_CS % 1024) == 0 && (O_BAR % 1024) == 0, "1 KB alignment of the swizzled tiles");
// cos / sin: row at, 16-B chunk k (of 4) in the 64-B swizzle is sm100.cuh's sw64 (chunk k ^ ((at >> 1) & 3)): conflict-free over 8 rows
constexpr uint32_t T_X = 0, T_ACC = 256;
constexpr uint32_t I_P = idesc_tf32(128, 64);

struct Bars {
  uint64_t afull[NA], aempty[NA], wfull[NW], wempty[NW], xfull[2], xfree[2], accfull[2], accfree[2], csfull, csempty;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

extern "C" __global__ void __launch_bounds__(384, 1)
swa_qkvg_fwd_tf32_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mmod,
                        const __grid_constant__ CUtensorMap mcos, const __grid_constant__ CUtensorMap msin,
                        int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, int NG, float eps, float qk_eps, int save,
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
    mbar_init(&B.csfull, 1); mbar_init(&B.csempty, 8);
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();                                                            // PDL: the next kernel may launch now

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: the activation ring
    if (lane == 0) {
      pdl_wait();                                                          // q / mod come from the previous kernels
      const uint32_t abytes = (uint32_t)(SP * AT * 128), mbytes = (uint32_t)(AT * 128);   // full boxes (out of range: zero-filled)
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0, grp; item(T, b, a0, s0, grp);
        for (int k = 0; k < NQK; ++k) {
          const int i = NQK * T + k, slot = i % NA;
          if (i >= NA) mbar_wait(&B.aempty[slot], ((i / NA) - 1) & 1);
          const uint32_t dst = su + O_A + slot * KB;
          if (k < 4) {
            mbar_expect_tx(&B.afull[slot], abytes);
            tma_load_4d(dst, &mq, &B.afull[slot], 32 * k, s0, b, a0);
          } else {                                                         // k = 4 + 2 kb + (0: shift_a | 1: scale_a)
            mbar_expect_tx(&B.afull[slot], mbytes);
            tma_load_2d(dst, &mmod, &B.afull[slot], C * ((k - 4) & 1) + 32 * ((k - 4) >> 1), b * S + s0);
          }
        }
      }
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA producer: cos / sin
    if (lane == 0) {
      pdl_wait();
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0, grp; item(T, b, a0, s0, grp);
        if (T >= 1) mbar_wait(&B.csempty, (T - 1) & 1);                   // item T - 1's epilogue has applied its RoPE
        mbar_expect_tx(&B.csfull, (uint32_t)(2 * AT * 64));
        tma_load_2d(su + O_CS, &mcos, &B.csfull, 0, b * S + s0);
        tma_load_2d(su + O_CS + 128 * 64, &msin, &B.csfull, 0, b * S + s0);
      }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ TMA producer: weight ring
    // per item, per projection p of its group: half hf (output rows 128 p + 64 hf ..), k-blocks 0-3 -- the MMA warp's order
    if (lane == 0) {
      pdl_wait();                                                          // W: the weight-form kernel (wprep_tf32) runs in the chain
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
    pdl_wait();                                                            // output stores (the tables now arrive by TMA)
    // ------------------------------------------------------------------------------------------------ row threads
    const int wg = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int sp = (int)r / AT, at = (int)r % AT;
    const bool rowok = (int)r < SP * AT;
    auto slot_of = [&](int T, int k) { return su + O_A + ((NQK * T + k) % NA) * KB; };   // ring position k of item T
    auto pro = [&](int T) {                                                // x of item T -> TMEM x[T & 1] (and the x save)
      int b, a0, s0, grp; item(T, b, a0, s0, grp);
      const int xb = T & 1;
      const int n = (a0 + sp) * Bn + b, s = s0 + at;
      const bool ok = rowok && a0 + sp < A && s < S;
      const uint32_t tx = trow + T_X + 128 * xb;
      for (int kb = 0; kb < 4; ++kb) mbar_wait(&B.afull[(NQK * T + kb) % NA], ((NQK * T + kb) / NA) & 1);
      if (T >= 2) mbar_wait(&B.xfree[xb], ((T >> 1) - 1) & 1);            // the MMAs of item T - 2 have read x[xb]
      tc_fence_after();
      // pass 1: the row's sum of squares over all 128 channels, 16 per step; this warpgroup's 64 raw q channels parked in x[xb]
      float ss0 = 0.f, ss1 = 0.f;
#pragma unroll 1
      for (int g = 0; g < 8; ++g) {
        uint32_t qr[16];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint4 u = lds128(slot_of(T, g >> 1) + sw128(r, 4 * (g & 1) + e));
          const float4 v = u2f4(u);
          ss0 = fmaf(v.x, v.x, fmaf(v.y, v.y, ss0));
          ss1 = fmaf(v.z, v.z, fmaf(v.w, v.w, ss1));
          qr[4 * e + 0] = u.x; qr[4 * e + 1] = u.y; qr[4 * e + 2] = u.z; qr[4 * e + 3] = u.w;
        }
        if ((g >> 2) == wg) tmem_st16(tx + 16 * g, qr);
      }
      __syncwarp();
      if (lane == 0)
        for (int kb = 0; kb < 4; ++kb) mbar_arrive(&B.aempty[(NQK * T + kb) % NA]);   // q read: the modulation k-blocks come in
      const float rstd = rsqrtf((ss0 + ss1) * (1.f / C) + eps);
      tmem_wait_st();
      float* xsave = Xs + ((size_t)n * S + s) * C;
      const bool sv = save && ok && grp == 0;
      for (int k = 0; k < 2; ++k) {                                        // this warpgroup's shift / scale k-blocks (2 wg + k)
        const int i = NQK * T + 4 + 2 * (2 * wg + k);
        mbar_wait(&B.afull[i % NA], (i / NA) & 1);
        mbar_wait(&B.afull[(i + 1) % NA], ((i + 1) / NA) & 1);
      }
#pragma unroll 1
      for (int u = 0; u < 4; ++u) {                                        // pass 2: this warpgroup's 64 channels, 16 per unit
        const int c0 = 64 * wg + 16 * u, kb = c0 >> 5, q0 = (c0 & 31) >> 2;
        const uint32_t ash = slot_of(T, 4 + 2 * kb), asc = slot_of(T, 5 + 2 * kb);
        uint32_t xr[16];
        tmem_ld16(tx + c0, xr);
        float4 sh[4], sc[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          sh[e] = u2f4(lds128(ash + sw128((uint32_t)at, q0 + e)));
          sc[e] = u2f4(lds128(asc + sw128((uint32_t)at, q0 + e)));
        }
        tmem_wait_ld();
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          xr[4 * e + 0] = tf32r(fmaf(__uint_as_float(xr[4 * e + 0]) * rstd, 1.f + sc[e].x, sh[e].x));
          xr[4 * e + 1] = tf32r(fmaf(__uint_as_float(xr[4 * e + 1]) * rstd, 1.f + sc[e].y, sh[e].y));
          xr[4 * e + 2] = tf32r(fmaf(__uint_as_float(xr[4 * e + 2]) * rstd, 1.f + sc[e].z, sh[e].z));
          xr[4 * e + 3] = tf32r(fmaf(__uint_as_float(xr[4 * e + 3]) * rstd, 1.f + sc[e].w, sh[e].w));
        }
        tmem_st16(tx + c0, xr);
        if (sv) {
#pragma unroll
          for (int k = 0; k < 4; ++k) stg128(xsave + c0 + 4 * k, make_uint4(xr[4 * k], xr[4 * k + 1], xr[4 * k + 2], xr[4 * k + 3]));
        }
      }
      // every warp releases all eight modulation k-blocks (the slots' count is 8), after waiting for the ones it did not read: a full
      // slot was loaded, so its previous use was released by all 8 warps and this arrival lands in the right phase
      for (int k = 4; k < NQK; ++k) {
        const int i = NQK * T + k;
        mbar_wait(&B.afull[i % NA], (i / NA) & 1);
      }
      __syncwarp();
      if (lane == 0)
        for (int k = 4; k < NQK; ++k) mbar_arrive(&B.aempty[(NQK * T + k) % NA]);
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.xfull[xb]);
    };
    auto epi = [&](int T) {
      int b, a0, s0, grp; item(T, b, a0, s0, grp);
      const int n = (a0 + sp) * Bn + b, s = s0 + at;
      const bool ok = rowok && a0 + sp < A && s < S;
      mbar_wait(&B.csfull, T & 1);                                         // cos / sin of the tile (row at: this thread's atom)
      const uint32_t csp = su + O_CS, snp = su + O_CS + 128 * 64;
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
            const float4 c4 = u2f4(lds128(csp + sw64((uint32_t)at, (uint32_t)k))), s4 = u2f4(lds128(snp + sw64((uint32_t)at, (uint32_t)k)));
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
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.csempty);                              // cos / sin of item T applied: item T + 1's may load
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
