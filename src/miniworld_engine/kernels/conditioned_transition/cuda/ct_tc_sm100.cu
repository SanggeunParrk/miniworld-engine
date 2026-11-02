// ct_tc_sm100.cu -- the fp32 (TF32) ConditionedTransition training GEMMs with their elementwise passes in the epilogue, B200 (sm_100a):
// one persistent, warp-specialised tcgen05 kernel (kind::tf32), built per epilogue (-DMODE):
//
//   MODE 0  SWAB   [a | b] = xn [Wa; Wb]^T, h = silu(a) b              -> h [M, ND], ab [M, 2 ND] (a at column c, b at hoff + c)
//                  (the expand GEMM + swiglu_fwd: the forward's [a | b] is written once and never read back for the SwiGLU)
//   MODE 1  GAB    dh = dout Ws (B = Ws^T [ND, D], K-major), a, b read from the forward's ab -> dab = [da | db] [M, 2 ND]
//                  (the dh GEMM + swiglu_bwd: dh never reaches HBM)
//   MODE 3  SW     MODE 0 without the [a | b] stores (inference: h only)
// (A squeeze GEMM with the gate in its epilogue was built and measured too: slower than cuBLAS + sigmul_fwd at every token row count, e.g.
// [36864, 1536] -> 768: 176 against 173 us, [3840, ...]: 35 against 27 us -- the TF32 tile at K = 1536 and few row tiles is intake-bound.)
//
// Operands are fp32 on the TF32 grid (the MMA truncates: the producers round them -- adaln_fwd(rna), the weight packs, sigmul_bwd_x),
// K-major (row-major activations, nn.Linear weights as stored), 128-B swizzled K-blocks of 32 elements. Any M (TMA zero-fills the tail
// rows of the last tile, the epilogue skips them), N a multiple of the tile width BN, K a multiple of 32.
//
// Roles (64 + 32 NEPI threads, one CTA per SM): warp 0 TMA producer, warp 1 MMA issuer (and TMEM owner), warps 2 .. the epilogue -- warp w
// reads TMEM lanes 32 (w % 4) .. + 31 (rows of the tile); the NEPI / 4 warps of a lane quarter split the 32-column units of the tile.
// Accumulators are double-buffered in tensor memory (2 x NACC x BN columns), so the epilogue of tile i overlaps the main loop of tile
// i + 1. The epilogue moves each warp's 32 x 32 blocks through a 4-KB swizzled slab of its own and stores them with coalesced 16-B stores
// (four whole 128-B row segments a warp instruction) -- no bulk-store staging, so a three-output epilogue never waits on a TMA store
// draining its buffer (the generic Transition GEMM's TMA-store epilogue, single-buffered at three fp32 outputs,
// bounded SWIGLU_AB at 392 us against cuBLAS + swiglu_fwd 325 us, [36864, 768] x 1536).
//
// -DCTA2=1: CTA pairs (cta_group::2): 256-row tiles, each CTA bringing its 128 A rows and half of the B rows (halving the operand intake
// per output, the bound of fp32 tiles), the leader issuing the M = 256 MMAs.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef MODE
#define MODE 0
#endif
#ifndef BN
#define BN 128
#endif
#ifndef CTA2
#define CTA2 0
#endif
#ifndef NEPI
#define NEPI 8
#endif
constexpr int NCTA = CTA2 ? 2 : 1;
constexpr int BNL = BN / NCTA;                       // B rows (output columns) each CTA of a pair brings
constexpr bool SWG = MODE == 0 || MODE == 3;         // the SwiGLU forward epilogues
constexpr bool PF2 = MODE == 1;                      // the epilogue reading two [M, *] operands, cp.async-prefetched one unit ahead
constexpr int NACC = MODE == 1 ? 1 : 2;              // accumulators per tile
constexpr int NPH = 1;                               // operand phases per tile (the producer / MMA loops keep the general form)
constexpr int NBS = SWG ? 2 : 1;                     // B tiles per ring stage
__host__ __device__ constexpr int ph_acc0(int p) { return p; }
__host__ __device__ constexpr int ph_nacc(int p) { return SWG ? 2 : 1; }

constexpr int BKE = 32;                              // fp32 elements per 128-byte K-block row
constexpr int A_BYTES = 128 * 128, B_BYTES = BNL * 128, STAGE = A_BYTES + NBS * B_BYTES;
constexpr int SLAB = 32 * 128;                       // one warp's 32 rows x 32 fp32 transposing slab
constexpr int NSLAB = PF2 ? 4 : 1;                   // GAB: a and b through two slabs, double-buffered (the next unit's prefetched)
constexpr int STG_BYTES = NEPI * NSLAB * SLAB;
constexpr int BUDGET = 232448 - 1024;
constexpr int NST_FIT = (BUDGET - STG_BYTES) / STAGE;
constexpr int NST = NST_FIT > 8 ? 8 : NST_FIT;
constexpr int O_STG = NST * STAGE, O_BAR = O_STG + STG_BYTES;
constexpr int SMEM_BYTES = O_BAR + 1024;
constexpr int TCOLS = NACC * BN;
constexpr int QW = NEPI / 4;                         // epilogue warps per lane quarter
static_assert(NEPI % 4 == 0 && NEPI >= 4 && NEPI <= 16 && BN / 32 >= NEPI / 4, "epilogue warps: every warp owns a column unit");
static_assert(NST >= 2, "ring depth");
static_assert(2 * TCOLS <= 512, "two accumulator buffers must fit tensor memory");
static_assert(BN % 32 == 0 && BN >= 32 && BN <= 256 && (BNL % 8) == 0, "tile width");
static_assert(STAGE % 1024 == 0, "128-B swizzle atoms");

// kind::tf32 instruction descriptor: TF32 A / B (format 2), fp32 D, K-major both, N, M
constexpr uint32_t IDESC = (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)((128 * NCTA) >> 4) << 24);
DEVI void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
#if CTA2
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC), "r"(acc) : "memory");
#else
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC), "r"(acc) : "memory");
#endif
}
DEVI void commit_all(uint64_t* bar) {
#if CTA2
  tc_commit2_mc(bar, (uint16_t)3);
#else
  tc_commit(bar);
#endif
}
DEVI void load_op(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1) {
#if CTA2
  tma_load_2d_2sm(dst, m, bar, c0, c1);
#else
  tma_load_2d(dst, m, bar, c0, c1);
#endif
}

DEVI float sgm(float x) { return 1.f / (1.f + __expf(-x)); }   // as ct_rows.cu's row passes
DEVI float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
// Moving a warp's 32 rows x 32 fp32 block between registers (thread = row, its 32 columns) and global memory through a 4-KB slab, 128-B
// swizzled: every global access of the warp is four whole 128-B row segments (lanes 8 i .. 8 i + 7: row 4 j + i, 16-B chunk lane % 8),
// where a thread's own row would be 32 different lines per instruction.
DEVI void slab_put(uint32_t slab, int lane, const float (&v)[32]) {
#pragma unroll
  for (int q = 0; q < 8; ++q)
    sts128(slab + sw128((uint32_t)lane, (uint32_t)q), make_uint4(__float_as_uint(v[4 * q]), __float_as_uint(v[4 * q + 1]), __float_as_uint(v[4 * q + 2]),
                                                                  __float_as_uint(v[4 * q + 3])));
}
DEVI void slab_get(uint32_t slab, int lane, float (&v)[32]) {
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const uint4 w = lds128(slab + sw128((uint32_t)lane, (uint32_t)q));
    v[4 * q] = __uint_as_float(w.x); v[4 * q + 1] = __uint_as_float(w.y); v[4 * q + 2] = __uint_as_float(w.z); v[4 * q + 3] = __uint_as_float(w.w);
  }
}
// slab -> global rows row0 .. row0 + 31 (those < M), columns c .. c + 31 of a row-major [*, ld] fp32 matrix
DEVI void slab_store(uint32_t slab, int lane, float* g, long ld, int row0, int c, int M) {
  __syncwarp();
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int rr = 4 * j + (lane >> 3), q = lane & 7;
    const uint4 w = lds128(slab + sw128((uint32_t)rr, (uint32_t)q));
    if (row0 + rr < M) stg128(g + (long)(row0 + rr) * ld + c + 4 * q, w);
  }
  __syncwarp();
}
// global -> slab without the registers (cp.async, rows past M zero-filled): the next unit's operands load while this one is computed
DEVI void slab_prefetch(uint32_t slab, int lane, const float* g, long ld, int row0, int c, int M) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int rr = 4 * j + (lane >> 3), q = lane & 7;
    const bool ok = row0 + rr < M;
    const float* src = g + (ok ? (long)(row0 + rr) * ld + c + 4 * q : 0);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(slab + sw128((uint32_t)rr, (uint32_t)q)), "l"(src), "r"(ok ? 16 : 0) : "memory");
  }
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
DEVI void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }

struct Bars {
  uint64_t full[NST], empty[NST], accf[2], acce[2];
  uint32_t tmem;
};

// maps: A0 / A1 [M][K_p] (box 32 x 128), B0 / B1 [N][K_p] (box 32 x BNL). Unused maps may repeat a used one.
extern "C" __global__ void __launch_bounds__(64 + 32 * NEPI, 1)
ctc_sm100(const __grid_constant__ CUtensorMap mA0, const __grid_constant__ CUtensorMap mA1, const __grid_constant__ CUtensorMap mB0,
          const __grid_constant__ CUtensorMap mB1, float* __restrict__ o0, float* __restrict__ o1, float* __restrict__ o2,
          const float* __restrict__ aux, const float* __restrict__ bias, int M, int N, int K0, int K1, int ld0, int ld1, int ldaux,
          int hoff) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NT = N / BN, MNT = ((M + 128 * NCTA - 1) / (128 * NCTA)) * NT, tiles = MNT;
  const uint32_t rank = CTA2 ? cluster_rank() : 0u;
  const bool leader = rank == 0;

  if (threadIdx.x == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.accf[s], 1); mbar_init(&B.acce[s], NEPI * NCTA); }
    fence_barrier_init();
    prefetch_map(&mA0); prefetch_map(&mB0);
    if (NPH > 1) prefetch_map(&mA1);
    if (NPH > 1 || NACC > 1) prefetch_map(&mB1);
  }
#if CTA2
  if (warp == 1) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
#else
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
#endif
  tc_fence_before();
  __syncthreads();
  if (CTA2) cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ producer
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x / NCTA; t < tiles; t += gridDim.x / NCTA) {
        const int m0 = (t / NT) * 128 * NCTA + (int)rank * 128, n0 = (t % NT) * BN + (int)rank * BNL;
#pragma unroll
        for (int p = 0; p < NPH; ++p) {
          const CUtensorMap* ma = p == 0 ? &mA0 : &mA1;
          const int nk = (p == 0 ? K0 : K1) / BKE, a0 = ph_acc0(p), na = ph_nacc(p);
          for (int k = 0; k < nk; ++k, ++it) {
            const int s = it % NST;
            mbar_wait(&B.empty[s], ((it / NST) & 1) ^ 1);
            if (leader) mbar_expect_tx(&B.full[s], (A_BYTES + na * B_BYTES) * NCTA);
            const uint32_t base = su + s * STAGE;
            load_op(base, ma, &B.full[s], k * BKE, m0);
            for (int j = 0; j < na; ++j) load_op(base + A_BYTES + j * B_BYTES, (a0 + j) == 0 ? &mB0 : &mB1, &B.full[s], k * BKE, n0);
          }
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    if (lane == 0 && leader) {
      int it = 0, tc = 0;
      for (int t = blockIdx.x / NCTA; t < tiles; t += gridDim.x / NCTA, ++tc) {
        const int buf = tc & 1;
#if CTA2
        mbar_wait_cl(&B.acce[buf], ((tc >> 1) & 1) ^ 1);
#else
        mbar_wait(&B.acce[buf], ((tc >> 1) & 1) ^ 1);
#endif
        tc_fence_after();
        const uint32_t dbase = tmem + buf * TCOLS;
#pragma unroll
        for (int p = 0; p < NPH; ++p) {
          const int nk = (p == 0 ? K0 : K1) / BKE, a0 = ph_acc0(p), na = ph_nacc(p);
          for (int k = 0; k < nk; ++k, ++it) {
            const int s = it % NST;
            mbar_wait(&B.full[s], (it / NST) & 1);
            tc_fence_after();
            const uint32_t base = su + s * STAGE;
            const uint64_t da = desc_k128(base);
            for (int j = 0; j < na; ++j) {
              const uint64_t db = desc_k128(base + A_BYTES + j * B_BYTES);
#pragma unroll
              for (int kk = 0; kk < 4; ++kk)
                umma(dbase + (a0 + j) * BN, da + (uint64_t)(kk * 2), db + (uint64_t)(kk * 2), (k > 0 || kk > 0) ? 1u : 0u);
            }
            commit_all(&B.empty[s]);
          }
        }
        commit_all(&B.accf[buf]);
      }
    }
  } else {
    // ------------------------------------------------------------------------------------------------ epilogue
    const int ew = warp - 2, q4 = warp & 3, part = ew >> 2;   // lane quarter, column share
    const uint32_t slab = su + O_STG + (uint32_t)(ew * NSLAB) * SLAB;
    int tc = 0;
#if MODE == 1
    // the saved a, b of each unit arrive by cp.async one unit ahead, into the slab pair the previous unit's stores have left
    int pi = 0;
    auto rows_of = [&](int tt) { return (tt / NT) * 128 * NCTA + (int)rank * 128 + q4 * 32; };
    {
      const int t0 = blockIdx.x / NCTA;
      if (t0 < tiles) {
        const int c0 = (t0 % NT) * BN + part * 32;
        slab_prefetch(slab, lane, aux, ldaux, rows_of(t0), c0, M);
        slab_prefetch(slab + SLAB, lane, aux, ldaux, rows_of(t0), hoff + c0, M);
      }
      cp_commit();
    }
#endif
    for (int t = blockIdx.x / NCTA; t < tiles; t += gridDim.x / NCTA, ++tc) {
      const int buf = tc & 1;
      const int m0 = (t / NT) * 128 * NCTA + (int)rank * 128, n0 = (t % NT) * BN;
      const int row0 = m0 + q4 * 32;
      mbar_wait(&B.accf[buf], (tc >> 1) & 1);
      tc_fence_after();
      const uint32_t tb = tmem + ((uint32_t)(q4 * 32) << 16) + buf * TCOLS;
      for (int u = part; u < BN / 32; u += QW) {
        const int col = u * 32, c = n0 + col;
        __syncwarp();
        uint32_t acc[NACC][32];
#pragma unroll
        for (int j = 0; j < NACC; ++j) tmem_ld32(tb + j * BN + col, acc[j]);
#if MODE == 1
        const uint32_t sa = slab + (uint32_t)(pi * 2) * SLAB, sb = sa + SLAB;
        {
          int tn = t, un = u + QW;
          if (un >= BN / 32) { tn = t + (int)(gridDim.x / NCTA); un = part; }
          if (tn < tiles) {
            const uint32_t na = slab + (uint32_t)((pi ^ 1) * 2) * SLAB;
            const int cn = (tn % NT) * BN + un * 32;
            slab_prefetch(na, lane, aux, ldaux, rows_of(tn), cn, M);
            slab_prefetch(na + SLAB, lane, aux, ldaux, rows_of(tn), hoff + cn, M);
          }
          cp_commit();
          cp_wait1();                                         // this unit's group has landed (the one just issued may be in flight)
          __syncwarp();
        }
#endif
        tmem_wait_ld();
        if (u + QW >= BN / 32) {   // the last TMEM read of this tile by this warp: hand the buffer back before the stores
          tc_fence_before();
          __syncwarp();
#if CTA2
          if (lane == 0) { if (leader) mbar_arrive(&B.acce[buf]); else mbar_arrive_remote_relaxed(&B.acce[buf], 0); }
#else
          if (lane == 0) mbar_arrive(&B.acce[buf]);
#endif
        }
#if MODE == 0 || MODE == 3
        {
          float v[32];
#pragma unroll
          for (int e = 0; e < 32; ++e) { const float a = __uint_as_float(acc[0][e]); v[e] = tf32r(a * sgm(a) * __uint_as_float(acc[1][e])); }   // h: TF32 operand
          slab_put(slab, lane, v);
          slab_store(slab, lane, o0, ld0, row0, c, M);
#if MODE == 0
#pragma unroll
          for (int e = 0; e < 32; ++e) v[e] = __uint_as_float(acc[0][e]);
          slab_put(slab, lane, v);
          slab_store(slab, lane, o1, ld1, row0, c, M);
#pragma unroll
          for (int e = 0; e < 32; ++e) v[e] = __uint_as_float(acc[1][e]);
          slab_put(slab, lane, v);
          slab_store(slab, lane, o1, ld1, row0, hoff + c, M);
#endif
        }
#elif MODE == 1
        {
          float av[32], bv[32];
          slab_get(sa, lane, av);
          slab_get(sb, lane, bv);
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            const float g = __uint_as_float(acc[0][e]), a = av[e], b = bv[e];
            const float s = sgm(a);
            av[e] = g * b * s * (1.f + a * (1.f - s));
            bv[e] = g * a * s;
          }
          __syncwarp();
          slab_put(sa, lane, av);
          slab_store(sa, lane, o0, ld0, row0, c, M);
          slab_put(sb, lane, bv);
          slab_store(sb, lane, o0, ld0, row0, hoff + c, M);
          pi ^= 1;
        }
#endif
      }
    }
  }
  tc_fence_before();
  __syncthreads();
#if CTA2
  cluster_sync();
  if (warp == 1) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
#else
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
#endif
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
