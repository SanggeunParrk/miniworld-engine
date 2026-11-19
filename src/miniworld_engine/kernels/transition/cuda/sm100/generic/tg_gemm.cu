// tg_gemm.cu -- the GEMMs of the generic sm_100a (B200) Transition and of the ConditionedTransition tail: one persistent, warp-specialised
// tcgen05 kernel, C[M, N] = sum over phases of A_p[M, K_p] B_j[N, K_p]^T, with the epilogue chosen at build time (-DMODE):
//
//   MODE 0  SWIGLU   h = silu(A Wa^T) * (A Wb^T)                                  (expand + SwiGLU; two accumulators)
//   MODE 1  RESID    out = A B^T + res                                           (squeeze + residual)
//   MODE 2  PLAIN    out = A B^T, fp32                                           (d_xn of the backward)
//   MODE 3  GATE     dh = dy Ws, a = xn Wa^T, b = xn Wb^T -> h, [dA | dB]         (SwiGLU backward; three accumulators, two phases)
//   MODE 4  CTGATE   y = res + sigmoid(cond Wsc^T + bsc) * (h Ws^T)               (ConditionedTransition squeeze + gate; two phases)
//   MODE 5  SWIGLU_AB  MODE 0 that also stores a, b (into [a | b], columns hoff apart): the forward at a few row tiles
//   MODE 6  GATE_AB  dh = dy Ws, a, b read from the forward's [a | b] (res, row stride ldres, b at column hoff) -> h, [dA | dB]
//                    (the backward at a few row tiles: one operand phase instead of GATE's two, whose recompute of a, b doubles the
//                    per-CTA operand intake that bounds a sub-wave grid)
//   MODE 7  GATE_AB without h: MODE 6 storing only [dA | dB] (the ConditionedTransition training backward, which keeps h from its forward)
//
// -DF32=1: fp32 activations on the TF32 tensor cores (kind::tf32; the operands are expected on the TF32 grid, the outputs that feed a
// later TF32 GEMM are rounded to it), else bf16 (kind::f16). Any M (TMA zero-fills the tail rows of the last 128-row tile and clips
// them on the store), N a multiple of the output group (64 bf16 / 32 fp32 columns), K a multiple of 64 (bf16) / 32 (fp32).
//
// Roles (320 threads, one CTA per SM): warp 0 TMA producer, warp 1 MMA issuer (and TMEM owner), warps 2-9 the epilogue -- warp w reads
// TMEM lanes 32 (w % 4) .. + 31, i.e. rows 32 (w % 4) .. of the tile; the two warps of a lane quarter split the 32-column units (a
// 128-B fp32 group each, or the two halves of a 128-B bf16 group, joined by a named barrier) and TMA-store 32-row slabs from a swizzled
// staging buffer. Accumulators are double-buffered in tensor memory (2 x NACC x BN columns), so the epilogue of tile i overlaps the main loop
// of tile i + 1. Operands stream through an NST-deep ring of [A | B_0 | B_1] K-blocks of 128 bytes per row (64 bf16 / 32 fp32), 128-B
// swizzled, K-major on both sides (nn.Linear weights as stored).
//
// Split K (KS > 1, single-phase epilogues; the host uses it with PLAIN): tile t is (K slice t / (M tiles x N tiles), output tile); slice s
// sums K-blocks [s nk / KS, (s + 1) nk / KS) and stores its fp32 partial at rows s MSTR + m of the output map (slabs of MSTR >= M rows),
// which the consumer adds in slice order (tg_ln_bwd_*, tg_splitsum). At a few row tiles this puts every SM on a GEMM whose output tiles
// alone would leave most of them idle, each walking 1 / KS of the K loop.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef MODE
#define MODE 0
#endif
#ifndef F32
#define F32 0
#endif
#ifndef H16
#define H16 0                                        // round 3 part 4: fp32 inference on the fp16 tensor cores (MODE 0 / 1 only, see below)
#endif
static_assert(!H16 || (!F32 && (MODE == 0 || MODE == 1)), "H16: the SWIGLU / RESID epilogues of the fp16-operand fp32 forward");
// -DH16=1: fp16 A / B (kind::f16, format 0) under fused_generic_sm100a._h16_forms' scales, passed as `bias` = sc[4]: SWIGLU stores
// h = fp16(silu(a) b sc[2]) with a, b = acc sc[1]; RESID reads the fp32 residual and stores fp32 out = res + acc sc[3].
#define OUTF_PP (F32 || MODE == 2 || (H16 && MODE == 1))
#if H16
#define PK2(a, b) f2h2(a, b)
#else
#define PK2(a, b) pack_bf16(a, b)
#endif
#ifndef BN
#define BN 128
#endif
#ifndef TG_TRACE
#define TG_TRACE(i) do { } while (0)                  // (analysis builds) epilogue trace points
#endif
#ifndef GSTORE
#define GSTORE 0                                     // (round 3, training; tg_bwd_small.cu) the epilogue stores rows from registers (st.global to gO0 /
#endif                                               // gO1) instead of staging + TMA stores: no bulk-store completion wait before a grid barrier
#ifndef BMN
#define BMN 0                                        // (round 3, training; tg_bwd_small.cu) the B operands MN-major: [K][N] row-major weights
#endif                                               // (no transposed copies), BNL / BKE 128-B N blocks of BKE K rows per stage
#ifndef CTA2
#define CTA2 0                                       // 1: CTA pairs (cta_group::2): M = 256 products, B split by N across the pair
#endif
constexpr int NCTA = CTA2 ? 2 : 1;
constexpr int BNL = BN / NCTA;                       // B rows (output columns) each CTA of a pair brings

constexpr bool SWG = MODE == 0 || MODE == 5;                             // the SwiGLU forward epilogues
constexpr bool GAT = MODE == 3 || MODE == 6 || MODE == 7;                             // the gate (SwiGLU backward) epilogues: h, dA, dB
constexpr int NACC = SWG ? 2 : MODE == 3 ? 3 : MODE == 4 ? 2 : 1;        // accumulators per tile
constexpr int NPH = (MODE == 3 || MODE == 4) ? 2 : 1;                     // operand phases (A, K) per tile
constexpr int NBS = (SWG || MODE == 3) ? 2 : 1;                           // B tiles per ring stage (the widest phase)
__host__ __device__ constexpr int ph_acc0(int p) { return p == 0 ? 0 : 1; }
__host__ __device__ constexpr int ph_nacc(int p) { return SWG ? 2 : (MODE == 3 && p == 1) ? 2 : 1; }

constexpr int ESZ = F32 ? 4 : 2;                     // activation / operand element bytes
constexpr int BKE = 128 / ESZ;                       // elements per 128-byte K-block row
constexpr bool OUT_F32 = OUTF_PP;                   // PLAIN (and the H16 RESID) write fp32 whatever the operands
constexpr int OCG = OUT_F32 ? 32 : 64;               // output columns per 128-byte staging row
constexpr int UPG = OCG / 32;                        // 32-column TMEM loads per output group
constexpr int NOUT = MODE == 7 ? 2 : (GAT || MODE == 5) ? 3 : 1;   // staged outputs per group (GATE: h, dA, dB; SWIGLU_AB: h, a, b; MODE 7: dA, dB)
constexpr bool LDAB = MODE == 6 || MODE == 7;            // the epilogue TMA-loads the saved a, b into its dA / dB slabs
constexpr int SL_A = MODE == 7 ? 0 : 1, SL_B = SL_A + 1;   // staging slots of dA (a) and dB (b)
// staging slabs: bf16 outputs (UPG 2) are staged by one warp per lane quarter (4 owners), fp32 ones by every epilogue warp (8); so the
// three-output bf16 epilogues (SWIGLU_AB, GATE, GATE_AB) fit double-buffered staging too
constexpr int NOWN = UPG == 1 ? 8 : 4;                 // (8 = NEPI, defined below)
constexpr int STG_PAR = (NOUT == 3 && UPG == 1) ? 1 : 2;   // staging buffers per output
constexpr int A_BYTES = 128 * 128, B_BYTES = BNL * 128, STAGE = A_BYTES + NBS * B_BYTES;
constexpr int SLAB = 32 * 128;                       // one warp's staging slab: 32 rows x 128 B
constexpr int NEPI = 8;                              // epilogue warps: two per TMEM lane quarter, splitting the columns
constexpr int STG_BYTES = NOWN * NOUT * STG_PAR * SLAB;
constexpr int BUDGET = 232448 - 1024;                // dynamic shared memory minus the barriers' kilobyte
constexpr int NST_FIT = (BUDGET - STG_BYTES) / STAGE;
constexpr int NST = NST_FIT > 8 ? 8 : NST_FIT;
constexpr int O_ST = 0, O_STG = NST * STAGE, O_BAR = O_STG + STG_BYTES;
constexpr int SMEM_BYTES = O_BAR + 1024;
constexpr int TCOLS = NACC * BN;                     // TMEM columns of one accumulator buffer
static_assert(NST >= 2, "ring depth");
static_assert(2 * TCOLS <= 512, "two accumulator buffers must fit tensor memory");
static_assert(BN % 32 == 0 && BN % OCG == 0 && BN >= 32 && BN <= 256, "tile width");
static_assert(STAGE % 1024 == 0 && SLAB % 1024 == 0, "128-B swizzle atoms");

#if F32
// kind::tf32 instruction descriptor: TF32 A / B (format 2), fp32 D, K-major both, N, M
constexpr uint32_t IDESC = (1u << 4) | (2u << 7) | (2u << 10) | (BMN ? (1u << 16) : 0u) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)((128 * NCTA) >> 4) << 24);
DEVI void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
#if CTA2
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC), "r"(acc) : "memory");
#else
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC), "r"(acc) : "memory");
#endif
}
#else
constexpr uint32_t IDESC = idesc_bf16(128 * NCTA, BN, 0, BMN) & ~(H16 ? ((1u << 7) | (1u << 10)) : 0u);   // H16: fp16 A / B
DEVI void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
#if CTA2
  umma_ss2(d, a, b, IDESC, acc);
#else
  umma_ss(d, a, b, IDESC, acc);
#endif
}
#endif
// the B operand's descriptor and its advance per MMA K step (16-B units): K-major 32 B; MN-major (BMN) KI K rows of 128 B (KI = 8 tf32,
// 16 bf16), N blocks BKE rows apart, tf32 in the 32-B-atom swizzle (tg_dw.cu's desc_op)
static_assert(!BMN || (!CTA2 && !H16 && BNL % BKE == 0), "BMN: single-CTA bf16 / TF32 builds, whole 128-B N blocks");
constexpr int B_KSTEP = BMN ? (F32 ? 8 : 16) * 128 / 16 : 2;
DEVI uint64_t desc_b(uint32_t saddr) {
#if BMN
  const uint64_t d = desc_mn128(saddr, BKE * 128);
  return F32 ? ((d & ~((uint64_t)7 << 61) & ~((uint64_t)0x3FFF << 32)) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)1 << 61)) : d;
#else
  return desc_k128(saddr);
#endif
}
// the pair's helpers (identity for single CTAs)
DEVI void commit_all(uint64_t* bar) {
#if CTA2
  tc_commit2_mc(bar, (uint16_t)3);
#else
  tc_commit(bar);
#endif
}
DEVI void load_op(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1) {
#if CTA2
  tma_load_2d_2sm(dst, m, bar, c0, c1);          // completes on the leader's barrier
#else
  tma_load_2d(dst, m, bar, c0, c1);
#endif
}

DEVI float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
DEVI float sigm(float a) { return sigmoid_kit(a); }   // rcp.approx(1 + ex2.approx(-a log2 e)), as the tuned kernels
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }

struct Bars {
  uint64_t full[NST], empty[NST], accf[2], acce[2];
  uint64_t ldab[LDAB ? NOWN : 1][2];            // GATE_AB: the saved a, b of a group TMA-loaded into its dA / dB staging slabs
  uint32_t tmem;
};

// 32 output values of one row -> 32 columns of its 128-B staging row (unit u of the group): bf16 64 B (4 chunks), fp32 128 B (8 chunks)
DEVI void stage32(uint32_t slab, uint32_t r, int u, const float (&v)[32]) {
#if OUTF_PP
#pragma unroll
  for (int q = 0; q < 8; ++q)
    sts128(slab + sw128(r, q), make_uint4(__float_as_uint(v[4 * q]), __float_as_uint(v[4 * q + 1]), __float_as_uint(v[4 * q + 2]),
                                          __float_as_uint(v[4 * q + 3])));
#else
#pragma unroll
  for (int q = 0; q < 4; ++q)
    sts128(slab + sw128(r, u * 4 + q), make_uint4(PK2(v[8 * q], v[8 * q + 1]), PK2(v[8 * q + 2], v[8 * q + 3]),
                                                  PK2(v[8 * q + 4], v[8 * q + 5]), PK2(v[8 * q + 6], v[8 * q + 7])));
#endif
}

// the inverse of stage32: the 32 values of a row's unit u from a 128-B swizzled staging row
DEVI void stage_ld32(uint32_t slab, uint32_t r, int u, float (&v)[32]) {
#if F32
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const uint4 w = lds128(slab + sw128(r, q));
    v[4 * q] = __uint_as_float(w.x); v[4 * q + 1] = __uint_as_float(w.y); v[4 * q + 2] = __uint_as_float(w.z); v[4 * q + 3] = __uint_as_float(w.w);
  }
#else
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint4 w = lds128(slab + sw128(r, u * 4 + q));
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) { v[8 * q + 2 * e] = bf16lo(ww[e]); v[8 * q + 2 * e + 1] = bf16hi(ww[e]); }
  }
#endif
}
#if GSTORE
#include "../widths/tsx_store.cuh"                    // (included first by tg_bwd_small.cu: a no-op there)
// 32 output values of each of the warp's 32 rows (lane = row) straight to global memory: off0 = the element offset of the warp's first row
// at this column, ld its row stride (fp32: 128 B per row; bf16 / fp16: 64 B). store_t's butterfly transpose: each store instruction writes
// whole 128-B rows (4 rows x 128 B fp32, 8 x 64 B bf16) instead of 32 rows x 16 B. Whole 32-row groups only (tg_bwd_small: M % 128 = 0).
DEVI void gst32(void* base, long off0, long ld, int lane, const float (&v)[32]) {
#if OUTF_PP
  uint32_t w[32];
#pragma unroll
  for (int e = 0; e < 32; ++e) w[e] = __float_as_uint(v[e]);
  float* p = reinterpret_cast<float*>(base) + off0;
  store_t<32>(p + store_t_off<32>(lane, lane, (size_t)ld, 4), (size_t)ld, w, lane);
#else
  uint32_t w[16];
#pragma unroll
  for (int e = 0; e < 16; ++e) w[e] = PK2(v[2 * e], v[2 * e + 1]);
  uint16_t* p = reinterpret_cast<uint16_t*>(base) + off0;
  store_t<16>(p + store_t_off<16>(lane, lane, (size_t)ld, 2), (size_t)ld, w, lane);
#endif
}
#endif

// 32 activation-dtype values of a row (residual) as fp32
DEVI void load_res32(const void* res, long off, bool ok, float (&v)[32]) {
#if F32 || H16
  const float* p = reinterpret_cast<const float*>(res) + off;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const uint4 w = ok ? ldg128(p + 4 * q) : make_uint4(0, 0, 0, 0);
    v[4 * q] = __uint_as_float(w.x); v[4 * q + 1] = __uint_as_float(w.y); v[4 * q + 2] = __uint_as_float(w.z); v[4 * q + 3] = __uint_as_float(w.w);
  }
#else
  const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(res) + off;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint4 w = ok ? ldg128(p + 8 * q) : make_uint4(0, 0, 0, 0);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) { v[8 * q + 2 * e] = bf16lo(ww[e]); v[8 * q + 2 * e + 1] = bf16hi(ww[e]); }
  }
#endif
}

// the barriers of one CTA (thread 0)
DEVI void tg_gemm_init(Bars& B) {
  for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
  for (int s = 0; s < 2; ++s) { mbar_init(&B.accf[s], 1); mbar_init(&B.acce[s], NEPI * NCTA); }
#if MODE == 6 || MODE == 7
  for (int o = 0; o < NOWN; ++o) { mbar_init(&B.ldab[o][0], 1); mbar_init(&B.ldab[o][1], 1); }
#endif
  fence_barrier_init();
}

// (round 3, training) the roles of one CTA over the tiles cta, cta + ncta, ... (the kernel below: blockIdx.x, gridDim.x), every thread of the
// CTA, barriers initialised and tensor memory allocated by the caller: tg_bwd_small.cu (-DTG_GEMM_BODY, no kernel) runs the gate and the
// d_xn tiles as phases of one launch through it. Maps as the kernel's.
DEVI void tg_gemm_roles(const CUtensorMap* pA0, const CUtensorMap* pA1, const CUtensorMap* pB0, const CUtensorMap* pB1, const CUtensorMap* pB2,
                        const CUtensorMap* pO0, const CUtensorMap* pO1, const void* __restrict__ res, const float* __restrict__ bias, int M, int N,
                        int K0, int K1, int ldres, int hoff, int KS, int MSTR, Bars& B, uint32_t su, uint32_t tmem, int cta, int ncta,
                        void* gO0 = nullptr, void* gO1 = nullptr) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#if H16
  const float hs1 = __ldg(bias + 1), hs2 = __ldg(bias + 2), hs3 = __ldg(bias + 3);
#else
  constexpr float hs3 = 1.f;                     // (x 1 is exact: the bf16 / TF32 builds' bits are unchanged)
#endif
  // a tile is 128 NCTA rows x BN columns; CTA `rank` of a pair owns rows rank * 128 .. of it (its TMEM lanes), and brings its A rows and
  // BNL of the BN B rows; the pair walks the same tile sequence
  const int NT = (N + BN - 1) / BN, MNT = ((M + 128 * NCTA - 1) / (128 * NCTA)) * NT, tiles = MNT * KS;
  const uint32_t rank = CTA2 ? cluster_rank() : 0u;
  const bool leader = rank == 0;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ producer
    if (lane == 0) {
      int it = 0;
      for (int t = cta / NCTA; t < tiles; t += ncta / NCTA) {
        const int ks = t / MNT, tt = t % MNT;
        const int m0 = (tt / NT) * 128 * NCTA + (int)rank * 128, n0 = (tt % NT) * BN + (int)rank * BNL;
#pragma unroll
        for (int p = 0; p < NPH; ++p) {
          const CUtensorMap* ma = p == 0 ? pA0 : pA1;
          const int nk = (p == 0 ? K0 : K1) / BKE / KS, a0 = ph_acc0(p), na = ph_nacc(p), kb0 = ks * nk;
          for (int k = kb0; k < kb0 + nk; ++k, ++it) {
            const int s = it % NST;
            mbar_wait(&B.empty[s], ((it / NST) & 1) ^ 1);
            if (leader) mbar_expect_tx(&B.full[s], (A_BYTES + na * B_BYTES) * NCTA);   // the pair's bytes land on the leader's barrier
            const uint32_t base = su + O_ST + s * STAGE;
            load_op(base, ma, &B.full[s], k * BKE, m0);
            for (int j = 0; j < na; ++j) {
              const int acc = a0 + j;
              const CUtensorMap* mb = acc == 0 ? pB0 : acc == 1 ? pB1 : pB2;
#if BMN
              for (int q = 0; q < BNL / BKE; ++q)            // [K][N] weights: BNL / BKE 128-B N blocks of BKE K rows
                load_op(base + A_BYTES + j * B_BYTES + q * BKE * 128, mb, &B.full[s], n0 + q * BKE, k * BKE);
#else
              load_op(base + A_BYTES + j * B_BYTES, mb, &B.full[s], k * BKE, n0);
#endif
            }
          }
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    if (lane == 0 && leader) {
      int it = 0, tc = 0;
      for (int t = cta / NCTA; t < tiles; t += ncta / NCTA, ++tc) {
        const int buf = tc & 1;
#if CTA2
        mbar_wait_cl(&B.acce[buf], ((tc >> 1) & 1) ^ 1);  // both CTAs' epilogues drained the buffer
#else
        mbar_wait(&B.acce[buf], ((tc >> 1) & 1) ^ 1);
#endif
        tc_fence_after();
        const uint32_t dbase = tmem + buf * TCOLS;
#pragma unroll
        for (int p = 0; p < NPH; ++p) {
          const int nk = (p == 0 ? K0 : K1) / BKE / KS, a0 = ph_acc0(p), na = ph_nacc(p);
          for (int k = 0; k < nk; ++k, ++it) {
            const int s = it % NST;
            mbar_wait(&B.full[s], (it / NST) & 1);
            tc_fence_after();
            const uint32_t base = su + O_ST + s * STAGE;
            const uint64_t da = desc_k128(base);
            for (int j = 0; j < na; ++j) {
              const uint64_t db = desc_b(base + A_BYTES + j * B_BYTES);
#pragma unroll
              for (int kk = 0; kk < 4; ++kk)
                umma(dbase + (a0 + j) * BN, da + (uint64_t)(kk * 2), db + (uint64_t)(kk * B_KSTEP), (k > 0 || kk > 0) ? 1u : 0u);
            }
            commit_all(&B.empty[s]);
          }
        }
        commit_all(&B.accf[buf]);
      }
    }
  } else {
    // ------------------------------------------------------------------------------------------------ epilogue (warps 2-9)
    const int ew = warp - 2, q4 = warp & 3, half = ew >> 2;  // lane quarter, column half
    const uint32_t r = (uint32_t)lane;                       // row within the quarter's 32-row slab
    // UPG == 1 (fp32 out): each warp owns whole groups (g % 2 == half) and its own slabs. UPG == 2 (bf16 out): the two warps of a
    // quarter write the two 64-B halves of one slab (the half-0 warp's), joined by named barrier 1 + q4, and the half-0 warp stores it.
    const int owner = UPG == 1 ? ew : (q4 + 0 * half);
    const uint32_t stg = su + O_STG + (uint32_t)(owner * NOUT * STG_PAR) * SLAB;
    const bool storer = UPG == 1 || half == 0;
    int tc = 0, grp = 0;
#if MODE == 6 || MODE == 7
    // GATE_AB with double-buffered staging (bf16): the a, b TMA loads run one staging group ahead of the group being computed (into the
    // other parity's dA / dB slabs, once that parity's previous stores have been read out), so their latency leaves the group's chain
    constexpr bool PREF = STG_PAR == 2;
    auto ab_issue = [&](int tq, int gq, int pq) {
      const int ksq = tq / MNT, ttq = tq % MNT;
      const int cq = (ttq % NT) * BN + gq * OCG;
      const int rq = ksq * MSTR + (ttq / NT) * 128 * NCTA + (int)rank * 128 + q4 * 32;
      uint64_t* bq = &B.ldab[owner][pq];
      mbar_expect_tx(bq, 2 * SLAB);
      tma_load_2d(stg + (uint32_t)(SL_A * STG_PAR + pq) * SLAB, pB1, bq, cq, rq);
      tma_load_2d(stg + (uint32_t)(SL_B * STG_PAR + pq) * SLAB, pB1, bq, hoff + cq, rq);
    };
    if (PREF && storer && lane == 0 && (int)(cta / NCTA) < tiles) ab_issue((int)(cta / NCTA), 0, 0);
#endif
    for (int t = cta / NCTA; t < tiles; t += ncta / NCTA, ++tc) {
      const int buf = tc & 1;
      const int ks = t / MNT, tt = t % MNT;
      const int m0 = (tt / NT) * 128 * NCTA + (int)rank * 128, n0 = (tt % NT) * BN;
      const int row = m0 + q4 * 32 + lane;
#if MODE == 6 || MODE == 7
      // (round 3, training) without PREF: the tile's first group's a, b are requested before its accumulator is ready, so their load
      // latency leaves the epilogue's chain (the one-group tiles of tg_bwd_small: ~1 us); same bits
      bool ab_early = false;
      if (!PREF) {
        const int g0 = UPG == 1 ? half : 0;
        if (g0 < BN / OCG && n0 + g0 * OCG < N) {
          if (storer && lane == 0) {
            tma_store_wait_read0();
            uint64_t* lb0 = &B.ldab[owner][0];
            const int rl = ks * MSTR + m0 + q4 * 32;
            mbar_expect_tx(lb0, 2 * SLAB);
            tma_load_2d(stg + (uint32_t)(SL_A * STG_PAR) * SLAB, pB1, lb0, n0 + g0 * OCG, rl);
            tma_load_2d(stg + (uint32_t)(SL_B * STG_PAR) * SLAB, pB1, lb0, hoff + n0 + g0 * OCG, rl);
          }
          ab_early = true;
        }
      }
#endif
      mbar_wait(&B.accf[buf], (tc >> 1) & 1);
      TG_TRACE(0);
      tc_fence_after();
      const uint32_t tb = tmem + ((uint32_t)(q4 * 32) << 16) + buf * TCOLS;
      for (int g = (UPG == 1 ? half : 0); g < BN / OCG; g += (UPG == 1 ? 2 : 1), ++grp) {
        const int c0 = n0 + g * OCG;                         // first output column of the group
        if (c0 >= N) break;
        const int par = STG_PAR == 2 ? (grp & 1) : 0;
#if MODE == 6 || MODE == 7
        if (PREF) {
          if (storer && lane == 0) {
            tma_store_wait_read0();                          // the other parity's slabs are free: prefetch the next group's a, b into them
            int tn = t, gn = g + 1;
            if (gn >= BN / OCG || n0 + gn * OCG >= N) { tn = t + (int)(ncta / NCTA); gn = 0; }
            if (tn < tiles) ab_issue(tn, gn, par ^ 1);
          }
        } else if (storer && lane == 0) { tma_store_wait_read0(); }
#else
        if (storer && lane == 0) { if (STG_PAR == 2) tma_store_wait_read1(); else tma_store_wait_read0(); }
#endif
#if MODE == 6 || MODE == 7
        // the group's a, b (the forward's [a | b], map mB1, box OCG x 32) into the dA / dB slabs, which their previous stores have left
        // (wait_read above): bulk loads in the layout stage32 writes, instead of 32 rows of per-thread loads per warp instruction
        uint64_t* lb = &B.ldab[owner][par];
        if (!PREF && storer && lane == 0 && !ab_early) {
          const int rl = ks * MSTR + m0 + q4 * 32;
          mbar_expect_tx(lb, 2 * SLAB);
          tma_load_2d(stg + (uint32_t)(SL_A * STG_PAR + par) * SLAB, pB1, lb, c0, rl);
          tma_load_2d(stg + (uint32_t)(SL_B * STG_PAR + par) * SLAB, pB1, lb, hoff + c0, rl);
        }
        ab_early = false;
#endif
        if (UPG == 2) named_bar_sync(1 + q4, 64); else __syncwarp();
        {
          const int u = UPG == 1 ? 0 : half;
          const int col = g * OCG + u * 32;                  // column within the tile
          uint32_t acc[NACC][32];
#pragma unroll
          for (int j = 0; j < NACC; ++j) tmem_ld32(tb + j * BN + col, acc[j]);
          tmem_wait_ld();
          TG_TRACE(1);
          float o0[32];
#if MODE == 0 || MODE == 5
#if MODE == 5
          float o1[32], o2[32];
#endif
#pragma unroll
          for (int e = 0; e < 32; ++e) {
#if H16
            const float a = __uint_as_float(acc[0][e]) * hs1, b = __uint_as_float(acc[1][e]) * hs1;
            const float h = a * sigm(a) * b * hs2;
#else
            const float a = __uint_as_float(acc[0][e]), b = __uint_as_float(acc[1][e]);
            const float h = a * sigm(a) * b;
#endif
            o0[e] = F32 ? tf32r(h) : h;
#if MODE == 5
            o1[e] = a; o2[e] = b;                            // as GATE's recompute sees them (fp32: unrounded; bf16: the saved rounding)
#endif
          }
#elif MODE == 1
          load_res32(res, (long)row * ldres + n0 + col, row < M, o0);
#pragma unroll
          for (int e = 0; e < 32; ++e) o0[e] += __uint_as_float(acc[0][e]) * hs3;
#elif MODE == 2
#pragma unroll
          for (int e = 0; e < 32; ++e) o0[e] = __uint_as_float(acc[0][e]);
#elif MODE == 3
          float o1[32], o2[32];
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            const float dh = __uint_as_float(acc[0][e]), a = __uint_as_float(acc[1][e]), b = __uint_as_float(acc[2][e]);
            const float s = sigm(a), l = a * s;
            o0[e] = F32 ? tf32r(l * b) : l * b;              // h (the dWs operand of tg_dw: on the TF32 grid, rounded to nearest)
            const float da = dh * b * (s + l * (1.f - s)), db = dh * l;
            o1[e] = F32 ? tf32r(da) : da;                    // dA, dB: the d_xn GEMM's A operand
            o2[e] = F32 ? tf32r(db) : db;
          }
#elif MODE == 6 || MODE == 7
          float o1[32], o2[32], av[32], bv[32];
          mbar_wait(lb, (grp >> (STG_PAR == 2 ? 1 : 0)) & 1);
          TG_TRACE(2);
          stage_ld32(stg + (uint32_t)(SL_A * STG_PAR + par) * SLAB, r, u, av);
          stage_ld32(stg + (uint32_t)(SL_B * STG_PAR + par) * SLAB, r, u, bv);
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            const float dh = __uint_as_float(acc[0][e]), a = av[e], b = bv[e];
            const float s = sigm(a), l = a * s;
            o0[e] = F32 ? tf32r(l * b) : l * b;              // h (the dWs operand of tg_dw: on the TF32 grid, rounded to nearest)
            const float da = dh * b * (s + l * (1.f - s)), db = dh * l;
            o1[e] = F32 ? tf32r(da) : da;
            o2[e] = F32 ? tf32r(db) : db;
          }
#elif MODE == 4
          if (ldres > 0) load_res32(res, (long)row * ldres + n0 + col, row < M, o0);   // ldres 0: no residual
          else {
#pragma unroll
            for (int e = 0; e < 32; ++e) o0[e] = 0.f;
          }
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            const float sp = __uint_as_float(acc[1][e]) + __ldg(bias + n0 + col + e);
            o0[e] += sigm(sp) * __uint_as_float(acc[0][e]);
          }
#endif
          TG_TRACE(3);
#if GSTORE
          {                                                  // (whole 32-row groups: the caller's M is a multiple of 128)
            const int u = UPG == 1 ? 0 : half;
            const long rr = (long)ks * MSTR + m0 + q4 * 32;    // the warp's first row
#if MODE != 7
            gst32(gO0, rr * N + c0 + u * 32, N, lane, o0);
#endif
#if MODE == 6 || MODE == 7
            gst32(gO1, rr * ldres + c0 + u * 32, ldres, lane, o1);
            gst32(gO1, rr * ldres + hoff + c0 + u * 32, ldres, lane, o2);
#endif
          }
#else
#if MODE != 7
          stage32(stg + (uint32_t)(0 * STG_PAR + par) * SLAB, r, u, o0);
#endif
#if MODE == 3 || MODE == 5 || MODE == 6 || MODE == 7
          stage32(stg + (uint32_t)(SL_A * STG_PAR + par) * SLAB, r, u, o1);
          stage32(stg + (uint32_t)(SL_B * STG_PAR + par) * SLAB, r, u, o2);
#endif
#endif  // GSTORE
        }
#if GSTORE
        if (UPG == 2) named_bar_sync(1 + q4, 64);          // both halves have read the group's a, b slabs before they are reloaded
#else
        fence_proxy_async();
        if (UPG == 2) named_bar_sync(1 + q4, 64); else __syncwarp();
        if (storer && lane == 0) {
          const int r0 = ks * MSTR + m0 + q4 * 32;
#if MODE != 7
          tma_store_2d(pO0, stg + (uint32_t)(0 * STG_PAR + par) * SLAB, c0, r0);
#endif
#if MODE == 3 || MODE == 5 || MODE == 6 || MODE == 7
          tma_store_2d(pO1, stg + (uint32_t)(SL_A * STG_PAR + par) * SLAB, c0, r0);
          tma_store_2d(pO1, stg + (uint32_t)(SL_B * STG_PAR + par) * SLAB, hoff + c0, r0);
#endif
          tma_store_commit();
        }
#endif
      }
      tc_fence_before();
      __syncwarp();
#if CTA2
      // the leader's warps arrive locally; the peer's relaxed (the TMEM reads are ordered by tcgen05.fence::before_thread_sync, and a
      // release at cluster scope costs ~0.5 us per tile on the critical path)
      if (lane == 0) { if (leader) mbar_arrive(&B.acce[buf]); else mbar_arrive_remote_relaxed(&B.acce[buf], 0); }
#else
      TG_TRACE(4);
      if (lane == 0) mbar_arrive(&B.acce[buf]);
#endif
    }
    if (!GSTORE && storer && lane == 0) tma_store_wait0();
  }
}

#ifndef TG_GEMM_BODY
// maps: A0 / A1 [M][K_p] (box BKE x 128), B0..B2 [N][K_p] (box BKE x BN), O0 / O1 outputs (box OCG x 32). Unused maps may repeat a used one.
extern "C" __global__ void __launch_bounds__(64 + 32 * NEPI, 1)
tg_gemm_sm100(const __grid_constant__ CUtensorMap mA0, const __grid_constant__ CUtensorMap mA1, const __grid_constant__ CUtensorMap mB0,
              const __grid_constant__ CUtensorMap mB1, const __grid_constant__ CUtensorMap mB2, const __grid_constant__ CUtensorMap mO0,
              const __grid_constant__ CUtensorMap mO1, const void* __restrict__ res, const float* __restrict__ bias, int M, int N, int K0,
              int K1, int ldres, int hoff, int KS, int MSTR) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) {
    tg_gemm_init(B);
    prefetch_map(&mA0); prefetch_map(&mB0); prefetch_map(&mO0);
    if (NPH > 1) prefetch_map(&mA1);
    if (NACC > 1) prefetch_map(&mB1);
  }
#if CTA2
  if (warp == 1) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
#else
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
#endif
  tc_fence_before();
  __syncthreads();
  if (CTA2) cluster_sync();                        // the leader's barriers exist before the peer's loads and arrivals target them
  pdl_wait();                                      // (round 3 part 5) programmatic dependent launches: the previous launch's outputs
  pdl_launch();                                    // are complete past this point; the next launch may start its prologue
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  tg_gemm_roles(&mA0, &mA1, &mB0, &mB1, &mB2, &mO0, &mO1, res, bias, M, N, K0, K1, ldres, hoff, KS, MSTR, B, su, tmem, (int)blockIdx.x,
                (int)gridDim.x);
  tc_fence_before();
  __syncthreads();
#if CTA2
  cluster_sync();                                  // the peer's last arrivals / the leader's last commits have landed
  if (warp == 1) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
#else
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
#endif
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
#endif
