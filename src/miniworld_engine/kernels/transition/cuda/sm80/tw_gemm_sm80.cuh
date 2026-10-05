// tw_gemm_sm80.cuh -- the tile GEMM of the wide-width A100 (sm_80) Transition kernels: C[BM x BN] += A[BM x K] B^T, bf16 -> fp32, mma.sync.m16n8k16.
//
//   * operands stream through a ST-stage cp.async ring (one __syncthreads per BK slice: the barrier publishes the landed slice AND frees
//     the stage the next load overwrites), A and B tiles are XOR-swizzled so ldmatrix and the 16-byte stores are conflict-free;
//   * A is [rows][K] (K contiguous); B is either [BN rows][K] (``BKN = false``: the weights of an nn.Linear, ldmatrix) or [K rows][BN]
//     (``BKN = true``: ldmatrix.trans);
//   * the warp grid is WM x WN, a warp owns MT m16 x NT n8 tiles of the CTA tile; the caller supplies the tile loaders (lambdas that
//     issue the cp.async copies of one k slice into a given stage), so a dual-B layout (a rows | b rows) or a gather is the caller's.
// Used by tw_kernels_sm80.cuh (the dual SwiGLU GEMM of the forward and the gate backward).
#pragma once
#include "sm80_common.cuh"

#ifndef TW_PIPE
#define TW_PIPE 1        // 1: the lean loop's barrier sits in the last k step of a slice (needs ST >= 4 to keep two slices in flight); 0: at the top of the slice
#endif

namespace a100 {

// byte offset of 16-byte chunk c of row r in a tile whose rows are CPR chunks (CPR * 16 bytes) long
template <int CPR>
DEVI uint32_t swz_off(uint32_t r, uint32_t c) {
  static_assert(CPR == 4 || CPR % 8 == 0, "tile row length");
  if constexpr (CPR == 4) return r * 64u + (((c ^ ((r >> 1) & 3u)) & 3u) << 4);
  else return r * (CPR * 16u) + ((c ^ (r & 7u)) << 4);
}

template <int BM_, int BN_, int BK_, int WM_, int WN_, int ST_, bool BKN_ = false>
struct GCfg {
  static constexpr int BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_, ST = ST_;
  static constexpr bool BKN = BKN_;
  static constexpr int NWARP = WM * WN, NTHR = 32 * NWARP;
  static constexpr int MINB = NTHR <= 128 ? 2 : 1;          // resident CTAs per SM the register budget is built for
  static constexpr int MT = BM / (16 * WM);                 // m16 tiles per warp
  static constexpr int NT = BN / (8 * WN);                  // n8 tiles per warp
  static constexpr int CPA = BK / 8;                        // 16-byte chunks per A row
  static constexpr int CPB = BKN ? BN / 8 : BK / 8;         // ... per B row
  static constexpr int A_STAGE = BM * BK * 2, B_STAGE = BN * BK * 2;
  static constexpr int SMEM = ST * (A_STAGE + B_STAGE);
  static_assert(BM % (16 * WM) == 0 && BN % (8 * WN) == 0 && NT % 2 == 0 && BK % 16 == 0, "warp tiling");
};

// one cp.async copy of a 16-byte chunk; src_bytes 0 zero-fills (a row past the end: the pointer stays valid)
DEVI void cp16(uint32_t dst, const void* src, bool ok) { cp_async16(dst, src, ok ? 16u : 0u); }

// Per-lane ldmatrix state of one warp: row / chunk-xor terms of the A fragment (rows wm * 16 MT + 16 mt + lane % 16, k chunk + lane / 16) and of the
// B fragment of an n8 pair (NK: rows n + lane % 8 + 8 (lane / 16), k chunk + (lane / 8) % 2; KN: k rows + lane % 8 + 8 ((lane / 8) % 2), n chunk + lane / 16)
template <class C>
struct Frag {
  uint32_t a_row, a_hi, b_row, b_hi;
  int wm, wn;
  DEVI Frag(int wm_, int wn_, int lane) : wm(wm_), wn(wn_) {
    a_row = (uint32_t)(wm * C::MT * 16 + (lane & 15));
    a_hi = (uint32_t)(lane >> 4);
    if constexpr (!C::BKN) { b_row = (uint32_t)((lane & 7) + 8 * (lane >> 4)); b_hi = (uint32_t)((lane >> 3) & 1); }
    else { b_row = (uint32_t)((lane & 7) + 8 * ((lane >> 3) & 1)); b_hi = (uint32_t)(lane >> 4); }
  }
  // A fragment of m16 tile mt, k step ks (16 k values) of the stage at a_base
  DEVI void a(uint32_t (&r)[4], uint32_t a_base, int mt, int ks) const {
    ldsm_x4(r, a_base + swz_off<C::CPA>(a_row + 16 * mt, 2 * ks + a_hi));
  }
  // B fragments of n8 tiles 2 np, 2 np + 1 (r[0..1], r[2..3]) at k step ks
  DEVI void b(uint32_t (&r)[4], uint32_t b_base, int np, int ks) const {
    if constexpr (!C::BKN) ldsm_x4(r, b_base + swz_off<C::CPB>((uint32_t)(wn * C::NT * 8 + 16 * np) + b_row, 2 * ks + b_hi));
    else ldsm_x4_t(r, b_base + swz_off<C::CPB>((uint32_t)(16 * ks) + b_row, (uint32_t)(wn * C::NT + 2 * np) + b_hi));
  }
};

// The mainloop.  load_a(stage_smem, kt) / load_b(stage_smem, kt) issue the cp.async copies of k slice kt (they do not commit).  smem: A stages, then B stages.
template <class C, class LA, class LB>
DEVI void gemm_mainloop(float (&acc)[C::MT][C::NT][4], uint32_t smem, int KT, const LA& load_a, const LB& load_b, const Frag<C>& fr) {
  const uint32_t sA = smem, sB = smem + C::ST * C::A_STAGE;
#pragma unroll
  for (int s = 0; s < C::ST - 1; ++s) {
    if (s < KT) { load_a(sA + s * C::A_STAGE, s); load_b(sB + s * C::B_STAGE, s); }
    cp_async_commit();
  }
  int rd = 0, wr = C::ST - 1;
#pragma unroll 1
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<C::ST - 2>();
    __syncthreads();
    if (kt + C::ST - 1 < KT) { load_a(sA + wr * C::A_STAGE, kt + C::ST - 1); load_b(sB + wr * C::B_STAGE, kt + C::ST - 1); }
    cp_async_commit();
    const uint32_t a_base = sA + rd * C::A_STAGE, b_base = sB + rd * C::B_STAGE;
    uint32_t fa[2][C::MT][4], fb[2][C::NT / 2][4];
#pragma unroll
    for (int mt = 0; mt < C::MT; ++mt) fr.a(fa[0][mt], a_base, mt, 0);
#pragma unroll
    for (int np = 0; np < C::NT / 2; ++np) fr.b(fb[0][np], b_base, np, 0);
#pragma unroll
    for (int ks = 0; ks < C::BK / 16; ++ks) {
      const int cb = ks & 1;
      if (ks + 1 < C::BK / 16) {
#pragma unroll
        for (int mt = 0; mt < C::MT; ++mt) fr.a(fa[cb ^ 1][mt], a_base, mt, ks + 1);
#pragma unroll
        for (int np = 0; np < C::NT / 2; ++np) fr.b(fb[cb ^ 1][np], b_base, np, ks + 1);
      }
#pragma unroll
      for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
        for (int np = 0; np < C::NT / 2; ++np) {
          mma16816(acc[mt][2 * np], fa[cb][mt], fb[cb][np][0], fb[cb][np][1]);
          mma16816(acc[mt][2 * np + 1], fa[cb][mt], fb[cb][np][2], fb[cb][np][3]);
        }
    }
    rd = rd + 1 == C::ST ? 0 : rd + 1;
    wr = wr + 1 == C::ST ? 0 : wr + 1;
  }
  cp_async_wait<0>();
  __syncthreads();                                   // every stage is free again: the epilogue may reuse the whole shared window
}

// ------------------------------------------------------------------------------------------------------------------ lean mainloop (whole tiles)
// The same ring and barrier protocol as gemm_mainloop for tiles whose rows all exist (no zero-fill, no predicates), with the address work moved out of the loop: the
// cp.async sources are per-thread pointers the caller computes ONCE (pa[j] / pb[j]: row of the thread's j-th copy at k = 0, its 16-byte chunk column included), the smem
// destinations are one per-thread term plus compile-time immediates (consecutive copies of a thread are NTHR / CPR rows apart: the swizzle term is the same), and the
// ldmatrix addresses are one per-thread offset per k step plus immediates (16-row steps leave the swizzle unchanged).  Measured with ncu: the generic loop spends ~140 address
// instructions on 64 HMMA per BK = 32 slice (the compiler rematerialises the lambdas' row pointers: 254 registers), cuBLAS ~25.
template <class C>
struct LeanFrag {
  static_assert(!C::BKN, "lean mainloop: weights are [rows][K]");
  static constexpr int KS = C::BK / 16;
  uint32_t a_off[KS], b_off[KS];            // smem byte offset (within a stage) of this lane's ldmatrix row, k step ks
  DEVI LeanFrag(int wm, int wn, int lane) {
    const uint32_t ar = (uint32_t)(wm * C::MT * 16 + (lane & 15)), ah = (uint32_t)(lane >> 4);
    const uint32_t br = (uint32_t)(wn * C::NT * 8 + (lane & 7) + 8 * (lane >> 4)), bh = (uint32_t)((lane >> 3) & 1);
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) { a_off[ks] = swz_off<C::CPA>(ar, 2 * ks + ah); b_off[ks] = swz_off<C::CPB>(br, 2 * ks + bh); }
  }
};

template <class C>
struct LeanCopies {                         // copies per thread of an A / B stage and the smem distance between a thread's consecutive copies
  static constexpr int NJA = C::BM * C::CPA / C::NTHR, NJB = C::BN * C::CPB / C::NTHR;
  static constexpr int RPJA = C::NTHR / C::CPA, RPJB = C::NTHR / C::CPB;      // tile rows between a thread's consecutive copies
  static_assert(NJA * C::NTHR == C::BM * C::CPA && NJB * C::NTHR == C::BN * C::CPB, "tile copy");
  static_assert(RPJA % 8 == 0 && RPJB % 8 == 0, "the swizzle term must not depend on the copy index");
};

template <class C>
DEVI void gemm_mainloop_lean(float (&acc)[C::MT][C::NT][4], uint32_t smem, int KT, const char* const (&pa)[LeanCopies<C>::NJA],
                             const char* const (&pb)[LeanCopies<C>::NJB], const LeanFrag<C>& fr, int tid) {
  constexpr int NJA = LeanCopies<C>::NJA, NJB = LeanCopies<C>::NJB;
  constexpr uint32_t JA = LeanCopies<C>::RPJA * C::CPA * 16, JB = LeanCopies<C>::RPJB * C::CPB * 16;      // smem bytes between a thread's consecutive copies
  constexpr uint32_t A_MT = 16 * C::CPA * 16, B_NP = 16 * C::CPB * 16;                                    // smem bytes of 16 tile rows
  const uint32_t sA = smem, sB = smem + C::ST * C::A_STAGE;
  const uint32_t dA = swz_off<C::CPA>(tid / C::CPA, tid % C::CPA), dB = swz_off<C::CPB>(tid / C::CPB, tid % C::CPB);
  auto issue = [&](int stage, int kt) {
    const size_t kb = (size_t)kt * (C::BK * 2);
#pragma unroll
    for (int j = 0; j < NJA; ++j) cp_async16(sA + stage * C::A_STAGE + dA + j * JA, pa[j] + kb);
#pragma unroll
    for (int j = 0; j < NJB; ++j) cp_async16(sB + stage * C::B_STAGE + dB + j * JB, pb[j] + kb);
  };
#pragma unroll
  for (int s = 0; s < C::ST - 1; ++s) {
    if (s < KT) issue(s, s);
    cp_async_commit();
  }
  constexpr int KS = C::BK / 16;
  static_assert(KS % 2 == 0, "fragment double buffer");
  uint32_t fa[2][C::MT][4], fb[2][C::NT / 2][4];
  auto load_frags = [&](int buf, int stage, int ks) {
    const uint32_t a_base = sA + stage * C::A_STAGE, b_base = sB + stage * C::B_STAGE;
#pragma unroll
    for (int mt = 0; mt < C::MT; ++mt) ldsm_x4(fa[buf][mt], a_base + fr.a_off[ks] + mt * A_MT);
#pragma unroll
    for (int np = 0; np < C::NT / 2; ++np) ldsm_x4(fb[buf][np], b_base + fr.b_off[ks] + np * B_NP);
  };
  auto mma_step = [&](int buf) {
#pragma unroll
    for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
      for (int np = 0; np < C::NT / 2; ++np) {
        mma16816(acc[mt][2 * np], fa[buf][mt], fb[buf][np][0], fb[buf][np][1]);
        mma16816(acc[mt][2 * np + 1], fa[buf][mt], fb[buf][np][2], fb[buf][np][3]);
      }
  };
#if TW_PIPE
  // The barrier of the ring sits in the LAST k step of a slice: the next slice's wait + __syncthreads + first fragment loads are issued before that step's MMAs, so their latency
  // hides under 32 MMAs per warp instead of idling the tensor pipe at every slice boundary (the CUTLASS multistage order).  Slice kt + 1 must have landed at slice kt's last step:
  // one group less of lead than the plain loop (wait_group ST - 3), so the ring needs ST >= 4 to keep two slices in flight.
  static_assert(C::ST >= 3, "ring depth");
  cp_async_wait<C::ST - 2>();
  __syncthreads();                                   // slice 0 landed
  load_frags(0, 0, 0);
  int rd = 0, wr = C::ST - 1;
#pragma unroll 1
  for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const int cur = ks & 1;
      if (ks + 1 < KS) {
        load_frags(cur ^ 1, rd, ks + 1);
      } else {
        cp_async_wait<C::ST - 3>();                  // slice kt + 1 landed (this thread's groups) ...
        __syncthreads();                             // ... for every thread; and nobody reads slice kt - 1's slot (wr) any more
        if (kt + C::ST - 1 < KT) issue(wr, kt + C::ST - 1);
        cp_async_commit();
        rd = rd + 1 == C::ST ? 0 : rd + 1;
        wr = wr + 1 == C::ST ? 0 : wr + 1;
        if (kt + 1 < KT) load_frags(cur ^ 1, rd, 0);
      }
      mma_step(cur);
    }
  }
#else
  int rd = 0, wr = C::ST - 1;
#pragma unroll 1
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<C::ST - 2>();
    __syncthreads();
    if (kt + C::ST - 1 < KT) issue(wr, kt + C::ST - 1);
    cp_async_commit();
    load_frags(0, rd, 0);
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      if (ks + 1 < KS) load_frags((ks & 1) ^ 1, rd, ks + 1);
      mma_step(ks & 1);
    }
    rd = rd + 1 == C::ST ? 0 : rd + 1;
    wr = wr + 1 == C::ST ? 0 : wr + 1;
  }
#endif
  cp_async_wait<0>();
  __syncthreads();                                   // every stage is free again: the epilogue may reuse the whole shared window
}

template <class C>
DEVI void zero_acc(float (&acc)[C::MT][C::NT][4]) {
#pragma unroll
  for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < C::NT; ++nt) { acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f; }
}

// Generic tile loader for an operand stored [rows][ld] (K contiguous): rows r0 .. r0 + R of columns k0 .. k0 + BK into a stage laid out [R][BK] (CPR = BK / 8 chunks per row)
// row_ptr(r) returns the global pointer of row r (r < R) of the tile or nullptr to zero-fill
template <int R, int CPR, int NTHR, class RowPtr>
DEVI void load_rows_k(uint32_t stage, int k0, const RowPtr& row_ptr, int tid) {
  static_assert((R * CPR) % NTHR == 0, "tile copy");
#pragma unroll
  for (int j = 0; j < (R * CPR) / NTHR; ++j) {
    const int i = tid + j * NTHR, r = i / CPR, c = i % CPR;
    const __nv_bfloat16* p = row_ptr(r);
    cp16(stage + swz_off<CPR>(r, c), (p != nullptr ? p : row_ptr(0)) + k0 + c * 8, p != nullptr);
  }
}

}  // namespace a100
