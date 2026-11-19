// gattn_fwd.cu — the SWA atom block's GLOBAL attention forward on sm_100a (tcgen05), in place of FlashAttention-4's on the B200:
//
//   O[n, i, h] = softmax_j(q[n, h, i] . k[n, h, j] scale) v[n, h, j]   over the keys j < seqused[n]     (bf16; query rows i >= seqused[n]: 0)
//   LSE[n, h, i] = lmul log2 sum_j exp(q . k scale)                     (lmul = ln 2: the natural log, FA4's convention; lmul = -1: the negated log2
//                                                                        gattn_bwd.cu reads; 0 on the rows i >= seqused[n]); LSE may be null
//
// Qh | Kh | Vh are the head-major [N, 4, S, 32] bf16 heads the qkvg stage writes (RMSNorm + 3D RoPE already applied), dense 64-B rows,
// 64-B swizzled tiles; O is the token-major [N S, 128] bf16 the out-projection stage reads. S is a multiple of 128.
// Contract: q and k are RMS-normalized over the head dim WITHOUT a weight (SWA3DRoPEAttention: F.rms_norm, no affine), so
// |q . k| scale <= |q| |k| / sqrt(32) = sqrt(32) and the exponent q . k scale log2(e) lies in [-8.2, 8.2]: no running maximum is kept
// (P = 2^(s scale log2 e) directly; no row maximum, no O rescale -- the softmax is the exponentials, their row sum and the bf16 packing).
// Schedule (from the token DiT's inference core, augmented_attention/cuda/sm100/attn_inf.cu): a work item is (sample, head, pair of
// 128-query tiles m0, m0 + 128); softmax warpgroup w takes tile m0 + 128 w, both read the same 64-key K | V blocks (loaded once per stage);
// the partner past the last tile (odd tile count) recomputes the last one and does not store. The key axis of an item stops at
// ceil(seqused[n] / 64) blocks (the last block's keys past seqused masked). S and P double-buffered per warpgroup (the exponentials of
// block G overlap PV(G - 1)); the two warpgroups take turns on the exponentials (PP).
// Commits per block and MMA warp: s_full (after QK(G + 2), which follows PV(G): it also says PV(G) is done, so P buffer G & 1 is free
// -- no separate P-free barrier) and kv_empty; o_done once per item (the epilogue's O is complete).
// Barriers: every signal a consumer can lag two phases behind is one barrier per buffer (s_full / p_full [w][G & 1]) -- the parity ABA
// rule of attn_inf.cu.
// CL > 1 (built, not routed: measured 0 - 2 % at N 1 - 48, worse at N 5 S 1024): a cluster of CL CTAs takes CL consecutive tile pairs of
// one head and multicasts each stage's K | V parts (L2 -> SM traffic / CL).
// TMEM: S[w][b] at w * 128 + b * 64 (64 cols), P[w][b] at 256 + w * 64 + b * 32 (bf16, 32 cols), O[w] at 384 + w * 64 (32 cols).
// SPLITB=1 (the split build, round 3 part 3): an item's key range cut across CTAs where whole items leave SMs idle (see the kernel's
// split comment; host: GattnFwd.splits). P16=1 (built, not routed): P in fp16, V converted to fp16 in shared memory by warp 3.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef CL
#define CL 1                             // cluster size (1, 2, 4): K | V multicast
#endif
#ifndef PP
#define PP 1                             // the two softmax warpgroups take turns on the exponentials
#endif
#ifndef ST
#define ST 6                             // K | V stages (8 KB each)
#endif
#ifndef QR
#define QR 2                             // q ring slots (2 x 8 KB each)
#endif
#ifndef LSUMR
#define LSUMR 0                          // 1: the row sum of the bf16-ROUNDED P (2026-11-17: no accuracy change in training, +13 - 16 % time)
#endif
#ifndef P16
#define P16 0                            // 1: P in fp16 (10 mantissa bits, 8x finer than bf16; P in [2^-8.2, 2^8.2] fits): PV on f16
                                         // operands, each stage's V converted bf16 -> fp16 in shared memory by warp 3 (training forward)
#endif
#ifndef EXH
#define EXH 0                            // r3-swa part 18 probe: with P16, P = ex2.approx.f16x2 of the f16-rounded scores (one MUFU op a pair)
#endif
#ifndef MIXV
#define MIXV 0                           // probe: P16 with V left bf16 (a mixed-type f16 / bf16 MMA)
#endif
#ifndef SPLITB
#define SPLITB -1                        // 0: whole items only, 1: split mode only, -1: the runtime flag
#endif
#ifndef NOMMA
#define NOMMA 0                         // timing probe (wrong values): 1 no QK MMAs, 2 no PV MMAs
#endif
#ifndef NOEXP
#define NOEXP 0                         // timing probe: no exponentials (wrong values)
#endif
#ifndef FPMOD                            // r3-swa part 18: of every FPMOD exponential pairs, FPCNT on the FMA pipe (ex2 by a degree-3 polynomial,
#define FPMOD 8                          // as gattn_bwd.cu / FA4), the rest on MUFU. F32IO builds 1 of 8 (S5 L 8192 522 -> 459 us alone, N 1 L 8192
#endif                                   // 114.7 -> 102.4; level at L 1024 - 6144 one sample); bf16 none (1 of 4: level to +3 %, 2 of 4: +20 %)
#ifndef FPCNT
#define FPCNT (F32IO ? 1 : 0)
#endif
#ifndef CMBS
#define CMBS (F32IO ? 0 : 2)             // N7: split-mode combine through shared-memory staging, CMBS foreign pieces a round (0: off; F32IO: its staging does not fit beside the raw tiles, and measured level)
#endif
DEVI f2 fsub2(f2 a, f2 b) { f2 r; asm("sub.rn.f32x2 %0, %1, %2;" : "=l"(r.v) : "l"(a.v), "l"(b.v)); return r; }
// 2^x of a pair on the FMA pipe (gattn_bwd.cu ex2_poly2); x >= -126 (callers clamp; masked -inf entries selected to 0 after)
DEVI f2 fex2_poly2(f2 x) {
  const f2 C = mk2(12582912.f, 12582912.f);
  const f2 j = add2(x, C);
  const f2 f = fsub2(x, fsub2(j, C));
  f2 p = fma2(mk2(0.0555041086648216f, 0.0555041086648216f), f, mk2(0.2402264923172785f, 0.2402264923172785f));
  p = fma2(p, f, mk2(0.6931471805599453f, 0.6931471805599453f));
  p = fma2(p, f, mk2(1.0f, 1.0f));
  return mk2(__int_as_float(__float_as_int(lo2(p)) + (__float_as_int(lo2(j)) << 23)), __int_as_float(__float_as_int(hi2(p)) + (__float_as_int(hi2(j)) << 23)));
}
#ifndef TRACE
#define TRACE 0                          // -DTRACE=1: clock64 stamps of CTA 0 into TRb[16 G + k] (k < 8: softmax row 0 of warpgroup 0; 8 ..: its MMA warp)
#endif
#define EV(k) do { if (TRACE && blockIdx.x == 0 && w == 0 && r == 0 && G < 128) TRb[16 * G + (k)] = clock64(); } while (0)
#define EVM(k, g) do { if (TRACE && blockIdx.x == 0 && w == 0 && lane == 0 && (g) < 128) TRb[16 * (g) + (k)] = clock64(); } while (0)
DEVI void cp_async16(uint32_t d, const void* s) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(s) : "memory"); }
DEVI void cp_async_wait_all() { asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory"); }
DEVI void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
DEVI uint32_t mapa_u32(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
DEVI void st_async4(uint32_t dst, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
               :: "r"(dst), "r"(a), "r"(b), "r"(c), "r"(d), "r"(bar) : "memory");
}
DEVI void st_async1(uint32_t dst, uint32_t a, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" :: "r"(dst), "r"(a), "r"(bar) : "memory");
}

#ifndef F32IO
#define F32IO 0                          // 1: fp32 io (the fp32 block's inference, round 3 part 4): Qh | Kh | Vh fp32 head-major, TMA-loaded raw
                                         // into staging and rounded to the bf16 tiles by warp 3; O written fp32 (bf16 values) by the softmax threads.
                                         // 2: only q fp32 (converted once per item); K | V bf16 as usual (one conversion per head, not per item)
#endif
#ifndef RS
#define RS 3                             // F32IO: raw K | V stages (16 KB each)
#endif
#ifndef BNK
#define BNK 64                          // keys per block; != 64: P aliased onto its S in TMEM (BNK 96: S[w][b] 96 columns, O at 384)
#endif
#ifndef CSPLIT
#define CSPLIT 0                         // 1 (r3-swa part 17): the CLUSTER split -- one item per cluster of CS = %cluster_nctarank CTAs
                                         // (launch attribute: 2 - 8), rank k the k-th contiguous share of the item's key blocks; the
                                         // partial (O, l) reduce-scattered over DSMEM by st.async (row r -> rank r / (128 / CS), lane = row:
                                         // one 512-B contiguous run per instruction), summed by the row's owner in rank order (fixed)
#endif
constexpr int BN = BNK, DH = 32, QM = 128, H = 4;
constexpr bool PAL = BN != 64;
static_assert(BN % 32 == 0 && (!PAL || 4 * BN + 64 <= 512), "TMEM: four S buffers + two O");
constexpr int TQ = QM * 64, TK = BN * 64, STB = 2 * TK;                     // q tile 8 KB; stage: K | V, 4 KB each
constexpr int O_Q = 0, O_ST = QR * 2 * TQ, O_X = O_ST + ST * STB, XS = QM * 64;
constexpr int NMAX = 256, PMAX = 1100;                                    // split mode: samples, pieces of one CTA
constexpr int TKR = BN * 128, TQR = QM * 128;                              // F32IO raw tiles: K or V block 8 KB, q tile 16 KB
constexpr int O_QR = O_X + 2 * XS, O_R = O_QR + (F32IO ? 2 * TQR : 0);    // F32IO: q tile pair raw (one slot), RS raw K | V stages
constexpr int O_BAR = O_R + (F32IO == 1 ? RS * 2 * TKR : 0), O_PF = O_BAR + 1024, O_PL = O_PF + 4 * (NMAX + 4);
// CSPLIT receive area: data [src rank][w][chunk 0..7][RO rows] x 16 B (2 x 128 x 128 B whatever CS), then l [src][w][RO] fp32
constexpr int O_RCV = (O_PL + 16 * PMAX + 1023) / 1024 * 1024, RCV_D = 2 * QM * DH * 4, RCV_BYTES = RCV_D + 2 * QM * 4;
constexpr int SMEM_BYTES = CSPLIT ? O_RCV + RCV_BYTES : O_PL + 16 * PMAX;
static_assert(!(CSPLIT && CL > 1), "CSPLIT: no multicast build");
constexpr int WSF = QM * DH + QM;                                          // split: floats of one partial (O [128][32], l [128])
constexpr int CMB_ROW = 144, CMB_P = QM * CMB_ROW + QM * 4, CMB_W = (CMBS ? CMBS + 1 : 1) * CMB_P;   // staging: a piece = 128 rows x 144 B (128 B of O + pad) + 128 l
constexpr int O_CMB = (O_PL + 16 * PMAX + 1023) / 1024 * 1024;
static_assert(!CMBS || O_CMB + 2 * CMB_W <= 232448, "combine staging");
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(O_ST % 1024 == 0 && O_X % 1024 == 0, "tile alignment");
static_assert(ST > 2, "QK(G + 2) is issued before block G's stage is released");
// TMEM column of S[w][b], P[w][b] (bf16; PAL: inside S[w][b]) and O[w]
__host__ __device__ constexpr uint32_t t_s(int w, int b) { return PAL ? (uint32_t)(w * 2 * BN + b * BN) : (uint32_t)(w * 128 + b * 64); }
__host__ __device__ constexpr uint32_t t_p(int w, int b) { return PAL ? t_s(w, b) : (uint32_t)(256 + w * 64 + b * 32); }
__host__ __device__ constexpr uint32_t t_o(int w) { return PAL ? (uint32_t)(4 * BN + w * 32) : (uint32_t)(384 + w * 64); }
constexpr uint32_t I_QK = idesc_bf16(128, BN),
                   I_PV = idesc_bf16(128, DH, 0, 1) & ~(P16 ? ((1u << 7) | (MIXV ? 0u : (1u << 10))) : 0u);   // P16: f16 A (and B)
constexpr bool VCONV = P16 && !MIXV;                                       // warp 3 converts each stage's V to fp16
static_assert(!(P16 && CL > 1), "P16: no multicast build");
static_assert(!EXH || P16, "EXH: P16 build");
static_assert(!(F32IO && (P16 || CL > 1 || PAL)), "F32IO: warp 3 converts (no P16), no multicast, 64-key blocks");
// F32IO: R raw fp32 rows (128 B, no swizzle) -> the bf16 64-B swizzled tile the MMAs read; lane-strided 16-B chunks
DEVI void cvt_rows(uint32_t src, uint32_t dst, int R, int lane) {
#pragma unroll 4
  for (int i = lane; i < R * 4; i += 32) {
    const uint32_t row = (uint32_t)i >> 2, c = (uint32_t)i & 3u;
    const uint4 a = lds128(src + row * 128u + c * 32u), b = lds128(src + row * 128u + c * 32u + 16u);
    sts128(dst + sw64(row, c), make_uint4(pack_bf16(__uint_as_float(a.x), __uint_as_float(a.y)), pack_bf16(__uint_as_float(a.z), __uint_as_float(a.w)),
                                          pack_bf16(__uint_as_float(b.x), __uint_as_float(b.y)), pack_bf16(__uint_as_float(b.z), __uint_as_float(b.w))));
  }
}
DEVI uint32_t bf2h2(uint32_t u) { return f2h2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;

struct Bars {
  uint64_t q_full[QR], q_empty[QR], kv_full[ST], kv_empty[ST], s_full[2][2], p_full[2][2], o_free[2], o_done[2], vc_full[ST];
  uint64_t qraw_full, qraw_empty, raw_full[RS], raw_empty[RS];             // F32IO: the raw fp32 staging
  uint64_t rcv;                                                            // CSPLIT: every rank's rows of mine have landed
  uint32_t tmem;
  int npieces, last[2];
};

extern "C" __global__ void __launch_bounds__(384, 1)
swa_gattn_fwd_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                    const __grid_constant__ CUtensorMap mo, const int* __restrict__ SEQU, float* __restrict__ LSE, int S, int N, float scale, float lmul,
                    float* __restrict__ WS, int* __restrict__ CNT, int split, unsigned long long* __restrict__ TRb, float* __restrict__ O32) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  int* PF = reinterpret_cast<int*>(sm + O_PF);                             // split: key-block prefix over the samples [N + 1]
  int4* PL = reinterpret_cast<int4*>(sm + O_PL);                           // split: this CTA's pieces {n, item of the sample, b0, nb}
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = S / QM, npt = (mt + 1) >> 1;
  // CL > 1: a cluster takes CL consecutive tile pairs of one (sample, head) at a time -- the same K | V block sequence, each CTA loading
  // 1 / CL of every stage multicast to all; a tile pair past the head's last (npt % CL != 0) is recomputed as the last one and not stored
  const int crank = CL > 1 ? (int)cluster_rank() : 0, ci = (int)blockIdx.x / CL, ncl = (int)gridDim.x / CL;
  const int npc = (npt + CL - 1) / CL;
  const int items = N * H * npc;
  // CSPLIT: cluster ci_s = item, rank csr of CS
  uint32_t csn = 1, csr = 0;
  if (CSPLIT) {
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(csn));
    csr = cluster_rank();
  }
  const int ci_s = (int)blockIdx.x / (int)csn;
  // split (CL 1, few items for the grid: one sample, S5 at L <= 2048, ...): the (item, key block) sequence of all items -- items in
  // (sample, head, tile pair) order, each ceil(sq / 64) key blocks -- is cut into gridDim.x equal contiguous ranges, one per CTA; an
  // item cut by a range boundary runs as pieces on consecutive CTAs, each piece's O (unnormalized fp32) and row sum go to WS
  // (slot (CTA, first / last piece of its range, warpgroup)), and the piece that completes the item's counter (CNT [item][w]) adds
  // the pieces in CTA order -- fixed, whoever comes last -- and writes O / LSE. No running maximum: partial sums add directly.
  const int GD = (int)gridDim.x, cid = (int)blockIdx.x;
  const bool SPL = CL == 1 && (SPLITB < 0 ? split != 0 : SPLITB == 1);   // SPLITB: the mode fixed at build time (the split code costs the whole-item
                                                                           // loop ~10 % when merely present)
  long long T = 0;
  auto start_of = [&](int cc) { return (long long)cc * T / GD; };
  auto cta_of = [&](long long x) { return (int)(((x + 1) * GD - 1) / T); };
  struct Pc { int n, h, m0, sq, nb, b0, it, c0, c1; long long ist; bool st; };
  auto item_of = [&](int li, Pc& p) {
    if (SPL) {
      const int4 e = PL[li];
      p.n = e.x; p.h = e.y / npt; p.m0 = (e.y - p.h * npt) * 2 * QM; p.b0 = e.z; p.nb = e.w;
      const int nbn = (PF[p.n + 1] - PF[p.n]) / (H * npt);
      p.ist = PF[p.n] + (long long)e.y * nbn;
      p.c0 = cta_of(p.ist); p.c1 = cta_of(p.ist + nbn - 1); p.it = p.n * H * npt + e.y; p.st = true;
    } else if (CSPLIT) {
      const int hx = ci_s / npt;
      p.m0 = (ci_s - hx * npt) * 2 * QM; p.n = hx / H; p.h = hx - p.n * H;
      p.c0 = p.c1 = cid; p.it = 0; p.ist = 0; p.st = true;
      const int sq = min(max(__ldg(SEQU + p.n), 0), S), nbe = max((sq + BN - 1) / BN, (int)csn);   // >= 1 block a rank (masked past sq)
      p.b0 = (int)csr * nbe / (int)csn;
      p.nb = ((int)csr + 1) * nbe / (int)csn - p.b0;
    } else {
      const int wi = ci + li * ncl;
      const int hx = wi / npc;
      p.m0 = min((wi - hx * npc) * CL + crank, npt - 1) * 2 * QM; p.n = hx / H; p.h = hx - p.n * H;
      p.b0 = 0; p.c0 = p.c1 = cid; p.it = 0; p.ist = 0;
      p.st = (wi - hx * npc) * CL + crank < npt;
    }
    p.sq = min(max(__ldg(SEQU + p.n), 0), S);
    if (!SPL && !CSPLIT) p.nb = max((p.sq + BN - 1) / BN, 1);
  };
  auto row_of = [&](int m0, int w) { return min(m0 + w * QM, (mt - 1) * QM); };

  if (tid == 0) {
    for (int s = 0; s < QR; ++s) { mbar_init(&B.q_full[s], 1); mbar_init(&B.q_empty[s], 2); }   // released by both MMA warps
    for (int s = 0; s < ST; ++s) { mbar_init(&B.kv_full[s], 1); mbar_init(&B.kv_empty[s], 2 * CL); mbar_init(&B.vc_full[s], 1); }
    for (int w = 0; w < 2; ++w) {
      mbar_init(&B.s_full[w][0], 1); mbar_init(&B.s_full[w][1], 1);
      mbar_init(&B.p_full[w][0], 4); mbar_init(&B.p_full[w][1], 4); mbar_init(&B.o_free[w], 4); mbar_init(&B.o_done[w], 1);
    }
    if (F32IO) {
      mbar_init(&B.qraw_full, 1); mbar_init(&B.qraw_empty, 1);
      for (int s = 0; s < RS; ++s) { mbar_init(&B.raw_full[s], 1); mbar_init(&B.raw_empty[s], 1); }
    }
    if (CSPLIT) { mbar_init(&B.rcv, 1); mbar_expect_tx(&B.rcv, RCV_BYTES); }   // armed before the cluster barrier below
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (CL > 1 || CSPLIT) cluster_sync();                                    // every CTA's barriers are initialized before any multicast / st.async
  pdl_launch();
  pdl_wait();                                                              // seqused / q / K / V come from the kernels before
  int my_items;
  if (SPL) {
    if (tid == 0) {
      int acc = 0;
      PF[0] = 0;
      for (int n = 0; n < N; ++n) { acc += H * npt * max((min(max(__ldg(SEQU + n), 0), S) + BN - 1) / BN, 1); PF[n + 1] = acc; }
      T = acc;
      const long long B1 = start_of(cid + 1);
      long long x = start_of(cid);
      int np = 0, n = 0;
      while (x < B1) {
        if (np == PMAX) __trap();                                          // the host keeps split launches far below PMAX pieces
        while (PF[n + 1] <= x) ++n;
        const int nbn = (PF[n + 1] - PF[n]) / (H * npt), off = (int)(x - PF[n]), itl = off / nbn;
        const long long e = min(B1, (long long)PF[n] + (long long)(itl + 1) * nbn);
        PL[np++] = make_int4(n, itl, off - itl * nbn, (int)(e - x));
        x = e;
      }
      B.npieces = np;
    }
    __syncthreads();
    T = PF[N];
    my_items = B.npieces;
  } else if (CSPLIT) {
    my_items = ci_s < N * H * npt ? 1 : 0;                                 // the host launches exactly items x CS CTAs
  } else {
    my_items = (items > ci) ? (items - ci + ncl - 1) / ncl : 0;
  }

  if (warp == 0) {
    // ------------------------------------------------------------------------------------ TMA producer: the q tile pair, then K | V blocks
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        Pc P; item_of(li, P);
        const int qs = li % QR, row = (P.n * H + P.h) * S, m0 = P.m0, nb = P.nb, kb0 = row + P.b0 * BN;
        if (F32IO) {                                                       // raw fp32 into staging; warp 3 makes the bf16 tiles
          if (li >= 1) mbar_wait(&B.qraw_empty, (li - 1) & 1);
          mbar_expect_tx(&B.qraw_full, 2 * TQR);
          for (int w = 0; w < 2; ++w) tma_load_2d(su + O_QR + w * TQR, &mq, &B.qraw_full, 0, row + row_of(m0, w));
        }
        if (F32IO == 1) {
          for (int b = 0; b < nb; ++b, ++g) {
            const int s = g % RS;
            if (g >= RS) mbar_wait(&B.raw_empty[s], ((g / RS) - 1) & 1);
            const uint32_t st = su + O_R + s * 2 * TKR;
            mbar_expect_tx(&B.raw_full[s], 2 * TKR);
            tma_load_2d(st, &mk, &B.raw_full[s], 0, kb0 + b * BN);
            tma_load_2d(st + TKR, &mv, &B.raw_full[s], 0, kb0 + b * BN);
          }
          continue;
        }
        if (!F32IO) {
          if (li >= QR) mbar_wait(&B.q_empty[qs], ((li / QR) - 1) & 1);
          mbar_expect_tx(&B.q_full[qs], 2 * TQ);
          for (int w = 0; w < 2; ++w) tma_load_2d(su + O_Q + qs * 2 * TQ + w * TQ, &mq, &B.q_full[qs], 0, row + row_of(m0, w));
        }
        for (int b = 0; b < nb; ++b, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.kv_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.kv_full[s], STB);                              // every part of the stage lands on every CTA's kv_full
          if (CL == 1) {
            tma_load_2d(st, &mk, &B.kv_full[s], 0, kb0 + b * BN);
            tma_load_2d(st + TK, &mv, &B.kv_full[s], 0, kb0 + b * BN);
          } else {                                                         // part crank: K or V (crank & 1), rows (crank >> 1) of CL / 2
            constexpr int PR = BN / (CL > 2 ? CL / 2 : 1);
            const int hp = crank >> 1;
            tma_load_2d_mc(st + (crank & 1) * TK + hp * PR * 64, (crank & 1) ? &mv : &mk, &B.kv_full[s], 0, kb0 + b * BN + hp * PR,
                           (uint16_t)((1u << CL) - 1));
          }
        }
      }
      if (CL > 1)                                                          // drain: every CTA's release of my last stages has landed
        for (int x = max(g - ST, 0); x < g; ++x) mbar_wait(&B.kv_empty[x % ST], (x / ST) & 1);
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------ MMA issuers: warp 1 + w serves warpgroup w
    // QK(G + 2) follows PV(G): S and P double-buffered. Two cursors over this CTA's (item, block) sequence: c (PV) and a (QK, 2 ahead).
    const int w = warp - 1;
    struct Cur { int li, b, nb; };
    auto start = [&](Cur& c) { c.li = 0; c.b = 0; c.nb = 0; if (my_items > 0) { Pc P; item_of(0, P); c.nb = P.nb; } };
    auto step = [&](Cur& c) {
      if (++c.b == c.nb) { c.b = 0; ++c.li; if (c.li < my_items) { Pc P; item_of(c.li, P); c.nb = P.nb; } }
    };
    auto qk = [&](int G, const Cur& a) {
      const int s = G % ST, qs = a.li % QR;
      if (a.b == 0) mbar_wait(&B.q_full[qs], (a.li / QR) & 1);
      mbar_wait(&B.kv_full[s], (G / ST) & 1);
      tc_fence_after();
      const uint64_t dq = desc_sw64(su + O_Q + qs * 2 * TQ + w * TQ), dk = desc_sw64(su + O_ST + s * STB);
      const bool last = a.b == a.nb - 1;
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks) if (!(NOMMA & 1)) umma_ss(tmem + t_s(w, G & 1), dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_QK, ks > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w][G & 1]);                                    // also: every MMA before it (PV(G - 2)) is done
        if (last) tc_commit(&B.q_empty[qs]);
      }
      __syncwarp();
    };
    Cur a, c;
    start(a); start(c);
    int GA = 0;
    for (; GA < 2 && a.li < my_items; ++GA) { qk(GA, a); step(a); }
    for (int G = 0; c.li < my_items; ++G) {
      const int s = G % ST;
      EVM(8, G);
      mbar_wait(&B.p_full[w][G & 1], (G >> 1) & 1);
      EVM(9, G);
      if (c.b == 0 && c.li >= 1) mbar_wait(&B.o_free[w], (c.li - 1) & 1);  // the previous item's O has been read out
      if (VCONV) mbar_wait(&B.vc_full[s], (G / ST) & 1);                   // block G's V is fp16 now
      tc_fence_after();
      const uint64_t dv = desc_sw64(su + O_ST + s * STB + TK);              // MN-major: 8-key groups 512 B apart
      const uint32_t acc0 = c.b > 0 ? 1u : 0u;
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < BN / 16; ++ks)
          if (!(NOMMA & 2)) umma_ts(tmem + t_o(w), tmem + t_p(w, G & 1) + ks * 8, dv + (uint64_t)(ks * 64), I_PV, ks > 0 ? 1u : acc0);
        if (c.b == c.nb - 1) tc_commit(&B.o_done[w]);                    // the item's O is complete
      }
      __syncwarp();
      EVM(10, G);
      if (a.li < my_items) { qk(GA, a); step(a); ++GA; }
      if (elect_one()) {                                                   // after QK(G + 2): block G's stage (its V read by PV(G))
        if (CL == 1) tc_commit(&B.kv_empty[s]);
        else tc_commit_mc(&B.kv_empty[s], (uint16_t)((1u << CL) - 1));    // released in every CTA of the cluster
      }
      __syncwarp();
      EVM(11, G);
      step(c);
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------ F32IO: raw fp32 q / K | V -> the bf16 tiles
    // (the producer's order: an item's q tile pair, then its blocks); each raw slot freed once read, each bf16 slot filled once free
    if (F32IO) {
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        Pc P; item_of(li, P);
        const int qs = li % QR;
        mbar_wait(&B.qraw_full, li & 1);
        if (li >= QR) mbar_wait(&B.q_empty[qs], ((li / QR) - 1) & 1);
        cvt_rows(su + O_QR, su + O_Q + qs * 2 * TQ, 2 * QM, lane);         // both tiles: raw rows 0 .. 255 -> two SW64 tiles back to back
        fence_proxy_async();                                               // the MMAs read them through the async proxy
        __syncwarp();
        if (lane == 0) { mbar_arrive(&B.q_full[qs]); mbar_arrive(&B.qraw_empty); }
        for (int b = 0; F32IO == 1 && b < P.nb; ++b, ++g) {
          const int s = g % ST, rs = g % RS;
          mbar_wait(&B.raw_full[rs], (g / RS) & 1);
          if (g >= ST) mbar_wait(&B.kv_empty[s], ((g / ST) - 1) & 1);
          const uint32_t src = su + O_R + rs * 2 * TKR, dst = su + O_ST + s * STB;
          cvt_rows(src, dst, BN, lane);
          cvt_rows(src + TKR, dst + TK, BN, lane);
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) { mbar_arrive(&B.kv_full[s]); mbar_arrive(&B.raw_empty[rs]); }
        }
      }
    }
    // ------------------------------------------------------------------------------------ P16: V of every stage bf16 -> fp16 in place
    // (element-wise, so the swizzle does not matter); the stage is released only after PV(G), which waits for vc_full
    if (VCONV) {
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        Pc P; item_of(li, P);
        for (int b = 0; b < P.nb; ++b, ++g) {
          const int s = g % ST;
          mbar_wait(&B.kv_full[s], (g / ST) & 1);
          const uint32_t vb = su + O_ST + s * STB + TK;
#pragma unroll
          for (int k = 0; k < TK / 512; ++k) {
            const uint32_t a = vb + (uint32_t)(k * 32 + lane) * 16;
            const uint4 x = lds128(a);
            sts128(a, make_uint4(bf2h2(x.x), bf2h2(x.y), bf2h2(x.z), bf2h2(x.w)));
          }
          fence_proxy_async();                                             // the MMA reads it through the async proxy
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.vc_full[s]);
        }
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------ softmax of tile m0 + 128 w, one query row per thread
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    if (PP && w == 1) named_bar_arrive(3, 256);                            // warpgroup 0 takes the first exp turn
    const uint32_t xs = su + O_X + w * XS;
    const float SC = scale * LOG2E;
    const f2 SC2 = mk2(SC, SC);
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      Pc P; item_of(li, P);
      const int n = P.n, h = P.h, m0 = P.m0, sq = P.sq, nb = P.nb;
      float l_i = 0.f;
      for (int b = 0; b < nb; ++b, ++G) {
        EV(0);
        mbar_wait(&B.s_full[w][G & 1], (G >> 1) & 1);                      // S(G) ready; PV(G - 2) done: P buffer G & 1 free
        tc_fence_after();
        uint32_t v[BN];
#pragma unroll
        for (int q = 0; q < BN / 32; ++q) tmem_ld32(trow + t_s(w, G & 1) + 32 * q, *reinterpret_cast<uint32_t(*)[32]>(v + 32 * q));
        tmem_wait_ld();
        EV(1);
        if (const int kv = sq - (P.b0 + b) * BN; kv < BN) {                         // the last block's keys past seqused: -inf (P = 0)
#pragma unroll
          for (int j = 0; j < BN; ++j) v[j] = j < kv ? v[j] : 0xff800000u;
        }
        tc_fence_after();                                                  // keeps the exp turn's arithmetic after it (without: +4 - 6 %)
        EV(3);
        if (PP) named_bar_sync(3 + w, 256);                                // my exp turn: the other warpgroup's exps are done
        EV(4);
        f2 ssp[4] = {mk2(0.f, 0.f), mk2(0.f, 0.f), mk2(0.f, 0.f), mk2(0.f, 0.f)};
#pragma unroll
        for (int cc = 0; cc < BN / 32; ++cc) {                              // 32 keys -> 16 packed P columns
          uint32_t pk[16];
#pragma unroll
          for (int k = 0; k < 16; ++k) {
            const f2 e = mul2(mk2u(v[cc * 32 + 2 * k], v[cc * 32 + 2 * k + 1]), SC2);
            if (NOEXP) { pk[k] = pack_bf16(lo2(e) * 1e-3f, hi2(e) * 1e-3f); continue; }
            if (EXH) {                                                     // one MUFU op per pair: ex2.approx.f16x2 (P16 only)
              const uint32_t xh = f2h2(lo2(e), hi2(e));
              uint32_t ph;
              asm("ex2.approx.f16x2 %0, %1;" : "=r"(ph) : "r"(xh));
              pk[k] = ph;
              ssp[k & 3] = add2(ssp[k & 3], mk2(h2f2(ph).x, h2f2(ph).y));
              continue;
            }
            float p0, p1;
            if (FPCNT > 0 && (k % FPMOD) < FPCNT) {                        // FMA pipe; masked (-inf) entries 0
              const float x0 = lo2(e), x1 = hi2(e);
              const f2 pe = fex2_poly2(mk2(fmaxf(x0, -126.f), fmaxf(x1, -126.f)));
              p0 = x0 < -126.f ? 0.f : lo2(pe);
              p1 = x1 < -126.f ? 0.f : hi2(pe);
            } else {
              p0 = ex2f(lo2(e)); p1 = ex2f(hi2(e));
            }
            pk[k] = P16 ? f2h2(p0, p1) : pack_bf16(p0, p1);
            ssp[k & 3] = add2(ssp[k & 3], LSUMR ? (P16 ? mk2(h2f2(pk[k]).x, h2f2(pk[k]).y) : mk2(bf16lo(pk[k]), bf16hi(pk[k]))) : mk2(p0, p1));   // LSUMR: the sum of the P the MMA reads
          }
          tmem_st16(trow + t_p(w, G & 1) + cc * 16, pk);
        }
        EV(5);
        if (PP) named_bar_arrive(4 - w, 256);                              // hand the MUFU to the other warpgroup
        const f2 ss = add2(add2(ssp[0], ssp[1]), add2(ssp[2], ssp[3]));
        l_i += lo2(ss) + hi2(ss);
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.p_full[w][G & 1]);
        EV(6);
        if (b == nb - 1) {
          // ---- epilogue of the item: O = acc / l (bf16; 0 on the rows past seqused), LSE = ln l
          mbar_wait(&B.o_done[w], li & 1);
          tc_fence_after();
          uint32_t ov[DH];
          tmem_ld32(trow + t_o(w), ov);
          tmem_wait_ld();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.o_free[w]);
          const int row = m0 + w * QM, i = row + (int)r;
          const bool live = row < S && P.st;
          if (CSPLIT) {
            // ---- reduce-scatter: row r -> rank r / RO, slot (my rank, w); chunk-major so a warp's lanes (consecutive rows) write one
            // contiguous 512-B (CS <= 4) run per instruction; every rank sends every row (the owner's byte count is fixed)
            const uint32_t RO = (uint32_t)QM / csn, own = r / RO, rr = r - own * RO;
            const uint32_t sl = (csr * 2u + (uint32_t)w) * RO;                           // slot row base: (src, w)
            const uint32_t rb = mapa_u32(smem_u32(&B.rcv), own);
            const uint32_t dd = mapa_u32(su + O_RCV + (sl * 8u) * 16u + rr * 16u, own);
#pragma unroll
            for (int q = 0; q < 8; ++q) st_async4(dd + (uint32_t)q * RO * 16u, ov[4 * q], ov[4 * q + 1], ov[4 * q + 2], ov[4 * q + 3], rb);
            st_async1(mapa_u32(su + O_RCV + RCV_D + (sl + rr) * 4u, own), __float_as_uint(l_i), rb);
            if (own != csr) continue;                                      // not my row: its owner stores it
            mbar_wait_cl(&B.rcv, 0);                                       // every rank's rows of mine (both warpgroups) have landed
#pragma unroll
            for (int k = 0; k < DH; ++k) ov[k] = 0u;
            l_i = 0.f;
            for (uint32_t s = 0; s < csn; ++s) {                           // rank order: deterministic
              const uint32_t so = (s * 2u + (uint32_t)w) * RO;
#pragma unroll
              for (int q = 0; q < 8; ++q) {
                const uint4 x = lds128(su + O_RCV + ((so * 8u) + (uint32_t)q * RO) * 16u + rr * 16u);
                ov[4 * q] = __float_as_uint(__uint_as_float(ov[4 * q]) + __uint_as_float(x.x));
                ov[4 * q + 1] = __float_as_uint(__uint_as_float(ov[4 * q + 1]) + __uint_as_float(x.y));
                ov[4 * q + 2] = __float_as_uint(__uint_as_float(ov[4 * q + 2]) + __uint_as_float(x.z));
                ov[4 * q + 3] = __float_as_uint(__uint_as_float(ov[4 * q + 3]) + __uint_as_float(x.w));
              }
              l_i += __uint_as_float(lds32(su + O_RCV + RCV_D + (so + rr) * 4u));
            }
            const bool ok = i < sq && l_i > 0.f;
            const float inv = ok ? 1.f / l_i : 0.f;
            if (live) {
              if (F32IO) {
                float4* dst = reinterpret_cast<float4*>(O32 + ((size_t)n * S + i) * (H * DH) + h * DH);
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                  const uint32_t u0 = pack_bf16(__uint_as_float(ov[4 * q]) * inv, __uint_as_float(ov[4 * q + 1]) * inv);
                  const uint32_t u1 = pack_bf16(__uint_as_float(ov[4 * q + 2]) * inv, __uint_as_float(ov[4 * q + 3]) * inv);
                  dst[q] = make_float4(__uint_as_float(u0 << 16), __uint_as_float(u0 & 0xffff0000u), __uint_as_float(u1 << 16),
                                       __uint_as_float(u1 & 0xffff0000u));
                }
              } else {                                                     // bf16 O (O32 carries its pointer): 64 B a row
                uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<uint16_t*>(O32) + ((size_t)n * S + i) * (H * DH) + h * DH);
#pragma unroll
                for (int q = 0; q < 4; ++q)
                  dst[q] = make_uint4(pack_bf16(__uint_as_float(ov[8 * q]) * inv, __uint_as_float(ov[8 * q + 1]) * inv),
                                      pack_bf16(__uint_as_float(ov[8 * q + 2]) * inv, __uint_as_float(ov[8 * q + 3]) * inv),
                                      pack_bf16(__uint_as_float(ov[8 * q + 4]) * inv, __uint_as_float(ov[8 * q + 5]) * inv),
                                      pack_bf16(__uint_as_float(ov[8 * q + 6]) * inv, __uint_as_float(ov[8 * q + 7]) * inv));
              }
              if (LSE) LSE[((size_t)n * H + h) * S + i] = ok ? __log2f(l_i) * lmul : 0.f;
            }
            continue;
          }
          if (live && P.c0 < P.c1) {
            // ---- a piece of a split item: partial O / l to WS; the piece completing the item's count adds all in CTA order
            float* wo = WS + ((size_t)(cid * 2 + (li == 0 ? 0 : 1)) * 2 + w) * WSF;   // slot: the first piece of my range, or its last
#pragma unroll
            for (int q = 0; q < 8; ++q)
              __stcg(reinterpret_cast<float4*>(wo + r * DH + 4 * q), make_float4(__uint_as_float(ov[4 * q]), __uint_as_float(ov[4 * q + 1]),
                                                                              __uint_as_float(ov[4 * q + 2]), __uint_as_float(ov[4 * q + 3])));
            __stcg(wo + QM * DH + r, l_i);
            __threadfence();
            named_bar_sync(1 + w, 128);
            if (r == 0) B.last[w] = atomicAdd(CNT + 2 * P.it + w, 1) == P.c1 - P.c0;
            named_bar_sync(1 + w, 128);
            if (!*reinterpret_cast<volatile int*>(&B.last[w])) continue;     // not the last piece: nothing to store
            __threadfence();
            // every piece's partial from WS, this one's included (written above): ov / l_i are free accumulators, CTA order
#if CMBS
            {                                                              // own piece: registers -> its slot (no L2 round trip)
              const uint32_t co = su + O_CMB + (uint32_t)w * CMB_W + CMBS * CMB_P;
#pragma unroll
              for (int q = 0; q < 8; ++q) sts128(co + (uint32_t)r * CMB_ROW + 16u * q, make_uint4(ov[4 * q], ov[4 * q + 1], ov[4 * q + 2], ov[4 * q + 3]));
              sts32(co + QM * CMB_ROW + (uint32_t)r * 4u, __float_as_uint(l_i));
            }
#endif
#pragma unroll
            for (int k = 0; k < DH; ++k) ov[k] = 0u;
            l_i = 0.f;
#if CMBS
            {                                                                // N7: the pieces' partials staged in shared memory (cp.async, all FOREIGN pieces of a group in
              const uint32_t cb = su + O_CMB + (uint32_t)w * CMB_W;           // flight; this CTA's own piece went from its registers to the own slot), then added in
              const uint32_t co = cb + CMBS * CMB_P;                          // CTA order: the same fp32 adds -- bitwise the same O / LSE
              for (int cc0 = P.c0; cc0 <= P.c1; cc0 += CMBS + 1) {
                const int ne = min(CMBS + 1, P.c1 - cc0 + 1);
                int sj = 0;
#pragma unroll
                for (int j = 0; j < CMBS + 1; ++j) {
                  const int cc = cc0 + j;
                  if (j < ne && cc != cid) {
                    const float* src = WS + ((size_t)(cc * 2 + (start_of(cc) >= P.ist ? 0 : 1)) * 2 + w) * WSF;
                    const uint32_t d = cb + (uint32_t)sj * CMB_P + (uint32_t)r * CMB_ROW;
#pragma unroll
                    for (int q = 0; q < 8; ++q) cp_async16(d + 16u * q, src + r * DH + 4 * q);
                    if (r < 32) cp_async16(cb + (uint32_t)sj * CMB_P + QM * CMB_ROW + (uint32_t)r * 16u, src + QM * DH + 4 * r);
                    ++sj;
                  }
                }
                cp_async_wait_all();
                named_bar_sync(1 + w, 128);                                  // the l chunks were copied by other threads
                sj = 0;
                for (int j = 0; j < CMBS + 1; ++j) {
                  const int cc = cc0 + j;
                  if (j < ne) {
                    const uint32_t pb = cc == cid ? co : cb + (uint32_t)(sj++) * CMB_P;
#pragma unroll
                    for (int q = 0; q < 8; ++q) {
                      const uint4 x = lds128(pb + (uint32_t)r * CMB_ROW + 16u * q);
                      ov[4 * q] = __float_as_uint(__uint_as_float(ov[4 * q]) + __uint_as_float(x.x)); ov[4 * q + 1] = __float_as_uint(__uint_as_float(ov[4 * q + 1]) + __uint_as_float(x.y));
                      ov[4 * q + 2] = __float_as_uint(__uint_as_float(ov[4 * q + 2]) + __uint_as_float(x.z)); ov[4 * q + 3] = __float_as_uint(__uint_as_float(ov[4 * q + 3]) + __uint_as_float(x.w));
                    }
                    l_i += __uint_as_float(lds32(pb + QM * CMB_ROW + (uint32_t)r * 4u));
                  }
                }
                named_bar_sync(1 + w, 128);                                  // the staging is free again
              }
            }
#else
            for (int cc = P.c0; cc <= P.c1; ++cc) {
              const float* src = WS + ((size_t)(cc * 2 + (start_of(cc) >= P.ist ? 0 : 1)) * 2 + w) * WSF;
#pragma unroll
              for (int q = 0; q < 8; ++q) {
                const float4 x = __ldcg(reinterpret_cast<const float4*>(src + r * DH + 4 * q));
                ov[4 * q] = __float_as_uint(__uint_as_float(ov[4 * q]) + x.x); ov[4 * q + 1] = __float_as_uint(__uint_as_float(ov[4 * q + 1]) + x.y);
                ov[4 * q + 2] = __float_as_uint(__uint_as_float(ov[4 * q + 2]) + x.z); ov[4 * q + 3] = __float_as_uint(__uint_as_float(ov[4 * q + 3]) + x.w);
              }
              l_i += __ldcg(src + QM * DH + r);
            }
#endif
            if (r == 0) CNT[2 * P.it + w] = 0;                              // every piece has counted: ready for the next launch
          }
          const bool ok = i < sq && l_i > 0.f;
          const float inv = ok ? 1.f / l_i : 0.f;
          if (F32IO) {                                                     // fp32 O holding the bf16-rounded values (as O16.float())
            if (live) {
              float4* dst = reinterpret_cast<float4*>(O32 + ((size_t)n * S + i) * (H * DH) + h * DH);
#pragma unroll
              for (int q = 0; q < 8; ++q) {
                const uint32_t u0 = pack_bf16(__uint_as_float(ov[4 * q]) * inv, __uint_as_float(ov[4 * q + 1]) * inv);
                const uint32_t u1 = pack_bf16(__uint_as_float(ov[4 * q + 2]) * inv, __uint_as_float(ov[4 * q + 3]) * inv);
                dst[q] = make_float4(__uint_as_float(u0 << 16), __uint_as_float(u0 & 0xffff0000u), __uint_as_float(u1 << 16),
                                     __uint_as_float(u1 & 0xffff0000u));
              }
              if (LSE) LSE[((size_t)n * H + h) * S + i] = ok ? __log2f(l_i) * lmul : 0.f;
            }
            continue;
          }
          if (r == 0) tma_store_wait_read0();                             // the previous item's store has left the staging buffer
          named_bar_sync(1 + w, 128);
#pragma unroll
          for (int q = 0; q < 4; ++q)
            sts128(xs + sw64(r, q), make_uint4(pack_bf16(__uint_as_float(ov[8 * q]) * inv, __uint_as_float(ov[8 * q + 1]) * inv),
                                               pack_bf16(__uint_as_float(ov[8 * q + 2]) * inv, __uint_as_float(ov[8 * q + 3]) * inv),
                                               pack_bf16(__uint_as_float(ov[8 * q + 4]) * inv, __uint_as_float(ov[8 * q + 5]) * inv),
                                               pack_bf16(__uint_as_float(ov[8 * q + 6]) * inv, __uint_as_float(ov[8 * q + 7]) * inv)));
          fence_proxy_async();
          named_bar_sync(1 + w, 128);
          if (r == 0 && live) { tma_store_2d(&mo, xs, h * DH, n * S + row); tma_store_commit(); }
          if (LSE && live) LSE[((size_t)n * H + h) * S + i] = ok ? __log2f(l_i) * lmul : 0.f;
        }
      }
    }
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (CL > 1 || CSPLIT) cluster_sync();                                    // CSPLIT: no rank exits while its peers may still write it
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
