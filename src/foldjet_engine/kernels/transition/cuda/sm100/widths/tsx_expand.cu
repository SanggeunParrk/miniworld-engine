// tsx_expand.cu -- the Transition forward at SMALL M (a few 128-row tiles: the single track [B, L, D] of a pairformer), second of
// three programmatic-dependent kernels (tsx_ln.cu -> this -> tsx_squeeze.cu): expand + SwiGLU (-DDIM=<D> -DHID=<H> -DCH=<hidden
// units per CTA, 32 / 64 / 128> -DF32=<0 | 1> -DSK=<K split = cluster size>), sm_100a, bf16 (kind::f16) or fp32 on the TF32 tensor cores.
// One cluster of SK CTAs per (128-row tile, CH-unit hidden chunk j): tiles x H / CH x SK CTAs, sized so that a [1, 768, 384] bf16 call
// fills the 148 SMs (6 x 24 at n = 2 with CH = 32, 6 x 24 at n = 4 with CH = 64; SK = 1). Each CTA TMA-loads its K slice of the tile's
// xn (written by tsx_ln.cu) and of its chunk of Wa and Wb (CH rows each), K-block by K-block (128 bytes of K per row: 64 bf16 / 32 fp32
// elements) with a barrier each so the products start as the data lands, runs [a | b] = xn [Wa_j; Wb_j]^T (M 128, N 2 CH) into tensor
// memory, and writes h = silu(a) b (and, with save, a and b for the backward). The K-blocks go through a ring where the whole K does not
// fit shared memory (D 768, fp32).
// SK > 1 (round 3): the per-CTA operand intake bounds this kernel at a few tiles (every CTA takes the tile's whole xn beside its weight
// chunk: 590 KB at D 768 fp32 with CH 32), so CTA k of a cluster takes K slice k and the fp32 partials of [a | b] meet in distributed
// shared memory as tsx_squeeze.cu's: CTA o owns units [o CH / SK, (o + 1) CH / SK) of the chunk, every CTA stores the other owners'
// columns into their shared memory (st.async completing on the owner's barrier) and each owner adds the SK partials in rank order
// (deterministic) before the SwiGLU. No cross-cluster reduction: h goes through L2 (M x H) to the squeeze.
// The weight K-blocks are loaded before griddepcontrol.wait when they are parameters or an earlier launch's cast (wpre = 1); when the
// LayerNorm launch this one follows writes them (the training forward's fused cast, wpre = 0), after it.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include "tsx_store.cuh"
using namespace s100;

#ifndef DIM
#define DIM 384
#endif
#ifndef HID
#define HID (2 * DIM)
#endif
#ifndef CH
#define CH 64
#endif
#ifndef F32
#define F32 0
#endif
#ifndef SK
#define SK 1
#endif
#ifndef H16
#define H16 0                                                   // round 3 part 4: fp32 activations on the fp16 tensor cores (inference)
#endif
static_assert(!(H16 && F32), "H16 is its own operand type");
#ifndef FOLD
#define FOLD 0                                                  // N1: the LayerNorm folded into the expand (bf16 inference, SK 1): raw x as A, W gamma as B, K16 correction MMA, statistics in the epilogue warps
#endif
#ifndef CHM
#define CHM 1                                                   // part 17: partials received chunk-major (512-B runs per warp store); 0: row-major swizzled
#endif
#if CHM
#define RXO(i) chm16((uint32_t)(i), (uint32_t)r)
#else
#define RXO(i) swz16((uint32_t)(r * (W_ / 2) + (i)))
#endif
#ifndef SIGNEW
#define SIGNEW (!F32 && !H16)                                                // N9: the epilogue's SwiGLU on packed pairs, the sigmoid's MUFU share moved to the FMA pipe (sigmoid_z2 modes per pair position)
#endif
#ifndef SIGPAT
#define SIGPAT 0xFF                                             // 2 bits per pair position (mod 4): 0 kit (2 MUFU an element), 1 Newton reciprocal, 2 polynomial exponential (both 1 MUFU, FMA pipe: A/B slower or equal), 3 one-MUFU tanh sigmoid (default: bf16 builds; the TR_SIG 1 / SIGT 1 form)
#endif
template <bool SC> DEVI f2 swiglu_pair(f2 a, f2 b, f2 cz, f2 ca, f2 ch, f2 cb, int mode) {
  if (SIGPAT == 0xFF) {
    // all pairs on the tanh sigmoid: ah = a sc / 2 is the tanh argument, a sigmoid(a sc) = ah (1 + tanh(ah)): three packed ops and the MUFU pair (cb = the scale of b)
    const f2 ah = mul2(a, ch);
    const f2 u = fma2(ah, mk2(tanhf_approx(lo2(ah)), tanhf_approx(hi2(ah))), ah);
    return mul2(u, SC ? mul2(b, cb) : b);
  }
  const f2 s = sigmoid_z2(mul2(a, cz), mode);
  return mul2(mul2(SC ? mul2(a, ca) : a, s), b);
}
constexpr int ESZ = F32 ? 4 : 2,
 BKE = 128 / ESZ;              // element bytes, elements of K per 128-byte K-block row
constexpr int D_ = DIM, H_ = HID, ROWS = 128, KD = D_ / SK, NKB = KD / BKE, NCHK = H_ / CH, HC = CH / 2;
static_assert((CH == 32 || CH == 64 || (CH == 128 && SK == 1)) && H_ % CH == 0 && D_ % (BKE * SK) == 0, "32- / 64- / 128-unit chunks, whole K-blocks per CTA");
constexpr int W_ = CH / SK;                                    // units of the chunk each CTA of a cluster owns
static_assert(SK == 1 || W_ == 16 || W_ == 32, "owner column groups of 16 or 32 units");
constexpr int KBT = ROWS * 128;                                // one [128 rows] K-block of xn, 16 KB
constexpr int WKB = 2 * CH * 128;                              // one K-block of [Wa_j; Wb_j]: 2 CH rows x 128 B
constexpr int RED = SK > 1 ? SK * ROWS * 2 * W_ * 4 : 0;       // partials received from the other ranks: [SK][128][a W | b W] fp32
constexpr int ZP = 2 * CH * 16 > 2048 ? 2 * CH * 16 : 2048;       // FOLD: the zero plane covers the B operand's 2 CH / 8 row groups (128 B each)
constexpr int FEX = FOLD ? 2 * CH * 16 + 2048 + ZP + 512 + 2048 : 0;   // FOLD: B correction plane, A plane, zero plane, rstd, statistics exchange
constexpr int CAP0 = (232448 - 1024 - RED - FEX) / (KBT + WKB);
// XD (FOLD where the K-blocks of x + weights do not fit as one ring, 128-unit chunks): the whole x tile stays resident (its K-blocks have their own barriers), the weights have a ring of their own
constexpr bool XD = FOLD && NKB > CAP0;
constexpr int CAPD = (232448 - 1024 - RED - FEX - NKB * KBT) / WKB;
constexpr int CAP = XD ? CAPD : CAP0;
constexpr int STG = NKB < CAP ? NKB : CAP;                     // ring stages (the whole K where it fits)
constexpr int XSTG = XD ? NKB : STG;                           // K-blocks of x in shared memory
constexpr int XTXV = XD ? 0 : KBT;                             // bytes of x a weight-stage barrier expects
constexpr int O_X = 0, O_W = XSTG * KBT, O_RED = O_W + STG * WKB, O_BP = O_RED + RED, O_AP = O_BP + 2 * CH * 16, O_AZ = O_AP + 2048,
              O_RS = O_AZ + ZP, O_EX = O_RS + 512, O_BAR = O_BP + FEX, SMEM_BYTES = O_BAR + 256;
static_assert(!FOLD || (SK == 1 && !F32 && !H16 && (XD || NKB <= STG) && NKB % 2 == 0 && NKB <= 8), "the fold: bf16, unsplit K, the whole x tile in shared memory");
static_assert(SMEM_BYTES <= 232448 && STG >= 2, "shared memory budget");
constexpr uint32_t TCOLS = 2 * CH;
#if F32
typedef float ACT;
// kind::tf32 instruction descriptor: TF32 A / B (format 2), fp32 D, K-major both, N, M
constexpr uint32_t IDESC = (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)((2 * CH) >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
DEVI void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC), "r"(acc) : "memory");
}
DEVI float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
#elif H16
// -DH16=1: xn (fp16, LN(x) / s_x, tsx_ln.cu -DH16) and Wa16 / Wb16 (/ s_w) on kind::f16 with fp16 A / B (format 0), fp32 D; the epilogue
// rescales [a | b] by sc[1] = s_x s_w and stores h = fp16(silu(a) b sc[2]) (sc[2] = 1 / s_h): fused_generic_sm100a._h16_forms' scales
typedef __half ACT;
constexpr uint32_t IDESC = (1u << 4) | ((uint32_t)((2 * CH) >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
DEVI void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) { umma_ss(d, a, b, IDESC, acc); }
#else
typedef __nv_bfloat16 ACT;
constexpr uint32_t IDESC = idesc_bf16(128, 2 * CH);
DEVI void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) { umma_ss(d, a, b, IDESC, acc); }
#endif

struct Bars {
  uint64_t full[STG], empty[STG], mma_done, red_full, aug_b_full, aug_a_full, xfull[8];
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 256, "barrier block");

template <int N> DEVI void tld(uint32_t taddr, uint32_t (&r)[N]) {
  if constexpr (N == 32) tmem_ld32(taddr, r); else tmem_ld16(taddr, r);
}

// no-swizzle K-major UMMA descriptor: core matrices of 8 rows x 16 B, LBO = distance of the K-adjacent core matrix, SBO = distance of the next 8-row group
DEVI uint64_t desc_none(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFFu) << 32) | ((uint64_t)1 << 46);
}
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

// NV values of a row (fp32 bit patterns) -> the activation dtype's words (bf16: packed pairs; fp32: as they are, `round` to the TF32 grid)
template <int NV> DEVI void pack_act(const uint32_t (&v)[NV], uint32_t (&o)[F32 ? NV : NV / 2], bool round) {
#if F32
#pragma unroll
  for (int k = 0; k < NV; ++k) o[k] = round ? __float_as_uint(tf32r(__uint_as_float(v[k]))) : v[k];
#elif H16
#pragma unroll
  for (int k = 0; k < NV / 2; ++k) o[k] = f2h2(__uint_as_float(v[2 * k]), __uint_as_float(v[2 * k + 1]));
#else
#pragma unroll
  for (int k = 0; k < NV / 2; ++k) o[k] = pack_bf16(__uint_as_float(v[2 * k]), __uint_as_float(v[2 * k + 1]));
#endif
}

// maps: xn [M][D] (box BKE x 128); wa, wb [H][D] (box BKE x CH). h [M][H]; a / b with row stride ldab, b at element boff of bs
// ([M][H] each: ldab = H, boff = 0; one packed [M][a | b]: ldab = 2 H, boff = H), in the activation dtype.
extern "C" __global__ void __launch_bounds__(384, 1)
transition_sx_expand(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mwa,
                     const __grid_constant__ CUtensorMap mwb, ACT* __restrict__ hs, ACT* __restrict__ as_,
                     ACT* __restrict__ bs, int save, int wpre, int ldab, int boff
#if H16
                     , const float* __restrict__ sc
#endif
#if FOLD
                     , const uint8_t* __restrict__ aug, float eps
#endif
                     TSX_STAMP_ARG
                     ) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cl = blockIdx.x / SK, rank = SK > 1 ? (int)cluster_rank() : 0;
  const int tile = cl / NCHK, j = cl % NCHK, row0 = tile * ROWS, h0 = j * CH, k0 = rank * KD;
  if (tid == 0) TSX_ST(0);

  if (tid == 0) {
    for (int s = 0; s < STG; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.mma_done, 1);
#if FOLD
    mbar_init(&B.aug_b_full, 1); mbar_init(&B.aug_a_full, 1);
    for (int g = 0; g < 8; ++g) mbar_init(&B.xfull[g], 1);
#endif
    if (SK > 1) { mbar_init(&B.red_full, 1); mbar_expect_tx(&B.red_full, (SK - 1) * ROWS * 2 * W_ * 4); }   // the other ranks' partials
    fence_barrier_init();
    fence_proxy_async();
    prefetch_map(&mxn); prefetch_map(&mwa); prefetch_map(&mwb);
#if FOLD
    mbar_expect_tx(&B.aug_b_full, 2 * CH * 16);                // the u / v correction rows of this chunk's Wa and Wb units (a prep form, cached with the weights)
    bulk_g2s(su + O_BP, aug + (size_t)h0 * 16, CH * 16, &B.aug_b_full);
    bulk_g2s(su + O_BP + CH * 16, aug + (size_t)(H_ + h0) * 16, CH * 16, &B.aug_b_full);
#endif
    if (wpre)
      for (int kb = 0; kb < STG; ++kb) {                        // weights first: they do not depend on the LayerNorm kernel
        mbar_expect_tx(&B.full[kb], XTXV + WKB);
        tma_load_2d(su + O_W + kb * WKB, &mwa, &B.full[kb], k0 + kb * BKE, h0);
        tma_load_2d(su + O_W + kb * WKB + CH * 128, &mwb, &B.full[kb], k0 + kb * BKE, h0);
      }
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TCOLS); tmem_relinquish(); }
#if FOLD
  if (tid < ZP / 16) *reinterpret_cast<uint4*>(sm + O_AZ + tid * 16) = make_uint4(0, 0, 0, 0);   // the zero plane (K 8 .. 15) of both operands
  fence_proxy_async();
#endif
  tc_fence_before();
  __syncthreads();
  if (SK > 1) cluster_sync_relaxed();                          // every CTA of the cluster runs before any distributed-shared store
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_wait();                                                  // xn is complete and visible
  pdl_launch();                                                // the squeeze may start its prologue
  if (tid == 0) TSX_ST(1);

  if (warp == 0) {
    if (lane == 0 && XD)                                       // the x tile first: every K-block on its own barrier
      for (int kb = 0; kb < NKB; ++kb) {
        mbar_expect_tx(&B.xfull[kb], KBT);
        tma_load_2d(su + O_X + kb * KBT, &mxn, &B.xfull[kb], k0 + kb * BKE, row0);
      }
    if (lane == 0)
      for (int kb = 0; kb < NKB; ++kb) {
        const int s = kb % STG;
        if (kb >= STG) mbar_wait(&B.empty[s], ((kb / STG) - 1) & 1);
        if (!wpre || kb >= STG) {                              // (wpre 0: the LayerNorm launch wrote the weights, complete only now)
          mbar_expect_tx(&B.full[s], XTXV + WKB);
          tma_load_2d(su + O_W + s * WKB, &mwa, &B.full[s], k0 + kb * BKE, h0);
          tma_load_2d(su + O_W + s * WKB + CH * 128, &mwb, &B.full[s], k0 + kb * BKE, h0);
        }
        if (!XD) tma_load_2d(su + O_X + s * KBT, &mxn, &B.full[s], k0 + kb * BKE, row0);
      }
    if (lane == 0) TSX_ST(2);
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------ MMA issue (converged warp)
#pragma unroll 1
    for (int kb = 0; kb < NKB; ++kb) {
      const int s = kb % STG;
      mbar_wait(&B.full[s], (kb / STG) & 1);
      if (XD) mbar_wait(&B.xfull[kb], 0);
      if (lane == 0 && kb == 0) TSX_ST(3);
      if (lane == 0 && kb == NKB - 1) TSX_ST(4);
      tc_fence_after();
      const uint64_t da = desc_k128(su + O_X + (XD ? kb : s) * KBT), db = desc_k128(su + O_W + s * WKB);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) mma(tmem, da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), (kb > 0 || ks > 0) ? 1u : 0u);
        if (NKB > STG) tc_commit(&B.empty[s]);
        if (!FOLD && kb == NKB - 1) tc_commit(&B.mma_done);
      }
      __syncwarp();
    }
#if FOLD
    // the rank-1 LayerNorm terms (-mean u + q v) of every unit: one K16 MMA accumulated into [a | b] once the statistics are in
    mbar_wait(&B.aug_b_full, 0);
    mbar_wait(&B.aug_a_full, 0);
    tc_fence_after();
    if (lane == 0) TSX_ST(10);
    if (elect_one()) {
      umma_ss(tmem, desc_none(su + O_AP, O_AZ - O_AP, 128), desc_none(su + O_BP, O_AZ - O_BP, 128), IDESC, 1u);
      tc_commit(&B.mma_done);
    }
    __syncwarp();
#endif
  } else if (warp >= 4) {
#if H16
    const float s1 = __ldg(sc + 1), s2 = __ldg(sc + 2);
#endif
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
#if SK == 1
    // ------------------------------------------------------------------------------------------ SwiGLU (one row per thread, warps
    // 4-7 the chunk's first CH / 2 units, 8-11 the second)
    const int g = (warp - 4) >> 2;
    constexpr int EC = HC > 32 ? 32 : HC;                      // units per step (CH 128: two steps of 32)
    constexpr int NW = F32 ? EC : EC / 2;                      // 32-bit words of a thread's row segment
#if FOLD
    // statistics of row r from the raw x tile in shared memory as its K-blocks land: shifted single pass (shift = the row's first element: sum (x - s),
    // sum (x - s)^2), the two halves of the K-blocks on warps 4-7 / 8-11, exchanged through shared memory; the first half also writes the A plane
    // [-mean_hi -mean_hi -mean_lo q_hi q_hi q_lo 0 0] (q = 1 / rstd) and rstd
    float rs;
    {
      const uint32_t xb = su + O_X;
      mbar_wait(XD ? &B.xfull[0] : &B.full[0], 0);
      const uint4 t0 = lds128(xb + sw128(r, 0));
      const float shift = __uint_as_float(t0.x << 16);
      const f2 nsh = mk2(-shift, -shift);
      f2 sa = mk2(0.f, 0.f), qa = mk2(0.f, 0.f);
#pragma unroll
      for (int kk = 0; kk < NKB / 2; ++kk) {
        const int kb = g * (NKB / 2) + kk;
        mbar_wait(XD ? &B.xfull[kb] : &B.full[kb], 0);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 t = lds128(xb + kb * KBT + sw128(r, q));
          const uint32_t w4[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const f2 d = add2(mk2u(w4[k] << 16, w4[k] & 0xffff0000u), nsh);
            sa = add2(sa, d);
            qa = fma2(d, d, qa);
          }
        }
      }
      float* ex = reinterpret_cast<float*>(sm + O_EX);
      ex[(g * ROWS + r) * 2] = lo2(sa) + hi2(sa);
      ex[(g * ROWS + r) * 2 + 1] = lo2(qa) + hi2(qa);
      named_bar_sync(2, 256);
      if (tid == 128) TSX_ST(11);
      const float sd = ex[r * 2] + ex[(ROWS + r) * 2], sq = ex[r * 2 + 1] + ex[(ROWS + r) * 2 + 1];
      const float md = sd * (1.f / D_), mean = shift + md;
      const float var = fmaxf(sq * (1.f / D_) - md * md, 0.f);
      rs = rsqrtf(var + eps);
      if (g == 0) {
        reinterpret_cast<float*>(sm + O_RS)[r] = rs;
        const float q = 1.f / rs;
        const __nv_bfloat16 mh = __float2bfloat16(-mean), qh = __float2bfloat16(q);
        const __nv_bfloat16 ml = __float2bfloat16(-mean - __bfloat162float(mh)), ql = __float2bfloat16(q - __bfloat162float(qh));
        const uint32_t e0 = *reinterpret_cast<const uint16_t*>(&mh), e2 = *reinterpret_cast<const uint16_t*>(&ml),
                       e3 = *reinterpret_cast<const uint16_t*>(&qh), e5 = *reinterpret_cast<const uint16_t*>(&ql);
        *reinterpret_cast<uint4*>(sm + O_AP + r * 16) = make_uint4(e0 | (e0 << 16), e2 | (e3 << 16), e3 | (e5 << 16), 0u);
      }
      fence_proxy_async();
      named_bar_sync(3, 256);
      if (tid == 128) TSX_ST(9);
      if (tid == 128) mbar_arrive(&B.aug_a_full);
    }
#endif
    mbar_wait(&B.mma_done, 0);
    if (warp == 4 && lane == 0) TSX_ST(5);
    tc_fence_after();
#pragma unroll
    for (int cc = 0; cc < HC; cc += EC) {
      uint32_t av[EC], bv[EC], hv[EC], hp[NW], ap[NW], bp[NW];
      tld<EC>(trow + g * HC + cc, av);
      tld<EC>(trow + CH + g * HC + cc, bv);
      tmem_wait_ld();
#pragma unroll
#if SIGNEW
      for (int k = 0; k < EC; k += 2) {
        const int pm = (SIGPAT >> (2 * (((cc + k) >> 1) & 3))) & 3;
#if H16
        const f2 hh = swiglu_pair<true>(mk2u(av[k], av[k + 1]), mk2u(bv[k], bv[k + 1]), mk2(-1.4426950408889634f * s1, -1.4426950408889634f * s1), mk2(s1 * s1 * s2, s1 * s1 * s2), mk2(0.5f * s1, 0.5f * s1), mk2(s1 * s2, s1 * s2), pm);
#elif FOLD
        const f2 hh = swiglu_pair<true>(mk2u(av[k], av[k + 1]), mk2u(bv[k], bv[k + 1]), mk2(-1.4426950408889634f * rs, -1.4426950408889634f * rs), mk2(rs * rs, rs * rs), mk2(0.5f * rs, 0.5f * rs), mk2(rs, rs), pm);
#else
        const f2 hh = swiglu_pair<false>(mk2u(av[k], av[k + 1]), mk2u(bv[k], bv[k + 1]), mk2(-1.4426950408889634f, -1.4426950408889634f), mk2(1.f, 1.f), mk2(0.5f, 0.5f), mk2(1.f, 1.f), pm);
#endif
        hv[k] = __float_as_uint(lo2(hh)); hv[k + 1] = __float_as_uint(hi2(hh));
      }
#else
      for (int k = 0; k < EC; ++k) {
#if H16
        const float a0 = __uint_as_float(av[k]) * s1;
        hv[k] = __float_as_uint(a0 * sigmoid_kit(a0) * (__uint_as_float(bv[k]) * s1) * s2);
#else
#if FOLD
        const float a0 = __uint_as_float(av[k]) * rs;
        hv[k] = __float_as_uint(a0 * sigmoid_kit(a0) * (__uint_as_float(bv[k]) * rs));
#else
        const float a0 = __uint_as_float(av[k]);
        hv[k] = __float_as_uint(a0 * sigmoid_kit(a0) * __uint_as_float(bv[k]));
#endif
#endif
      }
#endif
      pack_act<EC>(hv, hp, true);
      const size_t toff = (size_t)row0 * H_ + h0 + g * HC + cc + store_t_off<NW>(r, lane, H_, ESZ);
      const size_t aoff = (size_t)row0 * ldab + h0 + g * HC + cc + store_t_off<NW>(r, lane, ldab, ESZ);
      store_t(hs + toff, H_, hp, lane);
      if (!H16 && save) {
        pack_act<EC>(av, ap, false);
        pack_act<EC>(bv, bp, false);
        store_t(as_ + aoff, ldab, ap, lane);
        store_t(bs + boff + aoff, ldab, bp, lane);
      }
    }
    if (warp == 4 && lane == 0) TSX_ST(7);
    tc_fence_before();
#else
    // ------------------------------------------------------------------------------------------ the K slices' partials meet in the
    // owners' shared memory ([source rank][row][a W | b W] fp32), then the SwiGLU of this CTA's W units (warps 4-7: one row per thread)
    constexpr int NW = F32 ? W_ : W_ / 2;
    if (warp < 8) {
      mbar_wait(&B.mma_done, 0);
      tc_fence_after();
      uint32_t oa[W_], ob[W_];
#pragma unroll
      for (int oo = 1; oo <= SK; ++oo) {
        const int o = (rank + oo) % SK;                        // own block last; the others staggered over the cluster
        uint32_t va[W_], vb[W_];
        tld<W_>(trow + o * W_, va);
        tld<W_>(trow + CH + o * W_, vb);
        tmem_wait_ld();
        if (o == rank) {
#pragma unroll
          for (int i = 0; i < W_; ++i) { oa[i] = va[i]; ob[i] = vb[i]; }
        } else {
          const uint32_t dst = su + O_RED + (uint32_t)(rank * ROWS * 2 * W_ * 4), bar = smem_u32(&B.red_full);
#pragma unroll
          for (int i = 0; i < W_ / 4; ++i) {
            st_async_v4(dst + RXO(i), bar, (uint32_t)o, va[4 * i], va[4 * i + 1], va[4 * i + 2], va[4 * i + 3]);
            st_async_v4(dst + RXO(W_ / 4 + i), bar, (uint32_t)o, vb[4 * i], vb[4 * i + 1], vb[4 * i + 2],
                        vb[4 * i + 3]);
          }
        }
      }
      tc_fence_before();
      mbar_wait(&B.red_full, 0);                               // every other rank's partial of these units is here
      float sa[W_], sb[W_];
#pragma unroll
      for (int i = 0; i < W_; ++i) { sa[i] = 0.f; sb[i] = 0.f; }
#pragma unroll
      for (int o = 0; o < SK; ++o) {                           // fixed order over the K slices: deterministic
        if (o == rank) {
#pragma unroll
          for (int i = 0; i < W_; ++i) { sa[i] += __uint_as_float(oa[i]); sb[i] += __uint_as_float(ob[i]); }
        } else {
          const uint32_t src = su + O_RED + (uint32_t)(o * ROWS * 2 * W_ * 4);
#pragma unroll
          for (int i = 0; i < W_ / 4; ++i) {
            const uint4 t = lds128(src + RXO(i)), u = lds128(src + RXO(W_ / 4 + i));
            sa[4 * i] += __uint_as_float(t.x); sa[4 * i + 1] += __uint_as_float(t.y);
            sa[4 * i + 2] += __uint_as_float(t.z); sa[4 * i + 3] += __uint_as_float(t.w);
            sb[4 * i] += __uint_as_float(u.x); sb[4 * i + 1] += __uint_as_float(u.y);
            sb[4 * i + 2] += __uint_as_float(u.z); sb[4 * i + 3] += __uint_as_float(u.w);
          }
        }
      }
      uint32_t av[W_], bv[W_], hv[W_], hp[NW], ap[NW], bp[NW];
#pragma unroll
#if SIGNEW
#pragma unroll
      for (int k = 0; k < W_; ++k) { av[k] = __float_as_uint(sa[k]); bv[k] = __float_as_uint(sb[k]); }
#pragma unroll
      for (int k = 0; k < W_; k += 2) {
        const int pm = (SIGPAT >> (2 * ((k >> 1) & 3))) & 3;
#if H16
        const f2 hh = swiglu_pair<true>(mk2(sa[k], sa[k + 1]), mk2(sb[k], sb[k + 1]), mk2(-1.4426950408889634f * s1, -1.4426950408889634f * s1), mk2(s1 * s1 * s2, s1 * s1 * s2), mk2(0.5f * s1, 0.5f * s1), mk2(s1 * s2, s1 * s2), pm);
#else
        const f2 hh = swiglu_pair<false>(mk2(sa[k], sa[k + 1]), mk2(sb[k], sb[k + 1]), mk2(-1.4426950408889634f, -1.4426950408889634f), mk2(1.f, 1.f), mk2(0.5f, 0.5f), mk2(1.f, 1.f), pm);
#endif
        hv[k] = __float_as_uint(lo2(hh)); hv[k + 1] = __float_as_uint(hi2(hh));
      }
#else
#pragma unroll
      for (int k = 0; k < W_; ++k) {
        av[k] = __float_as_uint(sa[k]); bv[k] = __float_as_uint(sb[k]);
#if H16
        sa[k] *= s1; sb[k] *= s1;
        hv[k] = __float_as_uint(sa[k] * sigmoid_kit(sa[k]) * sb[k] * s2);
#else
        hv[k] = __float_as_uint(sa[k] * sigmoid_kit(sa[k]) * sb[k]);
#endif
      }
#endif
      pack_act<W_>(hv, hp, true);
      const size_t toff = (size_t)row0 * H_ + h0 + rank * W_ + store_t_off<NW>(r, lane, H_, ESZ);
      const size_t aoff = (size_t)row0 * ldab + h0 + rank * W_ + store_t_off<NW>(r, lane, ldab, ESZ);
      store_t(hs + toff, H_, hp, lane);
      if (!H16 && save) {
        pack_act<W_>(av, ap, false);
        pack_act<W_>(bv, bp, false);
        store_t(as_ + aoff, ldab, ap, lane);
        store_t(bs + boff + aoff, ldab, bp, lane);
      }
    }
#endif
  }
  __syncthreads();
  if (tid == 0) TSX_ST(8);
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
