// k1w.cu -- TriMul front on B200 (tcgen05) for any width D in {64, 128, 256, 384, 512}: input LayerNorm + gated projections +
// token mask -> channel-major bf16 planes. Same math and rounding points as k1.cu (the D128 kernel):
//   xn        = bf16(fma((x - mean) * rstd, g, b))    mean / rstd: fp32 two-pass statistics (k1w_stats), rstd = rsqrt.approx
//   plane[c]  = bf16(sigmoid(g_c) * p_c)              g_c = xn Wg_c^T, p_c = xn Wp_c^T on fp32 accumulators; ONE bf16 rounding
//   Pair (i, j) is masked unless tokens i and j both are kept (the pair mask is formed here from the [L] token mask); a masked
//   pair's xn row is zero, so its plane values are exactly 0.
// planes [P, L*L]: P = 4H (bidirectional: left out | left in | right out | right in) or 2H (one direction: left | right), H the
// hidden width (D, or 2D: one direction at D64 / D128, bidirectional at D64 -- P = 512, the D128 bidirectional plane count).
//
// Mapping. Tokens on M (128-token tiles, persistent CTAs). Unlike k1.cu the x tile does not stay in shared memory (at D512 it is
// 128 KB): x streams through a ring of 64-channel K-blocks, the transform warps normalise each K-block with precomputed statistics
// and write xn straight into TENSOR MEMORY (A operand, D/2 columns per tile). Weights stream as [128 rows][64 k] K-blocks through
// a second ring. Chunk c = plane channels 64c..64c+63 is one M128 N128 K=D product (TMEM cols [0,64) gate, [64,128) proj);
// the epilogue is k1.cu's (fragment-layout TMEM loads, sigmoid gate, stmatrix.trans into [64 ch][32 tok] boxes, TMA store).
// The weights are read in place (four row-major [P/2, D] matrices W_l, W_lg, W_r, W_rg): a chunk's K-block is its 64 gate rows
// and its 64 projection rows, two [64][64] TMA boxes into the two 8 KB halves of a slot (the same SW128 image as one packed
// [128][64] box), so no per-call weight pack is needed.
//
// 512 threads: warp 0 TMA x, warp 1 MMA, warp 2 TMEM alloc, warp 3 TMA weights; warps 4-7 transform (thread = token row);
// warps 8-11 / 12-15 epilogue for even / odd chunks.
#include "sm100.cuh"
#include "tmap.h"
#include <cstdlib>
#include <cuda_fp8.h>

using namespace sm100;
#ifdef TM_TL
static __device__ unsigned long long g_tl[160][16];
#define TL(slot) { g_tl[blockIdx.x][slot] = clock64(); }
#else
#define TL(slot) {}
#endif

#ifndef F8_ABL
#define F8_ABL 128   // dev ablations (time only, results wrong): 1 no atomics, 4 no amax, 16 no qinv multiply, 32 no cvt, 64 no stmatrix; 128: the scale as a packed FMUL2 (default, -1 % k1w8)
#endif
namespace k1w8 {

constexpr int TOK = 128, KBB = TOK * 64 * 2;   // one [128][64] bf16 K-block, SW128 = 16 KB
constexpr int OSTW = 64 * 32 * 2;              // per-warp output box: [64 ch][32 tok] bf16, 64-B swizzle = 4 KB
#ifndef K1W_INSTAT
#define K1W_INSTAT 256   // widths up to this compute the LayerNorm statistics in the transform warps (the row resident in the x ring)
#endif
#ifndef K1W_NWS256
#define K1W_NWS256 6
#endif
#ifndef K1W_EARLY
#define K1W_EARLY 1   // the first ring fills are issued by thread 0 at kernel entry (one-tile latency: L128 / L256)
#endif
#ifndef K1W_PDL
#define K1W_PDL 1   // launched as a programmatic dependent: barrier init, tensor-map prefetch and the TMEM allocation overlap the previous kernel
#endif
#ifndef K1W_XKEEP
#define K1W_XKEEP 24   // x stays in L2 for k3g / the backward (no evict-first hint) while it is at most this many MB
#endif
#ifndef K1W_TRIG
#define K1W_TRIG 1   // launch_dependents at entry: the inference contraction (ctr.cu, a programmatic dependent) schedules its CTAs as k1w's retire
#endif
#ifndef K1W_SPRE
#define K1W_SPRE 1   // D > K1W_INSTAT: k1w_stats (the previous kernel, plain launch) triggers at entry and writes only mean / rstd, so k1w loads
                     // x, the weights and the affine before griddepcontrol.wait; only the transform warps wait, before they read mean / rstd
#endif
#ifndef K1W_NWS512
#define K1W_NWS512 6   // D512 weight ring slots (8: 64 KB ops of 4 K-blocks)
#endif
#ifndef K1W_NXS512
#define K1W_NXS512 4
#endif
#ifndef K1W_NXS256
#define K1W_NXS256 4
#endif
constexpr int NTHREADS = 512;
constexpr float LOG2E = 1.4426950408889634f;
constexpr uint32_t IDESC = idesc_bf16(128, 128);

// FOLD (inference, D256 / D384): LN_in folded into the weights, W' = bf16(W o gamma_in). The raw x K-block is copied into TMEM by the
// MMA thread (tcgen05.cp) as it lands and multiplied as before (no normalise pass in front of the MMA); the statistics warps compute
// mu / rstd in parallel and write a per-token K16 correction row [a1 a1 a2 a1 a2 a3 | c1 c1 c2 c1 c2 c3 | 0 0 0 0] (a = -mu, c = 1 / rs,
// each split into three bf16 parts) into TMEM; one more K16 MMA per chunk against the forms' [u1 u2 u1 u3 u2 u1 | v1 v2 v1 v3 v2 v1]
// (u = sum_k W', v = W beta_in, three bf16 parts each) adds -mu u + v / rs to the accumulator in fp32, so the epilogue only scales by
// rs: plane = bf16(sigmoid(rs acc_g) (rs acc_p)), masked pairs 0.
constexpr int CRB = TOK * 16 * 2;   // a chunk's correction rows [128][16] bf16 (SW32) = 4 KB
template <int D, int FOLD = 0>
struct Cfg {
  static constexpr int KB = D / 64;                       // K-blocks per token row
  static constexpr int XC = D / 2;                        // TMEM columns of one xn tile (bf16 pairs)
  static constexpr int NXN = FOLD ? 1 : ((D <= 256) ? 2 : 1);   // xn tiles resident in TMEM
  static constexpr int NACC = (D <= 128) ? 3 : 2;         // 128-column accumulators
  static constexpr uint32_t T_ACC = 0, T_XN = NACC * 128, T_CR = T_XN + NXN * XC;   // T_CR (FOLD): 2 x 8 columns of correction rows
  static_assert(T_XN + NXN * XC + (FOLD ? 16 : 0) <= 512, "tmem");
  // FOLD at D64 (pass 15): a 4-slot weight ring (every K-block of one-direction hidden 128 / bidirectional resident) to fit the
  // correction boxes
  static constexpr int NWS = (D <= 128) ? ((FOLD && D == 64) ? 4 : 8) : (D == 256 ? K1W_NWS256 : (D == 512 ? K1W_NWS512 : 6));   // weight K-block ring
  // x K-block ring (FOLD at D128: one tile -- a slot is free as soon as it is copied into TMEM and read by the statistics warps)
  static constexpr int NXS = (D == 256) ? K1W_NXS256 : (D == 512 ? K1W_NXS512 : ((FOLD && D <= 128) ? 2 : 4));
  static_assert(D > K1W_INSTAT || D / 64 <= NXS, "in-kernel statistics: the row is resident in the x ring");
  static_assert(!FOLD || D == 64 || D == 128 || D == 256 || D == 384, "fold: D64 / D128 / D256 / D384");
  static constexpr int O_W = 0, O_X = O_W + NWS * KBB, O_OUT = O_X + NXS * KBB, O_GB = O_OUT + 8 * OSTW;
  // FOLD: no gamma / beta in shared memory; a [2 buffers][2][TOK] per-token table (rs, mask) for the epilogue, and a correction box
  // per chunk-ending weight slot (loaded with a chunk's last K-block): when KB divides the ring, slot s ends a chunk iff s % KB == KB - 1,
  // so the boxes take NWS / KB places (slot s -> s / KB), else one per slot
  static constexpr int NCR = (NWS % KB == 0) ? NWS / KB : NWS;
  static constexpr int O_CR = O_GB + (FOLD ? 2 * 2 * TOK * 4 : 2 * D * 4);
  static constexpr int O_BAR = O_CR + (FOLD ? NCR * CRB : 0);
  DEV static int cslot(int s) { return (NWS % KB == 0) ? s / KB : s; }
  static constexpr int SMEM = O_BAR + 512 + 1024;
  static_assert(SMEM <= 232448, "smem");
};

struct WMaps { CUtensorMap m[4]; };   // W_l, W_lg, W_r, W_rg

struct Bars {
  uint64_t w_full[8], w_empty[8], x_full[8], x_empty[8], xn_full[2], xn_empty[2], acc_full[3], acc_empty[3];
  uint64_t st_full[2], st_empty[2], cr_empty[2];   // FOLD: the per-token table + correction rows (statistics warps -> MMA / epilogue)
  uint32_t tmem;
};

DEV float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// k1.cu K1_SIG == 2: 0.5 + 0.5 tanh.approx(g / 2)
DEV float2 sigmoid2(float2 g) {
  const float2 h = __fmul2_rn(g, make_float2(0.5f, 0.5f));
  return __ffma2_rn(make_float2(tanh_approx(h.x), tanh_approx(h.y)), make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
}
DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV float4 lds128f(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
DEV void tmem_st8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
}
DEV void tmem_ld16x256_x4(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
                 "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
// smem (matrix descriptor) -> TMEM, 128 lanes x 256 bits (the issuing thread; asynchronous, ordered before its later tcgen05.mma)
DEV void tmem_cp_128x256b(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}
DEV void stmatrix_x4_trans(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d)
               : "memory");
}

template <int D, int P, int SMJ, int FOLD = 0>
__global__ void __launch_bounds__(NTHREADS, 1)
    k1w_kernel(const __grid_constant__ CUtensorMap mx, const __grid_constant__ WMaps mw, const __grid_constant__ CUtensorMap mplane,
               const uint8_t* __restrict__ tokmask, int L, float* __restrict__ mean, float* __restrict__ rstd,
               const float* __restrict__ gamma, const float* __restrict__ beta, int tiles, float eps, int stats_out, int n2, float rf3,
               const float* __restrict__ qinv, unsigned* __restrict__ amax, int ngrp) {
  // FOLD: gamma / beta unused (folded into the weights and the correction rows, mw.m[1])
  using G_ = Cfg<D, FOLD>;
  constexpr int KB = G_::KB, XC = G_::XC, NXN = G_::NXN, NACC = G_::NACC, NWS = G_::NWS, NXS = G_::NXS;
  constexpr int NCH = P / 64;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + G_::O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int n_local = tiles > cta ? (tiles - cta + G - 1) / G : 0;
  const uint64_t xhint = (long long)tiles * TOK * D * 2 <= ((long long)K1W_XKEEP << 20) ? EVICT_NORMAL : EVICT_FIRST;
  // flags bit 15 (S90): the packed weights come as 3-D boxes of gw K-blocks (one TMA op per gw blocks: a TMA op costs ~600 clk per SM whatever its size up to ~64 KB, so the stream is 2-3x faster than one 16 KB box per block)
  const int gw = (stats_out & 32768) ? ((stats_out >> 16) & 7) : 1;
  const int gsh = gw >> 1, gmk = gw - 1;   // gw is 1 or 2: shifts / masks, not integer divisions in the MMA and loader loops
#if K1W_EARLY
  // wres (flags bit 1, when every weight K-block of the P planes fits the ring): the weights are loaded once per CTA and kept
  const bool wres = (stats_out & 2) && NCH * KB <= NWS;
  // flags bit 5: no weight boxes in thread 0's entry fill (the weight warps issue them after the set-up; x alone at entry)
  const int nx0 = min(n_local * KB, NXS),
            nw0 = wres ? (n_local > 0 ? min(NCH * KB, NWS) : 0) : ((stats_out & 32) ? 0 : min(n_local * NCH * KB, NWS));
#else
  constexpr int nx0 = 0, nw0 = 0;
  const bool wres = false;
#endif
  float rf3s = 1.f;   // RoseTTAFold3 length factor (rf3 >= 0): the constant, or bf16(1 / max(n, 1)) from the token mask counted here
  if (rf3 >= 0.f) {
    if (rf3 > 0.f || tokmask == nullptr) rf3s = rf3;
    else {
      int n = 0;
      for (int i0 = 0; i0 < L; i0 += NTHREADS) n += __syncthreads_count(i0 + tid < L && tokmask[i0 + tid] != 0);
      rf3s = __bfloat162float(__float2bfloat16_rn(1.0f / (float)max(n, 1)));
    }
  }
  if (K1W_TRIG) pdl_trigger();
  if (tid == 0) TL(0)

  if (tid == 0) {
    for (int s = 0; s < NWS; ++s) { mbar_init(&B.w_full[s], 1); mbar_init(&B.w_empty[s], gw); }   // gw: K-blocks a weight op delivers (S90)
    // FOLD: an x slot is free once the statistics warps (4) read it and the tile's last MMA completed (1 commit)
    for (int s = 0; s < NXS; ++s) { mbar_init(&B.x_full[s], 1); mbar_init(&B.x_empty[s], FOLD ? 5 : 4); }
    for (int b = 0; b < NACC; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    for (int b = 0; b < NXN; ++b) { mbar_init(&B.xn_full[b], 1); mbar_init(&B.xn_empty[b], 1); }
    if (FOLD) for (int b = 0; b < 2; ++b) { mbar_init(&B.st_full[b], 1); mbar_init(&B.st_empty[b], 8); mbar_init(&B.cr_empty[b], 1); }
    fence_mbar_init();
    prefetch_tmap(&mx); prefetch_tmap(&mplane);
    for (int k = 0; k < 4; ++k) prefetch_tmap(&mw.m[k]);
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  constexpr bool SPRE = K1W_SPRE && D > K1W_INSTAT && !FOLD;   // the wait moves to the transform warps (mean / rstd)
  if (!SPRE) pdl_wait();   // nothing global is read before this (the previous kernel may write x, the weights, gamma / beta)
  if (tid == 0) {
#if K1W_EARLY
    // one-tile latency: the first fill of both rings is issued here, before the TMEM allocation and the CTA barrier (a cold box
    // lands ~1.3 us after its issue; the set-up below takes ~1.1 us of that)
    for (int q = 0; q < nx0; ++q) {
      mbar_expect_tx(&B.x_full[q], KBB);
      tma_load_2d(sm + G_::O_X + q * KBB, &mx, &B.x_full[q], (q % KB) * 64, (cta + (q / KB) * G) * TOK, xhint);
    }
    for (int q = 0; q < nw0; q += gw) {
      if (gw > 1) {   // grouped packed weights: one 3-D box of gw K-blocks of chunk c's [128 rows][64 k] images
        const int g = q >> gsh, kb0 = q % KB, row0 = ((q / KB) % NCH) * 64;
        const bool lastg = FOLD && kb0 + gw == KB;
        mbar_expect_tx(&B.w_full[g], gw * KBB + (lastg ? CRB : 0));
        tma_load_3d(sm + G_::O_W + q * KBB, &mw.m[2], &B.w_full[g], 0, 2 * row0, kb0, EVICT_LAST);
        if (lastg) tma_load_2d(sm + G_::O_CR + G_::cslot(q + gw - 1) * CRB, &mw.m[1], &B.w_full[g], 0, 2 * row0, EVICT_LAST);
        continue;
      }
      const int row = ((q / KB) % NCH) * 64, kb = q % KB, side = row < P / 2 ? 0 : 1, r = side ? row - P / 2 : row;
      mbar_expect_tx(&B.w_full[q], (FOLD && kb == KB - 1) ? KBB + CRB : KBB);
      if (stats_out & 16) {   // packed [2P, D] weights: chunk c's 64 gate | 64 projection rows at row 128 c, one [128][64] box
        tma_load_2d(sm + G_::O_W + q * KBB, &mw.m[0], &B.w_full[q], kb * 64, 2 * row, EVICT_LAST);
        if (FOLD && kb == KB - 1) tma_load_2d(sm + G_::O_CR + G_::cslot(q) * CRB, &mw.m[1], &B.w_full[q], 0, 2 * row, EVICT_LAST);   // correction
      } else {
        tma_load_2d(sm + G_::O_W + q * KBB, &mw.m[2 * side + 1], &B.w_full[q], kb * 64, r, EVICT_LAST);
        tma_load_2d(sm + G_::O_W + q * KBB + KBB / 2, &mw.m[2 * side], &B.w_full[q], kb * 64, r, EVICT_LAST);
      }
    }
#endif
  }
  if (!FOLD) {
    for (int k = tid; k < D; k += NTHREADS) {
      reinterpret_cast<float*>(sm + G_::O_GB)[k] = gamma[k];
      reinterpret_cast<float*>(sm + G_::O_GB)[D + k] = beta[k];
    }
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (tid == 0) TL(1)

  // weight issuers: warp 3 (fill q % wnp == 0), warp 2 (flags bit 3: == 1), warp 0 after its x stream (flags bit 6: == 2), with flags
  // bits 8-10 = wnp > 3 also the transform warps 4.. after their (only) tile (== 3 ..). Extra issuers only with one tile a CTA: warp 0
  // issues weights after its whole x stream and the transform warps after the transform (more tiles: they would wait on each other)
  const int wnx = (stats_out >> 8) & 7;
  const int wnp = ((stats_out & 64) && n_local <= 1) ? (wnx > 3 ? wnx : 3) : ((stats_out & 8) ? 2 : 1);
  const int wme = warp == 3 ? 0 : (warp == 2 ? 1 : (warp == 0 ? 2 : warp - 1));
  auto wloop = [&](const int me) {
      int s = 0, q = 0; uint32_t ph = 0;
      for (int i = 0; i < n_local; ++i)
        for (int c = 0; c < NCH; ++c)
          for (int kb = 0; kb < KB; ++kb) {
            if (gw > 1) {   // S90: grouped packed weights, one 3-D op per gw K-blocks (this loop visits every block; the op is issued at the group's first block)
              if (q >= nw0 && (q & gmk) == 0 && (q >> gsh) % wnp == me) {
                const int g = s >> gsh, row0 = c * 64;
                mbar_wait(&B.w_empty[g], ph ^ 1);
                const bool lastg = FOLD && kb + gw == KB;
                mbar_expect_tx(&B.w_full[g], gw * KBB + (lastg ? CRB : 0));
                tma_load_3d(sm + G_::O_W + s * KBB, &mw.m[2], &B.w_full[g], 0, 2 * row0, kb, EVICT_LAST);
                if (lastg) tma_load_2d(sm + G_::O_CR + G_::cslot(s + gw - 1) * CRB, &mw.m[1], &B.w_full[g], 0, 2 * row0, EVICT_LAST);
              }
            } else if (q >= nw0 && q % wnp == me) {
              mbar_wait(&B.w_empty[s], ph ^ 1);
              mbar_expect_tx(&B.w_full[s], (FOLD && kb == KB - 1) ? KBB + CRB : KBB);
              const int row = c * 64, side = row < P / 2 ? 0 : 1, r = side ? row - P / 2 : row;
              if (stats_out & 16) {   // packed weights: one box
                tma_load_2d(sm + G_::O_W + s * KBB, &mw.m[0], &B.w_full[s], kb * 64, 2 * row, EVICT_LAST);
                if (FOLD && kb == KB - 1) tma_load_2d(sm + G_::O_CR + G_::cslot(s) * CRB, &mw.m[1], &B.w_full[s], 0, 2 * row, EVICT_LAST);
              } else {
                tma_load_2d(sm + G_::O_W + s * KBB, &mw.m[2 * side + 1], &B.w_full[s], kb * 64, r, EVICT_LAST);            // gate rows
                tma_load_2d(sm + G_::O_W + s * KBB + KBB / 2, &mw.m[2 * side], &B.w_full[s], kb * 64, r, EVICT_LAST);      // projection
              }
            }
            ++q;
            if (++s == NWS) { s = 0; ph ^= 1; }
          }
  };
  const bool xiss = warp == 0 && lane == 0;
  const bool wiss = lane == 0 && !wres && (warp == 3 || (warp == 2 && wnp >= 2) || (warp == 0 && wnp >= 3));
  if (warp < 4) {
    if (xiss || wiss) {
     if (xiss) {
      // ---------------------------------------------------------------- x K-blocks -> ring (slot free once the transform read it)
      int s = 0, q = 0; uint32_t ph = 0;
      for (int i = 0; i < n_local; ++i) {
        const int row = (cta + i * G) * TOK;
        for (int kb = 0; kb < KB; ++kb) {
          if (q++ >= nx0) {
            mbar_wait(&B.x_empty[s], ph ^ 1);
            mbar_expect_tx(&B.x_full[s], KBB);
            tma_load_2d(sm + G_::O_X + s * KBB, &mx, &B.x_full[s], kb * 64, row, xhint);
          }
          if (++s == NXS) { s = 0; ph ^= 1; }
        }
      }
     }
     if (wiss) {   // wres: thread 0's first fill loaded every K-block
      wloop(wme);   // flags bit 3: two issuing warps (one thread issues ~1 TMA per 400-600 clk)
     }
    } else if (warp == 1) {
      // ---------------------------------------------------------------- MMA: acc[tok][0:64 gate | 64:128 proj] = xn (TMEM) . W_c^T
      const bool ldr = elect_one();
      const uint64_t w0 = desc_k_sw128(su + G_::O_W);
      constexpr uint64_t WSD = KBB >> 4;
      int s = 0, a = 0;
      uint32_t ph = 0, pa = 0;
      if constexpr (FOLD != 0) {
        // the raw x K-block is copied into TMEM by tcgen05.cp as it lands (the A image the transform warps write otherwise: lane =
        // token, 32 columns of bf16 pairs per K-block; the commit after the copy releases the slot together with the statistics
        // warps), acc = x . W'^T on the TS form, and at each chunk's last K-block one K16 MMA of the tile's correction rows
        const uint64_t x0 = desc_k_sw128(su + G_::O_X);
        for (int i = 0; i < n_local; ++i) {
          if (i >= 1) mbar_wait(&B.xn_empty[0], (i - 1) & 1);   // this thread's own commit: tile i - 1 has read the x image
          tc_fence_after();
          const uint32_t xa = tmem + G_::T_XN, cra = tmem + G_::T_CR + (i & 1) * 8;
#pragma unroll 1
          for (int c = 0; c < NCH; ++c) {
            mbar_wait(&B.acc_empty[a], pa ^ 1);
            tc_fence_after();
            const uint32_t d = tmem + G_::T_ACC + a * 128;
#pragma unroll 1
            for (int kb = 0; kb < KB; ++kb) {
              if (c == 0) {
                const int xq = i * KB + kb, xs = xq % NXS;
                mbar_wait(&B.x_full[xs], (xq / NXS) & 1);
                tc_fence_after();
                if (ldr) {
                  const uint64_t xd = x0 + (uint64_t)xs * WSD;
#pragma unroll
                  for (int j = 0; j < 4; ++j) tmem_cp_128x256b(xa + kb * 32 + j * 8, xd + 2 * j);
                  umma_commit(&B.x_empty[xs]);
                }
                __syncwarp();
              }
              if (wres) { s = (c * KB + kb) % NWS; ph = 0; }   // resident weights (D128 one direction): slot fixed, one fill
              if ((s & gmk) == 0) mbar_wait(&B.w_full[s >> gsh], ph);
              tc_fence_after();
              if (ldr && i == 0 && c == 0 && kb == 0) TL(4)
              if (ldr) {
                const uint64_t wd = w0 + (uint64_t)s * WSD;
                const uint32_t xk = xa + kb * 32;
                umma_ts(d, xk, wd, IDESC, kb > 0);
                umma_ts(d, xk + 8, wd + 2, IDESC, 1);
                umma_ts(d, xk + 16, wd + 4, IDESC, 1);
                umma_ts(d, xk + 24, wd + 6, IDESC, 1);
              }
              __syncwarp();
              if (kb == KB - 1) {
                if (c == 0) { mbar_wait(&B.st_full[i & 1], (i >> 1) & 1); tc_fence_after(); }   // the tile's correction rows
                if (ldr) umma_ts(d, cra, desc_k_sw32(su + G_::O_CR + G_::cslot(s) * CRB), IDESC, 1);
                __syncwarp();
              }
              if (ldr && !wres) umma_commit(&B.w_empty[s >> gsh]);
              __syncwarp();
              if (++s == NWS) { s = 0; ph ^= 1; }
            }
            if (ldr) umma_commit(&B.acc_full[a]);
            __syncwarp();
            if (++a == NACC) { a = 0; pa ^= 1; }
          }
          if (ldr && i == 0) TL(5)
          if (ldr) { umma_commit(&B.xn_empty[0]); umma_commit(&B.cr_empty[i & 1]); }
          __syncwarp();
        }
      } else
      for (int i = 0; i < n_local; ++i) {
        const int b = (NXN == 1) ? 0 : (i & 1);
        mbar_wait(&B.xn_full[b], (i / NXN) & 1);
        tc_fence_after();
        const uint32_t xa = tmem + G_::T_XN + b * XC;
#pragma unroll 1
        for (int c = 0; c < NCH; ++c) {
          mbar_wait(&B.acc_empty[a], pa ^ 1);
          tc_fence_after();
          const uint32_t d = tmem + G_::T_ACC + a * 128;
#pragma unroll 1
          for (int kb = 0; kb < KB; ++kb) {
            if (wres) { s = (c * KB + kb) % NWS; ph = 0; }   // resident: slot fixed, its one fill (phase 0) completes once
            if ((s & gmk) == 0) mbar_wait(&B.w_full[s >> gsh], ph);
            tc_fence_after();
            if (ldr && i == 0 && c == 0 && kb == 0) TL(4)
            if (ldr) {
              const uint64_t wd = w0 + (uint64_t)s * WSD;
              const uint32_t xk = xa + kb * 32;
              umma_ts(d, xk, wd, IDESC, kb > 0);
              umma_ts(d, xk + 8, wd + 2, IDESC, 1);
              umma_ts(d, xk + 16, wd + 4, IDESC, 1);
              umma_ts(d, xk + 24, wd + 6, IDESC, 1);
              if (!wres) umma_commit(&B.w_empty[s >> gsh]);
            }
            __syncwarp();
            if (++s == NWS) { s = 0; ph ^= 1; }
          }
          if (ldr) umma_commit(&B.acc_full[a]);
          __syncwarp();
          if (++a == NACC) { a = 0; pa ^= 1; }
        }
        if (ldr && i == 0) TL(5)
        if (ldr) umma_commit(&B.xn_empty[b]);
        __syncwarp();
      }
    }
  } else {
  // flags bit 7 (one tile a CTA, in-kernel statistics, D >= 128): the gate-epilogue warps 8-11 -- idle until the first accumulator --
  // transform the second half of the row's K-blocks (statistics: per-half sums exchanged through the output staging, which no
  // epilogue uses yet); warps 4-7 the first half
  const bool tsp = !FOLD && (D <= K1W_INSTAT && D >= 128) && (stats_out & 128) && n_local == 1;
  auto transform = [&](const int g, const int ng) {
    // ------------------------------------------------------------------ transform: thread = token row r, K-block by K-block
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    float* xs = reinterpret_cast<float*>(sm + G_::O_OUT);   // ng 2: [2 passes][2 halves][TOK]
    const int kb0 = g * KB / ng, kb1 = (g + 1) * KB / ng;
    int s = 0; uint32_t ph = 0;
    if (ng == 2) { s = kb0; }                                // one tile (n_local 1): K-block kb is ring slot kb, phase 0
    for (int i = 0; i < n_local; ++i) {
      const int row = (cta + i * G) * TOK + r, b = (NXN == 1) ? 0 : (i & 1);
      // tokens are b-major: row = (sample * L + i) * L + j, the token mask is [B, L]
      const int ll = L * L, sb = row / ll, rin = row - sb * ll;
      const uint32_t kmask = (tokmask == nullptr || (tokmask[sb * L + rin / L] && tokmask[sb * L + rin % L])) ? 0xffffffffu : 0u;
      float mu, rs;
      if constexpr (D <= K1W_INSTAT) {
        // whole row resident (KB <= 2 ring slots): two-pass statistics here, as k1.cu (packed f32x2 sums)
        int ss = s; uint32_t pp = ph;
        for (int kb = kb0; kb < kb1; ++kb) { mbar_wait(&B.x_full[ss], pp); if (++ss == NXS) { ss = 0; pp ^= 1; } }
        if (r == 0 && i == 0) TL(2)
        const int sbase = ng == 2 ? 0 : s;
        auto ldx = [&](int q) { const int kb = q >> 3; int sl = sbase + kb; if (sl >= NXS) sl -= NXS;
                                return lds128(su + G_::O_X + sl * KBB + sw128(r, q & 7)); };
        const int q0 = kb0 * 8, q1 = kb1 * 8;
        float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll (D <= 64 ? D / 8 : 4)   // a full unroll spilled 72 B at D128
        for (int q = q0; q < q1; ++q) {
          const uint4 t = ldx(q);
          a0 = __fadd2_rn(a0, up(t.x)); a1 = __fadd2_rn(a1, up(t.y)); a2 = __fadd2_rn(a2, up(t.z)); a3 = __fadd2_rn(a3, up(t.w));
        }
        a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
        if (ng == 2) {   // the two halves' sums, added in half order (deterministic)
          xs[g * TOK + r] = __fadd_rn(a0.x, a0.y);
          named_bar_sync(1, 256);
          mu = __fmul_rn(__fadd_rn(xs[r], xs[TOK + r]), 1.f / D);
        } else {
          mu = __fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / D);
        }
        const float2 nm0 = make_float2(-mu, -mu);
        a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll (D <= 64 ? D / 8 : 4)
        for (int q = q0; q < q1; ++q) {
          const uint4 t = ldx(q);
          float2 d;
          d = __fadd2_rn(up(t.x), nm0); a0 = __ffma2_rn(d, d, a0);
          d = __fadd2_rn(up(t.y), nm0); a1 = __ffma2_rn(d, d, a1);
          d = __fadd2_rn(up(t.z), nm0); a2 = __ffma2_rn(d, d, a2);
          d = __fadd2_rn(up(t.w), nm0); a3 = __ffma2_rn(d, d, a3);
        }
        a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
        if (ng == 2) {
          xs[(2 + g) * TOK + r] = __fadd_rn(a0.x, a0.y);
          named_bar_sync(1, 256);
          rs = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(xs[2 * TOK + r], xs[3 * TOK + r]), 1.f / D), eps));
        } else {
          rs = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / D), eps));
        }
        if ((stats_out & 1) && g == 0) { mean[row] = mu; rstd[row] = rs; }
      } else {
        if (SPRE && i == 0) pdl_wait();   // mean / rstd are k1w_stats' output
        mu = mean[row]; rs = rstd[row];
      }
      const float2 nm = make_float2(-mu, -mu), rr = make_float2(rs, rs);
      if (i >= NXN) mbar_wait(&B.xn_empty[b], ((i / NXN) - 1) & 1);
      tc_fence_after();
      for (int kb = kb0; kb < kb1; ++kb) {
        if constexpr (D > K1W_INSTAT) mbar_wait(&B.x_full[s], ph);
        const uint32_t xr = su + G_::O_X + s * KBB;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {          // 16 channels (8 packed words) per TMEM store
          uint32_t o[8];
#pragma unroll
          for (int qq = 0; qq < 2; ++qq) {
            const int q = q4 * 2 + qq;
            const uint4 t = lds128(xr + sw128(r, q));
            const int col = kb * 64 + q * 8;
            const float4 ga = lds128f(su + G_::O_GB + col * 4);
            const float4 gb4 = lds128f(su + G_::O_GB + col * 4 + 16);
            const float4 ba = lds128f(su + G_::O_GB + D * 4 + col * 4);
            const float4 bb = lds128f(su + G_::O_GB + D * 4 + col * 4 + 16);
            auto aff2 = [&](uint32_t w, float g0, float g1, float b0, float b1) {
              const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w), nm), rr), make_float2(g0, g1), make_float2(b0, b1));
              return pack_bf16(y.x, y.y) & kmask;
            };
            o[4 * qq + 0] = aff2(t.x, ga.x, ga.y, ba.x, ba.y);
            o[4 * qq + 1] = aff2(t.y, ga.z, ga.w, ba.z, ba.w);
            o[4 * qq + 2] = aff2(t.z, gb4.x, gb4.y, bb.x, bb.y);
            o[4 * qq + 3] = aff2(t.w, gb4.z, gb4.w, bb.z, bb.w);
          }
          tmem_st8(trow + G_::T_XN + b * XC + kb * 32 + q4 * 8, o);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.x_empty[s]);
        if (++s == NXS) { s = 0; ph ^= 1; }
      }
      tmem_wait_st();
      tc_fence_before();
      named_bar_sync(1, 128 * ng);
      if (r == 0 && i == 0 && g == 0) TL(3)
      if (warp == 4 && lane == 0) mbar_arrive(&B.xn_full[b]);
    }
  };
  if (FOLD && warp < 8) {
    // ------------------------------------------------------------------ FOLD statistics: thread = token row r, one pass over the
    // K-blocks as they land (the MMA thread copies the same slots into TMEM meanwhile): sums of x - k and (x - k)^2 with the row's
    // first value as pivot k (ln_stats' pivot-shifted form), each slot released as soon as it is read. -> mean / rstd (k3w), the
    // epilogue's table (rs, mask) and the tile's TMEM correction row (a = -mu, c = 1 / rs, three bf16 parts each).
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    float* tb = reinterpret_cast<float*>(sm + G_::O_GB);
    for (int i = 0; i < n_local; ++i) {
      const int row = (cta + i * G) * TOK + r, q0 = i * KB;
      const int ll = L * L, sb = row / ll, rin = row - sb * ll;
      const bool keep = tokmask == nullptr || (tokmask[sb * L + rin / L] && tokmask[sb * L + rin % L]);
      float2 s0 = make_float2(0.f, 0.f), s1 = s0, q0v = s0, q1v = s0;
      float piv = 0.f;
      for (int kb = 0; kb < KB; ++kb) {
        const int xq = q0 + kb, xs = xq % NXS;
        mbar_wait(&B.x_full[xs], (xq / NXS) & 1);
        if (r == 0 && i == 0 && kb == 0) TL(2)
        const uint32_t xr = su + G_::O_X + xs * KBB;
        if (kb == 0) piv = up(lds128(xr + sw128(r, 0)).x).x;
        const float2 np = make_float2(-piv, -piv);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 t = lds128(xr + sw128(r, q));
          float2 d;
          d = __fadd2_rn(up(t.x), np); s0 = __fadd2_rn(s0, d); q0v = __ffma2_rn(d, d, q0v);
          d = __fadd2_rn(up(t.y), np); s1 = __fadd2_rn(s1, d); q1v = __ffma2_rn(d, d, q1v);
          d = __fadd2_rn(up(t.z), np); s0 = __fadd2_rn(s0, d); q0v = __ffma2_rn(d, d, q0v);
          d = __fadd2_rn(up(t.w), np); s1 = __fadd2_rn(s1, d); q1v = __ffma2_rn(d, d, q1v);
        }
        fence_async_smem();   // generic reads of the slot before the TMA refill
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.x_empty[xs]);
      }
      s0 = __fadd2_rn(s0, s1); q0v = __fadd2_rn(q0v, q1v);
      const float m1 = __fmul_rn(__fadd_rn(s0.x, s0.y), 1.f / D);
      const float mu = __fadd_rn(piv, m1);
      const float var = fmaxf(__fmaf_rn(-m1, m1, __fmul_rn(__fadd_rn(q0v.x, q0v.y), 1.f / D)), 0.f);
      const float rs = rsqrt_ftz(__fadd_rn(var, eps));
      if (stats_out & 1) { mean[row] = mu; rstd[row] = rs; }
      const int b = i & 1;
      if (i >= 2) { mbar_wait(&B.st_empty[b], ((i >> 1) - 1) & 1); mbar_wait(&B.cr_empty[b], ((i >> 1) - 1) & 1); }
      tc_fence_after();
      tb[(b * 2 + 0) * TOK + r] = rs;
      tb[(b * 2 + 1) * TOK + r] = keep ? 1.f : 0.f;
      auto split3 = [](float v, float (&o)[3]) {
        o[0] = __bfloat162float(__float2bfloat16_rn(v));
        const float r1 = v - o[0];
        o[1] = __bfloat162float(__float2bfloat16_rn(r1));
        o[2] = __bfloat162float(__float2bfloat16_rn(r1 - o[1]));
      };
      float a3[3], c3[3];
      split3(-mu, a3);
      split3(__frcp_rn(rs), c3);
      uint32_t cr[8];
      cr[0] = pack_bf16(a3[0], a3[0]); cr[1] = pack_bf16(a3[1], a3[0]); cr[2] = pack_bf16(a3[1], a3[2]);
      cr[3] = pack_bf16(c3[0], c3[0]); cr[4] = pack_bf16(c3[1], c3[0]); cr[5] = pack_bf16(c3[1], c3[2]);
      cr[6] = 0u; cr[7] = 0u;
      tmem_st8(trow + G_::T_CR + b * 8, cr);
      tmem_wait_st();
      tc_fence_before();
      named_bar_sync(1, 128);
      if (r == 0 && i == 0) TL(3)
      if (warp == 4 && lane == 0) mbar_arrive(&B.st_full[b]);
    }
  } else if (warp < 8) {
    transform(0, tsp ? 2 : 1);
    if (lane == 0 && !wres && wme < wnp) wloop(wme);   // wnp > 3 (one tile a CTA): the transform warps become weight issuers
  }
  else {
    if (tsp && warp < 12) transform(1, 2);
    // ------------------------------------------------------------------ gate epilogue (k1.cu): warp quarter q = tokens 32q..32q+31
    const int grp = (warp - 8) >> 2, q = warp & 3;
    const uint32_t stg = su + G_::O_OUT + (warp - 8) * OSTW;
    const bool wst = true;   // e4m3 planes: the pair of warps (64 tokens) shares one [64 ch][64 tok] byte box (64-byte rows, SW64)
    const int pi = grp * 2 + (q >> 1), hq = q & 1;
    float fa[4] = {0.f, 0.f, 0.f, 0.f}, fm[4] = {0.f, 0.f, 0.f, 0.f};   // FOLD: rs and the pair mask of this thread's rows
    float amx = 0.f;   // largest |plane value * qinv| of the current guard group (non-negative floats order as unsigned)
    int gcur = 0;
    auto flush_amax = [&](int gsel) {
      if constexpr (F8_ABL & 1) return;
      float v = amx;
#pragma unroll
      for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
      if (lane == 0 && v > 0.f) atomicMax(amax + gsel, __float_as_uint(v));
      amx = 0.f;
    };
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + i * G) * TOK;
      if constexpr (FOLD) {   // this thread's 4 token rows: q 32 + hh 16 + e 8 + lane / 4 (the 16x256b fragment rows)
        mbar_wait(&B.st_full[i & 1], (i >> 1) & 1);
        const float* tb = reinterpret_cast<const float*>(sm + G_::O_GB) + (i & 1) * 2 * TOK;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int rr = q * 32 + (k >> 1) * 16 + (k & 1) * 8 + (lane >> 2);
          fa[k] = tb[rr]; fm[k] = tb[TOK + rr];
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.st_empty[i & 1]);
      }
      for (int c = grp; c < NCH; c += 2) {
        const int ci = i * NCH + c, acc = ci % NACC;
        mbar_wait(&B.acc_full[acc], (ci / NACC) & 1);
        tc_fence_after();
        if (warp == 8 && lane == 0 && i == 0 && c == 0) TL(6)
        if (warp == 12 && lane == 0 && i == 0 && c == 1) TL(7)
        if (wst) { if (hq == 0 && lane == 0) bulk_wait_read<0>(); __syncwarp(); named_bar_sync(2 + 2 * pi, 64); }
        else { if (lane == 0) bulk_wait_read<0>(); __syncwarp(); }
        __syncwarp();
        const uint32_t tq = tmem + G_::T_ACC + acc * 128 + ((uint32_t)(q * 32) << 16);
        uint32_t ga[16], pa[16], gb[16], pb[16];
        const float qs = __ldg(qinv + c);
        const int gc = (c * ngrp) / NCH;   // the guard group of this chunk: the running max is flushed when the group changes (a chunk range of one group is visited in a row)
        if (gc != gcur) { flush_amax(gcur); gcur = gc; }
        auto ldg = [&](int grp4, uint32_t (&g)[16], uint32_t (&p)[16]) {
          const uint32_t ta = tq + ((uint32_t)((grp4 >> 1) * 16) << 16) + (grp4 & 1) * 32;
          tmem_ld16x256_x4(ta, g);
          tmem_ld16x256_x4(ta + 64, p);
        };
        auto work = [&](int grp4, const uint32_t (&g)[16], const uint32_t (&p)[16]) {
          const int hh = grp4 >> 1, cq = grp4 & 1;
          float2 av[8];   // [2 j + e]: the plane values of channels (cq 32 + 8 j + 2 (lane & 3), +1) at token 8 (2 hh + e) + lane / 4
          if constexpr (FOLD) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
              for (int e = 0; e < 2; ++e) {
                const int k = hh * 2 + e;
                const float2 A2 = make_float2(fa[k], fa[k]);   // the correction MMA added -mu u + v / rs: scale by rs
                const float2 gg = __fmul2_rn(A2, make_float2(__uint_as_float(g[4 * j + 2 * e]), __uint_as_float(g[4 * j + 2 * e + 1])));
                const float2 pp = __fmul2_rn(A2, make_float2(__uint_as_float(p[4 * j + 2 * e]), __uint_as_float(p[4 * j + 2 * e + 1])));
                const float2 a = __fmul2_rn(sigmoid2(gg), pp);
                av[2 * j + e] = fm[k] != 0.f ? a : make_float2(0.f, 0.f);   // a masked pair's plane values are exactly 0
              }
            }
          } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float2 s0 = sigmoid2(make_float2(__uint_as_float(g[4 * j]), __uint_as_float(g[4 * j + 1])));
              const float2 s1 = sigmoid2(make_float2(__uint_as_float(g[4 * j + 2]), __uint_as_float(g[4 * j + 3])));
              float2 pq0 = make_float2(__uint_as_float(p[4 * j]), __uint_as_float(p[4 * j + 1])), pq1 = make_float2(__uint_as_float(p[4 * j + 2]), __uint_as_float(p[4 * j + 3]));
              if (rf3 >= 0.f && c < NCH / 2) { pq0 = __fmul2_rn(pq0, make_float2(rf3s, rf3s)); pq1 = __fmul2_rn(pq1, make_float2(rf3s, rf3s)); }   // RoseTTAFold3: f on the left projection
              av[2 * j] = __fmul2_rn(s0, pq0);
              av[2 * j + 1] = __fmul2_rn(s1, pq1);
            }
          }
          // e4m3: the plane value times the chunk's power-of-two qinv, cast saturating (never a NaN byte); the largest scaled magnitude goes to the guard
          uint32_t h[8];
#pragma unroll
          for (int w8 = 0; w8 < 8; ++w8) {
            const float2 sv = (F8_ABL & 16) ? av[w8] : __fmul2_rn(av[w8], make_float2(qs, qs));
            if constexpr (!(F8_ABL & 4)) amx = fmaxf(amx, fmaxf(fabsf(sv.x), fabsf(sv.y)));
            if constexpr (F8_ABL & 32) h[w8] = (__float_as_uint(sv.x) >> 24) | (__float_as_uint(sv.y) >> 16 & 0xff00u); else h[w8] = (uint32_t)__nv_cvt_float2_to_fp8x2(sv, __NV_SATFINITE, __NV_E4M3);
          }
          // The plane's token order is shuffled inside every 16-token group (position p holds token (p >> 1) + 8 (p & 1)): the two tokens a thread
          // holds, tr and tr + 8, are one 16-bit element, so ONE stmatrix.trans per work writes the channel-major byte rows (4 PRMT + 1 stmatrix, no
          // lane exchange). The contraction takes both operands in the same order (outgoing: the contracted axis; incoming: ctr8's epilogue undoes it).
          uint32_t r4[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) r4[j] = __byte_perm(h[2 * j], h[2 * j + 1], 0x5140);
          {
            const int k = lane >> 3, jr = lane & 7;
            const int chl = cq * 32 + k * 8 + jr;
            if constexpr (!(F8_ABL & 64)) stmatrix_x4_trans(su + G_::O_OUT + pi * 4096 + chl * 64 + ((((hq << 1) | hh) ^ ((chl >> 1) & 3)) << 4), r4[0], r4[1], r4[2], r4[3]);
            else asm volatile("" :: "r"(r4[0]), "r"(r4[1]), "r"(r4[2]), "r"(r4[3]));
          }
        };
        ldg(0, ga, pa);
        tmem_wait_ld();
        ldg(1, gb, pb);
        work(0, ga, pa);
        tmem_wait_ld();
        ldg(2, ga, pa);
        work(1, gb, pb);
        tmem_wait_ld();
        ldg(3, gb, pb);
        work(2, ga, pa);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_empty[acc]);
        work(3, gb, pb);
        fence_async_smem();
        __syncwarp();
        if (wst) {   // 128-byte plane rows: the pair of warps (tokens 64 pi .. 64 pi + 63) completes one [64 ch][64 tok] SW128 box; its first warp stores it
          named_bar_sync(3 + 2 * pi, 64);
          if (hq == 0 && lane == 0) {
            tma_store_2d(&mplane, sm + G_::O_OUT + pi * 4096, row0 + q * 32, c * 64);
            bulk_commit();
          }
        } else if (lane == 0) {
          if constexpr (SMJ != 0) {        // sample-major planes (n2 tokens per sample): the 4-D map of tmap::make_chmajor
            int tq, ts;
            tok_split(row0 + q * 32, n2, tq, ts);
            tma_store_4d(&mplane, sm + G_::O_OUT + (warp - 8) * OSTW, tq, 0, ts, c);
          } else {
            tma_store_2d(&mplane, sm + G_::O_OUT + (warp - 8) * OSTW, row0 + q * 32, c * 64);
          }
          bulk_commit();
        }
      }
    }
    flush_amax(gcur);
    if (warp == 8 && lane == 0) TL(8)
    if (warp == 12 && lane == 0) TL(9)
    if (lane == 0) bulk_wait<0>();
    if (warp == 8 && lane == 0) TL(10)
    if (warp == 12 && lane == 0) TL(11)
  }
  }
  tc_fence_before();
  __syncthreads();
  if (tid == 0) TL(12)
  if (warp == 2) tmem_dealloc(tmem, 512);
}

int num_sms() {
  static int n = 0;
  if (!n) { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
  return n;
}

template <int D, int P>
void launch(const CUtensorMap& mx, const WMaps& mw, const CUtensorMap& mp, const uint8_t* mask, int L, float* mean, float* rs,
            const float* g, const float* b, int tiles, int grid, cudaStream_t st, float eps, int stats_out, int n2, float rf3,
            const float* qv, unsigned* am, int ng) {
  static bool attr = false;
  if constexpr (D == 64) {   // sample-major planes: D64 training only (a compile-time variant: the [P, M] kernel is unchanged)
    static bool attr_s = false;
    if (n2 > 0) {
      if (!attr_s) { cudaFuncSetAttribute(k1w_kernel<D, P, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<D>::SMEM); attr_s = true; }
      tmap::launch_pdl(k1w_kernel<D, P, 1>, dim3(grid), dim3(NTHREADS), Cfg<D>::SMEM, st, K1W_PDL, mx, mw, mp, mask, L, mean, rs, g, b, tiles, eps, stats_out, n2, rf3, qv, am, ng);
      return;
    }
  }
  TORCH_CHECK(n2 == 0, "k1w: sample-major planes are D64 only");
  if constexpr (D == 64 || D == 128 || D == 256 || D == 384) {
    if (stats_out & 4096) {   // FOLD (the LN_in-folded front, inference)
      static bool attr_f = false;
      if (!attr_f) { cudaFuncSetAttribute(k1w_kernel<D, P, 0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<D, 1>::SMEM); attr_f = true; }
      tmap::launch_pdl(k1w_kernel<D, P, 0, 1>, dim3(grid), dim3(NTHREADS), Cfg<D, 1>::SMEM, st, K1W_PDL, mx, mw, mp, mask, L, mean, rs, g, b, tiles, eps, stats_out, 0, rf3, qv, am, ng);
      return;
    }
  }
  if (!attr) { cudaFuncSetAttribute(k1w_kernel<D, P, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<D>::SMEM); attr = true; }
  tmap::launch_pdl(k1w_kernel<D, P, 0>, dim3(grid), dim3(NTHREADS), Cfg<D>::SMEM, st, K1W_PDL, mx, mw, mp, mask, L, mean, rs, g, b, tiles, eps, stats_out, 0, rf3, qv, am, ng);
}

}  // namespace k1w8

// x [L*L, D] bf16; wl, wlg, wr, wrg [P/2, D] bf16 with unit column stride (planes 0..P/2 are left, P/2..P right);
// tokmask [L] bool / uint8 (None: every token kept), mean / rstd [L*L] fp32, gamma / beta [D] fp32, planes [P, L, L] bf16.
// D <= 128: statistics are computed in the kernel and written to mean / rstd when given; D > 128: read from mean / rstd (k1w_stats).
static void k1w_forward_impl(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                 c10::optional<torch::Tensor> tokmask, c10::optional<torch::Tensor> mean,
                 c10::optional<torch::Tensor> rstd, torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid,
                 double eps, double rf3, torch::Tensor qinv, torch::Tensor amax, int64_t ngrp) {
  using namespace k1w8;
  const int64_t M = x.size(0), D = x.size(1), P = tmap::chmajor_channels(planes);
  TORCH_CHECK(M % TOK == 0 && planes.numel() == P * M && x.is_contiguous() && planes.is_contiguous() &&
              (planes.dim() == 3 || planes.dim() == 4 || planes.dim() == 5));
  // planes [P, L, L] (one sample), [P, B, L, L] or sample-major [P / 64, B, 64, L, L] (tmap::make_chmajor); the tokens of x are
  // b-major, M = B L L
  const int L = (int)planes.size(planes.dim() - 1);
  const int64_t nb = M / ((int64_t)L * L);
  TORCH_CHECK(nb * L * L == M && planes.size(planes.dim() - 2) == L);
  TORCH_CHECK(!tokmask.has_value() || (tokmask->numel() == nb * L && tokmask->element_size() == 1 && tokmask->is_contiguous()));
  const auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto mx = tmap::make(x.data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  WMaps mw;
  const torch::Tensor* ws[4] = {&wl, &wlg, &wr, &wrg};
  // wl [2P, D] (row-major): the packed front (chunk c = rows 128 c .. 128 c + 127: its 64 gate rows, then its 64 projection rows;
  // b200_infer.pack_front), read as one [128][64] box per ring slot instead of two [64][64] boxes; wlg / wr / wrg are not read
  const bool packed = wl.size(0) == 2 * P;
  bool gw3 = false; int gwh = 1;
  if (packed) {
    TORCH_CHECK(wl.size(1) == D && wl.stride(1) == 1 && (wl.stride(0) * 2) % 16 == 0, "k1w_forward: packed front [2P, D], unit column stride");
    mw.m[0] = tmap::make(wl.data_ptr(), bf, {(uint64_t)D, (uint64_t)(2 * P)}, {(uint64_t)wl.stride(0) * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
    mw.m[1] = mw.m[2] = mw.m[3] = mw.m[0];
    // S90: the packed weights as 3-D boxes of gwh K-blocks of one chunk ([k in 64][row][k block], the slot layout of gwh consecutive 16 KB images), inference only (flags bit 15)
    const char* gw_env = std::getenv("TRIMUL_K1W_GW");
    const int kbh = (int)(D / 64);
    const int gw4 = (D == 256 && K1W_NWS256 % 4 == 0) || (D == 512 && K1W_NWS512 % 4 == 0);   // 64 KB ops where the ring holds whole groups of four
    gwh = kbh % 2 == 0 ? 2 : 1;   // 3 and 4 K-blocks (> 32 KB tensor boxes) fault (part 11 / 12)
    gw3 = grid < 0 && gwh > 1 && !(gw_env != nullptr && gw_env[0] == '0');
    if (gw3) mw.m[2] = tmap::make(wl.data_ptr(), bf, {64, (uint64_t)(2 * P), (uint64_t)kbh}, {(uint64_t)wl.stride(0) * 2, 128}, {64, 128, (uint32_t)gwh}, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  // FOLD (flags bit 12; D256 / D384 inference): wl is the LN_in-folded packed front (b200_infer.front_fold) and wlg its correction
  // rows [2P, 16] bf16 (a chunk's [128][16] box with the 32-byte swizzle, loaded with the chunk's last K-block); gamma / beta are
  // not read. The statistics are computed in the kernel and written to mean / rstd.
  const bool fold = packed && (D == 64 || D == 128 || D == 256 || D == 384) && wlg.dim() == 2 && wlg.size(0) == 2 * P && wlg.size(1) == 16;
  if (fold) {
    TORCH_CHECK(wlg.is_contiguous() && wlg.scalar_type() == torch::kBFloat16, "k1w_forward: fold correction rows [2P, 16] bf16");
    mw.m[1] = tmap::make(wlg.data_ptr(), bf, {16, (uint64_t)(2 * P)}, {32}, {16, 128}, CU_TENSOR_MAP_SWIZZLE_32B);
  }
  for (int k = 0; k < 4 && !packed; ++k) {
    const torch::Tensor& w = *ws[k];
    TORCH_CHECK(w.size(0) == P / 2 && w.size(1) == D && w.stride(1) == 1 && (w.stride(0) * 2) % 16 == 0,
                "k1w_forward: front weights must be [P/2, D] with unit column stride");
    mw.m[k] = tmap::make(w.data_ptr(), bf, {(uint64_t)D, (uint64_t)(P / 2)}, {(uint64_t)w.stride(0) * 2}, {64, 64},
                         CU_TENSOR_MAP_SWIZZLE_128B);
  }
  const int n2 = planes.dim() == 5 ? L * L : 0;   // 0: the [P, M] map as before
  // S90: inference planes written as [64 ch][64 tok] SW128 boxes (128-byte rows; flags bit 14) instead of [64 ch][32 tok] SW64 boxes (64-byte
  // rows). env TRIMUL_K1W_WST = 0 off (A/B)
  const char* wst_env = std::getenv("TRIMUL_K1W_WST");
  const bool wst = false;   // e4m3 planes: [32 tok] byte boxes a warp, no swizzle
  TORCH_CHECK(planes.scalar_type() == torch::kUInt8 && planes.dim() == 4 && qinv.scalar_type() == torch::kFloat32 && qinv.numel() >= P / 64 && amax.numel() >= ngrp && (P / 64) % ngrp == 0,
              "k1w8_forward: uint8 planes [P, B, L, L], qinv [P / 64] fp32, amax [ngrp] int32 / fp32 bits");
  auto mp = tmap::make(planes.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, {(uint64_t)M, (uint64_t)P}, {(uint64_t)M}, {64u, 64}, CU_TENSOR_MAP_SWIZZLE_64B);
  const int tiles = (int)(M / TOK);
  const bool inf13 = grid < 0;   // grid -1: an inference call (b200_infer): the pass-13 issue / transform defaults below apply
  int g = grid > 0 ? (int)grid : num_sms();
  g = std::min(g, tiles);
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(D <= K1W_INSTAT || (mean.has_value() && rstd.has_value()), "k1w_forward: this width needs precomputed statistics");
  const uint8_t* mk = tokmask.has_value() ? reinterpret_cast<const uint8_t*>(tokmask->data_ptr()) : nullptr;
  const float *ga = gamma.data_ptr<float>(), *be = beta.data_ptr<float>();
  const float* qv = qinv.data_ptr<float>();
  unsigned* am = reinterpret_cast<unsigned*>(amax.data_ptr());
  const int ng = (int)ngrp;
  float* mu = mean.has_value() ? mean->data_ptr<float>() : nullptr;
  float* rs = rstd.has_value() ? rstd->data_ptr<float>() : nullptr;
  const char* wr_env = std::getenv("TRIMUL_K1W_WRES");
  // resident weights (bit 1) by default where they fit the ring (D <= 128: 2026-11-17 pass 8, eval -0.4 to -1.7 %, training equal);
  // env TRIMUL_K1W_WRES = 0 off, 1 on (A/B)
  const int wr_mode = wr_env != nullptr ? wr_env[0] - '0' : 1;
  TORCH_CHECK(!fold || (mu != nullptr && rs != nullptr), "k1w_forward: the folded front writes mean / rstd");
  const int so = (((D <= K1W_INSTAT || fold) && mu != nullptr) ? 1 : 0) | (wr_mode == 1 ? 2 : 0) | (fold ? 4096 : 0) | (wst ? 16384 : 0) | (gw3 ? (32768 | (gwh << 16)) : 0);
  const char* np_env = std::getenv("TRIMUL_K1W_NP");
  // two weight-issuing warps (flags bit 3) at D >= 256 (2026-11-17, round 3 eighth pass: D384 L128 eval -5 %, D256 L128 -2.6 %, D384 L768
  // -7 %; D <= 128 no gain); env TRIMUL_K1W_NP = 1 / 2 overrides (A/B)
  int so_np = (np_env != nullptr ? np_env[0] >= '2' : D >= 256) ? 8 : 0;
  // 2026-11-18 pass 13: at D >= 256 a third weight issuer (warp 0 after its x stream; the kernel uses it only with one tile a CTA,
  // i.e. L128) -- rotated A/B, mean of 6 captures: bidirectional D256 L128 51.3 -> 50.9 (with the transform split 50.0-50.5), D384
  // 56.9 -> 55.6, D512 78.3 -> 76.7, bidirectional D384 85.6 -> 83.8; five / seven issuers (+ the transform warps) no better.
  // env TRIMUL_K1W_NP = 1 / 2 / 3 / 5 / 7 overrides (A/B)
  if (np_env != nullptr ? (np_env[0] >= '3' && np_env[0] <= '7') : (inf13 && D >= 256)) {   // 3: warp 0 after its x stream; 4-7: + the transform warps
    so_np |= 64 | ((np_env != nullptr ? np_env[0] - '0' : 3) << 8);
  }
  const char* ew_env = std::getenv("TRIMUL_K1W_EW");        // 1: no weight boxes in the entry fill (A/B; measured level, not used)
  if (ew_env != nullptr && ew_env[0] == '1') so_np |= 32;
  // the transform split over warps 4-7 and 8-11 (flags bit 7; one tile a CTA, in-kernel statistics; 2026-11-18 pass 13: D256 L128
  // 38.8 -> 38.3, bidirectional 50.9 -> 50.5 on top of the third issuer; D128 level). NOT used by default: the two-half statistics
  // change the bits, and a batch of samples (several tiles a CTA, no split) must give each sample's bits. env TRIMUL_K1W_TS = 1 (A/B)
  const char* ts_env = std::getenv("TRIMUL_K1W_TS");
  if (ts_env != nullptr && ts_env[0] == '1') so_np |= 128;
#define K1W_CASE(DD, PP) \
  if (D == DD && P == PP) { launch<DD, PP>(mx, mw, mp, mk, L, mu, rs, ga, be, tiles, g, st, (float)eps, so | so_np | (packed ? 16 : 0), n2, (float)rf3, qv, am, ng); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  K1W_CASE(64, 512) K1W_CASE(64, 256) K1W_CASE(64, 128) K1W_CASE(128, 512) K1W_CASE(128, 256) K1W_CASE(256, 1024) K1W_CASE(256, 512)
  K1W_CASE(384, 1536) K1W_CASE(384, 768) K1W_CASE(512, 2048) K1W_CASE(512, 1024)
#undef K1W_CASE
  TORCH_CHECK(false, "k1w_forward: unsupported (D, P) = (", D, ", ", P, ")");
}


// rf3 < 0: off. RoseTTAFold3's length factor f on the left projection (accumulator side): rf3 > 0 the constant f, rf3 == 0 f = bf16(1 / max(n, 1)),
// n the kept tokens of tokmask [L] (counted in the kernel); the planes of the left chunks are bf16(sigmoid(g) (f p)).
void k1w8_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                 c10::optional<torch::Tensor> tokmask, c10::optional<torch::Tensor> mean,
                 c10::optional<torch::Tensor> rstd, torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid,
                 double eps, torch::Tensor qinv, torch::Tensor amax, int64_t ngrp) {
  k1w_forward_impl(x, wl, wlg, wr, wrg, tokmask, mean, rstd, gamma, beta, planes, grid, eps, -1.0, qinv, amax, ngrp);
}
void k1w8_forward_rf3(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                     c10::optional<torch::Tensor> tokmask, c10::optional<torch::Tensor> mean,
                     c10::optional<torch::Tensor> rstd, torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid,
                     double eps, double rf3, torch::Tensor qinv, torch::Tensor amax, int64_t ngrp) {
  k1w_forward_impl(x, wl, wlg, wr, wrg, tokmask, mean, rstd, gamma, beta, planes, grid, eps, rf3, qinv, amax, ngrp);
}
