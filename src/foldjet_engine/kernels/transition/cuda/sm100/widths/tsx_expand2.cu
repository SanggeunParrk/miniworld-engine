// tsx_expand2.cu -- the small-M Transition expand + SwiGLU (tsx_expand.cu's place in the tsx_ln -> expand -> squeeze chain) on a CTA
// PAIR (cta_group::2), sm_100a, bf16, inference (-DDIM=<D> -DHID=<H> -DUH=<hidden units per CTA: 16 / 32 / 64>).
// One 2-CTA cluster per (128-row tile, 2 UH-unit hidden chunk): tiles x H / (2 UH) x 2 CTAs. The leader issues M = 128, N = 4 UH
// products (cta_group::2): CTA r holds rows [64 r, 64 r + 64) of the tile's xn and its own B half [Wa_r; Wb_r] (UH rows each, units
// [h0, h0 + UH)), and receives in its tensor memory its 64 rows of the WHOLE product [a_0 | b_0 | a_1 | b_1], folded over the 128 lanes
// (measured with tsx_pairtest.cu). So each SM takes half the tile's xn and half the pair's weights, and its accumulator is complete over K:
// no K split, no partial exchange (tsx_expand.cu -DSK=2 spends ~1 us of a [1, 128, 768] call exchanging partials in distributed shared memory).
// The weight K-blocks are loaded before griddepcontrol.wait (parameters), lane s of the producer warp issuing ring stage s.
// -DFOLD=1 (N5): the LayerNorm folded into this kernel (no tsx_ln launch; the pair-chain analogue of tsx_expand.cu -DFOLD): the RAW x is the
// A operand (resident, 64 rows a CTA, one barrier a K-block), Wp = bf16(W gamma) the B operand (its own ring), the row statistics come from
// the x tile in shared memory as its K-blocks land (shifted single pass), and the epilogue applies the rank-1 terms in fp32:
// a = rs (acc - mean u_n) + v_n with u_n = sum_k Wp_nk and v_n = sum_k beta_k W_nk (aug [H][4] fp32 = u_a v_a u_b v_b: a weight form).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include "tsx_store.cuh"
using namespace s100;

#ifndef DIM
#define DIM 768
#endif
#ifndef HID
#define HID (2 * DIM)
#endif
#ifndef UH
#define UH 32
#endif
#ifndef FOLD
#define FOLD 0
#endif
#ifndef SIGT
#define SIGT 1                                                  // FOLD epilogue: packed f32x2 + the one-MUFU tanh sigmoid (tsx_expand.cu keeps sigmoid_kit)
#endif
#ifndef FABL
#define FABL 0                                                  // timing ablations of the fold (wrong results): 1 no peer relay, 2 no statistics pass
#endif
constexpr int D_ = DIM, H_ = HID, ROWS = 128, HR = 64, NKB = D_ / 64, NP = H_ / (2 * UH);
static_assert((UH == 16 || UH == 32 || UH == 64) && H_ % (2 * UH) == 0 && D_ % 64 == 0, "16- / 32- / 64-unit halves, whole K-blocks");
constexpr int AKB = HR * 128;                                   // one K-block of this CTA's 64 xn rows, 8 KB
constexpr int WKB = 2 * UH * 128;                                // one K-block of [Wa_r; Wb_r]
#ifndef KBX
#define KBX 1                                                   // K-blocks per ring stage: the xn rows of a stage in ONE 3-D TMA box
#endif
constexpr int NS = NKB / KBX, AST = KBX * AKB, WST = KBX * WKB; // ring stages of the whole K, bytes of a stage
static_assert(NKB % KBX == 0, "whole stages");
#if FOLD
static_assert(KBX >= 2 && NKB % 2 == 0 && NKB <= 16, "the fold: >= 2 K-blocks a stage (one 3-D box each of x and of the interleaved Wp), the K-blocks in two halves");
constexpr int AF = NKB * AKB;                                   // the resident x tile of this CTA
constexpr int AUGB = 2 * UH * 16;                               // aug rows of the pair's units (u_a v_a u_b v_b fp32)
constexpr int CAP = (232448 - 1024 - AF - AUGB - 1024 - 512 - 1024) / WST;
constexpr int STG = NS < CAP ? NS : CAP;
static_assert(STG >= 2 && STG <= 32, "ring stages (one issuing lane each)");
constexpr int O_A = 0, O_W = AF, O_AUG = O_W + STG * WST, O_EX = O_AUG + AUGB, O_ST = O_EX + 1024, O_BAR = O_ST + 512, SMEM_BYTES = O_BAR + 1024;
#else
constexpr int CAP = (232448 - 1024) / (AST + WST);
constexpr int STG = NS < CAP ? NS : CAP;
static_assert(STG >= 1 && STG <= 32, "ring stages (one issuing lane each)");
constexpr int O_A = 0, O_W = STG * AST, O_BAR = O_W + STG * WST, SMEM_BYTES = O_BAR + 256;
#endif
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t TCOLS = 2 * UH < 32 ? 32 : 2 * UH;
#ifndef H16
#define H16 0                                                   // part 17: fp32 inference on the fp16 tensor cores (tsx_expand.cu -DH16's
#endif                                                          // scheme: xn / Wa16 / Wb16 fp16, [a | b] sc[1], h = fp16(silu(a) b sc[2]))
static_assert(!(FOLD && H16), "the fold is the bf16 chain's");
#if H16
constexpr uint32_t IDESC = (1u << 4) | ((uint32_t)((4 * UH) >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);   // fp16 A / B, fp32 D
typedef __half ACT;
#else
constexpr uint32_t IDESC = idesc_bf16(128, 4 * UH);
typedef __nv_bfloat16 ACT;
#endif
constexpr int NG = UH > 32 ? 2 : 1, EC = UH / NG;                 // epilogue warpgroups (column halves), units per group

struct Bars {
#if FOLD
  uint64_t full[STG], empty[STG], xfull[NKB], xready[NKB], mma_done, augfull;
#else
  uint64_t full[STG], empty[STG], mma_done;
#endif
  uint32_t tmem;
};
#if FOLD
static_assert(sizeof(Bars) <= 1024, "barrier block");
#else
static_assert(sizeof(Bars) <= 256, "barrier block");
#endif

// 3-D TMA (the kb_box tensor maps: element, row, K-block) into this CTA, completing on the pair leader's barrier
DEVI void tma_load_3d_2sm(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar) & 0xFEFFFFFFu), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
DEVI void load_a(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int st, int row) {
  if (KBX == 1) tma_load_2d_2sm(dst, m, bar, st * 64, row); else tma_load_3d_2sm(dst, m, bar, 0, row, st * KBX);
}
#if FOLD
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
#endif

template <int N> DEVI void tld(uint32_t taddr, uint32_t (&r)[N]) {
  if constexpr (N == 32) tmem_ld32(taddr, r); else tmem_ld16(taddr, r);
}

// maps: xn [M][D] (box 64 x 64); wa, wb [H][D] (box 64 x UH). h [M][H] bf16.
extern "C" __global__ void __launch_bounds__(128 + 128 * NG, 1)
transition_sx_expand2(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mwa,
                      const __grid_constant__ CUtensorMap mwb, ACT* __restrict__ hs
#if H16
                      , const float* __restrict__ sc
#endif
#if FOLD
                      , const float* __restrict__ aug, float eps, const uint8_t* __restrict__ xg
#endif
                      TSX_STAMP_ARG) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x >> 1, rank = (int)cluster_rank();
  const bool leader = rank == 0;
  const int tile = pair / NP, j = pair % NP, row0 = tile * ROWS, arow = row0 + rank * HR, h0 = j * 2 * UH + rank * UH;
  if (tid == 0) TSX_ST(0);

  if (tid == 0) {
    for (int s = 0; s < STG; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.mma_done, 1);
#if FOLD
    for (int s = 0; s < NKB; ++s) { mbar_init(&B.xfull[s], 1); mbar_init(&B.xready[s], 1); }
    mbar_init(&B.augfull, 1);
#endif
    fence_barrier_init();
    prefetch_map(&mxn); prefetch_map(&mwa); prefetch_map(&mwb);
#if FOLD
    if (!(FABL & 4))                                           // x is produced by the launch before this one: only an L2 prefetch is legal before griddepcontrol.wait
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(xg + (size_t)arow * (D_ * 2)), "r"(HR * D_ * 2) : "memory");
    mbar_expect_tx(&B.augfull, AUGB);                          // the pair's unit rows of the rank-1 terms (a weight form, not produced by an earlier launch)
    bulk_g2s(su + O_AUG, aug + (size_t)(j * 2 * UH) * 4, AUGB, &B.augfull);
#endif
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), TCOLS); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();                                      // the peer's TMA completes on the leader's barriers
  tc_fence_after();
  const uint32_t tmem = B.tmem;
#if FOLD
  // the interleaved Wp ([H / UH][Wa_c; Wb_c: 2 UH rows][D], mwa; mwb unused): this CTA's [Wa_r; Wb_r] of K-blocks st KBX .. st KBX + KBX - 1 in ONE 3-D box
  auto load_w = [&](int st) { tma_load_3d_2sm(su + O_W + (st % STG) * WST, &mwa, &B.full[st % STG], 0, 2 * h0, st * KBX); };
#else
  auto load_w = [&](int st) {                                  // ring stage st % STG: the weights of K-blocks st KBX .. st KBX + KBX - 1
    const uint32_t w = su + O_W + (st % STG) * WST;
#pragma unroll
    for (int i = 0; i < KBX; ++i) {
      tma_load_2d_2sm(w + i * WKB, &mwa, &B.full[st % STG], (st * KBX + i) * 64, h0);
      tma_load_2d_2sm(w + i * WKB + UH * 128, &mwb, &B.full[st % STG], (st * KBX + i) * 64, h0);
    }
  };
#endif
#if FOLD
  if (warp == 0 && lane < STG) {                               // the weights of ring stage `lane` (they do not depend on the previous launch)
    if (leader) mbar_expect_tx(&B.full[lane], 2 * WST);
    load_w(lane);
  }
  pdl_wait();                                                  // x is complete and visible
  if (tid == 0) TSX_ST(1);
  pdl_launch();
  if (warp == 3 && lane < NS) {                                // the resident x tile: stage `lane` (KBX K-blocks, one 3-D box), the leader's barrier counts both CTAs' bytes
    if (leader) mbar_expect_tx(&B.xfull[lane], 2 * AST);
    load_a(su + O_A + lane * AST, &mxn, &B.xfull[lane], lane, arow);
    if (leader) {                                              // the peer's threads cannot wait on the leader's barrier: tell it when K-block `lane` has landed
      mbar_wait(&B.xfull[lane], 0);
      if (!(FABL & 1)) mbar_arrive_remote(&B.xready[lane], 1);
    }
  }
#else
  if (warp == 0 && lane < STG) {                               // the weights of ring stage `lane` (they do not depend on the LayerNorm)
    if (leader) mbar_expect_tx(&B.full[lane], 2 * (AST + WST));
    load_w(lane);
  }
  pdl_wait();                                                  // xn is complete and visible
  if (tid == 0) TSX_ST(1);
  pdl_launch();
#endif

  if (warp == 0) {
#if FOLD
    if (lane < STG)
      for (int st = lane + STG; st < NS; st += STG) {
        mbar_wait(&B.empty[lane], ((st / STG) - 1) & 1);
        if (leader) mbar_expect_tx(&B.full[lane], 2 * WST);
        load_w(st);
      }
#else
    if (lane < STG)
      for (int st = lane; st < NS; st += STG) {
        if (st >= STG) {
          mbar_wait(&B.empty[lane], ((st / STG) - 1) & 1);
          if (leader) mbar_expect_tx(&B.full[lane], 2 * (AST + WST));
          load_w(st);
        }
        load_a(su + O_A + lane * AST, &mxn, &B.full[lane], st, arow);
      }
#endif
    if (lane == 0) TSX_ST(2);
  } else if (warp == 1) {
    if (leader) {
#pragma unroll 1
      for (int st = 0; st < NS; ++st) {
        const int s = st % STG;
        mbar_wait(&B.full[s], (st / STG) & 1);
#if FOLD
        mbar_wait(&B.xfull[st], 0);
#endif
        if (lane == 0 && st == 0) TSX_ST(3);
        if (lane == 0 && st == NS - 1) TSX_ST(4);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int i = 0; i < KBX; ++i) {
#if FOLD
            const uint64_t da = desc_k128(su + O_A + st * AST + i * AKB), db = desc_k128(su + O_W + s * WST + i * WKB);
#else
            const uint64_t da = desc_k128(su + O_A + s * AST + i * AKB), db = desc_k128(su + O_W + s * WST + i * WKB);
#endif
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ss2(tmem, da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), IDESC, (st > 0 || i > 0 || ks > 0) ? 1u : 0u);
          }
          if (NS > STG) tc_commit2_mc(&B.empty[s], 3);
          if (st == NS - 1) tc_commit2_mc(&B.mma_done, 3);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------ SwiGLU, one row per thread (all 128
    // rows of the tile: the pair product leaves every row of this CTA's units here); warps 4-7 units [0, EC), 8-11 [EC, 2 EC)
    // D layout of an M = 128 pair product (measured, tsx_pairtest.cu): CTA r holds ITS 64 rows x all N, folded: lanes 0-63 the leader's
    // B columns [a_0 | b_0], lanes 64-127 the peer's [a_1 | b_1] (same rows, same column addresses)
    const int g = (warp - 4) >> 2, q = warp & 3, r = (q & 1) * 32 + lane, hb = j * 2 * UH + (q >> 1) * UH;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    constexpr int ES = EC > 32 ? 32 : EC, NW = ES / 2;
#if H16
    const float s1 = __ldg(sc + 1), s2 = __ldg(sc + 2);
#endif
#if FOLD
    // statistics of row r (of this CTA's 64) from the raw x tile in shared memory as its K-blocks land: shifted single pass (shift = the row's
    // first element: sum (x - s), sum (x - s)^2), the two halves of the K-blocks on warps 4-5 + 6-7 (threads 0-63 / 64-127 of warps 4-7),
    // exchanged through shared memory. The peer reads its x through the leader's relay (xready), the leader through xfull.
    float mean_ = 0.f, rs_ = 1.f;
    {
      auto xwait = [&](int kb) { if (leader) mbar_wait(&B.xfull[kb / KBX], 0); else if (!(FABL & 1)) mbar_wait_cl(&B.xready[kb / KBX], 0); };
      const uint32_t xb = su + O_A;
      float* ex = reinterpret_cast<float*>(sm + O_EX);
      float2* stt = reinterpret_cast<float2*>(sm + O_ST);
      if (warp < 8 && warp >= 4) {
        const int t = tid - 128, rr = t & 63, hf = t >> 6;
        xwait(0);
        const uint4 t0 = lds128(xb + sw128(rr, 0));
        const float shift = __uint_as_float(t0.x << 16);
        const f2 nsh = mk2(-shift, -shift);
        f2 sa = mk2(0.f, 0.f), qa = mk2(0.f, 0.f);
#pragma unroll
        for (int kk = 0; kk < NKB / 2; ++kk) {
          const int kb = hf * (NKB / 2) + kk;
          xwait(kb);
#pragma unroll
          for (int qq = 0; qq < (FABL & 2 ? 0 : 8); ++qq) {
            const uint4 tt = lds128(xb + kb * AKB + sw128(rr, qq));
            const uint32_t w4[4] = {tt.x, tt.y, tt.z, tt.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const f2 d = add2(mk2u(w4[k] << 16, w4[k] & 0xffff0000u), nsh);
              sa = add2(sa, d);
              qa = fma2(d, d, qa);
            }
          }
        }
        ex[(hf * 64 + rr) * 2] = lo2(sa) + hi2(sa);
        ex[(hf * 64 + rr) * 2 + 1] = lo2(qa) + hi2(qa);
        named_bar_sync(2, 128);
        if (hf == 0) {
          const float sd = ex[rr * 2] + ex[(64 + rr) * 2], sq = ex[rr * 2 + 1] + ex[(64 + rr) * 2 + 1];
          const float md = sd * (1.f / D_), mean = shift + md;
          const float var = fmaxf(sq * (1.f / D_) - md * md, 0.f);
          stt[rr] = make_float2(mean, rsqrtf(var + eps));
        }
      }
      named_bar_sync(3, 128 * NG);                             // every epilogue thread: the statistics are in shared memory
      if (tid == 128) TSX_ST(9);
      const float2 stv = stt[r];
      mean_ = stv.x; rs_ = stv.y;
      mbar_wait(&B.augfull, 0);
    }
#endif
    mbar_wait(&B.mma_done, 0);
    if (warp == 4 && lane == 0) TSX_ST(5);
    tc_fence_after();
#pragma unroll
    for (int cc = 0; cc < EC; cc += ES) {
      uint32_t av[ES], bv[ES], hp[NW];
      tld<ES>(trow + g * EC + cc, av);
      tld<ES>(trow + UH + g * EC + cc, bv);
      tmem_wait_ld();
#pragma unroll
      for (int k = 0; k < NW; ++k) {
#if H16
        const float a0 = __uint_as_float(av[2 * k]) * s1, a1 = __uint_as_float(av[2 * k + 1]) * s1;
        hp[k] = f2h2(a0 * sigmoid_kit(a0) * (__uint_as_float(bv[2 * k]) * s1) * s2, a1 * sigmoid_kit(a1) * (__uint_as_float(bv[2 * k + 1]) * s1) * s2);
#elif FOLD
        const int ui = (q >> 1) * UH + g * EC + cc + 2 * k;
        const uint4 c0 = lds128(su + O_AUG + ui * 16), c1 = lds128(su + O_AUG + ui * 16 + 16);
#if SIGT
        // packed fp32x2 arithmetic, the sigmoid through one MUFU op (0.5 tanh(a / 2) + 0.5: the epilogue's 16 K MUFU ops a CTA at the two-op form are ~0.5 us)
        const f2 ta = fma2(mk2(rs_, rs_), fma2(mk2(-mean_, -mean_), mk2u(c0.x, c1.x), mk2u(av[2 * k], av[2 * k + 1])), mk2u(c0.y, c1.y));
        const f2 tb = fma2(mk2(rs_, rs_), fma2(mk2(-mean_, -mean_), mk2u(c0.z, c1.z), mk2u(bv[2 * k], bv[2 * k + 1])), mk2u(c0.w, c1.w));
        const f2 h2 = mul2(mul2(ta, sigmoid2(ta)), tb);
        hp[k] = pack_bf16(lo2(h2), hi2(h2));
#else
        const float a0 = fmaf(rs_, fmaf(-mean_, __uint_as_float(c0.x), __uint_as_float(av[2 * k])), __uint_as_float(c0.y));
        const float a1 = fmaf(rs_, fmaf(-mean_, __uint_as_float(c1.x), __uint_as_float(av[2 * k + 1])), __uint_as_float(c1.y));
        const float b0 = fmaf(rs_, fmaf(-mean_, __uint_as_float(c0.z), __uint_as_float(bv[2 * k])), __uint_as_float(c0.w));
        const float b1 = fmaf(rs_, fmaf(-mean_, __uint_as_float(c1.z), __uint_as_float(bv[2 * k + 1])), __uint_as_float(c1.w));
        hp[k] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_kit(a1) * b1);
#endif
#else
        const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
        hp[k] = pack_bf16(a0 * sigmoid_kit(a0) * __uint_as_float(bv[2 * k]), a1 * sigmoid_kit(a1) * __uint_as_float(bv[2 * k + 1]));
#endif
      }
      store_t(hs + (size_t)arow * H_ + hb + g * EC + cc + store_t_off<NW>(r, lane, H_, 2), H_, hp, lane);
    }
    if (warp == 4 && lane == 0) TSX_ST(7);
    tc_fence_before();
  }
  __syncthreads();
  cluster_sync_relaxed();                                      // the leader's products read the peer's shared memory until mma_done
  if (tid == 0) TSX_ST(8);
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, TCOLS); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
