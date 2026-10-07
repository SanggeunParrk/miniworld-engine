// bo_front_tf32.cu -- K1 of the bias-only token DiT's three-kernel fp32 (TF32) inference step on sm_100a (MINIWORLD_BIAS_ONLY_DIT_INF3=1):
//
//   xa = LN(x) * s1[tok] + sh1[tok];   v | g = xa [Wv; Wg]^T     (tcgen05.mma kind::tf32; xa and v rounded to the nearest TF32)
//
// x = the block's input [M, 768] fp32 (the residual stream: K3 reads it again, no copy is written); TAB [T, 6, 768] one block's slice
// of the hoisted tables (0 gate1, 1 gate2, 2 s1, 3 s2, 4 sh1, 5 sh2, sigmoids applied; row stride tstride; tok = row % T); W = [Wv; Wg]
// [2 DA, 768] fp32 TF32-rounded; VG [M, 2 DA] out (v TF32-rounded: the core's MMA operand; g unrounded); XA [M, 768] scratch.
//
// Cluster of CL CTAs per 128-row tile (CL = 8, 6 or 4: the host picks the fewest rounds of resident clusters, measured 15 / 22 / 33,
// and then the largest CL): CTA c owns input columns [c 768/CL, ..) (KO = 3, 4 or 6 k-blocks) and v | g columns [c NV, ..)
// (NV = 2 DA / CL: CL 8 192 / 256, CL 6 256 (DA 768 only), CL 4 384 / 512). The LayerNorm needs whole rows: per-row
// (mean, M2) of the own columns are all-reduced through DSMEM (1 KB per CTA, Chan's combination). The GEMM needs the full-K xa:
//   * each CTA writes its own xa k-blocks (built in shared memory, SW128) to an L2-resident scratch XA [M, 768] by TMA store (CL 4 /
//     6: st.global, round 9, below), waits
//     for the stores to complete (cp.async.bulk.wait_group 0, fence.proxy.async.global), then ONE release (fence.acq_rel.cluster) and
//     CL relaxed remote arrivals on the cluster's `xaready` (count CL) -- round 2 used CL release.cluster arrivals, ~0.5 us each
//     (xaready 1.3-4 us after the stores, with that much skew between CTAs);
//   * every CTA's A producer waits `xaready` once (acquire.cluster), then streams all 24 xa k-blocks from L2 with TMA through a
//     NAR = 6-stage ring, beside the W ring -- a plain TMA-fed GEMM, no per-block round trip.
// Round 1 exchanged the blocks through a DSMEM push ring: ~1.4 us of round trips per k-block (33-35 us at every L); deleted.
// Round 3 also prefetches the next k-block's x (statistics) and x / s1 / sh1 (xa) loads under the current one (round 2: one L2 /
// HBM latency per k-block, serially: 4.7 us of statistics and 7 us of xa at CL 4).
// The own blocks are staged in ring slots 0 .. KO - 1 = ring positions 0 .. KO - 1, and the GEMM takes them FIRST (round 4): the
// row workers arrive on their `afull` themselves, so the MMAs run on them while the stores complete and the cluster signal travels;
// the producer loads the other 24 - KO k-blocks (positions KO ..) after `xaready` (no CTA signals before its own stores have
// completed, so the producer never overwrites a staging tile still being stored). Round 4 also issues all of the statistics pass's x
// loads at once (CL 8 keeps them in registers for the xa pass, which then loads only s1 / sh1), and loads the weights evict_last.
// GEMM k-block order (the W producer's): KO c .. KO c + KO - 1, then 0 .. 23 without them.
//
// Warps: 0 W producer (TMA, lane 0); 1 MMA (whole warp waits, elect_one() issues); 2 TMEM allocator; 3 A producer (lane 0); 4-11 row
// workers (statistics, xa, its TMA stores) and the epilogue (TMEM 32x32b -> per-warp 32 x 32 transpose tile -> st.global.v4).
//   shared memory  A ring NAR = 6 x 16 KB (staging of the own blocks first) | W ring NWS x [WROWS][32] fp32 (CL 8: 5 x 24 KB / 3 x 32 KB;
//                  CL 4: 5 x 24 KB / 3 x 32 KB) | stats exchange CL KB | own stats 1 KB | barriers
//   TMEM           the [128][NV] accumulator (256 or 512 columns allocated)
// Per CTA and tile (DA 768, CL 8): 37.7 MFLOP (7.8 us at 4.86 TF/s per SM); TMA in: xa 384 KB + W 576 KB (7.8 us at ~123 GB/s).
// PDL: pdl_launch() after setup; the row workers pdl_wait() before reading x and before any store (VG / XA may still be read by the
// previous block's kernels); the W producer loads before the wait. Bit-identical reruns: fixed-order reductions, no atomics.
// Deleted after round 7's A/B: the Wvg prefetch into L2 (4-14 us slower in the whole step).
// Round 9 A/B (one switch each, whole step): at CL 4 / 6 (FXSTG) the own xa blocks go to XA with st.global from the registers that
// build the staging block (four whole 128-byte rows per warp instruction), then one release -- instead of TMA store +
// cp.async.bulk.wait_group 0 (2.3 us from the last store to xaready at CL 4, L768): front 25.1 -> 24.5 us at L768, whole step
// -0.4 .. -1.7 us at L512-768. CL 8 keeps the TMA store (+0.3 / +1.2 us at L256 / L384 with st.global). Not kept: the xa pass's
// first loads under the statistics exchange (+0.4 .. -0.8 us, and later statistics), launch_dependents as the first instruction.
// -DTRACE: %globaltimer per role (events below; bench_scripts/bo32_inf3_trace.py).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef CL
#define CL 8
#endif
#ifndef DATT
#define DATT 768
#endif
static_assert(CL == 4 || CL == 6 || CL == 8, "cluster of 4, 6 or 8");
#define FXSTG (CL != 8)                                        // the own xa blocks to L2 by st.global (CL 8: TMA store)
static_assert(CL != 6 || DATT == 768, "CL 6: 2 DA / 6 columns per CTA need DA 768");
static_assert(DATT == 768 || DATT == 1024, "768 or 1024 attention channels");

DEVI uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI float4 u2f4(uint4 u) { return make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
DEVI float4 ldg4(const float* p) { return u2f4(ldg128(p)); }
DEVI float sum8(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}
DEVI void tmem_ld32w(uint32_t taddr, uint32_t (&r)[32]) {    // tcgen05.ld fused with its wait (one asm statement)
  asm volatile("{\n\ttcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n\t"
               "tcgen05.wait::ld.sync.aligned;\n\t}"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
                 "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]),
                 "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
                 "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr) : "memory");
}
DEVI void cluster_sync_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned; barrier.cluster.wait.aligned;" ::: "memory"); }
DEVI uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
DEVI void push_bulk(uint32_t dst_local, uint32_t src, uint32_t bytes, uint32_t bar_local, uint32_t rank) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(mapa(dst_local, rank)), "r"(src), "r"(bytes), "r"(mapa(bar_local, rank)) : "memory");
}
// generic / async-proxy ordering of global memory: the scratch is written by TMA stores and read by other CTAs' TMA loads
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI void fence_acq_rel_cluster() { asm volatile("fence.acq_rel.cluster;" ::: "memory"); }

// -DTRACE: g_trace[16 CTAs][2048] %globaltimer ns (the tail's layout). Markers: 0 start | row workers (rw 0): 1 pdl_wait passed,
// 2 stats pushed, 3 xch seen, 4 own xa stores issued, 5 xaready signalled (stores complete), 10 accfull seen, 11 epilogue done |
// A producer: 16 xaready seen. Per ring position i: 256 + i afull seen by the MMA warp, 512 + i load issued, 768 + i the MMA warp
// began waiting; per W slot wj: 1024 + wj seen, 1280 + wj issued, 1536 + wj began waiting.
#ifdef TRACE
__device__ unsigned long long g_trace[16 * 2048];
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TR(i) do { if (blockIdx.x < 16 && (i) < 2048) g_trace[blockIdx.x * 2048 + (i)] = gtime(); } while (0)
#else
#define TR(i) do { } while (0)
#endif
constexpr int EV_AS = 256, EV_AI = 512, EV_AW = 768, EV_WS = 1024, EV_WI = 1280, EV_WW = 1536;

constexpr int D = 768, QM = 128, NK = D / 32;
constexpr int NC1 = D / CL, KO = NC1 / 32;
constexpr int NV = 2 * DATT / CL;
constexpr int WROWS = NV <= 256 ? NV : NV / 2, NWH = NV / WROWS;
constexpr int SLOT = QM * 128, NAR = 6;
constexpr int WSLOT = WROWS * 128;
constexpr int O_A = 0, O_W = NAR * SLOT;
constexpr int MISC = CL * 1024 + 1024 + 512;
constexpr int NWS = (232448 - O_W - MISC) / WSLOT;
constexpr int O_RED = O_W + NWS * WSLOT, O_OWN = O_RED + CL * 1024, O_BAR = O_OWN + 1024, SMEM_BYTES = O_BAR + 512;
static_assert(NWS >= 3 && SMEM_BYTES <= 232448, "shared memory");
static_assert(O_W % 1024 == 0 && WSLOT % 1024 == 0, "1 KB alignment of the 128-B-swizzled tiles");
static_assert(KO * CL == NK && KO <= NAR && NV % 64 == 0 && WROWS % 16 == 0 && WROWS <= 256, "tiling");
constexpr uint32_t TCOLS = NV <= 256 ? 256 : 512;
#ifndef FRONT_KEEPX
#define FRONT_KEEPX 1                                          // -DFRONT_KEEPX=0: reload x in the xa pass at CL 8 too
#endif
constexpr bool KEEPX = FRONT_KEEPX && KO <= 3;                 // CL 8: the statistics pass's x stays in registers for the xa pass
DEVI int kb_of(int t, int c) { if (t < KO) return KO * c + t; const int j = t - KO; return j < KO * c ? j : j + KO; }
constexpr uint32_t I_W = idesc_tf32(QM, WROWS);

struct Bars {
  uint64_t afull[NAR], aempty[NAR], wfull[NWS], wempty[NWS], xch, xaready, accfull;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

extern "C" __global__ void __launch_bounds__(384, 1)
bo_front_tf32_sm100(const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mxa, const float* __restrict__ X,
                       const float* __restrict__ TAB, float* __restrict__ VG, float* __restrict__ XA, int T, int tstride, int xastride,
                       float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t c = cluster_rank();
  const int m0 = (int)(blockIdx.x / CL) * QM, tr0 = m0 % T;

  if (tid == 0) {
    for (int s = 0; s < NAR; ++s) { mbar_init(&B.afull[s], 1); mbar_init(&B.aempty[s], 1); }
    for (int s = 0; s < NWS; ++s) { mbar_init(&B.wfull[s], 1); mbar_init(&B.wempty[s], 1); }
    mbar_init(&B.xch, 1);
    mbar_init(&B.xaready, CL);                                 // every cluster CTA's own xa blocks are in L2
    mbar_init(&B.accfull, 1);
    mbar_expect_tx(&B.xch, CL * 1024);
    fence_barrier_init();
    prefetch_map(&mw); prefetch_map(&mxa);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TCOLS); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();
  if (tid == 0) TR(0);

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ W producer (k-block order 0..23)
    if (lane == 0) {
      int wj = 0;
      const uint64_t keep = pol_evict_last();
      for (int t = 0; t < NK; ++t)
        for (int h = 0; h < NWH; ++h, ++wj) {
          const int ws = wj % NWS;
          if (wj >= NWS) mbar_wait(&B.wempty[ws], ((wj / NWS) - 1) & 1);
          mbar_expect_tx(&B.wfull[ws], WSLOT);
          tma_load_2d_h(su + O_W + ws * WSLOT, &mw, &B.wfull[ws], 32 * kb_of(t, (int)c), (int)c * NV + h * WROWS, keep);
          TR(EV_WI + wj);
        }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ A producer
    if (lane == 0) {
      mbar_wait_cl(&B.xaready, 0);                             // all CL CTAs' xa stores complete
      TR(16);
      fence_proxy_async_global();
      for (int i = KO; i < NK; ++i) {                          // positions 0 .. KO - 1: the own blocks, staged by the row workers
        const int slot = i % NAR;
        if (i >= NAR) mbar_wait(&B.aempty[slot], ((i / NAR) - 1) & 1);
        mbar_expect_tx(&B.afull[slot], SLOT);
        tma_load_2d(su + O_A + slot * SLOT, &mxa, &B.afull[slot], 32 * kb_of(i, (int)c), m0);
        TR(EV_AI + i);
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    int wj = 0;
    for (int i = 0; i < NK; ++i) {
      const int slot = i % NAR;
      if (lane == 0) TR(EV_AW + i);
      mbar_wait(&B.afull[slot], (i / NAR) & 1);
      tc_fence_after();
      if (lane == 0) TR(EV_AS + i);
      const uint64_t da = desc_k128(su + O_A + slot * SLOT);
      for (int h = 0; h < NWH; ++h, ++wj) {
        const int ws = wj % NWS;
        if (lane == 0) TR(EV_WW + wj);
        mbar_wait(&B.wfull[ws], (wj / NWS) & 1);
        tc_fence_after();
        if (lane == 0) TR(EV_WS + wj);
        if (elect_one()) {
          const uint64_t dw = desc_k128(su + O_W + ws * WSLOT);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ss_tf32(tmem + (uint32_t)(h * WROWS), da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_W, (i | ks) ? 1u : 0u);
          tc_commit(&B.wempty[ws]);
        }
        __syncwarp();
      }
      if (elect_one()) {
        tc_commit(&B.aempty[slot]);
        if (i == NK - 1) tc_commit(&B.accfull);
      }
      __syncwarp();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ row workers
    const int rw = tid - 128, w8 = rw >> 5, rsub = lane >> 3, e = lane & 7;
    pdl_wait();                                                // x: the previous kernel's output; VG / XA: read by earlier kernels
    if (rw == 0) TR(1);
    const float* xb = X + (size_t)m0 * D + (size_t)c * NC1 + 4 * e;
    const float* tb = TAB + (size_t)tr0 * tstride + (size_t)c * NC1 + 4 * e;
    float mu[4], m2[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) { mu[s] = 0.f; m2[s] = 0.f; }
    // ---- statistics of the own columns (per 32-column block, merged by Chan), all-reduced through DSMEM
    float4 xk[KO][4];                                          // every own k-block's x, all loads in flight at once (KO 6: 96 registers)
#pragma unroll
    for (int j = 0; j < KO; ++j)
#pragma unroll
      for (int s = 0; s < 4; ++s) xk[j][s] = ldg4(xb + (size_t)(16 * w8 + 4 * s + rsub) * D + 32 * j);
#pragma unroll
    for (int j = 0; j < KO; ++j) {
      const float n0 = 32.f * j, nn = n0 + 32.f;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const float4 v = xk[j][s];
        const float bm = sum8((v.x + v.y) + (v.z + v.w)) * (1.f / 32);
        const float a0 = v.x - bm, a1 = v.y - bm, a2 = v.z - bm, a3 = v.w - bm;
        const float bq = sum8((a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3));
        const float dl = bm - mu[s];
        mu[s] += dl * (32.f / nn);
        m2[s] += bq + dl * dl * (n0 * 32.f / nn);
      }
    }
    float4 xv[4], sv[4], hv[4], xn[4], sn[4], hn[4];          // the xa pass's k-block u inputs, and u + 1's in flight
    auto load = [&](int u, float4 (&xo)[4], float4 (&so)[4], float4 (&ho)[4]) {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int r = 16 * w8 + 4 * s + rsub;
        if constexpr (!KEEPX) xo[s] = ldg4(xb + (size_t)r * D + 32 * u);
        so[s] = ldg4(tb + (size_t)r * tstride + 2 * D + 32 * u);
        ho[s] = ldg4(tb + (size_t)r * tstride + 4 * D + 32 * u);
      }
    };
    float* own = reinterpret_cast<float*>(sm + O_OWN);
    if (e == 0) {
#pragma unroll
      for (int s = 0; s < 4; ++s) { own[16 * w8 + 4 * s + rsub] = mu[s]; own[QM + 16 * w8 + 4 * s + rsub] = m2[s]; }
    }
    fence_proxy_async();
    named_bar_sync(1, 256);
    if (rw == 0) {
      for (int p = 0; p < CL; ++p) push_bulk(su + O_RED + c * 1024, su + O_OWN, 1024, smem_u32(&B.xch), (uint32_t)p);
      TR(2);
    }
    mbar_wait_cl(&B.xch, 0);
    if (rw == 0) TR(3);
    const float* red = reinterpret_cast<const float*>(sm + O_RED);
    float rs[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int r = 16 * w8 + 4 * s + rsub;
      float mean = 0.f;
#pragma unroll
      for (int p = 0; p < CL; ++p) mean += red[p * 2 * QM + r];
      mean *= 1.f / CL;
      float M2 = 0.f;
#pragma unroll
      for (int p = 0; p < CL; ++p) { const float d = red[p * 2 * QM + r] - mean; M2 += red[p * 2 * QM + QM + r] + (float)NC1 * d * d; }
      mu[s] = mean;
      rs[s] = rsqrtf(M2 * (1.f / D) + eps);
    }
    // ---- own xa k-block u -> ring slot u = ring position u (SW128 rows) -> afull[u] (the MMAs start on it) and a TMA store to XA
    // (CL 4 / 6: the same values by st.global instead)
    load(0, xv, sv, hv);
#if FXSTG
    float* const xab = XA + (size_t)(m0 + 16 * w8 + rsub) * xastride + (size_t)c * NC1 + 4 * e;   // row 16 w8 + rsub, chunk e
#endif
#pragma unroll
    for (int u = 0; u < KO; ++u) {
      const uint32_t dst = su + O_A + (uint32_t)u * SLOT;
      if (u + 1 < KO) load(u + 1, xn, sn, hn);
      if constexpr (KEEPX) {
#pragma unroll
        for (int s = 0; s < 4; ++s) xv[s] = xk[u][s];
      }
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int r = 16 * w8 + 4 * s + rsub;
        const uint4 o =
            make_uint4(tf32r(fmaf((xv[s].x - mu[s]) * rs[s], sv[s].x, hv[s].x)), tf32r(fmaf((xv[s].y - mu[s]) * rs[s], sv[s].y, hv[s].y)),
                       tf32r(fmaf((xv[s].z - mu[s]) * rs[s], sv[s].z, hv[s].z)), tf32r(fmaf((xv[s].w - mu[s]) * rs[s], sv[s].w, hv[s].w)));
        sts128(dst + sw128((uint32_t)r, (uint32_t)e), o);
#if FXSTG
        stg128(xab + (size_t)(4 * s) * xastride + 32 * u, o);
#endif
      }
      fence_proxy_async();                                     // generic stores -> the MMA and the TMA store (async proxy) read them
#if FXSTG
      if (u == KO - 1) fence_proxy_async_global();             // this thread's XA stores -> the peers' TMA loads (after the release)
#endif
      named_bar_sync(1, 256);
      if (rw == 0) {
        mbar_arrive(&B.afull[u]);                              // ring position u: this CTA's own block, ready for the MMA warp
#if !FXSTG
        tma_store_2d(&mxa, dst, (int)c * NC1 + 32 * u, m0);
        tma_store_commit();
#endif
      }
#pragma unroll
      for (int s = 0; s < 4; ++s) { if constexpr (!KEEPX) xv[s] = xn[s]; sv[s] = sn[s]; hv[s] = hn[s]; }
    }
    if (rw == 0) {
      TR(4);
#if !FXSTG
      tma_store_wait0();                                       // the own blocks are in global memory (and the staging is free)
      fence_proxy_async_global();
#endif
      fence_acq_rel_cluster();                                 // one release for the CL relaxed arrivals
      for (int p = 0; p < CL; ++p) mbar_arrive_remote_relaxed(&B.xaready, (c + (uint32_t)p) % CL);
      TR(5);
    }
    // ---- epilogue: VG[:, c NV ..] = accumulator through per-warp transpose tiles (v rounded to TF32)
    mbar_wait(&B.accfull, 0);
    tc_fence_after();
    if (rw == 0) TR(10);
    const int q4 = warp & 3, hh = (warp - 4) >> 2;
    const uint32_t trow = tmem + ((uint32_t)(32 * q4) << 16);
    const uint32_t tile = su + O_A + (uint32_t)(warp - 4) * 4096;   // the A ring is idle: every block consumed, stores complete
    constexpr int NCH = NV / 64;
#pragma unroll 1
    for (int k = 0; k < NCH; ++k) {
      const int ch = hh * NCH + k, gc = (int)c * NV + 32 * ch;
      uint32_t v[32];
      tmem_ld32w(trow + (uint32_t)(32 * ch), v);
      if (gc < DATT) {
#pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = tf32r(__uint_as_float(v[i]));
      }
#pragma unroll
      for (int q = 0; q < 8; ++q) sts128(tile + (uint32_t)lane * 128 + (((uint32_t)q ^ (lane & 7)) << 4),
                                         make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint32_t rr = 4 * i + (lane >> 3), qq = lane & 7;
        const uint4 val = lds128(tile + rr * 128 + ((qq ^ (rr & 7)) << 4));
        stg128(VG + (size_t)(m0 + 32 * q4 + (int)rr) * (2 * DATT) + gc + 4 * qq, val);
      }
      __syncwarp();
    }
    if (rw == 0) TR(11);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}
