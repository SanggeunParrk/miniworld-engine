// bo_front_bf16.cu -- K1 of the bias-only token DiT's bf16-mixed three-kernel inference step on sm_100a
// (MINIWORLD_BIAS_ONLY_DIT_INF3_BF16=1), the bf16 twin of bo_front_tf32.cu:
//
//   xa = LN(x) * s1[tok] + sh1[tok];   v | g = xa [Wv; Wg]^T     (tcgen05.mma kind::f16; xa, v and g bf16, fp32 accumulation)
//
// x = the block's input [M, 768], fp32 or bf16 (-DXBF: the step's input, block 0), read again by the tail for the residual; the
// tables (map mtab) [T, 6, 768] bf16, one block's slice of the hoisted tables (0 gate1, 1 gate2, 2 s1, 3 s2, 4 sh1, 5 sh2, sigmoids
// applied; tok = row % T), read here as s1 / sh1 [128][32] boxes (64-B swizzle, 8 KB); W = [Wv; Wg] [2 DA, 768] bf16; VG [M, 2 DA]
// bf16 out; XA [M, 768] bf16 scratch.
//
// Cluster of CL CTAs per 128-row tile (CL = 8, 6 or 4; the host's round model): CTA c owns input columns [c 768 / CL, ..) and v | g
// columns [c NV, ..) (NV = 2 DA / CL). The LayerNorm statistics of the own columns (per 32-column block, Chan) are all-reduced
// through DSMEM; the GEMM's full-K xa is exchanged through L2: every CTA writes its own xa columns to XA with st.global, then ONE
// release (fence.proxy.async.global per thread, a barrier, fence.acq_rel.cluster, CL relaxed remote arrivals on xaready), and the
// A producer streams the k-blocks from L2. A k-block is 64 bf16 columns (one 128-B swizzle row, [128][64] = 16 KB per box):
//   CL 6 / 4: the own 2 / 3 k-blocks are also staged in ring slots 0 .. KX - 1 = positions 0 .. KX - 1 and the GEMM takes them
//             first (the MMAs run while the exchange completes);
//   CL 8:     a CTA's 96 columns are 1.5 k-blocks, so nothing is staged: all 12 xa k-blocks come from L2 after xaready.
// GEMM k-block order (both producers'): the own k-blocks first (CL 6 / 4), then the others in order.
// Warps: 0 W producer (TMA, lane 0); 1 MMA (whole warp waits, elect_one() issues); 2 TMEM allocator; 3 A producer (lane 0: the
// s1 / sh1 boxes, then the L2 xa k-blocks); 4-11 row workers (statistics, xa, its stores) and the epilogue (TMEM 32x32b -> bf16 ->
// per-warp [32][64 B] transpose tile -> st.global.v4). Row worker: warp w8 takes rows 16 w8 .. + 15 (four of them per lane group
// rsub), lane e columns 4 e .. 4 e + 3 of every 32-column block; its x stays in registers from the statistics to the xa pass (all
// of a row's own blocks loaded at once, after the PDL wait), its xa values are 8 bytes (4 bf16) per row and block.
// Table ring: before the exchange the A ring's slots KX .. 5 are idle, so the A producer loads every own block's s1 / sh1 there (slot
// KX + u % NT, s1 at +0, sh1 at +8 KB; NT = min(KO, 6 - KX): CL 8 3, CL 6 4, CL 4 3 -- CL 4 refills a slot once the row workers have
// read it, tempty) right after its PDL wait, under the x loads and the statistics; the xa pass reads them from shared memory.
//   shared memory  A ring NAR = 6 x 16 KB (tables, then xa) | W ring NWS x [WROWS][64] bf16 (as many as fit) | stats exchange
//                  CL KB | own stats 1 KB | barriers (bo_front_tf32.cu's layout: the same bytes per slot)
//   TMEM           the [128][NV] fp32 accumulator (256 or 512 columns allocated)
// PDL: pdl_launch() after setup; the row workers pdl_wait() before reading x and before any store (VG / XA may still be read by the
// previous kernels), the A producer before the tables; the W producer loads before the wait (weights are packed once). Bit-identical reruns: fixed-order
// reductions, no atomics. -DTRACE: %globaltimer markers into g_trace (bo_front_tf32.cu's layout).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef CL
#define CL 8
#endif
#ifndef DATT
#define DATT 768
#endif
#ifndef XBF
#define XBF 0                                                  // 1: x is bf16 (the step's input), else fp32
#endif
static_assert(CL == 4 || CL == 6 || CL == 8, "cluster of 4, 6 or 8");
static_assert(CL != 6 || DATT == 768, "CL 6: 2 DA / 6 columns per CTA need DA 768");
static_assert(DATT == 768 || DATT == 1024, "768 or 1024 attention channels");

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
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI void fence_acq_rel_cluster() { asm volatile("fence.acq_rel.cluster;" ::: "memory"); }
DEVI void sts64(uint32_t a, uint2 v) { asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(a), "r"(v.x), "r"(v.y) : "memory"); }
DEVI void stg64(void* p, uint2 v) { asm volatile("st.global.v2.b32 [%0], {%1,%2};" :: "l"(p), "r"(v.x), "r"(v.y) : "memory"); }
DEVI uint2 ldg64(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
DEVI uint2 lds64u(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v; }
DEVI float4 f4bf(uint2 u) { return make_float4(bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y)); }

#ifdef TRACE
__device__ unsigned long long g_trace[16 * 2048];
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TR(i) do { if (blockIdx.x < 16 && (i) < 2048) g_trace[blockIdx.x * 2048 + (i)] = gtime(); } while (0)
#else
#define TR(i) do { } while (0)
#endif
constexpr int EV_AS = 256, EV_AI = 512, EV_AW = 768, EV_WS = 1024, EV_WI = 1280, EV_WW = 1536;

#if XBF
using XT_ = __nv_bfloat16;
using XS = uint2;                                              // a row worker's 4 x values as kept in registers: raw bf16
DEVI XS ldx(const XT_* p) { return ldg64(p); }
DEVI float4 f4(XS u) { return f4bf(u); }
#else
using XT_ = float;
using XS = float4;
DEVI XS ldx(const XT_* p) { const uint4 u = ldg128(p); return make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
DEVI float4 f4(XS v) { return v; }
#endif
constexpr int D = 768, QM = 128, KB = 64, NK = D / KB;         // KB: bf16 columns per k-block
constexpr int NC1 = D / CL, KO = NC1 / 32;                     // own columns; 32-column blocks of the row workers
constexpr int KX = (NC1 % KB == 0) ? NC1 / KB : 0;             // own k-blocks staged in the ring (CL 6: 2, CL 4: 3, CL 8: 0)
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
static_assert(KO * 32 == NC1 && KX <= NAR && NV % 64 == 0 && WROWS % 16 == 0 && WROWS <= 256 && 8 * 2048 <= NAR * SLOT, "tiling");
constexpr uint32_t TCOLS = NV <= 256 ? 256 : 512;
constexpr int TBOX = QM * 64;                                  // a [128][32] bf16 table box (64-B swizzle): s1 at +0, sh1 at +TBOX
constexpr int NT = KO < NAR - KX ? KO : NAR - KX;             // table slots (A slots KX .. KX + NT - 1)
static_assert(NT >= 1 && 2 * TBOX == SLOT, "table ring");
DEVI int kb_of(int t, int c) {
  if (KX == 0) return t;
  if (t < KX) return KX * c + t;
  const int j = t - KX;
  return j < KX * c ? j : j + KX;
}
constexpr uint32_t I_W = idesc_bf16(QM, WROWS);

struct Bars {
  uint64_t afull[NAR], aempty[NAR], wfull[NWS], wempty[NWS], tfull[NT], tempty[NT], xch, xaready, accfull;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

extern "C" __global__ void __launch_bounds__(384, 1)
bo_front_bf16_sm100(const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mxa,
                    const __grid_constant__ CUtensorMap mtab, const XT_* __restrict__ X, __nv_bfloat16* __restrict__ VG,
                    __nv_bfloat16* __restrict__ XA, int T, int xastride, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t c = cluster_rank();
  const int m0 = (int)(blockIdx.x / CL) * QM, tr0 = m0 % T;

  if (tid == 0) {
    for (int s = 0; s < NAR; ++s) { mbar_init(&B.afull[s], 1); mbar_init(&B.aempty[s], 1); }
    for (int s = 0; s < NWS; ++s) { mbar_init(&B.wfull[s], 1); mbar_init(&B.wempty[s], 1); }
    for (int s = 0; s < NT; ++s) { mbar_init(&B.tfull[s], 1); mbar_init(&B.tempty[s], 8); }   // tempty: the 8 row-worker warps
    mbar_init(&B.xch, 1);
    mbar_init(&B.xaready, CL);                                 // every cluster CTA's own xa columns are in L2
    mbar_init(&B.accfull, 1);
    mbar_expect_tx(&B.xch, CL * 1024);
    fence_barrier_init();
    prefetch_map(&mw); prefetch_map(&mxa); prefetch_map(&mtab);
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
    // ------------------------------------------------------------------------------------------------ W producer (the GEMM's k order)
    if (lane == 0) {
      int wj = 0;
      const uint64_t keep = pol_evict_last();
      for (int t = 0; t < NK; ++t)
        for (int h = 0; h < NWH; ++h, ++wj) {
          const int ws = wj % NWS;
          if (wj >= NWS) mbar_wait(&B.wempty[ws], ((wj / NWS) - 1) & 1);
          mbar_expect_tx(&B.wfull[ws], WSLOT);
          tma_load_2d_h(su + O_W + ws * WSLOT, &mw, &B.wfull[ws], KB * kb_of(t, (int)c), (int)c * NV + h * WROWS, keep);
          TR(EV_WI + wj);
        }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ A producer
    if (lane == 0) {
      pdl_wait();                                              // the tables: the hoist's output
      const uint64_t once = pol_evict_first();                 // read once by this CTA
      for (int u = 0; u < KO; ++u) {                           // s1 / sh1 of own block u into table slot u % NT
        const int k = u % NT;
        if (u >= NT) mbar_wait(&B.tempty[k], ((u / NT) - 1) & 1);
        const uint32_t dst = su + O_A + (uint32_t)(KX + k) * SLOT;
        const int col = (int)c * NC1 + 32 * u;
        mbar_expect_tx(&B.tfull[k], 2 * TBOX);
        tma_load_2d_h(dst, &mtab, &B.tfull[k], 2 * D + col, tr0, once);
        tma_load_2d_h(dst + TBOX, &mtab, &B.tfull[k], 4 * D + col, tr0, once);
      }
      TR(6);
      mbar_wait_cl(&B.xaready, 0);                             // all CL CTAs' xa stores performed (every table read)
      TR(16);
      fence_proxy_async_global();
      for (int i = KX; i < NK; ++i) {                          // positions 0 .. KX - 1: the own k-blocks, staged by the row workers
        const int slot = i % NAR;
        if (i >= NAR) mbar_wait(&B.aempty[slot], ((i / NAR) - 1) & 1);
        mbar_expect_tx(&B.afull[slot], SLOT);
        tma_load_2d(su + O_A + slot * SLOT, &mxa, &B.afull[slot], KB * kb_of(i, (int)c), m0);
        TR(EV_AI + i);
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer (K = 16 per MMA)
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
            umma_ss(tmem + (uint32_t)(h * WROWS), da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_W, (i | ks) ? 1u : 0u);
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
    const XT_* xb = X + (size_t)m0 * D + (size_t)c * NC1 + 4 * e;
    float mu[4], m2[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) { mu[s] = 0.f; m2[s] = 0.f; }
    // ---- statistics of the own columns (per 32-column block, merged by Chan), all-reduced through DSMEM; x kept for the xa pass
    XS xk[KO][4];                                              // every own block's x, all loads in flight at once
#pragma unroll
    for (int j = 0; j < KO; ++j)
#pragma unroll
      for (int s = 0; s < 4; ++s) xk[j][s] = ldx(xb + (size_t)(16 * w8 + 4 * s + rsub) * D + 32 * j);
#pragma unroll
    for (int j = 0; j < KO; ++j) {
      const float n0 = 32.f * j, nn = n0 + 32.f;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const float4 v = f4(xk[j][s]);
        const float bm = sum8((v.x + v.y) + (v.z + v.w)) * (1.f / 32);
        const float a0 = v.x - bm, a1 = v.y - bm, a2 = v.z - bm, a3 = v.w - bm;
        const float bq = sum8((a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3));
        const float dl = bm - mu[s];
        mu[s] += dl * (32.f / nn);
        m2[s] += bq + dl * dl * (n0 * 32.f / nn);
      }
    }
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
    // ---- xa per own 32-column block u (4 bf16 per row and lane; s1 / sh1 from table slot u % NT) -> XA (st.global) and, CL 6 / 4,
    // into own k-block u / 2's ring slot (position u / 2: 128-B rows, the block's half u & 1 -> 16-B chunks 4 (u & 1) + e / 2, 8 B
    // each); after a k-block's second half its afull (the MMAs start on it)
    __nv_bfloat16* const xab = XA + (size_t)(m0 + 16 * w8 + rsub) * xastride + (size_t)c * NC1 + 4 * e;   // row 16 w8 + rsub, lane e
    const uint32_t tofs = 8u * (uint32_t)(e & 1);
#pragma unroll
    for (int u = 0; u < KO; ++u) {
      const uint32_t dst = su + O_A + (uint32_t)(u / 2) * SLOT;
      const uint32_t tsl = su + O_A + (uint32_t)(KX + u % NT) * SLOT;
      mbar_wait(&B.tfull[u % NT], (u / NT) & 1);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const uint32_t r = (uint32_t)(16 * w8 + 4 * s + rsub);
        const float4 xv = f4(xk[u][s]);
        const float4 sv = f4bf(lds64u(tsl + sw64(r, (uint32_t)(e >> 1)) + tofs));
        const float4 hv = f4bf(lds64u(tsl + TBOX + sw64(r, (uint32_t)(e >> 1)) + tofs));
        const uint2 o = make_uint2(pack_bf16(fmaf((xv.x - mu[s]) * rs[s], sv.x, hv.x), fmaf((xv.y - mu[s]) * rs[s], sv.y, hv.y)),
                                   pack_bf16(fmaf((xv.z - mu[s]) * rs[s], sv.z, hv.z), fmaf((xv.w - mu[s]) * rs[s], sv.w, hv.w)));
        if constexpr (KX > 0) sts64(dst + sw128(r, (uint32_t)(4 * (u & 1) + (e >> 1))) + 8 * (e & 1), o);
        stg64(xab + (size_t)(4 * s) * xastride + 32 * u, o);
      }
      if (u + NT < KO) {                                       // CL 4: the slot takes block u + NT's tables
        fence_proxy_async();                                   // these reads before the producer's TMA overwrites the slot
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.tempty[u % NT]);
      }
      if constexpr (KX > 0) {
        if (u & 1) {                                           // own k-block u / 2 complete
          fence_proxy_async();                                 // generic stores -> the MMA (async proxy) reads them
          named_bar_sync(1, 256);
          if (rw == 0) mbar_arrive(&B.afull[u / 2]);
        }
      }
    }
    fence_proxy_async();                                       // the table reads before the xa TMA loads reuse those slots
    fence_proxy_async_global();                                // this thread's XA stores -> the peers' TMA loads (after the release)
    named_bar_sync(1, 256);
    if (rw == 0) {
      TR(4);
      fence_acq_rel_cluster();                                 // one release (every row worker's stores: before the barrier)
      for (int p = 0; p < CL; ++p) mbar_arrive_remote_relaxed(&B.xaready, (c + (uint32_t)p) % CL);
      TR(5);
    }
    // ---- epilogue: VG[:, c NV ..] = the accumulator in bf16, through per-warp [32 rows][64 B] transpose tiles in the idle A ring
    mbar_wait(&B.accfull, 0);
    tc_fence_after();
    if (rw == 0) TR(10);
    const int q4 = warp & 3, hh = (warp - 4) >> 2;
    const uint32_t trow = tmem + ((uint32_t)(32 * q4) << 16);
    const uint32_t tile = su + O_A + (uint32_t)(warp - 4) * 2048;   // every block consumed (accfull)
    constexpr int NCH = NV / 64;
#pragma unroll 1
    for (int k = 0; k < NCH; ++k) {
      const int ch = hh * NCH + k, gc = (int)c * NV + 32 * ch;
      uint32_t v[32];
      tmem_ld32w(trow + (uint32_t)(32 * ch), v);
#pragma unroll
      for (int q = 0; q < 4; ++q)
        sts128(tile + sw64((uint32_t)lane, (uint32_t)q),
               make_uint4(pack_bf16(__uint_as_float(v[8 * q]), __uint_as_float(v[8 * q + 1])),
                          pack_bf16(__uint_as_float(v[8 * q + 2]), __uint_as_float(v[8 * q + 3])),
                          pack_bf16(__uint_as_float(v[8 * q + 4]), __uint_as_float(v[8 * q + 5])),
                          pack_bf16(__uint_as_float(v[8 * q + 6]), __uint_as_float(v[8 * q + 7]))));
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 4; ++i) {                            // rows 8 i .. 8 i + 7 of this warp's 32, 64 B each
        const uint32_t rr = 8 * i + (lane >> 2), qq = lane & 3;
        stg128(VG + (size_t)(m0 + 32 * q4 + (int)rr) * (2 * DATT) + gc + 8 * qq, lds128(tile + sw64(rr, qq)));
      }
      __syncwarp();
    }
    if (rw == 0) TR(11);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}
