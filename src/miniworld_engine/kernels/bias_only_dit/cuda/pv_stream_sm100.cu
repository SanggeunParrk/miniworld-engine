// pv_stream_sm100.cu -- the bias-only token DiT's attention core for LONG token lengths (L > 768), bf16, sm_100a:
//
//   a[s] = sigmoid(g[s]) * (P_h v[s]_h)      for every head h and sample s     (GATE=0: a = P v, the training backward's dV = P^T dO)
//
// P [H L, L] bf16 (normalised attention weights, or their per-head transpose; contiguous), v / g / a [S L, H DH] bf16 column views
// (v | g are the halves of the v|g GEMM output; any 16-byte aligned row stride). tcgen05 kind::f16 products, fp32 accumulation.
//
// Why a second bf16 core: pv_gate_inf.cu keeps the item's P tile [128 i][L j] in tensor memory (L / 2 columns) beside two
// accumulators, which bounds it at L <= 768. Past that P streams through shared memory as the K-major A operand, and the loop
// turns KEY-OUTER, SAMPLE-INNER: one 64-key P chunk feeds the products of the item's SG samples. This is the bf16 twin of
// pv_gate_tf32.cu (the fp32 core, which streams P for the same reason): the same kernel split, pipeline, warp roles, barriers,
// item order and double-buffered accumulators; what differs is only the operand format and the stage geometry:
//   * a stage is one P chunk [128 i][64 j] bf16 (128-B rows, SW128, 16 KB) + SG v tiles [64 j][DH d] (DH x 2 bytes inside a
//     128-byte swizzle row, 8 KB per tile), the group's tiles loaded as ONE 3-D TMA box [DH][64 keys][SG samples] (a box per
//     sample was 2-6 % slower); 4 K steps of 16 keys per stage (tf32: 32-key chunks, K steps of 8);
//   * B is the group's v tiles side by side as ONE MN-major operand of N = SG x 64 (the 128-B swizzle's 64-element atoms, LBO = 8 KB
//     between the samples' tiles): one MMA per K step reads the P chunk once for the whole group (the -DBATCHN form of the fp32
//     core). A head narrower than 64 computes 64 - DH unused columns per sample from the row's stale bytes; a partial last group
//     computes unused columns from stale tiles; the epilogue reads only each sample's first DH columns;
//   * the g / a staging tile is bf16 [128][DH -> 128 B] (16 KB) and the gate is pv_gate_inf.cu's packed f32x2 ``gate_pair``.
// Accumulators double-buffered by item parity (2 x SG x 64 <= 512 TMEM columns: SG <= 4), so an item's epilogue runs under the next
// item's products.
//
// Shared memory (bytes; 232448 per CTA): NST stages of STB = 16 KB + SG x 8 KB, NX = 3 g / a staging tiles of 16 KB, 512 B of
// barriers; NST = (232448 - 1024 - NX XG) / STB: SG 1 / 2 / 3 / 4 -> 7 / 5 / 4 / 3 stages. Every tile base is a multiple of 8 KB.
// Warps: 0 TMA producer, 1 MMA (whole warp waits, elect_one() issues), 2 TMEM allocator, 3 idle, 4-7 epilogue (one query row per
// thread). Work item = (head, 128-query tile, group of SG samples), the group fastest (consecutive CTAs read the same P tile, so
// the second group's P comes from L2).
// L a multiple of 128 (query tiles; 64-key chunks); no length bound of its own (the host serves L <= 3072).
// -DPDL_INF (the bf16 three-kernel inference step's K2, and the training backward's programmatic-dependent dV): launch_dependents
// after setup, griddepcontrol.wait before the producer's first load and before the epilogue reads g or stores a. No P_EARLY form:
// P and v share every stage.
// -DPMN (GATE=0; the training backward's dV = P^T dO WITHOUT a stored P^T): A is P itself, read as the MN-major operand. An item is
// (head, 128-KEY tile, group) and its K loop walks the queries: a stage's A is P[queries c KC .. + 63][keys m0 .. m0 + 127] as two
// TMA boxes [64 keys][64 queries] (SW128, 8 KB each; LBO = 8 KB between the two 64-key atoms, SBO 1 KB per 8 queries), 16 queries
// (2 KB) per K step; B is the group's dO tiles [64 queries][DH] in v's place (MN-major as before). The bytes per stage, the pipeline
// and the epilogue (rows (a0 + k) L + m0 of dV: the item's keys) are the K-major form's. Any L a multiple of 128: the shared-bias
// training path's dV at every length, and the per-sample one below.
// -DPSB (with SG = 1; a pair bias of its own per sample): P is [H, S, L, L], a plane per (head, sample); an item's P rows start at
// (h S + a0) L (pv_gate_inf.cu's -DPSB). With PMN this is the per-sample dV = P_a^T dO_a from P alone.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef SG
#define SG 4                                   // samples per work item
#endif
#ifndef NX
#define NX 3                                   // g staging tiles: loads run NX - 1 tiles ahead
#endif
#ifndef GATE
#define GATE 1
#endif
#ifndef NHEAD
#define NHEAD 16                               // heads x head width: 16 x 48, 24 x 32, 12 x 64, 16 x 64
#endif
#ifndef DHEAD
#define DHEAD 48
#endif
#ifdef PSB
static_assert(SG == 1, "PSB: one sample per work item (each sample its own P plane)");
constexpr bool PSB_P = true;
#else
constexpr bool PSB_P = false;
#endif
#ifdef PMN
static_assert(GATE == 0, "PMN: the backward's dV (no gate)");
constexpr bool P_MN = true;
#else
constexpr bool P_MN = false;
#endif
constexpr int DH = DHEAD, NH = NHEAD, QM = 128, KC = 64, NQC = DH / 8;       // KC: keys per stage; NQC: 16-byte chunks of a head row
static_assert(DH == 32 || DH == 48 || DH == 64, "head width 32, 48 or 64");
static_assert(SG >= 1 && SG <= 4, "two accumulator sets of SG x 64 columns within 512");
constexpr int TP = QM * 128;                                                 // P chunk [128 i][64 j] bf16 SW128: 16 KB
constexpr int TV = KC * 128;                                                 // v tile [64 j][DH d -> 128 B] SW128: 8 KB
constexpr int STB = TP + SG * TV;
constexpr int XG = QM * 128;                                                 // g in / gated a out [128][DH -> 128 B]: 16 KB
constexpr int NST = (232448 - 1024 - NX * XG) / STB;
constexpr int O_ST = 0, O_X = NST * STB, O_BAR = O_X + NX * XG, SMEM_BYTES = O_BAR + 512;
static_assert(NST >= 2 && SMEM_BYTES <= 232448, "shared memory");
static_assert(STB % 1024 == 0 && XG % 1024 == 0 && TV % 1024 == 0, "1 KB alignment of the swizzled tiles");
constexpr int NW = 64;                                                       // accumulator columns per sample (one MN atom)
constexpr int TNEED = 2 * SG * NW;
constexpr uint32_t TCOLS = TNEED <= 64 ? 64 : TNEED <= 128 ? 128 : TNEED <= 256 ? 256 : 512;
constexpr uint32_t I_PVB = idesc_bf16(QM, SG * NW, P_MN ? 1 : 0, 1);        // A = P K-major (PMN: MN-major), B = the v tiles MN-major

// o * sigmoid(g) for a pair of channels, packed f32x2 (pv_gate_inf.cu's gate_pair: exponentials on MUFU, the reciprocal on the FMA
// pipe, seed clamped so that g <= -87.3 stays finite)
DEVI uint32_t gate_pair(uint32_t o0, uint32_t o1, uint32_t g2) {
  const f2 t = mul2(mk2(bf16lo(g2), bf16hi(g2)), mk2(-1.4426950408889634f, -1.4426950408889634f));
  const f2 e = add2(mk2(ex2f(lo2(t)), ex2f(hi2(t))), mk2(1.f, 1.f));
  const f2 d = mk2(fmin_nan(lo2(e), RCP_SEED_MAX), fmin_nan(hi2(e), RCP_SEED_MAX));
  const f2 nd = neg2(d), two = mk2(2.f, 2.f);
  f2 r = mk2(__int_as_float(0x7EF311C3 - __float_as_int(lo2(d))), __int_as_float(0x7EF311C3 - __float_as_int(hi2(d))));
  r = mul2(r, fma2(nd, r, two));
  r = mul2(r, fma2(nd, r, two));
  const f2 o = mul2(mk2u(o0, o1), r);
  return pack_bf16(lo2(o), hi2(o));
}
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }

struct Bars {
  uint64_t full[NST], empty[NST], acc_full[2], acc_empty[2], g_full[NX];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(256, 1)
bo_pv_stream_sm100(const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mv,
                   const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mo, int L, int S) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = L / QM, nk = L / KC, ng = (S + SG - 1) / SG;
  const int items = NH * mt * ng;
  const int my = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  // item li of this CTA -> first sample a0, samples ns, query tile m0, head h (the group fastest)
  auto item_of = [&](int li, int& a0, int& ns, int& m0, int& h) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    a0 = (wi % ng) * SG; ns = min(SG, S - a0);
    const int r = wi / ng;
    m0 = (r % mt) * QM; h = r / mt;
  };

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    for (int x = 0; x < NX; ++x) mbar_init(&B.g_full[x], 1);
    fence_barrier_init();
    prefetch_map(&mp); prefetch_map(&mv); prefetch_map(&mo);
#if GATE
    prefetch_map(&mg);
#endif
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TCOLS); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
#ifdef PDL_INF
  pdl_launch();
#endif

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
#ifdef PDL_INF
      pdl_wait();                                                 // v (and in training P / P^T): earlier kernels' outputs
#endif
      int g = 0;
      for (int li = 0; li < my; ++li) {
        int a0, ns, m0, h; item_of(li, a0, ns, m0, h);
        for (int c = 0; c < nk; ++c, ++g) {
          const int s = g % NST;
          if (g >= NST) mbar_wait(&B.empty[s], ((g / NST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          // the P chunk, and the group's v tiles as ONE 3-D box [DH][64 keys][SG samples] (TV apart; samples past S zero-filled,
          // the full box counts): two TMA instructions per stage (a box per sample: 2-6 % slower, round 1)
          mbar_expect_tx(&B.full[s], TP + SG * (KC * DH * 2));
          const int prow = (PSB_P ? h * S + a0 : h) * L;          // the item's P plane ([H L, L], or [H S L, L] per sample)
          if (P_MN) {                                             // P[queries c KC .. + 63][keys m0 .. + 127]: two 64-key boxes
            tma_load_2d(st, &mp, &B.full[s], m0, prow + c * KC);
            tma_load_2d(st + TP / 2, &mp, &B.full[s], m0 + 64, prow + c * KC);
          } else {
            tma_load_2d(st, &mp, &B.full[s], c * KC, prow + m0);
          }
          tma_load_3d(st + TP, &mv, &B.full[s], h * DH, c * KC, a0);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    int g = 0;
    for (int li = 0; li < my; ++li) {
      const int b = li & 1;
      if (li >= 2) mbar_wait(&B.acc_empty[b], ((li >> 1) - 1) & 1);          // the epilogue has read item li - 2's accumulators
      tc_fence_after();
      for (int c = 0; c < nk; ++c, ++g) {
        const int s = g % NST;
        mbar_wait(&B.full[s], (g / NST) & 1);
        tc_fence_after();
        const uint32_t st = su + O_ST + s * STB;
        const uint64_t dp = P_MN ? desc_mn128(st, TP / 2) : desc_k128(st);   // PMN: the two 64-key atoms 8 KB apart
        const uint64_t dv = desc_mn128(st + TP, TV);              // every sample's v tile, TV apart: N = SG x 64
        if (elect_one()) {
          const uint32_t d = tmem + (uint32_t)(b * SG * NW);
#pragma unroll
          for (int ks = 0; ks < KC / 16; ++ks)                    // 16 keys: 32 B of a P row, 16 rows (2 KB) of a v tile
            umma_ss(d, dp + (uint64_t)(P_MN ? ((ks * 2048) >> 4) : ks * 2), dv + (uint64_t)((ks * 2048) >> 4), I_PVB,
                    (c > 0 || ks > 0) ? 1u : 0u);
          tc_commit(&B.empty[s]);
          if (c == nk - 1) tc_commit(&B.acc_full[b]);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ epilogue
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    // NX g staging tiles rotate over the running sequence of (item, sample) tiles. Thread r == 0 issues the store of tile t, then
    // waits for the store of tile t - 1 (not the one just issued) to have read its buffer and loads tile t - 1 + NX's g there.
    int pli = 0, pk = 0;                                          // the next tile whose g is loaded
    auto load_next_g = [&](int buf) {                             // thread r == 0 only
      if (pli >= my) return;
      int a0, ns, m0, h; item_of(pli, a0, ns, m0, h);
#if GATE
      mbar_expect_tx(&B.g_full[buf], QM * DH * 2);
      tma_load_2d(su + O_X + buf * XG, &mg, &B.g_full[buf], h * DH, (a0 + pk) * L + m0);
#else
      mbar_arrive(&B.g_full[buf]);                                // the staging tile is only an output: nothing to load
#endif
      if (++pk == ns) { pk = 0; ++pli; }
    };
#ifdef PDL_INF
    pdl_wait();                                                   // g: an earlier kernel's output; a: may still be read by one
#endif
    if (r == 0) for (int x = 0; x < NX - 1; ++x) load_next_g(x);
    int t = 0;
    for (int li = 0; li < my; ++li) {
      int a0, ns, m0, h; item_of(li, a0, ns, m0, h);
      const int b = li & 1;
      mbar_wait(&B.acc_full[b], (li >> 1) & 1);
      tc_fence_after();
      for (int k = 0; k < ns; ++k, ++t) {
        const int xb = t % NX;
        const uint32_t xg = su + O_X + xb * XG;
        uint32_t ov[DH];
#pragma unroll
        for (int cc = 0; cc < DH / 16; ++cc)
          tmem_ld16(trow + (uint32_t)((b * SG + k) * NW + cc * 16), *reinterpret_cast<uint32_t(*)[16]>(ov + 16 * cc));
        tmem_wait_ld();
        if (k == ns - 1) {                                        // every accumulator of the item read: the MMAs may reuse them
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.acc_empty[b]);
        }
        mbar_wait(&B.g_full[xb], (t / NX) & 1);
        // all loads first, then the math, then the stores (the shared accesses are volatile asm with a memory clobber)
        uint4 gw[NQC];
#if GATE
#pragma unroll
        for (int q = 0; q < NQC; ++q) gw[q] = lds128(xg + sw128(r, q));   // 8 channels per 16-byte chunk of this thread's row
#pragma unroll
        for (int q = 0; q < NQC; ++q) {
          const uint32_t gg[4] = {gw[q].x, gw[q].y, gw[q].z, gw[q].w};
          uint32_t o4[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) o4[e] = gate_pair(ov[8 * q + 2 * e], ov[8 * q + 2 * e + 1], gg[e]);
          gw[q] = make_uint4(o4[0], o4[1], o4[2], o4[3]);
        }
#else
#pragma unroll
        for (int q = 0; q < NQC; ++q)
          gw[q] = make_uint4(pack_bf16(__uint_as_float(ov[8 * q]), __uint_as_float(ov[8 * q + 1])),
                             pack_bf16(__uint_as_float(ov[8 * q + 2]), __uint_as_float(ov[8 * q + 3])),
                             pack_bf16(__uint_as_float(ov[8 * q + 4]), __uint_as_float(ov[8 * q + 5])),
                             pack_bf16(__uint_as_float(ov[8 * q + 6]), __uint_as_float(ov[8 * q + 7])));
#endif
#pragma unroll
        for (int q = 0; q < NQC; ++q) sts128(xg + sw128(r, q), gw[q]);
        fence_proxy_async();
        named_bar_sync(1, 128);
        if (r == 0) {
          tma_store_2d(&mo, xg, h * DH, (a0 + k) * L + m0);
          tma_store_commit();
          if (t == 0) load_next_g(NX - 1);                        // its buffer has never held a tile
          else { tma_store_wait_read1(); load_next_g((t - 1) % NX); }
        }
      }
    }
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}
