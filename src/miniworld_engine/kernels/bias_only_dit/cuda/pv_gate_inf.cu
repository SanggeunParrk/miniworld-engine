// pv_gate_inf.cu -- the bias-only token DiT's INFERENCE attention core, sm_100a:
//
//   a[s] = sigmoid(g[s]) * (P_h v[s]_h)      for every head h and sample s
//
// There is no query and no key: the attention weights P = softmax(pair bias) [H, L, L] (bf16, normalised) depend on neither the
// sample nor the solver step, so the runner makes them once per sample() and this kernel is a plain GEMM per head -- no online
// softmax, no running max, no rescale. v | g are column views of the v|g GEMM output [S L, 1536] bf16; a [S L, 768] bf16.
//
// Work item = (head, 128-query tile, group of SG samples); the host picks SG per shape. The item's P tile [128 i][L j] goes into
// TENSOR memory once (TMA in 64-key SW128 chunks, tcgen05.cp into the MMA's A-operand layout, L / 2 columns) and stays there while
// the group's samples stream their v tiles [128 j][48 d] through the shared-memory ring: per sample the K loop is 8 x (M 128, N 48,
// K 16) products per 128-key block, A from TMEM. Chunks are as large as a slot holds: the ring runs at a few hundred clocks per TMA
// instruction whatever its size (32-key P chunks and 64-key v tiles made it twice as slow). The loop is sample-OUTER, so one sample's accumulator is complete while the next
// sample's products run, and the epilogue of sample k (sigmoid(g) * o, g in by TMA, the gated tile out by TMA) hides under the
// MMAs of sample k + 1. (With the P tile in shared memory and the samples inner, the epilogue only started once the whole item
// was loaded and took ~40 % of the kernel.) Accumulators: two sets of 48 columns above P.
// Warps: 0 TMA producer, 1 MMA (and the P copies), 2 TMEM allocator, 3 idle, 4-7 epilogue (one query row per thread).
// -DPDL_INF (only the bf16 three-kernel inference step builds it: K2 between bo_front_bf16 and bo_tail_bf16 / bo_tail2_bf16):
// programmatic dependent launch -- launch_dependents after setup, griddepcontrol.wait before the producer's first load and before
// the epilogue reads g or stores a. Without it the cubin is the default inference / training path's, unchanged.
// L <= 768 (P plus two accumulators within 512 TMEM columns). Heads x width by -DNHEAD / -DDHEAD (16 x 48, 24 x 32, 12 x 64): a v or g
// row of the head is DH x 2 bytes inside a 128-byte swizzle row.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef SG
#define SG 2                                   // samples per work item
#endif
#ifndef NX
#define NX 3                                   // g staging tiles: loads run NX - 1 tiles ahead
#endif
#ifndef GATE
#define GATE 1                                 // 0: o = P v as it is (the training backward's dV = P^T do), no g read
#endif
#ifndef VB
#define VB 128                                 // keys per v tile (one TMA box [VB j][48 d]); L a multiple of VB
#endif
#ifndef NHEAD
#define NHEAD 16                               // heads x head width: 16 x 48, 24 x 32 or 12 x 64 (768 channels)
#endif
#ifndef DHEAD
#define DHEAD 48
#endif
constexpr int BN = VB, DH = DHEAD, QM = 128, NH = NHEAD, NQC = DH / 8;   // NQC: 16-byte chunks of a head row
static_assert(DH % 16 == 0 && DH <= 64, "head width: a multiple of 16 within one 128-byte swizzle row");
// ring slot: a v tile [VB j][48 d -> 128 B] or PCS P chunks [128 i][64 j], all SW128. Larger boxes take the SM's TMA intake from
// ~45 B/clk (12 KB) to ~60-70 (24-32 KB): with 128 keys per v tile the training shapes (A = 48) were intake-bound.
constexpr int SLOT = VB * 128 > 16384 ? VB * 128 : 16384, PCS = SLOT / 16384;
static_assert(VB % 16 == 0 && VB <= 256, "v tile");
constexpr int XG = QM * 128;                   // g in / gated o out: 16 KB
constexpr int NSLOT = (232448 - NX * XG - 1024) / SLOT;
constexpr int O_ST = 0, O_X = NSLOT * SLOT, O_BAR = O_X + NX * XG, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_ACC = 512 - 2 * DH;       // two DH-column accumulators at the top (416 / 448); P at [0, L / 2)
constexpr uint32_t I_PV = idesc_bf16(QM, DH, 0, 1);

// o * sigmoid(g) for a pair of channels, packed f32x2: 1 / (1 + 2^(-g log2 e)) with the exponentials on MUFU and the reciprocal on
// the FMA pipe (bit-trick seed, two Newton steps: relative error <= 2e-4, below the bf16 rounding of the product). The epilogue is
// one warp per SMSP and issue-bound: per pair this is ~14 instructions where the scalar kit sigmoid (two MUFU ops per channel) was
// MUFU-bound and the scalar Newton form issue-bound at twice the count.
DEVI uint32_t gate_pair(uint32_t o0, uint32_t o1, uint32_t g2) {
#ifdef SIG_KIT
  return pack_bf16(__uint_as_float(o0) * sigmoid_kit(bf16lo(g2)), __uint_as_float(o1) * sigmoid_kit(bf16hi(g2)));
#else
  const f2 t = mul2(mk2(bf16lo(g2), bf16hi(g2)), mk2(-1.4426950408889634f, -1.4426950408889634f));
  const f2 d = add2(mk2(ex2f(lo2(t)), ex2f(hi2(t))), mk2(1.f, 1.f));
  const f2 nd = neg2(d), two = mk2(2.f, 2.f);
  f2 r = mk2(__int_as_float(0x7EF311C3 - __float_as_int(lo2(d))), __int_as_float(0x7EF311C3 - __float_as_int(hi2(d))));
  r = mul2(r, fma2(nd, r, two));
  r = mul2(r, fma2(nd, r, two));
  const f2 o = mul2(mk2u(o0, o1), r);
  return pack_bf16(lo2(o), hi2(o));
#endif
}
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }

#ifdef TRACE
// CTA TRACE_CTA only: [event][index] = clock64 - start. 0 producer slot issued, 1 MMA slot consumed, 2 accumulator seen (per
// tile), 3 g ready (per tile), 4 store issued (per tile), 6 end (index 0)
__device__ long long g_tr[12][64];
#ifndef TRACE_CTA
#define TRACE_CTA 0
#endif
#define TR(ev, i) do { if (blockIdx.x == TRACE_CTA && (i) < 64) g_tr[ev][i] = clock64() - t0; } while (0)
#else
#define TR(ev, i) do { } while (0)
#endif

struct Bars {
  uint64_t full[NSLOT], empty[NSLOT], acc_full[2], acc_empty[2], g_full[NX], p_free;
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(256, 1)
bo_pv_gate_inf_sm100(const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mv,
                     const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mo, int L, int S) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = L / QM, nb = L / BN, nc = L / 64, ng = (S + SG - 1) / SG;
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
    for (int s = 0; s < NSLOT; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    for (int x = 0; x < NX; ++x) mbar_init(&B.g_full[x], 1);
    mbar_init(&B.p_free, 1);
    fence_barrier_init();
    prefetch_map(&mp); prefetch_map(&mv); prefetch_map(&mg); prefetch_map(&mo);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
#ifdef TRACE
  const long long t0 = clock64();
#endif
#ifdef PDL_INF
  pdl_launch();
#endif

  if (warp == 0) {
    if (lane == 0) {
#ifdef PDL_INF
      pdl_wait();                                                 // v: the v|g GEMM's output (P: hoisted, but in the chain's order)
#endif
      int g = 0;
      auto slot = [&](uint32_t bytes) {                           // the next ring slot, empty, armed for `bytes`
        const int s = g % NSLOT;
        if (g >= NSLOT) mbar_wait(&B.empty[s], ((g / NSLOT) - 1) & 1);
        mbar_expect_tx(&B.full[s], bytes);
        TR(0, g);
        ++g;
        return s;
      };
      for (int li = 0; li < my; ++li) {
        int a0, ns, m0, h; item_of(li, a0, ns, m0, h);
        for (int c = 0; c < nc; c += PCS) {                      // the P tile, 64 PCS keys per slot
          const int s = slot(PCS * QM * 64 * 2);
          for (int e = 0; e < PCS; ++e) tma_load_2d(su + O_ST + s * SLOT + e * 16384, &mp, &B.full[s], (c + e) * 64, h * L + m0);
        }
        for (int k = 0; k < ns; ++k)
          for (int n = 0; n < nb; ++n) {
            const int s = slot(BN * DH * 2);
            tma_load_2d(su + O_ST + s * SLOT, &mv, &B.full[s], h * DH, (a0 + k) * L + n * BN);
          }
      }
    }
  } else if (warp == 1) {
    int g = 0, t = 0;
    for (int li = 0; li < my; ++li) {
      int a0, ns, m0, h; item_of(li, a0, ns, m0, h);
      if (li >= 1) mbar_wait(&B.p_free, (li - 1) & 1);            // the previous item's products have read P
      for (int c = 0; c < nc; c += PCS, ++g) {                    // P chunk c -> TMEM columns 32 c .. 32 c + 31
        const int s = g % NSLOT;
        mbar_wait(&B.full[s], (g / NSLOT) & 1);
        if (lane == 0) TR(1, g);
        tc_fence_after();
        if (elect_one()) {
          for (int e = 0; e < PCS; ++e) {
            const uint64_t dp = desc_k128(su + O_ST + s * SLOT + e * 16384);
#pragma unroll
            for (int j = 0; j < 4; ++j) tmem_cp_128x256b(tmem + (c + e) * 32 + j * 8, dp + (uint64_t)(j * 2));   // 16 keys per copy
          }
          tc_commit(&B.empty[s]);
        }
        __syncwarp();
      }
      for (int k = 0; k < ns; ++k, ++t) {
        const int b = t & 1;
        if (t >= 2) mbar_wait(&B.acc_empty[b], ((t >> 1) - 1) & 1);
        tc_fence_after();
        for (int n = 0; n < nb; ++n, ++g) {
          const int s = g % NSLOT;
          mbar_wait(&B.full[s], (g / NSLOT) & 1);
          if (lane == 0) TR(1, g);
          tc_fence_after();
          const uint64_t dv = desc_mn128(su + O_ST + s * SLOT, 16384);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < BN / 16; ++ks)
              umma_ts(tmem + T_ACC + b * DH, tmem + n * (BN / 2) + ks * 8, dv + (uint64_t)(ks * 2048 >> 4), I_PV, (n > 0 || ks > 0) ? 1u : 0u);
            tc_commit(&B.empty[s]);
            if (n == nb - 1) {
              tc_commit(&B.acc_full[b]);
              if (k == ns - 1) tc_commit(&B.p_free);
            }
          }
          __syncwarp();
        }
      }
    }
  } else if (warp >= 4) {
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    // NX g staging tiles rotate over the running sequence of (item, sample) tiles. Thread r == 0 issues the store of tile t, then
    // waits for the store of tile t - 1 (not the one just issued) to have read its buffer and loads tile t - 1 + NX's g there.
#ifdef PDL_INF
    pdl_wait();                                                   // g: the v|g GEMM's output; a: may still be read by an earlier kernel
#endif
    int pli = 0, pk = 0;                                          // the next tile whose g is loaded
    auto load_next_g = [&](int buf) {                             // thread r == 0 only
      if (pli >= my) return;
#if GATE
      int a0, ns, m0, h; item_of(pli, a0, ns, m0, h);
      mbar_expect_tx(&B.g_full[buf], QM * DH * 2);
      tma_load_2d(su + O_X + buf * XG, &mg, &B.g_full[buf], h * DH, (a0 + pk) * L + m0);
#else
      int a0, ns, m0, h; item_of(pli, a0, ns, m0, h);
      mbar_arrive(&B.g_full[buf]);                                // the staging tile is only an output: nothing to load
#endif
      if (++pk == ns) { pk = 0; ++pli; }
    };
    if (r == 0) for (int x = 0; x < NX - 1; ++x) load_next_g(x);
    int t = 0;
    for (int li = 0; li < my; ++li) {
      int a0, ns, m0, h; item_of(li, a0, ns, m0, h);
      for (int k = 0; k < ns; ++k, ++t) {
        const int b = t & 1, xb = t % NX;
        const uint32_t xg = su + O_X + xb * XG;
        mbar_wait(&B.acc_full[b], (t >> 1) & 1);
        if (r == 0) TR(2, t);
        tc_fence_after();
        uint32_t ov[DH];
#pragma unroll
        for (int cc = 0; cc < DH / 16; ++cc) tmem_ld16(trow + T_ACC + b * DH + cc * 16, *reinterpret_cast<uint32_t(*)[16]>(ov + 16 * cc));
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_empty[b]);
        mbar_wait(&B.g_full[xb], (t / NX) & 1);
        if (r == 0) TR(3, t);
        // all six loads first, then the math, then the stores: the shared accesses are volatile asm with a memory clobber, so an
        // interleaved load / compute / store per chunk serialises six load latencies
        uint4 gw[NQC];
#if GATE
#pragma unroll
        for (int q = 0; q < NQC; ++q) gw[q] = lds128(xg + sw128(r, q));   // 8 channels per 16-byte chunk of this thread's row
#else
#pragma unroll
        for (int q = 0; q < NQC; ++q)
          gw[q] = make_uint4(pack_bf16(__uint_as_float(ov[8 * q]), __uint_as_float(ov[8 * q + 1])),
                             pack_bf16(__uint_as_float(ov[8 * q + 2]), __uint_as_float(ov[8 * q + 3])),
                             pack_bf16(__uint_as_float(ov[8 * q + 4]), __uint_as_float(ov[8 * q + 5])),
                             pack_bf16(__uint_as_float(ov[8 * q + 6]), __uint_as_float(ov[8 * q + 7])));
#endif
#pragma unroll
        for (int q = 0; q < (GATE ? NQC : 0); ++q) {
          const uint32_t gg[4] = {gw[q].x, gw[q].y, gw[q].z, gw[q].w};
          uint32_t o4[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) o4[e] = gate_pair(ov[8 * q + 2 * e], ov[8 * q + 2 * e + 1], gg[e]);
          gw[q] = make_uint4(o4[0], o4[1], o4[2], o4[3]);
        }
#pragma unroll
        for (int q = 0; q < NQC; ++q) sts128(xg + sw128(r, q), gw[q]);
        fence_proxy_async();
        named_bar_sync(1, 128);
        if (r == 0) {
          tma_store_2d(&mo, xg, h * DH, (a0 + k) * L + m0);
          tma_store_commit();
          TR(4, t);
          if (t == 0) load_next_g(NX - 1);                        // its buffer has never held a tile
          else { tma_store_wait_read1(); load_next_g((t - 1) % NX); }
        }
      }
    }
    if (r == 0) { tma_store_wait0(); TR(6, 0); }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
