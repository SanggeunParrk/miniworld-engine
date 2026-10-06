// pv_gate_tf32.cu -- the bias-only token DiT's attention core for the fp32 path, sm_100a, TF32 tensor cores (the fp32 twin of
// pv_gate_inf.cu):
//
//   a[s] = sigmoid(g[s]) * (P_h v[s]_h)      for every head h and sample s     (GATE=0: a = P v, the training backward's dV = P^T dO)
//
// P [H L, L] fp32 (normalised attention weights, or their per-head transpose), v / g / a [S L, H DH] fp32 column views (v | g are
// the halves of the v|g GEMM output; any row stride, 16-byte aligned). Products are tcgen05 kind::tf32 (fp32 operands rounded to
// tf32 by the MMA, fp32 accumulation); the gate is fp32.
//
// What the fp32 operands change against the bf16 core:
//   * P no longer fits tensor memory: a 128-query fp32 tile of P is L columns (768 > 512), so P streams through shared memory as
//     the K-major A operand, 32-key chunks [128 i][32 j] (128-B rows, SW128, 16 KB), and the loop is KEY-OUTER, SAMPLE-INNER: one
//     P chunk feeds the SG samples' products, each into its own accumulator (SG x DH TMEM columns). P is read once per item, as
//     in the bf16 core, and no P copy into tensor memory is needed.
//   * v is the B operand with the head's channels as N: MN-major. A kind::tf32 MMA reads an MN-major operand only from the 128-B
//     swizzle with 32-B atomicity (sm100.cuh: TMA SWIZZLE_128B_ATOM_32B, UMMA layout type 1): a v chunk of one sample is NA =
//     DH / 32 (rounded up) boxes [32 keys][32 channels] (4 KB each, LBO = 4 KB apart); 8 keys (1 KB) per K step. (The same
//     operand form as attn_dkv_tf32.cu's dV / dK products.) For DH = 48 the second box's channels 48..63 are the next head's
//     (or TMA zero fill past the last head): the N = 48 product never reads them.
//   * accumulators double-buffered by item parity (2 x SG x DH <= 512 columns): an item's epilogue runs under the next item's
//     products. The samples of a group finish together, so their epilogues are back to back.
//   * the gated tile is fp32: a 48-wide head row is 192 B, so the g / a staging tile is columns 0-31 (SW128, 16 KB) + columns
//     32..DH-1 (SW64, 8 KB for DH 48; SW128, 16 KB for DH 64; none for DH 32), loaded and stored by TMA.
//
// Shared memory (bytes; 232448 per CTA): NST stages of STB = 16 KB (P chunk) + SG x NA x 4 KB (v chunks), NX = 3 g / a staging
// tiles of XG = 16 + {0, 8, 16} KB, 512 B of barriers; NST = (232448 - 1024 - NX XG) / STB >= 2 (static_assert). Every tile base
// is a multiple of 4 KB (1 KB alignment of the swizzled tiles; the SW64 halves at 16 KB offsets).
//   16 x 48, SG 4: 3 x 48 KB + 3 x 24 KB;   24 x 32, SG 8: 3 x 48 KB + 3 x 16 KB;   16 x 64 / 12 x 64, SG 4: 2 x 48 KB + 3 x 32 KB
// Tensor memory: 2 SG DH columns (<= 512; the allocation is the next power of two).
// Warps: 0 TMA producer, 1 MMA (whole warp waits, elect_one() issues), 2 TMEM allocator, 3 idle, 4-7 epilogue (one query row per
// thread: DH accumulator registers + four 16-byte g chunks at a time; 128 registers at most -- __launch_bounds__(256, 2) caps them,
// shared memory keeps one CTA per SM).
// Work item = (head, 128-query tile, group of SG samples), the group fastest (consecutive CTAs share the P tile in L2).
// L a multiple of 128 (query tiles; 32-key chunks).
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
constexpr int DH = DHEAD, NH = NHEAD, QM = 128, KC = 32;                     // KC: keys per stage (one 128-B fp32 row of P)
static_assert(DH == 32 || DH == 48 || DH == 64, "head width 32, 48 or 64");
constexpr int NA = (DH + 31) / 32;                                           // 32-channel atoms of a v chunk
constexpr int WB = DH - 32;                                                  // channels of the second staging half: 0, 16, 32
constexpr int TP = QM * 128;                                                 // P chunk [128 i][32 j] fp32 SW128: 16 KB
constexpr int TVA = KC * 128, TV = NA * TVA;                                 // v atom [32 j][32 d] fp32: 4 KB; one sample's chunk
constexpr int STB = TP + SG * TV;
constexpr int XA = QM * 128, XB = QM * WB * 4, XG = XA + XB;                 // g / a staging tile
constexpr int NST = (232448 - 1024 - NX * XG) / STB;
constexpr int O_ST = 0, O_X = NST * STB, O_BAR = O_X + NX * XG, SMEM_BYTES = O_BAR + 512;
static_assert(NST >= 2 && SMEM_BYTES <= 232448, "shared memory");
static_assert(STB % 1024 == 0 && XG % 1024 == 0 && TV % 1024 == 0, "1 KB alignment of the swizzled tiles");
// -DBATCHN: one MMA per K step for ALL the group's samples -- B = their v chunks, which sit NA 4-KB atoms apart (TV = NA TVA), as ONE
// MN-major operand of N = SG NA 32 (LBO = TVA) -- instead of one MMA per sample: the P chunk (A) is read from shared memory once per
// K step, not SG times, and SG x fewer MMAs are issued (the kernel was bound by them: ~320 TF/s at L768). A 48-wide head computes 16
// columns of the next head's (or zero-filled) channels it never reads; a partial last group computes unused columns from stale
// shared memory. Accumulator of sample k: columns (buffer SG + k) NW.
#ifdef BATCHN
constexpr int NW = NA * 32;
static_assert(SG * NW <= 256 && (SG * NW) % 16 == 0, "N of the batched product");
#else
constexpr int NW = DH;
#endif
constexpr int TNEED = 2 * SG * NW;
static_assert(TNEED <= 512, "two accumulator sets of SG x NW columns");
constexpr uint32_t TCOLS = TNEED <= 32 ? 32 : TNEED <= 64 ? 64 : TNEED <= 128 ? 128 : TNEED <= 256 ? 256 : 512;
constexpr uint32_t I_PV = idesc_tf32(QM, DH, 0, 1);                          // A = P K-major, B = v MN-major
#ifdef BATCHN
constexpr uint32_t I_PVB = idesc_tf32(QM, SG * NW, 0, 1);                    // the group's samples side by side in N
#endif

DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }
// byte address of 16-byte chunk q (4 fp32 channels) of staging row r
DEVI uint32_t xchunk(uint32_t xa, uint32_t r, int q) {
  if (q < 8) return xa + sw128(r, (uint32_t)q);
  return WB == 16 ? xa + XA + sw64(r, (uint32_t)(q - 8)) : xa + XA + sw128(r, (uint32_t)(q - 8));
}

struct Bars {
  uint64_t full[NST], empty[NST], acc_full[2], acc_empty[2], g_full[NX];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(256, 2)
bo_pv_gate_tf32_sm100(const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mv,
                      const __grid_constant__ CUtensorMap mga, const __grid_constant__ CUtensorMap mgb,
                      const __grid_constant__ CUtensorMap moa, const __grid_constant__ CUtensorMap mob, int L, int S) {
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
    prefetch_map(&mp); prefetch_map(&mv); prefetch_map(&moa);
#if GATE
    prefetch_map(&mga);
#endif
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TCOLS); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my; ++li) {
        int a0, ns, m0, h; item_of(li, a0, ns, m0, h);
        for (int c = 0; c < nk; ++c, ++g) {
          const int s = g % NST;
          if (g >= NST) mbar_wait(&B.empty[s], ((g / NST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.full[s], TP + ns * TV);
          tma_load_2d(st, &mp, &B.full[s], c * KC, h * L + m0);
          for (int k = 0; k < ns; ++k)
#pragma unroll
            for (int a = 0; a < NA; ++a)
              tma_load_2d(st + TP + k * TV + a * TVA, &mv, &B.full[s], h * DH + 32 * a, (a0 + k) * L + c * KC);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    int g = 0;
    for (int li = 0; li < my; ++li) {
      int a0, ns, m0, h; item_of(li, a0, ns, m0, h);
      const int b = li & 1;
      if (li >= 2) mbar_wait(&B.acc_empty[b], ((li >> 1) - 1) & 1);          // the epilogue has read item li - 2's accumulators
      tc_fence_after();
      for (int c = 0; c < nk; ++c, ++g) {
        const int s = g % NST;
        mbar_wait(&B.full[s], (g / NST) & 1);
        tc_fence_after();
        const uint32_t st = su + O_ST + s * STB;
        const uint64_t dp = desc_k128(st);
        if (elect_one()) {
#ifdef BATCHN
          {
            const uint64_t dv = desc_mn32b(st + TP, TVA);         // every sample's atoms, TVA apart: N = SG NW
            const uint32_t d = tmem + (uint32_t)(b * SG * NW);
#pragma unroll
            for (int ks = 0; ks < KC / 8; ++ks)
              umma_ss_tf32(d, dp + (uint64_t)(ks * 2), dv + (uint64_t)((ks * 1024) >> 4), I_PVB, (c > 0 || ks > 0) ? 1u : 0u);
          }
#else
          for (int k = 0; k < ns; ++k) {
            // v of sample k: MN-major, two 32-channel atoms TVA apart (LBO); 8 keys = 1 KB per K step
            const uint64_t dv = desc_mn32b(st + TP + k * TV, TVA);
            const uint32_t d = tmem + (uint32_t)((b * SG + k) * DH);
#pragma unroll
            for (int ks = 0; ks < KC / 8; ++ks)
              umma_ss_tf32(d, dp + (uint64_t)(ks * 2), dv + (uint64_t)((ks * 1024) >> 4), I_PV, (c > 0 || ks > 0) ? 1u : 0u);
          }
#endif
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
      const uint32_t xg = su + O_X + buf * XG;
      mbar_expect_tx(&B.g_full[buf], XG);
      tma_load_2d(xg, &mga, &B.g_full[buf], h * DH, (a0 + pk) * L + m0);
      if (WB > 0) tma_load_2d(xg + XA, &mgb, &B.g_full[buf], h * DH + 32, (a0 + pk) * L + m0);
#else
      mbar_arrive(&B.g_full[buf]);                                // the staging tile is only an output: nothing to load
#endif
      if (++pk == ns) { pk = 0; ++pli; }
    };
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
        // four 16-byte chunks (16 channels) at a time: their loads first, then the math, then the stores
#pragma unroll
        for (int q0 = 0; q0 < DH / 4; q0 += 4) {
          uint4 gw[4];
#if GATE
#pragma unroll
          for (int e = 0; e < 4; ++e) gw[e] = lds128(xchunk(xg, r, q0 + e));
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int c = 4 * (q0 + e);
            gw[e] = make_uint4(__float_as_uint(__uint_as_float(ov[c + 0]) * sigmoid_kit(__uint_as_float(gw[e].x))),
                               __float_as_uint(__uint_as_float(ov[c + 1]) * sigmoid_kit(__uint_as_float(gw[e].y))),
                               __float_as_uint(__uint_as_float(ov[c + 2]) * sigmoid_kit(__uint_as_float(gw[e].z))),
                               __float_as_uint(__uint_as_float(ov[c + 3]) * sigmoid_kit(__uint_as_float(gw[e].w))));
          }
#else
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int c = 4 * (q0 + e);
            gw[e] = make_uint4(ov[c], ov[c + 1], ov[c + 2], ov[c + 3]);
          }
#endif
#pragma unroll
          for (int e = 0; e < 4; ++e) sts128(xchunk(xg, r, q0 + e), gw[e]);
        }
        fence_proxy_async();
        named_bar_sync(1, 128);
        if (r == 0) {
          tma_store_2d(&moa, xg, h * DH, (a0 + k) * L + m0);
          if (WB > 0) tma_store_2d(&mob, xg + XA, h * DH + 32, (a0 + k) * L + m0);
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
