// gattn_bwd.cu — the SWA atom block's GLOBAL attention backward on sm_100a (tcgen05), deterministic, in place of FlashAttention-4's:
//
//   S^T = K q^T, dP^T = V dO^T (TMEM, fp32);  P^T = 2^(S^T scale log2 e + LSE2);  dS^T = rn(P^T (dP^T - D))
//   dV = rn(sum_blocks rn(P^T) dO);  dK = rn(scale sum_blocks dS^T q);  dQ = rn(scale sum_keys dS K)
//   over the keys j < seqused[n] and the queries i < seqused[n]; every other row of dQ / dK / dV is written 0.
//
// LSE2 [N, H, S] is MINUS the log2 of the row sums the native forward (gattn_fwd.cu, lmul = -1) writes (added, not subtracted): the forward keeps no running maximum
// (q, k RMS-normalized without a weight: |q . k| scale <= sqrt(32)), so P = 2^(s scale log2 e + LSE2) exactly as the forward made it.
// D [N, H, S] = sum_d dO o per head (the out-projection backward writes it from the same bf16 dO this kernel reads).
// Qh | Kh | Vh / dQh | dKh | dVh head-major [N, 4, S, 32] bf16, dO token-major [N S, 128] bf16; S a multiple of 256.
//
// Key-major (FA2 / FA4 structure): an item is (sample n, head h, key pair kp): two 128-key tiles kp 256 + 128 w, one per softmax
// warpgroup w, both on the same 64-query blocks (q | dO | LSE2 | D, loaded once per stage). Per block: S^T / dP^T (SS MMAs, one MMA
// warp for both tiles) -> the row threads (one key row each) read both and release them at once (the next block's S^T / dP^T overlap
// the math) -> P^T to TMEM, dS^T to shared memory (bf16) -> dV += P^T dO, dK += dS^T q (TS / SS), dQ_b = dS K over both tiles' 256 keys
// (M = 64 queries, dS^T as the MN-major A, K as the MN-major B) into a double-buffered TMEM slot.
// dQ, deterministically: the P = S / 256 items of a head each add their dQ_b of query block qb into an fp32 accumulator ACC [N, H, S, 32]
// in a FIXED order, enforced by a counter per (n, h, qb) (ld.acquire / red.release, as FA4's deterministic mode). FA4 orders the adds by
// key block, which serializes the items of a head along the query axis; here item kp sweeps the query blocks starting at
// st_kp = kp Qn / Pn (wrapping), and the order at block qb is the order the items reach it (their step index (qb - st) mod Qn): in the
// steady state no item waits. Rank 0 stores (no zeroed ACC needed), the middle ranks bulk-add (cp.reduce.async.bulk .add.f32), the last
// rank reads ACC, adds its own, scales, writes dQ bf16 and resets the counter for the next launch.
// The items of a head must run at the same time (a waits-for cycle otherwise): the grid is a multiple of P (whole heads per round) and
// every CTA is resident (<= one CTA per SM, persistent). Items past seqused (kp >= ceil(sq / 256)) only write their zero rows.
// Warps: 0 TMA producer, 1 MMA issuer, 2 / 3 dQ reducers (blocks of even / odd index), 4-7 / 8-11 the row threads of tile 0 / 1.
// TMEM: S^T[w] at w 128, dP^T[w] at w 128 + 64, P^T[w] at 256 + w 64 (bf16), dQ_b[b] at 288 + b 64 (M = 64), dK[w] at 384 + w 64, dV[w] at
// 416 + w 64.
// SPLITB=1 (the split build, round 3 part 3): the (item, query step) sequence cut evenly across the CTAs where whole items leave SMs idle
// (one sample at L 3072 - 7168): dK / dV of a cut item combined in CTA order, dQ adders ordered by (step in their CTA's range, CTA).
// CSB=1 (the CLUSTER split build, r3-swa part 18): one item per cluster of CS = %cluster_nctarank CTAs (launch attribute 2 - 8), rank k
// the k-th contiguous share of the item's query blocks (its own staggered sweep inside the share); the dK | dV partials (fp32, 256 keys
// x 64) reduce-scattered over DSMEM by st.async to the key row's owner rank (rank k owns keys ceil(256 k / CS) ..) and summed there in rank
// order (deterministic), stored bf16 by plain stores. dQ: the counter-ordered adds as above, the adders of a block being rank k of each
// key pair's cluster, ordered by (step in the share, key pair). Every CTA resident (the host checks the co-resident clusters).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;
DEVI uint32_t mapa_u32(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
DEVI void st_async4(uint32_t dst, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
               :: "r"(dst), "r"(a), "r"(b), "r"(c), "r"(d), "r"(bar) : "memory");
}
template <int IMM> DEVI void tmem_ld16x2_16(uint32_t taddr, uint32_t (&r)[16]) {   // threads 0-15: lanes + t, columns from taddr; 16-31: + IMM
  asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16], %17;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
                 "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr), "n"(IMM));
}
DEVI f2 sub2(f2 a, f2 b) { f2 r; asm("sub.rn.f32x2 %0, %1, %2;" : "=l"(r.v) : "l"(a.v), "l"(b.v)); return r; }
// 2^x for a pair on the FMA pipe (FA4's technique, as augmented_attention/cuda/sm100/attn_fwd2.cu): degree-3 fit of 2^f, f = x - round(x), the
// exponent added to the bit pattern; x in [-30, 0] here (unmasked entries only), no clamp
DEVI f2 ex2_poly2(f2 x) {
  const f2 C = mk2(12582912.f, 12582912.f);
  const f2 j = add2(x, C);
  const f2 f = sub2(x, sub2(j, C));
  f2 p = fma2(mk2(0.0555041086648216f, 0.0555041086648216f), f, mk2(0.2402264923172785f, 0.2402264923172785f));
  p = fma2(p, f, mk2(0.6931471805599453f, 0.6931471805599453f));
  p = fma2(p, f, mk2(1.0f, 1.0f));
  return mk2(__int_as_float(__float_as_int(lo2(p)) + (__float_as_int(lo2(j)) << 23)), __int_as_float(__float_as_int(hi2(p)) + (__float_as_int(hi2(j)) << 23)));
}
// split mode: this piece's dK | dV partial (kv | vv, 64 floats of key row r) to WS slot `mine`; the piece completing the item's counter
// adds every piece's partial in CTA order (its own from registers) into kv | vv and returns true. Called before a piece's first block math (its registers
// free; inline: a call spills the kernel's live registers).
__device__ __forceinline__ bool split_combine(float* __restrict__ WS, int* __restrict__ ctr, int* last, uint32_t (&kv)[32], uint32_t (&vv)[32], int r, int w, int cid, int c0,
                                           int c1, int ist, unsigned T, int G0) {
  constexpr int WSF = 128 * 64;
  auto start_of = [&](int cc) { return (int)((unsigned)cc * T / (unsigned)G0); };
  float* wo = WS + ((size_t)(cid * 2 + (start_of(cid) >= ist ? 0 : 1)) * 2 + w) * WSF + r * 64;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    __stcg(reinterpret_cast<uint4*>(wo + 4 * q), make_uint4(kv[4 * q], kv[4 * q + 1], kv[4 * q + 2], kv[4 * q + 3]));
    __stcg(reinterpret_cast<uint4*>(wo + 32 + 4 * q), make_uint4(vv[4 * q], vv[4 * q + 1], vv[4 * q + 2], vv[4 * q + 3]));
  }
  __threadfence();
  asm volatile("bar.sync %0, 128;" :: "r"(1 + w) : "memory");
  if (r == 0) *last = atomicAdd(ctr, 1) == c1 - c0;
  asm volatile("bar.sync %0, 128;" :: "r"(1 + w) : "memory");
  if (!*reinterpret_cast<volatile int*>(last)) return false;
  __threadfence();
  // every partial from WS, its own included (written above): kv | vv are free accumulators, CTA order
#pragma unroll
  for (int k = 0; k < 32; ++k) { kv[k] = 0u; vv[k] = 0u; }
  for (int cc = c0; cc <= c1; ++cc) {
    const float* src = WS + ((size_t)(cc * 2 + (start_of(cc) >= ist ? 0 : 1)) * 2 + w) * WSF + r * 64;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const float4 x = __ldcg(reinterpret_cast<const float4*>(src + 4 * q)), y = __ldcg(reinterpret_cast<const float4*>(src + 32 + 4 * q));
      kv[4 * q] = __float_as_uint(__uint_as_float(kv[4 * q]) + x.x); kv[4 * q + 1] = __float_as_uint(__uint_as_float(kv[4 * q + 1]) + x.y);
      kv[4 * q + 2] = __float_as_uint(__uint_as_float(kv[4 * q + 2]) + x.z); kv[4 * q + 3] = __float_as_uint(__uint_as_float(kv[4 * q + 3]) + x.w);
      vv[4 * q] = __float_as_uint(__uint_as_float(vv[4 * q]) + y.x); vv[4 * q + 1] = __float_as_uint(__uint_as_float(vv[4 * q + 1]) + y.y);
      vv[4 * q + 2] = __float_as_uint(__uint_as_float(vv[4 * q + 2]) + y.z); vv[4 * q + 3] = __float_as_uint(__uint_as_float(vv[4 * q + 3]) + y.w);
    }
  }
  if (r == 0) *ctr = 0;                                                    // every piece has counted: ready for the next launch
  return true;
}
template <bool V> struct BoolC { static constexpr bool value = V; };
DEVI int ld_acquire(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
DEVI void red_release(int* p, int v) { asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI void bulk_s2g(void* g, uint32_t s, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(g), "r"(s), "r"(bytes) : "memory");
}
DEVI void bulk_add_f32(void* g, uint32_t s, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;" :: "l"(g), "r"(s), "r"(bytes) : "memory");
}
DEVI float4 ldcg4(const float* p) { float4 v; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory"); return v; }

#ifndef PMOD
#define PMOD 4                           // of every PMOD exponential pairs (unmasked blocks), PCNT on the FMA pipe (ex2_poly2), the rest on MUFU
#endif
#ifndef PCNT
#define PCNT 1
#endif
#ifndef SPIN_NS
#define SPIN_NS 64                       // back-off of the dQ counter poll
#endif
#ifndef NSTG
#define NSTG 4                           // dQ_b staging buffers (the reducers free one a block after its bulk add)
#endif
#ifndef PROBE
#define PROBE 0                          // timing probes (bits; wrong results): 1 no counter waits, 2 no dQ global traffic, 4 no exponentials,
                                         // 16 no dQ MMAs, 32 no dK / dV MMAs, 64 no S^T / dP^T MMAs, 128 no row math, 256 no P / dS stores
#endif
#ifndef ST
#define ST 6                                                               // query-block stages (q | dO | LSE2 | D, 9 KB each)
#endif
#ifndef QR
#define QR 2                                                               // item stages (K0 | K1 | V0 | V1, 32 KB each)
#endif
constexpr int BQ = 64, DH = 32, KM = 128, H = 4;
constexpr int TKV = KM * 64, TQ = BQ * 64;                                 // K / V tile 8 KB, q / dO block 4 KB
constexpr int IST = 4 * TKV;
constexpr int BLS = 2 * TQ + 1024;                                         // q | dO | LSE2 (256 B) | D (256 B) | pad
constexpr int O_IT = 0, O_BL = QR * IST, O_DS = O_BL + ST * BLS;          // then dS^T (2 x [128][64] bf16, SW128)
constexpr int XS = 128 * 64;
#ifndef MLI
#define MLI 0                            // 1: the TMA producer's loads issued from lanes 0 - 3 (part 18: level, 59.3 / 59.2 us alone at L 2048 -- not issue-bound)
#endif
#ifndef CSB
#define CSB 0                          // 1: the cluster split (one item per cluster, query shares, dK | dV reduce-scattered over DSMEM)
#endif
#ifndef CSN
#define CSN 0                            // CSB: the cluster size fixed at build time (0: %cluster_nctarank)
#endif
constexpr int O_X = O_DS + 2 * 16384, O_QS = O_X + (CSB ? 0 : 4 * XS);    // dK / dV staging (2 tiles x 2, SW64; CSB: none, plain stores); dQ_b staging (NSTG x [64][32] f32)
#ifndef SPLITB
#define SPLITB 0                         // 1: split mode (one sample / few items: the (item, query step) sequence cut evenly across the CTAs)
#endif
constexpr int NMAX = 256, PMAX = 600;                                      // split mode: samples, pieces of one CTA
constexpr int O_BAR = O_QS + NSTG * 8192, O_PF = O_BAR + 1024, O_PL = O_PF + 4 * (NMAX + 4);
// CSB receive area: [src rank][chunk 0..15][RS key rows] x 16 B (dK chunks 0..7, dV 8..15)
// rank o owns key rows rs(o) .. rs(o + 1) - 1, rs(o) = ceil(256 o / CS) (CS 3: 86 / 85 / 85 rows); RS = the largest count rounded up to 32
// (CS 2 / 3 / 4 / 8: 128 / 96 / 64 / 32) is the area's row stride: at most 72 KB
__host__ __device__ constexpr uint32_t row_start(uint32_t o, uint32_t cs) { return (o * 256u + cs - 1u) / cs; }
__host__ __device__ constexpr uint32_t rcv_stride(uint32_t cs) { return ((256u + cs - 1u) / cs + 31u) / 32u * 32u; }
constexpr int O_RCV = (O_BAR + 1024 + 1023) / 1024 * 1024, RCV_AREA = 3 * 16 * 96 * 16;
constexpr int SMEM_BYTES = CSB ? O_RCV + RCV_AREA : O_PL + 16 * PMAX;
static_assert(!(CSB && SPLITB), "CSB: whole-item build only");
static_assert(!CSB || QR == 1, "CSB: one item a CTA (-DQR=1 frees the receive area)");
constexpr int WSF = 128 * 64;                                              // split: floats of one partial (dK | dV of 128 keys, unscaled)
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(NSTG >= 4 && NSTG % 2 == 0, "a reducer frees a staging buffer at its next block: it needs two per reducer");
static_assert(O_BL % 1024 == 0 && BLS % 1024 == 0 && O_DS % 1024 == 0 && O_X % 1024 == 0 && O_QS % 1024 == 0, "tile alignment");
constexpr uint32_t T_S = 0, T_P = 256, T_DQ = 288, T_ACC = 384;
constexpr uint32_t I_S = idesc_bf16(128, BQ), I_KV = idesc_bf16(128, DH, 0, 1), I_DQ = idesc_bf16(64, DH, 1, 1);
constexpr float LOG2E = 1.4426950408889634f;
#ifndef TRACE
#define TRACE 0                          // -DTRACE=1: clock64 stamps of CTA 0, block g < 64, into TRb[16 g + k]
#endif
#define EVT(g, k) do { if (TRACE && blockIdx.x == 0 && (g) < 64) TRb[16 * (g) + (k)] = clock64(); } while (0)

struct Bars {
  uint64_t it_full[QR], it_empty[QR], bl_full[ST], bl_empty[ST], s_full[2], s_free[2], p_full[2], p_free, acc_full, acc_free[2],
      dq_full[2], dq_free[2], stg_full[NSTG], stg_free[NSTG];
  uint64_t rcv;                                                            // CSB: every rank's partial rows of mine have landed
  uint32_t tmem;
  int npieces, last[2];
};

extern "C" __global__ void __launch_bounds__(384, 1)
swa_gattn_bwd_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                    const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mlse, const __grid_constant__ CUtensorMap mdd,
                    const __grid_constant__ CUtensorMap mdk_o, const __grid_constant__ CUtensorMap mdv_o,
                    const int* __restrict__ SEQU, float* __restrict__ ACC, int* __restrict__ CNT, uint16_t* __restrict__ DQ,
                    uint16_t* __restrict__ DK, uint16_t* __restrict__ DV, int S, int N, float scale,
                    float* __restrict__ WS, int* __restrict__ CNT2, unsigned long long* __restrict__ TRb) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  int* PF = reinterpret_cast<int*>(sm + O_PF);                             // split: step prefix over the samples [N + 1]
  int4* PL = reinterpret_cast<int4*>(sm + O_PL);                           // split: this CTA's pieces {n, active item of the sample, t0, len}
  constexpr bool SPL = SPLITB == 1;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int P = S / (2 * KM), QB = S / BQ;                                // items per head, query blocks per head
  const int items = N * H * P, G0 = (int)gridDim.x;
  // split (SPLITB 1; few items for the grid, e.g. one sample: S / 256 x 4 items <= 128 of 148 SMs): the (active item, query step)
  // sequence -- items in (sample, head, key pair) order, Qn steps each -- is cut into gridDim.x equal contiguous ranges, one per CTA.
  // dK / dV of an item cut by a range boundary: each piece's fp32 partial to WS (slot (CTA, first / last piece of its range, warpgroup)),
  // the piece completing the item's counter (CNT2 [item][w]) adds them in CTA order and stores. dQ: the adders of a query block are
  // ordered by (step within their CTA's range, CTA) -- every CTA runs its range in that order and waits only on smaller keys (no cycle),
  // and the CTAs advance together, so the order is the arrival order in the steady state.
  const int cid = (int)blockIdx.x;
  unsigned T = 1;                                                          // split: total steps (< 2^24: host bound), 32-bit math
  auto start_of = [&](int cc) { return (int)((unsigned)cc * T / (unsigned)G0); };
  auto cta_of = [&](int x) { return (int)(((unsigned)(x + 1) * (unsigned)G0 - 1u) / T); };
  // CSB: CTAs cid / csn = the item, csr = cid mod csn its query share; csn = the cluster size or (CSN build) CSN, the cluster then holding
  // cluster size / CSN items (their rank groups at cluster ranks cb .. cb + CSN - 1)
  uint32_t csn = 1, csr = 0, cb = 0;
  if (CSB) {
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(csn));
    if (CSN) csn = CSN;                                                    // a per-size build: the exchange loops unrolled
    csr = (uint32_t)cid % csn;
    cb = cluster_rank() - csr;
  }
  int my_items = CSB ? 1 : (items > cid) ? (items - cid + G0 - 1) / G0 : 0;   // CSB: the host launches exactly items x CS CTAs
  // item li of this CTA: sample n, head h, key pair kp; seqused sq, query blocks Qn, contributing items Pn, the sweep's first block st;
  // its steps t0 .. t0 + len - 1 (whole: 0 .. Qn - 1); split: the item's first global step ist, its CTAs c0 .. c1, its index id;
  // the swept query blocks qb0 .. qb0 + qs - 1 (CSB: this rank's share; else 0 .. Qn - 1); CSB: clact = the item has keys (kp < Pn)
  struct It { int n, h, kp, sq, Qn, Pn, st, t0, len, c0, c1, id, ist, qb0, qs; bool act, clact; };
  auto item_of = [&](int li) {
    It e;
    if (SPL) {
      const int4 p = PL[li];
      e.n = p.x; e.sq = min(max(__ldg(SEQU + e.n), 0), S);
      e.Qn = (e.sq + BQ - 1) / BQ; e.Pn = (e.sq + 2 * KM - 1) / (2 * KM);
      e.h = p.y / e.Pn; e.kp = p.y - e.h * e.Pn; e.t0 = p.z; e.len = p.w; e.act = true;
      e.ist = PF[e.n] + p.y * e.Qn;
      e.c0 = cta_of(e.ist); e.c1 = cta_of(e.ist + e.Qn - 1); e.id = (e.n * H + e.h) * P + e.kp;
      e.qb0 = 0; e.qs = e.Qn; e.clact = true;
    } else {
      const int wi = CSB ? cid / (int)csn : cid + li * G0, hx = wi / P;
      e.kp = wi - hx * P; e.n = hx / H; e.h = hx - e.n * H;
      e.sq = min(max(__ldg(SEQU + e.n), 0), S);
      e.Qn = (e.sq + BQ - 1) / BQ; e.Pn = (e.sq + 2 * KM - 1) / (2 * KM);
      e.act = e.kp < e.Pn; e.t0 = 0; e.len = e.Qn; e.c0 = e.c1 = cid; e.id = 0; e.ist = 0;
      e.qb0 = 0; e.qs = e.Qn; e.clact = e.act;
      if (CSB) {
        e.qb0 = (int)csr * e.Qn / (int)csn;
        e.qs = ((int)csr + 1) * e.Qn / (int)csn - e.qb0;
        e.len = e.qs;
        e.act = e.act && e.qs > 0;                                         // a rank without query blocks still sends its (zero) partial
      }
    }
    e.st = e.act ? (e.kp * e.qs) / e.Pn : 0;
    return e;
  };
  auto qb_of = [](const It& e, int t) { const int q = e.st + t; return e.qb0 + (q >= e.qs ? q - e.qs : q); };

  if (tid == 0) EVT(61, 0);
  if (tid == 0) {
    for (int s = 0; s < QR; ++s) { mbar_init(&B.it_full[s], 1); mbar_init(&B.it_empty[s], 1); }
    for (int s = 0; s < ST; ++s) { mbar_init(&B.bl_full[s], 1); mbar_init(&B.bl_empty[s], 1); }
    for (int w = 0; w < 2; ++w) {
      mbar_init(&B.s_full[w], 1); mbar_init(&B.s_free[w], 4); mbar_init(&B.p_full[w], 4); mbar_init(&B.acc_free[w], 4);
      mbar_init(&B.dq_full[w], 1); mbar_init(&B.dq_free[w], 4);
    }
    mbar_init(&B.p_free, 1); mbar_init(&B.acc_full, 1);
    for (int s = 0; s < NSTG; ++s) { mbar_init(&B.stg_full[s], 4); mbar_init(&B.stg_free[s], 1); }
    if (CSB) {                                                             // armed before the cluster barrier below: CS partials of my rows
      mbar_init(&B.rcv, 1);
      mbar_expect_tx(&B.rcv, csn * (row_start(csr + 1u, csn) - row_start(csr, csn)) * 256u);
    }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (tid == 0) EVT(61, 1);
  if (CSB) cluster_sync();                                                 // every rank's receive barrier armed before any st.async
  pdl_wait();                                                            // every global access is below
  if (tid == 0) EVT(61, 2);
  pdl_launch();
  const uint32_t tmem = B.tmem;
  if (SPL) {
    if (tid == 0) {
      int acc = 0;
      PF[0] = 0;
      for (int n = 0; n < N; ++n) {
        const int sq = min(max(__ldg(SEQU + n), 0), S);
        acc += H * ((sq + 2 * KM - 1) / (2 * KM)) * ((sq + BQ - 1) / BQ);
        PF[n + 1] = acc;
      }
      T = (unsigned)max(acc, 1);
      const int B1 = min(start_of(cid + 1), acc);
      int x = start_of(cid);
      int np = 0, n = 0;
      while (x < B1) {
        if (np == PMAX) __trap();                                          // the host keeps split launches below PMAX pieces
        while (PF[n + 1] <= x) ++n;
        const int sq = min(max(__ldg(SEQU + n), 0), S), qn = (sq + BQ - 1) / BQ;
        const int off = (int)(x - PF[n]), itl = off / qn;
        const int e = min(B1, PF[n] + (itl + 1) * qn);
        PL[np++] = make_int4(n, itl, off - itl * qn, e - x);
        x = e;
      }
      B.npieces = np;
    }
    __syncthreads();
    T = (unsigned)max(PF[N], 1);
    my_items = B.npieces;
  }

  struct Cur { int li, ai, t; It e; };
  auto seek = [&](Cur& c) {                                              // the next active item from c.li on (c.t = 0)
    while (c.li < my_items) { c.e = item_of(c.li); if (c.e.act) return; ++c.li; }
  };
  auto step = [&](Cur& c) { if (++c.t == c.e.len) { c.t = 0; ++c.ai; ++c.li; seek(c); } };

  if (warp == 0 && MLI) {
    // ------------------------------------------------------------------------------------------------ TMA producer, multi-lane issue
    // (r3-swa part 18): one thread issues about one TMA per ~600 clocks whatever the box, so a block's four loads (q, dO, LSE2, D) from
    // one lane cost ~2400 clocks a block; lanes 0 - 3 issue one each (lane 0 waits and arms the barrier first)
    int g = 0, ai = 0;
    for (int li = 0; li < my_items; ++li) {
      const It e = item_of(li);
      if (!e.act) continue;
      const int is = ai % QR, hrow = (e.n * H + e.h) * S;
      if (lane == 0) {
        if (ai >= QR) mbar_wait(&B.it_empty[is], ((ai / QR) - 1) & 1);
        mbar_expect_tx(&B.it_full[is], IST);
      }
      __syncwarp();
      if (lane < 4) {
        const int w = lane & 1, row = hrow + e.kp * 2 * KM + w * KM;
        tma_load_2d(su + O_IT + is * IST + (lane >> 1) * 2 * TKV + w * TKV, (lane >> 1) ? &mv : &mk, &B.it_full[is], 0, row);
      }
      for (int t = 0; t < e.len; ++t, ++g) {
        const int s = g % ST, qb = qb_of(e, e.t0 + t);
        const uint32_t st = su + O_BL + s * BLS;
        if (lane == 0) {
          if (g >= ST) mbar_wait(&B.bl_empty[s], ((g / ST) - 1) & 1);
          EVT(g, 14);
          mbar_expect_tx(&B.bl_full[s], 2 * TQ + 2 * BQ * 4);
        }
        __syncwarp();
        const int row = hrow + qb * BQ;
        if (lane == 0) tma_load_2d(st, &mq, &B.bl_full[s], 0, row);
        else if (lane == 1) tma_load_2d(st + TQ, &mdo, &B.bl_full[s], e.h * DH, e.n * S + qb * BQ);
        else if (lane == 2) tma_load_2d(st + 2 * TQ, &mlse, &B.bl_full[s], 0, row / BQ);
        else if (lane == 3) tma_load_2d(st + 2 * TQ + 256, &mdd, &B.bl_full[s], 0, row / BQ);
      }
      ++ai;
    }
  } else if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      int g = 0, ai = 0;
      for (int li = 0; li < my_items; ++li) {
        const It e = item_of(li);
        if (!e.act) continue;
        const int is = ai % QR, hrow = (e.n * H + e.h) * S;
        if (ai >= QR) mbar_wait(&B.it_empty[is], ((ai / QR) - 1) & 1);
        mbar_expect_tx(&B.it_full[is], IST);
        for (int w = 0; w < 2; ++w) {
          const int row = hrow + e.kp * 2 * KM + w * KM;
          tma_load_2d(su + O_IT + is * IST + w * TKV, &mk, &B.it_full[is], 0, row);
          tma_load_2d(su + O_IT + is * IST + (2 + w) * TKV, &mv, &B.it_full[is], 0, row);
        }
        for (int t = 0; t < e.len; ++t, ++g) {
          const int s = g % ST, qb = qb_of(e, e.t0 + t);
          if (g >= ST) mbar_wait(&B.bl_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_BL + s * BLS;
          EVT(g, 14);
          mbar_expect_tx(&B.bl_full[s], 2 * TQ + 2 * BQ * 4);
          const int row = hrow + qb * BQ;
          tma_load_2d(st, &mq, &B.bl_full[s], 0, row);
          tma_load_2d(st + TQ, &mdo, &B.bl_full[s], e.h * DH, e.n * S + qb * BQ);
          tma_load_2d(st + 2 * TQ, &mlse, &B.bl_full[s], 0, row / BQ);
          tma_load_2d(st + 2 * TQ + 256, &mdd, &B.bl_full[s], 0, row / BQ);
        }
        ++ai;
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer (both tiles)
    auto issue_s = [&](int G, const Cur& c) {
      const int s = G % ST, is = c.ai % QR;
      if (c.t == 0) mbar_wait(&B.it_full[is], (c.ai / QR) & 1);
      mbar_wait(&B.bl_full[s], (G / ST) & 1);
      const uint64_t dq = desc_sw64(su + O_BL + s * BLS), ddo = desc_sw64(su + O_BL + s * BLS + TQ);
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        if (G >= 1) mbar_wait(&B.s_free[w], (G - 1) & 1);                // the row threads have read S^T / dP^T[w] of G - 1
        tc_fence_after();
        const uint64_t dk = desc_sw64(su + O_IT + is * IST + w * TKV), dv = desc_sw64(su + O_IT + is * IST + (2 + w) * TKV);
        const uint32_t tS = tmem + T_S + w * 128;
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < DH / 16; ++ks) if (!(PROBE & 64)) umma_ss(tS, dk + (uint64_t)(ks * 2), dq + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
#pragma unroll
          for (int ks = 0; ks < DH / 16; ++ks) if (!(PROBE & 64)) umma_ss(tS + 64, dv + (uint64_t)(ks * 2), ddo + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
          tc_commit(&B.s_full[w]);
        }
        __syncwarp();
      }
    };
    Cur c, nx;
    c.li = 0; c.ai = 0; c.t = 0; seek(c);
    nx = c;
    if (nx.li < my_items) { issue_s(0, nx); step(nx); }
    for (int G = 0; c.li < my_items; ++G) {
      const int s = G % ST, is = c.ai % QR;
      if (nx.li < my_items) { issue_s(G + 1, nx); step(nx); }              // S^T / dP^T(G + 1) first
      if (lane == 0) EVT(G, 9);
      const uint64_t dq = desc_sw64(su + O_BL + s * BLS), ddo = desc_sw64(su + O_BL + s * BLS + TQ);   // MN-major B
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        mbar_wait(&B.p_full[w], G & 1);
        if (lane == 0) EVT(G, 10 + w);
        if (c.t == 0 && c.ai >= 1) mbar_wait(&B.acc_free[w], (c.ai - 1) & 1);   // the previous item's dK / dV have been read out
        tc_fence_after();
        const uint32_t sds = su + O_DS + w * 16384;
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            if (!(PROBE & 32)) umma_ts(tmem + T_ACC + w * 64 + 32, tmem + T_P + w * 64 + ks * 8, ddo + (uint64_t)(ks * 1024 >> 4), I_KV, (c.t > 0 || ks > 0) ? 1u : 0u);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)                                   // dS^T from shared memory, K-major (16 queries = 32 B a step)
            if (!(PROBE & 32)) umma_ss(tmem + T_ACC + w * 64, desc_k128(sds) + (uint64_t)(ks * 2), dq + (uint64_t)(ks * 1024 >> 4), I_KV, (c.t > 0 || ks > 0) ? 1u : 0u);
        }
        __syncwarp();
      }
      const int b = G & 1;
      if (G >= 2) mbar_wait(&B.dq_free[b], ((G >> 1) - 1) & 1);  // dQ_b of G - 2 has been read out
      if (lane == 0) EVT(G, 12);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int w = 0; w < 2; ++w) {
          const uint64_t dk = desc_sw64(su + O_IT + is * IST + w * TKV);   // K as MN-major B (16 keys = 1 KB a step)
          const uint32_t sds = su + O_DS + w * 16384;
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            if (!(PROBE & 16)) umma_ss(tmem + T_DQ + b * 64, desc_mn128(sds, 0) + (uint64_t)(ks * 128), dk + (uint64_t)(ks * 64), I_DQ, (w > 0 || ks > 0) ? 1u : 0u);
        }
        tc_commit(&B.p_free);                                              // P^T, dS^T of both tiles read
        tc_commit(&B.dq_full[b]);
        tc_commit(&B.bl_empty[s]);
        if (c.t == c.e.len - 1) { tc_commit(&B.acc_full); tc_commit(&B.it_empty[is]); }
      }
      __syncwarp();
      if (lane == 0) EVT(G, 13);
      step(c);
    }
  } else if (warp == 2 || warp == 3) {
    // ------------------------------------------------------------------------------------------------ dQ reducers: warp 2 + b, blocks G & 1 == b
    // A block's bulk store / add is released (counter + staging buffer) at this warp's next block, BEFORE its counter poll: a release never
    // waits on a poll (no waits-for cycle between CTAs), and the bulk group has two blocks to complete.
    const int wb = warp - 2;
    int* pend = nullptr;                                                   // a counter whose release waits for this lane's bulk group
    int pend_sb = 0;
    auto flush = [&]() {
      if (pend != nullptr && lane == 0) { tma_store_wait0(); fence_proxy_async_global(); red_release(pend, 1); mbar_arrive(&B.stg_free[pend_sb]); }
      pend = nullptr;
      __syncwarp();
    };
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      const It e = item_of(li);
      if (!e.act) continue;
      const int tau0 = SPL ? e.ist - start_of(cid) : 0;           // split: my local step of the item's step 0
      for (int t = 0; t < e.len; ++t, ++G) {
        if ((G & 1) != wb) continue;
        const int ta = e.t0 + t, qb = qb_of(e, ta), sb = G % NSTG;
        const uint32_t stg = su + O_QS + sb * 8192;
        const float* stgp = reinterpret_cast<const float*>(sm + O_QS + sb * 8192);
        mbar_wait(&B.stg_full[sb], (G / NSTG) & 1);
        if (lane == 0) EVT(G, 15);
        flush();                                                           // the previous block of this warp: landed -> released
        // rank: the items that reach qb before this one (smaller step index (qb - st) mod Qn)
        int rank = 0;
        for (int k0 = 0; k0 < e.Pn; k0 += 32) {
          const int k = k0 + lane;
          bool before = false;
          if (CSB) {                                                         // rank csr of item k: step in the share, then key pair
            if (k < e.Pn) {
              int tk = qb - e.qb0 - (k * e.qs) / e.Pn;
              if (tk < 0) tk += e.qs;
              before = tk < ta || (tk == ta && k < e.kp);
            }
          } else if (k < e.Pn) {
            int tk = qb - (k * e.Qn) / e.Pn;
            if (tk < 0) tk += e.Qn;
            if (SPL) {                                                       // order: (step within its CTA's range, CTA)
              const int gk = e.ist + (k - e.kp) * e.Qn + tk;
              const int ck = cta_of(gk);
              const int tk2 = gk - start_of(ck), tm = tau0 + ta;
              before = tk2 < tm || (tk2 == tm && ck < cid);
            } else {
              before = tk < ta;
            }
          }
          rank += __popc(__ballot_sync(~0u, before));
        }
        const size_t row0 = (size_t)(e.n * H + e.h) * S + (size_t)qb * BQ;
        float* acc = ACC + row0 * DH;
        int* cnt = CNT + (e.n * H + e.h) * QB + qb;
        if (PROBE & 2) { __syncwarp(); if (lane == 0) mbar_arrive(&B.stg_free[sb]); __syncwarp(); continue; }
        if (rank > 0 && !(PROBE & 1)) {
          while (ld_acquire(cnt) != rank) { if (SPIN_NS) __nanosleep(SPIN_NS); }
          fence_proxy_async_global();
        }
        if (rank == e.Pn - 1) {
          // the last: ACC (ranks before) + own -> dQ (bf16, scaled); lane: rows 2 lane, 2 lane + 1
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int row = 2 * lane + rr;
            uint32_t o[16];
#pragma unroll
            for (int c4 = 0; c4 < 8; ++c4) {
              const int cc = (c4 + lane) & 7;                              // rotated: fewer bank conflicts
              float4 v = *reinterpret_cast<const float4*>(stgp + row * DH + 4 * cc);
              if (rank > 0) {
                const float4 a = ldcg4(acc + row * DH + 4 * cc);
                v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
              }
              o[2 * cc] = pack_bf16(v.x * scale, v.y * scale);
              o[2 * cc + 1] = pack_bf16(v.z * scale, v.w * scale);
            }
            uint4* dst = reinterpret_cast<uint4*>(DQ + (row0 + row) * DH);
#pragma unroll
            for (int q = 0; q < 4; ++q) dst[q] = make_uint4(o[4 * q], o[4 * q + 1], o[4 * q + 2], o[4 * q + 3]);
          }
          __syncwarp();
          if (lane == 0) {
            mbar_arrive(&B.stg_free[sb]);
            if (rank > 0) *cnt = 0;                                        // ready for the next launch
          }
        } else {
          if (lane == 0) {
            if (rank == 0) bulk_s2g(acc, stg, 8192); else bulk_add_f32(acc, stg, 8192);
            tma_store_commit();
          }
          pend = cnt; pend_sb = sb;
        }
        __syncwarp();
      }
    }
    flush();
  } else {
    // ------------------------------------------------------------------------------------------------ P^T / dS^T of tile w, one key row per thread
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const float SC = scale * LOG2E;
    const f2 SC2 = mk2(SC, SC);
    const uint32_t sds = su + O_DS + w * 16384;
    int G = 0, ai = 0;
    int pend = -1, pli = 0;
    auto epilogue = [&](int pai, int li) {                                 // dK / dV of the item -> SW64 staging -> TMA stores
      const It e = item_of(li);
      const int n = e.n, h = e.h, j0 = e.kp * 2 * KM + w * KM;
      mbar_wait(&B.acc_full, pai & 1);
      tc_fence_after();
      uint32_t kv[32], vv[32];
      tmem_ld32(trow + T_ACC + w * 64, kv);
      tmem_ld32(trow + T_ACC + w * 64 + 32, vv);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.acc_free[w]);
      if (SPL && e.c0 < e.c1) {
        if (!split_combine(WS, CNT2 + 2 * e.id + w, &B.last[w], kv, vv, (int)r, w, cid, e.c0, e.c1, e.ist, T, G0)) return;
      }
      const uint32_t xk = su + O_X + (2 * w) * XS, xv = xk + XS;
      if (r == 0) tma_store_wait_read0();
      named_bar_sync(1 + w, 128);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint4 a, c;
        a.x = pack_bf16(__uint_as_float(kv[8 * q]) * scale, __uint_as_float(kv[8 * q + 1]) * scale);
        a.y = pack_bf16(__uint_as_float(kv[8 * q + 2]) * scale, __uint_as_float(kv[8 * q + 3]) * scale);
        a.z = pack_bf16(__uint_as_float(kv[8 * q + 4]) * scale, __uint_as_float(kv[8 * q + 5]) * scale);
        a.w = pack_bf16(__uint_as_float(kv[8 * q + 6]) * scale, __uint_as_float(kv[8 * q + 7]) * scale);
        c.x = pack_bf16(__uint_as_float(vv[8 * q]), __uint_as_float(vv[8 * q + 1]));
        c.y = pack_bf16(__uint_as_float(vv[8 * q + 2]), __uint_as_float(vv[8 * q + 3]));
        c.z = pack_bf16(__uint_as_float(vv[8 * q + 4]), __uint_as_float(vv[8 * q + 5]));
        c.w = pack_bf16(__uint_as_float(vv[8 * q + 6]), __uint_as_float(vv[8 * q + 7]));
        sts128(xk + sw64(r, q), a);
        sts128(xv + sw64(r, q), c);
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (r == 0) {
        tma_store_2d(&mdk_o, xk, 0, (n * H + h) * S + j0);
        tma_store_2d(&mdv_o, xv, 0, (n * H + h) * S + j0);
        tma_store_commit();
      }
    };
    // dQ_b of block Gp (this tile's turn: Gp & 1 == w) -> fp32 staging [64][32] for the reducer warp 2 + w
    const uint32_t qr = 16 * (warp & 3) + (lane & 15), qh = lane >> 4;
    auto dq_out = [&](int Gp) {
      const int bb = Gp & 1;
      mbar_wait(&B.dq_full[bb], (Gp >> 1) & 1);
      tc_fence_after();
      uint32_t qv[16];
      tmem_ld16x2_16<16>(tmem + (lb << 16) + T_DQ + bb * 64, qv);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dq_free[bb]);
      const int sb = Gp % NSTG;
      if (Gp >= NSTG) mbar_wait(&B.stg_free[sb], ((Gp / NSTG) - 1) & 1);  // the reducer is done with this staging buffer's previous block
      const uint32_t dst = su + O_QS + sb * 8192 + qr * 128 + qh * 64;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int cc = (c + lane) & 3;
        sts128(dst + cc * 16, make_uint4(qv[4 * cc], qv[4 * cc + 1], qv[4 * cc + 2], qv[4 * cc + 3]));
      }
      fence_proxy_async();                                                 // the bulk copy reads it through the async proxy
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.stg_full[sb]);
    };
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    if (SPL)                                                      // split: the zero rows of every item, items wi = cid mod G0
      for (int wi = cid; wi < items; wi += G0) {
        const int hx = wi / P, kp = wi - hx * P, n = hx / H, h = hx - n * H;
        const int sq = min(max(__ldg(SEQU + n), 0), S), j = kp * 2 * KM + w * KM + (int)r;
        const size_t hrow = (size_t)(n * H + h) * S;
        if (j >= (sq + BQ - 1) / BQ * BQ) {
          uint4* d = reinterpret_cast<uint4*>(DQ + (hrow + j) * DH);
          d[0] = z4; d[1] = z4; d[2] = z4; d[3] = z4;
        }
        if (kp >= (sq + 2 * KM - 1) / (2 * KM)) {
          uint4* dk = reinterpret_cast<uint4*>(DK + (hrow + j) * DH);
          uint4* dv = reinterpret_cast<uint4*>(DV + (hrow + j) * DH);
          dk[0] = z4; dk[1] = z4; dk[2] = z4; dk[3] = z4;
          dv[0] = z4; dv[1] = z4; dv[2] = z4; dv[3] = z4;
        }
      }
    for (int li = 0; li < my_items; ++li) {
      const It e = item_of(li);
      const int j0 = e.kp * 2 * KM + w * KM, j = j0 + (int)r;
      const size_t hrow = (size_t)(e.n * H + e.h) * S;
      if (!SPL && (!CSB || csr == 0) && j >= e.Qn * BQ) {                          // dQ rows no query block covers
        uint4* d = reinterpret_cast<uint4*>(DQ + (hrow + j) * DH);
        d[0] = z4; d[1] = z4; d[2] = z4; d[3] = z4;
      }
      if (CSB && !e.act) {                                                         // CSB: rank 0 writes the zero dK | dV of an empty item
        if (!e.clact && csr == 0) {
          uint4* dk = reinterpret_cast<uint4*>(DK + (hrow + j) * DH);
          uint4* dv = reinterpret_cast<uint4*>(DV + (hrow + j) * DH);
          dk[0] = z4; dk[1] = z4; dk[2] = z4; dk[3] = z4;
          dv[0] = z4; dv[1] = z4; dv[2] = z4; dv[3] = z4;
        }
        continue;
      }
      if (!SPL && !e.act) {                                                        // keys past seqused: dK = dV = 0
        uint4* dk = reinterpret_cast<uint4*>(DK + (hrow + j) * DH);
        uint4* dv = reinterpret_cast<uint4*>(DV + (hrow + j) * DH);
        dk[0] = z4; dk[1] = z4; dk[2] = z4; dk[3] = z4;
        dv[0] = z4; dv[1] = z4; dv[2] = z4; dv[3] = z4;
        continue;
      }
      const bool kok = j < e.sq;
      for (int t = 0; t < e.len; ++t, ++G) {
        const int s = G % ST, qb = qb_of(e, e.t0 + t), qv = e.sq - qb * BQ;     // valid query columns of the block
        if (r == 0) EVT(G, w ? 7 : 0);
        if (SPL && t == 0 && pend >= 0) { epilogue(pend, pli); pend = -1; }   // split: before the math (its registers free)
        mbar_wait(&B.s_full[w], G & 1);
        tc_fence_after();
        const float* lse = reinterpret_cast<const float*>(sm + O_BL + s * BLS + 2 * TQ);
        const float* dd = lse + 64;
        if (r == 0 && w == 0) EVT(G, 1);
        uint32_t pp[32], ds[32];
        // the math of the block; MASK (the last query block of a ragged sequence, or a warp with keys past seqused): masked entries get
        // an infinite LSE (P = 0) -- one select per column there, no test at all in the common case
        auto math = [&](auto maskc) {
          constexpr bool MASK = decltype(maskc)::value;
          const float lpen = kok ? 0.f : INFINITY;
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            uint32_t sv[32], pv[32];
            tmem_ld32(trow + T_S + w * 128 + 32 * hf, sv);
            tmem_ld32(trow + T_S + w * 128 + 64 + 32 * hf, pv);
            tmem_wait_ld();
            if (hf == 1) {
              tc_fence_before();
              __syncwarp();
              if (lane == 0) mbar_arrive(&B.s_free[w]);
            }
#pragma unroll
            for (int k4 = 0; k4 < 8; ++k4) {                               // 4 query columns
              const int c0 = 32 * hf + 4 * k4;
              float4 l4 = *reinterpret_cast<const float4*>(lse + c0);
              const float4 d4 = *reinterpret_cast<const float4*>(dd + c0);
              if (MASK) {
                l4.x = c0 < qv ? l4.x - lpen : -INFINITY; l4.y = c0 + 1 < qv ? l4.y - lpen : -INFINITY;
                l4.z = c0 + 2 < qv ? l4.z - lpen : -INFINITY; l4.w = c0 + 3 < qv ? l4.w - lpen : -INFINITY;
              }
#pragma unroll
              for (int k = 0; k < 2; ++k) {
                const int cl = 4 * k4 + 2 * k;
                const f2 l2 = k == 0 ? mk2(l4.x, l4.y) : mk2(l4.z, l4.w), d2 = k == 0 ? mk2(d4.x, d4.y) : mk2(d4.z, d4.w);
                const f2 x = fma2(mk2u(sv[cl], sv[cl + 1]), SC2, l2);       // l2 = -LSE2
                float p0, p1;
                if (!MASK && ((2 * k4 + k) % PMOD) < PCNT) { const f2 e = ex2_poly2(x); p0 = lo2(e); p1 = hi2(e); }
                else { p0 = (PROBE & 4) ? lo2(x) : ex2f(lo2(x)); p1 = (PROBE & 4) ? hi2(x) : ex2f(hi2(x)); }
                const f2 g2 = mul2(mk2(p0, p1), sub2(mk2u(pv[cl], pv[cl + 1]), d2));
                pp[16 * hf + 2 * k4 + k] = pack_bf16(p0, p1);
                ds[16 * hf + 2 * k4 + k] = pack_bf16(lo2(g2), hi2(g2));
              }
            }
          }
        };
        if (PROBE & 128) {                                                // probe: no row math (S^T / dP^T released unread)
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.s_free[w]);
#pragma unroll
          for (int c = 0; c < 32; ++c) { pp[c] = 0u; ds[c] = 0u; }
        } else if (__all_sync(~0u, kok) && qv >= BQ) math(BoolC<false>{});
        else math(BoolC<true>{});
        if (r == 0 && w == 0) EVT(G, 3);
        if (!SPL && t == 0 && pend >= 0) { epilogue(pend, pli); pend = -1; }
        if (G >= 1) mbar_wait(&B.p_free, (G - 1) & 1);                    // the MMAs of G - 1 are done (P^T / dS^T read)
        if (r == 0 && w == 0) EVT(G, 4);
        tc_fence_after();
        if (!(PROBE & 256)) {
        tmem_st16(trow + T_P + w * 64, *reinterpret_cast<uint32_t(*)[16]>(pp));
        tmem_st16(trow + T_P + w * 64 + 16, *reinterpret_cast<uint32_t(*)[16]>(pp + 16));
#pragma unroll
        for (int c = 0; c < 8; ++c) sts128(sds + sw128(r, c), make_uint4(ds[4 * c], ds[4 * c + 1], ds[4 * c + 2], ds[4 * c + 3]));
        fence_proxy_async();
        }
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.p_full[w]);
        if (r == 0) EVT(G, w ? 8 : 5);
        if (G >= 1 && ((G - 1) & 1) == w) dq_out(G - 1);
        if (r == 0 && w == 0) EVT(G, 6);
      }
      pend = ai; pli = li;
      ++ai;
    }
    if (!CSB && pend >= 0) epilogue(pend, pli);
    if (G >= 1 && ((G - 1) & 1) == w) dq_out(G - 1);
    if (CSB) {
      // ---- the item's dK | dV: this rank's partial (zero without query blocks) reduce-scattered to the key rows' owners (rank o owns key
      // rows rs(o) .. rs(o + 1) - 1, rs(o) = ceil(256 o / CS)), summed there in rank order. After the last dq_out: no rank waits on the
      // exchange before its dQ adds are all staged.
      if (r == 0 && w == 0) EVT(60, 0);
      const It e = item_of(0);
      if (e.clact) {
        uint32_t kv[32], vv[32];
        if (e.act) {
          mbar_wait(&B.acc_full, 0);
          tc_fence_after();
          tmem_ld32(trow + T_ACC + w * 64, kv);
          tmem_ld32(trow + T_ACC + w * 64 + 32, vv);
          tmem_wait_ld();
          tc_fence_before();
        } else {
#pragma unroll
          for (int k = 0; k < 32; ++k) { kv[k] = 0u; vv[k] = 0u; }
        }
        const uint32_t RS = rcv_stride(csn), kk = (uint32_t)w * 128u + r, own = kk * csn / 256u, rr = kk - row_start(own, csn);
        const uint32_t rb = mapa_u32(smem_u32(&B.rcv), cb + own);
        const uint32_t dd = mapa_u32(su + O_RCV + (csr * 16u * RS + rr) * 16u, cb + own);   // chunk q at + q RS 16: a warp's rows = one contiguous run
#pragma unroll
        for (int q = 0; q < 8; ++q) st_async4(dd + (uint32_t)q * RS * 16u, kv[4 * q], kv[4 * q + 1], kv[4 * q + 2], kv[4 * q + 3], rb);
#pragma unroll
        for (int q = 0; q < 8; ++q) st_async4(dd + (uint32_t)(8 + q) * RS * 16u, vv[4 * q], vv[4 * q + 1], vv[4 * q + 2], vv[4 * q + 3], rb);
        if (r == 0 && w == 0) EVT(60, 1);
        mbar_wait_cl(&B.rcv, 0);                                           // every rank's rows of mine have landed
        if (r == 0 && w == 0) EVT(60, 2);
        // sum: (chunk pair cp, row) items, row fastest (conflict-free reads), 256 threads; two chunks (8 floats) -> one 16-B bf16 store
        const uint32_t r0 = row_start(csr, csn), nr = row_start(csr + 1u, csn) - r0;
        const size_t hb = (size_t)(e.n * H + e.h) * S + (size_t)e.kp * 2 * KM + r0;
        for (uint32_t ix = kk; ix < nr * 8u; ix += 256u) {
          const uint32_t cp = ix / nr, row = ix - cp * nr, q0 = 2u * cp;
          uint32_t o[4];
#pragma unroll
          for (uint32_t u = 0; u < 2; ++u) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (uint32_t s = 0; s < (CSN ? (uint32_t)CSN : csn); ++s) {   // rank order: deterministic
              const uint4 x = lds128(su + O_RCV + ((s * 16u + q0 + u) * RS + row) * 16u);
              a.x += __uint_as_float(x.x); a.y += __uint_as_float(x.y); a.z += __uint_as_float(x.z); a.w += __uint_as_float(x.w);
            }
            const float m = q0 < 8u ? scale : 1.f;
            o[2 * u] = pack_bf16(a.x * m, a.y * m);
            o[2 * u + 1] = pack_bf16(a.z * m, a.w * m);
          }
          uint16_t* dst = (q0 < 8u ? DK : DV) + (hb + row) * DH + (q0 & 7u) * 4u;   // q0 even: the pair never straddles dK | dV
          *reinterpret_cast<uint4*>(dst) = make_uint4(o[0], o[1], o[2], o[3]);
        }
      }
    }
    if (!CSB && r == 0) tma_store_wait0();
  }
  if (tid == 128) EVT(60, 3);
  tc_fence_before();
  __syncthreads();
  if (CSB) cluster_sync();                                                 // no rank exits while its peers may still write it
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
