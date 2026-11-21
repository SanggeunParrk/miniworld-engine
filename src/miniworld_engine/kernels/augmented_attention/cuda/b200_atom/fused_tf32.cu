// fused_tf32.cu — the fp32 (TF32) path's inference row stages fused around the attention (sm_100a, tcgen05 kind::tf32): two kernels in place
// of the eight launches (3 AdaLNs, the gate, 4 GEMMs) atom_gemm_tf32 / rows_tf32 take, with the same arithmetic.
//
//   atom_pre_tf32   x1 = LN(s) s1 + b1  (cross mode: xkv = LN(x1) mks + mkb; cross = 2, AlphaFold 3's parallel K / V norm: LN(s) mks + mkb)  ->  P = [x1 | x1 | xkv | xkv] . Wp^T  (+ bq on q),
//                   P [M, 512] fp32 with the q / k / v blocks rounded to tf32 (RNA: they only feed the attention's MMAs)
//   atom_post_tf32  gated = sigmoid(g) o;  a1 = s + so (gated Wo^T);  x2 = LN(a1) s2 + b2;  h = silu(x2 Wa^T) (x2 Wb^T);
//                   out = a1 + st (h Ws^T)                                                              out [M, 128] fp32
//
// Work units (persistent CTAs: unit blockIdx.x + i gridDim.x). The row-wise stages run on CUDA cores and write the MMA A operands straight
// into shared memory in the K-major 128-B-swizzled layout TMA would give (four K-slices of [128 rows][32 fp32] = 16 KB each, chunk q of row
// r at r * 128 + ((q ^ (r & 7)) << 4)), rounded to tf32 to nearest (cvt.rna; a kind::tf32 MMA truncates); the activations between the
// products never leave the SM. Weights (tf32-rounded host packs, K-major) stream through a 3-slot ring of 32 KB K-slices by TMA, in the order
// the MMAs take them. Work units: a CTA's unit is a serial latency chain (ncu, round 4: ~5 % issue utilisation), so the number of units
// against the 148 SMs decides the time. The first nfull 128-row tiles are full units, the remaining tiles are split in two half units each
// (a half unit costs ~0.7-0.8 of a full one, measured round 5): the host splits all tiles when the halves fit one wave (few tiles), and only
// the tail beyond the full waves when ITS halves fit one wave (4096 atoms: 148 full units + 24 halves instead of 160 tiles in two rounds).
//   pre   full: 128 rows x the 512 columns (Wp rows 0..255, then 256..511);  half: 128 rows x 256 columns (column half cb: Wp rows 256 cb ..;
//         A = x1, or xkv for the cross mode's k | v half)
//   post  full: 128 rows;  half: 64 rows (the MMAs still run M = 128, TMEM lanes / A rows 64..127 hold garbage nobody reads; the row phases
//         do half the rows)
// Row-wise phases come in two shapes: warp-per-row (coalesced global loads, a lane holds 4 channels; 4 rows per warp in flight, their
// LayerNorm reductions interleaved: the round-4 profile had a fifth of pre's samples in one-row-at-a-time 5-deep shuffle chains) and
// lane-per-row (TMEM lane = row: the accumulators leave TMEM 32 or 16 columns at a time and are parked in the operand tiles for the next
// warp-per-row phase).
// Warp roles (384 threads): warp 0 lane 0 TMA producer, warp 1 MMA issuer (whole warp, elect_one), warp 2 owns TMEM (512 columns), warp 3
// prefetches the CTA's first two units' input rows into L2 (keeping those pointers out of the row warps' registers), warps 4..11 the
// row-wise phases (lane quadrant = warp & 3, column half wg = (warp - 4) >> 2 in the lane-per-row phases).
// TMEM: pre  [0, 256) = x1 [Wq; Wg] (cross) or x1 [Wq; Wk] (a half unit: its column half), [256, 512) = xkv [Wk; Wv] or x1 [Wv; Wg];
//       post [0, 128) = y = gated Wo^T (kept to the end: a1 is recomputed from it), [128, 384) = [a | b] of one hidden half,
//            [384, 512) = t = h Ws^T (accumulated over the two halves).
// The sigmoid is rcp.approx(1 + 2^(-x log2 e)) everywhere in the fp32 path (gemm_tf32 / rows_tf32 too): the IEEE division it replaces was a
// quarter of post's warp samples.
// Budgets. smem: operand tiles A 64 KB | H 64 KB, weight ring 3 x 32 KB, barriers: 229632 B (one CTA per SM). Both launch 384 threads.
// Registers: pre __maxnreg__(128) (spill-free); post __maxnreg__(168): with one CTA per SM (smem) the register file allows 65536 / 384 =
// 170 per thread, and at 128 ptxas kept the post kernel's per-unit loop state (unit, first row, row count, unit count) on the stack.
// Programmatic dependent launch: griddepcontrol.wait after the smem / TMEM setup, before any global access.
// SPDX-License-Identifier: Apache-2.0
#include "../sm100/sm100.cuh"
using namespace s100;

constexpr int SLICE = 16384, TILE = 4 * SLICE, NST = 3, SLOT = 32768;
constexpr int O_A = 0, O_H = TILE, O_W = 2 * TILE, O_BAR = O_W + NST * SLOT, SMEM_BYTES = O_BAR + 256;
static_assert(O_H % 1024 == 0 && O_W % 1024 == 0 && SLOT % 1024 == 0, "1 KB alignment of the 128-B-swizzled tiles");
static_assert(SMEM_BYTES == 229632, "keep sm100_atom.KERNELS_FUSED32 in step");
constexpr int EPB = 4096;                                            // pre: per-warp TMA-store staging (2 x 4 KB, inside tile A)
#ifndef KQLN
#define KQLN 0     // B200 variant path (2026-11-17): RoseTTAFold3's affine LayerNorm of the flat q and k (pre; KQ = [wq | bq | wk | bk] fp32 [4][128])
#endif
#ifndef PARS
#define PARS 0     // ... the parallel schedule (post: x2 = LN(s) s2 + b2; RoseTTAFold3, Chai-1)
#endif
#ifndef UP
#define UP 0       // ... Boltz-2's up-gated SwiGLU (post: the hidden in four 64-wide quarters, [a | b | u] 192 accumulator columns each;
#endif             // mwu the pack [Wa_q; Wb_q; Wu_q] for q = 0..3, [768, 128], box 32 x 192)
#ifndef EARLY
#define EARLY (PARS && !UP)   // B200 variant path, round 3b (2026-11-17): post, the parallel schedule: the whole transition (x2 = LN(s) s2 + b2 -> [a | b] -> h -> t) reads only s and
#endif                        // the tables, complete when this kernel starts as the attention's dependent: it runs before the wait, under the
                              // attention; gated -> y = gated Wo^T (Wo last in the ring) and the output phase follow the wait
static_assert(!EARLY || (PARS && !UP), "EARLY: the parallel schedule's post build");
#ifndef PAIR
#define PAIR 0   // post (round 3 part 6): a CTA pair (cluster of 2) per unit: CTA rank r takes hidden half r ([Wa_r; Wb_r], Ws_r: 12 weight
#endif           // slices, not 20); both make y and x2; rank 1 hands its partial t_1 = h_1 Ws_1^T to rank 0 through DSMEM (into rank 0's tile H,
                 // once rank 0's MMAs are done: hfree), rank 0 adds t = t_0 + t_1 (always in this order) and writes the output
static_assert(!PAIR || (!EARLY && !PARS), "PAIR: the default / RMS / UP post builds");
#if PAIR
#define PCID ((int)blockIdx.x >> 1)   // post: the unit loop runs over clusters
#define PCN ((int)gridDim.x >> 1)
#else
#define PCID ((int)blockIdx.x)
#define PCN ((int)gridDim.x)
#endif
#if PAIR   // UP on a pair (round 3 part 7): CTA rank r takes the hidden quarters 2 r, 2 r + 1
#define NQ 2
#define QOF(qq) (2 * (int)rk + (qq))
#else
#define NQ 4
#define QOF(qq) (qq)
#endif

#ifndef TRACE
#define TRACE 0  // development only (-DTRACE=1, never a shipped build): %globaltimer stamps per CTA and phase in g_tr (read by _tools' probe)
#endif
#if TRACE
__device__ unsigned long long g_tr[296][32];
#define TRS(slot) do { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_tr[blockIdx.x][slot] = t_; } while (0)
#else
#define TRS(slot) do { } while (0)
#endif

constexpr int NSTMAX = 12;   // barrier slots (atom_post_tf32t's ring has 10)
struct Bars {
  uint64_t wfull[NSTMAX], wempty[NSTMAX], afull, hfull, accf, xfull, hfree;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 256, "barrier region");

#ifndef RMS
#define RMS 0   // ESMFold2 (-DRMS=1): every LayerNorm is an RMSNorm (no mean; the tables hold the identity-centred scales and raw gates)
#endif
DEVI float sigf(float x) { return rcpf(1.f + __expf(-x)); }          // the fp32 path's sigmoid (gemm_tf32 / rows_tf32 alike)
DEVI float rna(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
// x[k] (this lane's 4 channels of 4 rows) -> (x - mean) * rstd per row: rows_tf32's two-pass norm128 with the 4 rows' butterfly reductions
// interleaved (each row's additions in rows_tf32's order: lanes ^16, ^8, ^4, ^2, ^1)
DEVI void ln4x4(float (&x)[4][4], float eps) {
  float a[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) a[k] = x[k][0] + x[k][1] + x[k][2] + x[k][3];
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1)
#pragma unroll
    for (int k = 0; k < 4; ++k) a[k] += __shfl_xor_sync(0xffffffffu, a[k], o);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float mean = RMS ? 0.f : a[k] * (1.f / 128);
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { x[k][i] -= mean; v += x[k][i] * x[k][i]; }
    a[k] = v;
  }
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1)
#pragma unroll
    for (int k = 0; k < 4; ++k) a[k] += __shfl_xor_sync(0xffffffffu, a[k], o);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float rstd = 1.f / sqrtf(a[k] * (1.f / 128) + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[k][i] *= rstd;
  }
}
DEVI float4 ldg4(const float* p) { return __ldg(reinterpret_cast<const float4*>(p)); }
DEVI uint4 u4(float a, float b, float c, float d) { return make_uint4(__float_as_uint(a), __float_as_uint(b), __float_as_uint(c), __float_as_uint(d)); }
DEVI float4 lds4f(uint32_t a) { const uint4 v = lds128(a); return make_float4(__uint_as_float(v.x), __uint_as_float(v.y), __uint_as_float(v.z), __uint_as_float(v.w)); }
// the 16-byte group of columns 4 c4 .. 4 c4 + 3 of row r in a 128-row operand tile
DEVI uint32_t opnd(uint32_t base, int r, int c4) { return base + (uint32_t)(c4 >> 3) * SLICE + sw128((uint32_t)r, (uint32_t)(c4 & 7)); }
// lane-per-row: 32 / 16 TMEM columns of this warp's lanes
DEVI void tld(uint32_t taddr, float (&v)[32]) {
  uint32_t r[32];
  tmem_ld32(taddr, r);
  tmem_wait_ld();
#pragma unroll
  for (int j = 0; j < 32; ++j) v[j] = __uint_as_float(r[j]);
}
DEVI void tld16(uint32_t taddr, float (&v)[16]) {
  uint32_t r[16];
  tmem_ld16(taddr, r);
  tmem_wait_ld();
#pragma unroll
  for (int j = 0; j < 16; ++j) v[j] = __uint_as_float(r[j]);
}
// lane-per-row: row r's 32 values of K-slice j / its 16 values at columns 16 h16 .. of K-slice j -> operand tile
DEVI void put_row(uint32_t base, int r, int j, const float (&v)[32]) {
#pragma unroll
  for (int q = 0; q < 8; ++q) sts128(base + (uint32_t)j * SLICE + sw128((uint32_t)r, (uint32_t)q), u4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
}
DEVI void put_half(uint32_t base, int r, int j, int h16, const float (&v)[16]) {
#pragma unroll
  for (int q = 0; q < 4; ++q)
    sts128(base + (uint32_t)j * SLICE + sw128((uint32_t)r, (uint32_t)(4 * h16 + q)), u4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
}
// warp 3: L2 prefetch of rows [m0, m0 + rows) of a row-major fp32 [*, 128] operand (rows ld floats apart; 4 lines of 128 B per row)
DEVI void pf_rows(const float* t, int ld, int m0, int rows, int lane) {
  for (int l = lane; l < 4 * rows; l += 32) {
    const float* p = t + (size_t)(m0 + (l >> 2)) * ld + 32 * (l & 3);
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
  }
}
// one 32-row x 32-column chunk (v: this lane's row) -> staging buffer (nst & 1) of the warp -> TMA store at (col, row0) (gemm_tf32's)
DEVI void put_chunk(const CUtensorMap* map, uint32_t sbase, int& nst, int lane, int col, int row0, const float (&v)[32]) {
  const uint32_t buf = sbase + (uint32_t)(nst & 1) * EPB;
  if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
  __syncwarp();
#pragma unroll
  for (int q = 0; q < 8; ++q) sts128(buf + sw128((uint32_t)lane, (uint32_t)q), u4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
  fence_proxy_async();
  __syncwarp();
  if (lane == 0) {
    tma_store_2d(map, buf, col, row0);
    tma_store_commit();
  }
  ++nst;
}

// the common prologue: barriers, TMEM, PDL
// mode 2 (EARLY post): no wait here, the compute warps wait before the attention's outputs; mode 1: warp 3 (the L2 prefetcher) waits
// itself, after prefetching the rows that are complete before the chain's last launch (round 3 part 10)
// xn: xfull's arrival count (PAIR: 256 remote arrivals of the per-thread exchange; 1 for the bulk-copy exchange, round 3 part 14)
DEVI uint32_t setup(Bars& B, int warp, int mode = 0, uint32_t xn = 256) {
  if (threadIdx.x == 0) {
    for (int s = 0; s < NSTMAX; ++s) { mbar_init(&B.wfull[s], 1); mbar_init(&B.wempty[s], 1); }
    mbar_init(&B.afull, 8); mbar_init(&B.hfull, 8); mbar_init(&B.accf, 1); mbar_init(&B.xfull, xn); mbar_init(&B.hfree, 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
#if PAIR
  cluster_sync();                                                    // the peer's barriers are initialised before any remote arrive -- BEFORE
#endif                                                               // the wait (round 3 part 10: after it, the weight stream waited too)
  if (mode != 2 && warp != 0 && !(mode == 1 && warp == 3)) {         // warp 0 streams only the constant weight packs (round 3 part 8): it starts
    pdl_wait();                                                      // before the wait (the packs are made >= 2 launches earlier: complete by
    pdl_launch();                                                    // PDL's wait-then-launch chain); the others wait for the previous outputs
  }
  return B.tmem;
}
DEVI void teardown(uint32_t tmem, int warp) {
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
// MMA warp: D (+)= A (4 K-slices at abase) . the next 4 weight slots^T; fresh: the first product overwrites D
struct Mma {
  uint32_t su;
  int gi;
  DEVI void run(Bars& B, uint32_t d, uint32_t abase, uint32_t idesc, bool fresh, int ns = 4) {   // ns: K-slices (UP: 2 for a Ws quarter)
    for (int k = 0; k < ns; ++k, ++gi) {
      const int slot = gi % NST;
      mbar_wait(&B.wfull[slot], (gi / NST) & 1);
      tc_fence_after();
      const uint64_t da = desc_k128(abase + (uint32_t)k * SLICE), db = desc_k128(su + O_W + (uint32_t)slot * SLOT);
      if (elect_one()) {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)                               // K = 8 fp32 (32 B) per MMA
          umma_ss_tf32(d, da + (uint64_t)(2 * kk), db + (uint64_t)(2 * kk), idesc, (fresh && k == 0 && kk == 0) ? 0u : 1u);
        tc_commit(&B.wempty[slot]);
      }
      __syncwarp();
    }
  }
};
DEVI void commit_acc(Bars& B) {
  if (elect_one()) tc_commit(&B.accf);
  __syncwarp();
}
DEVI void wload(Bars& B, uint32_t su, int gi, const CUtensorMap* m, int c0, int c1, uint32_t bytes) {   // producer: weight slice gi
  const int slot = gi % NST;
  if ((int)(threadIdx.x & 31) != slot) return;                       // lane s of warp 0 issues ring slot s (round 3 part 8: one thread issues
  if (gi >= NST) mbar_wait(&B.wempty[slot], ((gi / NST) - 1) & 1);   // about one TMA load per ~600 clk, whatever the box)
  mbar_expect_tx(&B.wfull[slot], bytes);
  tma_load_2d(su + O_W + (uint32_t)slot * SLOT, m, &B.wfull[slot], c0, c1);
}
// post: unit u's (first row, row count), by value (by-reference outputs went through the stack: ptxas, round 7)
// nfull = -1 (round 3 part 5, 2026-11-17): every tile as four 32-row quarter units (few tiles: the quarters fit one wave)
DEVI int2 unit_rows(int u, int nfull) {
  return nfull < 0 ? make_int2(32 * u, 32) : u < nfull ? make_int2(128 * u, 128) : make_int2(128 * nfull + 64 * (u - nfull), 64);
}
DEVI int units_of(int units) {                                       // this CTA's share of the persistent unit loop
  return (int)blockIdx.x < units ? (units - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
}
DEVI int post_units(int M, int nfull) {                       // (PAIR: the cluster's share)
  const int u = nfull < 0 ? 4 * (M >> 7) : nfull + 2 * ((M >> 7) - nfull);
  return PCID < u ? (u - PCID + PCN - 1) / PCN : 0;
}
// PAIR: row r's 32 values of K-slice j -> the operand tile at the same offset in CTA `rank` of the cluster (DSMEM)
DEVI void put_row_peer(uint32_t base, int r, int j, const float (&v)[32], uint32_t rank) {
  uint32_t pb;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pb) : "r"(base), "r"(rank));
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const uint32_t a = pb + (uint32_t)j * SLICE + sw128((uint32_t)r, (uint32_t)q);
    asm volatile("st.shared::cluster.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(__float_as_uint(v[4 * q])), "r"(__float_as_uint(v[4 * q + 1])),
                 "r"(__float_as_uint(v[4 * q + 2])), "r"(__float_as_uint(v[4 * q + 3])) : "memory");
  }
}

// ==================================================================================================================== pre
// s [M, 128] (row stride 128), s1 / b1 rows mld apart, mks / mkb (cross) rows kld apart (the tables: mrows rows, row r read at r % mrows --
// mrows < M when the samples share the conditioning), bq [128]; mwp: Wp [512, 128] (box 32 x 256);
// mp: P [M, 512] (box 32 x 32). rndmask: the 128-column blocks of P stored tf32-rounded. nfull: see the header.
extern "C" __global__ void __maxnreg__(128)
atom_pre_tf32(const __grid_constant__ CUtensorMap mwp, const __grid_constant__ CUtensorMap mp, const float* __restrict__ s,
              const float* __restrict__ s1, const float* __restrict__ b1, int mld, const float* __restrict__ mks, const float* __restrict__ mkb,
              int kld, const float* __restrict__ bq, int cross, int rndmask, int M, float eps, int nfull, int mrows
#if KQLN
              , const float* __restrict__ KQ, float kq_eps
#endif
              ) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int units = nfull < 0 ? 4 * (M >> 7) : nfull + 2 * ((M >> 7) - nfull);
  const int my = units_of(units);
  const uint32_t sa = su + O_A, sh = su + O_H;
  if (threadIdx.x == 0) TRS(0);
  const uint32_t tmem = setup(B, warp, 1);
  if (threadIdx.x == 128) TRS(1);
  auto unit = [&](int i) { return (int)blockIdx.x + i * (int)gridDim.x; };
  // a unit's column half (a half unit), or -1 (a full unit: all 512 columns); its first row and row count. nfull = -1 (round 3 part 5,
  // 2026-11-17): every tile as four quarter units, 64 rows x one column half (unit 4 t + 2 rh + cb)
  auto half_of = [&](int u) { return nfull < 0 ? (u & 1) : u < nfull ? -1 : ((u - nfull) & 1); };
  auto row0_of = [&](int u) { return nfull < 0 ? 128 * (u >> 2) + 64 * ((u >> 1) & 1) : 128 * (u < nfull ? u : nfull + ((u - nfull) >> 1)); };
  const int rt = nfull < 0 ? 64 : 128;                              // a unit's rows

  if (warp == 0) {
    if (lane < NST) {                                               // (wload: lane s takes ring slot s)
      int gi = 0;
      for (int i = 0; i < my; ++i) {
        const int cb = half_of(unit(i));
        for (int l = (cb == 1 ? 4 : 0); l < (cb == 0 ? 4 : 8); ++l, ++gi) wload(B, su, gi, &mwp, 32 * (l & 3), 256 * (l >> 2), 32768u);
      }
    }
  } else if (warp == 1) {
    const uint32_t i256 = idesc_tf32(128, 256);
    Mma mm{su, 0};
    for (int i = 0; i < my; ++i) {
      const int cb = half_of(unit(i));
      mbar_wait(&B.afull, i & 1);
      tc_fence_after();
      mm.run(B, tmem, sa, i256, true);                               // x1 (or a half unit's A) against its first 256 Wp rows
      if (cb < 0) mm.run(B, tmem + 256, cross ? sh : sa, i256, true);  // a full unit: cols 256..511 from xkv (cross) or x1
      commit_acc(B);
    }
  } else if (warp == 3) {
    for (int i = 0; i < my && i < 2; ++i) pf_rows(s, 128, row0_of(unit(i)), rt, lane);   // s: the block input, complete before the chain
    pdl_wait();                                                      // (round 3 part 10: s before the wait, the tables after it)
    for (int i = 0; i < my && i < 2; ++i) {                          // the first two units' rows into L2
      const int m0 = row0_of(unit(i)), cb = half_of(unit(i)), t0 = m0 % mrows;
      pf_rows(s1, mld, t0, rt, lane); pf_rows(b1, mld, t0, rt, lane);
      if (cross && cb != 0) { pf_rows(mks, kld, t0, rt, lane); pf_rows(mkb, kld, t0, rt, lane); }
    }
  } else if (warp >= 4) {
    const int e = warp - 4, qd = warp & 3, wg = e >> 2;
    const uint32_t tl = tmem + ((uint32_t)(32 * qd) << 16);
    const uint32_t sbase = sa + (uint32_t)e * 2 * EPB;               // staging inside tile A (free once the MMAs have read it)
    int nst = 0;
    for (int i = 0; i < my; ++i) {
      const int cb = half_of(unit(i));
      const int m0 = row0_of(unit(i));
      const bool act = 32 * qd < rt;                                 // this warp's TMEM lane quadrant holds unit rows
      const size_t tsh = (size_t)(m0 - m0 % mrows);                   // table row = row - tsh (row % mrows: mrows a multiple of 128)
      const bool x1_to_a = cb != 1 || !cross;                       // tile A holds x1, or (cross, k | v half) xkv
      const bool want_kv = cross && cb != 0;
      const uint32_t kvt = cb < 0 ? sh : sa;
      // ---------------------------------------------------------------- warp per row: AdaLN 1 (and the K / V AdaLN) -> operand tiles
#pragma unroll 1
      for (int k0 = 0; k0 < (rt >> 3); k0 += 4) {                    // 4 rows per warp in flight
        float x[4][4];
        {
          float4 sc[4], sf[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const size_t row = (size_t)(m0 + e + 8 * (k0 + k));
            const float4 sv = ldg4(s + row * 128 + 4 * lane);
            x[k][0] = sv.x; x[k][1] = sv.y; x[k][2] = sv.z; x[k][3] = sv.w;
            sc[k] = ldg4(s1 + (row - tsh) * mld + 4 * lane);
            sf[k] = ldg4(b1 + (row - tsh) * mld + 4 * lane);
          }
          ln4x4(x, eps);
          if (want_kv && cross == 2) {                               // AlphaFold 3: xkv = LN(s) mks + mkb, from the same statistics
            float4 kc[4], kf[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const size_t row = (size_t)(m0 + e + 8 * (k0 + k));
              kc[k] = ldg4(mks + (row - tsh) * kld + 4 * lane);
              kf[k] = ldg4(mkb + (row - tsh) * kld + 4 * lane);
            }
#pragma unroll
            for (int k = 0; k < 4; ++k)
              sts128(opnd(kvt, e + 8 * (k0 + k), lane), u4(rna(fmaf(x[k][0], kc[k].x, kf[k].x)), rna(fmaf(x[k][1], kc[k].y, kf[k].y)),
                                                           rna(fmaf(x[k][2], kc[k].z, kf[k].z)), rna(fmaf(x[k][3], kc[k].w, kf[k].w))));
          }
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            x[k][0] = fmaf(x[k][0], sc[k].x, sf[k].x); x[k][1] = fmaf(x[k][1], sc[k].y, sf[k].y);
            x[k][2] = fmaf(x[k][2], sc[k].z, sf[k].z); x[k][3] = fmaf(x[k][3], sc[k].w, sf[k].w);
            if (x1_to_a) sts128(opnd(sa, e + 8 * (k0 + k), lane), u4(rna(x[k][0]), rna(x[k][1]), rna(x[k][2]), rna(x[k][3])));
          }
        }
        if (want_kv && cross != 2) {                                 // xkv = LN(x1) mks + mkb (x1 exact)
          float4 kc[4], kf[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const size_t row = (size_t)(m0 + e + 8 * (k0 + k));
            kc[k] = ldg4(mks + (row - tsh) * kld + 4 * lane);
            kf[k] = ldg4(mkb + (row - tsh) * kld + 4 * lane);
          }
          ln4x4(x, eps);
#pragma unroll
          for (int k = 0; k < 4; ++k)
            sts128(opnd(kvt, e + 8 * (k0 + k), lane), u4(rna(fmaf(x[k][0], kc[k].x, kf[k].x)), rna(fmaf(x[k][1], kc[k].y, kf[k].y)),
                                                         rna(fmaf(x[k][2], kc[k].z, kf[k].z)), rna(fmaf(x[k][3], kc[k].w, kf[k].w))));
        }
      }
      fence_proxy_async();                                           // generic-proxy writes -> the MMAs' (async-proxy) reads
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.afull);
      // ---------------------------------------------------------------- lane per row: P out of TMEM (bias on q, tf32 rounding on q / k / v)
      mbar_wait(&B.accf, i & 1);
      tc_fence_after();
      const int c0 = cb < 0 ? 0 : 8 * cb, nch = cb < 0 ? 16 : 8;   // P's 32-column chunks of this unit; TMEM column 32 (c - c0)
#if KQLN
      // whole 128-column blocks per warpgroup (block b to warpgroup b & 1): q (0) and k (1) get the flat LayerNorm of their row (lane = row:
      // two passes over the block's four TMEM chunks for the statistics, a third applies it), v / g leave as they are
      for (int b = c0 >> 2; b < (c0 + nch) >> 2; ++b) {
        if ((b & 1) != wg || !act) continue;
        float mean = 0.f, rstd = 1.f;
        const float* bqb = b == 0 ? bq : nullptr;
        if (b < 2) {
          float sum = 0.f;
          for (int c = 4 * b; c < 4 * b + 4; ++c) {
            float v[32];
            tld(tl + 32 * (c - c0), v);
#pragma unroll
            for (int j = 0; j < 32; ++j) sum += v[j] + (bqb ? __ldg(bqb + 32 * c + j) : 0.f);
          }
          mean = sum * (1.f / 128);
          float var = 0.f;
          for (int c = 4 * b; c < 4 * b + 4; ++c) {
            float v[32];
            tld(tl + 32 * (c - c0), v);
#pragma unroll
            for (int j = 0; j < 32; ++j) { const float d = v[j] + (bqb ? __ldg(bqb + 32 * c + j) : 0.f) - mean; var += d * d; }
          }
          rstd = 1.f / sqrtf(var * (1.f / 128) + kq_eps);
        }
        for (int c = 4 * b; c < 4 * b + 4; ++c) {
          float v[32];
          tld(tl + 32 * (c - c0), v);
          if (bqb) {
#pragma unroll
            for (int j = 0; j < 32; ++j) v[j] += __ldg(bqb + 32 * c + j);
          }
          if (b < 2) {
            const float* w = KQ + 256 * b + 32 * (c - 4 * b);
#pragma unroll
            for (int j = 0; j < 32; ++j) v[j] = fmaf((v[j] - mean) * rstd, __ldg(w + j), __ldg(w + 128 + j));
          }
          if ((rndmask >> b) & 1) {
#pragma unroll
            for (int j = 0; j < 32; ++j) v[j] = rna(v[j]);
          }
          put_chunk(&mp, sbase, nst, lane, 32 * c, m0 + 32 * qd, v);
        }
      }
      for (int cc = nch; cc < nch; cc += 2) {                        // (the per-chunk loop below: not taken)
#else
      for (int cc = wg; act && cc < nch; cc += 2) {
#endif
        const int c = c0 + cc, blk = c >> 2;
        float v[32];
        tld(tl + 32 * cc, v);
        if (blk == 0) {
          const float4* bp = reinterpret_cast<const float4*>(bq + 32 * c);
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const float4 b = __ldg(bp + q);
            v[4 * q] += b.x; v[4 * q + 1] += b.y; v[4 * q + 2] += b.z; v[4 * q + 3] += b.w;
          }
        }
        if ((rndmask >> blk) & 1) {
#pragma unroll
          for (int j = 0; j < 32; ++j) v[j] = rna(v[j]);
        }
        put_chunk(&mp, sbase, nst, lane, 32 * c, m0 + 32 * qd, v);
      }
      tc_fence_before();
      if (lane == 0) tma_store_wait_read0();                        // the stagings (tile A) are read before the next unit overwrites them
      __syncwarp();
      named_bar_sync(1, 256);
    }
    if (lane == 0) tma_store_wait0();
  }
  teardown(tmem, warp);
}

// ==================================================================================================================== post
// o [M, 128] (row stride 128), g rows gld apart, s [M, 128], so / s2 / b2 / st rows mld apart (mrows rows, read at r % mrows); mwo: Wo [128, 128] (box 32 x 128), mwu: the
// interleaved [Wa_0; Wb_0; Wa_1; Wb_1] [512, 128] (box 32 x 256), mws: Ws [128, 256] (box 32 x 128). out [M, 128]. nfull: see the header.
extern "C" __global__ void __maxnreg__(168)
atom_post_tf32(const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwu, const __grid_constant__ CUtensorMap mws,
               const float* __restrict__ o, const float* __restrict__ g, int gld, const float* __restrict__ s, const float* __restrict__ so,
               const float* __restrict__ s2, const float* __restrict__ b2, const float* __restrict__ st, int mld, float* __restrict__ out,
               int M, float eps, int nfull, int mrows) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#if PAIR
  const uint32_t rk = cluster_rank();                                // this CTA's hidden half
#endif
  // the CTA's unit count is recomputed at the top of each warp role (held across the role switch it lived on the stack: ptxas, round 7);
  // a unit's first row and row count (unit_rows): full units are 128-row tiles, then 64-row halves of the remaining tiles
  const uint32_t sa = su + O_A, sh = su + O_H;
  const uint32_t tmem = setup(B, warp, EARLY ? 2 : 1);               // (PAIR: the cluster barrier is inside setup, before the wait)

  if (warp == 0) {
    if (lane < NST) {                                               // (wload: lane s takes ring slot s)
      const int my = post_units(M, nfull);
      int gi = 0;
      for (int i = 0; i < my; ++i) {
#if EARLY
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mwu, 32 * k, 0, 32768u);           // [Wa_0; Wb_0]
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mws, 32 * k, 0, 16384u);           // Ws, hidden 0..127
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mwu, 32 * k, 256, 32768u);         // [Wa_1; Wb_1]
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mws, 128 + 32 * k, 0, 16384u);     // Ws, hidden 128..255
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mwo, 32 * k, 0, 16384u);           // Wo (after the attention)
        continue;
#endif
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mwo, 32 * k, 0, 16384u);           // Wo
#if UP
        for (int qq = 0; qq < NQ; ++qq) {                              // [Wa_q; Wb_q; Wu_q] (192 rows), then Ws of hidden 64 q .. (K 64)
          const int q = QOF(qq);                                     // (PAIR: this CTA's two quarters)
          for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mwu, 32 * k, 192 * q, 24576u);
          for (int k = 0; k < 2; ++k, ++gi) wload(B, su, gi, &mws, 64 * q + 32 * k, 0, 16384u);
        }
        continue;
#endif
#if PAIR
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mwu, 32 * k, 256 * (int)rk, 32768u);  // [Wa_r; Wb_r]
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mws, 128 * (int)rk + 32 * k, 0, 16384u);  // Ws, hidden 128 r ..
        continue;
#endif
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mwu, 32 * k, 0, 32768u);           // [Wa_0; Wb_0]
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mws, 32 * k, 0, 16384u);           // Ws, hidden 0..127
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mwu, 32 * k, 256, 32768u);         // [Wa_1; Wb_1]
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mws, 128 + 32 * k, 0, 16384u);     // Ws, hidden 128..255
      }
    }
  } else if (warp == 1) {
    const uint32_t i128 = idesc_tf32(128, 128), i256 = idesc_tf32(128, 256);
    Mma mm{su, 0};
    int na = 0, nh = 0;
    const int my = post_units(M, nfull);
    for (int i = 0; i < my; ++i) {
#if EARLY
      mbar_wait(&B.afull, (na++) & 1);                               // x2
      tc_fence_after();
      mm.run(B, tmem + 128, sa, i256, true);                         // [a | b] of hidden 0..127
      commit_acc(B);
      mbar_wait(&B.hfull, (nh++) & 1);                               // h_0
      tc_fence_after();
      mm.run(B, tmem + 384, sh, i128, true);                         // t = h_0 Ws_0^T
      mm.run(B, tmem + 128, sa, i256, true);                         // [a | b] of hidden 128..255
      commit_acc(B);
      mbar_wait(&B.hfull, (nh++) & 1);                               // h_1
      tc_fence_after();
      mm.run(B, tmem + 384, sh, i128, false);                        // t += h_1 Ws_1^T
      mbar_wait(&B.afull, (na++) & 1);                               // gated (after the attention)
      tc_fence_after();
      mm.run(B, tmem, sa, i128, true);                               // y = gated Wo^T
      commit_acc(B);                                                 // (covers the t MMAs too)
      continue;
#endif
      mbar_wait(&B.afull, (na++) & 1);                               // gated
      tc_fence_after();
      mm.run(B, tmem, sa, i128, true);                               // y = gated Wo^T
      commit_acc(B);
      mbar_wait(&B.afull, (na++) & 1);                               // x2
      tc_fence_after();
#if UP
      {
        const uint32_t i192 = idesc_tf32(128, 192);
        mm.run(B, tmem + 128, sa, i192, true);                       // [a | b | u] of hidden 0..63
        commit_acc(B);
        for (int q = 1; q <= NQ; ++q) {
          mbar_wait(&B.hfull, (nh++) & 1);                           // h_{q - 1} (tile H, K-slices 0..1)
          tc_fence_after();
          mm.run(B, tmem + 384, sh, i128, q == 1, 2);                // t (+)= h_{q - 1} Ws_{q - 1}^T
          if (q < NQ) mm.run(B, tmem + 128, sa, i192, true);          // [a | b | u] of hidden 64 q ..
          commit_acc(B);
        }
        (void)i256;
      }
      continue;
#endif
#if PAIR
      mm.run(B, tmem + 128, sa, i256, true);                         // [a | b] of hidden half r
      commit_acc(B);
      mbar_wait(&B.hfull, (nh++) & 1);                               // h_r
      tc_fence_after();
      mm.run(B, tmem + 384, sh, i128, true);                         // t_r = h_r Ws_r^T
      commit_acc(B);
      continue;
#endif
      mm.run(B, tmem + 128, sa, i256, true);                         // [a | b] of hidden 0..127
      commit_acc(B);
      mbar_wait(&B.hfull, (nh++) & 1);                               // h_0
      tc_fence_after();
      mm.run(B, tmem + 384, sh, i128, true);                         // t = h_0 Ws_0^T
      mm.run(B, tmem + 128, sa, i256, true);                         // [a | b] of hidden 128..255
      commit_acc(B);
      mbar_wait(&B.hfull, (nh++) & 1);                               // h_1
      tc_fence_after();
      mm.run(B, tmem + 384, sh, i128, false);                        // t += h_1 Ws_1^T
      commit_acc(B);
    }
  } else if (warp == 3) {
    const int my = post_units(M, nfull);
    for (int i = 0; i < my && i < 2; ++i) {                          // the first two units' rows into L2: g, s and the tables (complete before
      const int2 ur = unit_rows(PCID + i * PCN, nfull);              // the chain's last launch) before the wait, o after it (round 3 part 10)
      const int m0 = ur.x, rt = ur.y, t0 = m0 % mrows;
      pf_rows(g, gld, m0, rt, lane); pf_rows(s, 128, m0, rt, lane); pf_rows(so, mld, t0, rt, lane);
      pf_rows(s2, mld, t0, rt, lane); pf_rows(b2, mld, t0, rt, lane); pf_rows(st, mld, t0, rt, lane);
    }
#if !EARLY
    pdl_wait();
#endif
    for (int i = 0; i < my && i < 2; ++i) {
      const int2 ur = unit_rows(PCID + i * PCN, nfull);
      pf_rows(o, 128, ur.x, ur.y, lane);
    }
  } else if (warp >= 4) {
    const int e = warp - 4, qd = warp & 3, wg = e >> 2;
    const int r = 32 * qd + lane;                                    // lane-per-row phases: this thread's unit row
    const uint32_t tl = tmem + ((uint32_t)(32 * qd) << 16);
    int nc = 0, nx = 0;                                               // (nx: PAIR's hfree / xfull phases)
    (void)nx;
    const int my = post_units(M, nfull);
    for (int i = 0; i < my; ++i) {
      const int2 ur = unit_rows(PCID + i * PCN, nfull);
      const int m0 = ur.x, rt = ur.y;
      const size_t tsh = (size_t)(m0 - m0 % mrows);                   // table row = row - tsh (row % mrows: mrows a multiple of 128)
      const bool act = 32 * qd < rt;                                 // this warp's TMEM lane quadrant holds unit rows
      const int nr = rt >> 3;                                        // warp-per-row phases: rows per warp
#if EARLY
      // ---------------------------------------------------------------- warp per row: x2 = LN(s) s2 + b2 -> tile A (before the wait)
#pragma unroll 1
      for (int k0 = 0; k0 < nr; k0 += 4) {
        float x[4][4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float4 sv = ldg4(s + (size_t)(m0 + e + 8 * (k0 + k)) * 128 + 4 * lane);
          x[k][0] = sv.x; x[k][1] = sv.y; x[k][2] = sv.z; x[k][3] = sv.w;
        }
        ln4x4(x, eps);
#pragma unroll
        for (int h2 = 0; h2 < 4; h2 += 2) {
          float4 sc[2], sf[2];
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const size_t row = (size_t)(m0 + e + 8 * (k0 + h2 + k));
            sc[k] = ldg4(s2 + (row - tsh) * mld + 4 * lane);
            sf[k] = ldg4(b2 + (row - tsh) * mld + 4 * lane);
          }
#pragma unroll
          for (int k = 0; k < 2; ++k)
            sts128(opnd(sa, e + 8 * (k0 + h2 + k), lane),
                   u4(rna(fmaf(x[h2 + k][0], sc[k].x, sf[k].x)), rna(fmaf(x[h2 + k][1], sc[k].y, sf[k].y)),
                      rna(fmaf(x[h2 + k][2], sc[k].z, sf[k].z)), rna(fmaf(x[h2 + k][3], sc[k].w, sf[k].w))));
        }
      }
      tc_fence_before();
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.afull);
      // ---------------------------------------------------------------- lane per row: h = silu(a) b of each hidden half -> tile H
      for (int hb = 0; hb < 2; ++hb) {
        mbar_wait(&B.accf, (nc++) & 1);                              // hb = 1: the Ws_0 MMAs have read h_0 too
        tc_fence_after();
        if (act) {
          for (int q16 = 4 * wg; q16 < 4 * wg + 4; ++q16) {
            float va[16], vb[16];
            tld16(tl + 128 + 16 * q16, va);
            tld16(tl + 256 + 16 * q16, vb);
#pragma unroll
            for (int j = 0; j < 16; ++j) va[j] = rna(va[j] * sigf(va[j]) * vb[j]);
            put_half(sh, r, q16 >> 1, q16 & 1, va);
          }
        }
        tc_fence_before();
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.hfull);
      }
      // ---------------------------------------------------------------- the attention's outputs: wait, then gated = sigmoid(g) o -> tile A
      // (tile A is free: the [a | b] MMAs of both halves are done)
      pdl_wait();
      pdl_launch();
#pragma unroll 1
      for (int k0 = 0; k0 < nr; k0 += 4) {
        float4 ov[4], gv[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const size_t row = (size_t)(m0 + e + 8 * (k0 + k));
          ov[k] = ldg4(o + row * 128 + 4 * lane);
          gv[k] = ldg4(g + row * gld + 4 * lane);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k)
          sts128(opnd(sa, e + 8 * (k0 + k), lane), u4(rna(ov[k].x * sigf(gv[k].x)), rna(ov[k].y * sigf(gv[k].y)), rna(ov[k].z * sigf(gv[k].z)),
                                                      rna(ov[k].w * sigf(gv[k].w))));
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.afull);
#else
      // ---------------------------------------------------------------- warp per row: gated = sigmoid(g) o -> tile A
#pragma unroll 1
      for (int k0 = 0; k0 < nr; k0 += 4) {                           // 4 rows per warp in flight
        float4 ov[4], gv[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const size_t row = (size_t)(m0 + e + 8 * (k0 + k));
          ov[k] = ldg4(o + row * 128 + 4 * lane);
          gv[k] = ldg4(g + row * gld + 4 * lane);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k)
          sts128(opnd(sa, e + 8 * (k0 + k), lane), u4(rna(ov[k].x * sigf(gv[k].x)), rna(ov[k].y * sigf(gv[k].y)), rna(ov[k].z * sigf(gv[k].z)),
                                                      rna(ov[k].w * sigf(gv[k].w))));
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.afull);
      // ---------------------------------------------------------------- lane per row: y -> tile A (the Wo MMAs have read gated)
      mbar_wait(&B.accf, (nc++) & 1);
      tc_fence_after();
      if (act) {
        for (int c = 2 * wg; c < 2 * wg + 2; ++c) {
          float v[32];
          tld(tl + 32 * c, v);
          put_row(sa, r, c, v);
        }
      }
      named_bar_sync(1, 256);
      // ---------------------------------------------------------------- warp per row: a1 = s + so y, x2 = LN(a1) s2 + b2 -> tile A
#pragma unroll 1
      for (int k0 = 0; k0 < nr; k0 += 4) {                           // 4 rows per warp in flight, two load rounds (s, so; then s2, b2)
        float x[4][4];
        {
          float4 sv[4], gv[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const size_t row = (size_t)(m0 + e + 8 * (k0 + k));
            sv[k] = ldg4(s + row * 128 + 4 * lane);
            gv[k] = ldg4(so + (row - tsh) * mld + 4 * lane);
          }
#pragma unroll
#if PARS                                                             // the parallel schedule: the transition normalises s
          for (int k = 0; k < 4; ++k) { x[k][0] = sv[k].x; x[k][1] = sv[k].y; x[k][2] = sv[k].z; x[k][3] = sv[k].w; }
          (void)gv;
        }
        ln4x4(x, eps);
#else
          for (int k = 0; k < 4; ++k) {
            const float4 y = lds4f(opnd(sa, e + 8 * (k0 + k), lane));
            x[k][0] = fmaf(gv[k].x, y.x, sv[k].x); x[k][1] = fmaf(gv[k].y, y.y, sv[k].y);
            x[k][2] = fmaf(gv[k].z, y.z, sv[k].z); x[k][3] = fmaf(gv[k].w, y.w, sv[k].w);
          }
        }
        ln4x4(x, eps);
#endif
#pragma unroll
        for (int h2 = 0; h2 < 4; h2 += 2) {                          // s2, b2 two rows at a time (the phase's register peak: 16 + 16 values)
          float4 sc[2], sf[2];
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const size_t row = (size_t)(m0 + e + 8 * (k0 + h2 + k));
            sc[k] = ldg4(s2 + (row - tsh) * mld + 4 * lane);
            sf[k] = ldg4(b2 + (row - tsh) * mld + 4 * lane);
          }
#pragma unroll
          for (int k = 0; k < 2; ++k)
            sts128(opnd(sa, e + 8 * (k0 + h2 + k), lane),
                   u4(rna(fmaf(x[h2 + k][0], sc[k].x, sf[k].x)), rna(fmaf(x[h2 + k][1], sc[k].y, sf[k].y)),
                      rna(fmaf(x[h2 + k][2], sc[k].z, sf[k].z)), rna(fmaf(x[h2 + k][3], sc[k].w, sf[k].w))));
        }
      }
      tc_fence_before();
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.afull);
      // ---------------------------------------------------------------- lane per row: h = silu(a) b of each hidden half -> tile H
#if UP
      for (int q = 0; q < NQ; ++q) {                                 // h_q = silu(a) b u of hidden 64 q .. -> tile H K-slices 0..1
        mbar_wait(&B.accf, (nc++) & 1);                              // q > 0: the Ws_{q - 1} MMAs have read h_{q - 1} too
        tc_fence_after();
        if (act) {
          for (int q16 = 2 * wg; q16 < 2 * wg + 2; ++q16) {           // 16 columns at a time
            float va[16], vb[16], vu[16];
            tld16(tl + 128 + 16 * q16, va);
            tld16(tl + 192 + 16 * q16, vb);
            tld16(tl + 256 + 16 * q16, vu);
#pragma unroll
            for (int j = 0; j < 16; ++j) va[j] = rna(va[j] * sigf(va[j]) * vb[j] * vu[j]);
            put_half(sh, r, q16 >> 1, q16 & 1, va);
          }
        }
        tc_fence_before();
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.hfull);
      }
      for (int hb = 2; hb < 2; ++hb) {
#else
      for (int hb = 0; hb < (PAIR ? 1 : 2); ++hb) {                   // (PAIR: the CTA's one hidden half)
#endif
        mbar_wait(&B.accf, (nc++) & 1);                              // hb = 1: the Ws_0 MMAs have read h_0 too
        tc_fence_after();
        if (act) {
          for (int q16 = 4 * wg; q16 < 4 * wg + 4; ++q16) {           // 16 columns at a time
            float va[16], vb[16];
            tld16(tl + 128 + 16 * q16, va);
            tld16(tl + 256 + 16 * q16, vb);
#pragma unroll
            for (int j = 0; j < 16; ++j) va[j] = rna(va[j] * sigf(va[j]) * vb[j]);
            put_half(sh, r, q16 >> 1, q16 & 1, va);
          }
        }
        tc_fence_before();
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.hfull);
      }
#endif  // EARLY
      // ---------------------------------------------------------------- lane per row: y -> tile A, t -> tile H (every MMA of the unit is done)
      mbar_wait(&B.accf, (nc++) & 1);
      tc_fence_after();
#if PAIR
      if (rk == 1) {                                                 // t_1 -> rank 0's tile H (once its MMAs are done); this unit is done
        mbar_wait_cl(&B.hfree, (nx++) & 1);
        if (act)
          for (int c = 2 * wg; c < 2 * wg + 2; ++c) {
            float v[32];
            tld(tl + 384 + 32 * c, v);
            put_row_peer(sh, r, c, v, 0);
          }
        tc_fence_before();
        mbar_arrive_remote(&B.xfull, 0);                             // (release, cluster scope: this thread's stores)
        continue;
      }
      if (warp == 4 && lane == 0) mbar_arrive_remote(&B.hfree, 1);   // every MMA of the unit is done: tile H is free for t_1
      if (act)
        for (int c = 2 * wg; c < 2 * wg + 2; ++c) {
          float v[32];
          tld(tl + 32 * c, v);
          put_row(sa, r, c, v);
        }
      mbar_wait_cl(&B.xfull, (nx++) & 1);                            // t_1 has landed in tile H
      if (act)
        for (int c = 2 * wg; c < 2 * wg + 2; ++c) {                  // t = t_0 + t_1 (in this order)
          float v[32];
          tld(tl + 384 + 32 * c, v);
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const float4 h = lds4f(sh + (uint32_t)c * SLICE + sw128((uint32_t)r, (uint32_t)q));
            v[4 * q] += h.x; v[4 * q + 1] += h.y; v[4 * q + 2] += h.z; v[4 * q + 3] += h.w;
          }
          put_row(sh, r, c, v);
        }
#else
      if (act) {
        for (int c = 2 * wg; c < 2 * wg + 2; ++c) {
          float v[32];
          tld(tl + 32 * c, v);
          put_row(sa, r, c, v);
          tld(tl + 384 + 32 * c, v);
          put_row(sh, r, c, v);
        }
      }
#endif
      tc_fence_before();
      named_bar_sync(1, 256);
      // ---------------------------------------------------------------- warp per row: out = (s + so y) + st t
#pragma unroll 1
      for (int k0 = 0; k0 < nr; k0 += 4) {                           // 4 rows per warp in flight, two load rounds (s, so; then st)
        float4 a1[4];
        {
          float4 sv[4], gv[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const size_t row = (size_t)(m0 + e + 8 * (k0 + k));
            sv[k] = ldg4(s + row * 128 + 4 * lane);
            gv[k] = ldg4(so + (row - tsh) * mld + 4 * lane);
          }
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float4 y = lds4f(opnd(sa, e + 8 * (k0 + k), lane));
            a1[k] = make_float4(fmaf(gv[k].x, y.x, sv[k].x), fmaf(gv[k].y, y.y, sv[k].y), fmaf(gv[k].z, y.z, sv[k].z), fmaf(gv[k].w, y.w, sv[k].w));
          }
        }
        float4 tv[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) tv[k] = ldg4(st + ((size_t)(m0 + e + 8 * (k0 + k)) - tsh) * mld + 4 * lane);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int rr = e + 8 * (k0 + k);
          const float4 t = lds4f(opnd(sh, rr, lane));
          stg128(out + (size_t)(m0 + rr) * 128 + 4 * lane, u4(fmaf(tv[k].x, t.x, a1[k].x), fmaf(tv[k].y, t.y, a1[k].y), fmaf(tv[k].z, t.z, a1[k].z),
                                                             fmaf(tv[k].w, t.w, a1[k].w)));
        }
      }
      named_bar_sync(1, 256);                                        // tiles A / H read before the next unit writes them
    }
  }
#if PAIR
  cluster_sync();                                                    // no CTA leaves while its peer may still signal it
#endif
  teardown(tmem, warp);
}

// ==================================================================================================================== post, transposed
// atom_post_tf32t (round 3 part 10, 2026-11-18): atom_post_tf32's arithmetic for quarter units (32 rows; few tiles), with every product
// taken TRANSPOSED: D^T [128 features, 32 rows] = W [128 x K] . X^T, the weight slice the A operand (M = 128) and the unit's 32 activation
// rows the B operand (N = 32). The row-major M = 128 form ran 128-row MMAs for 32 real rows: the trace had the MMA chains (Wo, [a | b],
// Ws) at 0.8 / 1.2-1.6 / 0.8 us of a ~12.5 us post-attention chain -- the TF32 tensor rate (~3900 FLOP / clk / SM), not the weights.
// N = 32 does the real rows only (a quarter of the work), and the accumulators come out lane = feature: all 8 compute warps take part in
// the TMEM phases (16 rows x one feature a thread) where the row form had 2 warps (lane = row, 32 rows). The activations between the
// products go back into 32-row K-major tiles (K-slices 4 KB apart) with column writes (a warp writes 32 features of one row per store: no
// bank conflict). Same products, same summation order per output as atom_post_tf32 (PAIR: t = t_0 + t_1 in this order): bitwise equal.
// Every row the kernel reads except o is loaded before the grid-dependency wait (g from pre, s the block input, the tables).
// The hidden runs in blocks of 128 (PAIR: block rk); UP (Boltz-2): a / b / u of a block, each granule two 64-row boxes of the quarter pack
// [Wa_q; Wb_q; Wu_q] (quarters 2 hb, 2 hb + 1), h = silu(a) b u; EARLY (the parallel schedule, RoseTTAFold3 / Chai-1): x2 = LN(s) s2 + b2
// and the whole transition before the wait, gated -> y after it (Wo last in the ring).
// smem: X 16 KB (gated, x2, then t / t_0) | Y 16 KB (y) | H 32 KB (h: K = hidden; PAIR: h_r in the first 16 KB, rank 1's t_1 in the
// second) | EARLY: X2 16 KB (x2) | the weight ring of 16 KB granules ([128 rows x 32 K], one A operand each).
// TMEM: y^T 0..31, the parts (a, b[, u]) of hidden block hl at 32 + 32 (NP hl + p), t^T after them.
// -DTU64=1 (round 3 part 11, 2026-11-18): the same kernels on 64-row units (N = 64), for the tiles whose halves fit one wave but whose
// quarters do not (38..74 tiles: 1024 atoms x 5 samples, 5120 - 9472 rows; 80 CTAs at 40 tiles where the 32-row form needs 160). The
// tiles grow (X / Y 32 KB, H 64 KB; EARLY: x2 shares X -- every MMA that reads x2 is done before gated is written), the ring has 6 granules,
// the TMEM columns are 64 a part, a thread of the TMEM phases takes 32 rows (two 16-column halves), and the row phases run 8 rows a warp in
// two rounds of 4 (only g held from before the wait; a1 recomputed from Y for the output -- the same fmaf).
#ifndef TU64
#define TU64 0
#endif
static_assert(!(TU64 && PAIR), "TU64: single-CTA builds");
constexpr int TU = TU64 ? 64 : 32;
constexpr uint32_t TSL = TU * 128;
constexpr int GRT = 16384, OT_X = 0, OT_Y = TU * 512, OT_H = TU * 1024, OT_X2 = TU64 ? 0 : 65536,
              OT_W = TU64 ? 131072 : EARLY ? 81920 : 65536;
constexpr int NGT = (O_BAR - OT_W) / GRT;
static_assert(NGT >= (TU64 ? 6 : 9) && NGT <= NSTMAX, "atom_post_tf32t: the weight ring");
#if PAIR
constexpr int NHB = 1;                                                // hidden blocks of 128 a CTA (PAIR: block rk)
#else
constexpr int NHB = 2;
#endif
constexpr int NP = UP ? 3 : 2;                                        // parts of a hidden block: a, b (, u)
constexpr uint32_t TC_Y = 0, TC_A = TU, TC_T = TU + TU * NP * NHB;
static_assert(TC_T + TU <= 512, "TMEM");
DEVI uint32_t opT(uint32_t base, int r, int c4) { return base + (uint32_t)(c4 >> 3) * TSL + sw128((uint32_t)r, (uint32_t)(c4 & 7)); }
// feature f (K index) of rows r0 .. r0 + 15 -> a 32-row K-major tile
DEVI void put_col16(uint32_t base, int f, int r0, const float (&v)[16]) {
  const uint32_t a0 = base + (uint32_t)(f >> 5) * TSL + (uint32_t)(f & 3) * 4u, ch = (uint32_t)((f & 31) >> 2);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint32_t r = (uint32_t)(r0 + j);
    sts32(a0 + r * 128u + ((ch ^ (r & 7u)) << 4), __float_as_uint(v[j]));
  }
}
#if PAIR
DEVI void put_col16_peer(uint32_t base, int f, int r0, const float (&v)[16], uint32_t rank) {
  uint32_t pb;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pb) : "r"(base), "r"(rank));
  const uint32_t a0 = pb + (uint32_t)(f >> 5) * TSL + (uint32_t)(f & 3) * 4u, ch = (uint32_t)((f & 31) >> 2);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint32_t r = (uint32_t)(r0 + j);
    asm volatile("st.shared::cluster.b32 [%0], %1;" :: "r"(a0 + r * 128u + ((ch ^ (r & 7u)) << 4)), "r"(__float_as_uint(v[j])) : "memory");
  }
}
#endif
#ifndef XBULK
#define XBULK 0   // (round 3 part 14) 1: PAIR rank 1 writes t_1 into its own smem, then ONE bulk copy moves it to rank 0 (bitwise equal, level:
#endif            // the exchange phase 0.65-0.72 us either way, latency not stores; the F16 PAIR build always copies); 0: per-thread DSMEM stores
// bytes at src (this CTA) -> the same-layout buffer at local offset dst in CTA `rank` of the cluster, completing on that CTA's barrier bar
DEVI void bulk_to_peer(uint32_t dst, uint32_t src, uint32_t bytes, uint64_t* bar, uint32_t rank) {
  uint32_t d, b;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(dst), "r"(rank));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b) : "r"(smem_u32(bar)), "r"(rank));
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(d), "r"(src), "r"(bytes), "r"(b)
               : "memory");
}
// one ring granule: a [128 x 32] box at (c0, c1), or (c1b >= 0) two [64 x 32] boxes at rows c1 and c1b (UP's quarter-pack halves)
DEVI void wloadT(Bars& B, uint32_t wb, int gi, const CUtensorMap* m, int c0, int c1, int c1b = -1) {   // lane s of warp 0 issues slot s
  const int slot = gi % NGT;
  if ((int)(threadIdx.x & 31) != slot) return;
  if (gi >= NGT) mbar_wait(&B.wempty[slot], ((gi / NGT) - 1) & 1);
  mbar_expect_tx(&B.wfull[slot], (uint32_t)GRT);
  const uint32_t dst = wb + (uint32_t)slot * GRT;
  tma_load_2d(dst, m, &B.wfull[slot], c0, c1);
  if (c1b >= 0) tma_load_2d(dst + GRT / 2, m, &B.wfull[slot], c0, c1b);
}
// the parts of hidden block hb, K-slice by K-slice (the MMAs' order)
DEVI void wload_hid(Bars& B, uint32_t wb, int& gi, const CUtensorMap* mwu, int hb) {
  for (int k = 0; k < 4; ++k)
    for (int p = 0; p < NP; ++p) {
#if UP
      wloadT(B, wb, gi++, mwu, 32 * k, 192 * (2 * hb) + 64 * p, 192 * (2 * hb + 1) + 64 * p);
#else
      wloadT(B, wb, gi++, mwu, 32 * k, 256 * hb + 128 * p);
#endif
    }
}
struct MmaT {
  uint32_t wb;
  int gi;
  DEVI void one(Bars& B, uint32_t d, uint32_t bact, bool acc) {      // D^T (+)= the next weight granule [128 x 32 K] . X^T (32 rows, K-slice at bact)
    constexpr uint32_t IT = idesc_tf32(128, TU);
    const int slot = gi % NGT;
    mbar_wait(&B.wfull[slot], (gi / NGT) & 1);
    tc_fence_after();
    const uint64_t da = desc_k128(wb + (uint32_t)slot * GRT), db = desc_k128(bact);
    if (elect_one()) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) umma_ss_tf32(d, da + (uint64_t)(2 * kk), db + (uint64_t)(2 * kk), IT, (acc || kk) ? 1u : 0u);
      tc_commit(&B.wempty[slot]);
    }
    __syncwarp();
    ++gi;
  }
  DEVI void hid(Bars& B, uint32_t tmem, uint32_t xa, int hl) {      // the parts of hidden block hl from x2 (K-slices of xa)
    for (int k = 0; k < 4; ++k)
      for (int p = 0; p < NP; ++p) one(B, tmem + TC_A + TU * (NP * hl + p), xa + (uint32_t)k * TSL, k > 0);
  }
};

// arguments as atom_post_tf32 (nfull unused: always quarter units); mwu with box 32 x 128 (UP: 32 x 64)
extern "C" __global__ void __maxnreg__(168)
atom_post_tf32t(const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwu, const __grid_constant__ CUtensorMap mws,
                const float* __restrict__ o, const float* __restrict__ g, int gld, const float* __restrict__ s, const float* __restrict__ so,
                const float* __restrict__ s2, const float* __restrict__ b2, const float* __restrict__ st, int mld, float* __restrict__ out,
                int M, float eps, int nfull, int mrows) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  (void)nfull;
#if PAIR
  const uint32_t rk = cluster_rank();                                // this CTA's hidden block
  const int hb0 = (int)rk;
#else
  const int hb0 = 0;
#endif
  const uint32_t xa = su + OT_X, yt = su + OT_Y, ht = su + OT_H, wb = su + OT_W;
  const uint32_t x2t = EARLY ? su + OT_X2 : xa;                      // x2 (EARLY: its own tile, made before gated)
  if (threadIdx.x == 0) TRS(0);
  const uint32_t tmem = setup(B, warp, 2, (PAIR && XBULK) ? 1u : 256u);   // (no wait here: warp 3 and the row warps wait themselves)
  if (threadIdx.x == 128) TRS(1);
  const int my = TU64 ? units_of(M / TU) : post_units(M, -1);

  if (warp == 0) {
    if (lane == 0) TRS(14);
    if (lane < NGT) {
      int gi = 0;
      for (int i = 0; i < my; ++i) {
        if (!EARLY) for (int k = 0; k < 4; ++k) wloadT(B, wb, gi++, &mwo, 32 * k, 0);           // Wo
        for (int hb = hb0; hb < hb0 + NHB; ++hb) wload_hid(B, wb, gi, &mwu, hb);                 // the parts of hidden block hb
        for (int hb = hb0; hb < hb0 + NHB; ++hb)
          for (int k = 0; k < 4; ++k) wloadT(B, wb, gi++, &mws, 128 * hb + 32 * k, 0);        // Ws of hidden block hb
        if (EARLY) for (int k = 0; k < 4; ++k) wloadT(B, wb, gi++, &mwo, 32 * k, 0);            // (EARLY: Wo last, after the attention)
      }
    }
  } else if (warp == 1) {
    MmaT mm{wb, 0};
    int na = 0, nh = 0;
    for (int i = 0; i < my; ++i) {
      if (!EARLY) {
        mbar_wait(&B.afull, (na++) & 1);                             // gated
        tc_fence_after();
        if (lane == 0) TRS(10);
        for (int k = 0; k < 4; ++k) mm.one(B, tmem + TC_Y, xa + (uint32_t)k * TSL, k > 0);   // y^T = Wo gated^T
        commit_acc(B);
        if (lane == 0) TRS(11);
      }
      mbar_wait(&B.afull, (na++) & 1);                               // x2
      tc_fence_after();
      if (lane == 0) TRS(12);
      for (int hl = 0; hl < NHB; ++hl) {
        mm.hid(B, tmem, x2t, hl);
        commit_acc(B);
      }
      if (lane == 0) TRS(13);
      for (int hl = 0; hl < NHB; ++hl) {
        mbar_wait(&B.hfull, (nh++) & 1);                             // h of hidden block hl
        tc_fence_after();
        for (int k = 0; k < 4; ++k) mm.one(B, tmem + TC_T, ht + (uint32_t)(4 * hl + k) * TSL, hl > 0 || k > 0);   // t^T (+)= Ws h^T
      }
      if (EARLY) {
        mbar_wait(&B.afull, (na++) & 1);                             // gated (after the attention)
        tc_fence_after();
        for (int k = 0; k < 4; ++k) mm.one(B, tmem + TC_Y, xa + (uint32_t)k * TSL, k > 0);
      }
      commit_acc(B);                                                 // t (EARLY: and y)
      if (lane == 0) TRS(16);
    }
  } else if (warp == 3) {
    for (int i = 0; i < my && i < 2; ++i) {                          // g, s and the tables before the wait, o after it
      const int m0 = TU * (PCID + i * PCN), t0 = m0 % mrows;
      pf_rows(g, gld, m0, TU, lane); pf_rows(s, 128, m0, TU, lane); pf_rows(so, mld, t0, TU, lane);
      pf_rows(s2, mld, t0, TU, lane); pf_rows(b2, mld, t0, TU, lane); pf_rows(st, mld, t0, TU, lane);
    }
    pdl_wait();
    for (int i = 0; i < my && i < 2; ++i) pf_rows(o, 128, TU * (PCID + i * PCN), TU, lane);
#if TU64
  } else if (warp >= 4) {                                            // 64-row units: 8 rows a warp (two rounds of 4), 32 rows a TMEM thread
    const int e = warp - 4, qd = warp & 3, wg = e >> 2, f = 32 * qd + lane, r32 = 32 * wg;
    const uint32_t tl = tmem + ((uint32_t)(32 * qd) << 16) + (uint32_t)r32;
    int nc = 0;
    for (int i = 0; i < my; ++i) {
      const int m0 = TU * (PCID + i * PCN);
      const size_t tsh = (size_t)(m0 - m0 % mrows);
      float4 gv[8];                                                  // g (from pre, complete before the chain's last launch): before the wait
#pragma unroll
      for (int k = 0; k < 8; ++k) gv[k] = ldg4(g + (size_t)(m0 + e + 8 * k) * gld + 4 * lane);
      auto x2_phase = [&](bool from_s) {                            // x2 = LN(.) s2 + b2 -> X: of s (EARLY) or of a1 = s + so y (y in Y)
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 4) {
          float x[4][4];
          float4 sc4[4], sf4[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const size_t row = (size_t)(m0 + e + 8 * (k0 + k));
            const float4 sv = ldg4(s + row * 128 + 4 * lane);
            sc4[k] = ldg4(s2 + (row - tsh) * mld + 4 * lane);
            sf4[k] = ldg4(b2 + (row - tsh) * mld + 4 * lane);
            if (from_s) {
              x[k][0] = sv.x; x[k][1] = sv.y; x[k][2] = sv.z; x[k][3] = sv.w;
            } else {
              const float4 so4 = ldg4(so + (row - tsh) * mld + 4 * lane);
              const float4 y = lds4f(opT(yt, e + 8 * (k0 + k), lane));
              x[k][0] = fmaf(so4.x, y.x, sv.x); x[k][1] = fmaf(so4.y, y.y, sv.y); x[k][2] = fmaf(so4.z, y.z, sv.z); x[k][3] = fmaf(so4.w, y.w, sv.w);
            }
          }
          ln4x4(x, eps);
          if (warp == 4 && lane == 0 && k0 == 0) TRS(17);
#pragma unroll
          for (int k = 0; k < 4; ++k)
            sts128(opT(xa, e + 8 * (k0 + k), lane), u4(rna(fmaf(x[k][0], sc4[k].x, sf4[k].x)), rna(fmaf(x[k][1], sc4[k].y, sf4[k].y)),
                                                       rna(fmaf(x[k][2], sc4[k].z, sf4[k].z)), rna(fmaf(x[k][3], sc4[k].w, sf4[k].w))));
        }
        tc_fence_before();
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.afull);
      };
      auto gated_phase = [&]() {                                     // (unit 0: the wait first) o -> gated = sigmoid(g) o -> X
        if (i == 0) {
          pdl_wait();
          pdl_launch();
        }
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 4) {
          float4 ov[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) ov[k] = ldg4(o + (size_t)(m0 + e + 8 * (k0 + k)) * 128 + 4 * lane);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float4 gg = gv[k0 + k];
            sts128(opT(xa, e + 8 * (k0 + k), lane), u4(rna(ov[k].x * sigf(gg.x)), rna(ov[k].y * sigf(gg.y)), rna(ov[k].z * sigf(gg.z)),
                                                       rna(ov[k].w * sigf(gg.w))));
          }
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.afull);
      };
      auto y_out = [&]() {                                           // y^T -> Y (feature per lane, 32 rows in two halves)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          float v[16];
          tld16(tl + TC_Y + 16 * hh, v);
          put_col16(yt, f, r32 + 16 * hh, v);
        }
      };
      if (EARLY) {
        x2_phase(true);
      } else {
        gated_phase();
        if (warp == 4 && lane == 0) TRS(2);
        mbar_wait(&B.accf, (nc++) & 1);
        tc_fence_after();
        if (warp == 4 && lane == 0) TRS(3);
        y_out();
        named_bar_sync(1, 256);
        if (warp == 4 && lane == 0) TRS(15);
        x2_phase(false);
        if (warp == 4 && lane == 0) TRS(4);
      }
      // ---------------------------------------------------------------- feature per lane: h = silu(a) b (u) of each hidden block -> H
      for (int hl = 0; hl < NHB; ++hl) {
        mbar_wait(&B.accf, (nc++) & 1);
        tc_fence_after();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          float va[16], vb[16];
          tld16(tl + TC_A + TU * (NP * hl) + 16 * hh, va);
          tld16(tl + TC_A + TU * (NP * hl + 1) + 16 * hh, vb);
#if UP
          float vu[16];
          tld16(tl + TC_A + TU * (NP * hl + 2) + 16 * hh, vu);
#pragma unroll
          for (int j = 0; j < 16; ++j) va[j] = rna(va[j] * sigf(va[j]) * vb[j] * vu[j]);
#else
#pragma unroll
          for (int j = 0; j < 16; ++j) va[j] = rna(va[j] * sigf(va[j]) * vb[j]);
#endif
          put_col16(ht, 128 * hl + f, r32 + 16 * hh, va);
        }
        tc_fence_before();
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.hfull);
      }
      if (EARLY) gated_phase();                                      // (X held x2: every MMA that reads it is done -- the last h's accf)
      // ---------------------------------------------------------------- feature per lane: t^T -> X
      mbar_wait(&B.accf, (nc++) & 1);
      tc_fence_after();
      if (EARLY) y_out();
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float v[16];
        tld16(tl + TC_T + 16 * hh, v);
        put_col16(xa, f, r32 + 16 * hh, v);
      }
      tc_fence_before();
      named_bar_sync(1, 256);
      // ---------------------------------------------------------------- warp per row: out = (s + so y) + st t
#pragma unroll
      for (int k0 = 0; k0 < 8; k0 += 4) {
        float4 sv[4], so4[4], tv[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const size_t row = (size_t)(m0 + e + 8 * (k0 + k));
          sv[k] = ldg4(s + row * 128 + 4 * lane);
          so4[k] = ldg4(so + (row - tsh) * mld + 4 * lane);
          tv[k] = ldg4(st + (row - tsh) * mld + 4 * lane);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int rr = e + 8 * (k0 + k);
          const float4 y = lds4f(opT(yt, rr, lane)), t = lds4f(opT(xa, rr, lane));
          const float4 a1 = make_float4(fmaf(so4[k].x, y.x, sv[k].x), fmaf(so4[k].y, y.y, sv[k].y), fmaf(so4[k].z, y.z, sv[k].z),
                                        fmaf(so4[k].w, y.w, sv[k].w));
          stg128(out + (size_t)(m0 + rr) * 128 + 4 * lane, u4(fmaf(tv[k].x, t.x, a1.x), fmaf(tv[k].y, t.y, a1.y), fmaf(tv[k].z, t.z, a1.z),
                                                             fmaf(tv[k].w, t.w, a1.w)));
        }
      }
      named_bar_sync(1, 256);                                        // X / Y / H read before the next unit writes them
    }
  }
#else
  } else if (warp >= 4) {
    const int e = warp - 4, qd = warp & 3, wg = e >> 2, f = 32 * qd + lane, r16 = 16 * wg;
    const uint32_t tl = tmem + ((uint32_t)(32 * qd) << 16) + (uint32_t)r16;   // this thread's feature (lane), its 16 rows (columns)
    int nc = 0;
#if PAIR
    int nx = 0;
#endif
    for (int i = 0; i < my; ++i) {
      const int m0 = TU * (PCID + i * PCN);
      const size_t tsh = (size_t)(m0 - m0 % mrows);                   // table row = row - tsh
      // every row this warp reads except o, into registers (unit 0: before the grid-dependency wait -- g from pre, s the block input and
      // the tables are complete before the chain's last launch); a1 is kept from the x2 phase for the output phase
      float4 gv[4], sv[4], so4[4], sc4[4], sf4[4], st4v[4], a1[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const size_t row = (size_t)(m0 + e + 8 * k);
        gv[k] = ldg4(g + row * gld + 4 * lane);
        sv[k] = ldg4(s + row * 128 + 4 * lane);
        so4[k] = ldg4(so + (row - tsh) * mld + 4 * lane);
        sc4[k] = ldg4(s2 + (row - tsh) * mld + 4 * lane);
        sf4[k] = ldg4(b2 + (row - tsh) * mld + 4 * lane);
        st4v[k] = ldg4(st + (row - tsh) * mld + 4 * lane);
      }
      auto x2_phase = [&](bool from_s) {                            // x2 = LN(.) s2 + b2 -> x2t: of s (EARLY) or of a1 = s + so y (y in Y)
        float x[4][4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          if (from_s) {
            x[k][0] = sv[k].x; x[k][1] = sv[k].y; x[k][2] = sv[k].z; x[k][3] = sv[k].w;
          } else {
            const float4 y = lds4f(opT(yt, e + 8 * k, lane));
            a1[k] = make_float4(fmaf(so4[k].x, y.x, sv[k].x), fmaf(so4[k].y, y.y, sv[k].y), fmaf(so4[k].z, y.z, sv[k].z), fmaf(so4[k].w, y.w, sv[k].w));
            x[k][0] = a1[k].x; x[k][1] = a1[k].y; x[k][2] = a1[k].z; x[k][3] = a1[k].w;
          }
        }
        ln4x4(x, eps);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          sts128(opT(x2t, e + 8 * k, lane), u4(rna(fmaf(x[k][0], sc4[k].x, sf4[k].x)), rna(fmaf(x[k][1], sc4[k].y, sf4[k].y)),
                                               rna(fmaf(x[k][2], sc4[k].z, sf4[k].z)), rna(fmaf(x[k][3], sc4[k].w, sf4[k].w))));
        tc_fence_before();
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.afull);
      };
      auto gated_phase = [&]() {                                     // (unit 0: the wait first) o -> gated = sigmoid(g) o -> X
        if (i == 0) {
          pdl_wait();
          pdl_launch();
        }
        float4 ov[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) ov[k] = ldg4(o + (size_t)(m0 + e + 8 * k) * 128 + 4 * lane);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          sts128(opT(xa, e + 8 * k, lane), u4(rna(ov[k].x * sigf(gv[k].x)), rna(ov[k].y * sigf(gv[k].y)), rna(ov[k].z * sigf(gv[k].z)),
                                              rna(ov[k].w * sigf(gv[k].w))));
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.afull);
      };
      auto y_out = [&]() {                                           // y^T -> Y (feature per lane)
        float v[16];
        tld16(tl + TC_Y, v);
        put_col16(yt, f, r16, v);
      };
      if (EARLY) {
        x2_phase(true);
      } else {
        gated_phase();
        if (warp == 4 && lane == 0) TRS(2);
        mbar_wait(&B.accf, (nc++) & 1);
        tc_fence_after();
        if (warp == 4 && lane == 0) TRS(3);
        y_out();
        named_bar_sync(1, 256);
        x2_phase(false);
      }
      if (warp == 4 && lane == 0) TRS(4);
      // ---------------------------------------------------------------- feature per lane: h = silu(a) b (u) of each hidden block -> H
      for (int hl = 0; hl < NHB; ++hl) {
        mbar_wait(&B.accf, (nc++) & 1);
        tc_fence_after();
        if (warp == 4 && lane == 0) TRS(5 + hl);
        float va[16], vb[16];
        tld16(tl + TC_A + 32 * (NP * hl), va);
        tld16(tl + TC_A + 32 * (NP * hl + 1), vb);
#if UP
        float vu[16];
        tld16(tl + TC_A + 32 * (NP * hl + 2), vu);
#pragma unroll
        for (int j = 0; j < 16; ++j) va[j] = rna(va[j] * sigf(va[j]) * vb[j] * vu[j]);
#else
#pragma unroll
        for (int j = 0; j < 16; ++j) va[j] = rna(va[j] * sigf(va[j]) * vb[j]);
#endif
        put_col16(ht, 128 * hl + f, r16, va);
        tc_fence_before();
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.hfull);
      }
      if (EARLY) gated_phase();
      if (warp == 4 && lane == 0) TRS(7);
      // ---------------------------------------------------------------- feature per lane: t^T -> X (PAIR rank 1: t_1 -> rank 0's H, second half)
      mbar_wait(&B.accf, (nc++) & 1);
      tc_fence_after();
      if (warp == 4 && lane == 0) TRS(8);
      if (EARLY) y_out();                                            // (EARLY: y with t; the Wo MMAs read gated from X: done)
      {
        float v[16];
        tld16(tl + TC_T, v);
#if PAIR
        if (rk == 1) {
          if (i > 0) mbar_wait_cl(&B.hfree, (i - 1) & 1);           // rank 0 has read the previous unit's t_1
#if XBULK
          put_col16(ht + 16384u, f, r16, v);                         // t_1 -> this CTA's H second half (unused here), then one 16 KB bulk copy
          tc_fence_before();                                         // into rank 0's H second half, completing on its xfull (count 1, armed by
          fence_proxy_async();                                       // rank 0 with the bytes): one instruction where the 256 threads issued 16
          named_bar_sync(1, 256);                                    // DSMEM stores each (the copy's source is read before rank 0's xfull
          if (threadIdx.x == 128) bulk_to_peer(ht + 16384u, ht + 16384u, 16384u, &B.xfull, 0);   // completes, i.e. before its hfree)
#else
          put_col16_peer(ht + 16384u, f, r16, v, 0);
          tc_fence_before();
          mbar_arrive_remote(&B.xfull, 0);                           // (release, cluster scope: this thread's stores)
#endif
          continue;
        }
#if XBULK
        if (threadIdx.x == 128) mbar_expect_tx(&B.xfull, 16384u);   // rank 0 arms its xfull for the copy (arrival + bytes)
#endif
#endif
        put_col16(xa, f, r16, v);                                    // (X: every MMA that reads X is done -- the t MMAs completed after them)
      }
      tc_fence_before();
      named_bar_sync(1, 256);
#if PAIR
      mbar_wait_cl(&B.xfull, (nx++) & 1);                            // t_1 has landed in H's second half
#endif
      if (warp == 4 && lane == 0) TRS(9);
      // ---------------------------------------------------------------- warp per row: out = (s + so y) + st t
      {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int rr = e + 8 * k;
          if (EARLY) {                                               // (a1 = s + so y: y only now)
            const float4 y = lds4f(opT(yt, rr, lane));
            a1[k] = make_float4(fmaf(so4[k].x, y.x, sv[k].x), fmaf(so4[k].y, y.y, sv[k].y), fmaf(so4[k].z, y.z, sv[k].z), fmaf(so4[k].w, y.w, sv[k].w));
          }
          float4 t = lds4f(opT(xa, rr, lane));
#if PAIR
          const float4 t1 = lds4f(opT(ht + 16384u, rr, lane));     // t = t_0 + t_1 (in this order)
          t.x += t1.x; t.y += t1.y; t.z += t1.z; t.w += t1.w;
#endif
          stg128(out + (size_t)(m0 + rr) * 128 + 4 * lane, u4(fmaf(st4v[k].x, t.x, a1[k].x), fmaf(st4v[k].y, t.y, a1[k].y),
                                                             fmaf(st4v[k].z, t.z, a1[k].z), fmaf(st4v[k].w, t.w, a1[k].w)));
        }
      }
      named_bar_sync(1, 256);                                        // X / Y / H read before the next unit writes them
#if PAIR
      if (warp == 4 && lane == 0) mbar_arrive_remote(&B.hfree, 1);   // H's second half is free for the next t_1
#endif
      if (warp == 4 && lane == 0) TRS(18);
    }
  }
#endif  // TU64
#if PAIR
  cluster_sync();                                                    // no CTA leaves while its peer may still signal it
#endif
  teardown(tmem, warp);
  if (threadIdx.x == 0) TRS(19);
}

#if !UP && !PARS && !PAIR
// ==================================================================================================================== pre, transposed
// atom_pre_tf32t (round 3 part 10): atom_pre_tf32's arithmetic for 32-row units over all 512 columns (few tiles: 4 units a tile, one a CTA),
// the products transposed as in atom_post_tf32t: P^T [128 columns of block b, 32 rows] = Wp_b . X^T (block b's 128 Wp rows the A operand,
// in 16 KB granules [128 x 32 K]; the unit's x1 / xkv rows the N = 32 B operand). The row form ran 64-row units through M = 128 MMAs of
// N = 256 (the trace: ~1.1 us of MMA, two row-phase rounds of 4 rows a warp, 4 of 8 warps in the epilogue); here a warp normalises 4 rows in
// one round with all its loads in flight, the MMAs do the real rows only and every warp writes P (one 32 x 32 chunk per accumulator half).
// smem: X1 16 KB | XKV 16 KB (cross) | the ring: 10 granules from 32 KB on (the 16 Wp granules stream through it in block order) | the
// epilogue's stagings (4 KB a warp; blocks 0 | 1 leave while the MMAs of blocks 2 | 3 still stream through the ring).
// TMEM: P^T block b at columns 32 b .. 32 b + 31 (lane = column of the block).
// (TU64: 64-row units -- X1 / XKV 32 KB, 8 ring granules, P^T block b at TMEM columns 64 b .., the epilogue in two 32-row chunks, the row
// phase in two rounds of 4 rows a warp)
constexpr int OP_X = 0, OP_K = TU * 512, OP_W = TU * 1024;
constexpr int NGP = TU64 ? 8 : 10, OP_ST = OP_W + NGP * GRT;
static_assert(OP_ST + 8 * EPB <= O_BAR && NGP <= NSTMAX, "atom_pre_tf32t: smem");
// 32 rows x 32 lanes -> the sum over the lanes of row `lane` (a transposing butterfly: 31 shuffles for 32 sums)
DEVI float tsum32(const float (&x)[32], int lane) {
  float v[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) v[j] = x[j];
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    const bool up = (lane & off) != 0;
#pragma unroll
    for (int j = 0; j < off; ++j) {
      const float send = up ? v[j] : v[j + off], keep = up ? v[j + off] : v[j];
      v[j] = keep + __shfl_xor_sync(0xffffffffu, send, off);
    }
  }
  return v[0];
}
DEVI void wloadP(Bars& B, uint32_t wb, int gi, const CUtensorMap* m, int c0, int c1) {
  const int slot = gi % NGP;
  if ((int)(threadIdx.x & 31) != slot) return;
  if (gi >= NGP) mbar_wait(&B.wempty[slot], ((gi / NGP) - 1) & 1);
  mbar_expect_tx(&B.wfull[slot], (uint32_t)GRT);
  tma_load_2d(wb + (uint32_t)slot * GRT, m, &B.wfull[slot], c0, c1);
}

// arguments as atom_pre_tf32 (nfull unused); mwp with box 32 x 128
extern "C" __global__ void __maxnreg__(168)
atom_pre_tf32t(const __grid_constant__ CUtensorMap mwp, const __grid_constant__ CUtensorMap mp, const float* __restrict__ s,
               const float* __restrict__ s1, const float* __restrict__ b1, int mld, const float* __restrict__ mks, const float* __restrict__ mkb,
               int kld, const float* __restrict__ bq, int cross, int rndmask, int M, float eps, int nfull, int mrows
#if KQLN
               , const float* __restrict__ KQ, float kq_eps
#endif
               ) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  (void)nfull;
  const int units = M / TU;
  const int my = units_of(units);
  const uint32_t x1t = su + OP_X, xkt = su + OP_K, wb = su + OP_W;
  if (threadIdx.x == 0) TRS(0);
  const uint32_t tmem = setup(B, warp, 2);                           // (no wait here: warp 3 and the row warps wait themselves)
  if (threadIdx.x == 128) TRS(1);
  auto row0 = [&](int i) { return TU * ((int)blockIdx.x + i * (int)gridDim.x); };

  if (warp == 0) {
    if (lane < NGP) {
      int gi = 0;
      for (int i = 0; i < my; ++i)
        for (int b = 0; b < 4; ++b)
          for (int k = 0; k < 4; ++k) wloadP(B, wb, gi++, &mwp, 32 * k, 128 * b);   // Wp block b, K-slice k
    }
  } else if (warp == 1) {
    constexpr uint32_t IT = idesc_tf32(128, TU);
    int gi = 0;
    for (int i = 0; i < my; ++i) {
      mbar_wait(&B.afull, i & 1);
      tc_fence_after();
      if (lane == 0) TRS(10);
      for (int b = 0; b < 4; ++b) {
        const uint32_t xb = (cross && b >= 2) ? xkt : x1t;            // cross: k | v from xkv
        for (int k = 0; k < 4; ++k, ++gi) {
          const int slot = gi % NGP;
          mbar_wait(&B.wfull[slot], (gi / NGP) & 1);
          tc_fence_after();
          const uint64_t da = desc_k128(wb + (uint32_t)slot * GRT), db = desc_k128(xb + (uint32_t)k * TSL);
          if (elect_one()) {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) umma_ss_tf32(tmem + TU * b, da + (uint64_t)(2 * kk), db + (uint64_t)(2 * kk), IT, (k | kk) ? 1u : 0u);
            tc_commit(&B.wempty[slot]);
          }
          __syncwarp();
        }
        if (b & 1) commit_acc(B);                                    // blocks 0 | 1, then 2 | 3
      }
      if (lane == 0) TRS(11);
    }
  } else if (warp == 3) {
    for (int i = 0; i < my && i < 2; ++i) pf_rows(s, 128, row0(i), TU, lane);   // s: complete before the chain
    pdl_wait();
    for (int i = 0; i < my && i < 2; ++i) {
      const int t0 = row0(i) % mrows;
      pf_rows(s1, mld, t0, TU, lane); pf_rows(b1, mld, t0, TU, lane);
      if (cross) { pf_rows(mks, kld, t0, TU, lane); pf_rows(mkb, kld, t0, TU, lane); }
    }
  } else if (warp >= 4) {
    const int e = warp - 4, qd = warp & 3, wg = e >> 2, f = 32 * qd + lane;
    const uint32_t tl = tmem + ((uint32_t)(32 * qd) << 16);
    const uint32_t buf = su + OP_ST + (uint32_t)e * EPB;             // this warp's staging
    for (int i = 0; i < my; ++i) {
      const int m0 = row0(i);
      const size_t tsh = (size_t)(m0 - m0 % mrows);
      // ---------------------------------------------------------------- warp per row (4 rows a warp, every load in flight): AdaLN 1 (+ K / V)
#if TU64
#pragma unroll 1
#else
#pragma unroll
#endif
      for (int rg = 0; rg < TU / 32; ++rg) {                         // (TU64: two rounds of 4 rows, rows e + 8 (4 rg + k))
        float x[4][4];
        float4 sc[4], sf[4], kc[4], kf[4], sv[4];
        const int mr = m0 + 32 * rg, ro = 32 * rg;
#pragma unroll
        for (int k = 0; k < 4; ++k) sv[k] = ldg4(s + (size_t)(mr + e + 8 * k) * 128 + 4 * lane);   // s: the block input (complete)
        if (i == 0 && rg == 0) {                                     // the tables come from the previous kernel
          pdl_wait();
          pdl_launch();
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const size_t row = (size_t)(mr + e + 8 * k);
          x[k][0] = sv[k].x; x[k][1] = sv[k].y; x[k][2] = sv[k].z; x[k][3] = sv[k].w;
          sc[k] = ldg4(s1 + (row - tsh) * mld + 4 * lane);
          sf[k] = ldg4(b1 + (row - tsh) * mld + 4 * lane);
          if (cross) {
            kc[k] = ldg4(mks + (row - tsh) * kld + 4 * lane);
            kf[k] = ldg4(mkb + (row - tsh) * kld + 4 * lane);
          }
        }
        ln4x4(x, eps);
        if (cross == 2) {                                            // AlphaFold 3: xkv = LN(s) mks + mkb, from the same statistics
#pragma unroll
          for (int k = 0; k < 4; ++k)
            sts128(opT(xkt, ro + e + 8 * k, lane), u4(rna(fmaf(x[k][0], kc[k].x, kf[k].x)), rna(fmaf(x[k][1], kc[k].y, kf[k].y)),
                                                 rna(fmaf(x[k][2], kc[k].z, kf[k].z)), rna(fmaf(x[k][3], kc[k].w, kf[k].w))));
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          x[k][0] = fmaf(x[k][0], sc[k].x, sf[k].x); x[k][1] = fmaf(x[k][1], sc[k].y, sf[k].y);
          x[k][2] = fmaf(x[k][2], sc[k].z, sf[k].z); x[k][3] = fmaf(x[k][3], sc[k].w, sf[k].w);
          sts128(opT(x1t, ro + e + 8 * k, lane), u4(rna(x[k][0]), rna(x[k][1]), rna(x[k][2]), rna(x[k][3])));
        }
        if (cross == 1) {                                            // xkv = LN(x1) mks + mkb (x1 exact)
          ln4x4(x, eps);
#pragma unroll
          for (int k = 0; k < 4; ++k)
            sts128(opT(xkt, ro + e + 8 * k, lane), u4(rna(fmaf(x[k][0], kc[k].x, kf[k].x)), rna(fmaf(x[k][1], kc[k].y, kf[k].y)),
                                                 rna(fmaf(x[k][2], kc[k].z, kf[k].z)), rna(fmaf(x[k][3], kc[k].w, kf[k].w))));
        }
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.afull);
      if (warp == 4 && lane == 0) TRS(2);
      // ---------------------------------------------------------------- feature per lane: P^T blocks (2 h + wg) -> P (bias on q, tf32 rounding)
      for (int h = 0; h < 2; ++h) {
        mbar_wait(&B.accf, (2 * i + h) & 1);
        tc_fence_after();
        if (warp == 4 && lane == 0) TRS(3 + h);
        const int b = 2 * h + wg;
#if TU64
#pragma unroll 1
#else
#pragma unroll
#endif
        for (int c = 0; c < TU / 32; ++c) {                           // (TU64: the block's two 32-row chunks)
        float v[32];
        tld(tl + (uint32_t)(TU * b + 32 * c), v);
        float bias = b == 0 ? __ldg(bq + f) : 0.f;
#if KQLN
        if (b < 2) {                                                 // RoseTTAFold3: the flat LayerNorm of q (+ bq) / k over the block's 128
          const uint32_t rb = su + OP_K + (uint32_t)wg * 1024u;      // columns, per row: 4 warps (this warpgroup) hold them; XKV is free
#pragma unroll                                                       // (not cross) -- partial sums [2][4 warps][32 rows] there
          for (int j = 0; j < 32; ++j) v[j] += bias;
          bias = 0.f;
          sts32(rb + (uint32_t)(qd * 32 + lane) * 4u, __float_as_uint(tsum32(v, lane)));
          named_bar_sync(2 + wg, 128);
          float mean = 0.f;
#pragma unroll
          for (int q = 0; q < 4; ++q) mean += __uint_as_float(lds32(rb + (uint32_t)(q * 32 + lane) * 4u));
          mean *= 1.f / 128;                                         // (lane l: row l's mean)
          float d[32];
#pragma unroll
          for (int j = 0; j < 32; ++j) { const float dj = v[j] - __shfl_sync(0xffffffffu, mean, j); d[j] = dj * dj; v[j] = dj; }
          sts32(rb + 512u + (uint32_t)(qd * 32 + lane) * 4u, __float_as_uint(tsum32(d, lane)));
          named_bar_sync(2 + wg, 128);
          float var = 0.f;
#pragma unroll
          for (int q = 0; q < 4; ++q) var += __uint_as_float(lds32(rb + 512u + (uint32_t)(q * 32 + lane) * 4u));
          const float rstd = 1.f / sqrtf(var * (1.f / 128) + kq_eps);
          const float kw = __ldg(KQ + 256 * b + f), kb = __ldg(KQ + 256 * b + 128 + f);
#pragma unroll
          for (int j = 0; j < 32; ++j) v[j] = fmaf(v[j] * __shfl_sync(0xffffffffu, rstd, j), kw, kb);
        }
#endif
        const bool rnd = (rndmask >> b) & 1;
        if (lane == 0) tma_store_wait_read0();                      // the previous chunk's store has read the staging
        __syncwarp();
        const uint32_t ch = (uint32_t)(lane >> 2), cb = (uint32_t)(lane & 3) * 4u;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          float w = v[j] + bias;
          if (rnd) w = rna(w);
          sts32(buf + (uint32_t)j * 128u + ((ch ^ ((uint32_t)j & 7u)) << 4) + cb, __float_as_uint(w));
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) {
          tma_store_2d(&mp, buf, 128 * b + 32 * qd, m0 + 32 * c);
          tma_store_commit();
        }
        }
      }
      tc_fence_before();
      if (lane == 0) tma_store_wait_read0();                        // the stagings are read before the next unit (if any) rewrites them
      __syncwarp();
      named_bar_sync(1, 256);
      if (warp == 4 && lane == 0) TRS(5);
    }
    if (lane == 0) tma_store_wait0();
  }
  teardown(tmem, warp);
  if (threadIdx.x == 0) TRS(19);
}
#endif  // !UP && !PARS && !PAIR

// ==================================================================================================================== fp16-exact transposed units
// -DF16=1 (round 3 part 14, 2026-11-18): atom_post_f16t / atom_pre_f16t -- atom_post_tf32t / atom_pre_tf32t with kind::f16 products on fp16
// operands that hold the tf32 operands EXACTLY: a tf32 value has 11 significant bits, as an fp16 one has, so it IS an fp16 value whenever
// it lies in fp16's normal range [2^-14, 65504]. Every operand is scaled by a power of two first:
//   weights (host, b200_atom._f16pack): per pack block, the largest magnitude into [2^14, 2^15); the host checks that every element then is an
//     fp16 value bit for bit and runs the tf32 kernels otherwise (a block spanning more than ~2^29 in magnitude);
//   activations (here, per row): gated, x2, x1 / xkv: the row's largest magnitude (of the tf32-rounded values) into [2^14, 2^15) (warp butterfly
//     max in the warp-per-row phases); h (feature per lane: its rows' maxima are not at hand) from a bound, |h| <= (A1 X + 0.2785) B1 X
//     [U1 X] times 2 (X the row's max |x2|, A1 / B1 / U1 the largest L1 norm of a Wa / Wb / Wu row, host; |silu(a)| <= max(a, 0.2785)).
//   The accumulators are multiplied back by the exact inverse powers 2^-(e_w + e_row) (pow2 times pow2: exact in fp32's normal range).
//   Range: no overflow (|v| 2^e < 2^15 <= 65504); an element below 2^-29 of its row's maximum (h: of the bound) lands in fp16's subnormals and
//   may lose low bits (absolute error <= 2^-25 2^-e: below 2^-39 of the row's maximum, far under the tf32 rounding of that maximum).
// Half the MMA instructions (K = 16 a kind::f16 MMA, 8 a kind::tf32 one), half the weight bytes (granules [128 x 64 K] fp16 of 16 KB), half
// the operand tiles; the products sum the same exact partial products in the tensor core's own K grouping (16 against 8).
// Row factors live in smem (post: gated, x2 -> a / b, h scale, t; pre: x1, xkv), published to the TMEM-phase threads by a named barrier.
#ifndef F16
#define F16 0
#endif
#ifndef PRE1R
#define PRE1R 0
#endif
#ifndef POST1R
#define POST1R 0
#endif
#if F16
constexpr uint32_t IH = (1u << 4) | ((uint32_t)(TU >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);   // kind::f16: fp16 A / B, fp32 D, K-major
constexpr int HTX = TU * 256, HTF = TU * 512;                         // an fp16 operand tile [TU rows x 128] / an fp32 tile [TU x 128]
DEVI void sts64(uint32_t a, uint32_t x, uint32_t y) { asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(a), "r"(x), "r"(y) : "memory"); }
DEVI uint32_t h2(float lo, float hi) { uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI uint16_t h1(float v) { uint16_t r; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(v)); return r; }
// the scale exponent of a row whose largest magnitude is m: m 2^e in [2^14, 2^15) (e = 14 - floor(log2 m), at most 60); 0 / denormal m: 60;
// inf / nan: -120 (the row is lost anyway)
DEVI int sexp(float m) {
  const int E = (int)((__float_as_uint(m) >> 23) & 255u);
  return E == 0 ? 60 : E == 255 ? -120 : min(141 - E, 60);
}
DEVI float p2(int e) { return __uint_as_float((uint32_t)(127 + e) << 23); }
DEVI float amax4(float a, float b, float c, float d) { return fmaxf(fmaxf(fabsf(a), fabsf(b)), fmaxf(fabsf(c), fabsf(d))); }
// the 4 rows' maxima over the warp (lane: 4 channels each; m >= 0): non-negative floats order as their bit patterns, so ONE redux.sync per
// row (the 5-step shuffle butterfly cost ~1 us of the 64-row post's x2 phase)
DEVI void wmax4(float (&m)[4]) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint32_t r;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(__float_as_uint(m[k])));
    m[k] = __uint_as_float(r);
  }
}
// lane j (< n) gets v[j] (v uniform over the warp)
template <int N> DEVI float pick(const float (&v)[N], int lane) {
  float r = v[0];
#pragma unroll
  for (int j = 1; j < N; ++j) r = lane == j ? v[j] : r;
  return r;
}
// this lane's 4 channels (4 lane ..) of row r in an fp16 operand tile (two K-slices of 64 halves, TSL bytes apart, 128-B swizzle)
DEVI uint32_t opH(uint32_t base, int r, int lane) {
  return base + (uint32_t)(lane >> 4) * TSL + sw128((uint32_t)r, (uint32_t)((lane & 15) >> 1)) + (uint32_t)(lane & 1) * 8u;
}
DEVI void put4h(uint32_t a, float s, float x, float y, float z, float w) { sts64(a, h2(x * s, y * s), h2(z * s, w * s)); }
// hidden feature f (K index) of rows r0 .. r0 + 15 (v already scaled) -> the fp16 tile of K = hidden
DEVI void put_col16h(uint32_t base, int f, int r0, const float (&v)[16]) {
  const uint32_t a0 = base + (uint32_t)(f >> 6) * TSL + (uint32_t)(f & 7) * 2u, ch = (uint32_t)((f & 63) >> 3);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint32_t r = (uint32_t)(r0 + j);
    asm volatile("st.shared.b16 [%0], %1;" :: "r"(a0 + r * 128u + ((ch ^ (r & 7u)) << 4)), "h"(h1(v[j])) : "memory");
  }
}
DEVI void ld16f(uint32_t a, float (&v)[16]) {                         // 16 consecutive row factors (16-B aligned)
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float4 t = lds4f(a + 16u * (uint32_t)q);
    v[4 * q] = t.x; v[4 * q + 1] = t.y; v[4 * q + 2] = t.z; v[4 * q + 3] = t.w;
  }
}

// ---------------------------------------------------------------------------------------------------------------- post (fp16-exact)
// smem: X fp16 (gated; x2 unless EARLY) | X2 fp16 (EARLY) | Y fp32 (y) | H fp16 [TU x 128 NHB] (non-PAIR: t fp32 in its place once the Ws
// MMAs are done) | PAIR: T fp32 (t_0 / rank 1's t_1 before the copy) | T1 fp32 (rank 0: t_1, by ONE bulk copy from rank 1) | the ring of
// 16 KB granules | row factors (1 KB: gated -> y [x wfo], x2 -> a / b [x wfu], the h scale, h -> t [x wfs]) | barriers
constexpr int OH_X = 0, OH_X2 = HTX, OH_Y = OH_X2 + (EARLY ? HTX : 0), OH_H = OH_Y + HTF;
constexpr int OH_T = PAIR ? OH_H + HTX : OH_H, OH_T1 = OH_T + HTF, OH_W = PAIR ? OH_T1 + HTF : OH_H + 2 * HTX;
constexpr int OH_SC = O_BAR - 1024, NGH = (OH_SC - OH_W) / GRT;
static_assert(OH_W % 1024 == 0 && NGH >= 6 && NGH <= NSTMAX && NHB * HTX <= HTF, "atom_post_f16t: smem");
DEVI void wloadH(Bars& B, uint32_t wb, int gi, const CUtensorMap* m, int c0, int c1, int c1b = -1) {   // lane s of warp 0 issues slot s
  const int slot = gi % NGH;
  if ((int)(threadIdx.x & 31) != slot) return;
  if (gi >= NGH) mbar_wait(&B.wempty[slot], ((gi / NGH) - 1) & 1);
  mbar_expect_tx(&B.wfull[slot], (uint32_t)GRT);
  const uint32_t dst = wb + (uint32_t)slot * GRT;
  tma_load_2d(dst, m, &B.wfull[slot], c0, c1);
  if (c1b >= 0) tma_load_2d(dst + GRT / 2, m, &B.wfull[slot], c0, c1b);
}
struct MmaH {
  uint32_t wb;
  int gi;
  DEVI void one(Bars& B, uint32_t d, uint32_t bact, bool acc) {      // D^T (+)= the next granule [128 x 64 K] . X^T (TU rows, K-slice at bact)
    const int slot = gi % NGH;
    mbar_wait(&B.wfull[slot], (gi / NGH) & 1);
    tc_fence_after();
    const uint64_t da = desc_k128(wb + (uint32_t)slot * GRT), db = desc_k128(bact);
    if (elect_one()) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) umma_ss(d, da + (uint64_t)(2 * kk), db + (uint64_t)(2 * kk), IH, (acc || kk) ? 1u : 0u);   // K = 16 (32 B)
      tc_commit(&B.wempty[slot]);
    }
    __syncwarp();
    ++gi;
  }
};

// arguments as atom_post_tf32t, the packs fp16 (box 64 x 128; UP 64 x 64), then the weight factors 2^-e_w of Wo / Wu / Ws and the h bound's
// A1 / B1 / U1
extern "C" __global__ void __maxnreg__(168)
atom_post_f16t(const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwu, const __grid_constant__ CUtensorMap mws,
               const float* __restrict__ o, const float* __restrict__ g, int gld, const float* __restrict__ s, const float* __restrict__ so,
               const float* __restrict__ s2, const float* __restrict__ b2, const float* __restrict__ st, int mld, float* __restrict__ out,
               int M, float eps, int nfull, int mrows, float wfo, float wfu, float wfs, float ba, float bb, float bu) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  (void)nfull;
  (void)bu;
#if PAIR
  const uint32_t rk = cluster_rank();                                // this CTA's hidden block
  const int hb0 = (int)rk;
#else
  const int hb0 = 0;
#endif
  const uint32_t xa = su + OH_X, x2t = EARLY ? su + OH_X2 : xa, yt = su + OH_Y, ht = su + OH_H, tt = su + OH_T, wb = su + OH_W, sc = su + OH_SC;
  if (threadIdx.x == 0) TRS(0);
  const uint32_t tmem = setup(B, warp, 2, 1u);                       // (xfull: PAIR's bulk copy, count 1)
  if (threadIdx.x == 128) TRS(1);
  const int my = TU64 ? units_of(M / TU) : post_units(M, -1);

  if (warp == 0) {
    if (lane == 0) TRS(14);
    if (lane < NGH) {
      int gi = 0;
      for (int i = 0; i < my; ++i) {
        if (!EARLY) for (int k = 0; k < 2; ++k) wloadH(B, wb, gi++, &mwo, 64 * k, 0);           // Wo
        for (int hb = hb0; hb < hb0 + NHB; ++hb)                                                 // the parts of hidden block hb
          for (int k = 0; k < 2; ++k)
            for (int p = 0; p < NP; ++p) {
#if UP
              wloadH(B, wb, gi++, &mwu, 64 * k, 192 * (2 * hb) + 64 * p, 192 * (2 * hb + 1) + 64 * p);
#else
              wloadH(B, wb, gi++, &mwu, 64 * k, 256 * hb + 128 * p);
#endif
            }
        for (int hb = hb0; hb < hb0 + NHB; ++hb)
          for (int k = 0; k < 2; ++k) wloadH(B, wb, gi++, &mws, 128 * hb + 64 * k, 0);         // Ws of hidden block hb
        if (EARLY) for (int k = 0; k < 2; ++k) wloadH(B, wb, gi++, &mwo, 64 * k, 0);            // (EARLY: Wo last)
      }
    }
  } else if (warp == 1) {
    MmaH mm{wb, 0};
    int na = 0, nh = 0;
    for (int i = 0; i < my; ++i) {
      if (!EARLY) {
        mbar_wait(&B.afull, (na++) & 1);                             // gated
        tc_fence_after();
        if (lane == 0) TRS(10);
        for (int k = 0; k < 2; ++k) mm.one(B, tmem + TC_Y, xa + (uint32_t)k * TSL, k > 0);
        commit_acc(B);
        if (lane == 0) TRS(11);
      }
      mbar_wait(&B.afull, (na++) & 1);                               // x2
      tc_fence_after();
      if (lane == 0) TRS(12);
      for (int hl = 0; hl < NHB; ++hl) {
        for (int k = 0; k < 2; ++k)
          for (int p = 0; p < NP; ++p) mm.one(B, tmem + TC_A + TU * (NP * hl + p), x2t + (uint32_t)k * TSL, k > 0);
        commit_acc(B);
      }
      if (lane == 0) TRS(13);
      for (int hl = 0; hl < NHB; ++hl) {
        mbar_wait(&B.hfull, (nh++) & 1);                             // h of hidden block hl
        tc_fence_after();
        for (int k = 0; k < 2; ++k) mm.one(B, tmem + TC_T, ht + (uint32_t)(2 * hl + k) * TSL, hl > 0 || k > 0);
      }
      if (EARLY) {
        mbar_wait(&B.afull, (na++) & 1);                             // gated (after the attention)
        tc_fence_after();
        for (int k = 0; k < 2; ++k) mm.one(B, tmem + TC_Y, xa + (uint32_t)k * TSL, k > 0);
      }
      if (lane == 0) TRS(16);
      commit_acc(B);                                                 // t (EARLY: and y)
    }
  } else if (warp == 3) {
    for (int i = 0; i < my && i < 2; ++i) {                          // g, s and the tables before the wait, o after it
      const int m0 = TU * (PCID + i * PCN), t0 = m0 % mrows;
      pf_rows(g, gld, m0, TU, lane); pf_rows(s, 128, m0, TU, lane); pf_rows(so, mld, t0, TU, lane);
      pf_rows(s2, mld, t0, TU, lane); pf_rows(b2, mld, t0, TU, lane); pf_rows(st, mld, t0, TU, lane);
    }
    pdl_wait();
    for (int i = 0; i < my && i < 2; ++i) pf_rows(o, 128, TU * (PCID + i * PCN), TU, lane);
  } else if (warp >= 4) {
    constexpr int RPT = TU / 2, NHH = RPT / 16, NRD = TU / 32;       // a TMEM-phase thread: one feature x RPT rows; row phases: NRD rounds of 4
    const int e = warp - 4, qd = warp & 3, wg = e >> 2, f = 32 * qd + lane, rb = RPT * wg;
    const uint32_t tl = tmem + ((uint32_t)(32 * qd) << 16) + (uint32_t)rb;
    int nc = 0;
#if PAIR
    int nx = 0;
#endif
    for (int i = 0; i < my; ++i) {
      const int m0 = TU * (PCID + i * PCN);
      const size_t tsh = (size_t)(m0 - m0 % mrows);
#if PAIR
      if (rk == 0 && threadIdx.x == 128) mbar_expect_tx(&B.xfull, (uint32_t)HTF);   // this unit's t_1 (the bulk copy from rank 1)
#endif
      constexpr bool HO = TU == 32;                                 // 32-row units: every row input but o in registers before the wait
      float4 svh[4], soh[4], s2h[4], b2h[4], sth[4];               // (as atom_post_tf32t; s and the tables are complete before the chain)
      if (HO) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const size_t row = (size_t)(m0 + e + 8 * k);
          svh[k] = ldg4(s + row * 128 + 4 * lane);
          soh[k] = ldg4(so + (row - tsh) * mld + 4 * lane);
          s2h[k] = ldg4(s2 + (row - tsh) * mld + 4 * lane);
          b2h[k] = ldg4(b2 + (row - tsh) * mld + 4 * lane);
          sth[k] = ldg4(st + (row - tsh) * mld + 4 * lane);
        }
      }
      float4 gv[TU / 8];                                             // g (from pre, complete before the chain's last launch): before the wait
#pragma unroll
      for (int k = 0; k < TU / 8; ++k) gv[k] = ldg4(g + (size_t)(m0 + e + 8 * k) * gld + 4 * lane);
#if TU64 && POST1R
      // (round 3 part 15, -DPOST1R=1) the x2 phase's row inputs (s, so, s2, b2: complete before the chain) one round AHEAD: round 0's before
      // the wait for y (non-EARLY: issued under the Wo MMAs), round rd + 1's at the start of round rd (the two dependent load rounds took
      // ~3.3 us of the 64-row post, Protenix 1024 x 5)
      float4 xq[2][4][4];                                            // [buffer][s, so, s2, b2][row k]
      auto x2_ld = [&](int rd, float4 (&q)[4][4], bool from_s) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const size_t row = (size_t)(m0 + e + 8 * (4 * rd + k));
          q[0][k] = ldg4(s + row * 128 + 4 * lane);
          if (!from_s) q[1][k] = ldg4(so + (row - tsh) * mld + 4 * lane);
          q[2][k] = ldg4(s2 + (row - tsh) * mld + 4 * lane);
          q[3][k] = ldg4(b2 + (row - tsh) * mld + 4 * lane);
        }
      };
#endif
      auto x2_phase = [&](bool from_s) {                            // x2 = LN(.) s2 + b2 -> fp16 x2 tile: of s (EARLY) or of a1 = s + so y
        float mxa[TU / 8];                                           // the warp's rows' maxima (row e + 8 j)
#if TU64 && POST1R
        if (from_s) x2_ld(0, xq[0], true);                           // (EARLY: the first phase; non-EARLY: loaded before the y wait)
#endif
#pragma unroll
        for (int rd = 0; rd < NRD; ++rd) {
          float x[4][4], mx[4];
          float4 sc4[4], sf4[4];
#if TU64 && POST1R
          if (rd + 1 < NRD) x2_ld(rd + 1, xq[(rd + 1) & 1], from_s);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int rr = e + 8 * (4 * rd + k);
            const float4 sv = xq[rd & 1][0][k];
            sc4[k] = xq[rd & 1][2][k];
            sf4[k] = xq[rd & 1][3][k];
            if (from_s) {
              x[k][0] = sv.x; x[k][1] = sv.y; x[k][2] = sv.z; x[k][3] = sv.w;
            } else {
              const float4 so4 = xq[rd & 1][1][k];
              const float4 y = lds4f(opT(yt, rr, lane));
              x[k][0] = fmaf(so4.x, y.x, sv.x); x[k][1] = fmaf(so4.y, y.y, sv.y); x[k][2] = fmaf(so4.z, y.z, sv.z); x[k][3] = fmaf(so4.w, y.w, sv.w);
            }
          }
#else
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int rr = e + 8 * (4 * rd + k);
            const size_t row = (size_t)(m0 + rr);
            const float4 sv = HO ? svh[k] : ldg4(s + row * 128 + 4 * lane);
            sc4[k] = HO ? s2h[k] : ldg4(s2 + (row - tsh) * mld + 4 * lane);
            sf4[k] = HO ? b2h[k] : ldg4(b2 + (row - tsh) * mld + 4 * lane);
            if (from_s) {
              x[k][0] = sv.x; x[k][1] = sv.y; x[k][2] = sv.z; x[k][3] = sv.w;
            } else {
              const float4 so4 = HO ? soh[k] : ldg4(so + (row - tsh) * mld + 4 * lane);
              const float4 y = lds4f(opT(yt, rr, lane));
              x[k][0] = fmaf(so4.x, y.x, sv.x); x[k][1] = fmaf(so4.y, y.y, sv.y); x[k][2] = fmaf(so4.z, y.z, sv.z); x[k][3] = fmaf(so4.w, y.w, sv.w);
            }
          }
#endif
          ln4x4(x, eps);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            x[k][0] = rna(fmaf(x[k][0], sc4[k].x, sf4[k].x)); x[k][1] = rna(fmaf(x[k][1], sc4[k].y, sf4[k].y));
            x[k][2] = rna(fmaf(x[k][2], sc4[k].z, sf4[k].z)); x[k][3] = rna(fmaf(x[k][3], sc4[k].w, sf4[k].w));
            mx[k] = amax4(x[k][0], x[k][1], x[k][2], x[k][3]);
          }
          if (warp == 4 && lane == 0 && rd == 0) TRS(17);
          wmax4(mx);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int rr = e + 8 * (4 * rd + k);
            put4h(opH(x2t, rr, lane), p2(sexp(mx[k])), x[k][0], x[k][1], x[k][2], x[k][3]);
            mxa[4 * rd + k] = mx[k];
          }
        }
        if (lane < TU / 8) {                                         // lane j: row e + 8 j's factors
          const float m = pick(mxa, lane);
          const int rr = e + 8 * lane, ex = sexp(m);
          float hbnd = 2.f * fmaf(ba, m, 0.2785f) * (bb * m);         // |h| bound of the row
#if UP
          hbnd *= bu * m;
#endif
          const int eh = sexp(hbnd);
          sts32(sc + 256u + 4u * (uint32_t)rr, __float_as_uint(p2(-ex) * wfu));
          sts32(sc + 512u + 4u * (uint32_t)rr, __float_as_uint(p2(eh)));
          sts32(sc + 768u + 4u * (uint32_t)rr, __float_as_uint(p2(-eh) * wfs));
        }
        tc_fence_before();
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.afull);
        named_bar_sync(1, 256);                                      // the row factors, before any TMEM phase reads them
      };
      auto gated_phase = [&]() {                                     // (unit 0: the wait first) o -> gated = sigmoid(g) o -> fp16 X
        if (i == 0) {
          pdl_wait();
          pdl_launch();
        }
        float mxa[TU / 8];
#pragma unroll
        for (int rd = 0; rd < NRD; ++rd) {
          float4 ov[4];
          float v[4][4], mx[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) ov[k] = ldg4(o + (size_t)(m0 + e + 8 * (4 * rd + k)) * 128 + 4 * lane);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float4 gg = gv[4 * rd + k];
            v[k][0] = rna(ov[k].x * sigf(gg.x)); v[k][1] = rna(ov[k].y * sigf(gg.y)); v[k][2] = rna(ov[k].z * sigf(gg.z)); v[k][3] = rna(ov[k].w * sigf(gg.w));
            mx[k] = amax4(v[k][0], v[k][1], v[k][2], v[k][3]);
          }
          wmax4(mx);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            put4h(opH(xa, e + 8 * (4 * rd + k), lane), p2(sexp(mx[k])), v[k][0], v[k][1], v[k][2], v[k][3]);
            mxa[4 * rd + k] = mx[k];
          }
        }
        if (lane < TU / 8) sts32(sc + 4u * (uint32_t)(e + 8 * lane), __float_as_uint(p2(-sexp(pick(mxa, lane))) * wfo));
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.afull);
        named_bar_sync(1, 256);
      };
      auto y_out = [&]() {                                           // y^T (unscaled) -> Y fp32 (feature per lane)
#pragma unroll
        for (int hh = 0; hh < NHH; ++hh) {
          float v[16], fc[16];
          tld16(tl + TC_Y + 16 * hh, v);
          ld16f(sc + 4u * (uint32_t)(rb + 16 * hh), fc);
#pragma unroll
          for (int j = 0; j < 16; ++j) v[j] *= fc[j];
          put_col16(yt, f, rb + 16 * hh, v);
        }
      };
      if (EARLY) {
        x2_phase(true);
      } else {
        gated_phase();
        if (warp == 4 && lane == 0) TRS(2);
#if TU64 && POST1R
        x2_ld(0, xq[0], false);                                      // x2's first round under the Wo MMAs
#endif
        mbar_wait(&B.accf, (nc++) & 1);
        tc_fence_after();
        if (warp == 4 && lane == 0) TRS(3);
        y_out();
        named_bar_sync(1, 256);
        if (warp == 4 && lane == 0) TRS(15);
        x2_phase(false);
        if (warp == 4 && lane == 0) TRS(4);
      }
      // ---------------------------------------------------------------- feature per lane: h = silu(a) b (u) of each hidden block -> fp16 H
      for (int hl = 0; hl < NHB; ++hl) {
        mbar_wait(&B.accf, (nc++) & 1);
        tc_fence_after();
        if (warp == 4 && lane == 0) TRS(5 + hl);
#pragma unroll
        for (int hh = 0; hh < NHH; ++hh) {
          float va[16], vb[16], xs[16], hs[16];
          tld16(tl + TC_A + TU * (NP * hl) + 16 * hh, va);
          tld16(tl + TC_A + TU * (NP * hl + 1) + 16 * hh, vb);
          ld16f(sc + 256u + 4u * (uint32_t)(rb + 16 * hh), xs);
          ld16f(sc + 512u + 4u * (uint32_t)(rb + 16 * hh), hs);
#if UP
          float vu[16];
          tld16(tl + TC_A + TU * (NP * hl + 2) + 16 * hh, vu);
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            const float a = va[j] * xs[j];
            va[j] = rna(a * sigf(a) * (vb[j] * xs[j]) * (vu[j] * xs[j])) * hs[j];
          }
#else
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            const float a = va[j] * xs[j];
            va[j] = rna(a * sigf(a) * (vb[j] * xs[j])) * hs[j];
          }
#endif
          put_col16h(ht, 128 * hl + f, rb + 16 * hh, va);
        }
        tc_fence_before();
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.hfull);
      }
      if (warp == 4 && lane == 0) TRS(7);
      if (EARLY) gated_phase();
      // ---------------------------------------------------------------- feature per lane: t^T (unscaled) -> T fp32 (PAIR rank 1: then to rank 0)
      mbar_wait(&B.accf, (nc++) & 1);
      tc_fence_after();
      if (warp == 4 && lane == 0) TRS(8);
      if (EARLY) y_out();
#if PAIR
      if (rk == 1 && i > 0) mbar_wait_cl(&B.hfree, (i - 1) & 1);    // rank 0 has read the previous t_1: T (the copy's source) is free
#endif
#pragma unroll
      for (int hh = 0; hh < NHH; ++hh) {
        float v[16], fc[16];
        tld16(tl + TC_T + 16 * hh, v);
        ld16f(sc + 768u + 4u * (uint32_t)(rb + 16 * hh), fc);
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] *= fc[j];
        put_col16(tt, f, rb + 16 * hh, v);
      }
      tc_fence_before();
#if PAIR
      if (rk == 1) {                                                 // t_1 -> rank 0's T1: one bulk copy, completing on rank 0's xfull
        fence_proxy_async();
        named_bar_sync(1, 256);
        if (threadIdx.x == 128) bulk_to_peer(su + OH_T1, tt, (uint32_t)HTF, &B.xfull, 0);
        continue;
      }
#endif
      named_bar_sync(1, 256);
#if PAIR
      mbar_wait_cl(&B.xfull, (nx++) & 1);                            // t_1 has landed in T1
#endif
      if (warp == 4 && lane == 0) TRS(9);
      // ---------------------------------------------------------------- warp per row: out = (s + so y) + st t
#pragma unroll
      for (int rd = 0; rd < NRD; ++rd) {
        float4 sv[4], so4[4], tv[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const size_t row = (size_t)(m0 + e + 8 * (4 * rd + k));
          sv[k] = HO ? svh[k] : ldg4(s + row * 128 + 4 * lane);
          so4[k] = HO ? soh[k] : ldg4(so + (row - tsh) * mld + 4 * lane);
          tv[k] = HO ? sth[k] : ldg4(st + (row - tsh) * mld + 4 * lane);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int rr = e + 8 * (4 * rd + k);
          const float4 y = lds4f(opT(yt, rr, lane));
          float4 t = lds4f(opT(tt, rr, lane));
#if PAIR
          const float4 t1 = lds4f(opT(su + OH_T1, rr, lane));       // t = t_0 + t_1 (in this order)
          t.x += t1.x; t.y += t1.y; t.z += t1.z; t.w += t1.w;
#endif
          const float4 a1 = make_float4(fmaf(so4[k].x, y.x, sv[k].x), fmaf(so4[k].y, y.y, sv[k].y), fmaf(so4[k].z, y.z, sv[k].z),
                                        fmaf(so4[k].w, y.w, sv[k].w));
          stg128(out + (size_t)(m0 + rr) * 128 + 4 * lane, u4(fmaf(tv[k].x, t.x, a1.x), fmaf(tv[k].y, t.y, a1.y), fmaf(tv[k].z, t.z, a1.z),
                                                             fmaf(tv[k].w, t.w, a1.w)));
        }
      }
      named_bar_sync(1, 256);                                        // X / Y / H / T read before the next unit writes them
      if (warp == 4 && lane == 0) TRS(18);
#if PAIR
      if (warp == 4 && lane == 0) mbar_arrive_remote(&B.hfree, 1);   // T1 is free for the next t_1
#endif
    }
  }
#if PAIR
  cluster_sync();                                                    // no CTA leaves while its peer may still signal it (or copy from it)
#endif
  teardown(tmem, warp);
}

#if !UP && !PARS && !PAIR
// ---------------------------------------------------------------------------------------------------------------- pre (fp16-exact)
// smem: X1 fp16 | XKV fp16 | all of Wp (8 granules [128 x 64 K], block b's two at 2 b, 2 b + 1: loaded before the wait) | the stagings |
// row factors (x1, xkv: 2^-e_row)
constexpr int OQ_X = 0, OQ_K = HTX, OQ_W = 2 * HTX, NGQ = 8, OQ_ST = OQ_W + NGQ * GRT, OQ_SC = OQ_ST + 8 * EPB;
static_assert(OQ_W % 1024 == 0 && OQ_SC + 512 <= O_BAR && NGQ <= NSTMAX, "atom_pre_f16t: smem");

// arguments as atom_pre_tf32t (mwp: the fp16 pack, box 64 x 128), then the four blocks' weight factors 2^-e_b
extern "C" __global__ void __maxnreg__(168)
atom_pre_f16t(const __grid_constant__ CUtensorMap mwp, const __grid_constant__ CUtensorMap mp, const float* __restrict__ s,
              const float* __restrict__ s1, const float* __restrict__ b1, int mld, const float* __restrict__ mks, const float* __restrict__ mkb,
              int kld, const float* __restrict__ bq, int cross, int rndmask, int M, float eps, int nfull, int mrows,
#if KQLN
              const float* __restrict__ KQ, float kq_eps,
#endif
              float wf0, float wf1, float wf2, float wf3) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  (void)nfull;
  const int my = units_of(M / TU);
  const uint32_t x1t = su + OQ_X, xkt = su + OQ_K, wb = su + OQ_W, sc = su + OQ_SC;
  if (threadIdx.x == 0) TRS(0);
  const uint32_t tmem = setup(B, warp, 2);                           // (no wait here: warp 3 and the row warps wait themselves)
  if (threadIdx.x == 128) TRS(1);
  auto row0 = [&](int i) { return TU * ((int)blockIdx.x + i * (int)gridDim.x); };

  if (warp == 0) {
    static_assert(NGQ == 8, "atom_pre_f16t: all of Wp resident");
    if (lane < NGQ && my > 0) {                                      // all of Wp ONCE (granule 2 b + k from lane 2 b + k): resident for every
      const int b = lane >> 1, k = lane & 1;                         // unit of this CTA (the persistent many-unit grids load it once)
      mbar_expect_tx(&B.wfull[lane], (uint32_t)GRT);
      tma_load_2d(wb + (uint32_t)lane * GRT, &mwp, &B.wfull[lane], 64 * k, 128 * b);
    }
  } else if (warp == 1) {
    for (int i = 0; i < my; ++i) {
      mbar_wait(&B.afull, i & 1);
      tc_fence_after();
      if (lane == 0) TRS(10);
      for (int b = 0; b < 4; ++b) {
        const uint32_t xb = (cross && b >= 2) ? xkt : x1t;            // cross: k | v from xkv
        for (int k = 0; k < 2; ++k) {
          const int slot = 2 * b + k;
          if (i == 0) mbar_wait(&B.wfull[slot], 0);
          tc_fence_after();
          const uint64_t da = desc_k128(wb + (uint32_t)slot * GRT), db = desc_k128(xb + (uint32_t)k * TSL);
          if (elect_one()) {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) umma_ss(tmem + TU * b, da + (uint64_t)(2 * kk), db + (uint64_t)(2 * kk), IH, (k | kk) ? 1u : 0u);
          }
          __syncwarp();
        }
        if (b & 1) commit_acc(B);                                    // blocks 0 | 1, then 2 | 3
        if (lane == 0) TRS(11 + (b >> 1));
      }
    }
  } else if (warp == 3) {
    for (int i = 0; i < my && i < 2; ++i) pf_rows(s, 128, row0(i), TU, lane);   // s: complete before the chain
    pdl_wait();
    for (int i = 0; i < my && i < 2; ++i) {
      const int t0 = row0(i) % mrows;
      pf_rows(s1, mld, t0, TU, lane); pf_rows(b1, mld, t0, TU, lane);
      if (cross) { pf_rows(mks, kld, t0, TU, lane); pf_rows(mkb, kld, t0, TU, lane); }
    }
  } else if (warp >= 4) {
    const int e = warp - 4, qd = warp & 3, wg = e >> 2, f = 32 * qd + lane;
    const uint32_t tl = tmem + ((uint32_t)(32 * qd) << 16);
    const uint32_t buf = su + OQ_ST + (uint32_t)e * EPB;             // this warp's staging
    for (int i = 0; i < my; ++i) {
      const int m0 = row0(i);
      const size_t tsh = (size_t)(m0 - m0 % mrows);
      // ---------------------------------------------------------------- warp per row (4 rows a warp a round): AdaLN 1 (+ K / V), row scales
      // round rg of 4 rows from its loaded inputs (s, the AdaLN 1 table rows, the cross mode's K / V table rows)
      auto row_round = [&](int rg, const float4 (&sv)[4], const float4 (&scv)[4], const float4 (&sf)[4], const float4 (&kc)[4],
                           const float4 (&kf)[4]) {
        float x[4][4], xr[4][4], kv[4][4], m1[4], mk[4];
        const int ro = 32 * rg;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          x[k][0] = sv[k].x; x[k][1] = sv[k].y; x[k][2] = sv[k].z; x[k][3] = sv[k].w;
          mk[k] = 0.f;
        }
        if (warp == 4 && lane == 0) TRS(21 + rg);
        ln4x4(x, eps);
        if (cross == 2) {                                            // AlphaFold 3: xkv = LN(s) mks + mkb, from the same statistics
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            kv[k][0] = rna(fmaf(x[k][0], kc[k].x, kf[k].x)); kv[k][1] = rna(fmaf(x[k][1], kc[k].y, kf[k].y));
            kv[k][2] = rna(fmaf(x[k][2], kc[k].z, kf[k].z)); kv[k][3] = rna(fmaf(x[k][3], kc[k].w, kf[k].w));
            mk[k] = amax4(kv[k][0], kv[k][1], kv[k][2], kv[k][3]);
          }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          x[k][0] = fmaf(x[k][0], scv[k].x, sf[k].x); x[k][1] = fmaf(x[k][1], scv[k].y, sf[k].y);
          x[k][2] = fmaf(x[k][2], scv[k].z, sf[k].z); x[k][3] = fmaf(x[k][3], scv[k].w, sf[k].w);
#pragma unroll
          for (int c = 0; c < 4; ++c) xr[k][c] = rna(x[k][c]);
          m1[k] = amax4(xr[k][0], xr[k][1], xr[k][2], xr[k][3]);
        }
        if (cross == 1) {                                            // xkv = LN(x1) mks + mkb (x1 exact)
          ln4x4(x, eps);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            kv[k][0] = rna(fmaf(x[k][0], kc[k].x, kf[k].x)); kv[k][1] = rna(fmaf(x[k][1], kc[k].y, kf[k].y));
            kv[k][2] = rna(fmaf(x[k][2], kc[k].z, kf[k].z)); kv[k][3] = rna(fmaf(x[k][3], kc[k].w, kf[k].w));
            mk[k] = amax4(kv[k][0], kv[k][1], kv[k][2], kv[k][3]);
          }
        }
        wmax4(m1);
        if (warp == 4 && lane == 0) TRS(19 + rg);
        if (cross) wmax4(mk);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int rr = ro + e + 8 * k;
          put4h(opH(x1t, rr, lane), p2(sexp(m1[k])), xr[k][0], xr[k][1], xr[k][2], xr[k][3]);
          if (cross) put4h(opH(xkt, rr, lane), p2(sexp(mk[k])), kv[k][0], kv[k][1], kv[k][2], kv[k][3]);
        }
        if (lane < 4) {                                              // lane k: row ro + e + 8 k's factors
          const uint32_t ra = sc + 4u * (uint32_t)(ro + e + 8 * lane);
          sts32(ra, __float_as_uint(p2(-sexp(pick(m1, lane)))));
          if (cross) sts32(ra + 256u, __float_as_uint(p2(-sexp(pick(mk, lane)))));
        }
      };
      auto ld_rows = [&](const float* p, int ld, int mr, size_t sh, float4 (&v)[4]) {
#pragma unroll
        for (int k = 0; k < 4; ++k) v[k] = ldg4(p + ((size_t)(mr + e + 8 * k) - sh) * ld + 4 * lane);
      };
#if TU64 && PRE1R
      // (round 3 part 15, -DPRE1R=1) the 64-row unit's two rounds from ONE load round: s of both before the wait, the tables of both right
      // after it (the two dependent rounds took 3.3 + 2.4 us of L2 / HBM latency, Protenix 1024 x 5); the cross mode keeps the per-round form
      if (!cross) {
        float4 sa[2][4], ta[2][4], fa[2][4];
#pragma unroll
        for (int rg = 0; rg < 2; ++rg) ld_rows(s, 128, m0 + 32 * rg, 0, sa[rg]);   // s: the block input (complete)
        if (i == 0) {                                                // the tables come from the previous kernel
          pdl_wait();
          pdl_launch();
          if (warp == 4 && lane == 0) TRS(5);
        }
#pragma unroll
        for (int rg = 0; rg < 2; ++rg) {
          ld_rows(s1, mld, m0 + 32 * rg, tsh, ta[rg]);
          ld_rows(b1, mld, m0 + 32 * rg, tsh, fa[rg]);
        }
        if (warp == 4 && lane == 0) TRS(6);
#pragma unroll
        for (int rg = 0; rg < 2; ++rg) {
          row_round(rg, sa[rg], ta[rg], fa[rg], ta[rg], fa[rg]);
          if (warp == 4 && lane == 0 && rg == 0) TRS(7);
        }
      } else
#endif
      {
#if TU64
#pragma unroll 1
#else
#pragma unroll
#endif
        for (int rg = 0; rg < TU / 32; ++rg) {
          float4 scv[4], sf[4], kc[4], kf[4], sv[4];
          const int mr = m0 + 32 * rg;
          ld_rows(s, 128, mr, 0, sv);                                // s: the block input (complete)
          if (i == 0 && rg == 0) {                                   // the tables come from the previous kernel
            pdl_wait();
            pdl_launch();
            if (warp == 4 && lane == 0) TRS(5);
          }
          ld_rows(s1, mld, mr, tsh, scv);
          ld_rows(b1, mld, mr, tsh, sf);
          if (cross) {
            ld_rows(mks, kld, mr, tsh, kc);
            ld_rows(mkb, kld, mr, tsh, kf);
          }
          if (warp == 4 && lane == 0) TRS(6 + rg);
          row_round(rg, sv, scv, sf, kc, kf);
        }
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.afull);
      named_bar_sync(1, 256);                                        // the row factors, before the epilogue reads them
      if (warp == 4 && lane == 0) TRS(2);
      // ---------------------------------------------------------------- feature per lane: P^T blocks (2 h + wg) -> P (bias on q, tf32 rounding)
      for (int h = 0; h < 2; ++h) {
        mbar_wait(&B.accf, (2 * i + h) & 1);
        tc_fence_after();
        if (warp == 4 && lane == 0) TRS(3 + h);
        const int b = 2 * h + wg;
        const float wfb = b == 0 ? wf0 : b == 1 ? wf1 : b == 2 ? wf2 : wf3;
        const uint32_t fcb = sc + ((cross && b >= 2) ? 256u : 0u);
#if TU64
#pragma unroll 1
#else
#pragma unroll
#endif
        for (int c = 0; c < TU / 32; ++c) {                           // (TU64: the block's two 32-row chunks)
        float v[32];
        tld(tl + (uint32_t)(TU * b + 32 * c), v);
        {
          float fc[16];
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            ld16f(fcb + 4u * (uint32_t)(32 * c + 16 * hh), fc);
#pragma unroll
            for (int j = 0; j < 16; ++j) v[16 * hh + j] *= wfb * fc[j];
          }
        }
        float bias = b == 0 ? __ldg(bq + f) : 0.f;
#if KQLN
        if (b < 2) {                                                 // RoseTTAFold3: the flat LayerNorm of q (+ bq) / k (as atom_pre_tf32t)
          const uint32_t rb = su + OQ_K + (uint32_t)wg * 1024u;
#pragma unroll
          for (int j = 0; j < 32; ++j) v[j] += bias;
          bias = 0.f;
          sts32(rb + (uint32_t)(qd * 32 + lane) * 4u, __float_as_uint(tsum32(v, lane)));
          named_bar_sync(2 + wg, 128);
          float mean = 0.f;
#pragma unroll
          for (int q = 0; q < 4; ++q) mean += __uint_as_float(lds32(rb + (uint32_t)(q * 32 + lane) * 4u));
          mean *= 1.f / 128;
          float d[32];
#pragma unroll
          for (int j = 0; j < 32; ++j) { const float dj = v[j] - __shfl_sync(0xffffffffu, mean, j); d[j] = dj * dj; v[j] = dj; }
          sts32(rb + 512u + (uint32_t)(qd * 32 + lane) * 4u, __float_as_uint(tsum32(d, lane)));
          named_bar_sync(2 + wg, 128);
          float var = 0.f;
#pragma unroll
          for (int q = 0; q < 4; ++q) var += __uint_as_float(lds32(rb + 512u + (uint32_t)(q * 32 + lane) * 4u));
          const float rstd = 1.f / sqrtf(var * (1.f / 128) + kq_eps);
          const float kw = __ldg(KQ + 256 * b + f), kb = __ldg(KQ + 256 * b + 128 + f);
#pragma unroll
          for (int j = 0; j < 32; ++j) v[j] = fmaf(v[j] * __shfl_sync(0xffffffffu, rstd, j), kw, kb);
        }
#endif
        const bool rnd = (rndmask >> b) & 1;
        if (lane == 0) tma_store_wait_read0();                      // the previous chunk's store has read the staging
        __syncwarp();
        const uint32_t ch = (uint32_t)(lane >> 2), cb = (uint32_t)(lane & 3) * 4u;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          float w = v[j] + bias;
          if (rnd) w = rna(w);
          sts32(buf + (uint32_t)j * 128u + ((ch ^ ((uint32_t)j & 7u)) << 4) + cb, __float_as_uint(w));
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) {
          tma_store_2d(&mp, buf, 128 * b + 32 * qd, m0 + 32 * c);
          tma_store_commit();
        }
        }
      }
      tc_fence_before();
      if (lane == 0) tma_store_wait_read0();                        // the stagings are read before the next unit (if any) rewrites them
      __syncwarp();
      named_bar_sync(1, 256);
    }
    if (lane == 0) tma_store_wait0();
    if (warp == 4 && lane == 0) TRS(18);
  }
  teardown(tmem, warp);
}
#endif  // !UP && !PARS && !PAIR
#endif  // F16
