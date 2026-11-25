// fused_tf32.cu — the fp32 (TF32) path's inference row stages fused around the attention (sm_100a, tcgen05 kind::tf32): two kernels in place
// of the eight launches (3 AdaLNs, the gate, 4 GEMMs) atom_gemm_tf32 / rows_tf32 take, with the same arithmetic.
//
//   atom_pre_tf32   x1 = LN(s) s1 + b1  (cross mode: xkv = LN(x1) mks + mkb)  ->  P = [x1 | x1 | xkv | xkv] . Wp^T  (+ bq on q),
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

struct Bars {
  uint64_t wfull[NST], wempty[NST], afull, hfull, accf;
  uint32_t tmem;
};

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
    const float mean = a[k] * (1.f / 128);
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
DEVI uint32_t setup(Bars& B, int warp) {
  if (threadIdx.x == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.wfull[s], 1); mbar_init(&B.wempty[s], 1); }
    mbar_init(&B.afull, 8); mbar_init(&B.hfull, 8); mbar_init(&B.accf, 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  pdl_wait();                                                        // the previous kernel's outputs are complete and visible
  pdl_launch();
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
  DEVI void run(Bars& B, uint32_t d, uint32_t abase, uint32_t idesc, bool fresh) {
    for (int k = 0; k < 4; ++k, ++gi) {
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
  if (gi >= NST) mbar_wait(&B.wempty[slot], ((gi / NST) - 1) & 1);
  mbar_expect_tx(&B.wfull[slot], bytes);
  tma_load_2d(su + O_W + (uint32_t)slot * SLOT, m, &B.wfull[slot], c0, c1);
}
// post: unit u's (first row, row count), by value (by-reference outputs went through the stack: ptxas, round 7)
DEVI int2 unit_rows(int u, int nfull) { return u < nfull ? make_int2(128 * u, 128) : make_int2(128 * nfull + 64 * (u - nfull), 64); }
DEVI int units_of(int units) {                                       // this CTA's share of the persistent unit loop
  return (int)blockIdx.x < units ? (units - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
}
DEVI int post_units(int M, int nfull) { return units_of(nfull + 2 * ((M >> 7) - nfull)); }

// ==================================================================================================================== pre
// s [M, 128] (row stride 128), s1 / b1 rows mld apart, mks / mkb (cross) rows kld apart, bq [128]; mwp: Wp [512, 128] (box 32 x 256);
// mp: P [M, 512] (box 32 x 32). rndmask: the 128-column blocks of P stored tf32-rounded. nfull: see the header.
extern "C" __global__ void __maxnreg__(128)
atom_pre_tf32(const __grid_constant__ CUtensorMap mwp, const __grid_constant__ CUtensorMap mp, const float* __restrict__ s,
              const float* __restrict__ s1, const float* __restrict__ b1, int mld, const float* __restrict__ mks, const float* __restrict__ mkb,
              int kld, const float* __restrict__ bq, int cross, int rndmask, int M, float eps, int nfull) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int units = nfull + 2 * ((M >> 7) - nfull);
  const int my = units_of(units);
  const uint32_t sa = su + O_A, sh = su + O_H;
  const uint32_t tmem = setup(B, warp);
  auto unit = [&](int i) { return (int)blockIdx.x + i * (int)gridDim.x; };
  // a unit's column half (a half unit), or -1 (a full unit: all 512 columns); its 128-row tile
  auto half_of = [&](int u) { return u < nfull ? -1 : ((u - nfull) & 1); };
  auto tile_of = [&](int u) { return u < nfull ? u : nfull + ((u - nfull) >> 1); };

  if (warp == 0) {
    if (lane == 0) {
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
    for (int i = 0; i < my && i < 2; ++i) {                          // the first two units' rows into L2
      const int m0 = tile_of(unit(i)) * 128, cb = half_of(unit(i));
      pf_rows(s, 128, m0, 128, lane); pf_rows(s1, mld, m0, 128, lane); pf_rows(b1, mld, m0, 128, lane);
      if (cross && cb != 0) { pf_rows(mks, kld, m0, 128, lane); pf_rows(mkb, kld, m0, 128, lane); }
    }
  } else if (warp >= 4) {
    const int e = warp - 4, qd = warp & 3, wg = e >> 2;
    const uint32_t tl = tmem + ((uint32_t)(32 * qd) << 16);
    const uint32_t sbase = sa + (uint32_t)e * 2 * EPB;               // staging inside tile A (free once the MMAs have read it)
    int nst = 0;
    for (int i = 0; i < my; ++i) {
      const int cb = half_of(unit(i));
      const int m0 = tile_of(unit(i)) * 128;
      const bool x1_to_a = cb != 1 || !cross;                       // tile A holds x1, or (cross, k | v half) xkv
      const bool want_kv = cross && cb != 0;
      const uint32_t kvt = cb < 0 ? sh : sa;
      // ---------------------------------------------------------------- warp per row: AdaLN 1 (and the K / V AdaLN) -> operand tiles
#pragma unroll 1
      for (int k0 = 0; k0 < 16; k0 += 4) {                           // 4 rows per warp in flight
        float x[4][4];
        {
          float4 sc[4], sf[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const size_t row = (size_t)(m0 + e + 8 * (k0 + k));
            const float4 sv = ldg4(s + row * 128 + 4 * lane);
            x[k][0] = sv.x; x[k][1] = sv.y; x[k][2] = sv.z; x[k][3] = sv.w;
            sc[k] = ldg4(s1 + row * mld + 4 * lane);
            sf[k] = ldg4(b1 + row * mld + 4 * lane);
          }
          ln4x4(x, eps);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            x[k][0] = fmaf(x[k][0], sc[k].x, sf[k].x); x[k][1] = fmaf(x[k][1], sc[k].y, sf[k].y);
            x[k][2] = fmaf(x[k][2], sc[k].z, sf[k].z); x[k][3] = fmaf(x[k][3], sc[k].w, sf[k].w);
            if (x1_to_a) sts128(opnd(sa, e + 8 * (k0 + k), lane), u4(rna(x[k][0]), rna(x[k][1]), rna(x[k][2]), rna(x[k][3])));
          }
        }
        if (want_kv) {                                               // xkv = LN(x1) mks + mkb (x1 exact)
          float4 kc[4], kf[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const size_t row = (size_t)(m0 + e + 8 * (k0 + k));
            kc[k] = ldg4(mks + row * kld + 4 * lane);
            kf[k] = ldg4(mkb + row * kld + 4 * lane);
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
      for (int cc = wg; cc < nch; cc += 2) {
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
// o [M, 128] (row stride 128), g rows gld apart, s [M, 128], so / s2 / b2 / st rows mld apart; mwo: Wo [128, 128] (box 32 x 128), mwu: the
// interleaved [Wa_0; Wb_0; Wa_1; Wb_1] [512, 128] (box 32 x 256), mws: Ws [128, 256] (box 32 x 128). out [M, 128]. nfull: see the header.
extern "C" __global__ void __maxnreg__(168)
atom_post_tf32(const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwu, const __grid_constant__ CUtensorMap mws,
               const float* __restrict__ o, const float* __restrict__ g, int gld, const float* __restrict__ s, const float* __restrict__ so,
               const float* __restrict__ s2, const float* __restrict__ b2, const float* __restrict__ st, int mld, float* __restrict__ out,
               int M, float eps, int nfull) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // the CTA's unit count is recomputed at the top of each warp role (held across the role switch it lived on the stack: ptxas, round 7);
  // a unit's first row and row count (unit_rows): full units are 128-row tiles, then 64-row halves of the remaining tiles
  const uint32_t sa = su + O_A, sh = su + O_H;
  const uint32_t tmem = setup(B, warp);

  if (warp == 0) {
    if (lane == 0) {
      const int my = post_units(M, nfull);
      int gi = 0;
      for (int i = 0; i < my; ++i) {
        for (int k = 0; k < 4; ++k, ++gi) wload(B, su, gi, &mwo, 32 * k, 0, 16384u);           // Wo
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
      mbar_wait(&B.afull, (na++) & 1);                               // gated
      tc_fence_after();
      mm.run(B, tmem, sa, i128, true);                               // y = gated Wo^T
      commit_acc(B);
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
      commit_acc(B);
    }
  } else if (warp == 3) {
    const int my = post_units(M, nfull);
    for (int i = 0; i < my && i < 2; ++i) {                          // the first two units' rows into L2
      const int2 ur = unit_rows((int)blockIdx.x + i * (int)gridDim.x, nfull);
      const int m0 = ur.x, rt = ur.y;
      pf_rows(o, 128, m0, rt, lane); pf_rows(g, gld, m0, rt, lane); pf_rows(s, 128, m0, rt, lane); pf_rows(so, mld, m0, rt, lane);
      pf_rows(s2, mld, m0, rt, lane); pf_rows(b2, mld, m0, rt, lane); pf_rows(st, mld, m0, rt, lane);
    }
  } else if (warp >= 4) {
    const int e = warp - 4, qd = warp & 3, wg = e >> 2;
    const int r = 32 * qd + lane;                                    // lane-per-row phases: this thread's unit row
    const uint32_t tl = tmem + ((uint32_t)(32 * qd) << 16);
    int nc = 0;
    const int my = post_units(M, nfull);
    for (int i = 0; i < my; ++i) {
      const int2 ur = unit_rows((int)blockIdx.x + i * (int)gridDim.x, nfull);
      const int m0 = ur.x, rt = ur.y;
      const bool act = 32 * qd < rt;                                 // this warp's TMEM lane quadrant holds unit rows
      const int nr = rt >> 3;                                        // warp-per-row phases: rows per warp
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
            gv[k] = ldg4(so + row * mld + 4 * lane);
          }
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float4 y = lds4f(opnd(sa, e + 8 * (k0 + k), lane));
            x[k][0] = fmaf(gv[k].x, y.x, sv[k].x); x[k][1] = fmaf(gv[k].y, y.y, sv[k].y);
            x[k][2] = fmaf(gv[k].z, y.z, sv[k].z); x[k][3] = fmaf(gv[k].w, y.w, sv[k].w);
          }
        }
        ln4x4(x, eps);
#pragma unroll
        for (int h2 = 0; h2 < 4; h2 += 2) {                          // s2, b2 two rows at a time (the phase's register peak: 16 + 16 values)
          float4 sc[2], sf[2];
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const size_t row = (size_t)(m0 + e + 8 * (k0 + h2 + k));
            sc[k] = ldg4(s2 + row * mld + 4 * lane);
            sf[k] = ldg4(b2 + row * mld + 4 * lane);
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
      // ---------------------------------------------------------------- lane per row: y -> tile A, t -> tile H (every MMA of the unit is done)
      mbar_wait(&B.accf, (nc++) & 1);
      tc_fence_after();
      if (act) {
        for (int c = 2 * wg; c < 2 * wg + 2; ++c) {
          float v[32];
          tld(tl + 32 * c, v);
          put_row(sa, r, c, v);
          tld(tl + 384 + 32 * c, v);
          put_row(sh, r, c, v);
        }
      }
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
            gv[k] = ldg4(so + row * mld + 4 * lane);
          }
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float4 y = lds4f(opnd(sa, e + 8 * (k0 + k), lane));
            a1[k] = make_float4(fmaf(gv[k].x, y.x, sv[k].x), fmaf(gv[k].y, y.y, sv[k].y), fmaf(gv[k].z, y.z, sv[k].z), fmaf(gv[k].w, y.w, sv[k].w));
          }
        }
        float4 tv[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) tv[k] = ldg4(st + (size_t)(m0 + e + 8 * (k0 + k)) * mld + 4 * lane);
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
  teardown(tmem, warp);
}
