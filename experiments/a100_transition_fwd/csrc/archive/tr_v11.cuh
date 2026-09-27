// tr_fwd_sm80.cuh -- pair Transition forward, A100 / sm_80, one kernel:
//
//   out = x + Ws (silu(Wa xn) * (Wb xn)),  xn = LN(x),  D = 128, H = 512, bf16 in / out, fp32 accumulation.
//
// Warp = 32 token rows (2 m16 tiles) for the whole hidden dimension: LN(x) stays in registers as the GEMM1 A fragments (64 regs) and the
// output accumulator (32 x 128 fp32, 128 regs) stays in registers across all 512 hidden units, so the [M][512] activation never leaves
// the warp.  Per 16 hidden units: GEMM1 (a | b, 64 MMAs) -> SwiGLU in the C fragments, which ARE the GEMM2 A fragment (m16n8 C = m16k16 A)
// -> GEMM2 (32 MMAs).
//
// v3: one CTA of 8 warps per SM (256-row tile) sharing a 4-slot cp.async ring of 32-hidden weight chunks (24 KB each) -- half the L2 weight
// traffic of two 4-warp CTAs.  No CTA barrier: each warp waits on the slot's FULL mbarrier and arrives on its EMPTY mbarrier when done;
// at chunk u every warp copies its 1/8 of chunk u + 2 into the slot of chunk u - 2 (once all 8 warps retired it), so warps drift up to
// two chunks apart with no single refilling warp to wait for, and the two warps of an SM
// sub-partition fall out of phase (one's SwiGLU epilogue under the other's MMAs).  The tile count's remainder after the last full round
// runs as half tiles (16 rows per warp, first) so no round is a mostly-idle full one.  The residual's x re-read is issued right after
// the last GEMM1, into the A-fragment registers that just died.
//
// Host-side permutations (pack() in transition_a100.py) give thread q of a quad the four 16 B vectors i = 0..3 at columns 32 i + 8 q .. + 7
// of its rows, for x and for out alike (a quad's vector i is 64 contiguous bytes: full sectors for the store):
//   * GEMM1 k order: mma k-step s, index kk <-> logical column 32 (s / 2) + 8 (kk % 8 / 2) + 4 (s % 2) + 2 (kk / 8) + kk % 2, so the thread's
//     A fragments are its own x words (a straight 4 x 16 B read per row, no shuffle);
//   * GEMM2 n order: physical output column 8 J + 2 q + e <-> logical 32 (J / 4) + 8 q + 2 (J % 4) + e, so the accumulator of n8 tile J is
//     word J of the same 64 B the thread read for the LayerNorm: the accumulator starts at x (the residual) and the store is 4 x 16 B.
//   * Wa is pre-scaled by 1/2 (exact): silu(a) b = a' b (1 + tanh a'), one MUFU per hidden unit.
#pragma once
#include "sm80_common.cuh"

#include <cuda_fp16.h>

#include <type_traits>

namespace a100 {

struct TrParams {
  const __nv_bfloat16* x;      // [T][128]
  const __nv_bfloat16* w;      // packed [16 chunks][W1: 16 granules x 64 rows x 16 B | W2: 4 granules x 128 rows x 16 B]
  const float4* gb;            // LN affine as float4 slots [gamma | beta][i][q] = columns 32 q + 4 i .. 32 q + 4 i + 3
  __nv_bfloat16* out;          // [T][128]
  unsigned long long* trace;   // TR_TRACE builds: [warp][item][20] clock64 stamps
  int T, num_tiles;
  float eps;
};

#ifndef TANH16
#define TANH16 0
#endif
#ifndef TR_NWARP
#define TR_NWARP 8
#endif
struct TrCfg {
  // 8 warps: one CTA per SM, 4-slot ring; 4 warps: two independent CTAs per SM (the SM sub-partition's two warps belong to different
  // CTAs), 2-slot ring in lockstep
  static constexpr int D = 128, H = 512, NWARP = TR_NWARP, NTHR = 32 * NWARP, MINB = 8 / NWARP, BM = 32 * NWARP;
  static constexpr int CH = 32, NCHUNK = H / CH;            // hidden units per ring slot, chunks per tile
  static constexpr int SLOT_W1 = 2 * CH * D * 2;            // 16 KB: [16 k-granules][64 rows (a | b of 2 x 16 hidden)][16 B]
  static constexpr int SLOT = SLOT_W1 + CH * D * 2;         // + 8 KB: [4 hidden-granules][128 out rows][16 B]
#ifndef TR_NST
#define TR_NST (NWARP == 8 ? 4 : 2)
#endif
#ifndef TR_AHEAD
#define TR_AHEAD (NWARP == 8 ? 2 : 1)
#endif
  static constexpr int NST = TR_NST, AHEAD = TR_AHEAD;                   // ring slots; a warp at chunk u issues its share of chunk u + AHEAD
  static constexpr int XW = 32 * D * 2;                     // per-warp x staging, 8 KB
  static constexpr int SMEM_W = NST * SLOT, SMEM_X = NWARP * XW;
  static constexpr int NBAR = 2 * NST + NWARP;
  static constexpr int SMEM_GB = 2 * D * 4;                // LN affine, float4 slots [gamma | beta][i][q]
  static constexpr int MAX_ITEMS = 255, SMEM_SCHED = 4 * (MAX_ITEMS + 1);   // per-CTA work list, -1 terminated
  static constexpr int SMEM = SMEM_W + SMEM_X + NBAR * 8 + SMEM_GB + SMEM_SCHED;
  static_assert(MINB * (SMEM + 1024) <= 167936, "sm_80 shared memory");
};

// x staging row (256 B, 16 granules): granule G at G ^ ((r & 1) << 2).  A quarter-warp of the LN read is rows {r, r + 1} x quads
// q = 0..3 at granule 4 i + q: bits (G2 ^ r0, q1, q0) pick 8 distinct 16 B bank groups.
DEVI uint32_t swz_x(uint32_t r, uint32_t G) { return r * 256 + ((G ^ ((r & 1u) << 2)) << 4); }

// GEMM2 runs in f16 (h and Ws as f16, f16 accumulation): the same tensor rate on sm_80 as f32 accumulation and half the accumulator
// registers (64 instead of 128 for 32 x 128 outputs); the accumulator starts at x, so it also carries the residual.
DEVI void mma16816_h(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t pack_f16(float lo, float hi) {
  uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
DEVI float2 unpack_f16(uint32_t v) { return __half22float2(*reinterpret_cast<const __half2*>(&v)); }
DEVI uint32_t tanh_f16x2(uint32_t v) { uint32_t r; asm("tanh.approx.f16x2 %0, %1;\n" : "=r"(r) : "r"(v)); return r; }
DEVI uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
// SwiGLU of the element pair (a'0, b0), (a'1, b1) -> f16x2 h = a' b (1 + tanh a'): one MUFU for two elements (TANH16) or two
DEVI uint32_t swiglu2(float a0, float b0, float a1, float b1) {
#if TANH16
  const uint32_t pr = pack_f16(a0 * b0, a1 * b1);
  return hfma2(tanh_f16x2(pack_f16(a0, a1)), pr, pr);
#else
  const float p0 = a0 * b0, p1 = a1 * b1;
  return pack_f16(fmaf(tanh_approx(a0), p0, p0), fmaf(tanh_approx(a1), p1, p1));
#endif
}

// per-warp state that survives across tiles
struct TrWarp {
  uint32_t sW_u, xw, barX, sGB, w1_off, w2_off;
  const __nv_bfloat16* wsrc;   // this thread's first granule of the packed weights (p.w + 8 tid)
  int tid, lane, g8, q, u, total;
  unsigned long long* tr;       // this warp's trace row of the current item (TR_TRACE)
};
template <class G> DEVI uint32_t bar_full(const TrWarp& w, int s) { return w.sW_u + G::SMEM_W + G::SMEM_X + 8 * s; }
template <class G> DEVI uint32_t bar_empty(const TrWarp& w, int s) { return w.sW_u + G::SMEM_W + G::SMEM_X + 8 * (G::NST + s); }
// spin with a short back-off: a waiting warp should not take issue slots from the MMA warp of its sub-partition
DEVI void mbar_wait_bo(uint32_t bar, uint32_t parity) {
  if (mbar_test(bar, parity)) return;
#ifndef TR_SLEEP
#define TR_SLEEP 20
#endif
  while (!mbar_test(bar, parity)) { if (TR_SLEEP) __nanosleep(TR_SLEEP); }
}

DEVI void cp_async16_full(uint32_t dst, const void* src) {       // no src-size operand: the global address keeps its immediate offset
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
template <class G>
DEVI void issue_w(const TrParams& p, const TrWarp& w, int c, int s) {   // chunk c -> slot s: 1536 granules, 6 per thread of the CTA
  const __nv_bfloat16* src = w.wsrc + c * (G::SLOT / 2);
  const uint32_t dst = w.sW_u + s * G::SLOT + w.tid * 16;
#pragma unroll
  for (int i = 0; i < G::SLOT / 16 / G::NTHR; ++i) cp_async16_full(dst + i * G::NTHR * 16, src + i * G::NTHR * 8);
  cp_async_mbar_arrive(bar_full<G>(w, s));
}

template <class G>
DEVI void load_x(const TrParams& p, const TrWarp& w, int r0, int mt) {  // this warp's 16 mt rows from r0 -> its staging (zero-filled past T)
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    if (i < 8 * mt) {
      const int c = w.lane + 32 * i, r = c >> 4, gr = c & 15;
      const bool ok = r0 + r < p.T;
      cp_async16(w.xw + swz_x(r, gr), p.x + (size_t)(ok ? r0 + r : 0) * G::D + gr * 8, ok ? 16u : 0u);
    }
  }
  cp_async_mbar_arrive(w.barX);
}

// one work item: rows [r0, r0 + 16 MT) of this warp; the next item's rows are prefetched into the staging after the LayerNorm
#ifdef TR_TRACE
#define STAMP(k) do { if (w.lane == 0) w.tr[k] = clock64(); } while (0)
#else
#define STAMP(k) do { } while (0)
#endif
template <int MT, class G>
DEVI void tile(const TrParams& p, TrWarp& w, int r0, int xpar, int nr0, int nmt) {
  constexpr int D = G::D, NCHUNK = G::NCHUNK, NST = G::NST;
  const int g8 = w.g8, q = w.q;

  // ---- LayerNorm of rows (16 mt + 8 hr + g8) straight into the A fragments: word v = 2 s + e2 of the thread's 64 B -> fa[mt][s][hr + 2 e2].
  //      The same 64 B (as f16) initialise the output accumulator: word J = accumulator tile J, so the residual costs nothing at the end.
  uint32_t fa[MT][8][4];
  uint32_t acc2[MT][16][2];                                    // f16x2 [mt][n8 tile J][row g8 | g8 + 8]
  STAMP(0);
  mbar_wait_bo(w.barX, xpar);
  STAMP(1);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int r = 16 * mt + 8 * hr + g8;
      uint32_t wv[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 v = lds128(w.xw + swz_x(r, 4 * i + q));
        wv[4 * i] = v.x; wv[4 * i + 1] = v.y; wv[4 * i + 2] = v.z; wv[4 * i + 3] = v.w;
      }
      float xv[32];
#pragma unroll
      for (int e = 0; e < 16; ++e) { xv[2 * e] = bf16lo(wv[e]); xv[2 * e + 1] = bf16hi(wv[e]); }
#pragma unroll
      for (int e = 0; e < 16; ++e) acc2[mt][e][hr] = pack_f16(xv[2 * e], xv[2 * e + 1]);
      float sm = 0.f;
#pragma unroll
      for (int e = 0; e < 32; ++e) sm += xv[e];
      const float mean = quad_sum(sm) * (1.f / D);
      float sq = 0.f;
#pragma unroll
      for (int e = 0; e < 32; ++e) { xv[e] -= mean; sq = fmaf(xv[e], xv[e], sq); }
      const float rstd = rsqrtf(quad_sum(sq) * (1.f / D) + p.eps);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint4 gi = lds128(w.sGB + (i * 4 + q) * 16), bi = lds128(w.sGB + 512 + (i * 4 + q) * 16);
        const float4 gv = *reinterpret_cast<const float4*>(&gi), bv = *reinterpret_cast<const float4*>(&bi);
        fa[mt][i][hr] = pack_bf16(fmaf(xv[4 * i] * rstd, gv.x, bv.x), fmaf(xv[4 * i + 1] * rstd, gv.y, bv.y));
        fa[mt][i][hr + 2] = pack_bf16(fmaf(xv[4 * i + 2] * rstd, gv.z, bv.z), fmaf(xv[4 * i + 3] * rstd, gv.w, bv.w));
      }
    }
  __syncwarp();
  STAMP(2);
  if (nmt > 0) load_x<G>(p, w, nr0, nmt);                     // the staging is free: the next item's rows land under this one

#pragma unroll 1
  for (int c = 0; c < NCHUNK; ++c) {
    const int u = w.u, slot = u % NST;
    STAMP(21 + c);
    // this warp's share of chunk u + AHEAD, into the slot of chunk u + AHEAD - NST once every warp retired it.  A slow warp elsewhere
    // must not stall this one: if the slot is not free yet, the copy is issued at the end of the chunk instead (still a chunk early).
    constexpr int AH = G::AHEAD;
    const int s1 = (u + AH) % NST;
    const uint32_t par1 = ((u + AH) / NST - 1) & 1;
    bool pending = u + AH < w.total;
    if (pending && (u + AH < NST || mbar_test(bar_empty<G>(w, s1), par1))) {
      issue_w<G>(p, w, (u + AH) % NCHUNK, s1);
      pending = false;
    }
    mbar_wait_bo(bar_full<G>(w, slot), (u / NST) & 1);
    STAMP(3 + c);
    const uint32_t wb = w.sW_u + slot * G::SLOT;
#pragma unroll
    for (int ps = 0; ps < 2; ++ps) {
      // GEMM1: [a | b] of 16 hidden units, B fragments one k-step ahead of their MMAs
      float acc1[MT][4][4];                                    // [mt][a lo, a hi, b lo, b hi][4]
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc1[mt][n][e] = 0.f;
      uint32_t ba[2][4], bb[2][4];
      ldsm_x4(ba[0], wb + w.w1_off + ps * 512);
      ldsm_x4(bb[0], wb + w.w1_off + ps * 512 + 256);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int cb = s & 1;
        if (s < 7) {
          ldsm_x4(ba[cb ^ 1], wb + w.w1_off + ps * 512 + (s + 1) * 2048);
          ldsm_x4(bb[cb ^ 1], wb + w.w1_off + ps * 512 + 256 + (s + 1) * 2048);
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816(acc1[mt][0], fa[mt][s], ba[cb][0], ba[cb][1]);
          mma16816(acc1[mt][1], fa[mt][s], ba[cb][2], ba[cb][3]);
          mma16816(acc1[mt][2], fa[mt][s], bb[cb][0], bb[cb][1]);
          mma16816(acc1[mt][3], fa[mt][s], bb[cb][2], bb[cb][3]);
        }
      }
#ifndef G2_AHEAD
#define G2_AHEAD 1
#endif
      constexpr int NB = G2_AHEAD + 1;                         // GEMM2 B fragments G2_AHEAD j-blocks ahead of their MMAs
      uint32_t bs[NB][4];
#pragma unroll
      for (int j = 0; j < G2_AHEAD; ++j) ldsm_x4(bs[j], wb + w.w2_off + ps * 4096 + j * 256);   // under the epilogue
      // SwiGLU in the C fragments -> the GEMM2 A fragment (f16)
      uint32_t ha[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        // n8 tiles n (hidden 8 n + 2 q ..): (c0, c1) row g8 -> A-fragment word 2 n, (c2, c3) row g8 + 8 -> word 2 n + 1
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          ha[mt][2 * n] = swiglu2(acc1[mt][n][0], acc1[mt][2 + n][0], acc1[mt][n][1], acc1[mt][2 + n][1]);
          ha[mt][2 * n + 1] = swiglu2(acc1[mt][n][2], acc1[mt][2 + n][2], acc1[mt][n][3], acc1[mt][2 + n][3]);
        }
      }
      // GEMM2: acc2 += h Ws^T over these 16 hidden units (f16 accumulation)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int cb = j % NB;
        if (j + G2_AHEAD < 8) ldsm_x4(bs[(j + G2_AHEAD) % NB], wb + w.w2_off + ps * 4096 + (j + G2_AHEAD) * 256);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816_h(acc2[mt][2 * j], ha[mt], bs[cb][0], bs[cb][1]);
          mma16816_h(acc2[mt][2 * j + 1], ha[mt], bs[cb][2], bs[cb][3]);
        }
      }
    }
    __syncwarp();
    if (w.lane == 0) mbar_arrive(bar_empty<G>(w, slot));      // this warp retired the slot
    if (pending) {
      mbar_wait_bo(bar_empty<G>(w, s1), par1);
      issue_w<G>(p, w, (u + AH) % NCHUNK, s1);
    }
    ++w.u;
  }

  STAMP(19);
  // ---- store: word J of the thread's 64 B = accumulator tile J (x already in it)
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int r = r0 + 16 * mt + 8 * hr + g8;
      if (r < p.T) {
        __nv_bfloat16* dst = p.out + (size_t)r * D + 8 * q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          uint32_t o[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float2 v = unpack_f16(acc2[mt][4 * i + k][hr]);
            o[k] = pack_bf16(v.x, v.y);
          }
          stg128(dst + 32 * i, make_uint4(o[0], o[1], o[2], o[3]));    // a quad writes 64 contiguous bytes
        }
      }
    }
  STAMP(20);
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) tr_fwd_kernel(const TrParams p) {
  constexpr int NCHUNK = G::NCHUNK, BM = G::BM;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sX = smem + G::SMEM_W;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sX + G::SMEM_X);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int b = blockIdx.x, grid = gridDim.x;

  // schedule: at most one half tile (16 rows per warp) of the remainder FIRST -- the kernel's opening x fetch, when every SM reads DRAM
  // at once, is then half as large -- then full tiles b, b + grid, ...
  const int f = p.num_tiles / grid, rem = p.num_tiles - f * grid;
  const int rem_rows = p.T - f * grid * BM, nh = (rem_rows + BM / 2 - 1) / (BM / 2);
  const bool half_mode = rem > 0 && nh <= grid;
  const int n_full = half_mode ? f : f + (b < rem ? 1 : 0);
  const int n_half = (half_mode && b < nh) ? 1 : 0;
  const int n_items = n_full + n_half;
  // work list entry: (first row of the item's CTA tile) * 4 + m16 tiles per warp; a warp's rows start 16 mt warp further
  int* sched = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(bars) + 8 * G::NBAR + G::SMEM_GB);
  for (int it = tid; it <= G::MAX_ITEMS; it += G::NTHR) {
    int e = -1;
    if (it < n_items && it < G::MAX_ITEMS)
      e = it < n_half ? ((f * grid * BM + b * (BM / 2)) << 2) | 1 : (((b + (it - n_half) * grid) * BM) << 2) | 2;
    sched[it] = e;
  }

  TrWarp w;
  w.sW_u = smem_u32(smem);
  w.barX = smem_u32(bars) + 8 * (2 * G::NST + warp);
  w.sGB = smem_u32(bars) + 8 * G::NBAR;
  w.xw = smem_u32(sX) + warp * G::XW;
  w.wsrc = p.w + tid * 8;
  w.tid = tid; w.lane = lane; w.g8 = lane >> 2; w.q = lane & 3;
  w.u = 0; w.total = n_items * NCHUNK;
  // ldmatrix lane addresses: matrix mi = lane / 8 -> (row half mi >> 1, granule half mi & 1)
  const int lrow = ((lane >> 4) << 3) + (lane & 7), lgr = (lane >> 3) & 1;
  w.w1_off = lgr * 1024 + lrow * 16;                            // + p * 512 (step rows) + 256 (b rows) + s * 2048 (k-step)
  w.w2_off = G::SLOT_W1 + lgr * 2048 + lrow * 16;               // + p * 4096 (step granules) + j * 256 (out rows)

  if (tid == 0) {
    for (int s = 0; s < G::NST; ++s) { mbar_init(bar_full<G>(w, s), G::NTHR); mbar_init(bar_empty<G>(w, s), G::NWARP); }
    for (int k = 0; k < G::NWARP; ++k) mbar_init(smem_u32(bars) + 8 * (2 * G::NST + k), 32);
  }
  for (int k = tid; k < 64; k += G::NTHR) reinterpret_cast<float4*>(reinterpret_cast<uint8_t*>(bars) + 8 * G::NBAR)[k] = p.gb[k];
  __syncthreads();
  if (n_items == 0) return;
  int e = sched[0];
  const int mt0 = e & 3;
  load_x<G>(p, w, (e >> 2) + 16 * mt0 * warp, mt0);
  for (int c = 0; c < G::AHEAD && c < w.total; ++c) issue_w<G>(p, w, c % NCHUNK, c);
#pragma unroll 1
  for (int it = 0; e >= 0; ++it) {
    const int en = sched[it + 1];
    const int mt = e & 3, nmt = en >= 0 ? en & 3 : 0;
    const int r0 = (e >> 2) + 16 * mt * warp, nr0 = (en >> 2) + 16 * nmt * warp;
#ifdef TR_TRACE
    w.tr = p.trace + ((size_t)(b * G::NWARP + warp) * 32 + (it & 31)) * 40;
#endif
    if (mt == 2) tile<2, G>(p, w, r0, it & 1, nr0, nmt);
    else tile<1, G>(p, w, r0, it & 1, nr0, nmt);
    e = en;
  }
  cp_async_wait<0>();
}

}  // namespace a100
