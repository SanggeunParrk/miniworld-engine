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
// runs as half tiles (16 rows per warp) so the last round is not a mostly-idle full one.  The residual's x re-read is issued right after
// the last GEMM1, into the A-fragment registers that just died.
//
// Host-side permutations (pack() in transition_a100.py) make every thread's x / out footprint 32 CONTIGUOUS columns of its rows:
//   * GEMM1 k order: mma k-step s, index kk <-> logical column 32 (kk % 8 / 2) + 4 s + 2 (kk / 8) + kk % 2, so thread q of a quad holds
//     columns [32 q, 32 q + 32) of rows g, g + 8 in its A fragments (a straight 4 x 16 B read per row, no shuffle);
//   * GEMM2 n order: physical output column 8 J + 2 q + e <-> logical 32 q + 2 J + e, so the accumulator of n8 tile J is word J of the same
//     64 B the thread read for the LayerNorm: the residual is a word-for-word add and the store is 4 x 16 B per row.
//   * Wa is pre-scaled by 1/2 (exact): silu(a) b = a' b (1 + tanh a'), one MUFU per hidden unit.
#pragma once
#include "sm80_common.cuh"

#include <type_traits>

namespace a100 {

struct TrParams {
  const __nv_bfloat16* x;      // [T][128]
  const __nv_bfloat16* w;      // packed [16 chunks][W1: 16 granules x 64 rows x 16 B | W2: 4 granules x 128 rows x 16 B]
  const float4* gb;            // LN affine as float4 slots [gamma | beta][i][q] = columns 32 q + 4 i .. 32 q + 4 i + 3
  __nv_bfloat16* out;          // [T][128]
  int T, num_tiles;
  float eps;
};

struct TrCfg {
  static constexpr int D = 128, H = 512, NWARP = 8, NTHR = 32 * NWARP, MINB = 1, BM = 32 * NWARP;
  static constexpr int CH = 32, NCHUNK = H / CH;            // hidden units per ring slot, chunks per tile
  static constexpr int SLOT_W1 = 2 * CH * D * 2;            // 16 KB: [16 k-granules][64 rows (a | b of 2 x 16 hidden)][16 B]
  static constexpr int SLOT = SLOT_W1 + CH * D * 2;         // + 8 KB: [4 hidden-granules][128 out rows][16 B]
  static constexpr int NST = 4, AHEAD = 2;                   // ring slots; a warp at chunk u issues its share of chunk u + AHEAD
  static constexpr int XW = 32 * D * 2;                     // per-warp x staging, 8 KB
  static constexpr int SMEM_W = NST * SLOT, SMEM_X = NWARP * XW;
  static constexpr int NBAR = 2 * NST + NWARP;
  static constexpr int SMEM = SMEM_W + SMEM_X + NBAR * 8;
  static_assert(MINB * (SMEM + 1024) <= 167936, "sm_80 shared memory");
};

// x staging row (256 B, 16 granules): granule G at G ^ (r & 1) ^ ((G >> 3) << 1).  A quarter-warp of the LN / residual read is rows
// {r, r + 1} x quads q = 0..3 at granule 4 q + i: bits (r0, G3, G2) = (r0, q1, q0) pick 8 distinct 16 B bank groups.
DEVI uint32_t swz_x(uint32_t r, uint32_t G) { return r * 256 + ((G ^ (r & 1u) ^ ((G >> 3) << 1)) << 4); }

// per-warp state that survives across tiles
struct TrWarp {
  uint32_t sW_u, xw, barX, w1_off, w2_off;
  int tid, lane, g8, q, u, total;
};
template <class G> DEVI uint32_t bar_full(const TrWarp& w, int s) { return w.sW_u + G::SMEM_W + G::SMEM_X + 8 * s; }
template <class G> DEVI uint32_t bar_empty(const TrWarp& w, int s) { return w.sW_u + G::SMEM_W + G::SMEM_X + 8 * (G::NST + s); }
// spin with a short back-off: a waiting warp should not take issue slots from the MMA warp of its sub-partition
DEVI void mbar_wait_bo(uint32_t bar, uint32_t parity) {
  if (mbar_test(bar, parity)) return;
  while (!mbar_test(bar, parity)) __nanosleep(20);
}

template <class G>
DEVI void issue_w(const TrParams& p, const TrWarp& w, int c, int s) {   // chunk c -> slot s: 1536 granules, 6 per thread of the CTA
  const __nv_bfloat16* src = p.w + (size_t)c * (G::SLOT / 2) + w.tid * 8;
  const uint32_t dst = w.sW_u + s * G::SLOT + w.tid * 16;
#pragma unroll
  for (int i = 0; i < G::SLOT / 16 / G::NTHR; ++i) cp_async16(dst + i * G::NTHR * 16, src + i * G::NTHR * 8);
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
template <int MT, class G>
DEVI void tile(const TrParams& p, TrWarp& w, int r0, int xpar, int nr0, int nmt) {
  constexpr int D = G::D, NCHUNK = G::NCHUNK, NST = G::NST;
  const int g8 = w.g8, q = w.q;

  // ---- LayerNorm of rows (16 mt + 8 hr + g8) straight into the A fragments: word v = 2 s + e2 of the thread's 64 B -> fa[mt][s][hr + 2 e2]
  uint32_t fa[MT][8][4];
  mbar_wait_bo(w.barX, xpar);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int r = 16 * mt + 8 * hr + g8;
      uint32_t wv[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 v = lds128(w.xw + swz_x(r, 4 * q + i));
        wv[4 * i] = v.x; wv[4 * i + 1] = v.y; wv[4 * i + 2] = v.z; wv[4 * i + 3] = v.w;
      }
      float xv[32];
#pragma unroll
      for (int e = 0; e < 16; ++e) { xv[2 * e] = bf16lo(wv[e]); xv[2 * e + 1] = bf16hi(wv[e]); }
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
        const float4 gv = __ldg(p.gb + i * 4 + q), bv = __ldg(p.gb + 32 + i * 4 + q);
        fa[mt][i][hr] = pack_bf16(fmaf(xv[4 * i] * rstd, gv.x, bv.x), fmaf(xv[4 * i + 1] * rstd, gv.y, bv.y));
        fa[mt][i][hr + 2] = pack_bf16(fmaf(xv[4 * i + 2] * rstd, gv.z, bv.z), fmaf(xv[4 * i + 3] * rstd, gv.w, bv.w));
      }
    }
  __syncwarp();
  if (nmt > 0) load_x<G>(p, w, nr0, nmt);                     // the staging is free: the next item's rows land under this one

  float acc2[MT][16][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc2[mt][j][e] = 0.f;
  uint4 xr[MT][2][4];                                          // residual x, loaded under the last chunk's GEMM2

  auto chunk = [&](auto last_c) {
    constexpr bool LAST = decltype(last_c)::value;
    const int u = w.u, slot = u % NST;
    // this warp's share of chunk u + AHEAD, into the slot of chunk u + AHEAD - NST once every warp retired it
    constexpr int AH = G::AHEAD;
    if (u + AH < w.total) {
      const int s1 = (u + AH) % NST;
      if (u + AH >= NST) mbar_wait_bo(bar_empty<G>(w, s1), ((u + AH) / NST - 1) & 1);
      issue_w<G>(p, w, (u + AH) % NCHUNK, s1);
    }
    mbar_wait_bo(bar_full<G>(w, slot), (u / NST) & 1);
    const uint32_t wb = w.sW_u + slot * G::SLOT;
#pragma unroll
    for (int ps = 0; ps < 2; ++ps) {
      float acc1[MT][4][4];                                    // [mt][a lo, a hi, b lo, b hi][4]
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc1[mt][n][e] = 0.f;
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t ba[4], bb[4];
        ldsm_x4(ba, wb + w.w1_off + ps * 512 + s * 2048);
        ldsm_x4(bb, wb + w.w1_off + ps * 512 + 256 + s * 2048);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816(acc1[mt][0], fa[mt][s], ba[0], ba[1]);
          mma16816(acc1[mt][1], fa[mt][s], ba[2], ba[3]);
          mma16816(acc1[mt][2], fa[mt][s], bb[0], bb[1]);
          mma16816(acc1[mt][3], fa[mt][s], bb[2], bb[3]);
        }
      }
      if constexpr (LAST) {
        if (ps == 1) {                                         // fa is dead: fetch the residual rows into its registers
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
              const int r = r0 + 16 * mt + 8 * hr + g8;
              const uint4* src = reinterpret_cast<const uint4*>(p.x + (size_t)(r < p.T ? r : 0) * D + 32 * q);
#pragma unroll
              for (int i = 0; i < 4; ++i) xr[mt][hr][i] = __ldg(src + i);
            }
        }
      }
      uint32_t ha[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        float h[2][4];
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float av = acc1[mt][n][e], pr = av * acc1[mt][2 + n][e];
            h[n][e] = fmaf(tanh_approx(av), pr, pr);
          }
        ha[mt][0] = pack_bf16(h[0][0], h[0][1]);
        ha[mt][1] = pack_bf16(h[0][2], h[0][3]);
        ha[mt][2] = pack_bf16(h[1][0], h[1][1]);
        ha[mt][3] = pack_bf16(h[1][2], h[1][3]);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint32_t bs[4];
        ldsm_x4(bs, wb + w.w2_off + ps * 4096 + j * 256);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816(acc2[mt][2 * j], ha[mt], bs[0], bs[1]);
          mma16816(acc2[mt][2 * j + 1], ha[mt], bs[2], bs[3]);
        }
      }
    }
    __syncwarp();
    if (w.lane == 0) mbar_arrive(bar_empty<G>(w, slot));          // this warp retired the slot
    ++w.u;
  };
#pragma unroll 1
  for (int c = 0; c < NCHUNK - 1; ++c) chunk(std::false_type{});
  chunk(std::true_type{});

  // ---- residual + store: word J of the thread's 64 B = accumulator tile J
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int r = r0 + 16 * mt + 8 * hr + g8;
      if (r < p.T) {
        __nv_bfloat16* dst = p.out + (size_t)r * D + 32 * q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t xw4[4] = {xr[mt][hr][i].x, xr[mt][hr][i].y, xr[mt][hr][i].z, xr[mt][hr][i].w};
          uint32_t o[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int J = 4 * i + k;
            o[k] = pack_bf16(acc2[mt][J][2 * hr] + bf16lo(xw4[k]), acc2[mt][J][2 * hr + 1] + bf16hi(xw4[k]));
          }
          stg128(dst + 8 * i, make_uint4(o[0], o[1], o[2], o[3]));
        }
      }
    }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) tr_fwd_kernel(const TrParams p) {
  constexpr int NCHUNK = G::NCHUNK, BM = G::BM;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sX = smem + G::SMEM_W;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sX + G::SMEM_X);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int b = blockIdx.x, grid = gridDim.x;

  // schedule: full tiles b, b + grid, ... then at most one half tile (16 rows per warp) of the remainder
  const int f = p.num_tiles / grid, rem = p.num_tiles - f * grid;
  const int rem_rows = p.T - f * grid * BM, nh = (rem_rows + BM / 2 - 1) / (BM / 2);
  const bool half_mode = rem > 0 && nh <= grid;
  const int n_full = half_mode ? f : f + (b < rem ? 1 : 0);
  const bool has_half = half_mode && b < nh;
  const int n_items = n_full + (has_half ? 1 : 0);
  auto item = [&](int it, int& r0, int& mt) {                  // this warp's first row and m16-tile count of item it (mt 0: none)
    if (it >= n_items) { r0 = 0; mt = 0; }
    else if (it < n_full) { r0 = (b + it * grid) * BM + 32 * warp; mt = 2; }
    else { r0 = f * grid * BM + b * (BM / 2) + 16 * warp; mt = 1; }
  };

  TrWarp w;
  w.sW_u = smem_u32(smem);
  w.barX = smem_u32(bars) + 8 * (2 * G::NST + warp);
  w.xw = smem_u32(sX) + warp * G::XW;
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
  __syncthreads();
  if (n_items == 0) return;
  {
    int r0, mt;
    item(0, r0, mt);
    load_x<G>(p, w, r0, mt);
    for (int c = 0; c < G::AHEAD && c < w.total; ++c) issue_w<G>(p, w, c % NCHUNK, c);
  }
#pragma unroll 1
  for (int it = 0; it < n_items; ++it) {
    int r0, mt, nr0, nmt;
    item(it, r0, mt);
    item(it + 1, nr0, nmt);
    if (mt == 2) tile<2, G>(p, w, r0, it & 1, nr0, nmt);
    else tile<1, G>(p, w, r0, it & 1, nr0, nmt);
  }
  cp_async_wait<0>();
}

}  // namespace a100
