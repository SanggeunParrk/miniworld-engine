// tr_bwd_sm80.cuh -- pair Transition backward, A100 / sm_80 (D = 128, H = 512, bf16):
//
//   xn = LN(x)   a = xn Wa^T   b = xn Wb^T   dh = dy Ws        (recomputed here: nothing but x is saved by the forward)
//   silu = a sig(a)   h = silu b   dA = dh b (sig + silu (1 - sig))   dB = dh silu        (h, dA, dB rounded to bf16)
//   dWs = dy^T h   dWa = dA^T xn   dWb = dB^T xn   d_xn = dA Wa + dB Wb
//   dx = (gamma d_xn - xhat ca - cb) rstd + dy,  ca = mean(gamma d_xn xhat), cb = mean(gamma d_xn);  dgamma = sum d_xn xhat, dbeta = sum d_xn
//
// Three roles, 16 M D H FLOP in total (every product computed once):
//   P  x, dy -> (mean, rstd), and dA, dB, h in FRAGMENT-NATIVE blocks: block (R, K, m) = rows 16 R .., hidden 16 K .., of matrix m,
//      512 B = 32 lanes x the lane's m16k16 A-fragment words, so a consumer loads its A fragment (or, through movmatrix.trans, the
//      transposed one) with one 16 B load per lane and no shared memory.                                            (6 M D H)
//   X  d_xn = [dA | dB] [Wa; Wb] with f32 accumulation (A fragments straight from the blocks, B = [Wa; Wb] streamed through the ring and
//      read with ldmatrix.trans), then the LayerNorm backward + residual in the accumulator layout -> dx; dgamma / dbeta partials.  (4 M D H)
//   W  weight gradients (tr_bwd_w_sm80.cuh).                                                                          (6 M D H)
// P and X share the forward's skeleton: 8 warps x 32 rows per CTA, weights through a cp.async ring of 32-hidden chunks with per-slot
// FULL / EMPTY mbarriers and no CTA barrier, the remainder as half tiles first, and the forward's host permutations (thread q of a quad
// owns the 16 B vectors at columns 32 i + 8 q of its rows, for x, dy, xn and dx alike).
#pragma once
#include "tr_fwd_sm80.cuh"

namespace a100 {

struct BwdParams {
  const __nv_bfloat16* x;      // [T][128]
  const __nv_bfloat16* dy;     // [T][128]
  const __nv_bfloat16* wp;     // P weights: [16 chunks][W1: 16 k-granules x 64 rows (0.5 Wa | Wb) | W3: 16 k-granules x 32 rows (Ws^T)] x 16 B
  const __nv_bfloat16* wx;     // X weights: [16 chunks][Wa: 16 n-granules x 32 hidden | Wb: same] x 16 B, n in the output permutation
  const float4* gb;            // LN affine, the forward's float4 slots
  const float* gamma;          // [128] plain
  __nv_bfloat16* xn;           // P out: [T][128]
  float2* stats;               // P out: [T] (mean, rstd)
  uint4* ab;                   // P out: [T / 16][32 K][dA | dB | h][32 lanes] 16 B fragment-native blocks
  __nv_bfloat16* dx;           // X out: [T][128]
  float* dgb;                  // X out: [grid][2][128] (dgamma, dbeta) per CTA
  int T, num_tiles;
  float eps;
};

// ---------------------------------------------------------------------------------------------------------------- ring + schedule
template <int SLOT_, int NST_, int AHEAD_>
struct BwdCfg {
  static constexpr int D = 128, H = 512, NWARP = 8, NTHR = 256, BM = 256, CH = 32, NCHUNK = 16;
  static constexpr int SLOT = SLOT_, NST = NST_, AHEAD = AHEAD_;
  static constexpr int SMEM_W = NST * SLOT;
  static constexpr int BAR_BYTES = (2 * NST * 8 + 15) / 16 * 16;
  static constexpr int SMEM_GB = 2 * D * 4, MAX_ITEMS = 255, SMEM_SCHED = 4 * (MAX_ITEMS + 1), SMEM_DGB = 2 * D * 4;
  static constexpr int SMEM = SMEM_W + BAR_BYTES + SMEM_GB + SMEM_SCHED + SMEM_DGB;
  static_assert(SMEM + 1024 <= 167936, "sm_80 shared memory");
  static_assert(SLOT % (16 * NTHR) == 0, "slot copy");
};
using CfgP = BwdCfg<24576, 6, 2>;
using CfgX = BwdCfg<16384, 8, 2>;

template <class G>
struct Ring {
  uint32_t sW, bars;
  const __nv_bfloat16* src;     // this thread's first granule of the packed weights
  int tid, lane;
  DEVI uint32_t full(int s) const { return bars + 8 * s; }
  DEVI uint32_t empty(int s) const { return bars + 8 * (G::NST + s); }
  DEVI void issue(int c, int s) const {
    const __nv_bfloat16* g = src + c * (G::SLOT / 2);
    const uint32_t d = sW + s * G::SLOT + tid * 16;
#pragma unroll
    for (int i = 0; i < G::SLOT / 16 / G::NTHR; ++i) cp_async16_full(d + i * G::NTHR * 16, g + i * G::NTHR * 8);
    cp_async_mbar_arrive(full(s));
  }
  // start of chunk u: this warp's share of chunk u + AHEAD (deferred if a slow warp still holds its slot), then wait for chunk u
  DEVI uint32_t begin(int u, int total, bool& pending) const {
    const int s1 = (u + G::AHEAD) % G::NST;
    pending = u + G::AHEAD < total;
    if (pending && (u + G::AHEAD < G::NST || mbar_test(empty(s1), ((u + G::AHEAD) / G::NST - 1) & 1))) {
      issue((u + G::AHEAD) % G::NCHUNK, s1);
      pending = false;
    }
    mbar_wait_bo(full(u % G::NST), (u / G::NST) & 1);
    return sW + (u % G::NST) * G::SLOT;
  }
  DEVI void end(int u, bool pending) const {
    __syncwarp();
    if (lane == 0) mbar_arrive(empty(u % G::NST));
    if (pending) {
      const int s1 = (u + G::AHEAD) % G::NST;
      mbar_wait_bo(empty(s1), ((u + G::AHEAD) / G::NST - 1) & 1);
      issue((u + G::AHEAD) % G::NCHUNK, s1);
    }
  }
};

// per-CTA work list (half tile of the remainder first, then full tiles b, b + grid, ...), entries (tile row << 2) | m16 tiles per warp
template <class G>
DEVI int build_sched(int* sched, int T, int num_tiles, int tid) {
  const int b = blockIdx.x, grid = gridDim.x, BM = G::BM;
  const int f = num_tiles / grid, rem = num_tiles - f * grid;
  const int rem_rows = T - f * grid * BM, nh = (rem_rows + BM / 2 - 1) / (BM / 2);
  const bool half_mode = rem > 0 && nh <= grid;
  const int n_full = half_mode ? f : f + (b < rem ? 1 : 0);
  const int n_half = (half_mode && b < nh) ? 1 : 0;
  const int n_items = n_full + n_half;
  for (int it = tid; it <= G::MAX_ITEMS; it += G::NTHR) {
    int e = -1;
    if (it < n_items && it < G::MAX_ITEMS)
      e = it < n_half ? ((f * grid * BM + b * (BM / 2)) << 2) | 1 : (((b + (it - n_half) * grid) * BM) << 2) | 2;
    sched[it] = e;
  }
  return n_items;
}

DEVI void prefetch_rows(const __nv_bfloat16* base, int r0, int nrows, int T, int lane) {   // pull a warp's rows into L2
  if (lane < nrows && r0 + lane < T) {
    const char* a = reinterpret_cast<const char*>(base + (size_t)(r0 + lane) * 128);
    asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(a));
    asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(a + 128));
  }
}
DEVI uint32_t opq(uint32_t v) { uint32_t r; asm volatile("mov.b32 %0, %1;\n" : "=r"(r) : "r"(v)); return r; }
DEVI void stg128_na(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// ================================================================================================================== P
template <int MT>
DEVI void p_tile(const BwdParams& p, const Ring<CfgP>& w, uint32_t sGB, int r0, int nr0, int nmt, int u0, int total) {
  constexpr int D = 128;
  const int lane = w.lane, g8 = lane >> 2, q = lane & 3;
  // xn and dy A fragments (the forward's k order).  No other use may group these words differently (a 16 B load or store of a row):
  // the allocator would keep them in that grouping and re-copy each quad (4 MOVs) before every MMA -- so dy comes in 32-bit loads and
  // xn is not stored (W normalises its x tile itself)
  uint32_t fa[MT][8][4], fd[MT][8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    uint32_t xnw[2][16], dyw[2][16];                           // packed xn / dy words of rows g8 (hr 0) and g8 + 8 (hr 1)
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int r = r0 + 16 * mt + 8 * hr + g8;
      uint4 xin[4];
      const uint32_t* dyr = reinterpret_cast<const uint32_t*>(p.dy + (size_t)r * D + 8 * q);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        xin[i] = ldg_nc_na(p.x + (size_t)r * D + 32 * i + 8 * q);
#pragma unroll
        for (int k = 0; k < 4; ++k) dyw[hr][4 * i + k] = __ldg(dyr + 16 * i + k);   // 32-bit loads: each word lands in its fragment slot
      }
      // LayerNorm exactly as the forward computes it (the same xn, bit for bit)
      float xv[32];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t wv[4] = {xin[i].x, xin[i].y, xin[i].z, xin[i].w};
#pragma unroll
        for (int k = 0; k < 4; ++k) { xv[8 * i + 2 * k] = bf16lo(wv[k]); xv[8 * i + 2 * k + 1] = bf16hi(wv[k]); }
      }
      float sm = 0.f;
#pragma unroll
      for (int e = 0; e < 32; ++e) sm += xv[e];
      const float mean = quad_sum(sm) * (1.f / D);
      float sq = 0.f;
#pragma unroll
      for (int e = 0; e < 32; ++e) { xv[e] -= mean; sq = fmaf(xv[e], xv[e], sq); }
      const float rstd = rsqrtf(quad_sum(sq) * (1.f / D) + p.eps);
      if (q == 0) p.stats[r] = make_float2(mean, rstd);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint4 gi = lds128(sGB + (i * 4 + q) * 16), bi = lds128(sGB + 512 + (i * 4 + q) * 16);
        const float4 gv = *reinterpret_cast<const float4*>(&gi), bv = *reinterpret_cast<const float4*>(&bi);
        xnw[hr][2 * i] = pack_bf16(fmaf(xv[4 * i] * rstd, gv.x, bv.x), fmaf(xv[4 * i + 1] * rstd, gv.y, bv.y));
        xnw[hr][2 * i + 1] = pack_bf16(fmaf(xv[4 * i + 2] * rstd, gv.z, bv.z), fmaf(xv[4 * i + 3] * rstd, gv.w, bv.w));
      }
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {                              // word v = 2 s + e2 of row hr -> fragment word hr + 2 e2; the opaque
      fa[mt][s][0] = xnw[0][2 * s]; fa[mt][s][1] = xnw[1][2 * s]; fa[mt][s][2] = xnw[0][2 * s + 1]; fa[mt][s][3] = xnw[1][2 * s + 1];
      fd[mt][s][0] = dyw[0][2 * s]; fd[mt][s][1] = dyw[1][2 * s]; fd[mt][s][2] = dyw[0][2 * s + 1]; fd[mt][s][3] = dyw[1][2 * s + 1];
    }
  }
  __syncwarp();
  if (nmt > 0) { prefetch_rows(p.x, nr0, 16 * nmt, p.T, lane); prefetch_rows(p.dy, nr0, 16 * nmt, p.T, lane); }

  const int lrow = ((lane >> 4) << 3) + (lane & 7), lgr = (lane >> 3) & 1;
  const uint32_t w1_off = lgr * 1024 + lrow * 16, w3_off = 16384 + lgr * 512 + lrow * 16;
  uint4* const ab = p.ab + (size_t)(r0 / 16) * 3072 + lane;   // block (R, K, m) at ((R * 32 + K) * 3 + m) * 32 + lane
#pragma unroll 1
  for (int c = 0; c < 16; ++c) {
    bool pending;
    const uint32_t wb = w.begin(u0 + c, total, pending);
#pragma unroll
    for (int ps = 0; ps < 2; ++ps) {
      float acc1[MT][4][4], accd[MT][2][4];                    // [a lo, a hi, b lo, b hi], [dh lo, dh hi]
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
#pragma unroll
          for (int n = 0; n < 4; ++n) acc1[mt][n][e] = 0.f;
          accd[mt][0][e] = accd[mt][1][e] = 0.f;
        }
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t ba[4], bb[4], bd[4];
        ldsm_x4(ba, wb + w1_off + ps * 512 + s * 2048);
        ldsm_x4(bb, wb + w1_off + ps * 512 + 256 + s * 2048);
        ldsm_x4(bd, wb + w3_off + ps * 256 + s * 1024);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816(acc1[mt][0], fa[mt][s], ba[0], ba[1]);
          mma16816(acc1[mt][1], fa[mt][s], ba[2], ba[3]);
          mma16816(acc1[mt][2], fa[mt][s], bb[0], bb[1]);
          mma16816(acc1[mt][3], fa[mt][s], bb[2], bb[3]);
          mma16816(accd[mt][0], fd[mt][s], bd[0], bd[1]);
          mma16816(accd[mt][1], fd[mt][s], bd[2], bd[3]);
        }
      }
      const int K = 2 * c + ps;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        uint32_t wA[4], wB[4], wH[4];
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int hp = 0; hp < 2; ++hp) {                     // (c0, c1) row g8 -> word 2 n, (c2, c3) row g8 + 8 -> word 2 n + 1
            float vA[2], vB[2], vH[2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
              const int e = 2 * hp + k;
              const float ap = acc1[mt][n][e], bv = acc1[mt][2 + n][e], d = accd[mt][n][e];   // ap = a / 2 (Wa pre-scaled)
              const float t = tanh_approx(ap), sp = 1.f + t, silu = ap * sp;                // sig(a) = sp / 2, silu(a) = ap sp
              vH[k] = silu * bv;
              vB[k] = d * silu;
              vA[k] = d * bv * (0.5f * sp) * fmaf(ap, 1.f - t, 1.f);                        // sig + silu (1 - sig)
            }
            wA[2 * n + hp] = pack_bf16(vA[0], vA[1]);
            wB[2 * n + hp] = pack_bf16(vB[0], vB[1]);
            wH[2 * n + hp] = pack_bf16(vH[0], vH[1]);
          }
        uint4* const o = ab + mt * 3072 + K * 96;            // mt: the next m16 row block
        stg128(o, make_uint4(wA[0], wA[1], wA[2], wA[3]));
        stg128(o + 32, make_uint4(wB[0], wB[1], wB[2], wB[3]));
        stg128(o + 64, make_uint4(wH[0], wH[1], wH[2], wH[3]));
      }
    }
    w.end(u0 + c, pending);
  }
}

__global__ void __launch_bounds__(256, 1) tr_bwd_p_kernel(const BwdParams p) {
  using G = CfgP;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint8_t* bars = smem + G::SMEM_W;
  const uint32_t sGB = smem_u32(bars + G::BAR_BYTES);
  int* sched = reinterpret_cast<int*>(bars + G::BAR_BYTES + G::SMEM_GB);
  const int n_items = build_sched<G>(sched, p.T, p.num_tiles, tid);
  Ring<G> w;
  w.sW = smem_u32(smem); w.bars = smem_u32(bars); w.src = p.wp + tid * 8; w.tid = tid; w.lane = lane;
  const int total = n_items * 16;
  if (tid == 0)
    for (int s = 0; s < G::NST; ++s) { mbar_init(w.full(s), G::NTHR); mbar_init(w.empty(s), G::NWARP); }
  for (int k = tid; k < 64; k += G::NTHR) reinterpret_cast<float4*>(bars + G::BAR_BYTES)[k] = p.gb[k];
  __syncthreads();
  if (n_items == 0) return;
  for (int c = 0; c < G::AHEAD && c < total; ++c) w.issue(c % 16, c);
  int e = sched[0];
#pragma unroll 1
  for (int it = 0; e >= 0; ++it) {
    const int en = sched[it + 1];
    const int mt = e & 3, nmt = en >= 0 ? en & 3 : 0;
    const int r0 = (e >> 2) + 16 * mt * warp, nr0 = (en >> 2) + 16 * nmt * warp;
    if (mt == 2) p_tile<2>(p, w, sGB, r0, nr0, nmt, 16 * it, total);
    else p_tile<1>(p, w, sGB, r0, nr0, nmt, 16 * it, total);
    e = en;
  }
  cp_async_wait<0>();
}

// ================================================================================================================== X
template <int MT>
DEVI void x_tile(const BwdParams& p, const Ring<CfgX>& w, uint32_t sGam, uint32_t sDgb, int r0, int nr0, int nmt, int u0, int total) {
  constexpr int D = 128;
  const int lane = w.lane, g8 = lane >> 2, q = lane & 3;
  float acc[MT][16][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][j][e] = 0.f;
  const uint4* const ab = p.ab + (size_t)(r0 / 16) * 3072 + lane;
  // A fragments of step K straight from the blocks, one step ahead of their MMAs
  uint4 fA[MT], fB[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) { fA[mt] = ldg_nc_na(ab + mt * 3072); fB[mt] = ldg_nc_na(ab + mt * 3072 + 32); }
  // ldmatrix.trans lane address: matrix mi = lane / 8 -> (n-granule half mi >> 1, hidden half mi & 1)
  const uint32_t b_off = (lane >> 4) * 512 + ((((lane >> 3) & 1) << 3) + (lane & 7)) * 16;
#pragma unroll 1
  for (int c = 0; c < 16; ++c) {
    bool pending;
    const uint32_t wb = w.begin(u0 + c, total, pending);
#pragma unroll
    for (int ps = 0; ps < 2; ++ps) {
      const int K = 2 * c + ps;
      uint32_t aA[MT][4], aB[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        aA[mt][0] = fA[mt].x; aA[mt][1] = fA[mt].y; aA[mt][2] = fA[mt].z; aA[mt][3] = fA[mt].w;
        aB[mt][0] = fB[mt].x; aB[mt][1] = fB[mt].y; aB[mt][2] = fB[mt].z; aB[mt][3] = fB[mt].w;
      }
      if (K + 1 < 32) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          fA[mt] = ldg_nc_na(ab + mt * 3072 + (K + 1) * 96);
          fB[mt] = ldg_nc_na(ab + mt * 3072 + (K + 1) * 96 + 32);
        }
      }
#pragma unroll
      for (int ab = 0; ab < 2; ++ab)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          uint32_t bw[4];
          ldsm_x4_t(bw, wb + ab * 8192 + j * 1024 + ps * 256 + b_off);
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma16816(acc[mt][2 * j], ab ? aB[mt] : aA[mt], bw[0], bw[1]);
            mma16816(acc[mt][2 * j + 1], ab ? aB[mt] : aA[mt], bw[2], bw[3]);
          }
        }
    }
    w.end(u0 + c, pending);
  }
  if (nmt > 0) { prefetch_rows(p.x, nr0, 16 * nmt, p.T, lane); prefetch_rows(p.dy, nr0, 16 * nmt, p.T, lane); }

  // ---- LayerNorm backward + residual; accumulator tile J = word J of the thread's 64 B (columns 32 (J / 4) + 8 q + 2 (J % 4) + e)
  float dgs[32], dbs[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) dgs[k] = dbs[k] = 0.f;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int r = r0 + 16 * mt + 8 * hr + g8;
      const float2 st = p.stats[r];
      uint32_t xw[16], dw[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 xi = ldg_nc_na(p.x + (size_t)r * D + 32 * i + 8 * q), di = ldg_nc_na(p.dy + (size_t)r * D + 32 * i + 8 * q);
        xw[4 * i] = xi.x; xw[4 * i + 1] = xi.y; xw[4 * i + 2] = xi.z; xw[4 * i + 3] = xi.w;
        dw[4 * i] = di.x; dw[4 * i + 1] = di.y; dw[4 * i + 2] = di.z; dw[4 * i + 3] = di.w;
      }
      float xh[32], gd[32], s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int J = 0; J < 16; ++J) {
        const uint32_t col = 32 * (J >> 2) + 8 * q + 2 * (J & 3);
        const uint2 gi = lds64(sGam + col * 4);
        const float g0 = __uint_as_float(gi.x), g1 = __uint_as_float(gi.y);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float xv = e ? bf16hi(xw[J]) : bf16lo(xw[J]);
          const float dn = acc[mt][J][2 * hr + e];
          xh[2 * J + e] = (xv - st.x) * st.y;
          gd[2 * J + e] = (e ? g1 : g0) * dn;
          dgs[2 * J + e] = fmaf(dn, xh[2 * J + e], dgs[2 * J + e]);
          dbs[2 * J + e] += dn;
          s1 = fmaf(gd[2 * J + e], xh[2 * J + e], s1);
          s2 += gd[2 * J + e];
        }
      }
      const float ca = quad_sum(s1) * (1.f / D), cb = quad_sum(s2) * (1.f / D);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int J = 4 * i + k;
          const float v0 = fmaf(gd[2 * J] - fmaf(xh[2 * J], ca, cb), st.y, bf16lo(dw[J]));
          const float v1 = fmaf(gd[2 * J + 1] - fmaf(xh[2 * J + 1], ca, cb), st.y, bf16hi(dw[J]));
          o[k] = pack_bf16(v0, v1);
        }
        stg128(p.dx + (size_t)r * D + 32 * i + 8 * q, make_uint4(o[0], o[1], o[2], o[3]));
      }
    }
  // dgamma / dbeta: over the warp's 8 row groups (lanes with the same q), then one shared-memory atomic per column
#pragma unroll
  for (int k = 0; k < 32; ++k) {
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      dgs[k] += __shfl_xor_sync(0xffffffffu, dgs[k], o);
      dbs[k] += __shfl_xor_sync(0xffffffffu, dbs[k], o);
    }
  }
  if (g8 == 0) {
#pragma unroll
    for (int J = 0; J < 16; ++J)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const uint32_t col = 32 * (J >> 2) + 8 * q + 2 * (J & 3) + e;
        asm volatile("red.shared.add.f32 [%0], %1;\n" ::"r"(sDgb + col * 4), "f"(dgs[2 * J + e]) : "memory");
        asm volatile("red.shared.add.f32 [%0], %1;\n" ::"r"(sDgb + 512 + col * 4), "f"(dbs[2 * J + e]) : "memory");
      }
  }
}

__global__ void __launch_bounds__(256, 1) tr_bwd_x_kernel(const BwdParams p) {
  using G = CfgX;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint8_t* bars = smem + G::SMEM_W;
  float* sGam = reinterpret_cast<float*>(bars + G::BAR_BYTES);
  int* sched = reinterpret_cast<int*>(bars + G::BAR_BYTES + G::SMEM_GB);
  float* sDgb = reinterpret_cast<float*>(bars + G::BAR_BYTES + G::SMEM_GB + G::SMEM_SCHED);
  const int n_items = build_sched<G>(sched, p.T, p.num_tiles, tid);
  Ring<G> w;
  w.sW = smem_u32(smem); w.bars = smem_u32(bars); w.src = p.wx + tid * 8; w.tid = tid; w.lane = lane;
  const int total = n_items * 16;
  if (tid == 0)
    for (int s = 0; s < G::NST; ++s) { mbar_init(w.full(s), G::NTHR); mbar_init(w.empty(s), G::NWARP); }
  for (int k = tid; k < 128; k += G::NTHR) sGam[k] = p.gamma[k];
  for (int k = tid; k < 256; k += G::NTHR) sDgb[k] = 0.f;
  __syncthreads();
  if (n_items > 0) {
    for (int c = 0; c < G::AHEAD && c < total; ++c) w.issue(c % 16, c);
    int e = sched[0];
#pragma unroll 1
    for (int it = 0; e >= 0; ++it) {
      const int en = sched[it + 1];
      const int mt = e & 3, nmt = en >= 0 ? en & 3 : 0;
      const int r0 = (e >> 2) + 16 * mt * warp, nr0 = (en >> 2) + 16 * nmt * warp;
      if (mt == 2) x_tile<2>(p, w, smem_u32(sGam), smem_u32(sDgb), r0, nr0, nmt, 16 * it, total);
      else x_tile<1>(p, w, smem_u32(sGam), smem_u32(sDgb), r0, nr0, nmt, 16 * it, total);
      e = en;
    }
    cp_async_wait<0>();
  }
  __syncthreads();
  for (int k = tid; k < 256; k += G::NTHR) p.dgb[(size_t)blockIdx.x * 256 + k] = sDgb[k];
}

}  // namespace a100
