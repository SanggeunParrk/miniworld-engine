// tr_fwd_f8_sm120.cuh -- OPT-IN e4m3 Transition inference forward for the GeForce RTX 5090 (sm_120a), one kernel (fp8_sm120.py, docs/fp8.md "RTX 5090"):
//
//   out = x + Ws (silu(Wa xn) * (Wb xn)),  xn = LN(x),  D = 128 / 256, H = 2 D / 4 D, bf16 in / out, both GEMMs on mma.sync.m16n8k32 e4m3 (fp32 accumulation).
//
// The structure of tr_fwd_sm80.cuh (warp-owned rows for the whole hidden dimension, weights streaming from L2 through a cp.async ring of 32-hidden chunks with
// FULL / EMPTY mbarriers and no CTA barrier), with e4m3 operands:
//   * a warp owns 16 rows (one m16 tile: the fp32 output accumulator, D / 8 n8 tiles x 4, takes the registers the f16 one of the bf16 kernel gave to a second tile);
//   * LayerNorm in fp32 straight into the GEMM1 A fragments as e4m3 (xn sx, sx = the power of two of the LayerNorm bound: cannot saturate); the thread's
//     own 4 x 16 B of x per row are its fragments: GEMM1's K is permuted on the host (k32 step i, kernel k 4 q + j <-> column 32 i + 8 q + j, 16 + 4 q + j <->
//     32 i + 8 q + 4 + j);
//   * per 32 hidden units: GEMM1 (a | b, 8 n8 tiles x D / 32 k-steps) -> SwiGLU in fp32 in the C fragments -> h sh quantised in registers into the k32 A
//     fragment of GEMM2 (the C columns {2q, 2q+1, 8+2q, 9+2q} of a thread are its 4 A bytes: Ws's K is permuted to match, fp8_sm90a._perm_k) -> GEMM2 (D / 8 MMAs);
//   * the output accumulator starts at x (sh s2) and ends multiplied by 1 / (sh s2) (powers of two: the residual is exact in fp32).
// Scales sc (device, so a captured graph follows a rescale): [0] sx, [1] 0.5 / (sx s1) (a / 2 for silu(a) = a' (1 + tanh a')), [2] sh / (sx s1), [3] sh s2,
// [4] 1 / (sh s2).  peak: atomicMax of |h sh| (float bits) before the cast -- the guard (> 448: some value saturated).
#pragma once
#include "tr_fwd_sm80.cuh"

namespace a100 {

#ifndef TR8_NWARP
#define TR8_NWARP 8
#endif
#ifndef TR8_MINB
#define TR8_MINB 1
#endif
#ifndef TR8_SLEEP
#define TR8_SLEEP 0
#endif

template <int D_, int H_>
struct TrF8Cfg {
  static constexpr bool F32 = false;   // (load_x)
  static constexpr int D = D_, H = H_, NWARP = TR8_NWARP, NTHR = 32 * NWARP, MINB = TR8_MINB, MTF = 1, BM = 16 * NWARP;
  static constexpr int NS = D / 32, NJ = D / 8, NV = D / 32;   // GEMM1 k32 steps, GEMM2 n8 tiles, 16 B vectors of a quad thread per row
#ifndef TR8_FINE
#define TR8_FINE (D_ >= 256)
#endif
  static constexpr bool FINE = TR8_FINE;                        // GEMM1 B fragments one 16-row block ahead instead of one k-step (registers)
#ifndef TR8_RELOAD
#define TR8_RELOAD (D_ >= 256)
#endif
  static constexpr bool RELOAD = TR8_RELOAD;                    // the residual re-read at the store (registers across the LayerNorm)
  static constexpr int CH = 32, NCHUNK = H / CH;
  static constexpr int SLOT_W1 = 2 * CH * D;                    // [D / 16 k-granules][2 CH rows (Wa 32 | Wb 32)][16 B]
  static constexpr int GR1 = 2 * CH * 16;
  static constexpr int SLOT = SLOT_W1 + CH * D;                 // + [CH / 16 hidden granules][D out rows][16 B]
#ifndef TR8_NST
#define TR8_NST (((101376 - 4096) / TR8_MINB) / (96 * D_) > 8 ? 8 : ((101376 - 4096) / TR8_MINB) / (96 * D_))
#endif
#ifndef TR8_AHEAD
#define TR8_AHEAD 2
#endif
  static constexpr int NST = TR8_NST, AHEAD = TR8_AHEAD, RES = 0, NRING = NCHUNK;
  static constexpr int XW = 0, SMEM_W = NST * SLOT, SMEM_X = 0;
  static constexpr int NBAR = 2 * NST + NWARP + 1;
  static constexpr int BAR_BYTES = (NBAR * 8 + 15) / 16 * 16;
  static constexpr int SMEM_GB = 2 * D * 4;
  static constexpr int MAX_ITEMS = 255, SMEM_SCHED = 4 * (MAX_ITEMS + 1);
  static constexpr int SMEM = SMEM_W + SMEM_X + BAR_BYTES + SMEM_GB + SMEM_SCHED;
  static_assert(SMEM * MINB <= 101376 + 0 * 1024 && SMEM <= 101376, "sm_120 shared memory (99 KB a block)");
  static_assert(NST >= AHEAD + 1, "ring");
  static_assert(SLOT % (16 * NTHR) == 0, "slot copy split");
  static_assert(D % 32 == 0 && H % CH == 0, "widths");
};

struct Tr8Params {
  const __nv_bfloat16* x;
  const uint8_t* w;         // [NCHUNK][SLOT] e4m3 bytes
  const float4* gb;         // LN affine as float4 slots (tr_fwd_sm80 layout)
  __nv_bfloat16* out;
  const float* sc;          // [5] scales (above)
  float* peak;              // running max of |h sh| (float bits, atomicMax as int)
  int T, num_tiles;
  float eps;
};

DEVI void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// four floats -> four e4m3 bytes, x0 in the lowest byte (round to nearest, saturating)
DEVI uint32_t f8x4(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(x3), "f"(x2));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

// chunk c of the packed weights -> ring slot s (this thread's SLOT / 16 / NTHR granules), completion on the slot's FULL barrier
template <class G>
DEVI void issue_w8(const TrWarp& w, int c, int s) {
  const uint8_t* src = reinterpret_cast<const uint8_t*>(w.wsrc) + (size_t)c * G::SLOT;
  const uint32_t dst = w.sW_u + s * G::SLOT + w.tid * 16;
#pragma unroll
  for (int i = 0; i < G::SLOT / 16 / G::NTHR; ++i) cp_async16_full(dst + i * G::NTHR * 16, src + i * G::NTHR * 16);
  cp_async_mbar_arrive(bar_full<G>(w, s));
}

// per-warp state of the ring: the TrWarp of tr_fwd_sm80 (bar_full / bar_empty take this config)
template <class G>
DEVI void tile8(const Tr8Params& p, TrWarp& w, int r0, int nr0, int nmt, float& pk) {
  constexpr int D = G::D, NCHUNK = G::NCHUNK, NST = G::NST, NS = G::NS, NJ = G::NJ, NV = G::NV;
  const int g8 = w.g8, q = w.q;
  const float sx = __ldg(p.sc), sa = __ldg(p.sc + 1), sb = __ldg(p.sc + 2), c2 = __ldg(p.sc + 3), ic2 = __ldg(p.sc + 4);

  // ---- LayerNorm of rows r0 + 8 hr + g8 -> e4m3 A fragments fa[i] (k32 step i: words 0 / 2 row g8, 1 / 3 row g8 + 8); the accumulator starts at x c2
  uint32_t fa[NS][4];
  float acc2[NJ][4];
#pragma unroll
  for (int hr = 0; hr < 2; ++hr) {
    const int r = r0 + 8 * hr + g8;
    uint4 xin[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) xin[i] = r < p.T ? ldg_nc_na(p.x + (size_t)r * D + 32 * i + 8 * q) : make_uint4(0, 0, 0, 0);
    float xv[8 * NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      const uint32_t wv[4] = {xin[i].x, xin[i].y, xin[i].z, xin[i].w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        xv[8 * i + 2 * k] = bf16lo(wv[k]);
        xv[8 * i + 2 * k + 1] = bf16hi(wv[k]);
        if constexpr (!G::RELOAD) acc2[4 * i + k][2 * hr] = xv[8 * i + 2 * k] * c2;
        if constexpr (!G::RELOAD) acc2[4 * i + k][2 * hr + 1] = xv[8 * i + 2 * k + 1] * c2;
      }
    }
    float sm = 0.f;
#pragma unroll
    for (int e = 0; e < 8 * NV; ++e) sm += xv[e];
    const float mean = quad_sum(sm) * (1.f / D);
    float sq = 0.f;
#pragma unroll
    for (int e = 0; e < 8 * NV; ++e) { xv[e] -= mean; sq = fmaf(xv[e], xv[e], sq); }
    const float rs = rsqrtf(quad_sum(sq) * (1.f / D) + p.eps) * sx;
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      float o[8];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {   // the float4 slots 2 i + hh of the affine table: columns 32 i + 8 q + 4 hh .. + 3
        const uint4 gi = lds128(w.sGB + ((2 * i + hh) * 4 + q) * 16), bi = lds128(w.sGB + 4 * D + ((2 * i + hh) * 4 + q) * 16);
        const float4 gv = *reinterpret_cast<const float4*>(&gi), bv = *reinterpret_cast<const float4*>(&bi);
        o[4 * hh] = fmaf(xv[8 * i + 4 * hh] * rs, gv.x, bv.x * sx);
        o[4 * hh + 1] = fmaf(xv[8 * i + 4 * hh + 1] * rs, gv.y, bv.y * sx);
        o[4 * hh + 2] = fmaf(xv[8 * i + 4 * hh + 2] * rs, gv.z, bv.z * sx);
        o[4 * hh + 3] = fmaf(xv[8 * i + 4 * hh + 3] * rs, gv.w, bv.w * sx);
      }
      fa[i][hr] = f8x4(o[0], o[1], o[2], o[3]);
      fa[i][hr + 2] = f8x4(o[4], o[5], o[6], o[7]);
    }
  }
  if constexpr (G::RELOAD) {   // the accumulator starts at 0, x is added at the store (re-read from L2): x is not live across the LayerNorm
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc2[j][e] = 0.f;
  }
  __syncwarp();
  if (nmt > 0 && w.lane < 16 && nr0 + w.lane < p.T) {   // the next item's rows into L2 (as tr_fwd_sm80's unstaged load_x)
    const char* a = reinterpret_cast<const char*>(p.x + (size_t)(nr0 + w.lane) * D);
#pragma unroll
    for (int l = 0; l < D * 2 / 128; ++l) asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(a + 128 * l));
  }

#pragma unroll 1
  for (int c = 0; c < NCHUNK; ++c) {
    const int u = w.u, slot = u % NST;
    constexpr int AH = G::AHEAD;
    const int s1 = (u + AH) % NST;
    const uint32_t par1 = ((u + AH) / NST - 1) & 1;
    bool pending = u + AH < w.total;
    if (pending && (u + AH < NST || mbar_test(bar_empty<G>(w, s1), par1))) {
      issue_w8<G>(w, (u + AH) % G::NRING, s1);
      pending = false;
    }
    {
      const uint32_t bar = bar_full<G>(w, slot), par = (u / NST) & 1;
      while (!mbar_test(bar, par)) { if (TR8_SLEEP) __nanosleep(TR8_SLEEP); }
    }
    const uint32_t wb = w.sW_u + slot * G::SLOT;
    // GEMM1: 8 n8 tiles (a 0..3 = hidden 0..31, b 0..3), B fragments one k-step ahead
    float acc1[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc1[n][e] = 0.f;
    if constexpr (G::FINE) {   // B fragments one 16-row block ahead (4 registers in flight: the D 256 register budget)
      uint32_t bf[2][4];
      ldsm_x4(bf[0], wb + w.w1_off);
#pragma unroll
      for (int k = 0; k < 4 * NS; ++k) {
        const int s = k >> 2, rb = k & 3, cb = k & 1;
        if (k + 1 < 4 * NS) ldsm_x4(bf[cb ^ 1], wb + w.w1_off + ((k + 1) >> 2) * 2 * G::GR1 + ((k + 1) & 3) * 256);
        mma_e4m3(acc1[2 * rb], fa[s], bf[cb][0], bf[cb][1]);
        mma_e4m3(acc1[2 * rb + 1], fa[s], bf[cb][2], bf[cb][3]);
      }
    } else {
    uint32_t bf[2][4][4];
#pragma unroll
    for (int rb = 0; rb < 4; ++rb) ldsm_x4(bf[0][rb], wb + w.w1_off + rb * 256);
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      const int cb = s & 1;
      if (s + 1 < NS) {
#pragma unroll
        for (int rb = 0; rb < 4; ++rb) ldsm_x4(bf[cb ^ 1][rb], wb + w.w1_off + (s + 1) * 2 * G::GR1 + rb * 256);
      }
#pragma unroll
      for (int rb = 0; rb < 4; ++rb) {   // 16-row block rb of the slot: tiles 2 rb, 2 rb + 1
        mma_e4m3(acc1[2 * rb], fa[s], bf[cb][rb][0], bf[cb][rb][1]);
        mma_e4m3(acc1[2 * rb + 1], fa[s], bf[cb][rb][2], bf[cb][rb][3]);
      }
    }
    }
    uint32_t bs[2][4];
    ldsm_x4(bs[0], wb + w.w2_off);
    // SwiGLU (fp32) -> h sh as the GEMM2 A fragment: word 0 row g8 = tiles 0, 1 (c0, c1), word 1 row g8 + 8 (c2, c3), words 2 / 3 = tiles 2, 3
    uint32_t ha[4];
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {
      const int n0 = 2 * (wd >> 1), e0 = 2 * (wd & 1);
      float hv[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int n = n0 + (k >> 1), e = e0 + (k & 1);
        const float a = acc1[n][e] * sa, b = acc1[4 + n][e] * sb;
        const float ab = a * b;
        hv[k] = fmaf(tanh_approx(a), ab, ab);
        pk = fmaxf(pk, fabsf(hv[k]));
      }
      ha[wd] = f8x4(hv[0], hv[1], hv[2], hv[3]);
    }
    // GEMM2: acc2 += h Ws^T over these 32 hidden units
#pragma unroll
    for (int j = 0; j < NJ / 2; ++j) {
      if (j + 1 < NJ / 2) ldsm_x4(bs[(j + 1) & 1], wb + w.w2_off + (j + 1) * 256);
      mma_e4m3(acc2[2 * j], ha, bs[j & 1][0], bs[j & 1][1]);
      mma_e4m3(acc2[2 * j + 1], ha, bs[j & 1][2], bs[j & 1][3]);
    }
    __syncwarp();
    if (w.lane == 0) mbar_arrive(bar_empty<G>(w, slot));
    if (pending) {
      while (!mbar_test(bar_empty<G>(w, s1), par1)) { }
      issue_w8<G>(w, (u + AH) % G::NRING, s1);
    }
    ++w.u;
  }

  // ---- store: tiles 4 i .. 4 i + 3 = the thread's 16 B at columns 32 i + 8 q of its rows
#pragma unroll
  for (int hr = 0; hr < 2; ++hr) {
    const int r = r0 + 8 * hr + g8;
    if (r < p.T) {
      __nv_bfloat16* dst = p.out + (size_t)r * D + 8 * q;
#pragma unroll
      for (int i = 0; i < NV; ++i) {
        uint32_t o[4];
        if constexpr (G::RELOAD) {
          const uint4 xr = ldg_nc_na(p.x + (size_t)r * D + 32 * i + 8 * q);
          const uint32_t xw[4] = {xr.x, xr.y, xr.z, xr.w};
#pragma unroll
          for (int k = 0; k < 4; ++k)
            o[k] = pack_bf16(fmaf(acc2[4 * i + k][2 * hr], ic2, bf16lo(xw[k])), fmaf(acc2[4 * i + k][2 * hr + 1], ic2, bf16hi(xw[k])));
        } else {
#pragma unroll
          for (int k = 0; k < 4; ++k) o[k] = pack_bf16(acc2[4 * i + k][2 * hr] * ic2, acc2[4 * i + k][2 * hr + 1] * ic2);
        }
        stg128(dst + 32 * i, make_uint4(o[0], o[1], o[2], o[3]));
      }
    }
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) tr_fwd_f8_kernel(const Tr8Params p) {
  constexpr int BM = G::BM;
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + G::SMEM_W);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int b = blockIdx.x, grid = gridDim.x;
  const int f = p.num_tiles / grid, rem = p.num_tiles - f * grid;
  const int n_items = f + (b < rem ? 1 : 0);
  int* sched = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(bars) + G::BAR_BYTES + G::SMEM_GB);
  for (int it = tid; it <= G::MAX_ITEMS; it += G::NTHR) sched[it] = it < n_items && it < G::MAX_ITEMS ? (((b + it * grid) * BM) << 2) | 1 : -1;

  TrWarp w;
  w.sW_u = smem_u32(smem);
  w.barX = smem_u32(bars) + 8 * (2 * G::NST + warp);
  w.sGB = smem_u32(bars) + G::BAR_BYTES;
  w.xw = 0;
  w.wsrc = reinterpret_cast<const __nv_bfloat16*>(p.w) + tid * 8;
  w.tid = tid; w.lane = lane; w.g8 = lane >> 2; w.q = lane & 3;
  w.u = 0; w.total = n_items * G::NRING;
  const int lrow = ((lane >> 4) << 3) + (lane & 7), lgr = (lane >> 3) & 1;
  w.w1_off = lgr * G::GR1 + lrow * 16;                 // + s * 2 GR1 (k32 step) + rb * 256 (16-row block)
  w.w2_off = G::SLOT_W1 + lgr * (16 * G::D) + lrow * 16;   // + j * 256 (16 output rows)

  if (tid == 0) {
    for (int s = 0; s < G::NST; ++s) { mbar_init(bar_full<G>(w, s), G::NTHR); mbar_init(bar_empty<G>(w, s), G::NWARP); }
  }
  for (int k = tid; k < G::D / 2; k += G::NTHR) reinterpret_cast<float4*>(reinterpret_cast<uint8_t*>(bars) + G::BAR_BYTES)[k] = p.gb[k];
  __syncthreads();
  if (n_items == 0) return;
  for (int c = 0; c < G::AHEAD && c < w.total; ++c) issue_w8<G>(w, c % G::NRING, c);
  float pk = 0.f;
  int e = sched[0];
#pragma unroll 1
  for (int it = 0; e >= 0; ++it) {
    const int en = sched[it + 1];
    const int r0 = (e >> 2) + 16 * warp, nr0 = en >= 0 ? (en >> 2) + 16 * warp : 0;
    tile8<G>(p, w, r0, nr0, en >= 0 ? 1 : 0, pk);
    e = en;
  }
  cp_async_wait<0>();
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) pk = fmaxf(pk, __shfl_xor_sync(0xffffffffu, pk, o));
  if (lane == 0 && pk > 0.f) atomicMax(reinterpret_cast<int*>(p.peak), __float_as_int(pk));
}

}  // namespace a100
