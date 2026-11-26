// gp_kernels.cuh -- the gated projections of the A100 (sm_80), hand CUDA: the output projection of a gated activation (gate_out forward / dgrad), the dual-GEMM
// sigmoid gates of the triangle multiplication (tm1: four projections of one input, tm2: gate and output projection), and the one-pass elementwise gates.  All on
// the GEMM tile of the wide TriMul (``trimul_inproj/cuda/sm80/wide_gemm.cuh``: mma.sync m16n8k16, a cp.async ring, ldmatrix fragments, staged 16-byte epilogue
// stores); the accumulators are fp32 and every rounding to bf16 is the single one at a store, as the fused Triton kernels these replace.
//
//   gp_fwd        out[m, n]  = sum_k bf16(sigmoid(g[m, k]) v[m, k]) W[n, k]          the gated tile is formed in shared memory by the mainloop's hook
//   gp_dgrad      dA = dO W (fp32 accumulate), s = sigmoid(g):  dv = s dA,  dg = dA v s (1 - s),  a = bf16(s v)   (a feeds the dW GEMM)
//   tm2           out = sigmoid(x Wg) (y Wo);   backward (recompute): dB = d s, dA = dB (y Wo) (1 - s)
//   tm1           left = sigmoid(x WLg) (x WL), right = sigma(x WRg) (x WR);  backward: dLB = dL s_L, dLA = dLB (x WL) (1 - s_L), likewise right
//   sigmul / gres elementwise: a = bf16(sigmoid(g) o);  y = bf16(x + bf16(g b))   and their gradients
#pragma once
#include "wide_gemm.cuh"

namespace a100 {

// --------------------------------------------------------------------------------------------------------------------------- gated A operand
// The A tile of a stage is the VALUE tile (cp.async from v); the hook adds the GATE tile (cp.async from g, EXTRA bytes after A and B) and, once this thread's
// granules of the stage have landed, turns its own granules of the value tile into bf16(sigmoid(g) v) in place.
template <int BM, int NTHR, int GOFF>
struct GateHook {
  const __nv_bfloat16* g;
  size_t ldg;
  int i0, Mlim;
  DEVI void load(uint32_t sa, int kt) const {
#pragma unroll
    for (int e = 0; e < 4 * BM / NTHR; ++e) {
      const int gi = (int)threadIdx.x + NTHR * e, row = gi >> 2, c = gi & 3;
      const bool ok = i0 + row < Mlim;
      cp_async16(sa + GOFF + wkmaj(row, c), g + (size_t)(ok ? i0 + row : 0) * ldg + kt * 32 + c * 8, ok ? 16u : 0u);
    }
  }
  DEVI void transform(uint32_t sa) const {
#pragma unroll
    for (int e = 0; e < 4 * BM / NTHR; ++e) {
      const int gi = (int)threadIdx.x + NTHR * e, row = gi >> 2, c = gi & 3;
      const uint32_t av = sa + wkmaj(row, c), gv = sa + GOFF + wkmaj(row, c);
      const uint4 v = lds128(av), gg = lds128(gv);
      const uint32_t vw[4] = {v.x, v.y, v.z, v.w}, gw[4] = {gg.x, gg.y, gg.z, gg.w};
      uint32_t o[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j] = pack_bf16(sigmoid(bf16lo(gw[j])) * bf16lo(vw[j]), sigmoid(bf16hi(gw[j])) * bf16hi(vw[j]));
      sts128(av, make_uint4(o[0], o[1], o[2], o[3]));
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------------------------ gp_fwd
struct GpFwdParams {
  const __nv_bfloat16* v; size_t ldv;       // value (the attention output) [M][K]
  const __nv_bfloat16* g; size_t ldg;       // gate logits [M][K]
  const __nv_bfloat16* w;                   // [N][K] row-major: nn.Linear's (out, in)
  __nv_bfloat16* out; size_t ldo;           // [M][N]
  int M, N, K;
};

template <int BM, int BN>
struct GpFwdCfg {
  using Tile = WTile<BM, BN, 2, 2, 3, false, false, BM * 64>;
  static constexpr int GOFF = Tile::TILE_A + Tile::TILE_B;
  using Hook = GateHook<BM, Tile::NTHR, GOFF>;
};

template <int BM, int BN>
__global__ void __launch_bounds__(128, 2) gp_fwd_kernel(const GpFwdParams p) {
  using C = GpFwdCfg<BM, BN>;
  using Tile = typename C::Tile;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = smem_u32(smem);
  const int ntn = p.N / BN;
  const int mtile = (int)blockIdx.x / ntn, ntile = (int)blockIdx.x - mtile * ntn;
  const int t0 = mtile * BM, n0 = ntile * BN;
  Tile tl;
  const typename C::Hook hook{p.g, p.ldg, t0, p.M};
  const GemmOps ops{p.v, p.ldv, t0, p.M, p.w, (size_t)p.K, n0, p.N};
  if ((p.M & (BM - 1)) == 0) tl.run_fast(s0, p.K >> 5, ops, hook);          // whole tiles: the minimal-addressing mainloop; a ragged last tile needs run()'s bounds handling
  else tl.run(s0, p.K >> 5, ops, hook);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1, g8 = lane >> 2, q = lane & 3;
  constexpr int MT = Tile::MT, NT = 2 * Tile::NP, WTN = Tile::WTN;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh)
        stage_word(s0, Tile::WTM * wm + 16 * mt + g8 + 8 * rh, (WTN / 2) * wn + 4 * nt + q, BN, pack_bf16(tl.acc[mt][nt][2 * rh], tl.acc[mt][nt][2 * rh + 1]));
  __syncthreads();
  tile_store16<BM, BN, 128>(s0, p.out + (size_t)t0 * p.ldo + n0, p.ldo, p.M - t0);
}

// ---------------------------------------------------------------------------------------------------------------------------------- gp_dgrad
struct GpDgradParams {
  const __nv_bfloat16* dO; size_t ldo;      // [M][N]
  const __nv_bfloat16* wt;                  // W^T [K][N] (the reduction runs over N: K-major rows of the GEMM's B)
  const __nv_bfloat16* g; size_t ldg;       // [M][K]
  const __nv_bfloat16* v; size_t ldv;       // [M][K]
  __nv_bfloat16 *dv, *dg, *a;               // [M][K] each, row stride K
  int M, N, K;
};

// dynamic shared memory: the GEMM ring, or the staged g and v tiles of the epilogue (rows of BN * 2 + 16 bytes), whichever is larger
template <int BM, int BN>
struct GpDgradCfg {
  using Tile = WTile<BM, BN, 2, 2, 4, false, false>;
  static constexpr int SMEM = Tile::SMEM > 2 * BM * (BN * 2 + 16) ? Tile::SMEM : 2 * BM * (BN * 2 + 16);
};

// The epilogue needs g and v at every accumulator element.  Loading them per element (4-byte loads, each followed by its use) serialised one global latency per element;
// instead the g and v tiles are copied into the (now free) ring with coalesced 16-byte cp.async, and the elements are read from shared memory in the staged layout.
template <int BM, int BN>
__global__ void __launch_bounds__(128, 2) gp_dgrad_kernel(const GpDgradParams p) {
  using Tile = typename GpDgradCfg<BM, BN>::Tile;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = smem_u32(smem);
  const int ntn = p.K / BN;
  const int mtile = (int)blockIdx.x / ntn, ntile = (int)blockIdx.x - mtile * ntn;
  const int t0 = mtile * BM, n0 = ntile * BN;
  Tile tl;
  const GemmOps ops{p.dO, p.ldo, t0, p.M, p.wt, (size_t)p.N, n0, p.K};
  if ((p.M & (BM - 1)) == 0) tl.run_fast(s0, p.N >> 5, ops);
  else tl.run(s0, p.N >> 5, ops);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1, g8 = lane >> 2, q = lane & 3;
  constexpr int MT = Tile::MT, NT = 2 * Tile::NP, WTN = Tile::WTN, RB = BN * 2 + 16, GR = BN / 8;
  const uint32_t sg = s0, sv = s0 + BM * RB;
#pragma unroll
  for (int e = 0; e < BM * GR / 128; ++e) {
    const int idx = tid + 128 * e, r = idx / GR, c = idx % GR;
    const bool ok = t0 + r < p.M;
    const size_t row = (size_t)(ok ? t0 + r : 0);
    cp_async16(sg + r * RB + c * 16, p.g + row * p.ldg + n0 + 8 * c, ok ? 16u : 0u);
    cp_async16(sv + r * RB + c * 16, p.v + row * p.ldv + n0 + 8 * c, ok ? 16u : 0u);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  uint32_t wdv[MT][NT][2], wdg[MT][NT][2], wa[MT][NT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const int off = (Tile::WTM * wm + 16 * mt + g8 + 8 * rh) * RB + ((WTN / 2) * wn + 4 * nt + q) * 4;
        const uint32_t gw = lds32(sg + off), vw = lds32(sv + off);
        const float s0f = sigmoid(bf16lo(gw)), s1f = sigmoid(bf16hi(gw)), v0 = bf16lo(vw), v1 = bf16hi(vw);
        const float d0 = tl.acc[mt][nt][2 * rh], d1 = tl.acc[mt][nt][2 * rh + 1];
        wdv[mt][nt][rh] = pack_bf16(s0f * d0, s1f * d1);
        wdg[mt][nt][rh] = pack_bf16(d0 * v0 * s0f * (1.f - s0f), d1 * v1 * s1f * (1.f - s1f));
        wa[mt][nt][rh] = pack_bf16(s0f * v0, s1f * v1);
      }
  auto emit = [&](const uint32_t (&w)[MT][NT][2], __nv_bfloat16* dst) {
    __syncthreads();                                     // the g / v tiles (then the previous output tile) live where this one is staged
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) stage_word(s0, Tile::WTM * wm + 16 * mt + g8 + 8 * rh, (WTN / 2) * wn + 4 * nt + q, BN, w[mt][nt][rh]);
    __syncthreads();
    tile_store16<BM, BN, 128>(s0, dst + (size_t)t0 * p.K + n0, p.K, p.M - t0);
  };
  emit(wdv, p.dv);
  emit(wdg, p.dg);
  emit(wa, p.a);
}

// -------------------------------------------------------------------------------------------------------------------------------------- tm2
// Two accumulators over the same output tile (gate = x Wg, projection = y Wo, K = N = D), 8 warps, one CTA per SM (the TriMul's K3w structure).
struct Tm2Params {
  const __nv_bfloat16 *x, *y;               // [M][D]
  const __nv_bfloat16 *wgt, *wot;           // [D][D]: W^T, i.e. [out][in]
  const __nv_bfloat16* gout;                // backward: the output gradient [M][D]; nullptr in the forward
  __nv_bfloat16 *out, *da, *db;             // forward: out; backward: dA (gate logit), dB (projection)
  int M, D;
};

// The backward's epilogue needs the output gradient at every accumulator element: its tile is copied into the (free) ring with coalesced 16-byte cp.async and read from shared
// memory in the staged layout (a global load per element, each followed by its use, serialised one latency per element).
template <int BN>
__global__ void __launch_bounds__(256, BN == 64 ? 2 : 1) tm2_kernel(const Tm2Params p) {
  using TG = WTile<128, BN, 4, 2, 4, false, false>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = smem_u32(smem);
  const int ntn = p.D / BN;
  const int mtile = (int)blockIdx.x / ntn, ntile = (int)blockIdx.x - mtile * ntn;
  const int t0 = mtile * 128, n0 = ntile * BN;
  const bool whole = (p.M & 127) == 0;                   // whole tiles: the minimal-addressing mainloop; a ragged last tile needs run()'s bounds handling
  TG tg;
  const GemmOps og{p.x, (size_t)p.D, t0, p.M, p.wgt, (size_t)p.D, n0, p.D}, ob{p.y, (size_t)p.D, t0, p.M, p.wot, (size_t)p.D, n0, p.D};
  if (whole) tg.run_fast(s0, p.D >> 5, og); else tg.run(s0, p.D >> 5, og);
  TG tb;
  if (whole) tb.run_fast(s0, p.D >> 5, ob); else tb.run(s0, p.D >> 5, ob);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1, g8 = lane >> 2, q = lane & 3;
  constexpr int MT = TG::MT, NT = 2 * TG::NP, WTN = TG::WTN, RB = BN * 2 + 16, GR = BN / 8;
  if (p.gout != nullptr) {
#pragma unroll
    for (int e = 0; e < 128 * GR / 256; ++e) {
      const int idx = tid + 256 * e, r = idx / GR, c = idx % GR;
      const bool ok = t0 + r < p.M;
      cp_async16(s0 + r * RB + c * 16, p.gout + (size_t)(ok ? t0 + r : 0) * p.D + n0 + 8 * c, ok ? 16u : 0u);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
  }
  uint32_t w1[MT][NT][2], w2[MT][NT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const float a0 = tg.acc[mt][nt][2 * rh], a1 = tg.acc[mt][nt][2 * rh + 1], b0 = tb.acc[mt][nt][2 * rh], b1 = tb.acc[mt][nt][2 * rh + 1];
        const float s0f = sigmoid(a0), s1f = sigmoid(a1);
        if (p.gout == nullptr) {
          w1[mt][nt][rh] = pack_bf16(s0f * b0, s1f * b1);
        } else {
          const uint32_t gw = lds32(s0 + (32 * wm + 16 * mt + g8 + 8 * rh) * RB + ((WTN / 2) * wn + 4 * nt + q) * 4);
          const float db0 = bf16lo(gw) * s0f, db1 = bf16hi(gw) * s1f;
          w1[mt][nt][rh] = pack_bf16(db0 * (b0 * (1.f - s0f)), db1 * (b1 * (1.f - s1f)));      // dA
          w2[mt][nt][rh] = pack_bf16(db0, db1);                                                // dB
        }
      }
  auto emit = [&](const uint32_t (&w)[MT][NT][2], __nv_bfloat16* dst) {
    __syncthreads();                                     // the gradient tile (then the previous output tile) lives where this one is staged
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) stage_word(s0, 32 * wm + 16 * mt + g8 + 8 * rh, (WTN / 2) * wn + 4 * nt + q, BN, w[mt][nt][rh]);
    __syncthreads();
    tile_store16<128, BN, 256>(s0, dst + (size_t)t0 * p.D + n0, p.D, p.M - t0);
  };
  if (p.gout == nullptr) {
    emit(w1, p.out);
  } else {
    emit(w1, p.da);
    emit(w2, p.db);
  }
}

// -------------------------------------------------------------------------------------------------------------------------------------- tm1
// One GEMM over the packed weights (rows 16 j + 0..7 gate | 16 j + 8..15 projection of channels 8 j .. 8 j + 7; channels [0, D) left, [D, 2 D) right), the
// 128 x 128 tile of the TriMul's front (64 channels of one side per CTA), token-major outputs.
struct Tm1Params {
  const __nv_bfloat16* x;                   // [M][D]
  const __nv_bfloat16* w1;                  // [4 D][D] packed rows
  __nv_bfloat16 *left, *right;              // forward outputs [M][D]
  const __nv_bfloat16 *gl, *gr;             // backward: the output gradients [M][D]; nullptr in the forward
  __nv_bfloat16 *dla, *dlb, *dra, *drb;     // backward: (gate-logit, projection) gradients of the left and the right branch
  int M, D;
};

// (the backward's output-gradient tile is prefetched into the free ring with coalesced cp.async, as in tm2)
__global__ void __launch_bounds__(128, 2) tm1_kernel(const Tm1Params p) {
  using Tile = WTile<128, 128, 2, 2, 4, false, false>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = smem_u32(smem);
  const int ntn = (4 * p.D) / 128;
  const int mtile = (int)blockIdx.x / ntn, ntile = (int)blockIdx.x - mtile * ntn;
  const int t0 = mtile * 128, j0 = ntile * 128;
  const int pc0 = ntile * 64, side = pc0 >= p.D ? 1 : 0, c0 = pc0 - side * p.D;
  Tile tl;
  const GemmOps ops{p.x, (size_t)p.D, t0, p.M, p.w1, (size_t)p.D, j0, 4 * p.D};
  if ((p.M & 127) == 0) tl.run_fast(s0, p.D >> 5, ops); else tl.run(s0, p.D >> 5, ops);     // whole tiles: the minimal-addressing mainloop
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1, g8 = lane >> 2, q = lane & 3;
  constexpr int MT = Tile::MT, NP = Tile::NP, RB = 64 * 2 + 16;
  const bool bwd = p.gl != nullptr;
  if (bwd) {
    const __nv_bfloat16* gsrc = side ? p.gr : p.gl;
#pragma unroll
    for (int e = 0; e < 128 * 8 / 128; ++e) {
      const int idx = tid + 128 * e, r = idx >> 3, c = idx & 7;
      const bool ok = t0 + r < p.M;
      cp_async16(s0 + r * RB + c * 16, gsrc + (size_t)(ok ? t0 + r : 0) * p.D + c0 + 8 * c, ok ? 16u : 0u);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
  }
  uint32_t wu[MT][NP][2], wv[MT][NP][2];                 // forward: u;  backward: dA (gate logit), dB (projection)
#pragma unroll
  for (int np = 0; np < NP; ++np)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        float ga[2], pb[2];
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) { ga[cc] = tl.acc[mt][2 * np][2 * rh + cc]; pb[cc] = tl.acc[mt][2 * np + 1][2 * rh + cc]; }
        const float s0f = sigmoid(ga[0]), s1f = sigmoid(ga[1]);
        if (!bwd) {
          wu[mt][np][rh] = pack_bf16(s0f * pb[0], s1f * pb[1]);
        } else {
          const uint32_t gw = lds32(s0 + (64 * wm + 16 * mt + g8 + 8 * rh) * RB + (16 * wn + 4 * np + q) * 4);
          const float db0 = bf16lo(gw) * s0f, db1 = bf16hi(gw) * s1f;
          wu[mt][np][rh] = pack_bf16(db0 * pb[0] * (1.f - s0f), db1 * pb[1] * (1.f - s1f));    // dA
          wv[mt][np][rh] = pack_bf16(db0, db1);                                                   // dB
        }
      }
  auto emit = [&](const uint32_t (&w)[MT][NP][2], __nv_bfloat16* dst) {
    __syncthreads();                                     // the gradient tile (then the previous output tile) lives where this one is staged
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int np = 0; np < NP; ++np)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) stage_word(s0, 64 * wm + 16 * mt + g8 + 8 * rh, 16 * wn + 4 * np + q, 64, w[mt][np][rh]);
    __syncthreads();
    tile_store16<128, 64, 128>(s0, dst + (size_t)t0 * p.D + c0, p.D, p.M - t0);
  };
  if (!bwd) {
    emit(wu, side ? p.right : p.left);
  } else {
    emit(wu, side ? p.dra : p.dla);
    emit(wv, side ? p.drb : p.dlb);
  }
}

// --------------------------------------------------------------------------------------------------------------------------------- elementwise
// 16-byte vectors of 8 bf16; the < 8 trailing elements are handled by the first threads.
DEVI void unpack8w(uint4 v, float (&f)[8]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) { f[2 * j] = bf16lo(w[j]); f[2 * j + 1] = bf16hi(w[j]); }
}
DEVI uint4 pack8w(const float (&f)[8]) { return make_uint4(pack_bf16(f[0], f[1]), pack_bf16(f[2], f[3]), pack_bf16(f[4], f[5]), pack_bf16(f[6], f[7])); }
DEVI float bf16_at(const __nv_bfloat16* p, size_t i) { return __bfloat162float(p[i]); }

// a = bf16(sigmoid(g) o)
__global__ void __launch_bounds__(256) sigmul_fwd_kernel(const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ o, __nv_bfloat16* __restrict__ a, size_t n) {
  const size_t n8 = n >> 3, stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride) {
    float fg[8], fo[8], r[8];
    unpack8w(__ldg(reinterpret_cast<const uint4*>(g) + i), fg);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(o) + i), fo);
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = sigmoid(fg[j]) * fo[j];
    reinterpret_cast<uint4*>(a)[i] = pack8w(r);
  }
  for (size_t i = (n8 << 3) + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    a[i] = __float2bfloat16_rn(sigmoid(bf16_at(g, i)) * bf16_at(o, i));
}

// dg = da o s (1 - s),  do = da s
__global__ void __launch_bounds__(256) sigmul_bwd_kernel(const __nv_bfloat16* __restrict__ da, const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ o,
                                                          __nv_bfloat16* __restrict__ dg, __nv_bfloat16* __restrict__ dd, size_t n) {
  const size_t n8 = n >> 3, stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride) {
    float fa[8], fg[8], fo[8], rg[8], rd[8];
    unpack8w(__ldg(reinterpret_cast<const uint4*>(da) + i), fa);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(g) + i), fg);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(o) + i), fo);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float s = sigmoid(fg[j]);
      rd[j] = fa[j] * s;
      rg[j] = fa[j] * fo[j] * s * (1.f - s);
    }
    reinterpret_cast<uint4*>(dg)[i] = pack8w(rg);
    reinterpret_cast<uint4*>(dd)[i] = pack8w(rd);
  }
  for (size_t i = (n8 << 3) + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float s = sigmoid(bf16_at(g, i)), a = bf16_at(da, i);
    dd[i] = __float2bfloat16_rn(a * s);
    dg[i] = __float2bfloat16_rn(a * bf16_at(o, i) * s * (1.f - s));
  }
}

// y = bf16(x + bf16(g b))   (the eager product rounding before the residual add)
__global__ void __launch_bounds__(256) gres_fwd_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ b,
                                                        __nv_bfloat16* __restrict__ y, size_t n) {
  const size_t n8 = n >> 3, stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride) {
    float fx[8], fg[8], fb[8], r[8];
    unpack8w(__ldg(reinterpret_cast<const uint4*>(x) + i), fx);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(g) + i), fg);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(b) + i), fb);
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = fx[j] + round_bf16f(fg[j] * fb[j]);
    reinterpret_cast<uint4*>(y)[i] = pack8w(r);
  }
  for (size_t i = (n8 << 3) + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    y[i] = __float2bfloat16_rn(bf16_at(x, i) + round_bf16f(bf16_at(g, i) * bf16_at(b, i)));
}

// dg = dy b,  db = dy g
__global__ void __launch_bounds__(256) gres_bwd_kernel(const __nv_bfloat16* __restrict__ dy, const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ b,
                                                        __nv_bfloat16* __restrict__ dg, __nv_bfloat16* __restrict__ db, size_t n) {
  const size_t n8 = n >> 3, stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride) {
    float fy[8], fg[8], fb[8], rg[8], rb[8];
    unpack8w(__ldg(reinterpret_cast<const uint4*>(dy) + i), fy);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(g) + i), fg);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(b) + i), fb);
#pragma unroll
    for (int j = 0; j < 8; ++j) { rg[j] = fy[j] * fb[j]; rb[j] = fy[j] * fg[j]; }
    reinterpret_cast<uint4*>(dg)[i] = pack8w(rg);
    reinterpret_cast<uint4*>(db)[i] = pack8w(rb);
  }
  for (size_t i = (n8 << 3) + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    dg[i] = __float2bfloat16_rn(bf16_at(dy, i) * bf16_at(b, i));
    db[i] = __float2bfloat16_rn(bf16_at(dy, i) * bf16_at(g, i));
  }
}


// ---- the TriMul output gate (trimul_inproj/triton/gate_elem.py): forward y = bf16(res + ds[m % L] (p sigmoid(g))) and the saved gate; backward (d_proj, d_glogit)
__global__ void __launch_bounds__(256) gate_elem_fwd_kernel(const __nv_bfloat16* __restrict__ glogit, const __nv_bfloat16* __restrict__ proj, const __nv_bfloat16* __restrict__ res,
                                                             const __nv_bfloat16* __restrict__ ds, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ gate, size_t M, int N, int L) {
  const int g8 = N >> 3;
  const size_t n8 = M * g8, stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride) {
    const size_t m = i / g8;
    const int c = (int)(i - m * g8);
    float fg[8], fp[8], fr[8], fd[8], ry[8], rg[8];
    unpack8w(__ldg(reinterpret_cast<const uint4*>(glogit) + i), fg);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(proj) + i), fp);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(res) + i), fr);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(ds) + (size_t)(m % L) * g8 + c), fd);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float s = sigmoid(fg[j]);
      ry[j] = fp[j] * s * fd[j] + fr[j];
      rg[j] = s;
    }
    reinterpret_cast<uint4*>(y)[i] = pack8w(ry);
    reinterpret_cast<uint4*>(gate)[i] = pack8w(rg);
  }
}

// d_proj = dy ds gate;  d_glogit = dy ds proj gate (1 - gate)   (from_preact: ``gate`` holds the pre-activation and the sigmoid is taken here)
__global__ void __launch_bounds__(256) gate_elem_bwd_kernel(const __nv_bfloat16* __restrict__ dy, const __nv_bfloat16* __restrict__ proj, const __nv_bfloat16* __restrict__ gate,
                                                             const __nv_bfloat16* __restrict__ ds, __nv_bfloat16* __restrict__ dproj, __nv_bfloat16* __restrict__ dglogit, size_t M,
                                                             int N, int L, int from_preact) {
  const int g8 = N >> 3;
  const size_t n8 = M * g8, stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride) {
    const size_t m = i / g8;
    const int c = (int)(i - m * g8);
    float fy[8], fp[8], fg[8], fd[8], rp[8], rg[8];
    unpack8w(__ldg(reinterpret_cast<const uint4*>(dy) + i), fy);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(proj) + i), fp);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(gate) + i), fg);
    unpack8w(__ldg(reinterpret_cast<const uint4*>(ds) + (size_t)(m % L) * g8 + c), fd);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float s = from_preact ? sigmoid(fg[j]) : fg[j];
      const float d = fy[j] * fd[j];
      rp[j] = d * s;
      rg[j] = d * fp[j] * s * (1.f - s);
    }
    reinterpret_cast<uint4*>(dproj)[i] = pack8w(rp);
    reinterpret_cast<uint4*>(dglogit)[i] = pack8w(rg);
  }
}

}  // namespace a100
