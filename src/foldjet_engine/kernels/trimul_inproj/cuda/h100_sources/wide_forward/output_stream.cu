// SPDX-License-Identifier: Apache-2.0
// MiniWorld wide inference K3 (bidirectional, H = 2 D): streamed-operand GEMM with both LayerNorms folded into the
// weights.  Uses the engine's TMA/mbarrier/WGMMA primitives (tmn_kernels.cuh).
//
//   y[t, c] = x[t, c] + bf16(P[t, c]) * sigmoid(bf16(G[t, c]))
//   P = LN_out(tri[:, t]) . Wp[c] = rs_t (tri[:, t] . Wpf[c] - mu_t up[c]) + vp[c]
//       Wpf = bf16(Wp * gamma_out), up = sum_k Wpf[c, k], vp = Wp[c] . beta_out           (host fold, per call)
//   G = LN_in(x[t]) . Wg[c] = rsx_t (x[t] . Wgf[c] - mux_t ug[c]) + vg[c]                 (GFOLD; K1 / LN hand over (mux, rsx))
//     = xn[t] . Wg[c]                                                                     (!GFOLD; K1 hands over xn)
// tri is read in its native channel-major layout as an MN-major A operand: no transpose and no normalise pass.  The
// per-token output-LN statistics accumulate from the same shared-memory chunks (shifted one-pass sums) during the
// tile's first column block.  A CTA owns 128 tokens (two consumer warpgroups of 64) and walks the output in column
// blocks of BN; one producer thread streams [128 tok x 64 k] operand chunks with the matching [BN x 64 k] weight chunk
// through an NS-deep ring.  Epilogue: each warpgroup's x tile [64 tok x BN] arrives by TMA during the projection stream,
// y is written over it in shared memory and leaves by TMA store.
#include "tmn_kernels.cuh"
#include "wgmma_n.cuh"
using namespace tmn;
using namespace tmn::sm90;
using bf = __nv_bfloat16;
#ifndef GFOLD
#define GFOLD 0
#endif
constexpr int D = WIDTH, H = 2 * D, BN = BLOCKN, NS = STAGES, MT = 128;
constexpr int NCB = D / BN, KG = D / 64, KP = H / 64;
constexpr int ABYTES = MT * 128, WBYTES = BN * 128, STAGE = ABYTES + WBYTES;
constexpr int XBYTES = 64 * BN * 2;                              // per warpgroup x / y tile
constexpr int NACC = BN / 2;
static_assert(D % BN == 0 && (BN == 128 || BN == 192 || BN == 256), "column block");
// xa: gate A operand (x with GFOLD, else xn), box [64 k][128 tok]; wp, wg: [BN n][64 k] weight chunks; xm, ym: [64][64]
// uv: fp32 [4][D] = up, vp, ug, vg; xs: fp32 [M][2] = (mean, rstd) of LN_in (GFOLD)
struct Params { CUtensorMap tri, xa, wp, wg, xm, ym; const float* uv; const float* xs; int M; int tiles; };

TMN_DEVI uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0,[%1];" : "=r"(v) : "r"(a)); return v; }
template <int TA> TMN_DEVI void mma_n(float (&d)[NACC], uint64_t a, uint64_t b, int acc) {
  if constexpr (BN == 128) mw_mma128<TA>(d, a, b, acc);
  else if constexpr (BN == 192) mw_mma192<TA>(d, a, b, acc);
  else mw_mma256<TA>(d, a, b, acc);
}
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}

extern "C" __global__ void __launch_bounds__(384, 1) mw_wide_output_stream(__grid_constant__ const Params p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  uint8_t* ring = sm;
  uint8_t* xbuf = sm + NS * STAGE;                               // [2 wg][BN/64 chunks][64 tok][128 B]
  float* su = reinterpret_cast<float*>(xbuf + 2 * XBYTES);       // [4][D]
  const float* sv = su + D; const float* sug = su + 2 * D; const float* svg = su + 3 * D;
  float* red = su + 4 * D;                                       // [2 wg][4 grp][64 tok][2]  partial sums
  float* sk = red + 2 * 4 * 64 * 2;                              // [2 wg][64 tok] shift
  float* smu = sk + 2 * 64;                                      // [2 wg][64 tok] -rs*mu
  float* srs = smu + 2 * 64;                                     // [2 wg][64 tok] rs
  uint64_t* full = reinterpret_cast<uint64_t*>(srs + 2 * 64);
  uint64_t* empty = full + NS;
  uint64_t* xfull = empty + NS;                                  // [2 wg]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wg = __shfl_sync(0xffffffffu, tid >> 7, 0);
  for (int i = tid; i < (GFOLD ? 4 : 2) * D; i += 384) su[i] = p.uv[i];
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 8); }
    mbar_init(xfull, 1); mbar_init(xfull + 1, 1);
    fence_barrier_init();
    tma_prefetch_desc(&p.tri); tma_prefetch_desc(&p.xa); tma_prefetch_desc(&p.wp); tma_prefetch_desc(&p.wg);
    tma_prefetch_desc(&p.xm); tma_prefetch_desc(&p.ym);
  }
  __syncthreads();

  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (warp == 0 && lane == 0) {
      uint32_t it = 0;
      for (int tile = blockIdx.x; tile < p.tiles; tile += gridDim.x) {
        const int tok0 = tile * MT;
        for (int cb = 0; cb < NCB; ++cb) {
          for (int kc = 0; kc < KG + KP; ++kc, ++it) {
            const int s = it % NS; const uint32_t u = it / NS;
            if (u > 0) mbar_wait(empty + s, (u - 1) & 1);
            mbar_arrive_expect_tx(full + s, STAGE);
            uint8_t* a = ring + s * STAGE;
            if (kc < KG) {
              tma_load_2d(a + ABYTES, &p.wg, full + s, kc * 64, cb * BN);
              tma_load_2d(a, &p.xa, full + s, kc * 64, tok0);
            } else {
              const int k = kc - KG;
              tma_load_2d(a + ABYTES, &p.wp, full + s, k * 64, cb * BN);
              tma_load_2d(a, &p.tri, full + s, tok0, k * 64);
              tma_load_2d(a + 8192, &p.tri, full + s, tok0 + 64, k * 64);
            }
          }
        }
      }
    }
    __syncwarp();
    return;
  }

  setmaxnreg_inc<232>();
  const int w = wg - 1, wiw = warp & 3;
  const int rowA = 16 * wiw + (lane >> 2), rowB = rowA + 8;       // WG-relative accumulator rows
  const int q2 = 2 * (lane & 3);
  const bool elect = wiw == 0 && lane == 0;
  const uint32_t ring_u = smem_u32(ring), xb_u = smem_u32(xbuf) + w * XBYTES;
  uint32_t it = 0, xph = 0;
  float acc[NACC];
  uint32_t gp[NACC / 2];
  auto release = [&](uint32_t i) { fence_proxy_async(); __syncwarp(); if (lane == 0) mbar_arrive(empty + (i % NS)); };

  for (int tile = blockIdx.x; tile < p.tiles; tile += gridDim.x) {
    const int tok0 = tile * MT + 64 * w;
    float s1a = 0.f, s2a = 0.f, s1b = 0.f, s2b = 0.f, ka = 0.f, kb = 0.f;
    float2 xsA = make_float2(0.f, 1.f), xsB = xsA;
    if constexpr (GFOLD) { xsA = ldg64f(p.xs + 2 * (size_t)(tok0 + rowA)); xsB = ldg64f(p.xs + 2 * (size_t)(tok0 + rowB)); }
    for (int cb = 0; cb < NCB; ++cb) {
      // ---- gate: G = xa . Wg^T (K = D)
      for (int kc = 0; kc < KG; ++kc, ++it) {
        const int s = it % NS;
        mbar_wait(full + s, (it / NS) & 1);
        const uint32_t a = ring_u + s * STAGE + 8192 * w, b = ring_u + s * STAGE + ABYTES;
        fence_regs(acc); wgmma_fence();
#pragma unroll
        for (int q = 0; q < 4; ++q) mma_n<0>(acc, smem_desc(a + 32 * q, 16, 1024, 1), smem_desc(b + 32 * q, 16, 1024, 1), kc > 0 || q > 0);
        wgmma_commit();
        if (kc > 0) { wgmma_wait<1>(); fence_regs(acc); release(it - 1); }
      }
      wgmma_wait<0>(); fence_regs(acc); release(it - 1);
      if constexpr (GFOLD) {                            // G = rsx (x . Wgf - mux ug) + vg, rounded to bf16
        const float xrA = xsA.y, xnA = -xsA.y * xsA.x, xrB = xsB.y, xnB = -xsB.y * xsB.x;
#pragma unroll
        for (int g = 0; g < NACC / 4; ++g) {
          const int c = cb * BN + 8 * g + q2;
          const float2 uu = *reinterpret_cast<const float2*>(sug + c), vv = *reinterpret_cast<const float2*>(svg + c);
          gp[2 * g] = pack_bf16(fmaf(xrA, acc[4 * g], fmaf(xnA, uu.x, vv.x)), fmaf(xrA, acc[4 * g + 1], fmaf(xnA, uu.y, vv.y)));
          gp[2 * g + 1] = pack_bf16(fmaf(xrB, acc[4 * g + 2], fmaf(xnB, uu.x, vv.x)), fmaf(xrB, acc[4 * g + 3], fmaf(xnB, uu.y, vv.y)));
        }
      } else {
#pragma unroll
        for (int g = 0; g < NACC / 4; ++g) { gp[2 * g] = pack_bf16(acc[4 * g], acc[4 * g + 1]); gp[2 * g + 1] = pack_bf16(acc[4 * g + 2], acc[4 * g + 3]); }
      }
      // the residual tile: this warpgroup's previous y store has left shared memory -> load x [64 tok][BN] over it
      if (elect) {
        tma_store_wait_read<0>();
        mbar_arrive_expect_tx(xfull + w, XBYTES);
#pragma unroll
        for (int j = 0; j < BN / 64; ++j) tma_load_2d(xbuf + w * XBYTES + j * 8192, &p.xm, xfull + w, cb * BN + 64 * j, tok0);
      }
      // ---- projection: raw tri (MN-major) . Wpf^T (K = H); output-LN statistics on the tile's first column block
      for (int kc = 0; kc < KP; ++kc, ++it) {
        const int s = it % NS;
        mbar_wait(full + s, (it / NS) & 1);
        const uint32_t a = ring_u + s * STAGE + 8192 * w, b = ring_u + s * STAGE + ABYTES;
        fence_regs(acc); fence_regs(gp); wgmma_fence();
#pragma unroll
        for (int q = 0; q < 4; ++q) mma_n<1>(acc, smem_desc(a + 2048 * q, 16, 1024, 1), smem_desc(b + 32 * q, 16, 1024, 1), kc > 0 || q > 0);
        wgmma_commit();
        if (cb == 0) {                                   // warp wiw: channels 16 wiw .. +15 of the chunk, tokens 2 lane, 2 lane + 1
          if (kc == 0) { const uint32_t v0 = lds32(a + swz128(0, 4 * lane)); ka = bf16lo(v0); kb = bf16hi(v0); }
#pragma unroll
          for (int c = 0; c < 16; ++c) {
            const uint32_t v = lds32(a + swz128(16 * wiw + c, 4 * lane));
            const float da = bf16lo(v) - ka, db = bf16hi(v) - kb;
            s1a += da; s2a = fmaf(da, da, s2a); s1b += db; s2b = fmaf(db, db, s2b);
          }
        }
        if (kc > 0) { wgmma_wait<1>(); fence_regs(acc); release(it - 1); }
      }
      wgmma_wait<0>(); fence_regs(acc); release(it - 1);
      if (cb == 0) {
        float* r = red + ((w * 4 + wiw) * 64) * 2;
        r[4 * lane + 0] = s1a; r[4 * lane + 1] = s2a; r[4 * lane + 2] = s1b; r[4 * lane + 3] = s2b;
        if (wiw == 0) { sk[w * 64 + 2 * lane] = ka; sk[w * 64 + 2 * lane + 1] = kb; }
        named_bar_sync(1 + w, 128);
        if (tid - 128 * wg < 64) {
          const int t = tid - 128 * wg;
          float S1 = 0.f, S2 = 0.f;
#pragma unroll
          for (int g = 0; g < 4; ++g) { S1 += red[((w * 4 + g) * 64 + t) * 2]; S2 += red[((w * 4 + g) * 64 + t) * 2 + 1]; }
          const float m1 = S1 * (1.f / H), var = fmaxf(S2 * (1.f / H) - m1 * m1, 0.f);
          const float rs = rsqrtf(var + 1e-5f), mu = sk[w * 64 + t] + m1;
          srs[w * 64 + t] = rs; smu[w * 64 + t] = -rs * mu;
        }
        named_bar_sync(1 + w, 128);
      }
      // ---- epilogue: y = x + bf16(P) * sigmoid(bf16(G)), in place over the x tile, then one TMA store per 64 columns
      const float rsA = srs[w * 64 + rowA], nmA = smu[w * 64 + rowA], rsB = srs[w * 64 + rowB], nmB = smu[w * 64 + rowB];
      mbar_wait(xfull + w, xph); xph ^= 1;
#pragma unroll
      for (int g = 0; g < NACC / 4; ++g) {
        const int cc = 8 * g + q2, c = cb * BN + cc;
        const float2 uu = *reinterpret_cast<const float2*>(su + c), vv = *reinterpret_cast<const float2*>(sv + c);
        const uint32_t ad = xb_u + (cc / 64) * 8192 + swz128(rowA, (cc % 64) * 2), bd = xb_u + (cc / 64) * 8192 + swz128(rowB, (cc % 64) * 2);
        const uint32_t xa = lds32(ad), xb = lds32(bd);
        const float pa0 = math::round_bf16(fmaf(rsA, acc[4 * g + 0], fmaf(nmA, uu.x, vv.x)));
        const float pa1 = math::round_bf16(fmaf(rsA, acc[4 * g + 1], fmaf(nmA, uu.y, vv.y)));
        const float pb0 = math::round_bf16(fmaf(rsB, acc[4 * g + 2], fmaf(nmB, uu.x, vv.x)));
        const float pb1 = math::round_bf16(fmaf(rsB, acc[4 * g + 3], fmaf(nmB, uu.y, vv.y)));
        const float ya0 = bf16lo(xa) + pa0 * math::sigmoid(bf16lo(gp[2 * g])), ya1 = bf16hi(xa) + pa1 * math::sigmoid(bf16hi(gp[2 * g]));
        const float yb0 = bf16lo(xb) + pb0 * math::sigmoid(bf16lo(gp[2 * g + 1])), yb1 = bf16hi(xb) + pb1 * math::sigmoid(bf16hi(gp[2 * g + 1]));
        sts32(ad, pack_bf16(ya0, ya1));
        sts32(bd, pack_bf16(yb0, yb1));
      }
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      if (elect) {
#pragma unroll
        for (int j = 0; j < BN / 64; ++j) tma_store_2d(&p.ym, xbuf + w * XBYTES + j * 8192, cb * BN + 64 * j, tok0);
        tma_store_commit();
      }
    }
  }
  if (elect) tma_store_wait_all();
}
