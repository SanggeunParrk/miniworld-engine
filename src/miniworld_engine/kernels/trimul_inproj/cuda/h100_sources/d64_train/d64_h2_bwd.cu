// SPDX-License-Identifier: Apache-2.0
// Bidirectional TriMul training backward at pair width D = 64 with hidden width 128 PER DIRECTION (H = 256 contraction channels, the
// output LayerNorm normalises 256 channels), sm_90a, wgmma + TMA.  Derived from d64_bwd_wg_gatepipe.cu (hidden 64 per direction, H = 128):
// the registers of that kernel hold a whole 128-channel row of dxhat and the dWp partial, so the 256-channel block is split by channel
// halves; the only coupling between the halves is the output LayerNorm (the row sums s1, s2 and the projection p), which a first
// kernel computes for the whole row.
//
//   d64h2_b1a   (per 64-token tile; all 256 channels)   nhat = (tri - mu_out) r_out, xhat; p = nhat (Wp o go)^T + Wp bo, g = xhat (Wg o gi)^T
//               dp = dy ds s, dgl = dp p (1 - s); s1 = sum_c dxhat / 256 = dp . rowsum(Wp o go) / 256, s2 = sum_c dxhat nhat / 256 = dp . (p - pb) / 256
//               -> dp (bf16 [M][64]), dgl (bf16 [M][64]), s1 / s2 (fp32 [2][M]), partial dWg = Gg o gi + dglsum bi^T
//   d64h2_b1b   (per channel half; one launch per half)   dxhat_h = dp (Wp o go)_h ; G_h += dp^T nhat_h ; dpsum
//               -> dt_h = r_out (dxhat_h - s1 - nhat_h s2) (TMA store at channel base cb), partial dWp_h | dgo_h | dbo_h
//   d64h2_b2w   (per channel half, PASS 0 / 1)  the input side of the original B2 on the channels [cb, cb + 128) of both planes: pre = xn W1_h^T,
//               dxn partial = sum_b dpre_b W1_h[b] (+ the gate part dgl Wg in PASS 0), dW1_h.  PASS 0 stores the row's dxn (fp32 [M][64]);
//               PASS 1 adds it, then the input-LayerNorm backward with the identity residual (dx, dgamma, dbeta).
//   d64h2_fin   fixed-order reduction of the per-CTA partials into the gradients.
#include "d64_wg.cuh"

#ifndef PASS
#define PASS 0
#endif

namespace d64 {

// =====================================================================================================================
// B1a: whole row (256 channels), outputs dp / dgl / s1 / s2
// =====================================================================================================================
struct B1aParams {
  CUtensorMap tm_tri, tm_x, tm_dy, tm_ds, tm_st, tm_dp, tm_dgl;
  const bf16* wp;      // [64 out][256 in]
  const bf16* wg;      // [64 out][64 in]
  const float *go, *bo, *gi, *bi;
  float* part;         // [grid][ha::SLOT]: dWg [64][64]
  float* s12;          // [2][M]: s1, s2
  int ntiles, n, m;
};
namespace ha {
constexpr int WP = 0;               // 4 x [64 out][64 ch] (Wp o go): K-major B of p
constexpr int WGS = WP + 32768;     // [64 out][64 in] (Wg o gi) / 2: K-major B of g / 2
constexpr int ONES = WGS + 8192;    // [8][64] ones: K-major B of the row sums
constexpr int BIAS = ONES + 1024;   // pb[64] = Wp bo, gb[64] = Wg bi / 2, wsum[64] = rowsum(Wp o go)
constexpr int BAR = BIAS + 1024;    // [2 wg] full barriers
constexpr int RING = 44032;         // per-warpgroup region (1024-aligned)
// stage: TRI [256 ch][64 tok] (-> nhat), X [64 tok][64] (-> xhat), DY [64][64] (-> dgl), DS [64][64], ST [4][64] f32; then DP [64 tok][64 out]
constexpr int TRI = 0, XS = 32768, DYS = 40960, DSS = 49152, STS = 57344, STAGE = 58368;
constexpr int DP = STAGE;
constexpr int WGSZ = DP + 8192;
constexpr int SMEM = RING + 2 * WGSZ;
constexpr int SLOT = 4096;
constexpr uint32_t TX = 32768 + 3 * 8192 + 1024;
}  // namespace ha

extern "C" __global__ void __launch_bounds__(256, 1) d64h2_b1a(const __grid_constant__ B1aParams P) {
  using namespace ha;
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, wtid = tid & 127, w = wtid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;
  const int r = __shfl_sync(0xffffffffu, tid >> 7, 0);   // warp-uniform for ptxas
  const uint32_t sb = smem_u32(smem);
  const uint32_t ring = sb + RING + r * WGSZ;
  const uint32_t bar0 = sb + BAR + r * 8;
  float* sBias = reinterpret_cast<float*>(smem + BIAS);
  const bool leader = wtid == 0;

  // ---- resident operands (affine and the sigmoid's 1/2 folded into the weights); 8-element granules
#pragma unroll
  for (int it = 0; it < 8; ++it) {                        // Wp o go: [64 out][256 ch] -> four [64][64] tiles
    const int i = tid + 256 * it, o = i >> 5, c = (i & 31) * 8;
    const uint4 wv = *reinterpret_cast<const uint4*>(P.wp + o * 256 + c);
    const float4 g0 = *reinterpret_cast<const float4*>(P.go + c), g1 = *reinterpret_cast<const float4*>(P.go + c + 4);
    const uint4 ov = make_uint4(pack(lo(wv.x) * g0.x, hi(wv.x) * g0.y), pack(lo(wv.y) * g0.z, hi(wv.y) * g0.w),
                                pack(lo(wv.z) * g1.x, hi(wv.z) * g1.y), pack(lo(wv.w) * g1.z, hi(wv.w) * g1.w));
    *reinterpret_cast<uint4*>(smem + WP + (c >> 6) * 8192 + swz(o, (c & 63) * 2)) = ov;
  }
#pragma unroll
  for (int it = 0; it < 2; ++it) {                        // (Wg o gi) / 2, Wg: [64 out][64 in]
    const int i = tid + 256 * it, o = i >> 3, c = (i & 7) * 8;
    const uint4 wv = *reinterpret_cast<const uint4*>(P.wg + o * 64 + c);
    float4 g0 = *reinterpret_cast<const float4*>(P.gi + c), g1 = *reinterpret_cast<const float4*>(P.gi + c + 4);
    const uint4 ov = make_uint4(pack(0.5f * (lo(wv.x) * g0.x), 0.5f * (hi(wv.x) * g0.y)),
                                pack(0.5f * (lo(wv.y) * g0.z), 0.5f * (hi(wv.y) * g0.w)),
                                pack(0.5f * (lo(wv.z) * g1.x), 0.5f * (hi(wv.z) * g1.y)),
                                pack(0.5f * (lo(wv.w) * g1.z), 0.5f * (hi(wv.w) * g1.w)));
    *reinterpret_cast<uint4*>(smem + WGS + swz(o, c * 2)) = ov;
  }
  if (tid < 64) reinterpret_cast<uint4*>(smem + ONES)[tid] = make_uint4(0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u);
  {                                                       // pb = Wp bo, gb = Wg bi / 2, wsum: warp w8 rows 8 w8 .. 8 w8 + 7
    const int w8 = tid >> 5;
    const float2 c2 = *reinterpret_cast<const float2*>(P.bi + 2 * lane);
#pragma unroll
    for (int rr = 0; rr < 8; ++rr) {
      const int o = 8 * w8 + rr;
      float sp = 0.f, sw = 0.f;
#pragma unroll
      for (int k = 0; k < 2; ++k) {                       // channels 128 k + 4 lane .. + 3
        const int c = 128 * k + 4 * lane;
        const float4 b4 = *reinterpret_cast<const float4*>(P.bo + c);
        const float4 o4 = *reinterpret_cast<const float4*>(P.go + c);
        const uint2 wv = *reinterpret_cast<const uint2*>(P.wp + o * 256 + c);
        sp += fmaf(lo(wv.x), b4.x, fmaf(hi(wv.x), b4.y, fmaf(lo(wv.y), b4.z, hi(wv.y) * b4.w)));
        const uint32_t r0 = pack(lo(wv.x) * o4.x, hi(wv.x) * o4.y), r1 = pack(lo(wv.y) * o4.z, hi(wv.y) * o4.w);
        sw += lo(r0) + hi(r0) + lo(r1) + hi(r1);
      }
      const uint32_t gv = *reinterpret_cast<const uint32_t*>(P.wg + o * 64 + 2 * lane);
      float sg = fmaf(lo(gv), c2.x, hi(gv) * c2.y);
#pragma unroll
      for (int m = 16; m; m >>= 1) {
        sp += __shfl_xor_sync(0xffffffffu, sp, m);
        sg += __shfl_xor_sync(0xffffffffu, sg, m);
        sw += __shfl_xor_sync(0xffffffffu, sw, m);
      }
      if (lane == 0) { sBias[o] = sp; sBias[64 + o] = 0.5f * sg; sBias[128 + o] = sw; }
    }
  }
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) mbar_init(sb + BAR + 8 * i, 1);
    fence_barrier_init();
  }
  fence_async();
  __syncthreads();

  const int v = 2 * blockIdx.x + r, vstride = 2 * gridDim.x;
  auto load = [&](int tile) {
    const uint32_t s = ring, bar = bar0;
    // batched samples (tokens b-major, M = B n n): the ds row of token t0 is sample (t0 / n n) row j (ds is [B n, 64]); 0 for B = 1
    const int t0 = tile * 64, j0 = t0 % P.n + t0 / (P.n * P.n) * P.n;
    mbar_expect(bar, TX);
    tma_load(s + TRI, &P.tm_tri, bar, t0, 0);
    tma_load(s + TRI + 8192, &P.tm_tri, bar, t0, 64);
    tma_load(s + TRI + 16384, &P.tm_tri, bar, t0, 128);
    tma_load(s + TRI + 24576, &P.tm_tri, bar, t0, 192);
    tma_load(s + XS, &P.tm_x, bar, 0, t0);
    tma_load(s + DYS, &P.tm_dy, bar, 0, t0);
    tma_load(s + DSS, &P.tm_ds, bar, 0, j0);
    tma_load(s + STS, &P.tm_st, bar, t0, 0);
  };
  if (leader) {
    prefetch_map(&P.tm_tri); prefetch_map(&P.tm_x); prefetch_map(&P.tm_dy); prefetch_map(&P.tm_ds);
    prefetch_map(&P.tm_st); prefetch_map(&P.tm_dp); prefetch_map(&P.tm_dgl);
    if (v < P.ntiles) load(v);
  }

  float Gg[32], dgs[4];
#pragma unroll
  for (int i = 0; i < 32; ++i) Gg[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) dgs[i] = 0.f;
  const int rA = 16 * w + gq, rB = rA + 8;

  int k = 0;
  for (int tile = v; tile < P.ntiles; tile += vstride, ++k) {
    const uint32_t s = ring;
    mbar_wait(bar0, k & 1);
    const float* sSt = reinterpret_cast<const float*>(smem + (s - sb) + STS);
    const float miA = sSt[rA], riA = sSt[64 + rA], miB = sSt[rB], riB = sSt[64 + rB];
    const float moA = sSt[128 + rA], roA = sSt[192 + rA], moB = sSt[128 + rB], roB = sSt[192 + rB];
    // ---- xhat, nhat in place (bf16)
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t f[4];
      const uint32_t a = afrag(s + XS, 16 * w, 16 * ks, lane);
      ldsm4(f, a);
      f[0] = pack(fmaf(lo(f[0]), riA, -miA * riA), fmaf(hi(f[0]), riA, -miA * riA));
      f[1] = pack(fmaf(lo(f[1]), riB, -miB * riB), fmaf(hi(f[1]), riB, -miB * riB));
      f[2] = pack(fmaf(lo(f[2]), riA, -miA * riA), fmaf(hi(f[2]), riA, -miA * riA));
      f[3] = pack(fmaf(lo(f[3]), riB, -miB * riB), fmaf(hi(f[3]), riB, -miB * riB));
      stsm4(a, f);
    }
#pragma unroll
    for (int kc = 0; kc < 16; ++kc) {
      uint32_t f[4];
      const uint32_t a = afrag_t(s + TRI, 16 * kc, 16 * w, lane);
      ldsm4t(f, a);
      f[0] = pack(fmaf(lo(f[0]), roA, -moA * roA), fmaf(hi(f[0]), roA, -moA * roA));
      f[1] = pack(fmaf(lo(f[1]), roB, -moB * roB), fmaf(hi(f[1]), roB, -moB * roB));
      f[2] = pack(fmaf(lo(f[2]), roA, -moA * roA), fmaf(hi(f[2]), roA, -moA * roA));
      f[3] = pack(fmaf(lo(f[3]), roB, -moB * roB), fmaf(hi(f[3]), roB, -moB * roB));
      stsm4t(a, f);
    }
    fence_async();
    bar_sync(1 + r, 128);
    // ---- p = nhat Wp'^T + pb, g/2 = xhat Wg'^T + gb
    float pacc[32], gacc[32];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float2 pb = *reinterpret_cast<const float2*>(sBias + 8 * j + 2 * q);
      const float2 gb = *reinterpret_cast<const float2*>(sBias + 64 + 8 * j + 2 * q);
      pacc[4 * j] = pacc[4 * j + 2] = pb.x; pacc[4 * j + 1] = pacc[4 * j + 3] = pb.y;
      gacc[4 * j] = gacc[4 * j + 2] = gb.x; gacc[4 * j + 1] = gacc[4 * j + 3] = gb.y;
    }
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma64<0, 0>(gacc, kdesc(s + XS + 32 * ks), kdesc(sb + WGS + 32 * ks), 1);
#pragma unroll
    for (int ks = 0; ks < 16; ++ks)
      mma64<1, 0>(pacc, mdesc(s + TRI + 2048 * ks), kdesc(sb + WP + (ks >> 2) * 8192 + 32 * (ks & 3)), 1);
    wg_commit();
    wg_wait<0>();
    fence_acc(pacc); fence_acc(gacc);
    // ---- dProj = u s, dgl = dProj p (1 - s)   (u = dy ds, s = (1 + t) / 2, t = tanh(g / 2))
    uint32_t dpf[4][4], dgf[4][4];
    float s1A = 0.f, s1B = 0.f, s2A = 0.f, s2B = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = h ? rB : rA;
        const uint32_t off = swz(row, (8 * j + 2 * q) * 2);
        const uint32_t dyv = lds32(s + DYS + off), dsv = lds32(s + DSS + off);
        float dp2[2], dg2[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float hu = 0.5f * (e ? hi(dyv) : lo(dyv)) * (e ? hi(dsv) : lo(dsv));
          const float t = tanh_approx(gacc[4 * j + 2 * h + e]);
          dp2[e] = fmaf(hu, t, hu);
          dg2[e] = dp2[e] * pacc[4 * j + 2 * h + e] * fmaf(-0.5f, t, 0.5f);
        }
        const uint32_t dpr = pack(dp2[0], dp2[1]);   // the bf16 dProj the dxhat product reads
        dpf[j >> 1][(j & 1) * 2 + h] = dpr;
        const float2 pb2 = *reinterpret_cast<const float2*>(sBias + 8 * j + 2 * q);
        const float2 ws2 = *reinterpret_cast<const float2*>(sBias + 128 + 8 * j + 2 * q);
        const float s1v = fmaf(lo(dpr), ws2.x, hi(dpr) * ws2.y);
        const float s2v = fmaf(lo(dpr), pacc[4 * j + 2 * h] - pb2.x, hi(dpr) * (pacc[4 * j + 2 * h + 1] - pb2.y));
        if (h) { s1B += s1v; s2B += s2v; } else { s1A += s1v; s2A += s2v; }
        dgf[j >> 1][(j & 1) * 2 + h] = pack(dg2[0], dg2[1]);
      }
    }
    s1A = quad(s1A) * (1.f / 256); s1B = quad(s1B) * (1.f / 256);
    s2A = quad(s2A) * (1.f / 256); s2B = quad(s2B) * (1.f / 256);
    if (q == 0) {
      const int t0 = tile * 64;
      P.s12[t0 + rA] = s1A; P.s12[P.m + t0 + rA] = s2A;
      P.s12[t0 + rB] = s1B; P.s12[P.m + t0 + rB] = s2B;
    }
    __syncwarp();   // the warp's dy rows are read before dgl overwrites them
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      stsm4(afrag(ring + DP, 16 * w, 16 * t, lane), dpf[t]);
      stsm4(afrag(s + DYS, 16 * w, 16 * t, lane), dgf[t]);
    }
    fence_async();
    bar_sync(1 + r, 128);
    // ---- dpsum is B1b's; dWg: Gg += dgl^T xhat, dglsum
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma64<1, 1>(Gg, mdesc(s + DYS + 2048 * ks), mdesc(s + XS + 2048 * ks), 1);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma8<1, 0>(dgs, mdesc(s + DYS + 2048 * ks), kdesc(sb + ONES + 32 * ks), 1);
    wg_commit();
    wg_wait<0>();
    fence_acc(Gg); fence_acc(dgs);
    fence_async();
    bar_sync(1 + r, 128);
    if (leader) {
      const int t0 = tile * 64;
      tma_store(&P.tm_dp, ring + DP, 0, t0);
      tma_store(&P.tm_dgl, s + DYS, 0, t0);
      tma_commit();
      tma_wait_read();                 // the stores have read the stage: the next tile may overwrite it
      if (tile + vstride < P.ntiles) load(tile + vstride);
    }
  }
  if (leader) tma_wait_all();
  bar_sync(1 + r, 128);
  // ---- raw sums of both warpgroups ([o][c] fp32 in their ring regions), folded by all 256 threads
  {
    float* sGg = reinterpret_cast<float*>(smem + RING + r * WGSZ);
    float* sDgs = sGg + 4096;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = 8 * j + 2 * q;
      *reinterpret_cast<float2*>(sGg + rA * 64 + c) = make_float2(Gg[4 * j], Gg[4 * j + 1]);
      *reinterpret_cast<float2*>(sGg + rB * 64 + c) = make_float2(Gg[4 * j + 2], Gg[4 * j + 3]);
    }
    sDgs[rA] = dgs[0]; sDgs[rB] = dgs[2];   // same value from the 4 lanes
  }
  __syncthreads();
  {
    const float* G0 = reinterpret_cast<const float*>(smem + RING);
    const float* G1 = reinterpret_cast<const float*>(smem + RING + WGSZ);
    float* out = P.part + (size_t)blockIdx.x * SLOT;
    const int c = tid & 63;                                // dWg = Gg o gi + dglsum bi^T: column tid & 63
    const float gc = P.gi[c], bc = P.bi[c];
#pragma unroll 8
    for (int it = 0; it < 16; ++it) {
      const int i = tid + 256 * it, o = i >> 6;
      out[i] = fmaf(G0[i] + G1[i], gc, (G0[4096 + o] + G1[4096 + o]) * bc);
    }
  }
}

// =====================================================================================================================
// B1b: one channel half
// =====================================================================================================================
struct B1bParams {
  CUtensorMap tm_tri, tm_dp, tm_st, tm_s12, tm_dt;
  const bf16* wp;      // [64 out][256 in]
  const float *go, *bo;
  float* part;         // [grid][hb::SLOT]: dWp_h [64][128] | dgo_h [128] | dbo_h [128]
  int ntiles, cb;      // cb: first channel of the half (0 / 128)
};
namespace hb {
constexpr int WP = 0;               // 2 x [64 out][64 ch] (Wp o go)_h: K-major B of dp.. (MN-major B of dxhat)
constexpr int ONES = WP + 16384;    // [8][64] ones: K-major B of the row sums
constexpr int BAR = ONES + 1024;    // [2 wg][2 stage] full barriers
constexpr int RING = 18432;         // per-warpgroup region (1024-aligned)
// stage: TRI [128 ch][64 tok] (-> nhat -> dt), DP [64 tok][64 out], ST [2][64] f32 (mo, ro), S12 [2][64] f32 (s1, s2)
constexpr int TRI = 0, DPT = 16384, STS = 24576, S12S = 25088, STAGE = 26624;
constexpr int WGSZ = 2 * STAGE;
constexpr int SMEM = RING + 2 * WGSZ;
constexpr int SLOT = 8448;
constexpr uint32_t TX = 16384 + 8192 + 512 + 512;
}  // namespace hb

extern "C" __global__ void __launch_bounds__(256, 1) d64h2_b1b(const __grid_constant__ B1bParams P) {
  using namespace hb;
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, wtid = tid & 127, w = wtid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;
  const int r = __shfl_sync(0xffffffffu, tid >> 7, 0);   // warp-uniform for ptxas
  const uint32_t sb = smem_u32(smem);
  const uint32_t ring = sb + RING + r * WGSZ;
  const uint32_t bar0 = sb + BAR + r * 16;
  const bool leader = wtid == 0;
  const int cb = P.cb;

#pragma unroll
  for (int it = 0; it < 4; ++it) {                        // (Wp o go)_h: [64 out][128 ch] -> two [64][64] tiles
    const int i = tid + 256 * it, o = i >> 4, c = (i & 15) * 8;
    const uint4 wv = *reinterpret_cast<const uint4*>(P.wp + o * 256 + cb + c);
    const float4 g0 = *reinterpret_cast<const float4*>(P.go + cb + c), g1 = *reinterpret_cast<const float4*>(P.go + cb + c + 4);
    const uint4 ov = make_uint4(pack(lo(wv.x) * g0.x, hi(wv.x) * g0.y), pack(lo(wv.y) * g0.z, hi(wv.y) * g0.w),
                                pack(lo(wv.z) * g1.x, hi(wv.z) * g1.y), pack(lo(wv.w) * g1.z, hi(wv.w) * g1.w));
    *reinterpret_cast<uint4*>(smem + WP + (c >> 6) * 8192 + swz(o, (c & 63) * 2)) = ov;
  }
  if (tid < 64) reinterpret_cast<uint4*>(smem + ONES)[tid] = make_uint4(0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u);
  if (tid == 0) {
    for (int i = 0; i < 4; ++i) mbar_init(sb + BAR + 8 * i, 1);
    fence_barrier_init();
  }
  fence_async();
  __syncthreads();

  const int v = 2 * blockIdx.x + r, vstride = 2 * gridDim.x;
  auto load = [&](int tile, int st) {
    const uint32_t s = ring + st * STAGE, bar = bar0 + 8 * st;
    const int t0 = tile * 64;
    mbar_expect(bar, TX);
    tma_load(s + TRI, &P.tm_tri, bar, t0, cb);
    tma_load(s + TRI + 8192, &P.tm_tri, bar, t0, cb + 64);
    tma_load(s + DPT, &P.tm_dp, bar, 0, t0);
    tma_load(s + STS, &P.tm_st, bar, t0, 2);
    tma_load(s + S12S, &P.tm_s12, bar, t0, 0);
  };
  if (leader) {
    prefetch_map(&P.tm_tri); prefetch_map(&P.tm_dp); prefetch_map(&P.tm_st); prefetch_map(&P.tm_s12); prefetch_map(&P.tm_dt);
    if (v < P.ntiles) load(v, 0);
  }

  float G[64], dps[4];
#pragma unroll
  for (int i = 0; i < 64; ++i) G[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) dps[i] = 0.f;
  const int rA = 16 * w + gq, rB = rA + 8;

  int k = 0;
  for (int tile = v; tile < P.ntiles; tile += vstride, ++k) {
    const int st = k & 1;
    const uint32_t s = ring + st * STAGE;
    mbar_wait(bar0 + 8 * st, (k >> 1) & 1);
    const float* sSt = reinterpret_cast<const float*>(smem + (s - sb) + STS);
    const float* sS12 = reinterpret_cast<const float*>(smem + (s - sb) + S12S);
    const float moA = sSt[rA], roA = sSt[64 + rA], moB = sSt[rB], roB = sSt[64 + rB];
    // ---- nhat in place (bf16)
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      uint32_t f[4];
      const uint32_t a = afrag_t(s + TRI, 16 * kc, 16 * w, lane);
      ldsm4t(f, a);
      f[0] = pack(fmaf(lo(f[0]), roA, -moA * roA), fmaf(hi(f[0]), roA, -moA * roA));
      f[1] = pack(fmaf(lo(f[1]), roB, -moB * roB), fmaf(hi(f[1]), roB, -moB * roB));
      f[2] = pack(fmaf(lo(f[2]), roA, -moA * roA), fmaf(hi(f[2]), roA, -moA * roA));
      f[3] = pack(fmaf(lo(f[3]), roB, -moB * roB), fmaf(hi(f[3]), roB, -moB * roB));
      stsm4t(a, f);
    }
    fence_async();
    bar_sync(1 + r, 128);
    if (leader && tile + vstride < P.ntiles) {
      tma_wait_read();                 // the previous tile's store has read stage st ^ 1
      load(tile + vstride, st ^ 1);
    }
    // ---- dxhat = dp Wp' ; G += dp^T nhat ; dpsum
    float dxh[64];
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma128<0, 1>(dxh, kdesc(s + DPT + 32 * ks), mdesc(sb + WP + 2048 * ks), ks > 0);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma128<1, 0>(G, mdesc(s + DPT + 2048 * ks), kdesc(s + TRI + 32 * ks), 1);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma8<1, 0>(dps, mdesc(s + DPT + 2048 * ks), kdesc(sb + ONES + 32 * ks), 1);
    wg_commit();
    wg_wait<0>();
    fence_acc(dxh); fence_acc(G); fence_acc(dps);
    // ---- output-LN backward -> dt (in place of nhat); the row sums came from B1a
    const float s1A = sS12[rA], s2A = sS12[64 + rA], s1B = sS12[rB], s2B = sS12[64 + rB];
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      uint32_t f[4];
      const uint32_t a = afrag_t(s + TRI, 16 * kc, 16 * w, lane);
      ldsm4t(f, a);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float* d = dxh + 4 * (2 * kc + (i >> 1)) + 2 * (i & 1);
        const float ro = (i & 1) ? roB : roA, s1 = (i & 1) ? s1B : s1A, s2 = (i & 1) ? s2B : s2A;
        f[i] = pack(ro * (d[0] - s1 - lo(f[i]) * s2), ro * (d[1] - s1 - hi(f[i]) * s2));
      }
      stsm4t(a, f);
    }
    fence_async();
    bar_sync(1 + r, 128);
    if (leader) {
      const int t0 = tile * 64;
      tma_store(&P.tm_dt, s + TRI, t0, cb);
      tma_store(&P.tm_dt, s + TRI + 8192, t0, cb + 64);
      tma_commit();
    }
  }
  if (leader) tma_wait_all();
  bar_sync(1 + r, 128);   // this warpgroup's stores have read its ring: it now holds the raw sums
  {
    float* sG = reinterpret_cast<float*>(smem + RING + r * WGSZ);
    float* sDps = sG + 8192;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int c = 8 * j + 2 * q;
      *reinterpret_cast<float2*>(sG + rA * 128 + c) = make_float2(G[4 * j], G[4 * j + 1]);
      *reinterpret_cast<float2*>(sG + rB * 128 + c) = make_float2(G[4 * j + 2], G[4 * j + 3]);
    }
    sDps[rA] = dps[0]; sDps[rB] = dps[2];   // same value from the 4 lanes
  }
  __syncthreads();
  {
    const float* G0 = reinterpret_cast<const float*>(smem + RING);
    const float* G1 = reinterpret_cast<const float*>(smem + RING + WGSZ);
    float* out = P.part + (size_t)blockIdx.x * SLOT;
    {                                                      // dWp_h = G o go + dpsum bo^T: column tid & 127 per thread
      const int c = tid & 127;
      const float gc = P.go[cb + c], bc = P.bo[cb + c];
#pragma unroll 8
      for (int it = 0; it < 32; ++it) {
        const int i = tid + 256 * it, o = i >> 7;
        out[i] = fmaf(G0[i] + G1[i], gc, (G0[8192 + o] + G1[8192 + o]) * bc);
      }
    }
    const int c = tid & 127, half = tid >> 7;              // dgo = sum_o Wp o G, dbo = Wp^T dpsum (two halves of o)
    float sg = 0.f, sbv = 0.f;
#pragma unroll 8
    for (int o = 32 * half; o < 32 * half + 32; ++o) {
      const float wv = __bfloat162float(P.wp[o * 256 + cb + c]);
      sg = fmaf(wv, G0[o * 128 + c] + G1[o * 128 + c], sg);
      sbv = fmaf(wv, G0[8192 + o] + G1[8192 + o], sbv);
    }
    __syncthreads();                                       // every thread has read the raw sums: the first region is free
    float* sX = reinterpret_cast<float*>(smem + RING);
    if (half) { sX[c] = sg; sX[128 + c] = sbv; }
    __syncthreads();
    if (!half) { out[8192 + c] = sg + sX[c]; out[8320 + c] = sbv + sX[128 + c]; }
  }
}

// =====================================================================================================================
// B2: the input side on one channel half
// =====================================================================================================================
struct B2Params {
  CUtensorMap tm_dl, tm_dr, tm_x, tm_dxg, tm_dy, tm_st, tm_mk, tm_dx;
  const bf16* w1;      // packed [1024][64]: 16 blocks of 64 rows (32 gate | 32 projection channels), left 0-7, right 8-15
  const bf16* wg;      // packed [512][64]
  const float *gi, *bi;
  float* part;         // [grid][b2::SLOT]: dW1_h packed [512][64] | dgi [64] | dbi [64]
  float* dxn;          // [M][64] fp32: PASS 0 stores the row's dxn, PASS 1 adds it
  int ntiles, hh;      // hh: the channel half (0 / 1)
};
namespace b2 {
// W1 resident, rows of each 64-row block permuted to [gate 0-15 | proj 0-15 | gate 16-31 | proj 16-31] so that a half
// chunk (16 channels: gate + projection) is 32 contiguous rows; gate rows are halved (sigmoid via tanh(g / 2)).
constexpr int W1 = 0;
constexpr int DL = 0, DR = 16384, XS = 32768, DXG = 40960, DYS = 49152, STS = 57344, MKS = 57856, STAGE = 58368;
constexpr int WGATE = 65536;
constexpr int RING = 73728;
constexpr int XN = RING + 2 * STAGE;    // [64 tok][64] xn: K-major A of pre, MN-major B of dW1
constexpr int SCR = XN + 8192;          // [2 side][2 buffer][64 tok][64] dpre: K-major A of dxn, MN-major A of dW1
constexpr int GIB = SCR + 32768;        // gi[64] bi[64]
constexpr int REDW = RING; // final-only reduction reuses dead DL stage
constexpr int BAR = GIB + 512;
constexpr int SMEM = BAR + 64;
constexpr int SLOT = 32768 + 128;
constexpr uint32_t TX = 2 * 16384 + 3 * 8192 + 512 + 256;
// permuted smem row (within a block) -> packed W1 row (packed block rows: gate 0-31, projection 32-63)
DEVI int unperm(int r) { return ((r >> 4) & 1) * 32 + (r >> 5) * 16 + (r & 15); }
}  // namespace b2

// pre of half chunk (C, HC): xn . W1 rows [64 (4 side + C) + 32 HC, + 32)  (gate 16 | projection 16)
#define B2_ISSUE_PRE(ACC, C, HC) do { \
    const uint32_t b_ = sb + W1 + (4 * side + (C)) * 8192 + (HC) * 4096; \
    _Pragma("unroll") for (int ks = 0; ks < 4; ++ks) mma32<0, 0>(ACC, kdesc(sb + XN + 32 * ks), kdesc(b_ + 32 * ks), ks > 0); \
    wg_commit(); } while (0)
// half chunk (C, HC): mask / sigmoid derivatives -> scratch k-steps 2 HC (gate, x2) and 2 HC + 1 (projection)
#define B2_HALF(ACC, C, HC, SCRB) do { \
    uint32_t d4[4], ag[4], ap[4]; \
    ldsm4t(d4, afrag_t(dsrc, 32 * (C) + 16 * (HC), 16 * w, lane)); \
    _Pragma("unroll") for (int i = 0; i < 4; ++i) { \
      const int jj = i >> 1, h = i & 1; \
      const float mk = h ? mkB : mkA; \
      float vg[2], vp[2]; \
      _Pragma("unroll") for (int e = 0; e < 2; ++e) { \
        const float hd = (e ? hi(d4[i]) : lo(d4[i])) * mk; \
        const float t = tanh_approx(ACC[4 * jj + 2 * h + e]); \
        vp[e] = fmaf(hd, t, hd); \
        vg[e] = vp[e] * ACC[4 * (2 + jj) + 2 * h + e] * (1.f - t); \
      } \
      ag[i] = pack(vg[0], vg[1]); \
      ap[i] = pack(vp[0], vp[1]); \
    } \
    stsm4(afrag(SCRB, 16 * w, 32 * (HC), lane), ag); \
    stsm4(afrag(SCRB, 16 * w, 32 * (HC) + 16, lane), ap); } while (0)

extern "C" __global__ void __launch_bounds__(256, 1) d64h2_b2w(const __grid_constant__ B2Params P) {
  using namespace b2;
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, wtid = tid & 127, w = wtid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;
  const int side = __shfl_sync(0xffffffffu, tid >> 7, 0);   // warp-uniform for ptxas
  const uint32_t sb = smem_u32(smem);
  float* sGi = reinterpret_cast<float*>(smem + GIB);
  float* sBi = sGi + 64;
  float* sRedW = reinterpret_cast<float*>(smem + REDW) + (tid >> 5) * 128;
  const int cb = 128 * P.hh;

  for (int i = tid; i < 512 * 8; i += 256) {
    const int row = i >> 3, g = i & 7, blk = row >> 6, rr = row & 63;
    const int src = ((blk >> 2) * 8 + 4 * P.hh + (blk & 3)) * 64 + unperm(rr);   // the half's blocks of this side in the packed W1
    uint4 v4 = *reinterpret_cast<const uint4*>(P.w1 + src * 64 + g * 8);
    if (!((rr >> 4) & 1))   // gate rows / 2 (exact)
      v4 = make_uint4(pack(0.5f * lo(v4.x), 0.5f * hi(v4.x)), pack(0.5f * lo(v4.y), 0.5f * hi(v4.y)),
                      pack(0.5f * lo(v4.z), 0.5f * hi(v4.z)), pack(0.5f * lo(v4.w), 0.5f * hi(v4.w)));
    *reinterpret_cast<uint4*>(smem + W1 + swz(row, g * 16)) = v4;
  }
  if (tid < 64) { sGi[tid] = P.gi[tid]; sBi[tid] = P.bi[tid]; }
  float affine_sum[4] = {0.f, 0.f, 0.f, 0.f};
  for (int i = tid; i < 512; i += 256) {
    const int row = i >> 3, g = i & 7;
    *reinterpret_cast<uint4*>(smem + WGATE + swz(row, g * 16)) =
        *reinterpret_cast<const uint4*>(P.wg + row * 64 + g * 8);
  }
  if (tid == 0) {
    mbar_init(sb + BAR, 1); mbar_init(sb + BAR + 8, 1);
    fence_barrier_init();
  }
  fence_async();
  __syncthreads();

  auto load = [&](int tile, int st) {
    const uint32_t s = sb + RING + st * STAGE, bar = sb + BAR + 8 * st;
    const int t0 = tile * 64;
    mbar_expect(bar, TX);
    tma_load(s + DL, &P.tm_dl, bar, t0, cb);
    tma_load(s + DL + 8192, &P.tm_dl, bar, t0, cb + 64);
    tma_load(s + DR, &P.tm_dr, bar, t0, cb);
    tma_load(s + DR + 8192, &P.tm_dr, bar, t0, cb + 64);
    tma_load(s + XS, &P.tm_x, bar, 0, t0);
    tma_load(s + DXG, &P.tm_dxg, bar, 0, t0);
    tma_load(s + DYS, &P.tm_dy, bar, 0, t0);
    tma_load(s + STS, &P.tm_st, bar, t0, 0);
    tma_load(s + MKS, &P.tm_mk, bar, t0, 0);
  };
  if (tid == 0) {
    prefetch_map(&P.tm_dl); prefetch_map(&P.tm_dr); prefetch_map(&P.tm_x); prefetch_map(&P.tm_dxg);
    prefetch_map(&P.tm_dy); prefetch_map(&P.tm_st); prefetch_map(&P.tm_mk); prefetch_map(&P.tm_dx);
    if ((int)blockIdx.x < P.ntiles) load(blockIdx.x, 0);
  }

  float Gx[4][32];
#pragma unroll
  for (int c = 0; c < 4; ++c)
#pragma unroll
    for (int i = 0; i < 32; ++i) Gx[c][i] = 0.f;
  const int rA = 16 * w + gq, rB = rA + 8;
  const uint32_t scr0 = sb + SCR + side * 16384;

  int k = 0;
  for (int tile = blockIdx.x; tile < P.ntiles; tile += gridDim.x, ++k) {
    const int st = k & 1;
    const uint32_t s = sb + RING + st * STAGE;
    mbar_wait(sb + BAR + 8 * st, (k >> 1) & 1);
    const float* sSt = reinterpret_cast<const float*>(smem + RING + st * STAGE + STS);
    const float* sMk = reinterpret_cast<const float*>(smem + RING + st * STAGE + MKS);
    // ---- xn rows (the forward's operand statement: fma((x - mean) rstd, gamma, beta), one bf16 rounding)
    {   // warp (side, w): rows 16 w .., k-steps 2 side, 2 side + 1
      const float miA = sSt[rA], riA = sSt[64 + rA], miB = sSt[rB], riB = sSt[64 + rB];
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int ks = 2 * side + kk;
        uint32_t f[4];
        ldsm4(f, afrag(s + XS, 16 * w, 16 * ks, lane));
        const int c0 = 16 * ks + 2 * q;
#define XN_(v, m, rs, c) __fmaf_rn(__fmul_rn(__fsub_rn(v, m), rs), sGi[c], sBi[c])
        f[0] = pack(XN_(lo(f[0]), miA, riA, c0), XN_(hi(f[0]), miA, riA, c0 + 1));
        f[1] = pack(XN_(lo(f[1]), miB, riB, c0), XN_(hi(f[1]), miB, riB, c0 + 1));
        f[2] = pack(XN_(lo(f[2]), miA, riA, c0 + 8), XN_(hi(f[2]), miA, riA, c0 + 9));
        f[3] = pack(XN_(lo(f[3]), miB, riB, c0 + 8), XN_(hi(f[3]), miB, riB, c0 + 9));
#undef XN_
        stsm4(afrag(sb + XN, 16 * w, 16 * ks, lane), f);
      }
    }
    fence_async();
    __syncthreads();
    if (tid == 0 && tile + (int)gridDim.x < P.ntiles) {
      tma_wait_read();
      load(tile + gridDim.x, st ^ 1);
    }
    const float mkA = 0.5f * sMk[rA], mkB = 0.5f * sMk[rB];
    const uint32_t dsrc = s + (side ? DR : DL);
    float p0[16], p1[16], dxn[32];
    wg_fence();
    B2_ISSUE_PRE(p0, 0, 0);
    B2_ISSUE_PRE(p1, 0, 1);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint32_t scr = scr0 + (c & 1) * 8192;
      // in flight (oldest first): p0(c), p1(c), D(c - 1)
      if (c == 0) wg_wait<1>(); else wg_wait<2>();
      fence_acc(p0);
      B2_HALF(p0, c, 0, scr);
      if (c == 0) wg_wait<0>(); else wg_wait<1>();
      fence_acc(p1);
      B2_HALF(p1, c, 1, scr);
      fence_async();
      bar_sync(1 + side, 128);
      wg_fence();
      if (c < 3) {
        B2_ISSUE_PRE(p0, c + 1, 0);
        B2_ISSUE_PRE(p1, c + 1, 1);
      }
      const uint32_t wb = sb + W1 + (4 * side + c) * 8192;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma64<0, 1>(dxn, kdesc(scr + 32 * ks), mdesc(wb + 2048 * ks), c > 0 || ks > 0);
#if PASS == 0
      // the gate part dgl Wg of dxn is added once, in the first pass
      if (c == 3 && side == 0) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          mma64<0, 1>(dxn, kdesc(s + DXG + 32 * ks), mdesc(sb + WGATE + 2048 * ks), 1);
      }
#endif
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma64<1, 1>(Gx[c], mdesc(scr + 2048 * ks), mdesc(sb + XN + 2048 * ks), 1);
      wg_commit();
    }
    wg_wait<0>();
    fence_acc(dxn);
#pragma unroll
    for (int c = 0; c < 4; ++c) fence_acc(Gx[c]);
    // ---- exchange: side 0 finalises rows A (16 w + gq), side 1 rows B; each hands the other its half
    {
      const uint32_t mine = s + (side ? DR : DL), other = s + (side ? DL : DR);   // dl / dr consumed
#pragma unroll
      for (int g = 0; g < 4; ++g) {   // n-tiles 2 g, 2 g + 1 of the other side's row
        // Constant-index alternatives keep dxn in registers. Selecting an
        // index with runtime `side` forced a 128-byte local array in NVCC.
        const float4 send = side
            ? make_float4(dxn[8*g], dxn[8*g+1], dxn[8*g+4], dxn[8*g+5])
            : make_float4(dxn[8*g+2], dxn[8*g+3], dxn[8*g+6], dxn[8*g+7]);
        sts128f(mine + (g * 128 + wtid) * 16, send);
      }
      __syncthreads();
      const int row = side ? rB : rA;
      float d[16];
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const float4 v4 = lds128f(other + (g * 128 + wtid) * 16);
        d[4 * g] = (side ? dxn[8*g+2] : dxn[8*g]) + v4.x;
        d[4 * g + 1] = (side ? dxn[8*g+3] : dxn[8*g+1]) + v4.y;
        d[4 * g + 2] = (side ? dxn[8*g+6] : dxn[8*g+4]) + v4.z;
        d[4 * g + 3] = (side ? dxn[8*g+7] : dxn[8*g+5]) + v4.w;
      }
      float* part = P.dxn + (size_t)(tile * 64 + row) * 64 + 2 * q;
#if PASS == 0
      // ---- the row's partial dxn (this half + the gate part) for the second pass
#pragma unroll
      for (int j = 0; j < 8; ++j) *reinterpret_cast<float2*>(part + 8 * j) = make_float2(d[2 * j], d[2 * j + 1]);
#else
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 pv = *reinterpret_cast<const float2*>(part + 8 * j);
        d[2 * j] += pv.x;
        d[2 * j + 1] += pv.y;
      }
      // ---- input-LN backward with the identity residual, one row per thread quad
      const float rs = sSt[64 + row], nmr = -sSt[row] * rs;
      float xh[16], s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t off = swz(row, (8 * j + 2 * q) * 2);
        const uint32_t xv = lds32(s + XS + off);
        const float2 g2 = *reinterpret_cast<const float2*>(sGi + 8 * j + 2 * q);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = 2 * j + e;
          xh[i] = fmaf(e ? hi(xv) : lo(xv), rs, nmr);
          const float gd = d[i] * (e ? g2.y : g2.x);
          s1 += gd;
          s2 = fmaf(gd, xh[i], s2);
        }
      }
      s1 = quad(s1) * (1.f / 64);
      s2 = quad(s2) * (1.f / 64);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t off = swz(row, (8 * j + 2 * q) * 2);
        const uint32_t dyv = lds32(s + DYS + off);
        const float2 g2 = *reinterpret_cast<const float2*>(sGi + 8 * j + 2 * q);
        float o[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = 2 * j + e;
          o[e] = fmaf(rs, fmaf(d[i], e ? g2.y : g2.x, -fmaf(xh[i], s2, s1)), e ? hi(dyv) : lo(dyv));
        }
        sts32(s + XS + off, pack(o[0], o[1]));
      }
      // ---- dgamma / dbeta_in: reduce-scatter over the 8 row groups, then this warp's private sums
      float v[32];
#pragma unroll
      for (int i = 0; i < 16; ++i) { v[i] = d[i] * xh[i]; v[16 + i] = d[i]; }
#pragma unroll
      for (int i = 0; i < 16; ++i) {   // lane bit 4: keep [0, 16) or [16, 32)
        const bool up = lane & 16;
        const float keep = up ? v[16 + i] : v[i], send = up ? v[i] : v[16 + i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {    // lane bit 3
        const bool up = lane & 8;
        const float keep = up ? v[8 + i] : v[i], send = up ? v[i] : v[8 + i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {    // lane bit 2
        const bool up = lane & 4;
        const float keep = up ? v[4 + i] : v[i], send = up ? v[i] : v[4 + i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) affine_sum[i] += v[i];
#endif
    }
    fence_async();
    __syncthreads();
#if PASS == 1
    if (tid == 0) {
      tma_store(&P.tm_dx, s + XS, 0, tile * 64);
      tma_commit();
    }
#endif
  }
  if (tid == 0) tma_wait_all();
  float* out = P.part + (size_t)blockIdx.x * SLOT;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int row0 = 64 * (4 * side + c);
    const int pA = unperm(rA), pB = unperm(rB);
    const float fA = pA < 32 ? 0.5f : 1.f, fB = pB < 32 ? 0.5f : 1.f;   // gate rows carried 2 x the gradient
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int col = 8 * j + 2 * q;
      *reinterpret_cast<float2*>(out + (row0 + pA) * 64 + col) = make_float2(fA * Gx[c][4 * j], fA * Gx[c][4 * j + 1]);
      *reinterpret_cast<float2*>(out + (row0 + pB) * 64 + col) = make_float2(fB * Gx[c][4 * j + 2], fB * Gx[c][4 * j + 3]);
    }
  }
#if PASS == 1
  // All persistent input loads have finished; publish each lane's four sums
  // into the no-longer-live DL stage. Keep the original fixed warp sum order.
  const int affine_base = ((lane >> 4) & 1) * 64 + 8 * (4 * ((lane >> 3) & 1) + 2 * ((lane >> 2) & 1)) + 2 * q;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    sRedW[affine_base + 8 * (i >> 1) + (i & 1)] = affine_sum[i];
  __syncthreads();
  if (tid < 128) {
    const float* r8 = reinterpret_cast<const float*>(smem + REDW);
    float s = 0.f;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) s += r8[ww * 128 + tid];
    out[32768 + tid] = s;
  }
#endif
}

// =====================================================================================================================
// Partial reduction (fixed order).  Outputs: dW1 of both halves -> dWl / dWlg / dWr / dWrg [256][64]; dgi, dbi (PASS 1 slots); dWg
// (B1a slots); dWp [64][256] | dgo | dbo (B1b slots of both halves).
// =====================================================================================================================
#if MASTER_FP32
using MasterDW = float;
__device__ float master_value(float v) { return v; }
#else
using MasterDW = bf16;
__device__ bf16 master_value(float v) { return __float2bfloat16_rn(v); }
#endif
extern "C" __global__ void __launch_bounds__(256)
d64h2_fin(const float* __restrict__ pa, int ga, const float* __restrict__ pb0, const float* __restrict__ pb1, int gb,
          const float* __restrict__ pc0, const float* __restrict__ pc1, int gc,
          MasterDW* __restrict__ dwl, MasterDW* __restrict__ dwlg, MasterDW* __restrict__ dwr, MasterDW* __restrict__ dwrg,
          MasterDW* __restrict__ dwg, MasterDW* __restrict__ dwp, float* __restrict__ dgi, float* __restrict__ dbi,
          float* __restrict__ dgo, float* __restrict__ dbo) {
  int e = blockIdx.x * 256 + threadIdx.x;
  if (e < 65536) {                                          // dW1 of the two halves
    const int hh = e >> 15, f = e & 32767;
    const float* p = hh ? pc1 : pc0;
    float s = 0.f;
    for (int g = 0; g < gc; ++g) s += p[(size_t)g * b2::SLOT + f];
    const int pr = f >> 6, col = f & 63, blk = pr >> 6, within = pr & 63;
    const int c = 32 * blk + (within & 31), h = (c & 127) + 128 * hh;
    MasterDW* dst = c < 128 ? (within < 32 ? dwlg : dwl) : (within < 32 ? dwrg : dwr);
    dst[h * 64 + col] = master_value(s);
    return;
  }
  e -= 65536;
  if (e < 128) {                                            // dgi, dbi (second pass)
    float s = 0.f;
    for (int g = 0; g < gc; ++g) s += pc1[(size_t)g * b2::SLOT + 32768 + e];
    if (e < 64) dgi[e] = s; else dbi[e - 64] = s;
    return;
  }
  e -= 128;
  if (e < 4096) {                                           // dWg
    float s = 0.f;
    for (int g = 0; g < ga; ++g) s += pa[(size_t)g * ha::SLOT + e];
    dwg[e] = master_value(s);
    return;
  }
  e -= 4096;
  if (e < 16384) {                                          // dWp [64][256]
    const int o = e >> 8, cf = e & 255, hh = cf >> 7, cl = cf & 127;
    const float* p = hh ? pb1 : pb0;
    float s = 0.f;
    for (int g = 0; g < gb; ++g) s += p[(size_t)g * hb::SLOT + o * 128 + cl];
    dwp[e] = master_value(s);
    return;
  }
  e -= 16384;
  if (e < 512) {                                            // dgo | dbo
    const int which = e >> 8, cf = e & 255, hh = cf >> 7, cl = cf & 127;
    const float* p = hh ? pb1 : pb0;
    float s = 0.f;
    for (int g = 0; g < gb; ++g) s += p[(size_t)g * hb::SLOT + 8192 + which * 128 + cl];
    (which ? dbo : dgo)[cf] = s;
  }
}

}  // namespace d64
