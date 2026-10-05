// SPDX-License-Identifier: Apache-2.0
// Single-direction TriMul training backward at pair width D = 64 (hidden 64, one contraction), sm_90a, wgmma + TMA.
// Derived from d64_bwd_wg.cu (bidirectional, H = 128): the output side keeps one [64][64] Wp tile and a 64-channel
// output LN; the input side has two packed 64-row weight blocks per side instead of four.
//
// Notation (per token t, M = n * n tokens):  xhat = (x - mu_in) r_in, xn = xhat gi + bi (the forward's gate / projection operand),
// nhat = (tri - mu_out) r_out, norm = nhat go + bo, p = norm Wp^T, g = xn Wg^T, y = x + ds sigmoid(g) p,
// sigmoid(g) = (1 + tanh(g / 2)) / 2 (the 1/2 is folded into the resident gate weights: exact in bf16).
//
// uni64_b1w  (output side; two independent warpgroups, each its own 64-token tiles and 2-stage TMA ring)
//   p = nhat (Wp o go)^T + Wp bo,  g = xhat (Wg o gi)^T + Wg bi          (affine folded into the resident weights)
//   dp = dy ds s, dgl = dy ds p s (1 - s)
//   dxhat_out = dp (Wp o go);  dt = r_out (dxhat - mean(dxhat) - nhat mean(dxhat nhat))  -> dt [64][M] (TMA store)
//   dxg = dgl Wg  -> [M][64] bf16 (the gate part of dxn, consumed by b2)
//   G = dp^T nhat, dpsum = dp^T 1, Gg = dgl^T xhat, dglsum = dgl^T 1  (registers, whole kernel), folded per CTA:
//     dWp = G o go + dpsum bo^T,  dgo = colsum(Wp o G),  dbo = Wp^T dpsum,  dWg = Gg o gi + dglsum bi^T.
// uni64_b2w  (input side; warpgroup s = side (0 left, 1 right) of the same 64-token tile)
//   pre = xn W1[b]^T for its two packed blocks b in half chunks of 16 channels (gate + projection), dpre from dl / dr,
//   mask, sigmoid; dxn_s = sum_b dpre_b W1[b];  dW1[b] += dpre_b^T xn (registers, whole kernel).  The next half
//   chunks' products are in flight while the elementwise stage runs.  The sides exchange half rows of dxn, + dxg,
//   input-LN backward with the identity residual -> dx; dgamma / dbeta_in by a warp reduce-scatter per tile.
// uni64_finw  fixed-order reduction of the per-CTA partials.
#include "d64_wg.cuh"

namespace uni64 {
using namespace d64;

// =====================================================================================================================
// B1
// =====================================================================================================================
struct B1Params {
  CUtensorMap tm_tri, tm_x, tm_dy, tm_ds, tm_st, tm_dt, tm_dxg;
  const bf16* wp;      // [64 out][64 in]
  const bf16* wg;      // [64 out][64 in]
  const float *go, *bo, *gi, *bi;
  float* part;         // [grid][b1::SLOT]: dWp [64][64] | dgo [64] | dbo [64] | dWg [64][64]
  int ntiles, n;
  unsigned long long* prof;
};
namespace b1 {
constexpr int WP = 0;               // [64 out][64 ch] (Wp o go): K-major B of p, MN-major B of dxhat
constexpr int WGS = WP + 8192;      // [64 out][64 in] (Wg o gi) / 2: K-major B of g / 2
constexpr int WGU = WGS + 8192;     // [64 out][64 in] Wg: MN-major B of dxg
constexpr int ONES = WGU + 8192;    // [8][64] ones: K-major B of the row sums
constexpr int BIAS = ONES + 1024;   // pb[64] = Wp bo, gb[64] = Wg bi / 2, wsum[64] = rowsum(Wp o go)
constexpr int BAR = BIAS + 1024;    // [2 wg][2 stage] full barriers
constexpr int RING = 27648;         // per-warpgroup region (1024-aligned)
// stage: TRI [64 ch][64 tok] (-> nhat -> dt), X [64 tok][64] (-> xhat), DY [64][64] (-> dgl -> dxg), DS [64][64], ST [4][64] f32
constexpr int TRI = 0, XS = 8192, DYS = 16384, DSS = 24576, STS = 32768, STAGE = 33792;
constexpr int DP = 2 * STAGE;       // [64 tok][64 out] dProj
constexpr int WGSZ = DP + 8192;
constexpr int SMEM = RING + 2 * WGSZ;
constexpr int SLOT = 4096 + 128 + 4096;
constexpr uint32_t TX = 8192 + 3 * 8192 + 1024;
static_assert(BAR + 32 <= RING, "barriers inside the resident region");
}  // namespace b1

extern "C" __global__ void __launch_bounds__(256, 1) uni64_b1w(const __grid_constant__ B1Params P) {
  using namespace b1;
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, wtid = tid & 127, w = wtid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;
  const int r = __shfl_sync(0xffffffffu, tid >> 7, 0);   // warp-uniform for ptxas (the wgmma loop is not divergent)
  const uint32_t sb = smem_u32(smem);
  const uint32_t ring = sb + RING + r * WGSZ;
  const uint32_t bar0 = sb + BAR + r * 16;
  float* sBias = reinterpret_cast<float*>(smem + BIAS);
  const bool leader = wtid == 0;
  PF_DECL

  // ---- resident operands (affine and the sigmoid's 1/2 folded into the weights); 8-element granules
#pragma unroll
  for (int it = 0; it < 2; ++it) {                        // Wp o go: [64 out][64 ch]
    const int i = tid + 256 * it, o = i >> 3, c = (i & 7) * 8;
    const uint4 wv = *reinterpret_cast<const uint4*>(P.wp + o * 64 + c);
    const float4 g0 = *reinterpret_cast<const float4*>(P.go + c), g1 = *reinterpret_cast<const float4*>(P.go + c + 4);
    const uint4 ov = make_uint4(pack(lo(wv.x) * g0.x, hi(wv.x) * g0.y), pack(lo(wv.y) * g0.z, hi(wv.y) * g0.w),
                                pack(lo(wv.z) * g1.x, hi(wv.z) * g1.y), pack(lo(wv.w) * g1.z, hi(wv.w) * g1.w));
    *reinterpret_cast<uint4*>(smem + WP + swz(o, c * 2)) = ov;
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
    *reinterpret_cast<uint4*>(smem + WGU + swz(o, c * 2)) = wv;
  }
  if (tid < 64) reinterpret_cast<uint4*>(smem + ONES)[tid] = make_uint4(0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u);
  {                                                       // pb = Wp bo, gb = Wg bi / 2: warp w8 rows 8 w8 .. 8 w8 + 7
    const int w8 = tid >> 5;
    const float2 b2 = *reinterpret_cast<const float2*>(P.bo + 2 * lane);
    const float2 o2 = *reinterpret_cast<const float2*>(P.go + 2 * lane);
    const float2 c2 = *reinterpret_cast<const float2*>(P.bi + 2 * lane);
#pragma unroll
    for (int rr = 0; rr < 8; ++rr) {
      const int o = 8 * w8 + rr;
      const uint32_t wv = *reinterpret_cast<const uint32_t*>(P.wp + o * 64 + 2 * lane);
      const uint32_t gv = *reinterpret_cast<const uint32_t*>(P.wg + o * 64 + 2 * lane);
      float sp = fmaf(lo(wv), b2.x, hi(wv) * b2.y);
      float sg = fmaf(lo(gv), c2.x, hi(gv) * c2.y);
      const uint32_t r0 = pack(lo(wv) * o2.x, hi(wv) * o2.y);
      float sw = lo(r0) + hi(r0);
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
    for (int i = 0; i < 4; ++i) mbar_init(sb + BAR + 8 * i, 1);
    fence_barrier_init();
  }
  fence_async();
  __syncthreads();
  PF(9);

  const int v = 2 * blockIdx.x + r, vstride = 2 * gridDim.x;
  auto load = [&](int tile, int st) {
    const uint32_t s = ring + st * STAGE, bar = bar0 + 8 * st;
    const int t0 = tile * 64, j0 = t0 % P.n;
    mbar_expect(bar, TX);
    tma_load(s + TRI, &P.tm_tri, bar, t0, 0);
    tma_load(s + XS, &P.tm_x, bar, 0, t0);
    tma_load(s + DYS, &P.tm_dy, bar, 0, t0);
    tma_load(s + DSS, &P.tm_ds, bar, 0, j0);
    tma_load(s + STS, &P.tm_st, bar, t0, 0);
  };
  if (leader) {
    prefetch_map(&P.tm_tri); prefetch_map(&P.tm_x); prefetch_map(&P.tm_dy); prefetch_map(&P.tm_ds);
    prefetch_map(&P.tm_st); prefetch_map(&P.tm_dt); prefetch_map(&P.tm_dxg);
    if (v < P.ntiles) load(v, 0);
  }

  float G[32], Gg[32], dps[4], dgs[4];
#pragma unroll
  for (int i = 0; i < 32; ++i) G[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) Gg[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) dps[i] = dgs[i] = 0.f;
  const int rA = 16 * w + gq, rB = rA + 8;

  int k = 0;
  for (int tile = v; tile < P.ntiles; tile += vstride, ++k) {
    const int st = k & 1;
    const uint32_t s = ring + st * STAGE;
    mbar_wait(bar0 + 8 * st, (k >> 1) & 1);
    PF(0);
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
    for (int kc = 0; kc < 4; ++kc) {
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
      tma_wait_read();                 // the previous tile's stores have read stage st ^ 1
      load(tile + vstride, st ^ 1);
    }
    PF(1);
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
    for (int ks = 0; ks < 4; ++ks)
      mma64<1, 0>(pacc, mdesc(s + TRI + 2048 * ks), kdesc(sb + WP + 32 * ks), 1);
    wg_commit();
    wg_wait<0>();
    fence_acc(pacc); fence_acc(gacc);
    PF(2);
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
    s1A = quad(s1A) * (1.f / 64); s1B = quad(s1B) * (1.f / 64);
    s2A = quad(s2A) * (1.f / 64); s2B = quad(s2B) * (1.f / 64);
    __syncwarp();   // the warp's dy rows are read before dgl overwrites them
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      stsm4(afrag(ring + DP, 16 * w, 16 * t, lane), dpf[t]);
      stsm4(afrag(s + DYS, 16 * w, 16 * t, lane), dgf[t]);
    }
    fence_async();
    bar_sync(1 + r, 128);
    PF(3);
    // ---- 5a: dxhat = dp Wp' ; G += dp^T nhat ; dpsum
    float dxh[32];
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma64<0, 1>(dxh, kdesc(ring + DP + 32 * ks), mdesc(sb + WP + 2048 * ks), ks > 0);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma64<1, 0>(G, mdesc(ring + DP + 2048 * ks), kdesc(s + TRI + 32 * ks), 1);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma8<1, 0>(dps, mdesc(ring + DP + 2048 * ks), kdesc(sb + ONES + 32 * ks), 1);
    wg_commit();
    // ---- 5b: dxg = dgl Wg ; Gg += dgl^T xhat ; dglsum
    float dxg[32];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma64<0, 1>(dxg, kdesc(s + DYS + 32 * ks), mdesc(sb + WGU + 2048 * ks), ks > 0);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma64<1, 1>(Gg, mdesc(s + DYS + 2048 * ks), mdesc(s + XS + 2048 * ks), 1);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) mma8<1, 0>(dgs, mdesc(s + DYS + 2048 * ks), kdesc(sb + ONES + 32 * ks), 1);
    wg_commit();
    wg_wait<1>();
    fence_acc(dxh); fence_acc(G); fence_acc(dps);
    PF(4);
    // ---- output-LN backward -> dt (in place of nhat); row sums came from the dProj stage
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
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
    PF(5);
    wg_wait<0>();
    fence_acc(dxg); fence_acc(Gg); fence_acc(dgs);
    PF(6);
    // ---- dxg -> the dy / dgl region (5b has finished reading it)
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      uint32_t a[4];
      c2a<8>(a, dxg, t);
      stsm4(afrag(s + DYS, 16 * w, 16 * t, lane), a);
    }
    fence_async();
    bar_sync(1 + r, 128);
    if (leader) {
      const int t0 = tile * 64;
      tma_store(&P.tm_dt, s + TRI, t0, 0);
      tma_store(&P.tm_dxg, s + DYS, 0, t0);
      tma_commit();
    }
    PF(7);
  }
  if (leader) tma_wait_all();
  bar_sync(1 + r, 128);   // this warpgroup's stores have read its ring: it now holds the raw sums
  // ---- raw sums of both warpgroups ([o][c] fp32 in their ring regions), folded by all 256 threads
  {
    float* sG = reinterpret_cast<float*>(smem + RING + r * WGSZ);
    float* sGg = sG + 4096;
    float* sDps = sGg + 4096;
    float* sDgs = sDps + 64;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = 8 * j + 2 * q;
      *reinterpret_cast<float2*>(sG + rA * 64 + c) = make_float2(G[4 * j], G[4 * j + 1]);
      *reinterpret_cast<float2*>(sG + rB * 64 + c) = make_float2(G[4 * j + 2], G[4 * j + 3]);
      *reinterpret_cast<float2*>(sGg + rA * 64 + c) = make_float2(Gg[4 * j], Gg[4 * j + 1]);
      *reinterpret_cast<float2*>(sGg + rB * 64 + c) = make_float2(Gg[4 * j + 2], Gg[4 * j + 3]);
    }
    sDps[rA] = dps[0]; sDps[rB] = dps[2]; sDgs[rA] = dgs[0]; sDgs[rB] = dgs[2];   // same value from the 4 lanes
  }
  __syncthreads();
  {
    // per warpgroup region: G [64][64] | Gg [64][64] | dpsum [64] | dglsum [64]
    constexpr int OG = 4096, ODP = 8192, ODG = 8256;
    const float* G0 = reinterpret_cast<const float*>(smem + RING);
    const float* G1 = reinterpret_cast<const float*>(smem + RING + WGSZ);
    float* out = P.part + (size_t)blockIdx.x * SLOT;
    const int c = tid & 63;
    {                                                      // dWp = G o go + dpsum bo^T; dWg = Gg o gi + dglsum bi^T
      const float go_ = P.go[c], bo_ = P.bo[c], gi_ = P.gi[c], bi_ = P.bi[c];
#pragma unroll 8
      for (int it = 0; it < 16; ++it) {
        const int i = tid + 256 * it, o = i >> 6;
        out[i] = fmaf(G0[i] + G1[i], go_, (G0[ODP + o] + G1[ODP + o]) * bo_);
        out[4224 + i] = fmaf(G0[OG + i] + G1[OG + i], gi_, (G0[ODG + o] + G1[ODG + o]) * bi_);
      }
    }
    const int qtr = tid >> 6;                              // dgo = sum_o Wp o G, dbo = Wp^T dpsum (four quarters of o)
    float sg = 0.f, sbv = 0.f;
#pragma unroll 8
    for (int o = 16 * qtr; o < 16 * qtr + 16; ++o) {
      const float wv = __bfloat162float(P.wp[o * 64 + c]);
      sg = fmaf(wv, G0[o * 64 + c] + G1[o * 64 + c], sg);
      sbv = fmaf(wv, G0[ODP + o] + G1[ODP + o], sbv);
    }
    float* sX = reinterpret_cast<float*>(smem + WGS);     // (Wg o gi) / 2 no longer read: the quarter exchange
    __syncthreads();
    if (qtr) { sX[(qtr - 1) * 128 + c] = sg; sX[(qtr - 1) * 128 + 64 + c] = sbv; }
    __syncthreads();
    if (!qtr) {
      out[4096 + c] = sg + sX[c] + sX[128 + c] + sX[256 + c];
      out[4160 + c] = sbv + sX[64 + c] + sX[192 + c] + sX[320 + c];
    }
  }
  PF(8);
  PF_FLUSH(P.prof);
}

// =====================================================================================================================
// B2
// =====================================================================================================================
struct B2Params {
  CUtensorMap tm_dl, tm_dr, tm_x, tm_dxg, tm_dy, tm_st, tm_mk, tm_dx;
  const bf16* w1;      // packed [256][64]
  const float *gi, *bi;
  float* part;         // [grid][b2::SLOT]: dW1 packed [256][64] | dgi [64] | dbi [64]
  int ntiles;
  unsigned long long* prof;
};
namespace b2 {
// W1 resident, rows of each 64-row block permuted to [gate 0-15 | proj 0-15 | gate 16-31 | proj 16-31] so that a half
// chunk (16 channels: gate + projection) is 32 contiguous rows; gate rows are halved (sigmoid via tanh(g / 2)).
// K-major B of pre, MN-major B of dxn.  The dpre gate columns carry 2 x the gradient, so dxn is exact and the dW1
// gate rows are halved when stored.
constexpr int W1 = 0;
constexpr int DL = 0, DR = 8192, XS = 16384, DXG = 24576, DYS = 32768, STS = 40960, MKS = 41472, STAGE = 41984;
constexpr int RING = 32768;             // W1: 4 packed blocks [256][64]
constexpr int XN = RING + 2 * STAGE;    // [64 tok][64] xn: K-major A of pre, MN-major B of dW1
constexpr int SCR = XN + 8192;          // [2 side][2 buffer][64 tok][64] dpre: K-major A of dxn, MN-major A of dW1
constexpr int GIB = SCR + 32768;        // gi[64] bi[64]
constexpr int REDW = GIB + 512;         // [8 warps][dgi 64 | dbi 64] per-warp sums
constexpr int BAR = REDW + 8 * 128 * 4;
constexpr int SMEM = BAR + 64;
constexpr int SLOT = 16384 + 128;
constexpr uint32_t TX = 2 * 8192 + 3 * 8192 + 512 + 256;
// permuted smem row (within a block) -> packed W1 row (packed block rows: gate 0-31, projection 32-63)
DEVI int unperm(int r) { return ((r >> 4) & 1) * 32 + (r >> 5) * 16 + (r & 15); }
}  // namespace b2

// pre of half chunk (C, HC): xn . W1 rows [64 (4 side + C) + 32 HC, + 32)  (gate 16 | projection 16)
#define B2_ISSUE_PRE(ACC, C, HC) do { \
    const uint32_t b_ = sb + W1 + (2 * side + (C)) * 8192 + (HC) * 4096; \
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

extern "C" __global__ void __launch_bounds__(256, 1) uni64_b2w(const __grid_constant__ B2Params P) {
  using namespace b2;
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, wtid = tid & 127, w = wtid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;
  const int side = __shfl_sync(0xffffffffu, tid >> 7, 0);   // warp-uniform for ptxas
  const uint32_t sb = smem_u32(smem);
  float* sGi = reinterpret_cast<float*>(smem + GIB);
  float* sBi = sGi + 64;
  float* sRedW = reinterpret_cast<float*>(smem + REDW) + (tid >> 5) * 128;
  PF_DECL

  for (int i = tid; i < 256 * 8; i += 256) {
    const int row = i >> 3, g = i & 7, blk = row >> 6, rr = row & 63;
    uint4 v4 = *reinterpret_cast<const uint4*>(P.w1 + (blk * 64 + unperm(rr)) * 64 + g * 8);
    if (!((rr >> 4) & 1))   // gate rows / 2 (exact)
      v4 = make_uint4(pack(0.5f * lo(v4.x), 0.5f * hi(v4.x)), pack(0.5f * lo(v4.y), 0.5f * hi(v4.y)),
                      pack(0.5f * lo(v4.z), 0.5f * hi(v4.z)), pack(0.5f * lo(v4.w), 0.5f * hi(v4.w)));
    *reinterpret_cast<uint4*>(smem + W1 + swz(row, g * 16)) = v4;
  }
  if (tid < 64) { sGi[tid] = P.gi[tid]; sBi[tid] = P.bi[tid]; }
  for (int i = tid; i < 8 * 128; i += 256) reinterpret_cast<float*>(smem + REDW)[i] = 0.f;
  if (tid == 0) {
    mbar_init(sb + BAR, 1); mbar_init(sb + BAR + 8, 1);
    fence_barrier_init();
  }
  fence_async();
  __syncthreads();
  PF(9);

  auto load = [&](int tile, int st) {
    const uint32_t s = sb + RING + st * STAGE, bar = sb + BAR + 8 * st;
    const int t0 = tile * 64;
    mbar_expect(bar, TX);
    tma_load(s + DL, &P.tm_dl, bar, t0, 0);
    tma_load(s + DR, &P.tm_dr, bar, t0, 0);
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

  float Gx[2][32];
#pragma unroll
  for (int c = 0; c < 2; ++c)
#pragma unroll
    for (int i = 0; i < 32; ++i) Gx[c][i] = 0.f;
  const int rA = 16 * w + gq, rB = rA + 8;
  const uint32_t scr0 = sb + SCR + side * 16384;

  int k = 0;
  for (int tile = blockIdx.x; tile < P.ntiles; tile += gridDim.x, ++k) {
    const int st = k & 1;
    const uint32_t s = sb + RING + st * STAGE;
    mbar_wait(sb + BAR + 8 * st, (k >> 1) & 1);
    PF(0);
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
    PF(1);
    const float mkA = 0.5f * sMk[rA], mkB = 0.5f * sMk[rB];
    const uint32_t dsrc = s + (side ? DR : DL);
    float p0[16], p1[16], dxn[32];
    wg_fence();
    B2_ISSUE_PRE(p0, 0, 0);
    B2_ISSUE_PRE(p1, 0, 1);
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const uint32_t scr = scr0 + (c & 1) * 8192;
      // in flight (oldest first): p0(c), p1(c), D(c - 1)
      if (c == 0) wg_wait<1>(); else wg_wait<2>();
      fence_acc(p0);
      PF(2);
      B2_HALF(p0, c, 0, scr);
      if (c == 0) wg_wait<0>(); else wg_wait<1>();
      fence_acc(p1);
      B2_HALF(p1, c, 1, scr);
      PF(3);
      fence_async();
      bar_sync(1 + side, 128);
      PF(4);
      wg_fence();
      if (c < 1) {
        B2_ISSUE_PRE(p0, c + 1, 0);
        B2_ISSUE_PRE(p1, c + 1, 1);
      }
      const uint32_t wb = sb + W1 + (2 * side + c) * 8192;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma64<0, 1>(dxn, kdesc(scr + 32 * ks), mdesc(wb + 2048 * ks), c > 0 || ks > 0);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma64<1, 1>(Gx[c], mdesc(scr + 2048 * ks), mdesc(sb + XN + 2048 * ks), 1);
      wg_commit();
      PF(5);
    }
    wg_wait<0>();
    fence_acc(dxn);
#pragma unroll
    for (int c = 0; c < 2; ++c) fence_acc(Gx[c]);
    PF(2);
    // ---- exchange: side 0 finalises rows A (16 w + gq), side 1 rows B; each hands the other its half
    {
      const uint32_t mine = s + (side ? DR : DL), other = s + (side ? DL : DR);   // dl / dr consumed
#pragma unroll
      for (int g = 0; g < 4; ++g) {   // n-tiles 2 g, 2 g + 1 of the other side's row
        const int o0 = 4 * (2 * g) + (side ? 0 : 2), o1 = 4 * (2 * g + 1) + (side ? 0 : 2);
        sts128f(mine + (g * 128 + wtid) * 16, make_float4(dxn[o0], dxn[o0 + 1], dxn[o1], dxn[o1 + 1]));
      }
      __syncthreads();
      PF(6);
      const int row = side ? rB : rA, hb = side ? 2 : 0;
      float d[16];
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const float4 v4 = lds128f(other + (g * 128 + wtid) * 16);
        const int o0 = 4 * (2 * g) + hb, o1 = 4 * (2 * g + 1) + hb;
        d[4 * g] = dxn[o0] + v4.x; d[4 * g + 1] = dxn[o0 + 1] + v4.y;
        d[4 * g + 2] = dxn[o1] + v4.z; d[4 * g + 3] = dxn[o1 + 1] + v4.w;
      }
      // ---- input-LN backward with the identity residual, one row per thread quad
      const float rs = sSt[64 + row], nmr = -sSt[row] * rs;
      float xh[16], s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t off = swz(row, (8 * j + 2 * q) * 2);
        const uint32_t xv = lds32(s + XS + off), gv = lds32(s + DXG + off);
        const float2 g2 = *reinterpret_cast<const float2*>(sGi + 8 * j + 2 * q);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = 2 * j + e;
          d[i] += e ? hi(gv) : lo(gv);
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
      // v[i] = original index 16 b4 + 8 b3 + 4 b2 + i: quantity b4, column 8 (4 b3 + 2 b2 + i / 2) + 2 q + i % 2
      const int base = ((lane >> 4) & 1) * 64 + 8 * (4 * ((lane >> 3) & 1) + 2 * ((lane >> 2) & 1)) + 2 * q;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int a = base + 8 * (i >> 1) + (i & 1);
        sRedW[a] += v[i];
      }
    }
    PF(7);
    fence_async();
    __syncthreads();
    if (tid == 0) {
      tma_store(&P.tm_dx, s + XS, 0, tile * 64);
      tma_commit();
    }
    PF(8);
  }
  if (tid == 0) tma_wait_all();
  float* out = P.part + (size_t)blockIdx.x * SLOT;
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const int row0 = 64 * (2 * side + c);
    const int pA = unperm(rA), pB = unperm(rB);
    const float fA = pA < 32 ? 0.5f : 1.f, fB = pB < 32 ? 0.5f : 1.f;   // gate rows carried 2 x the gradient
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int col = 8 * j + 2 * q;
      *reinterpret_cast<float2*>(out + (row0 + pA) * 64 + col) = make_float2(fA * Gx[c][4 * j], fA * Gx[c][4 * j + 1]);
      *reinterpret_cast<float2*>(out + (row0 + pB) * 64 + col) = make_float2(fB * Gx[c][4 * j + 2], fB * Gx[c][4 * j + 3]);
    }
  }
  __syncthreads();
  if (tid < 128) {
    const float* r8 = reinterpret_cast<const float*>(smem + REDW);
    float s = 0.f;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) s += r8[ww * 128 + tid];
    out[16384 + tid] = s;
  }
  PF(10);
  PF_FLUSH(P.prof);
}

// =====================================================================================================================
// Partial reduction (fixed order).  Outputs: [0, 16384) dW1 packed rows -> dWl / dWlg / dWr / dWrg; [16384, 16512)
// dgi, dbi (B2 slots); [16512, 20608) dWp; [20608, 20736) dgo, dbo; [20736, 24832) dWg (B1 slots).
// =====================================================================================================================
constexpr int FIN_OUT = b2::SLOT + b1::SLOT;
#if MASTER_FP32
using MasterDW = float;
__device__ float master_value(float v) { return v; }
#else
using MasterDW = bf16;
__device__ bf16 master_value(float v) { return __float2bfloat16_rn(v); }
#endif
extern "C" __global__ void __launch_bounds__(256)
uni64_finw(const float* __restrict__ p1, int g1, const float* __restrict__ p2, int g2,
           MasterDW* __restrict__ dwl, MasterDW* __restrict__ dwlg, MasterDW* __restrict__ dwr, MasterDW* __restrict__ dwrg,
           MasterDW* __restrict__ dwg, MasterDW* __restrict__ dwp, float* __restrict__ dgi, float* __restrict__ dbi,
           float* __restrict__ dgo, float* __restrict__ dbo) {
  const int e = blockIdx.x * 256 + threadIdx.x;
  if (e < b2::SLOT) {
    float s = 0.f;
    for (int g = 0; g < g2; ++g) s += p2[(size_t)g * b2::SLOT + e];
    if (e < 16384) {
      // packed row pr: block pr / 64 = 32 channels of cat(left, right); rows 0-31 gate, 32-63 projection
      const int pr = e >> 6, col = e & 63, blk = pr >> 6, within = pr & 63;
      const int c = 32 * blk + (within & 31), h = c & 63;
      MasterDW* dst = c < 64 ? (within < 32 ? dwlg : dwl) : (within < 32 ? dwrg : dwr);
      dst[h * 64 + col] = master_value(s);
    } else if (e < 16448) {
      dgi[e - 16384] = s;
    } else {
      dbi[e - 16448] = s;
    }
  } else if (e < FIN_OUT) {
    const int f = e - b2::SLOT;
    float s = 0.f;
    for (int g = 0; g < g1; ++g) s += p1[(size_t)g * b1::SLOT + f];
    if (f < 4096) dwp[f] = master_value(s);
    else if (f < 4160) dgo[f - 4096] = s;
    else if (f < 4224) dbo[f - 4160] = s;
    else dwg[f - 4224] = master_value(s);
  }
}

}  // namespace uni64
