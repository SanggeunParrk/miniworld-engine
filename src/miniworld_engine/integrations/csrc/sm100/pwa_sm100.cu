// PWA (MSAPairWeightedAveraging) fused kernels for B200 (sm_100a): the H100 path's fusion boundaries (csrc/ ln_vg.cu,
// pwa_fwd3.cu, pwa_glue3.cu, dgv_bwd.cu, pair3.cu), designed for tcgen05 / TMEM / TMA.  The two contractions o = w . v and
// dv = w^T . do (and dw = do . v^T) run on cuBLAS bmm, which beat a tcgen05 kernel of ours by 5-15 % at L 256-1024 (same bits).
//
//   pair_fwd:   w = softmax_j(mask(LN(z) . Wb^T))          LN folded into the projection, tensor core on raw z
//   ln_vg:      y = LN(m), v = y . Wv^T head-major         [H, N, S*C]
//   gate_out:   out = m + drop(sum_h sigmoid(y Wg_h^T) . o_h . Wo_h)   (y recomputed)
//   glue2:      do, dyg = dgp . Wg, dWo, dWg               dgp never leaves the chip
//   dv_bwd:     dm = LN_bwd(dyg + dv . Wv) + dout, dWv, dgamma, dbeta (tensor core column sums)
//   pair_bwd:   dz, dWb, dgamma_z, dbeta_z                 parameter gradients from per-head sums (M = db^T xhat)
#include <torch/extension.h>
#include "sm100.cuh"

using namespace sm100;

namespace {
constexpr int H = 8, C = 32, D = 64, HC = H * C;

// ---------------------------------------------------------------------------------------------
// ln_vg.  Tile = one token n x 128 MSA rows.  warp 0 streams x tiles by TMA (Wv once); warps 2-9 run the
// LayerNorm in place, two threads per row (y overwrites x in its stage); warp 1 stores y by TMA and issues
// v = y . Wv^T (M = 128 s, N = 256 = all heads, K = 64) into one of two TMEM accumulators; warps 10-13 drain
// it: TMEM lane = s, so a thread holds one row's 256 values = eight 64-byte head rows, which go into eight
// 64B-swizzled [128 s][32 c] tiles -- exactly the head-major v boxes -- with no transposition at all.
// sigmoid(x) = 0.5 + 0.5 tanh(x / 2): one MUFU op (tanh.approx, ~2^-11 relative) where ex2 + rcp took two -- the gate
// passes evaluate 256 sigmoids per row and were bound on the special-function units
// u = bf16(sigmoid(bf16(x)) * o) on a packed bf16 pair: the module's own bf16 gate (its Linear output and sigmoid are bf16),
// 0.5 + 0.5 tanh(x / 2) in bf16x2 -- four instructions per pair where the fp32 form took eleven (the gate pass is issue-bound)
__device__ __forceinline__ uint32_t gate_mul_bf16x2(float x0, float x1, uint32_t o2) {
  uint32_t h, t, g, u;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(0.5f * x1), "f"(0.5f * x0));   // high half x1, low half x0
  asm("tanh.approx.bf16x2 %0, %1;" : "=r"(t) : "r"(h));
  asm("fma.rn.bf16x2 %0, %1, %2, %2;" : "=r"(g) : "r"(t), "r"(0x3F003F00u));         // 0.5 t + 0.5
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(u) : "r"(g), "r"(o2));
  return u;
}
__device__ __forceinline__ float sigmoid_t(float x) {
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return fmaf(0.5f, t, 0.5f);
}

namespace lv {
constexpr int BS = 128;
constexpr int NST = 3;
constexpr int XT = BS * D;                                  // x / y tile (16 KiB)
constexpr int VT = H * BS * C;                              // v staging [8 heads][128 s][32] (64 KiB)
constexpr int WT = HC * D;                                  // Wv [256][64] (32 KiB)
constexpr int THREADS = 448;
constexpr int SMEM = 1024 + (NST * XT + 2 * VT + WT) * 2 + 2 * D * 4 + 512;
constexpr uint32_t IDESC = idesc_bf16(128, HC);
static_assert(SMEM <= 232448, "one CTA per SM (TMEM: 2 x 256 columns)");
}  // namespace lv

template <typename WT_>
__global__ void __launch_bounds__(lv::THREADS, 1) ln_vg_sm100(
    int N, int S, int ntiles, float eps, int want_y,
    const __grid_constant__ CUtensorMap xmap,     // m [S][N][64]: (64, N, S), box (64, 1, 128), 128B swizzle
    const __grid_constant__ CUtensorMap ymap,     // y, same (written only if want_y: inference recomputes it downstream)
    const __grid_constant__ CUtensorMap wmap,     // Wv [256][64], box (64, 256), 128B swizzle
    const __grid_constant__ CUtensorMap vmap,     // v [H*N][S*C] as (32 c, S, H*N), box (32, 128, 1), 64B swizzle
    const WT_* __restrict__ LNW, const WT_* __restrict__ LNB) {
  using namespace lv;
  const int NSB = S / BS;                        // tile t = (token t / NSB, s block t % NSB)
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sV = sX + NST * XT;
  __nv_bfloat16* sW = sV + 2 * VT;
  float* sLN = reinterpret_cast<float*>(sW + WT);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLN + 2 * D);
  uint64_t* xfull = bars;              // [NST]
  uint64_t* yfull = xfull + NST;       // [NST] count 8 (LN warps)
  uint64_t* xempty = yfull + NST;      // [NST] count 2: the GEMM's commit + y's store has read the stage
  uint64_t* accf = xempty + NST;       // [2]
  uint64_t* acce = accf + 2;           // [2] count 4
  uint64_t* wful = acce + 2;           // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wful + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { bar_init(xfull + s, 1); bar_init(yfull + s, 8); bar_init(xempty + s, 2); }
    for (int b = 0; b < 2; ++b) { bar_init(accf + b, 1); bar_init(acce + b, 4); }
    bar_init(wful, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  if (tid < D) { sLN[tid] = to_f(LNW[tid]); sLN[D + tid] = to_f(LNB[tid]); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wful, WT * 2);
      load_2d(&wmap, sW, wful, 0, 0);
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++g) {
        const int n = t / NSB, s0 = (t % NSB) * BS, st = g % NST;
        if (g >= NST) wait(xempty + st, ((g / NST) - 1) & 1);
        expect_tx(xfull + st, XT * 2);
        load_3d(&xmap, sX + st * XT, xfull + st, 0, n, s0);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      wait(wful, 0);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int n = t / NSB, s0 = (t % NSB) * BS, st = lt % NST, b = lt & 1;
        wait(yfull + st, (lt / NST) & 1);
        if (want_y) {
          store_3d(&ymap, sX + st * XT, 0, n, s0);  // y leaves from the stage it was written in
          bulk_commit();
        } else {
          arrive(xempty + st);                   // no y store: the GEMM's commit is the stage's only other reader
        }
        if (lt >= 2) wait(acce + b, ((lt >> 1) - 1) & 1);
        tc_fence_after();
        const __nv_bfloat16* y = sX + st * XT;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          mma_ss(tmem + b * HC, desc_k128(y + ks * 16), desc_k128(sW + ks * 16), IDESC, ks ? 1u : 0u);
        mma_commit(accf + b);
        mma_commit(xempty + st);
        if (want_y && lt >= 1) {                 // the previous tile's y store has read its stage: its second release
          bulk_wait_read<1>();
          arrive(xempty + (lt - 1) % NST);
        }
      }
      bulk_wait<0>();
    }
  } else if (warp <= 9) {
    // ---------------- LayerNorm: row r, channel half hf ----------------
    const int r = (warp - 2) * 16 + (lane & 15), hf = lane >> 4;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int st = lt % NST;
      wait(xfull + st, (lt / NST) & 1);
      __nv_bfloat16* xr = sX + st * XT + r * 64;
      float2 x[16];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int c8 = hf * 4 + k;
        const uint4 u = *reinterpret_cast<const uint4*>(xr + ((c8 ^ (r & 7)) << 3));
        x[k * 4 + 0] = bf2f(u.x); x[k * 4 + 1] = bf2f(u.y); x[k * 4 + 2] = bf2f(u.z); x[k * 4 + 3] = bf2f(u.w);
      }
      float2 sa_ = make_float2(0.f, 0.f), sb = sa_;
#pragma unroll
      for (int k = 0; k < 16; k += 2) { sa_ = add2(sa_, x[k]); sb = add2(sb, x[k + 1]); }
      const float2 sm = add2(sa_, sb);
      float sum = sm.x + sm.y;
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      const float mean = sum * (1.f / D);
      const float2 nm = make_float2(-mean, -mean);
      float2 va = make_float2(0.f, 0.f), vb = va;
#pragma unroll
      for (int k = 0; k < 16; k += 2) {
        x[k] = add2(x[k], nm); x[k + 1] = add2(x[k + 1], nm);
        va = fma2(x[k], x[k], va); vb = fma2(x[k + 1], x[k + 1], vb);
      }
      const float2 vs = add2(va, vb);
      float var = vs.x + vs.y;
      var += __shfl_xor_sync(0xffffffffu, var, 16);
      const float rstd = 1.f / sqrtf(var * (1.f / D) + eps);
      const float2 rs = make_float2(rstd, rstd);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int c8 = hf * 4 + k;
        const float4 g0 = *reinterpret_cast<const float4*>(sLN + c8 * 8), g1 = *reinterpret_cast<const float4*>(sLN + c8 * 8 + 4);
        const float4 b0 = *reinterpret_cast<const float4*>(sLN + D + c8 * 8), b1 = *reinterpret_cast<const float4*>(sLN + D + c8 * 8 + 4);
        const float2 y0 = fma2(mul2(x[k * 4 + 0], rs), make_float2(g0.x, g0.y), make_float2(b0.x, b0.y));
        const float2 y1 = fma2(mul2(x[k * 4 + 1], rs), make_float2(g0.z, g0.w), make_float2(b0.z, b0.w));
        const float2 y2 = fma2(mul2(x[k * 4 + 2], rs), make_float2(g1.x, g1.y), make_float2(b1.x, b1.y));
        const float2 y3 = fma2(mul2(x[k * 4 + 3], rs), make_float2(g1.z, g1.w), make_float2(b1.z, b1.w));
        uint4 o;
        o.x = pack2(y0.x, y0.y); o.y = pack2(y1.x, y1.y); o.z = pack2(y2.x, y2.y); o.w = pack2(y3.x, y3.y);
        *reinterpret_cast<uint4*>(xr + ((c8 ^ (r & 7)) << 3)) = o;
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(yfull + st);
    }
  } else {
    // ---------------- drain: TMEM lane = s, 256 columns = 8 heads x 32 c ----------------
    const int q = warp & 3, r = q * 32 + lane;
    const bool leader = (warp == 10 && lane == 0);
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int n = t / NSB, s0 = (t % NSB) * BS, b = lt & 1;
      wait(accf + b, (lt >> 1) & 1);
      tc_fence_after();
      if (leader) bulk_wait_read<1>();           // the v stores of tile lt - 2 have left staging buffer b
      named_sync(2, 128);
      __nv_bfloat16* vb = sV + b * VT;
#pragma unroll 1
      for (int h = 0; h < H; ++h) {
        float v[32];
        tmem_ld32(tmem_at(tmem + b * HC, q * 32, h * C), v);
        tmem_wait_ld();
        if (h == H - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + b); }
        __nv_bfloat16* row = vb + (h * BS + r) * C;
#pragma unroll
        for (int c8 = 0; c8 < 4; ++c8) {
          uint4 o;
          o.x = pack2(v[c8 * 8 + 0], v[c8 * 8 + 1]); o.y = pack2(v[c8 * 8 + 2], v[c8 * 8 + 3]);
          o.z = pack2(v[c8 * 8 + 4], v[c8 * 8 + 5]); o.w = pack2(v[c8 * 8 + 6], v[c8 * 8 + 7]);
          *reinterpret_cast<uint4*>(row + ((c8 ^ ((r >> 1) & 3)) << 3)) = o;
        }
      }
      fence_proxy_async();
      named_sync(2, 128);
      if (leader) {
        for (int h = 0; h < H; ++h) store_3d(&vmap, vb + h * BS * C, 0, s0, h * N + n);
        bulk_commit();
      }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---------------------------------------------------------------------------------------------
// PWA forward, second half (the split fusion): with o = w . v already in global memory (cuBLAS bmm), per 128-row tile of
// the (s, i) rows:  g = sigmoid(y Wg^T) [128 x 256],  u = g . o,  out = msa + drop(bf16(u Wo^T)).
//   warp 0: o | y tiles (2-stage ring) and the resident Wg, Wo;  warp 1: tcgen05 issue;  warps 2-9: the gate drain
//   (u back to TMEM, packed, as the out GEMM's A);  warps 10-13: the output epilogue.
// The work goes in half-tile steps (heads 0-3 / 4-7): the gate GEMM of the next half runs while this half drains, and the
// out GEMM takes u in two K halves, so the drain warps never wait on a GEMM round trip.
namespace go {
constexpr int BM = 128;
constexpr int OT = BM * HC;                                // o tile [8 heads][128 rows][32], 64B swizzle
constexpr int YT = BM * D;                                 // msa tile [128][64] -> y in place (the gate GEMM's A)
constexpr int STAGE = OT + YT;                             // 80 KiB
constexpr int NST = 2;
constexpr int WGT = HC * D;                                // Wg [256][64]
constexpr int WOT = D * HC;                                // Wo [4 k-blocks][64 d][64 k]
constexpr int THREADS = 576;                               // + warps 14-17: the LayerNorm (y = LN(msa) in place)
constexpr int SMEM = 1024 + (NST * STAGE + WGT + WOT) * 2 + 1024;
constexpr int COL_G = 0, COL_U = 256, COL_OUT = 384;       // gate [256] fp32 | u [128] packed bf16 | out [2][64]
constexpr uint32_t ID_G = idesc_bf16(128, 128, 0, 0);
constexpr uint32_t ID_OUT = idesc_bf16(128, D, 0, 0);
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace go

template <typename WT_>
__global__ void __launch_bounds__(go::THREADS, 1) pwa_gate_out_sm100(
    int N, int ntiles, float eps, const WT_* __restrict__ LNW, const WT_* __restrict__ LNB,
    const __grid_constant__ CUtensorMap omap,     // o head-major [H*N][S*C], box (32, 128), 64B swizzle
    const __grid_constant__ CUtensorMap ymap,     // msa [M][64], box (64, 128), 128B swizzle: y = LN(msa) is recomputed here
    const __grid_constant__ CUtensorMap gmap,     // Wg [256][64], box (64, 256), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,     // Wo [64][256], box (64, 64), 128B swizzle
    const __nv_bfloat16* __restrict__ MSA, __nv_bfloat16* __restrict__ OUT,
    const __nv_bfloat16* __restrict__ DMASK, float dscale) {
  using namespace go;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sWg = sStage + NST * STAGE;
  __nv_bfloat16* sWo = sWg + WGT;
  float* sLN = reinterpret_cast<float*>(sWo + WOT);                // gamma[64], beta[64] (16-byte aligned: float4 reads)
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLN + 2 * D);
  uint64_t* full = bars;             // [NST]
  uint64_t* empty = full + NST;      // [NST] count 9: the gate GEMMs' commit (y) and the 8 drain warps (o)
  uint64_t* gf = empty + NST;        // [2]   gate half ready
  uint64_t* ge = gf + 2;             // [2]   count 8: gate half drained
  uint64_t* uf = ge + 2;             // [2]   count 8: u half written
  uint64_t* ue = uf + 2;             // [2]   the out GEMM read u half
  uint64_t* outf = ue + 2;           // [2]
  uint64_t* oute = outf + 2;         // [2]   count 4
  uint64_t* wf = oute + 2;           // [1]
  uint64_t* yrdy = wf + 1;           // [NST] count 4: the LayerNorm wrote y over the stage's msa tile
  uint32_t* tslot = reinterpret_cast<uint32_t*>(yrdy + NST);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(full + k, 1); bar_init(empty + k, 9); }
    for (int k = 0; k < 2; ++k) { bar_init(gf + k, 1); bar_init(ge + k, 8); bar_init(uf + k, 8); bar_init(ue + k, 1);
                                  bar_init(outf + k, 1); bar_init(oute + k, 4); }
    bar_init(wf, 1);
    for (int k2 = 0; k2 < NST; ++k2) bar_init(yrdy + k2, 4);
    bar_init_fence();
  }
  if (tid < D) { sLN[tid] = to_f(LNW[tid]); sLN[D + tid] = to_f(LNB[tid]); }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int nt = (int)blockIdx.x < ntiles ? (ntiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
  auto stage = [&](int lt) { return sStage + (lt % NST) * STAGE; };

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wf, (WGT + WOT) * 2);
      load_2d(&gmap, sWg, wf, 0, 0);
      for (int kb = 0; kb < 4; ++kb) load_2d(&wmap, sWo + kb * D * 64, wf, kb * 64, 0);
      for (int lt = 0; lt < nt; ++lt) {
        const int m0 = ((int)blockIdx.x + lt * (int)gridDim.x) * BM, st = lt % NST;
        if (lt >= NST) wait(empty + st, ((lt / NST) - 1) & 1);
        __nv_bfloat16* p = stage(lt);
        expect_tx(full + st, STAGE * 2);
        // a tile is 128 i of one s: o arrives as 64-byte pieces per (head, i), but the residual / output rows of a warp are 4 KiB
        // contiguous (one token x 128 s tiles read o as 8 KiB runs and measured 5-12 % slower: their row stores scatter)
        const int s = m0 / N, i0 = m0 % N;
        for (int h = 0; h < H; ++h) load_2d(&omap, p + h * BM * C, full + st, s * C, h * N + i0);
        load_2d(&ymap, p + OT, full + st, 0, m0);
      }
    }
  } else if (warp == 1) {
    wait(wf, 0);
    auto gate = [&](int k) {                     // gate half hf of tile lt: g[:, 128 hf + ...] = y . Wg[128 hf ...]^T
      const int lt = k >> 1, hf = k & 1, st = lt % NST;
      if (hf == 0) wait(yrdy + st, (lt / NST) & 1);
      if (k >= 2) wait(ge + hf, ((k >> 1) - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* y = stage(lt) + OT;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          mma_ss(tmem + COL_G + hf * 128, desc_k128(y + ks * 16), desc_k128(sWg + hf * 128 * 64 + ks * 16), ID_G, ks ? 1u : 0u);
        mma_commit(gf + hf);
        if (hf == 1) mma_commit(empty + st);
      }
      __syncwarp();
    };
    auto outg = [&](int k) {                     // out += u[:, half hf] . Wo[:, half hf]^T  (A from TMEM)
      const int lt = k >> 1, hf = k & 1, b = lt & 1;
      wait(uf + hf, (k >> 1) & 1);
      if (hf == 0 && lt >= 2) wait(oute + b, ((lt >> 1) - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const int kk = hf * 128 + ks * 16;
          mma_ts(tmem + COL_OUT + b * D, tmem + COL_U + hf * 64 + ks * 8, desc_k128(sWo + (kk >> 6) * D * 64 + ((kk & 63) >> 4) * 16),
                 ID_OUT, (hf | ks) ? 1u : 0u);
        }
        mma_commit(ue + hf);
        if (hf == 1) mma_commit(outf + b);
      }
      __syncwarp();
    };
    const int K = 2 * nt;
    if (K > 0) gate(0);
    for (int k = 0; k < K; ++k) {
      if (k + 1 < K) gate(k + 1);
      outg(k);
    }
  } else if (warp <= 9) {
    const int q = warp & 3, pp = (warp - 2) >> 2, r = q * 32 + lane;   // TMEM lane = row; pp: which 64 of the half
    for (int k = 0; k < 2 * nt; ++k) {
      const int lt = k >> 1, hf = k & 1, st = lt % NST;
      wait(gf + hf, (k >> 1) & 1);
      wait(full + st, (lt / NST) & 1);
      if (k >= 2) wait(ue + hf, ((k >> 1) - 1) & 1);
      tc_fence_after();
      const __nv_bfloat16* ob0 = stage(lt) + r * C;
#pragma unroll
      for (int ch = 0; ch < 2; ++ch) {
        float g[32];
        tmem_ld32(tmem_at(tmem + COL_G, q * 32, hf * 128 + pp * 64 + ch * 32), g);
        tmem_wait_ld();
        uint32_t up[16];
        const __nv_bfloat16* ob = ob0 + (hf * 4 + pp * 2 + ch) * BM * C;   // this chunk's head
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint4 u4 = *reinterpret_cast<const uint4*>(ob + ((e ^ ((r >> 1) & 3)) << 3));
          const uint32_t ow[4] = {u4.x, u4.y, u4.z, u4.w};
#pragma unroll
          for (int w2 = 0; w2 < 4; ++w2) {
            const int c = e * 8 + w2 * 2;
            up[e * 4 + w2] = gate_mul_bf16x2(g[c], g[c + 1], ow[w2]);
          }
        }
        tmem_st8(tmem_at(tmem + COL_U, q * 32, hf * 64 + pp * 32 + ch * 16), up);
        tmem_st8(tmem_at(tmem + COL_U, q * 32, hf * 64 + pp * 32 + ch * 16 + 8), up + 8);
      }
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) { arrive(ge + hf); arrive(uf + hf); if (hf == 1) arrive(empty + st); }
    }
  } else {
    // ---- output: bf16(update), dropout, + residual, straight to global (a warp's rows are 4 KiB contiguous) ----
    const int q = warp & 3, r = q * 32 + lane;
    // y = LN(msa) for tile lt, in place, one thread per row -- the same fp32 operations in the same order as ln_vg's two
    // half-row threads (each half's sums, then the two halves), so y is bit-identical to the y ln_vg keeps for training
    auto layernorm = [&](int lt) {
      const int st = lt % NST;
      wait(full + st, (lt / NST) & 1);
      __nv_bfloat16* xr = stage(lt) + OT + r * 64;
      float2 x[2][16];
      float hs[2];
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
          const int c8 = hf * 4 + k2;
          const uint4 u = *reinterpret_cast<const uint4*>(xr + ((c8 ^ (r & 7)) << 3));
          x[hf][k2 * 4 + 0] = bf2f(u.x); x[hf][k2 * 4 + 1] = bf2f(u.y); x[hf][k2 * 4 + 2] = bf2f(u.z); x[hf][k2 * 4 + 3] = bf2f(u.w);
        }
        float2 sa_ = make_float2(0.f, 0.f), sb = sa_;
#pragma unroll
        for (int k2 = 0; k2 < 16; k2 += 2) { sa_ = add2(sa_, x[hf][k2]); sb = add2(sb, x[hf][k2 + 1]); }
        const float2 sm = add2(sa_, sb);
        hs[hf] = sm.x + sm.y;
      }
      const float mean = (hs[0] + hs[1]) * (1.f / D);
      const float2 nm = make_float2(-mean, -mean);
      float hv[2];
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        float2 va = make_float2(0.f, 0.f), vb = va;
#pragma unroll
        for (int k2 = 0; k2 < 16; k2 += 2) {
          x[hf][k2] = add2(x[hf][k2], nm); x[hf][k2 + 1] = add2(x[hf][k2 + 1], nm);
          va = fma2(x[hf][k2], x[hf][k2], va); vb = fma2(x[hf][k2 + 1], x[hf][k2 + 1], vb);
        }
        const float2 vs = add2(va, vb);
        hv[hf] = vs.x + vs.y;
      }
      const float rstd = 1.f / sqrtf((hv[0] + hv[1]) * (1.f / D) + eps);
      const float2 rs = make_float2(rstd, rstd);
#pragma unroll
      for (int hf = 0; hf < 2; ++hf)
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
          const int c8 = hf * 4 + k2;
          const float4 g0 = *reinterpret_cast<const float4*>(sLN + c8 * 8), g1 = *reinterpret_cast<const float4*>(sLN + c8 * 8 + 4);
          const float4 b0 = *reinterpret_cast<const float4*>(sLN + D + c8 * 8), b1 = *reinterpret_cast<const float4*>(sLN + D + c8 * 8 + 4);
          const float2 y0 = fma2(mul2(x[hf][k2 * 4 + 0], rs), make_float2(g0.x, g0.y), make_float2(b0.x, b0.y));
          const float2 y1 = fma2(mul2(x[hf][k2 * 4 + 1], rs), make_float2(g0.z, g0.w), make_float2(b0.z, b0.w));
          const float2 y2 = fma2(mul2(x[hf][k2 * 4 + 2], rs), make_float2(g1.x, g1.y), make_float2(b1.x, b1.y));
          const float2 y3 = fma2(mul2(x[hf][k2 * 4 + 3], rs), make_float2(g1.z, g1.w), make_float2(b1.z, b1.w));
          uint4 o;
          o.x = pack2(y0.x, y0.y); o.y = pack2(y1.x, y1.y); o.z = pack2(y2.x, y2.y); o.w = pack2(y3.x, y3.y);
          *reinterpret_cast<uint4*>(xr + ((c8 ^ (r & 7)) << 3)) = o;
        }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(yrdy + st);
    };
    if (warp >= 14) {
      for (int lt = 0; lt < nt; ++lt) layernorm(lt);
    } else
    for (int lt = 0; lt < nt; ++lt) {
      const int row = ((int)blockIdx.x + lt * (int)gridDim.x) * BM + r, b = lt & 1;

      wait(outf + b, (lt >> 1) & 1);
      tc_fence_after();
      uint32_t pk[D / 2];
      {
        float v[D];
        tmem_ld32(tmem_at(tmem + COL_OUT + b * D, q * 32, 0), v);
        tmem_ld32(tmem_at(tmem + COL_OUT + b * D, q * 32, 32), v + 32);
        tmem_wait_ld();
#pragma unroll
        for (int c = 0; c < D / 2; ++c) pk[c] = pack2(v[2 * c], v[2 * c + 1]);
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(oute + b);
      const uint4* rp = reinterpret_cast<const uint4*>(MSA + (size_t)row * D);
      uint4* op = reinterpret_cast<uint4*>(OUT + (size_t)row * D);
      const __nv_bfloat16* dr = DMASK != nullptr ? DMASK + (size_t)(row % N) * D : nullptr;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        uint4 rv = __ldg(rp + c8);
        uint4 dmv = dr != nullptr ? *reinterpret_cast<const uint4*>(dr + c8 * 8) : make_uint4(0, 0, 0, 0);
        uint32_t* rw = reinterpret_cast<uint32_t*>(&rv);
        const uint32_t* dw = reinterpret_cast<const uint32_t*>(&dmv);
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
          // the stock module rounds the update to bf16 before the (dropout and the) residual add
          const float2 uu = bf2f(pk[c8 * 4 + k2]);
          float u0 = uu.x, u1 = uu.y;
          if (dr != nullptr) {
            const float2 m2 = bf2f(dw[k2]);
            u0 = __bfloat162float(__float2bfloat16(u0 * (m2.x * dscale))); u1 = __bfloat162float(__float2bfloat16(u1 * (m2.y * dscale)));
          }
          const float2 res = bf2f(rw[k2]);
          rw[k2] = pack2(res.x + u0, res.y + u1);
        }
        op[c8] = rv;
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---------------------------------------------------------------------------------------------
// PWA backward glue, second form: the dgp consumers move in, so dgp never reaches global memory.  Per (128-row i tile, 2 s),
// per head, as pwa_glue plus
//   dWg_h^T[d][c] += sum_(s,i) y[s,i,d] dgp[s,i,c]      (tensor core, TMEM over the CTA's tiles: the dWo scheme)
//   dyg[s,i,:]    += dgp_h[s,i,:] . Wg_h                 (tensor core, TMEM over the tile's heads)
// and dyg -- the gate branch's share of dy = dgp Wg + dv Wv -- leaves once per tile, bf16 natural [S][N][64] (dv_bwd adds
// dv Wv).  TMEM: g, du single-buffered (the drain loads them first thing, so the next head's GEMMs still run a whole head
// ahead), dWo and dWg^T 128 columns each (two lane halves), dyg 2 s x 64.  The dyg tile is staged in the head-7 o stage and
// go buffer, both free by then.
namespace gl2 {
constexpr int BI = 128, BS = 2, NO = BS * C;
constexpr int OT = BS * BI * C;                            // o / go / dgp tile [2 s][128 i][32], 64B swizzle
constexpr int DT = BI * NO;                                // do tile [128 i][64 (s,c)], 128B swizzle
constexpr int RT = BS * BI * D;                            // dres / y tile [2 s][128 i][64], 128B swizzle
constexpr int WH = 2 * C * D;                              // Wg_h | WoT_h, each [32 c][64 d], 128B swizzle
constexpr int NST = 3;
constexpr int THREADS = 320;                               // warp 0 producer, warp 1 MMA, warps 2-9 drain (warps q and q + 4: one s each)
constexpr int SMEM = 1024 + (NST * OT + 2 * DT + 2 * OT + 2 * OT + 2 * RT + 2 * WH) * 2 + 512;
constexpr int COL_G = 0, COL_DU = NO, COL_WO = 2 * NO, COL_WG = 2 * NO + 4 * C, COL_YG = 2 * NO + 8 * C;   // 0, 64, 128, 256, 384
constexpr int PW = 2 * HC;                                 // a CTA's slab row: dWo [d][256] | dWg^T [d][256]
constexpr uint32_t ID_GD = idesc_bf16(128, C, 0, 0);
constexpr uint32_t ID_W = idesc_bf16(64, C, 1, 1);         // A = dres'^T / y^T (MN-major), B = go / dgp (MN-major)
constexpr uint32_t ID_YG = idesc_bf16(128, D, 0, 1);       // A = dgp_h [128 i][32 c] (K-major), B = Wg_h [32 c][64 d] (MN-major)
static_assert(COL_YG + BS * D == 512, "TMEM: 512 columns");
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace gl2

__global__ void __launch_bounds__(gl2::THREADS, 1) pwa_glue2_sm100(
    int N, int S, int ntiles,
    const __grid_constant__ CUtensorMap omap,     // o head-major as (32 c, H*N, S), box (32, 128, 2), 64B swizzle
    const __grid_constant__ CUtensorMap rmap,     // dres [S][N][64] as (64, N, S), box (64, 128, 1), 128B swizzle
    const __grid_constant__ CUtensorMap ymap,     // y, same
    const __grid_constant__ CUtensorMap gmap,     // Wg [HC][64], box (64, 32)
    const __grid_constant__ CUtensorMap wotmap,   // Wo^T [HC][64], box (64, 32)
    const __grid_constant__ CUtensorMap domap,    // do [H*N][S*C], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap ygmap,    // dyg [S][N][64] as (64, N, S), box (64, 128, 1), 128B swizzle
    const __nv_bfloat16* __restrict__ DMASK, float dscale,
    float* __restrict__ DWO) {                    // [grid][64][512]: dWo | dWg^T
  using namespace gl2;
  const int NIB = N / BI;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(smb);   // [NST][OT]
  __nv_bfloat16* sDO = sO + NST * OT;                            // [2][DT]
  __nv_bfloat16* sDGP = sDO + 2 * DT;                            // [2][OT]
  __nv_bfloat16* sGO = sDGP + 2 * OT;                            // [2][OT]
  __nv_bfloat16* sR = sGO + 2 * OT;                              // dres' tile [RT]
  __nv_bfloat16* sY = sR + RT;                                   // y tile [RT]
  __nv_bfloat16* sWH = sY + RT;                                  // [2][WH]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sWH + 2 * WH);
  uint64_t* of = bars;               // [NST] o tile landed
  uint64_t* oe = of + NST;           // [NST] count 8: drain read it
  uint64_t* whf = oe + NST;          // [2]
  uint64_t* whe = whf + 2;           // [2]  the head's GEMMs reading Wg_h / WoT_h retired (the dyg GEMM is the last)
  uint64_t* tf = whe + 2;            // [1]  dres and y of the tile landed
  uint64_t* tr = tf + 1;             // [1]  count 8: dres masked (ready for the GEMMs)
  uint64_t* te = tr + 1;             // [1]  every GEMM reading the tile's dres / y retired
  uint64_t* accf = te + 1;           // [1]  g_h, du_h complete
  uint64_t* acce = accf + 1;         // [1]  count 8: loaded by the drain
  uint64_t* gof = acce + 1;          // [2]  count 8: go_h, dgp_h written
  uint64_t* goe = gof + 2;           // [2]  the GEMMs reading go_h / dgp_h retired
  uint64_t* ygf = goe + 2;           // [1]  the tile's dyg complete
  uint64_t* yge = ygf + 1;           // [1]  count 8: dyg loaded by the drain
  uint64_t* wdone = yge + 1;         // [1]  the CTA's last dWo / dWg GEMM retired
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wdone + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(of + k, 1); bar_init(oe + k, 8); }
    for (int k = 0; k < 2; ++k) { bar_init(whf + k, 1); bar_init(whe + k, 1); bar_init(gof + k, 8); bar_init(goe + k, 1); }
    bar_init(tf, 1); bar_init(tr, 8); bar_init(te, 1); bar_init(accf, 1); bar_init(acce, 8); bar_init(ygf, 1); bar_init(yge, 8); bar_init(wdone, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      int gh = 0, lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int i0 = (t % NIB) * BI, s0 = (t / NIB) * BS;
        if (lt >= 1) wait(te, (lt - 1) & 1);
        expect_tx(tf, 2 * RT * 2);
        for (int si = 0; si < BS; ++si) {
          load_3d(&rmap, sR + si * BI * D, tf, 0, i0, s0 + si);
          load_3d(&ymap, sY + si * BI * D, tf, 0, i0, s0 + si);
        }
        for (int h = 0; h < H; ++h, ++gh) {
          const int wb = gh & 1, st = gh % NST;
          if (gh >= 2) wait(whe + wb, ((gh >> 1) - 1) & 1);
          expect_tx(whf + wb, WH * 2);
          load_2d(&gmap, sWH + wb * WH, whf + wb, 0, h * C);
          load_2d(&wotmap, sWH + wb * WH + C * D, whf + wb, 0, h * C);
          if (gh >= NST) wait(oe + st, ((gh / NST) - 1) & 1);
          expect_tx(of + st, OT * 2);
          load_3d(&omap, sO + st * OT, of + st, 0, h * N + i0, s0);
        }
      }
    }
  } else if (warp == 1) {
    int gh = 0, lt = 0;
    // the GEMMs over go_hp / dgp_hp: dWo_hp += dres'^T go, dWg_hp^T += y^T dgp, dyg += dgp . Wg_hp
    auto dwo_gemm = [&](int ghp, int hp, int ltp, int tile_last, int cta_last) {
      const int b = ghp & 1;
      wait(gof + b, (ghp >> 1) & 1);
      if (hp == 0 && ltp >= 1) wait(yge, (ltp - 1) & 1);        // the previous tile's dyg has been read out of TMEM
      tc_fence_after();
      if (elect_one()) {
        const uint32_t lh = (uint32_t)(hp >> 2) << 20;          // lane half 16 * (hp >> 2)
        const __nv_bfloat16* go = sGO + b * OT;
        const __nv_bfloat16* dg = sDGP + b * OT;
        const __nv_bfloat16* wg = sWH + b * WH;
        // dyg first: its commit returns Wg_hp's buffer to the producer before the 32 weight-gradient MMAs, so the load of
        // Wg_(hp+2) is not queued behind them (the next-but-one head's gate GEMM waits on it)
#pragma unroll
        for (int si = 0; si < BS; ++si)
#pragma unroll
          for (int ks = 0; ks < C / 16; ++ks)
            mma_ss(tmem + COL_YG + si * D, desc_k64(dg + si * BI * C + ks * 16), desc_mn128(wg + ks * 16 * 64, 0), ID_YG, (hp | ks) ? 1u : 0u);
        mma_commit(whe + b);
#pragma unroll
        for (int ks = 0; ks < BS * BI / 16; ++ks) {
          mma_ss(tmem + lh + COL_WO + (hp & 3) * C, desc_mn128(sR + ks * 16 * 64, 0), sdesc(sa(go + ks * 16 * C), 0, 512, 4), ID_W, 1u);
          mma_ss(tmem + lh + COL_WG + (hp & 3) * C, desc_mn128(sY + ks * 16 * 64, 0), sdesc(sa(dg + ks * 16 * C), 0, 512, 4), ID_W, 1u);
        }
        mma_commit(goe + b);
        if (tile_last) { mma_commit(te); mma_commit(ygf); }
        if (cta_last) mma_commit(wdone);
      }
      __syncwarp();
    };
    int pgh = -1, ph = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      wait(tr, lt & 1);                          // dres' and y of the tile in place
      for (int h = 0; h < H; ++h, ++gh) {
        const int b = gh & 1;
        if (gh >= 1) wait(acce, (gh - 1) & 1);   // single g / du buffer: the drain has loaded head gh - 1's
        wait(whf + b, (gh >> 1) & 1);
        tc_fence_after();
        if (elect_one()) {
          const __nv_bfloat16* wg = sWH + b * WH;
          const __nv_bfloat16* wot = wg + C * D;
#pragma unroll
          for (int si = 0; si < BS; ++si)
#pragma unroll
            for (int ks = 0; ks < D / 16; ++ks) {
              mma_ss(tmem + COL_G + si * C, desc_k128(sY + si * BI * D + ks * 16), desc_k128(wg + ks * 16), ID_GD, ks ? 1u : 0u);
              mma_ss(tmem + COL_DU + si * C, desc_k128(sR + si * BI * D + ks * 16), desc_k128(wot + ks * 16), ID_GD, ks ? 1u : 0u);
            }
          mma_commit(accf);
        }
        __syncwarp();
        if (pgh >= 0) dwo_gemm(pgh, ph, lt, 0, 0);
        if (h == H - 1) {
          dwo_gemm(gh, h, lt, 1, t + (int)gridDim.x >= ntiles);
          pgh = -1;
        } else {
          pgh = gh; ph = h;
        }
      }
    }
  } else {
    const int q = warp & 3, r = q * 32 + lane;   // TMEM lane = i row of the tile
    const int si = (warp - 2) >> 2;              // this warp's s of the tile's two: half of every head's elementwise work
    const bool leader = (warp == 2 && lane == 0);
    if (si == 0) {                               // clear the dWo / dWg^T accumulators (lanes of this warp's sub-partition)
      uint32_t z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int c0 = 0; c0 < 8 * C; c0 += 8) tmem_st8(tmem_at(tmem + COL_WO, q * 32, c0), z);
      tmem_wait_st();
    }
    tc_fence_before();
    named_sync(1, 256);
    int gh = 0, lt = 0;
    int pst = 0, pb = 0, pi0 = 0, ps0 = 0;       // the previous tile's head-7 o stage, go buffer and coordinates
    // a tile's dyg: TMEM -> bf16 -> (its head-7 o stage | head-7 go buffer) -> global; then the o stage goes back to the producer
    auto dyg_out = [&](int ltp) {
      wait(ygf, ltp & 1);
      tc_fence_after();
      float yv[D];
      tmem_ld32(tmem_at(tmem + COL_YG + si * D, q * 32, 0), yv);
      tmem_ld32(tmem_at(tmem + COL_YG + si * D, q * 32, 32), yv + 32);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(yge);
      __nv_bfloat16* stg = si == 0 ? sO + pst * OT : sGO + pb * OT;   // [128 i][64 d], 128B swizzle (16 KiB each)
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        uint4 o;
        uint32_t* w4 = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for (int k = 0; k < 4; ++k) w4[k] = pack2(yv[c8 * 8 + 2 * k], yv[c8 * 8 + 2 * k + 1]);
        *reinterpret_cast<uint4*>(stg + r * 64 + ((c8 ^ (r & 7)) << 3)) = o;
      }
      fence_proxy_async();
      named_sync(1, 256);
      if (leader) {
        store_3d(&ygmap, sO + pst * OT, 0, pi0, ps0);
        store_3d(&ygmap, sGO + pb * OT, 0, pi0, ps0 + 1);
        bulk_commit();
        bulk_wait_read<0>();
      }
      named_sync(1, 256);
      if (lane == 0) arrive(oe + pst);         // the o stage is free again
    };
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i0 = (t % NIB) * BI, s0 = (t / NIB) * BS;
      wait(tf, lt & 1);
      if (DMASK != nullptr) {                    // dres' = dres . keep / (1 - p), once, in place (row r, this warp's s)
        const __nv_bfloat16* dr = DMASK + (size_t)(i0 + r) * D;
        __nv_bfloat16* row = sR + (si * BI + r) * D;
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          const int off = (c8 ^ (r & 7)) << 3;
          uint4 v = *reinterpret_cast<uint4*>(row + off);
          const uint4 kp = __ldg(reinterpret_cast<const uint4*>(dr + c8 * 8));
          const uint32_t* vw = reinterpret_cast<const uint32_t*>(&v);
          const uint32_t* kw = reinterpret_cast<const uint32_t*>(&kp);
          uint4 o;
          uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float2 a = bf2f(vw[k]), m = bf2f(kw[k]);
            ow[k] = pack2((a.x * m.x) * dscale, (a.y * m.y) * dscale);
          }
          *reinterpret_cast<uint4*>(row + off) = o;
        }
        fence_proxy_async();
      }
      __syncwarp();
      if (lane == 0) arrive(tr);
      for (int h = 0; h < H; ++h, ++gh) {
        const int b = gh & 1, st = gh % NST;
        wait(accf, gh & 1);
        tc_fence_after();
        float gt[C], du[C];                      // this warp's s: 32 channels of gate and du
        tmem_ld32(tmem_at(tmem + COL_G, q * 32, si * C), gt);
        tmem_ld32(tmem_at(tmem + COL_DU, q * 32, si * C), du);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(acce);
        wait(of + st, (gh / NST) & 1);
        if (gh >= 2) wait(goe + b, ((gh >> 1) - 1) & 1);        // the GEMMs of head gh - 2 have read go / dgp buffer b
        if (leader) bulk_wait_read<1>();                         // the do store of head gh - 2 has read staging b
        named_sync(1, 256);
        __nv_bfloat16* dob = sDO + b * DT;
        __nv_bfloat16* dgb = sDGP + b * OT;
        __nv_bfloat16* gob = sGO + b * OT;
        const __nv_bfloat16* ob = sO + st * OT;
        {
          const int orow = (si * BI + r) * C;
#pragma unroll
          for (int c8 = 0; c8 < 4; ++c8) {
            const int off64 = (c8 ^ ((r >> 1) & 3)) << 3;
            const uint4 ov = *reinterpret_cast<const uint4*>(ob + orow + off64);
            const uint32_t* ow = reinterpret_cast<const uint32_t*>(&ov);
            uint4 vdo, vdg, vgo;
            uint32_t* wdo = reinterpret_cast<uint32_t*>(&vdo);
            uint32_t* wdg = reinterpret_cast<uint32_t*>(&vdg);
            uint32_t* wgo = reinterpret_cast<uint32_t*>(&vgo);
#pragma unroll
            for (int k = 0; k < 4; ++k) {             // packed pairs: do = du g, dgp = du o g (1 - g), go = g o
              const int c = c8 * 8 + 2 * k;
              const float2 o2 = bf2f(ow[k]);
              const float2 g2 = make_float2(sigmoid_t(gt[c]), sigmoid_t(gt[c + 1]));
              const float2 d2 = make_float2(du[c], du[c + 1]);
              const float2 do2 = mul2(d2, g2);
              const float2 gg = fma2(g2, make_float2(-g2.x, -g2.y), g2);                           // g - g^2
              const float2 dg2 = mul2(mul2(d2, o2), gg);
              const float2 go2 = mul2(g2, o2);
              wdo[k] = pack2(do2.x, do2.y); wdg[k] = pack2(dg2.x, dg2.y); wgo[k] = pack2(go2.x, go2.y);
            }
            *reinterpret_cast<uint4*>(gob + orow + off64) = vgo;
            *reinterpret_cast<uint4*>(dgb + orow + off64) = vdg;                                   // go's layout: both are B operands
            const int cdo = si * 4 + c8;                                                          // do row r: 64 (s,c) columns
            *reinterpret_cast<uint4*>(dob + r * 64 + ((cdo ^ (r & 7)) << 3)) = vdo;
          }
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) { arrive(gof + b); if (h != H - 1) arrive(oe + st); }
        named_sync(1, 256);
        if (leader) {
          store_2d(&domap, dob, s0 * C, h * N + i0);
          bulk_commit();
        }
        if (h == 0 && lt >= 1) dyg_out(lt - 1);    // the previous tile's dyg, one head late: its GEMMs have long retired
        if (h == H - 1) { pst = st; pb = b; pi0 = i0; ps0 = s0; }
      }
    }
    if (lt >= 1) dyg_out(lt - 1);
    if (leader) bulk_wait<0>();
    // this CTA's slab: dWo [d][256] | dWg^T [d][256]; lane half 0 = heads 0-3, half 1 = heads 4-7; row d = q * 16 + (lane & 15)
    if (si == 0) {
      wait(wdone, 0);
      tc_fence_after();
      const int half = lane >> 4, d = q * 16 + (lane & 15);
      float* slab = DWO + (size_t)blockIdx.x * D * PW + (size_t)d * PW + half * 4 * C;
#pragma unroll 1
      for (int c0 = 0; c0 < 8 * C; c0 += 32) {         // columns 0-127: dWo, 128-255: dWg^T
        float v[32];
        tmem_ld32(tmem_at(tmem + COL_WO, q * 32, c0), v);
        tmem_wait_ld();
        float* dst = slab + (c0 >= 4 * C ? HC + c0 - 4 * C : c0);
#pragma unroll
        for (int k = 0; k < 32; k += 4) *reinterpret_cast<float4*>(dst + k) = make_float4(v[k], v[k + 1], v[k + 2], v[k + 3]);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// PWA backward tail, second form (with pwa_glue2): reads dv head-major and the gate branch's dyg, and produces
//   dy = dyg + dv . Wv (TMEM + the dyg tile), dm = LayerNorm_bwd(dy; x, gamma) + dout, dWv = dv^T . y, dgamma, dbeta.
// Tile = one token x 128 s.  dv streams as four [128][64] k-blocks (two heads each) with the matching Wv^T block; the four
// column blocks of dWv^T sit in two TMEM lane halves (blocks 0-1 / 2-3) of 128 columns.
namespace dv2 {
constexpr int BM = 128, KD = HC, KB = KD / 64;
constexpr int BLK = BM * 64, WB = D * 64;                  // k-block [2 heads][128 s][32 c] (64B swizzle), Wgv^T block [64 d][64 k]
constexpr int STAGE = BLK + WB;                            // 24 KiB
constexpr int NST = 4;                                     // the dv stream is latency-bound: ring depth is throughput
constexpr int RT = BM * D;                                 // y / x / dout / dm tiles [128][64]
constexpr int THREADS = 224;                               // warp 0: y + the ring, warp 1: MMA, warps 2-5: drain, warp 6: x / dout / dyg
constexpr int SMEM = 1024 + (NST * STAGE + 2 * RT + 3 * RT + RT + 2 * RT) * 2 + 512;   // y x2, x | dout | dyg x1, dm, P
constexpr int COL_DY = 0, COL_W = 128;                     // dy[2] 0/64, dWv^T 128..255 (two lane halves)
constexpr int COL_ONE = 256, COL_LN = 384;                 // a bf16 ones A operand [128][128] (64 columns), (dgamma | dbeta) x 128 rows
// dgamma / dbeta on the tensor core: P = [dy . xhat | dy] (bf16, [2 blocks][128 rows][64]) and D += 1^T P over the CTA's tiles
// (every row of D is the column sum); a per-tile 64-column warp reduce-scatter cost a third of the kernel
constexpr uint32_t ID_LN = idesc_bf16(128, 2 * D, 0, 1);   // A = ones (TMEM), B = P (MN-major)
constexpr uint32_t ID_DY = idesc_bf16(128, D, 0, 0);
constexpr uint32_t ID_W = idesc_bf16(64, 64, 1, 1);
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace dv2

template <typename WT_>
__global__ void __launch_bounds__(dv2::THREADS, 1) dv_bwd_sm100(
    int N, int S, int ntiles, float eps,
    const __grid_constant__ CUtensorMap vmap,     // dv head-major as (32 c, S, H*N), box (32, 128, 1), 64B swizzle
    const __grid_constant__ CUtensorMap wmap,     // [Wg; Wv]^T [64][512], box (64, 64): the Wv half (columns 256-511)
    const __grid_constant__ CUtensorMap ymap,     // y [S][N][64] as (64, N, S), box (64, 1, 128): a tile is one token x 128 s
    const __grid_constant__ CUtensorMap xmap,     // x (the msa input) [M][64]
    const __grid_constant__ CUtensorMap omap,     // dout (the residual gradient) [M][64]
    const __grid_constant__ CUtensorMap dmmap,    // dm [M][64]
    const __grid_constant__ CUtensorMap ygmap,    // dyg [M][64] (pwa_glue2)
    const WT_* __restrict__ LNW,
    float* __restrict__ DW,                       // [grid][64 d][256]  (dWv^T partials)
    float* __restrict__ DLN) {                    // [grid][2][64]      (dgamma, dbeta partials)
  using namespace dv2;
  const int NSB = S / BM;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sYb = sStage + NST * STAGE;                     // [2][y]: the GEMMs read it from the tile's first block
  __nv_bfloat16* sXO = sYb + 2 * RT;                             // [x | dout | dyg], single: the drain reads it only at the tile's end
  __nv_bfloat16* sDM = sXO + 3 * RT;
  __nv_bfloat16* sP = sDM + RT;                                  // [2 blocks][128 rows][64]: dy . xhat | dy
  uint64_t* bars = reinterpret_cast<uint64_t*>(sP + 2 * RT);
  uint64_t* full = bars;             // [NST]
  uint64_t* empty = full + NST;      // [NST]
  uint64_t* tf = empty + NST;        // [2] y of tile buffer landed
  uint64_t* te = tf + 2;             // [2] the dWgv GEMMs (commit) are done with it
  uint64_t* xf = te + 2;             // [1] x / dout landed
  uint64_t* xe = xf + 1;             // [1] count 4: the drain read them
  uint64_t* accf = xe + 1;           // [2]
  uint64_t* acce = accf + 2;         // [2] count 4
  uint64_t* wdone = acce + 2;        // [1]
  uint64_t* pf = wdone + 1;          // [1] count 4: P of the tile written
  uint64_t* pe = pf + 1;             // [1] the P GEMM retired
  uint32_t* tslot = reinterpret_cast<uint32_t*>(pe + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(full + k, 1); bar_init(empty + k, 1); }
    for (int k = 0; k < 2; ++k) { bar_init(tf + k, 1); bar_init(te + k, 1); bar_init(accf + k, 1); bar_init(acce + k, 4); }
    bar_init(xf, 1); bar_init(xe, 4); bar_init(wdone, 1); bar_init(pf, 4); bar_init(pe, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      int g = 0, lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int n = t / NSB, s0 = (t % NSB) * BM, b = lt & 1;
        if (lt >= 2) wait(te + b, ((lt >> 1) - 1) & 1);
        expect_tx(tf + b, RT * 2);
        load_3d(&ymap, sYb + b * RT, tf + b, 0, n, s0);
        for (int kb = 0; kb < KB; ++kb, ++g) {
          const int st = g % NST;
          if (g >= NST) wait(empty + st, ((g / NST) - 1) & 1);
          __nv_bfloat16* p = sStage + st * STAGE;
          expect_tx(full + st, STAGE * 2);
          for (int hh = 0; hh < 2; ++hh) load_3d(&vmap, p + hh * BM * 32, full + st, 0, s0, (2 * kb + hh) * N + n);
          load_2d(&wmap, p + BLK, full + st, HC + kb * 64, 0);
        }
      }
    }
  } else if (warp == 6) {
    if (lane == 0) {
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int n = t / NSB, s0 = (t % NSB) * BM;
        if (lt >= 1) wait(xe, (lt - 1) & 1);
        expect_tx(xf, 3 * RT * 2);
        load_3d(&xmap, sXO, xf, 0, n, s0);
        load_3d(&omap, sXO + RT, xf, 0, n, s0);
        load_3d(&ygmap, sXO + 2 * RT, xf, 0, n, s0);
      }
    }
  } else if (warp == 1) {
    int g = 0, lt = 0;
    auto p_gemm = [&](int ltp) {
      wait(pf, ltp & 1);
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < BM / 16; ++ks)
          mma_ts(tmem + COL_LN, tmem + COL_ONE, desc_mn128(sP + ks * 16 * 64, RT * 2), ID_LN, (ltp | ks) ? 1u : 0u);
        mma_commit(pe);
      }
      __syncwarp();
    };
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int b = lt & 1;
      wait(tf + b, (lt >> 1) & 1);
      if (lt >= 2) wait(acce + b, ((lt >> 1) - 1) & 1);
      tc_fence_after();
      const __nv_bfloat16* y = sYb + b * RT;
      for (int kb = 0; kb < KB; ++kb, ++g) {
        const int st = g % NST;
        wait(full + st, (g / NST) & 1);
        tc_fence_after();
        if (elect_one()) {
          const __nv_bfloat16* p = sStage + st * STAGE;
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)           // dy += dv_kb . Wv_kb  (A: the two heads' [128][32] 64B-swizzled halves)
            mma_ss(tmem + COL_DY + b * D, desc_k64(p + (ks >> 1) * BM * 32 + (ks & 1) * 16), desc_k128(p + BLK + ks * 16), ID_DY, (kb | ks) ? 1u : 0u);
          const uint32_t dw = tmem + ((uint32_t)(kb >> 1) << 20) + COL_W + (kb & 1) * 64;
#pragma unroll
          for (int ks = 0; ks < BM / 16; ++ks)     // dWv^T[:, kb block] += y^T . dv_kb over the tile's 128 rows
            mma_ss(dw, desc_mn128(y + ks * 16 * 64, 0), sdesc(sa(p + ks * 16 * 32), BM * 32 * 2, 512, 4), ID_W, (lt | ks) ? 1u : 0u);
          mma_commit(empty + st);
          if (kb == KB - 1) { mma_commit(accf + b); mma_commit(te + b); }
        }
        __syncwarp();
      }
      if (lt >= 1) p_gemm(lt - 1);               // the previous tile's P: its drain ran while this tile's GEMMs were queued
    }
    if (lt >= 1) p_gemm(lt - 1);
    if (elect_one()) mma_commit(wdone);
    __syncwarp();
  } else {
    const int q = warp & 3, r = q * 32 + lane;   // TMEM lane = row
    const bool leader = (warp == 2 && lane == 0);
    float2 gam[D / 2];                           // packed fp32 pairs throughout the drain (FADD2 / FFMA2 / FMUL2): it is power-bound
#pragma unroll
    for (int k = 0; k < D / 2; ++k) gam[k] = make_float2(to_f(LNW[2 * k]), to_f(LNW[2 * k + 1]));
    {                                            // the ones A operand of the P GEMM (bf16 1.0 pairs), this warp's 32 lanes
      const uint32_t one2[8] = {0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u};
      for (int c0 = 0; c0 < 64; c0 += 8) tmem_st8(tmem_at(tmem + COL_ONE, q * 32, c0), one2);
      tmem_wait_st();
      tc_fence_before();
    }
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int n = t / NSB, s0 = (t % NSB) * BM, b = lt & 1;
      wait(xf, lt & 1);
      wait(accf + b, (lt >> 1) & 1);
      tc_fence_after();
      float2 dy[D / 2];
      tmem_ld32(tmem_at(tmem + COL_DY + b * D, q * 32, 0), reinterpret_cast<float*>(dy));
      tmem_ld32(tmem_at(tmem + COL_DY + b * D, q * 32, 32), reinterpret_cast<float*>(dy) + 32);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(acce + b);
      const __nv_bfloat16* gr = sXO + 2 * RT + r * 64;   // + the gate branch's dyg (pwa_glue2)
      const __nv_bfloat16* xr = sXO + r * 64;
      const __nv_bfloat16* orow = sXO + RT + r * 64;
      float2 xh[D / 2];
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const int off = (c8 ^ (r & 7)) << 3;
        const uint4 u = *reinterpret_cast<const uint4*>(gr + off);
        dy[c8 * 4 + 0] = add2(dy[c8 * 4 + 0], bf2f(u.x)); dy[c8 * 4 + 1] = add2(dy[c8 * 4 + 1], bf2f(u.y));
        dy[c8 * 4 + 2] = add2(dy[c8 * 4 + 2], bf2f(u.z)); dy[c8 * 4 + 3] = add2(dy[c8 * 4 + 3], bf2f(u.w));
        const uint4 v = *reinterpret_cast<const uint4*>(xr + off);
        xh[c8 * 4 + 0] = bf2f(v.x); xh[c8 * 4 + 1] = bf2f(v.y); xh[c8 * 4 + 2] = bf2f(v.z); xh[c8 * 4 + 3] = bf2f(v.w);
      }
      float2 ra = make_float2(0.f, 0.f), rb = ra;
#pragma unroll
      for (int k = 0; k < D / 2; k += 2) { ra = add2(ra, xh[k]); rb = add2(rb, xh[k + 1]); }
      ra = add2(ra, rb);
      const float mean = (ra.x + ra.y) * (1.f / D);
      const float2 nm = make_float2(-mean, -mean);
      ra = make_float2(0.f, 0.f); rb = ra;
#pragma unroll
      for (int k = 0; k < D / 2; k += 2) {
        xh[k] = add2(xh[k], nm); xh[k + 1] = add2(xh[k + 1], nm);
        ra = fma2(xh[k], xh[k], ra); rb = fma2(xh[k + 1], xh[k + 1], rb);
      }
      ra = add2(ra, rb);
      const float rstd = rsqrtf((ra.x + ra.y) * (1.f / D) + eps);
      const float2 rs = make_float2(rstd, rstd);
      float2 gd[D / 2];
      float2 sa = make_float2(0.f, 0.f), sb = sa, ta = sa, tb = sa;
#pragma unroll
      for (int k = 0; k < D / 2; k += 2) {
        xh[k] = mul2(xh[k], rs); xh[k + 1] = mul2(xh[k + 1], rs);
        gd[k] = mul2(gam[k], dy[k]); gd[k + 1] = mul2(gam[k + 1], dy[k + 1]);
        sa = add2(sa, gd[k]); sb = add2(sb, gd[k + 1]);
        ta = fma2(gd[k], xh[k], ta); tb = fma2(gd[k + 1], xh[k + 1], tb);
      }
      sa = add2(sa, sb); ta = add2(ta, tb);
      const float s1 = (sa.x + sa.y) * (1.f / D), s2 = (ta.x + ta.y) * (1.f / D);
      if (leader) bulk_wait_read<0>();           // the previous tile's dm store has read the staging
      named_sync(1, 128);
      __nv_bfloat16* dmr = sDM + r * 64;
      {
        const float2 ns1 = make_float2(-s1, -s1), ns2 = make_float2(-s2, -s2);
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          const int off = (c8 ^ (r & 7)) << 3;
          const uint4 dv4 = *reinterpret_cast<const uint4*>(orow + off);
          const uint32_t* dw4 = reinterpret_cast<const uint32_t*>(&dv4);
          uint4 o;
          uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
          for (int k = 0; k < 4; ++k) {           // dm = rstd (gamma dy - s1 - xhat s2) + dout
            const int e = c8 * 4 + k;
            const float2 d2 = fma2(fma2(xh[e], ns2, add2(gd[e], ns1)), rs, bf2f(dw4[k]));
            ow[k] = pack2(d2.x, d2.y);
          }
          *reinterpret_cast<uint4*>(dmr + off) = o;
        }
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(xe);                 // x / dout / dyg are read
      // P = [dy . xhat | dy] for the P GEMM (the previous tile's must have retired)
      if (lt >= 1) wait(pe, (lt - 1) & 1);
#pragma unroll
      for (int blk = 0; blk < 2; ++blk)
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          uint4 o;
          uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int e = c8 * 4 + k;
            const float2 v2 = blk == 0 ? mul2(dy[e], xh[e]) : dy[e];
            ow[k] = pack2(v2.x, v2.y);
          }
          *reinterpret_cast<uint4*>(sP + blk * RT + r * 64 + ((c8 ^ (r & 7)) << 3)) = o;
        }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(pf);
      named_sync(1, 128);
      if (leader) { store_3d(&dmmap, sDM, 0, n, s0); bulk_commit(); }
    }
    if (leader) bulk_wait<0>();
    // this CTA's (dgamma, dbeta): every row of the P accumulator holds them; warp q = 0 writes them (lane l: columns l + 32 k)
    wait(wdone, 0);
    tc_fence_after();
    if (q == 0) {
      float* ln = DLN + (size_t)blockIdx.x * 2 * D;
#pragma unroll 1
      for (int c0 = 0; c0 < 2 * D; c0 += 32) {
        float v[32];
        tmem_ld32(tmem_at(tmem + COL_LN, 0, c0), v);
        tmem_wait_ld();
        float mine = v[0];
#pragma unroll
        for (int k = 1; k < 32; ++k) mine = lane == k ? v[k] : mine;
        ln[c0 + lane] = mine;
      }
    }
    // dWv^T slab: lane half 0 = column blocks 0-1, half 1 = 2-3; row d = q * 16 + (lane & 15)
    const int half = lane >> 4, d = q * 16 + (lane & 15);
    float* slab = DW + (size_t)blockIdx.x * D * KD + (size_t)d * KD + half * 128;
#pragma unroll 1
    for (int c0 = 0; c0 < 128; c0 += 32) {
      float v[32];
      tmem_ld32(tmem_at(tmem + COL_W, q * 32, c0), v);
      tmem_wait_ld();
#pragma unroll
      for (int k = 0; k < 32; k += 4) *reinterpret_cast<float4*>(slab + c0 + k) = make_float4(v[k], v[k + 1], v[k + 2], v[k + 3]);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---------------------------------------------------------------------------------------------
// PWA pair-side backward (the H100 path's Triton pair_bwd, same math), per pair (i, j):
//   zn = bf16(LN(z[i,j,:])), db[h] = w[h,i,j] (dw[h,i,j] - sum_j' w dw) -> bf16, dzn = db . Wb (fp32),
//   dz = LN_bwd(dzn),  dWb += db^T zn,  dgamma_z += dzn . xhat,  dbeta_z += dzn.
// The parameter gradients come from two per-head sums instead of per-channel column sums of dzn (a 64-value warp
// reduce-scatter per tile): M[h,d] = sum db[h] xhat[d] (tensor core, xhat^T as A) and cs[h] = sum db[h] (registers), then
// dWb = gamma o M + beta (x) cs, dgamma = sum_h Wb o M, dbeta = Wb^T cs (pair_param_finish).
// Tile = one pair row i x 128 j; z, the tile's w / dw slices and the row sums sum_j' w dw (a tiny pre-pass) land by TMA.
// Both products run on the tensor core: dzn = db . Wb (M = 128 j, N = 128 d, K = 16 (8 heads, padded)) into TMEM, and
// dWb^T += zn^T . db (M = 128 d, N = 16, K = 128 j) accumulated in TMEM over the CTA's tiles; db^T is the A operand of
// the first and the B operand of the second.  Thread = (row j = TMEM lane, channel half = warp / 4): the two halves of a
// row exchange their LayerNorm row sums through shared memory.  dgamma / dbeta are fp32 column sums (reduce-scatter).
namespace pb2 {
constexpr int DZ = 128, BJ = 128, HP = 16;
constexpr int ZT = BJ * DZ;                                // z / zn / dz tile [2 d blocks][128 j][64], 128B swizzle (32 KiB)
constexpr int DBT = HP * BJ;                               // db^T [2 j blocks][16 h][64 j], 128B swizzle (4 KiB)
constexpr int WBT = HP * DZ;                               // Wb [2 d blocks][16 h][64 d], rows 8-15 zero (4 KiB)
constexpr int WT = H * BJ;
constexpr int WDB = 7 * 1024;                              // w slice (2 KiB) | dw slice (4 KiB) | row sums (32 B), padded for TMA alignment
constexpr int THREADS = 320;                               // warps 0-7: compute, warp 8: producer, warp 9: MMA
constexpr int SMEM = 1024 + (2 * ZT + 2 * ZT + 2 * DBT + ZT + WBT) * 2 + 2 * WDB + 2 * DZ * 4 + 2 * 2 * 4 * BJ * 4 + 256;
constexpr uint32_t ID_DZN = idesc_bf16(128, DZ, 1, 1);     // A = db (db^T read MN-major), B = Wb (MN-major)
constexpr uint32_t ID_DWB = idesc_bf16(128, HP, 1, 0);     // A = zn^T (MN-major), B = db^T (K-major)
constexpr int COL_DZN = 0, COL_DWB = 256;                  // dzn[2] 0 / 128, dWb^T 256..271
constexpr int PSTRIDE = H * DZ + 4 * H;                    // a CTA's partials: M [8][128] | 4 warps x cs [8]
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace pb2

// sum_j w[h,i,j] dw[h,i,j] for every (i, h), written [N][H]: one warp per (h, i) row
template <typename DT_>
__global__ void __launch_bounds__(256) pair_sdot_sm100(const __nv_bfloat16* __restrict__ W16, const DT_* __restrict__ DW, float* __restrict__ SD, int N) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;   // row = h * N + i
  if (row >= H * N) return;
  const __nv_bfloat16* w = W16 + (size_t)row * N;
  const DT_* d = DW + (size_t)row * N;
  float sd = 0.f;
  for (int j = lane * 4; j < N; j += 128) {
    float4 dv;
    if constexpr (sizeof(DT_) == 4) dv = __ldg(reinterpret_cast<const float4*>(d + j));
    else { const uint2 u = __ldg(reinterpret_cast<const uint2*>(d + j)); const float2 a = bf2f(u.x), b = bf2f(u.y); dv = make_float4(a.x, a.y, b.x, b.y); }
    const uint2 wv = __ldg(reinterpret_cast<const uint2*>(w + j));
    const float2 w0 = bf2f(wv.x), w1 = bf2f(wv.y);
    sd += w0.x * dv.x + w0.y * dv.y + w1.x * dv.z + w1.y * dv.w;
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) sd += __shfl_xor_sync(0xffffffffu, sd, off);
  if (lane == 0) SD[(size_t)(row % N) * H + row / N] = sd;
}

// dWb[h,d] = gamma[d] M[h,d] + beta[d] cs[h],  dgamma[d] = sum_h Wb[h,d] M[h,d],  dbeta[d] = sum_h Wb[h,d] cs[h]  (one CTA, thread = d)
template <typename WT_, typename LT_, typename OT_, typename LO_>
__global__ void __launch_bounds__(128) pair_param_finish(const float* __restrict__ RED, const LT_* __restrict__ LNW, const LT_* __restrict__ LNB,
                                                         const WT_* __restrict__ WB, OT_* __restrict__ DWB, LO_* __restrict__ DG, LO_* __restrict__ DBE) {
  using namespace pb2;
  const int d = threadIdx.x;
  __shared__ float scs[H];
  if (d < H) scs[d] = RED[H * DZ + d] + RED[H * DZ + H + d] + RED[H * DZ + 2 * H + d] + RED[H * DZ + 3 * H + d];
  __syncthreads();
  const float g = to_f(LNW[d]), b = to_f(LNB[d]);
  float dg = 0.f, dbe = 0.f;
#pragma unroll
  for (int h = 0; h < H; ++h) {
    const float m = RED[h * DZ + d], wb = to_f(WB[h * DZ + d]);
    DWB[h * DZ + d] = from_f<OT_>(fmaf(g, m, b * scs[h]));
    dg = fmaf(wb, m, dg);
    dbe = fmaf(wb, scs[h], dbe);
  }
  DG[d] = from_f<LO_>(dg); DBE[d] = from_f<LO_>(dbe);
}

template <typename WT_, typename LT_, typename DT_>
__global__ void __launch_bounds__(pb2::THREADS, 1) pair_bwd_sm100(
    int N, int ntiles, float eps,
    const __grid_constant__ CUtensorMap zmap,     // z [N*N][128], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap dzmap,    // dz, same
    const __grid_constant__ CUtensorMap wmap,     // w16 [H][N][N] as (N j, N i, H), box (128, 1, 8)
    const __grid_constant__ CUtensorMap dwmap,    // dw (fp32 or bf16: DT_), same
    const __grid_constant__ CUtensorMap sdmap,    // the row sums [N][H] fp32, box (8, 1)
    const LT_* __restrict__ LNW, const LT_* __restrict__ LNB,
    const WT_* __restrict__ WB,                   // proj_z weight [H][128]
    float* __restrict__ PART) {                   // [grid][H * 128 (M) + 4 warps * 8 (cs)]: one column sum
  using namespace pb2;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sZ = reinterpret_cast<__nv_bfloat16*>(smb);   // [2][ZT]
  __nv_bfloat16* sZN = sZ + 2 * ZT;                              // [2][ZT] bf16 xhat, the M GEMM's A operand
  __nv_bfloat16* sDB = sZN + 2 * ZT;                             // [2][DBT]
  __nv_bfloat16* sOut = sDB + 2 * DBT;                           // dz staging [ZT]
  __nv_bfloat16* sWBt = sOut + ZT;                               // Wb [2][16][64]
  unsigned char* sWD = reinterpret_cast<unsigned char*>(sWBt + WBT);   // [2][WDB]
  float* sLN = reinterpret_cast<float*>(sWD + 2 * WDB);          // gamma[128], beta[128]
  float* sX = sLN + 2 * DZ;                                      // [2 exchanges][2 halves][4 values][128 rows] row-sum exchange
  uint64_t* bars = reinterpret_cast<uint64_t*>(sX + 2 * 2 * 4 * BJ);
  uint64_t* zf = bars;               // [2]
  uint64_t* ze = zf + 2;             // [2] count 8: z / w / dw read
  uint64_t* dbf = ze + 2;            // [2] count 4: db^T written (half-0 warps)
  uint64_t* znf = dbf + 2;           // [2] count 8: zn written
  uint64_t* dznf = znf + 2;          // [2] the dzn GEMM done
  uint64_t* dzne = dznf + 2;         // [2] count 8: dzn drained
  uint64_t* ae = dzne + 2;           // [2] the dWb GEMM retired (zn / db^T free)
  uint64_t* done = ae + 2;           // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(done + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < 2; ++k) { bar_init(zf + k, 1); bar_init(ze + k, 8); bar_init(dbf + k, 4); bar_init(znf + k, 8);
                                  bar_init(dznf + k, 1); bar_init(dzne + k, 8); bar_init(ae + k, 1); }
    bar_init(done, 1);
    bar_init_fence();
  }
  for (int v = tid; v < 2 * DBT / 8; v += THREADS) reinterpret_cast<uint4*>(sDB)[v] = make_uint4(0u, 0u, 0u, 0u);   // head rows 8-15 stay zero
  for (int v = tid; v < HP * DZ; v += THREADS) {             // Wb as the MN-major B operand: [d block][h][64 d]
    const int h = v / DZ, d = v % DZ;
    sWBt[(d >> 6) * HP * 64 + sw128(h, d & 63)] = h < H ? __float2bfloat16_rn(to_f(WB[h * DZ + d])) : __float2bfloat16_rn(0.f);
  }
  for (int v = tid; v < DZ; v += THREADS) { sLN[v] = to_f(LNW[v]); sLN[DZ + v] = to_f(LNB[v]); }
  if (warp == 9) tmem_alloc(tslot, 512);
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int NJB = N / BJ;

  if (warp == 8) {
    if (lane == 0) {
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int i = t / NJB, j0 = (t % NJB) * BJ, b = lt & 1;
        if (lt >= 2) wait(ze + b, ((lt >> 1) - 1) & 1);
        expect_tx(zf + b, ZT * 2 + WT * 2 + WT * (int)sizeof(DT_) + H * 4);
        for (int h = 0; h < 2; ++h) load_2d(&zmap, sZ + b * ZT + h * BJ * 64, zf + b, h * 64, i * N + j0);
        unsigned char* wd = sWD + b * WDB;
        load_3d(&wmap, wd, zf + b, j0, i, 0);
        load_3d(&dwmap, wd + WT * 2, zf + b, j0, i, 0);
        load_2d(&sdmap, wd + WT * 6, zf + b, 0, i);
      }
    }
  } else if (warp == 9) {
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int b = lt & 1;
      if (lt >= 2) wait(ae + b, ((lt >> 1) - 1) & 1);   // (db^T buffer b is rewritten only after this; the waits below imply it)
      wait(dbf + b, (lt >> 1) & 1);
      if (lt >= 2) wait(dzne + b, ((lt >> 1) - 1) & 1);
      tc_fence_after();
      if (elect_one()) {                                 // dzn = db . Wb: one K = 16 step
        mma_ss(tmem + COL_DZN + b * DZ, desc_mn128(sDB + b * DBT, HP * 64 * 2), desc_mn128(sWBt, HP * 64 * 2), ID_DZN, 0u);
        mma_commit(dznf + b);
      }
      __syncwarp();
      wait(znf + b, (lt >> 1) & 1);
      tc_fence_after();
      if (elect_one()) {                                 // dWb^T += zn^T . db over the tile's 128 j
        const __nv_bfloat16* zn = sZN + b * ZT;
        const __nv_bfloat16* db = sDB + b * DBT;
#pragma unroll
        for (int ks = 0; ks < BJ / 16; ++ks)
          mma_ss(tmem + COL_DWB, desc_mn128(zn + ks * 16 * 64, BJ * 64 * 2), desc_k128(db + (ks >> 2) * HP * 64 + (ks & 3) * 16), ID_DWB, (lt | ks) ? 1u : 0u);
        mma_commit(ae + b);
      }
      __syncwarp();
    }
    if (elect_one()) mma_commit(done);
    __syncwarp();
  } else {
    const int q = warp & 3, hf = warp >> 2, j = q * 32 + lane;   // row j = TMEM lane; channel half hf
    float cs[H];                                                 // this row's running sum of db per head (half-0 warps)
#pragma unroll
    for (int h = 0; h < H; ++h) cs[h] = 0.f;
    int lt = 0;
    auto xchg = [&](int slot, float a, float bb, float& ta, float& tb) {   // the two halves' row sums, via shared memory
      float* x = sX + ((slot * 2 + hf) * 4) * BJ;
      x[j] = a; x[BJ + j] = bb;
      named_sync(1, 256);
      const float* y = sX + ((slot * 2 + (hf ^ 1)) * 4) * BJ;
      ta = a + y[j]; tb = bb + y[BJ + j];
    };
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i = t / NJB, j0 = (t % NJB) * BJ, b = lt & 1;
      wait(zf + b, (lt >> 1) & 1);
      // ---- db (half-0 warps) -> db^T: the dzn GEMM's A operand and the dWb GEMM's B operand ----
      if (hf == 0) {
        const unsigned char* wd = sWD + b * WDB;
        const __nv_bfloat16* ws = reinterpret_cast<const __nv_bfloat16*>(wd);
        const DT_* dws = reinterpret_cast<const DT_*>(wd + WT * 2);
        const float* sdot = reinterpret_cast<const float*>(wd + WT * 6);
        if (lt >= 2) wait(ae + b, ((lt >> 1) - 1) & 1);
        __nv_bfloat16* dbt = sDB + b * DBT + (j >> 6) * HP * 64;
#pragma unroll
        for (int h = 0; h < H; ++h) {            // bf16: the stock proj_z backward is a bf16 GEMM
          const __nv_bfloat16 d16 = __float2bfloat16_rn(__bfloat162float(ws[h * BJ + j]) * (to_f(dws[h * BJ + j]) - sdot[h]));
          dbt[sw128(h, j & 63)] = d16;
          cs[h] += __bfloat162float(d16);
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) arrive(dbf + b);
      }
      // ---- LayerNorm recompute on this thread's 64 channels: packed fp32 pairs (FADD2 / FFMA2 / FMUL2) throughout -- the
      // kernel was issue-bound on scalar LN arithmetic; statistics in one pass shifted by the row's first element (one exchange)
      const __nv_bfloat16* zr = sZ + b * ZT + hf * BJ * 64 + j * 64;
      const float k0 = __bfloat162float(sZ[b * ZT + j * 64 + ((j & 7) << 3)]);   // z[i, j, 0]: the same shift for both halves
      const float2 nk = make_float2(-k0, -k0);
      float2 xh[32];
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const uint4 u = *reinterpret_cast<const uint4*>(zr + ((c8 ^ (j & 7)) << 3));
        xh[c8 * 4 + 0] = add2(bf2f(u.x), nk); xh[c8 * 4 + 1] = add2(bf2f(u.y), nk);
        xh[c8 * 4 + 2] = add2(bf2f(u.z), nk); xh[c8 * 4 + 3] = add2(bf2f(u.w), nk);
      }
      fence_proxy_async();                       // the loads may be in flight: order them before the next TMA write into the stage
      __syncwarp();
      if (lane == 0) arrive(ze + b);
      float2 s2a = make_float2(0.f, 0.f), s2b = s2a, q2a = s2a, q2b = s2a;
#pragma unroll
      for (int k = 0; k < 32; k += 2) {
        s2a = add2(s2a, xh[k]); s2b = add2(s2b, xh[k + 1]);
        q2a = fma2(xh[k], xh[k], q2a); q2b = fma2(xh[k + 1], xh[k + 1], q2b);
      }
      s2a = add2(s2a, s2b); q2a = add2(q2a, q2b);
      float tsum, tsq;
      xchg(0, s2a.x + s2a.y, q2a.x + q2a.y, tsum, tsq);
      const float m1 = tsum * (1.f / DZ);                  // mean - k0
      const float rstd = 1.f / sqrtf(fmaxf(tsq * (1.f / DZ) - m1 * m1, 0.f) + eps);
      {
        const float2 nm = make_float2(-m1, -m1), rs = make_float2(rstd, rstd);
#pragma unroll
        for (int k = 0; k < 32; ++k) xh[k] = mul2(add2(xh[k], nm), rs);
      }
      // ---- bf16 xhat into the M GEMM's A operand ----
      if (lt >= 2) wait(ae + b, ((lt >> 1) - 1) & 1);
      __nv_bfloat16* znr = sZN + b * ZT + hf * BJ * 64 + j * 64;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        uint4 o;
        o.x = pack2(xh[c8 * 4 + 0].x, xh[c8 * 4 + 0].y); o.y = pack2(xh[c8 * 4 + 1].x, xh[c8 * 4 + 1].y);
        o.z = pack2(xh[c8 * 4 + 2].x, xh[c8 * 4 + 2].y); o.w = pack2(xh[c8 * 4 + 3].x, xh[c8 * 4 + 3].y);
        *reinterpret_cast<uint4*>(znr + ((c8 ^ (j & 7)) << 3)) = o;
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(znf + b);
      // ---- dzn from TMEM, the LayerNorm backward -> dz ----
      wait(dznf + b, (lt >> 1) & 1);
      tc_fence_after();
      float2 g[32];                                        // dzn, then gamma o dzn
      tmem_ld32(tmem_at(tmem + COL_DZN + b * DZ, q * 32, hf * 64), reinterpret_cast<float*>(g));
      tmem_ld32(tmem_at(tmem + COL_DZN + b * DZ, q * 32, hf * 64 + 32), reinterpret_cast<float*>(g) + 32);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(dzne + b);
      const float2* gam2 = reinterpret_cast<const float2*>(sLN + hf * 64);
      float2 gsa = make_float2(0.f, 0.f), gsb = gsa, gxa = gsa, gxb = gsa;
#pragma unroll
      for (int k = 0; k < 32; k += 2) {
        g[k] = mul2(g[k], gam2[k]); g[k + 1] = mul2(g[k + 1], gam2[k + 1]);
        gsa = add2(gsa, g[k]); gsb = add2(gsb, g[k + 1]);
        gxa = fma2(g[k], xh[k], gxa); gxb = fma2(g[k + 1], xh[k + 1], gxb);
      }
      gsa = add2(gsa, gsb); gxa = add2(gxa, gxb);
      float gs, gx;
      xchg(1, gsa.x + gsa.y, gxa.x + gxa.y, gs, gx);
      gs *= (1.f / DZ); gx *= (1.f / DZ);
      if (warp == 0 && lane == 0) bulk_wait_read<0>();   // the previous tile's dz store has read the staging
      named_sync(1, 256);
      __nv_bfloat16* orow = sOut + hf * BJ * 64 + j * 64;
      {
        const float2 ngs = make_float2(-gs, -gs), ngx = make_float2(-gx, -gx), rs = make_float2(rstd, rstd);
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          uint4 o;
          uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
          for (int k = 0; k < 4; ++k) {                  // dz = rstd (g - gs - xhat gx)
            const float2 d2 = mul2(fma2(xh[c8 * 4 + k], ngx, add2(g[c8 * 4 + k], ngs)), rs);
            ow[k] = pack2(d2.x, d2.y);
          }
          *reinterpret_cast<uint4*>(orow + ((c8 ^ (j & 7)) << 3)) = o;
        }
      }
      fence_proxy_async();
      named_sync(1, 256);
      if (warp == 0 && lane == 0) { for (int h = 0; h < 2; ++h) store_2d(&dzmap, sOut + h * BJ * 64, h * 64, i * N + j0); bulk_commit(); }
    }
    if (warp == 0 && lane == 0) bulk_wait<0>();
    // this warp's cs partial (half-0 warps: one per row group)
    if (hf == 0) {
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) cs[h] += __shfl_xor_sync(0xffffffffu, cs[h], off);
      if (lane < H) {
        float v = cs[0];
#pragma unroll
        for (int h = 1; h < H; ++h) v = lane == h ? cs[h] : v;
        PART[(size_t)blockIdx.x * PSTRIDE + H * DZ + q * H + lane] = v;
      }
    }
    // this CTA's M partial: TMEM lane = d, columns = heads
    if (hf == 0) {
      wait(done, 0);
      tc_fence_after();
      float v[8];
      tmem_ld8(tmem_at(tmem + COL_DWB, q * 32, 0), v);
      tmem_wait_ld();
#pragma unroll
      for (int h = 0; h < H; ++h) PART[(size_t)blockIdx.x * PSTRIDE + h * DZ + q * 32 + lane] = v[h];
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 9) tmem_dealloc(tmem, 512);
}

// ---------------------------------------------------------------------------------------------
// PWA pair-side forward: w[h,i,:] = softmax_j(mask[j] ? bf16(LN(z[i,j,:]) . Wb_h) : -1e30)  (the H100 pair_fwd3's math).
// The LayerNorm folds into the projection: LN(z) . Wb_h = rstd (z . Wg_h - mean s_h) + c_h with Wg = bf16(Wb o gamma),
// s_h = sum_d Wg[h,d], c_h = Wb_h . beta -- the tensor core reads the z tile exactly as TMA lands it (no normalised copy) and a
// thread only needs its row's mean / rstd.  CTA = persistent over rows i; tile = row i x 128 j.
// (Writing bf16(LN(z)) over the tile first, as the module rounds, measured the same error against fp32 but 1.5x slower: the
// per-element normalisation made the kernel fp32-issue-bound.)
//   warp 12: z tiles by TMA (ring);  warp 13: acc[j][16] = z . Wg^T (M = 128 j, N = 16: 8 heads padded, K = 128);
//   warps 0-7: row statistics (thread = row j = TMEM lane), the logits -> a bf16 row buffer [8][N] in shared memory; two groups
//   of four take alternate tiles (the per-row sums are latency-bound at one warp per SMSP), each with its own TMEM accumulator;
//   warps 8-11: softmax over j of the previous row, straight to global (two row buffers: the rows overlap).
namespace pf2 {
constexpr int DZ = 128, BJ = 128, HP = 16, NMAX = 1024;
constexpr int ZT = BJ * DZ;                                // z tile [2 d blocks][128 j][64], 128B swizzle (32 KiB)
constexpr int NST = 5;
constexpr int WGT = HP * DZ;                               // Wg [2 d blocks][16 h][64], rows 8-15 zero (4 KiB)
constexpr int LB = H * NMAX;                               // one row's logits [8][NMAX] bf16 (16 KiB)
constexpr int THREADS = 448;
constexpr int SMEM = 1024 + (NST * ZT + WGT + 2 * LB) * 2 + 2 * H * 4 + 256;
constexpr uint32_t IDESC = idesc_bf16(128, HP, 0, 0);
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace pf2

template <typename WT_, typename LT_>
__global__ void __launch_bounds__(pf2::THREADS, 1) pair_fwd_sm100(
    int N, float eps,
    const __grid_constant__ CUtensorMap zmap,     // z [N*N][128], box (64, 128), 128B swizzle
    const uint8_t* __restrict__ MASK,             // key mask [N] (bool)
    const LT_* __restrict__ LNW, const LT_* __restrict__ LNB,
    const WT_* __restrict__ WB,                   // proj_z weight [H][128]
    __nv_bfloat16* __restrict__ W) {              // softmax [H][N][N]
  using namespace pf2;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sZ = reinterpret_cast<__nv_bfloat16*>(smb);   // [NST][ZT]
  __nv_bfloat16* sWg = sZ + NST * ZT;
  __nv_bfloat16* sL = sWg + WGT;                                 // [2][H][NMAX]
  float* sSC = reinterpret_cast<float*>(sL + 2 * LB);            // s[8], c[8]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sSC + 2 * H);
  uint64_t* full = bars;             // [NST]
  uint64_t* empty = full + NST;      // [NST] count 5: the MMA + the 4 statistics warps
  uint64_t* accf = empty + NST;      // [2]
  uint64_t* acce = accf + 2;         // [2] count 4
  uint64_t* rowf = acce + 2;         // [2] count 8: a row's logits written (both groups)
  uint64_t* rowe = rowf + 2;         // [2] count 4: its softmax stored
  uint32_t* tslot = reinterpret_cast<uint32_t*>(rowe + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(full + k, 1); bar_init(empty + k, 5); }
    for (int k = 0; k < 2; ++k) { bar_init(accf + k, 1); bar_init(acce + k, 4); bar_init(rowf + k, 8); bar_init(rowe + k, 4); }
    bar_init_fence();
  }
  for (int v = tid; v < HP * DZ; v += THREADS) {             // Wg = bf16(Wb o gamma) as the K-major B operand [d block][h][64 d]
    const int h = v / DZ, d = v % DZ;
    sWg[(d >> 6) * HP * 64 + sw128(h, d & 63)] = __float2bfloat16_rn(h < H ? to_f(WB[h * DZ + d]) * to_f(LNW[d]) : 0.f);
  }
  if (warp < H) {                                            // s_h over the rounded Wg (the MMA's own weights), c_h = Wb_h . beta
    float s = 0.f, c = 0.f;
    for (int d = lane; d < DZ; d += 32) {
      const float wb = to_f(WB[warp * DZ + d]);
      s += __bfloat162float(__float2bfloat16_rn(wb * to_f(LNW[d])));
      c += wb * to_f(LNB[d]);
    }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, off); c += __shfl_xor_sync(0xffffffffu, c, off); }
    if (lane == 0) { sSC[warp] = s; sSC[H + warp] = c; }
  }
  if (warp == 13) tmem_alloc(tslot, 32);
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int NJB = N / BJ;

  if (warp == 12) {
    if (lane == 0) {
      int g = 0;
      for (int i = blockIdx.x; i < N; i += gridDim.x)
        for (int jb = 0; jb < NJB; ++jb, ++g) {
          const int st = g % NST;
          if (g >= NST) wait(empty + st, ((g / NST) - 1) & 1);
          expect_tx(full + st, ZT * 2);
          for (int b = 0; b < 2; ++b) load_2d(&zmap, sZ + st * ZT + b * BJ * 64, full + st, b * 64, i * N + jb * BJ);
        }
    }
  } else if (warp == 13) {
    int g = 0;
    for (int i = blockIdx.x; i < N; i += gridDim.x)
      for (int jb = 0; jb < NJB; ++jb, ++g) {
        const int st = g % NST, b = g & 1;
        if (g >= 2) wait(acce + b, ((g >> 1) - 1) & 1);
        wait(full + st, (g / NST) & 1);
        tc_fence_after();
        if (elect_one()) {
          const __nv_bfloat16* zt = sZ + st * ZT;
#pragma unroll
          for (int ks = 0; ks < DZ / 16; ++ks)
            mma_ss(tmem + b * HP, desc_k128(zt + (ks >> 2) * BJ * 64 + (ks & 3) * 16), desc_k128(sWg + (ks >> 2) * HP * 64 + (ks & 3) * 16),
                   IDESC, ks ? 1u : 0u);
          mma_commit(empty + st);
          mma_commit(accf + b);
        }
        __syncwarp();
      }
  } else if (warp < 8) {
    const int grp = warp >> 2, j = (warp & 3) * 32 + lane;   // tile row = TMEM lane; group grp takes the tiles g = grp (mod 2)
    const __nv_bfloat16 masked = __float2bfloat16_rn(-1e30f);
    int g = 0, r = 0;
    for (int i = blockIdx.x; i < N; i += gridDim.x, ++r) {
      const int lb = r & 1;
      if (r >= 2) wait(rowe + lb, ((r >> 1) - 1) & 1);
      __nv_bfloat16* L = sL + lb * LB;
      for (int jb = 0; jb < NJB; ++jb, ++g) {
        const int st = g % NST, b = g & 1;
        if (b != grp) continue;
        wait(full + st, (g / NST) & 1);
        // one pass over the row in shared memory, shifted by its first element (no cancellation for rows far from zero mean;
        // holding the 128 values in registers spilled)
        const __nv_bfloat16* zr = sZ + st * ZT + j * 64;
        const float k0 = __bfloat162float(zr[0]);
        const float2 nk = make_float2(-k0, -k0);
        float2 a0 = make_float2(0.f, 0.f), a1 = a0, q0 = a0, q1 = a0;
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const uint4 v = *reinterpret_cast<const uint4*>(zr + (q >> 3) * BJ * 64 + (((q & 7) ^ (j & 7)) << 3));
          const float2 x0 = add2(bf2f(v.x), nk), x1 = add2(bf2f(v.y), nk), x2 = add2(bf2f(v.z), nk), x3 = add2(bf2f(v.w), nk);
          a0 = add2(a0, add2(x0, x1)); a1 = add2(a1, add2(x2, x3));
          q0 = fma2(x0, x0, fma2(x1, x1, q0)); q1 = fma2(x2, x2, fma2(x3, x3, q1));
        }
        fence_proxy_async();                     // order the stage's reads before the next TMA write into it (another proxy)
        __syncwarp();
        if (lane == 0) arrive(empty + st);
        a0 = add2(a0, a1); q0 = add2(q0, q1);
        const float m1 = (a0.x + a0.y) * (1.f / DZ);                    // mean - k0
        const float mean = k0 + m1;
        const float var = fmaxf((q0.x + q0.y) * (1.f / DZ) - m1 * m1, 0.f);
        const float rstd = 1.f / sqrtf(var + eps);
        wait(accf + b, (g >> 1) & 1);
        tc_fence_after();
        float a[16];
        tmem_ld16(tmem_at(tmem + b * HP, (warp & 3) * 32, 0), a);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(acce + b);
        const bool keep = MASK[jb * BJ + j] != 0;
#pragma unroll
        for (int h = 0; h < H; ++h)
          L[h * NMAX + jb * BJ + j] = keep ? __float2bfloat16_rn(fmaf(rstd, a[h] - mean * sSC[h], sSC[H + h])) : masked;
      }
      __syncwarp();
      if (lane == 0) arrive(rowf + lb);
    }
  } else if (warp < 12) {
    constexpr int KMAX = NMAX / 64;
    const int h0 = (warp - 8) * 2;
    int r = 0;
    for (int i = blockIdx.x; i < N; i += gridDim.x, ++r) {
      const int lb = r & 1;
      wait(rowf + lb, (r >> 1) & 1);
#pragma unroll 1
      for (int hh = 0; hh < 2; ++hh) {
        const int h = h0 + hh;
        const __nv_bfloat16* L = sL + lb * LB + h * NMAX;
        float2 e[KMAX];
        float mx = -INFINITY;
#pragma unroll
        for (int k = 0; k < KMAX; ++k) {
          const int jj = lane * 2 + k * 64;
          e[k] = jj < N ? bf2f(*reinterpret_cast<const uint32_t*>(L + jj)) : make_float2(-INFINITY, -INFINITY);
          mx = fmaxf(mx, fmaxf(e[k].x, e[k].y));
        }
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        float den = 0.f;
#pragma unroll
        for (int k = 0; k < KMAX; ++k) {             // (x - max) first: a fully masked row (all -1e30) is exactly uniform
          e[k].x = exp2f((e[k].x - mx) * 1.4426950408889634f); e[k].y = exp2f((e[k].y - mx) * 1.4426950408889634f);
          den += e[k].x + e[k].y;
        }
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) den += __shfl_xor_sync(0xffffffffu, den, off);
        const float inv = 1.f / den;
        uint32_t* out = reinterpret_cast<uint32_t*>(W + ((size_t)h * N + i) * N);
#pragma unroll
        for (int k = 0; k < KMAX; ++k) {
          const int jj = lane * 2 + k * 64;
          if (jj < N) out[jj >> 1] = pack2(e[k].x * inv, e[k].y * inv);
        }
      }
      __syncwarp();
      if (lane == 0) arrive(rowe + lb);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 13) tmem_dealloc(tmem, 32);
}

// sum_k P[k * W + c] over the splits for this block's 32 columns c = col0 + lane: 8 split groups (one warp each, two chains) then a
// fixed-order combine in shared memory (deterministic).  The total is valid in warp 0.  One thread per column with every split
// in series was latency-bound (10-16 us); 256 threads, one __syncthreads.
__device__ __forceinline__ float split_sum8(const float* __restrict__ P, int splits, long W, long c, float* sred) {
  const int g = threadIdx.x >> 5, l = threadIdx.x & 31;
  float a = 0.f, b = 0.f;
  int k = g;
  for (; k + 8 < splits; k += 16) { a += __ldg(P + (size_t)k * W + c); b += __ldg(P + (size_t)(k + 8) * W + c); }
  if (k < splits) a += __ldg(P + (size_t)k * W + c);
  sred[g * 32 + l] = a + b;
  __syncthreads();
  float t = 0.f;
  if (g == 0) {
#pragma unroll
    for (int q = 0; q < 8; ++q) t += sred[q * 32 + l];
  }
  return t;
}

// the glue2 slabs [splits][64][512] (dWo | dWg^T) -> dWo [64][256], dWg [256][64] in the output dtype, summed here (no colsum launch)
template <typename OT_>
__global__ void __launch_bounds__(256) glue2_finish(const float* __restrict__ PART, int splits, OT_* __restrict__ DWO, OT_* __restrict__ DWG) {
  __shared__ float sred[256];
  const int t = blockIdx.x * 32 + (threadIdx.x & 31);    // < 64 * 512
  const float v = split_sum8(PART, splits, D * 2 * HC, t, sred);
  if (threadIdx.x >= 32) return;
  const int d = t / (2 * HC), k = t % (2 * HC);
  if (k < HC) DWO[d * HC + k] = from_f<OT_>(v); else DWG[(k - HC) * D + d] = from_f<OT_>(v);
}

// the dv_bwd slabs [splits][64][256] (dWv^T) -> dWv [256][64], and [splits][2][64] -> dgamma, dbeta, summed here (no colsum launches):
// blocks 0 .. 511 take dWv, 512 .. 515 the LayerNorm sums
template <typename OT_, typename LO_>
__global__ void __launch_bounds__(256) dv_finish(const float* __restrict__ DW, const float* __restrict__ DLNP, int splits, OT_* __restrict__ DWV,
                                                 LO_* __restrict__ DLW, LO_* __restrict__ DLB) {
  __shared__ float sred[256];
  constexpr int NB = D * HC / 32;
  if ((int)blockIdx.x < NB) {
    const int t = blockIdx.x * 32 + (threadIdx.x & 31), d = t / HC, k = t % HC;   // slab element [d][k]
    const float v = split_sum8(DW, splits, D * HC, t, sred);
    if (threadIdx.x < 32) DWV[k * D + d] = from_f<OT_>(v);
  } else {
    const int t = (blockIdx.x - NB) * 32 + (threadIdx.x & 31);                      // < 2 * 64
    const float v = split_sum8(DLNP, splits, 2 * D, t, sred);
    if (threadIdx.x < 32) { if (t < D) DLW[t] = from_f<LO_>(v); else DLB[t - D] = from_f<LO_>(v); }
  }
}

// the weight layouts the backward reads: Wo^T [256][64] and [Wg; Wv]^T [64][512], bf16, in one launch
template <typename WT_>
__global__ void __launch_bounds__(256) pwa_wprep(const WT_* __restrict__ WG, const WT_* __restrict__ WV, const WT_* __restrict__ WO,
                                                 __nv_bfloat16* __restrict__ WOT, __nv_bfloat16* __restrict__ WGVT) {
  const int t = blockIdx.x * 256 + threadIdx.x;          // 0 .. 2 * HC * D + HC * D
  if (t < 2 * HC * D) {                                  // WGVT[d][k] = (k < 256 ? Wg[k][d] : Wv[k-256][d])
    const int d = t / (2 * HC), k = t % (2 * HC);
    WGVT[t] = __float2bfloat16_rn(to_f(k < HC ? WG[k * D + d] : WV[(k - HC) * D + d]));
  } else if (t < 3 * HC * D) {                           // WOT[c][d] = Wo[d][c]
    const int u = t - 2 * HC * D, c = u / D, d = u % D;
    WOT[u] = __float2bfloat16_rn(to_f(WO[d * HC + c]));
  }
}
}  // namespace

std::vector<torch::Tensor> pwa_wprep_host(torch::Tensor wg, torch::Tensor wv, torch::Tensor wo) {
  TORCH_CHECK(wg.is_contiguous() && wv.is_contiguous() && wo.is_contiguous() && wg.scalar_type() == wv.scalar_type() && wg.scalar_type() == wo.scalar_type() &&
              wg.numel() == HC * D && wv.numel() == HC * D && wo.numel() == D * HC, "wg, wv [256, 64], wo [64, 256], one dtype");
  auto wot = torch::empty({(long)HC, (long)D}, wg.options().dtype(torch::kBFloat16));
  auto wgvt = torch::empty({(long)D, (long)(2 * HC)}, wg.options().dtype(torch::kBFloat16));
  auto st = at::cuda::getCurrentCUDAStream();
  const int blocks = (3 * HC * D + 255) / 256;
  if (wg.scalar_type() == torch::kFloat32)
    pwa_wprep<float><<<blocks, 256, 0, st>>>(wg.data_ptr<float>(), wv.data_ptr<float>(), wo.data_ptr<float>(),
                                             reinterpret_cast<__nv_bfloat16*>(wot.data_ptr()), reinterpret_cast<__nv_bfloat16*>(wgvt.data_ptr()));
  else
    pwa_wprep<__nv_bfloat16><<<blocks, 256, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(wg.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(wv.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(wo.data_ptr()), reinterpret_cast<__nv_bfloat16*>(wot.data_ptr()), reinterpret_cast<__nv_bfloat16*>(wgvt.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {wot, wgvt};
}

// z [N, N, 128] bf16, key mask [N] (bool), LayerNorm affine [128] and proj_z weight [8, 128] (fp32 or bf16) -> w [H, N, N] bf16
torch::Tensor pair_fwd(torch::Tensor z, torch::Tensor mask, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor wb) {
  using namespace pf2;
  TORCH_CHECK(z.is_contiguous() && z.scalar_type() == torch::kBFloat16 && z.dim() == 3 && z.size(2) == DZ && z.size(0) == z.size(1), "z: [N, N, 128] bf16");
  const long N = z.size(0);
  TORCH_CHECK(N % BJ == 0 && N <= NMAX, "N must be a multiple of 128, at most 1024");
  TORCH_CHECK(mask.is_contiguous() && mask.scalar_type() == torch::kBool && mask.numel() == N, "mask: [N] bool (the key axis)");
  TORCH_CHECK(lnw.scalar_type() == lnb.scalar_type() && lnw.is_contiguous() && lnb.is_contiguous() && lnw.numel() == DZ && lnb.numel() == DZ, "LN affine [128]");
  TORCH_CHECK(wb.is_contiguous() && wb.numel() == H * DZ, "wb: [8, 128]");
  auto w = torch::empty({(long)H, N, N}, z.options());
  CUtensorMap zm = make_map<2>(z.data_ptr(), {(uint64_t)DZ, (uint64_t)(N * N)}, {(uint64_t)DZ}, {64, BJ}, CU_TENSOR_MAP_SWIZZLE_128B, "z");
  const int grid = std::min((int)N, num_sms(z.device().index()));
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* mp = reinterpret_cast<const uint8_t*>(mask.data_ptr<bool>());
  auto* wp = reinterpret_cast<__nv_bfloat16*>(w.data_ptr<at::BFloat16>());
  auto launch = [&](auto wt, auto lt) {
    using WT = decltype(wt); using LT = decltype(lt);
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pair_fwd_sm100<WT, LT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    pair_fwd_sm100<WT, LT><<<grid, THREADS, SMEM, st>>>((int)N, (float)eps, zm, mp, reinterpret_cast<const LT*>(lnw.data_ptr()),
                                                        reinterpret_cast<const LT*>(lnb.data_ptr()), reinterpret_cast<const WT*>(wb.data_ptr()), wp);
  };
  const bool wf = wb.scalar_type() == torch::kFloat32, lf = lnw.scalar_type() == torch::kFloat32;
  TORCH_CHECK((wf || wb.scalar_type() == torch::kBFloat16) && (lf || lnw.scalar_type() == torch::kBFloat16), "weights: fp32 or bf16");
  if (wf && lf) launch(float{}, float{});
  else if (wf) launch(float{}, __nv_bfloat16{});
  else if (lf) launch(__nv_bfloat16{}, float{});
  else launch(__nv_bfloat16{}, __nv_bfloat16{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return w;
}

// m [S, N, 64] bf16 -> (v head-major [H, N, S*C], y [S, N, 64])
std::vector<torch::Tensor> ln_vg(torch::Tensor m, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wv, double eps, bool want_y) {
  using namespace lv;
  TORCH_CHECK(m.is_cuda() && m.scalar_type() == torch::kBFloat16 && m.is_contiguous() && m.dim() == 3 && m.size(2) == D, "m: [S, N, 64] bf16");
  TORCH_CHECK(wv.scalar_type() == torch::kBFloat16 && wv.is_contiguous() && wv.size(0) == HC && wv.size(1) == D, "wv: [256, 64] bf16");
  TORCH_CHECK(lnw.scalar_type() == lnb.scalar_type() && lnw.is_contiguous() && lnb.is_contiguous() && lnw.numel() == D, "LN affine [64]");
  const long S = m.size(0), N = m.size(1);
  TORCH_CHECK(S % BS == 0, "S must be a multiple of ", BS);
  auto y = want_y ? torch::empty_like(m) : torch::empty({0}, m.options());
  auto v = torch::empty({H, N, S * C}, m.options());
  CUtensorMap xm = make_map<3>(m.data_ptr(), {(uint64_t)D, (uint64_t)N, (uint64_t)S}, {(uint64_t)D, (uint64_t)N * D}, {64, 1, BS}, CU_TENSOR_MAP_SWIZZLE_128B, "m");
  CUtensorMap ym = make_map<3>(want_y ? y.data_ptr() : m.data_ptr(), {(uint64_t)D, (uint64_t)N, (uint64_t)S}, {(uint64_t)D, (uint64_t)N * D}, {64, 1, BS}, CU_TENSOR_MAP_SWIZZLE_128B, "y");
  CUtensorMap wm = make_map<2>(wv.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, HC}, CU_TENSOR_MAP_SWIZZLE_128B, "wv");
  CUtensorMap vm = make_map<3>(v.data_ptr(), {(uint64_t)C, (uint64_t)S, (uint64_t)H * N}, {(uint64_t)C, (uint64_t)S * C}, {C, BS, 1}, CU_TENSOR_MAP_SWIZZLE_64B, "v");
  const int ntiles = (int)(N * (S / BS));
  const int grid = std::min(ntiles, num_sms(m.device().index()));
  auto st = at::cuda::getCurrentCUDAStream();
  if (lnw.scalar_type() == torch::kFloat32) {
    static bool a = false;
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(ln_vg_sm100<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
    ln_vg_sm100<float><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, want_y ? 1 : 0, xm, ym, wm, vm, lnw.data_ptr<float>(), lnb.data_ptr<float>());
  } else {
    TORCH_CHECK(lnw.scalar_type() == torch::kBFloat16, "LN affine: fp32 or bf16");
    static bool a = false;
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(ln_vg_sm100<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
    ln_vg_sm100<__nv_bfloat16><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, want_y ? 1 : 0, xm, ym, wm, vm,
        reinterpret_cast<const __nv_bfloat16*>(lnw.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(lnb.data_ptr<at::BFloat16>()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {v, y};
}

// the glue with the dgp consumers inside: o [H, N, S*C], y / dres [S, N, 64], wg [HC, 64], wot = Wo^T [HC, 64]
// -> (do head-major [H, N, S*C], dyg [S, N, 64] bf16 (= dgp . Wg), dWo [64, HC], dWg [HC, 64] in the requested dtype)
std::vector<torch::Tensor> pwa_glue2(torch::Tensor o, torch::Tensor y, torch::Tensor dres, torch::Tensor wgw, torch::Tensor wot,
                                     c10::optional<torch::Tensor> dmask, double dscale, int64_t out_bf16) {
  using namespace gl2;
  TORCH_CHECK(y.is_contiguous() && y.dim() == 3 && y.size(2) == D && dres.is_contiguous() && dres.sizes() == y.sizes(), "y, dres: [S, N, 64]");
  const long S = y.size(0), N = y.size(1);
  TORCH_CHECK(N % BI == 0 && S % BS == 0, "N must be a multiple of 128 and S even");
  TORCH_CHECK(o.is_contiguous() && o.sizes() == torch::IntArrayRef({H, N, S * C}), "o: head-major [H, N, S*C] (pwa_fwd2)");
  TORCH_CHECK(wgw.is_contiguous() && wgw.sizes() == torch::IntArrayRef({HC, D}) && wot.is_contiguous() && wot.sizes() == torch::IntArrayRef({HC, D}), "wg, wot: [256, 64]");
  auto dO = torch::empty({H, N, S * C}, y.options());
  auto dyg = torch::empty({S, N, (long)D}, y.options());
  const int ntiles = (int)((N / BI) * (S / BS));
  const int grid = std::min(ntiles, num_sms(y.device().index()));
  auto part = torch::empty({grid, (long)D, (long)PW}, y.options().dtype(torch::kFloat32));
  CUtensorMap om = make_map<3>(o.data_ptr(), {(uint64_t)C, (uint64_t)(H * N), (uint64_t)S}, {(uint64_t)(S * C), (uint64_t)C}, {C, BI, BS}, CU_TENSOR_MAP_SWIZZLE_64B, "o");
  auto nat = [&](void* p, const char* w) { return make_map<3>(p, {(uint64_t)D, (uint64_t)N, (uint64_t)S}, {(uint64_t)D, (uint64_t)N * D}, {64, BI, 1}, CU_TENSOR_MAP_SWIZZLE_128B, w); };
  CUtensorMap rm = nat(dres.data_ptr(), "dres"), ym = nat(y.data_ptr(), "y"), ygm = nat(dyg.data_ptr(), "dyg");
  CUtensorMap gm = make_map<2>(wgw.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, C}, CU_TENSOR_MAP_SWIZZLE_128B, "wg");
  CUtensorMap wtm = make_map<2>(wot.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, C}, CU_TENSOR_MAP_SWIZZLE_128B, "wot");
  CUtensorMap dom = make_map<2>(dO.data_ptr(), {(uint64_t)(S * C), (uint64_t)(H * N)}, {(uint64_t)(S * C)}, {64, BI}, CU_TENSOR_MAP_SWIZZLE_128B, "do");
  const __nv_bfloat16* dmp = nullptr;
  if (dmask.has_value() && dmask->numel()) {
    TORCH_CHECK(dmask->scalar_type() == torch::kBFloat16 && dmask->is_contiguous() && dmask->numel() == N * D, "dmask: [N, 64] bf16");
    dmp = reinterpret_cast<const __nv_bfloat16*>(dmask->data_ptr<at::BFloat16>());
  }
  auto st = at::cuda::getCurrentCUDAStream();
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_glue2_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  pwa_glue2_sm100<<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, om, rm, ym, gm, wtm, dom, ygm, dmp, (float)dscale, part.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const auto odt = out_bf16 ? torch::kBFloat16 : torch::kFloat32;
  auto dWo = torch::empty({(long)D, (long)HC}, y.options().dtype(odt)), dWg = torch::empty({(long)HC, (long)D}, y.options().dtype(odt));
  if (out_bf16) glue2_finish<__nv_bfloat16><<<D * PW / 32, 256, 0, st>>>(part.data_ptr<float>(), grid, reinterpret_cast<__nv_bfloat16*>(dWo.data_ptr()),
                                                                          reinterpret_cast<__nv_bfloat16*>(dWg.data_ptr()));
  else glue2_finish<float><<<D * PW / 32, 256, 0, st>>>(part.data_ptr<float>(), grid, dWo.data_ptr<float>(), dWg.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, dyg, dWo, dWg};
}

// dv head-major [H, N, S*C], dyg / y / x / dout [S, N, 64], wgvT = [Wg; Wv]^T [64, 512] (the Wv half is read)
// -> (dm, dWv [256, 64], dgamma, dbeta) in the requested dtypes
std::vector<torch::Tensor> dv_bwd(torch::Tensor dvh, torch::Tensor dyg, torch::Tensor y, torch::Tensor x, torch::Tensor dout, torch::Tensor wgvT,
                                  torch::Tensor lnw, double eps, int64_t w_bf16, int64_t ln_bf16) {
  using namespace dv2;
  TORCH_CHECK(x.dim() == 3 && x.size(2) == D, "x: [S, N, 64]");
  const long S = x.size(0), N = x.size(1);
  for (const auto* t : {&y, &x, &dout, &dyg}) TORCH_CHECK(t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->sizes() == x.sizes(), "y/x/dout/dyg: [S, N, 64] bf16");
  TORCH_CHECK(dvh.scalar_type() == torch::kBFloat16 && dvh.is_contiguous() && dvh.numel() == H * N * S * C, "dv: head-major [H, N, S*C]");
  TORCH_CHECK(wgvT.is_contiguous() && wgvT.size(0) == D && wgvT.size(1) == 2 * HC, "wgvT: [64, 512] bf16");
  TORCH_CHECK(S % BM == 0, "S must be a multiple of 128");
  auto dm = torch::empty_like(x);
  const int ntiles = (int)(N * (S / BM));
  const int grid = std::min(ntiles, num_sms(x.device().index()));
  auto dw = torch::empty({grid, (long)D, (long)KD}, x.options().dtype(torch::kFloat32));
  auto dln = torch::empty({grid, 2, (long)D}, x.options().dtype(torch::kFloat32));
  CUtensorMap vm = make_map<3>(dvh.data_ptr(), {(uint64_t)C, (uint64_t)S, (uint64_t)(H * N)}, {(uint64_t)C, (uint64_t)(S * C)}, {C, BM, 1}, CU_TENSOR_MAP_SWIZZLE_64B, "dv");
  auto nat = [&](const torch::Tensor& tt, const char* w) {
    return make_map<3>(tt.data_ptr(), {(uint64_t)D, (uint64_t)N, (uint64_t)S}, {(uint64_t)D, (uint64_t)N * D}, {64, 1, BM}, CU_TENSOR_MAP_SWIZZLE_128B, w);
  };
  CUtensorMap wm = make_map<2>(wgvT.data_ptr(), {(uint64_t)(2 * HC), (uint64_t)D}, {(uint64_t)(2 * HC)}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "wgvT");
  CUtensorMap ym = nat(y, "y"), xm = nat(x, "x"), om = nat(dout, "dout"), dmm = nat(dm, "dm"), ygm = nat(dyg, "dyg");
  auto st = at::cuda::getCurrentCUDAStream();
  if (lnw.scalar_type() == torch::kFloat32) {
    static bool a = false;
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(dv_bwd_sm100<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
    dv_bwd_sm100<float><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, vm, wm, ym, xm, om, dmm, ygm, lnw.data_ptr<float>(), dw.data_ptr<float>(), dln.data_ptr<float>());
  } else {
    static bool a = false;
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(dv_bwd_sm100<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
    dv_bwd_sm100<__nv_bfloat16><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, vm, wm, ym, xm, om, dmm, ygm,
        reinterpret_cast<const __nv_bfloat16*>(lnw.data_ptr<at::BFloat16>()), dw.data_ptr<float>(), dln.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const auto wdt = w_bf16 ? torch::kBFloat16 : torch::kFloat32, ldt = ln_bf16 ? torch::kBFloat16 : torch::kFloat32;
  auto dWv = torch::empty({(long)HC, (long)D}, x.options().dtype(wdt));
  auto dlw = torch::empty({(long)D}, x.options().dtype(ldt)), dlb = torch::empty({(long)D}, x.options().dtype(ldt));
  auto fin = [&](auto wo, auto lo) {
    using WO = decltype(wo); using LO = decltype(lo);
    dv_finish<WO, LO><<<HC * D / 32 + 2 * D / 32, 256, 0, st>>>(dw.data_ptr<float>(), dln.data_ptr<float>(), grid, reinterpret_cast<WO*>(dWv.data_ptr()),
                                                    reinterpret_cast<LO*>(dlw.data_ptr()), reinterpret_cast<LO*>(dlb.data_ptr()));
  };
  if (w_bf16) { if (ln_bf16) fin(__nv_bfloat16{}, __nv_bfloat16{}); else fin(__nv_bfloat16{}, float{}); }
  else { if (ln_bf16) fin(float{}, __nv_bfloat16{}); else fin(float{}, float{}); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dm, dWv, dlw, dlb};
}

// z [N, N, 128] bf16, w16 [H, N, N] bf16 (the forward's softmax), dw [H, N, N] fp32 -> (dz, dWb [H,128], dgamma, dbeta) fp32
std::vector<torch::Tensor> pair_bwd(torch::Tensor z, torch::Tensor w16, torch::Tensor dw, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor wb,
                                    int64_t wb_bf16, int64_t ln_bf16) {
  using namespace pb2;
  TORCH_CHECK(z.is_contiguous() && z.scalar_type() == torch::kBFloat16 && z.dim() == 3 && z.size(2) == DZ, "z: [N, N, 128] bf16");
  const long N = z.size(0);
  TORCH_CHECK(N % BJ == 0, "N must be a multiple of 128");
  TORCH_CHECK(w16.is_contiguous() && w16.numel() == H * N * N && dw.is_contiguous() && dw.numel() == H * N * N &&
              (dw.scalar_type() == torch::kFloat32 || dw.scalar_type() == torch::kBFloat16), "w16 [H, N, N] bf16, dw [H, N, N] fp32 or bf16");
  TORCH_CHECK(lnw.scalar_type() == lnb.scalar_type() && (lnw.scalar_type() == torch::kFloat32 || lnw.scalar_type() == torch::kBFloat16) &&
              lnw.is_contiguous() && lnb.is_contiguous() && lnw.numel() == DZ && lnb.numel() == DZ, "LN affine: [128] fp32 or bf16");
  TORCH_CHECK(wb.is_contiguous() && wb.numel() == H * DZ && (wb.scalar_type() == torch::kFloat32 || wb.scalar_type() == torch::kBFloat16), "wb: [8, 128]");
  auto dz = torch::empty_like(z);
  const int ntiles = (int)(N * (N / BJ));
  const int grid = std::min(ntiles, num_sms(z.device().index()));
  auto part = torch::empty({grid, (long)PSTRIDE}, z.options().dtype(torch::kFloat32));
  CUtensorMap zm = make_map<2>(z.data_ptr(), {(uint64_t)DZ, (uint64_t)(N * N)}, {(uint64_t)DZ}, {64, BJ}, CU_TENSOR_MAP_SWIZZLE_128B, "z");
  CUtensorMap dzm = make_map<2>(dz.data_ptr(), {(uint64_t)DZ, (uint64_t)(N * N)}, {(uint64_t)DZ}, {64, BJ}, CU_TENSOR_MAP_SWIZZLE_128B, "dz");
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* w16p = reinterpret_cast<const __nv_bfloat16*>(w16.data_ptr<at::BFloat16>());
  auto sd = torch::empty({N, (long)H}, z.options().dtype(torch::kFloat32));
  const bool dwf = dw.scalar_type() == torch::kFloat32;
  if (dwf) pair_sdot_sm100<float><<<(int)((H * N + 7) / 8), 256, 0, st>>>(w16p, dw.data_ptr<float>(), sd.data_ptr<float>(), (int)N);
  else pair_sdot_sm100<__nv_bfloat16><<<(int)((H * N + 7) / 8), 256, 0, st>>>(w16p, reinterpret_cast<const __nv_bfloat16*>(dw.data_ptr()), sd.data_ptr<float>(), (int)N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  CUtensorMap wm = make_map<3>(w16.data_ptr(), {(uint64_t)N, (uint64_t)N, (uint64_t)H}, {(uint64_t)N, (uint64_t)(N * N)}, {BJ, 1, H}, CU_TENSOR_MAP_SWIZZLE_NONE, "w16");
  CUtensorMap dwm = make_map<3>(dw.data_ptr(), {(uint64_t)N, (uint64_t)N, (uint64_t)H}, {(uint64_t)N, (uint64_t)(N * N)}, {BJ, 1, H}, CU_TENSOR_MAP_SWIZZLE_NONE, "dw",
                                dwf ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, dwf ? 4 : 2);
  CUtensorMap sdm = make_map<2>(sd.data_ptr(), {(uint64_t)H, (uint64_t)N}, {(uint64_t)H}, {H, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "sdot", CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  auto dWb = torch::empty({(long)H, (long)DZ}, z.options().dtype(wb_bf16 ? torch::kBFloat16 : torch::kFloat32));
  const auto ldt = ln_bf16 ? torch::kBFloat16 : torch::kFloat32;
  auto dg = torch::empty({(long)DZ}, z.options().dtype(ldt)), dbe = torch::empty({(long)DZ}, z.options().dtype(ldt));
  auto red = torch::Tensor();
  auto launch = [&](auto wt, auto lt, auto ot, auto lo) {
    using WT = decltype(wt); using LT = decltype(lt); using OT = decltype(ot); using LO = decltype(lo);
    auto kbody = [&](auto dt) {
      using DT = decltype(dt);
      static bool a = false;
      if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bwd_sm100<WT, LT, DT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
      pair_bwd_sm100<WT, LT, DT><<<grid, THREADS, SMEM, st>>>((int)N, ntiles, (float)eps, zm, dzm, wm, dwm, sdm, reinterpret_cast<const LT*>(lnw.data_ptr()),
        reinterpret_cast<const LT*>(lnb.data_ptr()), reinterpret_cast<const WT*>(wb.data_ptr()), part.data_ptr<float>());
    };
    if (dwf) kbody(float{}); else kbody(__nv_bfloat16{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    red = colsum(part);                                             // [PSTRIDE]: M, then the 4 warps' cs
    pair_param_finish<WT, LT, OT, LO><<<1, DZ, 0, st>>>(red.data_ptr<float>(), reinterpret_cast<const LT*>(lnw.data_ptr()), reinterpret_cast<const LT*>(lnb.data_ptr()),
        reinterpret_cast<const WT*>(wb.data_ptr()), reinterpret_cast<OT*>(dWb.data_ptr()), reinterpret_cast<LO*>(dg.data_ptr()), reinterpret_cast<LO*>(dbe.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  const bool wf = wb.scalar_type() == torch::kFloat32, lf = lnw.scalar_type() == torch::kFloat32;
  auto with_out = [&](auto wt, auto lt) {
    if (wb_bf16) { if (ln_bf16) launch(wt, lt, __nv_bfloat16{}, __nv_bfloat16{}); else launch(wt, lt, __nv_bfloat16{}, float{}); }
    else { if (ln_bf16) launch(wt, lt, float{}, __nv_bfloat16{}); else launch(wt, lt, float{}, float{}); }
  };
  if (wf && lf) with_out(float{}, float{});
  else if (wf) with_out(float{}, __nv_bfloat16{});
  else if (lf) with_out(__nv_bfloat16{}, float{});
  else with_out(__nv_bfloat16{}, __nv_bfloat16{});
  return {dz, dWb, dg, dbe};
}


// the split forward: o = pwa_ctr(w, v), then the gate / out-projection / residual pass over o
// y = LN(msa) is recomputed in the second pass (bit-identical to ln_vg's), so inference never writes or reads y
// the gate / out-projection / residual pass over a given o (head-major [H, N, S*C], e.g. cuBLAS bmm(w, v)) -> out
torch::Tensor pwa_gate_out(torch::Tensor o, torch::Tensor msa, torch::Tensor lnw, torch::Tensor lnb, double eps,
                           torch::Tensor wgw, torch::Tensor wow, c10::optional<torch::Tensor> dmask, double dscale) {
  using namespace go;
  TORCH_CHECK(o.is_contiguous() && o.dim() == 3 && o.size(0) == H && o.scalar_type() == torch::kBFloat16, "o: head-major [H, N, S*C] bf16");
  const long N = o.size(1), S = o.size(2) / C, M = S * N;
  TORCH_CHECK(M % BM == 0, "S * N must be a multiple of 128");
  TORCH_CHECK(msa.is_contiguous() && msa.numel() == M * D, "msa: [S, N, 64]");
  TORCH_CHECK(lnw.scalar_type() == lnb.scalar_type() && lnw.is_contiguous() && lnb.is_contiguous() && lnw.numel() == D, "LayerNorm affine [64]");
  TORCH_CHECK(wgw.is_contiguous() && wgw.sizes() == torch::IntArrayRef({HC, D}) && wow.is_contiguous() && wow.sizes() == torch::IntArrayRef({D, HC}), "wg [256, 64], wo [64, 256]");
  auto out = torch::empty({S, N, D}, msa.options());
  TORCH_CHECK(N % BM == 0, "N must be a multiple of 128");
  CUtensorMap om = make_map<2>(o.data_ptr(), {(uint64_t)(S * C), (uint64_t)(H * N)}, {(uint64_t)(S * C)}, {C, BM}, CU_TENSOR_MAP_SWIZZLE_64B, "o");
  CUtensorMap ym = make_map<2>(msa.data_ptr(), {(uint64_t)D, (uint64_t)M}, {(uint64_t)D}, {64, BM}, CU_TENSOR_MAP_SWIZZLE_128B, "msa");
  CUtensorMap gm = make_map<2>(wgw.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, HC}, CU_TENSOR_MAP_SWIZZLE_128B, "wg");
  CUtensorMap wm = make_map<2>(wow.data_ptr(), {(uint64_t)HC, (uint64_t)D}, {(uint64_t)HC}, {64, D}, CU_TENSOR_MAP_SWIZZLE_128B, "wo");
  const __nv_bfloat16* dmp = nullptr;
  if (dmask.has_value() && dmask->numel()) {
    TORCH_CHECK(dmask->scalar_type() == torch::kBFloat16 && dmask->is_contiguous() && dmask->numel() == N * D, "dmask: [N, 64] bf16");
    dmp = reinterpret_cast<const __nv_bfloat16*>(dmask->data_ptr<at::BFloat16>());
  }
  const int ntiles = (int)(M / BM);
  const int grid = std::min(ntiles, num_sms(msa.device().index()));
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* mp = reinterpret_cast<const __nv_bfloat16*>(msa.data_ptr<at::BFloat16>());
  auto* outp = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  if (lnw.scalar_type() == torch::kFloat32) {
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_gate_out_sm100<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    pwa_gate_out_sm100<float><<<grid, THREADS, SMEM, st>>>((int)N, ntiles, (float)eps, lnw.data_ptr<float>(), lnb.data_ptr<float>(), om, ym, gm, wm, mp, outp, dmp, (float)dscale);
  } else {
    TORCH_CHECK(lnw.scalar_type() == torch::kBFloat16, "LayerNorm affine: fp32 or bf16");
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_gate_out_sm100<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    pwa_gate_out_sm100<__nv_bfloat16><<<grid, THREADS, SMEM, st>>>((int)N, ntiles, (float)eps,
        reinterpret_cast<const __nv_bfloat16*>(lnw.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(lnb.data_ptr<at::BFloat16>()),
        om, ym, gm, wm, mp, outp, dmp, (float)dscale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_vg", &ln_vg, "sm100 LayerNorm + value projection: (v head-major [H,N,S*C], y [S,N,64])",
        py::arg("m"), py::arg("lnw"), py::arg("lnb"), py::arg("wv"), py::arg("eps") = 1e-5, py::arg("want_y") = true);
  m.def("pair_fwd", &pair_fwd, "sm100 pair-side forward: LN_z folded into proj_z (tensor core on raw z) -> key mask -> softmax over j",
        py::arg("z"), py::arg("mask"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("wb"));
  m.def("pair_bwd", &pair_bwd, "sm100 pair-side backward: softmax-bwd -> proj_z-bwd -> LN_z-bwd -> (dz, dWb, dgamma_z, dbeta_z)",
        py::arg("z"), py::arg("w16"), py::arg("dw"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("wb"), py::arg("wb_bf16") = 0, py::arg("ln_bf16") = 0);
  m.def("pwa_glue2", &pwa_glue2, "sm100 PWA backward glue with the dgp consumers inside: (do head-major, dyg = dgp Wg [S,N,64], dWo, dWg)",
        py::arg("o"), py::arg("y"), py::arg("dres"), py::arg("wg"), py::arg("wot"), py::arg("dmask") = py::none(), py::arg("dscale") = 1.0,
        py::arg("out_bf16") = 0);
  m.def("dv_bwd", &dv_bwd, "sm100 PWA backward tail with pwa_glue2: (dm = LN_bwd(dyg + dv Wv) + dout, dWv, dgamma, dbeta)",
        py::arg("dv"), py::arg("dyg"), py::arg("y"), py::arg("x"), py::arg("dout"), py::arg("wgvT"), py::arg("lnw"), py::arg("eps") = 1e-5,
        py::arg("w_bf16") = 0, py::arg("ln_bf16") = 0);
  m.def("pwa_wprep", &pwa_wprep_host, "Wo^T [256, 64] and [Wg; Wv]^T [64, 512] in bf16 (the layouts the backward reads), one launch",
        py::arg("wg"), py::arg("wv"), py::arg("wo"));
  m.def("pwa_gate_out", &pwa_gate_out, "sm100 PWA gate / out-projection / residual pass over a given head-major o",
        py::arg("o"), py::arg("msa"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("wg"), py::arg("wo"),
        py::arg("dmask") = py::none(), py::arg("dscale") = 1.0);
}
