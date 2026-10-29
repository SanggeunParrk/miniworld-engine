// PWA (MSAPairWeightedAveraging) value side on B200 (sm_100a) at the widths the MiniWorld kernels (pwa_sm100.cu: d_msa 64, 8 x 32) do not
// cover: d_msa PW_D in {64, 128}, 8 heads x PW_C in {8, 16, 32} value channels (AF3 / Protenix-v1 64 / 8 x 8, Protenix-v2 / OpenDDE
// 128 / 8 x 8, ESMFold2 128 / 8 x 16, Chai-1 64 / 8 x 32 with d_pair 256), and IntelliFold-v2's 256 / 8 x 32 (inference: its projection
// weights, 128 KiB each, stream through a shared-memory ring of k-blocks -- WS below), one build per (PW_D, PW_C). The pair side is
// pwa_sm100.cu's pair_fwd (d_pair 128 / 256 / 384 / 512, N <= 2048), the contraction o = w . v is cuBLAS bmm, as on the MiniWorld width.
//
//   ln_vg_w:     y = LN(m), v = y . Wv^T head-major [H, N, S*C]          (tile = one token x 128 MSA rows)
//   gate_out_w:  out = m + drop(bf16(sum_h (sigmoid(y Wg_h^T) * o_h) . Wo_h))   (y recomputed; tile = 128 tokens of one MSA row)
//
// The structure is pwa_sm100.cu's ln_vg / pwa_gate_out (warp-specialised TMA / tcgen05 / TMEM pipelines); what changes with the width:
// the m / y tiles are D / 64 128-byte-swizzled 64-channel blocks, the [rows][C] value / o tiles use the swizzle that fits a C * 2 byte
// row (64 B: 64B swizzle, 32 B: 32B, 16 B: none), the LayerNorm is one thread per row in three passes over shared memory (a 128-wide row
// does not fit the registers of a 448 / 576-thread CTA), and the epilogue drains the output in 32-channel pieces.
#include <torch/extension.h>
#include "sm100.cuh"

using namespace sm100;

#ifndef PW_D
#define PW_D 128
#endif
#ifndef PW_C
#define PW_C 16
#endif

// the backward kernels (glue_w, dvb_w) are built where the value width fits under d_msa: AF3 64 / 8 x 8, Protenix-v2 / OpenDDE 128 / 8 x 8,
// ESMFold2 128 / 8 x 16 (Chai-1's 64 / 8 x 32 is MiniWorld's MSA width: it trains on pwa_sm100.cu)
#define PW_BWD (8 * PW_C <= PW_D && PW_D <= 128)

namespace {
constexpr int H = 8, D = PW_D, C = PW_C, HC = H * C, KB = D / 64;
static_assert(D == 64 || D == 128 || (D == 256 && C == 32), "d_msa 64 or 128, or 256 x 32 (IntelliFold-v2)");
// d_msa 256: a projection's weights (128 KiB) do not fit next to the 64 KiB tiles: they stream through a ring of k-blocks
constexpr bool WS = D * HC * 2 > 64 * 1024;
static_assert(C == 8 || C == 16 || C == 32, "d_hidden 8, 16 or 32");

// a [rows][C] bf16 tile (C * 2 bytes a row) as TMA lands it: the 16-byte chunk c8 of row r
constexpr CUtensorMapSwizzle VSW = C == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : (C == 16 ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE);
__device__ __forceinline__ int vchunk(int r, int c8) {
  if constexpr (C == 32) return c8 ^ ((r >> 1) & 3);      // Swizzle<2,4,3>: byte bits 4-5 ^= bits 7-8, a row is 64 B
  else if constexpr (C == 16) return c8 ^ ((r >> 2) & 1); // Swizzle<1,4,3>: byte bit 4 ^= bit 7, a row is 32 B
  else return c8;
}
template <int N_> __device__ __forceinline__ void tmem_ldn(uint32_t taddr, float* v) {
  if constexpr (N_ == 8) tmem_ld8(taddr, v);
  else if constexpr (N_ == 16) tmem_ld16(taddr, v);
  else tmem_ld32(taddr, v);
}

// u = bf16(sigmoid(bf16(x)) * o) on a packed bf16 pair: the module's own bf16 gate (pwa_sm100.cu's gate_mul_bf16x2)
__device__ __forceinline__ uint32_t gate_mul_bf16x2(float x0, float x1, uint32_t o2) {
  uint32_t h, t, g, u;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(0.5f * x1), "f"(0.5f * x0));
  asm("tanh.approx.bf16x2 %0, %1;" : "=r"(t) : "r"(h));
  asm("fma.rn.bf16x2 %0, %1, %2, %2;" : "=r"(g) : "r"(t), "r"(0x3F003F00u));
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(u) : "r"(g), "r"(o2));
  return u;
}

// gamma / beta of channels d0 .. d0 + 7: from shared memory (sLN = gamma [D] | beta [D], fp32), or -- the weight-streaming build, whose
// shared memory is full -- from global through L1 (LT_: fp32 or bf16; the same values)
struct LnShared {
  const float* s;
  __device__ __forceinline__ void get(int d0, float4& g0, float4& g1, float4& b0, float4& b1) const {
    g0 = *reinterpret_cast<const float4*>(s + d0); g1 = *reinterpret_cast<const float4*>(s + d0 + 4);
    b0 = *reinterpret_cast<const float4*>(s + D + d0); b1 = *reinterpret_cast<const float4*>(s + D + d0 + 4);
  }
};
__device__ __forceinline__ void ln_ld8(const float* p, float4& a, float4& b) { a = __ldg(reinterpret_cast<const float4*>(p)); b = __ldg(reinterpret_cast<const float4*>(p) + 1); }
__device__ __forceinline__ void ln_ld8(const __nv_bfloat16* p, float4& a, float4& b) {
  const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
  const float2 x0 = bf2f(u.x), x1 = bf2f(u.y), x2 = bf2f(u.z), x3 = bf2f(u.w);
  a = make_float4(x0.x, x0.y, x1.x, x1.y); b = make_float4(x2.x, x2.y, x3.x, x3.y);
}
template <typename LT_>
struct LnGlobal {
  const LT_* w; const LT_* b;
  __device__ __forceinline__ void get(int d0, float4& g0, float4& g1, float4& b0, float4& b1) const { ln_ld8(w + d0, g0, g1); ln_ld8(b + d0, b0, b1); }
};

// y = LN(x) of row r of a [KB][128][64] 128B-swizzled bf16 tile, in place: one thread, three passes (mean, centred variance, write)
template <typename GB>
__device__ __forceinline__ void ln_row(__nv_bfloat16* tile, int r, const GB& gb, float eps) {
  float2 s2 = make_float2(0.f, 0.f);
#pragma unroll 1
  for (int kb = 0; kb < KB; ++kb)
#pragma unroll
    for (int c8 = 0; c8 < 8; ++c8) {
      const uint4 u = *reinterpret_cast<const uint4*>(tile + kb * 128 * 64 + r * 64 + ((c8 ^ (r & 7)) << 3));
      s2 = add2(s2, add2(add2(bf2f(u.x), bf2f(u.y)), add2(bf2f(u.z), bf2f(u.w))));
    }
  const float mean = (s2.x + s2.y) * (1.f / D);
  const float2 nm = make_float2(-mean, -mean);
  float2 v2 = make_float2(0.f, 0.f);
#pragma unroll 1
  for (int kb = 0; kb < KB; ++kb)
#pragma unroll
    for (int c8 = 0; c8 < 8; ++c8) {
      const uint4 u = *reinterpret_cast<const uint4*>(tile + kb * 128 * 64 + r * 64 + ((c8 ^ (r & 7)) << 3));
      const float2 x0 = add2(bf2f(u.x), nm), x1 = add2(bf2f(u.y), nm), x2 = add2(bf2f(u.z), nm), x3 = add2(bf2f(u.w), nm);
      v2 = fma2(x0, x0, fma2(x1, x1, fma2(x2, x2, fma2(x3, x3, v2))));
    }
  const float rstd = 1.f / sqrtf((v2.x + v2.y) * (1.f / D) + eps);
  const float2 rs = make_float2(rstd, rstd);
#pragma unroll 1
  for (int kb = 0; kb < KB; ++kb)
#pragma unroll
    for (int c8 = 0; c8 < 8; ++c8) {
      uint4* p = reinterpret_cast<uint4*>(tile + kb * 128 * 64 + r * 64 + ((c8 ^ (r & 7)) << 3));
      const uint4 u = *p;
      const int d0 = kb * 64 + c8 * 8;
      float4 g0, g1, b0, b1;
      gb.get(d0, g0, g1, b0, b1);
      const float2 y0 = fma2(mul2(add2(bf2f(u.x), nm), rs), make_float2(g0.x, g0.y), make_float2(b0.x, b0.y));
      const float2 y1 = fma2(mul2(add2(bf2f(u.y), nm), rs), make_float2(g0.z, g0.w), make_float2(b0.z, b0.w));
      const float2 y2 = fma2(mul2(add2(bf2f(u.z), nm), rs), make_float2(g1.x, g1.y), make_float2(b1.x, b1.y));
      const float2 y3 = fma2(mul2(add2(bf2f(u.w), nm), rs), make_float2(g1.z, g1.w), make_float2(b1.z, b1.w));
      uint4 o;
      o.x = pack2(y0.x, y0.y); o.y = pack2(y1.x, y1.y); o.z = pack2(y2.x, y2.y); o.w = pack2(y3.x, y3.y);
      *p = o;
    }
}

// ---------------------------------------------------------------------------------------------
// ln_vg_w.  Tile = token n x 128 MSA rows s.  warp 0: x tiles by TMA (Wv once); warps 2-5: the LayerNorm in place, a thread a row;
// warp 1: stores y by TMA (training) and issues v = y . Wv^T (M = 128 s, N = HC, K = D) into one of two TMEM accumulators;
// warps 6-9 drain it (TMEM lane = s): a thread holds its row's HC values = eight C-wide head rows, written into eight [128 s][C]
// tiles -- the head-major v boxes -- and stored by TMA.
namespace lvw {
constexpr int BS = 128, NST = WS ? 1 : 3, VB = WS ? 1 : 2;
constexpr int XT = BS * D;                      // x / y tile [KB][128][64]
constexpr int VT = H * BS * C;                  // v staging [8 heads][128 s][C] (VB of them)
constexpr int WT = HC * D;                      // Wv [KB][HC][64], or (WS) a ring of WRS k-blocks [HC][64]
constexpr int WRS = 2;
constexpr int WBUF = WS ? WRS * HC * 64 : WT;
constexpr int THREADS = 320;
constexpr int SMEM = 1024 + (NST * XT + VB * VT + WBUF) * 2 + 2 * D * 4 + 512;
constexpr uint32_t IDESC = idesc_bf16(128, HC);
constexpr int TCOLS = 2 * HC < 32 ? 32 : 2 * HC;
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace lvw

template <typename LT_>
__global__ void __launch_bounds__(lvw::THREADS, 1) ln_vg_w_sm100(
    int N, int S, int ntiles, float eps, int want_y,
    const __grid_constant__ CUtensorMap xmap,     // m [S][N][D]: (D, N, S), box (64, 1, 128), 128B swizzle
    const __grid_constant__ CUtensorMap ymap,     // y, same
    const __grid_constant__ CUtensorMap wmap,     // Wv [HC][D]: (D, HC), box (64, HC), 128B swizzle
    const __grid_constant__ CUtensorMap vmap,     // v [H*N][S*C] as (C, S, H*N), box (C, 128, 1)
    const LT_* __restrict__ LNW, const LT_* __restrict__ LNB) {
  using namespace lvw;
  const int NSB = S / BS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sV = sX + NST * XT;
  __nv_bfloat16* sW = sV + VB * VT;
  float* sLN = reinterpret_cast<float*>(sW + WBUF);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLN + 2 * D);
  uint64_t* xfull = bars;              // [NST]
  uint64_t* yfull = xfull + NST;       // [NST] count 4 (LN warps)
  uint64_t* xempty = yfull + NST;      // [NST] count 2: the GEMM's commit + y's store has read the stage (or the MMA warp's arrive)
  uint64_t* accf = xempty + NST;       // [2]
  uint64_t* acce = accf + 2;           // [2] count 4
  uint64_t* wful = acce + 2;           // [1]
  uint64_t* wrf = wful + 1;            // [WRS] (WS) a Wv k-block landed
  uint64_t* wre = wrf + WRS;           // [WRS] (WS) its GEMM retired
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wre + WRS);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { bar_init(xfull + s, 1); bar_init(yfull + s, 4); bar_init(xempty + s, 2); }
    for (int b = 0; b < 2; ++b) { bar_init(accf + b, 1); bar_init(acce + b, 4); }
    bar_init(wful, 1);
    for (int k = 0; k < WRS; ++k) { bar_init(wrf + k, 1); bar_init(wre + k, 1); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, TCOLS);
  if (tid < D) { sLN[tid] = to_f(LNW[tid]); sLN[D + tid] = to_f(LNB[tid]); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      if constexpr (!WS) {
        expect_tx(wful, WT * 2);
        for (int kb = 0; kb < KB; ++kb) load_2d(&wmap, sW + kb * HC * 64, wful, kb * 64, 0);
      }
      int g = 0, gk = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++g) {
        const int n = t / NSB, s0 = (t % NSB) * BS, st = g % NST;
        if (g >= NST) wait(xempty + st, ((g / NST) - 1) & 1);
        expect_tx(xfull + st, XT * 2);
        for (int kb = 0; kb < KB; ++kb) load_3d(&xmap, sX + st * XT + kb * BS * 64, xfull + st, kb * 64, n, s0);
        if constexpr (WS)                        // the tile's Wv k-blocks, in the order the GEMM takes them
          for (int kb = 0; kb < KB; ++kb, ++gk) {
            const int sl = gk % WRS;
            if (gk >= WRS) wait(wre + sl, ((gk / WRS) - 1) & 1);
            expect_tx(wrf + sl, HC * 64 * 2);
            load_2d(&wmap, sW + sl * HC * 64, wrf + sl, kb * 64, 0);
          }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      if constexpr (!WS) wait(wful, 0);
      int lt = 0, gk = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int n = t / NSB, s0 = (t % NSB) * BS, st = lt % NST, b = lt & 1;
        wait(yfull + st, (lt / NST) & 1);
        if (want_y) {
          for (int kb = 0; kb < KB; ++kb) store_3d(&ymap, sX + st * XT + kb * BS * 64, kb * 64, n, s0);
          bulk_commit();
        } else {
          arrive(xempty + st);
        }
        if (lt >= 2) wait(acce + b, ((lt >> 1) - 1) & 1);
        tc_fence_after();
        const __nv_bfloat16* y = sX + st * XT;
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
          const __nv_bfloat16* wk = sW + kb * HC * 64;
          if constexpr (WS) {
            const int sl = gk % WRS;
            wait(wrf + sl, (gk / WRS) & 1);
            tc_fence_after();
            wk = sW + sl * HC * 64;
          }
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            mma_ss(tmem + b * HC, desc_k128(y + kb * BS * 64 + ks * 16), desc_k128(wk + ks * 16), IDESC, (kb | ks) ? 1u : 0u);
          if constexpr (WS) { mma_commit(wre + gk % WRS); ++gk; }
        }
        mma_commit(accf + b);
        mma_commit(xempty + st);
        if (want_y && NST == 1) {                // one stage: the next tile's x waits on this release, so it cannot be deferred
          bulk_wait_read<0>();
          arrive(xempty + st);
        } else if (want_y && lt >= 1) {
          bulk_wait_read<1>();
          arrive(xempty + (lt - 1) % NST);
        }
      }
      if (want_y && NST > 1 && lt >= 1) {        // the last tile's y store: release its stage too (nothing waits on it, but keep the count)
        bulk_wait_read<0>();
        arrive(xempty + (lt - 1) % NST);
      }
      bulk_wait<0>();
    }
  } else if (warp <= 5) {
    const int r = (warp - 2) * 32 + lane;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int st = lt % NST;
      wait(xfull + st, (lt / NST) & 1);
      ln_row(sX + st * XT, r, LnShared{sLN}, eps);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(yfull + st);
    }
  } else {
    const int q = warp & 3, r = q * 32 + lane;
    const bool leader = (warp == 6 && lane == 0);
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int n = t / NSB, s0 = (t % NSB) * BS, b = lt & 1;
      wait(accf + b, (lt >> 1) & 1);
      tc_fence_after();
      if (leader) bulk_wait_read<VB - 1>();      // the v stores of tile lt - VB have left staging buffer lt % VB
      named_sync(2, 128);
      __nv_bfloat16* vb = sV + (lt % VB) * VT;
#pragma unroll 1
      for (int h = 0; h < H; ++h) {
        float v[C];
        tmem_ldn<C>(tmem_at(tmem + b * HC, q * 32, h * C), v);
        tmem_wait_ld();
        if (h == H - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + b); }
        __nv_bfloat16* row = vb + (h * BS + r) * C;
#pragma unroll
        for (int c8 = 0; c8 < C / 8; ++c8) {
          uint4 o;
          o.x = pack2(v[c8 * 8 + 0], v[c8 * 8 + 1]); o.y = pack2(v[c8 * 8 + 2], v[c8 * 8 + 3]);
          o.z = pack2(v[c8 * 8 + 4], v[c8 * 8 + 5]); o.w = pack2(v[c8 * 8 + 6], v[c8 * 8 + 7]);
          *reinterpret_cast<uint4*>(row + (vchunk(r, c8) << 3)) = o;
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
  if (warp == 1) tmem_dealloc(tmem, TCOLS);
}

// ---------------------------------------------------------------------------------------------
// gate_out_w.  Per 128-row tile of the (s, i) rows (128 tokens of one MSA row): g = sigmoid(y Wg^T) [128 x HC], u = g * o,
// out = msa + drop(bf16(u Wo^T)).  warp 0: o | msa tiles (2-stage ring), the resident Wg, Wo; warp 1: tcgen05 issue; warps 2-9: the gate
// drain (u back to TMEM, packed, as the out GEMM's A); warps 10-13: the output epilogue; warps 14-17: y = LN(msa) in place. The work goes
// in half-tile steps (heads 0-3 / 4-7): the gate GEMM of the next half runs while this half drains, the out GEMM takes u in two K halves.
namespace gow {
constexpr int BM = 128;
constexpr int OT = BM * HC;                     // o tile [8 heads][128 rows][C]
constexpr int YT = BM * D;                      // msa tile [KB][128][64] -> y in place (the gate GEMM's A)
constexpr int STAGE = OT + YT;
constexpr int NST = WS ? 1 : 2;
constexpr int WGT = HC * D;                     // Wg [KB][HC][64]
constexpr int KO = HC / 64 > 0 ? HC / 64 : 1;   // k-blocks of Wo
constexpr int WOT = D * KO * 64;                // Wo [KO k-blocks][D][64]
// WS (d_msa 256): Wg / Wo stream through WRS slots of 32 KiB in the order the MMA warp takes them -- a gate half's Wg rows are two
// slots ([2 k-blocks][HC / 2][64] each), an out half's Wo two ([D][64] each)
constexpr int WRS = 3, SLOT = 128 * 128;
constexpr int WBUF = WS ? WRS * SLOT : WGT + WOT;
constexpr int THREADS = 576;
constexpr int LNB_ = WS ? 0 : 2 * D * 4;          // gamma / beta in shared memory (WS: the LayerNorm reads them through L1: no room)
constexpr int SMEM = 1024 + (NST * STAGE + WBUF) * 2 + LNB_ + 512;
constexpr int GREG = WS ? 1 : 2, OB = WS ? 1 : 2;               // TMEM gate-half regions, out buffers
constexpr int COL_G = 0, COL_U = GREG * (HC / 2), COL_OUT = COL_U + HC / 2;   // gate [GREG][HC / 2] fp32 | u [HC / 2] packed bf16 | out [OB][D]
constexpr int TCOLS = 512;
constexpr uint32_t ID_G = idesc_bf16(128, HC / 2, 0, 0);
constexpr uint32_t ID_OUT = idesc_bf16(128, D, 0, 0);
static_assert(COL_OUT + OB * D <= TCOLS, "TMEM: 512 columns");
static_assert(!WS || (HC / 2 * 64 * 2 == SLOT && D * 64 == SLOT), "WS slots: two Wg k-blocks of a gate half, one Wo k-block");
static_assert(HC % 64 == 0, "Wo k-blocks");
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace gow

template <typename LT_>
__global__ void __launch_bounds__(gow::THREADS, 1) gate_out_w_sm100(
    int N, int ntiles, float eps, const LT_* __restrict__ LNW, const LT_* __restrict__ LNB,
    const __grid_constant__ CUtensorMap omap,     // o head-major [H*N][S*C], box (C, 128)
    const __grid_constant__ CUtensorMap ymap,     // msa [M][D], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap gmap,     // Wg [HC][D], box (64, HC), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,     // Wo [D][HC], box (64, D), 128B swizzle
    const __nv_bfloat16* __restrict__ MSA, __nv_bfloat16* __restrict__ OUT,
    const __nv_bfloat16* __restrict__ DMASK, float dscale) {
  using namespace gow;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sWg = sStage + NST * STAGE;   // WS: the ring
  __nv_bfloat16* sWo = sWg + (WS ? 0 : WGT);
  float* sLN = reinterpret_cast<float*>(sWg + WBUF);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLN + LNB_ / 4);
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
  uint64_t* wrf = yrdy + NST;        // [WRS] (WS) a weight slot landed
  uint64_t* wre = wrf + WRS;         // [WRS] (WS) its GEMMs retired
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wre + WRS);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(full + k, 1); bar_init(empty + k, 9); bar_init(yrdy + k, 4); }
    for (int k = 0; k < 2; ++k) { bar_init(gf + k, 1); bar_init(ge + k, 8); bar_init(uf + k, 8); bar_init(ue + k, 1);
                                  bar_init(outf + k, 1); bar_init(oute + k, 4); }
    bar_init(wf, 1);
    for (int k = 0; k < WRS; ++k) { bar_init(wrf + k, 1); bar_init(wre + k, 1); }
    bar_init_fence();
  }
  if (!WS && tid < D) { sLN[tid] = to_f(LNW[tid]); sLN[D + tid] = to_f(LNB[tid]); }
  if (warp == 1) tmem_alloc(tslot, TCOLS);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int nt = (int)blockIdx.x < ntiles ? (ntiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
  auto stage = [&](int lt) { return sStage + (lt % NST) * STAGE; };

  if (warp == 0) {
    if (lane == 0) {
      auto load_stage = [&](int lt) {
        const int m0 = ((int)blockIdx.x + lt * (int)gridDim.x) * BM, st = lt % NST;
        if (lt >= NST) wait(empty + st, ((lt / NST) - 1) & 1);
        __nv_bfloat16* p = stage(lt);
        expect_tx(full + st, STAGE * 2);
        const int s = m0 / N, i0 = m0 % N;
        for (int h = 0; h < H; ++h) load_2d(&omap, p + h * BM * C, full + st, s * C, h * N + i0);
        for (int kb = 0; kb < KB; ++kb) load_2d(&ymap, p + OT + kb * BM * 64, full + st, kb * 64, m0);
      };
      if constexpr (!WS) {
        expect_tx(wf, (WGT + WOT) * 2);
        for (int kb = 0; kb < KB; ++kb) load_2d(&gmap, sWg + kb * HC * 64, wf, kb * 64, 0);
        for (int kb = 0; kb < KO; ++kb) load_2d(&wmap, sWo + kb * D * 64, wf, kb * 64, 0);
        for (int lt = 0; lt < nt; ++lt) load_stage(lt);
      } else {                                   // the MMA warp's order: G(0), then G(k + 1), O(k); a tile's stage before its first gate
        int u = 0;
        auto unit = [&](int k, bool g) {
          const int hf = k & 1;
          for (int j = 0; j < 2; ++j, ++u) {
            const int sl = u % WRS;
            if (u >= WRS) wait(wre + sl, ((u / WRS) - 1) & 1);
            expect_tx(wrf + sl, SLOT * 2);
            if (g) for (int k2 = 0; k2 < 2; ++k2) load_2d(&gmap, sWg + sl * SLOT + k2 * (HC / 2) * 64, wrf + sl, (j * 2 + k2) * 64, hf * (HC / 2));
            else load_2d(&wmap, sWg + sl * SLOT, wrf + sl, (hf * 2 + j) * 64, 0);
          }
        };
        const int K = 2 * nt;
        if (K > 0) { load_stage(0); unit(0, true); }
        for (int k = 0; k < K; ++k) {
          if (k + 1 < K) { if (((k + 1) & 1) == 0) load_stage((k + 1) >> 1); unit(k + 1, true); }
          unit(k, false);
        }
      }
    }
  } else if (warp == 1) {
    if constexpr (!WS) wait(wf, 0);
    int u = 0;                                   // WS: weight slot uses, in the producer's order
    auto gate = [&](int k) {                     // gate half hf of tile lt: g[:, HC/2 hf + ...] = y . Wg[HC/2 hf ...]^T
      const int lt = k >> 1, hf = k & 1, st = lt % NST;
      if (hf == 0) wait(yrdy + st, (lt / NST) & 1);
      if (GREG == 2 && k >= 2) wait(ge + hf, ((k >> 1) - 1) & 1);
      if (GREG == 1 && k >= 1) wait(ge + ((k - 1) & 1), ((k - 1) >> 1) & 1);   // one gate region: the previous half drained
      tc_fence_after();
      const __nv_bfloat16* y = stage(lt) + OT;
      const uint32_t gcol = tmem + COL_G + (GREG == 2 ? hf * (HC / 2) : 0);
      if constexpr (!WS) {
        if (elect_one()) {
#pragma unroll
          for (int kb = 0; kb < KB; ++kb)
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              mma_ss(gcol, desc_k128(y + kb * BM * 64 + ks * 16), desc_k128(sWg + kb * HC * 64 + hf * (HC / 2) * 64 + ks * 16), ID_G, (kb | ks) ? 1u : 0u);
        }
      } else {
        for (int j = 0; j < 2; ++j, ++u) {
          const int sl = u % WRS;
          wait(wrf + sl, (u / WRS) & 1);
          tc_fence_after();
          if (elect_one()) {
#pragma unroll
            for (int k2 = 0; k2 < 2; ++k2)
#pragma unroll
              for (int ks = 0; ks < 4; ++ks) {
                const int kb = j * 2 + k2;
                mma_ss(gcol, desc_k128(y + kb * BM * 64 + ks * 16), desc_k128(sWg + sl * SLOT + k2 * (HC / 2) * 64 + ks * 16), ID_G, (kb | ks) ? 1u : 0u);
              }
            mma_commit(wre + sl);
          }
          __syncwarp();
        }
      }
      if (elect_one()) {
        mma_commit(gf + hf);
        if (hf == 1) mma_commit(empty + st);
      }
      __syncwarp();
    };
    auto outg = [&](int k) {                     // out += u[:, half hf] . Wo[:, half hf]^T  (A from TMEM)
      const int lt = k >> 1, hf = k & 1, ob = lt % OB;
      wait(uf + hf, (k >> 1) & 1);
      if (hf == 0 && lt >= OB) wait(oute + ob, ((lt / OB) - 1) & 1);
      tc_fence_after();
      if constexpr (!WS) {
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < HC / 32; ++ks) {
            const int kk = hf * (HC / 2) + ks * 16;
            mma_ts(tmem + COL_OUT + ob * D, tmem + COL_U + hf * (HC / 4) + ks * 8, desc_k128(sWo + (kk >> 6) * D * 64 + ((kk & 63) >> 4) * 16),
                   ID_OUT, (hf | ks) ? 1u : 0u);
          }
        }
      } else {
        for (int j = 0; j < 2; ++j, ++u) {       // the half's two Wo k-blocks [D][64]
          const int sl = u % WRS;
          wait(wrf + sl, (u / WRS) & 1);
          tc_fence_after();
          if (elect_one()) {
#pragma unroll
            for (int k4 = 0; k4 < 4; ++k4) {
              const int ks = j * 4 + k4;
              mma_ts(tmem + COL_OUT + ob * D, tmem + COL_U + hf * (HC / 4) + ks * 8, desc_k128(sWg + sl * SLOT + k4 * 16), ID_OUT, (hf | ks) ? 1u : 0u);
            }
            mma_commit(wre + sl);
          }
          __syncwarp();
        }
      }
      if (elect_one()) {
        mma_commit(ue + hf);
        if (hf == 1) mma_commit(outf + ob);
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
    const int q = warp & 3, pp = (warp - 2) >> 2, r = q * 32 + lane;   // TMEM lane = row; pp: which two heads of the half
    for (int k = 0; k < 2 * nt; ++k) {
      const int lt = k >> 1, hf = k & 1, st = lt % NST;
      wait(gf + hf, (k >> 1) & 1);
      wait(full + st, (lt / NST) & 1);
      if (k >= 2) wait(ue + hf, ((k >> 1) - 1) & 1);
      tc_fence_after();
      uint32_t up[C];                            // two heads x C / 2 packed words
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int head = hf * 4 + pp * 2 + hh;
        float g[C];
        tmem_ldn<C>(tmem_at(tmem + COL_G + (GREG == 2 ? hf * (HC / 2) : 0), q * 32, (pp * 2 + hh) * C), g);
        tmem_wait_ld();
        const __nv_bfloat16* ob = stage(lt) + (head * BM + r) * C;
#pragma unroll
        for (int e = 0; e < C / 8; ++e) {
          const uint4 u4 = *reinterpret_cast<const uint4*>(ob + (vchunk(r, e) << 3));
          const uint32_t ow[4] = {u4.x, u4.y, u4.z, u4.w};
#pragma unroll
          for (int w2 = 0; w2 < 4; ++w2) {
            const int c = e * 8 + w2 * 2;
            up[hh * (C / 2) + e * 4 + w2] = gate_mul_bf16x2(g[c], g[c + 1], ow[w2]);
          }
        }
      }
#pragma unroll
      for (int c8 = 0; c8 < C / 8; ++c8) tmem_st8(tmem_at(tmem + COL_U, q * 32, hf * (HC / 4) + pp * C + c8 * 8), up + c8 * 8);
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) { arrive(ge + hf); arrive(uf + hf); if (hf == 1) arrive(empty + st); }
    }
  } else if (warp >= 14) {
    const int r = (warp - 14) * 32 + lane;
    for (int lt = 0; lt < nt; ++lt) {
      const int st = lt % NST;
      wait(full + st, (lt / NST) & 1);
      if constexpr (WS) ln_row(stage(lt) + OT, r, LnGlobal<LT_>{LNW, LNB}, eps);
      else ln_row(stage(lt) + OT, r, LnShared{sLN}, eps);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(yrdy + st);
    }
  } else {
    // ---- output: bf16(update), dropout, + residual, straight to global, 32 channels at a time ----
    const int q = warp & 3, r = q * 32 + lane;
    for (int lt = 0; lt < nt; ++lt) {
      const int row = ((int)blockIdx.x + lt * (int)gridDim.x) * BM + r, b = lt % OB;
      wait(outf + b, (lt / OB) & 1);
      tc_fence_after();
      const uint4* rp = reinterpret_cast<const uint4*>(MSA + (size_t)row * D);
      uint4* op = reinterpret_cast<uint4*>(OUT + (size_t)row * D);
      const __nv_bfloat16* dr = DMASK != nullptr ? DMASK + (size_t)(row % N) * D : nullptr;
#pragma unroll 1
      for (int c0 = 0; c0 < D; c0 += 32) {
        float v[32];
        tmem_ld32(tmem_at(tmem + COL_OUT + b * D, q * 32, c0), v);
        tmem_wait_ld();
        if (c0 + 32 == D) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(oute + b); }
#pragma unroll
        for (int c8 = 0; c8 < 4; ++c8) {
          uint4 rv = __ldg(rp + c0 / 8 + c8);
          const uint4 dmv = dr != nullptr ? *reinterpret_cast<const uint4*>(dr + c0 + c8 * 8) : make_uint4(0, 0, 0, 0);
          uint32_t* rw = reinterpret_cast<uint32_t*>(&rv);
          const uint32_t* dw = reinterpret_cast<const uint32_t*>(&dmv);
#pragma unroll
          for (int k2 = 0; k2 < 4; ++k2) {
            // the stock module rounds the update to bf16 before the (dropout and the) residual add
            const float2 uu = bf2f(pack2(v[c8 * 8 + 2 * k2], v[c8 * 8 + 2 * k2 + 1]));
            float u0 = uu.x, u1 = uu.y;
            if (dr != nullptr) {
              const float2 m2 = bf2f(dw[k2]);
              u0 = __bfloat162float(__float2bfloat16(u0 * (m2.x * dscale))); u1 = __bfloat162float(__float2bfloat16(u1 * (m2.y * dscale)));
            }
            const float2 res = bf2f(rw[k2]);
            rw[k2] = pack2(res.x + u0, res.y + u1);
          }
          op[c0 / 8 + c8] = rv;
        }
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, TCOLS);
}
#if PW_BWD
// ---------------------------------------------------------------------------------------------
// Training at these widths: the backward's two MSA-side passes (the H100 wide path's boundaries, gate_bwd_w / dm_bwd_w, on tcgen05);
// the contractions dv = w^T . do, dw = do . v^T and the weight gradients dWo = dout'^T go, [dWg; dWv] = [dgp | dv]^T y are cuBLAS.
//
// glue_w.  Per 128-row tile of the (s, i) rows (128 tokens of one MSA row, as gate_out_w): dout' = dres . keep / (1 - p) in place,
// G = y Wg^T and DU = dout' Wo (tensor core, TMEM), then a thread a row: g = sigmoid(G), do = DU g, dgp = DU o g (1 - g), go = g o.
// do is written over o (head-major, as o arrived), dgp over y and go over dout' (natural [rows][HC], 64-channel blocks), and all
// three leave by TMA: do into [H, N, S*C], dgp into the first HC columns of dgv [S*N, 2 HC] (dvb_w puts dv next to it), go into
// [S*N, HC]; dout' too when the dropout is on (dWo's left operand).
//   warp 0: o | y | dres tiles (NST-stage ring) and the resident Wg, Wo;  warp 1: tcgen05 issue;  warps 2-9: the prep (dropout) and
//   the drain, a row and four heads a thread.
namespace glw {
constexpr int BM = 128;
constexpr int OT = BM * HC;                     // o -> do [H][128][C]
constexpr int YT = BM * D;                      // y [KB][128][64] -> dgp [HC / 64][128][64]
constexpr int RT = BM * D;                      // dres -> dout' -> go
constexpr int STAGE = OT + YT + RT;
constexpr int WGT = HC * D;                     // Wg [KB][HC][64] (K-major B of G)
constexpr int KO = HC / 64;
constexpr int WOT = D * HC;                     // Wo [KO][D][64] (MN-major B of DU)
constexpr int THREADS = 320;
constexpr int FIX = 1024 + (WGT + WOT) * 2 + 1024;
constexpr int NST = FIX + 2 * STAGE * 2 <= 232448 ? 2 : 1;
constexpr int SMEM = FIX + NST * STAGE * 2;
constexpr int NB = 4 * HC <= 512 ? 2 : 1;       // TMEM (G | DU) buffers
constexpr int TCOLS = NB * 2 * HC <= 256 ? 256 : 512;
constexpr uint32_t ID_G = idesc_bf16(128, HC, 0, 0);
constexpr uint32_t ID_DU = idesc_bf16(128, HC, 0, 1);
static_assert(HC % 64 == 0 && HC <= D && NB * 2 * HC <= 512, "glue_w widths");
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace glw

__device__ __forceinline__ float sigmoid_t(float x) {
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return fmaf(0.5f, t, 0.5f);
}

__global__ void __launch_bounds__(glw::THREADS, 1) glue_w_sm100(
    int N, int ntiles,
    const __grid_constant__ CUtensorMap omap,     // o head-major [H*N][S*C], box (C, 128)
    const __grid_constant__ CUtensorMap ymap,     // y [M][D], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap rmap,     // dres [M][D], same
    const __grid_constant__ CUtensorMap gmap,     // Wg [HC][D], box (64, HC)
    const __grid_constant__ CUtensorMap wmap,     // Wo [D][HC], box (64, D)
    const __grid_constant__ CUtensorMap domap,    // do, as o
    const __grid_constant__ CUtensorMap dgmap,    // dgv [M][2 HC], box (64, 128)
    const __grid_constant__ CUtensorMap gomap,    // go [M][HC], box (64, 128)
    const __grid_constant__ CUtensorMap dpmap,    // dout' [M][D], box (64, 128) (the dropout only)
    const __nv_bfloat16* __restrict__ DMASK, float dscale) {
  using namespace glw;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sWg = sStage + NST * STAGE;
  __nv_bfloat16* sWo = sWg + WGT;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sWo + WOT);
  uint64_t* full = bars;             // [NST]
  uint64_t* empty = full + NST;      // [NST] the tile's stores have read the stage
  uint64_t* rdy = empty + NST;       // [NST] count 8: dout' in place
  uint64_t* accf = rdy + NST;        // [NB]
  uint64_t* acce = accf + NB;        // [NB] count 8
  uint64_t* wf = acce + NB;          // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wf + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(full + k, 1); bar_init(empty + k, 1); bar_init(rdy + k, 8); }
    for (int k = 0; k < NB; ++k) { bar_init(accf + k, 1); bar_init(acce + k, 8); }
    bar_init(wf, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, TCOLS);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int nt = (int)blockIdx.x < ntiles ? (ntiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
  auto stage = [&](int lt) { return sStage + (lt % NST) * STAGE; };

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wf, (WGT + WOT) * 2);
      for (int kb = 0; kb < KB; ++kb) load_2d(&gmap, sWg + kb * HC * 64, wf, kb * 64, 0);
      for (int ko = 0; ko < KO; ++ko) load_2d(&wmap, sWo + ko * D * 64, wf, ko * 64, 0);
      for (int lt = 0; lt < nt; ++lt) {
        const int m0 = ((int)blockIdx.x + lt * (int)gridDim.x) * BM, st = lt % NST;
        if (lt >= NST) wait(empty + st, ((lt / NST) - 1) & 1);
        __nv_bfloat16* p = stage(lt);
        expect_tx(full + st, STAGE * 2);
        const int s = m0 / N, i0 = m0 % N;
        for (int h = 0; h < H; ++h) load_2d(&omap, p + h * BM * C, full + st, s * C, h * N + i0);
        for (int kb = 0; kb < KB; ++kb) {
          load_2d(&ymap, p + OT + kb * BM * 64, full + st, kb * 64, m0);
          load_2d(&rmap, p + OT + YT + kb * BM * 64, full + st, kb * 64, m0);
        }
      }
    }
  } else if (warp == 1) {
    wait(wf, 0);
    for (int lt = 0; lt < nt; ++lt) {
      const int st = lt % NST, b = lt % NB;
      wait(rdy + st, (lt / NST) & 1);
      if (lt >= NB) wait(acce + b, ((lt / NB) - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* y = stage(lt) + OT;
        const __nv_bfloat16* r = y + YT;
        const uint32_t acc = tmem + b * 2 * HC;
#pragma unroll
        for (int kb = 0; kb < KB; ++kb)
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            mma_ss(acc, desc_k128(y + kb * BM * 64 + ks * 16), desc_k128(sWg + kb * HC * 64 + ks * 16), ID_G, (kb | ks) ? 1u : 0u);
#pragma unroll
        for (int k = 0; k < D / 16; ++k)              // DU = dout' . Wo: B = Wo [k = d rows][64 hc] blocks (MN-major)
          mma_ss(acc + HC, desc_k128(r + (k >> 2) * BM * 64 + (k & 3) * 16), desc_mn128(sWo + k * 16 * 64, D * 64 * 2), ID_DU, k ? 1u : 0u);
        mma_commit(accf + b);
      }
      __syncwarp();
    }
  } else {
    const int q = warp & 3, pp = (warp - 2) >> 2, r = q * 32 + lane;   // TMEM lane = row; pp: heads 4 pp .. 4 pp + 3, channels of half pp
    const bool leader = (warp == 2 && lane == 0);
    for (int lt = 0; lt < nt; ++lt) {
      const int st = lt % NST, b = lt % NB, m0 = ((int)blockIdx.x + lt * (int)gridDim.x) * BM, s = m0 / N, i0 = m0 % N;
      __nv_bfloat16* O = stage(lt);
      __nv_bfloat16* Y = O + OT;
      __nv_bfloat16* R = Y + YT;
      wait(full + st, (lt / NST) & 1);
      if (DMASK != nullptr) {                    // dout' = dres . keep / (1 - p) in place: row r, channel half pp
        const __nv_bfloat16* dr = DMASK + (size_t)(i0 + r) * D;
#pragma unroll
        for (int e = 0; e < D / 16; ++e) {
          const int g8 = pp * (D / 16) + e, blk = g8 >> 3, c8 = g8 & 7;
          __nv_bfloat16* p = R + blk * BM * 64 + r * 64 + ((c8 ^ (r & 7)) << 3);
          uint4 v = *reinterpret_cast<uint4*>(p);
          const uint4 kp = __ldg(reinterpret_cast<const uint4*>(dr + g8 * 8));
          const uint32_t* vw = reinterpret_cast<const uint32_t*>(&v);
          const uint32_t* kw = reinterpret_cast<const uint32_t*>(&kp);
          uint4 o;
          uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float2 a = bf2f(vw[k]), m = bf2f(kw[k]);
            ow[k] = pack2((a.x * m.x) * dscale, (a.y * m.y) * dscale);
          }
          *reinterpret_cast<uint4*>(p) = o;
        }
        fence_proxy_async();
      }
      __syncwarp();
      if (lane == 0) arrive(rdy + st);
      if (DMASK != nullptr) {                    // dout' leaves for dWo's GEMM before go overwrites it
        named_sync(1, 256);
        if (leader) {
          for (int kb = 0; kb < KB; ++kb) store_2d(&dpmap, R + kb * BM * 64, kb * 64, m0);
          bulk_commit();
          bulk_wait_read<0>();
        }
        named_sync(1, 256);
      }
      wait(accf + b, (lt / NB) & 1);
      tc_fence_after();
#pragma unroll 1
      for (int hh = 0; hh < 4; ++hh) {
        const int h = pp * 4 + hh;
        float gt[C], du[C];
        tmem_ldn<C>(tmem_at(tmem + b * 2 * HC, q * 32, h * C), gt);
        tmem_ldn<C>(tmem_at(tmem + b * 2 * HC + HC, q * 32, h * C), du);
        tmem_wait_ld();
        if (hh == 3) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + b); }
        __nv_bfloat16* orow = O + (h * BM + r) * C;
#pragma unroll
        for (int e = 0; e < C / 8; ++e) {
          uint4* op = reinterpret_cast<uint4*>(orow + (vchunk(r, e) << 3));
          const uint4 ov = *op;
          const uint32_t* ow = reinterpret_cast<const uint32_t*>(&ov);
          uint4 vdo, vdg, vgo;
          uint32_t* wdo = reinterpret_cast<uint32_t*>(&vdo);
          uint32_t* wdg = reinterpret_cast<uint32_t*>(&vdg);
          uint32_t* wgo = reinterpret_cast<uint32_t*>(&vgo);
#pragma unroll
          for (int k = 0; k < 4; ++k) {                 // packed pairs: do = du g, dgp = du o g (1 - g), go = g o (pwa_glue2's math)
            const int c = e * 8 + 2 * k;
            const float2 o2 = bf2f(ow[k]);
            const float2 g2 = make_float2(sigmoid_t(gt[c]), sigmoid_t(gt[c + 1]));
            const float2 d2 = make_float2(du[c], du[c + 1]);
            const float2 do2 = mul2(d2, g2);
            const float2 gg = fma2(g2, make_float2(-g2.x, -g2.y), g2);
            const float2 dg2 = mul2(mul2(d2, o2), gg);
            const float2 go2 = mul2(g2, o2);
            wdo[k] = pack2(do2.x, do2.y); wdg[k] = pack2(dg2.x, dg2.y); wgo[k] = pack2(go2.x, go2.y);
          }
          *op = vdo;
          const int col = h * C + e * 8, blk = col >> 6, c8 = (col & 63) >> 3;
          const int off = blk * BM * 64 + r * 64 + ((c8 ^ (r & 7)) << 3);
          *reinterpret_cast<uint4*>(Y + off) = vdg;
          *reinterpret_cast<uint4*>(R + off) = vgo;
        }
      }
      fence_proxy_async();
      named_sync(1, 256);
      if (leader) {
        for (int h = 0; h < H; ++h) store_2d(&domap, O + h * BM * C, s * C, h * N + i0);
        for (int kb = 0; kb < KO; ++kb) { store_2d(&dgmap, Y + kb * BM * 64, kb * 64, m0); store_2d(&gomap, R + kb * BM * 64, kb * 64, m0); }
        bulk_commit();
        bulk_wait_read<0>();
        arrive(empty + st);
      }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, TCOLS);
}

// dvb_w.  Per tile of one token n x 128 MSA rows s: dy = [dgp | dv] . [Wg; Wv] (tensor core: dgp natural, 64-channel blocks; dv
// head-major as the cuBLAS contraction left it, its [128 s][C] head blocks read directly as K-major A), then a thread a row:
// dm = LayerNorm_bwd(dy; x, gamma) + dres; the dv tile also leaves in natural layout into dgv[..., HC:] for the weight GEMM.
// (dgamma, dbeta) are fp32 column sums: a warp reduce-scatter per 32-channel chunk leaves lane l the chunk's column l, accumulated per
// lane over the CTA's tiles and combined across the four warps in a fixed order. (The tensor-core sum of pwa_sm100.cu's dv_bwd rounds
// every dy . xhat to bf16 first: an error as large as the module's own bf16 dy, which put dgamma at 0.76-1.18x of the module's error.)
//   warp 0: dgp | dv | x | dres tiles and the resident [Wg; Wv];  warp 1: tcgen05;  warps 2-5: the drain (three passes over the row).
namespace dbw {
constexpr int BS = 128;
constexpr int DGT = BS * HC, DVT = BS * HC, XT = BS * D, RT = BS * D;
constexpr int STAGE = DGT + DVT + XT + RT;      // dgp | dv | x | dres (-> dm in place)
constexpr int WT = 2 * HC * D;                  // [Wg; Wv] as [D / 64][2 HC][64] (MN-major B)
constexpr int THREADS = 192;
constexpr int FIX = 1024 + WT * 2 + D * 4 + 512;
constexpr int NST = FIX + 2 * STAGE * 2 <= 232448 ? 2 : 1;
constexpr int SMEM = FIX + NST * STAGE * 2;
constexpr int NB = 2;                           // TMEM dy buffers
constexpr uint32_t ID_DY = idesc_bf16(128, D, 0, 1);
static_assert(NB * D <= 512, "TMEM");
static_assert(SMEM <= 232448 && STAGE * 2 >= 4 * 2 * D * 4, "one CTA per SM; the final reduction reuses a stage");
}  // namespace dbw

// the K-major A descriptor of the dv tile's k-step kk (16 value channels): [128 s][C] head blocks as TMA landed them
__device__ __forceinline__ uint64_t dv_desc(const __nv_bfloat16* dv, int kk) {
  if constexpr (C == 8) return sdesc(sa(dv + 2 * kk * 128 * 8), 128 * 8 * 2, 128, 0);       // no swizzle: core matrices 8 rows x 16 B; heads 2kk, 2kk+1 along K
  else if constexpr (C == 16) return sdesc(sa(dv + kk * 128 * 16), 16, 256, 6);             // 32B swizzle, one head a step
  else return desc_k64(dv + (kk >> 1) * 128 * 32 + (kk & 1) * 16);                         // 64B swizzle, two steps a head
}

// a warp's 32 rows x 32 columns (v: this lane's row, as 16 pairs) -> the column sum of column `lane`, fixed order (31 shuffles)
__device__ __forceinline__ float colsum32(float2 (&v)[16], int lane) {
  float* f = reinterpret_cast<float*>(v);
#pragma unroll
  for (int w = 16; w >= 1; w >>= 1) {
    const bool up = lane & w;
#pragma unroll
    for (int k = 0; k < w; ++k) {
      const float send = up ? f[k] : f[k + w], keep = up ? f[k + w] : f[k];
      f[k] = keep + __shfl_xor_sync(0xffffffffu, send, w);
    }
  }
  return f[0];
}

template <typename LT_>
__global__ void __launch_bounds__(dbw::THREADS, 1) dvb_w_sm100(
    int N, int S, int ntiles, float eps,
    const __grid_constant__ CUtensorMap dgmap,    // dgv [S][N][2 HC] as (2 HC, N, S), box (64, 1, 128), 128B swizzle (the dgp half)
    const __grid_constant__ CUtensorMap dvmap,    // dv head-major as (C, S, H*N), box (C, 128, 1)
    const __grid_constant__ CUtensorMap dvnmap,   // dgv as (2 HC, N, S), box (C, 1, 128) (dv out, natural)
    const __grid_constant__ CUtensorMap xmap,     // x [S][N][D] as (D, N, S), box (64, 1, 128), 128B swizzle
    const __grid_constant__ CUtensorMap rmap,     // dres, same
    const __grid_constant__ CUtensorMap dmmap,    // dm, same
    const __grid_constant__ CUtensorMap wgmap,    // Wg [HC][D], box (64, HC)
    const __grid_constant__ CUtensorMap wvmap,    // Wv, same
    const LT_* __restrict__ LNW,
    float* __restrict__ DLN) {                    // [grid][2 D]: (dgamma, dbeta) partials
  using namespace dbw;
  constexpr int NC = D / 32;                     // 32-channel chunks of a row
  const int NSB = S / BS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sW = sStage + NST * STAGE;
  float* sLN = reinterpret_cast<float*>(sW + WT);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLN + D);
  uint64_t* full = bars;             // [NST]
  uint64_t* empty = full + NST;      // [NST] the dm / dv stores read it
  uint64_t* accf = empty + NST;      // [NB]
  uint64_t* acce = accf + NB;        // [NB] count 4
  uint64_t* wf = acce + NB;          // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wf + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(full + k, 1); bar_init(empty + k, 1); }
    for (int k = 0; k < NB; ++k) { bar_init(accf + k, 1); bar_init(acce + k, 4); }
    bar_init(wf, 1);
    bar_init_fence();
  }
  if (tid < D) sLN[tid] = to_f(LNW[tid]);
  if (warp == 1) tmem_alloc(tslot, NB * D <= 128 ? 128 : 256);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  auto stage = [&](int lt) { return sStage + (lt % NST) * STAGE; };

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wf, WT * 2);
      for (int nb = 0; nb < KB; ++nb) {
        load_2d(&wgmap, sW + nb * 2 * HC * 64, wf, nb * 64, 0);
        load_2d(&wvmap, sW + nb * 2 * HC * 64 + HC * 64, wf, nb * 64, 0);
      }
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int n = t / NSB, s0 = (t % NSB) * BS, st = lt % NST;
        if (lt >= NST) wait(empty + st, ((lt / NST) - 1) & 1);
        __nv_bfloat16* p = stage(lt);
        expect_tx(full + st, STAGE * 2);
        for (int kb = 0; kb < HC / 64; ++kb) load_3d(&dgmap, p + kb * BS * 64, full + st, kb * 64, n, s0);
        for (int h = 0; h < H; ++h) load_3d(&dvmap, p + DGT + h * BS * C, full + st, 0, s0, h * N + n);
        for (int kb = 0; kb < KB; ++kb) {
          load_3d(&xmap, p + DGT + DVT + kb * BS * 64, full + st, kb * 64, n, s0);
          load_3d(&rmap, p + DGT + DVT + XT + kb * BS * 64, full + st, kb * 64, n, s0);
        }
      }
    }
  } else if (warp == 1) {
    wait(wf, 0);
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int st = lt % NST, b = lt % NB;
      wait(full + st, (lt / NST) & 1);
      if (lt >= NB) wait(acce + b, ((lt / NB) - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* p = stage(lt);
#pragma unroll
        for (int kk = 0; kk < HC / 16; ++kk)           // dy = dgp . Wg ...
          mma_ss(tmem + b * D, desc_k128(p + (kk >> 2) * BS * 64 + (kk & 3) * 16), desc_mn128(sW + kk * 16 * 64, 2 * HC * 64 * 2), ID_DY, kk ? 1u : 0u);
#pragma unroll
        for (int kk = 0; kk < HC / 16; ++kk)           // ... + dv . Wv
          mma_ss(tmem + b * D, dv_desc(p + DGT, kk), desc_mn128(sW + (HC + kk * 16) * 64, 2 * HC * 64 * 2), ID_DY, 1u);
        mma_commit(accf + b);
      }
      __syncwarp();
    }
  } else {
    const int q = warp & 3, r = q * 32 + lane;   // TMEM lane = row s
    const bool leader = (warp == 2 && lane == 0);
    float accg[NC], accb[NC];                    // this lane's columns 32 k + lane: dgamma, dbeta over the CTA's tiles (its warp's rows)
#pragma unroll
    for (int k = 0; k < NC; ++k) accg[k] = accb[k] = 0.f;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int n = t / NSB, s0 = (t % NSB) * BS, st = lt % NST, b = lt % NB;
      __nv_bfloat16* p = stage(lt);
      const __nv_bfloat16* X = p + DGT + DVT;
      __nv_bfloat16* Rr = p + DGT + DVT + XT;
      wait(full + st, (lt / NST) & 1);
      if (leader) {                              // dv out in natural layout, straight from the head blocks as they landed
        for (int h = 0; h < H; ++h) store_3d(&dvnmap, p + DGT + h * BS * C, HC + h * C, n, s0);
        bulk_commit();
      }
      // ---- pass 1: the LayerNorm statistics of x (as ln_vg_w: mean, then the centred variance) ----
      float2 s2 = make_float2(0.f, 0.f);
#pragma unroll 1
      for (int kb = 0; kb < KB; ++kb)
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          const uint4 u = *reinterpret_cast<const uint4*>(X + kb * BS * 64 + r * 64 + ((c8 ^ (r & 7)) << 3));
          s2 = add2(s2, add2(add2(bf2f(u.x), bf2f(u.y)), add2(bf2f(u.z), bf2f(u.w))));
        }
      const float mean = (s2.x + s2.y) * (1.f / D);
      const float2 nm = make_float2(-mean, -mean);
      float2 v2 = make_float2(0.f, 0.f);
#pragma unroll 1
      for (int kb = 0; kb < KB; ++kb)
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          const uint4 u = *reinterpret_cast<const uint4*>(X + kb * BS * 64 + r * 64 + ((c8 ^ (r & 7)) << 3));
          const float2 x0 = add2(bf2f(u.x), nm), x1 = add2(bf2f(u.y), nm), x2 = add2(bf2f(u.z), nm), x3 = add2(bf2f(u.w), nm);
          v2 = fma2(x0, x0, fma2(x1, x1, fma2(x2, x2, fma2(x3, x3, v2))));
        }
      const float rstd = 1.f / sqrtf((v2.x + v2.y) * (1.f / D) + eps);
      const float2 rs = make_float2(rstd, rstd);
      wait(accf + b, (lt / NB) & 1);
      tc_fence_after();
      // ---- pass 2: the row sums of gamma dy and gamma dy xhat; the column sums of dy xhat and dy (dgamma, dbeta) ----
      float2 sa_ = make_float2(0.f, 0.f), sb_ = sa_;
#pragma unroll
      for (int k = 0; k < NC; ++k) {
        const int c0 = k * 32;
        float2 dy[16], px[16];
        tmem_ld32(tmem_at(tmem + b * D, q * 32, c0), reinterpret_cast<float*>(dy));
        tmem_wait_ld();
        const float2* gam2 = reinterpret_cast<const float2*>(sLN + c0);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int c8 = ((c0 & 63) >> 3) + e;
          const uint4 u = *reinterpret_cast<const uint4*>(X + (c0 >> 6) * BS * 64 + r * 64 + ((c8 ^ (r & 7)) << 3));
          const uint32_t xw[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int kk = e * 4 + j;
            const float2 xh = mul2(add2(bf2f(xw[j]), nm), rs);
            const float2 gd = mul2(gam2[kk], dy[kk]);
            sa_ = add2(sa_, gd); sb_ = fma2(gd, xh, sb_);
            px[kk] = mul2(dy[kk], xh);
          }
        }
        accg[k] += colsum32(px, lane);
        accb[k] += colsum32(dy, lane);
      }
      const float s1 = (sa_.x + sa_.y) * (1.f / D), s2v = (sb_.x + sb_.y) * (1.f / D);
      const float2 ns1 = make_float2(-s1, -s1), ns2 = make_float2(-s2v, -s2v);
      // ---- pass 3: dm = rstd (gamma dy - s1 - xhat s2) + dres, over dres in place ----
#pragma unroll 1
      for (int c0 = 0; c0 < D; c0 += 32) {
        float2 dy[16];
        tmem_ld32(tmem_at(tmem + b * D, q * 32, c0), reinterpret_cast<float*>(dy));
        tmem_wait_ld();
        if (c0 + 32 == D) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + b); }
        const float2* gam2 = reinterpret_cast<const float2*>(sLN + c0);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int c8 = ((c0 & 63) >> 3) + e;
          const int off = (c0 >> 6) * BS * 64 + r * 64 + ((c8 ^ (r & 7)) << 3);
          const uint4 u = *reinterpret_cast<const uint4*>(X + off);
          const uint4 dv4 = *reinterpret_cast<const uint4*>(Rr + off);
          const uint32_t xw[4] = {u.x, u.y, u.z, u.w};
          const uint32_t rw[4] = {dv4.x, dv4.y, dv4.z, dv4.w};
          uint4 o;
          uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int kk = e * 4 + j;
            const float2 xh = mul2(add2(bf2f(xw[j]), nm), rs);
            const float2 gd = mul2(gam2[kk], dy[kk]);
            const float2 d2 = fma2(fma2(xh, ns2, add2(gd, ns1)), rs, bf2f(rw[j]));
            ow[j] = pack2(d2.x, d2.y);
          }
          *reinterpret_cast<uint4*>(Rr + off) = o;
        }
      }
      fence_proxy_async();
      named_sync(1, 128);
      if (leader) {
        for (int kb = 0; kb < KB; ++kb) store_3d(&dmmap, Rr + kb * BS * 64, kb * 64, n, s0);
        bulk_commit();
        bulk_wait_read<0>();
        arrive(empty + st);
      }
    }
    if (leader) bulk_wait<0>();
    named_sync(1, 128);                          // the last dm store has read its stage
    // the CTA's (dgamma, dbeta): the four warps' column sums, added in warp order (the stages are free: every tile is done)
    float* red = reinterpret_cast<float*>(sStage);   // [4 warps][2 D]
#pragma unroll
    for (int k = 0; k < NC; ++k) { red[q * 2 * D + k * 32 + lane] = accg[k]; red[q * 2 * D + D + k * 32 + lane] = accb[k]; }
    named_sync(1, 128);
    for (int c = r; c < 2 * D; c += 128)
      DLN[(size_t)blockIdx.x * 2 * D + c] = ((red[c] + red[2 * D + c]) + red[4 * D + c]) + red[6 * D + c];
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, NB * D <= 128 ? 128 : 256);
}
#endif  // PW_BWD
}  // namespace

static void check_ln(const torch::Tensor& lnw, const torch::Tensor& lnb) {
  TORCH_CHECK(lnw.is_contiguous() && lnb.is_contiguous() && lnw.numel() == D && lnb.numel() == D && lnw.scalar_type() == lnb.scalar_type() &&
              (lnw.scalar_type() == torch::kFloat32 || lnw.scalar_type() == torch::kBFloat16), "LayerNorm weight / bias [d_msa], fp32 or bf16");
}

// m [S, N, D] bf16 (S, N multiples of 128) -> (v head-major [H, N, S*C], y [S, N, D] (empty unless want_y))
std::vector<torch::Tensor> ln_vg_w(torch::Tensor m, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wv, double eps, bool want_y) {
  using namespace lvw;
  TORCH_CHECK(m.is_contiguous() && m.scalar_type() == torch::kBFloat16 && m.dim() == 3 && m.size(2) == D, "m: [S, N, d_msa] bf16");
  const long S = m.size(0), N = m.size(1);
  TORCH_CHECK(S % BS == 0 && N % 128 == 0, "S and N must be multiples of 128");
  TORCH_CHECK(wv.is_contiguous() && wv.scalar_type() == torch::kBFloat16 && wv.size(0) == HC && wv.size(1) == D, "wv: [8 d_hidden, d_msa] bf16");
  check_ln(lnw, lnb);
  auto v = torch::empty({(long)H, N, S * C}, m.options());
  auto y = want_y ? torch::empty_like(m) : torch::empty({0}, m.options());
  CUtensorMap xm = make_map<3>(m.data_ptr(), {(uint64_t)D, (uint64_t)N, (uint64_t)S}, {(uint64_t)D, (uint64_t)(N * D)}, {64, 1, BS},
                               CU_TENSOR_MAP_SWIZZLE_128B, "m");
  CUtensorMap ym = want_y ? make_map<3>(y.data_ptr(), {(uint64_t)D, (uint64_t)N, (uint64_t)S}, {(uint64_t)D, (uint64_t)(N * D)}, {64, 1, BS},
                                        CU_TENSOR_MAP_SWIZZLE_128B, "y") : xm;
  CUtensorMap wm = make_map<2>(wv.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, (uint32_t)HC}, CU_TENSOR_MAP_SWIZZLE_128B, "wv");
  CUtensorMap vm = make_map<3>(v.data_ptr(), {(uint64_t)C, (uint64_t)S, (uint64_t)(H * N)}, {(uint64_t)C, (uint64_t)(S * C)}, {(uint32_t)C, BS, 1},
                               VSW, "v");
  const int ntiles = (int)(N * (S / BS));
  const int grid = std::min(ntiles, num_sms(m.device().index()));
  auto st = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto lt) {
    using LT = decltype(lt);
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(ln_vg_w_sm100<LT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    ln_vg_w_sm100<LT><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, (int)want_y, xm, ym, wm, vm,
                                                   reinterpret_cast<const LT*>(lnw.data_ptr()), reinterpret_cast<const LT*>(lnb.data_ptr()));
  };
  if (lnw.scalar_type() == torch::kFloat32) launch(float{}); else launch(__nv_bfloat16{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {v, y};
}

// o head-major [H, N, S*C] bf16, msa [S, N, D] bf16 -> out = msa + drop(bf16(PWA update)) [S, N, D]
torch::Tensor gate_out_w(torch::Tensor o, torch::Tensor msa, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor wg, torch::Tensor wo,
                         c10::optional<torch::Tensor> dmask, double dscale) {
  using namespace gow;
  TORCH_CHECK(msa.is_contiguous() && msa.scalar_type() == torch::kBFloat16 && msa.dim() == 3 && msa.size(2) == D, "msa: [S, N, d_msa] bf16");
  const long S = msa.size(0), N = msa.size(1);
  TORCH_CHECK(N % BM == 0, "N must be a multiple of 128");
  TORCH_CHECK(o.is_contiguous() && o.scalar_type() == torch::kBFloat16 && o.numel() == (long)H * N * S * C, "o: [H, N, S*C] bf16");
  TORCH_CHECK(wg.is_contiguous() && wg.scalar_type() == torch::kBFloat16 && wg.size(0) == HC && wg.size(1) == D, "wg: [8 d_hidden, d_msa] bf16");
  TORCH_CHECK(wo.is_contiguous() && wo.scalar_type() == torch::kBFloat16 && wo.size(0) == D && wo.size(1) == HC, "wo: [d_msa, 8 d_hidden] bf16");
  check_ln(lnw, lnb);
  const __nv_bfloat16* dm = nullptr;
  if (dmask.has_value() && dmask->defined() && dmask->numel() > 0) {
    TORCH_CHECK(dmask->is_contiguous() && dmask->scalar_type() == torch::kBFloat16 && dmask->numel() == N * D, "dmask: [N, d_msa] bf16");
    dm = reinterpret_cast<const __nv_bfloat16*>(dmask->data_ptr<at::BFloat16>());
  }
  auto out = torch::empty_like(msa);
  const long M = S * N;
  CUtensorMap om = make_map<2>(o.data_ptr(), {(uint64_t)(S * C), (uint64_t)(H * N)}, {(uint64_t)(S * C)}, {(uint32_t)C, (uint32_t)BM}, VSW, "o");
  CUtensorMap ym = make_map<2>(msa.data_ptr(), {(uint64_t)D, (uint64_t)M}, {(uint64_t)D}, {64, (uint32_t)BM}, CU_TENSOR_MAP_SWIZZLE_128B, "msa");
  CUtensorMap gm = make_map<2>(wg.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, (uint32_t)(WS ? HC / 2 : HC)}, CU_TENSOR_MAP_SWIZZLE_128B, "wg");
  CUtensorMap wm = make_map<2>(wo.data_ptr(), {(uint64_t)HC, (uint64_t)D}, {(uint64_t)HC}, {64, (uint32_t)D}, CU_TENSOR_MAP_SWIZZLE_128B, "wo");
  const int ntiles = (int)(M / BM);
  const int grid = std::min(ntiles, num_sms(msa.device().index()));
  auto st = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto lt) {
    using LT = decltype(lt);
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(gate_out_w_sm100<LT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    gate_out_w_sm100<LT><<<grid, THREADS, SMEM, st>>>((int)N, ntiles, (float)eps, reinterpret_cast<const LT*>(lnw.data_ptr()),
                                                      reinterpret_cast<const LT*>(lnb.data_ptr()), om, ym, gm, wm,
                                                      reinterpret_cast<const __nv_bfloat16*>(msa.data_ptr<at::BFloat16>()),
                                                      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), dm, (float)dscale);
  };
  if (lnw.scalar_type() == torch::kFloat32) launch(float{}); else launch(__nv_bfloat16{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

#if PW_BWD

// o head-major [H, N, S*C], y / dres [S, N, D] bf16, wg [HC, D], wo [D, HC] bf16 (S, N multiples of 128)
// -> (do head-major [H, N, S*C], dgv [S, N, 2 HC] with dgp in [..., :HC] (dvb_w fills the rest), go [S, N, HC], dout' (dres itself
// without the dropout))
std::vector<torch::Tensor> glue_w(torch::Tensor o, torch::Tensor y, torch::Tensor dres, torch::Tensor wg, torch::Tensor wo,
                                  c10::optional<torch::Tensor> dmask, double dscale) {
  using namespace glw;
  TORCH_CHECK(y.is_contiguous() && y.scalar_type() == torch::kBFloat16 && y.dim() == 3 && y.size(2) == D, "y: [S, N, d_msa] bf16");
  TORCH_CHECK(dres.is_contiguous() && dres.scalar_type() == torch::kBFloat16 && dres.sizes() == y.sizes(), "dres: as y");
  const long S = y.size(0), N = y.size(1), M = S * N;
  TORCH_CHECK(N % BM == 0 && S % 128 == 0, "S and N must be multiples of 128");
  TORCH_CHECK(o.is_contiguous() && o.scalar_type() == torch::kBFloat16 && o.numel() == (long)H * N * S * C, "o: [H, N, S*C] bf16");
  TORCH_CHECK(wg.is_contiguous() && wg.scalar_type() == torch::kBFloat16 && wg.size(0) == HC && wg.size(1) == D, "wg: [8 d_hidden, d_msa] bf16");
  TORCH_CHECK(wo.is_contiguous() && wo.scalar_type() == torch::kBFloat16 && wo.size(0) == D && wo.size(1) == HC, "wo: [d_msa, 8 d_hidden] bf16");
  const __nv_bfloat16* dm = nullptr;
  if (dmask.has_value() && dmask->defined() && dmask->numel() > 0) {
    TORCH_CHECK(dmask->is_contiguous() && dmask->scalar_type() == torch::kBFloat16 && dmask->numel() == N * D, "dmask: [N, d_msa] bf16");
    dm = reinterpret_cast<const __nv_bfloat16*>(dmask->data_ptr<at::BFloat16>());
  }
  auto dO = torch::empty_like(o);
  auto dgv = torch::empty({S, N, 2L * HC}, y.options());
  auto go = torch::empty({S, N, (long)HC}, y.options());
  auto dout = dm != nullptr ? torch::empty_like(dres) : dres;
  CUtensorMap om = make_map<2>(o.data_ptr(), {(uint64_t)(S * C), (uint64_t)(H * N)}, {(uint64_t)(S * C)}, {(uint32_t)C, (uint32_t)BM}, VSW, "o");
  CUtensorMap dom = make_map<2>(dO.data_ptr(), {(uint64_t)(S * C), (uint64_t)(H * N)}, {(uint64_t)(S * C)}, {(uint32_t)C, (uint32_t)BM}, VSW, "do");
  auto rows = [&](void* p, long width, const char* w) {
    return make_map<2>(p, {(uint64_t)width, (uint64_t)M}, {(uint64_t)width}, {64, (uint32_t)BM}, CU_TENSOR_MAP_SWIZZLE_128B, w);
  };
  CUtensorMap ym = rows(y.data_ptr(), D, "y"), rm = rows(dres.data_ptr(), D, "dres"), dgm = rows(dgv.data_ptr(), 2 * HC, "dgv"),
              gom = rows(go.data_ptr(), HC, "go"), dpm = rows(dout.data_ptr(), D, "dout");
  CUtensorMap gm = make_map<2>(wg.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, (uint32_t)HC}, CU_TENSOR_MAP_SWIZZLE_128B, "wg");
  CUtensorMap wm = make_map<2>(wo.data_ptr(), {(uint64_t)HC, (uint64_t)D}, {(uint64_t)HC}, {64, (uint32_t)D}, CU_TENSOR_MAP_SWIZZLE_128B, "wo");
  const int ntiles = (int)(M / BM);
  const int grid = std::min(ntiles, num_sms(y.device().index()));
  auto st = at::cuda::getCurrentCUDAStream();
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(glue_w_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  glue_w_sm100<<<grid, THREADS, SMEM, st>>>((int)N, ntiles, om, ym, rm, gm, wm, dom, dgm, gom, dpm, dm, (float)dscale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, dgv, go, dout};
}

// dv head-major [H, N, S*C], dgv [S, N, 2 HC] (dgp in [..., :HC]), x / dres [S, N, D], wg / wv [HC, D] bf16, LayerNorm scale [D]
// -> (dm [S, N, D], dgamma, dbeta fp32); dv is written into dgv[..., HC:] (natural layout) for the weight GEMM
std::vector<torch::Tensor> dvb_w(torch::Tensor dvh, torch::Tensor dgv, torch::Tensor x, torch::Tensor dres, torch::Tensor wg, torch::Tensor wv,
                                 torch::Tensor lnw, double eps) {
  using namespace dbw;
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && x.dim() == 3 && x.size(2) == D, "x: [S, N, d_msa] bf16");
  TORCH_CHECK(dres.is_contiguous() && dres.scalar_type() == torch::kBFloat16 && dres.sizes() == x.sizes(), "dres: as x");
  const long S = x.size(0), N = x.size(1);
  TORCH_CHECK(S % BS == 0, "S must be a multiple of 128");
  TORCH_CHECK(dvh.is_contiguous() && dvh.scalar_type() == torch::kBFloat16 && dvh.numel() == (long)H * N * S * C, "dv: [H, N, S*C] bf16");
  TORCH_CHECK(dgv.is_contiguous() && dgv.scalar_type() == torch::kBFloat16 && dgv.numel() == S * N * 2 * HC, "dgv: [S, N, 2 HC] bf16");
  TORCH_CHECK(wg.is_contiguous() && wv.is_contiguous() && wg.scalar_type() == torch::kBFloat16 && wv.scalar_type() == torch::kBFloat16 &&
              wg.numel() == HC * D && wv.numel() == HC * D, "wg, wv: [8 d_hidden, d_msa] bf16");
  TORCH_CHECK(lnw.is_contiguous() && lnw.numel() == D && (lnw.scalar_type() == torch::kFloat32 || lnw.scalar_type() == torch::kBFloat16), "LN scale [d_msa]");
  auto dm = torch::empty_like(x);
  const int ntiles = (int)(N * (S / BS));
  const int grid = std::min(ntiles, num_sms(x.device().index()));
  auto dln = torch::empty({grid, 2L * D}, x.options().dtype(torch::kFloat32));
  CUtensorMap dgm = make_map<3>(dgv.data_ptr(), {(uint64_t)(2 * HC), (uint64_t)N, (uint64_t)S}, {(uint64_t)(2 * HC), (uint64_t)(N * 2 * HC)}, {64, 1, (uint32_t)BS},
                                CU_TENSOR_MAP_SWIZZLE_128B, "dgv");
  CUtensorMap dvnm = make_map<3>(dgv.data_ptr(), {(uint64_t)(2 * HC), (uint64_t)N, (uint64_t)S}, {(uint64_t)(2 * HC), (uint64_t)(N * 2 * HC)}, {(uint32_t)C, 1, (uint32_t)BS},
                                 VSW, "dgv (dv)");
  CUtensorMap dvm = make_map<3>(dvh.data_ptr(), {(uint64_t)C, (uint64_t)S, (uint64_t)(H * N)}, {(uint64_t)C, (uint64_t)(S * C)}, {(uint32_t)C, (uint32_t)BS, 1}, VSW, "dv");
  auto nat = [&](const torch::Tensor& t, const char* w) {
    return make_map<3>(t.data_ptr(), {(uint64_t)D, (uint64_t)N, (uint64_t)S}, {(uint64_t)D, (uint64_t)(N * D)}, {64, 1, (uint32_t)BS}, CU_TENSOR_MAP_SWIZZLE_128B, w);
  };
  CUtensorMap xm = nat(x, "x"), rm = nat(dres, "dres"), dmm = nat(dm, "dm");
  CUtensorMap wgm = make_map<2>(wg.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, (uint32_t)HC}, CU_TENSOR_MAP_SWIZZLE_128B, "wg");
  CUtensorMap wvm = make_map<2>(wv.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, (uint32_t)HC}, CU_TENSOR_MAP_SWIZZLE_128B, "wv");
  auto st = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto lt) {
    using LT = decltype(lt);
    static bool a = false;
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(dvb_w_sm100<LT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
    dvb_w_sm100<LT><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, dgm, dvm, dvnm, xm, rm, dmm, wgm, wvm,
                                                 reinterpret_cast<const LT*>(lnw.data_ptr()), dln.data_ptr<float>());
  };
  if (lnw.scalar_type() == torch::kFloat32) launch(float{}); else launch(__nv_bfloat16{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto red = colsum(dln);                                          // [2 D]: dgamma | dbeta
  return {dm, red.slice(0, 0, D), red.slice(0, D, 2 * D)};
}
#endif  // PW_BWD

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_vg_w", &ln_vg_w, "sm100 wide PWA LayerNorm + value projection: (v head-major [H,N,S*C], y [S,N,D] or empty)",
        py::arg("m"), py::arg("lnw"), py::arg("lnb"), py::arg("wv"), py::arg("eps") = 1e-5, py::arg("want_y") = false);
  m.def("gate_out_w", &gate_out_w, "sm100 wide PWA gate / out-projection / residual pass over a given head-major o",
        py::arg("o"), py::arg("msa"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("wg"), py::arg("wo"),
        py::arg("dmask") = py::none(), py::arg("dscale") = 1.0);
#if PW_BWD
  m.def("glue_w", &glue_w, "sm100 wide PWA backward glue: (do head-major, dgv [S,N,2HC] with dgp, go [S,N,HC], dout')",
        py::arg("o"), py::arg("y"), py::arg("dres"), py::arg("wg"), py::arg("wo"), py::arg("dmask") = py::none(), py::arg("dscale") = 1.0);
  m.def("dvb_w", &dvb_w, "sm100 wide PWA backward tail: (dm = LN_bwd([dgp | dv] [Wg; Wv]) + dres, dgamma, dbeta); dv into dgv[..., HC:]",
        py::arg("dv"), py::arg("dgv"), py::arg("x"), py::arg("dres"), py::arg("wg"), py::arg("wv"), py::arg("lnw"), py::arg("eps") = 1e-5);
#endif
}
