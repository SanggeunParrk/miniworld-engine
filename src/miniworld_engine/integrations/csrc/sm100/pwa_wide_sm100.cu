// PWA (MSAPairWeightedAveraging) value side on B200 (sm_100a) at the widths the MiniWorld kernels (pwa_sm100.cu: d_msa 64, 8 x 32) do not
// cover: d_msa PW_D in {64, 128}, 8 heads x PW_C in {8, 16, 32} value channels (AF3 / Protenix-v1 64 / 8 x 8, Protenix-v2 / OpenDDE
// 128 / 8 x 8, ESMFold2 128 / 8 x 16, Chai-1 64 / 8 x 32 with d_pair 256), one build per (PW_D, PW_C). The pair side is pwa_sm100.cu's
// pair_fwd (d_pair 128 / 256 / 384, N <= 2048), the contraction o = w . v is cuBLAS bmm, as on the MiniWorld width.
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

namespace {
constexpr int H = 8, D = PW_D, C = PW_C, HC = H * C, KB = D / 64;
static_assert(D == 64 || D == 128, "d_msa 64 or 128");
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

// y = LN(x) of row r of a [KB][128][64] 128B-swizzled bf16 tile, in place: one thread, three passes (mean, centred variance, write)
__device__ __forceinline__ void ln_row(__nv_bfloat16* tile, int r, const float* sLN, float eps) {
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
      const float4 g0 = *reinterpret_cast<const float4*>(sLN + d0), g1 = *reinterpret_cast<const float4*>(sLN + d0 + 4);
      const float4 b0 = *reinterpret_cast<const float4*>(sLN + D + d0), b1 = *reinterpret_cast<const float4*>(sLN + D + d0 + 4);
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
constexpr int BS = 128, NST = 3;
constexpr int XT = BS * D;                      // x / y tile [KB][128][64]
constexpr int VT = H * BS * C;                  // v staging [8 heads][128 s][C]
constexpr int WT = HC * D;                      // Wv [KB][HC][64]
constexpr int THREADS = 320;
constexpr int SMEM = 1024 + (NST * XT + 2 * VT + WT) * 2 + 2 * D * 4 + 512;
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
  __nv_bfloat16* sW = sV + 2 * VT;
  float* sLN = reinterpret_cast<float*>(sW + WT);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLN + 2 * D);
  uint64_t* xfull = bars;              // [NST]
  uint64_t* yfull = xfull + NST;       // [NST] count 4 (LN warps)
  uint64_t* xempty = yfull + NST;      // [NST] count 2: the GEMM's commit + y's store has read the stage (or the MMA warp's arrive)
  uint64_t* accf = xempty + NST;       // [2]
  uint64_t* acce = accf + 2;           // [2] count 4
  uint64_t* wful = acce + 2;           // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wful + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { bar_init(xfull + s, 1); bar_init(yfull + s, 4); bar_init(xempty + s, 2); }
    for (int b = 0; b < 2; ++b) { bar_init(accf + b, 1); bar_init(acce + b, 4); }
    bar_init(wful, 1);
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
      expect_tx(wful, WT * 2);
      for (int kb = 0; kb < KB; ++kb) load_2d(&wmap, sW + kb * HC * 64, wful, kb * 64, 0);
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++g) {
        const int n = t / NSB, s0 = (t % NSB) * BS, st = g % NST;
        if (g >= NST) wait(xempty + st, ((g / NST) - 1) & 1);
        expect_tx(xfull + st, XT * 2);
        for (int kb = 0; kb < KB; ++kb) load_3d(&xmap, sX + st * XT + kb * BS * 64, xfull + st, kb * 64, n, s0);
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
          for (int kb = 0; kb < KB; ++kb) store_3d(&ymap, sX + st * XT + kb * BS * 64, kb * 64, n, s0);
          bulk_commit();
        } else {
          arrive(xempty + st);
        }
        if (lt >= 2) wait(acce + b, ((lt >> 1) - 1) & 1);
        tc_fence_after();
        const __nv_bfloat16* y = sX + st * XT;
#pragma unroll
        for (int kb = 0; kb < KB; ++kb)
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            mma_ss(tmem + b * HC, desc_k128(y + kb * BS * 64 + ks * 16), desc_k128(sW + kb * HC * 64 + ks * 16), IDESC, (kb | ks) ? 1u : 0u);
        mma_commit(accf + b);
        mma_commit(xempty + st);
        if (want_y && lt >= 1) {
          bulk_wait_read<1>();
          arrive(xempty + (lt - 1) % NST);
        }
      }
      if (want_y && lt >= 1) {                   // the last tile's y store: release its stage too (nothing waits on it, but keep the count)
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
      ln_row(sX + st * XT, r, sLN, eps);
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
      if (leader) bulk_wait_read<1>();           // the v stores of tile lt - 2 have left staging buffer b
      named_sync(2, 128);
      __nv_bfloat16* vb = sV + b * VT;
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
constexpr int NST = 2;
constexpr int WGT = HC * D;                     // Wg [KB][HC][64]
constexpr int KO = HC / 64 > 0 ? HC / 64 : 1;   // k-blocks of Wo
constexpr int WOT = D * KO * 64;                // Wo [KO k-blocks][D][64]
constexpr int THREADS = 576;
constexpr int SMEM = 1024 + (NST * STAGE + WGT + WOT) * 2 + 1024;
constexpr int COL_G = 0, COL_U = HC, COL_OUT = HC + HC / 2;    // gate [HC] fp32 | u [HC / 2] packed bf16 | out [2][D]
constexpr int TCOLS = 512;
constexpr uint32_t ID_G = idesc_bf16(128, HC / 2, 0, 0);
constexpr uint32_t ID_OUT = idesc_bf16(128, D, 0, 0);
static_assert(COL_OUT + 2 * D <= TCOLS, "TMEM: 512 columns");
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
  __nv_bfloat16* sWg = sStage + NST * STAGE;
  __nv_bfloat16* sWo = sWg + WGT;
  float* sLN = reinterpret_cast<float*>(sWo + WOT);
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
    for (int k = 0; k < NST; ++k) { bar_init(full + k, 1); bar_init(empty + k, 9); bar_init(yrdy + k, 4); }
    for (int k = 0; k < 2; ++k) { bar_init(gf + k, 1); bar_init(ge + k, 8); bar_init(uf + k, 8); bar_init(ue + k, 1);
                                  bar_init(outf + k, 1); bar_init(oute + k, 4); }
    bar_init(wf, 1);
    bar_init_fence();
  }
  if (tid < D) { sLN[tid] = to_f(LNW[tid]); sLN[D + tid] = to_f(LNB[tid]); }
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
      for (int kb = 0; kb < KO; ++kb) load_2d(&wmap, sWo + kb * D * 64, wf, kb * 64, 0);
      for (int lt = 0; lt < nt; ++lt) {
        const int m0 = ((int)blockIdx.x + lt * (int)gridDim.x) * BM, st = lt % NST;
        if (lt >= NST) wait(empty + st, ((lt / NST) - 1) & 1);
        __nv_bfloat16* p = stage(lt);
        expect_tx(full + st, STAGE * 2);
        const int s = m0 / N, i0 = m0 % N;
        for (int h = 0; h < H; ++h) load_2d(&omap, p + h * BM * C, full + st, s * C, h * N + i0);
        for (int kb = 0; kb < KB; ++kb) load_2d(&ymap, p + OT + kb * BM * 64, full + st, kb * 64, m0);
      }
    }
  } else if (warp == 1) {
    wait(wf, 0);
    auto gate = [&](int k) {                     // gate half hf of tile lt: g[:, HC/2 hf + ...] = y . Wg[HC/2 hf ...]^T
      const int lt = k >> 1, hf = k & 1, st = lt % NST;
      if (hf == 0) wait(yrdy + st, (lt / NST) & 1);
      if (k >= 2) wait(ge + hf, ((k >> 1) - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* y = stage(lt) + OT;
#pragma unroll
        for (int kb = 0; kb < KB; ++kb)
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            mma_ss(tmem + COL_G + hf * (HC / 2), desc_k128(y + kb * BM * 64 + ks * 16),
                   desc_k128(sWg + kb * HC * 64 + hf * (HC / 2) * 64 + ks * 16), ID_G, (kb | ks) ? 1u : 0u);
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
        for (int ks = 0; ks < HC / 32; ++ks) {
          const int kk = hf * (HC / 2) + ks * 16;
          mma_ts(tmem + COL_OUT + b * D, tmem + COL_U + hf * (HC / 4) + ks * 8, desc_k128(sWo + (kk >> 6) * D * 64 + ((kk & 63) >> 4) * 16),
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
        tmem_ldn<C>(tmem_at(tmem + COL_G, q * 32, head * C), g);
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
      ln_row(stage(lt) + OT, r, sLN, eps);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(yrdy + st);
    }
  } else {
    // ---- output: bf16(update), dropout, + residual, straight to global, 32 channels at a time ----
    const int q = warp & 3, r = q * 32 + lane;
    for (int lt = 0; lt < nt; ++lt) {
      const int row = ((int)blockIdx.x + lt * (int)gridDim.x) * BM + r, b = lt & 1;
      wait(outf + b, (lt >> 1) & 1);
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
  CUtensorMap gm = make_map<2>(wg.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, (uint32_t)HC}, CU_TENSOR_MAP_SWIZZLE_128B, "wg");
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_vg_w", &ln_vg_w, "sm100 wide PWA LayerNorm + value projection: (v head-major [H,N,S*C], y [S,N,D] or empty)",
        py::arg("m"), py::arg("lnw"), py::arg("lnb"), py::arg("wv"), py::arg("eps") = 1e-5, py::arg("want_y") = false);
  m.def("gate_out_w", &gate_out_w, "sm100 wide PWA gate / out-projection / residual pass over a given head-major o",
        py::arg("o"), py::arg("msa"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("wg"), py::arg("wo"),
        py::arg("dmask") = py::none(), py::arg("dscale") = 1.0);
}
