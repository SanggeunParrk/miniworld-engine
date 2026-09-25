// PWA (MSAPairWeightedAveraging) fused kernels for B200 (sm_100a).  The same fusion algorithm, tensors and
// layouts as the H100 kernels in csrc/ (ln_vg.cu, pwa_fwd3.cu, pwa_glue3.cu, pwa_fwd2.cu's pwa_plain2,
// dgv_bwd.cu); the tensor-core work moves from wgmma to tcgen05 (TMEM accumulators, one issuing thread) and
// every operand tile is fetched by TMA.
//
//   ln_vg:  y[s,n,:] = bf16(LN(m[s,n,:]))                  [S][N][64]
//           v[h][n][s*C + c] = bf16(y[s,n,:] . Wv[h*C+c,:])  head-major, the layout the contraction reads
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
    int N, int S, int ntiles, float eps,
    const __grid_constant__ CUtensorMap xmap,     // m [S][N][64]: (64, N, S), box (64, 1, 128), 128B swizzle
    const __grid_constant__ CUtensorMap ymap,     // y, same
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
        store_3d(&ymap, sX + st * XT, 0, n, s0);  // y leaves from the stage it was written in
        bulk_commit();
        if (lt >= 2) wait(acce + b, ((lt >> 1) - 1) & 1);
        tc_fence_after();
        const __nv_bfloat16* y = sX + st * XT;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks)
          mma_ss(tmem + b * HC, desc_k128(y + ks * 16), desc_k128(sW + ks * 16), IDESC, ks ? 1u : 0u);
        mma_commit(accf + b);
        mma_commit(xempty + st);
        if (lt >= 1) {                           // the previous tile's y store has read its stage: its second release
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
// PWA forward: out[s,i,:] = msa[s,i,:] + drop(sum_h (sigmoid(y[s,i,:] Wg_h^T) .* o_h[s,i,:]) Wo_h),
//              o_h[s,i,:] = sum_j w[h,i,j] v[h][j][s*C + :]            (optionally keeping o, natural [S][N][HC])
// The H100 kernel's algorithm per (128-row i tile, a few s): for each head the contraction (M = 128 i, N = s x 32 c,
// K = N j in 64-wide chunks), the gate GEMM against the tile's y rows, sigmoid x o, and the out-projection
// accumulated over the heads; the residual (and the row-broadcast dropout) in the output epilogue.
//
// Three s per tile (N = 96): TMEM holds two o buffers (so the next head's contraction runs while the drain reads
// this one), one gate buffer (released as soon as the drain has read it) and the out accumulator:
// 2 x 96 + 96 + 3 x 64 = 480 of 512 columns.  Wider N re-reads each W tile for more output: at N = 64 the
// tensor core was bound by its own shared-memory operand reads.  A tile past the end of S is zero-filled by TMA
// and never stored.
//   warp 0: TMA producer -- (W chunk, v chunk) stages, per-head Wg_h / Wo_h, the tile's y rows
//   warp 1: tcgen05 issue -- o_h, gate_h, then out += u_{h-1} . Wo_{h-1}^T
//   warps 2-5: drain -- u = o / (1 + exp(-g)) -> the bf16 A tile of the out GEMM (single-buffered: the out GEMM of a head is
//              short and retires long before the next head's gate); o -> global for the backward
//   warps 6-9: the tile's output -- bf16(out), dropout, + residual, straight to global (a warp's rows are 4 KiB contiguous);
//              on separate warps it overlaps the next tile's first heads instead of stalling them
// Shared memory goes to the load ring: the stage loads are latency-bound, so the ring's depth is the throughput.
namespace pf {
constexpr int BI = 128, BS = 3, NO = BS * C;              // o / gate columns per head: 96
constexpr int JC = 64;                                     // j per stage
constexpr int WTL = BI * JC, VTL = JC * NO;               // W chunk [128 i][64 j] (128B swz), v chunk [3 s][64 j][32 c] (64B swz)
constexpr int STAGE = WTL + VTL;                           // 28 KiB
constexpr int NST = 4;                                     // 112 KiB of loads in flight: TMA here is latency-bound (bytes in flight / ~1.2 us)
constexpr int YT = BS * BI * D;                            // y rows [3 s][128 i][64]
constexpr int WH = C * D + D * C;                          // Wg_h [32 c][64 d] (128B swz) | Wo_h [64 d][32 c] (64B swz)
constexpr int UT = BS * BI * C;                            // u (or o) tile [3 s][128 i][32], 64B swizzle
constexpr int THREADS = 320;                               // + warps 6-9: the tile's output epilogue
constexpr int SMEM = 1024 + (NST * STAGE + YT + 2 * WH + UT) * 2 + 512;
constexpr int COL_O = 0, COL_G = 2 * NO, COL_OUT = 3 * NO;
constexpr uint32_t ID_CTR = idesc_bf16(128, NO, 0, 1);    // W K-major, v MN-major
constexpr uint32_t ID_GATE = idesc_bf16(128, C, 0, 0);
constexpr uint32_t ID_OUT = idesc_bf16(128, D, 0, 0);
static_assert(COL_OUT + BS * D <= 512, "TMEM columns");
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace pf

__global__ void __launch_bounds__(pf::THREADS, 1) pwa_fwd_sm100(
    int N, int S, int ntiles, int save_o,
    const __grid_constant__ CUtensorMap wmap,     // w16 [H*N][N], box (64 j, 128 i), 128B swizzle
    const __grid_constant__ CUtensorMap vmap,     // v [H*N][S*C] as (32 c, H*N rows, S), box (32, 64, 3), 64B swizzle
    const __grid_constant__ CUtensorMap ymap,     // y [S][N][64] as (64, N, S), box (64, 128, 1), 128B swizzle
    const __grid_constant__ CUtensorMap gmap,     // Wg [HC][64], box (64, 32), 128B swizzle
    const __grid_constant__ CUtensorMap omap_w,   // Wo [64][HC], box (32, 64), 64B swizzle
    __nv_bfloat16* __restrict__ OSAVE,            // o [S][N][HC] or nullptr
    const __nv_bfloat16* __restrict__ MSA,        // [S][N][64]
    __nv_bfloat16* __restrict__ OUT,              // [S][N][64]
    const __nv_bfloat16* __restrict__ DMASK,      // drop_msa keep-mask [N][64] (0/1, shared over s), or nullptr
    float dscale) {
  using namespace pf;
  const int NIB = N / BI;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sRing = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sY = sRing + NST * STAGE;
  __nv_bfloat16* sWH = sY + YT;                  // [2][WH]
  __nv_bfloat16* sU = sWH + 2 * WH;              // [UT]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sU + UT);
  uint64_t* full = bars;             // [NST]
  uint64_t* empty = full + NST;      // [NST]
  uint64_t* whf = empty + NST;       // [2]
  uint64_t* whe = whf + 2;           // [2]  the head's out GEMM retired (its gate GEMM before it)
  uint64_t* yf = whe + 2;            // [1]
  uint64_t* ye = yf + 1;             // [1]  the tile's last gate GEMM retired
  uint64_t* accf = ye + 1;           // [2]  o_h (buffer gh & 1) and gate_h complete
  uint64_t* acce = accf + 2;         // [2]  count 4: o buffer drained
  uint64_t* ge = acce + 2;           // [1]  count 4: gate drained
  uint64_t* uf = ge + 1;             // [2]  count 4: u_h written (parity by head)
  uint64_t* ue = uf + 2;             // [1]  the out GEMM reading u retired
  uint64_t* outf = ue + 1;           // [1]  the tile's out accumulator complete
  uint64_t* oute = outf + 1;         // [1]  count 4: drained
  uint32_t* tslot = reinterpret_cast<uint32_t*>(oute + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(full + k, 1); bar_init(empty + k, 1); }
    for (int k = 0; k < 2; ++k) { bar_init(whf + k, 1); bar_init(whe + k, 1); bar_init(accf + k, 1); bar_init(acce + k, 4);
                                  bar_init(uf + k, 4); }
    bar_init(ue, 1); bar_init(yf, 1); bar_init(ye, 1); bar_init(ge, 4); bar_init(outf, 1); bar_init(oute, 4);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int NJC = N / JC;

  if (warp == 0) {
    if (lane == 0) {
      int g = 0, gh = 0, lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int i0 = (t % NIB) * BI, s0 = (t / NIB) * BS;
        if (lt >= 1) wait(ye, (lt - 1) & 1);
        expect_tx(yf, YT * 2);
        for (int si = 0; si < BS; ++si) load_3d(&ymap, sY + si * BI * D, yf, 0, i0, s0 + si);
        for (int h = 0; h < H; ++h, ++gh) {
          const int wb = gh & 1;
          if (gh >= 2) wait(whe + wb, ((gh >> 1) - 1) & 1);
          expect_tx(whf + wb, WH * 2);
          load_2d(&gmap, sWH + wb * WH, whf + wb, 0, h * C);
          load_2d(&omap_w, sWH + wb * WH + C * D, whf + wb, h * C, 0);
          for (int jc = 0; jc < NJC; ++jc, ++g) {
            const int st = g % NST;
            if (g >= NST) wait(empty + st, ((g / NST) - 1) & 1);
            __nv_bfloat16* p = sRing + st * STAGE;
            expect_tx(full + st, STAGE * 2);
            load_2d(&wmap, p, full + st, jc * JC, h * N + i0);
            load_3d(&vmap, p + WTL, full + st, 0, h * N + jc * JC, s0);
          }
        }
      }
    }
  } else if (warp == 1) {
    // The whole warp walks the issue loop, converged; each batch of MMAs (and its commits) is issued by one elected
    // lane.  A lone lane-0 loop costs an ELECT round trip per tcgen05.mma -- ~70 cycles, more than an N = 96 MMA.
    int g = 0, gh = 0, lt = 0;
    auto out_gemm = [&](int ghp, int hp, int ltp) {   // out += u_{hp} . Wo_{hp}^T, every s of the tile
      const int b = ghp & 1;
      wait(uf + b, (ghp >> 1) & 1);
      if (hp == 0 && ltp >= 1) wait(oute, (ltp - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
        const __nv_bfloat16* wo = sWH + b * WH + C * D;
#pragma unroll
        for (int si = 0; si < BS; ++si)
#pragma unroll
          for (int ks = 0; ks < C / 16; ++ks)
            mma_ss(tmem + COL_OUT + si * D, desc_k64(sU + si * BI * C + ks * 16), desc_k64(wo + ks * 16), ID_OUT, (hp | ks) ? 1u : 0u);
        mma_commit(ue);
        mma_commit(whe + b);
        if (hp == H - 1) mma_commit(outf);
      }
      __syncwarp();
    };
    int pgh = -1, ph = 0, plt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      wait(yf, lt & 1);
      for (int h = 0; h < H; ++h, ++gh) {
        const int b = gh & 1;
        if (gh >= 2) wait(acce + b, ((gh >> 1) - 1) & 1);
        wait(whf + b, (gh >> 1) & 1);
        tc_fence_after();
        for (int jc = 0; jc < NJC; ++jc, ++g) {
          const int st = g % NST;
          wait(full + st, (g / NST) & 1);
          tc_fence_after();
          if (elect_one()) {
            const __nv_bfloat16* p = sRing + st * STAGE;
#pragma unroll
            for (int ks = 0; ks < JC / 16; ++ks)
              mma_ss(tmem + COL_O + b * NO, desc_k128(p + ks * 16), sdesc(sa(p + WTL + ks * 16 * C), JC * C * 2, 512, 4), ID_CTR, (jc | ks) ? 1u : 0u);
            mma_commit(empty + st);
          }
          __syncwarp();
        }
        if (gh >= 1) wait(ge, (gh - 1) & 1);     // the drain has read the previous head's gate
        tc_fence_after();
        if (elect_one()) {
          const __nv_bfloat16* wg = sWH + b * WH;
#pragma unroll
          for (int si = 0; si < BS; ++si)
#pragma unroll
            for (int ks = 0; ks < D / 16; ++ks)
              mma_ss(tmem + COL_G + si * C, desc_k128(sY + si * BI * D + ks * 16), desc_k128(wg + ks * 16), ID_GATE, ks ? 1u : 0u);
          mma_commit(accf + b);
          if (h == H - 1) mma_commit(ye);
        }
        __syncwarp();
        if (pgh >= 0) out_gemm(pgh, ph, plt);
        pgh = gh; ph = h; plt = lt;
      }
    }
    if (pgh >= 0) out_gemm(pgh, ph, plt);
  } else if (warp <= 5) {
    const int q = warp & 3, r = q * 32 + lane;   // TMEM lane = i row of the tile
    int gh = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
      const int i0 = (t % NIB) * BI, s0 = (t / NIB) * BS;
      for (int h = 0; h < H; ++h, ++gh) {
        const int b = gh & 1;
        wait(accf + b, (gh >> 1) & 1);
        tc_fence_after();
        float gt[NO];
        tmem_ld32(tmem_at(tmem + COL_G, q * 32, 0), gt);
        tmem_ld32(tmem_at(tmem + COL_G, q * 32, 32), gt + 32);
        tmem_ld32(tmem_at(tmem + COL_G, q * 32, 64), gt + 64);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(ge);               // the next head's gate GEMM may overwrite it
#pragma unroll
        for (int k = 0; k < NO; ++k) gt[k] = 1.f / (1.f + __expf(-gt[k]));
        if (gh >= 1) wait(ue, (gh - 1) & 1);     // the out GEMM of the previous head has read u
#pragma unroll
        for (int si = 0; si < BS; ++si) {
          float o[C];
          tmem_ld32(tmem_at(tmem + COL_O + b * NO, q * 32, si * C), o);
          tmem_wait_ld();
          if (si == BS - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + b); }
          __nv_bfloat16* ur = sU + (si * BI + r) * C;
          const bool live = OSAVE != nullptr && s0 + si < S;
          uint4* og = live ? reinterpret_cast<uint4*>(OSAVE + ((size_t)(s0 + si) * N + i0 + r) * HC + h * C) : nullptr;
#pragma unroll
          for (int c8 = 0; c8 < 4; ++c8) {
            uint4 uu, oo;
            uint32_t* uw = reinterpret_cast<uint32_t*>(&uu);
            uint32_t* ow = reinterpret_cast<uint32_t*>(&oo);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const int c = c8 * 8 + 2 * k;
              const float2 u2 = mul2(make_float2(o[c], o[c + 1]), make_float2(gt[si * C + c], gt[si * C + c + 1]));
              uw[k] = pack2(u2.x, u2.y);
              ow[k] = pack2(o[c], o[c + 1]);
            }
            *reinterpret_cast<uint4*>(ur + ((c8 ^ ((r >> 1) & 3)) << 3)) = uu;
            if (live) og[c8] = oo;
          }
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) arrive(uf + b);
      }
    }
  } else {
    // ---- the tile's output: bf16(update), dropout, + residual ----
    const int q = warp & 3, r = q * 32 + lane;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i0 = (t % NIB) * BI, s0 = (t / NIB) * BS;
      float dm[D];
      if (DMASK != nullptr) {
        const __nv_bfloat16* dr = DMASK + (size_t)(i0 + r) * D;
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          const uint4 u = *reinterpret_cast<const uint4*>(dr + c8 * 8);
          const float2 a0 = bf2f(u.x), a1 = bf2f(u.y), a2 = bf2f(u.z), a3 = bf2f(u.w);
          dm[c8 * 8 + 0] = a0.x * dscale; dm[c8 * 8 + 1] = a0.y * dscale; dm[c8 * 8 + 2] = a1.x * dscale; dm[c8 * 8 + 3] = a1.y * dscale;
          dm[c8 * 8 + 4] = a2.x * dscale; dm[c8 * 8 + 5] = a2.y * dscale; dm[c8 * 8 + 6] = a3.x * dscale; dm[c8 * 8 + 7] = a3.y * dscale;
        }
      }
      wait(outf, lt & 1);
      tc_fence_after();
#pragma unroll 1
      for (int si = 0; si < BS; ++si) {
        float v[D];
        tmem_ld32(tmem_at(tmem + COL_OUT + si * D, q * 32, 0), v);
        tmem_ld32(tmem_at(tmem + COL_OUT + si * D, q * 32, 32), v + 32);
        tmem_wait_ld();
        if (si == BS - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(oute); }
        if (s0 + si >= S) continue;
        const size_t rowoff = ((size_t)(s0 + si) * N + i0 + r) * D;
        const uint4* rp = reinterpret_cast<const uint4*>(MSA + rowoff);
        uint4* op = reinterpret_cast<uint4*>(OUT + rowoff);
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          uint4 rv = __ldg(rp + c8);
          uint32_t* rw = reinterpret_cast<uint32_t*>(&rv);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int c = c8 * 8 + 2 * k;
            // the stock module rounds the update to bf16 before the (dropout and the) residual add
            float u0 = __bfloat162float(__float2bfloat16(v[c])), u1 = __bfloat162float(__float2bfloat16(v[c + 1]));
            if (DMASK != nullptr) { u0 = __bfloat162float(__float2bfloat16(u0 * dm[c])); u1 = __bfloat162float(__float2bfloat16(u1 * dm[c + 1])); }
            const float2 res = bf2f(rw[k]);
            rw[k] = pack2(res.x + u0, res.y + u1);
          }
          op[c8] = rv;
        }
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---------------------------------------------------------------------------------------------
// PWA backward glue from the saved o (the H100 pwa_glue3 algorithm), per (128-row i tile, 2 s), per head:
//   g = sigmoid(y Wg_h^T), du = dres' Wo_h (dres' = dres . drop keep-mask / (1-p), masked once in shared memory)
//   do  = du . g                   -> head-major [H][N][S*C]  (the dv contraction's operand)
//   dgp = du . o . g (1 - g)       -> natural, into the first half of the [S][N][512] dgv buffer
//   go  = g . o                    -> shared memory only: dWo_h[d][c] += sum_(s,i) dres'[s,i,d] go[s,i,c], in TMEM
// dWo is an M = 64 product: two M = 64 accumulators share TMEM columns (lanes 0-15 / 16-31 of each sub-partition),
// so heads 0-3 and 4-7 take the two lane halves of the same 128 columns; per-CTA fp32 slabs are summed on the host.
namespace gl {
constexpr int BI = 128, BS = 2, NO = BS * C;
constexpr int OT = BS * BI * C;                            // o / go / dgp tile [2 s][128 i][32], 64B swizzle
constexpr int DT = BI * NO;                                // do tile [128 i][64 (s,c)], 128B swizzle
constexpr int RT = BS * BI * D;                            // dres / y tile [2 s][128 i][64], 128B swizzle
constexpr int WH = 2 * C * D;                              // Wg_h | WoT_h, each [32 c][64 d], 128B swizzle
constexpr int NST = 3;
constexpr int THREADS = 320;                               // warp 0 producer, warp 1 MMA, warps 2-9 drain (warps q and q + 4: one s each)
constexpr int SMEM = 1024 + (NST * OT + 2 * DT + 2 * OT + 2 * OT + 2 * RT + 2 * WH) * 2 + 512;
constexpr int COL_G = 0, COL_DU = 2 * NO, COL_W = 4 * NO;  // g[2] 0/64, du[2] 128/192, dWo 256..383 (two lane halves)
constexpr uint32_t ID_GD = idesc_bf16(128, C, 0, 0);
constexpr uint32_t ID_W = idesc_bf16(64, C, 1, 1);         // A = dres'^T (MN-major), B = go (MN-major)
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace gl

__global__ void __launch_bounds__(gl::THREADS, 1) pwa_glue_sm100(
    int N, int S, int ntiles,
    const __grid_constant__ CUtensorMap omap,     // o [S][N][HC] as (32 c, N, S), box (32, 128, 2), 64B swizzle
    const __grid_constant__ CUtensorMap rmap,     // dres [S][N][64] as (64, N, S), box (64, 128, 1), 128B swizzle
    const __grid_constant__ CUtensorMap ymap,     // y, same
    const __grid_constant__ CUtensorMap gmap,     // Wg [HC][64], box (64, 32)
    const __grid_constant__ CUtensorMap wotmap,   // Wo^T [HC][64], box (64, 32)
    const __grid_constant__ CUtensorMap domap,    // do [H*N][S*C], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap dgpmap,   // dgv [S][N][512] as (32, N, S) with row stride 512, box (32, 128, 2), 64B swizzle
    const __nv_bfloat16* __restrict__ DMASK, float dscale,
    float* __restrict__ DWO) {                    // [grid][64][HC]
  using namespace gl;
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
  uint64_t* whe = whf + 2;           // [2]  the head's gate / du GEMMs retired
  uint64_t* tf = whe + 2;            // [1]  dres and y of the tile landed
  uint64_t* tr = tf + 1;             // [1]  count 8: dres masked (ready for the GEMMs)
  uint64_t* te = tr + 1;             // [1]  every GEMM reading the tile's dres / y retired
  uint64_t* accf = te + 1;           // [2]  g_h, du_h complete
  uint64_t* acce = accf + 2;         // [2]  count 8: drained
  uint64_t* gof = acce + 2;          // [2]  count 8: go_h written
  uint64_t* goe = gof + 2;           // [2]  the dWo GEMM reading go_h retired
  uint64_t* wdone = goe + 2;         // [1]  the CTA's last dWo GEMM retired
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wdone + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(of + k, 1); bar_init(oe + k, 8); }
    for (int k = 0; k < 2; ++k) { bar_init(whf + k, 1); bar_init(whe + k, 1); bar_init(accf + k, 1); bar_init(acce + k, 8);
                                  bar_init(gof + k, 8); bar_init(goe + k, 1); }
    bar_init(tf, 1); bar_init(tr, 8); bar_init(te, 1); bar_init(wdone, 1);
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
          load_3d(&omap, sO + st * OT, of + st, h * C, i0, s0);
        }
      }
    }
  } else if (warp == 1) {
    int gh = 0, lt = 0;
    auto dwo_gemm = [&](int ghp, int hp, int last) {  // dWo_hp += dres'^T . go_hp over the tile's 256 (s, i) rows
      const int b = ghp & 1;
      wait(gof + b, (ghp >> 1) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint32_t d = tmem + ((uint32_t)(hp >> 2) << 20) + COL_W + (hp & 3) * C;   // lane half 16 * (hp >> 2)
        const __nv_bfloat16* go = sGO + b * OT;
#pragma unroll
        for (int ks = 0; ks < BS * BI / 16; ++ks)
          mma_ss(d, desc_mn128(sR + ks * 16 * 64, 0), sdesc(sa(go + ks * 16 * C), 0, 512, 4), ID_W, 1u);
        mma_commit(goe + b);
        if (last) mma_commit(wdone);
      }
      __syncwarp();
    };
    // zero the dWo accumulator once (an MMA with accumulate = 0 would need a first K step per head: simpler to clear it)
    int pgh = -1, ph = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      wait(tr, lt & 1);                          // dres' and y of the tile in place
      for (int h = 0; h < H; ++h, ++gh) {
        const int b = gh & 1;
        if (gh >= 2) wait(acce + b, ((gh >> 1) - 1) & 1);
        wait(whf + b, (gh >> 1) & 1);
        tc_fence_after();
        if (elect_one()) {
          const __nv_bfloat16* wg = sWH + b * WH;
          const __nv_bfloat16* wot = wg + C * D;
#pragma unroll
          for (int si = 0; si < BS; ++si)
#pragma unroll
            for (int ks = 0; ks < D / 16; ++ks) {
              mma_ss(tmem + COL_G + b * NO + si * C, desc_k128(sY + si * BI * D + ks * 16), desc_k128(wg + ks * 16), ID_GD, ks ? 1u : 0u);
              mma_ss(tmem + COL_DU + b * NO + si * C, desc_k128(sR + si * BI * D + ks * 16), desc_k128(wot + ks * 16), ID_GD, ks ? 1u : 0u);
            }
          mma_commit(accf + b);
          mma_commit(whe + b);
        }
        __syncwarp();
        if (pgh >= 0) dwo_gemm(pgh, ph, 0);
        if (h == H - 1) {                        // the tile's last dWo GEMM, then the tile's dres / y are free
          dwo_gemm(gh, h, t + (int)gridDim.x >= ntiles);
          if (elect_one()) mma_commit(te);
          __syncwarp();
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
    // clear the dWo accumulator (lanes of this warp's sub-partition, all 128 columns)
    if (si == 0) {
      uint32_t z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int c0 = 0; c0 < 4 * C; c0 += 8) tmem_st8(tmem_at(tmem + COL_W, q * 32, c0), z);
      tmem_wait_st();
    }
    tc_fence_before();
    named_sync(1, 256);
    int gh = 0, lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i0 = (t % NIB) * BI, s0 = (t / NIB) * BS;
      wait(tf, lt & 1);
      if (DMASK != nullptr) {                    // dres' = dres . keep / (1 - p), once, in place (row r, this warp's s)
        const __nv_bfloat16* dr = DMASK + (size_t)(i0 + r) * D;
        {
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
        }
        fence_proxy_async();
      }
      __syncwarp();
      if (lane == 0) arrive(tr);
      for (int h = 0; h < H; ++h, ++gh) {
        const int b = gh & 1, st = gh % NST;
        wait(accf + b, (gh >> 1) & 1);
        tc_fence_after();
        float gt[C], du[C];                      // this warp's s: 32 channels of gate and du
        tmem_ld32(tmem_at(tmem + COL_G + b * NO, q * 32, si * C), gt);
        tmem_ld32(tmem_at(tmem + COL_DU + b * NO, q * 32, si * C), du);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(acce + b);
        wait(of + st, (gh / NST) & 1);
        if (gh >= 2) wait(goe + b, ((gh >> 1) - 1) & 1);        // the dWo GEMM of head gh - 2 has read go buffer b
        if (leader) bulk_wait_read<1>();                         // the do / dgp stores of head gh - 2 have read staging b
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
              const float2 g2 = make_float2(1.f / (1.f + __expf(-gt[c])), 1.f / (1.f + __expf(-gt[c + 1])));
              const float2 d2 = make_float2(du[c], du[c + 1]);
              const float2 do2 = mul2(d2, g2);
              const float2 gg = fma2(g2, make_float2(-g2.x, -g2.y), g2);                           // g - g^2
              const float2 dg2 = mul2(mul2(d2, o2), gg);
              const float2 go2 = mul2(g2, o2);
              wdo[k] = pack2(do2.x, do2.y); wdg[k] = pack2(dg2.x, dg2.y); wgo[k] = pack2(go2.x, go2.y);
            }
            *reinterpret_cast<uint4*>(dgb + orow + off64) = vdg;
            *reinterpret_cast<uint4*>(gob + orow + off64) = vgo;
            const int cdo = si * 4 + c8;                                                          // do row r: 64 (s,c) columns
            *reinterpret_cast<uint4*>(dob + r * 64 + ((cdo ^ (r & 7)) << 3)) = vdo;
          }
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) { arrive(gof + b); arrive(oe + st); }
        named_sync(1, 256);
        if (leader) {
          store_2d(&domap, dob, s0 * C, h * N + i0);
          store_3d(&dgpmap, dgb, h * C, i0, s0);
          bulk_commit();
        }
      }
    }
    if (leader) bulk_wait<0>();
    // this CTA's dWo slab: lane half 0 = heads 0-3, half 1 = heads 4-7; row d = q * 16 + (lane & 15)
    if (si == 0) {
    wait(wdone, 0);
    tc_fence_after();
    const int half = lane >> 4, d = q * 16 + (lane & 15);
    float* slab = DWO + (size_t)blockIdx.x * D * HC + (size_t)d * HC + half * 4 * C;
#pragma unroll 1
    for (int c0 = 0; c0 < 4 * C; c0 += 32) {
      float v[32];
      tmem_ld32(tmem_at(tmem + COL_W, q * 32, c0), v);
      tmem_wait_ld();
#pragma unroll
      for (int k = 0; k < 32; k += 4) *reinterpret_cast<float4*>(slab + c0 + k) = make_float4(v[k], v[k + 1], v[k + 2], v[k + 3]);
    }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---------------------------------------------------------------------------------------------
// PWA backward, the plain contraction (the H100 pwa_plain2): dv[h][j][(s,c)] = sum_i w[h][i][j] do[h][i][(s,c)],
// written natural into the second half of the [S][N][512] dgv buffer.  Tile = one head x 128 j x 8 s: N = 256, the
// tensor core's full-rate shape.  A = w^T is read MN-major straight from w16 (j contiguous in each i row): no
// transposed copy of w; B = do chunks, MN-major.  Two 256-column accumulators alternate between tiles.
namespace pl {
constexpr int BJ = 128, BS = 8, NN = BS * C;               // 256 output columns
constexpr int KC = 64;                                     // i per stage
constexpr int AT = KC * BJ, BT = KC * NN;                  // w chunk [2 j blocks][64 i][64 j], do chunk [4 blocks][64 i][64]
constexpr int STAGE = AT + BT;                             // 48 KiB
constexpr int NST = 4;                                     // the stage loads are latency-bound: ring depth is throughput
constexpr int OUTT = BS / 2 * BJ * C;                      // natural staging for half the tile [4 s][128 j][32], 64B swizzle (32 KiB)
constexpr int THREADS = 192;
constexpr int SMEM = 1024 + (NST * STAGE + OUTT) * 2 + 256;
constexpr uint32_t IDESC = idesc_bf16(128, NN, 1, 1);
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace pl

__global__ void __launch_bounds__(pl::THREADS, 1) pwa_plain_sm100(
    int N, int S, int ntiles,
    const __grid_constant__ CUtensorMap wmap,     // w16 [H*N][N], box (64 j, 64 i), 128B swizzle
    const __grid_constant__ CUtensorMap dmap,     // do [H*N][S*C], box (64, 64 i), 128B swizzle
    const __grid_constant__ CUtensorMap vmap) {   // dgv as (32 c, N, S) at column offset 256 + h*32, row stride 512, box (32, 128, 8)
  using namespace pl;
  const int NJB = N / BJ, NSB = (S + BS - 1) / BS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sOut = sStage + NST * STAGE;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sOut + OUTT);
  uint64_t* full = bars;             // [NST]
  uint64_t* empty = full + NST;      // [NST]
  uint64_t* accf = empty + NST;      // [2]
  uint64_t* acce = accf + 2;         // [2] count 4
  uint32_t* tslot = reinterpret_cast<uint32_t*>(acce + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(full + k, 1); bar_init(empty + k, 1); }
    for (int k = 0; k < 2; ++k) { bar_init(accf + k, 1); bar_init(acce + k, 4); }
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int NKC = N / KC;
  // tile t = (s block fastest, then j block, then head): the w chunks of a (head, j block) stay in L2 across its s blocks
  auto decode = [&](int t, int& h, int& j0, int& s0) { s0 = (t % NSB) * BS; const int r = t / NSB; j0 = (r % NJB) * BJ; h = r / NJB; };

  if (warp == 0) {
    if (lane == 0) {
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        int h, j0, s0; decode(t, h, j0, s0);
        for (int kc = 0; kc < NKC; ++kc, ++g) {
          const int st = g % NST;
          if (g >= NST) wait(empty + st, ((g / NST) - 1) & 1);
          __nv_bfloat16* p = sStage + st * STAGE;
          expect_tx(full + st, STAGE * 2);
          for (int b = 0; b < 2; ++b) load_2d(&wmap, p + b * KC * 64, full + st, j0 + b * 64, h * N + kc * KC);
          for (int b = 0; b < 4; ++b) load_2d(&dmap, p + AT + b * KC * 64, full + st, s0 * C + b * 64, h * N + kc * KC);
        }
      }
    }
  } else if (warp == 1) {
    int g = 0, lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int b = lt & 1;
      if (lt >= 2) wait(acce + b, ((lt >> 1) - 1) & 1);
      tc_fence_after();
      for (int kc = 0; kc < NKC; ++kc, ++g) {
        const int st = g % NST;
        wait(full + st, (g / NST) & 1);
        tc_fence_after();
        if (elect_one()) {
          const __nv_bfloat16* p = sStage + st * STAGE;
#pragma unroll
          for (int ks = 0; ks < KC / 16; ++ks)
            mma_ss(tmem + b * NN, desc_mn128(p + ks * 16 * 64, KC * 64 * 2), desc_mn128(p + AT + ks * 16 * 64, KC * 64 * 2), IDESC, (kc | ks) ? 1u : 0u);
          mma_commit(empty + st);
          if (kc == NKC - 1) mma_commit(accf + b);
        }
        __syncwarp();
      }
    }
  } else {
    const int q = warp & 3, r = q * 32 + lane;   // TMEM lane = j row
    const bool leader = (warp == 2 && lane == 0);
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      int h, j0, s0; decode(t, h, j0, s0);
      const int b = lt & 1;
      wait(accf + b, (lt >> 1) & 1);
      tc_fence_after();
#pragma unroll 1
      for (int si = 0; si < BS; ++si) {
        if ((si & (BS / 2 - 1)) == 0) {          // each half of the tile: the staging must have been read by the last store
          if (leader) bulk_wait_read<0>();
          named_sync(1, 128);
        }
        float v[32];
        tmem_ld32(tmem_at(tmem + b * NN, q * 32, si * C), v);
        tmem_wait_ld();
        if (si == BS - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + b); }
        __nv_bfloat16* row = sOut + ((si & (BS / 2 - 1)) * BJ + r) * C;
#pragma unroll
        for (int c8 = 0; c8 < 4; ++c8) {
          uint4 o;
          o.x = pack2(v[c8 * 8 + 0], v[c8 * 8 + 1]); o.y = pack2(v[c8 * 8 + 2], v[c8 * 8 + 3]);
          o.z = pack2(v[c8 * 8 + 4], v[c8 * 8 + 5]); o.w = pack2(v[c8 * 8 + 6], v[c8 * 8 + 7]);
          *reinterpret_cast<uint4*>(row + ((c8 ^ ((r >> 1) & 3)) << 3)) = o;
        }
        if ((si & (BS / 2 - 1)) == BS / 2 - 1) {
          fence_proxy_async();
          named_sync(1, 128);
          if (leader) { store_3d(&vmap, sOut, HC + h * C, j0, s0 + si - (BS / 2 - 1)); bulk_commit(); }
        }
      }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---------------------------------------------------------------------------------------------
// PWA backward tail (the H100 dgv_bwd): reads the [M][512] dgv buffer (dgp | dv) once and produces
//   dy = dgv . Wgv (TMEM only), dm = LayerNorm_bwd(dy; x, gamma) + dout, dWgv = dgv^T . y, dgamma = sum dy xhat, dbeta = sum dy.
// Tile = 128 rows.  The 512 columns stream as eight [128][64] k-blocks (with the matching Wgv block); each block feeds
// the dy GEMM (A K-major) and, read MN-major, the dWgv^T GEMM (M = 64 d, N = the block's 64 columns): the eight
// column blocks of dWgv^T sit in two TMEM lane halves (blocks 0-3 / 4-7) of the same 256 columns.
namespace dv {
constexpr int BM = 128, KD = 2 * HC, KB = KD / 64;
constexpr int BLK = BM * 64, WB = D * 64;                  // dgv k-block [128][64], Wgv^T block [64 d][64 k]
constexpr int STAGE = BLK + WB;                            // 24 KiB
constexpr int NST = 6;                                     // the dgv stream is latency-bound: ring depth is throughput
constexpr int RT = BM * D;                                 // y / x / dout / dm tiles [128][64]
constexpr int THREADS = 224;                               // warp 0: y + the ring, warp 1: MMA, warps 2-5: drain, warp 6: x / dout
constexpr int SMEM = 1024 + (NST * STAGE + 2 * RT + 2 * RT + RT) * 2 + 512;   // y x2, x | dout x1, dm
constexpr int COL_DY = 0, COL_W = 128;                     // dy[2] 0/64, dWgv^T 128..383 (two lane halves)
constexpr uint32_t ID_DY = idesc_bf16(128, D, 0, 0);
constexpr uint32_t ID_W = idesc_bf16(64, 64, 1, 1);
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace dv

template <typename WT_>
__global__ void __launch_bounds__(dv::THREADS, 1) dgv_bwd_sm100(
    int M, int ntiles, float eps,
    const __grid_constant__ CUtensorMap gmap,     // dgv [M][512], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,     // Wgv^T [64][512], box (64, 64)
    const __grid_constant__ CUtensorMap ymap,     // y [M][64], box (64, 128)
    const __grid_constant__ CUtensorMap xmap,     // x (the msa input) [M][64]
    const __grid_constant__ CUtensorMap omap,     // dout (the residual gradient) [M][64]
    const __grid_constant__ CUtensorMap dmmap,    // dm [M][64]
    const WT_* __restrict__ LNW,
    float* __restrict__ DW,                       // [grid][64 d][512]  (dWgv^T partials)
    float* __restrict__ DLN) {                    // [grid][2][64]      (dgamma, dbeta partials)
  using namespace dv;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sYb = sStage + NST * STAGE;                     // [2][y]: the GEMMs read it from the tile's first block
  __nv_bfloat16* sXO = sYb + 2 * RT;                             // [x | dout], single: the drain reads it only at the tile's end
  __nv_bfloat16* sDM = sXO + 2 * RT;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sDM + RT);
  uint64_t* full = bars;             // [NST]
  uint64_t* empty = full + NST;      // [NST]
  uint64_t* tf = empty + NST;        // [2] y of tile buffer landed
  uint64_t* te = tf + 2;             // [2] the dWgv GEMMs (commit) are done with it
  uint64_t* xf = te + 2;             // [1] x / dout landed
  uint64_t* xe = xf + 1;             // [1] count 4: the drain read them
  uint64_t* accf = xe + 1;           // [2]
  uint64_t* acce = accf + 2;         // [2] count 4
  uint64_t* wdone = acce + 2;        // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wdone + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(full + k, 1); bar_init(empty + k, 1); }
    for (int k = 0; k < 2; ++k) { bar_init(tf + k, 1); bar_init(te + k, 1); bar_init(accf + k, 1); bar_init(acce + k, 4); }
    bar_init(xf, 1); bar_init(xe, 4); bar_init(wdone, 1);
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
        const int r0 = t * BM, b = lt & 1;
        if (lt >= 2) wait(te + b, ((lt >> 1) - 1) & 1);
        expect_tx(tf + b, RT * 2);
        load_2d(&ymap, sYb + b * RT, tf + b, 0, r0);
        for (int kb = 0; kb < KB; ++kb, ++g) {
          const int st = g % NST;
          if (g >= NST) wait(empty + st, ((g / NST) - 1) & 1);
          __nv_bfloat16* p = sStage + st * STAGE;
          expect_tx(full + st, STAGE * 2);
          load_2d(&gmap, p, full + st, kb * 64, r0);
          load_2d(&wmap, p + BLK, full + st, kb * 64, 0);
        }
      }
    }
  } else if (warp == 6) {
    if (lane == 0) {
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int r0 = t * BM;
        if (lt >= 1) wait(xe, (lt - 1) & 1);
        expect_tx(xf, 2 * RT * 2);
        load_2d(&xmap, sXO, xf, 0, r0);
        load_2d(&omap, sXO + RT, xf, 0, r0);
      }
    }
  } else if (warp == 1) {
    int g = 0, lt = 0;
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
          for (int ks = 0; ks < 4; ++ks)           // dy += dgv_kb . Wgv_kb
            mma_ss(tmem + COL_DY + b * D, desc_k128(p + ks * 16), desc_k128(p + BLK + ks * 16), ID_DY, (kb | ks) ? 1u : 0u);
          const uint32_t dw = tmem + ((uint32_t)(kb >> 2) << 20) + COL_W + (kb & 3) * 64;
#pragma unroll
          for (int ks = 0; ks < BM / 16; ++ks)     // dWgv^T[:, kb block] += y^T . dgv_kb over the tile's 128 rows
            mma_ss(dw, desc_mn128(y + ks * 16 * 64, 0), desc_mn128(p + ks * 16 * 64, 0), ID_W, (lt | ks) ? 1u : 0u);
          mma_commit(empty + st);
          if (kb == KB - 1) { mma_commit(accf + b); mma_commit(te + b); }
        }
        __syncwarp();
      }
    }
    if (elect_one()) mma_commit(wdone);
    __syncwarp();
  } else {
    const int q = warp & 3, r = q * 32 + lane;   // TMEM lane = row
    const bool leader = (warp == 2 && lane == 0);
    float gam[D];
#pragma unroll
    for (int c = 0; c < D; ++c) gam[c] = to_f(LNW[c]);
    float dga0 = 0.f, dga1 = 0.f, dbe0 = 0.f, dbe1 = 0.f;       // this lane's columns (lane, lane + 32) after the reduce-scatter
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int r0 = t * BM, b = lt & 1;
      wait(xf, lt & 1);
      wait(accf + b, (lt >> 1) & 1);
      tc_fence_after();
      float dy[D];
      tmem_ld32(tmem_at(tmem + COL_DY + b * D, q * 32, 0), dy);
      tmem_ld32(tmem_at(tmem + COL_DY + b * D, q * 32, 32), dy + 32);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(acce + b);
      const __nv_bfloat16* xr = sXO + r * 64;
      const __nv_bfloat16* orow = sXO + RT + r * 64;
      float xh[D];
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const uint4 u = *reinterpret_cast<const uint4*>(xr + ((c8 ^ (r & 7)) << 3));
        const float2 a0 = bf2f(u.x), a1 = bf2f(u.y), a2 = bf2f(u.z), a3 = bf2f(u.w);
        xh[c8 * 8 + 0] = a0.x; xh[c8 * 8 + 1] = a0.y; xh[c8 * 8 + 2] = a1.x; xh[c8 * 8 + 3] = a1.y;
        xh[c8 * 8 + 4] = a2.x; xh[c8 * 8 + 5] = a2.y; xh[c8 * 8 + 6] = a3.x; xh[c8 * 8 + 7] = a3.y;
      }
      float mean = 0.f;
#pragma unroll
      for (int c = 0; c < D; ++c) mean += xh[c];
      mean *= (1.f / D);
      float var = 0.f;
#pragma unroll
      for (int c = 0; c < D; ++c) { xh[c] -= mean; var += xh[c] * xh[c]; }
      const float rstd = rsqrtf(var * (1.f / D) + eps);
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int c = 0; c < D; ++c) { xh[c] *= rstd; const float gd = gam[c] * dy[c]; s1 += gd; s2 += gd * xh[c]; }
      s1 *= (1.f / D); s2 *= (1.f / D);
      if (leader) bulk_wait_read<0>();           // the previous tile's dm store has read the staging
      named_sync(1, 128);
      __nv_bfloat16* dmr = sDM + r * 64;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const int off = (c8 ^ (r & 7)) << 3;
        const uint4 dv4 = *reinterpret_cast<const uint4*>(orow + off);
        const uint32_t* dw4 = reinterpret_cast<const uint32_t*>(&dv4);
        uint4 o;
        uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int c = c8 * 8 + 2 * k;
          const float2 d2 = bf2f(dw4[k]);
          ow[k] = pack2(rstd * (gam[c] * dy[c] - s1 - xh[c] * s2) + d2.x, rstd * (gam[c + 1] * dy[c + 1] - s1 - xh[c + 1] * s2) + d2.y);
        }
        *reinterpret_cast<uint4*>(dmr + off) = o;
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(xe);                 // x / dout are read
      // dgamma / dbeta: column sums over the warp's 32 rows by a reduce-scatter (lane l ends with columns l and l + 32)
#pragma unroll
      for (int c = 0; c < D; ++c) xh[c] *= dy[c];               // xh now holds dy . xhat
#pragma unroll
      for (int off = 16; off >= 1; off >>= 1) {
#pragma unroll
        for (int k = 0; k < off; ++k) {
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            float* a = xh + hh * 32;
            float* d_ = dy + hh * 32;
            const bool up = (lane & off) != 0;
            const float sa_ = up ? a[k] : a[k + off], ka = up ? a[k + off] : a[k];
            const float sd = up ? d_[k] : d_[k + off], kd = up ? d_[k + off] : d_[k];
            a[k] = ka + __shfl_xor_sync(0xffffffffu, sa_, off);
            d_[k] = kd + __shfl_xor_sync(0xffffffffu, sd, off);
          }
        }
      }
      dga0 += xh[0]; dga1 += xh[32]; dbe0 += dy[0]; dbe1 += dy[32];
      named_sync(1, 128);
      if (leader) { store_2d(&dmmap, sDM, 0, r0); bulk_commit(); }
    }
    if (leader) bulk_wait<0>();
    // per-warp partials of dgamma / dbeta, folded on the host with the CTA's other warps
    float* ln = DLN + ((size_t)blockIdx.x * 4 + q) * 2 * D;
    ln[lane] = dga0; ln[lane + 32] = dga1; ln[D + lane] = dbe0; ln[D + lane + 32] = dbe1;
    // dWgv^T slab: lane half 0 = column blocks 0-3, half 1 = 4-7; row d = q * 16 + (lane & 15)
    wait(wdone, 0);
    tc_fence_after();
    const int half = lane >> 4, d = q * 16 + (lane & 15);
    float* slab = DW + (size_t)blockIdx.x * D * KD + (size_t)d * KD + half * 256;
#pragma unroll 1
    for (int c0 = 0; c0 < 256; c0 += 32) {
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
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace pb2

// sum_j w[h,i,j] dw[h,i,j] for every (i, h), written [N][H]: one warp per (h, i) row
__global__ void __launch_bounds__(256) pair_sdot_sm100(const __nv_bfloat16* __restrict__ W16, const float* __restrict__ DW, float* __restrict__ SD, int N) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;   // row = h * N + i
  if (row >= H * N) return;
  const __nv_bfloat16* w = W16 + (size_t)row * N;
  const float* d = DW + (size_t)row * N;
  float sd = 0.f;
  for (int j = lane * 4; j < N; j += 128) {
    const float4 dv = __ldg(reinterpret_cast<const float4*>(d + j));
    const uint2 wv = __ldg(reinterpret_cast<const uint2*>(w + j));
    const float2 w0 = bf2f(wv.x), w1 = bf2f(wv.y);
    sd += w0.x * dv.x + w0.y * dv.y + w1.x * dv.z + w1.y * dv.w;
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) sd += __shfl_xor_sync(0xffffffffu, sd, off);
  if (lane == 0) SD[(size_t)(row % N) * H + row / N] = sd;
}

template <typename WT_>
__global__ void __launch_bounds__(pb2::THREADS, 1) pair_bwd_sm100(
    int N, int ntiles, float eps,
    const __grid_constant__ CUtensorMap zmap,     // z [N*N][128], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap dzmap,    // dz, same
    const __grid_constant__ CUtensorMap wmap,     // w16 [H][N][N] as (N j, N i, H), box (128, 1, 8)
    const __grid_constant__ CUtensorMap dwmap,    // dw fp32, same
    const __grid_constant__ CUtensorMap sdmap,    // the row sums [N][H] fp32, box (8, 1)
    const float* __restrict__ LNW, const float* __restrict__ LNB,
    const WT_* __restrict__ WB,                   // proj_z weight [H][128]
    float* __restrict__ PWB,                      // [grid][H][128]
    float* __restrict__ PLN) {                    // [grid][8 warps][2][128]
  using namespace pb2;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sZ = reinterpret_cast<__nv_bfloat16*>(smb);   // [2][ZT]
  __nv_bfloat16* sZN = sZ + 2 * ZT;                              // [2][ZT] zn, the dWb GEMM's A operand
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
  for (int v = tid; v < DZ; v += THREADS) { sLN[v] = LNW[v]; sLN[DZ + v] = LNB[v]; }
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
        expect_tx(zf + b, ZT * 2 + WT * 2 + WT * 4 + H * 4);
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
    float ag[2] = {0.f, 0.f}, ab[2] = {0.f, 0.f};                // this lane's 2 columns of dgamma / dbeta
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
        const float* dws = reinterpret_cast<const float*>(wd + WT * 2);
        const float* sdot = reinterpret_cast<const float*>(wd + WT * 6);
        if (lt >= 2) wait(ae + b, ((lt >> 1) - 1) & 1);
        __nv_bfloat16* dbt = sDB + b * DBT + (j >> 6) * HP * 64;
#pragma unroll
        for (int h = 0; h < H; ++h)              // bf16: the stock proj_z backward is a bf16 GEMM
          dbt[sw128(h, j & 63)] = __float2bfloat16_rn(__bfloat162float(ws[h * BJ + j]) * (dws[h * BJ + j] - sdot[h]));
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) arrive(dbf + b);
      }
      // ---- LayerNorm recompute on this thread's 64 channels ----
      const __nv_bfloat16* zr = sZ + b * ZT + hf * BJ * 64 + j * 64;
      float xh[64];
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const uint4 u = *reinterpret_cast<const uint4*>(zr + ((c8 ^ (j & 7)) << 3));
        const float2 a0 = bf2f(u.x), a1 = bf2f(u.y), a2 = bf2f(u.z), a3 = bf2f(u.w);
        xh[c8 * 8 + 0] = a0.x; xh[c8 * 8 + 1] = a0.y; xh[c8 * 8 + 2] = a1.x; xh[c8 * 8 + 3] = a1.y;
        xh[c8 * 8 + 4] = a2.x; xh[c8 * 8 + 5] = a2.y; xh[c8 * 8 + 6] = a3.x; xh[c8 * 8 + 7] = a3.y;
      }
      __syncwarp();
      if (lane == 0) arrive(ze + b);
      float sm = 0.f;
#pragma unroll
      for (int c = 0; c < 64; ++c) sm += xh[c];
      float tot, dummy;
      xchg(0, sm, 0.f, tot, dummy);
      const float mean = tot * (1.f / DZ);
      float vv = 0.f;
#pragma unroll
      for (int c = 0; c < 64; ++c) { xh[c] -= mean; vv += xh[c] * xh[c]; }
      xchg(1, vv, 0.f, tot, dummy);
      const float rstd = 1.f / sqrtf(tot * (1.f / DZ) + eps);
#pragma unroll
      for (int c = 0; c < 64; ++c) xh[c] *= rstd;
      // ---- zn into the dWb GEMM's A operand ----
      if (lt >= 2) wait(ae + b, ((lt >> 1) - 1) & 1);
      __nv_bfloat16* znr = sZN + b * ZT + hf * BJ * 64 + j * 64;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        uint4 o;
        uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int c = c8 * 8 + 2 * k, d = hf * 64 + c;
          ow[k] = pack2(fmaf(xh[c], sLN[d], sLN[DZ + d]), fmaf(xh[c + 1], sLN[d + 1], sLN[DZ + d + 1]));
        }
        *reinterpret_cast<uint4*>(znr + ((c8 ^ (j & 7)) << 3)) = o;
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(znf + b);
      // ---- dzn from TMEM, the LayerNorm backward -> dz ----
      wait(dznf + b, (lt >> 1) & 1);
      tc_fence_after();
      float dzn[64];
      tmem_ld32(tmem_at(tmem + COL_DZN + b * DZ, q * 32, hf * 64), dzn);
      tmem_ld32(tmem_at(tmem + COL_DZN + b * DZ, q * 32, hf * 64 + 32), dzn + 32);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(dzne + b);
      float gs = 0.f, gx = 0.f;
#pragma unroll
      for (int c = 0; c < 64; ++c) { const float gg = dzn[c] * sLN[hf * 64 + c]; gs += gg; gx += gg * xh[c]; }
      xchg(0, gs, gx, gs, gx);                   // (exchange slot 0 is free again: the barrier inside xchg(1) separated its uses)
      gs *= (1.f / DZ); gx *= (1.f / DZ);
      if (warp == 0 && lane == 0) bulk_wait_read<0>();   // the previous tile's dz store has read the staging
      named_sync(1, 256);
      __nv_bfloat16* orow = sOut + hf * BJ * 64 + j * 64;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        uint4 o;
        uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int c = c8 * 8 + 2 * k, d = hf * 64 + c;
          ow[k] = pack2(rstd * (dzn[c] * sLN[d] - gs - xh[c] * gx), rstd * (dzn[c + 1] * sLN[d + 1] - gs - xh[c + 1] * gx));
        }
        *reinterpret_cast<uint4*>(orow + ((c8 ^ (j & 7)) << 3)) = o;
      }
      fence_proxy_async();
      named_sync(1, 256);
      if (warp == 0 && lane == 0) { for (int h = 0; h < 2; ++h) store_2d(&dzmap, sOut + h * BJ * 64, h * 64, i * N + j0); bulk_commit(); }
      // ---- dgamma += dzn . xhat, dbeta += dzn: column sums over the warp's 32 rows, a 32-lane reduce-scatter ----
#pragma unroll
      for (int c = 0; c < 64; ++c) xh[c] *= dzn[c];
#pragma unroll
      for (int st = 0; st < 5; ++st) {
        const int off = 16 >> st, half = 32 >> st;
        const bool up = (lane & off) != 0;
#pragma unroll
        for (int k = 0; k < half; ++k) {
          const float sa_ = up ? xh[k] : xh[k + half], ka = up ? xh[k + half] : xh[k];
          const float sb_ = up ? dzn[k] : dzn[k + half], kb = up ? dzn[k + half] : dzn[k];
          xh[k] = ka + __shfl_xor_sync(0xffffffffu, sa_, off);
          dzn[k] = kb + __shfl_xor_sync(0xffffffffu, sb_, off);
        }
      }
      ag[0] += xh[0]; ag[1] += xh[1]; ab[0] += dzn[0]; ab[1] += dzn[1];
    }
    if (warp == 0 && lane == 0) bulk_wait<0>();
    // this warp's dgamma / dbeta partials: columns hf * 64 + lane * 2 + k (the reduce-scatter's order), one writer each
    {
      const int c0 = hf * 64 + ((lane >> 4) & 1) * 32 + ((lane >> 3) & 1) * 16 + ((lane >> 2) & 1) * 8 + ((lane >> 1) & 1) * 4 + (lane & 1) * 2;
      float* pl = PLN + ((size_t)blockIdx.x * 8 + warp) * 2 * DZ;
      pl[c0] = ag[0]; pl[c0 + 1] = ag[1]; pl[DZ + c0] = ab[0]; pl[DZ + c0 + 1] = ab[1];
    }
    // this CTA's dWb partial: TMEM lane = d, columns = heads
    if (hf == 0) {
      wait(done, 0);
      tc_fence_after();
      float v[8];
      tmem_ld8(tmem_at(tmem + COL_DWB, q * 32, 0), v);
      tmem_wait_ld();
#pragma unroll
      for (int h = 0; h < H; ++h) PWB[((size_t)blockIdx.x * H + h) * DZ + q * 32 + lane] = v[h];
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 9) tmem_dealloc(tmem, 512);
}
}  // namespace

// m [S, N, 64] bf16 -> (v head-major [H, N, S*C], y [S, N, 64])
std::vector<torch::Tensor> ln_vg(torch::Tensor m, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wv, double eps) {
  using namespace lv;
  TORCH_CHECK(m.is_cuda() && m.scalar_type() == torch::kBFloat16 && m.is_contiguous() && m.dim() == 3 && m.size(2) == D, "m: [S, N, 64] bf16");
  TORCH_CHECK(wv.scalar_type() == torch::kBFloat16 && wv.is_contiguous() && wv.size(0) == HC && wv.size(1) == D, "wv: [256, 64] bf16");
  TORCH_CHECK(lnw.scalar_type() == lnb.scalar_type() && lnw.is_contiguous() && lnb.is_contiguous() && lnw.numel() == D, "LN affine [64]");
  const long S = m.size(0), N = m.size(1);
  TORCH_CHECK(S % BS == 0, "S must be a multiple of ", BS);
  auto y = torch::empty_like(m);
  auto v = torch::empty({H, N, S * C}, m.options());
  CUtensorMap xm = make_map<3>(m.data_ptr(), {(uint64_t)D, (uint64_t)N, (uint64_t)S}, {(uint64_t)D, (uint64_t)N * D}, {64, 1, BS}, CU_TENSOR_MAP_SWIZZLE_128B, "m");
  CUtensorMap ym = make_map<3>(y.data_ptr(), {(uint64_t)D, (uint64_t)N, (uint64_t)S}, {(uint64_t)D, (uint64_t)N * D}, {64, 1, BS}, CU_TENSOR_MAP_SWIZZLE_128B, "y");
  CUtensorMap wm = make_map<2>(wv.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, HC}, CU_TENSOR_MAP_SWIZZLE_128B, "wv");
  CUtensorMap vm = make_map<3>(v.data_ptr(), {(uint64_t)C, (uint64_t)S, (uint64_t)H * N}, {(uint64_t)C, (uint64_t)S * C}, {C, BS, 1}, CU_TENSOR_MAP_SWIZZLE_64B, "v");
  const int ntiles = (int)(N * (S / BS));
  const int grid = std::min(ntiles, num_sms(m.device().index()));
  auto st = at::cuda::getCurrentCUDAStream();
  if (lnw.scalar_type() == torch::kFloat32) {
    static bool a = false;
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(ln_vg_sm100<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
    ln_vg_sm100<float><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, xm, ym, wm, vm, lnw.data_ptr<float>(), lnb.data_ptr<float>());
  } else {
    TORCH_CHECK(lnw.scalar_type() == torch::kBFloat16, "LN affine: fp32 or bf16");
    static bool a = false;
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(ln_vg_sm100<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
    ln_vg_sm100<__nv_bfloat16><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, xm, ym, wm, vm,
        reinterpret_cast<const __nv_bfloat16*>(lnw.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(lnb.data_ptr<at::BFloat16>()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {v, y};
}

// w16 [H, N, N] (softmax over the last dim), vhm [H, N, S*C], y / msa [S, N, 64], wg [HC, 64], wo [64, HC] -> (out, o)
std::vector<torch::Tensor> pwa_fwd(torch::Tensor w16, torch::Tensor vhm, torch::Tensor y, torch::Tensor wgw, torch::Tensor wow, torch::Tensor msa,
                                   bool save_o, c10::optional<torch::Tensor> dmask, double dscale) {
  using namespace pf;
  TORCH_CHECK(w16.is_contiguous() && w16.scalar_type() == torch::kBFloat16 && w16.dim() == 3 && w16.size(0) == H, "w16: [H, N, N] bf16");
  const long N = w16.size(1);
  TORCH_CHECK(vhm.is_contiguous() && vhm.size(0) == H && vhm.size(1) == N && vhm.size(2) % C == 0, "vhm: [H, N, S*C]");
  const long S = vhm.size(2) / C;
  TORCH_CHECK(N % BI == 0, "N must be a multiple of 128");
  TORCH_CHECK(y.is_contiguous() && y.sizes() == torch::IntArrayRef({S, N, D}) && msa.is_contiguous() && msa.sizes() == torch::IntArrayRef({S, N, D}), "y, msa: [S, N, 64]");
  TORCH_CHECK(wgw.is_contiguous() && wgw.sizes() == torch::IntArrayRef({HC, D}) && wow.is_contiguous() && wow.sizes() == torch::IntArrayRef({D, HC}), "wg [256, 64], wo [64, 256]");
  auto out = torch::empty({S, N, D}, y.options());
  auto o = save_o ? torch::empty({S, N, HC}, y.options()) : torch::empty({0}, y.options());
  CUtensorMap wm = make_map<2>(w16.data_ptr(), {(uint64_t)N, (uint64_t)(H * N)}, {(uint64_t)N}, {JC, BI}, CU_TENSOR_MAP_SWIZZLE_128B, "w");
  CUtensorMap vm = make_map<3>(vhm.data_ptr(), {(uint64_t)C, (uint64_t)(H * N), (uint64_t)S}, {(uint64_t)(S * C), (uint64_t)C}, {C, JC, BS}, CU_TENSOR_MAP_SWIZZLE_64B, "v");
  CUtensorMap ym = make_map<3>(y.data_ptr(), {(uint64_t)D, (uint64_t)N, (uint64_t)S}, {(uint64_t)D, (uint64_t)N * D}, {64, BI, 1}, CU_TENSOR_MAP_SWIZZLE_128B, "y");
  CUtensorMap gm = make_map<2>(wgw.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, C}, CU_TENSOR_MAP_SWIZZLE_128B, "wg");
  CUtensorMap wom = make_map<2>(wow.data_ptr(), {(uint64_t)HC, (uint64_t)D}, {(uint64_t)HC}, {C, 64}, CU_TENSOR_MAP_SWIZZLE_64B, "wo");
  const __nv_bfloat16* dmp = nullptr;
  if (dmask.has_value() && dmask->numel()) {
    TORCH_CHECK(dmask->scalar_type() == torch::kBFloat16 && dmask->is_contiguous() && dmask->numel() == N * D, "dmask: [N, 64] bf16");
    dmp = reinterpret_cast<const __nv_bfloat16*>(dmask->data_ptr<at::BFloat16>());
  }
  const int ntiles = (int)((N / BI) * ((S + BS - 1) / BS));
  const int grid = std::min(ntiles, num_sms(y.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_fwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  pwa_fwd_sm100<<<grid, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>((int)N, (int)S, ntiles, save_o ? 1 : 0, wm, vm, ym, gm, wom,
      save_o ? reinterpret_cast<__nv_bfloat16*>(o.data_ptr<at::BFloat16>()) : nullptr,
      reinterpret_cast<const __nv_bfloat16*>(msa.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), dmp, (float)dscale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, o};
}

// o [S, N, HC] (saved by the forward), y / dres [S, N, 64], wg [HC, 64], wot = Wo^T [HC, 64], dgv [S, N, 512] (dgp -> [..., :256])
// -> (do head-major [H, N, S*C], dWo fp32 [64, HC])
std::vector<torch::Tensor> pwa_glue(torch::Tensor o, torch::Tensor y, torch::Tensor dres, torch::Tensor wgw, torch::Tensor wot, torch::Tensor dgv,
                                    c10::optional<torch::Tensor> dmask, double dscale) {
  using namespace gl;
  TORCH_CHECK(y.is_contiguous() && y.dim() == 3 && y.size(2) == D && dres.is_contiguous() && dres.sizes() == y.sizes(), "y, dres: [S, N, 64]");
  const long S = y.size(0), N = y.size(1);
  TORCH_CHECK(N % BI == 0 && S % BS == 0, "N must be a multiple of 128 and S even");
  TORCH_CHECK(o.is_contiguous() && o.sizes() == torch::IntArrayRef({S, N, HC}), "o: [S, N, 256]");
  TORCH_CHECK(dgv.is_contiguous() && dgv.sizes() == torch::IntArrayRef({S, N, 2 * HC}), "dgv: [S, N, 512]");
  TORCH_CHECK(wgw.is_contiguous() && wgw.sizes() == torch::IntArrayRef({HC, D}) && wot.is_contiguous() && wot.sizes() == torch::IntArrayRef({HC, D}), "wg, wot: [256, 64]");
  auto dO = torch::empty({H, N, S * C}, y.options());
  const int ntiles = (int)((N / BI) * (S / BS));
  const int grid = std::min(ntiles, num_sms(y.device().index()));
  auto dwo = torch::empty({grid, (long)D, (long)HC}, y.options().dtype(torch::kFloat32));
  CUtensorMap om = make_map<3>(o.data_ptr(), {(uint64_t)HC, (uint64_t)N, (uint64_t)S}, {(uint64_t)HC, (uint64_t)N * HC}, {C, BI, BS}, CU_TENSOR_MAP_SWIZZLE_64B, "o");
  auto nat = [&](void* p, const char* w) { return make_map<3>(p, {(uint64_t)D, (uint64_t)N, (uint64_t)S}, {(uint64_t)D, (uint64_t)N * D}, {64, BI, 1}, CU_TENSOR_MAP_SWIZZLE_128B, w); };
  CUtensorMap rm = nat(dres.data_ptr(), "dres"), ym = nat(y.data_ptr(), "y");
  CUtensorMap gm = make_map<2>(wgw.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, C}, CU_TENSOR_MAP_SWIZZLE_128B, "wg");
  CUtensorMap wtm = make_map<2>(wot.data_ptr(), {(uint64_t)D, (uint64_t)HC}, {(uint64_t)D}, {64, C}, CU_TENSOR_MAP_SWIZZLE_128B, "wot");
  CUtensorMap dom = make_map<2>(dO.data_ptr(), {(uint64_t)(S * C), (uint64_t)(H * N)}, {(uint64_t)(S * C)}, {64, BI}, CU_TENSOR_MAP_SWIZZLE_128B, "do");
  CUtensorMap dgm = make_map<3>(dgv.data_ptr(), {(uint64_t)HC, (uint64_t)N, (uint64_t)S}, {(uint64_t)(2 * HC), (uint64_t)N * 2 * HC}, {C, BI, BS}, CU_TENSOR_MAP_SWIZZLE_64B, "dgp");
  const __nv_bfloat16* dmp = nullptr;
  if (dmask.has_value() && dmask->numel()) {
    TORCH_CHECK(dmask->scalar_type() == torch::kBFloat16 && dmask->is_contiguous() && dmask->numel() == N * D, "dmask: [N, 64] bf16");
    dmp = reinterpret_cast<const __nv_bfloat16*>(dmask->data_ptr<at::BFloat16>());
  }
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_glue_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  pwa_glue_sm100<<<grid, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>((int)N, (int)S, ntiles, om, rm, ym, gm, wtm, dom, dgm, dmp, (float)dscale, dwo.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, colsum(dwo)};                                          // [64, HC], the CTAs' slabs in split order
}

// dv[h][j][(s,c)] = sum_i w16[h][i][j] do[h][i][(s,c)] into dgv[..., 256:] (natural [S][N][512])
void pwa_plain(torch::Tensor w16, torch::Tensor dO, torch::Tensor dgv) {
  using namespace pl;
  TORCH_CHECK(w16.is_contiguous() && w16.dim() == 3 && w16.size(0) == H, "w16: [H, N, N]");
  const long N = w16.size(1);
  TORCH_CHECK(dO.is_contiguous() && dO.size(0) == H && dO.size(1) == N && dO.size(2) % C == 0, "do: [H, N, S*C]");
  const long S = dO.size(2) / C;
  TORCH_CHECK(N % BJ == 0, "N must be a multiple of 128");
  TORCH_CHECK(dgv.is_contiguous() && dgv.sizes() == torch::IntArrayRef({S, N, 2 * HC}), "dgv: [S, N, 512]");
  CUtensorMap wm = make_map<2>(w16.data_ptr(), {(uint64_t)N, (uint64_t)(H * N)}, {(uint64_t)N}, {64, KC}, CU_TENSOR_MAP_SWIZZLE_128B, "w");
  CUtensorMap dm = make_map<2>(dO.data_ptr(), {(uint64_t)(S * C), (uint64_t)(H * N)}, {(uint64_t)(S * C)}, {64, KC}, CU_TENSOR_MAP_SWIZZLE_128B, "do");
  CUtensorMap vm = make_map<3>(dgv.data_ptr(), {(uint64_t)(2 * HC), (uint64_t)N, (uint64_t)S}, {(uint64_t)(2 * HC), (uint64_t)N * 2 * HC}, {C, BJ, BS / 2}, CU_TENSOR_MAP_SWIZZLE_64B, "dv");
  const int ntiles = (int)(H * (N / BJ) * ((S + BS - 1) / BS));
  const int grid = std::min(ntiles, num_sms(w16.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_plain_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  pwa_plain_sm100<<<grid, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>((int)N, (int)S, ntiles, wm, dm, vm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dgv [M, 512] (dgp | dv), y, x, dout [M, 64], wgvT = [Wg; Wv]^T [64, 512] -> (dm, dWgv [512, 64], dgamma, dbeta) fp32 weights
std::vector<torch::Tensor> dgv_bwd(torch::Tensor dgv, torch::Tensor y, torch::Tensor x, torch::Tensor dout, torch::Tensor wgvT, torch::Tensor lnw, double eps) {
  using namespace dv;
  const long M = x.numel() / D;
  TORCH_CHECK(dgv.is_contiguous() && dgv.scalar_type() == torch::kBFloat16 && dgv.numel() == M * KD, "dgv: [M, 512] bf16");
  for (const auto* t : {&y, &x, &dout}) TORCH_CHECK(t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == M * D, "y/x/dout: [M, 64] bf16");
  TORCH_CHECK(wgvT.is_contiguous() && wgvT.size(0) == D && wgvT.size(1) == KD, "wgvT: [64, 512] bf16");
  TORCH_CHECK(M % BM == 0, "rows must be a multiple of 128");
  auto dm = torch::empty_like(x);
  const int ntiles = (int)(M / BM);
  const int grid = std::min(ntiles, num_sms(x.device().index()));
  auto dw = torch::empty({grid, (long)D, (long)KD}, x.options().dtype(torch::kFloat32));
  auto dln = torch::empty({grid * 4, 2, (long)D}, x.options().dtype(torch::kFloat32));
  auto m2 = [&](const torch::Tensor& tt, long cols, const char* w) {
    return make_map<2>(tt.data_ptr(), {(uint64_t)cols, (uint64_t)(tt.numel() / cols)}, {(uint64_t)cols}, {64, (uint32_t)(cols == D && tt.numel() == D * KD ? 64 : BM)}, CU_TENSOR_MAP_SWIZZLE_128B, w);
  };
  CUtensorMap gm = m2(dgv, KD, "dgv");
  CUtensorMap wm = make_map<2>(wgvT.data_ptr(), {(uint64_t)KD, (uint64_t)D}, {(uint64_t)KD}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "wgvT");
  CUtensorMap ym = m2(y, D, "y"), xm = m2(x, D, "x"), om = m2(dout, D, "dout"), dmm = m2(dm, D, "dm");
  auto st = at::cuda::getCurrentCUDAStream();
  if (lnw.scalar_type() == torch::kFloat32) {
    static bool a = false;
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(dgv_bwd_sm100<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
    dgv_bwd_sm100<float><<<grid, THREADS, SMEM, st>>>((int)M, ntiles, (float)eps, gm, wm, ym, xm, om, dmm, lnw.data_ptr<float>(), dw.data_ptr<float>(), dln.data_ptr<float>());
  } else {
    static bool a = false;
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(dgv_bwd_sm100<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
    dgv_bwd_sm100<__nv_bfloat16><<<grid, THREADS, SMEM, st>>>((int)M, ntiles, (float)eps, gm, wm, ym, xm, om, dmm,
        reinterpret_cast<const __nv_bfloat16*>(lnw.data_ptr<at::BFloat16>()), dw.data_ptr<float>(), dln.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto dwt = colsum(dw);
  auto ln = colsum(dln).view({2, (long)D});
  return {dm, dwt.t(), ln[0], ln[1]};
}

// z [N, N, 128] bf16, w16 [H, N, N] bf16 (the forward's softmax), dw [H, N, N] fp32 -> (dz, dWb [H,128], dgamma, dbeta) fp32
std::vector<torch::Tensor> pair_bwd(torch::Tensor z, torch::Tensor w16, torch::Tensor dw, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor wb) {
  using namespace pb2;
  TORCH_CHECK(z.is_contiguous() && z.scalar_type() == torch::kBFloat16 && z.dim() == 3 && z.size(2) == DZ, "z: [N, N, 128] bf16");
  const long N = z.size(0);
  TORCH_CHECK(N % BJ == 0, "N must be a multiple of 128");
  TORCH_CHECK(w16.is_contiguous() && w16.numel() == H * N * N && dw.is_contiguous() && dw.scalar_type() == torch::kFloat32 && dw.numel() == H * N * N, "w16, dw: [H, N, N]");
  TORCH_CHECK(lnw.scalar_type() == torch::kFloat32 && lnb.scalar_type() == torch::kFloat32 && lnw.is_contiguous() && lnb.is_contiguous(), "LN affine: fp32 [128]");
  TORCH_CHECK(wb.is_contiguous() && wb.numel() == H * DZ, "wb: [8, 128]");
  auto dz = torch::empty_like(z);
  const int ntiles = (int)(N * (N / BJ));
  const int grid = std::min(ntiles, num_sms(z.device().index()));
  auto pwb = torch::empty({grid, (long)H, (long)DZ}, z.options().dtype(torch::kFloat32));
  auto pln = torch::zeros({grid * 8, 2, (long)DZ}, z.options().dtype(torch::kFloat32));   // a warp writes only its channel half
  CUtensorMap zm = make_map<2>(z.data_ptr(), {(uint64_t)DZ, (uint64_t)(N * N)}, {(uint64_t)DZ}, {64, BJ}, CU_TENSOR_MAP_SWIZZLE_128B, "z");
  CUtensorMap dzm = make_map<2>(dz.data_ptr(), {(uint64_t)DZ, (uint64_t)(N * N)}, {(uint64_t)DZ}, {64, BJ}, CU_TENSOR_MAP_SWIZZLE_128B, "dz");
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* w16p = reinterpret_cast<const __nv_bfloat16*>(w16.data_ptr<at::BFloat16>());
  auto sd = torch::empty({N, (long)H}, z.options().dtype(torch::kFloat32));
  pair_sdot_sm100<<<(int)((H * N + 7) / 8), 256, 0, st>>>(w16p, dw.data_ptr<float>(), sd.data_ptr<float>(), (int)N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  CUtensorMap wm = make_map<3>(w16.data_ptr(), {(uint64_t)N, (uint64_t)N, (uint64_t)H}, {(uint64_t)N, (uint64_t)(N * N)}, {BJ, 1, H}, CU_TENSOR_MAP_SWIZZLE_NONE, "w16");
  CUtensorMap dwm = make_map<3>(dw.data_ptr(), {(uint64_t)N, (uint64_t)N, (uint64_t)H}, {(uint64_t)N, (uint64_t)(N * N)}, {BJ, 1, H}, CU_TENSOR_MAP_SWIZZLE_NONE, "dw",
                                CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  CUtensorMap sdm = make_map<2>(sd.data_ptr(), {(uint64_t)H, (uint64_t)N}, {(uint64_t)H}, {H, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "sdot", CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  if (wb.scalar_type() == torch::kFloat32) {
    static bool a = false;
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bwd_sm100<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
    pair_bwd_sm100<float><<<grid, THREADS, SMEM, st>>>((int)N, ntiles, (float)eps, zm, dzm, wm, dwm, sdm, lnw.data_ptr<float>(), lnb.data_ptr<float>(),
        wb.data_ptr<float>(), pwb.data_ptr<float>(), pln.data_ptr<float>());
  } else {
    static bool a = false;
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(pair_bwd_sm100<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); a = true; }
    pair_bwd_sm100<__nv_bfloat16><<<grid, THREADS, SMEM, st>>>((int)N, ntiles, (float)eps, zm, dzm, wm, dwm, sdm, lnw.data_ptr<float>(), lnb.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(wb.data_ptr<at::BFloat16>()), pwb.data_ptr<float>(), pln.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto ln = colsum(pln).view({2, (long)DZ});
  return {dz, colsum(pwb), ln[0], ln[1]};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_vg", &ln_vg, "sm100 LayerNorm + value projection: (v head-major [H,N,S*C], y [S,N,64])",
        py::arg("m"), py::arg("lnw"), py::arg("lnb"), py::arg("wv"), py::arg("eps") = 1e-5);
  m.def("pwa_fwd", &pwa_fwd, "sm100 PWA forward: contraction + gate + out-projection + residual (+ dropout), optionally keeping o",
        py::arg("w16"), py::arg("vhm"), py::arg("y"), py::arg("wg"), py::arg("wo"), py::arg("msa"), py::arg("save_o") = false,
        py::arg("dmask") = py::none(), py::arg("dscale") = 1.0);
  m.def("pair_bwd", &pair_bwd, "sm100 pair-side backward: softmax-bwd -> proj_z-bwd -> LN_z-bwd -> (dz, dWb, dgamma_z, dbeta_z)",
        py::arg("z"), py::arg("w16"), py::arg("dw"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("wb"));
  m.def("dgv_bwd", &dgv_bwd, "sm100 fused dgv -> (dm = LN_bwd(dgv . Wgv) + dout, dWgv, dgamma, dbeta)",
        py::arg("dgv"), py::arg("y"), py::arg("x"), py::arg("dout"), py::arg("wgvT"), py::arg("lnw"), py::arg("eps") = 1e-5);
  m.def("pwa_plain", &pwa_plain, "sm100 PWA dv = w^T . do into dgv[..., 256:]", py::arg("w16"), py::arg("do"), py::arg("dgv"));
  m.def("pwa_glue", &pwa_glue, "sm100 PWA backward glue from the saved o: (do head-major, dWo fp32 [64][256]); dgp into dgv[..., :256]",
        py::arg("o"), py::arg("y"), py::arg("dres"), py::arg("wg"), py::arg("wot"), py::arg("dgv"), py::arg("dmask") = py::none(), py::arg("dscale") = 1.0);
}
