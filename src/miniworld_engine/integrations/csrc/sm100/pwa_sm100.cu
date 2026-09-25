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
//   warps 2-5: drain -- u = o / (1 + exp(-g)) -> bf16 A tiles of the out GEMM (and o -> its TMA box); at the end of
//              the tile: bf16(out), dropout, + residual, straight to global (a warp's rows are 4 KiB contiguous)
namespace pf {
constexpr int BI = 128, BS = 3, NO = BS * C;              // o / gate columns per head: 96
constexpr int JC = 64;                                     // j per stage
constexpr int WTL = BI * JC, VTL = JC * NO;               // W chunk [128 i][64 j] (128B swz), v chunk [3 s][64 j][32 c] (64B swz)
constexpr int STAGE = WTL + VTL;                           // 28 KiB
constexpr int NST = 3;
constexpr int YT = BS * BI * D;                            // y rows [3 s][128 i][64]
constexpr int WH = C * D + D * C;                          // Wg_h [32 c][64 d] (128B swz) | Wo_h [64 d][32 c] (64B swz)
constexpr int UT = BS * BI * C;                            // u (or o) tile [3 s][128 i][32], 64B swizzle
constexpr int THREADS = 192;
constexpr int SMEM = 1024 + (NST * STAGE + YT + 2 * WH + 2 * UT + UT) * 2 + 512;
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
    const __grid_constant__ CUtensorMap savemap,  // o [S][N][HC] as (32 c, N, S), box (32, 128, 3), 64B swizzle
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
  __nv_bfloat16* sU = sWH + 2 * WH;              // [2][UT]
  __nv_bfloat16* sOS = sU + 2 * UT;              // o save staging [UT]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sOS + UT);
  uint64_t* full = bars;             // [NST]
  uint64_t* empty = full + NST;      // [NST]
  uint64_t* whf = empty + NST;       // [2]
  uint64_t* whe = whf + 2;           // [2]  the head's out GEMM retired (its gate GEMM before it)
  uint64_t* yf = whe + 2;            // [1]
  uint64_t* ye = yf + 1;             // [1]  the tile's last gate GEMM retired
  uint64_t* accf = ye + 1;           // [2]  o_h (buffer gh & 1) and gate_h complete
  uint64_t* acce = accf + 2;         // [2]  count 4: o buffer drained
  uint64_t* ge = acce + 2;           // [1]  count 4: gate drained
  uint64_t* uf = ge + 1;             // [2]  count 4: u_h written
  uint64_t* ue = uf + 2;             // [2]  the out GEMM reading u_h retired
  uint64_t* outf = ue + 2;           // [1]  the tile's out accumulator complete
  uint64_t* oute = outf + 1;         // [1]  count 4: drained
  uint32_t* tslot = reinterpret_cast<uint32_t*>(oute + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { bar_init(full + k, 1); bar_init(empty + k, 1); }
    for (int k = 0; k < 2; ++k) { bar_init(whf + k, 1); bar_init(whe + k, 1); bar_init(accf + k, 1); bar_init(acce + k, 4);
                                  bar_init(uf + k, 4); bar_init(ue + k, 1); }
    bar_init(yf, 1); bar_init(ye, 1); bar_init(ge, 4); bar_init(outf, 1); bar_init(oute, 4);
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
            mma_ss(tmem + COL_OUT + si * D, desc_k64(sU + b * UT + si * BI * C + ks * 16), desc_k64(wo + ks * 16), ID_OUT, (hp | ks) ? 1u : 0u);
        mma_commit(ue + b);
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
  } else {
    const int q = warp & 3, r = q * 32 + lane;   // TMEM lane = i row of the tile
    const bool leader = (warp == 2 && lane == 0);
    int gh = 0, lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
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
        if (gh >= 2) wait(ue + b, ((gh >> 1) - 1) & 1);        // the out GEMM of head gh - 2 has read u buffer b
        if (save_o) { if (leader) bulk_wait_read<0>(); named_sync(1, 128); }
        __nv_bfloat16* ub = sU + b * UT;
#pragma unroll
        for (int si = 0; si < BS; ++si) {
          float o[C];
          tmem_ld32(tmem_at(tmem + COL_O + b * NO, q * 32, si * C), o);
          tmem_wait_ld();
          if (si == BS - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + b); }
          __nv_bfloat16* ur = ub + (si * BI + r) * C;
          __nv_bfloat16* orow = sOS + (si * BI + r) * C;
#pragma unroll
          for (int c8 = 0; c8 < 4; ++c8) {
            uint4 uu, oo;
            uint32_t* uw = reinterpret_cast<uint32_t*>(&uu);
            uint32_t* ow = reinterpret_cast<uint32_t*>(&oo);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const int c = c8 * 8 + 2 * k;
              uw[k] = pack2(o[c] * gt[si * C + c], o[c + 1] * gt[si * C + c + 1]);
              ow[k] = pack2(o[c], o[c + 1]);
            }
            const int off = (c8 ^ ((r >> 1) & 3)) << 3;
            *reinterpret_cast<uint4*>(ur + off) = uu;
            if (save_o) *reinterpret_cast<uint4*>(orow + off) = oo;
          }
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) arrive(uf + b);
        if (save_o) {
          named_sync(1, 128);
          if (leader) { store_3d(&savemap, sOS, h * C, i0, s0); bulk_commit(); }
        }
      }
      // ---- the tile's output: bf16(update), dropout, + residual ----
      wait(outf, lt & 1);
      tc_fence_after();
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
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
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
  CUtensorMap svm = make_map<3>(save_o ? o.data_ptr() : out.data_ptr(), {(uint64_t)HC, (uint64_t)N, (uint64_t)S}, {(uint64_t)HC, (uint64_t)N * HC}, {C, BI, BS}, CU_TENSOR_MAP_SWIZZLE_64B, "o");
  const __nv_bfloat16* dmp = nullptr;
  if (dmask.has_value() && dmask->numel()) {
    TORCH_CHECK(dmask->scalar_type() == torch::kBFloat16 && dmask->is_contiguous() && dmask->numel() == N * D, "dmask: [N, 64] bf16");
    dmp = reinterpret_cast<const __nv_bfloat16*>(dmask->data_ptr<at::BFloat16>());
  }
  const int ntiles = (int)((N / BI) * ((S + BS - 1) / BS));
  const int grid = std::min(ntiles, num_sms(y.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_fwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  pwa_fwd_sm100<<<grid, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>((int)N, (int)S, ntiles, save_o ? 1 : 0, wm, vm, ym, gm, wom, svm,
      reinterpret_cast<const __nv_bfloat16*>(msa.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), dmp, (float)dscale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, o};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_vg", &ln_vg, "sm100 LayerNorm + value projection: (v head-major [H,N,S*C], y [S,N,64])",
        py::arg("m"), py::arg("lnw"), py::arg("lnb"), py::arg("wv"), py::arg("eps") = 1e-5);
  m.def("pwa_fwd", &pwa_fwd, "sm100 PWA forward: contraction + gate + out-projection + residual (+ dropout), optionally keeping o",
        py::arg("w16"), py::arg("vhm"), py::arg("y"), py::arg("wg"), py::arg("wo"), py::arg("msa"), py::arg("save_o") = false,
        py::arg("dmask") = py::none(), py::arg("dscale") = 1.0);
}
