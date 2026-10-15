// MSAPairWeightedAveraging, MSA side, for any (d_msa, d_hidden) with d_msa a multiple of 64, 8 heads and
// H * d_hidden in {64, 128} (sm_90a).  The value / gate / out-projection passes of the PWA block as four per-token
// kernels; the two contractions (o = w . v, dv = w^T . do, dw = do . v^T) and the weight gradients that reduce over
// every (s, n) are cuBLAS bmm / mm in the caller.
//
//   ln_vg_w    y = LN(m) (+ the row statistics), v[h][n][s*C + c] = y . Wv^T           (head-major, the bmm's layout)
//   gate_out_w out = m + drop(Wo . (o * sigmoid(y . Wg^T)))                             (o head-major from the bmm)
//   gate_bwd_w do' = dout . Wo, gate recomputed: go = g * o, do = do' * g (head-major), dgp = do' * o * g (1 - g)
//   dm_bwd_w   dy = [dgp | dv] . [Wg; Wv], LayerNorm backward + the residual -> dm, dgamma / dbeta partials;
//              also copies dv into the natural-layout [S, N, 2*HC] buffer the dWg / dWv GEMM reads
//
// One warpgroup per CTA, persistent over tiles of (token n, 64 consecutive MSA rows s): an A tile is 64 rows x K
// in 128-byte-swizzled 64-column blocks, cp.async double-buffered; the weights are staged once per CTA.  Every
// epilogue works in the wgmma accumulator layout (row = 16 * warp + lane / 4 [+ 8], two adjacent columns per thread),
// so the head-major operands are 4-byte accesses into 16-byte-or-longer contiguous runs of the [H][N][S*C] layout and
// the gated product is the next GEMM's A operand straight from registers.  S need not be a multiple of 64.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <map>
#include "wgmma_n64.cuh"

#ifndef PW_D
#define PW_D 128
#endif
#ifndef PW_C
#define PW_C 16
#endif
#ifndef PW_NWG
#define PW_NWG 1                                          // warpgroups per CTA: each walks its own tiles, the weights are shared (2 measured no faster)
#endif

namespace {
using bf16 = __nv_bfloat16;
constexpr int H = 8, C = PW_C, D = PW_D, HC = H * C, TS = 64, NWG = PW_NWG, NT = 128 * NWG;
constexpr int NKD = D / 64, NKH = HC / 64, NK2 = 2 * HC / 64, NCD = D / 64, NCH = HC / 64;
constexpr int TILE = TS * 64;                              // elements of one [64 rows][64 cols] block
static_assert(D % 64 == 0 && HC % 64 == 0 && C % 8 == 0, "d_msa must be a multiple of 64, 8 * d_hidden of 64");
static_assert(HC <= 128 && D * HC <= 16384, "register / shared budget of the wide kernels");

__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
// 16-byte chunk `chunk` of row r in a [rows][64] block of block-index kb, 128-byte swizzle (element offset)
__device__ __forceinline__ int so(int kb, int rows, int r, int chunk) { return ((kb * rows + r) * 8 + (chunk ^ (r & 7))) * 8; }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool pred) {
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ float2 unpack2(uint32_t u) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u)); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }
// barrier of one warpgroup (named barrier 1 + wgi); __syncthreads() would couple the warpgroups' tile loops
__device__ __forceinline__ void wg_sync(int wgi) { asm volatile("bar.sync %0, 128;\n" :: "r"(1 + wgi) : "memory"); }
__device__ __forceinline__ unsigned char* align1k(unsigned char* p) {
  const uint32_t b = sa(p);
  return p + ((1024u - (b & 1023u)) & 1023u);
}

// LayerNorm of a [64 rows][D] tile in place (bf16 in, fp32 statistics, bf16 out) by one warpgroup: D / 32 threads per row, 32 contiguous
// columns each; g / b are this thread's 32 gamma / beta values.  A non-null Y also receives y and STATS the row's (mean, rstd).
__device__ __forceinline__ void ln_tile(bf16* sx, int wtid, const float* g, const float* b, float eps, bf16* Y, float* STATS, int n, int s0, int S, int N) {
  constexpr int TPR = D / 32, RP = 128 / TPR, PASSES = TS / RP;
  const int tq = wtid % TPR;
#pragma unroll
  for (int pass = 0; pass < PASSES; ++pass) {
    const int row = pass * RP + wtid / TPR;
    float x[32];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int ch = tq * 4 + q;
      const uint4 u = *reinterpret_cast<const uint4*>(sx + so(ch >> 3, TS, row, ch & 7));
      const uint32_t* p = reinterpret_cast<const uint32_t*>(&u);
#pragma unroll
      for (int k = 0; k < 4; ++k) { const float2 f = unpack2(p[k]); x[q * 8 + k * 2] = f.x; x[q * 8 + k * 2 + 1] = f.y; }
    }
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 32; ++k) sum += x[k];
#pragma unroll
    for (int o = 1; o < TPR; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float mean = sum * (1.f / D);
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 32; ++k) { x[k] -= mean; ss += x[k] * x[k]; }
#pragma unroll
    for (int o = 1; o < TPR; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float rstd = rsqrtf(ss * (1.f / D) + eps);
    const bool ok = s0 + row < S;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int ch = tq * 4 + q;
      uint4 u;
      u.x = pack2(g[q * 8 + 0] * (rstd * x[q * 8 + 0]) + b[q * 8 + 0], g[q * 8 + 1] * (rstd * x[q * 8 + 1]) + b[q * 8 + 1]);
      u.y = pack2(g[q * 8 + 2] * (rstd * x[q * 8 + 2]) + b[q * 8 + 2], g[q * 8 + 3] * (rstd * x[q * 8 + 3]) + b[q * 8 + 3]);
      u.z = pack2(g[q * 8 + 4] * (rstd * x[q * 8 + 4]) + b[q * 8 + 4], g[q * 8 + 5] * (rstd * x[q * 8 + 5]) + b[q * 8 + 5]);
      u.w = pack2(g[q * 8 + 6] * (rstd * x[q * 8 + 6]) + b[q * 8 + 6], g[q * 8 + 7] * (rstd * x[q * 8 + 7]) + b[q * 8 + 7]);
      *reinterpret_cast<uint4*>(sx + so(ch >> 3, TS, row, ch & 7)) = u;
      if (ok && Y != nullptr) *reinterpret_cast<uint4*>(Y + ((long)(s0 + row) * N + n) * D + ch * 8) = u;
    }
    if (ok && tq == 0 && STATS != nullptr) *reinterpret_cast<float2*>(STATS + ((long)(s0 + row) * N + n) * 2) = make_float2(mean, rstd);
  }
}

// Output staging.  Stores in the accumulator layout are 4 bytes per lane over eight rows, half-filled 32-byte sectors scattered across
// the natural [S, N, .] layout (a third of the time of gate_bwd_w went into them).  A [64 rows][NCOL] tile is therefore staged in shared
// memory (the operand tiles' 128-byte swizzle: the 32 lanes of a fragment store hit 32 distinct banks) and written back as full 16-byte,
// row-contiguous chunks.
__device__ __forceinline__ void st_put(bf16* st, int row, int col, uint32_t v) {
  *reinterpret_cast<uint32_t*>(st + so(col >> 6, TS, row, (col >> 3) & 7) + (col & 7)) = v;
}
__device__ __forceinline__ uint4 st_get(const bf16* st, int row, int ch) { return *reinterpret_cast<const uint4*>(st + so(ch >> 3, TS, row, ch & 7)); }
// natural layout [S][N][rowlen]: the staged tile into columns [col0, col0 + NCOL) of token n's rows s0 ..
template <int NCOL>
__device__ __forceinline__ void st_flush_nat(const bf16* st, int wtid, bf16* base, int rowlen, int col0, int n, int N, int s0, int S) {
  for (int v = wtid; v < TS * (NCOL / 8); v += 128) {
    const int row = v / (NCOL / 8), ch = v % (NCOL / 8);
    if (s0 + row < S) *reinterpret_cast<uint4*>(base + ((long)(s0 + row) * N + n) * rowlen + col0 + ch * 8) = st_get(st, row, ch);
  }
}
// head-major layout [H][N][S*C]: staged columns are (h, c); one head's 64 x C tile is contiguous
__device__ __forceinline__ void st_flush_hm(const bf16* st, int wtid, bf16* base, int n, int N, int s0, int S) {
  constexpr int PER = TS * (C / 8);
  for (int v = wtid; v < H * PER; v += 128) {
    // rows fastest: eight lanes of a quarter-warp read eight rows of ONE 16-byte column, which the 128-byte XOR swizzle spreads over
    // eight bank groups (row-pair-major order made every pair of lanes collide: half of this kernel's shared-load wavefronts)
    const int h = v / PER, rem = v % PER, row = rem % TS, c8 = rem / TS;
    if (s0 + row < S) *reinterpret_cast<uint4*>(base + ((long)h * N + n) * ((long)S * C) + (long)(s0 + row) * C + c8 * 8) = st_get(st, row, h * (C / 8) + c8);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// K1: LayerNorm + value projection
// ---------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(NT) ln_vg_kernel(const bf16* __restrict__ X, const float* __restrict__ LNW, const float* __restrict__ LNB, float eps,
                                                   const bf16* __restrict__ WV, bf16* __restrict__ Y, bf16* __restrict__ V, float* __restrict__ STATS,
                                                   int N, int S, int ntile) {
  extern __shared__ unsigned char smem_raw[];
  bf16* sX = reinterpret_cast<bf16*>(align1k(smem_raw));         // [2][NKD][TILE]
  bf16* sW = sX + NWG * 2 * NKD * TILE;                         // [NKD][HC][64]
  const int tid = threadIdx.x, wgi = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  auto load = [&](int buf, int t) {
    const int n = t % N, s0 = (t / N) * TS;
    for (int v = wtid; v < TS * (D / 8); v += 128) {
      const int row = v / (D / 8), ch = v % (D / 8);
      const bool ok = s0 + row < S;
      cp16(sa(sX + (wgi * 2 + buf) * NKD * TILE + so(ch >> 3, TS, row, ch & 7)), ok ? X + ((long)(s0 + row) * N + n) * D + ch * 8 : X, ok);
    }
  };
  for (int v = tid; v < HC * (D / 8); v += NT) {
    const int n = v / (D / 8), c8 = v % (D / 8);
    cp16(sa(sW + so(c8 >> 3, HC, n, c8 & 7)), WV + (long)n * D + c8 * 8, true);
  }
  cp_commit();                                                   // the weights
  cp_wait<0>();
  __syncthreads();                                               // (the only block-wide barrier: every thread arrives before any warpgroup runs out of tiles)
  if ((int)(blockIdx.x * NWG + wgi) < ntile) load(0, blockIdx.x * NWG + wgi);
  cp_commit();
  constexpr int TPR = D / 32;
  const int tq = wtid % TPR;
  float g[32], b[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) { g[k] = LNW[tq * 32 + k]; b[k] = LNB[tq * 32 + k]; }
  const int r0 = warp * 16 + (lane >> 2), q2 = (lane & 3) * 2;
  for (int it = 0;; ++it) {
    const int t = (int)(blockIdx.x * NWG + wgi) + it * (int)(gridDim.x * NWG);
    if (t >= ntile) break;
    const int buf = it & 1, tn = t + (int)(gridDim.x * NWG);
    if (tn < ntile) load(buf ^ 1, tn);
    cp_commit();
    cp_wait<1>();
    wg_sync(wgi);
    const int n = t % N, s0 = (t / N) * TS;
    bf16* sx = sX + (wgi * 2 + buf) * NKD * TILE;
    ln_tile(sx, wtid, g, b, eps, Y, STATS, n, s0, S, N);
    wgw::proxy_fence();
    wg_sync(wgi);
    float acc[NCH][32];
    wgw::fence();
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks)
#pragma unroll
      for (int j = 0; j < NCH; ++j)
        wgw::ss64(wgw::desc(sx + (ks >> 2) * TILE + (ks & 3) * 16), wgw::desc(sW + ((ks >> 2) * HC + 64 * j) * 64 + (ks & 3) * 16), acc[j], ks == 0 ? 0 : 1);
    wgw::commit();
    wgw::wait();
#pragma unroll
    for (int j = 0; j < NCH; ++j)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = j * 64 + nt * 8 + q2, h = col / C, c = col % C;
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int s = s0 + r0 + rr * 8;
          if (s < S) *reinterpret_cast<uint32_t*>(V + ((long)h * N + n) * ((long)S * C) + (long)s * C + c) = pack2(acc[j][nt * 4 + rr * 2], acc[j][nt * 4 + rr * 2 + 1]);
        }
      }
    wg_sync(wgi);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// K2: gate, gated product, out projection, dropout, residual.  LN: the tile source Y is the raw MSA and y = LN(m) is formed in place
// (inference, which keeps no y); otherwise Y is the y that ln_vg_w stored.
// ---------------------------------------------------------------------------------------------------------------
template <bool LN>
__global__ void __launch_bounds__(NT) gate_out_kernel(const bf16* __restrict__ O, const bf16* __restrict__ Y, const bf16* __restrict__ X,
                                                      const bf16* __restrict__ WG, const bf16* __restrict__ WO, const bf16* __restrict__ DMASK, float dscale,
                                                      const float* __restrict__ LNW, const float* __restrict__ LNB, float eps,
                                                      bf16* __restrict__ OUT, int N, int S, int ntile) {
  extern __shared__ unsigned char smem_raw[];
  bf16* sY = reinterpret_cast<bf16*>(align1k(smem_raw));         // [2][NKD][TILE]
  bf16* sWg = sY + NWG * 2 * NKD * TILE;                        // [NKD][HC][64]      Wg [HC][D]: rows = hidden, k = d
  bf16* sWo = sWg + NKD * HC * 64;                               // [NKH][D][64]       Wo [D][HC]: rows = d, k = hidden
  const int tid = threadIdx.x, wgi = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  auto load = [&](int buf, int t) {
    const int n = t % N, s0 = (t / N) * TS;
    for (int v = wtid; v < TS * (D / 8); v += 128) {
      const int row = v / (D / 8), ch = v % (D / 8);
      const bool ok = s0 + row < S;
      cp16(sa(sY + (wgi * 2 + buf) * NKD * TILE + so(ch >> 3, TS, row, ch & 7)), ok ? Y + ((long)(s0 + row) * N + n) * D + ch * 8 : Y, ok);
    }
  };
  for (int v = tid; v < HC * (D / 8); v += NT) {
    const int n = v / (D / 8), c8 = v % (D / 8);
    cp16(sa(sWg + so(c8 >> 3, HC, n, c8 & 7)), WG + (long)n * D + c8 * 8, true);
  }
  for (int v = tid; v < D * (HC / 8); v += NT) {
    const int n = v / (HC / 8), c8 = v % (HC / 8);
    cp16(sa(sWo + so(c8 >> 3, D, n, c8 & 7)), WO + (long)n * HC + c8 * 8, true);
  }
  __shared__ float sGB[LN ? 2 * D : 1];
  if (LN)
    for (int k = tid; k < 2 * D; k += NT) sGB[k] = k < D ? LNW[k] : LNB[k - D];
  cp_commit();                                                   // the weights
  cp_wait<0>();
  __syncthreads();                                               // (the only block-wide barrier: every thread arrives before any warpgroup runs out of tiles)
  if ((int)(blockIdx.x * NWG + wgi) < ntile) load(0, blockIdx.x * NWG + wgi);
  cp_commit();
  const int r0 = warp * 16 + (lane >> 2), q2 = (lane & 3) * 2;
  for (int it = 0;; ++it) {
    const int t = (int)(blockIdx.x * NWG + wgi) + it * (int)(gridDim.x * NWG);
    if (t >= ntile) break;
    const int buf = it & 1, tn = t + (int)(gridDim.x * NWG);
    if (tn < ntile) load(buf ^ 1, tn);
    cp_commit();
    const int n = t % N, s0 = (t / N) * TS;
    // fragments of o (head-major) and of the residual, in the accumulator layout, while the tile lands
    uint32_t of[HC / 8][2], xf[D / 8][2];
#pragma unroll
    for (int nt = 0; nt < HC / 8; ++nt) {
      const int col = nt * 8 + q2, h = col / C, c = col % C;
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int s = s0 + r0 + rr * 8;
        of[nt][rr] = s < S ? *reinterpret_cast<const uint32_t*>(O + ((long)h * N + n) * ((long)S * C) + (long)s * C + c) : 0u;
      }
    }
#pragma unroll
    for (int nt = 0; nt < D / 8; ++nt)
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int s = s0 + r0 + rr * 8;
        xf[nt][rr] = s < S ? *reinterpret_cast<const uint32_t*>(X + ((long)s * N + n) * D + nt * 8 + q2) : 0u;
      }
    cp_wait<1>();
    bf16* sy = sY + (wgi * 2 + buf) * NKD * TILE;
    if (LN) {
      wg_sync(wgi);                                              // the tile landed: normalise it in place
      float g[32], b[32];
#pragma unroll
      for (int k = 0; k < 32; ++k) { g[k] = sGB[(wtid % (D / 32)) * 32 + k]; b[k] = sGB[D + (wtid % (D / 32)) * 32 + k]; }
      ln_tile(sy, wtid, g, b, eps, nullptr, nullptr, n, s0, S, N);
    }
    wgw::proxy_fence();
    wg_sync(wgi);
    float gacc[NCH][32];
    wgw::fence();
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks)
#pragma unroll
      for (int j = 0; j < NCH; ++j)
        wgw::ss64(wgw::desc(sy + (ks >> 2) * TILE + (ks & 3) * 16), wgw::desc(sWg + ((ks >> 2) * HC + 64 * j) * 64 + (ks & 3) * 16), gacc[j], ks == 0 ? 0 : 1);
    wgw::commit();
    wgw::wait();
    uint32_t af[HC / 16][4];
#pragma unroll
    for (int kk = 0; kk < HC / 16; ++kk)
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int nt = 2 * kk + p, j = nt >> 3, nl = nt & 7;
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const float2 o = unpack2(of[nt][rr]);
          const float g0 = gacc[j][nl * 4 + rr * 2], g1 = gacc[j][nl * 4 + rr * 2 + 1];
          af[kk][p * 2 + rr] = pack2(o.x / (1.f + __expf(-g0)), o.y / (1.f + __expf(-g1)));
        }
      }
    float oacc[NCD][32];
    wgw::fence();
#pragma unroll
    for (int kk = 0; kk < HC / 16; ++kk)
#pragma unroll
      for (int jn = 0; jn < NCD; ++jn)
        wgw::rs64(af[kk][0], af[kk][1], af[kk][2], af[kk][3], wgw::desc(sWo + ((kk >> 2) * D + 64 * jn) * 64 + (kk & 3) * 16), oacc[jn], kk == 0 ? 0 : 1);
    wgw::commit();
    wgw::wait();
    wg_sync(wgi);                                                // every warp is past the gate GEMM: the y tile becomes the output staging
#pragma unroll
    for (int jn = 0; jn < NCD; ++jn)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = jn * 64 + nt * 8 + q2;
        float2 dm = make_float2(1.f, 1.f);
        if (DMASK != nullptr) { dm = unpack2(*reinterpret_cast<const uint32_t*>(DMASK + (long)n * D + col)); dm.x *= dscale; dm.y *= dscale; }
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          // the stock module rounds the update to bf16 before the (dropout and the) residual add
          float u0 = rbf(oacc[jn][nt * 4 + rr * 2]), u1 = rbf(oacc[jn][nt * 4 + rr * 2 + 1]);
          if (DMASK != nullptr) { u0 = rbf(u0 * dm.x); u1 = rbf(u1 * dm.y); }
          const float2 xr = unpack2(xf[jn * 8 + nt][rr]);
          st_put(sy, r0 + rr * 8, col, pack2(xr.x + u0, xr.y + u1));
        }
      }
    wg_sync(wgi);
    st_flush_nat<D>(sy, wtid, OUT, D, 0, n, N, s0, S);
    wg_sync(wgi);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// K3: gate backward.  do' = dout . Wo (the gradient of the gated product), g = sigmoid(y . Wg^T) recomputed from y.
// ---------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(NT) gate_bwd_kernel(const bf16* __restrict__ O, const bf16* __restrict__ Y, const bf16* __restrict__ DOUT,
                                                      const bf16* __restrict__ WG, const bf16* __restrict__ WOT,
                                                      bf16* __restrict__ DOH, bf16* __restrict__ GO, bf16* __restrict__ DGV,
                                                      int N, int S, int ntile) {
  extern __shared__ unsigned char smem_raw[];
  bf16* sY = reinterpret_cast<bf16*>(align1k(smem_raw));         // [2][NKD][TILE]
  bf16* sD = sY + NWG * 2 * NKD * TILE;                         // [NWG][2][NKD][TILE]
  bf16* sWg = sD + NWG * 2 * NKD * TILE;                         // [NKD][HC][64]
  bf16* sWt = sWg + NKD * HC * 64;                               // [NKD][HC][64]   Wo^T [HC][D]
  bf16* sSt = sWt + NKD * HC * 64;                               // [NWG][HC/64][64][64] output staging
  const int tid = threadIdx.x, wgi = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  auto load = [&](int buf, int t) {
    const int n = t % N, s0 = (t / N) * TS;
    for (int v = wtid; v < TS * (D / 8); v += 128) {
      const int row = v / (D / 8), ch = v % (D / 8);
      const bool ok = s0 + row < S;
      const long go = ((long)(s0 + row) * N + n) * D + ch * 8;
      const int so_ = so(ch >> 3, TS, row, ch & 7);
      cp16(sa(sY + (wgi * 2 + buf) * NKD * TILE + so_), ok ? Y + go : Y, ok);
      cp16(sa(sD + (wgi * 2 + buf) * NKD * TILE + so_), ok ? DOUT + go : DOUT, ok);
    }
  };
  for (int v = tid; v < HC * (D / 8); v += NT) {
    const int n = v / (D / 8), c8 = v % (D / 8);
    cp16(sa(sWg + so(c8 >> 3, HC, n, c8 & 7)), WG + (long)n * D + c8 * 8, true);
    cp16(sa(sWt + so(c8 >> 3, HC, n, c8 & 7)), WOT + (long)n * D + c8 * 8, true);
  }
  cp_commit();                                                   // the weights
  cp_wait<0>();
  __syncthreads();                                               // (the only block-wide barrier: every thread arrives before any warpgroup runs out of tiles)
  if ((int)(blockIdx.x * NWG + wgi) < ntile) load(0, blockIdx.x * NWG + wgi);
  cp_commit();
  const int r0 = warp * 16 + (lane >> 2), q2 = (lane & 3) * 2;
  for (int it = 0;; ++it) {
    const int t = (int)(blockIdx.x * NWG + wgi) + it * (int)(gridDim.x * NWG);
    if (t >= ntile) break;
    const int buf = it & 1, tn = t + (int)(gridDim.x * NWG);
    if (tn < ntile) load(buf ^ 1, tn);
    cp_commit();
    const int n = t % N, s0 = (t / N) * TS;
    uint32_t of[HC / 8][2];
#pragma unroll
    for (int nt = 0; nt < HC / 8; ++nt) {
      const int col = nt * 8 + q2, h = col / C, c = col % C;
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int s = s0 + r0 + rr * 8;
        of[nt][rr] = s < S ? *reinterpret_cast<const uint32_t*>(O + ((long)h * N + n) * ((long)S * C) + (long)s * C + c) : 0u;
      }
    }
    cp_wait<1>();
    wgw::proxy_fence();
    wg_sync(wgi);
    const bf16* sy = sY + (wgi * 2 + buf) * NKD * TILE;
    const bf16* sd = sD + (wgi * 2 + buf) * NKD * TILE;
    float gacc[NCH][32], dacc[NCH][32];
    wgw::fence();
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks)
#pragma unroll
      for (int j = 0; j < NCH; ++j) {
        wgw::ss64(wgw::desc(sy + (ks >> 2) * TILE + (ks & 3) * 16), wgw::desc(sWg + ((ks >> 2) * HC + 64 * j) * 64 + (ks & 3) * 16), gacc[j], ks == 0 ? 0 : 1);
        wgw::ss64(wgw::desc(sd + (ks >> 2) * TILE + (ks & 3) * 16), wgw::desc(sWt + ((ks >> 2) * HC + 64 * j) * 64 + (ks & 3) * 16), dacc[j], ks == 0 ? 0 : 1);
      }
    wgw::commit();
    wgw::wait();
    bf16* st = sSt + wgi * HC * 64;
#pragma unroll
    for (int j = 0; j < NCH; ++j)
#pragma unroll
      for (int k = 0; k < 32; ++k) gacc[j][k] = 1.f / (1.f + __expf(-gacc[j][k]));        // g = sigmoid(gate), in place
    // three outputs, each staged and written as 16-byte rows: go = g o, dgp = do' o g (1 - g), do = do' g
#pragma unroll
    for (int round = 0; round < 3; ++round) {
#pragma unroll
      for (int j = 0; j < NCH; ++j)
#pragma unroll
        for (int nl = 0; nl < 8; ++nl) {
          const int nt = j * 8 + nl, col = nt * 8 + q2;
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const float2 o = unpack2(of[nt][rr]);
            const float g0 = gacc[j][nl * 4 + rr * 2], g1 = gacc[j][nl * 4 + rr * 2 + 1];
            const float d0 = dacc[j][nl * 4 + rr * 2], d1 = dacc[j][nl * 4 + rr * 2 + 1];
            uint32_t v;
            if (round == 0) v = pack2(o.x * g0, o.y * g1);
            else if (round == 1) v = pack2(d0 * o.x * g0 * (1.f - g0), d1 * o.y * g1 * (1.f - g1));
            else v = pack2(d0 * g0, d1 * g1);
            st_put(st, r0 + rr * 8, col, v);
          }
        }
      wg_sync(wgi);
      if (round == 0) st_flush_nat<HC>(st, wtid, GO, HC, 0, n, N, s0, S);
      else if (round == 1) st_flush_nat<HC>(st, wtid, DGV, 2 * HC, 0, n, N, s0, S);
      else st_flush_hm(st, wtid, DOH, n, N, s0, S);
      wg_sync(wgi);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// K4: dy = [dgp | dv] . [Wg; Wv] (K = 2 HC), LayerNorm backward, residual gradient; dgamma / dbeta per-CTA partials
// ---------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(NT) dm_bwd_kernel(const bf16* __restrict__ DVH, bf16* __restrict__ DGV, const bf16* __restrict__ X, const float* __restrict__ STATS,
                                                    const bf16* __restrict__ DRES, const bf16* __restrict__ WGVT, const float* __restrict__ LNW,
                                                    bf16* __restrict__ DM, float* __restrict__ PART, int N, int S, int ntile) {
  extern __shared__ unsigned char smem_raw[];
  bf16* sA = reinterpret_cast<bf16*>(align1k(smem_raw));         // [2][NK2][TILE]  [dgp | dv]
  bf16* sB = sA + NWG * 2 * NK2 * TILE;                         // [NK2][D][64]    [Wg | Wv]^T: rows = d, k = 2 HC
  bf16* sX = sB + NK2 * D * 64;                                 // [2][NWG][NCD][TILE]  x tile of the same (n, s0): prefetched, not read from HBM in the epilogue
  bf16* sR = sX + NWG * 2 * NCD * TILE;                         // [2][NWG][NCD][TILE]  the residual gradient tile
  __shared__ float sGam[D];
  __shared__ float sRed[4 * NWG][2][D];
  const int tid = threadIdx.x, wgi = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  auto load = [&](int buf, int t) {
    const int n = t % N, s0 = (t / N) * TS;
    for (int v = wtid; v < TS * (D / 8); v += 128) {
      const int row = v / (D / 8), ch = v % (D / 8);
      const bool ok = s0 + row < S;
      const long g = ((long)(s0 + row) * N + n) * D + ch * 8;
      const int o = (wgi * 2 + buf) * NCD * TILE + so(ch >> 3, TS, row, ch & 7);
      cp16(sa(sX + o), ok ? X + g : X, ok);
      cp16(sa(sR + o), ok ? DRES + g : DRES, ok);
    }
    for (int v = wtid; v < TS * (2 * HC / 8); v += 128) {
      const int row = v / (2 * HC / 8), ch = v % (2 * HC / 8);
      const bool ok = s0 + row < S;
      const bf16* src;
      if (ch < HC / 8) src = DGV + ((long)(s0 + row) * N + n) * (2 * HC) + ch * 8;
      else { const int col = (ch - HC / 8) * 8, h = col / C, c = col % C; src = DVH + ((long)h * N + n) * ((long)S * C) + (long)(s0 + row) * C + c; }
      cp16(sa(sA + (wgi * 2 + buf) * NK2 * TILE + so(ch >> 3, TS, row, ch & 7)), ok ? src : DGV, ok);
    }
  };
  for (int v = tid; v < D * (2 * HC / 8); v += NT) {
    const int n = v / (2 * HC / 8), c8 = v % (2 * HC / 8);
    cp16(sa(sB + so(c8 >> 3, D, n, c8 & 7)), WGVT + (long)n * (2 * HC) + c8 * 8, true);
  }
  for (int k = tid; k < D; k += NT) sGam[k] = LNW[k];
  cp_commit();                                                   // the weights
  cp_wait<0>();
  __syncthreads();                                               // (the only block-wide barrier: every thread arrives before any warpgroup runs out of tiles)
  if ((int)(blockIdx.x * NWG + wgi) < ntile) load(0, blockIdx.x * NWG + wgi);
  cp_commit();
  const int r0 = warp * 16 + (lane >> 2), q2 = (lane & 3) * 2;
  float dgs[D / 8][2], dbs[D / 8][2];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) { dgs[i][0] = dgs[i][1] = dbs[i][0] = dbs[i][1] = 0.f; }
  for (int it = 0;; ++it) {
    const int t = (int)(blockIdx.x * NWG + wgi) + it * (int)(gridDim.x * NWG);
    if (t >= ntile) break;
    const int buf = it & 1, tn = t + (int)(gridDim.x * NWG);
    if (tn < ntile) load(buf ^ 1, tn);
    cp_commit();
    cp_wait<1>();
    wgw::proxy_fence();
    wg_sync(wgi);
    const int n = t % N, s0 = (t / N) * TS;
    bf16* sa_ = sA + (wgi * 2 + buf) * NK2 * TILE;
    // dv into the natural-layout buffer (the dWv GEMM's operand): copied out of the tile that just landed
    for (int v = wtid; v < TS * (HC / 8); v += 128) {
      const int row = v / (HC / 8), ch = v % (HC / 8), a = HC / 8 + ch;
      if (s0 + row < S)
        *reinterpret_cast<uint4*>(DGV + ((long)(s0 + row) * N + n) * (2 * HC) + HC + ch * 8) = *reinterpret_cast<const uint4*>(sa_ + so(a >> 3, TS, row, a & 7));
    }
    // the row statistics are fetched before the wgmma is issued: their global latency hides under it
    float mean[2], rstd[2];
    bool valid[2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int s = s0 + r0 + rr * 8;
      valid[rr] = s < S;
      const float2 st = valid[rr] ? *reinterpret_cast<const float2*>(STATS + ((long)s * N + n) * 2) : make_float2(0.f, 0.f);
      mean[rr] = st.x; rstd[rr] = st.y;
    }
    float dacc[NCD][32];
    wgw::fence();
#pragma unroll
    for (int kk = 0; kk < NK2 * 4; ++kk)
#pragma unroll
      for (int jn = 0; jn < NCD; ++jn)
        wgw::ss64(wgw::desc(sa_ + (kk >> 2) * TILE + (kk & 3) * 16), wgw::desc(sB + ((kk >> 2) * D + 64 * jn) * 64 + (kk & 3) * 16), dacc[jn], kk == 0 ? 0 : 1);
    wgw::commit();
    wgw::wait();
    const bf16* xt = sX + (wgi * 2 + buf) * NCD * TILE;
    const bf16* rt = sR + (wgi * 2 + buf) * NCD * TILE;
    float sg[2] = {0.f, 0.f}, sgx[2] = {0.f, 0.f};
#pragma unroll
    for (int jn = 0; jn < NCD; ++jn)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = jn * 64 + nt * 8 + q2;
        const float ga = sGam[col], gb = sGam[col + 1];
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const float2 mf = valid[rr] ? unpack2(*reinterpret_cast<const uint32_t*>(xt + so(jn, TS, r0 + rr * 8, nt) + q2)) : make_float2(0.f, 0.f);
          const float x0 = (mf.x - mean[rr]) * rstd[rr], x1 = (mf.y - mean[rr]) * rstd[rr];
          const float d0 = dacc[jn][nt * 4 + rr * 2], d1 = dacc[jn][nt * 4 + rr * 2 + 1];
          sg[rr] += d0 * ga + d1 * gb;
          sgx[rr] += d0 * ga * x0 + d1 * gb * x1;
          if (valid[rr]) { dgs[jn * 8 + nt][0] += d0 * x0; dgs[jn * 8 + nt][1] += d1 * x1; dbs[jn * 8 + nt][0] += d0; dbs[jn * 8 + nt][1] += d1; }
        }
      }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      sg[rr] += __shfl_xor_sync(0xffffffffu, sg[rr], 1); sg[rr] += __shfl_xor_sync(0xffffffffu, sg[rr], 2);
      sgx[rr] += __shfl_xor_sync(0xffffffffu, sgx[rr], 1); sgx[rr] += __shfl_xor_sync(0xffffffffu, sgx[rr], 2);
      sg[rr] *= (1.f / D); sgx[rr] *= (1.f / D);
    }
    wg_sync(wgi);                                                // the [dgp | dv] tile has been read by every warp: it becomes the dm staging
#pragma unroll
    for (int jn = 0; jn < NCD; ++jn)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = jn * 64 + nt * 8 + q2;
        const float ga = sGam[col], gb = sGam[col + 1];
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          if (!valid[rr]) continue;
          const float2 mf = unpack2(*reinterpret_cast<const uint32_t*>(xt + so(jn, TS, r0 + rr * 8, nt) + q2));
          const float2 rf = unpack2(*reinterpret_cast<const uint32_t*>(rt + so(jn, TS, r0 + rr * 8, nt) + q2));
          const float x0 = (mf.x - mean[rr]) * rstd[rr], x1 = (mf.y - mean[rr]) * rstd[rr];
          const float d0 = dacc[jn][nt * 4 + rr * 2] * ga, d1 = dacc[jn][nt * 4 + rr * 2 + 1] * gb;
          st_put(sa_, r0 + rr * 8, col, pack2(rstd[rr] * (d0 - sg[rr] - x0 * sgx[rr]) + rf.x, rstd[rr] * (d1 - sg[rr] - x1 * sgx[rr]) + rf.y));
        }
      }
    wg_sync(wgi);
    st_flush_nat<D>(sa_, wtid, DM, D, 0, n, N, s0, S);
    wg_sync(wgi);
  }
  // dgamma / dbeta: sum over the eight row lanes of the quad layout, then over the four warps, in a fixed order
#pragma unroll
  for (int i = 0; i < D / 8; ++i)
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      float a = dgs[i][k], b = dbs[i][k];
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); b += __shfl_xor_sync(0xffffffffu, b, o); }
      if ((lane >> 2) == 0) { sRed[warp + 4 * wgi][0][i * 8 + q2 + k] = a; sRed[warp + 4 * wgi][1][i * 8 + q2 + k] = b; }
    }
  __syncthreads();
  for (int k = tid; k < 2 * D; k += NT) {
    const int w = k / D, col = k % D;
    {
    float a = 0.f;
    for (int q = 0; q < 4 * NWG; ++q) a += sRed[q][w][col];                     // fixed order
    PART[((long)blockIdx.x * 2 + w) * D + col] = a;
  }
  }
}

// persistent grid: one CTA per resident slot; the launch attribute and the occupancy are queried once per kernel (keyed by the
// function, not its type: gate_out_kernel<true> and <false> share one)
template <typename K>
int grid_for(K kernel, int smem, int ntile, int device) {
  static std::map<const void*, int> bps_of;
  static int sms = 0;
  const void* key = reinterpret_cast<const void*>(kernel);
  auto it = bps_of.find(key);
  if (it == bps_of.end()) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, NT, smem);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    it = bps_of.emplace(key, bps < 1 ? 1 : bps).first;
  }
  return std::max(1, std::min(ntile, sms * it->second));
}
inline const bf16* P(const torch::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr<at::BFloat16>()); }
inline bf16* W(torch::Tensor& t) { return reinterpret_cast<bf16*>(t.data_ptr<at::BFloat16>()); }
void check_bf16(const torch::Tensor& t, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous(), what, ": contiguous cuda bf16");
}
}  // namespace

std::vector<torch::Tensor> ln_vg_w(torch::Tensor m, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wv, double eps, bool save_y) {
  check_bf16(m, "m"); check_bf16(wv, "wv");
  TORCH_CHECK(m.dim() == 3 && m.size(2) == D, "m: [S, N, ", D, "]");
  TORCH_CHECK(wv.dim() == 2 && wv.size(0) == HC && wv.size(1) == D, "wv: [", HC, ", ", D, "]");
  TORCH_CHECK(lnw.scalar_type() == torch::kFloat && lnw.numel() == D && lnb.scalar_type() == torch::kFloat && lnb.numel() == D, "LN params fp32 [D]");
  const int S = (int)m.size(0), N = (int)m.size(1);
  auto y = save_y ? torch::empty_like(m) : torch::empty({0}, m.options());                      // inference keeps neither y nor the statistics
  auto v = torch::empty({H, (long)N, (long)S * C}, m.options());
  auto stats = save_y ? torch::empty({S, N, 2}, m.options().dtype(torch::kFloat)) : torch::empty({0}, m.options().dtype(torch::kFloat));
  const int ntile = N * ((S + TS - 1) / TS);
  const int smem = 1024 + (NWG * 2 * NKD * TILE + NKD * HC * 64) * 2;
  const int grid = grid_for(ln_vg_kernel, smem, (ntile + NWG - 1) / NWG, m.device().index());
  ln_vg_kernel<<<grid, NT, smem, at::cuda::getCurrentCUDAStream()>>>(P(m), lnw.data_ptr<float>(), lnb.data_ptr<float>(), (float)eps, P(wv), save_y ? W(y) : nullptr, W(v), save_y ? stats.data_ptr<float>() : nullptr, N, S, ntile);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {v, y, stats};
}

// y is the stored LN(m) from ln_vg_w, or -- with lnw / lnb -- the raw m, normalised inside (inference: nothing stores y)
torch::Tensor gate_out_w(torch::Tensor o, torch::Tensor y, torch::Tensor m, torch::Tensor wg, torch::Tensor wo, c10::optional<torch::Tensor> dmask, double dscale,
                         c10::optional<torch::Tensor> lnw, c10::optional<torch::Tensor> lnb, double eps) {
  check_bf16(o, "o"); check_bf16(y, "y"); check_bf16(m, "m"); check_bf16(wg, "wg"); check_bf16(wo, "wo");
  const int S = (int)m.size(0), N = (int)m.size(1);
  TORCH_CHECK(m.size(2) == D && y.sizes() == m.sizes() && o.sizes() == torch::IntArrayRef({H, N, (long)S * C}), "shapes");
  TORCH_CHECK(wg.sizes() == torch::IntArrayRef({HC, D}) && wo.sizes() == torch::IntArrayRef({D, HC}), "weights: Wg [HC, D], Wo [D, HC]");
  if (dmask.has_value()) { check_bf16(*dmask, "dmask"); TORCH_CHECK(dmask->sizes() == torch::IntArrayRef({N, D}), "dmask: [N, D]"); }
  auto out = torch::empty_like(m);
  const int ntile = N * ((S + TS - 1) / TS);
  const int smem = 1024 + (NWG * 2 * NKD * TILE + NKD * HC * 64 + NKH * D * 64) * 2;
  const bool ln = lnw.has_value();
  TORCH_CHECK(ln == lnb.has_value(), "lnw and lnb come together");
  if (ln) TORCH_CHECK(lnw->scalar_type() == torch::kFloat && lnw->numel() == D && lnb->scalar_type() == torch::kFloat && lnb->numel() == D, "LN params fp32 [D]");
  const float* gp = ln ? lnw->data_ptr<float>() : nullptr;
  const float* bp = ln ? lnb->data_ptr<float>() : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (ln) {
    const int grid = grid_for(gate_out_kernel<true>, smem, (ntile + NWG - 1) / NWG, m.device().index());
    gate_out_kernel<true><<<grid, NT, smem, stream>>>(P(o), P(y), P(m), P(wg), P(wo), dmask.has_value() ? P(*dmask) : nullptr, (float)dscale, gp, bp, (float)eps, W(out), N, S, ntile);
  } else {
    const int grid = grid_for(gate_out_kernel<false>, smem, (ntile + NWG - 1) / NWG, m.device().index());
    gate_out_kernel<false><<<grid, NT, smem, stream>>>(P(o), P(y), P(m), P(wg), P(wo), dmask.has_value() ? P(*dmask) : nullptr, (float)dscale, gp, bp, 0.f, W(out), N, S, ntile);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// -> {do (head-major), go [S, N, HC]}; dgp lands in dgv[..., :HC]
std::vector<torch::Tensor> gate_bwd_w(torch::Tensor o, torch::Tensor y, torch::Tensor dout, torch::Tensor wg, torch::Tensor wot, torch::Tensor dgv) {
  check_bf16(o, "o"); check_bf16(y, "y"); check_bf16(dout, "dout"); check_bf16(wg, "wg"); check_bf16(wot, "wot"); check_bf16(dgv, "dgv");
  const int S = (int)y.size(0), N = (int)y.size(1);
  TORCH_CHECK(y.size(2) == D && dout.sizes() == y.sizes() && o.sizes() == torch::IntArrayRef({H, N, (long)S * C}), "shapes");
  TORCH_CHECK(wg.sizes() == torch::IntArrayRef({HC, D}) && wot.sizes() == torch::IntArrayRef({HC, D}), "weights: Wg [HC, D], Wo^T [HC, D]");
  TORCH_CHECK(dgv.sizes() == torch::IntArrayRef({S, N, 2 * HC}), "dgv: [S, N, 2 HC]");
  auto doh = torch::empty_like(o);
  auto go = torch::empty({S, N, (long)HC}, y.options());
  const int ntile = N * ((S + TS - 1) / TS);
  const int smem = 1024 + (NWG * 4 * NKD * TILE + 2 * NKD * HC * 64 + NWG * HC * 64) * 2;
  const int grid = grid_for(gate_bwd_kernel, smem, (ntile + NWG - 1) / NWG, y.device().index());
  gate_bwd_kernel<<<grid, NT, smem, at::cuda::getCurrentCUDAStream()>>>(P(o), P(y), P(dout), P(wg), P(wot), W(doh), W(go), W(dgv), N, S, ntile);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {doh, go};
}

// -> {dm [S, N, D], dgamma [D], dbeta [D]}; dv is copied into dgv[..., HC:]
std::vector<torch::Tensor> dm_bwd_w(torch::Tensor dvh, torch::Tensor dgv, torch::Tensor m, torch::Tensor stats, torch::Tensor dres, torch::Tensor wgvt, torch::Tensor lnw) {
  check_bf16(dvh, "dv"); check_bf16(dgv, "dgv"); check_bf16(m, "m"); check_bf16(dres, "dres"); check_bf16(wgvt, "wgvt");
  const int S = (int)m.size(0), N = (int)m.size(1);
  TORCH_CHECK(m.size(2) == D && dres.sizes() == m.sizes() && dvh.sizes() == torch::IntArrayRef({H, N, (long)S * C}), "shapes");
  TORCH_CHECK(dgv.sizes() == torch::IntArrayRef({S, N, 2 * HC}) && wgvt.sizes() == torch::IntArrayRef({D, 2 * HC}), "dgv: [S, N, 2 HC], wgvt: [D, 2 HC]");
  TORCH_CHECK(stats.scalar_type() == torch::kFloat && stats.is_contiguous() && stats.numel() == (long)S * N * 2, "stats: fp32 [S, N, 2]");
  TORCH_CHECK(lnw.scalar_type() == torch::kFloat && lnw.numel() == D, "lnw: fp32 [D]");
  auto dm = torch::empty_like(m);
  const int ntile = N * ((S + TS - 1) / TS);
  const int smem = 1024 + (NWG * 2 * NK2 * TILE + NK2 * D * 64 + 2 * NWG * 2 * NCD * TILE) * 2;
  const int grid = grid_for(dm_bwd_kernel, smem, (ntile + NWG - 1) / NWG, m.device().index());
  auto part = torch::empty({grid, 2, (long)D}, m.options().dtype(torch::kFloat));
  dm_bwd_kernel<<<grid, NT, smem, at::cuda::getCurrentCUDAStream()>>>(P(dvh), W(dgv), P(m), stats.data_ptr<float>(), P(dres), P(wgvt), lnw.data_ptr<float>(), W(dm), part.data_ptr<float>(), N, S, ntile);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto red = part.sum(0);
  return {dm, red[0], red[1]};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("ln_vg_w", &ln_vg_w, "LayerNorm + value projection: (v head-major [H, N, S*C], y [S, N, D], stats [S, N, 2]); save_y=False leaves y and stats empty",
          py::arg("m"), py::arg("lnw"), py::arg("lnb"), py::arg("wv"), py::arg("eps"), py::arg("save_y") = true);
  mod.def("gate_out_w", &gate_out_w, "gate, gated product, out projection, dropout, residual", py::arg("o"), py::arg("y"), py::arg("m"), py::arg("wg"), py::arg("wo"), py::arg("dmask") = py::none(), py::arg("dscale") = 1.0, py::arg("lnw") = py::none(), py::arg("lnb") = py::none(), py::arg("eps") = 1e-5);
  mod.def("gate_bwd_w", &gate_bwd_w, "gate backward: (do head-major, go); dgp into dgv[..., :HC]");
  mod.def("dm_bwd_w", &dm_bwd_w, "dy = [dgp | dv] . [Wg; Wv], LayerNorm backward + residual: (dm, dgamma, dbeta); dv into dgv[..., HC:]");
}
