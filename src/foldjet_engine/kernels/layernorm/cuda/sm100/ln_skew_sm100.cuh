// B200 (sm_100a only) LayerNorm, skewed rows (2026-11-17): bf16 / fp32 rows of an ODD width (267 / 451 / 651 / 831 / 833: a row is not
// a whole number of 16-byte vectors, so it starts at any of V = 8 (bf16) / 4 (fp32) element offsets inside a vector, with a period
// of V rows; the comments below say 8 for V). The H100 kernels stage such rows through
// shared memory (ln_stage_fwd_kernel) or run them element-wise (ln_small_fwd_kernel / ln_rw_bwd_kernel).
//
// Here a row is TPR lanes x CH aligned 16-byte vectors held in registers, like the aligned row kernels: slot j of row r is the aligned
// vector at element r N - e + 8 j, with e = (r N) mod 8 the row's phase. A CTA takes the rows of ONE phase (CTA b: rows 8 t + (b mod
// 8)), so every thread of the CTA sees the same column in the same slot element: the affine parameters are one phase-shifted copy in
// shared memory (read as float4), not registers -- a build that kept them in registers (168 forward / 251 backward) was resident on
// only 384 / 256 threads a SM and measured 1.1-1.2x slower forward. The first and the last slot of a row are shared with the neighbour rows: they
// are loaded whole and masked to the row's own elements (a fixed word mask per thread), and written element-wise (2-byte stores of
// the row's own elements: nothing outside the row is written); every other slot is one 16-byte load and one 16-byte store. The host
// guarantees that the 16-byte vector holding the tensor's last element is readable. Reductions are sub-warp shuffles.
//
//   skew::fwd  y, (mean, rstd)
//   skew::bwd  dx, and one fp32 partial row of dw / db per CTA (the row slots of a warp folded by shuffles, the warps in order),
//              then rows::col_reduce adds the partial rows in a fixed order: a replay is bit-identical.
#pragma once
#include <algorithm>
#include "ln_common.cuh"

namespace mwln {
namespace skew {

using BF = __nv_bfloat16;
template <typename T> constexpr int vec_of() { return 16 / int(sizeof(T)); }

// bf16 -> fp32 is exact: the bf16 bits are the fp32's upper half
template <typename T>
__device__ __forceinline__ void unpack(const uint4& u, float* f) {
  if constexpr (sizeof(T) == 4) {
    f[0] = __uint_as_float(u.x);
    f[1] = __uint_as_float(u.y);
    f[2] = __uint_as_float(u.z);
    f[3] = __uint_as_float(u.w);
    return;
  }
  f[0] = __uint_as_float(u.x << 16);
  f[1] = __uint_as_float(u.x & 0xffff0000u);
  f[2] = __uint_as_float(u.y << 16);
  f[3] = __uint_as_float(u.y & 0xffff0000u);
  f[4] = __uint_as_float(u.z << 16);
  f[5] = __uint_as_float(u.z & 0xffff0000u);
  f[6] = __uint_as_float(u.w << 16);
  f[7] = __uint_as_float(u.w & 0xffff0000u);
}

// the word mask of the slot's elements lo <= i < hi
template <typename T>
__device__ __forceinline__ uint4 slot_mask(int lo, int hi) {
  uint32_t m[4];
  if constexpr (sizeof(T) == 4) {
#pragma unroll
    for (int k = 0; k < 4; ++k) m[k] = (k >= lo && k < hi) ? 0xffffffffu : 0u;
    return make_uint4(m[0], m[1], m[2], m[3]);
  }
#pragma unroll
  for (int k = 0; k < 4; ++k)
    m[k] = ((2 * k >= lo && 2 * k < hi) ? 0x0000ffffu : 0u) | ((2 * k + 1 >= lo && 2 * k + 1 < hi) ? 0xffff0000u : 0u);
  return make_uint4(m[0], m[1], m[2], m[3]);
}

// the phase-shifted copy of an affine parameter: element k of a row's vector stream is column k - e (``fill`` outside the row)
// (every load of a thread is issued before the first store: the copy is one load round, not S / NT of them -- measured on the B200
// at a few hundred rows, where the table copy was most of the kernel: 451 / 831 / 833 token streams)
template <int S, int NT>
__device__ __forceinline__ void fill_table(float* tab, const void* __restrict__ pv, int bf, float fill, int e, int N) {
  constexpr int R = (S + NT - 1) / NT;
  float v[R];
#pragma unroll
  for (int i = 0; i < R; ++i) {
    const int col = int(threadIdx.x) + i * NT - e;
    v[i] = fill;
    if (pv && col >= 0 && col < N) v[i] = bf ? __bfloat162float(reinterpret_cast<const BF*>(pv)[col]) : reinterpret_cast<const float*>(pv)[col];
  }
#pragma unroll
  for (int i = 0; i < R; ++i)
    if (int(threadIdx.x) + i * NT < S) tab[threadIdx.x + i * NT] = v[i];
}

template <int V>
__device__ __forceinline__ void loadv(const float* tab, float* out) {
  const float4 a = *reinterpret_cast<const float4*>(tab);
  out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
  if constexpr (V == 8) {
    const float4 b = *reinterpret_cast<const float4*>(tab + 4);
    out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
  }
}

// only the first and the last chunk hold a partial or an idle slot (``covers``: the host checks the layout against the width)
template <int CH> constexpr bool edge(int c) { return c == 0 || c == CH - 1; }

template <typename T, int TPR, int CH>
__device__ __forceinline__ uint4 load_slot(const T* __restrict__ x, int64_t base, int c, int lane, bool live, int hi,
                                           const uint4& mask) {
  constexpr int V = vec_of<T>();
  const int j = c * TPR + lane;
  uint4 u = make_uint4(0u, 0u, 0u, 0u);
  if (edge<CH>(c)) {
    if (live && hi > 0) {
      u = *reinterpret_cast<const uint4*>(x + base + V * j);
      u.x &= mask.x;
      u.y &= mask.y;
      u.z &= mask.z;
      u.w &= mask.w;
    }
  } else {
    if (live) u = *reinterpret_cast<const uint4*>(x + base + V * j);
  }
  return u;
}

// a partial slot (the row's first or last one) is written element by element: the other elements of the vector are a neighbour row's.
// Measured against at most three aligned 2 / 4 / 8-byte stores of the packed words: slower (more live registers), the plain form stays.
template <typename T, int CH>
__device__ __forceinline__ void store_slot(T* p, const float* o, int c, int lo, int hi) {
  constexpr int V = vec_of<T>();
  if (edge<CH>(c)) {
    if (lo == 0 && hi == V) {
      store_vec<T, V>(p, o);
    } else {
#pragma unroll
      for (int i = 0; i < V; ++i)
        if (i >= lo && i < hi) p[i] = from_f<T>(o[i]);
    }
  } else {
    store_vec<T, V>(p, o);
  }
}

// CTA b of a grid of 8 G: phase g = b mod 8 (rows 8 t + g), tiles u = b / 8, b / 8 + G, ... of RPC rows of that phase
template <typename T, int TPR, int CH, int WARPS, bool ONE, bool PF = false, bool PDL = false>
__global__ void __launch_bounds__(WARPS * 32) fwd(const T* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b, int wbf,
                                                  int bbf, T* __restrict__ y, float* __restrict__ mo, float* __restrict__ ro, int64_t M,
                                                  int N, float eps) {
  constexpr int V = vec_of<T>(), RPC = WARPS * 32 / TPR, S = TPR * CH * V;
  static_assert(TPR <= 32, "sub-warp rows");
  __shared__ __align__(16) float tw[S], tb[S];
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const int g = blockIdx.x & (V - 1);
  const int e = (g * N) & (V - 1);
  const float inv_n = 1.f / float(N);
  if constexpr (PDL) {  // x / the parameters may be the previous kernel's output
    pdl_wait();
    pdl_trigger();
  }
  int lo[CH], hi[CH];
  uint4 mask[CH];
  uint32_t em[2][V];  // all ones for the row's own elements of the first / last chunk
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const int j = c * TPR + lane;
    lo[c] = j == 0 ? e : 0;
    hi[c] = min(V, N + e - V * j);
    mask[c] = slot_mask<T>(lo[c], hi[c]);
#pragma unroll
    for (int i = 0; i < V; ++i)
      if (edge<CH>(c)) em[c != 0][i] = (i >= lo[c] && i < hi[c]) ? 0xffffffffu : 0u;
  }
  const int64_t stride = gridDim.x / V;
  // the first tile's x is loaded before the affine tables are copied: the two cold DRAM round trips overlap
  int64_t u = blockIdx.x / V;
  uint4 raw[CH];
  if constexpr (ONE || PF) {  // the first tile's loads go out before the table copy
    const int64_t row = V * (u * RPC + r_in) + g;
#pragma unroll
    for (int c = 0; c < CH; ++c) raw[c] = load_slot<T, TPR, CH>(x, row * N - e, c, lane, row < M, hi[c], mask[c]);
  }
  fill_table<S, WARPS * 32>(tw, w, wbf, 1.f, e, N);
  fill_table<S, WARPS * 32>(tb, b, bbf, 0.f, e, N);
  __syncthreads();
  for (bool first = ONE || PF; V * (u * RPC) + g < M; u += stride, first = false) {
    const int64_t row = V * (u * RPC + r_in) + g;
    const bool live = row < M;
    const int64_t base = row * N - e;
    if (!PF && !first) {
#pragma unroll
      for (int c = 0; c < CH; ++c) raw[c] = load_slot<T, TPR, CH>(x, base, c, lane, live, hi[c], mask[c]);
    }
    uint4 nxt[CH];  // PF: the next tile's x is in flight while this tile is reduced and stored
    if constexpr (PF) {
      const int64_t rn = V * ((u + stride) * RPC + r_in) + g;
#pragma unroll
      for (int c = 0; c < CH; ++c) nxt[c] = load_slot<T, TPR, CH>(x, rn * N - e, c, lane, rn < M, hi[c], mask[c]);
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      float f[V];
      unpack<T>(raw[c], f);
#pragma unroll
      for (int i = 0; i < V; ++i) s += f[i];
    }
    const float mean = row_sum<TPR>(s, nullptr, 0, lane) * inv_n;
    float q = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      float f[V];
      unpack<T>(raw[c], f);
#pragma unroll
      for (int i = 0; i < V; ++i) {
        float d = f[i] - mean;
        if (edge<CH>(c)) d = __uint_as_float(__float_as_uint(d) & em[c != 0][i]);
        q += d * d;
      }
    }
    const float rs = rsqrtf(row_sum<TPR>(q, nullptr, 0, lane) * inv_n + eps);
    if (live) {
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        if (!edge<CH>(c) || hi[c] > 0) {
          const int j = c * TPR + lane;
          float f[V], o[V], wv[V], bv[V];
          unpack<T>(raw[c], f);
          loadv<V>(tw + V * j, wv);
          loadv<V>(tb + V * j, bv);
#pragma unroll
          for (int i = 0; i < V; ++i) o[i] = (f[i] - mean) * rs * wv[i] + bv[i];
          store_slot<T, CH>(y + base + V * j, o, c, lo[c], hi[c]);
        }
      }
      if (lane == 0) {
        if (mo) mo[row] = mean;
        if (ro) ro[row] = rs;
      }
    }
    if constexpr (PF) {
#pragma unroll
      for (int c = 0; c < CH; ++c) raw[c] = nxt[c];
    }
  }
}

template <typename T, int TPR, int CH, int WARPS, bool ONE, bool PDL = false, bool FUSE = false>
__global__ void __launch_bounds__(WARPS * 32) bwd(const T* __restrict__ dy, const T* __restrict__ x, const void* __restrict__ w, int wbf,
                                                  const float* __restrict__ mean, const float* __restrict__ rstd, T* __restrict__ dx,
                                                  float* __restrict__ part, int pstride, int64_t M, int N, ColArgs ca) {
  if constexpr (FUSE) {  // the column CTAs of a fused backward (col_role)
    if (int(blockIdx.x) >= ca.gx) {
      if constexpr (PDL) {
        pdl_wait();
        pdl_trigger();
      }
      col_role<T, WARPS * 32, false, true>(ca, dy, x, mean, rstd, M, N);
      return;
    }
  }
  const int GX = FUSE ? ca.gx : int(gridDim.x);
  constexpr int V = vec_of<T>(), RPC = WARPS * 32 / TPR, E = CH * V, S = TPR * CH * V;
  static_assert(TPR <= 32, "sub-warp rows");
  __shared__ __align__(16) float tw[S];
  __shared__ float red[WARPS][2][S];
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const int g = blockIdx.x & (V - 1);
  const int e = (g * N) & (V - 1);
  const float inv_n = 1.f / float(N);
  const bool wgrad = !FUSE && part != nullptr;
  int lo[CH], hi[CH];
  uint4 mask[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const int j = c * TPR + lane;
    lo[c] = j == 0 ? e : 0;
    hi[c] = min(V, N + e - V * j);
    mask[c] = slot_mask<T>(lo[c], hi[c]);
  }
  float aw[E], ab[E];
#pragma unroll
  for (int k = 0; k < E; ++k) aw[k] = ab[k] = 0.f;
  const int64_t stride = GX / V;
  // the first tile's x / dy / statistics are loaded before the weight table is copied (the cold round trips overlap)
  int64_t u = blockIdx.x / V;
  uint4 xr[CH], dr[CH];
  float mu, rs;
  // x (saved by the forward) before the programmatic wait, the statistics (the forward's) and dy (the previous kernel's) after it
  const auto load_x = [&](int64_t uu) {
    const int64_t row = V * (uu * RPC + r_in) + g;
#pragma unroll
    for (int c = 0; c < CH; ++c) xr[c] = load_slot<T, TPR, CH>(x, row * N - e, c, lane, row < M, hi[c], mask[c]);
  };
  const auto load_dy = [&](int64_t uu) {
    const int64_t row = V * (uu * RPC + r_in) + g;
    const bool live = row < M;
    mu = live ? mean[row] : 0.f;
    rs = live ? rstd[row] : 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c)
      dr[c] = load_slot<T, TPR, CH>(dy, row * N - e, c, lane, live, hi[c], mask[c]);  // zeros outside the row: no contribution below
  };
  if constexpr (ONE) load_x(u);  // one tile a CTA: its loads go out before the table copy
  fill_table<S, WARPS * 32>(tw, w, wbf, 1.f, e, N);
  if constexpr (PDL) {
    pdl_wait();
    pdl_trigger();
  }
  if constexpr (ONE) load_dy(u);
  __syncthreads();
  for (bool first = ONE; V * (u * RPC) + g < M; u += stride, first = false) {
    const int64_t row = V * (u * RPC + r_in) + g;
    const bool live = row < M;
    const int64_t base = row * N - e;
    if (!first) {
      load_x(u);
      load_dy(u);
    }
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      float fx[V], fd[V], wv[V];
      unpack<T>(xr[c], fx);
      unpack<T>(dr[c], fd);
      loadv<V>(tw + V * (c * TPR + lane), wv);
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const int k = c * V + i;
        const float xh = (fx[i] - mu) * rs;
        const float gg = fd[i] * wv[i];
        s1 += gg;
        s2 += gg * xh;
        if (wgrad) {
          aw[k] += fd[i] * xh;
          ab[k] += fd[i];
        }
      }
    }
    s1 = row_sum<TPR>(s1, nullptr, 0, lane) * inv_n;
    s2 = row_sum<TPR>(s2, nullptr, 0, lane) * inv_n;
    if (live) {
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        if (!edge<CH>(c) || hi[c] > 0) {
          const int j = c * TPR + lane;
          float fx[V], fd[V], o[V], wv[V];
          unpack<T>(xr[c], fx);
          unpack<T>(dr[c], fd);
          loadv<V>(tw + V * j, wv);
#pragma unroll
          for (int i = 0; i < V; ++i) {
            const float xh = (fx[i] - mu) * rs;
            o[i] = rs * (fd[i] * wv[i] - s1 - xh * s2);
          }
          store_slot<T, CH>(dx + base + V * j, o, c, lo[c], hi[c]);
        }
      }
    }
  }
  if (!wgrad) return;
  // the row slots of a warp hold the same columns (one phase a CTA): fold them (fixed shuffle tree), then the CTA's warps in order
#pragma unroll
  for (int k = 0; k < E; ++k) {
#pragma unroll
    for (int o = 16; o >= TPR; o >>= 1) {
      aw[k] += __shfl_xor_sync(0xffffffffu, aw[k], o);
      ab[k] += __shfl_xor_sync(0xffffffffu, ab[k], o);
    }
  }
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) < TPR) {
#pragma unroll
    for (int k = 0; k < E; ++k) {
      red[warp][0][(k / V * TPR + lane) * V + k % V] = aw[k];
      red[warp][1][(k / V * TPR + lane) * V + k % V] = ab[k];
    }
  }
  __syncthreads();
  for (int col = threadIdx.x; col < N; col += WARPS * 32) {  // column col is element col + e of the vector stream
    float sw = 0.f, sb = 0.f;
#pragma unroll
    for (int wi = 0; wi < WARPS; ++wi) {
      sw += red[wi][0][col + e];
      sb += red[wi][1][col + e];
    }
    part[int64_t(blockIdx.x) * N + col] = sw;
    part[(int64_t(pstride) + blockIdx.x) * N + col] = sb;
  }
}

// the builds: (TPR, CH, WARPS)
#define MWLN_SKEW_LAYOUTS(X) X(8, 5, 2) X(8, 11, 2) X(32, 3, 2) X(16, 4, 2) X(32, 4, 2) X(8, 9, 2) X(32, 6, 2) X(32, 7, 2)

template <typename T>
bool dispatch(bool is_fwd, bool pf, int tpr, int ch, int warps, int grid, cudaStream_t s, const T* x, const T* dy, const void* w,
                     const void* b, int wbf, int bbf, T* out, float* mean, float* rstd, float* part, int pstride, int64_t M, int N, float eps, ColArgs ca = ColArgs{}) {
  const bool fuse = !is_fwd && ca.gx > 0 && ca.gx < grid;  // a fused backward: column CTAs after the gx dx CTAs (PDL builds only)
  // one tile a CTA when the grid covers every row tile of a phase (grid_for): that build loads its tile before the table copy
  // (pf: a forward over several tiles a CTA keeps the next tile's loads in flight while it reduces and stores the current one)
  const int gdx = fuse ? ca.gx : grid;  // the dx CTAs
  const bool one = int64_t(gdx / vec_of<T>()) * (warps * 32 / tpr) * vec_of<T>() >= M;
  const bool pdl = pdl_on(is_fwd, M * N * int64_t(sizeof(T)));  // the PDL builds (pdl_wait / pdl_trigger) for small activations only
#define MWLN_SL(TPR_, CH_, W_, ONE_)                                                                                         \
  if (is_fwd && pdl) launch(fwd<T, TPR_, CH_, W_, ONE_, false, true>, dim3(grid), dim3(W_ * 32), 0, s, true, x, w, b, wbf, bbf, out, mean, rstd, M, N, eps); \
  else if (is_fwd) launch(fwd<T, TPR_, CH_, W_, ONE_>, dim3(grid), dim3(W_ * 32), 0, s, false, x, w, b, wbf, bbf, out, mean, rstd, M, N, eps);           \
  else if (pdl && fuse) launch_cl(bwd<T, TPR_, CH_, W_, ONE_, true, true>, dim3(grid), dim3(W_ * 32), 0, s, true, ca.K, dy, x, w, wbf, mean, rstd, out, part, pstride, M, N, ca); \
  else if (pdl) launch(bwd<T, TPR_, CH_, W_, ONE_, true>, dim3(grid), dim3(W_ * 32), 0, s, true, dy, x, w, wbf, mean, rstd, out, part, pstride, M, N, ca); \
  else if (fuse) return false; \
  else launch(bwd<T, TPR_, CH_, W_, ONE_>, dim3(grid), dim3(W_ * 32), 0, s, false, dy, x, w, wbf, mean, rstd, out, part, pstride, M, N, ca);
#define MWLN_SX(TPR_, CH_, W_)                                                                                        \
  if (tpr == TPR_ && ch == CH_ && warps == W_) {                                                                      \
    if (one) { MWLN_SL(TPR_, CH_, W_, true) }                                                                         \
    else if (is_fwd && pf) { launch(fwd<T, TPR_, CH_, W_, false, true>, dim3(grid), dim3(W_ * 32), 0, s, false, x, w, b, wbf, bbf, out, mean, rstd, M, N, eps); } \
    else { MWLN_SL(TPR_, CH_, W_, false) }                                   \
    return true;                                                                                                      \
  }
  MWLN_SKEW_LAYOUTS(MWLN_SX)
#undef MWLN_SL
#undef MWLN_SX
  return false;
}

// whether layout (tpr, ch) holds every row of N elements with its partial / idle slots in the first and the last chunk only
inline bool covers(int64_t tpr, int64_t ch, int N, int V) {
  return N % 2 == 1 && ch >= 2 && tpr * ch * V >= N + V - 1 && (N - 1) / V >= (ch - 1) * tpr;
}

// the grid for M rows on RPC-row tiles: 8 phases x (tiles of a phase, at most ``ctas`` CTAs a SM in all)
inline int grid_for(int64_t M, int64_t rpc, int64_t ctas, int V) {
  const int64_t tiles = ((M + V - 1) / V + rpc - 1) / rpc, cap = std::max<int64_t>(1, int64_t(num_sms()) * ctas / V);
  return int(V * std::max<int64_t>(1, tiles < cap ? tiles : cap));
}

}  // namespace skew
}  // namespace mwln
