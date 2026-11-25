// B200 (sm_100a only) LayerNorm / RMSNorm rows: the sub-warp row kernels of the B200 port branch (origin/b200/ladder-final,
// kernels/layernorm/cuda/general/ln_rows.cu, dc425caa5), built for sm_100a alone as their own extension (layernorm_rows_sm100):
// no other card builds or runs this file. Changes against the branch: the column pass (the branch's ln_bwd.cu ln_col_reduce) lives
// here, the (16, 3) / (8, 3) / (32, 6) layouts are built too (a 384-wide bf16 row is 16 lanes x 3 vectors, a 768-wide fp32 row 32 x 6),
// and the bindings are in this file.
//
// General LayerNorm / RMSNorm, the B200 form for rows a (sub-)warp holds without idle lanes (TPR <= 32 threads a row, CHUNKS
// 16-byte vectors a thread, shuffle reductions only, no shared-memory round trip): small CTAs (WARPS warps) on a resident grid of
// ``ctas`` CTAs a SM, RPT rows a thread loaded before any reduction. The forward reads x through the non-coherent path without L1
// allocation and writes y with streaming stores (both touched once). Measured on B200 against the Triton kernels (``config_sm100``).
//
//   ln_rows_fwd  y = (x - mean) rstd w + b (RMS: no mean, no bias); mean / rstd per row when asked
//   ln_rows_bwd  dx, and per-CTA partial rows of dw / db: per thread over its rows, the row slots of a warp folded by shuffles, the
//                warps of the CTA in order -- fixed order, then ln_bwd.cu's column pass (one launch): a replay is bit-identical.
#include <algorithm>
#include "ln_common.cuh"
#include "ln_skew_sm100.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#ifndef MWLN_STREAM_ALL
#define MWLN_STREAM_ALL 0  // 1: the streaming load / store hints for 2-byte rows too (experiment)
#endif

namespace mwln {
namespace rows {

// the streaming hints are taken for fp32 rows only: measured on B200 they took an fp32 [147456, 128] forward 23.3 -> 21.1 us and
// cost the bf16 ones (a [147456, 128] bf16 forward 9.7 -> 12.5 us)
template <typename T, int VEC>
__device__ __forceinline__ Pack<T, VEC> ld_stream(const T* p) {
  if constexpr ((sizeof(T) == 4 || MWLN_STREAM_ALL) && sizeof(Pack<T, VEC>) == 16) {
    uint4 u;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w)
                 : "l"(p));
    return *reinterpret_cast<Pack<T, VEC>*>(&u);
  } else {
    return load_pack<T, VEC>(p);
  }
}

template <typename T, int VEC>
__device__ __forceinline__ void st_stream(T* p, const float* o) {
  Pack<T, VEC> pk;
#pragma unroll
  for (int i = 0; i < VEC; ++i) pk.v[i] = from_f<T>(o[i]);
  if constexpr ((sizeof(T) == 4 || MWLN_STREAM_ALL) && sizeof(Pack<T, VEC>) == 16) __stcs(reinterpret_cast<uint4*>(p), *reinterpret_cast<uint4*>(&pk));
  else *reinterpret_cast<Pack<T, VEC>*>(p) = pk;
}

// the row-sum exchange buffer of a row wider than a warp (TPR 64: one row a CTA of TPR threads, e.g. bf16 2560 = 160 lanes x 2 vectors)
template <int TPR>
__device__ __forceinline__ float* wide_buf() {
  if constexpr (TPR > 32) {
    __shared__ float buf[TPR / 32];
    return buf;
  } else {
    return nullptr;
  }
}

template <typename T, int TPR, int CH, int VEC, bool RMS, int RPT, int WARPS, bool PF = false, bool PDL = false>
__global__ void __launch_bounds__(WARPS * 32) fwd(const T* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b, int wbf,
                                                  int bbf, T* __restrict__ y, float* __restrict__ mo, float* __restrict__ ro, int64_t M,
                                                  int N, float eps) {
  static_assert(TPR <= 32 || TPR == WARPS * 32, "sub-warp rows, or one row a CTA");
  constexpr int RPC = WARPS * 32 / TPR, RT = RPC * RPT, E = CH * VEC;
  float* const rsum = wide_buf<TPR>();
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const float inv_n = 1.f / float(N);
  if constexpr (PDL) {  // x / the parameters may be the previous kernel's output
    pdl_wait();
    pdl_trigger();
  }
  float wr[E], br[E];
  load_affine<TPR, CH, VEC>(w, wbf, 1.f, lane, N, wr);
  if constexpr (!RMS) load_affine<TPR, CH, VEC>(b, bbf, 0.f, lane, N, br);
  Pack<T, VEC> raw[RPT][CH];
  const auto load = [&](int64_t tt) {
#pragma unroll
    for (int k = 0; k < RPT; ++k) {
      const int64_t row = tt * RT + k * RPC + r_in;
#pragma unroll
      for (int c = 0; c < CH; ++c)
        raw[k][c] = row < M ? ld_stream<T, VEC>(x + row * N + (c * TPR + lane) * VEC) : zero_pack<T, VEC>();
    }
  };
  if constexpr (PF) load(blockIdx.x);
  for (int64_t t = blockIdx.x; t * RT < M; t += gridDim.x) {
    if constexpr (!PF) load(t);
    Pack<T, VEC> cur[RPT][CH];  // PF: this tile's rows; the next tile's loads go out before the reductions
    if constexpr (PF) {
#pragma unroll
      for (int k = 0; k < RPT; ++k)
#pragma unroll
        for (int c = 0; c < CH; ++c) cur[k][c] = raw[k][c];
      load(t + gridDim.x);
    }
    auto& rw = PF ? cur : raw;
#pragma unroll
    for (int k = 0; k < RPT; ++k) {
      const int64_t row = t * RT + k * RPC + r_in;
      float mean = 0.f;
      if constexpr (!RMS) {
        float s = 0.f;
#pragma unroll
        for (int c = 0; c < CH; ++c)
#pragma unroll
          for (int i = 0; i < VEC; ++i) s += to_f<T>(rw[k][c].v[i]);
        mean = row_sum<TPR>(s, rsum, r_in, lane) * inv_n;
      }
      float q = 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c)
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const float d = to_f<T>(rw[k][c].v[i]) - mean;
          q += d * d;
        }
      const float rs = rsqrtf(row_sum<TPR>(q, rsum, r_in, lane) * inv_n + eps);
      if (row < M) {
#pragma unroll
        for (int c = 0; c < CH; ++c) {
          float o[VEC];
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const float xn = (to_f<T>(rw[k][c].v[i]) - mean) * rs;
            o[i] = RMS ? xn * wr[c * VEC + i] : xn * wr[c * VEC + i] + br[c * VEC + i];
          }
          st_stream<T, VEC>(y + row * N + (c * TPR + lane) * VEC, o);
        }
        if (lane == 0) {
          if (mo) mo[row] = mean;
          if (ro) ro[row] = rs;
        }
      }
    }
  }
}

// The non-persistent forward (``ctas`` 0): one RT-row tile a CTA, every load of the thread's rows issued before the affine loads and
// before any reduction (a resident CTA would otherwise pay the parameter-load latency, then the row-load latency, per tile).
template <typename T, int TPR, int CH, int VEC, bool RMS, int RPT, int WARPS, bool PDL = false>
__global__ void __launch_bounds__(WARPS * 32) fwd_direct(const T* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b, int wbf,
                                                         int bbf, T* __restrict__ y, float* __restrict__ mo, float* __restrict__ ro, int64_t M,
                                                         int N, float eps) {
  static_assert(TPR <= 32 || TPR == WARPS * 32, "sub-warp rows, or one row a CTA");
  constexpr int RPC = WARPS * 32 / TPR, RT = RPC * RPT, E = CH * VEC;
  float* const rsum = wide_buf<TPR>();
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const float inv_n = 1.f / float(N);
  if constexpr (PDL) {
    pdl_wait();
    pdl_trigger();
  }
  const int64_t t = blockIdx.x;
  Pack<T, VEC> raw[RPT][CH];
#pragma unroll
  for (int k = 0; k < RPT; ++k) {
    const int64_t row = t * RT + k * RPC + r_in;
#pragma unroll
    for (int c = 0; c < CH; ++c)
      raw[k][c] = row < M ? ld_stream<T, VEC>(x + row * N + (c * TPR + lane) * VEC) : zero_pack<T, VEC>();
  }
  float wr[E], br[E];
  load_affine<TPR, CH, VEC>(w, wbf, 1.f, lane, N, wr);
  if constexpr (!RMS) load_affine<TPR, CH, VEC>(b, bbf, 0.f, lane, N, br);
#pragma unroll
  for (int k = 0; k < RPT; ++k) {
    const int64_t row = t * RT + k * RPC + r_in;
    float mean = 0.f;
    if constexpr (!RMS) {
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c)
#pragma unroll
        for (int i = 0; i < VEC; ++i) s += to_f<T>(raw[k][c].v[i]);
      mean = row_sum<TPR>(s, rsum, r_in, lane) * inv_n;
    }
    float q = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c)
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const float d = to_f<T>(raw[k][c].v[i]) - mean;
        q += d * d;
      }
    const float rs = rsqrtf(row_sum<TPR>(q, rsum, r_in, lane) * inv_n + eps);
    if (row < M) {
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        float o[VEC];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const float xn = (to_f<T>(raw[k][c].v[i]) - mean) * rs;
          o[i] = RMS ? xn * wr[c * VEC + i] : xn * wr[c * VEC + i] + br[c * VEC + i];
        }
        st_stream<T, VEC>(y + row * N + (c * TPR + lane) * VEC, o);
      }
      if (lane == 0) {
        if (mo) mo[row] = mean;
        if (ro) ro[row] = rs;
      }
    }
  }
}

// The lean non-persistent forward (``ctas`` -1, 2026-11-17 round 6): one-warp CTAs of RPT rows a thread, every row load issued first,
// the affine values read at use (one 16-byte chunk a time, L1-resident) instead of held in registers, and a launch bound asking for
// the whole SM (2048 threads): the HBM-bound pair-stream forwards keep more bytes in flight (measured, docs/b200/transition.md round 6:
// bf16 [4194304, 128] 349 -> 329 us, Triton 321; the plain copy of the same bytes 316).
template <typename T, int TPR, int CH, int VEC, bool RMS, int RPT, int LW = 1>
__global__ void __launch_bounds__(LW * 32, 2048 / (LW * 32)) fwd_lean(const T* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b, int wbf,
                                                   int bbf, T* __restrict__ y, float* __restrict__ mo, float* __restrict__ ro, int64_t M,
                                                   int N, float eps) {
  static_assert(TPR <= 32, "sub-warp rows");
  constexpr int RPC = LW * 32 / TPR, RT = RPC * RPT;  // LW 4 (round 8): four-warp CTAs, 16 a SM (one-warp CTAs stop at 32 a SM)
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const float inv_n = 1.f / float(N);
  const int64_t t = blockIdx.x;
  Pack<T, VEC> raw[RPT][CH];
#pragma unroll
  for (int k = 0; k < RPT; ++k) {
    const int64_t row = t * RT + k * RPC + r_in;
#pragma unroll
    for (int c = 0; c < CH; ++c)
      raw[k][c] = row < M ? ld_stream<T, VEC>(x + row * N + (c * TPR + lane) * VEC) : zero_pack<T, VEC>();
  }
  // the affine loads go out right behind the row loads (read after the reductions, their latency sat on the CTA's critical path)
  float wr[CH * VEC], br[CH * VEC];
  load_affine<TPR, CH, VEC>(w, wbf, 1.f, lane, N, wr);
  if constexpr (!RMS) load_affine<TPR, CH, VEC>(b, bbf, 0.f, lane, N, br);
  float mean[RPT], rs[RPT];
#pragma unroll
  for (int k = 0; k < RPT; ++k) {
    float m = 0.f;
    if constexpr (!RMS) {
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c)
#pragma unroll
        for (int i = 0; i < VEC; ++i) s += to_f<T>(raw[k][c].v[i]);
      m = row_sum<TPR>(s, nullptr, r_in, lane) * inv_n;
    }
    float q = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c)
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const float d = to_f<T>(raw[k][c].v[i]) - m;
        q += d * d;
      }
    mean[k] = m;
    rs[k] = rsqrtf(row_sum<TPR>(q, nullptr, r_in, lane) * inv_n + eps);
  }
#pragma unroll
  for (int c = 0; c < CH; ++c) {
#pragma unroll
    for (int k = 0; k < RPT; ++k) {
      const int64_t row = t * RT + k * RPC + r_in;
      if (row >= M) continue;
      float o[VEC];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const float xn = (to_f<T>(raw[k][c].v[i]) - mean[k]) * rs[k];
        o[i] = RMS ? xn * wr[c * VEC + i] : xn * wr[c * VEC + i] + br[c * VEC + i];
      }
      st_stream<T, VEC>(y + row * N + (c * TPR + lane) * VEC, o);
    }
  }
#pragma unroll
  for (int k = 0; k < RPT; ++k) {
    const int64_t row = t * RT + k * RPC + r_in;
    if (row < M && lane == 0) {
      if (mo) mo[row] = mean[k];
      if (ro) ro[row] = rs[k];
    }
  }
}

// PF (round 8): the next tile's x / dy / statistics in flight while the current one is reduced (Triton's two-stage schedule) -- measured
// slower on every launch of bf16 [589824 / 262144, 256] (best 300.8 / 145.3 us against 270.1 / 130.8 without), so no dispatch builds it.
template <typename T, int TPR, int CH, int VEC, bool RMS, int RPT, int WARPS, bool PDL = false, bool FUSE = false, bool PF = false>
__global__ void __launch_bounds__(WARPS * 32) bwd(const T* __restrict__ dy, const T* __restrict__ x, const void* __restrict__ w, int wbf,
                                                  const float* __restrict__ mean, const float* __restrict__ rstd, T* __restrict__ dx,
                                                  float* __restrict__ part, int pstride, int64_t M, int N, ColArgs ca) {
  if constexpr (FUSE) {  // the column CTAs of a fused backward (col_role)
    if (int(blockIdx.x) >= ca.gx) {
      if constexpr (PDL) {
        pdl_wait();
        pdl_trigger();
      }
      col_role<T, WARPS * 32, RMS, false>(ca, dy, x, mean, rstd, M, N);
      return;
    }
  }
  const int64_t GX = FUSE ? int64_t(ca.gx) : int64_t(gridDim.x);
  static_assert(TPR <= 32 || TPR == WARPS * 32, "sub-warp rows, or one row a CTA");
  constexpr int RPC = WARPS * 32 / TPR, RT = RPC * RPT, E = CH * VEC;
  float* const rsum = wide_buf<TPR>();
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const float inv_n = 1.f / float(N);
  const bool wgrad = !FUSE && part != nullptr;
  float wr[E], aw[E], ab[E];
  load_affine<TPR, CH, VEC>(w, wbf, 1.f, lane, N, wr);
#pragma unroll
  for (int e = 0; e < E; ++e) aw[e] = ab[e] = 0.f;
  Pack<T, VEC> xr[RPT][CH], dr[RPT][CH];
  float mu[RPT], rs[RPT];
  const auto load_x = [&](int64_t tt, Pack<T, VEC> (&xa)[RPT][CH]) {
#pragma unroll
    for (int k = 0; k < RPT; ++k) {
      const int64_t row = tt * RT + k * RPC + r_in;
      const bool v = row < M;
#pragma unroll
      for (int c = 0; c < CH; ++c) xa[k][c] = v ? load_pack<T, VEC>(x + row * N + (c * TPR + lane) * VEC) : zero_pack<T, VEC>();
    }
  };
  const auto load_d = [&](int64_t tt, Pack<T, VEC> (&da)[RPT][CH], float (&ma)[RPT], float (&ra)[RPT]) {
#pragma unroll
    for (int k = 0; k < RPT; ++k) {
      const int64_t row = tt * RT + k * RPC + r_in;
      const bool v = row < M;
      ma[k] = v && !RMS ? mean[row] : 0.f;
      ra[k] = v ? rstd[row] : 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c) da[k][c] = v ? load_pack<T, VEC>(dy + row * N + (c * TPR + lane) * VEC) : zero_pack<T, VEC>();
    }
  };
  int64_t t = blockIdx.x;
  if constexpr (PF) {  // the first tile; every later tile's loads go out while the one before it is reduced and stored
    if (t * RT < M) {
      load_x(t, xr);
      load_d(t, dr, mu, rs);
    }
  }
  for (; t * RT < M; t += GX) {
    if constexpr (!PF) {
      // x (saved by the forward, written long before) and the weights go out before the wait; the statistics (the forward's) and dy
      // (the previous kernel's) after it
      load_x(t, xr);
      if constexpr (PDL) {
        pdl_wait();
        pdl_trigger();
      }
      load_d(t, dr, mu, rs);
    }
    Pack<T, VEC> xn[RPT][CH], dn[RPT][CH];
    float mn[RPT], rn[RPT];
    if constexpr (PF) {
      if ((t + GX) * RT < M) {
        load_x(t + GX, xn);
        load_d(t + GX, dn, mn, rn);
      }
    }
#pragma unroll
    for (int k = 0; k < RPT; ++k) {
      const int64_t row = t * RT + k * RPC + r_in;
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c)
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const int e = c * VEC + i;
          const float dv = to_f<T>(dr[k][c].v[i]);
          const float xh = (to_f<T>(xr[k][c].v[i]) - mu[k]) * rs[k];
          const float g = dv * wr[e];
          s1 += g;
          s2 += g * xh;
          if (wgrad) {
            aw[e] += dv * xh;
            if constexpr (!RMS) ab[e] += dv;
          }
        }
      if constexpr (RMS) s1 = 0.f;
      else s1 = row_sum<TPR>(s1, rsum, r_in, lane) * inv_n;
      s2 = row_sum<TPR>(s2, rsum, r_in, lane) * inv_n;
      if (row < M) {
#pragma unroll
        for (int c = 0; c < CH; ++c) {
          float o[VEC];
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const int e = c * VEC + i;
            const float xh = (to_f<T>(xr[k][c].v[i]) - mu[k]) * rs[k];
            o[i] = rs[k] * (to_f<T>(dr[k][c].v[i]) * wr[e] - s1 - xh * s2);
          }
          store_vec<T, VEC>(dx + row * N + (c * TPR + lane) * VEC, o);
        }
      }
    }
    if constexpr (PF) {
#pragma unroll
      for (int k = 0; k < RPT; ++k) {
        mu[k] = mn[k];
        rs[k] = rn[k];
#pragma unroll
        for (int c = 0; c < CH; ++c) {
          xr[k][c] = xn[k][c];
          dr[k][c] = dn[k][c];
        }
      }
    }
  }
  if (!wgrad) return;
  if constexpr (TPR > 32) {  // one row a CTA: a column is one thread's
#pragma unroll
    for (int e = 0; e < E; ++e) {
      const int col = ((e / VEC) * TPR + lane) * VEC + e % VEC;
      part[int64_t(blockIdx.x) * N + col] = aw[e];
      if constexpr (!RMS) part[(int64_t(pstride) + blockIdx.x) * N + col] = ab[e];
    }
    return;
  }
  // the row slots of a warp hold the same columns: fold them (fixed shuffle tree), then the CTA's warps in order
  __shared__ float red[WARPS][2][E * TPR];
#pragma unroll
  for (int e = 0; e < E; ++e) {
#pragma unroll
    for (int o = 16; o >= TPR; o >>= 1) {
      aw[e] += __shfl_xor_sync(0xffffffffu, aw[e], o);
      if constexpr (!RMS) ab[e] += __shfl_xor_sync(0xffffffffu, ab[e], o);
    }
  }
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) < TPR) {
#pragma unroll
    for (int e = 0; e < E; ++e) {
      red[warp][0][e * TPR + lane] = aw[e];
      if constexpr (!RMS) red[warp][1][e * TPR + lane] = ab[e];
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < E * TPR; idx += WARPS * 32) {
    const int e = idx / TPR, l = idx % TPR;
    const int col = ((e / VEC) * TPR + l) * VEC + e % VEC;
    float s = 0.f, sb = 0.f;
#pragma unroll
    for (int wi = 0; wi < WARPS; ++wi) {
      s += red[wi][0][idx];
      if constexpr (!RMS) sb += red[wi][1][idx];
    }
    part[int64_t(blockIdx.x) * N + col] = s;
    if constexpr (!RMS) part[(int64_t(pstride) + blockIdx.x) * N + col] = sb;
  }
}


// out[col] = sum over the P partial rows in a fixed order (the branch's ln_bwd.cu column pass): 32 columns x 32 row slices per CTA,
// the slices summed in order; blockIdx.y 0: dw (part, out), 1: db (part + P N, out2).
__global__ void __launch_bounds__(1024) col_reduce(const float* __restrict__ part, int P, int N, void* __restrict__ out,
                                                  void* __restrict__ out2, int obf) {
  __shared__ float sm[32][33];
  pdl_wait();  // the partial rows are the backward's
  pdl_trigger();
  const float* src = part + (blockIdx.y ? int64_t(P) * N : 0);
  void* dst = blockIdx.y ? out2 : out;
  const int tx = threadIdx.x & 31, sl = threadIdx.x >> 5;
  const int col = blockIdx.x * 32 + tx;
  float s = 0.f;
  if (col < N) {
    int p = sl;
    for (; p + 96 < P; p += 128) {
      const float a = src[int64_t(p) * N + col], b = src[int64_t(p + 32) * N + col];
      const float c = src[int64_t(p + 64) * N + col], d = src[int64_t(p + 96) * N + col];
      s += a; s += b; s += c; s += d;
    }
    for (; p < P; p += 32) s += src[int64_t(p) * N + col];
  }
  sm[sl][tx] = s;
  __syncthreads();
  if (sl == 0 && col < N) {
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) t += sm[i][tx];
    if (obf) reinterpret_cast<__nv_bfloat16*>(dst)[col] = __float2bfloat16(t);
    else reinterpret_cast<float*>(dst)[col] = t;
  }
}
// the builds: (TPR, CH) layouts x (RPT, WARPS) in {1, 2} x {1, 2, 4} and {4, 8} x {2, 4}
#define MWLN_ROWS_LAYOUTS(X, ...)                                                                                            \
  X(4, 1, __VA_ARGS__) X(4, 2, __VA_ARGS__) X(4, 4, __VA_ARGS__) X(8, 1, __VA_ARGS__) X(8, 2, __VA_ARGS__) X(8, 4, __VA_ARGS__) \
  X(8, 3, __VA_ARGS__) X(16, 1, __VA_ARGS__) X(16, 2, __VA_ARGS__) X(16, 3, __VA_ARGS__) X(16, 4, __VA_ARGS__) X(32, 1, __VA_ARGS__) X(32, 2, __VA_ARGS__) X(32, 3, __VA_ARGS__) \
  X(32, 4, __VA_ARGS__) X(32, 6, __VA_ARGS__)

template <typename T, int VEC, bool RMS>
bool dispatch(bool is_fwd, bool pf, bool lean, int tpr, int ch, int rpt, int warps, int grid, bool direct, cudaStream_t s, const T* x, const T* dy, const void* w,
              const void* b, int wbf, int bbf, T* out, float* mean, float* rstd, float* part, int pstride, int64_t M, int N, float eps, ColArgs ca = ColArgs{}) {
  const bool fuse = !is_fwd && ca.gx > 0 && ca.gx < grid;  // a fused backward: column CTAs after the gx dx CTAs (PDL builds only)
  const bool pdl = pdl_on(is_fwd, M * N * int64_t(sizeof(T)));  // the PDL builds (pdl_wait / pdl_trigger) for small activations only
  if (is_fwd && lean) {  // the lean non-persistent forward (fwd_lean): one-warp CTAs, RPT 1 / 2 / 4
#define MWLN_LEAN(TPR_, CH_)                                                                                          \
    if (tpr == TPR_ && ch == CH_) {                                                                                 \
      if (rpt == 1) launch(fwd_lean<T, TPR_, CH_, VEC, RMS, 1>, dim3(grid), dim3(32), 0, s, false, x, w, b, wbf, bbf, out, mean, rstd, M, N, eps);       \
      else if (rpt == 2) launch(fwd_lean<T, TPR_, CH_, VEC, RMS, 2>, dim3(grid), dim3(32), 0, s, false, x, w, b, wbf, bbf, out, mean, rstd, M, N, eps);  \
      else if (rpt == 4) launch(fwd_lean<T, TPR_, CH_, VEC, RMS, 4>, dim3(grid), dim3(32), 0, s, false, x, w, b, wbf, bbf, out, mean, rstd, M, N, eps);  \
      else return false;                                                                                            \
      return true;                                                                                                  \
    }
    MWLN_LEAN(4, 2) MWLN_LEAN(8, 1) MWLN_LEAN(8, 2) MWLN_LEAN(16, 1) MWLN_LEAN(16, 2) MWLN_LEAN(16, 3) MWLN_LEAN(8, 4) MWLN_LEAN(16, 4)
    MWLN_LEAN(32, 1) MWLN_LEAN(32, 2) MWLN_LEAN(32, 3) MWLN_LEAN(32, 4) MWLN_LEAN(4, 4)
#undef MWLN_LEAN
    return false;
  }
  if (is_fwd && pf && !direct && rpt == 1 && warps == 2) {  // the resident forward with the next tile's loads in flight (fwd<PF>)
#define MWLN_PF(TPR_, CH_)                                                                                        \
    if (tpr == TPR_ && ch == CH_) {                                                                               \
      launch(fwd<T, TPR_, CH_, VEC, RMS, 1, 2, true>, dim3(grid), dim3(64), 0, s, false, x, w, b, wbf, bbf, out, mean, rstd, M, N, eps);   \
      return true;                                                                                                \
    }
    MWLN_PF(16, 3)
#undef MWLN_PF
  }
#define MWLN_RL(R, W, TPR_, CH_)                                                                             \
  if (rpt == R && warps == W) {                                                                              \
    if constexpr (pdl_rl(R, W)) {                                                                            \
      if (pdl && is_fwd && direct) launch(fwd_direct<T, TPR_, CH_, VEC, RMS, R, W, true>, dim3(grid), dim3(W * 32), 0, s, true, x, w, b, wbf, bbf, out, mean, rstd, M, N, eps); \
      else if (pdl && is_fwd) launch(fwd<T, TPR_, CH_, VEC, RMS, R, W, false, true>, dim3(grid), dim3(W * 32), 0, s, true, x, w, b, wbf, bbf, out, mean, rstd, M, N, eps); \
      else if (pdl && fuse) launch_cl(bwd<T, TPR_, CH_, VEC, RMS, R, W, true, true>, dim3(grid), dim3(W * 32), 0, s, true, ca.K, dy, x, w, wbf, mean, rstd, out, part, pstride, M, N, ca); \
      else if (pdl) launch(bwd<T, TPR_, CH_, VEC, RMS, R, W, true>, dim3(grid), dim3(W * 32), 0, s, true, dy, x, w, wbf, mean, rstd, out, part, pstride, M, N, ca); \
      if (pdl) return true;                                                                                  \
    }                                                                                                        \
    if (is_fwd && direct) launch(fwd_direct<T, TPR_, CH_, VEC, RMS, R, W>, dim3(grid), dim3(W * 32), 0, s, false, x, w, b, wbf, bbf, out, mean, rstd, M, N, eps); \
    else if (is_fwd) launch(fwd<T, TPR_, CH_, VEC, RMS, R, W>, dim3(grid), dim3(W * 32), 0, s, false, x, w, b, wbf, bbf, out, mean, rstd, M, N, eps); \
    else if (fuse) return false;                                                                             \
    else launch(bwd<T, TPR_, CH_, VEC, RMS, R, W>, dim3(grid), dim3(W * 32), 0, s, false, dy, x, w, wbf, mean, rstd, out, part, pstride, M, N, ca);  \
    return true;                                                                                             \
  }
#define MWLN_RX(TPR_, CH_, unused)                                                                                       \
  if (tpr == TPR_ && ch == CH_) {                                                                                         \
    MWLN_RL(1, 1, TPR_, CH_) MWLN_RL(2, 1, TPR_, CH_) MWLN_RL(1, 2, TPR_, CH_) MWLN_RL(2, 2, TPR_, CH_) MWLN_RL(1, 4, TPR_, CH_) MWLN_RL(2, 4, TPR_, CH_) MWLN_RL(4, 2, TPR_, CH_) MWLN_RL(4, 4, TPR_, CH_) MWLN_RL(8, 2, TPR_, CH_) MWLN_RL(8, 4, TPR_, CH_)                   \
  }
  MWLN_ROWS_LAYOUTS(MWLN_RX, 0)
#undef MWLN_RX
  if (tpr == 160 && ch == 2) {  // rows wider than a warp: one row a five-warp CTA (bf16 2560), 1 / 2 / 4 rows a thread
    MWLN_RL(1, 5, 160, 2)
    MWLN_RL(2, 5, 160, 2)
    MWLN_RL(4, 5, 160, 2)
  }
#undef MWLN_RL
  return false;
}

}  // namespace rows
}  // namespace mwln


// x [M, N] bf16 / fp32 contiguous, rows 16-byte aligned; w / b fp32 or bf16 [N] or empty. (y, mean, rstd) like ln_fwd.
std::vector<torch::Tensor> ln_rows_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, bool rms, int64_t tpr,
                                       int64_t ch, int64_t rpt, int64_t warps, int64_t ctas, bool pf) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x");
  TORCH_CHECK(!rms || b.numel() == 0, "RMSNorm has no bias");
  TORCH_CHECK(tpr * ch * (16 / x.element_size()) == x.size(1), "B200 LayerNorm rows: layout ", tpr, " x ", ch, " does not cover width ", x.size(1));
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  auto y = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto mean = save && !rms ? torch::empty({M}, f32) : torch::empty({0}, f32);
  auto rstd = save ? torch::empty({M}, f32) : torch::empty({0}, f32);
  if (M == 0) return {y, mean, rstd};
  for (auto* t : {&w, &b})
    if (t->numel() && (reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 || !t->is_contiguous())) *t = t->contiguous().clone();
  const int wbf = w.numel() && w.scalar_type() == torch::kBFloat16, bbf = b.numel() && b.scalar_type() == torch::kBFloat16;
  const int64_t rt = int64_t(warps) * 32 / tpr * rpt;
  const int64_t tiles = (M + rt - 1) / rt, cap = int64_t(mwln::num_sms()) * ctas;
  const int grid = int(ctas <= 0 || tiles < cap ? tiles : cap);  // ctas 0 / -1: the non-persistent / lean forward, a tile a CTA
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
  const void* wp = w.numel() ? w.data_ptr() : nullptr;
  const void* bp = b.numel() ? b.data_ptr() : nullptr;
  float* mp = mean.numel() ? mean.data_ptr<float>() : nullptr;
  float* rp = rstd.numel() ? rstd.data_ptr<float>() : nullptr;
#define MWLN_GO(CT, V)                                                                                                         \
  ok = (rms ? mwln::rows::dispatch<CT, V, true> : mwln::rows::dispatch<CT, V, false>)(                                         \
      true, pf, ctas < 0, int(tpr), int(ch), int(rpt), int(warps), grid, ctas == 0, s, reinterpret_cast<const CT*>(x.data_ptr()), nullptr, wp, bp, wbf, bbf, \
      reinterpret_cast<CT*>(y.data_ptr()), mp, rp, nullptr, 0, M, N, float(eps), mwln::ColArgs{});
  if (x.scalar_type() == torch::kBFloat16) { MWLN_GO(__nv_bfloat16, 8) }
  else if (x.scalar_type() == torch::kFloat32) { MWLN_GO(float, 4) }
#undef MWLN_GO
  TORCH_CHECK(ok, "B200 LayerNorm rows forward: no build for tpr=", tpr, " ch=", ch, " rpt=", rpt, " warps=", warps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}

// (dx, dw, db) like ln_bwd; `ctas` CTAs a SM = the partial rows' count / SMs.
std::vector<torch::Tensor> ln_rows_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd, bool wgrad,
                                       bool rms, int64_t tpr, int64_t ch, int64_t rpt, int64_t warps, int64_t ctas, bool gbf16, bool pf, int64_t fk) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && dy.is_contiguous() && dy.sizes() == x.sizes() &&
                  dy.scalar_type() == x.scalar_type() && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(dy.data_ptr()) % 16 == 0,
              "dy, x");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  TORCH_CHECK(tpr * ch * (16 / x.element_size()) == N, "B200 LayerNorm rows: layout ", tpr, " x ", ch, " does not cover width ", N);
  auto dx = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto gopt = x.options().dtype(gbf16 ? torch::kBFloat16 : torch::kFloat32);
  if (w.numel() && (reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 || !w.is_contiguous())) w = w.contiguous().clone();
  const int wbf = w.numel() && w.scalar_type() == torch::kBFloat16;
  const int64_t rt = int64_t(warps) * 32 / tpr * rpt;
  const int64_t tiles = (M + rt - 1) / rt, cap = int64_t(mwln::num_sms()) * ctas;
  const int grid = int(std::max<int64_t>(1, tiles < cap ? tiles : cap));
  // fused (round 8): column CTAs after the dx CTAs sum dw / db (mwln::col_role), no partial rows, no column-pass launch
  // fk: the cluster size of the column CTAs (0: not fused); set_fuse overrides it (A/B)
  const int64_t kt = mwln::fuse_max_rows() < 0 ? fk : M <= mwln::fuse_max_rows() ? mwln::fuse_k() : 0;
  const bool fuse = wgrad && M > 0 && kt > 0 && mwln::pdl_on(false, M * N * int64_t(x.element_size())) &&
                    mwln::pdl_rl(int(rpt), int(warps));
  const mwln::ColPlan cp = fuse ? mwln::col_plan(M, N, 16 / int(x.element_size()), false, int(warps), int(kt), grid) : mwln::ColPlan{0, 0, grid, grid};
  auto part = fuse ? torch::empty({0}, f32) : wgrad ? torch::empty({rms ? 1 : 2, grid, N}, f32) : torch::empty({0}, f32);
  auto dw = wgrad ? (M == 0 ? torch::zeros({N}, gopt) : torch::empty({N}, gopt)) : torch::empty({0}, f32);
  auto db = wgrad && !rms ? (M == 0 ? torch::zeros({N}, gopt) : torch::empty({N}, gopt)) : torch::empty({0}, f32);
  if (M == 0) return {dx, dw, db};
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
  float* pp = wgrad && !fuse ? part.data_ptr<float>() : nullptr;
  mwln::ColArgs ca{};
  if (fuse)
    ca = mwln::ColArgs{dw.data_ptr(), rms ? nullptr : db.data_ptr(), int(gbf16), cp.gx, cp.K, cp.nrt, mwln::fuse_dbg()};
  const int lgrid = cp.grid;
#define MWLN_GO(CT, V)                                                                                                             \
  ok = (rms ? mwln::rows::dispatch<CT, V, true> : mwln::rows::dispatch<CT, V, false>)(                                             \
      false, false, false, int(tpr), int(ch), int(rpt), int(warps), lgrid, false, s, reinterpret_cast<const CT*>(x.data_ptr()),                            \
      reinterpret_cast<const CT*>(dy.data_ptr()), w.numel() ? w.data_ptr() : nullptr, nullptr, wbf, 0, reinterpret_cast<CT*>(dx.data_ptr()), \
      rms ? nullptr : mean.data_ptr<float>(), rstd.data_ptr<float>(), pp, grid, M, N, 0.f, ca);
  if (x.scalar_type() == torch::kBFloat16) { MWLN_GO(__nv_bfloat16, 8) }
  else if (x.scalar_type() == torch::kFloat32) { MWLN_GO(float, 4) }
#undef MWLN_GO
  TORCH_CHECK(ok, "B200 LayerNorm rows backward: no build for tpr=", tpr, " ch=", ch, " rpt=", rpt, " warps=", warps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (wgrad && !fuse) {
    mwln::launch(mwln::rows::col_reduce, dim3((N + 31) / 32, rms ? 1 : 2), dim3(1024), 0, s, (mwln::pdl_mode() & 4) != 0, pp, grid, N, dw.data_ptr(), rms ? nullptr : db.data_ptr(), int(gbf16));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dx, dw, db};
}


// Skewed rows (ln_skew_sm100.cuh): the kernels read whole 16-byte vectors, so the vector holding the tensor's last element must lie
// inside its storage (the Python side pads a tensor that ends mid-vector at the end of its storage).
static bool skew_tail_ok(const torch::Tensor& t) {
  const int64_t es = t.element_size(), v = 16 / es, n = t.numel(), r = n % v;
  return r == 0 || int64_t(t.storage().nbytes()) - (t.storage_offset() + n) * es >= (v - r) * es;
}
static bool skew_dtype(const torch::Tensor& t) { return t.scalar_type() == torch::kBFloat16 || t.scalar_type() == torch::kFloat32; }

// x [M, N] bf16 / fp32 contiguous, 16-byte aligned, N odd, a layout that covers N; w / b fp32 or bf16 [N] or empty. (y, mean, rstd).
std::vector<torch::Tensor> ln_skew_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, int64_t tpr, int64_t ch,
                                       int64_t warps, int64_t ctas, bool pf) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && skew_dtype(x) && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x");
  const int64_t M = x.size(0);
  const int N = int(x.size(1)), V = 16 / int(x.element_size());
  TORCH_CHECK(mwln::skew::covers(tpr, ch, N, V) && ctas > 0, "B200 LayerNorm skewed rows: layout ", tpr, " x ", ch, " does not cover width ", N);
  TORCH_CHECK(skew_tail_ok(x), "B200 LayerNorm skewed rows: the last 16-byte vector of x crosses its storage");
  auto y = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto mean = save ? torch::empty({M}, f32) : torch::empty({0}, f32);
  auto rstd = save ? torch::empty({M}, f32) : torch::empty({0}, f32);
  if (M == 0) return {y, mean, rstd};
  for (auto* t : {&w, &b})
    if (t->numel() && !t->is_contiguous()) *t = t->contiguous();
  const int wbf = w.numel() && w.scalar_type() == torch::kBFloat16, bbf = b.numel() && b.scalar_type() == torch::kBFloat16;
  const int grid = mwln::skew::grid_for(M, warps * 32 / tpr, ctas, V);
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
#define MWLN_GO(CT)                                                                                                                     \
  ok = mwln::skew::dispatch<CT>(true, pf, int(tpr), int(ch), int(warps), grid, s, reinterpret_cast<const CT*>(x.data_ptr()), nullptr,       \
                                w.numel() ? w.data_ptr() : nullptr, b.numel() ? b.data_ptr() : nullptr, wbf, bbf,                       \
                                reinterpret_cast<CT*>(y.data_ptr()), save ? mean.data_ptr<float>() : nullptr,                           \
                                save ? rstd.data_ptr<float>() : nullptr, nullptr, 0, M, N, float(eps));
  if (V == 8) { MWLN_GO(__nv_bfloat16) }
  else { MWLN_GO(float) }
#undef MWLN_GO
  TORCH_CHECK(ok, "B200 LayerNorm skewed rows forward: no build for tpr=", tpr, " ch=", ch, " warps=", warps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}

// (dx, dw, db) like ln_rows_bwd; the grid (8 phases x tiles, at most `ctas` CTAs a SM) is the partial rows' count.
std::vector<torch::Tensor> ln_skew_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd, bool wgrad,
                                       int64_t tpr, int64_t ch, int64_t warps, int64_t ctas, bool gbf16, int64_t fk) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && dy.is_contiguous() && dy.sizes() == x.sizes() &&
                  skew_dtype(x) && dy.scalar_type() == x.scalar_type() &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(dy.data_ptr()) % 16 == 0,
              "dy, x");
  const int64_t M = x.size(0);
  const int N = int(x.size(1)), V = 16 / int(x.element_size());
  TORCH_CHECK(mwln::skew::covers(tpr, ch, N, V) && ctas > 0, "B200 LayerNorm skewed rows: layout ", tpr, " x ", ch, " does not cover width ", N);
  TORCH_CHECK(skew_tail_ok(x) && skew_tail_ok(dy), "B200 LayerNorm skewed rows: the last 16-byte vector of x / dy crosses its storage");
  auto dx = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto gopt = x.options().dtype(gbf16 ? torch::kBFloat16 : torch::kFloat32);
  if (w.numel() && !w.is_contiguous()) w = w.contiguous();
  const int wbf = w.numel() && w.scalar_type() == torch::kBFloat16;
  const int grid = mwln::skew::grid_for(M, warps * 32 / tpr, ctas, V);
  // fused (round 8): column CTAs after the dx CTAs sum dw / db (mwln::col_role), no partial rows, no column-pass launch
  // fk: the cluster size of the column CTAs (0: not fused); set_fuse overrides it (A/B)
  const int64_t kt = mwln::fuse_max_rows() < 0 ? fk : M <= mwln::fuse_max_rows() ? mwln::fuse_k() : 0;
  const bool fuse = wgrad && M > 0 && kt > 0 && mwln::pdl_on(false, M * N * int64_t(x.element_size()));
  const mwln::ColPlan cp = fuse ? mwln::col_plan(M, N, 16 / int(x.element_size()), true, int(warps), int(kt), grid) : mwln::ColPlan{0, 0, grid, grid};
  auto part = fuse ? torch::empty({0}, f32) : wgrad ? torch::empty({2, grid, N}, f32) : torch::empty({0}, f32);
  auto dw = wgrad ? (M == 0 ? torch::zeros({N}, gopt) : torch::empty({N}, gopt)) : torch::empty({0}, f32);
  auto db = wgrad ? (M == 0 ? torch::zeros({N}, gopt) : torch::empty({N}, gopt)) : torch::empty({0}, f32);
  if (M == 0) return {dx, dw, db};
  auto s = at::cuda::getCurrentCUDAStream();
  float* pp = wgrad && !fuse ? part.data_ptr<float>() : nullptr;
  mwln::ColArgs ca{};
  if (fuse)
    ca = mwln::ColArgs{dw.data_ptr(), db.data_ptr(), int(gbf16), cp.gx, cp.K, cp.nrt, mwln::fuse_dbg()};
  const int lgrid = cp.grid;
  bool ok = false;
#define MWLN_GO(CT)                                                                                                                     \
  ok = mwln::skew::dispatch<CT>(false, false, int(tpr), int(ch), int(warps), lgrid, s, reinterpret_cast<const CT*>(x.data_ptr()),               \
                                reinterpret_cast<const CT*>(dy.data_ptr()), w.numel() ? w.data_ptr() : nullptr, nullptr, wbf, 0,        \
                                reinterpret_cast<CT*>(dx.data_ptr()), mean.data_ptr<float>(), rstd.data_ptr<float>(), pp, grid, M, N,   \
                                0.f, ca);
  if (V == 8) { MWLN_GO(__nv_bfloat16) }
  else { MWLN_GO(float) }
#undef MWLN_GO
  TORCH_CHECK(ok, "B200 LayerNorm skewed rows backward: no build for tpr=", tpr, " ch=", ch, " warps=", warps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (wgrad && !fuse) {
    mwln::launch(mwln::rows::col_reduce, dim3((N + 31) / 32, 2), dim3(1024), 0, s, (mwln::pdl_mode() & 4) != 0, pp, grid, N, dw.data_ptr(), db.data_ptr(), int(gbf16));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dx, dw, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rows_fwd", &ln_rows_fwd, "LayerNorm / RMSNorm forward, B200 sub-warp rows");
  m.def("rows_bwd", &ln_rows_bwd, "LayerNorm / RMSNorm backward, B200 sub-warp rows (dx, per-CTA partials, fixed-order column pass)");
  m.def("skew_fwd", &ln_skew_fwd, "LayerNorm forward, B200 skewed (odd-width bf16) rows");
  m.def("skew_bwd", &ln_skew_bwd, "LayerNorm backward, B200 skewed (odd-width bf16) rows");
  m.def("set_pdl", [](int64_t mode) { mwln::pdl_mode() = int(mode); }, "programmatic dependent launches: bits 1 forward, 2 backward, 4 column pass");
  m.def("set_fuse_dbg", [](int64_t d) { mwln::fuse_dbg() = int(d); }, "fused-backward experiment bits");
  m.def("set_fuse", [](int64_t max_rows, int64_t k) { mwln::fuse_max_rows() = max_rows; mwln::fuse_k() = int(k); },
        "fused backward (column CTAs): the rows up to which it is taken (0 never) and the row groups aimed at");
}
