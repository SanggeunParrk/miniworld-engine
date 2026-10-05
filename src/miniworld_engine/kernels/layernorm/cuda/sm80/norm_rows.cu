// A100 (sm_80) row kernels of LayerNorm and RMSNorm, forward and backward, bf16 or fp32 rows, fp32 statistics always.
//
//   y = (x - mean) rstd w + b   (LayerNorm)      y = x rstd w   (RMSNorm: no mean, no bias)      rstd = 1 / sqrt(mean((x - mean)^2) + eps)
//   dx = rstd (w dy - xhat mean(w dy xhat) - mean(w dy))   dw = sum_rows dy xhat   db = sum_rows dy   (RMSNorm: no mean(w dy) term, no db)
//
// An optional per-row scale `RS` (the AF pair mask folded into the epilogue: y = LN(x) rs) scales y in the forward and dy in the backward.
//
// Vector path (width a multiple of one 16-byte chunk, 16-byte aligned rows): a row is NV = N / CE chunks (CE = 8 bf16 or 4 fp32); G lanes (a power of two <= 32, by default the largest
// that divides NV) own it, lane g holding chunks g, g + G, ... (V of them: coalesced), so a warp serves 32 / G rows and the statistics are a shuffle over G lanes.
//   forward:  a one-shot grid of CTAs of 128 threads, a warp per row group or, for a mid-sized bf16 problem, per two or four of them (the loads of all issued first, the weight / bias
//             read once for all: `fwd_cfg_bf16` picks the geometry by width and size).  A persistent grid-stride loop and the `.cs` cache hints were 3-10 % slower on a streaming kernel.
//   backward: the dw / db column partials stay in registers over a persistent loop, are folded through shared memory into one fp32 partial row per CTA, and a small kernel adds
//             the partials in a fixed order.  The grid is one wave of CTAs, no more than one per 64 rows (every CTA is a partial row to write and read back: that is what a small
//             M pays for).  Large M: the persistent warps take chunks of row groups from a work counter (static assignment left 10-30 % on the table at wide rows: the CTAs
//             finish at different times), which makes the order of the rows inside the partial rows -- dw / db, not dx -- vary from run to run (fp32 rounding); small M: static
//             assignment, bit-reproducible.
// Scalar path (any other width / alignment): a warp per row, lane l on columns l, l + 32, ...; with the row cached in registers up to 32 * VS columns.
// Staged path (odd widths, many rows): tiles of rows copied to shared memory with cp.async, normalised there, copied back out.
#include "norm_common.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <map>
#include <type_traits>
#include <utility>

using namespace norms;

namespace {

constexpr int NTF = 128;      // threads per CTA, forward kernels
constexpr int NTB = 256;      // threads per CTA, vector backward
constexpr int CHUNK_GROUPS = 4;   // row groups a persistent backward warp takes from the work counter at a time

// the (lane group, chunks per lane) pairs the vector kernels are built for: G = the largest power of two dividing NV (cap 32), V = NV / G
#define NORMS_PAIRS_FWD(X) X(2, 1) X(4, 1) X(8, 1) X(16, 1) X(32, 1) X(2, 3) X(4, 3) X(8, 3) X(16, 3) X(32, 2) X(32, 3) X(32, 4) X(32, 6) X(32, 8) X(32, 10)
#define NORMS_PAIRS_BWD(X) X(2, 1) X(4, 1) X(8, 1) X(16, 1) X(32, 1) X(2, 3) X(4, 3) X(8, 3) X(16, 3) X(32, 2) X(32, 3) X(32, 4)

__device__ __forceinline__ const void* param_ptr(const void* p, bool is_bf16, long i) {
  return reinterpret_cast<const char*>(p) + (is_bf16 ? 2 : 4) * i;
}

// ------------------------------------------------------------------------------------------------------------------ forward, vector path
// UN row groups per warp (a warp's 32 / G rows each, UN groups `wstride` apart): the loads of all of them are issued before any is computed and the weight / bias are loaded once for all
// of them -- a warp moves UN V 512 bytes at a time instead of V 512, which is what a mid-sized problem (a few waves of one-shot CTAs) needs to keep its bytes in flight.
template <typename XT, int G, int V, int UN, bool RMS>
__global__ void __launch_bounds__(NTF) fwd_vec(const XT* __restrict__ X, XT* __restrict__ Y, const void* __restrict__ W, const void* __restrict__ B, bool w_bf,
                                               const XT* __restrict__ RS, float* __restrict__ MEAN, float* __restrict__ RSTD, long M, int N, float eps) {
  constexpr int CE = Chunk<XT>::CE, RPW = 32 / G;
  constexpr bool WEARLY = V <= 4;        // weight / bias loaded up front (wide rows load them per chunk to stay inside the register budget)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane & (G - 1), sub = lane / G;
  const long rg0 = (long)blockIdx.x * (NTF / 32) + warp;
  if (rg0 * RPW >= M) return;
  const long wstride = (long)gridDim.x * (NTF / 32);

  uint4 xr[UN][V];
  long row[UN];
#pragma unroll
  for (int u = 0; u < UN; ++u) {
    row[u] = (rg0 + (long)u * wstride) * RPW + sub;
#pragma unroll
    for (int v = 0; v < V; ++v) xr[u][v] = row[u] < M ? Chunk<XT>::load_raw(X + row[u] * N + (long)(g + G * v) * CE) : make_uint4(0u, 0u, 0u, 0u);
  }
  float w[WEARLY ? V : 1][WEARLY ? CE : 1], b[WEARLY ? V : 1][WEARLY ? CE : 1];
  if constexpr (WEARLY) {
#pragma unroll
    for (int v = 0; v < V; ++v) {
#pragma unroll
      for (int e = 0; e < CE; ++e) { w[v][e] = 1.f; b[v][e] = 0.f; }
      if (W) load_param<CE>(param_ptr(W, w_bf, (long)(g + G * v) * CE), w_bf, w[v]);
      if (B) load_param<CE>(param_ptr(B, w_bf, (long)(g + G * v) * CE), w_bf, b[v]);
    }
  } else {
#pragma unroll
    for (int v = 0; v < V; ++v) {                    // wide rows read their parameters per chunk after the statistics: warm L1 now, together
      if (W) prefetch_l1(param_ptr(W, w_bf, (long)(g + G * v) * CE));
      if (B) prefetch_l1(param_ptr(B, w_bf, (long)(g + G * v) * CE));
    }
  }
  const float invN = 1.f / (float)N;
#pragma unroll
  for (int u = 0; u < UN; ++u) {
    float x[V][CE];
#pragma unroll
    for (int v = 0; v < V; ++v) Chunk<XT>::unpack(xr[u][v], x[v]);
    float mean = 0.f;
    if constexpr (!RMS) {
      float s = 0.f;
#pragma unroll
      for (int v = 0; v < V; ++v)
#pragma unroll
        for (int e = 0; e < CE; ++e) s += x[v][e];
      mean = group_sum<G>(s) * invN;
#pragma unroll
      for (int v = 0; v < V; ++v)
#pragma unroll
        for (int e = 0; e < CE; ++e) x[v][e] -= mean;
    }
    float q = 0.f;
#pragma unroll
    for (int v = 0; v < V; ++v)
#pragma unroll
      for (int e = 0; e < CE; ++e) q += x[v][e] * x[v][e];
    const float rstd = rsqrtf(group_sum<G>(q) * invN + eps);
    if (row[u] < M) {
      const float rs = RS ? to_f(RS[row[u]]) : 1.f;
#pragma unroll
      for (int v = 0; v < V; ++v) {
        float wv[CE], bv[CE], y[CE];
        if constexpr (WEARLY) {
#pragma unroll
          for (int e = 0; e < CE; ++e) { wv[e] = w[v][e]; bv[e] = b[v][e]; }
        } else {
#pragma unroll
          for (int e = 0; e < CE; ++e) { wv[e] = 1.f; bv[e] = 0.f; }
          if (W) load_param<CE>(param_ptr(W, w_bf, (long)(g + G * v) * CE), w_bf, wv);
          if (B) load_param<CE>(param_ptr(B, w_bf, (long)(g + G * v) * CE), w_bf, bv);
        }
#pragma unroll
        for (int e = 0; e < CE; ++e) y[e] = fmaf(x[v][e] * rstd, wv[e], bv[e]) * rs;
        Chunk<XT>::store(Y + row[u] * N + (long)(g + G * v) * CE, y);
      }
      if (g == 0) {
        if (MEAN) MEAN[row[u]] = mean;
        if (RSTD) RSTD[row[u]] = rstd;
      }
    }
  }
}

// UN row groups (a warp's 32 / G rows each, at rg0, rg0 + step, ...; the first `cnt` of them valid): the loads of all of them are issued before any is computed (a warp keeps UN groups
// in flight: narrow rows move little data per group), then dx and the dw / db column partials added into the lane's registers.  The x / dy chunks stay packed (widened where used).
// MODE 0: LayerNorm with db, 1: LayerNorm without db, 2: RMSNorm.
template <typename XT, int G, int V, int MODE, int UN>
__device__ __forceinline__ void bwd_groups(long rg0, long step, int cnt, const XT* __restrict__ DY, const XT* __restrict__ X, const float (&w)[V][Chunk<XT>::CE],
                                           const XT* __restrict__ RS, const float* __restrict__ MEAN, const float* __restrict__ RSTD, XT* __restrict__ DX,
                                           float (&aw)[V][Chunk<XT>::CE], float (&ab)[MODE == 0 ? V : 1][MODE == 0 ? Chunk<XT>::CE : 1], long M, int N, float invN, int g, int sub) {
  constexpr int CE = Chunk<XT>::CE, RPW = 32 / G;
  constexpr bool RMS = MODE == 2, DB = MODE == 0;
  uint4 xr[UN][V], dr[UN][V];
  long row[UN];
  bool valid[UN];
  float mean[UN], rstd[UN], rs[UN];
#pragma unroll
  for (int u = 0; u < UN; ++u) {
    row[u] = (rg0 + (long)u * step) * RPW + sub;
    valid[u] = u < cnt && row[u] < M;
#pragma unroll
    for (int v = 0; v < V; ++v) {
      xr[u][v] = valid[u] ? Chunk<XT>::load_raw(X + row[u] * N + (long)(g + G * v) * CE) : make_uint4(0u, 0u, 0u, 0u);
      dr[u][v] = valid[u] ? Chunk<XT>::load_raw(DY + row[u] * N + (long)(g + G * v) * CE) : make_uint4(0u, 0u, 0u, 0u);
    }
    mean[u] = (RMS || !valid[u]) ? 0.f : MEAN[row[u]];
    rstd[u] = valid[u] ? RSTD[row[u]] : 0.f;
    rs[u] = (RS && valid[u]) ? to_f(RS[row[u]]) : 1.f;
  }
#pragma unroll
  for (int u = 0; u < UN; ++u) {
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int v = 0; v < V; ++v) {
      float xf[CE], df[CE];
      Chunk<XT>::unpack(xr[u][v], xf);
      Chunk<XT>::unpack(dr[u][v], df);
#pragma unroll
      for (int e = 0; e < CE; ++e) {
        const float wdy = w[v][e] * (df[e] * rs[u]);
        s1 += wdy * ((xf[e] - mean[u]) * rstd[u]);
        s2 += wdy;
      }
    }
    s1 = group_sum<G>(s1);
    if constexpr (!RMS) s2 = group_sum<G>(s2);
    const float c1 = s1 * invN, c2 = RMS ? 0.f : s2 * invN;
#pragma unroll
    for (int v = 0; v < V; ++v) {
      float xf[CE], df[CE], o[CE];
      Chunk<XT>::unpack(xr[u][v], xf);
      Chunk<XT>::unpack(dr[u][v], df);
#pragma unroll
      for (int e = 0; e < CE; ++e) {
        const float xh = (xf[e] - mean[u]) * rstd[u];
        const float d = df[e] * rs[u];
        o[e] = (w[v][e] * d - (xh * c1 + c2)) * rstd[u];
        aw[v][e] = fmaf(d, xh, aw[v][e]);
        if constexpr (DB) ab[v][e] += d;
      }
      if (valid[u]) Chunk<XT>::store(DX + row[u] * N + (long)(g + G * v) * CE, o);
    }
  }
}

// row groups in flight per iteration of a backward warp: more when a group is little data
template <int V> constexpr int bwd_unroll() { return V == 1 ? 4 : (V == 2 ? 2 : 1); }


// DYN: the warps take chunks of CHUNK_GROUPS row groups from the work counter CTR (zeroed by the host); otherwise a static grid-stride assignment.
// The CTA is capped at two per SM (128 registers) for V >= 2, which is what a row of 768 bf16 needs (V = 3 chunks per lane, 24 dw and 24 db partials).
template <typename XT, int G, int V, int MODE, bool DYN>
__global__ void __launch_bounds__(NTB, V >= 2 ? 2 : 1) bwd_vec(const XT* __restrict__ DY, const XT* __restrict__ X, const void* __restrict__ W, bool w_bf,
                                                               const XT* __restrict__ RS, const float* __restrict__ MEAN, const float* __restrict__ RSTD, XT* __restrict__ DX,
                                                               float* __restrict__ PDW, float* __restrict__ PDB, int* __restrict__ CTR, long M, int N) {
  constexpr int CE = Chunk<XT>::CE, RPW = 32 / G;
  constexpr bool DB = MODE == 0;
  constexpr int UN = bwd_unroll<V>();
  extern __shared__ float red[];                    // [warps][N]
  const int nthr = blockDim.x, nwarp = nthr >> 5;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane & (G - 1), sub = lane / G;

  float w[V][CE], aw[V][CE], ab[DB ? V : 1][DB ? CE : 1];
#pragma unroll
  for (int v = 0; v < V; ++v) {
#pragma unroll
    for (int e = 0; e < CE; ++e) { w[v][e] = 1.f; aw[v][e] = 0.f; }
    if (W) load_param<CE>(param_ptr(W, w_bf, (long)(g + G * v) * CE), w_bf, w[v]);
  }
  if constexpr (DB) {
#pragma unroll
    for (int v = 0; v < V; ++v)
#pragma unroll
      for (int e = 0; e < CE; ++e) ab[v][e] = 0.f;
  }
  const float invN = 1.f / (float)N;
  const long ngroups = (M + RPW - 1) / RPW;
  if constexpr (DYN) {
    for (;;) {
      int c = 0;
      if (lane == 0) c = atomicAdd(CTR, 1);
      c = __shfl_sync(0xffffffffu, c, 0);
      const long rg0 = (long)c * CHUNK_GROUPS;
      if (rg0 >= ngroups) break;
      const long rg1 = min(ngroups, rg0 + CHUNK_GROUPS);
      for (long rg = rg0; rg < rg1; rg += UN)
        bwd_groups<XT, G, V, MODE, UN>(rg, 1, (int)min((long)UN, rg1 - rg), DY, X, w, RS, MEAN, RSTD, DX, aw, ab, M, N, invN, g, sub);
    }
  } else {
    const long stride = (long)gridDim.x * nwarp;
    for (long rg = (long)blockIdx.x * nwarp + warp; rg < ngroups; rg += stride * UN)
      bwd_groups<XT, G, V, MODE, UN>(rg, stride, (int)min((long)UN, (ngroups - rg + stride - 1) / stride), DY, X, w, RS, MEAN, RSTD, DX, aw, ab, M, N, invN, g, sub);
  }
  // the rows of a warp's sub-groups -> one row of partial sums per warp (every sub-group gets the sum), then one per CTA through shared memory
  if constexpr (RPW > 1) {
#pragma unroll
    for (int o = G; o < 32; o <<= 1)
#pragma unroll
      for (int v = 0; v < V; ++v)
#pragma unroll
        for (int e = 0; e < CE; ++e) {
          aw[v][e] += __shfl_xor_sync(0xffffffffu, aw[v][e], o);
          if constexpr (DB) ab[v][e] += __shfl_xor_sync(0xffffffffu, ab[v][e], o);
        }
  }
  if (PDW) {
    if (sub == 0) {
#pragma unroll
      for (int v = 0; v < V; ++v)
#pragma unroll
        for (int e = 0; e < CE; ++e) red[warp * N + (g + G * v) * CE + e] = aw[v][e];
    }
    __syncthreads();
    for (int t = threadIdx.x; t < N; t += nthr) {
      float s = 0.f;
      for (int k = 0; k < nwarp; ++k) s += red[k * N + t];
      PDW[(long)blockIdx.x * N + t] = s;
    }
    __syncthreads();
  }
  if constexpr (DB) {
    if (PDB) {
      if (sub == 0) {
#pragma unroll
        for (int v = 0; v < V; ++v)
#pragma unroll
          for (int e = 0; e < CE; ++e) red[warp * N + (g + G * v) * CE + e] = ab[v][e];
      }
      __syncthreads();
      for (int t = threadIdx.x; t < N; t += nthr) {
        float s = 0.f;
        for (int k = 0; k < nwarp; ++k) s += red[k * N + t];
        PDB[(long)blockIdx.x * N + t] = s;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------ partial sums -> dw, db
// P partial rows [P, N] added in a fixed order.  Block (8, 32): thread (x, y) owns the column quad 8 blockIdx.x + x (float4 loads: N a multiple of 4) and the rows y, y + 32, ...;
// eight rows are loaded before any is added (the partials sit in L2: it is the load latency, not the bandwidth, that a small reduction pays for), so up to 256 partial rows take one
// round of loads.  The 32 row slices fold through the shuffles of a warp (lane = 8 y' + x: the four slices of a warp) and one shared-memory round across the eight warps.
__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 shfl_xor4(float4 v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o), __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}
__global__ void __launch_bounds__(256) reduce_partials4(const float* __restrict__ PDW, const float* __restrict__ PDB, void* __restrict__ DW, void* __restrict__ DB, bool o_bf, int P, int N) {
  __shared__ float4 sw[8][8], sb[8][8];                    // [warp][quad]
  const int q = blockIdx.x * 8 + threadIdx.x;
  const int tid = threadIdx.y * 8 + threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
  if (q * 4 < N) {
    for (int p0 = 0; p0 < P; p0 += 256) {
      float4 ua[8], ub[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int p = p0 + threadIdx.y + 32 * i;
        ua[i] = (PDW && p < P) ? *(reinterpret_cast<const float4*>(PDW + (long)p * N) + q) : make_float4(0.f, 0.f, 0.f, 0.f);
        ub[i] = (PDB && p < P) ? *(reinterpret_cast<const float4*>(PDB + (long)p * N) + q) : make_float4(0.f, 0.f, 0.f, 0.f);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) { a = add4(a, ua[i]); b = add4(b, ub[i]); }
    }
  }
  a = add4(a, shfl_xor4(a, 8));
  a = add4(a, shfl_xor4(a, 16));
  b = add4(b, shfl_xor4(b, 8));
  b = add4(b, shfl_xor4(b, 16));
  if (lane < 8) { sw[warp][lane] = a; sb[warp][lane] = b; }
  __syncthreads();
  if (tid < 8 && q * 4 < N) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f), t = s;
#pragma unroll
    for (int k = 0; k < 8; ++k) { s = add4(s, sw[k][tid]); t = add4(t, sb[k][tid]); }
    if (DW) {
      if (o_bf) { uint2 u = make_uint2(pack_bf2(s.x, s.y), pack_bf2(s.z, s.w)); *(reinterpret_cast<uint2*>(DW) + q) = u; }
      else *(reinterpret_cast<float4*>(DW) + q) = s;
    }
    if (DB) {
      if (o_bf) { uint2 u = make_uint2(pack_bf2(t.x, t.y), pack_bf2(t.z, t.w)); *(reinterpret_cast<uint2*>(DB) + q) = u; }
      else *(reinterpret_cast<float4*>(DB) + q) = t;
    }
  }
}

// the same for any width: block (32 columns, 16 row slices), 128 partial rows a round of loads
__global__ void __launch_bounds__(512) reduce_partials(const float* __restrict__ PDW, const float* __restrict__ PDB, void* __restrict__ DW, void* __restrict__ DB, bool o_bf, int P, int N) {
  __shared__ float sw[16][33], sb[16][33];
  const int col = blockIdx.x * 32 + threadIdx.x;
  float a = 0.f, b = 0.f;
  if (col < N) {
    for (int p0 = 0; p0 < P; p0 += 128) {
      float ua[8], ub[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int p = p0 + threadIdx.y + 16 * i;
        ua[i] = (PDW && p < P) ? PDW[(long)p * N + col] : 0.f;
        ub[i] = (PDB && p < P) ? PDB[(long)p * N + col] : 0.f;
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) { a += ua[i]; b += ub[i]; }
    }
  }
  sw[threadIdx.y][threadIdx.x] = a;
  sb[threadIdx.y][threadIdx.x] = b;
  __syncthreads();
  if (threadIdx.y == 0 && col < N) {
    float s = 0.f, t = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) { s += sw[k][threadIdx.x]; t += sb[k][threadIdx.x]; }
    if (DW) { if (o_bf) reinterpret_cast<bf*>(DW)[col] = __float2bfloat16_rn(s); else reinterpret_cast<float*>(DW)[col] = s; }
    if (DB) { if (o_bf) reinterpret_cast<bf*>(DB)[col] = __float2bfloat16_rn(t); else reinterpret_cast<float*>(DB)[col] = t; }
  }
}

// ------------------------------------------------------------------------------------------------------------------ scalar path (any width, any alignment)
// VS > 0: the row is cached in registers, lane l holding columns l + 32 k (k < VS, N <= 32 VS); VS = 0: the row is read again from L1 for each pass.
// one row held at `x` (global or shared memory) normalised into `y` (may be the same place) by the 32 lanes of a warp; mean / rstd come back in every lane
template <typename XT, int VS, bool RMS>
__device__ __forceinline__ void norm_row_scalar(const XT* x, XT* y, int N, int lane, const void* W, const void* B, bool w_bf, float rs, float eps, float& mean_out, float& rstd_out) {
  const float invN = 1.f / (float)N;
  constexpr int VR = VS > 0 ? VS : 1;
  float xr[VR], wr[VR], br[VR];
  float s = 0.f;
  if constexpr (VS > 0) {
    // every load of the row and of the parameters is issued before any is used (the stores come last: a load placed after a store is serialised behind it, which cost a 27-column
    // row 5 of its 9 microseconds)
#pragma unroll
    for (int k = 0; k < VS; ++k) {
      const int c = lane + 32 * k;
      const bool in = c < N;
      xr[k] = in ? to_f(x[c]) : 0.f;
      wr[k] = (W && in) ? param_at(W, w_bf, c) : 1.f;
      br[k] = (B && in) ? param_at(B, w_bf, c) : 0.f;
    }
#pragma unroll
    for (int k = 0; k < VS; ++k) s += xr[k];
  } else {
    for (int c = lane; c < N; c += 32) s += to_f(x[c]);
  }
  float mean = 0.f;
  if constexpr (!RMS) mean = warp_sum(s) * invN;
  float q = 0.f;
  if constexpr (VS > 0) {
#pragma unroll
    for (int k = 0; k < VS; ++k) {
      const int c = lane + 32 * k;
      xr[k] = c < N ? xr[k] - mean : 0.f;
      q += xr[k] * xr[k];
    }
  } else {
    for (int c = lane; c < N; c += 32) { const float d = to_f(x[c]) - mean; q += d * d; }
  }
  const float rstd = rsqrtf(warp_sum(q) * invN + eps);
  if constexpr (VS > 0) {
#pragma unroll
    for (int k = 0; k < VS; ++k) {
      const int c = lane + 32 * k;
      if (c < N) y[c] = from_f<XT>(fmaf(xr[k] * rstd, wr[k], br[k]) * rs);
    }
  } else {
    for (int c = lane; c < N; c += 32) {
      const float wv = W ? param_at(W, w_bf, c) : 1.f, bv = B ? param_at(B, w_bf, c) : 0.f;
      y[c] = from_f<XT>(fmaf((to_f(x[c]) - mean) * rstd, wv, bv) * rs);
    }
  }
  mean_out = mean;
  rstd_out = rstd;
}

template <typename XT, int VS, bool RMS>
__global__ void __launch_bounds__(NTF) fwd_scalar(const XT* __restrict__ X, XT* __restrict__ Y, const void* __restrict__ W, const void* __restrict__ B, bool w_bf,
                                                  const XT* __restrict__ RS, float* __restrict__ MEAN, float* __restrict__ RSTD, long M, int N, float eps) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const long row = (long)blockIdx.x * (NTF / 32) + warp;
  if (row >= M) return;
  float mean, rstd;
  norm_row_scalar<XT, VS, RMS>(X + row * N, Y + row * N, N, lane, W, B, w_bf, RS ? to_f(RS[row]) : 1.f, eps, mean, rstd);
  if (lane == 0) {
    if (MEAN) MEAN[row] = mean;
    if (RSTD) RSTD[row] = rstd;
  }
}

// ------------------------------------------------------------------------------------------------------------------ staged path (odd widths, M large)
// A row of N elements is not a whole number of 16-byte chunks, but U rows are (U = 16 / gcd(16, row bytes), 8 bf16 rows of 267): a unit of U rows is contiguous, 16-byte
// aligned and a whole number of chunks.  The CTA copies a tile of R rows (a multiple of U) into shared memory with cp.async (coalesced, 16 bytes a lane), the warps normalise
// their rows there (lane l on columns l + 32 k: scalar shared-memory accesses, which never miss) and the tile is copied back out the same way.  The tail rows (M mod U) are the
// scalar kernel's.  NTG threads per CTA; VS = ceil(N / 32) rounded up to a built value.
constexpr int NTG = 256, NWG = NTG / 32;

template <typename XT, int VS, bool RMS>
__global__ void __launch_bounds__(NTG) fwd_stage(const XT* __restrict__ X, XT* __restrict__ Y, const void* __restrict__ W, const void* __restrict__ B, bool w_bf,
                                                 const XT* __restrict__ RS, float* __restrict__ MEAN, float* __restrict__ RSTD, long Mfull, int N, int R, float eps) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long rowb = (long)N * sizeof(XT);
  const long row0 = (long)blockIdx.x * R;
  const int nr = (int)min((long)R, Mfull - row0);
  const int nchunks = (int)((long)nr * rowb / 16);
  const unsigned char* gx = reinterpret_cast<const unsigned char*>(X) + row0 * rowb;
  for (int c = tid; c < nchunks; c += NTG) cp_async16(smem + 16 * c, gx + 16 * c);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  for (int r = warp; r < nr; r += NWG) {
    XT* xs = reinterpret_cast<XT*>(smem + (long)r * rowb);
    float mean, rstd;
    norm_row_scalar<XT, VS, RMS>(xs, xs, N, lane, W, B, w_bf, RS ? to_f(RS[row0 + r]) : 1.f, eps, mean, rstd);
    if (lane == 0) {
      if (MEAN) MEAN[row0 + r] = mean;
      if (RSTD) RSTD[row0 + r] = rstd;
    }
  }
  __syncthreads();
  unsigned char* gy = reinterpret_cast<unsigned char*>(Y) + row0 * rowb;
  for (int c = tid; c < nchunks; c += NTG) *reinterpret_cast<uint4*>(gy + 16 * c) = *reinterpret_cast<const uint4*>(smem + 16 * c);
}

// one row of the backward: xs = x, ds = dy (scaled by the row scale) in, dxs = dx out (ds and dxs may be the same place); the lane's columns l + 32 k keep their dw / db partials in aw / ab
template <typename XT, int VS, bool RMS>
__device__ __forceinline__ void bwd_row_scalar(const XT* xs, const XT* ds, XT* dxs, int N, int lane, const float (&wreg)[VS], float mean, float rstd, float rs, float (&aw)[VS], float (&ab)[VS]) {
  const float invN = 1.f / (float)N;
  float xh[VS], dv[VS];
  float s1 = 0.f, s2 = 0.f;
#pragma unroll
  for (int k = 0; k < VS; ++k) {
    const int c = lane + 32 * k;
    const bool in = c < N;
    xh[k] = in ? (to_f(xs[c]) - mean) * rstd : 0.f;
    dv[k] = in ? to_f(ds[c]) * rs : 0.f;
    const float wdy = wreg[k] * dv[k];
    s1 += wdy * xh[k];
    s2 += wdy;
  }
  s1 = warp_sum(s1);
  if constexpr (!RMS) s2 = warp_sum(s2);
  const float c1 = s1 * invN, c2 = RMS ? 0.f : s2 * invN;
#pragma unroll
  for (int k = 0; k < VS; ++k) {
    const int c = lane + 32 * k;
    if (c < N) dxs[c] = from_f<XT>((wreg[k] * dv[k] - (xh[k] * c1 + c2)) * rstd);
    aw[k] = fmaf(dv[k], xh[k], aw[k]);
    ab[k] += dv[k];
  }
}

// Persistent CTAs (static tile assignment, two stages of cp.async prefetch); the dw / db partials of the lanes' columns are added in registers and folded into one partial row per CTA.
template <typename XT, int VS, bool RMS, int RPT>
__global__ void __launch_bounds__(NTG) bwd_stage(const XT* __restrict__ DY, const XT* __restrict__ X, const void* __restrict__ W, bool w_bf, const XT* __restrict__ RS,
                                                 const float* __restrict__ MEAN, const float* __restrict__ RSTD, XT* __restrict__ DX, float* __restrict__ PDW,
                                                 float* __restrict__ PDB, long Mfull, int N, int R) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long rowb = (long)N * sizeof(XT);
  const long TB = (long)R * rowb;                     // bytes of one tile of one tensor
  float wreg[VS], aw[VS], ab[VS];
#pragma unroll
  for (int k = 0; k < VS; ++k) {
    const int c = lane + 32 * k;
    wreg[k] = c < N && W ? param_at(W, w_bf, c) : 1.f;
    aw[k] = 0.f;
    ab[k] = 0.f;
  }
  const long ntiles = (Mfull + R - 1) / R;
  auto issue = [&](long t, int stage) {
    const long row0 = t * R;
    const int nr = (int)min((long)R, Mfull - row0);
    const int nchunks = (int)((long)nr * rowb / 16);
    const unsigned char* gx = reinterpret_cast<const unsigned char*>(X) + row0 * rowb;
    const unsigned char* gd = reinterpret_cast<const unsigned char*>(DY) + row0 * rowb;
    unsigned char* sx = smem + (long)stage * 2 * TB;
    for (int c = tid; c < nchunks; c += NTG) {
      cp_async16(sx + 16 * c, gx + 16 * c);
      cp_async16(sx + TB + 16 * c, gd + 16 * c);
    }
    cp_async_commit();
  };
  long tile = blockIdx.x;
  int s = 0;
  if (tile < ntiles) issue(tile, 0);
  for (; tile < ntiles; tile += gridDim.x, s ^= 1) {
    const long nxt = tile + gridDim.x;
    const long row0 = tile * R;
    const int nr = (int)min((long)R, Mfull - row0);
    // this warp's rows' statistics, loaded while the tile is in flight
    float mr[RPT], sr[RPT];                              // this warp's rows of the tile (RPT >= R / NWG)
#pragma unroll
    for (int j = 0; j < RPT; ++j) {
      const int r = warp + NWG * j;
      mr[j] = 0.f;
      sr[j] = 0.f;
      if (r < nr) {
        mr[j] = RMS ? 0.f : MEAN[row0 + r];
        sr[j] = RSTD[row0 + r];
      }
    }
    if (nxt < ntiles) {
      issue(nxt, s ^ 1);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    unsigned char* sx = smem + (long)s * 2 * TB;
    unsigned char* sd = sx + TB;
#pragma unroll
    for (int j = 0; j < RPT; ++j) {
      const int r = warp + NWG * j;
      if (r < nr) {
        const float rs = RS ? to_f(RS[row0 + r]) : 1.f;
        bwd_row_scalar<XT, VS, RMS>(reinterpret_cast<const XT*>(sx + (long)r * rowb), reinterpret_cast<const XT*>(sd + (long)r * rowb), reinterpret_cast<XT*>(sd + (long)r * rowb), N, lane, wreg, mr[j], sr[j], rs, aw, ab);
      }
    }
    __syncthreads();
    unsigned char* gdx = reinterpret_cast<unsigned char*>(DX) + row0 * rowb;
    const int nchunks = (int)((long)nr * rowb / 16);
    for (int c = tid; c < nchunks; c += NTG) *reinterpret_cast<uint4*>(gdx + 16 * c) = *reinterpret_cast<const uint4*>(sd + 16 * c);
  }
  // the lanes' partials -> one partial row per CTA through shared memory
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);          // [NWG][N]
  if (PDW) {
#pragma unroll
    for (int k = 0; k < VS; ++k) {
      const int c = lane + 32 * k;
      if (c < N) red[warp * N + c] = aw[k];
    }
    __syncthreads();
    for (int t = tid; t < N; t += NTG) {
      float a = 0.f;
      for (int q = 0; q < NWG; ++q) a += red[q * N + t];
      PDW[(long)blockIdx.x * N + t] = a;
    }
    __syncthreads();
  }
  if (PDB && !RMS) {
#pragma unroll
    for (int k = 0; k < VS; ++k) {
      const int c = lane + 32 * k;
      if (c < N) red[warp * N + c] = ab[k];
    }
    __syncthreads();
    for (int t = tid; t < N; t += NTG) {
      float a = 0.f;
      for (int q = 0; q < NWG; ++q) a += red[q * N + t];
      PDB[(long)blockIdx.x * N + t] = a;
    }
  }
}

// four warps per CTA; each warp keeps its own dw / db partial row in shared memory (a lane touches only its own columns), one partial row per CTA at the end
constexpr int NTS = 128, NWS = NTS / 32;

template <typename XT, int VS, bool RMS>
__global__ void __launch_bounds__(NTS) bwd_scalar(const XT* __restrict__ DY, const XT* __restrict__ X, const void* __restrict__ W, bool w_bf, const XT* __restrict__ RS,
                                                  const float* __restrict__ MEAN, const float* __restrict__ RSTD, XT* __restrict__ DX, float* __restrict__ PDW,
                                                  float* __restrict__ PDB, long M, int N) {
  extern __shared__ float sm[];                     // [NWS][N] dw | [NWS][N] db
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float* sdw = sm + (long)warp * N;
  float* sdb = sm + (long)(NWS + warp) * N;
  for (int c = lane; c < N; c += 32) { sdw[c] = 0.f; sdb[c] = 0.f; }
  __syncwarp();
  const float invN = 1.f / (float)N;
  for (long row = (long)blockIdx.x * NWS + warp; row < M; row += (long)gridDim.x * NWS) {
    const XT* x = X + row * N;
    const XT* dy = DY + row * N;
    const float mean = RMS ? 0.f : MEAN[row];
    const float rstd = RSTD[row];
    const float rs = RS ? to_f(RS[row]) : 1.f;
    float s1 = 0.f, s2 = 0.f;
    for (int c = lane; c < N; c += 32) {
      const float xh = (to_f(x[c]) - mean) * rstd;
      const float wdy = (W ? param_at(W, w_bf, c) : 1.f) * (to_f(dy[c]) * rs);
      s1 += wdy * xh;
      s2 += wdy;
    }
    s1 = warp_sum(s1);
    if constexpr (!RMS) s2 = warp_sum(s2);
    const float c1 = s1 * invN, c2 = RMS ? 0.f : s2 * invN;
    XT* dx = DX + row * N;
    for (int c = lane; c < N; c += 32) {
      const float xh = (to_f(x[c]) - mean) * rstd;
      const float d = to_f(dy[c]) * rs;
      const float wdy = (W ? param_at(W, w_bf, c) : 1.f) * d;
      dx[c] = from_f<XT>((wdy - (xh * c1 + c2)) * rstd);
      sdw[c] += d * xh;
      sdb[c] += d;
    }
    __syncwarp();
  }
  __syncthreads();
  for (int t = threadIdx.x; t < N; t += NTS) {
    float a = 0.f, b = 0.f;
    for (int k = 0; k < NWS; ++k) { a += sm[(long)k * N + t]; b += sm[(long)(NWS + k) * N + t]; }
    if (PDW) PDW[(long)blockIdx.x * N + t] = a;
    if (PDB) PDB[(long)blockIdx.x * N + t] = b;
  }
}

// Scalar backward with the row cached in registers (N <= 32 VS): per row every load (x, dy) is issued before any is used and dx is written last (`bwd_row_scalar`); the dw / db partials stay
// in the lane's registers over the warp's rows (no shared-memory read-modify-write per element) and are folded through shared memory into one partial row per CTA.
template <typename XT, int VS, bool RMS>
__global__ void __launch_bounds__(NTS) bwd_scalar_reg(const XT* __restrict__ DY, const XT* __restrict__ X, const void* __restrict__ W, bool w_bf, const XT* __restrict__ RS,
                                                      const float* __restrict__ MEAN, const float* __restrict__ RSTD, XT* __restrict__ DX, float* __restrict__ PDW,
                                                      float* __restrict__ PDB, long M, int N) {
  extern __shared__ float sm[];                     // [NWS][N]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float wreg[VS], aw[VS], ab[VS];
#pragma unroll
  for (int k = 0; k < VS; ++k) {
    const int c = lane + 32 * k;
    wreg[k] = (c < N && W) ? param_at(W, w_bf, c) : 1.f;
    aw[k] = 0.f;
    ab[k] = 0.f;
  }
  for (long row = (long)blockIdx.x * NWS + warp; row < M; row += (long)gridDim.x * NWS)
    bwd_row_scalar<XT, VS, RMS>(X + row * N, DY + row * N, DX + row * N, N, lane, wreg, RMS ? 0.f : MEAN[row], RSTD[row], RS ? to_f(RS[row]) : 1.f, aw, ab);
  if (PDW) {
#pragma unroll
    for (int k = 0; k < VS; ++k) {
      const int c = lane + 32 * k;
      if (c < N) sm[warp * N + c] = aw[k];
    }
    __syncthreads();
    for (int t = threadIdx.x; t < N; t += NTS) {
      float a = 0.f;
      for (int q = 0; q < NWS; ++q) a += sm[q * N + t];
      PDW[(long)blockIdx.x * N + t] = a;
    }
    __syncthreads();
  }
  if (PDB && !RMS) {
#pragma unroll
    for (int k = 0; k < VS; ++k) {
      const int c = lane + 32 * k;
      if (c < N) sm[warp * N + c] = ab[k];
    }
    __syncthreads();
    for (int t = threadIdx.x; t < N; t += NTS) {
      float a = 0.f;
      for (int q = 0; q < NWS; ++q) a += sm[q * N + t];
      PDB[(long)blockIdx.x * N + t] = a;
    }
  }
}

// Wide rows (bf16, N = 256 V, V = 6, 8, 10: 1536 / 2048 / 2560 columns): a warp per row, the 32 lanes owning chunks lane + 32 v, the row's x and dy chunks packed in registers (16 V registers).  The
// dw / db partials of a lane (8 V floats each) do not fit the registers next to them: they are accumulated into the warp's own rows of shared memory (a lane touches only its own columns, so no
// atomics and no barrier), then folded across the warps into one partial row per CTA like the other backward kernels.
template <typename XT, int V, int MODE>
__global__ void __launch_bounds__(NTS) bwd_wide(const XT* __restrict__ DY, const XT* __restrict__ X, const void* __restrict__ W, bool w_bf, const XT* __restrict__ RS,
                                                const float* __restrict__ MEAN, const float* __restrict__ RSTD, XT* __restrict__ DX, float* __restrict__ PDW,
                                                float* __restrict__ PDB, long M, int N) {
  constexpr int CE = Chunk<XT>::CE;
  constexpr bool RMS = MODE == 2, DB = MODE == 0;
  extern __shared__ __align__(16) float sm[];                    // [NWS][N] dw | [NWS][N] db (LayerNorm with db)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float* sdw = sm + (long)warp * N;
  float* sdb = sm + (long)(NWS + warp) * N;
#pragma unroll
  for (int v = 0; v < V; ++v) {
    float4* p = reinterpret_cast<float4*>(sdw + (long)(lane + 32 * v) * CE);
    float4* q = reinterpret_cast<float4*>(sdb + (long)(lane + 32 * v) * CE);
#pragma unroll
    for (int k = 0; k < CE / 4; ++k) {
      p[k] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (DB) q[k] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    if (W) prefetch_l1(param_ptr(W, w_bf, (long)(lane + 32 * v) * CE));      // the weight chunks are read twice a row: warm L1 together
  }
  const float invN = 1.f / (float)N;
  for (long row = (long)blockIdx.x * NWS + warp; row < M; row += (long)gridDim.x * NWS) {
    uint4 xr[V], dr[V];
#pragma unroll
    for (int v = 0; v < V; ++v) {
      xr[v] = Chunk<XT>::load_raw(X + row * N + (long)(lane + 32 * v) * CE);
      dr[v] = Chunk<XT>::load_raw(DY + row * N + (long)(lane + 32 * v) * CE);
    }
    const float mean = RMS ? 0.f : MEAN[row], rstd = RSTD[row], rs = RS ? to_f(RS[row]) : 1.f;
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int v = 0; v < V; ++v) {
      float xf[CE], df[CE], wv[CE];
      Chunk<XT>::unpack(xr[v], xf);
      Chunk<XT>::unpack(dr[v], df);
#pragma unroll
      for (int e = 0; e < CE; ++e) wv[e] = 1.f;
      if (W) load_param<CE>(param_ptr(W, w_bf, (long)(lane + 32 * v) * CE), w_bf, wv);
#pragma unroll
      for (int e = 0; e < CE; ++e) {
        const float wdy = wv[e] * (df[e] * rs);
        s1 += wdy * ((xf[e] - mean) * rstd);
        s2 += wdy;
      }
    }
    s1 = warp_sum(s1);
    if constexpr (!RMS) s2 = warp_sum(s2);
    const float c1 = s1 * invN, c2 = RMS ? 0.f : s2 * invN;
#pragma unroll
    for (int v = 0; v < V; ++v) {
      float xf[CE], df[CE], wv[CE], o[CE], gw[CE], gb[CE];
      Chunk<XT>::unpack(xr[v], xf);
      Chunk<XT>::unpack(dr[v], df);
#pragma unroll
      for (int e = 0; e < CE; ++e) wv[e] = 1.f;
      if (W) load_param<CE>(param_ptr(W, w_bf, (long)(lane + 32 * v) * CE), w_bf, wv);
#pragma unroll
      for (int e = 0; e < CE; ++e) {
        const float xh = (xf[e] - mean) * rstd, d = df[e] * rs;
        o[e] = (wv[e] * d - (xh * c1 + c2)) * rstd;
        gw[e] = d * xh;
        gb[e] = d;
      }
      Chunk<XT>::store(DX + row * N + (long)(lane + 32 * v) * CE, o);
      float4* p = reinterpret_cast<float4*>(sdw + (long)(lane + 32 * v) * CE);
      float4* q = reinterpret_cast<float4*>(sdb + (long)(lane + 32 * v) * CE);
#pragma unroll
      for (int k = 0; k < CE / 4; ++k) {
        float4 a = p[k];
        a.x += gw[4 * k]; a.y += gw[4 * k + 1]; a.z += gw[4 * k + 2]; a.w += gw[4 * k + 3];
        p[k] = a;
        if (DB) {
          float4 b = q[k];
          b.x += gb[4 * k]; b.y += gb[4 * k + 1]; b.z += gb[4 * k + 2]; b.w += gb[4 * k + 3];
          q[k] = b;
        }
      }
    }
  }
  __syncthreads();
  if (PDW) {
    for (int t = threadIdx.x; t < N; t += NTS) {
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < NWS; ++k) a += sm[(long)k * N + t];
      PDW[(long)blockIdx.x * N + t] = a;
    }
  }
  if (DB && PDB) {
    for (int t = threadIdx.x; t < N; t += NTS) {
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < NWS; ++k) a += sm[(long)(NWS + k) * N + t];
      PDB[(long)blockIdx.x * N + t] = a;
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------ host side
int sm_count() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

template <typename K> int blocks_per_sm(K kernel, int threads, size_t dyn) {
  static std::map<std::pair<const void*, size_t>, int> cache;
  const auto key = std::make_pair(reinterpret_cast<const void*>(kernel), dyn);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  if (dyn > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  int n = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, threads, dyn);
  n = std::max(n, 1);
  cache[key] = n;
  return n;
}

bool is_bf16(const at::Tensor& t) { return t.scalar_type() == at::kBFloat16; }
const void* optp(const c10::optional<at::Tensor>& t) { return t.has_value() && t->defined() ? t->data_ptr() : nullptr; }
template <typename T> const T* optp_t(const c10::optional<at::Tensor>& t) { return reinterpret_cast<const T*>(optp(t)); }
template <typename T> T* optp_w(const c10::optional<at::Tensor>& t) { return t.has_value() && t->defined() ? reinterpret_cast<T*>(t->data_ptr()) : nullptr; }
bool aligned16(const void* p) { return p == nullptr || (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

// the lane group of the vector path for a row of NV chunks: (G, V), or (0, 0) when the row is not served (odd NV: a lane per row is not worth a path)
std::pair<int, int> lane_group(int NV) {
  int g = 1;
  while (g < 32 && NV % (2 * g) == 0) g *= 2;
  if (g < 2) return {0, 0};
  return {g, NV / g};
}

// What a bf16 forward is run with, by width and size (E = M x N elements; measured in the production op, a fresh output each call, CUDA graph: probes/ln_fwd_knobs.py).  A problem of a
// few waves of one-shot CTAs with one narrow row group a warp keeps too few bytes in flight per warp: two row groups a warp (the loads of both issued first, the weight / bias loaded once for
// both) took 5-20 % off the mid-sized forwards; at the sizes where HBM is the limit every configuration is the same speed (and below ~1M elements the second group only adds latency).
// 64 wide rows: the lane group follows the size (8 lanes a row at the ends, 4 in between: 8 rows a warp, a shorter shuffle chain).
struct FwdCfg { int G, UN; };
FwdCfg fwd_cfg_bf16(int N, long M, int G0) {
  const double E = (double)M * (double)N;
  switch (N) {
    case 16: return {2, E < 1e6 ? 1 : (E < 6e6 ? 2 : (E < 1.4e7 ? 4 : 1))};
    case 64:
      if (E < 1e6) return {8, 1};
      if (E < 2.5e6) return {4, 1};
      if (E < 8e6) return {4, 2};
      if (E < 1.5e7) return {8, 2};
      return {8, 1};
    case 128: return {16, E < 1.2e6 ? 1 : 2};
    case 256: return {32, (E >= 1.2e6 && E < 1.2e7) ? 2 : 1};
    case 384: return {16, (E >= 4e6 && E < 1.2e7) ? 2 : 1};
    case 512: return {32, (E >= 2e6 && E < 2e7) ? 2 : 1};
    case 768: return {32, (E >= 2e6 && E < 8e6) ? 2 : 1};
    default: return {G0, 1};
  }
}
// (lane group, chunks per lane, row groups per warp) the bf16 configurations above add to the default pairs
#define NORMS_TRIPLES_FWD_BF(X) X(2, 1, 2) X(2, 1, 4) X(8, 1, 2) X(4, 2, 1) X(4, 2, 2) X(16, 1, 2) X(32, 1, 2) X(16, 3, 2) X(32, 2, 2) X(32, 3, 2)

template <typename XT, bool RMS, int G, int V, int UN>
void run_fwd_vec(const XT* X, XT* Y, const void* W, const void* B, bool w_bf, const XT* RS, float* MEAN, float* RSTD, long M, int N, float eps, cudaStream_t st) {
  constexpr int RPW = 32 / G;
  const long groups = (M + RPW - 1) / RPW;
  const long warps = (groups + UN - 1) / UN;
  const long grid = (warps + NTF / 32 - 1) / (NTF / 32);
  fwd_vec<XT, G, V, UN, RMS><<<(unsigned)grid, NTF, 0, st>>>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps);
}

template <typename XT, bool RMS>
bool launch_fwd_vec(int NV, const XT* X, XT* Y, const void* W, const void* B, bool w_bf, const XT* RS, float* MEAN, float* RSTD, long M, int N, float eps,
                    cudaStream_t st) {
  auto [G, V] = lane_group(NV);
  int un = 1;
  if constexpr (std::is_same_v<XT, bf>) {
    if (G > 0) {
      const FwdCfg c = fwd_cfg_bf16(N, M, G);
      if (c.G != G && NV % c.G == 0) { G = c.G; V = NV / c.G; }
      un = c.UN;
    }
  }
#define TRY(g, v) if (G == g && V == v && un == 1) { run_fwd_vec<XT, RMS, g, v, 1>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st); return true; }
  NORMS_PAIRS_FWD(TRY)
#undef TRY
  if constexpr (std::is_same_v<XT, bf>) {
#define TRY3(g, v, u) if (G == g && V == v && un == u) { run_fwd_vec<XT, RMS, g, v, u>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st); return true; }
    NORMS_TRIPLES_FWD_BF(TRY3)
#undef TRY3
  }
  return false;
}

template <typename XT, bool RMS, int VS>
void run_fwd_scalar(const XT* X, XT* Y, const void* W, const void* B, bool w_bf, const XT* RS, float* MEAN, float* RSTD, long M, int N, float eps, cudaStream_t st) {
  const long grid = (M + NTF / 32 - 1) / (NTF / 32);
  fwd_scalar<XT, VS, RMS><<<(unsigned)grid, NTF, 0, st>>>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps);
}

template <typename XT, bool RMS>
void launch_fwd_scalar(const XT* X, XT* Y, const void* W, const void* B, bool w_bf, const XT* RS, float* MEAN, float* RSTD, long M, int N, float eps, cudaStream_t st) {
  if (N <= 64) run_fwd_scalar<XT, RMS, 2>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st);
  else if (N <= 128) run_fwd_scalar<XT, RMS, 4>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st);
  else if (N <= 256) run_fwd_scalar<XT, RMS, 8>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st);
  else if (N <= 384) run_fwd_scalar<XT, RMS, 12>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st);
  else if (N <= 512) run_fwd_scalar<XT, RMS, 16>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st);
  else if (N <= 864) run_fwd_scalar<XT, RMS, 27>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st);
  else if (N <= 1024) run_fwd_scalar<XT, RMS, 32>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st);
  else run_fwd_scalar<XT, RMS, 0>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st);
}

// ---- staged path geometry: a unit is U rows = U N sizeof(XT) bytes, a whole number of 16-byte chunks
int gcd_i(int a, int b) { while (b) { const int t = a % b; a = b; b = t; } return a; }
constexpr int STAGE_MAX_N = 1024;                     // widest odd row of the staged path (the scalar path beyond)
constexpr long STAGE_MIN_ROWS = 4096;                 // fewer rows: the scalar kernels (a CTA per tile is not worth it)
constexpr size_t STAGE_TILE_BYTES = 10 * 1024;        // target bytes of one tensor of a tile

struct StageGeom { int U, R; long rowb; };
template <typename XT> StageGeom stage_geom(int N, bool bwd) {
  const long rowb = (long)N * (long)sizeof(XT);
  const int U = 16 / gcd_i(16, (int)(rowb % 16 == 0 ? 16 : rowb % 16));
  const long UB = (long)U * rowb;
  long TU = std::max<long>(1, (long)STAGE_TILE_BYTES / UB);
  if (bwd) TU = std::min<long>(TU, std::max(1, 16 / U));          // at most 16 rows a tile: two rows a warp
  return {U, (int)(TU * U), rowb};
}
int vs_bucket(int N) { return N <= 96 ? 3 : N <= 160 ? 5 : N <= 288 ? 9 : N <= 384 ? 12 : N <= 480 ? 15 : N <= 576 ? 18 : N <= 864 ? 27 : 32; }

template <typename XT, bool RMS, int VS>
void run_fwd_stage(const XT* X, XT* Y, const void* W, const void* B, bool w_bf, const XT* RS, float* MEAN, float* RSTD, long Mfull, int N, const StageGeom& g, float eps, cudaStream_t st) {
  const size_t smem = (size_t)g.R * g.rowb;
  auto k = fwd_stage<XT, VS, RMS>;
  if (smem > 48 * 1024) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  const long grid = (Mfull + g.R - 1) / g.R;
  k<<<(unsigned)grid, NTG, smem, st>>>(X, Y, W, B, w_bf, RS, MEAN, RSTD, Mfull, N, g.R, eps);
}

template <typename XT, bool RMS>
bool launch_fwd_stage(const XT* X, XT* Y, const void* W, const void* B, bool w_bf, const XT* RS, float* MEAN, float* RSTD, long M, int N, float eps, cudaStream_t st) {
  if (N > STAGE_MAX_N || M < STAGE_MIN_ROWS || !aligned16(X) || !aligned16(Y)) return false;
  const StageGeom g = stage_geom<XT>(N, false);
  const long Mfull = (M / g.U) * g.U;
  switch (vs_bucket(N)) {
    case 3: run_fwd_stage<XT, RMS, 3>(X, Y, W, B, w_bf, RS, MEAN, RSTD, Mfull, N, g, eps, st); break;
    case 5: run_fwd_stage<XT, RMS, 5>(X, Y, W, B, w_bf, RS, MEAN, RSTD, Mfull, N, g, eps, st); break;
    case 9: run_fwd_stage<XT, RMS, 9>(X, Y, W, B, w_bf, RS, MEAN, RSTD, Mfull, N, g, eps, st); break;
    case 12: run_fwd_stage<XT, RMS, 12>(X, Y, W, B, w_bf, RS, MEAN, RSTD, Mfull, N, g, eps, st); break;
    case 15: run_fwd_stage<XT, RMS, 15>(X, Y, W, B, w_bf, RS, MEAN, RSTD, Mfull, N, g, eps, st); break;
    case 18: run_fwd_stage<XT, RMS, 18>(X, Y, W, B, w_bf, RS, MEAN, RSTD, Mfull, N, g, eps, st); break;
    case 27: run_fwd_stage<XT, RMS, 27>(X, Y, W, B, w_bf, RS, MEAN, RSTD, Mfull, N, g, eps, st); break;
    default: run_fwd_stage<XT, RMS, 32>(X, Y, W, B, w_bf, RS, MEAN, RSTD, Mfull, N, g, eps, st); break;
  }
  if (Mfull < M)   // the tail rows
    launch_fwd_scalar<XT, RMS>(X + Mfull * N, Y + Mfull * N, W, B, w_bf, RS ? RS + Mfull : nullptr, MEAN ? MEAN + Mfull : nullptr, RSTD ? RSTD + Mfull : nullptr, M - Mfull, N, eps, st);
  return true;
}

template <typename XT, bool RMS>
void dispatch_fwd(const at::Tensor& x, const c10::optional<at::Tensor>& w, const c10::optional<at::Tensor>& b, const c10::optional<at::Tensor>& rs, at::Tensor& y,
                  const c10::optional<at::Tensor>& mean, const c10::optional<at::Tensor>& rstd, float eps) {
  const long M = x.size(0);
  const int N = (int)x.size(1);
  const XT* X = reinterpret_cast<const XT*>(x.data_ptr());
  XT* Y = reinterpret_cast<XT*>(y.data_ptr());
  const void* W = optp(w);
  const void* B = optp(b);
  const bool w_bf = w.has_value() && w->defined() ? is_bf16(*w) : (b.has_value() && b->defined() ? is_bf16(*b) : false);
  const XT* RS = optp_t<XT>(rs);
  float* MEAN = optp_w<float>(mean);
  float* RSTD = optp_w<float>(rstd);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int CE = Chunk<XT>::CE;
  const bool vec = N % CE == 0 && aligned16(X) && aligned16(Y) && aligned16(W) && aligned16(B);
  if (vec && launch_fwd_vec<XT, RMS>(N / CE, X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st)) return;
  if (launch_fwd_stage<XT, RMS>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st)) return;
  launch_fwd_scalar<XT, RMS>(X, Y, W, B, w_bf, RS, MEAN, RSTD, M, N, eps, st);
}

// partial rows -> dw / db (the vector kernel when the width allows float4 columns)
void run_reduce(const float* PDW, const float* PDB, void* DW, void* DB, bool o_bf, int P, int N, cudaStream_t st) {
  if (N % 4 == 0 && aligned16(PDW) && aligned16(PDB) && aligned16(DW) && aligned16(DB))
    reduce_partials4<<<(N / 4 + 7) / 8, dim3(8, 32), 0, st>>>(PDW, PDB, DW, DB, o_bf, P, N);
  else
    reduce_partials<<<(N + 31) / 32, dim3(32, 16), 0, st>>>(PDW, PDB, DW, DB, o_bf, P, N);
}

template <typename XT, int MODE, int G, int V>
void run_bwd_vec(const XT* DY, const XT* X, const void* W, bool w_bf, const XT* RS, const float* MEAN, const float* RSTD, XT* DX, bool need_dw, bool need_db,
                 long M, int N, void* DW, void* DB, bool o_bf, const at::Tensor& like, cudaStream_t st) {
  constexpr int RPW = 32 / G;
  const long groups = (M + RPW - 1) / RPW;
  const int sms = sm_count();
  // the persistent grid: at most one wave of CTAs of NTB threads.  Little work (a small M) takes smaller CTAs and, past one CTA per SM, no more than one per 64 rows: every CTA
  // costs one partial row of dw / db (written, then read by the reduction kernel, whose latency is what a small M pays for), and a warp keeps several row groups in flight anyway.
  const int nt = groups <= 2048 ? 128 : NTB;
  const size_t dyn = (size_t)(nt / 32) * N * sizeof(float);
  auto stat = bwd_vec<XT, G, V, MODE, false>;
  const int resident = sms * blocks_per_sm(stat, nt, dyn);
  const int grid = (int)std::min<long>({(groups + nt / 32 - 1) / (nt / 32), (long)resident, std::max<long>(sms, (M + 63) / 64)});
  const bool dyn_sched = groups > 32 * (long)grid * (nt / 32);
  at::Tensor pdw, pdb, ctr;
  float *PDW = nullptr, *PDB = nullptr;
  if (need_dw) { pdw = at::empty({grid, N}, like.options().dtype(at::kFloat)); PDW = pdw.data_ptr<float>(); }
  if (need_db) { pdb = at::empty({grid, N}, like.options().dtype(at::kFloat)); PDB = pdb.data_ptr<float>(); }
  if (dyn_sched) {
    ctr = at::zeros({1}, like.options().dtype(at::kInt));
    bwd_vec<XT, G, V, MODE, true><<<grid, nt, dyn, st>>>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, PDW, PDB, ctr.data_ptr<int>(), M, N);
  } else {
    stat<<<grid, nt, dyn, st>>>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, PDW, PDB, nullptr, M, N);
  }
  if (need_dw || need_db) run_reduce(PDW, PDB, DW, DB, o_bf, grid, N, st);
}

// the wide-row backward (bf16, V = 6, 8, 10)
template <typename XT, int MODE, int V>
void run_bwd_wide(const XT* DY, const XT* X, const void* W, bool w_bf, const XT* RS, const float* MEAN, const float* RSTD, XT* DX, bool need_dw, bool need_db,
                  long M, int N, void* DW, void* DB, bool o_bf, const at::Tensor& like, cudaStream_t st) {
  auto k = bwd_wide<XT, V, MODE>;
  const size_t dyn = (size_t)(MODE == 0 ? 2 : 1) * NWS * N * sizeof(float);
  const int grid = (int)std::min<long>({(M + NWS - 1) / NWS, (long)sm_count() * blocks_per_sm(k, NTS, dyn), std::max<long>(sm_count(), (M + 63) / 64)});   // a CTA = one partial row of dw / db
  at::Tensor pdw, pdb;
  float *PDW = nullptr, *PDB = nullptr;
  if (need_dw) { pdw = at::empty({grid, N}, like.options().dtype(at::kFloat)); PDW = pdw.data_ptr<float>(); }
  if (need_db) { pdb = at::empty({grid, N}, like.options().dtype(at::kFloat)); PDB = pdb.data_ptr<float>(); }
  k<<<grid, NTS, dyn, st>>>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, PDW, PDB, M, N);
  if (need_dw || need_db) run_reduce(PDW, PDB, DW, DB, o_bf, grid, N, st);
}

// the register-cached scalar backward (N <= 32 VS)
template <typename XT, bool RMS, int VS>
void run_bwd_scalar_reg(const XT* DY, const XT* X, const void* W, bool w_bf, const XT* RS, const float* MEAN, const float* RSTD, XT* DX, bool need_dw, bool need_db,
                        long M, int N, void* DW, void* DB, bool o_bf, const at::Tensor& like, cudaStream_t st) {
  auto k = bwd_scalar_reg<XT, VS, RMS>;
  const size_t dyn = (size_t)NWS * N * sizeof(float);
  const int grid = (int)std::min<long>({(M + NWS - 1) / NWS, (long)sm_count() * blocks_per_sm(k, NTS, dyn), std::max<long>(sm_count(), (M + 63) / 64)});   // a CTA = one partial row of dw / db
  at::Tensor pdw, pdb;
  float *PDW = nullptr, *PDB = nullptr;
  if (need_dw) { pdw = at::empty({grid, N}, like.options().dtype(at::kFloat)); PDW = pdw.data_ptr<float>(); }
  if (need_db) { pdb = at::empty({grid, N}, like.options().dtype(at::kFloat)); PDB = pdb.data_ptr<float>(); }
  k<<<grid, NTS, dyn, st>>>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, PDW, PDB, M, N);
  if (need_dw || need_db) run_reduce(PDW, PDB, DW, DB, o_bf, grid, N, st);
}

template <typename XT, bool RMS, int VS>
void run_bwd_scalar(const XT* DY, const XT* X, const void* W, bool w_bf, const XT* RS, const float* MEAN, const float* RSTD, XT* DX, bool need_dw, bool need_db,
                    long M, int N, void* DW, void* DB, bool o_bf, const at::Tensor& like, cudaStream_t st) {
  auto k = bwd_scalar<XT, VS, RMS>;
  const size_t dyn = (size_t)2 * NWS * N * sizeof(float);
  const int grid = (int)std::min<long>({(M + NWS - 1) / NWS, (long)sm_count() * blocks_per_sm(k, NTS, dyn), std::max<long>(sm_count(), (M + 63) / 64)});   // a CTA = one partial row of dw / db
  at::Tensor pdw, pdb;
  float *PDW = nullptr, *PDB = nullptr;
  if (need_dw) { pdw = at::empty({grid, N}, like.options().dtype(at::kFloat)); PDW = pdw.data_ptr<float>(); }
  if (need_db) { pdb = at::empty({grid, N}, like.options().dtype(at::kFloat)); PDB = pdb.data_ptr<float>(); }
  k<<<grid, NTS, dyn, st>>>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, PDW, PDB, M, N);
  if (need_dw || need_db) run_reduce(PDW, PDB, DW, DB, o_bf, grid, N, st);
}

template <typename XT, bool RMS, int VS, int RT>
void run_bwd_stage(const XT* DY, const XT* X, const void* W, bool w_bf, const XT* RS, const float* MEAN, const float* RSTD, XT* DX, bool need_dw, bool need_db, long M, int N,
                   void* DW, void* DB, bool o_bf, const at::Tensor& like, const StageGeom& g, cudaStream_t st) {
  const long Mfull = (M / g.U) * g.U;
  const long ntiles = (Mfull + g.R - 1) / g.R;
  const size_t smem = (size_t)4 * g.R * g.rowb;                      // two stages of (x tile, dy tile)
  auto k = bwd_stage<XT, VS, RMS, RT>;
  const int resident = sm_count() * blocks_per_sm(k, NTG, smem);
  const int grid = (int)std::min<long>({ntiles, (long)resident, std::max<long>(sm_count(), (Mfull + 63) / 64)});
  const int P = grid + (Mfull < M ? 1 : 0);                           // + the partial row of the tail
  at::Tensor pdw, pdb;
  float *PDW = nullptr, *PDB = nullptr;
  if (need_dw) { pdw = at::empty({P, N}, like.options().dtype(at::kFloat)); PDW = pdw.data_ptr<float>(); }
  if (need_db) { pdb = at::empty({P, N}, like.options().dtype(at::kFloat)); PDB = pdb.data_ptr<float>(); }
  k<<<grid, NTG, smem, st>>>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, PDW, PDB, Mfull, N, g.R);
  if (Mfull < M) {
    auto kt = bwd_scalar<XT, 0, RMS>;
    const size_t dyn = (size_t)2 * NWS * N * sizeof(float);
    if (dyn > 48 * 1024) cudaFuncSetAttribute(kt, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
    kt<<<1, NTS, dyn, st>>>(DY + Mfull * N, X + Mfull * N, W, w_bf, RS ? RS + Mfull : nullptr, MEAN ? MEAN + Mfull : nullptr, RSTD + Mfull, DX + Mfull * N,
                            PDW ? PDW + (long)grid * N : nullptr, PDB ? PDB + (long)grid * N : nullptr, M - Mfull, N);
  }
  if (need_dw || need_db) run_reduce(PDW, PDB, DW, DB, o_bf, P, N, st);
}

template <typename XT, bool RMS>
bool launch_bwd_stage(const XT* DY, const XT* X, const void* W, bool w_bf, const XT* RS, const float* MEAN, const float* RSTD, XT* DX, bool need_dw, bool need_db, long M, int N,
                      void* DW, void* DB, bool o_bf, const at::Tensor& like, cudaStream_t st) {
  if (N > STAGE_MAX_N || M < STAGE_MIN_ROWS || !aligned16(X) || !aligned16(DY) || !aligned16(DX)) return false;
  const StageGeom g = stage_geom<XT>(N, true);
  const int rt = (g.R + NWG - 1) / NWG;
#define RUN(vs, rtv) run_bwd_stage<XT, RMS, vs, rtv>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, need_dw, need_db, M, N, DW, DB, o_bf, like, g, st)
#define BY_RT(vs) if (rt <= 1) RUN(vs, 1); else RUN(vs, 2)
  switch (vs_bucket(N)) {
    case 3: BY_RT(3); break;
    case 5: BY_RT(5); break;
    case 9: BY_RT(9); break;
    case 12: BY_RT(12); break;
    case 15: BY_RT(15); break;
    case 18: BY_RT(18); break;
    case 27: BY_RT(27); break;
    default: BY_RT(32); break;
  }
#undef BY_RT
#undef RUN
  return true;
}

template <typename XT, bool RMS>
void dispatch_bwd(const at::Tensor& dy, const at::Tensor& x, const c10::optional<at::Tensor>& w, const c10::optional<at::Tensor>& rs,
                  const c10::optional<at::Tensor>& mean, const at::Tensor& rstd, at::Tensor& dx, c10::optional<at::Tensor>& dw, c10::optional<at::Tensor>& db) {
  const long M = x.size(0);
  const int N = (int)x.size(1);
  const XT* DY = reinterpret_cast<const XT*>(dy.data_ptr());
  const XT* X = reinterpret_cast<const XT*>(x.data_ptr());
  XT* DX = reinterpret_cast<XT*>(dx.data_ptr());
  const void* W = optp(w);
  const bool w_bf = w.has_value() && w->defined() && is_bf16(*w);
  const XT* RS = optp_t<XT>(rs);
  const float* MEAN = optp_t<float>(mean);
  const float* RSTD = rstd.data_ptr<float>();
  const bool need_dw = dw.has_value() && dw->defined(), need_db = db.has_value() && db->defined();
  void* DW = need_dw ? dw->data_ptr() : nullptr;
  void* DB = need_db ? db->data_ptr() : nullptr;
  const bool o_bf = need_dw ? is_bf16(*dw) : (need_db ? is_bf16(*db) : false);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int CE = Chunk<XT>::CE;
  const bool vec = N % CE == 0 && aligned16(X) && aligned16(DY) && aligned16(DX) && aligned16(W);
  if (vec) {
    const auto [G, V] = lane_group(N / CE);
    if constexpr (std::is_same_v<XT, bf>) {
      if (G == 32 && (V == 6 || V == 8 || V == 10)) {             // wide rows: the partials go to shared memory
#define WIDE(vv)                                                                                                                                       \
  if (V == vv) {                                                                                                                                       \
    if constexpr (RMS) run_bwd_wide<XT, 2, vv>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, need_dw, need_db, M, N, DW, DB, o_bf, x, st);                      \
    else if (need_db) run_bwd_wide<XT, 0, vv>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, need_dw, need_db, M, N, DW, DB, o_bf, x, st);                        \
    else run_bwd_wide<XT, 1, vv>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, need_dw, need_db, M, N, DW, DB, o_bf, x, st);                                    \
    return;                                                                                                                                            \
  }
        WIDE(6) WIDE(8) WIDE(10)
#undef WIDE
      }
    }
    // MODE 0: LayerNorm with db, 1: LayerNorm without, 2: RMSNorm
#define TRY(g, v)                                                                                                                                      \
  if (G == g && V == v) {                                                                                                                              \
    if constexpr (RMS) run_bwd_vec<XT, 2, g, v>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, need_dw, need_db, M, N, DW, DB, o_bf, x, st);                    \
    else if (need_db) run_bwd_vec<XT, 0, g, v>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, need_dw, need_db, M, N, DW, DB, o_bf, x, st);                      \
    else run_bwd_vec<XT, 1, g, v>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, need_dw, need_db, M, N, DW, DB, o_bf, x, st);                                  \
    return;                                                                                                                                            \
  }
    NORMS_PAIRS_BWD(TRY)
#undef TRY
  }
  if (launch_bwd_stage<XT, RMS>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, need_dw, need_db, M, N, DW, DB, o_bf, x, st)) return;
  // the row in registers (up to 1024 columns), per width bucket; wider rows re-read the row from L1 for each pass
#define RUN_REG(vs) run_bwd_scalar_reg<XT, RMS, vs>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, need_dw, need_db, M, N, DW, DB, o_bf, x, st)
  if (N <= STAGE_MAX_N) {
    switch (vs_bucket(N)) {
      case 3: RUN_REG(3); break;
      case 5: RUN_REG(5); break;
      case 9: RUN_REG(9); break;
      case 12: RUN_REG(12); break;
      case 15: RUN_REG(15); break;
      case 18: RUN_REG(18); break;
      case 27: RUN_REG(27); break;
      default: RUN_REG(32); break;
    }
  } else {
    run_bwd_scalar<XT, RMS, 0>(DY, X, W, w_bf, RS, MEAN, RSTD, DX, need_dw, need_db, M, N, DW, DB, o_bf, x, st);
  }
#undef RUN_REG
}

void check_params(const char* what, const c10::optional<at::Tensor>& t, int64_t N) {
  if (!t.has_value() || !t->defined()) return;
  TORCH_CHECK(t->is_cuda() && t->dim() == 1 && t->size(0) == N && t->is_contiguous(), what, ": a contiguous [N] CUDA tensor");
  TORCH_CHECK(t->scalar_type() == at::kFloat || t->scalar_type() == at::kBFloat16, what, ": fp32 or bf16");
}

}  // namespace

// y = norm(x) over the rows of x [M, N] (contiguous, bf16 or fp32).  w / b: [N] fp32 or bf16 (None: ones / zeros; one dtype for both); rs: [M] in x's dtype or None; mean / rstd:
// [M] fp32 or None (the statistics for the backward; mean is None for RMSNorm).
void norm_fwd(at::Tensor x, c10::optional<at::Tensor> w, c10::optional<at::Tensor> b, c10::optional<at::Tensor> rs, at::Tensor y, c10::optional<at::Tensor> mean,
              c10::optional<at::Tensor> rstd, double eps, bool rms) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && y.is_contiguous() && y.sizes() == x.sizes() && y.scalar_type() == x.scalar_type(), "norm_fwd: x / y");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "norm_fwd: bf16 or fp32 rows");
  const int64_t N = x.size(1);
  check_params("norm_fwd w", w, N);
  check_params("norm_fwd b", b, N);
  TORCH_CHECK(!(rms && b.has_value() && b->defined()), "norm_fwd: RMSNorm has no bias");
  TORCH_CHECK(!(w.has_value() && w->defined() && b.has_value() && b->defined()) || w->scalar_type() == b->scalar_type(), "norm_fwd: w and b share a dtype");
  c10::cuda::CUDAGuard guard(x.device());
  if (x.numel() == 0) return;
  if (x.scalar_type() == at::kBFloat16) {
    if (rms) dispatch_fwd<bf, true>(x, w, b, rs, y, mean, rstd, (float)eps);
    else dispatch_fwd<bf, false>(x, w, b, rs, y, mean, rstd, (float)eps);
  } else {
    if (rms) dispatch_fwd<float, true>(x, w, b, rs, y, mean, rstd, (float)eps);
    else dispatch_fwd<float, false>(x, w, b, rs, y, mean, rstd, (float)eps);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "norm_fwd: launch failed");
}

// dx (same dtype as x), and dw / db (fp32 or bf16 [N]) when given: filled by this call.
void norm_bwd(at::Tensor dy, at::Tensor x, c10::optional<at::Tensor> w, c10::optional<at::Tensor> rs, c10::optional<at::Tensor> mean, at::Tensor rstd, at::Tensor dx,
              c10::optional<at::Tensor> dw, c10::optional<at::Tensor> db, bool rms) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && dy.is_contiguous() && dx.is_contiguous() && dy.sizes() == x.sizes() && dx.sizes() == x.sizes(), "norm_bwd: x / dy / dx");
  TORCH_CHECK(dy.scalar_type() == x.scalar_type() && dx.scalar_type() == x.scalar_type(), "norm_bwd: dy, dx and x share a dtype");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "norm_bwd: bf16 or fp32 rows");
  TORCH_CHECK(rstd.is_cuda() && rstd.scalar_type() == at::kFloat && rstd.numel() == x.size(0), "norm_bwd: rstd [M] fp32");
  TORCH_CHECK(rms || (mean.has_value() && mean->defined()), "norm_bwd: LayerNorm needs the mean");
  const int64_t N = x.size(1);
  check_params("norm_bwd w", w, N);
  check_params("norm_bwd dw", dw, N);
  check_params("norm_bwd db", db, N);
  TORCH_CHECK(!(rms && db.has_value() && db->defined()), "norm_bwd: RMSNorm has no bias");
  TORCH_CHECK(!(dw.has_value() && dw->defined() && db.has_value() && db->defined()) || dw->scalar_type() == db->scalar_type(), "norm_bwd: dw and db share a dtype");
  c10::cuda::CUDAGuard guard(x.device());
  if (x.numel() == 0) {
    if (dw.has_value() && dw->defined()) dw->zero_();
    if (db.has_value() && db->defined()) db->zero_();
    return;
  }
  if (x.scalar_type() == at::kBFloat16) {
    if (rms) dispatch_bwd<bf, true>(dy, x, w, rs, mean, rstd, dx, dw, db);
    else dispatch_bwd<bf, false>(dy, x, w, rs, mean, rstd, dx, dw, db);
  } else {
    if (rms) dispatch_bwd<float, true>(dy, x, w, rs, mean, rstd, dx, dw, db);
    else dispatch_bwd<float, false>(dy, x, w, rs, mean, rstd, dx, dw, db);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "norm_bwd: launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("norm_fwd", &norm_fwd);
  m.def("norm_bwd", &norm_bwd);
}
