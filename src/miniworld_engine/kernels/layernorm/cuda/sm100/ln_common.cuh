// General LayerNorm over the last dimension (any width, any row count): shared pieces of the forward and backward kernels.
//
// A row of N elements is held by TPR threads (TPR in 4..1024, TPR <= 32 is a sub-warp group of a warp); each thread keeps CHUNKS
// vectors of VEC elements of its row in registers -- column (c * TPR + lane) * VEC for chunk c, so a warp's loads are contiguous.
// VEC = 16 bytes / sizeof(T) when N is a multiple of it (and the rows are 16-byte aligned), otherwise 1 (scalar, any N). A CTA is
// max(TPR, 256) threads = RPC = CTA / TPR rows at a time, and grid-strides over the row groups. Statistics are fp32.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cooperative_groups.h>

namespace mwln {

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }

template <typename T, int VEC> struct alignas(sizeof(T) * VEC) Pack { T v[VEC]; };

template <int TPR> constexpr int cta_threads() { return TPR >= 256 ? TPR : 256; }

// Multiprocessor count of the current device, looked up once per device (cudaDeviceGetAttribute is not free per launch).
inline int num_sms() {
  static int cached[64] = {0};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= 64) dev = 0;
  if (!cached[dev]) cudaDeviceGetAttribute(&cached[dev], cudaDevAttrMultiProcessorCount, dev);
  return cached[dev];
}

// Programmatic dependent launch (round 7, 2026-11-18): the kernels of this extension are launched as programmatic dependents of the
// previous kernel of their stream (``launch`` below) and enter through ``pdl_wait`` before they read anything the previous kernel may
// have written; ``pdl_trigger`` (issued after the wait, so the grid before the previous one has completed when a dependent starts)
// lets the next launch's CTAs become resident while this grid still runs. Without the launch attribute both are no-ops. Bits of
// ``pdl_mode``: 1 forward, 2 backward, 4 column pass.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

inline int& pdl_mode() {
  static int m = 7;
  return m;
}

// whether a forward (bit 1) / backward (bit 2) launch is a programmatic dependent: activations under 8 MB only (the kernels with the
// wait are separate builds, PDL true; measured on the B200, round 7: the wait / trigger in every build cost 8-40 % on the large
// HBM-bound training steps and 1.7x on the lean forward, and a large grid has no launch latency left to hide)
inline bool pdl_on(bool is_fwd, int64_t bytes) {
  return (pdl_mode() & (is_fwd ? 1 : 2)) != 0 && bytes < (int64_t(8) << 20);
}

// the (rows a thread, warps a CTA) launches of the row kernels that get a PDL build: the ones the dispatch tables pick (rows_config,
// BWD_LAUNCH, FWD_DIRECT4, the five-warp 2560 rows); the others run without PDL
constexpr bool pdl_rl(int r, int w) {
  return (r == 1 && (w == 2 || w == 4 || w == 5)) || (r == 2 && (w == 2 || w == 4)) || (r == 4 && (w == 4 || w == 5));
}

template <typename... KArgs, typename... Args>
inline void launch(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, bool pdl, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...);
}

// The column role of a fused backward (round 8, 2026-11-18): dw / db summed by extra CTAs of the backward's own grid, concurrently
// with the dx CTAs (blocks [0, gx)), so a training step is two launches and the dx CTAs carry no partial sums. The grid is launched
// in thread-block clusters of K CTAs (gx a multiple of K): column CTA gx + K cr + k is rank k of the cluster of column range cr (CW
// columns: LP = 16 lanes a row x V elements a lane, 16-byte loads; two row slots a warp) and sums the rows of group k (W x 2 x nrt
// consecutive rows a slot). An odd width (SK) reads the aligned vectors a row spans and realigns them through a shuffle from the
// next lane (lane 15 only feeds lane 14: CW = 15 V). The K groups of a range are added over DSMEM after a cluster barrier, rank k
// the k-th slice of the outputs, the ranks in order: every sum has a fixed order, a replay is bit-identical; no partial rows in
// global memory, no counters, no atomics.
struct ColArgs {
  void* out;   // dw [N]
  void* out2;  // db [N] (null: RMS)
  int obf;     // the outputs are bf16
  int gx;      // the dx CTAs (blocks [0, gx), a multiple of K); the column CTAs follow
  int K;       // row groups = the cluster size
  int nrt;     // rows a row slot
  int dbg;     // experiment bits (set_fuse_dbg): 1 the column CTAs only meet at the barriers
};

template <bool SK, int LP>
__host__ __device__ constexpr int col_width(int V) { return (SK ? LP - 1 : LP) * V; }

// the V elements of a lane that start m elements into its aligned 16-byte vector a (b: the next lane's vector)
template <int V>
__device__ __forceinline__ uint4 realign(const uint4& a, const uint4& b, int m) {
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  uint32_t o[4];
  if constexpr (V == 4) {  // fp32: m words
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = m == 0 ? w[i] : m == 1 ? w[i + 1] : m == 2 ? w[i + 2] : w[i + 3];
  } else {  // bf16: m half-words
    const int q = m >> 1;
    uint32_t s[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) s[j] = q == 0 ? w[j] : q == 1 ? w[j + 1] : q == 2 ? w[j + 2] : w[j + 3];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = (m & 1) ? __funnelshift_r(s[i], s[i + 1], 16) : s[i];
  }
  return make_uint4(o[0], o[1], o[2], o[3]);
}

template <typename T>
__device__ __forceinline__ void unpack_vec(const uint4& u, float* f) {
  if constexpr (sizeof(T) == 4) {
    f[0] = __uint_as_float(u.x);
    f[1] = __uint_as_float(u.y);
    f[2] = __uint_as_float(u.z);
    f[3] = __uint_as_float(u.w);
  } else {
    const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[2 * i] = __uint_as_float(q[i] << 16);
      f[2 * i + 1] = __uint_as_float(q[i] & 0xffff0000u);
    }
  }
}

template <typename T, int NT, bool RMS, bool SK>
__device__ __forceinline__ void col_role(const ColArgs& a, const T* __restrict__ dy, const T* __restrict__ x, const float* __restrict__ mean,
                                         const float* __restrict__ rstd, int64_t M, int N) {
  namespace cg = cooperative_groups;
  constexpr int V = 16 / int(sizeof(T)), W = NT / 32, LP = 16, RS = 32 / LP, CW = col_width<SK, LP>(V), LO = SK ? LP - 1 : LP;
  constexpr int RB = 8;  // rows a slot loads at a time
  static_assert(NT % 32 == 0, "whole warps");
  __shared__ float cred[W * RS][2][CW];
  __shared__ float cpart[2 * CW];
  cg::cluster_group cl = cg::this_cluster();
  const int k = int(cl.block_rank()), cr = (int(blockIdx.x) - a.gx) / a.K;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, sl = lane / LP, li = lane % LP;
  const int slot = w * RS + sl;
  const int c0 = cr * CW + li * V;
  const int64_t total = M * int64_t(N);
  const int64_t r0 = (int64_t(k) * (W * RS) + slot) * a.nrt;  // this slot's first row
  float sw[V], sb[V];
#pragma unroll
  for (int i = 0; i < V; ++i) sw[i] = sb[i] = 0.f;
  if (!(a.dbg & 1)) {
    for (int i0 = 0; i0 < a.nrt; i0 += LP) {
      const int64_t rb = r0 + i0;
      const int nr = min(LP, a.nrt - i0);
      float mu_l = 0.f, rs_l = 0.f;  // lane li of the slot: the statistics of row rb + li
      if (li < nr && rb + li < M) {
        if constexpr (!RMS) mu_l = mean[rb + li];
        rs_l = rstd[rb + li];
      }
      for (int j0 = 0; j0 < nr; j0 += RB) {
        uint4 xv[RB], dv[RB];
        int mm[RB];
#pragma unroll
        for (int j = 0; j < RB; ++j) {
          const int64_t r = rb + j0 + j;
          const int64_t o = r * N + c0;
          const int m = SK ? int(o & (V - 1)) : 0;
          const int64_t base = o - m;
          const bool ok = j0 + j < nr && r < M && base < total && (SK || c0 < N);
          mm[j] = m;
          xv[j] = ok ? *reinterpret_cast<const uint4*>(x + base) : make_uint4(0u, 0u, 0u, 0u);
          dv[j] = ok ? *reinterpret_cast<const uint4*>(dy + base) : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int j = 0; j < RB; ++j) {
          uint4 xa = xv[j], da = dv[j];
          if constexpr (SK) {
            uint4 xn, dn;
            xn.x = __shfl_down_sync(0xffffffffu, xa.x, 1, LP);
            xn.y = __shfl_down_sync(0xffffffffu, xa.y, 1, LP);
            xn.z = __shfl_down_sync(0xffffffffu, xa.z, 1, LP);
            xn.w = __shfl_down_sync(0xffffffffu, xa.w, 1, LP);
            dn.x = __shfl_down_sync(0xffffffffu, da.x, 1, LP);
            dn.y = __shfl_down_sync(0xffffffffu, da.y, 1, LP);
            dn.z = __shfl_down_sync(0xffffffffu, da.z, 1, LP);
            dn.w = __shfl_down_sync(0xffffffffu, da.w, 1, LP);
            xa = realign<V>(xa, xn, mm[j]);
            da = realign<V>(da, dn, mm[j]);
          }
          const float mu = __shfl_sync(0xffffffffu, mu_l, (j0 + j) & (LP - 1), LP);
          const float rs = __shfl_sync(0xffffffffu, rs_l, (j0 + j) & (LP - 1), LP);
          float fx[V], fd[V];
          unpack_vec<T>(xa, fx);
          unpack_vec<T>(da, fd);
          const bool rowok = j0 + j < nr && rb + j0 + j < M && li < LO;
#pragma unroll
          for (int i = 0; i < V; ++i) {
            const float d = rowok && c0 + i < N ? fd[i] : 0.f;
            const float xh = (fx[i] - mu) * rs;
            sw[i] += d * xh;
            if constexpr (!RMS) sb[i] += d;
          }
        }
      }
    }
  }
  if (li < LO) {
#pragma unroll
    for (int i = 0; i < V; ++i) {
      cred[slot][0][li * V + i] = sw[i];
      cred[slot][1][li * V + i] = sb[i];
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < 2 * CW; idx += NT) {
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < W * RS; ++q) s += cred[q][idx / CW][idx % CW];
    cpart[idx] = s;
  }
  cl.sync();  // every rank's partial is in its shared memory
  const int per = (2 * CW + a.K - 1) / a.K;
  for (int idx = k * per + threadIdx.x; idx < min(2 * CW, (k + 1) * per); idx += NT) {
    float v = 0.f;
    for (int q = 0; q < a.K; ++q) v += cl.map_shared_rank(cpart, q)[idx];
    const int o = idx / CW, c = cr * CW + idx % CW;
    void* dst = o ? a.out2 : a.out;
    if (c < N && dst) {
      if (a.obf) reinterpret_cast<__nv_bfloat16*>(dst)[c] = __float2bfloat16(v);
      else reinterpret_cast<float*>(dst)[c] = v;
    }
  }
  cl.sync();  // no rank leaves while another still reads its shared memory
}

// the column CTAs of a fused backward: ranges of CW columns, K each
__host__ __device__ inline int col_ctas(int N, int V, bool sk, int K) { return (N + (sk ? 15 : 16) * V - 1) / ((sk ? 15 : 16) * V) * K; }

// the dx CTAs of a fused backward each row slot... a dx grid padded to whole clusters
inline int pad_to(int g, int K) { return (g + K - 1) / K * K; }

// fused-backward knobs (set_fuse): rows up to which a backward with affine gradients is fused (0: never), the row groups aimed at
inline int64_t& fuse_max_rows() {
  static int64_t m = -1;  // -1: the per-call row groups (the Python table)
  return m;
}
inline int& fuse_k() {
  static int k = 16;
  return k;
}

// a fused backward's launch: K-CTA clusters, nrt rows a row slot (two slots a warp), the dx grid padded to whole clusters
struct ColPlan {
  int K, nrt, gx, grid;
};
inline ColPlan col_plan(int64_t M, int N, int V, bool sk, int W, int K, int grid) {
  ColPlan p;
  p.K = K;
  p.nrt = int((M + int64_t(K) * W * 2 - 1) / (int64_t(K) * W * 2));
  p.gx = pad_to(grid, K);
  p.grid = p.gx + col_ctas(N, V, sk, K);
  return p;
}

inline int& fuse_dbg() {
  static int d = 0;
  return d;
}


// launch() in thread-block clusters of `cluster` CTAs along x (more than 8: the non-portable size is allowed first)
template <typename... KArgs, typename... Args>
inline void launch_cl(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, bool pdl, int cluster, Args... args) {
  if (cluster > 8) cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = unsigned(cluster);
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 2 : 1;
  cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...);
}

// Sum over the TPR threads of a row. `red` holds RPC * (TPR / 32) floats when TPR > 32; every thread of the CTA must call it.
template <int TPR>
__device__ __forceinline__ float row_sum(float s, float* red, int r_in, int lane) {
#pragma unroll
  for (int o = (TPR < 32 ? TPR : 32) / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if constexpr (TPR > 32) {
    constexpr int W = TPR / 32;
    if ((lane & 31) == 0) red[r_in * W + (lane >> 5)] = s;
    __syncthreads();
    s = 0.f;
#pragma unroll
    for (int i = 0; i < W; ++i) s += red[r_in * W + i];
    __syncthreads();
  }
  return s;
}

template <typename T, int VEC>
__device__ __forceinline__ void load_vec(const T* p, float* out) {
  if constexpr (VEC == 1) {
    out[0] = to_f<T>(*p);
  } else {
    Pack<T, VEC> pk = *reinterpret_cast<const Pack<T, VEC>*>(p);
#pragma unroll
    for (int i = 0; i < VEC; ++i) out[i] = to_f<T>(pk.v[i]);
  }
}

template <typename T, int VEC>
__device__ __forceinline__ void store_vec(T* p, const float* in) {
  if constexpr (VEC == 1) {
    *p = from_f<T>(in[0]);
  } else {
    Pack<T, VEC> pk;
#pragma unroll
    for (int i = 0; i < VEC; ++i) pk.v[i] = from_f<T>(in[i]);
    *reinterpret_cast<Pack<T, VEC>*>(p) = pk;
  }
}

// A thread's affine values for its columns (fixed across the rows it visits): `fill` where `p` is null or past N. `bf` says the
// parameter is bf16 (else fp32): fp32 values are read as 16-byte vectors, bf16 ones as VEC-wide packs (the host passes aligned rows).
template <int TPR, int CHUNKS, int VEC>
__device__ __forceinline__ void load_affine(const void* __restrict__ pv, int bf, float fill, int lane, int N, float* out) {
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int col = (c * TPR + lane) * VEC;
    if (pv && col < N) {
      if (bf) {
        load_vec<__nv_bfloat16, VEC>(reinterpret_cast<const __nv_bfloat16*>(pv) + col, out + c * VEC);
      } else if constexpr (VEC % 4 == 0) {
        const float* p = reinterpret_cast<const float*>(pv);
#pragma unroll
        for (int i = 0; i < VEC; i += 4) {
          const float4 t = *reinterpret_cast<const float4*>(p + col + i);
          out[c * VEC + i] = t.x;
          out[c * VEC + i + 1] = t.y;
          out[c * VEC + i + 2] = t.z;
          out[c * VEC + i + 3] = t.w;
        }
      } else {
        const float* p = reinterpret_cast<const float*>(pv);
#pragma unroll
        for (int i = 0; i < VEC; ++i) out[c * VEC + i] = p[col + i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < VEC; ++i) out[c * VEC + i] = fill;
    }
  }
}

// A row's VEC-wide pack as loaded (kept raw in registers: half the registers of its fp32 values for bf16 / fp16), and its values.
template <typename T, int VEC>
__device__ __forceinline__ Pack<T, VEC> load_pack(const T* p) {
  return *reinterpret_cast<const Pack<T, VEC>*>(p);
}
template <typename T, int VEC>
__device__ __forceinline__ Pack<T, VEC> zero_pack() {
  Pack<T, VEC> z;
#pragma unroll
  for (int i = 0; i < VEC; ++i) z.v[i] = from_f<T>(0.f);
  return z;
}

}  // namespace mwln

// Instantiation lists, shared by the launchers. Scalar path: TPR 32..1024 x CHUNKS {1, 2, 4, 8}. Vector path: also CHUNKS 3 (a 768 /
// 1536 / 3072-wide bf16 row at a warp or two per row without idle lanes). RPT (rows a thread holds) is a template parameter, built at 1:
// two rows a thread, batched or software-pipelined, measured slower on B200 than the one-wave grid at one row.
#define MWLN_FOR_CHUNKS(M, ...) M(1, __VA_ARGS__) M(2, __VA_ARGS__) M(4, __VA_ARGS__) M(8, __VA_ARGS__)
#define MWLN_FOR_CHUNKS_V(M, ...) MWLN_FOR_CHUNKS(M, __VA_ARGS__) M(3, __VA_ARGS__)
