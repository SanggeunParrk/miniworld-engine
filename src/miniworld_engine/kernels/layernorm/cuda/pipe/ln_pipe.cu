// Warp-per-row-group LayerNorm / RMSNorm with a cp.async ring per warp (sm80+; tuned on H100).
//
// A row of N elements is N * itemsize / 16 chunks of 16 bytes: TPR lanes of a warp hold it, CH chunks per lane (TPR 16 or 32, CH > 1 only at
// TPR = 32), so a warp-pass covers RPP = 32 / TPR rows. A warp streams its passes through a ring of STAGES shared-memory stages filled by
// cp.async: every lane reads back exactly the bytes it copied, so a warp needs no barrier, and a few passes of every warp are in flight at
// once (the HBM latency x bandwidth product of an H100 needs ~20 KB per SM). The grid is persistent (SMs x resident CTAs); warp w takes passes
// w, w + W, w + 2W ... so the whole device sweeps the tensor front to back.
//
// forward : y = (x - mean) * rstd * w + b (RMS: mean = 0, no bias), fp32 statistics. The training forward also stores (mean, rstd) per row (8
//           bytes, 3% of a D128 bf16 row) so the backward is one reduction per row instead of three.
// backward: dx from x, dy and the saved statistics; the affine gradients are kept in registers per lane, summed over the warps of the CTA in
//           shared memory and stored as one fp32 partial row per CTA (plain stores, no atomics); pipe_reduce, launched behind it with
//           programmatic dependent launch, adds the partial rows in a fixed order and writes dw / db in the weight dtype. Deterministic:
//           the gradients are bit-reproducible from call to call and replay to replay.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <stdint.h>
#include <type_traits>

namespace mwpipe {

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }

#ifndef MWP_WARPS
#define MWP_WARPS 8
#endif
#ifndef MWP_S1
#define MWP_S1 4
#endif
#ifndef MWP_S2
#define MWP_S2 4
#endif
#ifndef MWP_S3
#define MWP_S3 6  // 96-chunk rows (RMS d768 bf16): six stages put the persistent forward on the Triton kernel at L384 (24.8 -> 23.8 us)
#endif
#ifndef MWP_S4
#define MWP_S4 3
#endif
#ifndef MWP_MINB_BWD
#define MWP_MINB_BWD (CH * VEC > 12 ? 1 : 2)
#endif
#ifndef MWP_MINB_FWD
#define MWP_MINB_FWD (CH * VEC > 16 ? 1 : 2)
#endif
#ifndef MWP_BWD_CTAS1
#define MWP_BWD_CTAS1 1
#endif
#ifndef MWP_BWD_CTAS
#define MWP_BWD_CTAS 0
#endif
#ifndef MWP_DIRECT_DEFAULT
#define MWP_DIRECT_DEFAULT 1
#endif
constexpr int WARPS = MWP_WARPS;  // warps per CTA
constexpr int THREADS = WARPS * 32;
#ifndef MWP_D2
#define MWP_D2 1  // exact-fit single-chunk rows with weights of the activation dtype: direct2_fwd
#endif
#ifndef MWP_D2PW
#define MWP_D2PW 2  // row passes per warp
#endif
#ifndef MWP_D3
#define MWP_D3 1  // the lean inference direct forward (direct3_fwd)
#endif
#ifndef MWP_D3PW
#define MWP_D3PW 2  // row passes per warp
#endif
#ifndef MWP_D3PW2
#define MWP_D3PW2 4  // row passes per warp when the PW-2 grid would be more than one wave of resident CTAs
#endif
#ifndef MWP_D3CH
#define MWP_D3CH 1  // the lean multi-chunk direct forward
#endif
#ifndef MWP_D3CH_MIN
#define MWP_D3CH_MIN (int64_t(32) << 20)  // activation bytes from which it runs (below: the ring, 1-2% ahead at x[1, 128, 128, 256] fp32, 16.8 MB)
#endif
#ifndef MWP_D3CH_EMAX
#define MWP_D3CH_EMAX 12  // values a lane holds of a row (CH * VEC) at most: wider rows (bf16 512, fp32 512) stay on the PW 1 direct forward, 3% ahead of it at x[1, 768, 768, 512] bf16
#endif
#ifndef MWP_D3CH_PWE
#define MWP_D3CH_PWE 48  // values a lane holds over its rows (PW * CH * VEC) at most
#endif
#ifndef MWP_RED_H100_PMAX
#define MWP_RED_H100_PMAX 192  // per-CTA partial rows up to which it runs (P = 264 at bf16 N = 384: 4.2 us against pipe_reduce 4.0 us; P = 132: 3.0 against 3.8)
#endif
#ifndef MWP_RED_H100
#define MWP_RED_H100 1  // the 1024-thread backward reduce on sm_90
#endif
#ifndef MWP_D3STAT_SMEM
#define MWP_D3STAT_SMEM 1  // the training forward stores a CTA's (mean, rstd) rows together
#endif
#ifndef MWP_D3MINB_PW
#define MWP_D3MINB_PW 4  // from this many rows a warp the build is held to MWP_D3MINB_REGS registers a thread (more CTAs resident, so the grid stays one wave)
#endif
#ifndef MWP_D3MINB_REGS
#define MWP_D3MINB_REGS 64
#endif
#ifndef MWP_D3W
#define MWP_D3W 4  // warps per CTA
#endif
#ifndef MWP_D2PDL
#define MWP_D2PDL 1  // programmatic dependent launch of the direct2 forward (small activations)
#endif
#ifndef MWP_D2W
#define MWP_D2W 4  // warps per CTA
#endif

// stages of the ring (per warp and stream: STAGES x CH x 512 bytes)
template <int CH> constexpr int stages() { return CH == 1 ? MWP_S1 : (CH == 2 ? MWP_S2 : (CH == 3 ? MWP_S3 : MWP_S4)); }

__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async16(unsigned s, const void* gmem, bool pred) {
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz) : "memory");
}
__device__ __forceinline__ void cp_async8(unsigned s, const void* gmem, bool pred) {
  const int sz = pred ? 8 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(gmem), "r"(sz) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

template <typename T, int VEC> struct alignas(sizeof(T) * VEC) Pack { T v[VEC]; };

// sum over the TPR lanes of a row group
template <int TPR> __device__ __forceinline__ float group_sum(float s) {
#pragma unroll
  for (int o = TPR / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}
template <int TPR> __device__ __forceinline__ void group_sum2(float& a, float& b) {
#pragma unroll
  for (int o = TPR / 2; o > 0; o >>= 1) {
    a += __shfl_xor_sync(0xffffffffu, a, o);
    b += __shfl_xor_sync(0xffffffffu, b, o);
  }
}

// One lane's fp32 affine values for its columns (fill where `p` is null or past N); `p` is 16-byte aligned, read as 4-byte or 16-byte vectors.
template <int TPR, int CH, int VEC>
__device__ __forceinline__ void load_affine(const void* __restrict__ p, int wcode, float fill, int l, int N, float* out) {
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const int col = (c * TPR + l) * VEC;
    if (p != nullptr && col < N) {
      if (wcode == 0) {
#pragma unroll
        for (int i = 0; i < VEC; i += 4) {
          const float4 t = __ldg(reinterpret_cast<const float4*>(reinterpret_cast<const float*>(p) + col + i));
          out[c * VEC + i] = t.x;
          out[c * VEC + i + 1] = t.y;
          out[c * VEC + i + 2] = t.z;
          out[c * VEC + i + 3] = t.w;
        }
      } else {
#pragma unroll
        for (int i = 0; i < VEC; i += 2) {
          const unsigned u = __ldg(reinterpret_cast<const unsigned*>(reinterpret_cast<const uint16_t*>(p) + col + i));
          if (wcode == 1) {
            out[c * VEC + i] = __uint_as_float(u << 16);
            out[c * VEC + i + 1] = __uint_as_float(u & 0xffff0000u);
          } else {
            const __half2 h = *reinterpret_cast<const __half2*>(&u);
            out[c * VEC + i] = __low2float(h);
            out[c * VEC + i + 1] = __high2float(h);
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < VEC; ++i) out[c * VEC + i] = fill;
    }
  }
}

// Per-lane stream cursor: the row of this lane's group in the next pass to copy / compute, and the matching element pointer
// (base + row * N + l * VEC).
template <typename P> struct Cursor {
  P* p;
  int64_t row;
};

template <typename T, int TPR, int CH, int VEC>
__device__ __forceinline__ void issue_x(unsigned sm, Cursor<const T>& cur, const T* base, int64_t M, int64_t rstep, int64_t estep, const bool* cvalid) {
  const bool rv = cur.row < M;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const bool ok = rv && cvalid[c];
    cp_async16(sm + c * 512, ok ? cur.p + c * TPR * VEC : base, ok);
  }
  cur.row += rstep;
  cur.p += estep;
}

template <typename T, int CH, int VEC>
__device__ __forceinline__ void read_pass(const uint4* ring_lane, float* v) {
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const uint4 u = ring_lane[c * 32];
    const Pack<T, VEC> pk = *reinterpret_cast<const Pack<T, VEC>*>(&u);
#pragma unroll
    for (int i = 0; i < VEC; ++i) v[c * VEC + i] = to_f<T>(pk.v[i]);
  }
}

// mean (0 for RMS) and rstd of the row from the lane's values (pad columns hold 0 and are excluded from the variance)
template <int TPR, int CH, int VEC, bool RMS>
__device__ __forceinline__ void row_stats(const float* v, const bool* cvalid, bool pad, float inv_n, float eps, float& mean, float& rstd) {
  constexpr int E = CH * VEC;
  mean = 0.f;
  if constexpr (!RMS) {
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) s += v[e];
    mean = group_sum<TPR>(s) * inv_n;
  }
  float q = 0.f;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      float d = v[c * VEC + i] - mean;
      if (pad) d = cvalid[c] ? d : 0.f;
      q = fmaf(d, d, q);
    }
  }
  rstd = rsqrtf(group_sum<TPR>(q) * inv_n + eps);
}

template <typename T, int TPR, int CH, int VEC, bool RMS, bool PAD>
__global__ void __launch_bounds__(THREADS, MWP_MINB_FWD)
    pipe_fwd(const T* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b, int wcode, T* __restrict__ y,
             float* __restrict__ ws, float2* __restrict__ stats, int64_t M, int N, float eps) {
  constexpr int RPP = 32 / TPR;
  constexpr int S = stages<CH>();
  constexpr int E = CH * VEC;
  extern __shared__ __align__(16) uint4 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane / TPR, l = lane % TPR;
  uint4* ring = smem + warp * (S * CH * 32) + lane;  // this lane's chunk 0 of stage 0
  const unsigned ring_u32 = smem_u32(ring);
  if (ws != nullptr && blockIdx.x == 0 && threadIdx.x == 0) ws[0] = 0.f;  // ws only flags "the backward wants affine gradients"
  const int64_t W = int64_t(gridDim.x) * WARPS, wid = int64_t(blockIdx.x) * WARPS + warp;
  const int64_t npass = (M + RPP - 1) / RPP;
  const int64_t rstep = W * RPP, estep = rstep * N;
  bool cvalid[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) cvalid[c] = PAD ? ((c * TPR + l) * VEC < N) : true;
  constexpr bool pad = PAD;
  const float inv_n = 1.f / float(N);
  Cursor<const T> cin{x + (wid * RPP + g) * int64_t(N) + l * VEC, wid * RPP + g};
#pragma unroll
  for (int s = 0; s < S; ++s) {
    issue_x<T, TPR, CH, VEC>(ring_u32 + s * CH * 512, cin, x, M, rstep, estep, cvalid);
    cp_commit();
  }
  float wr[E], br[E];
  load_affine<TPR, CH, VEC>(w, wcode, 1.f, l, N, wr);
  load_affine<TPR, CH, VEC>(b, wcode, 0.f, l, N, br);
  Cursor<T> cout{y + (wid * RPP + g) * int64_t(N) + l * VEC, wid * RPP + g};
  int slot = 0;
  for (int64_t u = wid; u < npass; u += W) {
    cp_wait<S - 1>();
    float v[E];
    read_pass<T, CH, VEC>(ring + slot * (CH * 32), v);
    issue_x<T, TPR, CH, VEC>(ring_u32 + slot * (CH * 512), cin, x, M, rstep, estep, cvalid);
    cp_commit();
    slot = slot + 1 == S ? 0 : slot + 1;
    float mean, rstd;
    row_stats<TPR, CH, VEC, RMS>(v, cvalid, pad, inv_n, eps, mean, rstd);
    if (cout.row < M) {
      if (stats != nullptr && l == 0) stats[cout.row] = make_float2(mean, rstd);
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        if (cvalid[c]) {
          Pack<T, VEC> o;
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const int e = c * VEC + i;
            o.v[i] = from_f<T>((v[e] - mean) * rstd * wr[e] + br[e]);
          }
          *reinterpret_cast<Pack<T, VEC>*>(cout.p + c * TPR * VEC) = o;
        }
      }
    }
    cout.row += rstep;
    cout.p += estep;
  }
  cp_wait<0>();
}

// Non-persistent forward for single-chunk rows: a warp takes PW consecutive passes, issues all their loads, then computes and stores; the
// hardware schedules the CTAs. On rows of 256-512 bytes this keeps HBM busier than the cp.async ring (no startup latency per warp, no lock-step).
template <typename T, int TPR, int VEC, bool RMS, int PW, bool PAD>
__global__ void __launch_bounds__(THREADS) direct_fwd(const T* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b, int wcode,
                                                      T* __restrict__ y, float* __restrict__ ws, float2* __restrict__ stats, int64_t M, int N,
                                                      float eps) {
  constexpr int RPP = 32 / TPR;
  constexpr int E = VEC;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane / TPR, l = lane % TPR;
  if (ws != nullptr && blockIdx.x == 0 && threadIdx.x == 0) ws[0] = 0.f;
  const int64_t wid = int64_t(blockIdx.x) * WARPS + warp;
  const bool cvalid[1] = {PAD ? (l * VEC < N) : true};
  constexpr bool pad = PAD;
  uint4 raw[PW];
#pragma unroll
  for (int k = 0; k < PW; ++k) {
    const int64_t row = (wid * PW + k) * RPP + g;
    raw[k] = make_uint4(0, 0, 0, 0);
    if (row < M && cvalid[0]) raw[k] = __ldg(reinterpret_cast<const uint4*>(x + row * int64_t(N) + l * VEC));
  }
  float wr[E], br[E];
  load_affine<TPR, 1, VEC>(w, wcode, 1.f, l, N, wr);
  load_affine<TPR, 1, VEC>(b, wcode, 0.f, l, N, br);
  const float inv_n = 1.f / float(N);
#pragma unroll
  for (int k = 0; k < PW; ++k) {
    const int64_t row = (wid * PW + k) * RPP + g;
    float v[E];
    const Pack<T, VEC> pk = *reinterpret_cast<const Pack<T, VEC>*>(&raw[k]);
#pragma unroll
    for (int i = 0; i < VEC; ++i) v[i] = to_f<T>(pk.v[i]);
    float mean, rstd;
    row_stats<TPR, 1, VEC, RMS>(v, cvalid, pad, inv_n, eps, mean, rstd);
    if (row < M) {
      if (stats != nullptr && l == 0) stats[row] = make_float2(mean, rstd);
      if (cvalid[0]) {
        Pack<T, VEC> o;
#pragma unroll
        for (int i = 0; i < VEC; ++i) o.v[i] = from_f<T>((v[i] - mean) * rstd * wr[i] + br[i]);
        *reinterpret_cast<Pack<T, VEC>*>(y + row * int64_t(N) + l * VEC) = o;
      }
    }
  }
}

// Exact-fit single-chunk rows whose weight / bias have the activation's dtype (the common case): the direct forward again, written the way a
// plain copy kernel is -- no per-element dtype branches, vector affine loads -- which keeps it at 40 registers (the generic one needs 56: 4
// resident CTAs of 8 warps per SM instead of 6), which is worth 2-3% on the pair LayerNorm forward (within ~1% of a plain copy). PW row passes per warp, DW warps per CTA.
template <typename WT, int VEC> __device__ __forceinline__ void load_vec_f(const WT* p, float* o) {
  if constexpr (std::is_same<WT, float>::value) {  // fp32 affine values of 2-byte activations (the master-weight regime): VEC / 4 float4 loads
#pragma unroll
    for (int j = 0; j < VEC / 4; ++j) {
      const float4 v = *reinterpret_cast<const float4*>(p + 4 * j);
      o[4 * j] = v.x, o[4 * j + 1] = v.y, o[4 * j + 2] = v.z, o[4 * j + 3] = v.w;
    }
  } else {
    const Pack<WT, VEC> pk = *reinterpret_cast<const Pack<WT, VEC>*>(p);
#pragma unroll
    for (int i = 0; i < VEC; ++i) o[i] = to_f<WT>(pk.v[i]);
  }
}

template <typename T, typename WT, int TPR, int VEC, bool RMS, int PW, int DW>
__global__ void __launch_bounds__(DW * 32) direct2_fwd(const T* __restrict__ x, const WT* __restrict__ w, const WT* __restrict__ b, T* __restrict__ y,
                                                       float* __restrict__ ws, float2* __restrict__ stats, int64_t M, int N, float eps) {
  constexpr int RPP = 32 / TPR;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900  // programmatic dependent launch (launched with it on sm_90+): the next launch overlaps this one
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  asm volatile("griddepcontrol.wait;\n" ::: "memory");  // x and the weights may be the predecessor's output
#endif
  const int g = lane / TPR, l = lane % TPR;
  if (ws != nullptr && blockIdx.x == 0 && threadIdx.x == 0) ws[0] = 0.f;
  const int64_t wid = int64_t(blockIdx.x) * DW + warp;
  uint4 raw[PW];
#pragma unroll
  for (int p = 0; p < PW; ++p) {
    const int64_t row = (wid * PW + p) * RPP + g;
    raw[p] = make_uint4(0, 0, 0, 0);
    // a plain global load, not ld.global.nc: at the HBM floor (x[1, 768, 768, 128] bf16, the Triton kernel at a plain copy's 2.87 TB/s) the
    // non-coherent path measured 0.3-0.4% slower
    if (row < M)
      asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(raw[p].x), "=r"(raw[p].y), "=r"(raw[p].z), "=r"(raw[p].w) : "l"(x + row * N + l * VEC));
  }
  float wr[VEC], br[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) { wr[i] = 1.f; br[i] = 0.f; }
  if (w != nullptr) {
    load_vec_f<WT, VEC>(w + l * VEC, wr);
  }
  if (!RMS && b != nullptr) {
    load_vec_f<WT, VEC>(b + l * VEC, br);
  }
  const float inv_n = 1.f / float(N);
#pragma unroll
  for (int p = 0; p < PW; ++p) {
    const int64_t row = (wid * PW + p) * RPP + g;
    float v[VEC];
    const Pack<T, VEC> pk = *reinterpret_cast<const Pack<T, VEC>*>(&raw[p]);
#pragma unroll
    for (int i = 0; i < VEC; ++i) v[i] = to_f<T>(pk.v[i]);
    float mean = 0.f;
    if constexpr (!RMS) {
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < VEC; ++i) s += v[i];
      mean = group_sum<TPR>(s) * inv_n;
    }
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const float d = v[i] - mean;
      q = fmaf(d, d, q);
    }
    const float rstd = rsqrtf(group_sum<TPR>(q) * inv_n + eps);
    if (row < M) {
      if (stats != nullptr && l == 0) stats[row] = make_float2(mean, rstd);
      Pack<T, VEC> o;
#pragma unroll
      for (int i = 0; i < VEC; ++i) o.v[i] = from_f<T>((v[i] - mean) * rstd * wr[i] + br[i]);
      *reinterpret_cast<Pack<T, VEC>*>(y + row * N + l * VEC) = o;
    }
  }
}

// The direct forward written for the fewest instructions (the cold-cache small calls are latency, not bandwidth, bound: at
// x[1, 4096, 128] fp32 the Triton kernel issues 65 instructions a row against direct2's 117, and finishes 0.4 us sooner; the wide rows
// pay it as 16-byte loads of the weights for every row). N = TPR * CH * VEC is a compile-time constant (no division), the weights are
// loaded once a warp, the reductions of the PW rows of a warp run interleaved (one shuffle chain for all of them, not one after the other
// behind branches), a CTA whose rows are all in range takes the guard-free path (FULL), and rsqrt is the bare approximate instruction
// (var + eps >= eps, never denormal). STATS: the training forward (statistics, and the backward's affine flag).
__device__ __forceinline__ float rsqrt_fast(float a) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

template <typename T, typename WT, int TPR, int CH, int VEC, bool RMS, int PW, int DW, bool STATS, bool FULL>
__device__ __forceinline__ void d3_body(const T* __restrict__ x, const WT* __restrict__ w, const WT* __restrict__ b, T* __restrict__ y, float2* __restrict__ stats, float2* sst,
                                        int64_t M, float eps) {
  constexpr int RPP = 32 / TPR;
  constexpr int E = CH * VEC;
  constexpr int N = TPR * E;
  constexpr float inv_n = 1.f / float(N);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane / TPR, l = lane % TPR;
  const int64_t row0 = (int64_t(blockIdx.x) * DW + warp) * (PW * RPP) + g;
  const T* xp = x + row0 * N + l * VEC;
  T* yp = y + row0 * N + l * VEC;
  uint4 raw[PW][CH];
#pragma unroll
  for (int p = 0; p < PW; ++p)
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      raw[p][c] = make_uint4(0, 0, 0, 0);
      if (FULL || row0 + p * RPP < M)
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(raw[p][c].x), "=r"(raw[p][c].y), "=r"(raw[p][c].z), "=r"(raw[p][c].w)
                     : "l"(xp + p * RPP * N + c * TPR * VEC));
    }
  float wr[E], br[E];
#pragma unroll
  for (int i = 0; i < E; ++i) { wr[i] = 1.f; br[i] = 0.f; }
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    if (w != nullptr) load_vec_f<WT, VEC>(w + (c * TPR + l) * VEC, wr + c * VEC);
    if (!RMS && b != nullptr) load_vec_f<WT, VEC>(b + (c * TPR + l) * VEC, br + c * VEC);
  }
  float v[PW][E], mean[PW], q[PW];
#pragma unroll
  for (int p = 0; p < PW; ++p) {
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      const Pack<T, VEC> pk = *reinterpret_cast<const Pack<T, VEC>*>(&raw[p][c]);
#pragma unroll
      for (int i = 0; i < VEC; ++i) v[p][c * VEC + i] = to_f<T>(pk.v[i]);
    }
    mean[p] = 0.f;
  }
  if constexpr (!RMS) {
#pragma unroll
    for (int p = 0; p < PW; ++p) {
      float s0 = v[p][0], s1 = v[p][1];
#pragma unroll
      for (int i = 2; i < E; i += 2) { s0 += v[p][i]; s1 += v[p][i + 1]; }
      mean[p] = s0 + s1;
    }
#pragma unroll
    for (int o = TPR / 2; o > 0; o >>= 1) {
#pragma unroll
      for (int p = 0; p < PW; ++p) mean[p] += __shfl_xor_sync(0xffffffffu, mean[p], o);
    }
#pragma unroll
    for (int p = 0; p < PW; ++p) mean[p] *= inv_n;
  }
#pragma unroll
  for (int p = 0; p < PW; ++p) {
    float q0 = 0.f, q1 = 0.f;
#pragma unroll
    for (int i = 0; i < E; i += 2) {
      const float d0 = v[p][i] - mean[p], d1 = v[p][i + 1] - mean[p];
      q0 = fmaf(d0, d0, q0);
      q1 = fmaf(d1, d1, q1);
    }
    q[p] = q0 + q1;
  }
#pragma unroll
  for (int o = TPR / 2; o > 0; o >>= 1) {
#pragma unroll
    for (int p = 0; p < PW; ++p) q[p] += __shfl_xor_sync(0xffffffffu, q[p], o);
  }
#pragma unroll
  for (int p = 0; p < PW; ++p) {
    const float rstd = rsqrt_fast(fmaf(q[p], inv_n, eps));
    if constexpr (STATS) {
#if MWP_D3STAT_SMEM
      if (l == 0) sst[(warp * PW + p) * RPP + g] = make_float2(mean[p], rstd);
#else
      if (l == 0 && (FULL || row0 + p * RPP < M)) stats[row0 + p * RPP] = make_float2(mean[p], rstd);
#endif
    }
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      Pack<T, VEC> o;
#pragma unroll
      for (int i = 0; i < VEC; ++i) o.v[i] = from_f<T>((v[p][c * VEC + i] - mean[p]) * rstd * wr[c * VEC + i] + br[c * VEC + i]);
      if (FULL || row0 + p * RPP < M) *reinterpret_cast<Pack<T, VEC>*>(yp + p * RPP * N + c * TPR * VEC) = o;
    }
  }
#if MWP_D3STAT_SMEM
  if constexpr (STATS) {  // the CTA's rows are contiguous in `stats`: whole 32-byte sectors in one store, not 8 bytes a lane from 4 warps
    __syncthreads();
    const int64_t base = int64_t(blockIdx.x) * (DW * PW * RPP);
    if (threadIdx.x < DW * PW * RPP && (FULL || base + threadIdx.x < M)) stats[base + threadIdx.x] = sst[threadIdx.x];
  }
#endif
}

template <typename T, typename WT, int TPR, int CH, int VEC, bool RMS, int PW, int DW, bool STATS>
__global__ void __launch_bounds__(DW * 32, (CH == 1 && PW >= MWP_D3MINB_PW ? 65536 / (DW * 32 * MWP_D3MINB_REGS) : 1)) direct3_fwd(const T* __restrict__ x, const WT* __restrict__ w, const WT* __restrict__ b, T* __restrict__ y,
                                                        float* __restrict__ ws, float2* __restrict__ stats, int64_t M, float eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900  // programmatic dependent launch (see launch_direct2)
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  constexpr int RPP = 32 / TPR;
  __shared__ float2 sst[STATS ? DW * PW * RPP : 1];  // the CTA's statistics, stored together (MWP_D3STAT_SMEM)
  if (STATS && ws != nullptr && blockIdx.x == 0 && threadIdx.x == 0) ws[0] = 0.f;  // the backward's affine-gradient flag
  if ((int64_t(blockIdx.x) + 1) * (DW * PW * RPP) <= M) d3_body<T, WT, TPR, CH, VEC, RMS, PW, DW, STATS, true>(x, w, b, y, stats, sst, M, eps);
  else d3_body<T, WT, TPR, CH, VEC, RMS, PW, DW, STATS, false>(x, w, b, y, stats, sst, M, eps);
}

// Inference forward of multi-chunk rows (CH = 2..4 chunks of 16 bytes per lane, a warp per row): the direct forward for wide rows. The
// persistent cp.async ring (pipe_fwd) measured 2-5% behind the HBM floor here (bf16 384 / 512, fp32 256 / 384 / 512 at L384-768 pair
// streams, from 48 MB); a warp that loads its whole row up front (CH x 16 bytes per lane in flight) and lets the hardware schedule the CTAs keeps
// HBM busier. The affine values (WT: the activation dtype or fp32) are loaded after the row, vectorized, once per warp.
#ifndef MWP_DCH
#define MWP_DCH 1
#endif
#ifndef MWP_DCH_MIN
#define MWP_DCH_MIN (int64_t(48) << 20)  // activation bytes from which it runs (below: 4-11% slower than the ring at L128, 12-34 MB)
#endif
#ifndef MWP_DCH_TRAIN
#define MWP_DCH_TRAIN 1
#endif
#ifndef MWP_DCHW
#define MWP_DCHW 4  // warps per CTA
#endif
template <typename T, typename WT, int CH, int VEC, bool RMS, bool PAD>
__global__ void __launch_bounds__(MWP_DCHW * 32) directch_fwd(const T* __restrict__ x, const WT* __restrict__ w, const WT* __restrict__ b,
                                                              T* __restrict__ y, float* __restrict__ ws, float2* __restrict__ stats, int64_t M, int N, float eps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t row = int64_t(blockIdx.x) * MWP_DCHW + warp;
  if (ws != nullptr && blockIdx.x == 0 && threadIdx.x == 0) ws[0] = 0.f;  // the training forward: the backward's affine-gradient flag
  if (row >= M) return;
  uint4 raw[CH];
  bool cv[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const int col = (c * 32 + lane) * VEC;
    cv[c] = !PAD || col < N;
    raw[c] = make_uint4(0, 0, 0, 0);
    if (cv[c])
      asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(raw[c].x), "=r"(raw[c].y), "=r"(raw[c].z), "=r"(raw[c].w) : "l"(x + row * N + col));
  }
  float v[CH * VEC];
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const Pack<T, VEC> pk = *reinterpret_cast<const Pack<T, VEC>*>(&raw[c]);
#pragma unroll
    for (int i = 0; i < VEC; ++i) v[c * VEC + i] = to_f<T>(pk.v[i]);
  }
  const float inv_n = 1.f / float(N);
  float mean = 0.f;
  if constexpr (!RMS) {
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < CH * VEC; ++e) s += v[e];
    mean = group_sum<32>(s) * inv_n;
  }
  float q = 0.f;
#pragma unroll
  for (int c = 0; c < CH; ++c)
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const float d = cv[c] ? v[c * VEC + i] - mean : 0.f;
      q = fmaf(d, d, q);
    }
  const float rstd = rsqrtf(group_sum<32>(q) * inv_n + eps);
  if (stats != nullptr && lane == 0) stats[row] = make_float2(mean, rstd);
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    if (!cv[c]) continue;
    const int col = (c * 32 + lane) * VEC;
    float wr[VEC], br[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) { wr[i] = 1.f; br[i] = 0.f; }
    if (w != nullptr) {
      const Pack<WT, VEC> pw = *reinterpret_cast<const Pack<WT, VEC>*>(w + col);
#pragma unroll
      for (int i = 0; i < VEC; ++i) wr[i] = to_f<WT>(pw.v[i]);
    }
    if (!RMS && b != nullptr) {
      const Pack<WT, VEC> pb = *reinterpret_cast<const Pack<WT, VEC>*>(b + col);
#pragma unroll
      for (int i = 0; i < VEC; ++i) br[i] = to_f<WT>(pb.v[i]);
    }
    Pack<T, VEC> o;
#pragma unroll
    for (int i = 0; i < VEC; ++i) o.v[i] = from_f<T>((v[c * VEC + i] - mean) * rstd * wr[i] + br[i]);
    *reinterpret_cast<Pack<T, VEC>*>(y + row * N + col) = o;
  }
}

// Inference forward of exact-fit single-chunk rows on sm_90 (no statistics, no workspace): rows are contiguous, so a tile of BK_TB bytes is ONE
// TMA bulk copy (cp.async.bulk, no tensor map) into shared memory, normalised in place by the CTA and written back by ONE bulk store. A
// persistent CTA keeps BK_ST tiles in flight; the copy engine moves the bytes instead of 16-byte thread loads and stores. Taken for the activation
// sizes where it measured ahead of the direct forward (MWP_BK_MIN .. MWP_BK_MAX below).
#ifndef MWP_BK
#define MWP_BK 1
#endif
#ifndef MWP_BK_TB
#define MWP_BK_TB 16384  // bytes per tile
#endif
#ifndef MWP_BK_ST
#define MWP_BK_ST 4  // tiles in flight per CTA
#endif
#ifndef MWP_BK_W
#define MWP_BK_W 8  // warps per CTA
#endif
// the activation sizes (bytes) the bulk forward takes: measured against the direct forward on an H100 (L2 flushed between calls, as the ladder
// times), x[1, L, L, 128] bf16: L256 (17 MB) 0.98x and L384 (38 MB) 0.96x the Triton kernel where the direct forward is 1.03x / 1.01x; at
// L128 (4 MB) the persistent ring does not fill (1.14x) and from ~75 MB (L768, or L384 at 256 columns) it falls behind the direct forward.
#ifndef MWP_BK_MIN
#define MWP_BK_MIN (int64_t(12) << 20)  // was 8 MB: from the lean direct3 forward the 8.4 MB calls (16384 rows of 512 bytes, fp32 weights) are 10% faster than the bulk kernel; 16.8 MB and up stay ahead on it
#endif
#ifndef MWP_BK_MAX
#define MWP_BK_MAX (int64_t(48) << 20)
#endif
__device__ __forceinline__ void bk_wait(unsigned mbar, unsigned phase) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}\n" ::"r"(mbar), "r"(phase)
      : "memory");
}
template <int TB> __device__ __forceinline__ void bk_load(unsigned dst, const void* src, unsigned bytes, unsigned mbar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(mbar), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst), "l"(src), "r"(bytes),
               "r"(mbar)
               : "memory");
}
template <typename T, int TPR, int VEC, bool RMS>
__global__ void __launch_bounds__(MWP_BK_W * 32) bulk_fwd(const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ b,
                                                          T* __restrict__ y, int64_t M, int N, float eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int RPP = 32 / TPR;                  // rows per warp pass
  constexpr int RB = TPR * VEC * int(sizeof(T));  // row bytes
  constexpr int R = MWP_BK_TB / RB;              // rows per tile
  constexpr int PASSES = R / (RPP * MWP_BK_W);
  static_assert(R % (RPP * MWP_BK_W) == 0, "a tile is a whole number of CTA passes");
  extern __shared__ __align__(128) unsigned char bk_smem[];
  __shared__ __align__(8) unsigned long long bars[MWP_BK_ST];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane / TPR, l = lane % TPR;
  const int64_t tiles = (M + R - 1) / R;
  const unsigned sbase = smem_u32(bk_smem);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < MWP_BK_ST; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&bars[s])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < MWP_BK_ST; ++s) {
      const int64_t t = blockIdx.x + int64_t(s) * gridDim.x;
      if (t < tiles) {
        const int64_t r0 = t * R;
        const unsigned bytes = unsigned((M - r0 < R ? M - r0 : R) * RB);
        bk_load<MWP_BK_TB>(sbase + s * MWP_BK_TB, x + r0 * N, bytes, smem_u32(&bars[s]));
      }
    }
  }
  float wr[VEC], br[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) { wr[i] = 1.f; br[i] = 0.f; }
  if (w != nullptr) {
    const Pack<T, VEC> pw = *reinterpret_cast<const Pack<T, VEC>*>(w + l * VEC);
#pragma unroll
    for (int i = 0; i < VEC; ++i) wr[i] = to_f<T>(pw.v[i]);
  }
  if (!RMS && b != nullptr) {
    const Pack<T, VEC> pb = *reinterpret_cast<const Pack<T, VEC>*>(b + l * VEC);
#pragma unroll
    for (int i = 0; i < VEC; ++i) br[i] = to_f<T>(pb.v[i]);
  }
  const float inv_n = 1.f / float(N);
  for (int64_t i = 0;; ++i) {
    const int64_t t = blockIdx.x + i * gridDim.x;
    if (t >= tiles) break;
    const int s = int(i % MWP_BK_ST);
    const unsigned phase = unsigned((i / MWP_BK_ST) & 1);
    const int64_t r0 = t * R;
    const int rows = int(M - r0 < R ? M - r0 : R);
    bk_wait(smem_u32(&bars[s]), phase);
    const unsigned buf = sbase + s * MWP_BK_TB;
#pragma unroll
    for (int p = 0; p < PASSES; ++p) {
      const int r = (p * MWP_BK_W + warp) * RPP + g;
      const unsigned a = buf + r * RB + l * 16;
      uint4 raw = make_uint4(0, 0, 0, 0);
      if (r < rows) asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n" : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w) : "r"(a));
      const Pack<T, VEC> pk = *reinterpret_cast<const Pack<T, VEC>*>(&raw);
      float v[VEC];
#pragma unroll
      for (int k = 0; k < VEC; ++k) v[k] = to_f<T>(pk.v[k]);
      float mean = 0.f;
      if constexpr (!RMS) {
        float sm = 0.f;
#pragma unroll
        for (int k = 0; k < VEC; ++k) sm += v[k];
        mean = group_sum<TPR>(sm) * inv_n;
      }
      float q = 0.f;
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        const float d = v[k] - mean;
        q = fmaf(d, d, q);
      }
      const float rstd = rsqrtf(group_sum<TPR>(q) * inv_n + eps);
      Pack<T, VEC> o;
#pragma unroll
      for (int k = 0; k < VEC; ++k) o.v[k] = from_f<T>((v[k] - mean) * rstd * wr[k] + br[k]);
      const uint4 ov = *reinterpret_cast<const uint4*>(&o);
      if (r < rows) asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(ov.x), "r"(ov.y), "r"(ov.z), "r"(ov.w) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // the generic-proxy writes, before the bulk store reads them
    __syncthreads();
    if (threadIdx.x == 0) {
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(y + r0 * N), "r"(buf), "r"(unsigned(rows * RB))
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      const int64_t tn = t + int64_t(MWP_BK_ST) * gridDim.x;
      if (tn < tiles) {
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");  // the store has read the stage: refill it
        const int64_t rn = tn * R;
        const unsigned bytes = unsigned((M - rn < R ? M - rn : R) * RB);
        bk_load<MWP_BK_TB>(buf, x + rn * N, bytes, smem_u32(&bars[s]));
      }
    }
  }
  if (threadIdx.x == 0) asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
#else
  __trap();
#endif
}

// the backward's per-row (mean, rstd): the lanes of a row group read one address
template <int TPR>
__device__ __forceinline__ void issue_stats(unsigned sm, const float2*& ps, int64_t& row, const float2* base, int64_t M, int64_t rstep) {
  const bool ok = row < M;
  cp_async8(sm, ok ? ps : base, ok);
  row += rstep;
  ps += rstep;
}

template <typename T, int TPR, int CH, int VEC, bool RMS, bool WG>
__global__ void __launch_bounds__(THREADS, MWP_MINB_BWD)
    pipe_bwd(const T* __restrict__ dy, const T* __restrict__ x, const void* __restrict__ w, const float2* __restrict__ stats,
             T* __restrict__ dx, float* __restrict__ ws, void* __restrict__ dw_out, void* __restrict__ db_out, int wcode, int64_t M, int N) {
  constexpr int RPP = 32 / TPR;
  constexpr int S = stages<CH>();
  constexpr int E = CH * VEC;
  constexpr int SU = S * CH * 32;  // uint4 per warp and stream
  extern __shared__ __align__(16) uint4 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane / TPR, l = lane % TPR;
  const uint4* ring_x = smem + warp * SU + lane;
  const uint4* ring_d = smem + WARPS * SU + warp * SU + lane;
  const float2* ring_s = reinterpret_cast<const float2*>(smem + 2 * WARPS * SU) + warp * (S * 32) + lane;
  const unsigned x_u32 = smem_u32(ring_x), d_u32 = smem_u32(ring_d), s_u32 = smem_u32(ring_s);
  const int64_t W = int64_t(gridDim.x) * WARPS, wid = int64_t(blockIdx.x) * WARPS + warp;
  const int64_t npass = (M + RPP - 1) / RPP;
  const int64_t rstep = W * RPP, estep = rstep * N;
  bool cvalid[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) cvalid[c] = (c * TPR + l) * VEC < N;
  const float inv_n = 1.f / float(N);
  const int64_t row0 = wid * RPP + g;
  Cursor<const T> cx{x + row0 * int64_t(N) + l * VEC, row0}, cd{dy + row0 * int64_t(N) + l * VEC, row0};
  const float2* ps = stats + row0;
  int64_t srow = row0;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    issue_x<T, TPR, CH, VEC>(x_u32 + s * CH * 512, cx, x, M, rstep, estep, cvalid);
    issue_x<T, TPR, CH, VEC>(d_u32 + s * CH * 512, cd, dy, M, rstep, estep, cvalid);
    issue_stats<TPR>(s_u32 + s * 256, ps, srow, stats, M, rstep);
    cp_commit();
  }
  float wr[E], aw[E], ab[E];
  load_affine<TPR, CH, VEC>(w, wcode, 1.f, l, N, wr);
#pragma unroll
  for (int e = 0; e < E; ++e) aw[e] = ab[e] = 0.f;
  Cursor<T> cout{dx + row0 * int64_t(N) + l * VEC, row0};
  int slot = 0;
  for (int64_t u = wid; u < npass; u += W) {
    cp_wait<S - 1>();
    float xv[E], dv[E];
    read_pass<T, CH, VEC>(ring_x + slot * (CH * 32), xv);
    read_pass<T, CH, VEC>(ring_d + slot * (CH * 32), dv);
    const float2 st = ring_s[slot * 32];
    issue_x<T, TPR, CH, VEC>(x_u32 + slot * (CH * 512), cx, x, M, rstep, estep, cvalid);
    issue_x<T, TPR, CH, VEC>(d_u32 + slot * (CH * 512), cd, dy, M, rstep, estep, cvalid);
    issue_stats<TPR>(s_u32 + slot * 256, ps, srow, stats, M, rstep);
    cp_commit();
    slot = slot + 1 == S ? 0 : slot + 1;
    const float rstd = st.y, nmr = -st.x * st.y;  // xhat = x * rstd + nmr
    float s1 = 0.f, s2 = 0.f;
    float xh[E <= 8 ? E : 1];
    // columns past N carry x = dy = 0: g = 0 there, and so do the sums
#pragma unroll
    for (int e = 0; e < E; ++e) {
      const float xhe = fmaf(xv[e], rstd, nmr);
      if constexpr (E <= 8) xh[e] = xhe;
      const float gg = dv[e] * wr[e];
      s1 += gg;
      s2 = fmaf(gg, xhe, s2);
      if constexpr (WG) {
        aw[e] = fmaf(dv[e], xhe, aw[e]);
        if constexpr (!RMS) ab[e] += dv[e];
      }
    }
    if constexpr (RMS) {
      s2 = group_sum<TPR>(s2) * inv_n;
      s1 = 0.f;
    } else {
      group_sum2<TPR>(s1, s2);
      s1 *= inv_n;
      s2 *= inv_n;
    }
    if (cout.row < M) {
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        if (cvalid[c]) {
          Pack<T, VEC> o;
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const int e = c * VEC + i;
            const float xhe = E <= 8 ? xh[E <= 8 ? e : 0] : fmaf(xv[e], rstd, nmr);
            o.v[i] = from_f<T>(rstd * fmaf(-xhe, s2, fmaf(dv[e], wr[e], -s1)));
          }
          *reinterpret_cast<Pack<T, VEC>*>(cout.p + c * TPR * VEC) = o;
        }
      }
    }
    cout.row += rstep;
    cout.p += estep;
  }
  cp_wait<0>();
  if constexpr (WG) {
    // let the reduce kernel launch behind this one (programmatic dependent launch); it waits for this grid's memory before it reads
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    // lanes l and l + TPR hold the same columns of different rows: fold them, then fold the warps of the CTA in shared memory
#pragma unroll
    for (int e = 0; e < E; ++e) {
#pragma unroll
      for (int o = TPR; o < 32; o <<= 1) {
        aw[e] += __shfl_xor_sync(0xffffffffu, aw[e], o);
        if constexpr (!RMS) ab[e] += __shfl_xor_sync(0xffffffffu, ab[e], o);
      }
    }
    __syncthreads();  // the ring is reused as scratch
    float* red = reinterpret_cast<float*>(smem);  // [warp][aw | ab][E][TPR]
    constexpr int PER = E * TPR;
    if (g == 0) {
#pragma unroll
      for (int e = 0; e < E; ++e) {
        red[(warp * 2 + 0) * PER + e * TPR + l] = aw[e];
        red[(warp * 2 + 1) * PER + e * TPR + l] = ab[e];
      }
    }
    __syncthreads();
    // one partial row per CTA, plain stores: pipe_reduce adds the rows in a fixed order (bit-reproducible)
    float* mine = ws + int64_t(blockIdx.x) * 2 * N;
    for (int idx = threadIdx.x; idx < PER; idx += THREADS) {
      float tw = 0.f, tb = 0.f;
#pragma unroll
      for (int wi = 0; wi < WARPS; ++wi) {
        tw += red[(wi * 2 + 0) * PER + idx];
        if constexpr (!RMS) tb += red[(wi * 2 + 1) * PER + idx];
      }
      const int e = idx / TPR, ll = idx % TPR;
      const int col = ((e / VEC) * TPR + ll) * VEC + e % VEC;
      if (col < N) {
        mine[col] = tw;
        if constexpr (!RMS) mine[N + col] = tb;
      }
    }
  }
}

// dw / db = the sum of the P per-CTA partial rows in a fixed order: 32 columns x 16 row slices per block, converted to the weight dtype
template <bool RMS>
__global__ void __launch_bounds__(512) pipe_reduce(const float* __restrict__ part, int P, int N, void* __restrict__ dw_out, void* __restrict__ db_out,
                                                   int wcode) {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  __shared__ float sm[2][16][33];
  const int tx = threadIdx.x & 31, sl = threadIdx.x >> 5;
  const int col = blockIdx.x * 32 + tx;
  float a = 0.f, b = 0.f;
  if (col < N) {
#pragma unroll 16
    for (int p = sl; p < P; p += 16) {
      a += part[int64_t(p) * 2 * N + col];
      if constexpr (!RMS) b += part[int64_t(p) * 2 * N + N + col];
    }
  }
  sm[0][sl][tx] = a;
  sm[1][sl][tx] = b;
  __syncthreads();
  if (sl == 0 && col < N) {
    float ta = 0.f, tb = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      ta += sm[0][i][tx];
      tb += sm[1][i][tx];
    }
    if (wcode == 0) {
      reinterpret_cast<float*>(dw_out)[col] = ta;
      if constexpr (!RMS) reinterpret_cast<float*>(db_out)[col] = tb;
    } else if (wcode == 1) {
      reinterpret_cast<__nv_bfloat16*>(dw_out)[col] = __float2bfloat16(ta);
      if constexpr (!RMS) reinterpret_cast<__nv_bfloat16*>(db_out)[col] = __float2bfloat16(tb);
    } else {
      reinterpret_cast<__half*>(dw_out)[col] = __float2half(ta);
      if constexpr (!RMS) reinterpret_cast<__half*>(db_out)[col] = __float2half(tb);
    }
  }
}
// dw / db = the sum of the P per-CTA partial rows in a fixed order: a block is 32 columns x 32 row slices of ONE of the two (dw, db), so a
// thread has P / 32 independent loads in flight and the blocks are twice as many; it sits on the critical path (it starts when the backward ends:
// 6.2 us after it with 16 slices and both sums a thread at N = 256), converted to the weight dtype
template <bool RMS>
__global__ void __launch_bounds__(1024) pipe_reduce_h100(const float* __restrict__ part, int P, int N, void* __restrict__ dw_out, void* __restrict__ db_out,
                                                    int wcode) {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  __shared__ float sm[32][33];
  const int tx = threadIdx.x & 31, sl = threadIdx.x >> 5;
  const int col = blockIdx.x * 32 + tx;
  const int which = RMS ? 0 : blockIdx.y;  // 0: dw, 1: db
  float a = 0.f;
  if (col < N) {
#pragma unroll 16
    for (int p = sl; p < P; p += 32) a += part[int64_t(p) * 2 * N + which * N + col];
  }
  sm[sl][tx] = a;
  __syncthreads();
  if (sl == 0 && col < N) {
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) t += sm[i][tx];
    void* out = which ? db_out : dw_out;
    if (wcode == 0) reinterpret_cast<float*>(out)[col] = t;
    else if (wcode == 1) reinterpret_cast<__nv_bfloat16*>(out)[col] = __float2bfloat16(t);
    else reinterpret_cast<__half*>(out)[col] = __float2half(t);
  }
}

inline int sm_major() {
  static int cached[64] = {0};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= 64) dev = 0;
  if (!cached[dev]) cudaDeviceGetAttribute(&cached[dev], cudaDevAttrComputeCapabilityMajor, dev);
  return cached[dev];
}

inline int num_sms() {
  static int cached[64] = {0};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= 64) dev = 0;
  if (!cached[dev]) cudaDeviceGetAttribute(&cached[dev], cudaDevAttrMultiProcessorCount, dev);
  return cached[dev];
}

template <typename K> int resident_ctas(K kernel, size_t smem) {
  int n = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, THREADS, smem);
  return n < 1 ? 1 : n;
}

// direct3_fwd with PW row passes a warp: its launch (PDL for small activations, see launch_direct2); grid_out != nullptr: only the CTAs it would take
template <typename T, typename WT, int TPR, int CH, int VEC, bool RMS, int PW, bool STATS>
void launch_d3(const T* x, const WT* w, const WT* b, T* y, float* ws, float2* stats, int64_t M, int N, float eps, cudaStream_t s, int64_t* grid_out) {
  constexpr int RPP = 32 / TPR;
  const int64_t grid = (((M + RPP - 1) / RPP + PW - 1) / PW + MWP_D3W - 1) / MWP_D3W;
  if (grid_out != nullptr) {
    *grid_out = grid;
    return;
  }
  if (MWP_D2PDL && sm_major() >= 9 && M * int64_t(N) * int64_t(sizeof(T)) < MWP_BK_MIN) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(unsigned(grid));
    cfg.blockDim = dim3(MWP_D3W * 32);
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, direct3_fwd<T, WT, TPR, CH, VEC, RMS, PW, MWP_D3W, STATS>, x, w, b, y, ws, stats, M, eps);
    return;
  }
  direct3_fwd<T, WT, TPR, CH, VEC, RMS, PW, MWP_D3W, STATS><<<int(grid), MWP_D3W * 32, 0, s>>>(x, w, b, y, ws, stats, M, eps);
}

// One-wave rule of the direct3 forward: PW row passes a warp (MWP_D3PW) when that grid is one wave of resident CTAs, else MWP_D3PW2: the
// 48-register 2-byte builds hold 10 CTAs an SM, so 2048 CTAs of PW 2 would be two waves with a tail (9.1 us against 7.7 us at 16384 rows).
template <typename T, typename WT, int TPR, int CH, int VEC, bool RMS, bool STATS>
void launch_d3_auto(const T* x, const WT* w, const WT* b, T* y, float* ws, float2* stats, int64_t M, int N, float eps, cudaStream_t s) {
  static int occ = 0;
  if (!occ) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, direct3_fwd<T, WT, TPR, CH, VEC, RMS, MWP_D3PW, MWP_D3W, STATS>, MWP_D3W * 32, 0);
    if (occ < 1) occ = 1;
  }
  int64_t g = 0;
  launch_d3<T, WT, TPR, CH, VEC, RMS, MWP_D3PW, STATS>(x, w, b, y, ws, stats, M, N, eps, s, &g);
  if (g <= int64_t(num_sms()) * occ) launch_d3<T, WT, TPR, CH, VEC, RMS, MWP_D3PW, STATS>(x, w, b, y, ws, stats, M, N, eps, s, nullptr);
  else launch_d3<T, WT, TPR, CH, VEC, RMS, MWP_D3PW2, STATS>(x, w, b, y, ws, stats, M, N, eps, s, nullptr);
}

// The multi-chunk rows (CH >= 2, a warp a row): the smallest of PW 1 / 2 / 4 / 8 rows a warp whose grid is one wave of resident CTAs, else the largest
// allowed, where a lane holds at most MWP_D3CH_PWE values over its rows (PW * CH * VEC: past that the build spills or fits few CTAs: 2x slower at
// x[1, 512, 512, 512] fp32 when the build was held to 64 registers).
template <typename T, typename WT, int CH, int VEC, bool RMS, int PW, bool STATS>
bool d3ch_fits(int64_t M, int N) {
  static int occ = 0;
  if (!occ) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, direct3_fwd<T, WT, 32, CH, VEC, RMS, PW, MWP_D3W, STATS>, MWP_D3W * 32, 0);
    if (occ < 1) occ = 1;
  }
  int64_t g = 0;
  launch_d3<T, WT, 32, CH, VEC, RMS, PW, STATS>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, M, N, 0.f, nullptr, &g);
  return g <= int64_t(num_sms()) * occ;
}
template <typename T, typename WT, int CH, int VEC, bool RMS, bool STATS>
void launch_d3ch(const T* x, const WT* w, const WT* b, T* y, float* ws, float2* stats, int64_t M, int N, float eps, cudaStream_t s) {
  constexpr int E = CH * VEC;
  constexpr int PWMAX = MWP_D3CH_PWE / E >= 8 ? 8 : MWP_D3CH_PWE / E >= 4 ? 4 : MWP_D3CH_PWE / E >= 2 ? 2 : 1;
  if constexpr (PWMAX >= 8) {
    if (!d3ch_fits<T, WT, CH, VEC, RMS, 4, STATS>(M, N) && d3ch_fits<T, WT, CH, VEC, RMS, 8, STATS>(M, N)) {
      launch_d3<T, WT, 32, CH, VEC, RMS, 8, STATS>(x, w, b, y, ws, stats, M, N, eps, s, nullptr);
      return;
    }
  }
  if constexpr (PWMAX >= 4) {
    if (!d3ch_fits<T, WT, CH, VEC, RMS, 2, STATS>(M, N)) {
      launch_d3<T, WT, 32, CH, VEC, RMS, 4, STATS>(x, w, b, y, ws, stats, M, N, eps, s, nullptr);
      return;
    }
  }
  if constexpr (PWMAX >= 2) {
    launch_d3<T, WT, 32, CH, VEC, RMS, 2, STATS>(x, w, b, y, ws, stats, M, N, eps, s, nullptr);
  } else {
    launch_d3<T, WT, 32, CH, VEC, RMS, 1, STATS>(x, w, b, y, ws, stats, M, N, eps, s, nullptr);
  }
}

// The direct2 forward's launch: PDL on sm_90+ for small activations (see MWP_BK_MIN), plain otherwise. WT: the dtype of the affine values.
template <typename T, typename WT, int TPR, int VEC, bool RMS>
void launch_direct2(const T* x, const WT* w, const WT* b, T* y, float* ws, float2* stats, int64_t M, int N, float eps, cudaStream_t s) {
  constexpr int RPP = 32 / TPR;
  if (MWP_D3 && sm_major() == 9 && (stats != nullptr || ws == nullptr)) {  // the lean kernel, inference or training
    if (stats != nullptr) launch_d3_auto<T, WT, TPR, 1, VEC, RMS, true>(x, w, b, y, ws, stats, M, N, eps, s);
    else launch_d3_auto<T, WT, TPR, 1, VEC, RMS, false>(x, w, b, y, ws, stats, M, N, eps, s);
    return;
  }
  constexpr int PW2 = MWP_D2PW;
  const int64_t units2 = ((M + RPP - 1) / RPP + PW2 - 1) / PW2;
  const int64_t grid2 = (units2 + MWP_D2W - 1) / MWP_D2W;
  // programmatic dependent launch only for small activations, where overlapping the predecessor's tail pays: past MWP_BK_MIN every
  // CTA of a launch this large waiting at griddepcontrol.wait cost 0.4-0.8% at the HBM floor (and 1.7x with one-warp CTAs)
  if (MWP_D2PDL && sm_major() >= 9 && M * int64_t(N) * int64_t(sizeof(T)) < MWP_BK_MIN) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(unsigned(grid2));
    cfg.blockDim = dim3(MWP_D2W * 32);
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, direct2_fwd<T, WT, TPR, VEC, RMS, PW2, MWP_D2W>, x, w, b, y, ws, stats, M, N, eps);
    return;
  }
  direct2_fwd<T, WT, TPR, VEC, RMS, PW2, MWP_D2W><<<int(grid2), MWP_D2W * 32, 0, s>>>(x, w, b, y, ws, stats, M, N, eps);
}

template <typename T, int TPR, int CH, int VEC, bool RMS, bool PAD>
void launch_fwd_pad(const T* x, const void* w, const void* b, int wcode, T* y, float* ws, float2* stats, int64_t M, int N, float eps, int per_sm,
                cudaStream_t s) {
  constexpr int RPP = 32 / TPR;
  if constexpr (CH == 1) {
    if (per_sm >= 1000 || (per_sm == 0 && MWP_DIRECT_DEFAULT)) {
#if MWP_D2
      constexpr int WC = std::is_same<T, float>::value ? 0 : (std::is_same<T, __nv_bfloat16>::value ? 1 : 2);
      if constexpr (!PAD) {
#if MWP_BK
        const int64_t bytes = M * int64_t(N) * int64_t(sizeof(T));
        if (wcode == WC && stats == nullptr && ws == nullptr && sm_major() >= 9 && bytes >= MWP_BK_MIN && bytes <= MWP_BK_MAX) {
          auto k = bulk_fwd<T, TPR, VEC, RMS>;
          constexpr int smem = MWP_BK_ST * MWP_BK_TB;
          static int occ = 0;
          if (!occ) {
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, MWP_BK_W * 32, smem);
            if (occ < 1) occ = 1;
          }
          constexpr int64_t R = MWP_BK_TB / (TPR * VEC * int64_t(sizeof(T)));
          const int64_t tiles = (M + R - 1) / R;
          const int64_t cap = int64_t(num_sms()) * occ;
          k<<<int(tiles < cap ? tiles : cap), MWP_BK_W * 32, smem, s>>>(x, static_cast<const T*>(w), static_cast<const T*>(b), y, M, N, eps);
          return;
        }
#endif
        if (wcode == WC) {
          launch_direct2<T, T, TPR, VEC, RMS>(x, static_cast<const T*>(w), static_cast<const T*>(b), y, ws, stats, M, N, eps, s);
          return;
        }
        if constexpr (WC != 0) {  // fp32 affine values on 2-byte activations (master weights)
          if (wcode == 0) {
            launch_direct2<T, float, TPR, VEC, RMS>(x, static_cast<const float*>(w), static_cast<const float*>(b), y, ws, stats, M, N, eps, s);
            return;
          }
        }
      }
#endif
      constexpr int PW = sizeof(T) == 4 ? 1 : 2;
      const int64_t units = ((M + RPP - 1) / RPP + PW - 1) / PW;
      const int64_t grid = (units + WARPS - 1) / WARPS;
      direct_fwd<T, TPR, VEC, RMS, PW, PAD><<<int(grid), THREADS, 0, s>>>(x, w, b, wcode, y, ws, stats, M, N, eps);
      return;
    }
  }
#if MWP_DCH
  if constexpr (CH > 1 && TPR == 32 && !PAD) {  // the lean multi-chunk forward (direct3_fwd with CH chunks a lane, weights loaded once a warp), inference and training
    if (MWP_D3CH && CH * VEC <= MWP_D3CH_EMAX && sm_major() == 9 && (stats != nullptr || ws == nullptr) && per_sm == 0 && M * int64_t(N) * int64_t(sizeof(T)) >= MWP_D3CH_MIN) {
      constexpr int WC = std::is_same<T, float>::value ? 0 : (std::is_same<T, __nv_bfloat16>::value ? 1 : 2);
      const bool st = stats != nullptr;
      if (wcode == 0) {
        if (st) launch_d3ch<T, float, CH, VEC, RMS, true>(x, static_cast<const float*>(w), static_cast<const float*>(b), y, ws, stats, M, N, eps, s);
        else launch_d3ch<T, float, CH, VEC, RMS, false>(x, static_cast<const float*>(w), static_cast<const float*>(b), y, ws, stats, M, N, eps, s);
        return;
      }
      if constexpr (WC != 0) {
        if (wcode == WC) {
          if (st) launch_d3ch<T, T, CH, VEC, RMS, true>(x, static_cast<const T*>(w), static_cast<const T*>(b), y, ws, stats, M, N, eps, s);
          else launch_d3ch<T, T, CH, VEC, RMS, false>(x, static_cast<const T*>(w), static_cast<const T*>(b), y, ws, stats, M, N, eps, s);
          return;
        }
      }
    }
  }
  if constexpr (CH > 1 && TPR == 32) {
    if ((MWP_DCH_TRAIN || (stats == nullptr && ws == nullptr)) && per_sm == 0 && M * int64_t(N) * int64_t(sizeof(T)) >= MWP_DCH_MIN) {  // large: the direct multi-chunk forward (training too: it writes the statistics)
      constexpr int WC = std::is_same<T, float>::value ? 0 : (std::is_same<T, __nv_bfloat16>::value ? 1 : 2);
      const int g = int((M + MWP_DCHW - 1) / MWP_DCHW);
      if (wcode == 0) {
        directch_fwd<T, float, CH, VEC, RMS, PAD><<<g, MWP_DCHW * 32, 0, s>>>(x, static_cast<const float*>(w), static_cast<const float*>(b), y, ws, stats, M, N, eps);
        return;
      }
      if constexpr (WC != 0) {
        if (wcode == WC) {
          directch_fwd<T, T, CH, VEC, RMS, PAD><<<g, MWP_DCHW * 32, 0, s>>>(x, static_cast<const T*>(w), static_cast<const T*>(b), y, ws, stats, M, N, eps);
          return;
        }
      }
    }
  }
#endif
  constexpr size_t smem = size_t(WARPS) * stages<CH>() * CH * 32 * 16;
  auto k = pipe_fwd<T, TPR, CH, VEC, RMS, PAD>;
  if (smem > 48 * 1024) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  static int occ = 0;
  if (!occ) occ = resident_ctas(k, smem);
  const int64_t cap = int64_t(num_sms()) * (per_sm > 0 && per_sm < occ ? per_sm : occ);
  const int64_t need = ((M + RPP - 1) / RPP + WARPS - 1) / WARPS;
  const int grid = int(need < cap ? need : cap);
  k<<<grid > 0 ? grid : 1, THREADS, smem, s>>>(x, w, b, wcode, y, ws, stats, M, N, eps);
}

template <typename T, int TPR, int CH, int VEC, bool RMS>
void launch_fwd(const T* x, const void* w, const void* b, int wcode, T* y, float* ws, float2* stats, int64_t M, int N, float eps, int per_sm,
                cudaStream_t s) {
  if (TPR * CH * VEC == N) launch_fwd_pad<T, TPR, CH, VEC, RMS, false>(x, w, b, wcode, y, ws, stats, M, N, eps, per_sm, s);
  else launch_fwd_pad<T, TPR, CH, VEC, RMS, true>(x, w, b, wcode, y, ws, stats, M, N, eps, per_sm, s);
}

template <typename T, int TPR, int CH, int VEC, bool RMS, bool WG>
void launch_bwd_wg(const T* dy, const T* x, const void* w, const float2* stats, T* dx, float* ws, void* dw, void* db, int wcode, int64_t M, int N,
                   int per_sm, cudaStream_t s) {
  constexpr int RPP = 32 / TPR;
  constexpr size_t smem = size_t(WARPS) * stages<CH>() * (CH * 32 * 16 * 2 + 32 * 8);
  auto k = pipe_bwd<T, TPR, CH, VEC, RMS, WG>;
  if (smem > 48 * 1024) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  static int occ = 0;
  if (!occ) occ = resident_ctas(k, smem);
  if (per_sm == 0) per_sm = CH == 1 ? MWP_BWD_CTAS1 : MWP_BWD_CTAS;  // fewer CTAs per SM end the backward sooner (less contention on the partial rows and the tail)
  const int64_t cap = int64_t(num_sms()) * (per_sm > 0 && per_sm < occ ? per_sm : occ);
  const int64_t need = ((M + RPP - 1) / RPP + WARPS - 1) / WARPS;
  const int grid = int(need < cap ? need : cap);
  k<<<grid > 0 ? grid : 1, THREADS, smem, s>>>(dy, x, w, stats, dx, ws, dw, db, wcode, M, N);
#ifndef MWP_NOREDUCE
  if constexpr (WG) {
    const bool h100 = sm_major() == 9 && MWP_RED_H100 && grid <= MWP_RED_H100_PMAX;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = h100 ? dim3((N + 31) / 32, RMS ? 1 : 2) : dim3((N + 31) / 32);
    cfg.blockDim = dim3(h100 ? 1024 : 512);
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (h100) cudaLaunchKernelEx(&cfg, pipe_reduce_h100<RMS>, static_cast<const float*>(ws), grid > 0 ? grid : 1, N, dw, db, wcode);
    else cudaLaunchKernelEx(&cfg, pipe_reduce<RMS>, static_cast<const float*>(ws), grid > 0 ? grid : 1, N, dw, db, wcode);
  }
#endif
}

template <typename T, int TPR, int CH, int VEC, bool RMS>
void launch_bwd(const T* dy, const T* x, const void* w, const float2* stats, T* dx, float* ws, void* dw, void* db, int wcode, int64_t M, int N,
                int per_sm, cudaStream_t s) {
  if (ws != nullptr) launch_bwd_wg<T, TPR, CH, VEC, RMS, true>(dy, x, w, stats, dx, ws, dw, db, wcode, M, N, per_sm, s);
  else launch_bwd_wg<T, TPR, CH, VEC, RMS, false>(dy, x, w, stats, dx, ws, dw, db, wcode, M, N, per_sm, s);
}

// (tpr, ch) from the number of 16-byte chunks per row; false: no build (rows under 9 chunks or over 64 chunks)
inline bool pick(int nchunk, int& tpr, int& ch) {
  if (nchunk < 9 || nchunk > 128) return false;
  if (nchunk <= 16) { tpr = 16; ch = 1; return true; }
  tpr = 32;
  ch = nchunk <= 32 ? 1 : nchunk <= 64 ? 2 : nchunk <= 96 ? 3 : 4;
  return true;
}

#ifdef MWP_PROBE
#define MWP_FOR_CONFIG(M_, ...) M_(16, 1, __VA_ARGS__) M_(32, 1, __VA_ARGS__) M_(32, 2, __VA_ARGS__) M_(32, 3, __VA_ARGS__)
#else
#define MWP_FOR_CONFIG(M_, ...) M_(16, 1, __VA_ARGS__) M_(32, 1, __VA_ARGS__) M_(32, 2, __VA_ARGS__) M_(32, 3, __VA_ARGS__) M_(32, 4, __VA_ARGS__)
#endif

template <typename T, int VEC, bool RMS>
bool dispatch_fwd(int tpr, int ch, const T* x, const void* w, const void* b, int wcode, T* y, float* ws, float2* stats, int64_t M, int N,
                  float eps, int gm, cudaStream_t s) {
  if (tpr == 8) {  // 128-byte rows (bf16 / fp16 64, fp32 32 values): the inference forward only, eight lanes a row (the backward and the ring have no such build)
    constexpr int WC = std::is_same<T, float>::value ? 0 : (std::is_same<T, __nv_bfloat16>::value ? 1 : 2);
    if (stats != nullptr || ws != nullptr) return false;
    if (wcode == WC) {
      launch_direct2<T, T, 8, VEC, RMS>(x, static_cast<const T*>(w), static_cast<const T*>(b), y, ws, stats, M, N, eps, s);
      return true;
    }
    if constexpr (WC != 0) {
      if (wcode == 0) {
        launch_direct2<T, float, 8, VEC, RMS>(x, static_cast<const float*>(w), static_cast<const float*>(b), y, ws, stats, M, N, eps, s);
        return true;
      }
    }
    return false;
  }
#define MWP_CASE(TPR_, CH_, ...)                                                                                         \
  if (tpr == TPR_ && ch == CH_) {                                                                                       \
    launch_fwd<T, TPR_, CH_, VEC, RMS>(x, w, b, wcode, y, ws, stats, M, N, eps, gm, s);                                 \
    return true;                                                                                                        \
  }
  MWP_FOR_CONFIG(MWP_CASE, 0)
#undef MWP_CASE
  return false;
}

template <typename T, int VEC, bool RMS>
bool dispatch_bwd(int tpr, int ch, const T* dy, const T* x, const void* w, const float2* stats, T* dx, float* ws, void* dw, void* db,
                  int wcode, int64_t M, int N, int gm, cudaStream_t s) {
#define MWP_CASE(TPR_, CH_, ...)                                                                                         \
  if (tpr == TPR_ && ch == CH_) {                                                                                       \
    launch_bwd<T, TPR_, CH_, VEC, RMS>(dy, x, w, stats, dx, ws, dw, db, wcode, M, N, gm, s);                          \
    return true;                                                                                                        \
  }
  MWP_FOR_CONFIG(MWP_CASE, 0)
#undef MWP_CASE
  return false;
}

}  // namespace mwpipe

// x [M, N] contiguous, rows 16-byte aligned; w, b [N] (bf16 / fp16 / fp32, 16-byte aligned) or empty. Returns y; `ws` (fp32 [1]) only flags
// that the backward wants affine gradients (the kernel zeroes it when given) and `stats` (fp32 [M, 2]: mean, rstd) written when given: the training forward hands both to the backward.
torch::Tensor pipe_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool rms, torch::Tensor ws, torch::Tensor stats,
                           int64_t per_sm) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be contiguous CUDA [M, N]");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  const int esz = int(x.element_size());
  TORCH_CHECK((int64_t(N) * esz) % 16 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "rows must be 16-byte aligned");
  TORCH_CHECK(!rms || b.numel() == 0, "RMSNorm has no bias");
  if (w.numel() && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16) w = w.clone();
  if (b.numel() && reinterpret_cast<uintptr_t>(b.data_ptr()) % 16) b = b.clone();
  auto y = torch::empty_like(x);
  if (M == 0) return y;
  int tpr = 0, ch = 0;
  if (N * esz == 128) {  // 128-byte rows: the eight-lane direct forward (inference only, see dispatch_fwd)
    tpr = 8, ch = 1;
    TORCH_CHECK(ws.numel() == 0 && stats.numel() == 0, "128-byte rows: inference forward only");
  } else {
    TORCH_CHECK(mwpipe::pick(N * esz / 16, tpr, ch), "width ", N, " is beyond the pipelined kernels");
  }
  const auto wt = w.numel() ? w.scalar_type() : b.scalar_type();
  const int wcode = w.numel() || b.numel() ? (wt == torch::kFloat32 ? 0 : wt == torch::kBFloat16 ? 1 : 2) : 0;
  TORCH_CHECK(!(w.numel() && b.numel()) || w.scalar_type() == b.scalar_type(), "weight and bias must share a dtype");
  const void* wp = w.numel() ? w.data_ptr() : nullptr;
  const void* bp = b.numel() ? b.data_ptr() : nullptr;
  float* wsp = ws.numel() ? ws.data_ptr<float>() : nullptr;
  float2* stp = stats.numel() ? reinterpret_cast<float2*>(stats.data_ptr<float>()) : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
#define MWP_GO(CT, V)                                                                                                    \
  (rms ? mwpipe::dispatch_fwd<CT, V, true>(tpr, ch, xp, wp, bp, wcode, yp, wsp, stp, M, N, float(eps), int(per_sm), s) \
       : mwpipe::dispatch_fwd<CT, V, false>(tpr, ch, xp, wp, bp, wcode, yp, wsp, stp, M, N, float(eps), int(per_sm), s))
#define MWP_DT(TORCH_T, CT, V)                           \
  if (x.scalar_type() == TORCH_T) {                      \
    auto xp = reinterpret_cast<const CT*>(x.data_ptr()); \
    auto yp = reinterpret_cast<CT*>(y.data_ptr());       \
    ok = MWP_GO(CT, V);                                  \
  }
  MWP_DT(torch::kBFloat16, __nv_bfloat16, 8)
#ifndef MWP_PROBE
  MWP_DT(torch::kHalf, __half, 8)
#endif
  MWP_DT(torch::kFloat32, float, 4)
#undef MWP_DT
#undef MWP_GO
  TORCH_CHECK(ok, "pipelined LayerNorm forward: unsupported dtype or config");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// dw / db come back in `wcode`'s dtype (empty when `ws` is empty: no affine gradients); `stats` and `ws` are the training forward's.
std::vector<torch::Tensor> pipe_backward(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor stats, torch::Tensor ws, bool rms,
                                         int64_t wcode, int64_t per_sm) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && dy.is_contiguous() && dy.sizes() == x.sizes() &&
                  dy.scalar_type() == x.scalar_type(),
              "dy, x must be contiguous CUDA [M, N] of one dtype");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  const int esz = int(x.element_size());
  TORCH_CHECK((int64_t(N) * esz) % 16 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(dy.data_ptr()) % 16 == 0,
              "rows must be 16-byte aligned");
  TORCH_CHECK(stats.numel() == 2 * M, "stats must be the forward's [M, 2]");
  if (w.numel() && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16) w = w.clone();
  auto dx = torch::empty_like(x);
  const bool wgrad = ws.numel() > 0;
  torch::Dtype wt = wcode == 0 ? torch::kFloat32 : wcode == 1 ? torch::kBFloat16 : torch::kHalf;
  auto wopt = x.options().dtype(wt);
  auto dw = wgrad ? torch::empty({N}, wopt) : torch::empty({0}, wopt);
  auto db = wgrad && !rms ? torch::empty({N}, wopt) : torch::empty({0}, wopt);
  if (M == 0) {
    if (wgrad) {
      dw.zero_();
      if (!rms) db.zero_();
    }
    return {dx, dw, db};
  }
  int tpr = 0, ch = 0;
  TORCH_CHECK(mwpipe::pick(N * esz / 16, tpr, ch), "width ", N, " is beyond the pipelined kernels");
  const void* wp = w.numel() ? w.data_ptr() : nullptr;
  // one partial row per CTA (at most 32 CTAs per SM), written by the kernel and reduced by pipe_reduce
  auto part = wgrad ? torch::empty({int64_t(mwpipe::num_sms()) * 32, 2 * int64_t(N)}, x.options().dtype(torch::kFloat32)) : torch::empty({0}, x.options().dtype(torch::kFloat32));
  float* wsp = wgrad ? part.data_ptr<float>() : nullptr;
  const float2* stp = reinterpret_cast<const float2*>(stats.data_ptr<float>());
  void* dwp = wgrad ? dw.data_ptr() : nullptr;
  void* dbp = wgrad && !rms ? db.data_ptr() : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
#define MWP_GO(CT, V)                                                                                                      \
  (rms ? mwpipe::dispatch_bwd<CT, V, true>(tpr, ch, a, xp, wp, stp, o, wsp, dwp, dbp, int(wcode), M, N, int(per_sm), s)  \
       : mwpipe::dispatch_bwd<CT, V, false>(tpr, ch, a, xp, wp, stp, o, wsp, dwp, dbp, int(wcode), M, N, int(per_sm), s))
#define MWP_DT(TORCH_T, CT, V)                           \
  if (x.scalar_type() == TORCH_T) {                      \
    auto a = reinterpret_cast<const CT*>(dy.data_ptr()); \
    auto xp = reinterpret_cast<const CT*>(x.data_ptr()); \
    auto o = reinterpret_cast<CT*>(dx.data_ptr());       \
    ok = MWP_GO(CT, V);                                  \
  }
  MWP_DT(torch::kBFloat16, __nv_bfloat16, 8)
#ifndef MWP_PROBE
  MWP_DT(torch::kHalf, __half, 8)
#endif
  MWP_DT(torch::kFloat32, float, 4)
#undef MWP_DT
#undef MWP_GO
  TORCH_CHECK(ok, "pipelined LayerNorm backward: unsupported dtype or config");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dw, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &pipe_forward, "pipelined LayerNorm / RMSNorm forward");
  m.def("bwd", &pipe_backward, "pipelined LayerNorm / RMSNorm backward");
}
