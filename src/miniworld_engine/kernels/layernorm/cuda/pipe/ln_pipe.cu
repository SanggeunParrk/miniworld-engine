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
//           shared memory and added into a zeroed fp32 workspace with one atomic per column per CTA; the last CTA to finish converts the sums to
//           the weight dtype, writes dw / db and clears the workspace for the next call. One kernel, no reduce launch, no cast. The atomics
//           add in an unspecified order (fp32 rounding differs from replay to replay; the general kernels in ../general are bit-reproducible
//           and stay the choice for a caller that needs that).
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
template <typename T, int TPR, int VEC, bool RMS, int PW, int DW>
__global__ void __launch_bounds__(DW * 32) direct2_fwd(const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ b, T* __restrict__ y,
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
    raw[p] = row < M ? __ldg(reinterpret_cast<const uint4*>(x + row * N + l * VEC)) : make_uint4(0, 0, 0, 0);
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
#define MWP_BK_MIN (int64_t(8) << 20)
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
          constexpr int PW2 = MWP_D2PW;
          const int64_t units2 = ((M + RPP - 1) / RPP + PW2 - 1) / PW2;
          const int64_t grid2 = (units2 + MWP_D2W - 1) / MWP_D2W;
          if (sm_major() >= 9) {
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3(unsigned(grid2));
            cfg.blockDim = dim3(MWP_D2W * 32);
            cfg.stream = s;
            cudaLaunchAttribute at[1];
            at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            at[0].val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = at;
            cfg.numAttrs = 1;
            cudaLaunchKernelEx(&cfg, direct2_fwd<T, TPR, VEC, RMS, PW2, MWP_D2W>, x, static_cast<const T*>(w), static_cast<const T*>(b), y, ws, stats, M,
                               N, eps);
            return;
          }
          direct2_fwd<T, TPR, VEC, RMS, PW2, MWP_D2W><<<int(grid2), MWP_D2W * 32, 0, s>>>(x, static_cast<const T*>(w), static_cast<const T*>(b), y, ws, stats, M, N, eps);
          return;
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
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((N + 31) / 32);
    cfg.blockDim = dim3(512);
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, pipe_reduce<RMS>, static_cast<const float*>(ws), grid > 0 ? grid : 1, N, dw, db, wcode);
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
#define MWP_FOR_CONFIG(M_, ...) M_(16, 1, __VA_ARGS__) M_(32, 1, __VA_ARGS__) M_(32, 3, __VA_ARGS__)
#else
#define MWP_FOR_CONFIG(M_, ...) M_(16, 1, __VA_ARGS__) M_(32, 1, __VA_ARGS__) M_(32, 2, __VA_ARGS__) M_(32, 3, __VA_ARGS__) M_(32, 4, __VA_ARGS__)
#endif

template <typename T, int VEC, bool RMS>
bool dispatch_fwd(int tpr, int ch, const T* x, const void* w, const void* b, int wcode, T* y, float* ws, float2* stats, int64_t M, int N,
                  float eps, int gm, cudaStream_t s) {
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
    launch_bwd<T, TPR_, CH_, VEC, RMS>(dy, x, w, stats, dx, ws, dw, db, wcode, M, N, gm, s);                            \
    return true;                                                                                                        \
  }
  MWP_FOR_CONFIG(MWP_CASE, 0)
#undef MWP_CASE
  return false;
}

}  // namespace mwpipe

// x [M, N] contiguous, rows 16-byte aligned; w, b [N] (bf16 / fp16 / fp32, 16-byte aligned) or empty. Returns y; `ws` (fp32 [2N + 1]) is
// zeroed by the kernel when given and `stats` (fp32 [M, 2]: mean, rstd) written when given: the training forward hands both to the backward.
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
  TORCH_CHECK(mwpipe::pick(N * esz / 16, tpr, ch), "width ", N, " is beyond the pipelined kernels");
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
