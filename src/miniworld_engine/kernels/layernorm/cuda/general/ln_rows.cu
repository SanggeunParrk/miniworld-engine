// H100 LayerNorm for rows whose width is not a 16-byte multiple (N * sizeof(T) % 16 != 0: 267 / 451 / 651 / 831 / 833 ...), forward and
// backward, fp32 statistics. Such rows are not 16-byte aligned, so no thread can read its share of one row as vectors; the general
// kernels then run one scalar element per thread with 128 threads per row (half the lanes idle at these widths, CTA barriers per
// reduction). Measured on H100 (graph timing, L2 flushed) against cuEquivariance's layer_norm_transpose and the af3_torch kit's Triton
// LayerNorm, the kernels here are:
//
// forward, 2-byte activations, from STAGE_MIN_ROWS rows: ln_stage_fwd_kernel. U consecutive rows are a whole number of 16-byte chunks;
//           a 2-warp CTA copies a ~16 KB tile through shared memory with 16-byte cp.async / stores and normalises it there (the
//           element kernel is issue bound at 2-byte accesses: ncu 87% SM throughput at 51% DRAM; 267 bf16 L768: 248 -> 220 us).
// forward, up to SMALL_MAX_ROWS rows: ln_small_fwd_kernel, a 128-thread CTA per row (latency bound token streams).
// forward, otherwise (fp32 pairs, tails): ln_rw_fwd_kernel, a warp per row (lane l on columns l, l + 32, ...: coalesced), the
//           statistics warp shuffles only, the affine in registers loaded once.
// backward: ln_rw_bwd_kernel, persistent CTAs grid-striding over the rows, a warp per row (narrow rows prefetch the next one); dx per
//           row; dw = sum dy * xhat, db = sum dy per lane in registers, summed over the CTA's warps in shared memory in a fixed order into
//           one partial row per CTA; ln_rw_bwd_col_reduce adds the partial rows in a fixed order: a replay is bit-identical, no atomics.
#include "ln_common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace mwln {

constexpr int RW_WARPS = 8;
constexpr int RW_THREADS = RW_WARPS * 32;

template <typename T, int EPL, int PW>
__global__ void __launch_bounds__(RW_THREADS) ln_rw_fwd_kernel(const T* __restrict__ x, const float* __restrict__ w,
                                                              const float* __restrict__ b, T* __restrict__ y,
                                                              float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t M,
                                                              int N, float eps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t row0 = (int64_t(blockIdx.x) * RW_WARPS + warp) * PW;
  T raw[PW][EPL];
#pragma unroll
  for (int p = 0; p < PW; ++p) {
    const int64_t row = row0 + p;
    const T* xr = x + row * int64_t(N);
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const int col = k * 32 + lane;
      raw[p][k] = (row < M && col < N) ? xr[col] : from_f<T>(0.f);
    }
  }
  float wr[EPL], br[EPL];  // the affine, loaded once up front (a load next to each store would serialise on its latency)
#pragma unroll
  for (int k = 0; k < EPL; ++k) {
    const int col = k * 32 + lane;
    wr[k] = (w != nullptr && col < N) ? __ldg(w + col) : 1.f;
    br[k] = (b != nullptr && col < N) ? __ldg(b + col) : 0.f;
  }
  const float inv_n = 1.f / float(N);
#pragma unroll
  for (int p = 0; p < PW; ++p) {
    const int64_t row = row0 + p;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < EPL; ++k) s += to_f<T>(raw[p][k]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s * inv_n;
    float q = 0.f;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const float d = k * 32 + lane < N ? to_f<T>(raw[p][k]) - mean : 0.f;
      q = fmaf(d, d, q);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rstd = rsqrtf(q * inv_n + eps);
    if (row < M) {
      T* yr = y + row * int64_t(N);
#pragma unroll
      for (int k = 0; k < EPL; ++k) {
        const int col = k * 32 + lane;
        if (col < N) {
          yr[col] = from_f<T>((to_f<T>(raw[p][k]) - mean) * rstd * wr[k] + br[k]);
        }
      }
      if (lane == 0 && mean_out != nullptr) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
      }
    }
  }
}

// one row of x and dy (zeros past N, or past M)
template <typename T, int EPL>
__device__ __forceinline__ void rw_load_row(const T* __restrict__ x, const T* __restrict__ dy, int64_t r, int64_t M, int N, int lane, T (&xo)[EPL],
                                            T (&dout)[EPL]) {
  const int64_t o = r * int64_t(N);
#pragma unroll
  for (int k = 0; k < EPL; ++k) {
    const int col = k * 32 + lane;
    const bool ok = r < M && col < N;
    xo[k] = ok ? x[o + col] : from_f<T>(0.f);
    dout[k] = ok ? dy[o + col] : from_f<T>(0.f);
  }
}

template <typename T, int EPL>
__global__ void __launch_bounds__(RW_THREADS) ln_rw_bwd_kernel(const T* __restrict__ dy, const T* __restrict__ x,
                                                              const float* __restrict__ w, const float* __restrict__ mean,
                                                              const float* __restrict__ rstd, T* __restrict__ dx,
                                                              float* __restrict__ dw_part, float* __restrict__ db_part, int64_t M,
                                                              int N) {
  __shared__ float red[RW_WARPS][32 + 1];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool wgrad = dw_part != nullptr;
  const float inv_n = 1.f / float(N);
  float aw[EPL], ab[EPL], wr[EPL];
#pragma unroll
  for (int k = 0; k < EPL; ++k) {
    const int col = k * 32 + lane;
    wr[k] = (w != nullptr && col < N) ? __ldg(w + col) : 1.f;
    aw[k] = ab[k] = 0.f;
  }
  // narrow rows (PF): the warp loads its next row while it works on this one (a persistent warp has nothing else in flight)
  constexpr bool PF = EPL <= 16;
  const int64_t rstep = int64_t(gridDim.x) * RW_WARPS;
  T xn[EPL], dn[EPL];
  if constexpr (PF) rw_load_row<T, EPL>(x, dy, int64_t(blockIdx.x) * RW_WARPS + warp, M, N, lane, xn, dn);
  for (int64_t row = int64_t(blockIdx.x) * RW_WARPS + warp; row < M; row += rstep) {
    const int64_t off = row * int64_t(N);
    T xr[EPL], dr[EPL];
    if constexpr (PF) {
#pragma unroll
      for (int k = 0; k < EPL; ++k) {
        xr[k] = xn[k];
        dr[k] = dn[k];
      }
      rw_load_row<T, EPL>(x, dy, row + rstep, M, N, lane, xn, dn);
    } else {
      rw_load_row<T, EPL>(x, dy, row, M, N, lane, xr, dr);
    }
    const float mu = mean[row], rs = rstd[row];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const int col = k * 32 + lane;
      const float wv = wr[k];
      const float xh = col < N ? (to_f<T>(xr[k]) - mu) * rs : 0.f;
      const float dv = to_f<T>(dr[k]);
      const float g = dv * wv;
      s1 += g;
      s2 = fmaf(g, xh, s2);
      aw[k] = fmaf(dv, xh, aw[k]);
      ab[k] += dv;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      s1 += __shfl_xor_sync(0xffffffffu, s1, o);
      s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    }
    s1 *= inv_n;
    s2 *= inv_n;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const int col = k * 32 + lane;
      if (col < N) {
        const float xh = (to_f<T>(xr[k]) - mu) * rs;
        dx[off + col] = from_f<T>(rs * (to_f<T>(dr[k]) * wr[k] - s1 - xh * s2));
      }
    }
  }
  if (!wgrad) return;
  // one partial row per CTA: the warps' column partials summed in a fixed order (warp 0, 1, ...)
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    float* out = (pass == 0 ? dw_part : db_part) + int64_t(blockIdx.x) * N;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      red[warp][lane] = pass == 0 ? aw[k] : ab[k];
      __syncthreads();
      if (warp == 0) {
        float t = 0.f;
#pragma unroll
        for (int wi = 0; wi < RW_WARPS; ++wi) t += red[wi][lane];
        const int col = k * 32 + lane;
        if (col < N) out[col] = t;
      }
      __syncthreads();
    }
  }
}

// Few rows (a token stream: 128 - 768 rows of 831 / 833): the call is latency bound, so a row is spread over a CTA of SM_THREADS threads
// (PT values each: thread t on columns t, t + SM_THREADS, ...; x, w and b all loaded up front), the sum and the centred sum of squares
// each a warp shuffle tree plus one pass over the warps' partials in shared memory (fixed order).
constexpr int SM_THREADS = 128;
template <int NWARP>
__device__ __forceinline__ float cta_sum(float v, float* red, int lane, int warp) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < NWARP; ++i) t += red[i];
  return t;
}
template <typename T, int PT>
__global__ void __launch_bounds__(SM_THREADS) ln_small_fwd_kernel(const T* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                                                 T* __restrict__ y, float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                                                 int N, float eps) {
  constexpr int NWARP = SM_THREADS / 32;
  __shared__ float red[2][NWARP];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int64_t row = blockIdx.x;
  const T* xr = x + row * int64_t(N);
  float v[PT], wr[PT], br[PT];
#pragma unroll
  for (int k = 0; k < PT; ++k) {
    const int col = k * SM_THREADS + tid;
    v[k] = col < N ? to_f<T>(xr[col]) : 0.f;
    wr[k] = (w != nullptr && col < N) ? __ldg(w + col) : 1.f;
    br[k] = (b != nullptr && col < N) ? __ldg(b + col) : 0.f;
  }
  const float inv_n = 1.f / float(N);
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < PT; ++k) s += v[k];
  const float mean = cta_sum<NWARP>(s, red[0], lane, warp) * inv_n;
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < PT; ++k) {
    const float d = k * SM_THREADS + tid < N ? v[k] - mean : 0.f;
    q = fmaf(d, d, q);
  }
  const float rstd = rsqrtf(cta_sum<NWARP>(q, red[1], lane, warp) * inv_n + eps);
  T* yr = y + row * int64_t(N);
#pragma unroll
  for (int k = 0; k < PT; ++k) {
    const int col = k * SM_THREADS + tid;
    if (col < N) yr[col] = from_f<T>((v[k] - mean) * rstd * wr[k] + br[k]);
  }
  if (tid == 0 && mean_out != nullptr) {
    mean_out[row] = mean;
    rstd_out[row] = rstd;
  }
}

// Staged forward (rows that are not a 16-byte multiple, many rows): U consecutive rows (U = 16 / gcd(16, row bytes): 8 bf16 rows of
// 267) are a whole number of 16-byte chunks, 16-byte aligned. A CTA of NW warps copies a tile of R rows (a multiple of U) into shared
// memory with 16-byte cp.async, its warps normalise the rows there in place (lane l on columns l + 32 k: conflict-free) and the CTA
// copies the tile back out with 16-byte stores: every global access is a full vector, which the element-wise kernel's 2-byte accesses
// are not. Small CTAs and small tiles (many CTAs resident per SM) overlap one CTA's copy with another's arithmetic. The tail rows
// (M mod U) are the element kernel's.
template <typename T, int EPL, int NW>
__global__ void __launch_bounds__(NW * 32) ln_stage_fwd_kernel(const T* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                                              T* __restrict__ y, float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                                              int64_t Mfull, int N, int R, float eps) {
  extern __shared__ __align__(16) unsigned char stg[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int64_t rowb = int64_t(N) * sizeof(T);
  const int64_t row0 = int64_t(blockIdx.x) * R;
  const int nr = int(Mfull - row0 < R ? Mfull - row0 : R);
  const int nchunks = int(nr * rowb / 16);
  const unsigned char* gx = reinterpret_cast<const unsigned char*>(x) + row0 * rowb;
  const unsigned s0 = static_cast<unsigned>(__cvta_generic_to_shared(stg));
  for (int c = tid; c < nchunks; c += NW * 32)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s0 + 16u * c), "l"(gx + 16 * int64_t(c)) : "memory");
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  float wr[EPL], br[EPL];
#pragma unroll
  for (int k = 0; k < EPL; ++k) {
    const int col = k * 32 + lane;
    wr[k] = (w != nullptr && col < N) ? __ldg(w + col) : 1.f;
    br[k] = (b != nullptr && col < N) ? __ldg(b + col) : 0.f;
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  const float inv_n = 1.f / float(N);
  for (int r = warp; r < nr; r += NW) {
    T* xs = reinterpret_cast<T*>(stg + r * rowb);
    float v[EPL];
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const int col = k * 32 + lane;
      v[k] = col < N ? to_f<T>(xs[col]) : 0.f;
      s += v[k];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s * inv_n;
    float q = 0.f;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const float d = k * 32 + lane < N ? v[k] - mean : 0.f;
      q = fmaf(d, d, q);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rstd = rsqrtf(q * inv_n + eps);
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const int col = k * 32 + lane;
      if (col < N) xs[col] = from_f<T>((v[k] - mean) * rstd * wr[k] + br[k]);
    }
    if (lane == 0 && mean_out != nullptr) {
      mean_out[row0 + r] = mean;
      rstd_out[row0 + r] = rstd;
    }
  }
  __syncthreads();
  uint4* gy = reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(y) + row0 * rowb);
  const uint4* sy = reinterpret_cast<const uint4*>(stg);
  for (int c = tid; c < nchunks; c += NW * 32) gy[c] = sy[c];
}

__global__ void __launch_bounds__(256) ln_rw_bwd_col_reduce(const float* __restrict__ part, int P, int N, float* __restrict__ out) {
  __shared__ float sm[8][33];
  const int tx = threadIdx.x & 31, sl = threadIdx.x >> 5;
  const int col = blockIdx.x * 32 + tx;
  float s = 0.f;
  if (col < N)
    for (int p = sl; p < P; p += 8) s += part[int64_t(p) * N + col];
  sm[sl][tx] = s;
  __syncthreads();
  if (sl == 0 && col < N) {
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) t += sm[i][tx];
    out[col] = t;
  }
}

}  // namespace mwln

// the values per lane the kernels are built for: exact fits of the requirement widths (9: 267, 15: 451, 21: 651, 27: 831 / 833; an idle
// slot is wasted arithmetic: the 2-byte kernels are issue bound) and multiples of 4 for any other width up to 1024
#define MWLN_RW_EPLS(M_, ...) M_(4, __VA_ARGS__) M_(8, __VA_ARGS__) M_(9, __VA_ARGS__) M_(12, __VA_ARGS__) M_(15, __VA_ARGS__) \
  M_(16, __VA_ARGS__) M_(20, __VA_ARGS__) M_(21, __VA_ARGS__) M_(24, __VA_ARGS__) M_(27, __VA_ARGS__) M_(28, __VA_ARGS__) M_(32, __VA_ARGS__)

static int rw_epl(int N) {
  const int need = (N + 31) / 32;
  if (need == 9 || need == 15 || need == 21 || need == 27) return need;
  const int e = ((need + 3) / 4) * 4;
  return e <= 32 ? e : -1;
}

static int gcd_int(int a, int b) {
  while (b) {
    const int t = a % b;
    a = b;
    b = t;
  }
  return a;
}

template <typename CT>
static void launch_rw_fwd(int epl, const CT* xp, const float* wp, const float* bp, CT* yp, float* mp, float* rp, int64_t M, int N, float eps,
                          cudaStream_t s) {
  if (M <= 0) return;
  const int64_t grid = (M + mwln::RW_WARPS - 1) / mwln::RW_WARPS;
#define MWLN_RW_F(E, ...) \
  if (epl == E) mwln::ln_rw_fwd_kernel<CT, E, 1><<<unsigned(grid), mwln::RW_THREADS, 0, s>>>(xp, wp, bp, yp, mp, rp, M, N, eps);
  MWLN_RW_EPLS(MWLN_RW_F, 0)
#undef MWLN_RW_F
}

template <typename CT, int NW>
static void launch_stage_fwd(int epl, const CT* xp, const float* wp, const float* bp, CT* yp, float* mp, float* rp, int64_t Mfull, int N, int R,
                             float eps, cudaStream_t s) {
  const int smem = R * N * int(sizeof(CT));
  const int64_t grid = (Mfull + R - 1) / R;
#define MWLN_ST_F(E, ...)                                                                                             \
  if (epl == E) {                                                                                                    \
    auto k = mwln::ln_stage_fwd_kernel<CT, E, NW>;                                                                   \
    if (smem > 48 * 1024) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);                 \
    k<<<unsigned(grid), NW * 32, smem, s>>>(xp, wp, bp, yp, mp, rp, Mfull, N, R, eps);                               \
  }
  MWLN_RW_EPLS(MWLN_ST_F, 0)
#undef MWLN_ST_F
}

//: rows from which the staged forward runs (fewer: the element kernel; a CTA per tile is not worth it on a few rows)
constexpr int64_t STAGE_MIN_ROWS = 4096;
//: rows up to which the forward runs a CTA per row (ln_small_fwd_kernel: the token streams, latency bound)
constexpr int64_t SMALL_MAX_ROWS = 1024;
//: the staged forward's default geometry: warps per CTA, bytes per tile
constexpr int STAGE_NW = 2;
constexpr int STAGE_TILE = 16384;

template <typename CT>
static void rows_fwd_impl(int64_t mode, int epl, const CT* xp, const float* wp, const float* bp, CT* yp, float* mp, float* rp, int64_t M, int N,
                          float eps, cudaStream_t s) {
  const int rowb = N * int(sizeof(CT));
  const bool aligned = reinterpret_cast<uintptr_t>(xp) % 16 == 0 && reinterpret_cast<uintptr_t>(yp) % 16 == 0;
  int nw = STAGE_NW, target = STAGE_TILE;
  if (mode >= 100000) {
    nw = int(mode / 100000);
    target = int(mode % 100000);
  }
  if (mode == 3 || (mode == 0 && M <= SMALL_MAX_ROWS)) {  // few rows: a CTA per row
    const int pt = (N + mwln::SM_THREADS - 1) / mwln::SM_THREADS;
#define MWLN_SM_F(P) \
  if (pt == P) mwln::ln_small_fwd_kernel<CT, P><<<unsigned(M), mwln::SM_THREADS, 0, s>>>(xp, wp, bp, yp, mp, rp, N, eps);
    MWLN_SM_F(1) MWLN_SM_F(2) MWLN_SM_F(3) MWLN_SM_F(4) MWLN_SM_F(5) MWLN_SM_F(6) MWLN_SM_F(7) MWLN_SM_F(8)
#undef MWLN_SM_F
    return;
  }
  const bool stage = mode != 1 && aligned && (mode >= 100000 || (sizeof(CT) == 2 && M >= STAGE_MIN_ROWS));
  if (!stage) {
    launch_rw_fwd<CT>(epl, xp, wp, bp, yp, mp, rp, M, N, eps, s);
    return;
  }
  const int U = 16 / gcd_int(16, rowb % 16 == 0 ? 16 : rowb % 16);
  int units = target / (U * rowb);
  if (units < 1) units = 1;
  const int R = units * U;
  const int64_t Mfull = (M / U) * U;
  if (Mfull > 0) {
    if (nw == 4) launch_stage_fwd<CT, 4>(epl, xp, wp, bp, yp, mp, rp, Mfull, N, R, eps, s);
    else if (nw == 8) launch_stage_fwd<CT, 8>(epl, xp, wp, bp, yp, mp, rp, Mfull, N, R, eps, s);
    else launch_stage_fwd<CT, 2>(epl, xp, wp, bp, yp, mp, rp, Mfull, N, R, eps, s);
  }
  if (Mfull < M)  // the tail rows
    launch_rw_fwd<CT>(epl, xp + Mfull * N, wp, bp, yp + Mfull * N, mp ? mp + Mfull : nullptr, rp ? rp + Mfull : nullptr, M - Mfull, N, eps, s);
}

// x [M, N] contiguous (bf16 / fp16 / fp32), N <= 1024; w, b fp32 [N] or empty. Returns (y, mean, rstd); mean / rstd fp32 [M] when
// `save`, else empty. `mode` 0: the default (the staged kernel from STAGE_MIN_ROWS rows on a 16-byte-aligned x, else the element kernel),
// 1: the element kernel, 3: the CTA-per-row kernel, NW * 100000 + tile bytes: the staged kernel with that geometry (NW 2 / 4 / 8).
std::vector<torch::Tensor> ln_rows_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, int64_t mode) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be contiguous CUDA [M, N]");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  const int epl = rw_epl(N);
  TORCH_CHECK(epl > 0, "warp-per-row LayerNorm: width ", N, " > 1024");
  auto y = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto mean = save ? torch::empty({M}, f32) : torch::empty({0}, f32);
  auto rstd = save ? torch::empty({M}, f32) : torch::empty({0}, f32);
  if (M == 0) return {y, mean, rstd};
  const float* wp = w.numel() ? w.data_ptr<float>() : nullptr;
  const float* bp = b.numel() ? b.data_ptr<float>() : nullptr;
  float* mp = save ? mean.data_ptr<float>() : nullptr;
  float* rp = save ? rstd.data_ptr<float>() : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = true;
  if (x.scalar_type() == torch::kBFloat16)
    rows_fwd_impl<__nv_bfloat16>(mode, epl, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), wp, bp,
                                 reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), mp, rp, M, N, float(eps), s);
  else if (x.scalar_type() == torch::kHalf)
    rows_fwd_impl<__half>(mode, epl, reinterpret_cast<const __half*>(x.data_ptr()), wp, bp, reinterpret_cast<__half*>(y.data_ptr()), mp, rp, M,
                          N, float(eps), s);
  else if (x.scalar_type() == torch::kFloat32)
    rows_fwd_impl<float>(mode, epl, x.data_ptr<float>(), wp, bp, y.data_ptr<float>(), mp, rp, M, N, float(eps), s);
  else
    ok = false;
  TORCH_CHECK(ok, "warp-per-row LayerNorm forward: unsupported dtype");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}

// dy, x [M, N] contiguous of one dtype; w fp32 [N] or empty; mean / rstd [M] fp32 from the forward. Returns (dx, dw, db): dw / db fp32 [N]
// when `wgrad`, else empty. `ctas_per_sm` bounds the persistent grid (0: as many as fit).
std::vector<torch::Tensor> ln_rows_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd, bool wgrad,
                                       int64_t unused, int64_t ctas_per_sm) {
  (void)unused;
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && dy.is_contiguous() && dy.sizes() == x.sizes() &&
                  dy.scalar_type() == x.scalar_type(),
              "dy, x must be contiguous CUDA [M, N] of one dtype");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  // a multiple of 4 values per lane, not the exact fit: the backward at 21 (651) measured 2.3x slower than at 24 (fp32, L128-768)
  const int epl = ((N + 31) / 32 + 3) / 4 * 4 <= 32 ? ((N + 31) / 32 + 3) / 4 * 4 : -1;
  TORCH_CHECK(epl > 0, "warp-per-row LayerNorm: width ", N, " > 1024");
  auto dx = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto dw = wgrad ? torch::empty({N}, f32) : torch::empty({0}, f32);
  auto db = wgrad ? torch::empty({N}, f32) : torch::empty({0}, f32);
  if (M == 0) {
    if (wgrad) {
      dw.zero_();
      db.zero_();
    }
    return {dx, dw, db};
  }
  const float* wp = w.numel() ? w.data_ptr<float>() : nullptr;
  const int64_t groups = (M + mwln::RW_WARPS - 1) / mwln::RW_WARPS;
  auto s = at::cuda::getCurrentCUDAStream();
  int grid = 0;
  torch::Tensor part;
#define MWLN_RW_B(E, CT)                                                                                                    \
  if (epl == E) {                                                                                                          \
    auto k = mwln::ln_rw_bwd_kernel<CT, E>;                                                                                \
    static int occ = 0;                                                                                                    \
    if (!occ) {                                                                                                            \
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, mwln::RW_THREADS, 0);                                         \
      if (occ < 1) occ = 1;                                                                                                \
    }                                                                                                                      \
    const int per = ctas_per_sm > 0 && ctas_per_sm < occ ? int(ctas_per_sm) : occ;                                         \
    int64_t cap = int64_t(mwln::num_sms()) * per;                                                                          \
    /* the partial rows stay a few % of the traffic: no more CTAs than ~M / 32 rows (at least one per SM) */              \
    const int64_t lim = std::max<int64_t>(mwln::num_sms(), M / 32);                                                       \
    if (cap > lim) cap = lim;                                                                                              \
    grid = int(groups < cap ? groups : cap);                                                                               \
    part = wgrad ? torch::empty({2, grid, N}, f32) : torch::empty({0}, f32);                                               \
    float* dwp = wgrad ? part.data_ptr<float>() : nullptr;                                                                 \
    float* dbp = wgrad ? part.data_ptr<float>() + int64_t(grid) * N : nullptr;                                             \
    k<<<grid, mwln::RW_THREADS, 0, s>>>(reinterpret_cast<const CT*>(dy.data_ptr()), reinterpret_cast<const CT*>(x.data_ptr()), wp,     \
                                        mean.data_ptr<float>(), rstd.data_ptr<float>(), reinterpret_cast<CT*>(dx.data_ptr()), dwp,   \
                                        dbp, M, N);                                                                       \
  }
#define MWLN_RW_DT(TORCH_T, CT) \
  if (x.scalar_type() == TORCH_T) { MWLN_RW_EPLS(MWLN_RW_B, CT) }
  MWLN_RW_DT(torch::kBFloat16, __nv_bfloat16)
  MWLN_RW_DT(torch::kHalf, __half)
  MWLN_RW_DT(torch::kFloat32, float)
#undef MWLN_RW_DT
#undef MWLN_RW_B
  TORCH_CHECK(grid > 0, "warp-per-row LayerNorm backward: unsupported dtype");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (wgrad) {
    const int blocks = (N + 31) / 32;
    mwln::ln_rw_bwd_col_reduce<<<blocks, 256, 0, s>>>(part.data_ptr<float>(), grid, N, dw.data_ptr<float>());
    mwln::ln_rw_bwd_col_reduce<<<blocks, 256, 0, s>>>(part.data_ptr<float>() + int64_t(grid) * N, grid, N, db.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dx, dw, db};
}
