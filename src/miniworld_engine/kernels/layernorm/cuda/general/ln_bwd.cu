// General LayerNorm backward: dx per row, and the affine gradients dw = sum_rows dy * xhat, db = sum_rows dy reduced without atomics
// (per-CTA partials in a fixed order, then a column pass), so a replay is bit-identical.
#include "ln_common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace mwln {

template <typename T, int TPR, int CHUNKS, int VEC>
__global__ void __launch_bounds__(cta_threads<TPR>()) ln_bwd_kernel(
    const T* __restrict__ dy, const T* __restrict__ x, const float* __restrict__ w, const float* __restrict__ mean,
    const float* __restrict__ rstd, T* __restrict__ dx, float* __restrict__ dw_part, float* __restrict__ db_part, int64_t M,
    int N) {
  constexpr int CTA = cta_threads<TPR>();
  constexpr int RPC = CTA / TPR;
  constexpr int E = CHUNKS * VEC;
  extern __shared__ float part_smem[];  // RPC * TPR * E floats when RPC > 1 and the affine gradients are wanted
  __shared__ float red[TPR > 32 ? RPC * (TPR / 32) : 1];
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const float inv_n = 1.f / float(N);
  const bool wgrad = dw_part != nullptr;
  float aw[E], ab[E], wr[E];
  load_affine<TPR, CHUNKS, VEC>(w, 1.f, lane, N, wr);
#pragma unroll
  for (int e = 0; e < E; ++e) aw[e] = ab[e] = 0.f;
  for (int64_t base = int64_t(blockIdx.x) * RPC; base < M; base += int64_t(gridDim.x) * RPC) {
    const int64_t row = base + r_in;
    const bool valid = row < M;
    const int64_t off = (valid ? row : 0) * int64_t(N);
    const float mu = valid ? mean[row] : 0.f, rs = valid ? rstd[row] : 0.f;
    float xh[E], g[E];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int col = (c * TPR + lane) * VEC;
      float xv[VEC], dv[VEC];
      if (valid && col < N) {
        load_vec<T, VEC>(x + off + col, xv);
        load_vec<T, VEC>(dy + off + col, dv);
      } else {
#pragma unroll
        for (int i = 0; i < VEC; ++i) xv[i] = dv[i] = 0.f;
      }
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const int e = c * VEC + i;
        xh[e] = (xv[i] - mu) * rs;
        g[e] = dv[i] * wr[e];
        s1 += g[e];
        s2 += g[e] * xh[e];
        if (wgrad) {
          aw[e] += dv[i] * xh[e];
          ab[e] += dv[i];
        }
      }
    }
    s1 = row_sum<TPR>(s1, red, r_in, lane) * inv_n;
    s2 = row_sum<TPR>(s2, red, r_in, lane) * inv_n;
    if (valid) {
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c) {
        const int col = (c * TPR + lane) * VEC;
        if (col < N) {
          float o[VEC];
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const int e = c * VEC + i;
            o[i] = rs * (g[e] - s1 - xh[e] * s2);
          }
          store_vec<T, VEC>(dx + off + col, o);
        }
      }
    }
  }
  if (!wgrad) return;
  // the CTA's rows-in-flight hold partials of the same columns: sum them in shared memory, then one partial row per CTA
  float* out_w = dw_part + int64_t(blockIdx.x) * N;
  float* out_b = db_part + int64_t(blockIdx.x) * N;
  if constexpr (RPC == 1) {
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int col = (c * TPR + lane) * VEC;
      if (col < N) {
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          out_w[col + i] = aw[c * VEC + i];
          out_b[col + i] = ab[c * VEC + i];
        }
      }
    }
  } else {
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const float* acc = pass == 0 ? aw : ab;
      float* out = pass == 0 ? out_w : out_b;
#pragma unroll
      for (int e = 0; e < E; ++e) part_smem[(r_in * TPR + lane) * E + e] = acc[e];
      __syncthreads();
      for (int idx = threadIdx.x; idx < TPR * E; idx += CTA) {
        float t = 0.f;
#pragma unroll 4
        for (int r = 0; r < RPC; ++r) t += part_smem[r * TPR * E + idx];
        const int l = idx / E, e = idx % E;
        const int col = ((e / VEC) * TPR + l) * VEC + e % VEC;
        if (col < N) out[col] = t;
      }
      __syncthreads();
    }
  }
}

// out[col] = sum over the P partial rows, 32 columns x 8 row slices per CTA, fixed order.
__global__ void __launch_bounds__(256) ln_col_reduce(const float* __restrict__ part, int P, int N, float* __restrict__ out) {
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

template <typename T, int TPR, int CHUNKS, int VEC>
int launch_bwd(const T* dy, const T* x, const float* w, const float* mean, const float* rstd, T* dx, float* dwp, float* dbp,
               int64_t M, int N, bool wgrad, int grid_cap, cudaStream_t stream) {
  constexpr int CTA = cta_threads<TPR>();
  constexpr int RPC = CTA / TPR;
  constexpr int E = CHUNKS * VEC;
  const size_t smem = RPC > 1 && wgrad ? size_t(RPC) * TPR * E * sizeof(float) : 0;
  if (smem > 48 * 1024)
    cudaFuncSetAttribute(ln_bwd_kernel<T, TPR, CHUNKS, VEC>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  static int occ[2] = {0, 0};  // resident CTAs per SM, without / with the partial-sum shared memory
  if (!occ[wgrad]) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[wgrad], ln_bwd_kernel<T, TPR, CHUNKS, VEC>, CTA, smem);
    if (occ[wgrad] < 1) occ[wgrad] = 1;
  }
  const int64_t groups = (M + RPC - 1) / RPC;
  int64_t cap = int64_t(num_sms()) * occ[wgrad];
  if (cap > grid_cap) cap = grid_cap;
  int grid = int(groups < cap ? groups : cap);
  if (grid < 1) grid = 1;
  if (!wgrad) {
    ln_bwd_kernel<T, TPR, CHUNKS, VEC><<<grid, CTA, 0, stream>>>(dy, x, w, mean, rstd, dx, nullptr, nullptr, M, N);
    return grid;
  }
  ln_bwd_kernel<T, TPR, CHUNKS, VEC><<<grid, CTA, smem, stream>>>(dy, x, w, mean, rstd, dx, dwp, dbp, M, N);
  return grid;
}

template <typename T, int VEC>
int dispatch_bwd(int tpr, int chunks, const T* dy, const T* x, const float* w, const float* mean, const float* rstd, T* dx,
                 float* dwp, float* dbp, int64_t M, int N, bool wgrad, int grid_cap, cudaStream_t s) {
#define MWLN_CASE(CH, TPR_) \
  if (tpr == TPR_ && chunks == CH) return launch_bwd<T, TPR_, CH, VEC>(dy, x, w, mean, rstd, dx, dwp, dbp, M, N, wgrad, grid_cap, s);
#define MWLN_TPR(TPR_) MWLN_FOR_CHUNKS(MWLN_CASE, TPR_)
  if constexpr (VEC > 1) {
    MWLN_TPR(4) MWLN_TPR(8) MWLN_TPR(16)
  }
  MWLN_TPR(32) MWLN_TPR(64) MWLN_TPR(128) MWLN_TPR(256) MWLN_TPR(512) MWLN_TPR(1024)
#undef MWLN_TPR
#undef MWLN_CASE
  return -1;
}

}  // namespace mwln

// dy, x [M, N] contiguous, same dtype; w fp32 [N] or empty; mean / rstd [M] fp32 from the forward. Returns (dx, dw, db): dw / db
// fp32 [N] when `wgrad`, else empty. `grid_cap` bounds the CTA count (= the partial rows of the affine gradients).
std::vector<torch::Tensor> ln_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd,
                                  bool wgrad, int64_t tpr, int64_t chunks, int64_t vec, int64_t grid_cap) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && dy.is_contiguous() && dy.sizes() == x.sizes() &&
                  dy.scalar_type() == x.scalar_type(),
              "dy, x must be contiguous CUDA [M, N] of one dtype");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  auto dx = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  // partial rows: up to every resident CTA (<= 8 per SM), but no more than ~M / 32 of them (their traffic stays a few % of x's)
  if (grid_cap < 1) {
    const int64_t sms = mwln::num_sms();
    grid_cap = std::min(sms * 8, std::max(sms, M / 32));
  }
  if (w.numel() && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16) w = w.clone();
  auto part = wgrad ? torch::empty({2, grid_cap, N}, f32) : torch::empty({0}, f32);
  auto dw = wgrad ? torch::zeros({N}, f32) : torch::empty({0}, f32);
  auto db = wgrad ? torch::zeros({N}, f32) : torch::empty({0}, f32);
  if (M == 0) return {dx, dw, db};
  const float* wp = w.numel() ? w.data_ptr<float>() : nullptr;
  float* dwp = wgrad ? part.data_ptr<float>() : nullptr;
  float* dbp = wgrad ? part.data_ptr<float>() + grid_cap * N : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  int grid = -1;
#define MWLN_DT(TORCH_T, CT, VMAX)                                                                                        \
  if (x.scalar_type() == TORCH_T) {                                                                                        \
    auto a = reinterpret_cast<const CT*>(dy.data_ptr());                                                                   \
    auto xp = reinterpret_cast<const CT*>(x.data_ptr());                                                                   \
    auto o = reinterpret_cast<CT*>(dx.data_ptr());                                                                         \
    grid = vec > 1 ? mwln::dispatch_bwd<CT, VMAX>(int(tpr), int(chunks), a, xp, wp, mean.data_ptr<float>(),               \
                                                  rstd.data_ptr<float>(), o, dwp, dbp, M, N, wgrad, int(grid_cap), s)      \
                   : mwln::dispatch_bwd<CT, 1>(int(tpr), int(chunks), a, xp, wp, mean.data_ptr<float>(),                   \
                                               rstd.data_ptr<float>(), o, dwp, dbp, M, N, wgrad, int(grid_cap), s);        \
  }
  MWLN_DT(torch::kBFloat16, __nv_bfloat16, 8)
  MWLN_DT(torch::kHalf, __half, 8)
  MWLN_DT(torch::kFloat32, float, 4)
#undef MWLN_DT
  TORCH_CHECK(grid > 0, "general LayerNorm backward: no build for tpr=", tpr, " chunks=", chunks, " vec=", vec);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (wgrad) {
    const int blocks = (N + 31) / 32;
    mwln::ln_col_reduce<<<blocks, 256, 0, s>>>(dwp, grid, N, dw.data_ptr<float>());
    mwln::ln_col_reduce<<<blocks, 256, 0, s>>>(dbp, grid, N, db.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dx, dw, db};
}
