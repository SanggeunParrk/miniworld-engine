// General LayerNorm forward: y = (x - mean) * rstd * w + b per row, fp32 statistics, y in x's dtype; mean / rstd saved on request.
#include "ln_common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace mwln {

template <typename T, int TPR, int CHUNKS, int VEC>
__global__ void __launch_bounds__(cta_threads<TPR>()) ln_fwd_kernel(
    const T* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, T* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t M, int N, float eps) {
  constexpr int CTA = cta_threads<TPR>();
  constexpr int RPC = CTA / TPR;
  constexpr int E = CHUNKS * VEC;
  __shared__ float red[TPR > 32 ? RPC * (TPR / 32) : 1];
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const float inv_n = 1.f / float(N);
  float wr[E], br[E];
  load_affine<TPR, CHUNKS, VEC>(w, 1.f, lane, N, wr);
  load_affine<TPR, CHUNKS, VEC>(b, 0.f, lane, N, br);
  for (int64_t base = int64_t(blockIdx.x) * RPC; base < M; base += int64_t(gridDim.x) * RPC) {
    const int64_t row = base + r_in;
    const bool valid = row < M;
    const T* xr = x + (valid ? row : 0) * int64_t(N);
    float v[E];
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int col = (c * TPR + lane) * VEC;
      if (valid && col < N) {
        load_vec<T, VEC>(xr + col, v + c * VEC);
      } else {
#pragma unroll
        for (int i = 0; i < VEC; ++i) v[c * VEC + i] = 0.f;
      }
#pragma unroll
      for (int i = 0; i < VEC; ++i) s += v[c * VEC + i];
    }
    const float mean = row_sum<TPR>(s, red, r_in, lane) * inv_n;
    float q = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int col = (c * TPR + lane) * VEC;
      if (col < N) {
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const float d = v[c * VEC + i] - mean;
          q += d * d;
        }
      }
    }
    const float rstd = rsqrtf(row_sum<TPR>(q, red, r_in, lane) * inv_n + eps);
    if (valid) {
      T* yr = y + row * int64_t(N);
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c) {
        const int col = (c * TPR + lane) * VEC;
        if (col < N) {
          float o[VEC];
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            o[i] = (v[c * VEC + i] - mean) * rstd * wr[c * VEC + i] + br[c * VEC + i];
          }
          store_vec<T, VEC>(yr + col, o);
        }
      }
      if (lane == 0 && mean_out) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
      }
    }
  }
}

template <typename T, int TPR, int CHUNKS, int VEC>
void launch_fwd(const T* x, const float* w, const float* b, T* y, float* mean, float* rstd, int64_t M, int N, float eps,
                cudaStream_t stream) {
  constexpr int CTA = cta_threads<TPR>();
  constexpr int RPC = CTA / TPR;
  const int sms = num_sms();
  const int64_t groups = (M + RPC - 1) / RPC;
  const int64_t cap = int64_t(sms) * (2048 / CTA) * 4;  // full residency, a few waves of grid-stride
  const int grid = int(groups < cap ? groups : cap);
  ln_fwd_kernel<T, TPR, CHUNKS, VEC><<<grid > 0 ? grid : 1, CTA, 0, stream>>>(x, w, b, y, mean, rstd, M, N, eps);
}

template <typename T, int VEC>
bool dispatch_fwd(int tpr, int chunks, const T* x, const float* w, const float* b, T* y, float* mean, float* rstd, int64_t M,
                  int N, float eps, cudaStream_t s) {
#define MWLN_CASE(CH, TPR_)                                                              \
  if (tpr == TPR_ && chunks == CH) {                                                     \
    launch_fwd<T, TPR_, CH, VEC>(x, w, b, y, mean, rstd, M, N, eps, s);                  \
    return true;                                                                         \
  }
#define MWLN_TPR(TPR_) MWLN_FOR_CHUNKS(MWLN_CASE, TPR_)
  if constexpr (VEC > 1) {
    MWLN_TPR(4) MWLN_TPR(8) MWLN_TPR(16)
  }
  MWLN_TPR(32) MWLN_TPR(64) MWLN_TPR(128) MWLN_TPR(256) MWLN_TPR(512) MWLN_TPR(1024)
#undef MWLN_TPR
#undef MWLN_CASE
  return false;
}

}  // namespace mwln

// x [M, N] contiguous (bf16 / fp16 / fp32); w, b fp32 [N] or empty. Returns (y, mean, rstd); mean / rstd are [M] fp32 when
// `save`, else empty.
std::vector<torch::Tensor> ln_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, int64_t tpr,
                                  int64_t chunks, int64_t vec) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be contiguous CUDA [M, N]");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  auto y = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto mean = save ? torch::empty({M}, f32) : torch::empty({0}, f32);
  auto rstd = save ? torch::empty({M}, f32) : torch::empty({0}, f32);
  if (M == 0) return {y, mean, rstd};
  if (w.numel() && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16) w = w.clone();
  if (b.numel() && reinterpret_cast<uintptr_t>(b.data_ptr()) % 16) b = b.clone();
  const float* wp = w.numel() ? w.data_ptr<float>() : nullptr;
  const float* bp = b.numel() ? b.data_ptr<float>() : nullptr;
  float* mp = save ? mean.data_ptr<float>() : nullptr;
  float* rp = save ? rstd.data_ptr<float>() : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
#define MWLN_DT(TORCH_T, CT, VMAX)                                                                                      \
  if (x.scalar_type() == TORCH_T) {                                                                                      \
    auto xp = reinterpret_cast<const CT*>(x.data_ptr());                                                                 \
    auto yp = reinterpret_cast<CT*>(y.data_ptr());                                                                       \
    ok = vec > 1 ? mwln::dispatch_fwd<CT, VMAX>(int(tpr), int(chunks), xp, wp, bp, yp, mp, rp, M, N, float(eps), s)     \
                 : mwln::dispatch_fwd<CT, 1>(int(tpr), int(chunks), xp, wp, bp, yp, mp, rp, M, N, float(eps), s);       \
  }
  MWLN_DT(torch::kBFloat16, __nv_bfloat16, 8)
  MWLN_DT(torch::kHalf, __half, 8)
  MWLN_DT(torch::kFloat32, float, 4)
#undef MWLN_DT
  TORCH_CHECK(ok, "general LayerNorm forward: no build for tpr=", tpr, " chunks=", chunks, " vec=", vec);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}
