// General LayerNorm / RMSNorm forward: y = (x - mean) * rstd * w + b per row (RMS: mean = 0, no bias), fp32 statistics, y in x's
// dtype; mean / rstd saved on request. A row is held as the packs it was loaded as.
#include "ln_common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace mwln {

template <typename T, int TPR, int CHUNKS, int VEC, bool RMS, int RPT>
__global__ void __launch_bounds__(cta_threads<TPR>()) ln_fwd_kernel(
    const T* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b, int wbf, int bbf, T* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t M, int N, float eps) {
  constexpr int CTA = cta_threads<TPR>();
  constexpr int RPC = CTA / TPR;
  constexpr int E = CHUNKS * VEC;
  static_assert(RPT == 1, "built at one row a thread");
  __shared__ float red[TPR > 32 ? RPC * (TPR / 32) : 1];
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const float inv_n = 1.f / float(N);
  float wr[E], br[E];
  load_affine<TPR, CHUNKS, VEC>(w, wbf, 1.f, lane, N, wr);
  if constexpr (!RMS) load_affine<TPR, CHUNKS, VEC>(b, bbf, 0.f, lane, N, br);
  // one row of this thread, from its packs: statistics, the normalized row, the stats (lane 0)
  auto process = [&](const Pack<T, VEC>* raw, int64_t row) {
    float mean = 0.f;
    if constexpr (!RMS) {
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
        for (int i = 0; i < VEC; ++i) s += to_f<T>(raw[c].v[i]);
      mean = row_sum<TPR>(s, red, r_in, lane) * inv_n;
    }
    float q = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int col = (c * TPR + lane) * VEC;
      if (col < N) {
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const float d = to_f<T>(raw[c].v[i]) - mean;
          q += d * d;
        }
      }
    }
    const float rstd = rsqrtf(row_sum<TPR>(q, red, r_in, lane) * inv_n + eps);
    if (row < M) {
      T* yr = y + row * int64_t(N);
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c) {
        const int col = (c * TPR + lane) * VEC;
        if (col < N) {
          float o[VEC];
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const float xn = (to_f<T>(raw[c].v[i]) - mean) * rstd;
            o[i] = RMS ? xn * wr[c * VEC + i] : xn * wr[c * VEC + i] + br[c * VEC + i];
          }
          store_vec<T, VEC>(yr + col, o);
        }
      }
      if (lane == 0) {
        if (mean_out) mean_out[row] = mean;
        if (rstd_out) rstd_out[row] = rstd;
      }
    }
  };
  auto load_row = [&](Pack<T, VEC>* raw, int64_t row) {
    const bool valid = row < M;
    const T* xr = x + (valid ? row : 0) * int64_t(N);
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int col = (c * TPR + lane) * VEC;
      raw[c] = (valid && col < N) ? load_pack<T, VEC>(xr + col) : zero_pack<T, VEC>();
    }
  };
  const int64_t stride = int64_t(gridDim.x) * RPC;
  for (int64_t base = int64_t(blockIdx.x) * RPC; base < M; base += stride) {
    Pack<T, VEC> raw[CHUNKS];
    load_row(raw, base + r_in);
    process(raw, base + r_in);
  }
}

template <typename T, int TPR, int CHUNKS, int VEC, bool RMS, int RPT>
void launch_fwd(const T* x, const void* w, const void* b, int wbf, int bbf, T* y, float* mean, float* rstd, int64_t M, int N,
                float eps, int waves, cudaStream_t stream) {
  constexpr int CTA = cta_threads<TPR>();
  constexpr int RPC = CTA / TPR;
  const int sms = num_sms();
  const int64_t groups = (M + RPC - 1) / RPC;
  int64_t cap;
  if (waves > 0) {   // `waves` resident waves: the CTAs the SMs actually hold (registers), not the 2048-thread bound
    static int occ = 0;
    if (!occ) {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ln_fwd_kernel<T, TPR, CHUNKS, VEC, RMS, RPT>, CTA, 0);
      if (occ < 1) occ = 1;
    }
    cap = int64_t(sms) * occ * waves;
  } else {
    cap = int64_t(sms) * (2048 / CTA) * 4;  // full residency, a few waves of grid-stride (the H100 launch)
  }
  const int grid = int(groups < cap ? groups : cap);
  ln_fwd_kernel<T, TPR, CHUNKS, VEC, RMS, RPT>
      <<<grid > 0 ? grid : 1, CTA, 0, stream>>>(x, w, b, wbf, bbf, y, mean, rstd, M, N, eps);
}

template <typename T, int VEC, bool RMS>
bool dispatch_fwd(int tpr, int chunks, int rpt, const T* x, const void* w, const void* b, int wbf, int bbf, T* y, float* mean,
                  float* rstd, int64_t M, int N, float eps, int waves, cudaStream_t s) {
#define MWLN_CASE(CH, TPR_)                                                                     \
  if (tpr == TPR_ && chunks == CH && rpt == 1) {                                                \
    launch_fwd<T, TPR_, CH, VEC, RMS, 1>(x, w, b, wbf, bbf, y, mean, rstd, M, N, eps, waves, s); \
    return true;                                                                                \
  }
#define MWLN_TPR(TPR_) MWLN_FOR_CHUNKS(MWLN_CASE, TPR_)
#define MWLN_TPR_V(TPR_) MWLN_FOR_CHUNKS_V(MWLN_CASE, TPR_)
  if constexpr (VEC > 1) {
    MWLN_TPR_V(4) MWLN_TPR_V(8) MWLN_TPR_V(16) MWLN_TPR_V(32) MWLN_TPR_V(64) MWLN_TPR_V(128) MWLN_TPR_V(256) MWLN_TPR_V(512) MWLN_TPR_V(1024)
  } else {
    MWLN_TPR(32) MWLN_TPR(64) MWLN_TPR(128) MWLN_TPR(256) MWLN_TPR(512) MWLN_TPR(1024)
  }
#undef MWLN_TPR_V
#undef MWLN_TPR
#undef MWLN_CASE
  return false;
}

}  // namespace mwln

// x [M, N] contiguous (bf16 / fp16 / fp32); w, b fp32 or bf16 [N] (16-byte aligned) or empty. Returns (y, mean, rstd); rstd is [M]
// fp32 when `save`, mean too unless `rms` (RMSNorm: no mean, and b must be empty), else they are empty. `rpt`: rows per thread in
// flight (TPR <= 32 only); `waves`: the grid cap in resident waves (0: 4).
std::vector<torch::Tensor> ln_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, bool rms, int64_t tpr,
                                  int64_t chunks, int64_t vec, int64_t rpt, int64_t waves) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be contiguous CUDA [M, N]");
  TORCH_CHECK(!rms || b.numel() == 0, "RMSNorm has no bias");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  auto y = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto mean = save && !rms ? torch::empty({M}, f32) : torch::empty({0}, f32);
  auto rstd = save ? torch::empty({M}, f32) : torch::empty({0}, f32);
  if (M == 0) return {y, mean, rstd};
  for (auto* t : {&w, &b}) {
    if (!t->numel()) continue;
    TORCH_CHECK(t->scalar_type() == torch::kFloat32 || t->scalar_type() == torch::kBFloat16, "affine parameters fp32 or bf16");
    if (reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 || !t->is_contiguous()) *t = t->contiguous().clone();
  }
  const void* wp = w.numel() ? w.data_ptr() : nullptr;
  const void* bp = b.numel() ? b.data_ptr() : nullptr;
  const int wbf = w.numel() && w.scalar_type() == torch::kBFloat16, bbf = b.numel() && b.scalar_type() == torch::kBFloat16;
  float* mp = mean.numel() ? mean.data_ptr<float>() : nullptr;
  float* rp = save ? rstd.data_ptr<float>() : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
#define MWLN_GO(CT, V)                                                                                                            \
  (rms ? mwln::dispatch_fwd<CT, V, true>(int(tpr), int(chunks), int(rpt), xp, wp, bp, wbf, bbf, yp, mp, rp, M, N, float(eps), int(waves), s) \
       : mwln::dispatch_fwd<CT, V, false>(int(tpr), int(chunks), int(rpt), xp, wp, bp, wbf, bbf, yp, mp, rp, M, N, float(eps), int(waves), s))
#define MWLN_DT(TORCH_T, CT, VMAX)                                     \
  if (x.scalar_type() == TORCH_T) {                                     \
    auto xp = reinterpret_cast<const CT*>(x.data_ptr());                \
    auto yp = reinterpret_cast<CT*>(y.data_ptr());                      \
    ok = vec > 1 ? MWLN_GO(CT, VMAX) : MWLN_GO(CT, 1);                  \
  }
  MWLN_DT(torch::kBFloat16, __nv_bfloat16, 8)
  MWLN_DT(torch::kHalf, __half, 8)
  MWLN_DT(torch::kFloat32, float, 4)
#undef MWLN_DT
#undef MWLN_GO
  TORCH_CHECK(ok, "general LayerNorm forward: no build for tpr=", tpr, " chunks=", chunks, " vec=", vec, " rpt=", rpt);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}
