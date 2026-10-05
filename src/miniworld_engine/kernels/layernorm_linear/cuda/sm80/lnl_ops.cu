// lnl_ops.cu -- torch bindings of the A100 fused LayerNorm + projection kernels (lnl_sm80.cuh), on the current CUDA stream.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <algorithm>

#include "lnl_sm80.cuh"

namespace {

using bf = __nv_bfloat16;

int nsm() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }
cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
bool aligned16(const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; }

// resident CTAs per SM of a kernel with `smem` dynamic shared bytes (at least 1)
template <class K>
int occupancy(K kernel, int nthr, int smem) {
  int n = 1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, nthr, smem));
  return std::max(n, 1);
}

template <int D, int NT>
void run_fwd(const lnl::FwdParams& p) {
  using G = lnl::FwdCfg<D, NT, 256>;
  const auto kernel = lnl::lnl_fwd_kernel<G, D, NT>;
  const long long ntile = (p.M + 15) / 16;
  const long long want = (ntile + G::NW - 1) / G::NW;
  const long long cap = (long long)nsm() * occupancy(kernel, G::NTHR, G::SMEM);
  const unsigned grid = (unsigned)std::max<long long>(1, std::min(want, cap));
  kernel<<<grid, G::NTHR, G::SMEM, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int D>
void run_bwd(const lnl::BwdParams& p0, torch::Tensor dx, torch::Tensor dw, torch::Tensor dgamma, torch::Tensor w, torch::Tensor gamma, int ln_bf16) {
  using S = lnl::BwdShape<D>;
  const auto kernel = lnl::lnl_bwd_kernel<D>;
  static_assert(S::SMEM <= 49152, "the backward's shared memory stays below the default limit");
  const long long ntile = (p0.M + 15) / 16;
  const long long want = (ntile + S::RP - 1) / S::RP;
  const long long cap = (long long)nsm() * occupancy(kernel, S::NTHR, S::SMEM);
  const unsigned grid = (unsigned)std::max<long long>(1, std::min(want, cap));
  auto part = torch::empty({(long long)grid, p0.nh, D}, torch::TensorOptions().dtype(at::kFloat).device(dx.device()));
  lnl::BwdParams p = p0;
  p.part = part.data_ptr<float>();
  kernel<<<grid, S::NTHR, S::SMEM, stream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto gn = torch::empty({p0.nh, D}, torch::TensorOptions().dtype(at::kFloat).device(dx.device()));
  lnl::FinParams f;
  f.part = p.part; f.gn = gn.data_ptr<float>(); f.w = p.w; f.gamma = p.gamma; f.dw = reinterpret_cast<bf*>(dw.data_ptr()); f.dgamma = dgamma.data_ptr();
  f.P = (int)grid; f.nh = p.nh; f.D = D; f.ln_bf16 = ln_bf16;
  lnl::lnl_reduce_kernel<<<(p0.nh * D + 255) / 256, 256, 0, stream()>>>(f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  lnl::lnl_dgamma_kernel<<<(D + 127) / 128, 128, 0, stream()>>>(f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_common(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& gamma, int64_t D, int64_t nh) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == D && aligned16(x), "lnl: x [M, D] bf16 contiguous, 16-byte aligned");
  TORCH_CHECK(x.size(0) < (int64_t(1) << 31) && x.size(0) >= 1, "lnl: M");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.dim() == 2 && w.size(0) == nh && w.size(1) == D && aligned16(w), "lnl: w [nh, D] bf16");
  TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == at::kFloat && gamma.is_contiguous() && gamma.numel() == D, "lnl: gamma [D] fp32");
  TORCH_CHECK(nh >= 1 && nh <= 16, "lnl: 1 .. 16 outputs");
}

}  // namespace

// out [M, nh] bf16 = LayerNorm(x) gamma W^T; stats [M, 2] fp32 (mean, rstd) and u [M, nh] fp32 (the result before rounding) when not empty
void lnl_forward(torch::Tensor x, torch::Tensor w, torch::Tensor gamma, torch::Tensor out, torch::Tensor stats, torch::Tensor u, double eps) {
  const int64_t M = x.size(0), D = x.size(1), nh = w.size(0);
  check_common(x, w, gamma, D, nh);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == M * nh && aligned16(out), "lnl_forward: out [M, nh] bf16");
  TORCH_CHECK(!stats.numel() || (stats.scalar_type() == at::kFloat && stats.is_contiguous() && stats.numel() == 2 * M && aligned16(stats)), "lnl_forward: stats [M, 2] fp32");
  TORCH_CHECK(!u.numel() || (u.scalar_type() == at::kFloat && u.is_contiguous() && u.numel() == M * nh), "lnl_forward: u [M, nh] fp32");
  lnl::FwdParams p;
  p.x = reinterpret_cast<const bf*>(x.data_ptr()); p.w = reinterpret_cast<const bf*>(w.data_ptr()); p.gamma = gamma.data_ptr<float>();
  p.out = reinterpret_cast<bf*>(out.data_ptr()); p.stats = stats.numel() ? stats.data_ptr<float>() : nullptr; p.u = u.numel() ? u.data_ptr<float>() : nullptr;
  p.M = M; p.nh = (int)nh; p.eps = (float)eps;
  const c10::cuda::CUDAGuard guard(x.device());
#define FWD_CASE(DD) case DD: if (nh <= 8) run_fwd<DD, 1>(p); else run_fwd<DD, 2>(p); break;
  switch (D) { FWD_CASE(16) FWD_CASE(64) FWD_CASE(128) FWD_CASE(256) FWD_CASE(384) FWD_CASE(512) default: TORCH_CHECK(false, "lnl_forward: unsupported width ", D); }
#undef FWD_CASE
}

// dx [M, D] bf16, dw [nh, D] bf16, dgamma [D] (fp32, or bf16 when the LayerNorm scale was) from dout [M, nh] bf16 and the forward's stats / u
void lnl_backward(torch::Tensor x, torch::Tensor w, torch::Tensor gamma, torch::Tensor dout, torch::Tensor stats, torch::Tensor u, torch::Tensor dx, torch::Tensor dw,
                  torch::Tensor dgamma, int64_t ln_bf16) {
  const int64_t M = x.size(0), D = x.size(1), nh = w.size(0);
  check_common(x, w, gamma, D, nh);
  TORCH_CHECK(dout.is_cuda() && dout.scalar_type() == at::kBFloat16 && dout.is_contiguous() && dout.numel() == M * nh && aligned16(dout), "lnl_backward: dout [M, nh] bf16");
  TORCH_CHECK(stats.is_cuda() && stats.scalar_type() == at::kFloat && stats.is_contiguous() && stats.numel() == 2 * M && aligned16(stats), "lnl_backward: stats [M, 2] fp32");
  TORCH_CHECK(u.is_cuda() && u.scalar_type() == at::kFloat && u.is_contiguous() && u.numel() == M * nh && aligned16(u), "lnl_backward: u [M, nh] fp32");
  TORCH_CHECK(dx.is_cuda() && dx.scalar_type() == at::kBFloat16 && dx.is_contiguous() && dx.numel() == M * D && aligned16(dx), "lnl_backward: dx [M, D] bf16");
  TORCH_CHECK(dw.is_cuda() && dw.scalar_type() == at::kBFloat16 && dw.is_contiguous() && dw.numel() == nh * D, "lnl_backward: dw [nh, D] bf16");
  TORCH_CHECK(dgamma.is_cuda() && dgamma.is_contiguous() && dgamma.numel() == D && dgamma.scalar_type() == (ln_bf16 ? at::kBFloat16 : at::kFloat), "lnl_backward: dgamma [D]");
  lnl::BwdParams p;
  p.x = reinterpret_cast<const bf*>(x.data_ptr()); p.w = reinterpret_cast<const bf*>(w.data_ptr()); p.gamma = gamma.data_ptr<float>();
  p.dout = reinterpret_cast<const bf*>(dout.data_ptr()); p.stats = stats.data_ptr<float>(); p.u = u.data_ptr<float>();
  p.dx = reinterpret_cast<bf*>(dx.data_ptr()); p.part = nullptr; p.M = M; p.nh = (int)nh;
  const c10::cuda::CUDAGuard guard(x.device());
#define BWD_CASE(DD) case DD: run_bwd<DD>(p, dx, dw, dgamma, w, gamma, (int)ln_bf16); break;
  switch (D) { BWD_CASE(16) BWD_CASE(64) BWD_CASE(128) BWD_CASE(256) BWD_CASE(384) BWD_CASE(512) default: TORCH_CHECK(false, "lnl_backward: unsupported width ", D); }
#undef BWD_CASE
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lnl_forward", &lnl_forward);
  m.def("lnl_backward", &lnl_backward);
}
