// ops.cu -- the torch extension of the A100 (sm_80) MPNN hidden-message kernels (mpnn_message/cuda/sm80.py)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "msg_fwd_sm80.cuh"
#include "msg_bwd_sm80.cuh"
#include "msg_bwd_fused_sm80.cuh"

#define CHECK_CUDA_BF16(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kBFloat16, #x " must be a CUDA bf16 tensor")
#define CHECK_CUDA_F32(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kFloat32, #x " must be a CUDA fp32 tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_ALIGN16(x) TORCH_CHECK((reinterpret_cast<uintptr_t>((x).data_ptr()) & 15) == 0, #x " must be 16-byte aligned")

namespace {
using bf = __nv_bfloat16;
inline const bf* bptr(const torch::Tensor& t) { return reinterpret_cast<const bf*>(t.data_ptr()); }
inline int num_sms() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }
}  // namespace

// P [G * 48, 128] bf16, W [128, 128] bf16, bias [128] bf16, mask [G * 48] fp32 -> reduced [G, 128] fp32
torch::Tensor msg_fwd(torch::Tensor p, torch::Tensor w, torch::Tensor bias, torch::Tensor mask, double scale) {
  CHECK_CUDA_BF16(p); CHECK_CUDA_BF16(w); CHECK_CUDA_BF16(bias); CHECK_CUDA_F32(mask);
  CHECK_CONTIG(p); CHECK_CONTIG(w); CHECK_CONTIG(bias); CHECK_CONTIG(mask);
  CHECK_ALIGN16(p); CHECK_ALIGN16(w); CHECK_ALIGN16(mask);
  TORCH_CHECK(p.numel() > 0 && p.numel() % (48 * 128) == 0, "P must hold groups x 48 x 128 elements");
  TORCH_CHECK(w.numel() == 128 * 128 && bias.numel() == 128, "bad weight sizes");
  const int64_t groups = p.numel() / (48 * 128);
  TORCH_CHECK(mask.numel() == groups * 48, "mask must hold one weight per edge");
  c10::cuda::CUDAGuard guard(p.device());
  auto out = torch::empty({groups, 128}, p.options().dtype(torch::kFloat32));
  mp80::MsgFwdParams prm{bptr(p), bptr(w), bptr(bias), mask.data_ptr<float>(), out.data_ptr<float>(), groups, (float)(1.0 / scale)};
  using C = mp80::MsgFwdCfg;
  const unsigned ctas = (unsigned)std::min<int64_t>((int64_t)num_sms() * C::MINB, (groups + C::NW - 1) / C::NW);
  cudaFuncSetAttribute(mp80::msg_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
  mp80::msg_fwd_kernel<<<ctas, C::NTHR, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(prm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// the backward but for dW (see msg_bwd_sm80.cuh): P, mask, gred = d reduced [groups, 128] fp32 -> dp [rows, 128] (the gradient of P), a = bf16(gelu(P)) and dproj [rows, 128] (for the weight-gradient GEMM) and
// db_part [ctas, 128] fp32 (the bias gradient, one partial row per CTA).  All outputs are the caller's (views of larger buffers when the rows are chunked).
int64_t msg_bwd(torch::Tensor p, torch::Tensor w, torch::Tensor bias, torch::Tensor mask, torch::Tensor gred, torch::Tensor dp, torch::Tensor a, torch::Tensor dproj,
                torch::Tensor db_part, double scale) {
  CHECK_CUDA_BF16(p); CHECK_CUDA_BF16(w); CHECK_CUDA_BF16(bias); CHECK_CUDA_F32(mask); CHECK_CUDA_F32(gred); CHECK_CUDA_BF16(dp); CHECK_CUDA_BF16(a); CHECK_CUDA_BF16(dproj); CHECK_CUDA_F32(db_part);
  CHECK_CONTIG(p); CHECK_CONTIG(w); CHECK_CONTIG(bias); CHECK_CONTIG(mask); CHECK_CONTIG(gred); CHECK_CONTIG(dp); CHECK_CONTIG(a); CHECK_CONTIG(dproj); CHECK_CONTIG(db_part);
  CHECK_ALIGN16(p); CHECK_ALIGN16(w); CHECK_ALIGN16(bias); CHECK_ALIGN16(mask); CHECK_ALIGN16(dp); CHECK_ALIGN16(a); CHECK_ALIGN16(dproj);
  TORCH_CHECK(p.numel() > 0 && p.numel() % (48 * 128) == 0, "P must hold groups x 48 x 128 elements");
  const int64_t groups = p.numel() / (48 * 128);
  TORCH_CHECK(w.numel() == 128 * 128 && bias.numel() == 128 && mask.numel() == groups * 48 && gred.numel() == groups * 128, "bad operand sizes");
  TORCH_CHECK(dp.numel() == p.numel() && a.numel() == p.numel() && dproj.numel() == p.numel(), "bad output sizes");
  using C = mp80::MsgBwdCfg;
  const int64_t ctas = std::min<int64_t>((int64_t)num_sms() * C::MINB, (groups + C::NW - 1) / C::NW);
  TORCH_CHECK(db_part.numel() >= ctas * 128, "db_part is too small");
  c10::cuda::CUDAGuard guard(p.device());
  mp80::MsgBwdParams prm{bptr(p), bptr(w), bptr(bias), mask.data_ptr<float>(), gred.data_ptr<float>(), reinterpret_cast<bf*>(dp.data_ptr()), reinterpret_cast<bf*>(a.data_ptr()),
                         reinterpret_cast<bf*>(dproj.data_ptr()), db_part.data_ptr<float>(), groups, (float)(1.0 / scale)};
  cudaFuncSetAttribute(mp80::msg_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
  mp80::msg_bwd_kernel<<<(unsigned)ctas, C::NTHR, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(prm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ctas;
}

// the WHOLE backward, dW included (see msg_bwd_fused_sm80.cuh): dp [rows, 128] bf16 (the gradient of P), dw_part [ctas, 128 * 128] fp32 (one slice per CTA), db_part [ctas, 128] fp32; returns ctas
int64_t msg_bwd_fused(torch::Tensor p, torch::Tensor w, torch::Tensor bias, torch::Tensor mask, torch::Tensor gred, torch::Tensor dp, torch::Tensor dw_part, torch::Tensor db_part, double scale) {
  CHECK_CUDA_BF16(p); CHECK_CUDA_BF16(w); CHECK_CUDA_BF16(bias); CHECK_CUDA_F32(mask); CHECK_CUDA_F32(gred); CHECK_CUDA_BF16(dp); CHECK_CUDA_F32(dw_part); CHECK_CUDA_F32(db_part);
  CHECK_CONTIG(p); CHECK_CONTIG(w); CHECK_CONTIG(bias); CHECK_CONTIG(mask); CHECK_CONTIG(gred); CHECK_CONTIG(dp); CHECK_CONTIG(dw_part); CHECK_CONTIG(db_part);
  CHECK_ALIGN16(p); CHECK_ALIGN16(w); CHECK_ALIGN16(bias); CHECK_ALIGN16(mask); CHECK_ALIGN16(dp); CHECK_ALIGN16(dw_part);
  TORCH_CHECK(p.numel() > 0 && p.numel() % (48 * 128) == 0, "P must hold groups x 48 x 128 elements");
  const int64_t groups = p.numel() / (48 * 128), tiles = groups * 3;
  TORCH_CHECK(w.numel() == 128 * 128 && bias.numel() == 128 && mask.numel() == groups * 48 && gred.numel() == groups * 128, "bad operand sizes");
  TORCH_CHECK(dp.numel() == p.numel(), "bad output sizes");
  using C = mp80::MsgBwdFCfg;
  const int64_t ctas = std::min<int64_t>((int64_t)num_sms() * C::MINB, (tiles + C::NW - 1) / C::NW);
  TORCH_CHECK(db_part.numel() >= ctas * 128 && dw_part.numel() >= ctas * 128 * 128, "dw_part / db_part are too small");
  c10::cuda::CUDAGuard guard(p.device());
  mp80::MsgBwdFParams prm{bptr(p), bptr(w), bptr(bias), mask.data_ptr<float>(), gred.data_ptr<float>(), reinterpret_cast<bf*>(dp.data_ptr()), dw_part.data_ptr<float>(), db_part.data_ptr<float>(),
                          tiles, (float)(1.0 / scale)};
  cudaFuncSetAttribute(mp80::msg_bwd_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
  mp80::msg_bwd_fused_kernel<<<(unsigned)ctas, C::NTHR, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(prm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ctas;
}

// dW [128, 128] fp32 = the sum of the first `nparts` slices in order
torch::Tensor dw_reduce(torch::Tensor dw_part, int64_t nparts) {
  CHECK_CUDA_F32(dw_part); CHECK_CONTIG(dw_part);
  TORCH_CHECK(nparts >= 1 && dw_part.numel() >= nparts * 128 * 128, "bad slice count");
  c10::cuda::CUDAGuard guard(dw_part.device());
  auto out = torch::empty({128, 128}, dw_part.options());
  mp80::dw_reduce_kernel<<<64, 256, 0, at::cuda::getCurrentCUDAStream()>>>(dw_part.data_ptr<float>(), out.data_ptr<float>(), (int)nparts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

__global__ void gelu_test_kernel(const float* x, float* g, float* d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { g[i] = mp80::gelu_f(x[i]); d[i] = mp80::gelu_grad_f(x[i]); }
}

// the device GELU / GELU' on fp32 inputs (tests)
std::vector<torch::Tensor> gelu_test(torch::Tensor x) {
  CHECK_CUDA_F32(x); CHECK_CONTIG(x);
  c10::cuda::CUDAGuard guard(x.device());
  auto g = torch::empty_like(x), d = torch::empty_like(x);
  const int n = (int)x.numel();
  gelu_test_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), g.data_ptr<float>(), d.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {g, d};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("msg_fwd", &msg_fwd);
  m.def("msg_bwd", &msg_bwd);
  m.def("msg_bwd_fused", &msg_bwd_fused);
  m.def("dw_reduce", &dw_reduce);
  m.def("gelu_test", &gelu_test);
}
