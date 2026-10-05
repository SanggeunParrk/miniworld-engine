// ops.cu -- the torch extension of the A100 (sm_80) MPNN relative-position bucket reduction (mpnn_relative_position/cuda/sm80.py)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "relpos_sm80.cuh"

// grad [rows, 16] bf16 or fp32 (contiguous), bucket [rows] int64 -> (grad_table [nbuckets, 16] fp32, grad_bias [16] fp32); nbuckets <= 79
std::vector<torch::Tensor> relpos_reduce(torch::Tensor grad, torch::Tensor bucket, int64_t nbuckets) {
  using namespace mp80;
  TORCH_CHECK(grad.is_cuda() && bucket.is_cuda() && grad.device() == bucket.device(), "grad and bucket must be CUDA tensors of one device");
  TORCH_CHECK(grad.is_contiguous() && bucket.is_contiguous(), "grad and bucket must be contiguous");
  TORCH_CHECK(bucket.scalar_type() == torch::kLong, "bucket must be int64");
  const bool f32 = grad.scalar_type() == torch::kFloat32;
  TORCH_CHECK(f32 || grad.scalar_type() == torch::kBFloat16, "grad must be bf16 or fp32");
  const int64_t rows = bucket.numel();
  TORCH_CHECK(rows > 0 && grad.numel() == rows * RP_W, "grad must hold 16 channels per bucket index");
  TORCH_CHECK(nbuckets >= 1 && nbuckets <= RP_ROWS - 1, "nbuckets must be in [1, 79]");
  c10::cuda::CUDAGuard guard(grad.device());
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t nsteps = (rows + 15) / 16;
  const int nw = RelposCfg<false>::NW;
  const int ctas = (int)std::max<int64_t>(1, std::min<int64_t>((int64_t)sms * RelposCfg<false>::MINB, (nsteps + nw - 1) / nw));
  auto fopt = grad.options().dtype(torch::kFloat32);
  auto partial = torch::empty({ctas, RP_TAB}, fopt);
  auto table = torch::empty({nbuckets, RP_W}, fopt);
  auto bias = torch::empty({RP_W}, fopt);
  RelposParams p{grad.data_ptr(), bucket.data_ptr<int64_t>(), partial.data_ptr<float>(), rows, (int)nbuckets};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (f32) relpos_kernel<true><<<ctas, RelposCfg<true>::NTHR, RelposCfg<true>::SMEM, stream>>>(p);
  else relpos_kernel<false><<<ctas, RelposCfg<false>::NTHR, RelposCfg<false>::SMEM, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  relpos_reduce_kernel<<<(RP_TAB + 127) / 128, 128, 0, stream>>>(partial.data_ptr<float>(), table.data_ptr<float>(), bias.data_ptr<float>(), ctas, (int)nbuckets);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {table, bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relpos_reduce", &relpos_reduce);
}
