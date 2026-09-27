// ops.cu -- torch binding of the A100 Transition forward kernel, on the current CUDA stream.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "tr_fwd_sm80.cuh"

namespace {

int num_sms() {
  static int n = 0;
  if (n == 0) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, at::cuda::current_device());
  return n;
}

}  // namespace

// x [T,128] bf16, w packed bf16, gb packed LN affine fp32, out [T,128] bf16 (written)
void fwd(torch::Tensor x, torch::Tensor w, torch::Tensor gb, torch::Tensor out, double eps, int64_t grid, torch::Tensor trace, torch::Tensor stats, torch::Tensor xn) {
  using G = a100::TrCfg;
  TORCH_CHECK(x.is_contiguous() && x.size(1) == 128 && x.scalar_type() == at::kBFloat16, "x: [T,128] bf16 contiguous");
  TORCH_CHECK(w.numel() == (int64_t)G::NCHUNK * G::SLOT / 2, "w: packed size");
  a100::TrParams p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  TORCH_CHECK(gb.numel() == 256 && gb.scalar_type() == at::kFloat, "gb: packed [2][8][4][4] fp32");
  p.gb = reinterpret_cast<const float4*>(gb.data_ptr<float>());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.stats = stats.numel() ? reinterpret_cast<float2*>(stats.data_ptr<float>()) : nullptr;
  p.xn = xn.numel() ? reinterpret_cast<__nv_bfloat16*>(xn.data_ptr()) : nullptr;
  p.trace = trace.numel() ? reinterpret_cast<unsigned long long*>(trace.data_ptr()) : nullptr;
  p.T = (int)x.size(0);
  p.num_tiles = (p.T + G::BM - 1) / G::BM;
  p.eps = (float)eps;
  const int g = grid > 0 ? (int)grid : std::min(p.num_tiles, num_sms() * G::MINB);
  TORCH_CHECK(p.num_tiles / g + 2 <= G::MAX_ITEMS, "fwd: more work items per CTA than the schedule table holds");
  static bool set = false;
  if (!set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::tr_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set = true;
  }
  a100::tr_fwd_kernel<G><<<g, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fwd", &fwd); }
