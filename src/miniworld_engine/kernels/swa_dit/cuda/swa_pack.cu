// ESMFold2 SWA atom block: the FFN weight layouts of the team-gm wgmma kernels (swa_ffn_fwd.cu / swa_ffn_bwd.cu), all four in ONE launch
// (they were four torch ``.contiguous()`` copies per block per step):
//   wab64 [512, 128]  per 64-wide hidden chunk j rows [Wu[64j .. 64j+63] ; Wu[256 + 64j ..]]      (forward a|b tile)
//   wab32 [512, 128]  per 32-wide hidden chunk j rows [Wu[32j .. 32j+31] ; Wu[256 + 32j ..]]      (backward a|b tile)
//   wdt   [256, 128]  Wd^T                                                                         (backward)
//   wabt  [128, 512]  wab32^T                                                                      (backward)
// bf16 [Wu: 512 x 128, Wd: 128 x 256] only.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {
constexpr int C = 128, NH = 256, R2 = 2 * NH;
__global__ void __launch_bounds__(256) pack_kernel(const __nv_bfloat16* __restrict__ wu, const __nv_bfloat16* __restrict__ wd,
                                                   __nv_bfloat16* __restrict__ wab64, __nv_bfloat16* __restrict__ wab32,
                                                   __nv_bfloat16* __restrict__ wdt, __nv_bfloat16* __restrict__ wabt) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < R2 * C) {
    const int r = i / C, c = i % C, h = r / NH, k = r % NH;           // source row r = h * 256 + k
    const __nv_bfloat16 v = wu[i];
    wab64[((k / 64) * 2 + h) * 64 * C + (k % 64) * C + c] = v;
    const int r32 = ((k / 32) * 2 + h) * 32 + (k % 32);
    wab32[r32 * C + c] = v;
    wabt[c * R2 + r32] = v;
  } else if (i < R2 * C + C * NH) {
    const int j = i - R2 * C, r = j / NH, n = j % NH;                  // wd [128][256]
    wdt[n * C + r] = wd[j];
  }
}
}  // namespace

std::vector<torch::Tensor> pack_ffn(torch::Tensor wu, torch::Tensor wd) {
  TORCH_CHECK(wu.is_cuda() && wu.scalar_type() == torch::kBFloat16 && wu.is_contiguous() && wu.size(0) == R2 && wu.size(1) == C, "wu [512, 128] bf16");
  TORCH_CHECK(wd.is_cuda() && wd.scalar_type() == torch::kBFloat16 && wd.is_contiguous() && wd.size(0) == C && wd.size(1) == NH, "wd [128, 256] bf16");
  auto o = wu.options();
  auto wab64 = torch::empty({R2, C}, o), wab32 = torch::empty({R2, C}, o), wdt = torch::empty({NH, C}, o), wabt = torch::empty({C, R2}, o);
  const int n = R2 * C + C * NH;
  pack_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(wu.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(wd.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(wab64.data_ptr()), reinterpret_cast<__nv_bfloat16*>(wab32.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(wdt.data_ptr()), reinterpret_cast<__nv_bfloat16*>(wabt.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {wab64, wab32, wdt, wabt};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_ffn", &pack_ffn, "the wgmma FFN kernels' weight layouts in one launch");
}
