// chai_f8.cu -- EXPERIMENTAL (engine 3.0.x line, opt-in, default off): the operand pass of Chai-1's grouped OuterProductMean on e4m3
// (docs/fp8.md "Chai-1 grouped OPM").  a, b [G, S, W] bf16 (the prologue's group-major layout, rows s, W = N * 8) -> out [G, W, S] e4m3 = round(a * inv):
// the K = S axis contiguous, the form torch._scaled_mm / cuBLASLt takes (TN, both operands K-major) for the batched product O[g] = a_g^T b_g.
// A static scale (`inv`, a power of two from the weights: opm_fp8.bound_scales) -- the cast cannot saturate, the satfinite flag is a safeguard only.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {
constexpr int T = 32;
// grid (W / 32, S / 32, G), block (32, 8): a 32 x 32 tile transposed through shared memory (bytes), 32 contiguous output bytes per row
__global__ void __launch_bounds__(256) qt_kernel(const __nv_bfloat16* __restrict__ a, uint8_t* __restrict__ out, long S, long W, const float* __restrict__ inv_p) {
  __shared__ uint8_t tile[T][T + 4];
  const float inv = *inv_p;
  const long g = blockIdx.z;
  const long s0 = (long)blockIdx.y * T, w0 = (long)blockIdx.x * T;
  const __nv_bfloat16* src = a + g * S * W;
  uint8_t* dst = out + g * W * S;
#pragma unroll
  for (int i = 0; i < T; i += 8) {
    const long s = s0 + threadIdx.y + i, w = w0 + threadIdx.x;
    float v = 0.f;
    if (s < S && w < W) v = __bfloat162float(src[s * W + w]) * inv;
    tile[threadIdx.y + i][threadIdx.x] = (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < T; i += 8) {
    const long w = w0 + threadIdx.y + i, s = s0 + threadIdx.x;
    if (w < W && s < S) dst[w * S + s] = tile[threadIdx.x][threadIdx.y + i];
  }
}
}  // namespace

// a: bf16 [G, S, W] contiguous; inv: fp32 scalar on the device (1 / scale; read at run time: a captured graph follows an in-place rescale)
at::Tensor chai_quant_t(at::Tensor a, at::Tensor inv) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 && a.is_contiguous() && a.dim() == 3, "a: contiguous bf16 [G, S, W]");
  TORCH_CHECK(inv.is_cuda() && inv.scalar_type() == at::kFloat && inv.numel() == 1, "inv: fp32 scalar");
  const long G = a.size(0), S = a.size(1), W = a.size(2);
  auto out = at::empty({G, W, S}, a.options().dtype(at::kFloat8_e4m3fn));
  const c10::cuda::CUDAGuard guard(a.device());
  const dim3 grid((unsigned)((W + T - 1) / T), (unsigned)((S + T - 1) / T), (unsigned)G);
  qt_kernel<<<grid, dim3(T, 8), 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
                                                                          reinterpret_cast<uint8_t*>(out.data_ptr()), S, W, inv.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("chai_quant_t", &chai_quant_t, "bf16 [G, S, W] -> e4m3 [G, W, S] (x inv)"); }
