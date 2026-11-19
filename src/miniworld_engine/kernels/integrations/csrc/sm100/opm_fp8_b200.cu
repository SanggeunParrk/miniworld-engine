// opm_fp8_b200.cu -- EXPERIMENTAL (opt-in, MINIWORLD_OPM_FP8=bound, B200): the e4m3 operands of OuterProductMean's grouped product
// O[(i,c),(j,e)] = sum_s A2[(i,c), s] BT[(j,e), s]. A2 / BT are the bf16 outputs of the default prologue; here one launch casts both to e4m3 with a static
// power-of-two scale (q = 1 / dequantisation scale, device scalars: the scale bound of opm_fp8.bound_scales cannot be exceeded by any input), saturating
// cvt.rn.satfinite.e4m3x2 (no NaN byte). The product is torch._scaled_mm (cuBLASLt e4m3, fp32 accumulation, bf16 O).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__global__ void __launch_bounds__(256) cast2_kernel(const uint4* __restrict__ a, const uint4* __restrict__ b, uint2* __restrict__ ao, uint2* __restrict__ bo,
                                                    const float* __restrict__ q, long n8) {
  const uint4* src = blockIdx.y ? b : a;
  uint2* dst = blockIdx.y ? bo : ao;
  const float inv = q[blockIdx.y];
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (long)gridDim.x * blockDim.x) {
    const uint4 v = src[i];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
    uint32_t o[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const float2 f0 = __bfloat1622float2(p[2 * k]), f1 = __bfloat1622float2(p[2 * k + 1]);
      const __nv_fp8x2_storage_t r0 = __nv_cvt_float2_to_fp8x2(make_float2(f0.x * inv, f0.y * inv), __NV_SATFINITE, __NV_E4M3);
      const __nv_fp8x2_storage_t r1 = __nv_cvt_float2_to_fp8x2(make_float2(f1.x * inv, f1.y * inv), __NV_SATFINITE, __NV_E4M3);
      o[k] = (uint32_t)r0 | ((uint32_t)r1 << 16);
    }
    dst[i] = make_uint2(o[0], o[1]);
  }
}

void opm_cast1(at::Tensor a, at::Tensor ao, at::Tensor q, int64_t idx) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && a.is_contiguous() && ao.scalar_type() == at::kFloat8_e4m3fn && ao.is_contiguous() && ao.sizes() == a.sizes(), "opm_cast1: bf16 -> e4m3");
  TORCH_CHECK(q.scalar_type() == at::kFloat && idx >= 0 && idx < q.numel() && a.numel() % 8 == 0, "opm_cast1: q fp32, numel % 8");
  const at::cuda::CUDAGuard g(a.device());
  const long n8 = a.numel() / 8;
  const int blocks = (int)std::min<long>((n8 + 255) / 256, 148 * 8);
  cast2_kernel<<<dim3(blocks, 1), 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint4*>(a.data_ptr()), reinterpret_cast<const uint4*>(a.data_ptr()),
      reinterpret_cast<uint2*>(ao.data_ptr()), reinterpret_cast<uint2*>(ao.data_ptr()), q.data_ptr<float>() + idx, n8);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "opm_cast1 launch");
}

void opm_cast2(at::Tensor a, at::Tensor b, at::Tensor ao, at::Tensor bo, at::Tensor q) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 && a.is_contiguous() && b.is_contiguous() && a.sizes() == b.sizes(), "opm_cast2: bf16 contiguous");
  TORCH_CHECK(ao.scalar_type() == at::kFloat8_e4m3fn && bo.scalar_type() == at::kFloat8_e4m3fn && ao.is_contiguous() && bo.is_contiguous() && ao.sizes() == a.sizes() && bo.sizes() == a.sizes(), "opm_cast2: e4m3 out");
  TORCH_CHECK(q.scalar_type() == at::kFloat && q.numel() == 2 && a.numel() % 8 == 0, "opm_cast2: q [2] fp32, numel % 8");
  const at::cuda::CUDAGuard g(a.device());
  const long n8 = a.numel() / 8;
  const int blocks = (int)std::min<long>((n8 + 255) / 256, 148 * 8);
  cast2_kernel<<<dim3(blocks, 2), 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint4*>(a.data_ptr()), reinterpret_cast<const uint4*>(b.data_ptr()),
      reinterpret_cast<uint2*>(ao.data_ptr()), reinterpret_cast<uint2*>(bo.data_ptr()), q.data_ptr<float>(), n8);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "opm_cast2 launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("opm_cast2", &opm_cast2); m.def("opm_cast1", &opm_cast1); }
