// EXPERIMENTAL, opt-in (pwa_fp8_train.py, MINIWORLD_PWA_FP8_TRAIN_O8=1): the PWA training step's saved attention output ``o`` ([H, N, S*C] bf16, with
// v the largest saved tensor) stored as e4m3 with a delayed per-tensor power-of-two scale, and dequantised to bf16 for the backward.
//   o8_quant(o, inv, amax)   -> e4m3(o * inv); amax (fp32 [1], NOT zeroed here) takes max |o * inv| (the next call's scale)
//   o8_dequant(o8, scale)    -> bf16(float(o8) * scale)
// One pass each over the tensor (8 elements per thread, 16 B loads / 8 B stores): memory-bound.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "../../_sm_count.cuh"

namespace {
__device__ __forceinline__ uint16_t e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

__global__ void quant_kernel(const uint4* __restrict__ x, uint2* __restrict__ q, long n8, const float* __restrict__ inv_p, float* __restrict__ amax_p) {
  const float inv = *inv_p;
  float amax = 0.f;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n8; i += (long)gridDim.x * blockDim.x) {
    const uint4 u = x[i];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
    uint16_t o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 f = __bfloat1622float2(p[k]);
      const float a = f.x * inv, b = f.y * inv;
      amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
      o[k] = e4m3x2(a, b);
    }
    uint2 r;
    r.x = (uint32_t)o[0] | ((uint32_t)o[1] << 16);
    r.y = (uint32_t)o[2] | ((uint32_t)o[3] << 16);
    q[i] = r;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((threadIdx.x & 31) == 0) atomicMax(reinterpret_cast<int*>(amax_p), __float_as_int(amax));   // non-negative floats order as ints
}

__global__ void dequant_kernel(const uint2* __restrict__ q, uint4* __restrict__ y, long n8, const float* __restrict__ scale_p) {
  const float sc = *scale_p;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n8; i += (long)gridDim.x * blockDim.x) {
    const uint2 r = q[i];
    uint4 u;
    uint32_t* w = reinterpret_cast<uint32_t*>(&u);
    const uint32_t src[2] = {r.x, r.y};
#pragma unroll
    for (int k = 0; k < 2; ++k) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const unsigned short h = (unsigned short)((src[k] >> (16 * j)) & 0xffffu);
        const __nv_fp8x2_e4m3 pr = *reinterpret_cast<const __nv_fp8x2_e4m3*>(&h);
        const float2 f = __half22float2(__half2(pr));
        const __nv_bfloat162 b = __float22bfloat162_rn(make_float2(f.x * sc, f.y * sc));
        w[k * 2 + j] = *reinterpret_cast<const uint32_t*>(&b);
      }
    }
    y[i] = u;
  }
}
}  // namespace

torch::Tensor o8_quant(torch::Tensor o, torch::Tensor inv, torch::Tensor amax) {
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == torch::kBFloat16 && o.is_contiguous() && o.numel() % 8 == 0, "o: contiguous bf16, numel % 8 == 0");
  TORCH_CHECK(inv.scalar_type() == torch::kFloat && inv.numel() == 1 && amax.scalar_type() == torch::kFloat && amax.numel() == 1, "inv / amax: fp32[1]");
  auto q = torch::empty(o.sizes(), o.options().dtype(torch::kFloat8_e4m3fn));
  const long n8 = o.numel() / 8;
  const int sms = mw::sm_count();
  const int grid = (int)std::min<long>((n8 + 255) / 256, (long)sms * 8);
  quant_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint4*>(o.data_ptr()), reinterpret_cast<uint2*>(q.data_ptr()), n8,
                                                                  inv.data_ptr<float>(), amax.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q;
}

torch::Tensor o8_dequant(torch::Tensor q, torch::Tensor scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kFloat8_e4m3fn && q.is_contiguous() && q.numel() % 8 == 0, "q: contiguous e4m3, numel % 8 == 0");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat && scale.numel() == 1, "scale: fp32[1]");
  auto y = torch::empty(q.sizes(), q.options().dtype(torch::kBFloat16));
  const long n8 = q.numel() / 8;
  const int sms = mw::sm_count();
  const int grid = (int)std::min<long>((n8 + 255) / 256, (long)sms * 8);
  dequant_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint2*>(q.data_ptr()), reinterpret_cast<uint4*>(y.data_ptr()), n8,
                                                                    scale.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("o8_quant", &o8_quant, "bf16 -> e4m3 of o * inv; amax of |o * inv|", py::arg("o"), py::arg("inv"), py::arg("amax"));
  mod.def("o8_dequant", &o8_dequant, "e4m3 -> bf16 of o8 * scale", py::arg("q"), py::arg("scale"));
}
