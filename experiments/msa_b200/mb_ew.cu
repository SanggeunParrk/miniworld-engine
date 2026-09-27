// Elementwise ceiling of the backward grad loop: 8 warps / SM run the per-stage math (32 elements / thread) on registers.
// VAR 0: full (ex2, P pack, dS, dS pack)   1: no packs   2: ex2 only   3: no ex2 (FMA chain + packs)   4: packs only
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;
__device__ __forceinline__ float ex2f(float x) { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
template <int VAR>
__global__ void __launch_bounds__(256, 1) mb(int iters, float* sink, long long* clk) {
  float sv[32], dp[32], bl[32];
  for (int i = 0; i < 32; ++i) { sv[i] = threadIdx.x * 1e-3f + i; dp[i] = i * 0.5f; bl[i] = -i * 0.1f; }
  const float2 sc2 = make_float2(0.25f, 0.25f), neg1 = make_float2(-1.f, -1.f);
  float2 lm = make_float2(0.1f, 0.2f), dm = make_float2(0.3f, 0.4f);
  uint32_t acc = 0;
  __syncthreads();
  long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < 32; i += 2) {
      float2 x = fma2(lm, neg1, fma2(make_float2(sv[i], sv[i + 1]), sc2, make_float2(bl[i], bl[i + 1])));
      float2 pp;
      if (VAR == 3) pp = x; else pp = make_float2(ex2f(x.x), ex2f(x.y));
      float2 ds = mul2(pp, fma2(dm, neg1, make_float2(dp[i], dp[i + 1])));
      if (VAR == 2) { acc += __float_as_uint(pp.x) ^ __float_as_uint(pp.y); continue; }
      if (VAR == 1) { acc += __float_as_uint(ds.x) + __float_as_uint(pp.y); continue; }
      if (VAR == 4) { acc += pack2(sv[i], dp[i]) + pack2(bl[i], sv[i + 1]); continue; }
      acc += pack2(pp.x, pp.y) + pack2(ds.x, ds.y);
    }
    lm.x += 1e-7f; dm.y += 1e-7f;
  }
  __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) clk[blockIdx.x] = t1 - t0;
  if (acc == 0x12345678) sink[0] = 1.f;
}
torch::Tensor run(int var, int iters) {
  auto out = torch::zeros({148}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto sink = torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto k = var == 0 ? mb<0> : var == 1 ? mb<1> : var == 2 ? mb<2> : var == 3 ? mb<3> : mb<4>;
  k<<<148, 256>>>(iters, sink.data_ptr<float>(), reinterpret_cast<long long*>(out.data_ptr<int64_t>()));
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
