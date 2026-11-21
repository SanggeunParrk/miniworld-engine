// software grid barrier microbenchmark (B200): NB CTAs, one per SM (big smem), `n` barriers back to back
#include <torch/extension.h>
#include <cuda_runtime.h>
namespace {
__device__ __forceinline__ void barrier(unsigned long long* ctr, unsigned int target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;" :: "l"(ctr) : "memory");
    unsigned long long v;
    do { asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(ctr) : "memory"); } while (v < target);
  }
  __syncthreads();
}
__global__ void __launch_bounds__(128, 1) kern(unsigned long long* ctr, long long* out, int n, int work) {
  extern __shared__ unsigned char sm[];
  const unsigned int nb = gridDim.x;
  unsigned long long t0 = 0;
  float acc = 0.f;
  for (int i = 0; i < n; ++i) {
    if (i == 10 && threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); t0 = t; }
    for (int w = 0; w < work; ++w) acc = acc * 1.0001f + 1.f;   // a little per-CTA work between the barriers
    barrier(ctr, (unsigned int)(i + 1) * nb);
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); out[0] = (long long)(t - t0); out[1] = n - 10; }
  if (acc == 123.f) out[2] = 1;
}
}
std::vector<double> run(int nb, int n, int work) {
  auto ctr = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto out = torch::zeros({4}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
  kern<<<nb, 128, 200 * 1024>>>(reinterpret_cast<unsigned long long*>(ctr.data_ptr<int64_t>()), reinterpret_cast<long long*>(out.data_ptr<int64_t>()), n, work);
  cudaDeviceSynchronize();
  auto o = out.cpu();
  return {(double)o[0].item<int64_t>() / (double)o[1].item<int64_t>() / 1e3};   // us a barrier
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
