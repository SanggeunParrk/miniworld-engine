// cluster barrier cost (B200): cluster of CS CTAs, one CTA an SM (large smem), n arrive/wait pairs back to back
#include <torch/extension.h>
#include <cuda_runtime.h>
namespace {
__global__ void __launch_bounds__(128, 1) kern(long long* out, int n, int mode) {
  unsigned long long t0 = 0;
  for (int i = 0; i < n + 20; ++i) {
    if (i == 20 && threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    if (mode == 0) asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    else if (mode == 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
    else { asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory"); }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); out[0] = (long long)(t - t0); }
}
}
double run(int cs, int ctas, int n, int mode) {
  auto out = torch::zeros({4}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
  cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(ctas, 1, 1); cfg.blockDim = dim3(128, 1, 1); cfg.dynamicSmemBytes = 200 * 1024;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = cs; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, reinterpret_cast<long long*>(out.data_ptr<int64_t>()), n, mode);
  if (e != cudaSuccess) return -1.0;
  cudaDeviceSynchronize();
  auto o = out.cpu();
  return (double)o[0].item<int64_t>() / (double)n / 1e3;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
