// distributed shared memory all-to-all read bandwidth: cluster of CS CTAs; each CTA reads `kb` KiB from its peers (spread evenly), 192 threads, 16 loads in flight
#include <torch/extension.h>
#include <cuda_runtime.h>
namespace {
__global__ void __launch_bounds__(192, 1) kern(long long* out, float* sink, int iters, int kb, int mode) {
  extern __shared__ __align__(16) unsigned char sm[];
  const uint32_t cs = 0, dummy = 0; (void)cs; (void)dummy;
  uint32_t rank; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  uint32_t nr; asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(nr));
  for (int i = threadIdx.x; i < 16384; i += blockDim.x) reinterpret_cast<float*>(sm)[i] = (float)i;
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  float acc = 0.f;
  unsigned long long t0 = 0;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sm);
  const int nf = kb * 1024 / 16;                       // float4 loads a CTA
  for (int it = 0; it < iters + 3; ++it) {
    if (it == 3 && threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
    for (int e0 = threadIdx.x; e0 < nf; e0 += blockDim.x * 16) {
      float4 t[16];
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        const int e = e0 + q * blockDim.x;
        const uint32_t peer = (rank + 1 + e % (nr - 1 > 0 ? nr - 1 : 1)) % nr;
        uint32_t a; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(base + (uint32_t)((e * 16) % 65536)), "r"(peer));
        if (e < nf) asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(t[q].x), "=f"(t[q].y), "=f"(t[q].z), "=f"(t[q].w) : "r"(a) : "memory");
        else t[q] = make_float4(0, 0, 0, 0);
      }
#pragma unroll
      for (int q = 0; q < 16; ++q) acc += t[q].x + t[q].y + t[q].z + t[q].w;
    }
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
  if (threadIdx.x == 0 && blockIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); out[0] = (long long)(t - t0); }
  if (acc == 1.2345f) sink[0] = acc;
}
}
double run(int cs, int ctas, int iters, int kb) {
  auto out = torch::zeros({4}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto sink = torch::zeros({4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
  cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(ctas, 1, 1); cfg.blockDim = dim3(192, 1, 1); cfg.dynamicSmemBytes = 200 * 1024;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = cs; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  if (cudaLaunchKernelEx(&cfg, kern, reinterpret_cast<long long*>(out.data_ptr<int64_t>()), sink.data_ptr<float>(), iters, kb, 0) != cudaSuccess) return -1.0;
  cudaDeviceSynchronize();
  auto o = out.cpu();
  return (double)o[0].item<int64_t>() / (double)iters / 1e3;     // us a round
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
