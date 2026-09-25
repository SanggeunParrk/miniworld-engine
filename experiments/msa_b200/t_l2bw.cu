// L2 -> shared TMA bandwidth probe: every CTA streams `iters` 16 KiB boxes of an L2-resident buffer through an NST ring.
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;
template <int NST>
__global__ void __launch_bounds__(64) probe(const __grid_constant__ CUtensorMap m, int iters, int rows, int* sink) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __shared__ uint64_t full[NST], empty[NST];
  const int tid = threadIdx.x;
  if (tid == 0) { for (int s = 0; s < NST; ++s) { bar_init(full + s, 1); bar_init(empty + s, 1); } bar_init_fence(); }
  __syncthreads();
  if (tid == 0) {
    for (int g = 0; g < iters; ++g) {
      const int s = g % NST;
      if (g >= NST) wait(empty + s, ((g / NST) - 1) & 1);
      expect_tx(full + s, 16384);
      load_2d(&m, smb + s * 16384, full + s, 0, ((blockIdx.x * 7 + g) * 128) % rows);
    }
  } else if (tid == 32) {
    int acc = 0;
    for (int g = 0; g < iters; ++g) {
      const int s = g % NST;
      wait(full + s, (g / NST) & 1);
      acc += smb[s * 16384 + (g & 1023)];
      arrive(empty + s);
    }
    if (acc == 123456789) sink[0] = acc;
  }
}
double run(torch::Tensor buf, int64_t iters, int64_t nst, int64_t ctas) {
  const int rows = (int)buf.size(0);
  CUtensorMap m = make_map<2>(buf.data_ptr(), {64, (uint64_t)rows}, {64}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "buf");
  auto sink = torch::zeros({1}, buf.options().dtype(torch::kInt32));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto st = at::cuda::getCurrentCUDAStream();
  auto go = [&]() {
    const int smem = 1024 + (int)nst * 16384;
    if (nst == 4) { cudaFuncSetAttribute(probe<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); probe<4><<<ctas, 64, smem, st>>>(m, iters, rows, sink.data_ptr<int>()); }
    if (nst == 8) { cudaFuncSetAttribute(probe<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); probe<8><<<ctas, 64, smem, st>>>(m, iters, rows, sink.data_ptr<int>()); }
    if (nst == 12) { cudaFuncSetAttribute(probe<12>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); probe<12><<<ctas, 64, smem, st>>>(m, iters, rows, sink.data_ptr<int>()); }
  };
  go(); go();
  cudaEventRecord(a, st); go(); cudaEventRecord(b, st); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  return (double)ctas * iters * 16384 / (ms * 1e-3) / 1e12;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
