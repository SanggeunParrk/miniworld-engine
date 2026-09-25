// TMA box-shape throughput probe: every CTA streams boxes of a [rows][cols] bf16 tensor through an NST ring.
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;
__global__ void __launch_bounds__(64) probe(const __grid_constant__ CUtensorMap m, int iters, int nboxr, int nboxc, int bytes, int rank3, int* sink) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  constexpr int NST = 6;
  __shared__ uint64_t full[NST], empty[NST];
  const int tid = threadIdx.x;
  const int slot = (bytes + 1023) / 1024 * 1024;
  if (tid == 0) { for (int s = 0; s < NST; ++s) { bar_init(full + s, 1); bar_init(empty + s, 1); } bar_init_fence(); }
  __syncthreads();
  if (tid == 0) {
    for (int g = 0; g < iters; ++g) {
      const int s = g % NST;
      if (g >= NST) wait(empty + s, ((g / NST) - 1) & 1);
      expect_tx(full + s, bytes);
      const int k = blockIdx.x * 131 + g * 7;
      if (rank3) load_3d(&m, smb + s * slot, full + s, 0, (k % nboxr) * 64, (k / nboxr) % nboxc * 3);
      else load_2d(&m, smb + s * slot, full + s, (k / nboxr) % nboxc * 64, (k % nboxr) * 64);
    }
  } else if (tid == 32) {
    int acc = 0;
    for (int g = 0; g < iters; ++g) { const int s = g % NST; wait(full + s, (g / NST) & 1); acc += smb[s * slot + (g & 511)]; arrive(empty + s); }
    if (acc == 123456789) sink[0] = acc;
  }
}
double run(torch::Tensor buf, int64_t mode) {
  // buf: [R][Ccols] bf16.  mode 0: box (64 cols, 64 rows) 128B swizzle (8 KiB);  mode 1: the PWA v box: (32 c, 64 rows, 3 s) 64B swizzle over
  // the same tensor viewed (32, R, Ccols/32) (12 KiB);  mode 2: box (64 cols, 192 rows) 128B swizzle (24 KiB)
  const uint64_t R = buf.size(0), CC = buf.size(1);
  CUtensorMap m; int bytes, rank3 = 0, nboxr, nboxc;
  if (mode == 1) { m = make_map<3>(buf.data_ptr(), {32, R, CC / 32}, {CC, 32}, {32, 64, 3}, CU_TENSOR_MAP_SWIZZLE_64B, "v"); bytes = 32 * 64 * 3 * 2; rank3 = 1; nboxr = R / 64; nboxc = CC / 96; }
  else if (mode == 0) { m = make_map<2>(buf.data_ptr(), {CC, R}, {CC}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "b"); bytes = 8192; nboxr = R / 64; nboxc = CC / 64; }
  else { m = make_map<2>(buf.data_ptr(), {CC, R}, {CC}, {64, 192}, CU_TENSOR_MAP_SWIZZLE_128B, "b"); bytes = 24576; nboxr = R / 192; nboxc = CC / 64; }
  auto sink = torch::zeros({1}, buf.options().dtype(torch::kInt32));
  const int iters = 3000, ctas = 148, smem = 1024 + 6 * ((bytes + 1023) / 1024 * 1024);
  cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  auto st = at::cuda::getCurrentCUDAStream();
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  probe<<<ctas, 64, smem, st>>>(m, iters, nboxr, nboxc, bytes, rank3, sink.data_ptr<int>());
  cudaEventRecord(a, st);
  probe<<<ctas, 64, smem, st>>>(m, iters, nboxr, nboxc, bytes, rank3, sink.data_ptr<int>());
  cudaEventRecord(b, st); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  return (double)ctas * iters * bytes / (ms * 1e-3) / 1e12;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
