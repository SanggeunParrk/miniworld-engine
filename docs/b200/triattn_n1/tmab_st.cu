#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "sm100.cuh"
using namespace sm100;
namespace {
constexpr int TB = 32768;
template <int V>
__global__ void __launch_bounds__(32, 1) st(const __grid_constant__ CUtensorMap m, int ntiles, int L) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* sm = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  if (threadIdx.x != 0) return;
  const int nb = L >> 7;
  fence_proxy_async();
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
    int tt = t;
    if (V == 2) { const int nt1 = L * nb, b = t / nt1, r = t - b * nt1; tt = b * nt1 + (r % L) * nb + r / L; }
    if (V == 0) { for (int kh = 0; kh < 2; ++kh) store_2d(&m, sm + kh * 16384, kh * 64, t * 128); }
    else if (V == 1 || V == 2) { int n, s, b; rows_nsb(tt * 128, L, n, s, b); for (int kh = 0; kh < 2; ++kh) store_4d(&m, sm + kh * 16384, kh * 64, n, s, b); }
    else if (V == 3) { int n, s, b; rows_nsb(t * 128, L, n, s, b); store_4d(&m, sm, 0, n, s, b); }
    else { const int n0 = (t % (L / 8)) * 8, s0 = (t / (L / 8)) * 16; store_4d(&m, sm, 0, n0, s0, 0); }
    bulk_commit();
    bulk_wait_read<3>();
  }
  bulk_wait<0>();
}
template <int V> void run(const CUtensorMap& mp, int ntiles, int L, int grid) {
  const size_t smem = TB + 1024; cudaFuncSetAttribute(st<V>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  st<V><<<grid, 32, smem, at::cuda::getCurrentCUDAStream()>>>(mp, ntiles, L);
}
}
void tma_st(torch::Tensor x, int64_t L, int64_t V, int64_t grid) {
  const int ntiles = (int)(L * L / 128); const uint64_t C = 128; const char* base = (const char*)x.data_ptr(); CUtensorMap m;
  switch (V) {
    case 0: m = make_map<2>(base, {C, (uint64_t)(L * L)}, {C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "v0"); break;
    case 1: case 2: m = make_map<4>(base, {C, (uint64_t)L, (uint64_t)L, 1}, {C, C * L, C * L * L}, {64, 1, 128, 1}, CU_TENSOR_MAP_SWIZZLE_128B, "v1"); break;
    case 3: m = make_map<4>(base, {C, (uint64_t)L, (uint64_t)L, 1}, {C, C * L, C * L * L}, {128, 1, 128, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "v3"); break;
    default: m = make_map<4>(base, {C, (uint64_t)L, (uint64_t)L, 1}, {C, C * L, C * L * L}, {128, 8, 16, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "v4"); break;
  }
  switch (V) { case 0: run<0>(m, ntiles, L, grid); break; case 1: run<1>(m, ntiles, L, grid); break; case 2: run<2>(m, ntiles, L, grid); break; case 3: run<3>(m, ntiles, L, grid); break; default: run<4>(m, ntiles, L, grid); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) { mm.def("tma_st", &tma_st); }
