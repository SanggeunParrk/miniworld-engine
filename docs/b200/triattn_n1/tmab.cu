// TMA box-shape microbenchmark for the ending node's row tiles: every variant moves 32 KiB a tile, 2048 tiles at L512
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "sm100.cuh"
using namespace sm100;
namespace {
constexpr int NS = 4, TB = 32768;
__device__ __forceinline__ void bulk1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
template <int V>
__global__ void __launch_bounds__(32, 1) bw(const __grid_constant__ CUtensorMap m, const char* base, int ntiles, int L) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* sm = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __shared__ uint64_t bars[NS];
  if (threadIdx.x != 0) return;
  for (int i = 0; i < NS; ++i) bar_init(bars + i, 1);
  bar_init_fence();
  const int nb = L >> 7;
  auto issue = [&](int tile, int st) {
    unsigned char* dst = sm + st * TB; uint64_t* bar = bars + st;
    expect_tx(bar, TB);
    if (V == 0) { for (int kh = 0; kh < 2; ++kh) load_2d(&m, dst + kh * 16384, bar, kh * 64, tile * 128); }
    else if (V == 1 || V == 2) {
      int t = tile;
      if (V == 2) { const int nt1 = L * nb, b = t / nt1, r = t - b * nt1; t = b * nt1 + (r % L) * nb + r / L; }
      int n, s, b; rows_nsb(t * 128, L, n, s, b);
      for (int kh = 0; kh < 2; ++kh) load_4d(&m, dst + kh * 16384, bar, kh * 64, n, s, b);
    } else if (V == 3) { int n, s, b; rows_nsb(tile * 128, L, n, s, b); load_4d(&m, dst, bar, 0, n, s, b); }
    else if (V == 4) { const int n0 = (tile % (L / 8)) * 8, s0 = (tile / (L / 8)) * 16; load_4d(&m, dst, bar, 0, n0, s0, 0); }
    else if (V == 5) { int n, s, b; rows_nsb(tile * 128, L, n, s, b);
      for (int r = 0; r < 128; ++r) bulk1d(sa(dst) + r * 256, base + ((long)((b * L + s + r) * L + n)) * 256, 256, sa(bar)); }
  };
  int k = 0;
  for (int t = blockIdx.x; t < ntiles && k < NS; t += gridDim.x, ++k) issue(t, k);
  k = 0; unsigned ph = 0;
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++k) {
    const int st = k % NS; wait(bars + st, (k / NS) & 1);
    const int tn = t + NS * gridDim.x; if (tn < ntiles) issue(tn, st);
  }
}
template <int V> void run(const CUtensorMap& mp, const char* base, int ntiles, int L, int grid) {
  const size_t smem = NS * TB + 1024;
  cudaFuncSetAttribute(bw<V>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  bw<V><<<grid, 32, smem, at::cuda::getCurrentCUDAStream()>>>(mp, base, ntiles, L);
}
}
void tma_bw(torch::Tensor x, int64_t L, int64_t V, int64_t grid) {   // x: [L, L, 128] bf16
  const int ntiles = (int)(L * L / 128); const uint64_t C = 128;
  const char* base = (const char*)x.data_ptr();
  CUtensorMap m;
  switch (V) {
    case 0: m = make_map<2>(base, {C, (uint64_t)(L * L)}, {C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "v0"); break;
    case 1: case 2: m = make_map<4>(base, {C, (uint64_t)L, (uint64_t)L, 1}, {C, C * L, C * L * L}, {64, 1, 128, 1}, CU_TENSOR_MAP_SWIZZLE_128B, "v1"); break;
    case 3: m = make_map<4>(base, {C, (uint64_t)L, (uint64_t)L, 1}, {C, C * L, C * L * L}, {128, 1, 128, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "v3"); break;
    case 4: m = make_map<4>(base, {C, (uint64_t)L, (uint64_t)L, 1}, {C, C * L, C * L * L}, {128, 8, 16, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "v4"); break;
    default: m = make_map<2>(base, {C, (uint64_t)(L * L)}, {C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "v5"); break;
  }
  switch (V) { case 0: run<0>(m, base, ntiles, L, grid); break; case 1: run<1>(m, base, ntiles, L, grid); break; case 2: run<2>(m, base, ntiles, L, grid); break;
    case 3: run<3>(m, base, ntiles, L, grid); break; case 4: run<4>(m, base, ntiles, L, grid); break; default: run<5>(m, base, ntiles, L, grid); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) { mm.def("tma_bw", &tma_bw); }
