// tail input per tile: x (32 KiB, the caller's transposed pair) + o (32 KiB plane rows): today's shapes vs an 8 n x 16 s tile
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "sm100.cuh"
using namespace sm100;
namespace {
constexpr int NS = 3, TB = 65536;
template <int V>
__global__ void __launch_bounds__(32, 1) bw2(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mo, int ntiles, int L) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* sm = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __shared__ uint64_t bars[NS];
  if (threadIdx.x != 0) return;
  for (int i = 0; i < NS; ++i) bar_init(bars + i, 1);
  bar_init_fence();
  auto issue = [&](int tile, int st) {
    unsigned char* dst = sm + st * TB; uint64_t* bar = bars + st; expect_tx(bar, TB);
    if (V == 0) {   // today: x = 2 boxes 64 x (1 n, 128 s); o = 2 boxes 64 ch x 128 rows
      int n, s, b; rows_nsb(tile * 128, L, n, s, b);
      for (int kh = 0; kh < 2; ++kh) { load_4d(&mx, dst + kh * 16384, bar, kh * 64, n, s, b); load_2d(&mo, dst + 32768 + kh * 16384, bar, kh * 64, tile * 128); }
    } else {        // 8 n x 16 s tile: x = one box 128 ch x (8 n, 16 s) (2 KiB runs); o = 8 pieces of 16 rows x 2 halves
      const int n0 = (tile % (L / 8)) * 8, s0 = (tile / (L / 8)) * 16;
      load_4d(&mx, dst, bar, 0, n0, s0, 0);
      for (int j = 0; j < 8; ++j) for (int kh = 0; kh < 2; ++kh) load_2d(&mo, dst + 32768 + kh * 16384 + j * 2048, bar, kh * 64, (n0 + j) * L + s0);
    }
  };
  int k = 0;
  for (int t = blockIdx.x; t < ntiles && k < NS; t += gridDim.x, ++k) issue(t, k);
  k = 0;
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++k) { const int st = k % NS; wait(bars + st, (k / NS) & 1); const int tn = t + NS * gridDim.x; if (tn < ntiles) issue(tn, st); }
}
template <int V> void run(const CUtensorMap& a, const CUtensorMap& b, int nt, int L, int grid) {
  const size_t smem = NS * TB + 1024; cudaFuncSetAttribute(bw2<V>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  bw2<V><<<grid, 32, smem, at::cuda::getCurrentCUDAStream()>>>(a, b, nt, L);
}
}
void tma_bw2(torch::Tensor x, torch::Tensor o, int64_t L, int64_t V, int64_t grid) {
  const int nt = (int)(L * L / 128); const uint64_t C = 128; CUtensorMap mx, mo;
  if (V == 0) mx = make_map<4>(x.data_ptr(), {C, (uint64_t)L, (uint64_t)L, 1}, {C, C * L, C * L * L}, {64, 1, 128, 1}, CU_TENSOR_MAP_SWIZZLE_128B, "x0");
  else mx = make_map<4>(x.data_ptr(), {C, (uint64_t)L, (uint64_t)L, 1}, {C, C * L, C * L * L}, {128, 8, 16, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "x1");
  mo = make_map<2>(o.data_ptr(), {C, (uint64_t)(L * L)}, {C}, {64, V == 0 ? 128u : 16u}, CU_TENSOR_MAP_SWIZZLE_128B, "o");
  if (V == 0) run<0>(mx, mo, nt, L, grid); else run<1>(mx, mo, nt, L, grid);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) { mm.def("tma_bw2", &tma_bw2); }
