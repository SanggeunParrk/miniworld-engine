// dump the raw shared-memory image of one TMA box
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;
__global__ void dump(const __grid_constant__ CUtensorMap m, int nbytes, uint8_t* out, int c1, int c2, int c3) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __shared__ uint64_t bar;
  if (threadIdx.x == 0) { bar_init(&bar, 1); bar_init_fence(); }
  __syncthreads();
  if (threadIdx.x == 0) { expect_tx(&bar, nbytes); load_4d(&m, smb, &bar, 0, c1, c2, c3); }
  wait(&bar, 0);
  for (int i = threadIdx.x; i < nbytes; i += blockDim.x) out[i] = smb[i];
}
torch::Tensor run(torch::Tensor O, int64_t ni, int64_t nj, int64_t sw) {
  const uint64_t M = nj * 32;
  // dims (e, j, i, c): box (32, 32, 4, 2) -> smem [c][i][j][e]: two [128 rows][32] tiles
  CUtensorMap am = make_map<4>(O.data_ptr(), {32, (uint64_t)nj, (uint64_t)ni, 32}, {32, 32 * M, M}, {32, 32, 4, 2},
                               sw ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_NONE, "O");
  auto out = torch::empty({16384}, O.options().dtype(torch::kUInt8));
  cudaFuncSetAttribute(dump, cudaFuncAttributeMaxDynamicSharedMemorySize, 20000);
  dump<<<1, 128, 20000, at::cuda::getCurrentCUDAStream()>>>(am, 16384, out.data_ptr<uint8_t>(), 0, 0, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
