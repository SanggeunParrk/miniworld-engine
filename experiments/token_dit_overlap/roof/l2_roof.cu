// The pattern floor for the attention core's loads: how fast this card can pull L2-resident tiles through TMA, with the
// same shapes and stage count the core uses. Measured, not assumed, because "% of SoL" means nothing without it.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

constexpr int TILE_ROWS = 64, TILE_COLS = 64, TILE_B = TILE_ROWS * TILE_COLS * 2, ST = 3;

__global__ void __launch_bounds__(160, 2)
roof_kernel(const __grid_constant__ CUtensorMap map, int iters, int rows, float* __restrict__ sink) {
  extern __shared__ __align__(1024) uint8_t sm[];
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + ST * TILE_B);
  uint64_t* empty = full + ST;
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 4); }   // four consumer warps
    fence_barrier_init();
  }
  __syncthreads();
  const int rblocks = rows / TILE_ROWS;
  if (tid >= 128) {                                                 // producer warp, one issuer
    if (tid == 128) {
      for (int n = 0; n < iters; ++n) {
        const int s = n % ST;
        mbar_wait(&empty[s], ((n / ST) & 1) ^ 1);
        mbar_arrive_expect_tx(&full[s], TILE_B);
        tma_load_2d(sm + s * TILE_B, &map, &full[s], 0, ((blockIdx.x + n * 37) % rblocks) * TILE_ROWS);
      }
    }
    return;
  }
  float acc = 0.f;
  for (int n = 0; n < iters; ++n) {
    const int s = n % ST;
    mbar_wait(&full[s], (n / ST) & 1);
    acc += __int_as_float(*reinterpret_cast<const int*>(sm + s * TILE_B + ((tid & 31) * 4)));
    if ((tid & 31) == 0) mbar_arrive(&empty[s]);
  }
  if (acc == 1234.5f) sink[0] = acc;
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <array>
#include <map>
#include <mutex>
namespace {
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr; cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p, "no TMA");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}
}  // namespace

void roof(torch::Tensor t, int64_t iters, int64_t ctas) {
  CUtensorMap map{};
  const cuuint64_t dims[2] = {(cuuint64_t)t.size(1), (cuuint64_t)t.size(0)};
  const cuuint64_t strides[1] = {(cuuint64_t)t.size(1) * 2};
  const cuuint32_t box[2] = {TILE_COLS, TILE_ROWS}, elem[2] = {1, 1};
  TORCH_CHECK(encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, elem,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "encode failed");
  const size_t smem = ST * TILE_B + 256;
  cudaFuncSetAttribute(roof_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  auto sink = torch::zeros({1}, t.options().dtype(torch::kFloat32));
  roof_kernel<<<(int)ctas, 160, smem, at::cuda::getCurrentCUDAStream()>>>(map, (int)iters, (int)t.size(0),
                                                                          sink.data_ptr<float>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("roof", &roof); }
