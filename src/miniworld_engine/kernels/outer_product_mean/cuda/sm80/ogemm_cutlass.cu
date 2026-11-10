// The OuterProductMean's grouped outer product on A100 with the dead tokens skipped: O[(i, c), (j, e)] = sum_s a[s, i, c] b[s, j, e], i.e. O = A^T B with
// A, B [S, 32 L] (the prologue's s-major operands), on a CUTLASS 2.x sm_80 GEMM (128 x 128 x 32 tiles, 64 x 64 warps, 4-stage cp.async pipeline: 0.96 of
// cuBLAS on the bare product).  A token whose mask column is all zero (padding: masked in every MSA row) has a = b = 0, so every O block of a pair with
// such a token is exactly 0: a threadblock whose 4 row tokens (or 4 column tokens) are all dead writes its zero tile instead of running the K loop.  The
// liveness comes from the device ([L] uint8), so the launch is shape-static (CUDA graphs) and the result is the dense product's, bit for bit.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

namespace {
using bf = cutlass::bfloat16_t;
constexpr int TM = 128, TN = 128, TK = 32, CH = 32;
using Gemm = cutlass::gemm::device::Gemm<bf, cutlass::layout::ColumnMajor, bf, cutlass::layout::RowMajor, bf, cutlass::layout::RowMajor, float,
                                         cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TM, TN, TK>,
                                         cutlass::gemm::GemmShape<64, 64, TK>, cutlass::gemm::GemmShape<16, 8, 16>,
                                         cutlass::epilogue::thread::LinearCombination<bf, 8, float, float>,
                                         cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4>;
using GK = Gemm::GemmKernel;

__global__ void __launch_bounds__(GK::kThreadCount) ogemm_skip_kernel(typename GK::Params params, const uint8_t* __restrict__ live, bf* __restrict__ out,
                                                                     int ldo) {
  extern __shared__ __align__(16) unsigned char smem[];
  typename GK::ThreadblockSwizzle sw;
  const cutlass::gemm::GemmCoord tile = sw.get_tile_offset(params.swizzle_log_tile);
  if (params.grid_tiled_shape.m() <= tile.m() || params.grid_tiled_shape.n() <= tile.n()) return;
  const int t0 = tile.m() * (TM / CH), u0 = tile.n() * (TN / CH);
  const uint32_t lm = *reinterpret_cast<const uint32_t*>(live + t0), ln = *reinterpret_cast<const uint32_t*>(live + u0);   // 4 tokens each
  if (lm != 0u && ln != 0u) {
    GK op;
    op(params, *reinterpret_cast<typename GK::SharedStorage*>(smem));
    return;
  }
  // a dead tile: zeros (128 rows x 256 B)
  const int r0 = tile.m() * TM, c0 = tile.n() * TN;
  for (int v = threadIdx.x; v < TM * (TN / 8); v += GK::kThreadCount) {
    const int r = v / (TN / 8), c = v % (TN / 8);
    *reinterpret_cast<uint4*>(out + (long)(r0 + r) * ldo + c0 + c * 8) = make_uint4(0, 0, 0, 0);
  }
}
}  // namespace

// a, b [S, 32 L] bf16 (row-major), live [L] uint8 (0 = a token masked in every MSA row), out [32 L, 32 L] bf16 = a^T b
void ogemm_skip(torch::Tensor a, torch::Tensor b, torch::Tensor live, torch::Tensor out) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 && a.is_contiguous() && b.sizes() == a.sizes() && b.is_contiguous() && b.scalar_type() == at::kBFloat16,
              "ogemm_skip: a / b contiguous bf16 [S, 32 L]");
  const int K = (int)a.size(0), M = (int)a.size(1);
  TORCH_CHECK(M % TM == 0 && K % 8 == 0, "ogemm_skip: 32 L a multiple of 128 (L % 4), S % 8");
  TORCH_CHECK(live.is_cuda() && live.scalar_type() == at::kByte && live.is_contiguous() && live.numel() == M / CH && reinterpret_cast<uintptr_t>(live.data_ptr()) % 4 == 0,
              "ogemm_skip: live [L] uint8, 4-byte aligned");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == M && out.size(1) == M, "ogemm_skip: out [32 L, 32 L] bf16");
  const at::cuda::CUDAGuard guard(a.device());
  bf* A = reinterpret_cast<bf*>(a.data_ptr());
  bf* B = reinterpret_cast<bf*>(b.data_ptr());
  bf* C = reinterpret_cast<bf*>(out.data_ptr());
  const cutlass::gemm::GemmCoord problem(M, M, K);
  typename GK::ThreadblockSwizzle sw;
  const cutlass::gemm::GemmCoord grid_shape = sw.get_tiled_shape(problem, {TM, TN, TK}, 1);
  typename GK::Params params(problem, grid_shape, {A, M}, {B, M}, {C, M}, {C, M}, {1.f, 0.f});
  const dim3 grid = sw.get_grid_shape(grid_shape);
  const int smem = (int)sizeof(typename GK::SharedStorage);
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ogemm_skip_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured = true;
  }
  ogemm_skip_kernel<<<grid, GK::kThreadCount, smem, at::cuda::getCurrentCUDAStream()>>>(params, reinterpret_cast<const uint8_t*>(live.data_ptr()), C, M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("ogemm_skip", &ogemm_skip); }
