// Minimal CUTLASS SM90 TF32 WGMMA GEMM (build de-risk): D = A @ W^T, fp32 in/out, tf32 tensor core.
// A (M,K) row-major, W (N,K) row-major (nn.Linear layout) → D (M,N) row-major.
// ElementA=ElementB=float on Sm90 OpClassTensorOp ⇒ implicit TF32 (example 48), fp32 accumulate.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAcc = float;

using LayoutA = cutlass::layout::RowMajor;     // A (M,K)
using LayoutB = cutlass::layout::ColumnMajor;  // B = W^T (K,N) col-major == W (N,K) row-major
using LayoutC = cutlass::layout::RowMajor;     // D (M,N)

constexpr int AlignA = 4;  // 128/32
constexpr int AlignB = 4;
constexpr int AlignC = 4;

using TileShape = Shape<_128, _128, _32>;
using ClusterShape = Shape<_1, _2, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutC, AlignC,
    ElementC, LayoutC, AlignC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

torch::Tensor gemm_tf32(torch::Tensor A, torch::Tensor W) {
  // A (M,K) row-major fp32, W (N,K) row-major fp32 → D (M,N) = A @ W^T
  TORCH_CHECK(A.is_cuda() && W.is_cuda(), "cuda only");
  TORCH_CHECK(A.scalar_type() == torch::kFloat32, "fp32 only");
  A = A.contiguous();
  W = W.contiguous();
  int M = A.size(0), K = A.size(1), N = W.size(0);
  auto D = torch::empty({M, N}, A.options());

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {reinterpret_cast<ElementA*>(A.data_ptr<float>()), sA,
       reinterpret_cast<ElementB*>(W.data_ptr<float>()), sB},
      {{1.0f, 0.0f},
       reinterpret_cast<ElementC*>(D.data_ptr<float>()), sC,
       reinterpret_cast<ElementC*>(D.data_ptr<float>()), sD}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(long)ws}, A.options().dtype(torch::kUInt8));
  auto st = gemm.can_implement(args);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "can_implement failed");
  st = gemm.initialize(args, workspace.data_ptr());
  TORCH_CHECK(st == cutlass::Status::kSuccess, "initialize failed");
  st = gemm.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(st == cutlass::Status::kSuccess, "run failed");
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_tf32", &gemm_tf32, "CUTLASS SM90 TF32 GEMM D=A@W^T");
}
