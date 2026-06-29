// Minimal Hopper TF32 WGMMA GEMM gate (CUTLASS 3.x CollectiveBuilder).
// Computes D = A @ B^T  with A:(M,K) row-major, B:(N,K) row-major (i.e. B^T is K x N),
// D:(M,N) row-major.  fp32 in/out, tfloat32 operands, fp32 accumulate.
// This is the x @ W^T form used everywhere in the transition tail.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// A:(M,K) row-major  -> LayoutA RowMajor
// We want D = A @ B^T where B is stored (N,K) row-major.  B^T is (K,N).
// In CUTLASS terms the GEMM computes D(M,N) = A(M,K) * Bop(K,N).
// Bop = B^T.  If B is (N,K) row-major then B^T is (K,N) and reading B as
// ColumnMajor (K,N) gives exactly B^T.  So LayoutB = ColumnMajor with B's (N,K) buffer
// reinterpreted: a (K,N) column-major matrix has stride (1,K) == B's (N,K) row-major
// stride viewed transposed.  Net: ElementB col-major over the (N,K) buffer == B^T.
using ElementA = float;
using LayoutA  = cutlass::layout::RowMajor;
constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = float;
using LayoutB  = cutlass::layout::ColumnMajor;   // gives B^T from an (N,K) row-major buffer
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = float;
using LayoutC  = cutlass::layout::RowMajor;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAcc = float;
using ArchTag    = cutlass::arch::Sm90;
using OpClass    = cutlass::arch::OpClassTensorOp;
using TileShape    = Shape<_128,_128,_32>;
using ClusterShape = Shape<_1,_2,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutC, AlignC,
    ElementC, LayoutC, AlignC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// D = A @ B^T ; A:(M,K) row-major, B:(N,K) row-major, D:(M,N) row-major.
torch::Tensor gate_gemm(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "fp32 only");
  A = A.contiguous();
  B = B.contiguous();
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "K mismatch");

  auto D = torch::empty({M, N}, A.options());

  StrideA dA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB dB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC dC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD dD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* pA = reinterpret_cast<ElementA const*>(A.data_ptr<float>());
  auto* pB = reinterpret_cast<ElementB const*>(B.data_ptr<float>());
  auto* pD = reinterpret_cast<ElementC*>(D.data_ptr<float>());

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {pA, dA, pB, dB},
    {{}, nullptr, dC, pD, dD}
  };
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta  = 0.0f;

  Gemm gemm;
  auto stream = at::cuda::getCurrentCUDAStream();

  size_t workspace_size = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)workspace_size},
                                torch::dtype(torch::kUInt8).device(A.device()));

  cutlass::Status status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "cutlass can_implement failed: ", cutlass::cutlassGetStatusString(status));
  status = gemm.initialize(args, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "cutlass initialize failed: ", cutlass::cutlassGetStatusString(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "cutlass run failed: ", cutlass::cutlassGetStatusString(status));
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate_gemm", &gate_gemm, "D = A @ B^T (Hopper TF32 WGMMA)");
}
