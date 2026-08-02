// CUTLASS SM90 TF32 fused adaLN forward (fp32 IO, tf32 WGMMA, fused gate epilogue).
//
// Two GEMMs over cond_aff = LN(cond)*lnw (materialized in triton; avoids epilogue LN of x which
// would need full-N subtile, infeasible for d>256):
//   fwd1: y    = sigmoid(cond_aff @ Ws^T + scale_b[n]) * x_hat[m,n]   (x_hat via C operand)
//   fwd2: y   += cond_aff @ Wb^T                                       (residual add, beta=1, C=y)
// scale_b is a per-column (length-N) vector; x_hat and y_prev are full (M,N) tensors passed as the
// standard C source operand (Sm90SrcFetch) — no Sm90AuxLoad needed.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace fusion = cutlass::epilogue::fusion;

using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAcc = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;     // cond_aff (M,K)
using LayoutB = cutlass::layout::ColumnMajor;  // W (N,K) row-major == W^T (K,N) col-major
using LayoutC = cutlass::layout::RowMajor;     // C/D (M,N)
constexpr int AlignA = 4, AlignB = 4, AlignC = 4;
constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

using TileShape = Shape<_128, _128, _32>;
using ClusterShape = Shape<_1, _1, _1>;
using EpiSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using MainSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

// ── EVT for fwd1: D = sigmoid(acc + scale_b[n]) * C   (C = x_hat) ────────────────────────────
using EVTSigmoid = fusion::Sm90EVT<
    fusion::Sm90Compute<cutlass::multiplies, ElementC, ElementCompute, RoundStyle>,  // (*) C
    fusion::Sm90EVT<
        fusion::Sm90Compute<cutlass::epilogue::thread::Sigmoid, ElementCompute, ElementCompute, RoundStyle>,
        fusion::Sm90EVT<
            fusion::Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, RoundStyle>,
            fusion::Sm90AccFetch,
            fusion::Sm90RowBroadcast<0, TileShape, ElementCompute>>>,
    fusion::Sm90SrcFetch<ElementC>>;

template <class FusionOp>
struct GemmT {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementC, LayoutC, AlignC,
      EpiSchedule, FusionOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainSchedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Gemm1 = GemmT<EVTSigmoid>::Gemm;
// fwd2: standard linear combination D = alpha*acc + beta*C
using LinComb = fusion::LinearCombination<ElementC, ElementCompute, ElementC, ElementCompute, RoundStyle>;
using Gemm2 = GemmT<LinComb>::Gemm;

template <class Gemm>
static void run_gemm(typename Gemm::Arguments& args, torch::Tensor ref) {
  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(long)ws}, ref.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "can_implement failed");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess, "initialize failed");
  TORCH_CHECK(gemm.run(at::cuda::getCurrentCUDAStream()) == cutlass::Status::kSuccess, "run failed");
}

// y = sigmoid(cond_aff @ Ws^T + scale_b) * x_hat
torch::Tensor adaln_fwd1(torch::Tensor cond_aff, torch::Tensor Ws, torch::Tensor scale_b, torch::Tensor x_hat) {
  cond_aff = cond_aff.contiguous(); Ws = Ws.contiguous(); x_hat = x_hat.contiguous(); scale_b = scale_b.contiguous();
  int M = cond_aff.size(0), K = cond_aff.size(1), N = Ws.size(0);
  auto y = torch::empty({M, N}, cond_aff.options());
  using S = typename Gemm1::GemmKernel::StrideA;
  using SB = typename Gemm1::GemmKernel::StrideB;
  using SC = typename Gemm1::GemmKernel::StrideC;
  auto sA = cutlass::make_cute_packed_stride(S{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(SB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
  typename Gemm1::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {cond_aff.data_ptr<float>(), sA, Ws.data_ptr<float>(), sB},
      {{},  // fusion args filled below
       x_hat.data_ptr<float>(), sC, y.data_ptr<float>(), sC}};
  // EVT args (tree depth-first): multiply{ sigmoid{ plus{ acc{}, rowbcast{scale_b} } }, srcfetch{} }
  args.epilogue.thread = {
      {                                  // sigmoid( plus(acc, scale_b) )
          {                              // plus(acc, scale_b)
              {},                        // AccFetch
              {scale_b.data_ptr<float>(), ElementCompute(0)},  // Sm90RowBroadcast args
              {}                         // plus compute
          },
          {}                             // sigmoid compute
      },
      {},                                // Sm90SrcFetch (C = x_hat) — no args
      {}                                 // multiply compute
  };
  run_gemm<Gemm1>(args, cond_aff);
  return y;
}

// y_out = cond_aff @ Wb^T + y_prev
torch::Tensor adaln_fwd2(torch::Tensor cond_aff, torch::Tensor Wb, torch::Tensor y_prev) {
  cond_aff = cond_aff.contiguous(); Wb = Wb.contiguous(); y_prev = y_prev.contiguous();
  int M = cond_aff.size(0), K = cond_aff.size(1), N = Wb.size(0);
  auto y = torch::empty({M, N}, cond_aff.options());
  using S = typename Gemm2::GemmKernel::StrideA;
  using SB = typename Gemm2::GemmKernel::StrideB;
  using SC = typename Gemm2::GemmKernel::StrideC;
  auto sA = cutlass::make_cute_packed_stride(S{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(SB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
  typename Gemm2::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {cond_aff.data_ptr<float>(), sA, Wb.data_ptr<float>(), sB},
      {{1.0f, 1.0f}, y_prev.data_ptr<float>(), sC, y.data_ptr<float>(), sC}};
  run_gemm<Gemm2>(args, cond_aff);
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adaln_fwd1", &adaln_fwd1, "y = sigmoid(cond_aff@Ws^T + scale_b) * x_hat");
  m.def("adaln_fwd2", &adaln_fwd2, "y = cond_aff@Wb^T + y_prev");
}
