// Per-shape config sweep for the Hopper TF32 WGMMA GEMM (D = A@B^T).
// Instantiates several tile/cluster/schedule configs; python picks best per shape vs cuBLAS.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

using EA = float; using EB = float; using EC = float; using EAcc = float;
using LA = cutlass::layout::RowMajor;
using LB = cutlass::layout::ColumnMajor;
using LC = cutlass::layout::RowMajor;
constexpr int AL = 4;
using Arch = cutlass::arch::Sm90;
using Op   = cutlass::arch::OpClassTensorOp;

template <class Tile, class Cluster, class MainSched, class EpiSched>
struct GemmCfg {
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, Op, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      EAcc, EAcc, EC, LC, AL, EC, LC, AL, EpiSched>::CollectiveOp;
  using CollMain = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, Op, EA, LA, AL, EB, LB, AL, EAcc, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,
      MainSched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollMain, CollEpi>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// StreamK variant: same mainloop/epilogue but StreamK tile scheduler.
template <class Tile, class Cluster, class MainSched, class EpiSched>
struct GemmCfgSK {
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, Op, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      EAcc, EAcc, EC, LC, AL, EC, LC, AL, EpiSched>::CollectiveOp;
  using CollMain = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, Op, EA, LA, AL, EB, LB, AL, EAcc, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,
      MainSched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollMain, CollEpi,
      cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <class Cfg>
static bool run_cfg(const float* A, const float* B, float* D, int M, int N, int K,
                    at::Device dev) {
  using G = typename Cfg::Gemm;
  using SA = typename G::GemmKernel::StrideA;
  using SB = typename G::GemmKernel::StrideB;
  using SC = typename G::GemmKernel::StrideC;
  using SD = typename G::GemmKernel::StrideD;
  auto dA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
  auto dB = cutlass::make_cute_packed_stride(SB{}, cute::make_shape(N, K, 1));
  auto dC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
  auto dD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));
  typename G::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    {A, dA, B, dB}, {{}, nullptr, dC, D, dD}};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta  = 0.0f;
  G gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  size_t ws = G::get_workspace_size(args);
  auto wbuf = torch::empty({(int64_t)ws}, torch::dtype(torch::kUInt8).device(dev));
  if (gemm.initialize(args, wbuf.data_ptr()) != cutlass::Status::kSuccess) return false;
  if (gemm.run(at::cuda::getCurrentCUDAStream()) != cutlass::Status::kSuccess) return false;
  return true;
}

// StreamK runner: decomposition_mode defaults to Heuristic (auto stream-k/split-k).
template <class Cfg>
static bool run_cfg_sk(const float* A, const float* B, float* D, int M, int N, int K,
                       at::Device dev, int splits) {
  using G = typename Cfg::Gemm;
  using SA = typename G::GemmKernel::StrideA;
  using SB = typename G::GemmKernel::StrideB;
  using SC = typename G::GemmKernel::StrideC;
  using SD = typename G::GemmKernel::StrideD;
  auto dA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
  auto dB = cutlass::make_cute_packed_stride(SB{}, cute::make_shape(N, K, 1));
  auto dC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
  auto dD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));
  typename G::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    {A, dA, B, dB}, {{}, nullptr, dC, D, dD}};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta  = 0.0f;
  using DM = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  if (splits > 0) {
    args.scheduler.splits = splits;
    args.scheduler.decomposition_mode = DM::SplitK;
  } else {
    args.scheduler.decomposition_mode = DM::StreamK;
  }
  G gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  size_t ws = G::get_workspace_size(args);
  auto wbuf = torch::empty({(int64_t)ws}, torch::dtype(torch::kUInt8).device(dev));
  if (gemm.initialize(args, wbuf.data_ptr()) != cutlass::Status::kSuccess) return false;
  if (gemm.run(at::cuda::getCurrentCUDAStream()) != cutlass::Status::kSuccess) return false;
  return true;
}

using Coop = cutlass::gemm::collective::KernelScheduleAuto; // builder picks cooperative for TF32
using EpiAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;
using CoopK = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using PingK = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpiCoop = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpiPing = cutlass::epilogue::TmaWarpSpecialized;

// Config table. Index -> a distinct tile/cluster/schedule.
#define CFG(IDX, T0,T1,T2, C0,C1,C2, MS, ES) \
  case IDX: return run_cfg<GemmCfg<Shape<_##T0,_##T1,_##T2>, Shape<_##C0,_##C1,_##C2>, MS, ES>>(A,B,D,M,N,K,dev);
// StreamK / SplitK config (SPL=0 => StreamK auto; SPL>0 => forced split-K).
#define CFGSK(IDX, T0,T1,T2, C0,C1,C2, MS, ES, SPL) \
  case IDX: return run_cfg_sk<GemmCfgSK<Shape<_##T0,_##T1,_##T2>, Shape<_##C0,_##C1,_##C2>, MS, ES>>(A,B,D,M,N,K,dev,SPL);

static bool dispatch(int cfg, const float* A, const float* B, float* D,
                     int M, int N, int K, at::Device dev) {
  switch (cfg) {
    CFG(0, 128,128,32, 1,1,1, CoopK, EpiCoop)
    CFG(1, 128,128,32, 1,2,1, CoopK, EpiCoop)
    CFG(2, 128,128,32, 2,1,1, CoopK, EpiCoop)
    CFG(3, 128,256,32, 1,1,1, CoopK, EpiCoop)
    CFG(4, 128,256,32, 1,2,1, CoopK, EpiCoop)
    CFG(5, 256,128,32, 1,1,1, CoopK, EpiCoop)
    CFG(6, 256,128,32, 2,1,1, CoopK, EpiCoop)
    CFG(7, 64,128,32,  1,1,1, PingK, EpiPing)
    CFG(8, 64,256,32,  1,1,1, PingK, EpiPing)
    CFG(9, 128,128,32, 1,1,1, PingK, EpiPing)
    CFG(10,128,256,32, 1,1,1, PingK, EpiPing)
    CFG(11,64,128,64,  1,1,1, PingK, EpiPing)
    CFG(12,128,128,64, 1,1,1, CoopK, EpiCoop)
    // large-K tuned (token squeeze K=1536, dx K=3072): wider K tile, smaller M, varied N/cluster
    CFG(13,64,128,64,  1,2,1, PingK, EpiPing)
    CFG(14,64,128,64,  2,1,1, PingK, EpiPing)
    CFG(15,128,128,64, 1,2,1, CoopK, EpiCoop)
    CFG(16,128,128,64, 2,1,1, CoopK, EpiCoop)
    CFG(17,128,256,32, 1,1,1, CoopK, EpiCoop)
    CFG(18,256,128,32, 1,1,1, CoopK, EpiCoop)
    CFG(19,128,128,32, 2,1,1, CoopK, EpiCoop)
    CFG(20,256,128,32, 1,1,1, CoopK, EpiCoop)
    CFG(21,128,128,32, 1,1,1, PingK, EpiPing)
    CFG(22,128,256,32, 1,1,1, PingK, EpiPing)
    // StreamK (auto) — canonical for large-K small-M token squeeze/dx (cooperative only)
    CFGSK(23,128,128,32, 1,1,1, CoopK, EpiCoop, 0)
    CFGSK(24,128,128,64, 1,1,1, CoopK, EpiCoop, 0)
    CFGSK(25,128,64,64,  1,1,1, CoopK, EpiCoop, 0)
    CFGSK(26,128,256,32, 1,1,1, CoopK, EpiCoop, 0)
    CFGSK(27,128,64,32,  1,1,1, CoopK, EpiCoop, 0)
    // forced Split-K (cooperative)
    CFGSK(28,128,128,32, 1,1,1, CoopK, EpiCoop, 2)
    CFGSK(29,128,128,64, 1,1,1, CoopK, EpiCoop, 4)
    CFGSK(30,128,64,64,  1,1,1, CoopK, EpiCoop, 4)
    default: return false;
  }
}

constexpr int NUM_CFG = 31;

torch::Tensor gemm_cfg(torch::Tensor A, torch::Tensor B, int64_t cfg) {
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  auto D = torch::empty({M, N}, A.options());
  bool ok = dispatch((int)cfg, A.data_ptr<float>(), B.data_ptr<float>(),
                     D.data_ptr<float>(), M, N, K, A.device());
  TORCH_CHECK(ok, "config ", cfg, " failed/unimplemented for shape M=", M, " N=", N, " K=", K);
  return D;
}

int64_t num_cfg() { return NUM_CFG; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_cfg", &gemm_cfg, "D=A@B^T with config index");
  m.def("num_cfg", &num_cfg, "number of configs");
}
