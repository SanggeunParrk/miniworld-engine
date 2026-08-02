// Tuned pure-CUTLASS plain GEMMs for the ConditionedTransition tail (no cuBLAS anywhere).
//   gemm_nt(A,B,cfg): D = A @ B^T   (A:(M,K) row-major, B:(N,K) row-major)  -- dgrad/forward
//   gemm_tn(A,B,cfg): D = A^T @ B   (A:(M,N) row-major, B:(M,K) row-major -> D:(N,K)) -- wgrad
// Config index selects tile/cluster/schedule (+ StreamK/SplitK). Python picks best per shape.
#include "ct_common.cuh"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

namespace ct {
using cutlass::epilogue::collective::EpilogueTileAuto;
using cutlass::epilogue::collective::EpilogueScheduleAuto;

using CoopK = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using PingK = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpiCoop = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpiPing = cutlass::epilogue::TmaWarpSpecialized;

// ---------- D = A @ B^T : A row-major (M,K), B row-major (N,K) -> B read col-major = B^T -------
template <class Tile, class Clu, class MS, class ES, class Sched>
struct NT {
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OpClass, Tile, Clu, EpilogueTileAuto, ElementAcc, ElementCompute,
      ElementC, cutlass::layout::RowMajor, Align, ElementC, cutlass::layout::RowMajor, Align,
      ES>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass, ElementA, cutlass::layout::RowMajor, Align,
      ElementB, cutlass::layout::ColumnMajor, Align, ElementAcc, Tile, Clu,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>, MS>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi, Sched>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// ---------- D = A^T @ B : A row-major (M,N), B row-major (M,K) -> D:(N,K) (contract over M) ----
// In CUTLASS terms D(N,K) = Aop(N,M) @ Bop(M,K). Aop = A^T: A is (M,N) row-major; reading A as
// ColumnMajor (N,M) gives A^T. Bop = B: B is (M,K) row-major = RowMajor. So LayoutA=ColumnMajor,
// LayoutB=RowMajor, problem (M=N, N=K, K=M_contract).
template <class Tile, class Clu, class MS, class ES, class Sched>
struct TN {
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OpClass, Tile, Clu, EpilogueTileAuto, ElementAcc, ElementCompute,
      ElementC, cutlass::layout::RowMajor, Align, ElementC, cutlass::layout::RowMajor, Align,
      ES>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass, ElementA, cutlass::layout::ColumnMajor, Align,
      ElementB, cutlass::layout::RowMajor, Align, ElementAcc, Tile, Clu,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>, MS>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi, Sched>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <bool IsSK, class G>
static bool run(const float* A, const float* B, float* D, int M, int N, int K,
                at::Device dev, int splits, bool streamk) {
  using SA = typename G::GemmKernel::StrideA; using SB = typename G::GemmKernel::StrideB;
  using SC = typename G::GemmKernel::StrideC; using SD = typename G::GemmKernel::StrideD;
  auto dA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
  auto dB = cutlass::make_cute_packed_stride(SB{}, cute::make_shape(N, K, 1));
  auto dC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
  auto dD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));
  typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    {A, dA, B, dB}, {{}, nullptr, dC, D, dD}};
  args.epilogue.thread.alpha = 1.0f; args.epilogue.thread.beta = 0.0f;
  using DM = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  if constexpr (IsSK) {
    if (splits > 0) { args.scheduler.splits = splits; args.scheduler.decomposition_mode = DM::SplitK; }
    else { args.scheduler.decomposition_mode = DM::StreamK; }
  }
  (void)splits; (void)streamk;
  G gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  size_t ws = G::get_workspace_size(args);
  auto wbuf = torch::empty({(int64_t)ws}, torch::dtype(torch::kUInt8).device(dev));
  if (gemm.initialize(args, wbuf.data_ptr()) != cutlass::Status::kSuccess) return false;
  return gemm.run(at::cuda::getCurrentCUDAStream()) == cutlass::Status::kSuccess;
}

using PS = cutlass::gemm::PersistentScheduler;
using SK = cutlass::gemm::StreamKScheduler;

// NT config table (D=A@B^T): BLOCK_K 32/64, persistent + StreamK + forced splitK.
static bool dispatch_nt(int cfg, const float* A, const float* B, float* D,
                        int M, int N, int Kd, at::Device dev) {
  switch (cfg) {
    case 0: return run<false,typename NT<Shape<_128,_128,_32>,Shape<_1,_1,_1>,CoopK,EpiCoop,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 1: return run<false,typename NT<Shape<_64,_128,_32>,Shape<_1,_1,_1>,PingK,EpiPing,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 2: return run<false,typename NT<Shape<_64,_128,_64>,Shape<_1,_1,_1>,PingK,EpiPing,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 3: return run<false,typename NT<Shape<_128,_256,_32>,Shape<_1,_1,_1>,CoopK,EpiCoop,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 4: return run<false,typename NT<Shape<_256,_128,_32>,Shape<_1,_1,_1>,CoopK,EpiCoop,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 5: return run<false,typename NT<Shape<_128,_128,_64>,Shape<_1,_1,_1>,CoopK,EpiCoop,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 6: return run<true, typename NT<Shape<_128,_128,_32>,Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,0,true);
    case 7: return run<true, typename NT<Shape<_128,_128,_64>,Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,0,true);
    case 8: return run<true, typename NT<Shape<_128,_64,_64>, Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,0,true);
    case 9:  return run<true,typename NT<Shape<_128,_128,_64>,Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,4,true);
    case 10: return run<true,typename NT<Shape<_128,_128,_64>,Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,8,true);
    case 11: return run<true,typename NT<Shape<_128,_128,_32>,Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,16,true);
    default: return false;
  }
}
// TN config table (D=A^T@B wgrad): col-major A transpose requires BLOCK_K (contraction tile)
// EXACTLY 32 (=128B for fp32). Persistent + StreamK + forced splitK over the contraction.
static bool dispatch_tn(int cfg, const float* A, const float* B, float* D,
                        int M, int N, int Kd, at::Device dev) {
  switch (cfg) {
    case 0: return run<false,typename TN<Shape<_128,_128,_32>,Shape<_1,_1,_1>,CoopK,EpiCoop,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 1: return run<false,typename TN<Shape<_64,_128,_32>, Shape<_1,_1,_1>,PingK,EpiPing,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 2: return run<false,typename TN<Shape<_128,_256,_32>,Shape<_1,_1,_1>,CoopK,EpiCoop,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 3: return run<false,typename TN<Shape<_256,_128,_32>,Shape<_1,_1,_1>,CoopK,EpiCoop,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 4: return run<false,typename TN<Shape<_128,_64,_32>, Shape<_1,_1,_1>,CoopK,EpiCoop,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 5: return run<false,typename TN<Shape<_64,_64,_32>,  Shape<_1,_1,_1>,PingK,EpiPing,PS>::Gemm>(A,B,D,M,N,Kd,dev,0,false);
    case 6: return run<true, typename TN<Shape<_128,_128,_32>,Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,0,true);
    case 7: return run<true, typename TN<Shape<_128,_64,_32>, Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,0,true);
    // atom wgrad is split-K bound (tiny output, huge M): high split factors + small tiles for CTA count
    case 8: return run<true, typename TN<Shape<_128,_128,_32>,Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,32,true);
    case 9:  return run<true,typename TN<Shape<_128,_64,_32>, Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,16,true);
    case 10: return run<true,typename TN<Shape<_128,_64,_32>, Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,32,true);
    case 11: return run<true,typename TN<Shape<_128,_64,_32>, Shape<_1,_1,_1>,CoopK,EpiCoop,SK>::Gemm>(A,B,D,M,N,Kd,dev,64,true);
    default: return false;
  }
}
constexpr int NUM_CFG = 12;

torch::Tensor gemm_nt(torch::Tensor A, torch::Tensor B, int64_t cfg) {
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  auto D = torch::empty({M, N}, A.options());
  TORCH_CHECK(dispatch_nt((int)cfg, A.data_ptr<float>(), B.data_ptr<float>(),
              D.data_ptr<float>(), M, N, K, A.device()), "gemm_nt cfg ", cfg, " failed M",M," N",N," K",K);
  return D;
}
// D = A^T @ B : A:(Mc,N), B:(Mc,K) -> (N,K). problem M=N, N=K, contract=Mc.
torch::Tensor gemm_tn(torch::Tensor A, torch::Tensor B, int64_t cfg) {
  A = A.contiguous(); B = B.contiguous();
  int Mc = A.size(0), N = A.size(1), K = B.size(1);
  TORCH_CHECK(B.size(0) == Mc, "gemm_tn contract mismatch");
  auto D = torch::empty({N, K}, A.options());
  TORCH_CHECK(dispatch_tn((int)cfg, A.data_ptr<float>(), B.data_ptr<float>(),
              D.data_ptr<float>(), N, K, Mc, A.device()), "gemm_tn cfg ", cfg, " failed N",N," K",K," Mc",Mc);
  return D;
}
int64_t num_cfg() { return NUM_CFG; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_nt", &gemm_nt, "D = A @ B^T");
  m.def("gemm_tn", &gemm_tn, "D = A^T @ B (wgrad)");
  m.def("num_cfg", &num_cfg);
}

} // namespace ct
