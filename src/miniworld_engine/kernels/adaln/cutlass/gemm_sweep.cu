// Sweep CUTLASS SM90 TF32 GEMM configs (tile×cluster) to find a cuBLAS-matching one.
// D = A @ W^T, A (M,K) row-major, W (N,K) row-major, fp32 in/out, tf32 wgmma.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;
using EA=cutlass::tfloat32_t; using EB=cutlass::tfloat32_t; using EC=float; using EAcc=float;
using LA=cutlass::layout::RowMajor; using LB=cutlass::layout::ColumnMajor; using LC=cutlass::layout::RowMajor;
constexpr int AL=4;

template<class Tile, class Cluster>
struct G {
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, EAcc, EAcc,
      EC, LC, AL, EC, LC, AL, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, EA, LA, AL, EB, LB, AL, EAcc, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using K = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int>, Main, Epi>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<K>;
};

template<class Gemm>
void run(torch::Tensor A, torch::Tensor W, torch::Tensor D){
  int M=A.size(0),Kd=A.size(1),N=W.size(0);
  using SA=typename Gemm::GemmKernel::StrideA; using SB=typename Gemm::GemmKernel::StrideB; using SC=typename Gemm::GemmKernel::StrideC;
  auto sA=cutlass::make_cute_packed_stride(SA{},cute::make_shape(M,Kd,1));
  auto sB=cutlass::make_cute_packed_stride(SB{},cute::make_shape(N,Kd,1));
  auto sC=cutlass::make_cute_packed_stride(SC{},cute::make_shape(M,N,1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,Kd},
    {reinterpret_cast<EA*>(A.data_ptr<float>()),sA,reinterpret_cast<EB*>(W.data_ptr<float>()),sB},
    {{1.0f,0.0f},D.data_ptr<float>(),sC,D.data_ptr<float>(),sC}};
  Gemm g; size_t ws=Gemm::get_workspace_size(args);
  auto wsp=torch::empty({(long)ws},A.options().dtype(torch::kUInt8));
  TORCH_CHECK(g.can_implement(args)==cutlass::Status::kSuccess,"can_implement");
  TORCH_CHECK(g.initialize(args,wsp.data_ptr())==cutlass::Status::kSuccess,"init");
  TORCH_CHECK(g.run(at::cuda::getCurrentCUDAStream())==cutlass::Status::kSuccess,"run");
}

// Kernel-only timing: setup ONCE, time `iters` run() with cudaEvents (excludes host setup).
template<class Gemm>
double bench_run(torch::Tensor A, torch::Tensor W, torch::Tensor D, int warmup, int iters){
  int M=A.size(0),Kd=A.size(1),N=W.size(0);
  using SA=typename Gemm::GemmKernel::StrideA; using SB=typename Gemm::GemmKernel::StrideB; using SC=typename Gemm::GemmKernel::StrideC;
  auto sA=cutlass::make_cute_packed_stride(SA{},cute::make_shape(M,Kd,1));
  auto sB=cutlass::make_cute_packed_stride(SB{},cute::make_shape(N,Kd,1));
  auto sC=cutlass::make_cute_packed_stride(SC{},cute::make_shape(M,N,1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,Kd},
    {reinterpret_cast<EA*>(A.data_ptr<float>()),sA,reinterpret_cast<EB*>(W.data_ptr<float>()),sB},
    {{1.0f,0.0f},D.data_ptr<float>(),sC,D.data_ptr<float>(),sC}};
  Gemm g; size_t ws=Gemm::get_workspace_size(args);
  auto wsp=torch::empty({(long)ws},A.options().dtype(torch::kUInt8));
  TORCH_CHECK(g.can_implement(args)==cutlass::Status::kSuccess,"can_implement");
  TORCH_CHECK(g.initialize(args,wsp.data_ptr())==cutlass::Status::kSuccess,"init");
  auto stream=at::cuda::getCurrentCUDAStream();
  for(int i=0;i<warmup;i++) g.run(stream);
  cudaEvent_t b,e; cudaEventCreate(&b); cudaEventCreate(&e);
  cudaStreamSynchronize(stream); cudaEventRecord(b,stream);
  for(int i=0;i<iters;i++) g.run(stream);
  cudaEventRecord(e,stream); cudaEventSynchronize(e);
  float ms=0; cudaEventElapsedTime(&ms,b,e);
  cudaEventDestroy(b); cudaEventDestroy(e);
  return (double)ms/iters*1000.0; // us
}
double bench_cfg(torch::Tensor A, torch::Tensor W, int cfg, int warmup, int iters){
  A=A.contiguous();W=W.contiguous();
  auto D=torch::empty({A.size(0),W.size(0)},A.options());
  switch(cfg){
    case 0: return bench_run<G<Shape<_128,_128,_32>,Shape<_1,_1,_1>>::Gemm>(A,W,D,warmup,iters);
    case 1: return bench_run<G<Shape<_128,_128,_64>,Shape<_1,_1,_1>>::Gemm>(A,W,D,warmup,iters);
    case 2: return bench_run<G<Shape<_128,_256,_64>,Shape<_1,_1,_1>>::Gemm>(A,W,D,warmup,iters);
    case 3: return bench_run<G<Shape<_256,_128,_64>,Shape<_1,_1,_1>>::Gemm>(A,W,D,warmup,iters);
    case 4: return bench_run<G<Shape<_128,_128,_32>,Shape<_1,_2,_1>>::Gemm>(A,W,D,warmup,iters);
    case 5: return bench_run<G<Shape<_128,_256,_32>,Shape<_2,_1,_1>>::Gemm>(A,W,D,warmup,iters);
    case 6: return bench_run<G<Shape<_64,_128,_64>,Shape<_1,_1,_1>>::Gemm>(A,W,D,warmup,iters);
    case 7: return bench_run<G<Shape<_128,_64,_64>,Shape<_2,_1,_1>>::Gemm>(A,W,D,warmup,iters);
    default: TORCH_CHECK(false,"bad cfg"); return 0;
  }
}

torch::Tensor run_cfg(torch::Tensor A, torch::Tensor W, int cfg){
  A=A.contiguous();W=W.contiguous();
  auto D=torch::empty({A.size(0),W.size(0)},A.options());
  switch(cfg){
    case 0: run<G<Shape<_128,_128,_32>,Shape<_1,_1,_1>>::Gemm>(A,W,D); break;
    case 1: run<G<Shape<_128,_128,_64>,Shape<_1,_1,_1>>::Gemm>(A,W,D); break;
    case 2: run<G<Shape<_128,_256,_64>,Shape<_1,_1,_1>>::Gemm>(A,W,D); break;
    case 3: run<G<Shape<_256,_128,_64>,Shape<_1,_1,_1>>::Gemm>(A,W,D); break;
    case 4: run<G<Shape<_128,_128,_32>,Shape<_1,_2,_1>>::Gemm>(A,W,D); break;
    case 5: run<G<Shape<_128,_256,_32>,Shape<_2,_1,_1>>::Gemm>(A,W,D); break;
    case 6: run<G<Shape<_64,_128,_64>,Shape<_1,_1,_1>>::Gemm>(A,W,D); break;
    case 7: run<G<Shape<_128,_64,_64>,Shape<_2,_1,_1>>::Gemm>(A,W,D); break;
    default: TORCH_CHECK(false,"bad cfg");
  }
  return D;
}
int n_cfg(){return 8;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gemm",&run_cfg);m.def("bench",&bench_cfg);m.def("n_cfg",&n_cfg);}
