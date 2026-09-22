// CUTLASS SM90 TF32 GEMM sweep with K-tile=128 (canonical CUTLASS test config) + cluster<1,2,1>.
// D = A @ W^T, fp32 in/out, tf32. Kernel-only timing (setup excluded).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;
using EA=float; using EB=float; using EC=float; using EAcc=float;
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
double bench_run(torch::Tensor A, torch::Tensor W, torch::Tensor D, int warmup, int iters){
  int M=A.size(0),Kd=A.size(1),N=W.size(0);
  using SA=typename Gemm::GemmKernel::StrideA; using SB=typename Gemm::GemmKernel::StrideB; using SC=typename Gemm::GemmKernel::StrideC;
  auto sA=cutlass::make_cute_packed_stride(SA{},cute::make_shape(M,Kd,1));
  auto sB=cutlass::make_cute_packed_stride(SB{},cute::make_shape(N,Kd,1));
  auto sC=cutlass::make_cute_packed_stride(SC{},cute::make_shape(M,N,1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,Kd},
    {A.data_ptr<float>(),sA,W.data_ptr<float>(),sB},
    {{1.0f,0.0f},D.data_ptr<float>(),sC,D.data_ptr<float>(),sC}};
  Gemm g; size_t ws=Gemm::get_workspace_size(args);
  auto wsp=torch::empty({(long)ws},A.options().dtype(torch::kUInt8));
  if(g.can_implement(args)!=cutlass::Status::kSuccess) return -1.0;
  if(g.initialize(args,wsp.data_ptr())!=cutlass::Status::kSuccess) return -2.0;
  auto stream=at::cuda::getCurrentCUDAStream();
  for(int i=0;i<warmup;i++) g.run(stream);
  cudaEvent_t b,e; cudaEventCreate(&b); cudaEventCreate(&e);
  cudaStreamSynchronize(stream); cudaEventRecord(b,stream);
  for(int i=0;i<iters;i++) g.run(stream);
  cudaEventRecord(e,stream); cudaEventSynchronize(e);
  float ms=0; cudaEventElapsedTime(&ms,b,e); cudaEventDestroy(b); cudaEventDestroy(e);
  return (double)ms/iters*1000.0;
}

torch::Tensor run_one(torch::Tensor A, torch::Tensor W, int cfg);  // fwd decl for correctness

double bench_cfg(torch::Tensor A, torch::Tensor W, int cfg, int warmup, int iters){
  A=A.contiguous();W=W.contiguous();
  auto D=torch::empty({A.size(0),W.size(0)},A.options());
  switch(cfg){
    case 0: return bench_run<G<Shape<_64,_128,_128>,Shape<_1,_2,_1>>::Gemm>(A,W,D,warmup,iters);  // CUTLASS Test 1 (K128)
    case 1: return bench_run<G<Shape<_64,_128,_128>,Shape<_1,_1,_1>>::Gemm>(A,W,D,warmup,iters);
    case 2: return bench_run<G<Shape<_128,_128,_64>,Shape<_1,_2,_1>>::Gemm>(A,W,D,warmup,iters);
    case 3: return bench_run<G<Shape<_128,_128,_64>,Shape<_1,_1,_1>>::Gemm>(A,W,D,warmup,iters);
    case 4: return bench_run<G<Shape<_128,_256,_64>,Shape<_1,_2,_1>>::Gemm>(A,W,D,warmup,iters);
    case 5: return bench_run<G<Shape<_256,_128,_64>,Shape<_1,_2,_1>>::Gemm>(A,W,D,warmup,iters);
    case 6: return bench_run<G<Shape<_64,_256,_64>,Shape<_1,_1,_1>>::Gemm>(A,W,D,warmup,iters);
    case 7: return bench_run<G<Shape<_64,_128,_64>,Shape<_1,_2,_1>>::Gemm>(A,W,D,warmup,iters);
    default: return -9;
  }
}
int n_cfg(){return 8;}

// correctness: run cfg0 and return D
torch::Tensor gemm_ref(torch::Tensor A, torch::Tensor W){
  A=A.contiguous();W=W.contiguous(); auto D=torch::empty({A.size(0),W.size(0)},A.options());
  bench_run<G<Shape<_64,_128,_128>,Shape<_1,_2,_1>>::Gemm>(A,W,D,0,1); return D;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("bench",&bench_cfg);m.def("n_cfg",&n_cfg);m.def("gemm_ref",&gemm_ref);}
