// B200-style grouped projection dgrad + LN backward + residual on Hopper.
// A two-slot TMA ring feeds WGMMA. Rounded dZ remains in shared memory;
// no FP32 projection dgrad tensor is written to HBM.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include "fa3_utils.h"
using namespace cute;
using Element=cutlass::bfloat16_t;
__device__ float sum_warp(float x) {
  #pragma unroll
  for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);
  return x;
}
template<int C> struct Config {
  static constexpr int N=C==64?64:128;
  using AL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));
  using BL=decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{},Shape<Int<N>,_64>{}));
  using MMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,Int<N>,_64>,GMMA::Major::K,GMMA::Major::MN>()));
  using GS=Shape<int32_t,int32_t>;
  using AS=Stride<int64_t,_1>;
  using BS=Stride<_1,int64_t>;
  using TA=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GS{},AS{}),AL{},Shape<_64,_64>{},_1{}));
  using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GS{},BS{}),BL{},Shape<Int<N>,_64>{},_1{}));
  struct Shared {
    array_aligned<Element,4096,1024> a[2];
    array_aligned<Element,N*64,1024> b[2];
    array_aligned<Element,64*C,1024> dz;
    cutlass::arch::ClusterTransactionBarrier full[2];
  };
  struct Params {
    TA a;TB b;Element const* x;Element const* dy;float const* stats;float const* lw;float const* lb;
    Element* dx;Element* z;float* parts;int rows;
  };
};
template<int C> __global__ __launch_bounds__(128)
void projection_ln(CUTE_GRID_CONSTANT typename Config<C>::Params const p) {
  using T=Config<C>;extern __shared__ char memory[];auto& s=*reinterpret_cast<typename T::Shared*>(memory);
  int tid=threadIdx.x,warp=tid/32,lane=tid%32;constexpr int K=4*C+64;
  if(tid==0){s.full[0].init(1);s.full[1].init(1);prefetch_tma_descriptor(p.a.get_tma_descriptor());prefetch_tma_descriptor(p.b.get_tma_descriptor());cutlass::arch::fence_barrier_init();}
  __syncthreads();
  auto ag=p.a.get_tma_tensor(make_shape(p.rows,K));auto bg=p.b.get_tma_tensor(make_shape(C,K));
  typename T::MMA mma;auto thr=mma.get_slice(tid);
  auto coord=thr.partition_C(make_identity_tensor(Shape<_64,Int<T::N>>{}));
  for(int nb=0;nb<C/T::N;++nb) {
    auto load=[&](int kt) {
      int slot=kt%2;s.full[slot].arrive_and_expect_tx((4096+T::N*64)*sizeof(Element));
      auto at=local_tile(ag,Shape<_64,_64>{},make_coord(blockIdx.x,kt));
      auto bt=local_tile(bg,Shape<Int<T::N>,_64>{},make_coord(nb,kt));
      auto sa=make_tensor(make_smem_ptr(s.a[slot].data()),typename T::AL{});
      auto sb=make_tensor(make_smem_ptr(s.b[slot].data()),typename T::BL{});
      auto ac=p.a.get_slice(_0{});auto bc=p.b.get_slice(_0{});
      copy(p.a.with(reinterpret_cast<uint64_t&>(s.full[slot])),ac.partition_S(at),ac.partition_D(sa));
      copy(p.b.with(reinterpret_cast<uint64_t&>(s.full[slot])),bc.partition_S(bt),bc.partition_D(sb));
    };
    // Each barrier has completed K/64 (always odd) transactions per N block.
    // Track independent slot phases across output-column blocks.
    int phase0=(nb*((K/64+1)/2))%2,phase1=(nb*(K/64/2))%2;
    if(tid==0){load(0);load(1);}
    auto acc=partition_fragment_C(mma,Shape<_64,Int<T::N>>{});clear(acc);
    for(int kt=0;kt<K/64;++kt) {
      int slot=kt%2,phase=((slot?phase1:phase0)+kt/2)%2;
      s.full[slot].wait(phase);
      auto sa=make_tensor(make_smem_ptr(s.a[slot].data()),typename T::AL{});
      auto sb=make_tensor(make_smem_ptr(s.b[slot].data()),typename T::BL{});
      auto aa=thr.partition_fragment_A(sa);auto bb=thr.partition_fragment_B(sb);
      flash::gemm<false,0>(mma,aa,bb,acc);
      __syncthreads();
      if(tid==0 && kt+2<K/64)load(kt+2);
    }
    #pragma unroll
    for(int j=0;j<size(acc);++j) {int r=get<0>(coord(j)),c=nb*T::N+get<1>(coord(j));s.dz[r*C+c]=Element(acc(j));}
    __syncthreads();
  }
  float ga[C/32]={},be[C/32]={};
  for(int r=warp;r<64;r+=4) {
    int row=blockIdx.x*64+r;
    if(row<p.rows) {
      float mu=p.stats[row*2],inv=p.stats[row*2+1],xh[C/32],d[C/32],a=0.f,b=0.f;
      #pragma unroll
      for(int j=0;j<C/32;++j) {
        int c=j*32+lane;xh[j]=(float(p.x[row*C+c])-mu)*inv;
        float raw=float(s.dz[r*C+c]);d[j]=raw*p.lw[c];a+=d[j];b+=d[j]*xh[j];ga[j]+=raw*xh[j];be[j]+=raw;
        p.z[row*C+c]=Element(xh[j]*p.lw[c]+p.lb[c]);
      }
      a=sum_warp(a)/C;b=sum_warp(b)/C;
      #pragma unroll
      for(int j=0;j<C/32;++j) {int c=j*32+lane;Element dx=Element(inv*(d[j]-a-xh[j]*b));p.dx[row*C+c]=Element(float(dx)+float(p.dy[row*C+c]));}
    }
  }
  #pragma unroll
  for(int j=0;j<C/32;++j) {int c=j*32+lane;p.parts[((blockIdx.x*4+warp)*2)*C+c]=ga[j];p.parts[((blockIdx.x*4+warp)*2+1)*C+c]=be[j];}
}
template<int C> std::vector<torch::Tensor> launch(torch::Tensor x,torch::Tensor dall,torch::Tensor wall,torch::Tensor dy,torch::Tensor stats,torch::Tensor lw,torch::Tensor lb) {
  using T=Config<C>;int rows=x.size(0),K=4*C+64;
  auto dx=torch::empty_like(x),z=torch::empty_like(x),parts=torch::empty({(rows+63)/64*4,2,C},x.options().dtype(torch::kFloat32));
  auto a=make_tensor(make_gmem_ptr((Element const*)dall.data_ptr()),make_shape(rows,K),typename T::AS{K,_1{}});
  auto b=make_tensor(make_gmem_ptr((Element const*)wall.data_ptr()),make_shape(C,K),typename T::BS{_1{},C});
  auto ta=make_tma_copy(SM90_TMA_LOAD{},a,typename T::AL{},Shape<_64,_64>{},_1{});
  auto tb=make_tma_copy(SM90_TMA_LOAD{},b,typename T::BL{},Shape<Int<T::N>,_64>{},_1{});
  typename T::Params p{ta,tb,(Element const*)x.data_ptr(),(Element const*)dy.data_ptr(),stats.data_ptr<float>(),lw.data_ptr<float>(),lb.data_ptr<float>(),(Element*)dx.data_ptr(),(Element*)z.data_ptr(),parts.data_ptr<float>(),rows};
  C10_CUDA_CHECK(cudaFuncSetAttribute(projection_ln<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename T::Shared)));
  projection_ln<C><<<(rows+63)/64,128,sizeof(typename T::Shared),at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,z,parts};
}
std::vector<torch::Tensor> backward(torch::Tensor x,torch::Tensor dall,torch::Tensor wall,torch::Tensor dy,torch::Tensor stats,torch::Tensor lw,torch::Tensor lb) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.is_contiguous() && x.dim()==2,"BF16 x required");
  c10::cuda::CUDAGuard guard(x.device());int C=x.size(1),rows=x.size(0);
  for(auto t:{dall,wall,dy})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==torch::kBFloat16 && t.is_contiguous(),"BF16 grouped operands required");
  TORCH_CHECK(dall.sizes()==torch::IntArrayRef({rows,4*C+64}) && wall.sizes()==torch::IntArrayRef({4*C+64,C}) && dy.sizes()==x.sizes(),"grouped projection shape mismatch");
  TORCH_CHECK(stats.device()==x.device() && stats.scalar_type()==torch::kFloat32 && stats.is_contiguous() && stats.sizes()==torch::IntArrayRef({rows,2}),"LN stats mismatch");
  for(auto t:{lw,lb})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous() && t.numel()==C,"FP32 LN affines required");
  switch(C){
#define GO(C_) case C_: return launch<C_>(x,dall,wall,dy,stats,lw,lb);
    GO(64) GO(128) GO(256) GO(384) GO(512)
#undef GO
    default:TORCH_CHECK(false,"unsupported width");
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("backward",&backward);}
