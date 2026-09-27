// Output projection dgrad, gate derivatives, gated output for dW, and delta.
// The rounded BF16 dO feeds both the returned gradient and the FP32 delta.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include "fa3_utils.h"
using namespace cute;
using Element=cutlass::bfloat16_t;
__device__ __forceinline__ float ld(uint32_t p){uint16_t v;asm volatile("ld.shared.b16 %0,[%1];":"=h"(v):"r"(p):"memory");return float(Element::bitcast(v));}
__device__ __forceinline__ float ex2(float x){float v;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(v):"f"(x));return v;}
__device__ __forceinline__ float rcp(float x){float v;asm("rcp.approx.ftz.f32 %0,%1;":"=f"(v):"f"(x));return v;}
__device__ __forceinline__ float2 ld2(uint32_t p){uint32_t v;asm volatile("ld.shared.b32 %0,[%1];":"=r"(v):"r"(p):"memory");return make_float2(float(Element::bitcast(uint16_t(v))),float(Element::bitcast(uint16_t(v>>16))));}
__device__ __forceinline__ void st2(Element* p,float lo,float hi){uint32_t v;asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(v):"f"(hi),"f"(lo));*reinterpret_cast<uint32_t*>(p)=v;}
__device__ __forceinline__ void ss2(uint32_t p,float lo,float hi){uint32_t v;asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(v):"f"(hi),"f"(lo));asm volatile("st.shared.b32 [%0],%1;"::"r"(p),"r"(v):"memory");}
struct Config {
 using A=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_128>{}));
 using B=decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{},Shape<_64,_128>{}));
 using G=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));
 using TA=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),make_shape(int32_t{},_128{}),make_stride(_128{},_1{})),A{},Shape<_64,_128>{},_1{}));
 using TG=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),make_shape(int32_t{},_128{}),make_stride(_128{},_1{})),G{},Shape<_64,_64>{},_1{}));
 using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),Shape<_128,_128>{},Stride<_1,_128>{}),B{},Shape<_64,_128>{},_1{}));
 using TS=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),make_shape(int32_t{},_128{}),make_stride(_128{},_1{})),G{},Shape<_64,_64>{},_1{}));
 using MMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_64,_128>,GMMA::Major::K,GMMA::Major::MN>()));
 struct Shared {
  array_aligned<Element,8192,1024> dy;
  array_aligned<Element,8192,1024> w;
  cutlass::arch::ClusterTransactionBarrier ready,pointwise_ready;
 };
 struct Params {TA dy;TG gate,out;TB w;Element *dr,*dg,*a;float* delta;int rows;TS sr,sg,sa;};
};
template<class T,class G,class S> __device__ __forceinline__ void cp(T const& t,G const& g,S const& s,cutlass::arch::ClusterTransactionBarrier& b){auto c=t.get_slice(_0{});copy(t.with(reinterpret_cast<uint64_t&>(b)),c.partition_S(g),c.partition_D(s));}
__global__ __launch_bounds__(128,6) void gate_delta_tma(CUTE_GRID_CONSTANT Config::Params const p){
 extern __shared__ char buf[];auto& s=*reinterpret_cast<Config::Shared*>(buf);int tid=threadIdx.x;int mt=blockIdx.x/2,nt=blockIdx.x%2;
 if(tid==0){s.ready.init(1);s.pointwise_ready.init(1);prefetch_tma_descriptor(p.dy.get_tma_descriptor());prefetch_tma_descriptor(p.w.get_tma_descriptor());prefetch_tma_descriptor(p.gate.get_tma_descriptor());prefetch_tma_descriptor(p.out.get_tma_descriptor());cutlass::arch::fence_barrier_init();}
 __syncthreads();
 if(tid==0){
   s.ready.arrive_and_expect_tx(2*8192*sizeof(Element));auto shape=make_shape(p.rows,_128{});
   auto dy=local_tile(p.dy.get_tma_tensor(shape),Shape<_64,_128>{},make_coord(mt,0));
   auto w=local_tile(p.w.get_tma_tensor(Shape<_128,_128>{}),Shape<_64,_128>{},make_coord(nt,0));
   cp(p.dy,dy,make_tensor(make_smem_ptr(s.dy.data()),Config::A{}),s.ready);
   cp(p.w,w,make_tensor(make_smem_ptr(s.w.data()),Config::B{}),s.ready);
 }
 Config::MMA mma;auto t=mma.get_slice(tid);auto acc=partition_fragment_C(mma,Shape<_64,_64>{});
 auto sa=make_tensor(make_smem_ptr(s.dy.data()),Config::A{});auto sb=make_tensor(make_smem_ptr(s.w.data()),Config::B{});
 auto a=t.partition_fragment_A(sa);auto b=t.partition_fragment_B(sb);s.ready.wait(0);flash::gemm<true,0>(mma,a,b,acc);
 // Weight reads have retired. Reuse the same16KB for gate and attention output.
 cutlass::arch::NamedBarrier::sync(128,1);
 if(tid==0){
  s.pointwise_ready.arrive_and_expect_tx(2*4096*sizeof(Element));
  auto shape=make_shape(p.rows,_128{});
  auto gate=local_tile(p.gate.get_tma_tensor(shape),Shape<_64,_64>{},make_coord(mt,nt));
  auto out=local_tile(p.out.get_tma_tensor(shape),Shape<_64,_64>{},make_coord(mt,nt));
  cp(p.gate,gate,make_tensor(make_smem_ptr(s.w.data()),Config::G{}),s.pointwise_ready);
  cp(p.out,out,make_tensor(make_smem_ptr(s.w.data()+4096),Config::G{}),s.pointwise_ready);
 }
 s.pointwise_ready.wait(0);
 auto coords=t.partition_C(make_identity_tensor(Shape<_64,_64>{}));
 uint32_t gp=cast_smem_ptr_to_uint(s.w.data()),op=cast_smem_ptr_to_uint(s.w.data()+4096);
 float sums[2][2]={};
 #pragma unroll
 for(int i=0;i<size(acc);i+=2){
  int row=get<0>(coords(i)),col=get<1>(coords(i));int off=as_position_independent_swizzle_layout(Config::G{})(make_coord(row,col))*2;
  float2 gg=ld2(gp+off),oo=ld2(op+off);
  float g0=rcp(1.f+ex2(-gg.x*1.4426950408889634f)),g1=rcp(1.f+ex2(-gg.y*1.4426950408889634f));
  float dr0=float(Element(g0*acc(i))),dr1=float(Element(g1*acc(i+1)));
  int idx=(mt*64+row)*128+nt*64+col;
  ss2(cast_smem_ptr_to_uint(s.dy.data())+off,dr0,dr1);ss2(gp+off,((acc(i)*oo.x)*g0)*(1.f-g0),((acc(i+1)*oo.y)*g1)*(1.f-g1));ss2(op+off,g0*oo.x,g1*oo.y);
  sums[i/16][(i%4)/2]+=dr0*oo.x+dr1*oo.y;
 }
 #pragma unroll
 for(int h=0;h<2;++h){
  #pragma unroll
  for(int r=0;r<2;++r){
   float sum=sums[h][r];sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);
   int row=mt*64+get<0>(coords(r*2));
   if((tid%4)==0)p.delta[(nt*2+h)*p.rows+row]=sum;
  }
 }
 cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,1);
 if(tid==0){
  auto shape=make_shape(p.rows,_128{});
  auto sr=make_tensor(make_smem_ptr(s.dy.data()),Config::G{}),sg=make_tensor(make_smem_ptr(s.w.data()),Config::G{}),sa=make_tensor(make_smem_ptr(s.w.data()+4096),Config::G{});
  auto gr=local_tile(p.sr.get_tma_tensor(shape),Shape<_64,_64>{},make_coord(mt,nt)),gg=local_tile(p.sg.get_tma_tensor(shape),Shape<_64,_64>{},make_coord(mt,nt)),ga=local_tile(p.sa.get_tma_tensor(shape),Shape<_64,_64>{},make_coord(mt,nt));
  auto cr=p.sr.get_slice(_0{}),cg=p.sg.get_slice(_0{}),ca=p.sa.get_slice(_0{});
  copy(p.sr,cr.partition_S(sr),cr.partition_D(gr));copy(p.sg,cg.partition_S(sg),cg.partition_D(gg));copy(p.sa,ca.partition_S(sa),ca.partition_D(ga));tma_store_arrive();tma_store_wait<0>();
 }

}
std::vector<torch::Tensor> backward(torch::Tensor dy,torch::Tensor w,torch::Tensor gate,torch::Tensor out){
 TORCH_CHECK(dy.is_cuda() && dy.dim()==2 && dy.size(1)==128 && dy.size(0)%64==0,"[M,128], M divisible by64 required");c10::cuda::CUDAGuard guard(dy.device());
 C10_CUDA_CHECK(cudaSetDevice(dy.get_device()));
 int rows=dy.size(0);for(auto const& t:{dy,w,gate,out})TORCH_CHECK(t.device()==dy.device() && t.scalar_type()==torch::kBFloat16 && t.is_contiguous(),"contiguous BF16 inputs required");
 TORCH_CHECK(gate.sizes()==dy.sizes() && out.sizes()==dy.sizes() && w.sizes()==torch::IntArrayRef({128,128}),"shape mismatch");
 auto dr=torch::empty_like(gate),dg=torch::empty_like(gate),a=torch::empty_like(gate);auto delta=torch::empty({4,rows},dy.options().dtype(torch::kFloat32));
 auto make_a=[&](torch::Tensor const& t){return make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)t.data_ptr()),make_shape(rows,_128{}),make_stride(_128{},_1{})),Config::A{},Shape<_64,_128>{},_1{});};
 auto make_g=[&](torch::Tensor const& t){return make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)t.data_ptr()),make_shape(rows,_128{}),make_stride(_128{},_1{})),Config::G{},Shape<_64,_64>{},_1{});};
 auto bw=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)w.data_ptr()),Shape<_128,_128>{},Stride<_1,_128>{}),Config::B{},Shape<_64,_128>{},_1{});
 auto make_s=[&](torch::Tensor const& t){return make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)t.data_ptr()),make_shape(rows,_128{}),make_stride(_128{},_1{})),Config::G{},Shape<_64,_64>{},_1{});};
 Config::Params p{make_a(dy),make_g(gate),make_g(out),bw,(Element*)dr.data_ptr(),(Element*)dg.data_ptr(),(Element*)a.data_ptr(),delta.data_ptr<float>(),rows,make_s(dr),make_s(dg),make_s(a)};
 C10_CUDA_CHECK(cudaFuncSetAttribute(gate_delta_tma,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Config::Shared)));gate_delta_tma<<<rows/32,128,sizeof(Config::Shared),at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();return {dr,dg,a,delta};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("backward",&backward);m.def("smem",[](){return sizeof(Config::Shared);});}
