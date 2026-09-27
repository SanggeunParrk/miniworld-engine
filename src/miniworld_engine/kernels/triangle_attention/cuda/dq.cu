// Query-owned dQ; resident Q/dO/stats, double-buffered TMA K/V/bias.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include "fa3_utils.h"
using namespace cute;
using Element=cutlass::bfloat16_t;
constexpr float SCALE=0.1767766952966369f,LOG2E=1.44269504f;
__device__ __forceinline__ float ex2(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float sload(uint32_t p){uint16_t v;asm volatile("ld.shared.b16 %0,[%1];":"=h"(v):"r"(p):"memory");return float(Element::bitcast(v));}
__device__ __forceinline__ float sloadf(uint32_t p){float v;asm volatile("ld.shared.f32 %0,[%1];":"=f"(v):"r"(p):"memory");return v;}
__device__ __forceinline__ void spair(uint32_t p,float lo,float hi){uint32_t x;asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(x):"f"(hi),"f"(lo));asm volatile("st.shared.b32 [%0],%1;"::"r"(p),"r"(x):"memory");}
struct Config {
 using QL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));
 using KT=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_64>{}));
 using SL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));
 using QS=Shape<int32_t,_32,_4,int32_t>;
 using QStride=Stride<int64_t,_1,_32,int64_t>;
 using BS=Shape<int32_t,int32_t,_4>;
 using BStride=Stride<int64_t,_1,int64_t>;
 using TQ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),QS{},QStride{}),QL{},Shape<_64,_32>{},_1{}));
 using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),BS{},BStride{}),SL{},Shape<_64,_64>{},_1{}));
 using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),QS{},QStride{}),QL{},Shape<_64,_32>{},_1{}));
 using Score=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_64,_32>>()));
 using Grad=decltype(make_tiled_mma(GMMA::rs_op_selector<Element,Element,float,Shape<_64,_32,_64>,GMMA::Major::K,GMMA::Major::MN>()));
 struct Shared {
  array_aligned<Element,2048,1024> q,dout,k[2],v[2];
  array_aligned<Element,4096,1024> bias[2];
  array_aligned<float,64,128> lse,delta;
  cutlass::arch::ClusterTransactionBarrier resident,full[2];
  cutlass::arch::ClusterBarrier empty[2];
 };
 struct Params {TQ q,k,v,dout;TB bias;float const* lse;float const* delta;Element* dq;int L;TO out;};
};
template<class TMA,class G,class S> __device__ __forceinline__ void copy_tile(TMA const& tma,G const& src,S const& dst,cutlass::arch::ClusterTransactionBarrier& ready){auto c=tma.get_slice(_0{});copy(tma.with(reinterpret_cast<uint64_t&>(ready)),c.partition_S(src),c.partition_D(dst));}
__global__ __launch_bounds__(128,4) void dq_tma(CUTE_GRID_CONSTANT Config::Params const p){
 extern __shared__ char storage[];auto& s=*reinterpret_cast<Config::Shared*>(storage);
 int tid=threadIdx.x,row=blockIdx.y,h=blockIdx.z,qt=blockIdx.x,L=p.L;
 if(tid==0){s.resident.init(1);for(int i=0;i<2;++i){s.full[i].init(1);s.empty[i].init(1);}prefetch_tma_descriptor(p.q.get_tma_descriptor());prefetch_tma_descriptor(p.k.get_tma_descriptor());prefetch_tma_descriptor(p.v.get_tma_descriptor());prefetch_tma_descriptor(p.dout.get_tma_descriptor());prefetch_tma_descriptor(p.bias.get_tma_descriptor());cutlass::arch::fence_barrier_init();}
 __syncthreads();
 auto load=[&](int kt){
   int slot=kt%2;
   s.full[slot].arrive_and_expect_tx((2*2048+4096)*sizeof(Element));
   auto shape=make_shape(L,_32{},_4{},L);
   auto kg=p.k.get_tma_tensor(shape);auto vg=p.v.get_tma_tensor(shape);auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
   auto kk=local_tile(kg(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));auto vv=local_tile(vg(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qt,kt));
   copy_tile(p.k,kk,make_tensor(make_smem_ptr(s.k[slot].data()),Config::QL{}),s.full[slot]);copy_tile(p.v,vv,make_tensor(make_smem_ptr(s.v[slot].data()),Config::QL{}),s.full[slot]);copy_tile(p.bias,bb,make_tensor(make_smem_ptr(s.bias[slot].data()),Config::SL{}),s.full[slot]);
 };
 if(tid==0){
   auto shape=make_shape(L,_32{},_4{},L);auto qg=p.q.get_tma_tensor(shape);auto dog=p.dout.get_tma_tensor(shape);
   s.resident.arrive_and_expect_tx(2*2048*sizeof(Element)+2*64*sizeof(float));
   auto qtile=local_tile(qg(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));auto dotile=local_tile(dog(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
   copy_tile(p.q,qtile,make_tensor(make_smem_ptr(s.q.data()),Config::QL{}),s.resident);copy_tile(p.dout,dotile,make_tensor(make_smem_ptr(s.dout.data()),Config::QL{}),s.resident);
   int stat=(h*L+row)*L+qt*64;
   SM90_BULK_COPY_G2S::copy(p.lse+stat,reinterpret_cast<uint64_t*>(&s.resident),s.lse.data(),64*sizeof(float));SM90_BULK_COPY_G2S::copy(p.delta+stat,reinterpret_cast<uint64_t*>(&s.resident),s.delta.data(),64*sizeof(float));
   load(0);
 }
 int lane=tid;Config::Score smma;Config::Grad gmma;auto st=smma.get_slice(lane);auto gt=gmma.get_slice(lane);
 auto sc=st.partition_C(make_identity_tensor(Shape<_64,_64>{}));auto gc=gt.partition_C(make_identity_tensor(Shape<_64,_32>{}));auto dq=partition_fragment_C(gmma,Shape<_64,_32>{});clear(dq);
 s.resident.wait(0);
 auto sq=make_tensor(make_smem_ptr(s.q.data()),Config::QL{}),sd=make_tensor(make_smem_ptr(s.dout.data()),Config::QL{});auto qa=st.partition_fragment_A(sq),da=st.partition_fragment_A(sd);
 uint32_t mp=cast_smem_ptr_to_uint(s.lse.data()),ddp=cast_smem_ptr_to_uint(s.delta.data());
 int qr0=get<0>(sc(0)),qr1=get<0>(sc(2));float m0=fmaxf(sloadf(mp+qr0*4),-1e38f),m1=fmaxf(sloadf(mp+qr1*4),-1e38f),d0=sloadf(ddp+qr0*4),d1=sloadf(ddp+qr1*4);
 for(int kt=0;kt<L/64;++kt){
  if(tid==0 && kt+1<L/64)load(kt+1);
  int slot=kt%2,phase=(kt/2)%2;s.full[slot].wait(phase);
  auto sk=make_tensor(make_smem_ptr(s.k[slot].data()),Config::QL{}),sv=make_tensor(make_smem_ptr(s.v[slot].data()),Config::QL{});auto kb=st.partition_fragment_B(sk),vb=st.partition_fragment_B(sv);
  auto score=partition_fragment_C(smma,Shape<_64,_64>{}),dp=partition_fragment_C(smma,Shape<_64,_64>{});
  flash::gemm<true,-1>(smma,qa,kb,score);flash::gemm<true,-1>(smma,da,vb,dp);
  // Score is the older committed group. Overlap softmax with the dP group.
  warpgroup_wait<1>();warpgroup_fence_operand(score);
  uint32_t bp=cast_smem_ptr_to_uint(s.bias[slot].data());
  #pragma unroll
  for(int xb=0;xb<size(score);xb+=8){
   // Four 8x8 blocks cover a 16x16 sub-tile in C-fragment order.
   int qr=(lane/32)*16+(lane%16),kr=xb*2+(lane%32/16)*8;
   uint32_t addr=bp+as_position_independent_swizzle_layout(Config::SL{})(make_coord(qr,kr))*2,bv[4];
   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
     :"=r"(bv[0]),"=r"(bv[1]),"=r"(bv[2]),"=r"(bv[3]):"r"(addr):"memory");
   #pragma unroll
   for(int xi=0;xi<8;++xi){
    int x=xb+xi;float bias=float(Element::bitcast(uint16_t(bv[xi/2]>>((xi%2)*16))));
    float logit=score(x)+bias*(1.f/SCALE);float pr=ex2(logit*(SCALE*LOG2E)-((x%4)/2?m1:m0));score(x)=pr;
   }
  }
  warpgroup_wait<0>();warpgroup_fence_operand(dp);
  #pragma unroll
  for(int x=0;x<size(dp);++x)dp(x)=score(x)*(dp(x)-((x%4)/2?d1:d0));
  auto acc_a=make_tensor(dp.data(),flash::convert_layout_acc_Aregs<Config::Grad>(dp.layout()));
  auto dsa=make_tensor_like<Element>(acc_a);flash::convert_type_out(acc_a,dsa);
  auto kk=make_tensor(make_smem_ptr(s.k[slot].data()),Config::KT{});auto ktb=gt.partition_fragment_B(kk);
  flash::gemm<false,0>(gmma,dsa,ktb,dq);__syncthreads();
 }
 uint32_t op=cast_smem_ptr_to_uint(s.q.data());
 #pragma unroll
 for(int x=0;x<size(dq);x+=2){int qr=get<0>(gc(x)),d=get<1>(gc(x));int off=as_position_independent_swizzle_layout(Config::QL{})(make_coord(qr,d))*2;spair(op+off,dq(x)*SCALE,dq(x+1)*SCALE);}
 cutlass::arch::fence_view_async_shared();__syncthreads();
 if(lane==0){auto og=p.out.get_tma_tensor(make_shape(L,_32{},_4{},L));auto tile=local_tile(og(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));auto src=make_tensor(make_smem_ptr(s.q.data()),Config::QL{});auto c=p.out.get_slice(_0{});copy(p.out,c.partition_S(src),c.partition_D(tile));tma_store_arrive();tma_store_wait<0>();}

}
torch::Tensor backward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor b,torch::Tensor m,torch::Tensor delta,torch::Tensor dy){
 TORCH_CHECK(q.is_cuda() && q.dim()==5 && q.size(0)==1 && q.size(1)==4 && q.size(4)==32,"B1 H4 D32 required");c10::cuda::CUDAGuard guard(q.device());int L=q.size(2);TORCH_CHECK(L>=64 && L<=1024 && L%64==0 && q.size(3)==L,"square L multiple64 <=1024");
 for(auto const& t:{q,k,v,dy}){TORCH_CHECK(t.device()==q.device() && t.scalar_type()==torch::kBFloat16 && t.sizes()==q.sizes(),"QKV/dO metadata mismatch");TORCH_CHECK(t.stride(4)==1 && t.stride(1)==32 && t.stride(3)==128 && t.stride(2)==L*128,"projection layout required");}
 TORCH_CHECK(b.device()==q.device() && b.scalar_type()==torch::kBFloat16 && b.is_contiguous() && b.sizes()==torch::IntArrayRef({1,4,L,L}),"bias metadata mismatch");for(auto const& t:{m,delta})TORCH_CHECK(t.device()==q.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous() && t.numel()==4*L*L,"FP32 contiguous stats required");
 auto result=torch::empty({1,L,L,128},q.options());auto shape=make_shape(L,_32{},_4{},L);
 auto make_q=[&](torch::Tensor const& x){auto src=make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),shape,Config::QStride{x.stride(3),_1{},_32{},x.stride(2)});return make_tma_copy(SM90_TMA_LOAD{},src,Config::QL{},Shape<_64,_32>{},_1{});};
 auto bg=make_tensor(make_gmem_ptr((Element const*)b.data_ptr()),make_shape(L,L,_4{}),Config::BStride{L,_1{},int64_t(L)*L});auto tb=make_tma_copy(SM90_TMA_LOAD{},bg,Config::SL{},Shape<_64,_64>{},_1{});
 Config::Params p{make_q(q),make_q(k),make_q(v),make_q(dy),tb,m.data_ptr<float>(),delta.data_ptr<float>(),(Element*)result.data_ptr(),L,{}};
 auto og=make_tensor(make_gmem_ptr((Element*)result.data_ptr()),shape,Config::QStride{128,_1{},_32{},int64_t(L)*128});p.out=make_tma_copy(SM90_TMA_STORE{},og,Config::QL{},Shape<_64,_32>{},_1{});
 C10_CUDA_CHECK(cudaFuncSetAttribute(dq_tma,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Config::Shared)));dq_tma<<<dim3(L/64,L,4),128,sizeof(Config::Shared),at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();return result.view({1,L,L,4,32}).permute({0,3,1,2,4});
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("backward",&backward);m.def("smem",[](){return sizeof(Config::Shared);});}
