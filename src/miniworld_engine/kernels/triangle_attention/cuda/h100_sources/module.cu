// Hopper implementation of B200 front/core/tail boundaries.
// LN and sigmoid-gated attention tiles stay in shared memory before WGMMA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include "fa3_utils.h"
using namespace cute;
using Element=cutlass::bfloat16_t;
using TileLayout=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));

__device__ float warp_sum(float x) {
  #pragma unroll
  for(int d=16;d;d>>=1) x+=__shfl_xor_sync(0xffffffff,x,d);
  return x;
}
__device__ float sigmoid(float x) { return 1.f/(1.f+expf(-x)); }
template<int C,bool Tail> struct Stage {
  static constexpr int N=Tail?(C==64?64:128):(C==64?128:256);
  using WL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<Int<N>,_64>{}));
  using GS=Shape<int32_t,int32_t>;
  using ST=Stride<int64_t,_1>;
  using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GS{},ST{}),WL{},Shape<Int<N>,_64>{},_1{}));
  struct Shared {
    array_aligned<Element,Tail || C>=384?1:64*C,1024> z;
    array_aligned<float,Tail?1:C,128> lw,lb;
    array_aligned<float,128,128> stats;
    array_aligned<Element,4096,1024> a;
    array_aligned<Element,N*64,1024> w[2];
    cutlass::arch::ClusterTransactionBarrier full[2];
  };
};
template<int C, bool Tail> __global__ __launch_bounds__(128)
void fused_stage(CUTE_GRID_CONSTANT typename Stage<C,Tail>::TW const tw,
                 Element const* x,float const* lnw,float const* lnb,Element const* weights,
                 Element const* o,Element const* gate,Element const* drop,bool const* mask,
                 Element* proj,Element* bias,Element* result,float* stats,int rows,int L,int H,float eps,int gate_stride) {
  using T=Stage<C,Tail>;
  extern __shared__ char memory[]; auto& s=*reinterpret_cast<typename T::Shared*>(memory);
  constexpr int N=T::N,cols=Tail?C:4*C+64;constexpr int blocks=(cols+N-1)/N;
  int tid=threadIdx.x,lane=tid%32,warp=tid/32,base=(blockIdx.x/blocks)*64,nb=(blockIdx.x%blocks)*N;
  if(tid==0){s.full[0].init(1);s.full[1].init(1);prefetch_tma_descriptor(tw.get_tma_descriptor());cutlass::arch::fence_barrier_init();}
  __syncthreads();
  auto wg=tw.get_tma_tensor(make_shape(cols,C));
  auto load=[&](int kk){int slot=(kk/64)%2;s.full[slot].arrive_and_expect_tx(N*64*sizeof(Element));
    auto ww=local_tile(wg,Shape<Int<N>,_64>{},make_coord(nb/N,kk/64));
    auto sw=make_tensor(make_smem_ptr(s.w[slot].data()),typename T::WL{});auto t=tw.get_slice(_0{});
    copy(tw.with(reinterpret_cast<uint64_t&>(s.full[slot])),t.partition_S(ww),t.partition_D(sw));};
  if(tid==0){load(0);if(C>64)load(64);}
  if constexpr(!Tail) {
    for(int c=tid;c<C;c+=128){s.lw[c]=lnw[c];s.lb[c]=lnb[c];}
    __syncthreads();
    // Four lanes own one row;32 row groups cover the64-row tile in two rounds.
    int group=tid/4,sub=tid%4;
    for(int r=group;r<64;r+=32) {
      int row=base+r;float vals[C/4],mean=0.f,var=0.f;
      #pragma unroll
      for(int k=0;k<C/4;k+=8){
        uint4 vv=row<rows?*reinterpret_cast<uint4 const*>(x+row*C+sub*(C/4)+k):make_uint4(0,0,0,0);
        auto* v=reinterpret_cast<Element*>(&vv);
        #pragma unroll
        for(int j=0;j<8;++j){vals[k+j]=float(v[j]);mean+=vals[k+j];}
      }
      mean+=__shfl_xor_sync(0xffffffff,mean,1);mean+=__shfl_xor_sync(0xffffffff,mean,2);mean/=C;
      #pragma unroll
      for(int k=0;k<C/4;++k){float d=vals[k]-mean;var+=d*d;}
      var+=__shfl_xor_sync(0xffffffff,var,1);var+=__shfl_xor_sync(0xffffffff,var,2);
      float inv=rsqrtf(var/C+eps);
      if(sub==0){s.stats[r*2]=mean;s.stats[r*2+1]=inv;
        if(nb==0 && row<rows){stats[row*2]=mean;stats[row*2+1]=inv;}}
      if constexpr(C<384){
        #pragma unroll
        for(int k=0;k<C/4;k+=8){uint4 vec;auto* v=reinterpret_cast<Element*>(&vec);
          #pragma unroll
          for(int j=0;j<8;++j){int c=sub*(C/4)+k+j;v[j]=Element((vals[k+j]-mean)*inv*s.lw[c]+s.lb[c]);vals[k+j]=float(v[j]);}
          *reinterpret_cast<uint4*>(s.z.data()+r*C+sub*(C/4)+k)=vec;}
      }
    }
  }
  __syncthreads();
  using TileMMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,Int<N>,_64>>()));
  TileMMA mma;auto thr=mma.get_slice(tid);
  auto acc=partition_fragment_C(mma,Shape<_64,Int<N>>{});clear(acc);
  auto al=as_position_independent_swizzle_layout(TileLayout{});
  for(int kk=0;kk<C;kk+=64) {
    // The swizzle preserves16-byte vectors. Both read and write use128 bits.
    for(int t=tid;t<4096/8;t+=128) {
      int r=t/8,k=(t%8)*8;uint4 vec;
      if constexpr(Tail) {
        int row=base+r;
        if(row<rows){
          uint4 gv=*reinterpret_cast<uint4 const*>(gate+row*gate_stride+kk+k),ov=*reinterpret_cast<uint4 const*>(o+row*C+kk+k);
          auto const* gg=reinterpret_cast<Element const*>(&gv);auto const* oo=reinterpret_cast<Element const*>(&ov);auto* vv=reinterpret_cast<Element*>(&vec);
          #pragma unroll
          for(int j=0;j<8;++j)vv[j]=Element(sigmoid(float(gg[j]))*float(oo[j]));
        }else vec=make_uint4(0,0,0,0);
      }else if constexpr(C<384)vec=*reinterpret_cast<uint4 const*>(s.z.data()+r*C+kk+k);
      else {
        int row=base+r;uint4 raw=row<rows?*reinterpret_cast<uint4 const*>(x+row*C+kk+k):make_uint4(0,0,0,0);
        auto* in=reinterpret_cast<Element*>(&raw);auto* out=reinterpret_cast<Element*>(&vec);
        float mu=s.stats[r*2],inv=s.stats[r*2+1];
        #pragma unroll
        for(int j=0;j<8;++j)out[j]=Element((float(in[j])-mu)*inv*s.lw[kk+k+j]+s.lb[kk+k+j]);
      }
      *reinterpret_cast<uint4*>(s.a.data()+al(make_coord(r,k)))=vec;
    }
    cutlass::arch::fence_view_async_shared();__syncthreads();
    int slot=(kk/64)%2;s.full[slot].wait((kk/128)%2);
    auto aa=make_tensor(make_smem_ptr(s.a.data()),TileLayout{});auto ww=make_tensor(make_smem_ptr(s.w[slot].data()),typename T::WL{});
    auto ra=thr.partition_fragment_A(aa);auto rb=thr.partition_fragment_B(ww);
    flash::gemm<false,0>(mma,ra,rb,acc);
    __syncthreads();
    if(tid==0 && kk+128<C)load(kk+128);
  }
  auto coord=thr.partition_C(make_identity_tensor(Shape<_64,Int<N>>{}));
  #pragma unroll
  for(int j=0;j<size(acc);j+=2) {
    int row=base+get<0>(coord(j)),col=nb+get<1>(coord(j));
    if(row<rows) {
      Element lo=Element(acc(j)),hi=Element(acc(j+1));
      if constexpr(Tail) {
        int64_t idx=int64_t(row)*C+col;
        if(drop){int off=((row/(L*L))*L+row%L)*C+col;lo=Element(float(lo)*float(drop[off]));hi=Element(float(hi)*float(drop[off+1]));}
        lo=Element(float(x[idx])+float(lo));hi=Element(float(x[idx+1])+float(hi));
        *reinterpret_cast<uint32_t*>(result+idx)=uint32_t(lo.raw())|(uint32_t(hi.raw())<<16);
      }else if(col<4*C){
        int p=col/C,c=col%C;int64_t idx=(int64_t(p)*rows+row)*C+c;
        *reinterpret_cast<uint32_t*>(proj+idx)=uint32_t(lo.raw())|(uint32_t(hi.raw())<<16);
      }else{
        int b=row/(L*L),q=(row/L)%L,key=row%L;
        if(col<4*C+H)bias[((b*H+col-4*C)*L+q)*L+key]=mask && !mask[b*L+key]?Element(-3.3895313892515355e38f):lo;
        if(col+1<4*C+H)bias[((b*H+col+1-4*C)*L+q)*L+key]=mask && !mask[b*L+key]?Element(-3.3895313892515355e38f):hi;
      }
    }
  }
}

template<int C,bool Tail> auto weight_tma(torch::Tensor w) {
  using T=Stage<C,Tail>;constexpr int cols=Tail?C:4*C+64;
  auto g=make_tensor(make_gmem_ptr((Element const*)w.data_ptr()),make_shape(cols,C),typename T::ST{C,_1{}});
  return make_tma_copy(SM90_TMA_LOAD{},g,typename T::WL{},Shape<Int<T::N>,_64>{},_1{});
}
void metadata(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.is_contiguous() && x.dim()==2,"contiguous BF16 matrix required");
  TORCH_CHECK((x.size(1)==64 || x.size(1)==128 || x.size(1)==256 || x.size(1)==384 || x.size(1)==512) && x.size(0)>0,"C64..512, multiple64 required");
}
std::vector<torch::Tensor> h100_front(torch::Tensor x,torch::Tensor lnw,torch::Tensor lnb,torch::Tensor weights,torch::Tensor mask,int L,int H,double eps) {
  metadata(x);c10::cuda::CUDAGuard guard(x.device());int C=x.size(1),rows=x.size(0),B=rows/(L*L);
  TORCH_CHECK(L>0 && L%128==0 && rows==B*L*L && H==C/32,"invalid pair/head dimensions");
  TORCH_CHECK(weights.device()==x.device() && weights.scalar_type()==torch::kBFloat16 && weights.is_contiguous() && weights.sizes()==torch::IntArrayRef({4*C+64,C}),"packed weights mismatch");
  for(auto t:{lnw,lnb}) TORCH_CHECK(t.device()==x.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous() && t.numel()==C,"FP32 LN affines required");
  TORCH_CHECK(mask.device()==x.device() && mask.is_contiguous() && mask.scalar_type()==torch::kBool && (mask.numel()==0 || mask.sizes()==torch::IntArrayRef({B,L})),"mask mismatch");
  auto p=torch::empty({4,rows,C},x.options()),bias=torch::empty({B,H,L,L},x.options());
  auto stats=torch::empty({rows,2},x.options().dtype(torch::kFloat32));
  int N=C==64?128:256;dim3 grid(((rows+63)/64)*((4*C+64+N-1)/N));auto stream=at::cuda::getCurrentCUDAStream();
#define F(C_) case C_: C10_CUDA_CHECK(cudaFuncSetAttribute(fused_stage<C_,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename Stage<C_,false>::Shared))); fused_stage<C_,false><<<grid,128,sizeof(typename Stage<C_,false>::Shared),stream>>>(weight_tma<C_,false>(weights),(Element*)x.data_ptr(),lnw.data_ptr<float>(),lnb.data_ptr<float>(),(Element*)weights.data_ptr(),nullptr,nullptr,nullptr,mask.numel()?mask.data_ptr<bool>():nullptr,(Element*)p.data_ptr(),(Element*)bias.data_ptr(),nullptr,stats.data_ptr<float>(),rows,L,H,float(eps),0);break;
  switch(C){F(64) F(128) F(256) F(384) F(512)}
#undef F
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {p,bias,stats};
}
torch::Tensor h100_tail(torch::Tensor x,torch::Tensor gate,torch::Tensor o,torch::Tensor wo,torch::Tensor drop,int L) {
  metadata(x);c10::cuda::CUDAGuard guard(x.device());int C=x.size(1),rows=x.size(0);
  TORCH_CHECK(gate.device()==x.device() && gate.scalar_type()==x.scalar_type() && gate.sizes()==x.sizes() && gate.stride(1)==1 && gate.stride(0)>=C && gate.stride(0)%8==0,"gate mismatch");
  TORCH_CHECK(o.device()==x.device() && o.scalar_type()==x.scalar_type() && o.is_contiguous() && o.sizes()==x.sizes(),"output mismatch");
  TORCH_CHECK(wo.device()==x.device() && wo.scalar_type()==x.scalar_type() && wo.is_contiguous() && wo.sizes()==torch::IntArrayRef({C,C}),"Wo mismatch");
  TORCH_CHECK(drop.device()==x.device() && drop.scalar_type()==x.scalar_type() && drop.is_contiguous() && (drop.numel()==0 || drop.numel()==rows/L*C),"dropout scale mismatch");
  auto y=torch::empty_like(x);int N=C==64?64:128;dim3 grid(((rows+63)/64)*(C/N));auto stream=at::cuda::getCurrentCUDAStream();
#define T(C_) case C_: C10_CUDA_CHECK(cudaFuncSetAttribute(fused_stage<C_,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename Stage<C_,true>::Shared))); fused_stage<C_,true><<<grid,128,sizeof(typename Stage<C_,true>::Shared),stream>>>(weight_tma<C_,true>(wo),(Element*)x.data_ptr(),nullptr,nullptr,(Element*)wo.data_ptr(),(Element*)o.data_ptr(),(Element*)gate.data_ptr(),drop.numel()?(Element*)drop.data_ptr():nullptr,nullptr,nullptr,nullptr,(Element*)y.data_ptr(),nullptr,rows,L,C/32,0.f,gate.stride(0));break;
  switch(C){T(64) T(128) T(256) T(384) T(512)}
#undef T
  C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
// CUDA-only backward helpers; cuBLAS handles the grouped projection products.
__global__ void gate_rows(Element const* du,Element const* gate,Element const* o,Element const* dy,Element const* drop,
                         Element* dout,Element* dg,Element* u,Element* dys,float* delta,int rows,int C,int L,int H,int gate_stride) {
  int row=blockIdx.x,warp=threadIdx.x/32,lane=threadIdx.x%32;
  for(int h=warp;h<H;h+=4) {
    int col=h*32+lane,idx=row*C+col;
    float sig=sigmoid(float(gate[row*gate_stride+col]));Element d=Element(float(du[idx])*sig);
    dout[idx]=d;dg[idx]=Element(float(du[idx])*float(o[idx])*sig*(1.f-sig));
    u[idx]=Element(float(o[idx])*sig);
    float ds=drop?float(drop[((row/(L*L))*L+row%L)*C+col]):1.f;
    dys[idx]=Element(float(dy[idx])*ds);
    float sum=warp_sum(float(d)*float(o[idx]));
    if(lane==0) delta[(((row/(L*L))*H+h)*L+(row/L)%L)*L+row%L]=sum;
  }
}
__global__ void scale_rows(Element const* dy,Element const* drop,Element* out,int64_t n,int C,int L) {
  int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;
  if(i<n) {int64_t row=i/C;out[i]=drop?Element(float(dy[i])*float(drop[((row/(L*L))*L+row%L)*C+i%C])):dy[i];}
}
torch::Tensor h100_scale(torch::Tensor dy,torch::Tensor drop,int L) {
  metadata(dy);c10::cuda::CUDAGuard guard(dy.device());auto out=torch::empty_like(dy);
  scale_rows<<<(dy.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>((Element*)dy.data_ptr(),drop.numel()?(Element*)drop.data_ptr():nullptr,(Element*)out.data_ptr(),dy.numel(),dy.size(1),L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> h100_gate_backward(torch::Tensor du,torch::Tensor gate,torch::Tensor o,torch::Tensor dy,torch::Tensor drop,int L) {
  metadata(du);c10::cuda::CUDAGuard guard(du.device());int C=du.size(1),rows=du.size(0),H=C/32;
  auto dout=torch::empty_like(du),dg=torch::empty_like(du),u=torch::empty_like(du),dys=torch::empty_like(du);
  auto delta=torch::empty({rows/(L*L),H,L,L},du.options().dtype(torch::kFloat32));
  gate_rows<<<rows,128,0,at::cuda::getCurrentCUDAStream()>>>((Element*)du.data_ptr(),(Element*)gate.data_ptr(),(Element*)o.data_ptr(),(Element*)dy.data_ptr(),drop.numel()?(Element*)drop.data_ptr():nullptr,(Element*)dout.data_ptr(),(Element*)dg.data_ptr(),(Element*)u.data_ptr(),(Element*)dys.data_ptr(),delta.data_ptr<float>(),rows,C,L,H,gate.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dout,dg,u,dys,delta};
}
__global__ void pack_grad(Element const* dq,Element const* dk,Element const* dv,Element const* dg,Element const* db,bool const* mask,Element* dall,int rows,int C,int L,int H) {
  int64_t idx=int64_t(blockIdx.x)*256+threadIdx.x,n=int64_t(rows)*(4*C+64);
  if(idx<n) {
    int row=idx/(4*C+64),col=idx%(4*C+64);float val=0.f;
    if(col<4*C) {int p=col/C,c=col%C;Element const* src=p==0?dq:p==1?dk:p==2?dv:dg;val=float(src[row*C+c]);}
    else if(col<4*C+H) {int b=row/(L*L),q=(row/L)%L,k=row%L,h=col-4*C;
      if(!mask || mask[b*L+k]) val=float(db[((b*H+h)*L+q)*L+k]);}
    dall[idx]=Element(val);
  }
}
torch::Tensor h100_pack_backward(torch::Tensor dq,torch::Tensor dk,torch::Tensor dv,torch::Tensor dg,torch::Tensor db,torch::Tensor mask,int L) {
  metadata(dq);c10::cuda::CUDAGuard guard(dq.device());int C=dq.size(1),rows=dq.size(0);auto out=torch::empty({rows,4*C+64},dq.options());
  pack_grad<<<(out.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>((Element*)dq.data_ptr(),(Element*)dk.data_ptr(),(Element*)dv.data_ptr(),(Element*)dg.data_ptr(),(Element*)db.data_ptr(),mask.numel()?mask.data_ptr<bool>():nullptr,(Element*)out.data_ptr(),rows,C,L,C/32);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
template<int C> __global__ __launch_bounds__(128)
void ln_rows(Element const* x,float const* dz,Element const* dy,float const* stats,float const* lw,float const* lb,
             Element* dx,Element* z,float* parts,int rows,int tile_rows) {
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  float ga[C/32]={},be[C/32]={};
  for(int rr=warp;rr<tile_rows;rr+=4) {
    int row=blockIdx.x*tile_rows+rr;
    if(row<rows) {
      float mu=stats[row*2],inv=stats[row*2+1],xh[C/32],d[C/32],a=0.f,b=0.f;
      #pragma unroll
      for(int k=0;k<C/32;++k) {
        int c=k*32+lane;xh[k]=(float(x[row*C+c])-mu)*inv;
        // BF16 normalized activations match the forward projection input.
        z[row*C+c]=Element(xh[k]*lw[c]+lb[c]);
        float raw=float(Element(dz[row*C+c]));d[k]=raw*lw[c];a+=d[k];b+=d[k]*xh[k];ga[k]+=raw*xh[k];be[k]+=raw;
      }
      a=warp_sum(a)/C;b=warp_sum(b)/C;
      #pragma unroll
      for(int k=0;k<C/32;++k) {int c=k*32+lane;Element v=Element(inv*(d[k]-a-xh[k]*b));dx[row*C+c]=Element(float(v)+float(dy[row*C+c]));}
    }
  }
  #pragma unroll
  for(int k=0;k<C/32;++k) {int c=k*32+lane;parts[((blockIdx.x*4+warp)*2)*C+c]=ga[k];parts[((blockIdx.x*4+warp)*2+1)*C+c]=be[k];}
}
std::vector<torch::Tensor> h100_ln_backward(torch::Tensor x,torch::Tensor dz,torch::Tensor dy,torch::Tensor stats,torch::Tensor lw,torch::Tensor lb) {
  metadata(x);c10::cuda::CUDAGuard guard(x.device());int C=x.size(1),rows=x.size(0);
  TORCH_CHECK(dz.device()==x.device() && dz.scalar_type()==torch::kFloat32 && dz.is_contiguous() && dz.sizes()==x.sizes(),"FP32 dZ required");
  int tile_rows=rows<=32768?32:128;
  auto dx=torch::empty_like(x),z=torch::empty_like(x),parts=torch::empty({(rows+tile_rows-1)/tile_rows*4,2,C},dz.options());
  auto stream=at::cuda::getCurrentCUDAStream();dim3 grid((rows+tile_rows-1)/tile_rows);
#define LN(C_) case C_: ln_rows<C_><<<grid,128,0,stream>>>((Element*)x.data_ptr(),dz.data_ptr<float>(),(Element*)dy.data_ptr(),stats.data_ptr<float>(),lw.data_ptr<float>(),lb.data_ptr<float>(),(Element*)dx.data_ptr(),(Element*)z.data_ptr(),parts.data_ptr<float>(),rows,tile_rows);break;
  switch(C){LN(64) LN(128) LN(256) LN(384) LN(512)}
#undef LN
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,z,parts};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("front",&h100_front);m.def("tail",&h100_tail);m.def("scale",&h100_scale);m.def("gate_backward",&h100_gate_backward);
  m.def("pack_backward",&h100_pack_backward);m.def("ln_backward",&h100_ln_backward);
}
