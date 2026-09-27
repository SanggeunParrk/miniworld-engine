// Fused projection dgrad + LayerNorm backward + residual, including LN parameter gradients.
// One producer warp fills a two-stage TMA ring; one consumer warpgroup runs
// WGMMA. The narrow four-head bias projection is accumulated in the epilogue.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include "fa3_utils.h"
using namespace cute;
using Element = cutlass::bfloat16_t;

__device__ __forceinline__ float sm_load_bf16(uint32_t p) {
  uint16_t v;asm volatile("ld.shared.b16 %0, [%1];":"=h"(v):"r"(p):"memory");return float(Element::bitcast(v));
}
__device__ __forceinline__ void sm_store_bf16(uint32_t p,Element v) {
  asm volatile("st.shared.b16 [%0], %1;"::"r"(p),"h"(v.raw()):"memory");
}


__device__ __forceinline__ uint32_t sm_load_pair(uint32_t p) {
 uint32_t v;asm volatile("ld.shared.b32 %0,[%1];":"=r"(v):"r"(p):"memory");return v;
}
__device__ __forceinline__ void sm_store_pair(uint32_t p,float lo,float hi) {
 uint32_t v;asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(v):"f"(hi),"f"(lo));
 asm volatile("st.shared.b32 [%0],%1;"::"r"(p),"r"(v):"memory");
}

template<int M> struct Config {
  using A = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{}, Shape<Int<M>,_64>{}));
  using B = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{}, Shape<_128,_64>{}));
  using MMA = decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<Int<M>,_128,_64>,GMMA::Major::K,GMMA::Major::MN>()));
  using TA = decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),make_shape(int32_t{},_128{}),make_stride(_128{},_1{})),A{},Shape<Int<M>,_64>{},_1{}));
  using TB = decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),Shape<_128,_128>{},Stride<_1,_128>{}),B{},Shape<_128,_64>{},_1{}));
  using XL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<Int<M>,_128>{}));
  using TX=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),make_shape(int32_t{},_128{}),make_stride(_128{},_1{})),XL{},Shape<Int<M>,_128>{},_1{}));
  using RS=Shape<int32_t,_128,int32_t>;
  using RT=Stride<int64_t,_1,int64_t>;
  using TR=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),RS{},RT{}),XL{},Shape<Int<M>,_128>{},_1{}));
  using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),RS{},RT{}),XL{},Shape<Int<M>,_128>{},_1{}));
  struct Shared {
    array_aligned<Element,cosize_v<A>,1024> a[2];
    array_aligned<Element,cosize_v<B>,1024> b[2];
    // X/residual reuse retired B stages; rounded dZ reuses retired A stages.
    array_aligned<float,M,128> mean,rstd;
    array_aligned<Element,M*4,128> dbias;
    array_aligned<Element,512,128> wbias;
    array_aligned<float,128,128> gamma;
    array_aligned<float,4*128,128> dgamma,dbeta;
    cutlass::arch::ClusterTransactionBarrier x_full,epilogue_full;
    cutlass::arch::ClusterTransactionBarrier full[2];
    cutlass::arch::ClusterBarrier empty[2];
  };
  struct Params {
    TA a[4]; TB b[4]; TX x; TR res_tma; TO out_tma;
    Element const *db,*wb,*residual;
    float const *mean,*rstd,*gamma;
    Element *dx;
    float *partial;
    int rows,L;bool ending;
  };
};

template<int M> __global__ __launch_bounds__(160,4)
void projection_ln_residual_tma(CUTE_GRID_CONSTANT typename Config<M>::Params const p) {
  using C=Config<M>;
  extern __shared__ char storage[];
  auto &s=*reinterpret_cast<typename C::Shared*>(storage);
  int tid=threadIdx.x;
  if(tid==0) {
    for(int j=0;j<2;++j) { s.full[j].init(1);s.empty[j].init(1); }
    s.x_full.init(1);s.epilogue_full.init(1);
    prefetch_tma_descriptor(p.x.get_tma_descriptor());
    prefetch_tma_descriptor(p.res_tma.get_tma_descriptor());
    for(int j=0;j<4;++j) { prefetch_tma_descriptor(p.a[j].get_tma_descriptor());prefetch_tma_descriptor(p.b[j].get_tma_descriptor()); }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if(tid>=128) {
    if(tid==128) {
      s.x_full.arrive_and_expect_tx(2*M*sizeof(float)+M*4*sizeof(Element)+512*sizeof(Element)+128*sizeof(float));
      SM90_BULK_COPY_G2S::copy(p.mean+blockIdx.x*M,reinterpret_cast<uint64_t*>(&s.x_full),s.mean.data(),M*sizeof(float));
      SM90_BULK_COPY_G2S::copy(p.rstd+blockIdx.x*M,reinterpret_cast<uint64_t*>(&s.x_full),s.rstd.data(),M*sizeof(float));
      SM90_BULK_COPY_G2S::copy(p.db+blockIdx.x*M*4,reinterpret_cast<uint64_t*>(&s.x_full),s.dbias.data(),M*4*sizeof(Element));
      SM90_BULK_COPY_G2S::copy(p.wb,reinterpret_cast<uint64_t*>(&s.x_full),s.wbias.data(),512*sizeof(Element));
      SM90_BULK_COPY_G2S::copy(p.gamma,reinterpret_cast<uint64_t*>(&s.x_full),s.gamma.data(),128*sizeof(float));
      for(int it=0;it<8;++it) {
        int stage=it%2,phase=(it/2)%2,proj=it/2,kt=it%2;
        s.empty[stage].wait(phase^1);
        s.full[stage].arrive_and_expect_tx((M*64+128*64)*sizeof(Element));
        auto ga=local_tile(p.a[proj].get_tma_tensor(make_shape(p.rows,_128{})),Shape<Int<M>,_64>{},make_coord(blockIdx.x,kt));
        auto gb=local_tile(p.b[proj].get_tma_tensor(Shape<_128,_128>{}),Shape<_128,_64>{},make_coord(0,kt));
        auto sa=make_tensor(make_smem_ptr(s.a[stage].data()),typename C::A{});
        auto sb=make_tensor(make_smem_ptr(s.b[stage].data()),typename C::B{});
        auto ca=p.a[proj].get_slice(_0{});
        auto cb=p.b[proj].get_slice(_0{});
        copy(p.a[proj].with(reinterpret_cast<uint64_t&>(s.full[stage])),ca.partition_S(ga),ca.partition_D(sa));
        copy(p.b[proj].with(reinterpret_cast<uint64_t&>(s.full[stage])),cb.partition_S(gb),cb.partition_D(sb));
      }
      // Both B stages become X/residual only after all WGMMA readers retire.
      s.empty[0].wait(1);s.empty[1].wait(1);
      s.epilogue_full.arrive_and_expect_tx(M*128*sizeof(Element)*(p.L%M==0?2:1));
      auto gx=local_tile(p.x.get_tma_tensor(make_shape(p.rows,_128{})),Shape<Int<M>,_128>{},make_coord(blockIdx.x,0));
      auto sx=make_tensor(make_smem_ptr(s.b[0].data()),typename C::XL{});
      auto cx=p.x.get_slice(_0{});
      copy(p.x.with(reinterpret_cast<uint64_t&>(s.epilogue_full)),cx.partition_S(gx),cx.partition_D(sx));
      if(p.L%M==0) {
        int tiles=p.L/M;
        auto rg=p.res_tma.get_tma_tensor(make_shape(p.L,_128{},p.L));
        auto gr=local_tile(rg(_,_,blockIdx.x/tiles),Shape<Int<M>,_128>{},make_coord(blockIdx.x%tiles,0));
        auto sr=make_tensor(make_smem_ptr(s.b[1].data()),typename C::XL{});
        auto cr=p.res_tma.get_slice(_0{});
        copy(p.res_tma.with(reinterpret_cast<uint64_t&>(s.epilogue_full)),cr.partition_S(gr),cr.partition_D(sr));
      }
    }
    return;
  }
  typename C::MMA mma;
  auto thr=mma.get_slice(tid);
  auto acc=partition_fragment_C(mma,Shape<Int<M>,_128>{});
  clear(acc);
  for(int it=0;it<8;++it) {
    int stage=it%2,phase=(it/2)%2;
    s.full[stage].wait(phase);
    auto sa=make_tensor(make_smem_ptr(s.a[stage].data()),typename C::A{});
    auto sb=make_tensor(make_smem_ptr(s.b[stage].data()),typename C::B{});
    auto ra=thr.partition_fragment_A(sa);
    auto rb=thr.partition_fragment_B(sb);
    flash::gemm<false,0>(mma,ra,rb,acc);
    // All 128 consumers have completed their WGMMA before releasing this stage.
    cutlass::arch::NamedBarrier::sync(128,1);
    if(tid==0) s.empty[stage].arrive();
  }
  s.x_full.wait(0);
  // The last WGMMA has completed. Reuse B stage 0 for the rounded dZ tile.
  uint32_t dzp=cast_smem_ptr_to_uint(s.a[0].data());
  auto coord=thr.partition_C(make_identity_tensor(Shape<Int<M>,_128>{}));
  uint32_t dbp=cast_smem_ptr_to_uint(s.dbias.data()),wbp=cast_smem_ptr_to_uint(s.wbias.data());
  #pragma unroll
  for(int i=0;i<size(acc);i+=2) {
    int lr=get<0>(coord(i)),col=get<1>(coord(i));
    float v0=acc(i),v1=acc(i+1);
    #pragma unroll
    for(int h=0;h<4;++h) {
      float d=sm_load_bf16(dbp+(lr*4+h)*2);
      uint32_t ww=sm_load_pair(wbp+(h*128+col)*2);
      v0=fmaf(d,float(Element::bitcast(uint16_t(ww))),v0);
      v1=fmaf(d,float(Element::bitcast(uint16_t(ww>>16))),v1);
    }
    int off=as_position_independent_swizzle_layout(typename C::XL{})(make_coord(lr,col))*2;
    sm_store_pair(dzp+off,v0,v1);
  }
  cutlass::arch::NamedBarrier::sync(128,1);
  s.x_full.wait(0);
  s.epilogue_full.wait(0);
  uint32_t xp=cast_smem_ptr_to_uint(s.b[0].data());
  uint32_t rp=cast_smem_ptr_to_uint(s.b[1].data());
  Element* __restrict__ final_out=p.dx;
  int consumer=tid,warp=consumer/32,lane=consumer%32;
  float dg[4]={},db[4]={},gamma[4];
  #pragma unroll
  for(int t=0;t<4;++t) gamma[t]=s.gamma[lane*2+(t/2)*64+t%2];
  int outer0=(blockIdx.x*M)/p.L,inner0=(blockIdx.x*M)%p.L;
  #pragma unroll 1
  for(int rr=0;rr<M/4;++rr) {
    int local=warp+rr*4,row=blockIdx.x*M+local;
    float mu=s.mean[local],inv=s.rstd[local];
    float xv[4],gv[4],wv[4];
    float c1=0.f,c2=0.f;
    #pragma unroll
    for(int t=0;t<4;t+=2) {
      int col=lane*2+(t/2)*64;
      int off=as_position_independent_swizzle_layout(typename C::XL{})(make_coord(local,col))*2;
      uint32_t xx=sm_load_pair(xp+off),zz=sm_load_pair(dzp+off);
      #pragma unroll
      for(int u=0;u<2;++u) {
        xv[t+u]=(float(Element::bitcast(uint16_t(xx>>(u*16))))-mu)*inv;
        gv[t+u]=float(Element::bitcast(uint16_t(zz>>(u*16))));wv[t+u]=gv[t+u]*gamma[t+u];
        c1+=xv[t+u]*wv[t+u];c2+=wv[t+u];
        dg[t+u]+=gv[t+u]*xv[t+u];db[t+u]+=gv[t+u];
      }
    }
    #pragma unroll
    for(int offset=16;offset>0;offset/=2) {
      c1+=__shfl_xor_sync(0xffffffff,c1,offset);
      c2+=__shfl_xor_sync(0xffffffff,c2,offset);
    }
    c1*=1.f/128.f;c2*=1.f/128.f;
    #pragma unroll
    for(int t=0;t<4;t+=2) {
      int col=lane*2+(t/2)*64;
      int final_row=p.ending?(p.L%M==0?(inner0+local)*p.L+outer0:(row%p.L)*p.L+row/p.L):row;
      int off=as_position_independent_swizzle_layout(typename C::XL{})(make_coord(local,col))*2;
      uint32_t rr=p.L%M==0?sm_load_pair(rp+off):0;
      float fv[2];
      #pragma unroll
      for(int u=0;u<2;++u) {
        Element dxbranch=Element((wv[t+u]-(xv[t+u]*c1+c2))*inv);
        float rv=p.L%M==0?float(Element::bitcast(uint16_t(rr>>(u*16)))):float(p.residual[final_row*128+col+u]);
        fv[u]=float(dxbranch)+rv;
        if(p.L%M!=0)final_out[final_row*128+col+u]=Element(fv[u]);
      }
      if(p.L%M==0)sm_store_pair(xp+off,fv[0],fv[1]);
    }
  }

  if(p.L%M==0) {
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::NamedBarrier::sync(128,1);
    if(consumer==0) {
      auto og=p.out_tma.get_tma_tensor(make_shape(p.L,_128{},p.L));
      auto go=local_tile(og(_,_,outer0),Shape<Int<M>,_128>{},make_coord(inner0/M,0));
      auto so=make_tensor(make_smem_ptr(s.b[0].data()),typename C::XL{});
      auto co=p.out_tma.get_slice(_0{});
      copy(p.out_tma,co.partition_S(so),co.partition_D(go));
      tma_store_arrive();tma_store_wait<0>();
    }
  }
  #pragma unroll
  for(int t=0;t<4;++t) {
    int col=lane*2+(t/2)*64+t%2;
    s.dgamma[warp*128+col]=dg[t];s.dbeta[warp*128+col]=db[t];
  }
  cutlass::arch::NamedBarrier::sync(128,1);
  float dg_sum=0.f,db_sum=0.f;
  #pragma unroll
  for(int w=0;w<4;++w) {dg_sum+=s.dgamma[w*128+consumer];db_sum+=s.dbeta[w*128+consumer];}
  p.partial[blockIdx.x*256+consumer]=dg_sum;
  p.partial[blockIdx.x*256+128+consumer]=db_sum;
}

__global__ void reduce_norm(float const* partial,float* dw,float* db,int tiles) {
  // Sixteen independent eight-channel reductions; coalesced 32-byte sectors.
  int lane=threadIdx.x%8,chunk=threadIdx.x/8,col=blockIdx.x*8+lane;
  float g=0.f,b=0.f;
  for(int i=blockIdx.y*16+chunk;i<tiles;i+=256) {g+=partial[i*256+col];b+=partial[i*256+128+col];}
  __shared__ float gg[128],bb[128];
  gg[threadIdx.x]=g;bb[threadIdx.x]=b;
  __syncthreads();
  if(threadIdx.x<8) {
    float gsum=0.f,bsum=0.f;
    #pragma unroll
    for(int i=0;i<16;++i) {gsum+=gg[i*8+lane];bsum+=bb[i*8+lane];}
    dw[blockIdx.y*256+col]=gsum;db[blockIdx.y*256+col]=bsum;
  }
}

__global__ void finish_norm(float const* partial,float* dw,float* db) {
 int c=threadIdx.x;float g=0.f,b=0.f;
 #pragma unroll
 for(int i=0;i<16;++i){g+=partial[i*256+c];b+=partial[i*256+128+c];}
 dw[c]=g;db[c]=b;
}

template<int M> std::vector<torch::Tensor> launch(std::vector<torch::Tensor> const& dy,std::vector<torch::Tensor> const& w,
    torch::Tensor x,torch::Tensor mean,torch::Tensor rstd,torch::Tensor gamma,torch::Tensor residual,int L,bool ending) {
  using C=Config<M>;
  int rows=dy[0].numel()/128;
  auto dx=torch::empty({rows,128},dy[0].options());
  auto partial=torch::empty({rows/M,256},mean.options());
  auto dw=torch::empty({128},mean.options());auto db=torch::empty_like(dw);
  typename C::Params p;
  for(int j=0;j<4;++j) {
    auto a=make_tensor(make_gmem_ptr((Element const*)dy[j].data_ptr()),make_shape(rows,_128{}),make_stride(_128{},_1{}));
    auto b=make_tensor(make_gmem_ptr((Element const*)w[j].data_ptr()),Shape<_128,_128>{},Stride<_1,_128>{});
    p.a[j]=make_tma_copy(SM90_TMA_LOAD{},a,typename C::A{},Shape<Int<M>,_64>{},_1{});
    p.b[j]=make_tma_copy(SM90_TMA_LOAD{},b,typename C::B{},Shape<_128,_64>{},_1{});
  }
  auto xg=make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),make_shape(rows,_128{}),make_stride(_128{},_1{}));
  p.x=make_tma_copy(SM90_TMA_LOAD{},xg,typename C::XL{},Shape<Int<M>,_128>{},_1{});
  auto rg=make_tensor(make_gmem_ptr((Element const*)residual.data_ptr()),make_shape(L,_128{},L),typename C::RT{ending?int64_t(L)*128:128,_1{},ending?128:int64_t(L)*128});
  p.res_tma=make_tma_copy(SM90_TMA_LOAD{},rg,typename C::XL{},Shape<Int<M>,_128>{},_1{});
  auto og=make_tensor(make_gmem_ptr((Element*)dx.data_ptr()),make_shape(L,_128{},L),typename C::RT{ending?int64_t(L)*128:128,_1{},ending?128:int64_t(L)*128});
  p.out_tma=make_tma_copy(SM90_TMA_STORE{},og,typename C::XL{},Shape<Int<M>,_128>{},_1{});
  p.db=(Element const*)dy[4].data_ptr();p.wb=(Element const*)w[4].data_ptr();p.dx=(Element*)dx.data_ptr();p.rows=rows;
  p.mean=mean.data_ptr<float>();p.rstd=rstd.data_ptr<float>();p.gamma=gamma.data_ptr<float>();
  p.residual=(Element const*)residual.data_ptr();p.L=L;p.ending=ending;p.partial=partial.data_ptr<float>();
  C10_CUDA_CHECK(cudaFuncSetAttribute(projection_ln_residual_tma<M>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename C::Shared)));
  auto stream=at::cuda::getCurrentCUDAStream();
  projection_ln_residual_tma<M><<<rows/M,160,sizeof(typename C::Shared),stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto small=torch::empty({16,256},mean.options());
  reduce_norm<<<dim3(16,16),128,0,stream>>>(partial.data_ptr<float>(),small.data_ptr<float>(),small.data_ptr<float>()+128,rows/M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  finish_norm<<<1,128,0,stream>>>(small.data_ptr<float>(),dw.data_ptr<float>(),db.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx,dw,db};
}
std::vector<torch::Tensor> backward(std::vector<torch::Tensor> dy,std::vector<torch::Tensor> w,
    torch::Tensor x,torch::Tensor mean,torch::Tensor rstd,torch::Tensor gamma,torch::Tensor residual,int L,bool ending) {
  TORCH_CHECK(dy.size()==5 && w.size()==5,"five gradients and weights required");
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.is_contiguous() && x.numel()==int64_t(L)*L*128 && L%8==0,"square BF16 input required, L multiple8");
  c10::cuda::CUDAGuard guard(x.device());
  int rows=L*L;
  for(int j=0;j<5;++j) {
    TORCH_CHECK(w[j].device()==x.device() && dy[j].device()==x.device(),"device mismatch");
    TORCH_CHECK(dy[j].scalar_type()==torch::kBFloat16 && w[j].scalar_type()==torch::kBFloat16,"BF16 required");
    TORCH_CHECK(dy[j].is_contiguous() && w[j].is_contiguous(),"contiguous inputs required");
    TORCH_CHECK(w[j].sizes()==torch::IntArrayRef({j<4?128:4,128}),"weight shape");
    TORCH_CHECK(dy[j].numel()==rows*(j<4?128:4),"gradient shape");
  }
  for(auto const& t:{mean,rstd,gamma}) TORCH_CHECK(t.device()==x.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous(),"FP32 contiguous statistics/scale required");
  TORCH_CHECK(mean.numel()==rows && rstd.numel()==rows && gamma.numel()==128,"statistics/scale shape");
  TORCH_CHECK(residual.device()==x.device() && residual.scalar_type()==torch::kBFloat16 && residual.is_contiguous() && residual.numel()==x.numel(),"residual gradient metadata mismatch");
  return launch<64>(dy,w,x,mean,rstd,gamma,residual,L,ending);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("backward",&backward);m.def("smem",[](){return sizeof(Config<64>::Shared);});}
