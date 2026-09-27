// Training forward: one score buffer, complete QK groups before scalar score reads.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include "fa3_utils.h"
using namespace cute;
using Element = cutlass::bfloat16_t;
constexpr float SCALE=0.1767766952966369f, LOG2E=1.44269504f;
__device__ __forceinline__ float ex2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ void store_pair(uint32_t p,float a,float b) {
  uint32_t v; asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(v):"f"(b),"f"(a));
  asm volatile("st.shared.b32 [%0],%1;"::"r"(p),"r"(v):"memory");
}

template<int Capacity,int Consumers,int Stages,int Projectors> struct Config {
  using QL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));
  using VT=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_64>{}));
  using BL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));
  using ZL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_128>{}));
  using WL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_32,_64>{}));
  using ZS=Shape<int32_t,_128,int32_t>; using ZStride=Stride<_128,_1,int64_t>;
  using WS=Shape<_128,_128>; using WStride=Stride<_128,_1>;
  using QS=Shape<int32_t,_32,_4,int32_t>; using QStride=Stride<int64_t,_1,_32,int64_t>;
  using BS=Shape<int32_t,int32_t,_4>; using BStride=Stride<int64_t,_1,int64_t>;
  using TZ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),ZS{},ZStride{}),ZL{},Shape<_64,_128>{},_1{}));
  using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),WS{},WStride{}),WL{},Shape<_32,_64>{},_1{}));
  using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),BS{},BStride{}),BL{},Shape<_64,_64>{},_1{}));
  using TQ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),QS{},QStride{}),QL{},Shape<_64,_32>{},_1{}));
  using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),QS{},QStride{}),QL{},Shape<_64,_32>{},_1{}));
  using Proj=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_64,_128>>()));
  using Score=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_64,_32>>()));
  using PV=decltype(make_tiled_mma(GMMA::rs_op_selector<Element,Element,float,Shape<_64,_32,_64>,GMMA::Major::K,GMMA::Major::MN>()));
  union alignas(1024) Scratch {
    struct { array_aligned<Element,8192,1024> z[Projectors]; array_aligned<Element,8192,1024> w[2]; } proj;
    struct {
      array_aligned<Element,2048,1024> q[Consumers];
      array_aligned<Element,4096,1024> bias[Consumers][Stages];
    } attn;
  };
  struct Shared {
    array_aligned<Element,2048,1024> kv[2][Capacity/64];
    Scratch scratch;
    cutlass::arch::ClusterTransactionBarrier zfull[Consumers],qfull[Consumers],wfull,bfull[Consumers][Stages];
  };
  struct Params { TZ z; TW wq,wg,wk,wv; TB bias; TQ loadq; TO saveq,saveg,savek,savev,out; float* lse; int L; };
};
template<class T,class G,class S>
__device__ __forceinline__ void tma_load(T const& t,G const& g,S const& s,
    cutlass::arch::ClusterTransactionBarrier& bar) {
  auto c=t.get_slice(_0{});
  copy(t.with(reinterpret_cast<uint64_t&>(bar)),c.partition_S(g),c.partition_D(s));
}
template<int Capacity,int Consumers,int Stages,int Projectors>
__global__ __launch_bounds__(Consumers*128,1)
void qkv_attention_compact(CUTE_GRID_CONSTANT typename Config<Capacity,Consumers,Stages,Projectors>::Params const p) {
  using C=Config<Capacity,Consumers,Stages,Projectors>;
  extern __shared__ char storage[];
  auto& s=*reinterpret_cast<typename C::Shared*>(storage);
  int tid=threadIdx.x,c=tid/128,lane=tid%128,h=blockIdx.x,row=blockIdx.y,L=p.L,nt=L/64;
  if(tid==0) {
    for(int cc=0;cc<Consumers;++cc) {s.zfull[cc].init(1);s.qfull[cc].init(1);} s.wfull.init(1);
    for(int cc=0;cc<Consumers;++cc)for(int b=0;b<Stages;++b)s.bfull[cc][b].init(1);
    prefetch_tma_descriptor(p.z.get_tma_descriptor());
    prefetch_tma_descriptor(p.wq.get_tma_descriptor());
    prefetch_tma_descriptor(p.wg.get_tma_descriptor());
    prefetch_tma_descriptor(p.wk.get_tma_descriptor());
    prefetch_tma_descriptor(p.wv.get_tma_descriptor());
    prefetch_tma_descriptor(p.bias.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  // Fewer projection WGs keep full K128 Z tiles without increasing attention scratch.
  if(c<Projectors) {
    typename C::Proj mma;auto mt=mma.get_slice(lane);
    auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_64>{}));
    auto acc=partition_fragment_C(mma,Shape<_64,_64>{});
    auto zg=p.z.get_tma_tensor(make_shape(L,_128{},L));
    auto zs=make_tensor(make_smem_ptr(s.scratch.proj.z[c].data()),typename C::ZL{});
    auto za=mt.partition_fragment_A(zs);
    if(tid==0) {
      s.wfull.arrive_and_expect_tx(4*4096*sizeof(Element));
      #pragma unroll
      for(int which=0;which<4;++which) {
        auto const& wt=which==0?p.wq:(which==1?p.wg:(which==2?p.wk:p.wv));
        auto wg=wt.get_tma_tensor(make_shape(_128{},_128{}));
        auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w[which/2].data()),typename C::ZL{});
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk) {
          auto src=local_tile(wg,Shape<_32,_64>{},make_coord(h,chunk));
          auto dst=local_tile(ws,Shape<_32,_64>{},make_coord(which%2,chunk));
          tma_load(wt,src,dst,s.wfull);
        }
      }
    }
    s.wfull.wait(0);
    for(int qt=c;qt<nt;qt+=Projectors) {
      if(lane==0) {
        s.zfull[c].arrive_and_expect_tx(8192*sizeof(Element));
        auto zt=local_tile(zg(_,_,row),Shape<_64,_128>{},make_coord(qt,0));
        tma_load(p.z,zt,zs,s.zfull[c]);
      }
      s.zfull[c].wait((qt/Projectors)%2);
      #pragma unroll
      for(int turn=0;turn<2;++turn) {
        int pair=1-turn;
        auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w[pair].data()),typename C::ZL{});
        auto wb=mt.partition_fragment_B(ws);
        flash::gemm<true,0>(mma,za,wb,acc);
        #pragma unroll
        for(int x=0;x<size(acc);x+=2) {
          int mr=get<0>(coord(x)),nd=get<1>(coord(x)),which=nd/32;
          Element* dst=pair==0?s.scratch.proj.z[c].data()+which*2048:s.kv[which][qt].data();
          int off=as_position_independent_swizzle_layout(typename C::QL{})(make_coord(mr,nd%32))*2;
          store_pair(cast_smem_ptr_to_uint(dst)+off,acc(x),acc(x+1));
        }
        cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,c+1);
        if(lane==0) {
          #pragma unroll
          for(int which=0;which<2;++which) {
            auto const& save=pair==0?(which==0?p.saveq:p.saveg):(which==0?p.savek:p.savev);
            auto sg=save.get_tma_tensor(make_shape(L,_32{},_4{},L));
            auto tile=local_tile(sg(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
            auto src=make_tensor(make_smem_ptr(pair==0?s.scratch.proj.z[c].data()+which*2048:s.kv[which][qt].data()),typename C::QL{});
            auto cp=save.get_slice(_0{});copy(save,cp.partition_S(src),cp.partition_D(tile));
          }
          tma_store_arrive();
        }
      }
      // Q/gate sources alias Z: retire source reads before loading the next Z tile.
      if(lane==0)tma_store_wait<0>();
      cutlass::arch::NamedBarrier::sync(128,c+1);
    }
    // Unlike source-only retirement, this also completes global Q writes for reload.
    if(lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
  }
  // QKV saves are complete. The projection scratch may become bias storage.
  __syncthreads();
  auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
  typename C::Score smma; typename C::PV pmma;
  auto st=smma.get_slice(lane); auto pt=pmma.get_slice(lane);
  for(int qt=c;qt<nt;qt+=Consumers) {
    auto load_bias=[&](int kt) {
      int stage=kt%Stages;
      s.bfull[c][stage].arrive_and_expect_tx(4096*sizeof(Element));
      auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qt,kt));
      tma_load(p.bias,bb,make_tensor(make_smem_ptr(s.scratch.attn.bias[c][stage].data()),typename C::BL{}),s.bfull[c][stage]);
    };
    if(lane==0) {
      auto qg=p.loadq.get_tma_tensor(make_shape(L,_32{},_4{},L));
      auto qq=local_tile(qg(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
      s.qfull[c].arrive_and_expect_tx(2048*sizeof(Element));
      tma_load(p.loadq,qq,make_tensor(make_smem_ptr(s.scratch.attn.q[c].data()),typename C::QL{}),s.qfull[c]);
      for(int b=0;b<Stages && b<nt;++b)load_bias(b);
    }
    s.qfull[c].wait((qt/Consumers)%2);
    auto score=partition_fragment_C(smma,Shape<_64,_64>{});
    auto out=partition_fragment_C(pmma,Shape<_64,_32>{}); clear(out);
    float m[2]={-INFINITY,-INFINITY},l[2]={1.f,1.f};
    auto sq=make_tensor(make_smem_ptr(s.scratch.attn.q[c].data()),typename C::QL{});
    auto qa=st.partition_fragment_A(sq);
    for(int kt=0;kt<nt;++kt) {
      int stage=kt%Stages;
      auto sk=make_tensor(make_smem_ptr(s.kv[0][kt].data()),typename C::QL{});
      auto kb=st.partition_fragment_B(sk);
      flash::gemm<true,0>(smma,qa,kb,score);
      cutlass::arch::NamedBarrier::sync(128,c+1);
      if constexpr(Stages==2) {
        if(lane==0 && kt>0 && kt+1<nt)load_bias(kt+1);
      }
      s.bfull[c][stage].wait(((qt/Consumers)*(nt/Stages)+kt/Stages)%2);
      asm volatile("":::"memory");
        uint32_t bp=cast_smem_ptr_to_uint(s.scratch.attn.bias[c][stage].data());
        float mx[2]={-INFINITY,-INFINITY};
        #pragma unroll
        for(int xb=0;xb<size(score);xb+=8) {
          int qr=(lane/32)*16+(lane%16), kr=xb*2+(lane%32/16)*8;
          uint32_t addr=bp+as_position_independent_swizzle_layout(typename C::BL{})(make_coord(qr,kr))*2;
          uint32_t bv[4];
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
            :"=r"(bv[0]),"=r"(bv[1]),"=r"(bv[2]),"=r"(bv[3]):"r"(addr):"memory");
          #pragma unroll
          for(int xi=0;xi<8;++xi) {
            int x=xb+xi, mi=(x%4)/2;
            float bias=float(Element::bitcast(uint16_t(bv[xi/2]>>((xi%2)*16))));
            score(x)=score(x)+bias*(1.f/SCALE);
            mx[mi]=fmaxf(mx[mi],score(x));
          }
        }
        if constexpr(Stages==1) {
          cutlass::arch::NamedBarrier::sync(128,c+1);
          if(lane==0 && kt+1<nt)load_bias(kt+1);
        }
        float alpha[2], ls[2]={0.f,0.f};
        #pragma unroll
        for(int mi=0;mi<2;++mi) {
          mx[mi]=fmaxf(mx[mi],__shfl_xor_sync(0xffffffffu,mx[mi],1));
          mx[mi]=fmaxf(mx[mi],__shfl_xor_sync(0xffffffffu,mx[mi],2));
          float mn=fmaxf(fmaxf(m[mi],mx[mi]*(SCALE*LOG2E)),-1e38f);
          alpha[mi]=ex2(m[mi]-mn); m[mi]=mn;
        }
        #pragma unroll
        for(int x=0;x<size(score);++x) {
          int mi=(x%4)/2;
          score(x)=ex2(score(x)*(SCALE*LOG2E)-m[mi]);
          ls[mi]+=score(x);
        }
        #pragma unroll
        for(int mi=0;mi<2;++mi) {
          ls[mi]+=__shfl_xor_sync(0xffffffffu,ls[mi],1);
          ls[mi]+=__shfl_xor_sync(0xffffffffu,ls[mi],2);
          l[mi]=l[mi]*alpha[mi]+ls[mi];
        }
        auto ar=make_tensor(score.data(),flash::convert_layout_acc_Aregs<typename C::PV>(score.layout()));
        auto pr=make_tensor_like<Element>(ar); flash::convert_type_out(ar,pr);
        warpgroup_wait<0>(); warpgroup_fence_operand(out);
        #pragma unroll
        for(int x=0;x<size(out);++x) out(x)*=alpha[(x%4)/2];
        auto vv=make_tensor(make_smem_ptr(s.kv[1][kt].data()),typename C::VT{});
        auto vb=pt.partition_fragment_B(vv);
        flash::gemm<false,0>(pmma,pr,vb,out);

    }
    warpgroup_wait<0>(); warpgroup_fence_operand(out);
    int ep_tid,ep_h,ep_row,ep_nt;
    asm volatile("mov.u32 %0, %%tid.x;":"=r"(ep_tid));
    asm volatile("mov.u32 %0, %%ctaid.x;":"=r"(ep_h));
    asm volatile("mov.u32 %0, %%ctaid.y;":"=r"(ep_row));
    asm volatile("mov.u32 %0, %1;":"=r"(ep_nt):"r"(nt));
    int ep_lane=ep_tid%128,ep_c=ep_tid/128,ep_L=ep_nt*64;
    auto sc=smma.get_slice(ep_lane).partition_C(make_identity_tensor(Shape<_64,_64>{}));
    auto oc=pmma.get_slice(ep_lane).partition_C(make_identity_tensor(Shape<_64,_32>{}));
    float inv[2];
    #pragma unroll
    for(int mi=0;mi<2;++mi) {
      float den=l[mi]>0.f?l[mi]:1.f; inv[mi]=1.f/den;
      if(ep_lane%4==0) {
        int qr=get<0>(sc(2*mi));
        p.lse[(ep_h*ep_L+ep_row)*ep_L+qt*64+qr]=m[mi]+log2f(den);
      }
    }
    uint32_t op=cast_smem_ptr_to_uint(s.scratch.attn.q[ep_c].data());
    #pragma unroll
    for(int x=0;x<size(out);x+=2) {
      int qr=get<0>(oc(x)), d=get<1>(oc(x));
      int off=as_position_independent_swizzle_layout(typename C::QL{})(make_coord(qr,d))*2;
      store_pair(op+off,out(x)*inv[(x%4)/2],out(x+1)*inv[(x%4)/2]);
    }
    cutlass::arch::fence_view_async_shared(); cutlass::arch::NamedBarrier::sync(128,ep_c+1);
    if(ep_lane==0) {
      auto og=p.out.get_tma_tensor(make_shape(ep_L,_32{},_4{},ep_L));
      auto tile=local_tile(og(_,_,ep_h,ep_row),Shape<_64,_32>{},make_coord(qt,0));
      auto src=make_tensor(make_smem_ptr(s.scratch.attn.q[ep_c].data()),typename C::QL{}); auto ep_c=p.out.get_slice(_0{});
      copy(p.out,ep_c.partition_S(src),ep_c.partition_D(tile)); tma_store_arrive(); tma_store_wait<0>();
    }
  }
}

template<int Capacity,int Consumers,int Stages,int Projectors>
void launch(torch::Tensor z,torch::Tensor wq,torch::Tensor wk,torch::Tensor wv,torch::Tensor wg,torch::Tensor b,
            torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor gate,torch::Tensor out,torch::Tensor lse) {
  using C=Config<Capacity,Consumers,Stages,Projectors>; int L=z.size(1);
  auto zg=make_tensor(make_gmem_ptr((Element const*)z.data_ptr()),make_shape(L,_128{},L),typename C::ZStride{_128{},_1{},int64_t(L)*128});
  auto tz=make_tma_copy(SM90_TMA_LOAD{},zg,typename C::ZL{},Shape<_64,_128>{},_1{});
  auto weight=[&](torch::Tensor const& w) {
    auto g=make_tensor(make_gmem_ptr((Element const*)w.data_ptr()),typename C::WS{},typename C::WStride{});
    return make_tma_copy(SM90_TMA_LOAD{},g,typename C::WL{},Shape<_32,_64>{},_1{});
  };
  auto bg=make_tensor(make_gmem_ptr((Element const*)b.data_ptr()),make_shape(L,L,_4{}),typename C::BStride{L,_1{},int64_t(L)*L});
  auto tb=make_tma_copy(SM90_TMA_LOAD{},bg,typename C::BL{},Shape<_64,_64>{},_1{});
  auto save=[&](torch::Tensor const& t) {
    auto g=make_tensor(make_gmem_ptr((Element*)t.data_ptr()),make_shape(L,_32{},_4{},L),typename C::QStride{128,_1{},_32{},int64_t(L)*128});
    return make_tma_copy(SM90_TMA_STORE{},g,typename C::QL{},Shape<_64,_32>{},_1{});
  };
  auto qg=make_tensor(make_gmem_ptr((Element const*)q.data_ptr()),make_shape(L,_32{},_4{},L),typename C::QStride{128,_1{},_32{},int64_t(L)*128});
  auto tq=make_tma_copy(SM90_TMA_LOAD{},qg,typename C::QL{},Shape<_64,_32>{},_1{});
  typename C::Params p{tz,weight(wq),weight(wg),weight(wk),weight(wv),tb,tq,save(q),save(gate),save(k),save(v),save(out),lse.data_ptr<float>(),L};
  C10_CUDA_CHECK(cudaFuncSetAttribute(qkv_attention_compact<Capacity,Consumers,Stages,Projectors>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename C::Shared)));
  qkv_attention_compact<Capacity,Consumers,Stages,Projectors><<<dim3(4,L),Consumers*128,sizeof(typename C::Shared),at::cuda::getCurrentCUDAStream()>>>(p);
}
std::vector<torch::Tensor> launch_forward(torch::Tensor z,torch::Tensor wq,torch::Tensor wk,torch::Tensor wv,torch::Tensor wg,torch::Tensor b) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type()==torch::kBFloat16 && z.is_contiguous() && z.dim()==4 && z.size(0)==1 && z.size(3)==128,"contiguous B1 L L C128 BF16 Z required");
  c10::cuda::CUDAGuard guard(z.device()); int L=z.size(1);
  TORCH_CHECK(z.size(2)==L && (L==64 || L==128 || L==256 || L==384 || L==768 || L==1024),"unsupported shape");
  for(auto const& w:{wq,wk,wv,wg})TORCH_CHECK(w.device()==z.device() && w.scalar_type()==z.scalar_type() && w.is_contiguous() && w.sizes()==torch::IntArrayRef({128,128}),"invalid projection weight");
  TORCH_CHECK(b.device()==z.device() && b.scalar_type()==z.scalar_type() && b.is_contiguous() && b.sizes()==torch::IntArrayRef({1,4,L,L}),"invalid bias");
  auto gate=torch::empty_like(z);
  auto q=torch::empty_like(z),k=torch::empty_like(z),v=torch::empty_like(z),o=torch::empty_like(z);
  auto lse=torch::empty({1,4,L,L},z.options().dtype(torch::kFloat32));
  if(L<=128)launch<128,2,2,1>(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse);
  else if(L<=384)launch<384,6,2,6>(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse);
  else if(L==768)launch<768,6,2,6>(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse);
  else launch<1024,6,1,4>(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto view=[&](torch::Tensor const& x){return x.view({1,L,L,4,32}).permute({0,3,1,2,4});};
  return {view(o),lse,view(q),view(k),view(v),gate};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("forward",&launch_forward);
  m.def("smem",[](){return sizeof(Config<1024,6,1,4>::Shared);});
}
