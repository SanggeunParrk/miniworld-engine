// Row-group dK/dV + shared-bias gradient. Native CUDA/TMA/WGMMA, B1 H4 D32.
// One producer warpgroup and two consumer warpgroups; each consumer owns four outer rows.
// dK/dV are final outputs. Only one FP32 bias partial per R rows reaches HBM.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include "fa3_utils.h"
using namespace cute;
using Element=cutlass::bfloat16_t;
constexpr float SCALE=0.1767766952966369f;
constexpr float LOG2E=1.44269504f;
__device__ __forceinline__ float exp2_approx(float x) {
  float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}


__device__ __forceinline__ float sm_load_bf16(uint32_t p) {
  uint16_t v; asm volatile("ld.shared.b16 %0, [%1];" : "=h"(v) : "r"(p) : "memory");
  return float(Element::bitcast(v));
}
__device__ __forceinline__ void sm_store_bf16(uint32_t p,Element v) {
  asm volatile("st.shared.b16 [%0], %1;" :: "r"(p),"h"(v.raw()) : "memory");
}
__device__ __forceinline__ float sm_load_f32(uint32_t p) {
  float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(p) : "memory"); return v;
}
__device__ __forceinline__ void sm_store_f32(uint32_t p,float v) {
  asm volatile("st.shared.f32 [%0], %1;" :: "r"(p),"f"(v) : "memory");
}

// PTX cvt packs its first input high, second input low (CUTLASS convention).
__device__ __forceinline__ void sm_store_pair(uint32_t p,float lo,float hi) {
  uint32_t packed;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(hi),"f"(lo));
  asm volatile("st.shared.b32 [%0], %1;" :: "r"(p),"r"(packed) : "memory");
}


// Two independent MMAs share one WGMMA fence/commit/wait group.
template<bool Zero, class MMA, class A0, class B0, class C0, class A1, class B1, class C1>
__device__ __forceinline__ void gemm_pair(MMA& mma,A0& a0,B0& b0,C0& c0,A1& a1,B1& b1,C1& c1){
  constexpr bool RS=!cute::is_base_of<cute::GMMA::DescriptorIterator,typename MMA::FrgTypeA>::value;
  if constexpr(RS){warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);}
  warpgroup_fence_operand(c0);warpgroup_fence_operand(c1);warpgroup_arrive();
  mma.accumulate_=Zero?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One;
  #pragma unroll
  for(int k=0;k<size<2>(a0);++k){cute::gemm(mma,a0(_,_,k),b0(_,_,k),c0);mma.accumulate_=GMMA::ScaleOut::One;}
  mma.accumulate_=Zero?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One;
  #pragma unroll
  for(int k=0;k<size<2>(a1);++k){cute::gemm(mma,a1(_,_,k),b1(_,_,k),c1);mma.accumulate_=GMMA::ScaleOut::One;}
  warpgroup_commit_batch();warpgroup_wait<0>();
  warpgroup_fence_operand(c0);warpgroup_fence_operand(c1);
  if constexpr(RS){warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);}
}

template<int R> struct Config {
  static constexpr int NWG=2, RP=R/NWG, QS=4;
  static_assert(R==8);
  using KL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));
  using QL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_32,_32>{}));
  using QT=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_32>{}));
  using SL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));
  using BL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_32,_64>{}));
  using ShapeQ=Shape<int32_t,_32,_4,int32_t>;
  using StrideQ=Stride<int64_t,_1,_32,int64_t>;
  using ShapeB=Shape<int32_t,int32_t,_4>;
  using StrideB=Stride<int64_t,_1,int64_t>;
  using TK=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),ShapeQ{},StrideQ{}),KL{},Shape<_64,_32>{},_1{}));
  using TQ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),ShapeQ{},StrideQ{}),QL{},Shape<_32,_32>{},_1{}));
  using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),ShapeB{},StrideB{}),BL{},Shape<_32,_64>{},_1{}));
  using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),ShapeQ{},StrideQ{}),KL{},Shape<_64,_32>{},_1{}));
  using ScoreMMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_32,_32>>()));
  using GradMMA=decltype(make_tiled_mma(GMMA::rs_op_selector<Element,Element,float,Shape<_64,_32,_32>,GMMA::Major::K,GMMA::Major::MN>()));
  struct Shared {
    array_aligned<Element,2048,1024> k[R],v[R];
    array_aligned<Element,1024,1024> q[NWG][QS],dout[NWG][QS];
    array_aligned<Element,2048,1024> bias[2],ds[NWG][2][RP];
    array_aligned<float,32,128> lse[NWG][QS],delta[NWG][QS];
    // dBias reduction reads the eight BF16 dS tiles directly.
    cutlass::arch::ClusterTransactionBarrier kv_full;
    cutlass::arch::ClusterTransactionBarrier q_full[NWG][QS],b_full[2];
    cutlass::arch::ClusterBarrier ds_ready[2],ds_empty[2];
    cutlass::arch::ClusterBarrier q_empty[NWG][QS],b_empty[2];
  };
  struct Params {
    TQ q; TK k,v; TQ dout; TB bias;
    float const *lse,*delta;
    Element *dk,*dv;
    float *db;
    int L; TO outk,outv;
  };
};

template<class TMA,class GT,class ST>
__device__ __forceinline__ void tma_copy(TMA const& tma,GT const& src,ST const& dst,cutlass::arch::ClusterTransactionBarrier& full) {
  auto c=tma.get_slice(_0{});
  copy(tma.with(reinterpret_cast<uint64_t&>(full)),c.partition_S(src),c.partition_D(dst));
}

template<int R> __global__ __launch_bounds__(384,1)
void grouped_dkdv(CUTE_GRID_CONSTANT typename Config<R>::Params const p) {
  using C=Config<R>;
  extern __shared__ char smem[];
  auto& s=*reinterpret_cast<typename C::Shared*>(smem);
  int tid=threadIdx.x, wg=tid/128-1, lane=tid%128;
  int h=blockIdx.z, group=blockIdx.y, kt=blockIdx.x;
  int L=p.L, nt=L/32, nkt=L/64;
  if(tid==0) {
    s.kv_full.init(C::NWG);
    for(int st=0;st<2;++st) {
      s.b_full[st].init(1);s.b_empty[st].init(C::NWG);
      s.ds_ready[st].init(C::NWG);s.ds_empty[st].init(1);
    }
    for(int st=0;st<C::QS;++st)
      for(int w=0;w<C::NWG;++w) {s.q_full[w][st].init(1);s.q_empty[w][st].init(1);}
    prefetch_tma_descriptor(p.q.get_tma_descriptor());prefetch_tma_descriptor(p.k.get_tma_descriptor());
    prefetch_tma_descriptor(p.v.get_tma_descriptor());prefetch_tma_descriptor(p.dout.get_tma_descriptor());
    prefetch_tma_descriptor(p.bias.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if(tid<128) {
    cutlass::arch::warpgroup_reg_dealloc<56>();
    if(tid%32==0 && tid/32<C::NWG) {
      int producer=tid/32;
      auto shape=make_shape(L,_32{},_4{},L);
      auto mk=p.k.get_tma_tensor(shape);auto mv=p.v.get_tma_tensor(shape);
      s.kv_full.arrive_and_expect_tx(C::RP*2*2048*sizeof(Element));
      for(int rr=0;rr<C::RP;++rr) {
        int r=producer*C::RP+rr;
        auto gk=local_tile(mk(_,_,h,group*R+r),Shape<_64,_32>{},make_coord(kt,0));
        auto gv=local_tile(mv(_,_,h,group*R+r),Shape<_64,_32>{},make_coord(kt,0));
        tma_copy(p.k,gk,make_tensor(make_smem_ptr(s.k[r].data()),typename C::KL{}),s.kv_full);
        tma_copy(p.v,gv,make_tensor(make_smem_ptr(s.v[r].data()),typename C::KL{}),s.kv_full);
      }
      auto mq=p.q.get_tma_tensor(shape);auto md=p.dout.get_tma_tensor(shape);
      auto mb=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
      for(int jt=0;jt<nt;++jt) {
        int bs=jt%2,bphase=(jt/2)%2;
        if(producer==0){
        s.b_empty[bs].wait(bphase^1);
        s.b_full[bs].arrive_and_expect_tx(2048*sizeof(Element));
        auto gb=local_tile(mb(_,_,h),Shape<_32,_64>{},make_coord(jt,kt));
        tma_copy(p.bias,gb,make_tensor(make_smem_ptr(s.bias[bs].data()),typename C::BL{}),s.b_full[bs]);
        }
        for(int rr=0;rr<C::RP;++rr) {
          int it=jt*C::RP+rr,st=it%C::QS,phase=(it/C::QS)%2;
          {
            int w=producer;
            int row=group*R+w*C::RP+rr;
            s.q_empty[w][st].wait(phase^1);
            s.q_full[w][st].arrive_and_expect_tx(2*1024*sizeof(Element)+2*32*sizeof(float));
            auto gq=local_tile(mq(_,_,h,row),Shape<_32,_32>{},make_coord(jt,0));
            auto gd=local_tile(md(_,_,h,row),Shape<_32,_32>{},make_coord(jt,0));
            tma_copy(p.q,gq,make_tensor(make_smem_ptr(s.q[w][st].data()),typename C::QL{}),s.q_full[w][st]);
            tma_copy(p.dout,gd,make_tensor(make_smem_ptr(s.dout[w][st].data()),typename C::QL{}),s.q_full[w][st]);
            int stat=(h*L+row)*L+jt*32;
            SM90_BULK_COPY_G2S::copy(p.lse+stat,reinterpret_cast<uint64_t*>(&s.q_full[w][st]),s.lse[w][st].data(),32*sizeof(float));
            SM90_BULK_COPY_G2S::copy(p.delta+stat,reinterpret_cast<uint64_t*>(&s.q_full[w][st]),s.delta[w][st].data(),32*sizeof(float));
          }
        }
      }
    }
    if(tid>=64){
      int rl=tid-64;
      for(int jt=0;jt<nt;++jt){
        int bs=jt%2,phase=(jt/2)%2;
        s.ds_ready[bs].wait(phase);
        int64_t base=(((int64_t(h)*(L/R)+group)*nt+jt)*nkt+kt)*2048;
        #pragma unroll
        for(int chunk=0;chunk<8;++chunk){
          int vec=chunk*64+rl;
          int lane=vec%128,xbase=(vec/128)*4;
          float values[4];
          #pragma unroll
          for(int local=0;local<4;local+=2){
            int x=xbase+local;
        int key=(lane/32)*16+(lane%32)/4+((x%4)/2)*8;
        int query=(lane%4)*2+(x/4)*8;
        int off=as_position_independent_swizzle_layout(typename C::SL{})(make_coord(key,query))*2;
        float sum0=0.f,sum1=0.f;
        #pragma unroll
        for(int w=0;w<C::NWG;++w) {
          #pragma unroll
          for(int rr=0;rr<C::RP;++rr){
            uint32_t pair,ptr=cast_smem_ptr_to_uint(s.ds[w][bs][rr].data())+off;
            asm volatile("ld.shared.b32 %0,[%1];":"=r"(pair):"r"(ptr):"memory");
            sum0+=float(Element::bitcast(uint16_t(pair)));
            sum1+=float(Element::bitcast(uint16_t(pair>>16)));
          }
        }
            values[local]=sum0;values[local+1]=sum1;
          }
          reinterpret_cast<float4*>(p.db+base)[vec]=make_float4(values[0],values[1],values[2],values[3]);
        }
        cutlass::arch::NamedBarrier::sync(64,6);
        if(rl==0)s.ds_empty[bs].arrive();
      }
    }
    return;
  }
  cutlass::arch::warpgroup_reg_alloc<224>();
  typename C::ScoreMMA smma;typename C::GradMMA gmma;
  auto st=smma.get_slice(lane);auto gt=gmma.get_slice(lane);
  auto gc=partition_fragment_C(gmma,Shape<_64,_32>{});
  auto gcoords=gt.partition_C(make_identity_tensor(Shape<_64,_32>{}));
  auto scoords=st.partition_C(make_identity_tensor(Shape<_64,_32>{}));
  constexpr int NG=decltype(size(gc))::value;
  float dkbuf[C::RP][NG],dvbuf[C::RP][NG];
  #pragma unroll
  for(int r=0;r<C::RP;++r) {
    #pragma unroll
    for(int x=0;x<NG;++x) dkbuf[r][x]=dvbuf[r][x]=0;
  }
  s.kv_full.wait(0);
  for(int jt=0;jt<nt;++jt) {
    int bs=jt%2,bphase=(jt/2)%2;
    s.ds_empty[bs].wait(bphase^1);
    s.b_full[bs].wait(bphase);
    uint32_t bp=cast_smem_ptr_to_uint(s.bias[bs].data());
    // Static row iteration keeps each row's persistent gradient in registers.
    cute::for_each(make_seq<C::RP>{},[&](auto rr_) {
      constexpr int rr=decltype(rr_)::value;
      uint32_t ps=cast_smem_ptr_to_uint(s.ds[wg][bs][rr].data());
      int r=wg*C::RP+rr,row=group*R+r;
      int it=jt*C::RP+rr,slot=it%C::QS,phase=(it/C::QS)%2;
      s.q_full[wg][slot].wait(phase);
      uint32_t mp=cast_smem_ptr_to_uint(s.lse[wg][slot].data()),deltap=cast_smem_ptr_to_uint(s.delta[wg][slot].data());
      auto sq=make_tensor(make_smem_ptr(s.q[wg][slot].data()),typename C::QL{});
      auto sd=make_tensor(make_smem_ptr(s.dout[wg][slot].data()),typename C::QL{});
      auto sk=make_tensor(make_smem_ptr(s.k[r].data()),typename C::KL{});
      auto sv=make_tensor(make_smem_ptr(s.v[r].data()),typename C::KL{});
      auto score=partition_fragment_C(smma,Shape<_64,_32>{});
      auto dp=partition_fragment_C(smma,Shape<_64,_32>{});
      auto ka=st.partition_fragment_A(sk);auto qb=st.partition_fragment_B(sq);
      auto va=st.partition_fragment_A(sv);auto dob=st.partition_fragment_B(sd);
      gemm_pair<true>(smma,ka,qb,score,va,dob,dp);
      // ldmatrix transposes each bias 8x8 block directly into score-fragment
      // register order. No extra shared buffer or CTA barrier is required.
      #pragma unroll
      for(int xb=0;xb<size(score);xb+=8) {
        int qbase=xb*2+(lane%32/16)*8;
        int kbase=(lane/32)*16+(lane%16/8)*8;
        uint32_t addr=bp+as_position_independent_swizzle_layout(typename C::BL{})(make_coord(qbase+lane%8,kbase))*2;
        uint32_t bv[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
          :"=r"(bv[0]),"=r"(bv[1]),"=r"(bv[2]),"=r"(bv[3]):"r"(addr):"memory");
        #pragma unroll
        for(int yi=0;yi<8;yi+=4) {
          int query0=get<1>(scoords(xb+yi));
          float mm0=fmaxf(sm_load_f32(mp+query0*4),-1e38f);
          float mm1=fmaxf(sm_load_f32(mp+(query0+1)*4),-1e38f);
          float dd0=sm_load_f32(deltap+query0*4);
          float dd1=sm_load_f32(deltap+(query0+1)*4);
          #pragma unroll
          for(int xi=0;xi<4;++xi) {
            int x=xb+yi+xi;
            float bias=float(Element::bitcast(uint16_t(bv[(yi+xi)/2]>>((xi%2)*16))));
            float logit=score(x)+bias*(1.f/SCALE);
            float pr=exp2_approx(logit*(SCALE*LOG2E)-(xi%2?mm1:mm0));
            dp(x)=pr*(dp(x)-(xi%2?dd1:dd0));score(x)=pr;
          }
        }
      }
      #pragma unroll
      for(int x=0;x<size(score);x+=2) {
        int key=get<0>(scoords(x)),query=get<1>(scoords(x));
        int off=as_position_independent_swizzle_layout(typename C::SL{})(make_coord(key,query))*2;
        sm_store_pair(ps+off,dp(x),dp(x+1));
      }
      auto qt=make_tensor(make_smem_ptr(s.q[wg][slot].data()),typename C::QT{});
      auto dot=make_tensor(make_smem_ptr(s.dout[wg][slot].data()),typename C::QT{});
      auto acc_p=make_tensor(score.data(),flash::convert_layout_acc_Aregs<typename C::GradMMA>(score.layout()));
      auto acc_ds=make_tensor(dp.data(),flash::convert_layout_acc_Aregs<typename C::GradMMA>(dp.layout()));
      auto pa=make_tensor_like<Element>(acc_p);auto dsa=make_tensor_like<Element>(acc_ds);
      flash::convert_type_out(acc_p,pa);flash::convert_type_out(acc_ds,dsa);
      auto qbt=gt.partition_fragment_B(qt);auto dbt=gt.partition_fragment_B(dot);
      auto dk=make_tensor(make_rmem_ptr(dkbuf[rr]),gc.layout());
      auto dv=make_tensor(make_rmem_ptr(dvbuf[rr]),gc.layout());
      gemm_pair<false>(gmma,pa,dbt,dv,dsa,qbt,dk);
      cutlass::arch::NamedBarrier::sync(128,wg+1);
      if(lane==0) s.q_empty[wg][slot].arrive();
    });
    // The last per-WG reader barrier also publishes every dS store.
    if(lane==0)s.ds_ready[bs].arrive();
    // Bias input can retire; dS reuse waits for the independent reducer.
    if(lane==0) s.b_empty[bs].arrive();
  }
  #pragma unroll
  for(int rr=0;rr<C::RP;++rr) {
    int r=wg*C::RP+rr,row=group*R+r;
    uint32_t skp=cast_smem_ptr_to_uint(s.k[r].data()),svp=cast_smem_ptr_to_uint(s.v[r].data());
    #pragma unroll
    for(int x=0;x<NG;x+=2) {
      int key=get<0>(gcoords(x)),d=get<1>(gcoords(x));
      int off=as_position_independent_swizzle_layout(typename C::KL{})(make_coord(key,d))*2;
      sm_store_pair(skp+off,dkbuf[rr][x]*SCALE,dkbuf[rr][x+1]*SCALE);
      sm_store_pair(svp+off,dvbuf[rr][x],dvbuf[rr][x+1]);
    }
    cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,wg+1);
    if(lane==0){
      auto shape=make_shape(L,_32{},_4{},L);
      auto gk=p.outk.get_tma_tensor(shape);auto gv=p.outv.get_tma_tensor(shape);
      auto ktile=local_tile(gk(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
      auto vtile=local_tile(gv(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
      auto ks=make_tensor(make_smem_ptr(s.k[r].data()),typename C::KL{});
      auto vs=make_tensor(make_smem_ptr(s.v[r].data()),typename C::KL{});
      auto ck=p.outk.get_slice(_0{});auto cv=p.outv.get_slice(_0{});
      copy(p.outk,ck.partition_S(ks),ck.partition_D(ktile));
      copy(p.outv,cv.partition_S(vs),cv.partition_D(vtile));
      tma_store_arrive();tma_store_wait<0>();
    }
  }

}

__global__ void reduce_bias(float const* part,Element* out,int L,int groups) {
  int h=blockIdx.z,nt=L/32,nkt=L/64;
  float acc[8]={};
  for(int g=0;g<groups;++g) {
    int64_t base=(((int64_t(h)*groups+g)*nt+blockIdx.y)*nkt+blockIdx.x)*512+threadIdx.x;
    #pragma unroll
    for(int j=0;j<2;++j) {
      float4 x=reinterpret_cast<float4 const*>(part)[base+j*256];
      acc[j*4+0]+=x.x;acc[j*4+1]+=x.y;acc[j*4+2]+=x.z;acc[j*4+3]+=x.w;
    }
  }
  #pragma unroll
  for(int j=0;j<2;++j) {
    #pragma unroll
    for(int c=0;c<4;++c) {
      int idx=(threadIdx.x+j*256)*4+c,lane=(idx/4)%128,x=(idx/512)*4+((idx%4)/2)*2+(idx%2);
      int k=blockIdx.x*64+(lane/32)*16+(lane%32)/4+((x%4)/2)*8;
      int q=blockIdx.y*32+(lane%4)*2+(x%2)+(x/4)*8;
      out[(h*L+q)*L+k]=Element(acc[j*4+c]);
    }
  }
}

// Small lengths need more independent output CTAs to fill the GPU.
__global__ void reduce_bias_small(float const* part,Element* out,int L,int groups) {
  int h=blockIdx.z,nt=L/32,nkt=L/64,jt=blockIdx.y/4,chunk=blockIdx.y%4;
  float a=0.f,b=0.f;
  for(int g=0;g<groups;++g) {
    int64_t base=(((int64_t(h)*groups+g)*nt+jt)*nkt+blockIdx.x)*1024+chunk*256+threadIdx.x;
    float2 x=reinterpret_cast<float2 const*>(part)[base];a+=x.x;b+=x.y;
  }
  #pragma unroll
  for(int c=0;c<2;++c) {
    int idx=chunk*512+threadIdx.x*2+c,lane=(idx/4)%128,x=(idx/512)*4+((idx%4)/2)*2+(idx%2);
    int k=blockIdx.x*64+(lane/32)*16+(lane%32)/4+((x%4)/2)*8;
    int q=jt*32+(lane%4)*2+(x%2)+(x/4)*8;
    out[(h*L+q)*L+k]=Element(c?b:a);
  }
}

template<int R> std::vector<torch::Tensor> launch(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor b,torch::Tensor lse,torch::Tensor delta,torch::Tensor dout) {
  using C=Config<R>;int L=q.size(2);
  auto dk=torch::empty({1,L,L,128},q.options());auto dv=torch::empty_like(dk);
  auto part=torch::empty({4,L/R,L,L},q.options().dtype(torch::kFloat32));
  auto db=torch::empty({1,4,L,L},q.options());
  auto shape=make_shape(L,_32{},_4{},L);
  auto make_q=[&](torch::Tensor const& x,auto tile_rows) {
    using Layout=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<decltype(tile_rows),_32>{}));
    auto mx=make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),shape,typename C::StrideQ{x.stride(3),_1{},_32{},x.stride(2)});
    return make_tma_copy(SM90_TMA_LOAD{},mx,Layout{},Shape<decltype(tile_rows),_32>{},_1{});
  };
  auto mb=make_tensor(make_gmem_ptr((Element const*)b.data_ptr()),make_shape(L,L,_4{}),typename C::StrideB{L,_1{},int64_t(L)*L});
  auto tb=make_tma_copy(SM90_TMA_LOAD{},mb,typename C::BL{},Shape<_32,_64>{},_1{});
  typename C::Params p{make_q(q,_32{}),make_q(k,_64{}),make_q(v,_64{}),make_q(dout,_32{}),tb,lse.data_ptr<float>(),delta.data_ptr<float>(),(Element*)dk.data_ptr(),(Element*)dv.data_ptr(),part.data_ptr<float>(),L,{},{}};
  auto make_out=[&](torch::Tensor const& x){auto g=make_tensor(make_gmem_ptr((Element*)x.data_ptr()),shape,typename C::StrideQ{128,_1{},_32{},int64_t(L)*128});return make_tma_copy(SM90_TMA_STORE{},g,typename C::KL{},Shape<_64,_32>{},_1{});};
  p.outk=make_out(dk);p.outv=make_out(dv);
  C10_CUDA_CHECK(cudaFuncSetAttribute(grouped_dkdv<R>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename C::Shared)));
  auto stream=at::cuda::getCurrentCUDAStream();
  grouped_dkdv<R><<<dim3(L/64,L/R,4),384,sizeof(typename C::Shared),stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if(L<=384) reduce_bias_small<<<dim3(L/64,L/8,4),256,0,stream>>>(part.data_ptr<float>(),(Element*)db.data_ptr(),L,L/R);
  else reduce_bias<<<dim3(L/64,L/32,4),256,0,stream>>>(part.data_ptr<float>(),(Element*)db.data_ptr(),L,L/R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dk.view({1,L,L,4,32}).permute({0,3,1,2,4}),dv.view({1,L,L,4,32}).permute({0,3,1,2,4}),db};
}

std::vector<torch::Tensor> backward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor b,torch::Tensor lse,torch::Tensor delta,torch::Tensor dout,int R) {
  TORCH_CHECK(q.is_cuda() && q.dim()==5 && q.size(0)==1 && q.size(1)==4 && q.size(4)==32,"B1 H4 D32 required");
  c10::cuda::CUDAGuard guard(q.device());int L=q.size(2);
  TORCH_CHECK(L>=64 && L<=1024 && L%64==0 && q.size(3)==L,"square length multiple64, at most1024 required");
  for(auto const& x:{q,k,v,dout}) {
    TORCH_CHECK(x.device()==q.device() && x.scalar_type()==torch::kBFloat16 && x.sizes()==q.sizes(),"QKV/dO shape/device/dtype mismatch");
    TORCH_CHECK(x.stride(4)==1 && x.stride(1)==32 && x.stride(3)==128 && x.stride(2)==L*128,"projection layout required");
  }
  TORCH_CHECK(b.device()==q.device() && b.scalar_type()==torch::kBFloat16 && b.sizes()==torch::IntArrayRef({1,4,L,L}) && b.is_contiguous(),"contiguous BF16 bias required");
  for(auto const& x:{lse,delta}) TORCH_CHECK(x.device()==q.device() && x.scalar_type()==torch::kFloat32 && x.is_contiguous() && x.numel()==4*L*L,"FP32 contiguous statistics required");
  TORCH_CHECK(R==4 || R==8,"legacy group4 ABI or native group8 required");
  return launch<8>(q,k,v,b,lse,delta,dout);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.attr("row_group")=8;m.def("backward",&backward);m.def("smem",[](){return std::vector<int>{sizeof(Config<8>::Shared)};});}
