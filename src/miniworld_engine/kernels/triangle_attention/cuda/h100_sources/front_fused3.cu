// Hopper front stage v3: LayerNorm + [Wq;Wk;Wv;Wg;Wb] projection in one kernel.
// v2 re-streamed the whole weight matrix from L2 for every 64-row tile, which made
// L2 bandwidth the limit at C>=256. Here one CTA owns 128 rows: two warpgroups
// (64 rows each) walk the same 128-column weight blocks through one shared TMA ring,
// so each weight byte is fetched once per 128 rows. LN is computed once per tile into
// shared memory; Q/K/V/G leave through TMA stores, the head bias is transposed in
// shared memory so each head writes contiguous keys.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include "fa3_utils.h"
using namespace cute;
using Element=cutlass::bfloat16_t;
#ifndef MW_F3_STAGES
#define MW_F3_STAGES 4
#endif
constexpr int BN=128;
#ifndef MW_F3_SLOTS
#define MW_F3_SLOTS 1
#endif
constexpr int SLOTS=MW_F3_SLOTS;
constexpr int stages_for(int){return MW_F3_STAGES;}
#ifdef MW_F3_MINB_FORCE
constexpr int minb_for(int){return MW_F3_MINB_FORCE;}
#else
constexpr int minb_for(int C){return C<=128?2:1;}
#endif
using TileLayout=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));
using WL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<Int<BN>,_64>{}));
using GS=Shape<int32_t,int32_t>;
using ST=Stride<int64_t,_1>;
using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GS{},ST{}),WL{},Shape<Int<BN>,_64>{},_1{}));
using OS=Shape<int32_t,int32_t,int32_t>;
using OStride=Stride<int64_t,_1,int64_t>;
using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),OS{},OStride{}),TileLayout{},Shape<_64,_64>{},_1{}));

template<int C> struct Shared {
  array_aligned<Element,128*C,1024> z;                 // [wg][k tile][64 x 64]
  array_aligned<Element,BN*64,1024> w[stages_for(C)];  // shared weight ring
  array_aligned<Element,64*BN,1024> st[2*SLOTS];
  cutlass::arch::ClusterTransactionBarrier full[stages_for(C)];
  cutlass::arch::ClusterBarrier empty[stages_for(C)];
};

template<int C> __global__ __launch_bounds__(256,minb_for(C))
void front_v3(CUTE_GRID_CONSTANT TW const tw,CUTE_GRID_CONSTANT TO const to,
              Element const* x,float const* lnw,float const* lnb,bool const* mask,
              Element* bias,float* stats,int rows,int L,int H,float eps) {
  constexpr int NK=C/64,COLS=4*C+64,NB=(COLS+BN-1)/BN,STAGES=stages_for(C),total=NB*NK;
  extern __shared__ char memory[]; auto& s=*reinterpret_cast<Shared<C>*>(memory);
  int tid=threadIdx.x,wgi=tid/128,wt=tid%128,base=blockIdx.x*128;
  if(tid==0) {
    for(int i=0;i<STAGES;++i){s.full[i].init(1);s.empty[i].init(2);}
    prefetch_tma_descriptor(tw.get_tma_descriptor());prefetch_tma_descriptor(to.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  auto wg=tw.get_tma_tensor(make_shape(COLS,C));
  auto issue=[&](int item) {
    int stage=item%STAGES;
    if(item>=STAGES) s.empty[stage].wait(((item/STAGES)-1)&1);
    s.full[stage].arrive_and_expect_tx(BN*64*sizeof(Element));
    auto ww=local_tile(wg,Shape<Int<BN>,_64>{},make_coord(item/NK,item%NK));
    auto sw=make_tensor(make_smem_ptr(s.w[stage].data()),WL{});
    auto t=tw.get_slice(_0{});
    copy(tw.with(reinterpret_cast<uint64_t&>(s.full[stage])),t.partition_S(ww),t.partition_D(sw));
  };
  if(tid==0) for(int i=0;i<STAGES && i<total;++i) issue(i);

  // ---- LayerNorm: T lanes per row, 16-byte chunks interleaved across the lanes ----
  auto al=as_position_independent_swizzle_layout(TileLayout{});
  {
    constexpr int T=C<256?4:8,RP=256/T,CH=C/8,PER=CH/T;
    int sub=tid%T,rg=tid/T;
#ifdef MW_F3_ABL_NOLN
    for(int r=rg;r<0;r+=RP) {
#else
    for(int r=rg;r<128;r+=RP) {
#endif
      int64_t row=base+r;float vals[PER*8],mean=0.f,var=0.f;
      #pragma unroll
      for(int i=0;i<PER;++i) {
        uint4 vv=*reinterpret_cast<uint4 const*>(x+row*C+(sub+T*i)*8);
        auto const* v=reinterpret_cast<Element const*>(&vv);
        #pragma unroll
        for(int e=0;e<8;++e){vals[i*8+e]=float(v[e]);mean+=vals[i*8+e];}
      }
      #pragma unroll
      for(int d=1;d<T;d<<=1) mean+=__shfl_xor_sync(0xffffffff,mean,d);
      mean/=C;
      #pragma unroll
      for(int i=0;i<PER*8;++i){float d=vals[i]-mean;var+=d*d;}
      #pragma unroll
      for(int d=1;d<T;d<<=1) var+=__shfl_xor_sync(0xffffffff,var,d);
      float inv=rsqrtf(var/C+eps);
      if(sub==0){stats[row*2]=mean;stats[row*2+1]=inv;}
      int rw=r/64,rr=r%64;
      #pragma unroll
      for(int i=0;i<PER;++i) {
        int col=(sub+T*i)*8;uint4 vec;auto* o=reinterpret_cast<Element*>(&vec);
        #pragma unroll
        for(int e=0;e<8;++e) o[e]=Element((vals[i*8+e]-mean)*inv*__ldg(lnw+col+e)+__ldg(lnb+col+e));
        *reinterpret_cast<uint4*>(s.z.data()+(rw*NK+col/64)*4096+al(make_coord(rr,col%64)))=vec;
      }
    }
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  // ---- projection: both warpgroups consume every ring stage ----
  using TileMMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,Int<BN>,_64>>()));
  TileMMA mma;auto thr=mma.get_slice(wt);
  auto acc=partition_fragment_C(mma,Shape<_64,Int<BN>>{});
  auto coord=thr.partition_C(make_identity_tensor(Shape<_64,Int<BN>>{}));
  auto og=to.get_tma_tensor(make_shape(rows,C,4));
  auto tile=to.get_slice(_0{});
  Element* stg=s.st[wgi*SLOTS].data();
  bool leader=wt==0;
  int it=0;
  for(int blk=0;blk<NB;++blk) {
    clear(acc);
    for(int kc=0;kc<NK;++kc,++it) {
      int stage=it%STAGES;
      s.full[stage].wait((it/STAGES)&1);
      auto aa=make_tensor(make_smem_ptr(s.z.data()+(wgi*NK+kc)*4096),TileLayout{});
      auto ww=make_tensor(make_smem_ptr(s.w[stage].data()),WL{});
      auto ra=thr.partition_fragment_A(aa);auto rb=thr.partition_fragment_B(ww);
#ifndef MW_F3_ABL_NOMMA
      flash::gemm<false,1>(mma,ra,rb,acc);
#endif
      // Item it-1's WGMMA has retired in every warp of this warpgroup: release its stage.
      cutlass::arch::NamedBarrier::sync(128,1+wgi);
      if(leader && it>0) s.empty[(it-1)%STAGES].arrive(uint32_t(0),uint32_t(1));
      if(tid==0 && it>0 && it-1+STAGES<total) issue(it-1+STAGES);
    }
    warpgroup_wait<0>();warpgroup_fence_operand(acc);
    int col0=blk*BN;
    stg=s.st[wgi*SLOTS+blk%SLOTS].data();
    // The previous block's TMA store has had a whole block of MMA time to drain.
#ifndef MW_F3_ABL_NOSTOREWAIT
    if(leader) tma_store_wait<SLOTS-1>();
#endif
    cutlass::arch::NamedBarrier::sync(128,1+wgi);
#ifdef MW_F3_ABL_NOEPI
    if(false) {
#else
    if(col0<4*C) {
#endif
      #pragma unroll
      for(int j=0;j<size(acc);j+=2) {
        int rr=get<0>(coord(j)),cc=get<1>(coord(j));
        Element lo=Element(acc(j)),hi=Element(acc(j+1));
        *reinterpret_cast<uint32_t*>(stg+(cc/64)*4096+al(make_coord(rr,cc%64)))=uint32_t(lo.raw())|(uint32_t(hi.raw())<<16);
      }
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(128,1+wgi);
      if(leader) {
        #pragma unroll
        for(int chunk=0;chunk<BN/64;++chunk) {
          int col=col0+chunk*64;
          auto dst=local_tile(og(_,_,col/C),Shape<_64,_64>{},make_coord(base/64+wgi,(col%C)/64));
          auto src=make_tensor(make_smem_ptr(stg+chunk*4096),TileLayout{});
          copy(to,tile.partition_S(src),tile.partition_D(dst));
        }
        tma_store_arrive();
      }
    } else {
#ifdef MW_F3_ABL_NOEPI
      if(false)
#endif
      {
      // Head bias: stage [head][64 keys], then each head writes its 64 contiguous keys.
      #pragma unroll
      for(int j=0;j<size(acc);j+=2) {
        int rr=get<0>(coord(j)),cc=get<1>(coord(j));
        if(cc<H) stg[cc*64+rr]=Element(acc(j));
        if(cc+1<H) stg[(cc+1)*64+rr]=Element(acc(j+1));
      }
      cutlass::arch::NamedBarrier::sync(128,1+wgi);
      if(wt<H*8) {
        int h=wt/8,c=wt%8;
        int64_t row0=base+wgi*64;int b=row0/(int64_t(L)*L),q=(row0/L)%L,key0=row0%L;
        uint4 vec=*reinterpret_cast<uint4 const*>(stg+h*64+c*8);
        auto* v=reinterpret_cast<Element*>(&vec);
        if(mask) {
          #pragma unroll
          for(int e=0;e<8;++e) if(!mask[b*L+key0+c*8+e]) v[e]=Element(-3.3895313892515355e38f);
        }
        *reinterpret_cast<uint4*>(bias+((int64_t(b)*H+h)*L+q)*L+key0+c*8)=vec;
      }
      }
    }
  }
  if(leader) tma_store_wait<0>();
}

template<int C> void launch(torch::Tensor x,torch::Tensor lnw,torch::Tensor lnb,torch::Tensor weights,torch::Tensor mask,
                            torch::Tensor p,torch::Tensor bias,torch::Tensor stats,int L,int H,float eps) {
  int rows=x.size(0);
  auto gw=make_tensor(make_gmem_ptr((Element const*)weights.data_ptr()),make_shape(4*C+64,C),ST{C,_1{}});
  auto tw=make_tma_copy(SM90_TMA_LOAD{},gw,WL{},Shape<Int<BN>,_64>{},_1{});
  auto go=make_tensor(make_gmem_ptr((Element*)p.data_ptr()),make_shape(rows,C,4),OStride{C,_1{},int64_t(rows)*C});
  auto to=make_tma_copy(SM90_TMA_STORE{},go,TileLayout{},Shape<_64,_64>{},_1{});
  size_t smem=sizeof(Shared<C>);
  C10_CUDA_CHECK(cudaFuncSetAttribute(front_v3<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  front_v3<C><<<rows/128,256,smem,at::cuda::getCurrentCUDAStream()>>>(tw,to,(Element*)x.data_ptr(),lnw.data_ptr<float>(),lnb.data_ptr<float>(),
    mask.numel()?mask.data_ptr<bool>():nullptr,(Element*)bias.data_ptr(),stats.data_ptr<float>(),rows,L,H,eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> front_forward(torch::Tensor x,torch::Tensor lnw,torch::Tensor lnb,torch::Tensor weights,torch::Tensor mask,int L,int H,double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.is_contiguous() && x.dim()==2,"contiguous BF16 matrix required");
  c10::cuda::CUDAGuard guard(x.device());int C=x.size(1),rows=x.size(0),B=rows/(L*L);
  TORCH_CHECK(L>0 && L%128==0 && rows==B*L*L && H==C/32 && H<=16,"invalid pair/head dimensions");
  TORCH_CHECK(weights.device()==x.device() && weights.scalar_type()==torch::kBFloat16 && weights.is_contiguous() && weights.sizes()==torch::IntArrayRef({4*C+64,C}),"packed weights mismatch");
  for(auto t:{lnw,lnb}) TORCH_CHECK(t.device()==x.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous() && t.numel()==C,"FP32 LN affines required");
  TORCH_CHECK(mask.device()==x.device() && mask.is_contiguous() && mask.scalar_type()==torch::kBool && (mask.numel()==0 || mask.sizes()==torch::IntArrayRef({B,L})),"mask mismatch");
  auto p=torch::empty({4,rows,C},x.options()),bias=torch::empty({B,H,L,L},x.options());
  auto stats=torch::empty({rows,2},x.options().dtype(torch::kFloat32));
  switch(C){
    case 64: launch<64>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps));break;
    case 128: launch<128>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps));break;
    case 256: launch<256>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps));break;
    case 384: launch<384>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps));break;
    case 512: launch<512>(x,lnw,lnb,weights,mask,p,bias,stats,L,H,float(eps));break;
    default: TORCH_CHECK(false,"C must be 64/128/256/384/512");
  }
  return {p,bias,stats};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("front",&front_forward);}
