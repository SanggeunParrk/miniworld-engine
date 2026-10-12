// Hopper tail stage v3: sigmoid(gate) * o -> output projection -> dropout scale -> residual.
// One CTA owns 128 rows (two warpgroups of 64). The gated tile u is built once in shared
// memory; both warpgroups walk the same 128-column blocks of Wo through one TMA ring.
// The residual rows are staged in the output tile and updated in place, then leave through
// TMA stores. Rounding order matches the v1 tail: bf16(acc) -> * drop -> bf16 -> + x -> bf16.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include "fa3_utils.h"
using namespace cute;
using Element=cutlass::bfloat16_t;
#ifndef MW_T3_STAGES
#define MW_T3_STAGES 4
#endif
constexpr int BN=128;
constexpr int STAGES=MW_T3_STAGES;
using TileLayout=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));
using WL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<Int<BN>,_64>{}));
using GS=Shape<int32_t,int32_t>;
using ST=Stride<int64_t,_1>;
using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GS{},ST{}),WL{},Shape<Int<BN>,_64>{},_1{}));
using OS=Shape<int32_t,int32_t>;
using OStride=Stride<int64_t,_1>;
using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),OS{},OStride{}),TileLayout{},Shape<_64,_64>{},_1{}));

__device__ __forceinline__ float sigmoidf(float x){return 1.f/(1.f+expf(-x));}

template<int C> struct Shared {
  array_aligned<Element,128*C,1024> z;
  array_aligned<Element,BN*64,1024> w[STAGES];
  array_aligned<Element,64*BN,1024> st[2];
  cutlass::arch::ClusterTransactionBarrier full[STAGES];
  cutlass::arch::ClusterBarrier empty[STAGES];
};

template<int C> __global__ __launch_bounds__(256,1)
void tail_v3(CUTE_GRID_CONSTANT TW const tw,CUTE_GRID_CONSTANT TO const to,
             Element const* x,Element const* gate,Element const* o,Element const* drop,
             int rows,int L,int gate_stride) {
  constexpr int NK=C/64,NB=(C+BN-1)/BN,total=NB*NK;
  extern __shared__ char memory[]; auto& s=*reinterpret_cast<Shared<C>*>(memory);
  int tid=threadIdx.x,wgi=tid/128,wt=tid%128,base=blockIdx.x*128;
  if(tid==0) {
    for(int i=0;i<STAGES;++i){s.full[i].init(1);s.empty[i].init(2);}
    prefetch_tma_descriptor(tw.get_tma_descriptor());prefetch_tma_descriptor(to.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  auto wg=tw.get_tma_tensor(make_shape(C,C));
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

  // ---- u = sigmoid(gate) * o for the whole 128 x C tile ----
  auto al=as_position_independent_swizzle_layout(TileLayout{});
  {
    constexpr int CH=C/8;
    for(int idx=tid;idx<128*CH;idx+=256) {
      int r=idx/CH,col=(idx%CH)*8;int64_t row=base+r;
      uint4 gv=*reinterpret_cast<uint4 const*>(gate+row*gate_stride+col),ov=*reinterpret_cast<uint4 const*>(o+row*C+col);
      auto const* gg=reinterpret_cast<Element const*>(&gv);auto const* oo=reinterpret_cast<Element const*>(&ov);
      uint4 vec;auto* vv=reinterpret_cast<Element*>(&vec);
      #pragma unroll
      for(int e=0;e<8;++e) vv[e]=Element(sigmoidf(float(gg[e]))*float(oo[e]));
      int rw=r/64,rr=r%64;
      *reinterpret_cast<uint4*>(s.z.data()+(rw*NK+col/64)*4096+al(make_coord(rr,col%64)))=vec;
    }
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  using TileMMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,Int<BN>,_64>>()));
  TileMMA mma;auto thr=mma.get_slice(wt);
  auto acc=partition_fragment_C(mma,Shape<_64,Int<BN>>{});
  auto coord=thr.partition_C(make_identity_tensor(Shape<_64,Int<BN>>{}));
  auto og=to.get_tma_tensor(make_shape(rows,C));
  auto tile=to.get_slice(_0{});
  Element* stg=s.st[wgi].data();
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
      flash::gemm<false,1>(mma,ra,rb,acc);
      cutlass::arch::NamedBarrier::sync(128,1+wgi);
      if(leader && it>0) s.empty[(it-1)%STAGES].arrive(uint32_t(0),uint32_t(1));
      if(tid==0 && it>0 && it-1+STAGES<total) issue(it-1+STAGES);
    }
    warpgroup_wait<0>();warpgroup_fence_operand(acc);
    int col0=blk*BN;
    if(leader) tma_store_wait<0>();
    cutlass::arch::NamedBarrier::sync(128,1+wgi);
    // Stage the residual rows (zero past column C), then fold the projection in place.
    for(int t=wt;t<64*BN/8;t+=128) {
      int r=t/(BN/8),k=(t%(BN/8))*8,col=col0+k;
      uint4 vec=col<C?*reinterpret_cast<uint4 const*>(x+int64_t(base+wgi*64+r)*C+col):make_uint4(0,0,0,0);
      *reinterpret_cast<uint4*>(stg+(k/64)*4096+al(make_coord(r,k%64)))=vec;
    }
    cutlass::arch::NamedBarrier::sync(128,1+wgi);
    #pragma unroll
    for(int j=0;j<size(acc);j+=2) {
      int rr=get<0>(coord(j)),cc=get<1>(coord(j));
      int col=col0+cc;
      if(col<C) {
        int64_t row=base+wgi*64+rr;
        Element lo=Element(acc(j)),hi=Element(acc(j+1));
        if(drop) {
          int64_t off=((row/(int64_t(L)*L))*L+row%L)*C+col;
          lo=Element(float(lo)*float(drop[off]));hi=Element(float(hi)*float(drop[off+1]));
        }
        int so=(cc/64)*4096+al(make_coord(rr,cc%64));
        uint32_t sv=*reinterpret_cast<uint32_t*>(stg+so);
        Element slo=Element::bitcast(uint16_t(sv&0xffff)),shi=Element::bitcast(uint16_t(sv>>16));
        lo=Element(float(slo)+float(lo));hi=Element(float(shi)+float(hi));
        *reinterpret_cast<uint32_t*>(stg+so)=uint32_t(lo.raw())|(uint32_t(hi.raw())<<16);
      }
    }
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::NamedBarrier::sync(128,1+wgi);
    if(leader) {
      #pragma unroll
      for(int chunk=0;chunk<BN/64;++chunk) {
        int col=col0+chunk*64;
        if(col<C) {
          auto dst=local_tile(og,Shape<_64,_64>{},make_coord(base/64+wgi,col/64));
          auto src=make_tensor(make_smem_ptr(stg+chunk*4096),TileLayout{});
          copy(to,tile.partition_S(src),tile.partition_D(dst));
        }
      }
      tma_store_arrive();
    }
  }
  if(leader) tma_store_wait<0>();
}

template<int C> void launch(torch::Tensor x,torch::Tensor gate,torch::Tensor o,torch::Tensor wo,torch::Tensor drop,torch::Tensor y,int L) {
  int rows=x.size(0);
  auto gw=make_tensor(make_gmem_ptr((Element const*)wo.data_ptr()),make_shape(C,C),ST{C,_1{}});
  auto tw=make_tma_copy(SM90_TMA_LOAD{},gw,WL{},Shape<Int<BN>,_64>{},_1{});
  auto go=make_tensor(make_gmem_ptr((Element*)y.data_ptr()),make_shape(rows,C),OStride{C,_1{}});
  auto to=make_tma_copy(SM90_TMA_STORE{},go,TileLayout{},Shape<_64,_64>{},_1{});
  size_t smem=sizeof(Shared<C>);
  C10_CUDA_CHECK(cudaFuncSetAttribute(tail_v3<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  tail_v3<C><<<rows/128,256,smem,at::cuda::getCurrentCUDAStream()>>>(tw,to,(Element*)x.data_ptr(),(Element*)gate.data_ptr(),(Element*)o.data_ptr(),
    drop.numel()?(Element*)drop.data_ptr():nullptr,rows,L,gate.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor tail_forward(torch::Tensor x,torch::Tensor gate,torch::Tensor o,torch::Tensor wo,torch::Tensor drop,int L) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.is_contiguous() && x.dim()==2,"contiguous BF16 matrix required");
  c10::cuda::CUDAGuard guard(x.device());int C=x.size(1),rows=x.size(0);
  TORCH_CHECK(L>0 && L%128==0 && rows%128==0,"L must be a multiple of 128");
  TORCH_CHECK(gate.device()==x.device() && gate.scalar_type()==x.scalar_type() && gate.sizes()==x.sizes() && gate.stride(1)==1 && gate.stride(0)>=C && gate.stride(0)%8==0,"gate mismatch");
  TORCH_CHECK(o.device()==x.device() && o.scalar_type()==x.scalar_type() && o.is_contiguous() && o.sizes()==x.sizes(),"output mismatch");
  TORCH_CHECK(wo.device()==x.device() && wo.scalar_type()==x.scalar_type() && wo.is_contiguous() && wo.sizes()==torch::IntArrayRef({C,C}),"Wo mismatch");
  TORCH_CHECK(drop.device()==x.device() && drop.scalar_type()==x.scalar_type() && drop.is_contiguous() && (drop.numel()==0 || drop.numel()==rows/L*C),"dropout scale mismatch");
  auto y=torch::empty_like(x);
  switch(C){
    case 64: launch<64>(x,gate,o,wo,drop,y,L);break;
    case 128: launch<128>(x,gate,o,wo,drop,y,L);break;
    case 256: launch<256>(x,gate,o,wo,drop,y,L);break;
    case 384: launch<384>(x,gate,o,wo,drop,y,L);break;
    case 512: launch<512>(x,gate,o,wo,drop,y,L);break;
    default: TORCH_CHECK(false,"C must be 64/128/256/384/512");
  }
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("tail",&tail_forward);}
