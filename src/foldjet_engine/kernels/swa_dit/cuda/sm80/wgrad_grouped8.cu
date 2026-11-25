// E103: grouped QKV/G, Wu and Wo; BM128/ST5/eight warps/S32.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "../../../trimul_inproj/cuda/sm80/sm80_common.cuh"

namespace swa_wgrad_grouped8_80 {
using namespace a100;

template<int BM_, int BN_, int BK_, int ST_, int THREADS_=128,int MINB_=-1>
struct Config {
  static constexpr int BM=BM_, BN=BN_, BK=BK_, ST=ST_, NT=BN/16, MT=BM/(THREADS_/4);
  static constexpr int THREADS=THREADS_, AS=BM*BK*2, BS=BN*BK*2, STAGE=AS+BS, SMEM=ST*STAGE;
  // Eight-warp tiles need more than 128 registers per thread. Requiring two CTAs
  // would force the compiler to spill under the 64K-register SM budget.
  static constexpr int MINB=MINB_>=0 ? MINB_ : (THREADS==256 || SMEM>83000 ? 1 : 2);
};
template<int CPR>
DEVI uint32_t offset(int row, int chunk) {
  return row*(CPR*16u) + ((chunk ^ (CPR==4 ? ((row>>1)&3) : (row&7)))<<4);
}
struct Params { const __nv_bfloat16 *a,*b; float *out; int m,n,k,splits; };

template<class C,bool AM,bool BMN>
DEVI void tile(const Params &p,int ti,int tj,uint8_t *storage) {
  constexpr int CA=AM ? C::BM/8 : C::BK/8, CB=BMN ? C::BN/8 : C::BK/8;
  constexpr int NA=C::BM*C::BK/8/C::THREADS, NB=C::BN*C::BK/8/C::THREADS;
  const int tid=threadIdx.x, warp=tid>>5, lane=tid&31, wm=warp>>1, wn=warp&1;
  const int i0=ti*C::BM,j0=tj*C::BN,steps=p.k/C::BK;

  const uint32_t s0=smem_u32(storage);
  const __nv_bfloat16 *pa[NA],*pb[NB];
  uint32_t sa[NA],sb[NB];
#pragma unroll
  for(int q=0;q<NA;++q) {
    const int idx=tid+q*C::THREADS,r=idx/CA,c=idx%CA;
    pa[q]=p.a+(AM ? (size_t)r*p.m+i0+c*8 : (size_t)(i0+r)*p.k+c*8);
    sa[q]=offset<CA>(r,c);
  }
#pragma unroll
  for(int q=0;q<NB;++q) {
    const int idx=tid+q*C::THREADS,r=idx/CB,c=idx%CB;
    pb[q]=p.b+(BMN ? (size_t)r*p.n+j0+c*8 : (size_t)(j0+r)*p.k+c*8);
    sb[q]=C::AS+offset<CB>(r,c);
  }
  auto issue=[&](int kt,int slot) {
    const uint32_t base=s0+slot*C::STAGE;
    const size_t ka=(size_t)kt*C::BK*(AM ? p.m : 1),kb=(size_t)kt*C::BK*(BMN ? p.n : 1);
#pragma unroll
    for(int q=0;q<NA;++q)cp_async16(base+sa[q],pa[q]+ka);
#pragma unroll
    for(int q=0;q<NB;++q)cp_async16(base+sb[q],pb[q]+kb);
  };
  float acc[C::MT][C::NT][4];
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int nt=0;nt<C::NT;++nt)
#pragma unroll
      for(int e=0;e<4;++e)acc[mt][nt][e]=0.f;
  constexpr int KS=C::BK/16;
  uint32_t af[2][C::MT][4],bf[2][C::NT/2][4];
  uint32_t ao[C::MT][KS],bo[C::NT/2][KS];
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int ks=0;ks<KS;++ks)
      ao[mt][ks]=AM ? offset<CA>(ks*16+(lane&7)+8*(lane>>4),2*(wm*C::MT+mt)+((lane>>3)&1))
                    : offset<CA>(wm*C::MT*16+mt*16+(lane&15),ks*2+(lane>>4));
#pragma unroll
  for(int np=0;np<C::NT/2;++np)
#pragma unroll
    for(int ks=0;ks<KS;++ks)
      bo[np][ks]=C::AS+(BMN ? offset<CB>(ks*16+(lane&7)+8*((lane>>3)&1),wn*C::NT+np*2+(lane>>4))
                          : offset<CB>(wn*C::NT*8+np*16+(lane&7)+8*(lane>>4),ks*2+((lane>>3)&1)));
  auto fragments=[&](int buf,int slot,int ks) {
    const uint32_t base=s0+slot*C::STAGE;
#pragma unroll
    for(int mt=0;mt<C::MT;++mt) {
      if constexpr(AM)ldsm_x4_t(af[buf][mt],base+ao[mt][ks]);
      else ldsm_x4(af[buf][mt],base+ao[mt][ks]);
    }
#pragma unroll
    for(int np=0;np<C::NT/2;++np) {
      if constexpr(BMN)ldsm_x4_t(bf[buf][np],base+bo[np][ks]);
      else ldsm_x4(bf[buf][np],base+bo[np][ks]);
    }
  };
#pragma unroll
  for(int slot=0;slot<C::ST-1;++slot) {if(slot<steps)issue(slot,slot);cp_async_commit();}
  cp_async_wait<C::ST-2>();__syncthreads();fragments(0,0,0);
  int rd=0,wr=C::ST-1;
#pragma unroll 1
  for(int kt=0;kt<steps;++kt) {
#pragma unroll
    for(int ks=0;ks<KS;++ks) {
      if(ks+1<KS)fragments((ks&1)^1,rd,ks+1);
      else {
        cp_async_wait<C::ST-3>();__syncthreads();
        if(kt+C::ST-1<steps)issue(kt+C::ST-1,wr);
        cp_async_commit();rd=rd+1==C::ST ? 0 : rd+1;wr=wr+1==C::ST ? 0 : wr+1;
        if(kt+1<steps)fragments((ks&1)^1,rd,0);
      }
#pragma unroll
      for(int mt=0;mt<C::MT;++mt)
#pragma unroll
        for(int np=0;np<C::NT/2;++np) {
          mma16816(acc[mt][np*2],af[ks&1][mt],bf[ks&1][np][0],bf[ks&1][np][1]);
          mma16816(acc[mt][np*2+1],af[ks&1][mt],bf[ks&1][np][2],bf[ks&1][np][3]);
        }
    }
  }
  cp_async_wait<0>();__syncthreads();
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int nt=0;nt<C::NT;++nt)
#pragma unroll
      for(int hh=0;hh<2;++hh){
        const int row=i0+wm*C::MT*16+mt*16+8*hh+(lane>>2);
        const int col=j0+(wn*C::NT+nt)*8+2*(lane&3);
        p.out[(size_t)row*p.n+col]=acc[mt][nt][2*hh];
        p.out[(size_t)row*p.n+col+1]=acc[mt][nt][2*hh+1];
      }

}


struct GroupParams{const __nv_bfloat16 *dp,*x,*dab,*y,*datt,*gated;float *partial;int total_k;};
template<class C,class W,int SQ,int SW>
__global__ void __launch_bounds__(C::THREADS,C::MINB) grouped_kernel(GroupParams g){
 constexpr int part_elements=SQ*512*128;
 constexpr int large_cols=128/C::BN,large_tiles=(512/C::BM)*large_cols,large_blocks=SQ*large_tiles;
 constexpr int small_cols=128/W::BN,small_tiles=(128/W::BM)*small_cols;
 int id=blockIdx.x,split,tile_id;Params p;
 extern __shared__ __align__(128) uint8_t storage[];
 if(id<2*large_blocks){
  if(id<large_blocks){p.a=g.dp;p.b=g.x;p.out=g.partial;}else{id-=large_blocks;p.a=g.dab;p.b=g.y;p.out=g.partial+part_elements;}
  split=id/large_tiles;tile_id=id%large_tiles;p.m=512;p.n=128;p.k=g.total_k/SQ;p.splits=SQ;
  p.a+=(size_t)split*p.k*p.m;p.b+=(size_t)split*p.k*p.n;p.out+=(size_t)split*p.m*p.n;
  tile<C,true,true>(p,tile_id/large_cols,tile_id%large_cols,storage);
 }else{
  id-=2*large_blocks;split=id/small_tiles;tile_id=id%small_tiles;p.a=g.datt;p.b=g.gated;p.out=g.partial+2*part_elements;p.m=128;p.n=128;p.k=g.total_k/SW;p.splits=SW;
  p.a+=(size_t)split*p.k*p.m;p.b+=(size_t)split*p.k*p.n;p.out+=(size_t)split*p.m*p.n;
  tile<W,true,true>(p,tile_id/small_cols,tile_id%small_cols,storage);
 }
}
template<int S>DEVI float reduce_value(const float *partial,int i,int elements){
 float values[S];
 #pragma unroll
 for(int s=0;s<S;++s)values[s]=partial[(size_t)s*elements+i];
 float sum=0.f;
 #pragma unroll
 for(int s=0;s<S;++s)sum+=values[s];
 return sum;
}
template<int SQ,int SW>
__global__ void grouped_reduce_kernel(const float *partial,float *qkv,float *gate,float *wu,float *wo){
 constexpr int big=512*128,part=SQ*big,qkv_elements=384*128,small=128*128;int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<big){float s=reduce_value<SQ>(partial,i,big);if(i<qkv_elements)qkv[i]=s;else gate[i-qkv_elements]=s;}
 else if(i<2*big){i-=big;wu[i]=reduce_value<SQ>(partial+part,i,big);}
 else{i-=2*big;wo[i]=reduce_value<SW>(partial+2*part,i,small);}
}
template<class C,class W,int SQ,int SW>
void launch(GroupParams p,float *qkv,float *gate,float *wu,float *wo){
 constexpr int smem=C::SMEM>W::SMEM ? C::SMEM : W::SMEM;
 constexpr int blocks=2*SQ*(512/C::BM)*(128/C::BN)+SW*(128/W::BM)*(128/W::BN);
 static bool configured=false;if(!configured){C10_CUDA_CHECK(cudaFuncSetAttribute(grouped_kernel<C,W,SQ,SW>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));configured=true;}
 grouped_kernel<C,W,SQ,SW><<<blocks,C::THREADS,smem,at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
 grouped_reduce_kernel<SQ,SW><<<(2*512*128+128*128)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(p.partial,qkv,gate,wu,wo);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(torch::Tensor dp,torch::Tensor x,torch::Tensor dab,torch::Tensor y,torch::Tensor datt,torch::Tensor gated,
         torch::Tensor partial,torch::Tensor qkv,torch::Tensor gate,torch::Tensor wu,torch::Tensor wo){
 TORCH_CHECK(dp.is_cuda(),"CUDA operands required");auto dev=dp.device();const int K=dp.size(0);constexpr int SQ=32;constexpr int SW=32;
 for(auto t:{dp,x,dab,y,datt,gated})TORCH_CHECK(t.device()==dev&&t.scalar_type()==at::kBFloat16&&t.is_contiguous(),"actual contiguous BF16 operand layout");
 for(auto t:{partial,qkv,gate,wu,wo})TORCH_CHECK(t.device()==dev&&t.scalar_type()==at::kFloat&&t.is_contiguous(),"unrounded contiguous FP32 independent gradients");
 TORCH_CHECK((K==196608||K==393216)&&dp.sizes()==at::IntArrayRef({K,512})&&dab.sizes()==at::IntArrayRef({K,512}),"fixed QKV/G and FFN operand dimensions");
 for(auto t:{x,y,datt,gated})TORCH_CHECK(t.sizes()==at::IntArrayRef({K,128}),"fixed 128-channel full operand");
 TORCH_CHECK(partial.numel()==2*SQ*512*128+SW*128*128&&qkv.sizes()==at::IntArrayRef({384,128})&&gate.sizes()==at::IntArrayRef({128,128})&&wu.sizes()==at::IntArrayRef({512,128})&&wo.sizes()==at::IntArrayRef({128,128}),"fixed workspace/output sizes");
 at::cuda::CUDAGuard guard(dev);GroupParams p{(const __nv_bfloat16*)dp.data_ptr(),(const __nv_bfloat16*)x.data_ptr(),(const __nv_bfloat16*)dab.data_ptr(),(const __nv_bfloat16*)y.data_ptr(),(const __nv_bfloat16*)datt.data_ptr(),(const __nv_bfloat16*)gated.data_ptr(),(float*)partial.data_ptr(),K};
 float *pq=(float*)qkv.data_ptr(),*pg=(float*)gate.data_ptr(),*pu=(float*)wu.data_ptr(),*po=(float*)wo.data_ptr();

 using LargeEight5=Config<128,128,32,5,256,1>;
 launch<LargeEight5,LargeEight5,32,32>(p,pq,pg,pu,po);
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&swa_wgrad_grouped8_80::run);}
