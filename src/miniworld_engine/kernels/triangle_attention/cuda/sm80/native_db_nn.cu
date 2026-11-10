// E109 own CUDA DB-only4xK NN gradient, logical N128 and tail8. Original augmented TN candidate preserved.
// E054: rectangular adaptation of our own contract_tuned.cu; original preserved.
// A100 plane GEMM candidate. Separate from the established path until full-module measurements pass.
// NT/TN/NN/TT layouts share a cp.async ring; operand pointers and fragment offsets are hoisted out of K.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "sm80_common.cuh"

namespace triattn_db_nn80 {
using namespace a100;

template<int BM_, int BN_, int BK_, int ST_, int THREADS_=128>
struct Config {
  static constexpr int BM=BM_, BN=BN_, BK=BK_, ST=ST_, NT=BN/16, MT=BM/(THREADS_/4);
  static constexpr int THREADS=THREADS_, AS=BM*BK*2, BS=BN*BK*2, TAIL=BK*32, STAGE=AS+BS+TAIL, SMEM=ST*STAGE;
  // Eight-warp tiles need more than 128 registers per thread. Requiring two CTAs
  // would force the compiler to spill under the 64K-register SM budget.
  static constexpr int MINB=THREADS==256 || SMEM>83000 ? 1 : 2;
};
template<int CPR>
DEVI uint32_t offset(int row, int chunk) {
  return row*(CPR*16u) + ((chunk ^ (CPR==2 ? (row&1) : CPR==4 ? ((row>>1)&3) : (row&7)))<<4);
}
struct Params { const __nv_bfloat16 *a,*b; float *out; int m,n,k,mode,ldb,ldo,lda; };

template<class C,bool AM,bool BMN>
DEVI void tile(const Params &p,int ti,int tj,uint8_t *storage) {
  constexpr int CA=AM ? C::BM/8 : C::BK/8, CB=BMN ? C::BN/8 : C::BK/8;
  constexpr int NA=C::BM*C::BK/8/C::THREADS, NB=C::BN*C::BK/8/C::THREADS;
  const int tid=threadIdx.x, warp=tid>>5, lane=tid&31, wm=warp>>1, wn=warp&1;
  const int i0=ti*C::BM,j0=tj*C::BN,steps=(p.k+C::BK-1)/C::BK;

  const uint32_t s0=smem_u32(storage);
  const __nv_bfloat16 *pa[NA],*pb[NB];
  uint32_t sa[NA],sb[NB];
#pragma unroll
  for(int q=0;q<NA;++q) {
    const int idx=tid+q*C::THREADS,r=idx/CA,c=idx%CA;
    pa[q]=p.a+(AM ? (size_t)r*p.m+i0+c*8 : (size_t)(i0+r)*p.lda+c*8);
    sa[q]=offset<CA>(r,c);
  }
#pragma unroll
  for(int q=0;q<NB;++q) {
    const int idx=tid+q*C::THREADS,r=idx/CB,c=idx%CB;
    pb[q]=p.b+(BMN ? (size_t)r*p.ldb+j0+c*8 : (size_t)(j0+r)*p.k+c*8);
    sb[q]=C::AS+offset<CB>(r,c);
  }
  auto issue=[&](int kt,int slot) {
    const uint32_t base=s0+slot*C::STAGE;
    const size_t ka=(size_t)kt*C::BK*(AM ? p.m : 1),kb=(size_t)kt*C::BK*(BMN ? p.ldb : 1);
#pragma unroll
    for(int q=0;q<NA;++q){const int idx=tid+q*C::THREADS,r=idx/CA,c=idx%CA;cp_async16(base+sa[q],pa[q]+ka,(AM ? i0+c*8<p.m && kt*C::BK+r<p.k : i0+r<p.m && kt*C::BK+c*8<p.k)?16:0);}
#pragma unroll
    for(int q=0;q<NB;++q){const int idx=tid+q*C::THREADS,r=idx/CB,c=idx%CB;cp_async16(base+sb[q],pb[q]+kb,(BMN ? j0+c*8<p.n && kt*C::BK+r<p.k : j0+r<p.n && kt*C::BK+c*8<p.k)?16:0);}

    for(int idx=tid;idx<C::BK*2;idx+=C::THREADS){
      const int r=idx/2,c=idx%2;const bool ok=c==0 && kt*C::BK+r<p.k;
      const __nv_bfloat16*src=p.b+(size_t)(kt*C::BK+r)*p.ldb+128;
      cp_async16(base+C::AS+C::BS+offset<2>(r,c),src,ok?16:0);
    }
  };
  float acc[C::MT][C::NT][4],tailacc[C::MT][2][4];
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)for(int nt=0;nt<2;++nt)for(int e=0;e<4;++e)tailacc[mt][nt][e]=0.f;
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int nt=0;nt<C::NT;++nt)
#pragma unroll
      for(int e=0;e<4;++e)acc[mt][nt][e]=0.f;
  constexpr int KS=C::BK/16;
  uint32_t af[2][C::MT][4],bf[2][C::NT/2][4],tbf[2][4];
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

    if(wn==0)ldsm_x4_t(tbf[buf],base+C::AS+C::BS+offset<2>(ks*16+(lane&7)+8*((lane>>3)&1),lane>>4));
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
      for(int mt=0;mt<C::MT;++mt) {
        if(wn==0){mma16816(tailacc[mt][0],af[ks&1][mt],tbf[ks&1][0],tbf[ks&1][1]);mma16816(tailacc[mt][1],af[ks&1][mt],tbf[ks&1][2],tbf[ks&1][3]);}
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
  cp_async_wait<0>();
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int nt=0;nt<C::NT;++nt)
#pragma unroll
      for(int hh=0;hh<2;++hh) {
        const int r=i0+wm*C::MT*16+16*mt+8*hh+(lane>>2);
        const int c=j0+wn*C::NT*8+nt*8+2*(lane&3);
        if(r<p.m && c<p.n) {
          p.out[(size_t)r*p.ldo+c]=acc[mt][nt][2*hh];
          if(c+1<p.n)p.out[(size_t)r*p.ldo+c+1]=acc[mt][nt][2*hh+1];
        }
      }

  if(wn==0){
#pragma unroll
    for(int mt=0;mt<C::MT;++mt)
#pragma unroll
      for(int hh=0;hh<2;++hh){
        const int r=i0+wm*C::MT*16+16*mt+8*hh+(lane>>2),c=128+2*(lane&3);
        if(r<p.m){p.out[(size_t)r*p.ldo+c]=tailacc[mt][0][2*hh];p.out[(size_t)r*p.ldo+c+1]=tailacc[mt][0][2*hh+1];}
      }
  }
}
template<class C>
__global__ void __launch_bounds__(C::THREADS,C::MINB) kernel(Params p,int chunk) {
 extern __shared__ __align__(128) uint8_t storage[];
 const int cols=(p.n+C::BN-1)/C::BN,rows=(p.m+C::BM-1)/C::BM,tpc=rows*cols;
 const int split=blockIdx.x/tpc,idx=blockIdx.x-split*tpc,begin=split*chunk;
 p.a+=begin;p.b+=(size_t)begin*p.ldb;p.out+=(size_t)split*p.m*p.ldo;
 p.k=min(chunk,p.k-begin);tile<C,false,true>(p,idx/cols,idx%cols,storage);
}
__global__ void reduce_kernel(const float*part,float*out,int splits,int width) {
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<width){float v=0;for(int s=0;s<splits;++s)v+=part[(size_t)s*width+i];out[i]=v;}
}
template<class C> void launch(const Params&p,int chunk,int splits) {
 static bool configured=false;
 if(!configured){C10_CUDA_CHECK(cudaFuncSetAttribute(kernel<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));configured=true;}
 kernel<C><<<splits*((p.m+C::BM-1)/C::BM)*((p.n+C::BN-1)/C::BN),C::THREADS,C::SMEM,at::cuda::getCurrentCUDAStream()>>>(p,chunk);
}
void run(torch::Tensor a,torch::Tensor b,torch::Tensor part,torch::Tensor out,int64_t cfg,int64_t chunk) {
 TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&out.is_cuda()&&part.is_cuda()&&a.device()==b.device()&&a.device()==part.device()&&a.device()==out.device(),"same CUDA device");
 const c10::cuda::CUDAGuard guard(a.device());
 TORCH_CHECK(a.scalar_type()==at::kBFloat16&&b.scalar_type()==at::kBFloat16&&part.scalar_type()==at::kFloat&&out.scalar_type()==at::kFloat,"BF16 operands, FP32 output");
 TORCH_CHECK(a.dim()==2&&b.dim()==2&&out.dim()==2&&part.dim()==3&&a.is_contiguous()&&b.is_contiguous()&&out.is_contiguous()&&part.is_contiguous(),"contiguous matrices");
 const int K=a.size(1),M=a.size(0),N=128,splits=(K+chunk-1)/chunk;
 TORCH_CHECK(K>0&&M>0&&N>0&&M==4&&N%8==0&&K%32==0&&chunk>0&&chunk%64==0&&b.size(0)==K&&b.size(1)==136&&out.sizes()==at::IntArrayRef({M,136})&&part.sizes()==at::IntArrayRef({splits,M,136}),"TN logical and partial shapes");
 Params p{(const __nv_bfloat16*)a.data_ptr(),(const __nv_bfloat16*)b.data_ptr(),part.data_ptr<float>(),M,N,K,0,136,136,K};
 switch(cfg){
 case 0:launch<Config<16,128,32,4,64>>(p,chunk,splits);break;
 case 1:launch<Config<16,128,64,3,64>>(p,chunk,splits);break;
 case 2:launch<Config<16,128,64,4,64>>(p,chunk,splits);break;
 case 3:launch<Config<32,128,32,4,128>>(p,chunk,splits);break;
 default:TORCH_CHECK(false,"DB-onlyNN configuration");}
 const int width=M*136;reduce_kernel<<<(width+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(part.data_ptr<float>(),out.data_ptr<float>(),splits,width);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&triattn_db_nn80::run);}
