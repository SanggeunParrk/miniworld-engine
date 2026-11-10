// E054: rectangular adaptation of our own contract_tuned.cu; original preserved.
// A100 plane GEMM candidate. Separate from the established path until full-module measurements pass.
// NT/TN/NN/TT layouts share a cp.async ring; operand pointers and fragment offsets are hoisted out of K.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "../../../trimul_inproj/cuda/sm80/sm80_common.cuh"

namespace opm_rect80 {
using namespace a100;

template<int BM_, int BN_, int BK_, int ST_, int THREADS_=128>
struct Config {
  static constexpr int BM=BM_, BN=BN_, BK=BK_, ST=ST_, NT=BN/16, MT=BM/(THREADS_/4);
  static constexpr int THREADS=THREADS_, AS=BM*BK*2, BS=BN*BK*2, STAGE=AS+BS, SMEM=ST*STAGE;
  // Eight-warp tiles need more than 128 registers per thread. Requiring two CTAs
  // would force the compiler to spill under the 64K-register SM budget.
  static constexpr int MINB=THREADS==256 || SMEM>83000 ? 1 : 2;
};
template<int CPR>
DEVI uint32_t offset(int row, int chunk) {
  return row*(CPR*16u) + ((chunk ^ (CPR==4 ? ((row>>1)&3) : (row&7)))<<4);
}
struct Params { const __nv_bfloat16 *a,*b; __nv_bfloat16 *out; int m,n,k,mode; };

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
  constexpr int CO=C::BN/8;
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int nt=0;nt<C::NT;++nt)
#pragma unroll
      for(int hh=0;hh<2;++hh) {
        const int r=wm*C::MT*16+16*mt+8*hh+(lane>>2),c=wn*C::NT+nt;
        sts32(s0+offset<CO>(r,c)+4*(lane&3),pack_bf16(acc[mt][nt][2*hh],acc[mt][nt][2*hh+1]));
      }
  __syncthreads();
#pragma unroll
  for(int q=0;q<C::BM*CO/C::THREADS;++q) {
    const int idx=tid+q*C::THREADS,r=idx/CO,c=idx%CO;
    stg128(p.out+(size_t)(i0+r)*p.n+j0+c*8,lds128(s0+offset<CO>(r,c)));
  }
  __syncthreads();
}
template<class C,int MODE>
__global__ void __launch_bounds__(C::THREADS,C::MINB) kernel(Params p) {
  extern __shared__ __align__(128) uint8_t storage[];
  const int cols=p.n/C::BN,tpc=(p.m/C::BM)*cols;
  for(int idx=blockIdx.x;idx<tpc;idx+=gridDim.x) {
    const int ti=idx/cols,tj=idx-ti*cols;
    if constexpr(MODE==0)tile<C,false,false>(p,ti,tj,storage);
    else if constexpr(MODE==1)tile<C,true,true>(p,ti,tj,storage);
    else if constexpr(MODE==2)tile<C,false,true>(p,ti,tj,storage);
    else tile<C,true,false>(p,ti,tj,storage);
  }
}

template<class C,int MODE>
void launch_mode(const Params&p,int grid) {
  static bool configured=false;
  if(!configured) {C10_CUDA_CHECK(cudaFuncSetAttribute(kernel<C,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));configured=true;}
  TORCH_CHECK(p.m%C::BM==0 && p.n%C::BN==0 && p.k%C::BK==0,"plane GEMM tile divisibility");
  const int blocks=grid>0 ? grid : (p.m/C::BM)*(p.n/C::BN);
  kernel<C,MODE><<<blocks,C::THREADS,C::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);
}
template<class C>
void launch(const Params&p,int grid) {
  switch(p.mode) {

    case 0:launch_mode<C,0>(p,grid);break;
    case 1:launch_mode<C,1>(p,grid);break;
    case 2:launch_mode<C,2>(p,grid);break;
    case 3:launch_mode<C,3>(p,grid);break;
  }
}
void run(torch::Tensor a,torch::Tensor b,torch::Tensor out,int64_t mode,int64_t cfg,int64_t grid) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda() && a.device()==b.device() && a.device()==out.device(),"same CUDA device required");
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(a.scalar_type()==at::kBFloat16 && b.scalar_type()==a.scalar_type() && out.scalar_type()==a.scalar_type(),"BF16 required");
  TORCH_CHECK(a.dim()==2&&b.dim()==2&&out.dim()==2&&mode>=0&&mode<=3,"matrix operands and mode0..3");
  TORCH_CHECK(a.is_contiguous()&&b.is_contiguous()&&out.is_contiguous(),"contiguous operands");
  const int M=out.size(0),N=out.size(1);const bool AM=mode==1||mode==3,BMN=mode==1||mode==2;
  const int K=AM?a.size(0):a.size(1);
  TORCH_CHECK(a.sizes()==(AM?at::IntArrayRef({K,M}):at::IntArrayRef({M,K})),"A shape");
  TORCH_CHECK(b.sizes()==(BMN?at::IntArrayRef({K,N}):at::IntArrayRef({N,K})),"B shape");
  Params p{reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),M,N,K,(int)mode};
  switch(cfg) {
    case 0:launch<Config<128,128,32,4>>(p,grid);break;
    case 1:launch<Config<128,128,32,3>>(p,grid);break;
    case 2:launch<Config<64,128,32,4>>(p,grid);break;
    case 3:launch<Config<128,64,32,4>>(p,grid);break;
    case 4:launch<Config<64,64,32,4>>(p,grid);break;
    case 5:launch<Config<128,128,64,3>>(p,grid);break;
    case 6:launch<Config<128,128,64,4>>(p,grid);break;
    case 7:launch<Config<64,64,64,4>>(p,grid);break;
    case 8:launch<Config<64,128,64,4>>(p,grid);break;
    case 9:launch<Config<128,64,64,4>>(p,grid);break;
    case 10:launch<Config<128,128,32,4,256>>(p,grid);break;
    case 11:launch<Config<128,128,64,3,256>>(p,grid);break;
    case 12:launch<Config<128,128,64,4,256>>(p,grid);break;
    case 13:launch<Config<64,128,32,4,256>>(p,grid);break;
    case 14:launch<Config<128,64,32,4,256>>(p,grid);break;
    case 15:launch<Config<256,128,32,3,256>>(p,grid);break;
    case 16:launch<Config<128,128,32,5>>(p,grid);break;
    case 17:launch<Config<128,128,32,5,256>>(p,grid);break;
    default:TORCH_CHECK(false,"unknown rectangular GEMM tile");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace opm_rect80
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&opm_rect80::run);}
