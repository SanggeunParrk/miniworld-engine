// Candidate A100 BF16 x BF16 -> FP32 full-width input-gradient GEMM.
// Own mma.sync/cp.async kernel. One CTA owns all output channels of 32 rows.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include "sm80_common.cuh"
namespace rect_dx {
using namespace a100;
template<int D_,int ST_> struct Cfg {
 static constexpr int BM=64,BN=D_,BK=32,ST=ST_,THREADS=256,MT=2,NT=BN/32;
 static constexpr int CA=BK/8,CB=BN/8,AS=BM*BK*2,BS=BN*BK*2,STAGE=AS+BS,SMEM=ST*STAGE;
 static constexpr int NA=(BM*BK/8+THREADS-1)/THREADS,NB=BN*BK/8/THREADS;
};
template<int CPR> DEVI uint32_t off(int row,int chunk) {
 return row*(CPR*16u)+((chunk^(CPR==4 ? ((row>>1)&3) : (row&7)))<<4);
}
template<class C> __global__ void __launch_bounds__(256,1) kernel(const __nv_bfloat16* __restrict__ a,const __nv_bfloat16* __restrict__ b,float* __restrict__ out,int m,int k) {
 extern __shared__ __align__(128) uint8_t storage[];
 const int tid=threadIdx.x,warp=threadIdx.x>>5,lane=tid&31,steps=k/C::BK,wm=warp/4,wn=warp%4;
 const uint32_t s0=smem_u32(storage);
 for(int ti=blockIdx.x;ti<m/C::BM;ti+=gridDim.x) {
  const int i0=ti*C::BM;
  const __nv_bfloat16 *pa[C::NA],*pb[C::NB];uint32_t sa[C::NA],sb[C::NB];
#pragma unroll
  for(int q=0;q<C::NA;++q) {int idx=tid+q*C::THREADS,r=idx/C::CA,c=idx%C::CA;pa[q]=a+(size_t)(i0+r)*k+c*8;sa[q]=off<C::CA>(r,c);}
#pragma unroll
  for(int q=0;q<C::NB;++q) {int idx=tid+q*C::THREADS,r=idx/C::CB,c=idx%C::CB;pb[q]=b+(size_t)r*C::BN+c*8;sb[q]=C::AS+off<C::CB>(r,c);}
  auto issue=[&](int kt,int slot) {const uint32_t base=s0+slot*C::STAGE;
#pragma unroll
   for(int q=0;q<C::NA;++q)if(tid+q*C::THREADS<C::BM*C::BK/8)cp_async16(base+sa[q],pa[q]+kt*C::BK);
#pragma unroll
   for(int q=0;q<C::NB;++q)cp_async16(base+sb[q],pb[q]+(size_t)kt*C::BK*C::BN);
  };
  float acc[C::MT][C::NT][4];
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
   for(int nt=0;nt<C::NT;++nt)
#pragma unroll
    for(int e=0;e<4;++e)acc[mt][nt][e]=0.f;
  uint32_t af[2][C::MT][4],bf[2][C::NT/2][4];
  uint32_t ao[2],bo[C::NT/2][2];
#pragma unroll
  for(int ks=0;ks<2;++ks)ao[ks]=off<C::CA>(wm*C::MT*16+(lane&15),ks*2+(lane>>4));
#pragma unroll
  for(int np=0;np<C::NT/2;++np)
#pragma unroll
   for(int ks=0;ks<2;++ks)bo[np][ks]=C::AS+off<C::CB>(ks*16+(lane&7)+8*((lane>>3)&1),wn*C::NT+np*2+(lane>>4));
  auto fragments=[&](int buf,int slot,int ks) {const uint32_t base=s0+slot*C::STAGE;
#pragma unroll
   for(int mt=0;mt<C::MT;++mt)ldsm_x4(af[buf][mt],base+ao[ks]+mt*16*C::BK*2);
#pragma unroll
   for(int np=0;np<C::NT/2;++np)ldsm_x4_t(bf[buf][np],base+bo[np][ks]);
  };
#pragma unroll
  for(int slot=0;slot<C::ST-1;++slot){if(slot<steps)issue(slot,slot);cp_async_commit();}
  cp_async_wait<C::ST-2>();__syncthreads();fragments(0,0,0);int rd=0,wr=C::ST-1;
#pragma unroll 1
  for(int kt=0;kt<steps;++kt) {
#pragma unroll
   for(int ks=0;ks<2;++ks) {
    if(ks==0)fragments(1,rd,1);
    else {cp_async_wait<C::ST-3>();__syncthreads();if(kt+C::ST-1<steps)issue(kt+C::ST-1,wr);cp_async_commit();rd=(rd+1)%C::ST;wr=(wr+1)%C::ST;if(kt+1<steps)fragments(0,rd,0);}
#pragma unroll
    for(int mt=0;mt<C::MT;++mt)
#pragma unroll
     for(int np=0;np<C::NT/2;++np){mma16816(acc[mt][np*2],af[ks&1][mt],bf[ks&1][np][0],bf[ks&1][np][1]);mma16816(acc[mt][np*2+1],af[ks&1][mt],bf[ks&1][np][2],bf[ks&1][np][3]);}
   }
  }
  cp_async_wait<0>();__syncthreads();
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
   for(int nt=0;nt<C::NT;++nt)
#pragma unroll
    for(int hh=0;hh<2;++hh){int row=i0+wm*C::MT*16+mt*16+8*hh+(lane>>2),col=wn*C::NT*8+nt*8+2*(lane&3);reinterpret_cast<float2*>(out+(size_t)row*C::BN+col)[0]=make_float2(acc[mt][nt][2*hh],acc[mt][nt][2*hh+1]);}
  __syncthreads();
 }
}
template<int D,int ST> void launch(torch::Tensor a,torch::Tensor b,torch::Tensor out,int grid) {
 using C=Cfg<D,ST>;C10_CUDA_CHECK(cudaFuncSetAttribute(kernel<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));
 kernel<C><<<grid>0?grid:a.size(0)/64,256,C::SMEM,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),out.data_ptr<float>(),a.size(0),a.size(1));
}
void run(torch::Tensor a,torch::Tensor b,torch::Tensor out,int64_t stages,int64_t grid) {
 TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&out.is_cuda()&&a.device()==b.device()&&a.device()==out.device(),"same CUDA device");const c10::cuda::CUDAGuard guard(a.device());
 TORCH_CHECK(a.scalar_type()==at::kBFloat16&&b.scalar_type()==at::kBFloat16&&out.scalar_type()==at::kFloat,"BF16 inputs / FP32 output");
 TORCH_CHECK(a.dim()==2&&b.dim()==2&&out.dim()==2&&a.size(1)==b.size(0)&&out.size(0)==a.size(0)&&out.size(1)==b.size(1),"GEMM shapes");
 TORCH_CHECK(a.is_contiguous()&&b.is_contiguous()&&out.is_contiguous()&&a.size(0)%64==0&&a.size(1)%32==0,"whole contiguous tiles");
 TORCH_CHECK(stages==3||stages==4,"stage count");
#define L(D) if(b.size(1)==D){if(stages==3)launch<D,3>(a,b,out,grid);else launch<D,4>(a,b,out,grid);C10_CUDA_KERNEL_LAUNCH_CHECK();return;}
 L(256) L(384) L(512)
#undef L
 TORCH_CHECK(false,"width not built");
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&rect_dx::run);}
