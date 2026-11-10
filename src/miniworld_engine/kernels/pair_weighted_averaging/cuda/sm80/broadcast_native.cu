// Own A100 CUDA broadcast GEMM for PWA: C[h,split] = W[h] V[h,split].
// Reads the shared weight plane directly; no repeated weight tensor or cuBLAS.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "../../../transition/cuda/sm80/sm80_common.cuh"
namespace a100 {
template<int BM_,int BN_,int BK_,int WM_,int WN_,int ST_>
struct PBCfg {
  static constexpr int BM=BM_,BN=BN_,BK=BK_,WM=WM_,WN=WN_,ST=ST_;
  static constexpr int NTHR=32*WM*WN,MT=BM/(16*WM),NT=BN/(8*WN);
  static constexpr int AS=BM*BK*2,BS=BK*BN*2,SMEM=ST*(AS+BS);
  static constexpr int MINB=NTHR==128 && SMEM<=81920 ? 2 : 1;
};
template<int CPR>
DEVI uint32_t pb_swz(uint32_t row,uint32_t chunk) {
  if constexpr(CPR==4)return row*64+((chunk^((row>>1)&3))<<4);
  else return row*(CPR*16)+((chunk^(row&7))<<4);
}
template<class C,bool TRANSPOSE>
__global__ void __launch_bounds__(C::NTHR,C::MINB) pwa_broadcast_native_kernel(
    const __nv_bfloat16* __restrict__ weights,const __nv_bfloat16* __restrict__ values,
    __nv_bfloat16* __restrict__ output,int L,int N,int splits) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,wm=warp/C::WN,wn=warp%C::WN;
  const int tiles=(L/C::BM)*(N/C::BN),batch=blockIdx.x/tiles,tile=blockIdx.x%tiles;
  const int im=(tile/(N/C::BN))*C::BM,jn=(tile%(N/C::BN))*C::BN,nk=L/C::BK;
  const __nv_bfloat16* A=weights+(size_t)(batch/splits)*L*L;
  const __nv_bfloat16* B=values+(size_t)batch*L*N;
  const uint32_t base=smem_u32(smem);
  auto load_stage=[&](int step,int slot) {
    const int k0=step*C::BK;const uint32_t sa=base+slot*(C::AS+C::BS),sb=sa+C::AS;
#pragma unroll
    for(int e=0;e<C::AS/16/C::NTHR;++e) {
      const int g=tid+C::NTHR*e;
      if constexpr(!TRANSPOSE) {
        const int r=g/(C::BK/8),c=g%(C::BK/8);
        cp_async16(sa+pb_swz<C::BK/8>(r,c),A+(size_t)(im+r)*L+k0+8*c);
      } else {
        const int r=g/(C::BM/8),c=g%(C::BM/8);
        cp_async16(sa+pb_swz<C::BM/8>(r,c),A+(size_t)(k0+r)*L+im+8*c);
      }
    }
#pragma unroll
    for(int e=0;e<C::BS/16/C::NTHR;++e) {
      const int g=tid+C::NTHR*e,r=g/(C::BN/8),c=g%(C::BN/8);
      cp_async16(sb+pb_swz<C::BN/8>(r,c),B+(size_t)(k0+r)*N+jn+8*c);
    }
  };
  float acc[C::MT][C::NT][4];
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int nt=0;nt<C::NT;++nt)
#pragma unroll
      for(int e=0;e<4;++e)acc[mt][nt][e]=0.f;
  uint32_t af[2][C::MT][4],bf[2][C::NT/2][4];
  auto fragments=[&](int buffer,int slot,int kk) {
    const uint32_t sa=base+slot*(C::AS+C::BS),sb=sa+C::AS;
#pragma unroll
    for(int mt=0;mt<C::MT;++mt) {
      if constexpr(!TRANSPOSE)
        ldsm_x4(af[buffer][mt],sa+pb_swz<C::BK/8>(wm*(C::BM/C::WM)+16*mt+(lane&7)+8*((lane>>3)&1),2*kk+(lane>>4)));
      else
        ldsm_x4_t(af[buffer][mt],sa+pb_swz<C::BM/8>(16*kk+(lane&7)+8*(lane>>4),wm*(C::BM/C::WM/8)+2*mt+((lane>>3)&1)));
    }
#pragma unroll
    for(int np=0;np<C::NT/2;++np)
      ldsm_x4_t(bf[buffer][np],sb+pb_swz<C::BN/8>(16*kk+(lane&7)+8*((lane>>3)&1),wn*C::NT+2*np+(lane>>4)));
  };
  auto multiply=[&](int buffer) {
#pragma unroll
    for(int mt=0;mt<C::MT;++mt)
#pragma unroll
      for(int np=0;np<C::NT/2;++np) {
        mma16816(acc[mt][2*np],af[buffer][mt],bf[buffer][np][0],bf[buffer][np][1]);
        mma16816(acc[mt][2*np+1],af[buffer][mt],bf[buffer][np][2],bf[buffer][np][3]);
      }
  };
#pragma unroll
  for(int slot=0;slot<C::ST-1;++slot){if(slot<nk)load_stage(slot,slot);cp_async_commit();}
  cp_async_wait<C::ST-2>();__syncthreads();fragments(0,0,0);
#pragma unroll 1
  for(int step=0;step<nk;++step) {
#pragma unroll
    for(int kk=0;kk<C::BK/16;++kk) {
      if(kk==C::BK/16-1) {
        cp_async_wait<C::ST-3>();__syncthreads();
        if(step+C::ST-1<nk)load_stage(step+C::ST-1,(step+C::ST-1)%C::ST);
        cp_async_commit();
        if(step+1<nk)fragments((kk+1)&1,(step+1)%C::ST,0);
      }else fragments((kk+1)&1,step%C::ST,kk+1);
      multiply(kk&1);
    }
  }
  cp_async_wait<0>();__syncthreads();
  const int g8=lane>>2,q=lane&3;
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int half=0;half<2;++half) {
      const int row=wm*(C::BM/C::WM)+16*mt+g8+8*half;
#pragma unroll
      for(int nt=0;nt<C::NT;++nt) {
        const int chunk=(wn*(C::BN/C::WN)+8*nt)/8;
        sts32(base+pb_swz<C::BN/8>(row,chunk)+q*4,pack_bf16(acc[mt][nt][2*half],acc[mt][nt][2*half+1]));
      }
    }
  __syncthreads();
  __nv_bfloat16* destination=output+(size_t)batch*L*N+(size_t)im*N+jn;
#pragma unroll
  for(int e=0;e<C::BM*C::BN/8/C::NTHR;++e) {
    const int g=tid+C::NTHR*e,row=g/(C::BN/8),chunk=g%(C::BN/8);
    stg128(destination+(size_t)row*N+chunk*8,lds128(base+pb_swz<C::BN/8>(row,chunk)));
  }
}
}
template<class C,bool TRANSPOSE>
void launch(torch::Tensor w,torch::Tensor v,torch::Tensor out) {
  const int L=w.size(1),N=v.size(2),splits=v.size(0)/w.size(0);
  TORCH_CHECK(L%C::BM==0 && L%C::BK==0 && N%C::BN==0,"native broadcast tile dimensions");
  const at::cuda::CUDAGuard guard(w.device());
  static std::set<int> configured;
  if(configured.insert(at::cuda::current_device()).second)
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_broadcast_native_kernel<C,TRANSPOSE>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));
  a100::pwa_broadcast_native_kernel<C,TRANSPOSE><<<v.size(0)*(L/C::BM)*(N/C::BN),C::NTHR,C::SMEM,at::cuda::getCurrentCUDAStream()>>>(
    reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),L,N,splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<bool TRANSPOSE>
void dispatch(torch::Tensor w,torch::Tensor v,torch::Tensor out,int cfg) {
  if(cfg==0)launch<a100::PBCfg<128,128,32,2,2,4>,TRANSPOSE>(w,v,out);
  else if(cfg==1)launch<a100::PBCfg<128,256,32,2,4,3>,TRANSPOSE>(w,v,out);
  else if(cfg==2)launch<a100::PBCfg<128,128,64,2,2,3>,TRANSPOSE>(w,v,out);
  else if(cfg==3)launch<a100::PBCfg<64,128,64,2,2,3>,TRANSPOSE>(w,v,out);
  else TORCH_CHECK(false,"native broadcast cfg");
}
void run(torch::Tensor w,torch::Tensor v,torch::Tensor out,bool transpose,int64_t cfg) {
  for(auto t:{w,v,out})TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type()==at::kBFloat16 && t.device()==w.device(),"native broadcast matching BF16 CUDA tensors");
  TORCH_CHECK(w.dim()==3 && v.dim()==3 && out.sizes()==v.sizes() && w.size(1)==w.size(2) && v.size(1)==w.size(1) && w.size(0)>0 && v.size(0)%w.size(0)==0,"native broadcast shapes");
  if(transpose)dispatch<true>(w,v,out,cfg);else dispatch<false>(w,v,out,cfg);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
