// Isolated lean-fragment A100 weight-gradient candidate. A[K,M]^T B[K,N], FP32 partials
// and a fixed-order reduction; no cuBLAS, atomics or reduced-precision partials.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <algorithm>
#include <type_traits>
#include "sm80_common.cuh"

namespace a100 {
template<int BM_, int BK_, int ST_>
struct WGCfg {
  static constexpr int BM=BM_, BN=128, BK=BK_, ST=ST_, NTHR=128;
  static constexpr int MT=BM/32, NT=8;
  static constexpr int AS=BK*BM*2, BS=BK*BN*2, SMEM=ST*(AS+BS);
  static constexpr int MINB=SMEM<=81920 ? 2 : 1;
};
template<int C>
DEVI uint32_t wg_swz(uint32_t row,uint32_t chunk) {
  return row*(C*16u)+((chunk^(row&7u))<<4);
}
template<class C>
__global__ void __launch_bounds__(128,C::MINB) weight_grad_splitk_kernel(
    const __nv_bfloat16* __restrict__ a,const __nv_bfloat16* __restrict__ b,
    float* __restrict__ part,int M,int N,int K,int replicas) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,wm=warp>>1,wn=warp&1;
  const int tiles=(M/C::BM)*(N/C::BN),rep=blockIdx.x/tiles,tile=blockIdx.x%tiles;
  const int im=(tile/(N/C::BN))*C::BM,jn=(tile%(N/C::BN))*C::BN;
  const int total_k=K/C::BK,begin=(total_k*rep)/replicas,end=(total_k*(rep+1))/replicas,nk=end-begin;
  const uint32_t base=smem_u32(smem);
  auto load_stage=[&](int step,int slot) {
    const int k0=(begin+step)*C::BK;
    const uint32_t sa=base+slot*(C::AS+C::BS),sb=sa+C::AS;
#pragma unroll
    for(int e=0;e<C::AS/16/128;++e) {
      const int g=tid+128*e,r=g/(C::BM/8),c=g%(C::BM/8);
      cp_async16(sa+wg_swz<C::BM/8>(r,c),a+(size_t)(k0+r)*M+im+c*8);
    }
#pragma unroll
    for(int e=0;e<C::BS/16/128;++e) {
      const int g=tid+128*e,r=g/16,c=g%16;
      cp_async16(sb+wg_swz<16>(r,c),b+(size_t)(k0+r)*N+jn+c*8);
    }
  };
  float acc[C::MT][8][4];
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int nt=0;nt<8;++nt)
#pragma unroll
      for(int e=0;e<4;++e)acc[mt][nt][e]=0.f;
  const int ar=(lane&7)+8*(lane>>4),ac=(lane>>3)&1;
  const int br=(lane&7)+8*((lane>>3)&1),bc=lane>>4;
  uint32_t af[C::MT][4],bf[4][4],aoff[C::MT],boff[4];
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)aoff[mt]=wg_swz<C::BM/8>(ar,(C::BM/16)*wm+2*mt+ac);
#pragma unroll
  for(int np=0;np<4;++np)boff[np]=wg_swz<16>(br,8*wn+2*np+bc);
  auto fragments=[&](int slot,int kk) {
    const uint32_t sa=base+slot*(C::AS+C::BS),sb=sa+C::AS;
#pragma unroll
    for(int mt=0;mt<C::MT;++mt)
      ldsm_x4_t(af[mt],sa+aoff[mt]+kk*16*C::BM*2);
#pragma unroll
    for(int np=0;np<4;++np)
      ldsm_x4_t(bf[np],sb+boff[np]+kk*16*C::BN*2);
  };
  auto multiply=[&]() {
#pragma unroll
    for(int mt=0;mt<C::MT;++mt)
#pragma unroll
      for(int np=0;np<4;++np) {
        mma16816(acc[mt][2*np],af[mt],bf[np][0],bf[np][1]);
        mma16816(acc[mt][2*np+1],af[mt],bf[np][2],bf[np][3]);
      }
  };
#pragma unroll
  for(int slot=0;slot<C::ST-1;++slot) {
    if(slot<nk)load_stage(slot,slot);
    cp_async_commit();
  }
  if(nk) {
#pragma unroll 1
    for(int step=0;step<nk;++step) {
      cp_async_wait<C::ST-2>();__syncthreads();
#pragma unroll
      for(int kk=0;kk<C::BK/16;++kk) {
        fragments(step%C::ST,kk);
        multiply();
        if(kk==C::BK/16-1) {
          cp_async_wait<C::ST-3>();__syncthreads();
          if(step+C::ST-1<nk)load_stage(step+C::ST-1,(step+C::ST-1)%C::ST);
          cp_async_commit();
        }
      }
    }
  }
  cp_async_wait<0>();__syncthreads();
  const int g8=lane>>2,q=lane&3;
  float* dest=part+(size_t)rep*M*N+(size_t)im*N+jn;
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int nt=0;nt<8;++nt)
#pragma unroll
      for(int half=0;half<2;++half)
        *reinterpret_cast<float2*>(dest+(size_t)(wm*(C::BM/2)+16*mt+g8+8*half)*N+64*wn+8*nt+2*q)
          =make_float2(acc[mt][nt][2*half],acc[mt][nt][2*half+1]);
}

template<class T>
__global__ void weight_grad_reduce_kernel(const float* part,T* out,int64_t elements,int replicas) {
  for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<elements;i+=(int64_t)blockDim.x*gridDim.x) {
    float value=0.f;
    for(int r=0;r<replicas;++r)value+=part[(int64_t)r*elements+i];
    if constexpr(std::is_same_v<T,float>)out[i]=value;
    else out[i]=__float2bfloat16_rn(value);
  }
}
}

template<class C>
void launch_wg(torch::Tensor a,torch::Tensor b,torch::Tensor part,torch::Tensor out,int replicas) {
  const int M=a.size(1),N=b.size(1),K=a.size(0);
  TORCH_CHECK(M%C::BM==0 && N%C::BN==0 && K%C::BK==0,"weight-gradient tile dimensions");
  static bool configured=false;
  if(!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::weight_grad_splitk_kernel<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));
    configured=true;
  }
  auto stream=at::cuda::getCurrentCUDAStream();
  a100::weight_grad_splitk_kernel<C><<<(M/C::BM)*(N/C::BN)*replicas,128,C::SMEM,stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
      part.data_ptr<float>(),M,N,K,replicas);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int blocks=std::min<int64_t>((out.numel()+255)/256,432);
  if(out.scalar_type()==at::kFloat)
    a100::weight_grad_reduce_kernel<float><<<blocks,256,0,stream>>>(part.data_ptr<float>(),out.data_ptr<float>(),out.numel(),replicas);
  else
    a100::weight_grad_reduce_kernel<__nv_bfloat16><<<blocks,256,0,stream>>>(part.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),out.numel(),replicas);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(torch::Tensor a,torch::Tensor b,torch::Tensor part,torch::Tensor out,int64_t replicas,int64_t cfg) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && part.is_cuda() && out.is_cuda() && a.device()==b.device() && a.device()==part.device() && a.device()==out.device(),"matching CUDA devices");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && part.is_contiguous() && out.is_contiguous() && a.dim()==2 && b.dim()==2,"contiguous operands");
  TORCH_CHECK(a.scalar_type()==at::kBFloat16 && b.scalar_type()==at::kBFloat16 && part.scalar_type()==at::kFloat && (out.scalar_type()==at::kFloat || out.scalar_type()==at::kBFloat16),"BF16 operands, FP32 partials, FP32/BF16 result");
  TORCH_CHECK(replicas>=1 && a.size(0)==b.size(0) && a.size(0)>=64*replicas && out.dim()==2 && out.size(0)==a.size(1) && out.size(1)==b.size(1) && part.numel()==replicas*out.numel(),"weight-gradient shapes/replicas");
  if(cfg==0)launch_wg<a100::WGCfg<128,32,4>>(a,b,part,out,replicas);
  else if(cfg==1)launch_wg<a100::WGCfg<128,64,3>>(a,b,part,out,replicas);
  else if(cfg==2)launch_wg<a100::WGCfg<64,64,3>>(a,b,part,out,replicas);
  else if(cfg==3)launch_wg<a100::WGCfg<64,32,4>>(a,b,part,out,replicas);
  else TORCH_CHECK(false,"weight-gradient cfg");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
