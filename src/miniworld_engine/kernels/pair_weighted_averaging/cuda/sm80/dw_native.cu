// Own A100 CUDA PWA dw = dO V^T. FP32 output/partials; fixed-order split-K reduction.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "../../../transition/cuda/sm80/sm80_common.cuh"
namespace a100 {
template<int BM_,int BN_,int BK_,int WM_,int WN_,int ST_>
struct PWD {
 static constexpr int BM=BM_,BN=BN_,BK=BK_,WM=WM_,WN=WN_,ST=ST_;
 static constexpr int NTHR=32*WM*WN,MT=BM/(16*WM),NT=BN/(8*WN);
 static constexpr int AS=BM*BK*2,BS=BN*BK*2,SMEM=ST*(AS+BS);
 static constexpr int MINB=SMEM<=81920 ? 2 : 1;
};
template<int CPR> DEVI uint32_t pwd_swz(uint32_t r,uint32_t c) {
 if constexpr(CPR==4)return r*64+((c^((r>>1)&3))<<4);
 else return r*(CPR*16)+((c^(r&7))<<4);
}
template<class C>
__global__ void __launch_bounds__(C::NTHR,C::MINB) pwa_dw_native_kernel(
 const __nv_bfloat16* __restrict__ av,const __nv_bfloat16* __restrict__ bv,
 float* __restrict__ partial,int L,int K,int batches,int replicas) {
 extern __shared__ __align__(128) uint8_t shared[];
 const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,wm=warp/C::WN,wn=warp%C::WN;
 const int im=(blockIdx.x/(L/C::BN))*C::BM,jn=(blockIdx.x%(L/C::BN))*C::BN;
 const int batch=blockIdx.y,rep=blockIdx.z,kt=K/C::BK,begin=kt*rep/replicas,end=kt*(rep+1)/replicas,nk=end-begin;
 const auto* a=av+(size_t)batch*L*K;const auto* b=bv+(size_t)batch*L*K;
 const uint32_t base=smem_u32(shared);
 auto load=[&](int step,int slot) {
  const int k0=(begin+step)*C::BK;const uint32_t sa=base+slot*(C::AS+C::BS),sb=sa+C::AS;
#pragma unroll
  for(int e=0;e<C::AS/16/C::NTHR;++e){const int g=tid+C::NTHR*e,r=g/(C::BK/8),c=g%(C::BK/8);cp_async16(sa+pwd_swz<C::BK/8>(r,c),a+(size_t)(im+r)*K+k0+8*c);}
#pragma unroll
  for(int e=0;e<C::BS/16/C::NTHR;++e){const int g=tid+C::NTHR*e,r=g/(C::BK/8),c=g%(C::BK/8);cp_async16(sb+pwd_swz<C::BK/8>(r,c),b+(size_t)(jn+r)*K+k0+8*c);}
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
  for(int mt=0;mt<C::MT;++mt)ldsm_x4(af[buffer][mt],sa+pwd_swz<C::BK/8>(wm*(C::BM/C::WM)+16*mt+(lane&15),2*kk+(lane>>4)));
#pragma unroll
  for(int np=0;np<C::NT/2;++np)ldsm_x4(bf[buffer][np],sb+pwd_swz<C::BK/8>(wn*(C::BN/C::WN)+16*np+(lane&7)+8*(lane>>4),2*kk+((lane>>3)&1)));
 };
 auto multiply=[&](int buffer) {
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
   for(int np=0;np<C::NT/2;++np){mma16816(acc[mt][2*np],af[buffer][mt],bf[buffer][np][0],bf[buffer][np][1]);mma16816(acc[mt][2*np+1],af[buffer][mt],bf[buffer][np][2],bf[buffer][np][3]);}
 };
#pragma unroll
 for(int slot=0;slot<C::ST-1;++slot){if(slot<nk)load(slot,slot);cp_async_commit();}
 cp_async_wait<C::ST-2>();__syncthreads();fragments(0,0,0);
#pragma unroll 1
 for(int step=0;step<nk;++step){
#pragma unroll
  for(int kk=0;kk<C::BK/16;++kk){
   if(kk==C::BK/16-1){cp_async_wait<C::ST-3>();__syncthreads();if(step+C::ST-1<nk)load(step+C::ST-1,(step+C::ST-1)%C::ST);cp_async_commit();if(step+1<nk)fragments((kk+1)&1,(step+1)%C::ST,0);}
   else fragments((kk+1)&1,step%C::ST,kk+1);
   multiply(kk&1);
  }
 }
 cp_async_wait<0>();
 float* dest=partial+((size_t)rep*batches+batch)*L*L+(size_t)im*L+jn;
#pragma unroll
 for(int mt=0;mt<C::MT;++mt)
#pragma unroll
  for(int nt=0;nt<C::NT;++nt)
#pragma unroll
   for(int half=0;half<2;++half){const int r=wm*(C::BM/C::WM)+16*mt+(lane>>2)+8*half,c=wn*(C::BN/C::WN)+8*nt+2*(lane&3);*reinterpret_cast<float2*>(dest+(size_t)r*L+c)=make_float2(acc[mt][nt][2*half],acc[mt][nt][2*half+1]);}
}
__global__ void pwa_dw_reduce_kernel(const float* p,float* out,long elements,int replicas){const long i=(long)blockIdx.x*256+threadIdx.x;if(i>=elements)return;float sum=0.f;for(int r=0;r<replicas;++r)sum+=p[(long)r*elements+i];out[i]=sum;}
}
template<class C> void launch(torch::Tensor a,torch::Tensor b,torch::Tensor part,torch::Tensor out,int replicas){
 const int L=a.size(1),K=a.size(2),batch=a.size(0);TORCH_CHECK(L%C::BM==0 && L%C::BN==0 && K%C::BK==0 && K/C::BK>=replicas,"PWA dw tile dimensions");
 static std::set<int> configured;if(configured.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_dw_native_kernel<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));
 auto stream=at::cuda::getCurrentCUDAStream();float* p=replicas==1?out.data_ptr<float>():part.data_ptr<float>();
 a100::pwa_dw_native_kernel<C><<<dim3((L/C::BM)*(L/C::BN),batch,replicas),C::NTHR,C::SMEM,stream>>>((const __nv_bfloat16*)a.data_ptr(),(const __nv_bfloat16*)b.data_ptr(),p,L,K,batch,replicas);
 C10_CUDA_KERNEL_LAUNCH_CHECK();if(replicas>1){const long n=out.numel();a100::pwa_dw_reduce_kernel<<<(n+255)/256,256,0,stream>>>(p,out.data_ptr<float>(),n,replicas);C10_CUDA_KERNEL_LAUNCH_CHECK();}
}
void run(torch::Tensor a,torch::Tensor b,torch::Tensor part,torch::Tensor out,int64_t replicas,int64_t cfg){
 const at::cuda::CUDAGuard guard(a.device());for(auto t:{a,b,part,out})TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device()==a.device(),"PWA dw matching contiguous CUDA tensors");
 TORCH_CHECK(a.scalar_type()==at::kBFloat16 && b.scalar_type()==at::kBFloat16 && part.scalar_type()==at::kFloat && out.scalar_type()==at::kFloat,"PWA dw BF16 operands / FP32 result");
 TORCH_CHECK(a.dim()==3 && b.sizes()==a.sizes() && out.sizes()==torch::IntArrayRef({a.size(0),a.size(1),a.size(1)}) && replicas>=1 && part.numel()==(replicas==1?0:replicas*out.numel()),"PWA dw shapes");
 if(cfg==0)launch<a100::PWD<128,128,32,2,2,4>>(a,b,part,out,replicas);
 else if(cfg==1)launch<a100::PWD<64,128,32,2,2,5>>(a,b,part,out,replicas);
 else if(cfg==2)launch<a100::PWD<128,64,32,2,2,5>>(a,b,part,out,replicas);
 else if(cfg==3)launch<a100::PWD<64,64,32,2,2,5>>(a,b,part,out,replicas);
 else if(cfg==4)launch<a100::PWD<128,128,32,4,2,5>>(a,b,part,out,replicas);
 else if(cfg==5)launch<a100::PWD<128,128,32,4,2,4>>(a,b,part,out,replicas);
 else TORCH_CHECK(false,"PWA dw config");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
