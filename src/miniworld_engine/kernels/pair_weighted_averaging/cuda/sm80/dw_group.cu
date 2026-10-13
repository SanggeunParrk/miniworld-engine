// Own CUDA PWA short dw: split K among warp groups of one CTA and reduce on chip.
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
template<int BM_,int BN_,int ST_,int GROUPS_>
struct PWDGroup: PWD<BM_,BN_,32,2,2,ST_> {
 static constexpr int GROUPS=GROUPS_,NTHR=128*GROUPS;
 static constexpr int GROUP_SMEM=PWD<BM_,BN_,32,2,2,ST_>::SMEM;
 static constexpr int SMEM=GROUPS*GROUP_SMEM,MINB=1;
};
template<int CPR> DEVI uint32_t pwd_swz(uint32_t r,uint32_t c) {
 if constexpr(CPR==4)return r*64+((c^((r>>1)&3))<<4);
 else return r*(CPR*16)+((c^(r&7))<<4);
}
template<class C>
__global__ void __launch_bounds__(C::NTHR,C::MINB) pwa_dw_group_kernel(
 const __nv_bfloat16* __restrict__ av,const __nv_bfloat16* __restrict__ bv,
 float* __restrict__ partial,int L,int K,int batches,int replicas) {
 extern __shared__ __align__(128) uint8_t shared[];
 const int group=threadIdx.x/128,tid=threadIdx.x%128,warp=tid>>5,lane=tid&31,wm=warp/C::WN,wn=warp%C::WN;
 const int im=(blockIdx.x/(L/C::BN))*C::BM,jn=(blockIdx.x%(L/C::BN))*C::BN;
 const int batch=blockIdx.y,kt=K/C::BK,begin=kt*group/C::GROUPS,end=kt*(group+1)/C::GROUPS,nk=end-begin;
 const auto* a=av+(size_t)batch*L*K;const auto* b=bv+(size_t)batch*L*K;
 const uint32_t base=smem_u32(shared)+group*C::GROUP_SMEM;
 auto load=[&](int step,int slot) {
  const int k0=(begin+step)*C::BK;const uint32_t sa=base+slot*(C::AS+C::BS),sb=sa+C::AS;
#pragma unroll
  for(int e=0;e<C::AS/16/128;++e){const int g=tid+128*e,r=g/(C::BK/8),c=g%(C::BK/8);cp_async16(sa+pwd_swz<C::BK/8>(r,c),a+(size_t)(im+r)*K+k0+8*c);}
#pragma unroll
  for(int e=0;e<C::BS/16/128;++e){const int g=tid+128*e,r=g/(C::BK/8),c=g%(C::BK/8);cp_async16(sb+pwd_swz<C::BK/8>(r,c),b+(size_t)(jn+r)*K+k0+8*c);}
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
 __syncthreads();
 float* dest=reinterpret_cast<float*>(shared)+group*C::BM*C::BN;
#pragma unroll
 for(int mt=0;mt<C::MT;++mt)
#pragma unroll
  for(int nt=0;nt<C::NT;++nt)
#pragma unroll
   for(int half=0;half<2;++half){const int r=wm*(C::BM/C::WM)+16*mt+(lane>>2)+8*half,c=wn*(C::BN/C::WN)+8*nt+2*(lane&3);*reinterpret_cast<float2*>(dest+r*C::BN+c)=make_float2(acc[mt][nt][2*half],acc[mt][nt][2*half+1]);}
 __syncthreads();
 for(int i=threadIdx.x;i<C::BM*C::BN/4;i+=C::NTHR){
  float4 sum=make_float4(0.f,0.f,0.f,0.f);
#pragma unroll
  for(int g=0;g<C::GROUPS;++g){const float4 v=*reinterpret_cast<float4*>(reinterpret_cast<float*>(shared)+g*C::BM*C::BN+4*i);sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;}
  const int row=(4*i)/C::BN,col=(4*i)%C::BN;
  *reinterpret_cast<float4*>(partial+(size_t)batch*L*L+(size_t)(im+row)*L+jn+col)=sum;
 }

}
}
template<class C> void launch(torch::Tensor a,torch::Tensor b,torch::Tensor out){
 const int L=a.size(1),K=a.size(2),batch=a.size(0);TORCH_CHECK(L%C::BM==0 && L%C::BN==0 && K%(C::BK*C::GROUPS)==0,"PWA dw group tile dimensions");
 static std::set<int> configured;if(configured.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(a100::pwa_dw_group_kernel<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));
 a100::pwa_dw_group_kernel<C><<<dim3((L/C::BM)*(L/C::BN),batch),C::NTHR,C::SMEM,at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)a.data_ptr(),(const __nv_bfloat16*)b.data_ptr(),out.data_ptr<float>(),L,K,batch,1);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(torch::Tensor a,torch::Tensor b,torch::Tensor out,int64_t cfg){
 const at::cuda::CUDAGuard guard(a.device());for(auto t:{a,b,out})TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device()==a.device(),"PWA dw group matching contiguous CUDA tensors");
 TORCH_CHECK(a.scalar_type()==at::kBFloat16 && b.scalar_type()==at::kBFloat16 && out.scalar_type()==at::kFloat,"PWA dw group BF16 operands / FP32 result");
 TORCH_CHECK(a.dim()==3 && b.sizes()==a.sizes() && out.sizes()==torch::IntArrayRef({a.size(0),a.size(1),a.size(1)}),"PWA dw group shapes");
 if(cfg==0)launch<a100::PWDGroup<64,64,5,2>>(a,b,out);
 else if(cfg==1)launch<a100::PWDGroup<64,64,5,4>>(a,b,out);
 else if(cfg==2)launch<a100::PWDGroup<64,64,4,4>>(a,b,out);
 else TORCH_CHECK(false,"PWA dw group config");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
