// E102 cooperative one-launch own DW plus ns4 fold; all 64 CTAs must reside concurrently. No module dispatch.
// Own A100 CUDA PWA dw = dO V^T. FP32 output/partials; fixed-order split-K reduction.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include <cooperative_groups.h>
#include "../../../transition/cuda/sm80/sm80_common.cuh"
namespace pwa_dw_coop80 {
using namespace a100;
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
__global__ void __launch_bounds__(C::NTHR,C::MINB) pwa_dw_coop_contract_kernel(
 const __nv_bfloat16* __restrict__ av,const __nv_bfloat16* __restrict__ bv,
 float* __restrict__ partial,float* __restrict__ output,int L,int K,int batches,int replicas) {
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
 uint32_t af[C::MT][4],bf[C::NT/2][4];
 auto fragments=[&](int slot,int kk) {
  const uint32_t sa=base+slot*(C::AS+C::BS),sb=sa+C::AS;
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)ldsm_x4(af[mt],sa+pwd_swz<C::BK/8>(wm*(C::BM/C::WM)+16*mt+(lane&15),2*kk+(lane>>4)));
#pragma unroll
  for(int np=0;np<C::NT/2;++np)ldsm_x4(bf[np],sb+pwd_swz<C::BK/8>(wn*(C::BN/C::WN)+16*np+(lane&7)+8*(lane>>4),2*kk+((lane>>3)&1)));
 };
 auto multiply=[&]() {
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
   for(int np=0;np<C::NT/2;++np){mma16816(acc[mt][2*np],af[mt],bf[np][0],bf[np][1]);mma16816(acc[mt][2*np+1],af[mt],bf[np][2],bf[np][3]);}
 };
#pragma unroll
 for(int slot=0;slot<C::ST-1;++slot){if(slot<nk)load(slot,slot);cp_async_commit();}
#pragma unroll 1
 for(int step=0;step<nk;++step){
  cp_async_wait<C::ST-2>();__syncthreads();
  if(step+C::ST-1<nk)load(step+C::ST-1,(step+C::ST-1)%C::ST);
  cp_async_commit();
#pragma unroll
  for(int kk=0;kk<C::BK/16;++kk){fragments(step%C::ST,kk);multiply();}
  __syncthreads();
 }
 cp_async_wait<0>();
 float* dest=partial+((size_t)rep*batches+batch)*L*L+(size_t)im*L+jn;
#pragma unroll
 for(int mt=0;mt<C::MT;++mt)
#pragma unroll
  for(int nt=0;nt<C::NT;++nt)
#pragma unroll
   for(int half=0;half<2;++half){const int r=wm*(C::BM/C::WM)+16*mt+(lane>>2)+8*half,c=wn*(C::BN/C::WN)+8*nt+2*(lane&3);*reinterpret_cast<float2*>(dest+(size_t)r*L+c)=make_float2(acc[mt][nt][2*half],acc[mt][nt][2*half+1]);}
 cooperative_groups::this_grid().sync();
 const long block=((long)blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x;
 const long stride=(long)gridDim.x*gridDim.y*gridDim.z*C::NTHR;
 const long plane=(long)L*L,groups=8*plane/4;
 for(long i=block*C::NTHR+tid;i<groups;i+=stride) {
  const long index=4*i,head=index/plane,offset=index%plane;
  float4 total=make_float4(0.f,0.f,0.f,0.f);
#pragma unroll
  for(int ns=0;ns<4;++ns) {
   const long batch_index=(head*4+ns)*plane+offset;
   const float4 v0=*reinterpret_cast<const float4*>(partial+batch_index);
   const float4 v1=*reinterpret_cast<const float4*>(partial+32*plane+batch_index);
   float4 sum=make_float4(0.f,0.f,0.f,0.f);
   sum.x+=v0.x;sum.y+=v0.y;sum.z+=v0.z;sum.w+=v0.w;
   sum.x+=v1.x;sum.y+=v1.y;sum.z+=v1.z;sum.w+=v1.w;
   total.x+=sum.x;total.y+=sum.y;total.z+=sum.z;total.w+=sum.w;
  }
  *reinterpret_cast<float4*>(output+index)=total;
 }
}
template<int R>
__global__ void pwa_dw_coop_reduce_kernel(const float* p,float* out,long groups,long plane) {
 const long i=(long)blockIdx.x*256+threadIdx.x;if(i>=groups)return;
 const long index=4*i,head=index/plane,offset=index%plane;
 float4 total=make_float4(0.f,0.f,0.f,0.f);
#pragma unroll
 for(int ns=0;ns<4;++ns) {
  float4 sum=make_float4(0.f,0.f,0.f,0.f);
#pragma unroll
  for(int r=0;r<R;r+=4) {
   const long batch=(head*4+ns)*plane+offset;
   const float4 v0=*reinterpret_cast<const float4*>(p+(long)r*32*plane+batch);
   const float4 v1=*reinterpret_cast<const float4*>(p+(long)(r+1)*32*plane+batch);
   if constexpr(R>=4) {
   const float4 v2=*reinterpret_cast<const float4*>(p+(long)(r+2)*32*plane+batch);
   const float4 v3=*reinterpret_cast<const float4*>(p+(long)(r+3)*32*plane+batch);
   sum.x+=v0.x;sum.y+=v0.y;sum.z+=v0.z;sum.w+=v0.w;
   sum.x+=v1.x;sum.y+=v1.y;sum.z+=v1.z;sum.w+=v1.w;
   sum.x+=v2.x;sum.y+=v2.y;sum.z+=v2.z;sum.w+=v2.w;
   sum.x+=v3.x;sum.y+=v3.y;sum.z+=v3.z;sum.w+=v3.w;
   } else {
   sum.x+=v0.x;sum.y+=v0.y;sum.z+=v0.z;sum.w+=v0.w;
   sum.x+=v1.x;sum.y+=v1.y;sum.z+=v1.z;sum.w+=v1.w;
   }
  }
  total.x+=sum.x;total.y+=sum.y;total.z+=sum.z;total.w+=sum.w;
 }
 *reinterpret_cast<float4*>(out+index)=total;
}

}
template<class C> void launch(torch::Tensor a,torch::Tensor b,torch::Tensor part,torch::Tensor out,int replicas){
 const int L=a.size(1),K=a.size(2),batch=a.size(0);TORCH_CHECK(L%C::BM==0 && L%C::BN==0 && K%C::BK==0 && K/C::BK>=replicas,"PWA dw tile dimensions");
 static std::set<int> configured;if(configured.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_dw_coop80::pwa_dw_coop_contract_kernel<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));
 auto stream=at::cuda::getCurrentCUDAStream();
 int active=0;
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,pwa_dw_coop80::pwa_dw_coop_contract_kernel<C>,C::NTHR,C::SMEM));
 const auto* properties=at::cuda::getCurrentDeviceProperties();
 const dim3 grid((L/C::BM)*(L/C::BN),batch,replicas);
 TORCH_CHECK(properties->cooperativeLaunch && (long)grid.x*grid.y*grid.z<=active*properties->multiProcessorCount,"PWA cooperative grid must fit resident capacity");
 const auto* ap=(const __nv_bfloat16*)a.data_ptr();const auto* bp=(const __nv_bfloat16*)b.data_ptr();
 float* pp=part.data_ptr<float>();float* op=out.data_ptr<float>();
 int lm=L,km=K,bm=batch,rm=replicas;
 void* args[]={&ap,&bp,&pp,&op,&lm,&km,&bm,&rm};
 C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)pwa_dw_coop80::pwa_dw_coop_contract_kernel<C>,grid,dim3(C::NTHR),args,C::SMEM,stream));
 C10_CUDA_KERNEL_LAUNCH_CHECK();

}
void run(torch::Tensor a,torch::Tensor b,torch::Tensor part,torch::Tensor out,int64_t replicas,int64_t cfg){
 const at::cuda::CUDAGuard guard(a.device());for(auto t:{a,b,part,out})TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device()==a.device(),"PWA dw matching contiguous CUDA tensors");
 TORCH_CHECK(a.scalar_type()==at::kBFloat16 && b.scalar_type()==at::kBFloat16 && part.scalar_type()==at::kFloat && out.scalar_type()==at::kFloat,"PWA dw BF16 operands / FP32 result");
 TORCH_CHECK(a.dim()==3 && b.sizes()==a.sizes() && out.sizes()==torch::IntArrayRef({8,a.size(1),a.size(1)}) && a.size(0)==32 && a.size(1)==128 && replicas==2 && part.numel()==replicas*a.size(0)*a.size(1)*a.size(1),"PWA dw shapes");
 if(cfg==0)launch<pwa_dw_coop80::PWD<128,128,32,2,2,4>>(a,b,part,out,replicas);
 else if(cfg==1)launch<pwa_dw_coop80::PWD<128,128,32,2,2,5>>(a,b,part,out,replicas);
 else if(cfg==2)launch<pwa_dw_coop80::PWD<128,128,64,2,2,3>>(a,b,part,out,replicas);
 else if(cfg==3)launch<pwa_dw_coop80::PWD<128,128,64,2,2,4>>(a,b,part,out,replicas);
 else TORCH_CHECK(false,"PWA dw config");
}
torch::Tensor result(torch::Tensor a,torch::Tensor b,int64_t replicas,int64_t cfg) {
 const at::cuda::CUDAGuard guard(a.device());
 auto out=torch::empty({8,a.size(1),a.size(1)},a.options().dtype(at::kFloat));
 auto partial=torch::empty({replicas,a.size(0),a.size(1),a.size(1)},a.options().dtype(at::kFloat));
 run(a,b,partial,out,replicas,cfg);return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("result",&result);}
