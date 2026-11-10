// E098 isolated early cp.async and single fragment DW with ns4 fold. No module dispatch.
// Own A100 CUDA PWA dw = dO V^T. FP32 output/partials; fixed-order split-K reduction.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "../../../transition/cuda/sm80/sm80_common.cuh"
namespace pwa_pair_partial80 {
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
__global__ void __launch_bounds__(C::NTHR,C::MINB) pwa_partial_contract_kernel(
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
}

} // namespace pwa_pair_partial80
#include "pwa_pair_partial_sm80.cuh"
torch::Tensor contract(torch::Tensor a,torch::Tensor b) {
 const at::cuda::CUDAGuard guard(a.device());
 TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device()==b.device() && a.is_contiguous() && b.is_contiguous() && a.scalar_type()==at::kBFloat16 && b.scalar_type()==at::kBFloat16 && a.sizes()==torch::IntArrayRef({32,128,8192}) && b.sizes()==a.sizes(),"PWA partial frozen operands");
 auto partial=torch::empty({2,32,128,128},a.options().dtype(at::kFloat));
 using C=pwa_pair_partial80::PWD<128,128,64,2,2,4>;
 static std::set<int> configured;if(configured.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_pair_partial80::pwa_partial_contract_kernel<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));
 pwa_pair_partial80::pwa_partial_contract_kernel<C><<<dim3(1,32,2),C::NTHR,C::SMEM,at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)a.data_ptr(),(const __nv_bfloat16*)b.data_ptr(),partial.data_ptr<float>(),128,8192,32,2);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return partial;
}
std::vector<torch::Tensor> pair(torch::Tensor z,torch::Tensor w,torch::Tensor dw,torch::Tensor mask,torch::Tensor lnw,torch::Tensor lnb,torch::Tensor wb,double eps) {
 const at::cuda::CUDAGuard guard(z.device());
 for(auto t:{z,w,dw,lnw,lnb,wb})TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device()==z.device(),"PWA pair matching contiguous CUDA tensors");
 TORCH_CHECK(z.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kBFloat16 && wb.scalar_type()==at::kBFloat16 && dw.scalar_type()==at::kFloat && lnw.scalar_type()==at::kFloat && lnb.scalar_type()==at::kFloat,"PWA pair dtypes");
 TORCH_CHECK(z.sizes()==torch::IntArrayRef({128,128,128}) && w.sizes()==torch::IntArrayRef({8,128,128}) && dw.sizes()==torch::IntArrayRef({2,32,128,128}) && lnw.numel()==128 && lnb.numel()==128 && wb.sizes()==torch::IntArrayRef({8,128}),"PWA pair frozen shapes");
 TORCH_CHECK(mask.numel()==0 || (mask.is_cuda() && mask.device()==z.device() && mask.scalar_type()==at::kByte && mask.is_contiguous() && mask.numel()==128),"PWA pair uint8 mask");
 auto dz=torch::empty_like(z);auto pm=torch::empty({128,1024},z.options().dtype(at::kFloat));auto ps=torch::empty({128,8},z.options().dtype(at::kFloat));
 pwa_pair_partial80::PairBwdParams p;
 p.z=(const __nv_bfloat16*)z.data_ptr();p.w=(const __nv_bfloat16*)w.data_ptr();p.dw=dw.data_ptr<float>();p.mask=mask.numel()?(const uint8_t*)mask.data_ptr():nullptr;
 p.lnw=lnw.data_ptr<float>();p.lnb=lnb.data_ptr<float>();p.wb=(const __nv_bfloat16*)wb.data_ptr();p.dz=(__nv_bfloat16*)dz.data_ptr();p.pM=pm.data_ptr<float>();p.pS=ps.data_ptr<float>();p.L=128;p.ns=4;p.eps=(float)eps;
 using C=pwa_pair_partial80::PairBwdCfg<128>;const int bytes=C::TILES+C::PREP+16*128;
 static std::set<int> configured;if(configured.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(pwa_pair_partial80::pwa_pair_bwd_kernel<128,2>,cudaFuncAttributeMaxDynamicSharedMemorySize,166912));
 pwa_pair_partial80::pwa_pair_bwd_kernel<128,2><<<128,C::NTHR,bytes,at::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {dz,pm,ps};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("contract",&contract);m.def("pair",&pair);}
