#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "pwa_glue_tuned.cuh"
#include "pwa_glue_v2.cuh"
template<int MODE>
void launch(pwa80::GlueTunedParams p,int nbx){
 using G=pwa80::GlueTunedCfg<64,32,MODE>;p.ntile_s=(p.S+G::NTOK-1)/G::NTOK;p.ntile=p.L*p.ntile_s;
 static std::set<int> configured;if(configured.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(pwa80::pwa_glue_tuned_kernel<64,32,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
 pwa80::pwa_glue_tuned_kernel<64,32,MODE><<<dim3(nbx,G::NG),G::NTHR,G::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// mode 2: the prefetch-pipelined tile of the same kernel (pwa_glue_v2.cuh), bit-identical to mode 1
void launch_v2(const pwa80::GlueTunedParams& tp,int nbx){
 using G=pwa80::GlueV2Cfg<64,32>;pwa80::GlueV2Params p{};p.m=tp.m;p.dres=tp.dres;p.o=tp.o;p.lnw=tp.lnw;p.lnb=tp.lnb;p.wg=tp.wg;p.wo=tp.wo;p.keep=tp.keep;p.dO=tp.dO;p.dgp=tp.dgp;p.part=tp.part;
 p.S=tp.S;p.L=tp.L;p.kp=tp.kp;p.ns=tp.ns;p.eps=tp.eps;p.dscale=tp.dscale;p.ntile_s=(p.S+G::NTOK-1)/G::NTOK;p.ntile=p.L*p.ntile_s;
 static std::set<int> configured;if(configured.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(pwa80::pwa_glue_v2_kernel<64,32>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
 pwa80::pwa_glue_v2_kernel<64,32><<<dim3(nbx,G::NG),G::NTHR,G::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void glue(torch::Tensor m,torch::Tensor dres,torch::Tensor o,torch::Tensor lnw,torch::Tensor lnb,torch::Tensor wg,torch::Tensor wo,torch::Tensor keep,torch::Tensor dO,torch::Tensor dgp,torch::Tensor part,double eps,double dscale,int64_t mode){
 const at::cuda::CUDAGuard guard(m.device());for(auto t:{m,dres,o,wg,wo,dO,dgp})TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device()==m.device() && t.scalar_type()==at::kBFloat16,"tunedglue BF16 inputs");
 for(auto t:{lnw,lnb,part})TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device()==m.device() && t.scalar_type()==at::kFloat,"tunedglue FP32 inputs/partials");
 const int S=m.size(0),L=m.size(1),D=m.size(2),ns=o.size(0)/8;TORCH_CHECK(m.dim()==3 && D==64 && dres.sizes()==m.sizes() && wg.sizes()==torch::IntArrayRef({256,64}) && wo.sizes()==torch::IntArrayRef({64,256}),"tunedglue fixedwidth geometry");
 TORCH_CHECK(ns>=1 && S%ns==0 && o.sizes()==torch::IntArrayRef({8*ns,L,(S/ns)*32}) && dO.sizes()==o.sizes() && dgp.sizes()==o.sizes(),"tunedglue headmajor layout");
 TORCH_CHECK(part.dim()==3 && part.size(0)>0 && part.size(1)==4 && part.size(2)==8192 && lnw.numel()==64 && lnb.numel()==64,"tunedglue legacy partials");
 TORCH_CHECK(keep.numel()==0 || (keep.is_cuda() && keep.device()==m.device() && keep.is_contiguous() && keep.scalar_type()==at::kBFloat16 && keep.numel()==L*64),"tunedglue dropout keep");
 pwa80::GlueTunedParams p{};p.m=(const __nv_bfloat16*)m.data_ptr();p.dres=(const __nv_bfloat16*)dres.data_ptr();p.o=(const __nv_bfloat16*)o.data_ptr();p.lnw=lnw.data_ptr<float>();p.lnb=lnb.data_ptr<float>();p.wg=(const __nv_bfloat16*)wg.data_ptr();p.wo=(const __nv_bfloat16*)wo.data_ptr();p.keep=keep.numel()?(const __nv_bfloat16*)keep.data_ptr():nullptr;p.dO=(__nv_bfloat16*)dO.data_ptr();p.dgp=(__nv_bfloat16*)dgp.data_ptr();p.part=part.data_ptr<float>();p.S=S;p.L=L;p.kp=S/ns;p.ns=ns;p.eps=eps;p.dscale=dscale;
 if(mode==0)launch<0>(p,part.size(0));else if(mode==1)launch<1>(p,part.size(0));else if(mode==2)launch_v2(p,part.size(0));else TORCH_CHECK(false,"tunedglue mode");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("glue",&glue);}
