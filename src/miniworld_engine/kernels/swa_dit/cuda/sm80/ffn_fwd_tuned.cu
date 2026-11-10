#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <vector>
#include "ffn_fwd_tuned.cuh"
namespace swa_ffn_fwd_tuned80 {
std::vector<torch::Tensor> compute(torch::Tensor qi,torch::Tensor g,torch::Tensor o,
 torch::Tensor mod,torch::Tensor wo,torch::Tensor wu,torch::Tensor wd,
 int64_t S,int64_t B,double eps){
 TORCH_CHECK(qi.is_cuda()&&S>0&&B>0,"CUDA operands and positive physical dimensions");
 auto dev=qi.device();c10::cuda::CUDAGuard guard(dev);const int M=qi.size(0);
 for(auto t:{qi,g,o,wo,wu,wd})TORCH_CHECK(t.device()==dev&&t.scalar_type()==at::kBFloat16&&t.is_contiguous(),"original contiguous BF16 storage");
 TORCH_CHECK(mod.device()==dev&&mod.scalar_type()==at::kFloat&&mod.is_contiguous()&&mod.numel()==B*S*768,"physical FP32 modulation shape");
 for(auto t:{qi,g,o})TORCH_CHECK(t.sizes()==at::IntArrayRef({M,128}),"actual full row storage");
 TORCH_CHECK(M%S==0&&(M/S)%B==0&&wo.sizes()==at::IntArrayRef({128,128})&&wu.sizes()==at::IntArrayRef({512,128})&&wd.sizes()==at::IntArrayRef({128,256}),"physical conditioning and weight dimensions");
 auto out=torch::empty_like(qi),q1=torch::empty_like(qi),att=torch::empty_like(qi),y=torch::empty_like(qi),ff=torch::empty_like(qi);
 auto bp=[](torch::Tensor t){return (const __nv_bfloat16*)t.data_ptr();};auto bw=[](torch::Tensor t){return (__nv_bfloat16*)t.data_ptr();};
 FfnFwdParams p{bp(qi),bp(g),bp(o),mod.data_ptr<float>(),bp(wo),bp(wu),bp(wd),bw(out),bw(q1),bw(att),bw(y),bw(ff),M,(int)S,(int)B,(float)eps,nullptr};
 using G=FfnCfg<4,1,1>;
 static bool set=false;if(!set){C10_CUDA_CHECK(cudaFuncSetAttribute(ffn_fwd_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));set=true;}
 int active=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,ffn_fwd_kernel<G>,G::NTHR,G::SMEM));
 TORCH_CHECK(active>=1,"No valid resident CTA");const int ntile=((p.S+15)/16)*(p.M/p.S);
 const int ctas=std::min(at::cuda::getCurrentDeviceProperties()->multiProcessorCount*active,(ntile+G::NW-1)/G::NW);
 ffn_fwd_kernel<G><<<ctas,G::NTHR,G::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
 return {out,q1,att,y,ff};
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("compute",&swa_ffn_fwd_tuned80::compute);}
