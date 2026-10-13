// E053 binding-free snapshot of finalize_typed.cu; kernel mathematics unchanged.
// Five non-aliasing parameter gradients, with the original cast rounding.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

struct OpmFinalizeField {const float* src;void* dst;int n;bool bf16;};
struct OpmFinalizeParams {OpmFinalizeField field[5];};
__global__ void opm_finalize_typed_kernel(OpmFinalizeParams p,int n){
  const int idx=blockIdx.x*256+threadIdx.x;if(idx>=n)return;
#pragma unroll
  for(int j=0;j<5;++j){const auto f=p.field[j];if(idx<f.n){
    const float value=f.src[idx];
    if(f.bf16)((__nv_bfloat16*)f.dst)[idx]=__float2bfloat16_rn(value);
    else ((float*)f.dst)[idx]=value;
  }}
}
std::vector<torch::Tensor> result(std::vector<torch::Tensor> src,std::vector<torch::Tensor> parameters){
  TORCH_CHECK(src.size()==5&&parameters.size()==5,"OPM five gradients and parameters");
  const at::cuda::CUDAGuard guard(src[0].device());std::vector<torch::Tensor> out;OpmFinalizeParams args{};int n=0;
  for(int i=0;i<5;++i){auto s=src[i],p=parameters[i];
    TORCH_CHECK(s.is_cuda()&&s.is_contiguous()&&s.scalar_type()==at::kFloat&&s.device()==src[0].device(),"OPM contiguous FP32 gradients");
    TORCH_CHECK(p.is_cuda()&&p.device()==s.device()&&(p.scalar_type()==at::kFloat||p.scalar_type()==at::kBFloat16)&&p.sizes()==s.sizes(),"OPM matching parameter shapes/dtypes");
    auto o=torch::empty(s.sizes(),p.options());out.push_back(o);
    args.field[i]={s.data_ptr<float>(),o.data_ptr(),(int)s.numel(),p.scalar_type()==at::kBFloat16};n=std::max(n,(int)s.numel());
  }
  if(n)opm_finalize_typed_kernel<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(args,n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
