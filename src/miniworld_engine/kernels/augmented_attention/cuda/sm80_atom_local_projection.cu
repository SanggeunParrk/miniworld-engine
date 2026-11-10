// E059: batched FP32 master-parameter -> BF16 casts, same round-to-nearest-even as Tensor.to.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
namespace local_projection_cast_sm80 {
struct Field{const float* in;__nv_bfloat16* out;int64_t n;int64_t begin,end;};
struct Params{Field f[16];int count;};
__global__ void parameter_cast_bf16_kernel(Params p){
  const int64_t block=blockIdx.x;
#pragma unroll
  for(int j=0;j<16;++j){if(j<p.count&&block>=p.f[j].begin&&block<p.f[j].end){
    const Field f=p.f[j];const int64_t i=(block-f.begin)*1024+threadIdx.x*4;
    if(i+3<f.n){const float4 v=__ldg(reinterpret_cast<const float4*>(f.in+i));
      uint32_t a,b;asm("cvt.rn.bf16x2.f32 %0, %1, %2;":"=r"(a):"f"(v.y),"f"(v.x));asm("cvt.rn.bf16x2.f32 %0, %1, %2;":"=r"(b):"f"(v.w),"f"(v.z));
      *reinterpret_cast<uint2*>(f.out+i)=make_uint2(a,b);
    }else for(int t=0;t<4;++t)if(i+t<f.n)f.out[i+t]=__float2bfloat16_rn(f.in[i+t]);
  }}
}
std::vector<torch::Tensor> result(std::vector<torch::Tensor> src){
  TORCH_CHECK(!src.empty()&&src.size()<=16,"one to16parameter tensors");at::cuda::CUDAGuard guard(src[0].device());std::vector<torch::Tensor> out;Params args{};int64_t blocks=0;
  for(auto t:src){TORCH_CHECK(t.is_cuda()&&t.device()==src[0].device()&&t.is_contiguous()&&(t.scalar_type()==at::kFloat||t.scalar_type()==at::kBFloat16),"contiguous FP32/BF16 same-device parameter");
    if(t.scalar_type()==at::kBFloat16){out.push_back(t);continue;}
    auto o=torch::empty(t.sizes(),t.options().dtype(at::kBFloat16));out.push_back(o);const int64_t end=blocks+(t.numel()+1023)/1024;
    args.f[args.count++]={t.data_ptr<float>(),(__nv_bfloat16*)o.data_ptr(),t.numel(),blocks,end};blocks=end;
  }
  if(blocks){parameter_cast_bf16_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(args);C10_CUDA_KERNEL_LAUNCH_CHECK();}return out;
}
} // namespace local_projection_cast_sm80
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("result",&local_projection_cast_sm80::result);}
