// E062: round master parameters once and write their packed projection copies
// in the same CUDA launch. Distinct allocations keep custom-op outputs unaliased.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
namespace param_cast_pack80 {
struct Field {const float* in;__nv_bfloat16 *out,*packed;int64_t n,begin,end;};
struct Params {Field f[16];int count;};
__global__ void parameter_cast_pack_bf16_kernel(Params p) {
  const int64_t block=blockIdx.x;
#pragma unroll
  for(int j=0;j<16;++j) if(j<p.count && block>=p.f[j].begin && block<p.f[j].end) {
    const Field f=p.f[j];const int64_t i=(block-f.begin)*1024+threadIdx.x*4;
    if(i+3<f.n) {
      const float4 v=__ldg(reinterpret_cast<const float4*>(f.in+i));uint32_t a,b;
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;":"=r"(a):"f"(v.y),"f"(v.x));
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;":"=r"(b):"f"(v.w),"f"(v.z));
      const uint2 result=make_uint2(a,b);*reinterpret_cast<uint2*>(f.out+i)=result;
      if(f.packed) *reinterpret_cast<uint2*>(f.packed+i)=result;
    } else for(int t=0;t<4;++t) if(i+t<f.n) {
      const auto result=__float2bfloat16_rn(f.in[i+t]);f.out[i+t]=result;
      if(f.packed) f.packed[i+t]=result;
    }
  }
}
std::vector<torch::Tensor> result(std::vector<torch::Tensor> src,std::vector<std::vector<int64_t>> pairs) {
  TORCH_CHECK(!src.empty() && src.size()<=16 && pairs.size()<=8,"cast pack arity");
  const at::cuda::CUDAGuard guard(src[0].device());std::vector<torch::Tensor> out;Params args{};int64_t blocks=0;
  for(auto t:src) {
    TORCH_CHECK(t.is_cuda() && t.device()==src[0].device() && t.is_contiguous() && t.scalar_type()==at::kFloat,"cast pack: contiguous same-device FP32 master tensors");
    auto o=torch::empty(t.sizes(),t.options().dtype(at::kBFloat16));out.push_back(o);const int64_t end=blocks+(t.numel()+1023)/1024;
    args.f[args.count++]={t.data_ptr<float>(),(__nv_bfloat16*)o.data_ptr(),nullptr,t.numel(),blocks,end};blocks=end;
  }
  for(const auto& pair:pairs) {
    TORCH_CHECK(pair.size()==2 && pair[0]>=0 && pair[1]>=0 && pair[0]<(int64_t)src.size() && pair[1]<(int64_t)src.size() && pair[0]!=pair[1],"cast pack index pair");
    const int i=pair[0],j=pair[1];const auto a=src[i],b=src[j];
    TORCH_CHECK(a.dim()==2 && b.dim()==2 && a.size(1)==b.size(1) && a.numel()%4==0 && !args.f[i].packed && !args.f[j].packed,"cast pack compatible aligned non-repeated matrices");
    auto packed=torch::empty({a.size(0)+b.size(0),a.size(1)},a.options().dtype(at::kBFloat16));out.push_back(packed);
    args.f[i].packed=(__nv_bfloat16*)packed.data_ptr();args.f[j].packed=args.f[i].packed+a.numel();
  }
  if(blocks) {parameter_cast_pack_bf16_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(args);C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return out;
}
} // namespace param_cast_pack80
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("result",&param_cast_pack80::result);}
