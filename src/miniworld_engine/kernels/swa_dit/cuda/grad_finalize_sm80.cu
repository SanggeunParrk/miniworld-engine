#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
namespace swa_grad_finalize80 {
struct Params {const float *src[5];void *dst[5];int blocks[5],mask;};
__global__ void kernel(Params p){
 int block=blockIdx.x,field=0;
#pragma unroll
 for(int i=0;i<4;++i){if(block>=p.blocks[i]){block-=p.blocks[i];field=i+1;}else break;}
 const int pair=block*blockDim.x+threadIdx.x;
 const float2 v=reinterpret_cast<const float2*>(p.src[field])[pair];
 if(p.mask&(1<<field))reinterpret_cast<__nv_bfloat162*>(p.dst[field])[pair]=__floats2bfloat162_rn(v.x,v.y);
 else reinterpret_cast<float2*>(p.dst[field])[pair]=v;
}
void run(std::vector<torch::Tensor> src,std::vector<torch::Tensor> dst,int64_t mask){
 TORCH_CHECK(src.size()==5&&dst.size()==5,"five weight gradients");
 const auto device=src[0].device();at::cuda::CUDAGuard guard(device);Params p{};p.mask=mask;int blocks=0;
 for(int i=0;i<5;++i){
  TORCH_CHECK(src[i].is_cuda()&&src[i].device()==device&&dst[i].device()==device&&src[i].scalar_type()==at::kFloat,"FP32 gradients on one CUDA device");
  TORCH_CHECK(src[i].is_contiguous()&&dst[i].is_contiguous()&&src[i].sizes()==dst[i].sizes()&&src[i].numel()%256==0,"contiguous matching full tiles");
  TORCH_CHECK(dst[i].scalar_type()==((mask&(1<<i))?at::kBFloat16:at::kFloat),"parameter gradient dtype");
  p.src[i]=(const float*)src[i].data_ptr();p.dst[i]=dst[i].data_ptr();p.blocks[i]=src[i].numel()/256;blocks+=p.blocks[i];
 }
 kernel<<<blocks,128,0,at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<torch::Tensor> finalize(std::vector<torch::Tensor> src,int64_t mask){
 TORCH_CHECK(src.size()==5,"five weight gradients");at::cuda::CUDAGuard guard(src[0].device());
 std::vector<torch::Tensor> out;out.reserve(5);
 for(int i=0;i<5;++i)out.push_back(torch::empty_like(src[i],src[i].options().dtype((mask&(1<<i))?at::kBFloat16:at::kFloat)));
 run(src,out,mask);return out;
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&swa_grad_finalize80::run);m.def("finalize",&swa_grad_finalize80::finalize);}
