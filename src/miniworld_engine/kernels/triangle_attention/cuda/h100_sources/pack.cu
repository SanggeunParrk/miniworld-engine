// One CUDA launch stages either BF16 weights or FP32 master weights.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
struct Pack {void const* ptr[6];bool fp32[6];};
__global__ void pack_parameters(Pack p,__nv_bfloat16* wall,__nv_bfloat16* wo,int C,int H) {
  int64_t idx=int64_t(blockIdx.x)*256+threadIdx.x,n=int64_t(4*C+64)*C;
  if(idx<n+int64_t(C)*C) {
    int which;int64_t off;__nv_bfloat16* dst;
    if(idx<n) {
      dst=wall+idx;which=idx/(int64_t(C)*C);off=idx%(int64_t(C)*C);
      if(which==4 && off>=H*C) {*dst=__float2bfloat16(0.f);return;}
    } else {which=5;off=idx-n;dst=wo+off;}
    *dst=p.fp32[which]?__float2bfloat16(reinterpret_cast<float const*>(p.ptr[which])[off]):reinterpret_cast<__nv_bfloat16 const*>(p.ptr[which])[off];
  }
}
std::vector<torch::Tensor> pack(std::vector<torch::Tensor> w) {
  TORCH_CHECK(w.size()==6,"six projection weights required");
  TORCH_CHECK(w[0].is_cuda() && w[0].dim()==2,"CUDA weights required");
  c10::cuda::CUDAGuard guard(w[0].device());int C=w[0].size(1),H=C/32;
  TORCH_CHECK(C>=64 && C<=512 && C%64==0,"C64..512 required");
  Pack p;
  for(int j=0;j<6;++j) {
    TORCH_CHECK(w[j].device()==w[0].device() && w[j].is_contiguous() && w[j].sizes()==torch::IntArrayRef({j==4?H:C,C}),"weight shape/device/layout mismatch");
    TORCH_CHECK(w[j].scalar_type()==torch::kBFloat16 || w[j].scalar_type()==torch::kFloat32,"BF16 or FP32 weights required");
    p.ptr[j]=w[j].data_ptr();p.fp32[j]=w[j].scalar_type()==torch::kFloat32;
  }
  auto wall=torch::empty({4*C+64,C},w[0].options().dtype(torch::kBFloat16));
  auto wo=torch::empty({C,C},wall.options());
  int64_t total=wall.numel()+wo.numel();
  // Resolve the kernel handle explicitly: CUDA12.9 otherwise probes the host
  // symbol with cuKernelGetFunction and logs INVALID_HANDLE before falling back.
  cudaKernel_t kernel;C10_CUDA_CHECK(cudaGetKernel(&kernel,pack_parameters));
  auto wp=(__nv_bfloat16*)wall.data_ptr(),op=(__nv_bfloat16*)wo.data_ptr();
  void* args[]={&p,&wp,&op,&C,&H};
  C10_CUDA_CHECK(cudaLaunchKernel((void*)kernel,dim3((total+255)/256),dim3(256),args,0,at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {wall,wo};
}
struct Finish {void* ptr[8];bool fp32[8];};
__global__ void finish_parameters(float const* g,float const* wo,float const* norm,Finish out,int C,int H) {
  int64_t idx=int64_t(blockIdx.x)*256+threadIdx.x;
  int64_t n0=2*C,n1=n0+int64_t(4)*C*C,n2=n1+H*C,total=n2+int64_t(C)*C;
  if(idx>=total)return;
  int which;int64_t off;float val;
  if(idx<n0){which=idx/C;off=idx%C;val=norm[idx];}
  else if(idx<n1){int64_t pos=idx-n0;which=2+pos/(int64_t(C)*C);off=pos%(int64_t(C)*C);val=g[pos];}
  else if(idx<n2){which=6;off=idx-n1;val=g[int64_t(4)*C*C+off];}
  else {which=7;off=idx-n2;val=wo[off];}
  if(out.fp32[which])reinterpret_cast<float*>(out.ptr[which])[off]=val;
  else reinterpret_cast<__nv_bfloat16*>(out.ptr[which])[off]=__float2bfloat16(val);
}
std::vector<torch::Tensor> finish(torch::Tensor g,torch::Tensor wo,torch::Tensor norm,std::vector<bool> fp32) {
  TORCH_CHECK(g.is_cuda() && g.dim()==2 && g.scalar_type()==torch::kFloat32 && g.is_contiguous(),"FP32 gradient matrix required");
  c10::cuda::CUDAGuard guard(g.device());int C=g.size(1),H=C/32;
  TORCH_CHECK(C>=64 && C<=512 && C%64==0 && g.size(0)==4*C+64 && fp32.size()==8,"gradient metadata mismatch");
  TORCH_CHECK(wo.device()==g.device() && wo.scalar_type()==torch::kFloat32 && wo.is_contiguous() && wo.sizes()==torch::IntArrayRef({C,C}),"dWo mismatch");
  TORCH_CHECK(norm.device()==g.device() && norm.scalar_type()==torch::kFloat32 && norm.is_contiguous() && norm.sizes()==torch::IntArrayRef({2,C}),"LN gradient mismatch");
  std::vector<torch::Tensor> result;Finish out;
  for(int j=0;j<8;++j) {
    auto opt=g.options().dtype(fp32[j]?torch::kFloat32:torch::kBFloat16);
    result.push_back(j<2?torch::empty({C},opt):torch::empty({j==6?H:C,C},opt));
    out.ptr[j]=result.back().data_ptr();out.fp32[j]=fp32[j];
  }
  int64_t n=2*C+int64_t(5)*C*C+H*C;
  finish_parameters<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(g.data_ptr<float>(),wo.data_ptr<float>(),norm.data_ptr<float>(),out,C,H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("pack",&pack);m.def("finish",&finish);}
