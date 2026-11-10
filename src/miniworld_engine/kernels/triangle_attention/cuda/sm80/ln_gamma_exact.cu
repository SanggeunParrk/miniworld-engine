// E083 exact-normalized-input LayerNorm scale gradient; separate candidate.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "ln_gamma_exact.cuh"
namespace triattn_gamma80 {
void run(torch::Tensor dd,torch::Tensor db,torch::Tensor x,torch::Tensor stats,torch::Tensor wt,torch::Tensor wbt,torch::Tensor gamma,torch::Tensor partial,torch::Tensor out,int64_t length,bool transposed){
 const c10::cuda::CUDAGuard guard(x.device()); const int T=x.numel()/128;
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&T>0&&T%(length*length)==0&&length%128==0,"pair shape");
 TORCH_CHECK(dd.is_contiguous()&&dd.scalar_type()==at::kBFloat16&&dd.numel()==(int64_t)T*512&&db.is_contiguous()&&db.numel()==(int64_t)T*4,"gradient shapes");
 TORCH_CHECK(stats.scalar_type()==at::kFloat&&stats.is_contiguous()&&stats.numel()==(int64_t)T*2,"stats");
 TORCH_CHECK(wt.is_contiguous()&&wt.scalar_type()==at::kBFloat16&&wt.numel()==4*128*128&&wbt.is_contiguous()&&wbt.scalar_type()==at::kBFloat16&&wbt.numel()==128*4,"packed weights");
 TORCH_CHECK(gamma.scalar_type()==at::kFloat&&gamma.is_contiguous()&&gamma.numel()==128,"gamma32");
 const int grid=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
 TORCH_CHECK(partial.scalar_type()==at::kFloat&&partial.is_contiguous()&&partial.numel()==grid*8*128,"partial scratch");
 TORCH_CHECK(out.numel()==128&&out.is_contiguous()&&(out.scalar_type()==at::kBFloat16||out.scalar_type()==at::kFloat),"gradient output");
 GammaParams p{};p.d=(const __nv_bfloat16*)dd.data_ptr();p.db=(const __nv_bfloat16*)db.data_ptr();p.x=(const __nv_bfloat16*)x.data_ptr();p.lnst=stats.data_ptr<float>();p.dout=(const __nv_bfloat16*)x.data_ptr();p.wt=(const __nv_bfloat16*)wt.data_ptr();p.wbt=(const __nv_bfloat16*)wbt.data_ptr();p.gamma=gamma.data_ptr<float>();p.ldd=512;p.ntile=T/16;p.L=length;p.transposed=transposed;p.partial=partial.data_ptr<float>();
 using G=GammaCfg<8>;auto kernel=gamma_partial_kernel<G>;static std::set<int> configured;
 if(configured.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
 auto st=at::cuda::getCurrentCUDAStream();kernel<<<grid,G::NTHR,G::SMEM,st>>>(p);gamma_reduce_kernel<<<128,256,0,st>>>(p.partial,out.data_ptr(),grid*8,out.scalar_type()==at::kBFloat16);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&triattn_gamma80::run);}
