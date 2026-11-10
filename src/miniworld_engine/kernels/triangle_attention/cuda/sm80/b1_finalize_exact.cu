// E083 exact-normalized-input LayerNorm scale gradient; separate candidate.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "b1_finalize_exact.cuh"
namespace triattn_b1_finish80 {
void run(torch::Tensor dd,torch::Tensor db,torch::Tensor x,torch::Tensor stats,torch::Tensor dout,torch::Tensor dpair,torch::Tensor wt,torch::Tensor wbt,torch::Tensor gamma,torch::Tensor partial,torch::Tensor out,int64_t length,bool transposed){
 const c10::cuda::CUDAGuard guard(x.device()); const int T=x.numel()/128;
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&T>0&&T%(length*length)==0&&length%128==0,"pair shape");
 TORCH_CHECK(dd.is_contiguous()&&dd.scalar_type()==at::kBFloat16&&dd.numel()==(int64_t)T*512&&db.is_contiguous()&&db.numel()==(int64_t)T*4,"gradient shapes");
 TORCH_CHECK(stats.scalar_type()==at::kFloat&&stats.is_contiguous()&&stats.numel()==(int64_t)T*2,"stats");
 TORCH_CHECK(wt.is_contiguous()&&wt.scalar_type()==at::kBFloat16&&wt.numel()==4*128*128&&wbt.is_contiguous()&&wbt.scalar_type()==at::kBFloat16&&wbt.numel()==128*4,"packed weights");
 TORCH_CHECK(gamma.scalar_type()==at::kFloat&&gamma.is_contiguous()&&gamma.numel()==128,"gamma32");
 const int grid=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
 TORCH_CHECK(partial.scalar_type()==at::kFloat&&partial.is_contiguous()&&partial.numel()==grid*8*128,"partial scratch");
 TORCH_CHECK(out.numel()==128&&out.is_contiguous()&&out.scalar_type()==at::kFloat,"FP32 gamma output");
 TORCH_CHECK(dout.sizes()==x.sizes()&&dpair.sizes()==x.sizes()&&dout.is_contiguous()&&dpair.is_contiguous()&&dout.scalar_type()==at::kBFloat16&&dpair.scalar_type()==at::kBFloat16,"input gradient output");
 GammaB1Params p{};p.d=(const __nv_bfloat16*)dd.data_ptr();p.db=(const __nv_bfloat16*)db.data_ptr();p.x=(const __nv_bfloat16*)x.data_ptr();p.lnst=stats.data_ptr<float>();p.dout=(const __nv_bfloat16*)dout.data_ptr();p.dpair=(__nv_bfloat16*)dpair.data_ptr();p.wt=(const __nv_bfloat16*)wt.data_ptr();p.wbt=(const __nv_bfloat16*)wbt.data_ptr();p.gamma=gamma.data_ptr<float>();p.ldd=512;p.ntile=T/16;p.L=length;p.transposed=transposed;p.partial=partial.data_ptr<float>();
 using G=GammaB1Cfg<8>;auto kernel=b1_gamma_kernel<G>;static std::set<int> configured;
 if(configured.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
 auto st=at::cuda::getCurrentCUDAStream();kernel<<<grid,G::NTHR,G::SMEM,st>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
namespace triattn_b1_finish80 {
void finish(torch::Tensor g,std::vector<torch::Tensor> w,std::vector<torch::Tensor> dw,torch::Tensor gamma,torch::Tensor beta,torch::Tensor partial,torch::Tensor dgamma,torch::Tensor dbeta) {
 const c10::cuda::CUDAGuard guard(g.device());
 TORCH_CHECK(g.is_cuda()&&g.scalar_type()==at::kFloat&&g.is_contiguous()&&g.numel()==516*136&&w.size()==5&&dw.size()==5,"gradient and weights");
 FinalizeParams p{};p.g=g.data_ptr<float>();p.partial=partial.data_ptr<float>();p.count=partial.numel()/128;
 TORCH_CHECK(partial.is_cuda()&&partial.is_contiguous()&&partial.scalar_type()==at::kFloat&&p.count==at::cuda::getCurrentDeviceProperties()->multiProcessorCount*8,"gamma partial shape");
 for(int i=0;i<5;++i){TORCH_CHECK(w[i].is_contiguous()&&dw[i].is_contiguous()&&w[i].scalar_type()==at::kBFloat16&&dw[i].scalar_type()==at::kBFloat16&&w[i].numel()==(i==4?4:128)*128&&dw[i].sizes()==w[i].sizes(),"BF16 weights");p.w[i]=(const __nv_bfloat16*)w[i].data_ptr();p.dw[i]=(__nv_bfloat16*)dw[i].data_ptr();}
 for(auto *t:{&gamma,&beta,&dgamma,&dbeta})TORCH_CHECK(t->is_cuda()&&t->scalar_type()==at::kFloat&&t->is_contiguous()&&t->numel()==128,"FP32 affine");
 p.gamma=gamma.data_ptr<float>();p.beta=beta.data_ptr<float>();p.dgamma=dgamma.data_ptr<float>();p.dbeta=dbeta.data_ptr<float>();
 finalize_kernel<<<516+128,256,0,at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&triattn_b1_finish80::run);m.def("finish",&triattn_b1_finish80::finish);}
