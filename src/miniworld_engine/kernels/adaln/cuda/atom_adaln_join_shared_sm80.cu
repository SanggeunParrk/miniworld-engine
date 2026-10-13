#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <string>
#include "atom_adaln_join_shared_sm80.cuh"
#include "adaln_launch.cuh"
namespace atom_adaln_join_shared80 {
template<class T> const T* C(const at::Tensor&t){return reinterpret_cast<const T*>(t.data_ptr());}
template<class T> T* P(const at::Tensor&t){return reinterpret_cast<T*>(t.data_ptr());}
int64_t grid(int64_t rows){return std::max<int64_t>(1,std::min<int64_t>((rows+127)/128,at::cuda::getCurrentDeviceProperties()->multiProcessorCount));}
void check(const at::Tensor&t,const at::Tensor&x,int64_t rows,int64_t cols,at::ScalarType dt){
 TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.is_contiguous()&&t.scalar_type()==dt&&t.numel()==rows*cols&&reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,"aligned contiguous current physical tensor required");
}
template<bool INTERMEDIATE>
void launch(const AdalnAtomBwdParams&p){using G=AdalnAtomBwdCfg<8>;adl::launch_persistent<G>(adaln_atom_bwd_kernel<G,INTERMEDIATE>,p,(p.M+15)/16,at::cuda::getCurrentCUDAStream(),1);C10_CUDA_KERNEL_LAUNCH_CHECK();}
void run(at::Tensor dy,at::Tensor x,at::Tensor cond,at::Tensor xst,at::Tensor cst,at::Tensor lnw,at::Tensor ws,at::Tensor wt1,at::Tensor wt2,at::Tensor sb,at::Tensor res,at::Tensor dc_transition,at::Tensor dc_attention,at::Tensor dx,at::Tensor dc,at::Tensor dsc,at::Tensor aff,at::Tensor psb,at::Tensor plw,bool intermediate){
 TORCH_CHECK(x.is_cuda()&&x.dim()==2&&x.size(1)==128&&x.is_contiguous()&&x.scalar_type()==at::kBFloat16,"physical BF [M,128] required");const int64_t M=x.size(0);c10::cuda::CUDAGuard guard(x.device());
 for(const auto&t:{dy,x,cond,res,dc_transition,dc_attention,dx,dc,dsc,aff})check(t,x,M,128,at::kBFloat16);
 for(const auto&t:{ws,wt1,wt2})check(t,x,128,128,at::kBFloat16);
 check(sb,x,1,128,at::kBFloat16);check(lnw,x,1,128,at::kFloat);check(xst,x,M,2,at::kFloat);check(cst,x,M,2,at::kFloat);check(psb,x,grid(M),128,at::kFloat);check(plw,x,grid(M),128,at::kFloat);
 AdalnAtomBwdParams p{C<bf>(dy),C<bf>(x),C<bf>(cond),C<float2>(xst),C<float2>(cst),C<float>(lnw),C<bf>(ws),C<bf>(wt1),C<bf>(wt2),C<bf>(sb),C<bf>(res),C<bf>(dc_transition),C<bf>(dc_attention),P<bf>(dx),P<bf>(dc),P<bf>(dsc),P<bf>(aff),P<float>(psb),P<float>(plw),M};
 if(intermediate)launch<true>(p);else launch<false>(p);
}
template<bool INTERMEDIATE>std::map<std::string,int64_t> attributes(){
 using G=AdalnAtomBwdCfg<8>;cudaFuncAttributes attrs;int blocks=0;C10_CUDA_CHECK(cudaFuncSetAttribute(adaln_atom_bwd_kernel<G,INTERMEDIATE>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs,adaln_atom_bwd_kernel<G,INTERMEDIATE>));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,adaln_atom_bwd_kernel<G,INTERMEDIATE>,G::NTHR,G::SMEM));
 return {{"registers_per_thread",attrs.numRegs},{"local_bytes",attrs.localSizeBytes},{"dynamic_shared_bytes",G::SMEM},{"static_shared_bytes",attrs.sharedSizeBytes},{"max_active_blocks_per_SM",blocks},{"threads",G::NTHR}};
}
std::map<std::string,int64_t> stats(bool intermediate){if(intermediate)return attributes<true>();return attributes<false>();}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&atom_adaln_join_shared80::run);m.def("grid",&atom_adaln_join_shared80::grid);m.def("stats",&atom_adaln_join_shared80::stats);}
