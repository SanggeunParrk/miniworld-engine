#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "ct_atom_narrow_adaln.cuh"
#include "ct_atom_narrow_tail.cuh"
namespace ct_atom_narrow80 {
template<class G,class P,class K>void launch(K kernel,const P&p,cudaStream_t stream){
 static std::set<int> devices;if(devices.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
 const int sm=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
 const int grid=std::min<int64_t>((p.M+16*G::NW-1)/(16*G::NW),sm*G::MINB);
 kernel<<<grid,G::NTHR,G::SMEM,stream>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<int NW,int NC>void adaln(AdalnAtomFwdParams p,cudaStream_t s){
 using G=AdalnAtomFwdCfg<NW,(NC==128||NW==8?2:4),NC>;
 auto kernel=adaln_atom_fwd_kernel<G>;static std::set<int> devices;
 if(devices.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
 const int sm=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
 const int grid=std::min<int64_t>((p.M+16*NW-1)/(16*NW),sm*G::MINB);
 kernel<<<dim3(grid,128/NC),G::NTHR,G::SMEM,s>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<int NW>void tail(CtTailFwdParams p,cudaStream_t s){using G=CtTailFwdCfg<NW>;launch<G>(ct_atom_fwd_kernel<G>,p,s);}
void run(torch::Tensor x,torch::Tensor c,std::vector<torch::Tensor> ps,torch::Tensor wab,torch::Tensor xa,torch::Tensor y,int64_t cfg,double ex,double ec,bool residual){
 TORCH_CHECK(ps.size()==9&&cfg>=0&&cfg<48,"nine parameters, cfg0..47");
 auto check=[&](torch::Tensor t,at::ScalarType dt){TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.is_contiguous()&&t.scalar_type()==dt,"same-device contiguous operand dtype");};
 check(x,at::kBFloat16);check(c,at::kBFloat16);check(xa,at::kBFloat16);check(y,at::kBFloat16);check(wab,at::kBFloat16);check(ps[0],at::kFloat);for(int i=1;i<9;++i)check(ps[i],at::kBFloat16);
 TORCH_CHECK(x.dim()==2&&x.size(1)==128&&c.dim()==2&&c.size(1)==128&&x.size(0)>0&&c.size(0)>0&&x.size(0)%c.size(0)==0&&xa.sizes()==x.sizes()&&y.sizes()==x.sizes(),"x/cond rows and outputs");
 TORCH_CHECK(ps[0].numel()==128&&ps[1].sizes()==at::IntArrayRef({128,128})&&ps[2].numel()==128&&ps[3].sizes()==at::IntArrayRef({128,128})&&ps[4].sizes()==at::IntArrayRef({256,128})&&ps[5].sizes()==at::IntArrayRef({256,128})&&ps[6].sizes()==at::IntArrayRef({128,256})&&ps[7].sizes()==at::IntArrayRef({128,128})&&ps[8].numel()==128&&wab.sizes()==at::IntArrayRef({512,128}),"atom128/hidden256 weights");
 at::cuda::CUDAGuard guard(x.device());auto s=at::cuda::getCurrentCUDAStream();const long M=x.size(0),P=c.size(0);
 AdalnAtomFwdParams ap{(const bf*)x.data_ptr(),(const bf*)c.data_ptr(),(const float*)ps[0].data_ptr(),(const bf*)ps[1].data_ptr(),(const bf*)ps[3].data_ptr(),(const bf*)ps[2].data_ptr(),(bf*)xa.data_ptr(),nullptr,nullptr,M,P,(float)ex,(float)ec};
 CtTailFwdParams tp{(const bf*)xa.data_ptr(),residual?(const bf*)x.data_ptr():nullptr,(const bf*)c.data_ptr(),(const bf*)wab.data_ptr(),(const bf*)ps[6].data_ptr(),(const bf*)ps[7].data_ptr(),(const bf*)ps[8].data_ptr(),(bf*)y.data_ptr(),nullptr,M,P};
 switch(cfg/4){case 0:adaln<1,32>(ap,s);break;case 1:adaln<2,32>(ap,s);break;case 2:adaln<4,32>(ap,s);break;case 3:adaln<8,32>(ap,s);break;case 4:adaln<1,64>(ap,s);break;case 5:adaln<2,64>(ap,s);break;case 6:adaln<4,64>(ap,s);break;case 7:adaln<8,64>(ap,s);break;case 8:adaln<1,128>(ap,s);break;case 9:adaln<2,128>(ap,s);break;case 10:adaln<4,128>(ap,s);break;case 11:adaln<8,128>(ap,s);break;}
 switch(cfg%4){case 0:tail<1>(tp,s);break;case 1:tail<2>(tp,s);break;case 2:tail<4>(tp,s);break;case 3:tail<8>(tp,s);break;}
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&ct_atom_narrow80::run);}
