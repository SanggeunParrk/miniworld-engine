#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "affine263.cuh"
namespace triattn_affine_sm80 {
__global__ void pack_kernel(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,const __nv_bfloat16* g,const __nv_bfloat16* b,__nv_bfloat16* out,float* zeros){
 int s=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(s>=264)return;
 int c=s<256?a100::f1_channel(s):s<258?s:-1;uint32_t value=0;
 if(c>=0){const __nv_bfloat16* src=c<64?q:c<128?k:c<192?v:c<256?g:b;int row=c<256?c%64:c-256;value=reinterpret_cast<const uint32_t*>(src+(int64_t)row*64)[lane];}
 reinterpret_cast<uint32_t*>(out+(int64_t)s*64)[lane]=value;if(lane==0)zeros[s]=0.f;
}
__global__ void store_kernel(const float* G,__nv_bfloat16* q,__nv_bfloat16* k,__nv_bfloat16* v,__nv_bfloat16* g,__nv_bfloat16* b){
 int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=258*64)return;int row=at/64,col=at%64;__nv_bfloat16 bf=__float2bfloat16_rn(G[row*72+col]);
 __nv_bfloat16* dest=row<64?q:row<128?k:row<192?v:row<256?g:b;dest[(row<256?row%64:row-256)*64+col]=bf;
}
}
void pack(std::vector<torch::Tensor> w,torch::Tensor out,torch::Tensor zeros){
 TORCH_CHECK(w.size()==5&&out.is_cuda()&&out.scalar_type()==at::kBFloat16&&out.is_contiguous()&&out.numel()==264*64&&zeros.scalar_type()==at::kFloat&&zeros.numel()==264&&zeros.is_contiguous()&&zeros.device()==out.device());
 c10::cuda::CUDAGuard guard(out.device());for(int i=0;i<5;++i)TORCH_CHECK(w[i].device()==out.device()&&w[i].scalar_type()==at::kBFloat16&&w[i].is_contiguous()&&w[i].numel()==(i==4?2:64)*64);
 triattn_affine_sm80::pack_kernel<<<33,256,0,at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)w[0].data_ptr(),(const __nv_bfloat16*)w[1].data_ptr(),(const __nv_bfloat16*)w[2].data_ptr(),(const __nv_bfloat16*)w[3].data_ptr(),(const __nv_bfloat16*)w[4].data_ptr(),(__nv_bfloat16*)out.data_ptr(),zeros.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(torch::Tensor x,torch::Tensor w,torch::Tensor zeros,torch::Tensor gamma,torch::Tensor beta,torch::Tensor mask,torch::Tensor y,torch::Tensor bias,torch::Tensor stats,torch::Tensor xh,int L,bool transposed,double eps,bool direct){
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==at::kBFloat16&&x.dim()==2&&x.size(1)==64&&L>=128&&L%128==0);int T=x.size(0),Z=T/(L*L);TORCH_CHECK(T%(L*L)==0);
 TORCH_CHECK(w.is_contiguous()&&w.scalar_type()==at::kBFloat16&&w.numel()==264*64&&zeros.is_contiguous()&&zeros.scalar_type()==at::kFloat&&zeros.numel()==264);
 TORCH_CHECK(gamma.scalar_type()==at::kFloat&&beta.scalar_type()==at::kFloat&&gamma.is_contiguous()&&beta.is_contiguous()&&gamma.numel()==64&&beta.numel()==64);
 TORCH_CHECK(mask.numel()==0||(mask.is_contiguous()&&mask.scalar_type()==at::kByte&&mask.numel()==Z*L));
 TORCH_CHECK(y.is_contiguous()&&y.scalar_type()==at::kBFloat16&&y.numel()==(int64_t)T*256&&bias.is_contiguous()&&bias.scalar_type()==at::kBFloat16&&bias.numel()==(int64_t)T*2);
 TORCH_CHECK(stats.numel()==0||(stats.is_contiguous()&&stats.scalar_type()==at::kFloat&&stats.numel()==(int64_t)T*2));TORCH_CHECK(xh.numel()==0||(xh.is_contiguous()&&xh.scalar_type()==at::kBFloat16&&xh.numel()==(int64_t)T*72));
 c10::cuda::CUDAGuard guard(x.device());for(auto tensor:{w,zeros,gamma,beta,mask,y,bias,stats,xh})TORCH_CHECK(tensor.device()==x.device());
 triattn_affine_sm80::Params p{};p.x=(const __nv_bfloat16*)x.data_ptr();p.w=(const __nv_bfloat16*)w.data_ptr();p.bvec=zeros.data_ptr<float>();p.mask=mask.numel()?mask.data_ptr<uint8_t>():nullptr;p.out=(__nv_bfloat16*)y.data_ptr();p.bias=(__nv_bfloat16*)bias.data_ptr();p.lnst=stats.numel()?stats.data_ptr<float>():nullptr;p.xh=xh.numel()?(__nv_bfloat16*)xh.data_ptr():nullptr;p.ntile=T/32;p.L=L;p.transposed=transposed;p.eps=eps;p.gamma=gamma.data_ptr<float>();p.beta=beta.data_ptr<float>();p.save_affine=direct;
 using G=a100::FusedCfg<64,64,8,1,32>;int grid=std::min<int>(at::cuda::getDeviceProperties(x.device().index())->multiProcessorCount,(p.ntile+G::NW-1)/G::NW);
 triattn_affine_sm80::native_f1_kernel<G><<<grid,G::NTHR,G::F1_SMEM+2*G::DIN*4,at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void store(torch::Tensor G,std::vector<torch::Tensor> dw){
 TORCH_CHECK(G.is_cuda()&&G.scalar_type()==at::kFloat&&G.is_contiguous()&&G.numel()==258*72&&dw.size()==5);c10::cuda::CUDAGuard guard(G.device());
 for(int i=0;i<5;++i)TORCH_CHECK(dw[i].device()==G.device()&&dw[i].scalar_type()==at::kBFloat16&&dw[i].is_contiguous()&&dw[i].numel()==(i==4?2:64)*64);
 triattn_affine_sm80::store_kernel<<<65,256,0,at::cuda::getCurrentCUDAStream()>>>(G.data_ptr<float>(),(__nv_bfloat16*)dw[0].data_ptr(),(__nv_bfloat16*)dw[1].data_ptr(),(__nv_bfloat16*)dw[2].data_ptr(),(__nv_bfloat16*)dw[3].data_ptr(),(__nv_bfloat16*)dw[4].data_ptr());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("pack",&pack);m.def("run",&run);m.def("store",&store);}
