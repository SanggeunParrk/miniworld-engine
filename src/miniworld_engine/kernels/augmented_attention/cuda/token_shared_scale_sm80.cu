// E874 exact shared gate table; original sigmoid/FMA/BF16 rounding.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "sm80/glue_sm80.cuh"
namespace token_shared_scale_sm80 {
template<bool RES>
__global__ void __launch_bounds__(256) shared_res_gate(
 const __nv_bfloat16* __restrict__ y,const __nv_bfloat16* __restrict__ gate,
 const __nv_bfloat16* __restrict__ res,__nv_bfloat16* __restrict__ out,
 long long M,int d,int length,int padded) {
 const int chunks=d/8;
 for(long long c=(long long)blockIdx.x*256+threadIdx.x;c<M*chunks;c+=(long long)gridDim.x*256){
  const long long row=c/chunks;const int col=(int)(c-row*chunks)*8;
  const int token=(int)(row%padded);float result[8];
  if(token>=length){
   #pragma unroll
   for(int e=0;e<8;++e)result[e]=0.f;
  }else{
   float xy[8],xg[8],xr[8];
   aa80::V8<__nv_bfloat16>::load(y+row*d+col,xy);
   aa80::V8<__nv_bfloat16>::load(gate+(long long)token*d+col,xg);
   if constexpr(RES)aa80::V8<__nv_bfloat16>::load(res+row*d+col,xr);
   #pragma unroll
   for(int e=0;e<8;++e)result[e]=RES?fmaf(aa80::sigmoid(xg[e]),xy[e],xr[e]):aa80::sigmoid(xg[e])*xy[e];
  }
  aa80::V8<__nv_bfloat16>::store(out+row*d+col,result);
 }
}
void launch(torch::Tensor y,torch::Tensor gate,torch::Tensor res,torch::Tensor out,int64_t length,int64_t padded){
 TORCH_CHECK(y.is_cuda()&&y.scalar_type()==at::kBFloat16&&y.dim()==2&&y.is_contiguous(),"E874 y contiguous BF16 rows");
 const int64_t M=y.size(0),d=y.size(1);
 TORCH_CHECK(d==768&&length>0&&length<=padded&&padded%128==0&&M==5*padded,"E874 exact frozen Token geometry");
 TORCH_CHECK(gate.device()==y.device()&&gate.scalar_type()==y.scalar_type()&&gate.is_contiguous()&&gate.dim()==2&&gate.size(0)==length&&gate.size(1)==d,"E874 one exact shared gate table");
 TORCH_CHECK(out.device()==y.device()&&out.scalar_type()==y.scalar_type()&&out.is_contiguous()&&out.sizes()==y.sizes(),"E874 output geometry");
 const bool has_res=res.numel()>0;
 if(has_res)TORCH_CHECK(res.device()==y.device()&&res.scalar_type()==y.scalar_type()&&res.is_contiguous()&&res.sizes()==y.sizes(),"E874 residual geometry");
 const at::cuda::CUDAGuard guard(y.device());
 const int64_t blocks=(M*(d/8)+255)/256;
 const int64_t cap=(int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount*16;
 const unsigned grid=(unsigned)std::max<int64_t>(1,std::min(blocks,cap));
 auto stream=at::cuda::getCurrentCUDAStream();
 const auto* yp=reinterpret_cast<const __nv_bfloat16*>(y.data_ptr());
 const auto* gp=reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr());
 auto* op=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
 if(has_res)shared_res_gate<true><<<grid,256,0,stream>>>(yp,gp,reinterpret_cast<const __nv_bfloat16*>(res.data_ptr()),op,M,(int)d,(int)length,(int)padded);
 else shared_res_gate<false><<<grid,256,0,stream>>>(yp,gp,nullptr,op,M,(int)d,(int)length,(int)padded);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("shared_res_gate",&token_shared_scale_sm80::launch);}

