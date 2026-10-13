// E126 original Inductor sum3: left-associated F32 additions, one final BF16 store. Independent experimental DSO.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
namespace atom_gradient_join_compiled80 {
__device__ __forceinline__ uint32_t add_pair(uint32_t a,uint32_t b){
 const float2 fa=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a));
 const float2 fb=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b));
 const __nv_bfloat162 out=__floats2bfloat162_rn(__fadd_rn(fa.x,fb.x),__fadd_rn(fa.y,fb.y));
 return *reinterpret_cast<const uint32_t*>(&out);
}
template<int PLANES,int VEC,int THREADS>
__global__ void join(const uint32_t* __restrict__ a,const uint32_t* __restrict__ b,const uint32_t* __restrict__ c,uint32_t* __restrict__ out,int64_t count){
 const int64_t base=(int64_t(blockIdx.x)*THREADS+threadIdx.x)*VEC;
#pragma unroll
 for(int q=0;q<VEC;++q){const int64_t i=base+q;if(i<count){uint32_t value;
 if constexpr(PLANES==2)value=add_pair(a[i],b[i]);
 else{
  const uint32_t va=a[i],vb=b[i],vc=c[i];
  const float2 fa=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&va));
  const float2 fb=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vb));
  const float2 fc=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vc));
  const __nv_bfloat162 result=__floats2bfloat162_rn(__fadd_rn(__fadd_rn(fa.x,fb.x),fc.x),__fadd_rn(__fadd_rn(fa.y,fb.y),fc.y));
  value=*reinterpret_cast<const uint32_t*>(&result);
 }out[i]=value;}}
}
template<int PLANES,int VEC,int THREADS>
void launch(const torch::Tensor&a,const torch::Tensor&b,const torch::Tensor&c,const torch::Tensor&out){
 const int64_t count=a.numel()/2;
 join<PLANES,VEC,THREADS><<<(count+VEC*THREADS-1)/(VEC*THREADS),THREADS,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint32_t*>(a.data_ptr()),reinterpret_cast<const uint32_t*>(b.data_ptr()),reinterpret_cast<const uint32_t*>(c.data_ptr()),reinterpret_cast<uint32_t*>(out.data_ptr()),count);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(const torch::Tensor&a,const torch::Tensor&b,const torch::Tensor&c,const torch::Tensor&out,int64_t planes,int64_t cfg){
 TORCH_CHECK(a.is_cuda()&&a.scalar_type()==at::kBFloat16&&a.is_contiguous()&&a.numel()%2==0,"contiguous CUDA BF16 packed pairs required");
 for(const auto&t:{b,c,out})TORCH_CHECK(t.is_cuda()&&t.device()==a.device()&&t.scalar_type()==a.scalar_type()&&t.sizes()==a.sizes()&&t.is_contiguous(),"identical physical BF16 tensor metadata required");
 c10::cuda::CUDAGuard guard(a.device());TORCH_CHECK(planes==2||planes==3,"planes2/3 required");
#define CASE(I,V,T) case I:if(planes==2)launch<2,V,T>(a,b,c,out);else launch<3,V,T>(a,b,c,out);break;
 switch(cfg){CASE(0,1,128);CASE(1,1,256);CASE(2,2,128);CASE(3,2,256);CASE(4,4,128);CASE(5,4,256);CASE(6,8,128);CASE(7,8,256);CASE(8,4,512);default:TORCH_CHECK(false,"join cfg0..8");}
#undef CASE
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&atom_gradient_join_compiled80::run);}
