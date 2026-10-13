// E137. Actual uint2/uint4 global transfers, independent of E126 scalar namespaces.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <map>
#include <string>
namespace atom_gradient_join_vector80 {
template<int PACK> struct Vector;
template<> struct Vector<2>{using type=uint2;};
template<> struct Vector<4>{using type=uint4;};
__device__ __forceinline__ float2 unpack(uint32_t value){
 return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&value));
}
__device__ __forceinline__ uint32_t pack(float x,float y){
 const __nv_bfloat162 value=__floats2bfloat162_rn(x,y);
 return *reinterpret_cast<const uint32_t*>(&value);
}
template<int PLANES,bool INTERMEDIATE>
__device__ __forceinline__ uint32_t sum(uint32_t a,uint32_t b,uint32_t c){
 const float2 fa=unpack(a),fb=unpack(b);
 float x=__fadd_rn(fa.x,fb.x),y=__fadd_rn(fa.y,fb.y);
 if constexpr(PLANES==3){
  if constexpr(INTERMEDIATE){const float2 rounded=unpack(pack(x,y));x=rounded.x;y=rounded.y;}
  const float2 fc=unpack(c);x=__fadd_rn(x,fc.x);y=__fadd_rn(y,fc.y);
 }
 return pack(x,y);
}
template<int PLANES,bool INTERMEDIATE,int PACK,int THREADS,int ITER>
__global__ void join(const typename Vector<PACK>::type* __restrict__ a,const typename Vector<PACK>::type* __restrict__ b,const typename Vector<PACK>::type* __restrict__ c,typename Vector<PACK>::type* __restrict__ out,int64_t count){
 const int64_t base=(int64_t(blockIdx.x)*THREADS+threadIdx.x)*ITER;
#pragma unroll
 for(int it=0;it<ITER;++it){
  const int64_t i=base+it;
  if(i<count){
   const typename Vector<PACK>::type av=a[i],bv=b[i];
   typename Vector<PACK>::type cv{},result;
   if constexpr(PLANES==3)cv=c[i];
   const uint32_t *ap=reinterpret_cast<const uint32_t*>(&av),*bp=reinterpret_cast<const uint32_t*>(&bv),*cp=reinterpret_cast<const uint32_t*>(&cv);
   uint32_t *rp=reinterpret_cast<uint32_t*>(&result);
#pragma unroll
   for(int q=0;q<PACK;++q)rp[q]=sum<PLANES,INTERMEDIATE>(ap[q],bp[q],cp[q]);
   out[i]=result;
  }
 }
}
template<int PLANES,bool INTERMEDIATE,int PACK,int THREADS,int ITER>
void launch(const torch::Tensor&a,const torch::Tensor&b,const torch::Tensor&c,const torch::Tensor&out){
 using V=typename Vector<PACK>::type;TORCH_CHECK(a.numel()%(2*PACK)==0,"aligned vector extent required");
 for(const auto&t:{a,b,c,out})TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%sizeof(V)==0,"aligned vector pointer required");
 const int64_t count=a.numel()/(2*PACK);
 join<PLANES,INTERMEDIATE,PACK,THREADS,ITER><<<(count+THREADS*ITER-1)/(THREADS*ITER),THREADS,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const V*>(a.data_ptr()),reinterpret_cast<const V*>(b.data_ptr()),reinterpret_cast<const V*>(c.data_ptr()),reinterpret_cast<V*>(out.data_ptr()),count);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<int PLANES,bool INTERMEDIATE>
void select(const torch::Tensor&a,const torch::Tensor&b,const torch::Tensor&c,const torch::Tensor&out,int64_t cfg){
#define CASE(I,P,T,N) case I:launch<PLANES,INTERMEDIATE,P,T,N>(a,b,c,out);break;
 switch(cfg){CASE(0,2,128,1);CASE(1,2,256,1);CASE(2,2,512,1);CASE(3,4,128,1);CASE(4,4,256,1);CASE(5,4,512,1);CASE(6,2,128,2);CASE(7,2,256,2);CASE(8,2,512,2);CASE(9,4,128,2);CASE(10,4,256,2);CASE(11,4,512,2);default:TORCH_CHECK(false,"vector cfg0..11");}
#undef CASE
}
void run(const torch::Tensor&a,const torch::Tensor&b,const torch::Tensor&c,const torch::Tensor&out,int64_t planes,int64_t cfg,bool intermediate){
 TORCH_CHECK(a.is_cuda()&&a.scalar_type()==at::kBFloat16&&a.is_contiguous(),"contiguous CUDA BF16 required");
 for(const auto&t:{b,c,out})TORCH_CHECK(t.is_cuda()&&t.device()==a.device()&&t.scalar_type()==a.scalar_type()&&t.sizes()==a.sizes()&&t.is_contiguous(),"identical physical BF16 tensor metadata required");
 c10::cuda::CUDAGuard guard(a.device());TORCH_CHECK(planes==2||planes==3,"planes2/3 required");
 if(planes==2){if(intermediate)select<2,true>(a,b,c,out,cfg);else select<2,false>(a,b,c,out,cfg);}
 else{if(intermediate)select<3,true>(a,b,c,out,cfg);else select<3,false>(a,b,c,out,cfg);}
}
template<int PLANES,bool INTERMEDIATE,int PACK,int THREADS,int ITER>
std::map<std::string,int64_t> properties(){
 cudaFuncAttributes attrs;int blocks=0;
 C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs,join<PLANES,INTERMEDIATE,PACK,THREADS,ITER>));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,join<PLANES,INTERMEDIATE,PACK,THREADS,ITER>,THREADS,0));
 return {{"registers_per_thread",attrs.numRegs},{"local_bytes",attrs.localSizeBytes},{"static_shared_bytes",attrs.sharedSizeBytes},{"max_active_blocks_per_SM",blocks},{"threads",THREADS},{"packed_uint32_words",PACK},{"iterations",ITER}};
}
template<int PLANES,bool INTERMEDIATE>
std::map<std::string,int64_t> select_properties(int64_t cfg){
#define PROP(I,P,T,N) case I:return properties<PLANES,INTERMEDIATE,P,T,N>();
 switch(cfg){PROP(0,2,128,1);PROP(1,2,256,1);PROP(2,2,512,1);PROP(3,4,128,1);PROP(4,4,256,1);PROP(5,4,512,1);PROP(6,2,128,2);PROP(7,2,256,2);PROP(8,2,512,2);PROP(9,4,128,2);PROP(10,4,256,2);PROP(11,4,512,2);default:TORCH_CHECK(false,"vector cfg0..11");}
#undef PROP
}
std::map<std::string,int64_t> stats(int64_t planes,int64_t cfg,bool intermediate){
 TORCH_CHECK(planes==2||planes==3,"planes2/3 required");
 if(planes==2){if(intermediate)return select_properties<2,true>(cfg);return select_properties<2,false>(cfg);}
 if(intermediate)return select_properties<3,true>(cfg);return select_properties<3,false>(cfg);
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&atom_gradient_join_vector80::run);m.def("stats",&atom_gradient_join_vector80::stats);}
