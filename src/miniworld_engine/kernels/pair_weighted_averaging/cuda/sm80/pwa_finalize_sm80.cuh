#pragma once
// Finalize PWA's fixed-order reduction buffers directly in parameter dtypes.
// The pair products preserve separate FP32 multiply/add rounding, as in Torch.
namespace pwa80 {
struct FinalizeOutputs { void* data[8]; unsigned bf16; };
template<int I>
__device__ __forceinline__ void finalize_store(const FinalizeOutputs& out,int64_t i,float value) {
  if(out.bf16&(1u<<I))reinterpret_cast<__nv_bfloat16*>(out.data[I])[i]=__float2bfloat16_rn(value);
  else reinterpret_cast<float*>(out.data[I])[i]=value;
}
__device__ __forceinline__ float pair_head_sum(const __nv_bfloat16* wb,const float* value,int column,int dz,bool head_scalar) {
  float acc[4]={0.f,0.f,0.f,0.f};
#pragma unroll
  for(int h=0;h<8;++h) {
    const float product=__fmul_rn(__bfloat162float(wb[h*dz+column]),value[head_scalar ? h : h*dz+column]);
    acc[h%4]=__fadd_rn(acc[h%4],product);
  }
  return __fadd_rn(__fadd_rn(__fadd_rn(acc[0],acc[1]),acc[2]),acc[3]);
}
__global__ void __launch_bounds__(256) pwa_finalize_typed_kernel(
    const float* dgm,const float* dbm,const float* dwv,const float* glue,
    const float* big_m,const float* sums,const float* lnw,const float* lnb,
    const __nv_bfloat16* wb,FinalizeOutputs out,int dm,int hc,int dz) {
  const int gch=hc<64 ? hc : 64;
  const int64_t matrix=(int64_t)dm*hc;
  const int64_t total=2*dm+3*matrix+10*dz;
  for(int64_t index=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;index<total;index+=(int64_t)blockDim.x*gridDim.x) {
    int64_t i=index;
    if(i<dm){finalize_store<0>(out,i,dgm[i]);continue;}i-=dm;
    if(i<dm){finalize_store<1>(out,i,dbm[i]);continue;}i-=dm;
    if(i<matrix){finalize_store<2>(out,i,dwv[i]);continue;}i-=matrix;
    if(i<matrix){
      const int group=i/(dm*gch),inner=i%(dm*gch);
      finalize_store<3>(out,i,glue[group*(2*dm*gch)+dm*gch+inner]);continue;
    }i-=matrix;
    if(i<matrix){
      const int row=i/hc,column=i%hc,group=column/gch;
      finalize_store<7>(out,i,glue[group*(2*dm*gch)+row*gch+column%gch]);continue;
    }i-=matrix;
    if(i<dz){finalize_store<4>(out,i,pair_head_sum(wb,big_m,i,dz,false));continue;}i-=dz;
    if(i<dz){finalize_store<5>(out,i,pair_head_sum(wb,sums,i,dz,true));continue;}i-=dz;
    if(i<8*dz){
      const int column=i%dz,head=i/dz;
      const float value=__fadd_rn(__fmul_rn(lnw[column],big_m[i]),__fmul_rn(lnb[column],sums[head]));
      finalize_store<6>(out,i,value);
    }
  }
}
} // namespace pwa80
