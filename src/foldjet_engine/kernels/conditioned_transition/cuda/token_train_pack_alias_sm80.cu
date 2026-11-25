// The training pack retains unrounded FP32 matrices for all parameter gradients.
// Fold conditioning/pair LN weights in FP32, then round GEMM operands once.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
namespace token_train_pack_alias80 {
struct Pack {
  const void* src[23]; bool f32[23]; void* dst[20]; bool out_f32[20]; bool skip[20];
  int end[20], d, dc, dp;
};
__device__ __forceinline__ float read(const Pack& p,int w,int i) {
  return p.f32[w] ? static_cast<const float*>(p.src[w])[i]
                 : __bfloat162float(static_cast<const __nv_bfloat16*>(p.src[w])[i]);
}
__device__ __forceinline__ void read8(const Pack& p,int w,int i,float* v) {
  if(p.f32[w]) {
    float4 a=*reinterpret_cast<const float4*>(static_cast<const float*>(p.src[w])+i);
    float4 b=*reinterpret_cast<const float4*>(static_cast<const float*>(p.src[w])+i+4);
    v[0]=a.x;v[1]=a.y;v[2]=a.z;v[3]=a.w;v[4]=b.x;v[5]=b.y;v[6]=b.z;v[7]=b.w;
  } else {
    uint4 a=*reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.src[w])+i);
    unsigned u[4]={a.x,a.y,a.z,a.w};
#pragma unroll
    for(int k=0;k<4;++k){float2 t=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u[k]));v[2*k]=t.x;v[2*k+1]=t.y;}
  }
}
__global__ void pack_kernel(Pack p) {
  int i=(blockIdx.x*blockDim.x+threadIdx.x)*8;if(i>=p.end[19])return;
  int part=0;while(i>=p.end[part])++part;i-=part?p.end[part-1]:0;
  if(p.skip[part])return; // Match original detach/to(dtype): no copy when dtype already matches.
  float v[8]={};
  if(part==0||part==1)read8(p,part==0?0:14,i,v);
  else if(part==2||part==3){
    int q=i/(p.d*p.dc),j=i%(p.d*p.dc);int w=q==0?1:q==1?3:q==2?15:17;
    read8(p,w,j,v);
    if(part==3){float scale[8];read8(p,q<2?0:14,j%p.dc,scale);
#pragma unroll
      for(int k=0;k<8;++k)v[k]*=scale[k];}
  } else if(part==4)read8(p,i<p.d*p.dc?4:18,i%(p.d*p.dc),v);
  else if(part>=5&&part<=8)read8(p,part==5?2:part==6?16:part==7?5:19,i,v);
  else if(part==9){int q=i/(p.d*p.d),j=i%(p.d*p.d);read8(p,q==0?6:q==1?8:q==2?9:10,j,v);}
  else if(part==10){if(i<p.d)read8(p,7,i,v);}
  else if(part==11||part==12){
#pragma unroll
    for(int k=0;k<8;++k)v[k]=1.f;
  } else if(part==13)read8(p,12,i,v);
  else if(part==14)read8(p,13,i,v);
  else if(part==15||part==16){float scale[8];read8(p,13,i,v);read8(p,12,i%p.dp,scale);
#pragma unroll
    for(int k=0;k<8;++k)v[k]*=scale[k];
  } else if(part==17)read8(p,11,i,v);
  else if(part==18)read8(p,i<2*p.d*p.d?20:21,i%(2*p.d*p.d),v);
  else read8(p,22,i,v);
  if(p.out_f32[part]){
    float* out=static_cast<float*>(p.dst[part])+i;
    *reinterpret_cast<float4*>(out)=make_float4(v[0],v[1],v[2],v[3]);
    *reinterpret_cast<float4*>(out+4)=make_float4(v[4],v[5],v[6],v[7]);
  } else {
    unsigned u[4];
#pragma unroll
    for(int k=0;k<4;++k){__nv_bfloat162 t=__floats2bfloat162_rn(v[2*k],v[2*k+1]);u[k]=*reinterpret_cast<unsigned*>(&t);}
    *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(p.dst[part])+i)=make_uint4(u[0],u[1],u[2],u[3]);
  }
}
std::vector<torch::Tensor> result(std::vector<torch::Tensor> weights) {
  TORCH_CHECK(weights.size()==23,"no-QK-norm Token training pack requires23 parameters");
  c10::cuda::CUDAGuard guard(weights[0].device());
  const int D=768,DC=384,DP=128,H=16,DH=48;
  const int sizes[23]={DC,D*DC,D,D*DC,D*DC,D,D*D,D,D*D,D*D,D*D,D*D,DP,H*DP,DC,D*DC,D,D*DC,D*DC,D,2*D*D,2*D*D,2*D*D};
  Pack p{};p.d=D;p.dc=DC;p.dp=DP;
  for(int w=0;w<23;++w){const auto& t=weights[w];
    TORCH_CHECK(t.is_cuda()&&t.device()==weights[0].device()&&t.is_contiguous()&&t.numel()==sizes[w]
      &&(t.scalar_type()==at::kFloat||t.scalar_type()==at::kBFloat16)
      &&!(reinterpret_cast<uintptr_t>(t.data_ptr())&15),"Token train pack parameter",w);
    p.src[w]=t.data_ptr();p.f32[w]=t.scalar_type()==at::kFloat;
  }
  auto f=weights[0].options().dtype(at::kFloat),b=f.dtype(at::kBFloat16);
  auto leaf=[&](int field,int source,at::IntArrayRef shape,bool f32){
    const auto dtype=f32?at::kFloat:at::kBFloat16;
    p.skip[field]=weights[source].scalar_type()==dtype;
    return p.skip[field]?weights[source].detach().view(shape):torch::empty(shape,f32?f:b);
  };
  std::vector<torch::Tensor> out={leaf(0,0,{DC},true),leaf(1,14,{DC},true),torch::empty({4*D,DC},f),torch::empty({4*D,DC},b),torch::empty({2*D,DC},b),
    leaf(5,2,{D},true),leaf(6,16,{D},true),leaf(7,5,{D},true),leaf(8,19,{D},true),torch::empty({4*D,D},b),torch::empty({4*D},b),
    torch::empty({DH},f),torch::empty({DH},f),leaf(13,12,{DP},true),leaf(14,13,{H,DP},true),torch::empty({H,DP},f),torch::empty({H,DP},b),
    leaf(17,11,{D,D},false),torch::empty({4*D,D},b),leaf(19,22,{D,2*D},false)};
  int total=0;for(int i=0;i<20;++i){total+=out[i].numel();p.end[i]=total;p.dst[i]=out[i].data_ptr();p.out_f32[i]=out[i].scalar_type()==at::kFloat;}
  pack_kernel<<<(total+2047)/2048,256,0,at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("result",&token_train_pack_alias80::result);}
