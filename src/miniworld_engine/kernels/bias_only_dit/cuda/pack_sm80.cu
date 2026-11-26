// One inference weight-pack launch. Source parameters stay live on graph replay.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace {
struct Pack {
  const void* src[20];
  bool fp32[20];
  void* dst[7];
  int end[7], da;
};
__device__ __forceinline__ float read(const Pack& p, int w, int i) {
  return p.fp32[w] ? static_cast<const float*>(p.src[w])[i]
                   : __bfloat162float(static_cast<const __nv_bfloat16*>(p.src[w])[i]);
}
__device__ __forceinline__ void read8(const Pack& p, int w, int i, float* v) {
  if (reinterpret_cast<uintptr_t>(p.src[w]) & 15) {
#pragma unroll
    for (int k=0;k<8;++k) v[k]=read(p,w,i+k);
    return;
  }
  if (p.fp32[w]) {
    float4 a = *reinterpret_cast<const float4*>(static_cast<const float*>(p.src[w]) + i);
    float4 b = *reinterpret_cast<const float4*>(static_cast<const float*>(p.src[w]) + i + 4);
    v[0]=a.x;v[1]=a.y;v[2]=a.z;v[3]=a.w;
    v[4]=b.x;v[5]=b.y;v[6]=b.z;v[7]=b.w;
  } else {
    uint4 a = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.src[w]) + i);
    unsigned u[4]={a.x,a.y,a.z,a.w};
#pragma unroll
    for(int k=0;k<4;++k) {
      float2 f=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u[k]));
      v[2*k]=f.x;v[2*k+1]=f.y;
    }
  }
}
template <bool F32>
__global__ void pack_kernel(Pack p) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= p.end[6]) return;
  int part = 0;
  while (i >= p.end[part]) ++part;
  i -= part ? p.end[part - 1] : 0;
  float v[8]={};
  if (part == 0) { // V | gate
    read8(p, i < p.da * 768 ? 0 : 1, i % (p.da * 768),v);
  } else if (part == 1) { // expand A | B
    read8(p, i < 1536 * 768 ? 17 : 18, i % (1536 * 768),v);
  } else if (part == 2) { // conditioning scale/shift, with LN weight folded
    int q = i / (768 * 384), j = i % (768 * 384);
    int w = q == 0 ? 4 : q == 1 ? 6 : q == 2 ? 12 : 14;
    float scale[8];read8(p,w,j,v);read8(p,q<2?3:11,j%384,scale);
#pragma unroll
    for(int k=0;k<8;++k)v[k]*=scale[k];
  } else if (part == 3) {
    if (i < 768) read8(p, 5, i,v);
    else if (i >= 1536 && i < 2304) read8(p, 13, i - 1536,v);
  } else if (part == 4) {
    read8(p, i < 768 * 384 ? 7 : 15, i % (768 * 384),v);
  } else if (part == 5) {
    read8(p, i < 768 ? 8 : 16, i % 768,v);
  } else { // pair projection [heads, 128]
    float scale[8];read8(p,10,i,v);read8(p,9,i%128,scale);
#pragma unroll
    for(int k=0;k<8;++k)v[k]*=scale[k];
  }
  if (F32) {
    float4* d = reinterpret_cast<float4*>(static_cast<float*>(p.dst[part]) + i);
    d[0] = make_float4(v[0], v[1], v[2], v[3]); d[1] = make_float4(v[4], v[5], v[6], v[7]);
    return;
  }
  uint4 out;unsigned u[4];
#pragma unroll
  for(int k=0;k<4;++k){__nv_bfloat162 bf=__floats2bfloat162_rn(v[k*2],v[k*2+1]);u[k]=*reinterpret_cast<unsigned*>(&bf);}
  out.x=u[0];out.y=u[1];out.z=u[2];out.w=u[3];
  *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(p.dst[part])+i)=out;
}
}

std::vector<torch::Tensor> pack_weights(std::vector<torch::Tensor> weights, int64_t heads, int64_t da, bool f32) {
  TORCH_CHECK(weights.size() == 20, "bias-only pack needs 20 parameters");
  TORCH_CHECK((heads == 16 && (da == 768 || da == 1024)) || ((heads == 12 || heads == 24) && da == 768), "bias-only head layout");
  c10::cuda::CUDAGuard guard(weights[0].device());
  const int64_t sizes[20] = {da*768,da*768,768*da,384,768*384,768,768*384,
    768*384,768,128,heads*128,384,768*384,768,768*384,768*384,768,
    1536*768,1536*768,768*1536};
  Pack p{}; p.da=da;
  for (int j=0;j<20;++j) {
    const auto& w=weights[j];
    TORCH_CHECK(w.is_cuda() && w.device()==weights[0].device() && w.is_contiguous()
      && w.numel()==sizes[j] && (w.scalar_type()==at::kBFloat16 || w.scalar_type()==at::kFloat), "bias-only pack parameter ",j);
    p.src[j]=w.data_ptr();p.fp32[j]=w.scalar_type()==at::kFloat;
  }
  auto opts=weights[0].options().dtype(f32 ? at::kFloat : at::kBFloat16);
  std::vector<torch::Tensor> out={torch::empty({2*da,768},opts),torch::empty({3072,768},opts),
    torch::empty({3072,384},opts),torch::empty({3072},opts),torch::empty({1536,384},opts),
    torch::empty({1536},opts),torch::empty({heads,128},opts)};
  int total=0;
  for(int j=0;j<7;++j){total+=out[j].numel();p.end[j]=total;p.dst[j]=out[j].data_ptr();}
  if (f32) pack_kernel<true><<<(total+2047)/2048,256,0,at::cuda::getCurrentCUDAStream()>>>(p);
  else pack_kernel<false><<<(total+2047)/2048,256,0,at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("pack_weights",&pack_weights);}
