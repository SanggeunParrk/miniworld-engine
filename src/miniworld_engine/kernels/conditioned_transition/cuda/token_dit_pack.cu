// Pack one inference block in one launch. Every source is read on graph replay.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdlib>

namespace {
struct Pack {
  const void* src[23];
  bool fp32[23];
  __nv_bfloat16* dst[8];
  int end[8];
  float qscale, bscale;
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
__global__ void pack_scalar_kernel(Pack p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.end[7]) return;
  int part = 0;
  while (i >= p.end[part]) ++part;
  i -= part ? p.end[part - 1] : 0;
  float v = 0.f;
  if (part == 0) {  // Q | K | V | gate, Q pre-scaled for exp2 softmax
    int quarter = i / (768 * 768), j = i % (768 * 768);
    int w = quarter == 0 ? 0 : quarter + 1;
    v = read(p, w, j) * (quarter == 0 ? p.qscale : 1.f);
  } else if (part == 1) {
    if (i < 768) v = read(p, 1, i) * p.qscale;
  } else if (part == 2) {  // transition expand A | B
    v = read(p, i < 1536 * 768 ? 20 : 21, i % (1536 * 768));
  } else if (part == 3) {  // conditioning scale/shift of both halves, LN weight folded
    int quarter = i / (768 * 384), j = i % (768 * 384);
    int w = quarter == 0 ? 7 : quarter == 1 ? 9 : quarter == 2 ? 15 : 17;
    v = read(p, w, j) * read(p, quarter < 2 ? 6 : 14, j % 384);
  } else if (part == 4) {
    if (i < 768) v = read(p, 8, i);
    else if (i >= 1536 && i < 2304) v = read(p, 16, i - 1536);
  } else if (part == 5) {
    v = read(p, i < 768 * 384 ? 10 : 18, i % (768 * 384));
  } else if (part == 6) {
    v = read(p, i < 768 ? 11 : 19, i % 768);
  } else {  // pair projection: [128, 16], with LN weight and log2(e) folded
    int c = i / 16, h = i % 16;
    v = (read(p, 13, h * 128 + c) * read(p, 12, c)) * p.bscale;
  }
  p.dst[part][i] = __float2bfloat16_rn(v);
}
__global__ void pack_vector_kernel(Pack p) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= p.end[7]) return;
  int part = 0;
  while (i >= p.end[part]) ++part;
  i -= part ? p.end[part - 1] : 0;
  float v[8] = {};
  if (part == 0) {
    int q=i/(768*768),j=i%(768*768);
    read8(p,q==0?0:q+1,j,v);
#pragma unroll
    for(int k=0;k<8;++k)v[k]*=q==0?p.qscale:1.f;
  } else if (part == 1) {
    if(i<768){read8(p,1,i,v);
#pragma unroll
      for(int k=0;k<8;++k)v[k]*=p.qscale;
    }
  } else if (part == 2) {
    read8(p,i<1536*768?20:21,i%(1536*768),v);
  } else if (part == 3) {
    int q=i/(768*384),j=i%(768*384);
    int w=q==0?7:q==1?9:q==2?15:17;
    float scale[8];read8(p,w,j,v);read8(p,q<2?6:14,j%384,scale);
#pragma unroll
    for(int k=0;k<8;++k)v[k]*=scale[k];
  } else if (part == 4) {
    if(i<768)read8(p,8,i,v);
    else if(i>=1536 && i<2304)read8(p,16,i-1536,v);
  } else if (part == 5) {
    read8(p,i<768*384?10:18,i%(768*384),v);
  } else if (part == 6) {
    read8(p,i<768?11:19,i%768,v);
  } else {
    float scale[8];read8(p,13,i,v);read8(p,12,i%128,scale);
#pragma unroll
    for(int k=0;k<8;++k)v[k]=(v[k]*scale[k])*p.bscale;
  }
  unsigned u[4];
#pragma unroll
  for(int k=0;k<4;++k){__nv_bfloat162 bf=__floats2bfloat162_rn(v[2*k],v[2*k+1]);u[k]=*reinterpret_cast<unsigned*>(&bf);}
  *reinterpret_cast<uint4*>(p.dst[part]+i)=make_uint4(u[0],u[1],u[2],u[3]);
}
}

std::vector<torch::Tensor> pack_token(std::vector<torch::Tensor> weights, double qscale, double bscale) {
  TORCH_CHECK(weights.size() == 23, "token pack needs 23 parameters");
  c10::cuda::CUDAGuard guard(weights[0].device());
  const int sizes[23] = {768*768,768,768*768,768*768,768*768,768*768,
    384,768*384,768,768*384,768*384,768,128,16*128,384,768*384,768,
    768*384,768*384,768,1536*768,1536*768,768*1536};
  Pack p{};p.qscale=qscale;p.bscale=bscale;
  for (int j=0;j<23;++j) {
    const auto& w=weights[j];
    TORCH_CHECK(w.is_cuda() && w.device()==weights[0].device() && w.is_contiguous()
      && w.numel()==sizes[j] && (w.scalar_type()==at::kBFloat16 || w.scalar_type()==at::kFloat), "token pack parameter ",j);
    p.src[j]=w.data_ptr();p.fp32[j]=w.scalar_type()==at::kFloat;
  }
  auto opts=weights[0].options().dtype(at::kBFloat16);
  std::vector<torch::Tensor> out={torch::empty({3072,768},opts),torch::empty({3072},opts),
    torch::empty({3072,768},opts),torch::empty({3072,384},opts),torch::empty({3072},opts),
    torch::empty({1536,384},opts),torch::empty({1536},opts),torch::empty({128,16},opts)};
  int total=0;
  for(int j=0;j<8;++j){total+=out[j].numel();p.end[j]=total;p.dst[j]=reinterpret_cast<__nv_bfloat16*>(out[j].data_ptr());}
  const char* mode=std::getenv("MINIWORLD_A100_CLEANUP");
  if(mode && mode[0]=='0') pack_scalar_kernel<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(p);
  else pack_vector_kernel<<<(total+2047)/2048,256,0,at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The CUDA pair producer wants [heads, channels]; expose its transpose as
  // the runner's [channels, heads] contract without another device copy.
  if (!(mode && mode[0]=='0')) out[7]=out[7].view({16,128}).t();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("pack_token",&pack_token);}
