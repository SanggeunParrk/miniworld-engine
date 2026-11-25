// rows_sm80.cuh -- the row stages of the composed SWA atom block on sm_80 (fp32 / bf16 elementwise and per-row reductions around the
// cuBLAS projections and the bf16 window attention): the A100 port of the H100 native row stages (swa_qkvg_fwd.cu, namespace
// swa_native; same stage numbers, operands, rounding points and fixed-order modulation reductions, no floating atomics). Plain CUDA:
// one warp per row (4 rows per 128-thread block), nothing arch-specific. Used by the fp32 block (cuda/sm80/__init__.py block_fwd_fp32 /
// block_bwd_fp32) and the fp32 modulation.
//   stage 0  x = RMS(q) (1 + scale_a) + shift_a                       stage 6   dffn = dy gate_f; d gate_f (sum over augments)
//   stage 1  qk head RMSNorm + RoPE -> Qh / Kh / Vh (bf16), G, pq, pk stage 7   SwiGLU backward
//   stage 2  gated = o sigmoid(g)                                     stage 8   dq1 (FFN adaLN backward); d shift_f / d scale_f
//   stage 3  q1 = q + gate_a att; y = RMS(q1) (1 + scale_f) + shift_f stage 9   datt = dq1 gate_a, gated; d gate_a
//   stage 4  hh = silu(a) b                                           stage 10  dO (bf16), dG, the attention's D = rowsum(dO o)
//   stage 5  out = q1 + gate_f ffn                                    stage 11  dP (qk norm + RoPE backward)
//   stage 14 dc = silu'(c) dy (the modulation's silu)                 stage 12  dq (attention adaLN backward); d shift_a / d scale_a
#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <vector>

namespace swa_rows80 {
struct Rows { const void* x[12]; void* y[8]; bool xb[12],yb[8],bf; int A,B,S,stage; float eps; };
__device__ float get(const Rows& p,int k,long i) { return p.xb[k]?__bfloat162float(((const __nv_bfloat16*)p.x[k])[i]):((const float*)p.x[k])[i]; }
__device__ void put(const Rows& p,int k,long i,float v) { if(p.yb[k]) ((__nv_bfloat16*)p.y[k])[i]=__float2bfloat16_rn(v); else ((float*)p.y[k])[i]=v; }
__device__ float rnd(float v,bool bf) { return bf?__bfloat162float(__float2bfloat16_rn(v)):v; }
__device__ float sum(float v) { for(int d=16;d;d>>=1)v+=__shfl_xor_sync(0xffffffff,v,d); return v; }
__device__ float sig(float v) { return 1.f/(1.f+expf(-v)); }
template<typename K,typename...T> void launch(K fn,int grid,T...args) {
 cudaKernel_t k; C10_CUDA_CHECK(cudaGetKernel(&k,fn)); void* a[]={static_cast<void*>(&args)...};
 C10_CUDA_CHECK(cudaLaunchKernel((const void*)k,dim3(grid),dim3(128),a,0,at::cuda::getCurrentCUDAStream())); C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void rows(Rows p) {
 long r=(long)blockIdx.x*4+threadIdx.x/32; int lane=threadIdx.x%32; long M=(long)p.A*p.B*p.S;
 bool reduce=p.stage==8||p.stage==9||p.stage==12||p.stage==6;
 if(r>=(reduce?(long)p.B*p.S:M))return;
 long mr=r%(p.B*p.S);
 if(p.stage==0) {
  float z[4],ss=0; for(int h=0;h<4;++h){z[h]=get(p,0,r*128+h*32+lane);ss+=z[h]*z[h];}float rs=rsqrtf(sum(ss)/128+p.eps);
  for(int h=0;h<4;++h){int c=h*32+lane;put(p,0,r*128+c,z[h]*rs*(1+get(p,1,mr*768+128+c))+get(p,1,mr*768+c));}
 } else if(p.stage==1) {
  for(int h=0;h<4;++h){int c=h*32+lane;long hm=((r/p.S*4+h)*p.S+r%p.S)*32+lane;
   float cs=get(p,1,mr*16+lane%16),sn=get(p,2,mr*16+lane%16);
   for(int k=0;k<2;++k){float z=get(p,0,r*512+k*128+c);put(p,4+k,r*128+c,z);float v=rnd(z*rsqrtf(sum(z*z)/32+p.eps),p.bf);float pair=__shfl_xor_sync(0xffffffff,v,16);put(p,k,hm,v*cs+(lane<16?-pair:pair)*sn);}
   put(p,2,hm,get(p,0,r*512+256+c));put(p,3,r*128+c,get(p,0,r*512+384+c));
  }
 } else if(p.stage==2) {
  for(int h=0;h<4;++h){long i=r*128+h*32+lane;put(p,0,i,get(p,0,i)*sig(get(p,1,i)));}
 } else if(p.stage==3) {
  float z[4],ss=0;for(int h=0;h<4;++h){int c=h*32+lane;long i=r*128+c;z[h]=rnd(get(p,0,i)+rnd(rnd(get(p,2,mr*768+256+c),p.bf)*get(p,1,i),p.bf),p.bf);ss+=z[h]*z[h];put(p,0,i,z[h]);}
  float rs=rsqrtf(sum(ss)/128+p.eps);for(int h=0;h<4;++h){int c=h*32+lane;put(p,1,r*128+c,z[h]*rs*(1+get(p,2,mr*768+512+c))+get(p,2,mr*768+384+c));}
 } else if(p.stage==4||p.stage==7) {
  for(int h=0;h<8;++h){int c=h*32+lane;float a=get(p,0,r*512+c),b=get(p,0,r*512+256+c),s=sig(a);if(p.stage==4)put(p,0,r*256+c,a*s*b);else{float dh=get(p,1,r*256+c);put(p,0,r*512+c,dh*b*s*(1+a*(1-s)));put(p,0,r*512+256+c,dh*a*s);put(p,1,r*256+c,a*s*b);}}
 } else if(p.stage==5) {
  for(int h=0;h<4;++h){int c=h*32+lane;long i=r*128+c;put(p,0,i,get(p,0,i)+rnd(rnd(get(p,2,mr*768+640+c),p.bf)*get(p,1,i),p.bf));}
 } else if(p.stage==6||p.stage==9) {
  int col=p.stage==6?640:256;float dm[4]={};
  for(int a=0;a<p.A;++a){long rr=r+(long)a*p.B*p.S;for(int h=0;h<4;++h){int c=h*32+lane;long i=rr*128+c;float d=get(p,0,i);dm[h]+=d*get(p,2,i);put(p,0,i,d*rnd(get(p,1,mr*768+col+c),p.bf));if(p.stage==9)put(p,1,i,get(p,3,i)*sig(get(p,4,i)));}}
  for(int h=0;h<4;++h)put(p,p.stage==6?1:2,mr*768+col+h*32+lane,dm[h]);
 } else if(p.stage==8||p.stage==12) {
  int col=p.stage==8?384:0;float ds[4]={},dt[4]={};
  for(int a=0;a<p.A;++a){long rr=r+(long)a*p.B*p.S;float q[4],g[4],ss=0,dot=0;for(int h=0;h<4;++h){int c=h*32+lane;q[h]=get(p,2,rr*128+c);ss+=q[h]*q[h];}
   float rs=rsqrtf(sum(ss)/128+p.eps);for(int h=0;h<4;++h){int c=h*32+lane;float d=get(p,1,rr*128+c);q[h]*=rs;ds[h]+=d;dt[h]+=d*q[h];g[h]=d*(1+get(p,3,mr*768+col+128+c));dot+=g[h]*q[h];}dot=sum(dot)/128;
   for(int h=0;h<4;++h){int c=h*32+lane;put(p,0,rr*128+c,get(p,0,rr*128+c)+rs*(g[h]-q[h]*dot));}}
  for(int h=0;h<4;++h){int c=h*32+lane;put(p,1,mr*768+col+c,ds[h]);put(p,1,mr*768+col+128+c,dt[h]);}
 } else if(p.stage==10) {
  for(int h=0;h<4;++h){long i=r*128+h*32+lane;float d=get(p,0,i),o=get(p,1,i),s=sig(get(p,2,i));float dO=rnd(d*s,true);put(p,0,i,dO);put(p,1,i,d*o*s*(1-s));float delta=sum(dO*o);if(lane==0)put(p,2,(r/p.S*4+h)*p.S+r%p.S,delta);}
 } else if(p.stage==11) {
  for(int h=0;h<4;++h){int c=h*32+lane;long hm=((r/p.S*4+h)*p.S+r%p.S)*32+lane;float cs=get(p,6,mr*16+lane%16),sn=get(p,7,mr*16+lane%16);
   for(int k=0;k<2;++k){float g=get(p,2+k,hm),pair=__shfl_xor_sync(0xffffffff,g,16);g=rnd(g*cs+(lane<16?pair:-pair)*sn,p.bf);float z=get(p,k,r*128+c),rs=rsqrtf(sum(z*z)/32+p.eps),xh=z*rs;float dz=rs*(g-xh*sum(g*xh)/32);put(p,0,r*512+k*128+c,dz);}
   put(p,0,r*512+256+c,get(p,4,hm));put(p,0,r*512+384+c,get(p,5,r*128+c));}
 }
}
struct Copy {const void* x[8];void* y[8];bool xb[8],yb[8];long sizes[8];int count,stage;};
__global__ void copies(Copy p){long j=(long)blockIdx.x*128+threadIdx.x;for(int k=0;k<p.count;++k){if(j>=p.sizes[k])continue;float z=p.xb[k]?__bfloat162float(((const __nv_bfloat16*)p.x[k])[j]):((const float*)p.x[k])[j];
 if(p.stage==1)z=rnd(z*sig(z),p.xb[k]);long dest=j;if(p.stage==2){long row=j/128,c=j%128;dest=((row%256)/64*128+(row/256)*64+row%64)*128+c;}
 if(p.stage==3){if(k==1)dest=(j%256)*128+j/256;else{long row=j/128,c=j%128,pr=(row%256)/32*64+(row/256)*32+row%32;dest=k==0?pr*128+c:c*512+pr;}}
 if(p.yb[k])((__nv_bfloat16*)p.y[k])[dest]=__float2bfloat16_rn(z);else((float*)p.y[k])[dest]=z;}}
__global__ void mod_bwd(Rows p,int count){int i=blockIdx.x*128+threadIdx.x;if(i>=count)return;float c=get(p,0,i),s=sig(c),d=rnd(get(p,1,i),p.bf);put(p,0,i,d*s*(1+c*(1-s)));}
}
inline void native_rows(std::vector<torch::Tensor> x,std::vector<torch::Tensor> y,int stage,int A,int B,int S,bool bf,double eps){
 TORCH_CHECK(x.size()<=12&&y.size()<=8,"native row arity");swa_rows80::Rows p{};p.A=A;p.B=B;p.S=S;p.stage=stage;p.bf=bf;p.eps=eps;
 for(int k=0;k<x.size();++k){TORCH_CHECK(x[k].is_cuda()&&x[k].is_contiguous(),"native row inputs must be contiguous");p.x[k]=x[k].data_ptr();p.xb[k]=x[k].scalar_type()==torch::kBFloat16;}
 for(int k=0;k<y.size();++k){TORCH_CHECK(y[k].is_cuda()&&y[k].is_contiguous(),"native row outputs must be contiguous");p.y[k]=y[k].data_ptr();p.yb[k]=y[k].scalar_type()==torch::kBFloat16;}
 if(stage==14)swa_rows80::launch(swa_rows80::mod_bwd,(x[0].numel()+127)/128,p,(int)x[0].numel());
 else {int nr=(stage==6||stage==8||stage==9||stage==12)?B*S:A*B*S;swa_rows80::launch(swa_rows80::rows,(nr+3)/4,p);}
}
inline void native_copy(std::vector<torch::Tensor>x,std::vector<torch::Tensor>y,int stage){TORCH_CHECK(x.size()==y.size()&&x.size()<=8,"native copy arity");swa_rows80::Copy p{};p.count=x.size();p.stage=stage;long mx=0;
 for(int k=0;k<x.size();++k){TORCH_CHECK(x[k].is_cuda()&&y[k].is_cuda()&&x[k].is_contiguous()&&y[k].is_contiguous()&&x[k].numel()==y[k].numel(),"native copy shape/layout");p.x[k]=x[k].data_ptr();p.y[k]=y[k].data_ptr();p.xb[k]=x[k].scalar_type()==torch::kBFloat16;p.yb[k]=y[k].scalar_type()==torch::kBFloat16;p.sizes[k]=x[k].numel();mx=std::max(mx,p.sizes[k]);}swa_rows80::launch(swa_rows80::copies,(mx+127)/128,p);}
