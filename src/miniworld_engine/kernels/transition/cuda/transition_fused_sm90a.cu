#if defined(MW_TRANSITION_AUX_ONLY)
#ifndef MW_N
#define MW_N 4              // hidden = MW_N * D (the n of Transition(d, n)); the D128 fused kernels are built per n
#endif
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
using bf = __nv_bfloat16;
namespace {
__device__ float rd(const void* p, int i, bool fp32) {
  return fp32 ? static_cast<const float*>(p)[i] : __bfloat162float(static_cast<const bf*>(p)[i]);
}
struct Prep {
  const void* src[3]; bf* dst[4];
  int rows[3], cols[3], s0[3], s1[3], fp32[3], tasks[4];
};
__global__ void prepare_k(Prep p) {
  __shared__ float tile[32][33];
  int op=p.tasks[blockIdx.y],s=op==3?2:op;
  int ntx=(p.cols[s]+31)/32,r0=(blockIdx.x/ntx)*32,c0=(blockIdx.x%ntx)*32;
  int lane=threadIdx.x&31,rr=threadIdx.x>>5;
  for(int j=rr;j<32;j+=8){
    int r=r0+j,c=c0+lane;
    float v=(r<p.rows[s]&&c<p.cols[s])?rd(p.src[s],r*p.s0[s]+c*p.s1[s],p.fp32[s]):0.f;
    if(op==3)tile[j][lane]=v;
    else if(r<p.rows[s]&&c<p.cols[s])p.dst[op][r*p.cols[s]+c]=__float2bfloat16_rn(v);
  }
  if(op==3){
    __syncthreads();
    for(int j=rr;j<32;j+=8){
      int r=c0+j,c=r0+lane;
      if(r<p.cols[s]&&c<p.rows[s])p.dst[3][r*p.rows[s]+c]=__float2bfloat16_rn(tile[lane][j]);
    }
  }
}
__global__ void pair_layout_k(const bf* a,const bf* b,bf* out,int h,int d,int block,bool transpose) {
  __shared__ float tile[32][33];
  int ntx=(d+31)/32,r0=(blockIdx.x/ntx)*32,c0=(blockIdx.x%ntx)*32;
  int lane=threadIdx.x&31,rr=threadIdx.x>>5;
  for(int j=rr;j<32;j+=8){
    int r=r0+j,c=c0+lane;
    int leaf=block?(r/block)%2:r/h;
    int sr=block?(r/(2*block))*block+r%block:r%h;
    float v=r<2*h&&c<d?__bfloat162float((leaf?b:a)[sr*d+c]):0.f;
    if(transpose)tile[j][lane]=v;
    else if(r<2*h&&c<d)out[r*d+c]=__float2bfloat16_rn(v);
  }
  if(transpose){
    __syncthreads();
    for(int j=rr;j<32;j+=8){
      int r=c0+j,c=r0+lane;
      if(r<d&&c<2*h)out[r*(2*h)+c]=__float2bfloat16_rn(tile[lane][j]);
    }
  }
}
struct Cast { const void* src[8];void* dst[8];int n[8],sf[8],df[8]; };
__global__ void cast_k(Cast p){
  int leaf=blockIdx.y,i=blockIdx.x*256+threadIdx.x;
  if(i>=p.n[leaf])return;
  float v=rd(p.src[leaf],i,p.sf[leaf]);
  if(p.df[leaf])static_cast<float*>(p.dst[leaf])[i]=v;
  else static_cast<bf*>(p.dst[leaf])[i]=__float2bfloat16_rn(v);
}
__device__ float wsum(float x){for(int q=16;q;q>>=1)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,q));return x;}
// Each persistent warp owns rows and keeps its affine-gradient sums in registers.
// A second fixed-order reduction replaces the old global floating-point atomics.
template<int NV,bool HOIST=(NV<=4)> __global__ __launch_bounds__(128,NV>4?3:4)
void ln_bwd_k(const bf* dy,const bf* x,const float* gamma,const float* rs,const float* c1,
              const bf* residual,bf* dx,float* partial,int rows,int d){
  int lane=threadIdx.x&31,warp=threadIdx.x>>5,gw=blockIdx.x*4+warp,nw=gridDim.x*4;
  float ag[4*NV]={0},ab[4*NV]={0};
  union Vec{uint2 u;bf v[4];};
  float4 gm[HOIST?NV:1];                               // gamma stays in registers across the rows of this warp (not at NV = 6: registers)
  if(HOIST){
    #pragma unroll
    for(int v=0;v<NV;++v){int c=(v*32+lane)*4;gm[HOIST?v:0]=c<d?*reinterpret_cast<const float4*>(gamma+c):make_float4(0.f,0.f,0.f,0.f);}
  }
  for(int row=gw;row<rows;row+=nw){
    float xh[4*NV],grad[4*NV],wd[4*NV],a=0.f,b=0.f;
    Vec rv[NV];                                        // the residual is issued with the first loads: it was a second dependent round trip
    #pragma unroll
    for(int v=0;v<NV;++v){int c=(v*32+lane)*4;if(c<d)rv[v].u=*reinterpret_cast<const uint2*>(residual+row*d+c);}
    float rstd=rs[row],offset=c1[row];
    #pragma unroll
    for(int v=0;v<NV;++v){
      int c=(v*32+lane)*4;
      if(c<d){
        Vec xv,gv;xv.u=*reinterpret_cast<const uint2*>(x+row*d+c);gv.u=*reinterpret_cast<const uint2*>(dy+row*d+c);
        const float4 gq=HOIST?gm[HOIST?v:0]:*reinterpret_cast<const float4*>(gamma+c);
        const float g4[4]={gq.x,gq.y,gq.z,gq.w};
        #pragma unroll
        for(int e=0;e<4;++e){
          int k=4*v+e;
          xh[k]=fmaf(__bfloat162float(xv.v[e]),rstd,-offset);
          grad[k]=__bfloat162float(gv.v[e]);wd[k]=g4[e]*grad[k];
          a=__fadd_rn(a,wd[k]*xh[k]);b=__fadd_rn(b,wd[k]);
        }
      }
    }
    a=wsum(a)/d;b=wsum(b)/d;
    #pragma unroll
    for(int v=0;v<NV;++v){
      int c=(v*32+lane)*4;
      if(c<d){
        Vec out;
        #pragma unroll
        for(int e=0;e<4;++e){
          int k=4*v+e;
          out.v[e]=__float2bfloat16_rn((wd[k]-(xh[k]*a+b))*rstd+__bfloat162float(rv[v].v[e]));
          ag[k]=__fadd_rn(ag[k],grad[k]*xh[k]);ab[k]=__fadd_rn(ab[k],grad[k]);
        }
        *reinterpret_cast<uint2*>(dx+row*d+c)=out.u;
      }
    }
  }
  #pragma unroll
  for(int v=0;v<NV;++v){
    int c=(v*32+lane)*4;
    if(c<d){
      #pragma unroll
      for(int e=0;e<4;++e){partial[gw*d+c+e]=ag[4*v+e];partial[(nw+gw)*d+c+e]=ab[4*v+e];}
    }
  }
}
__global__ void ln_reduce_k(const float* partial,float* dg,float* db,int nw,int d){
  __shared__ float sg[256],sb[256];
  int c=blockIdx.x,t=threadIdx.x;float g=0.f,b=0.f;
  for(int r=t;r<nw;r+=256){g=__fadd_rn(g,partial[r*d+c]);b=__fadd_rn(b,partial[(nw+r)*d+c]);}
  sg[t]=g;sb[t]=b;__syncthreads();
  for(int s=128;s;s>>=1){if(t<s){sg[t]=__fadd_rn(sg[t],sg[t+s]);sb[t]=__fadd_rn(sb[t],sb[t+s]);}__syncthreads();}
  if(!t){dg[c]=sg[0];db[c]=sb[0];}
}
// D512 candidate: consume the raw gradient in the input pass, so it does not
// remain live across the row reduction and output pass. Keep affine reduction
// order and the saved-statistics/residual boundary identical to ln_bwd_k.
__global__ __launch_bounds__(128,5)
void ln_bwd_compact512_k(const bf* dy,const bf* x,const float* gamma,const float* rs,const float* c1,
                        const bf* residual,bf* dx,float* partial,int rows){
  constexpr int d=512;
  int lane=threadIdx.x&31,warp=threadIdx.x>>5,gw=blockIdx.x*4+warp,nw=gridDim.x*4;
  float ag[16]={0},ab[16]={0};
  union Vec{uint2 u;bf v[4];};
  for(int row=gw;row<rows;row+=nw){
    float xh[16],wd[16],a=0.f,b=0.f;
    float rstd=rs[row],offset=c1[row];
    #pragma unroll
    for(int v=0;v<4;++v){
      int c=(v*32+lane)*4;
      Vec xv,gv;xv.u=*reinterpret_cast<const uint2*>(x+row*d+c);gv.u=*reinterpret_cast<const uint2*>(dy+row*d+c);
      #pragma unroll
      for(int e=0;e<4;++e){
        int k=4*v+e;float grad=__bfloat162float(gv.v[e]);
        xh[k]=fmaf(__bfloat162float(xv.v[e]),rstd,-offset);wd[k]=gamma[c+e]*grad;
        a=__fadd_rn(a,wd[k]*xh[k]);b=__fadd_rn(b,wd[k]);
        ag[k]=__fadd_rn(ag[k],grad*xh[k]);ab[k]=__fadd_rn(ab[k],grad);
      }
    }
    a=wsum(a)/d;b=wsum(b)/d;
    #pragma unroll
    for(int v=0;v<4;++v){
      int c=(v*32+lane)*4;Vec out,res;res.u=*reinterpret_cast<const uint2*>(residual+row*d+c);
      #pragma unroll
      for(int e=0;e<4;++e){
        int k=4*v+e;
        out.v[e]=__float2bfloat16_rn((wd[k]-(xh[k]*a+b))*rstd+__bfloat162float(res.v[e]));
      }
      *reinterpret_cast<uint2*>(dx+row*d+c)=out.u;
    }
  }
  #pragma unroll
  for(int v=0;v<4;++v){
    int c=(v*32+lane)*4;
    #pragma unroll
    for(int e=0;e<4;++e){partial[gw*d+c+e]=ag[4*v+e];partial[(nw+gw)*d+c+e]=ab[4*v+e];}
  }
}
// D384 candidate: same fused tensor lifetimes, with twelve live channels per lane.
__global__ __launch_bounds__(128,6)
void ln_bwd_compact384_k(const bf* dy,const bf* x,const float* gamma,const float* rs,const float* c1,
                        const bf* residual,bf* dx,float* partial,int rows){
  constexpr int d=384;
  int lane=threadIdx.x&31,warp=threadIdx.x>>5,gw=blockIdx.x*4+warp,nw=gridDim.x*4;
  float ag[12]={0},ab[12]={0};
  union Vec{uint2 u;bf v[4];};
  for(int row=gw;row<rows;row+=nw){
    float xh[12],wd[12],a=0.f,b=0.f;
    float rstd=rs[row],offset=c1[row];
    #pragma unroll
    for(int v=0;v<3;++v){
      int c=(v*32+lane)*4;
      Vec xv,gv;xv.u=*reinterpret_cast<const uint2*>(x+row*d+c);gv.u=*reinterpret_cast<const uint2*>(dy+row*d+c);
      #pragma unroll
      for(int e=0;e<4;++e){
        int k=4*v+e;float grad=__bfloat162float(gv.v[e]);
        xh[k]=fmaf(__bfloat162float(xv.v[e]),rstd,-offset);wd[k]=gamma[c+e]*grad;
        a=__fadd_rn(a,wd[k]*xh[k]);b=__fadd_rn(b,wd[k]);
        ag[k]=__fadd_rn(ag[k],grad*xh[k]);ab[k]=__fadd_rn(ab[k],grad);
      }
    }
    a=wsum(a)/d;b=wsum(b)/d;
    #pragma unroll
    for(int v=0;v<3;++v){
      int c=(v*32+lane)*4;Vec out,res;res.u=*reinterpret_cast<const uint2*>(residual+row*d+c);
      #pragma unroll
      for(int e=0;e<4;++e){
        int k=4*v+e;
        out.v[e]=__float2bfloat16_rn((wd[k]-(xh[k]*a+b))*rstd+__bfloat162float(res.v[e]));
      }
      *reinterpret_cast<uint2*>(dx+row*d+c)=out.u;
    }
  }
  #pragma unroll
  for(int v=0;v<3;++v){
    int c=(v*32+lane)*4;
    #pragma unroll
    for(int e=0;e<4;++e){partial[gw*d+c+e]=ag[4*v+e];partial[(nw+gw)*d+c+e]=ab[4*v+e];}
  }
}
void check(const at::Tensor& t){
  TORCH_CHECK(t.is_cuda()&&(t.scalar_type()==at::kFloat||t.scalar_type()==at::kBFloat16),"CUDA BF16/FP32 tensor required");
}
}
std::vector<at::Tensor> prepare(at::Tensor a,at::Tensor b,at::Tensor s,bool transpose){
  c10::cuda::CUDAGuard guard(a.device());
  std::vector<at::Tensor> in={a,b,s},out;Prep p{};int count=0;
  TORCH_CHECK(a.dim()==2&&b.sizes()==a.sizes()&&s.dim()==2&&s.size(0)==a.size(1)&&s.size(1)==a.size(0),"transition weight shapes");
  for(int j=0;j<3;++j){
    check(in[j]);TORCH_CHECK(in[j].device()==a.device(),"weight device mismatch");
    bool copy=in[j].scalar_type()!=at::kBFloat16||!in[j].is_contiguous();
    out.push_back(copy?at::empty(in[j].sizes(),in[j].options().dtype(at::kBFloat16)):in[j]);
    p.src[j]=in[j].data_ptr();p.dst[j]=reinterpret_cast<bf*>(out[j].data_ptr());
    p.rows[j]=in[j].size(0);p.cols[j]=in[j].size(1);p.s0[j]=in[j].stride(0);p.s1[j]=in[j].stride(1);
    p.fp32[j]=in[j].scalar_type()==at::kFloat;
    if(copy)p.tasks[count++]=j;
  }
  out.push_back(at::empty(transpose?std::vector<int64_t>{s.size(1),s.size(0)}:std::vector<int64_t>{0},s.options().dtype(at::kBFloat16)));
  p.dst[3]=reinterpret_cast<bf*>(out[3].data_ptr());if(transpose)p.tasks[count++]=3;
  int tiles=((a.size(0)+31)/32)*((a.size(1)+31)/32);
  if(count){
    // Pass an explicit kernel handle, as in the native dropout launcher. This
    // avoids the runtime probing a host symbol as a CU kernel under memcheck.
    cudaKernel_t kernel;C10_CUDA_CHECK(cudaGetKernel(&kernel,prepare_k));
    void* args[]={&p};
    C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel),
      dim3(tiles,count),dim3(256),args,0,at::cuda::getCurrentCUDAStream()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
at::Tensor pair_layout(at::Tensor a,at::Tensor b,int block,bool transpose){
  c10::cuda::CUDAGuard guard(a.device());check(a);check(b);
  TORCH_CHECK(a.dim()==2&&a.sizes()==b.sizes()&&a.is_contiguous()&&b.is_contiguous()
              &&a.scalar_type()==at::kBFloat16&&b.scalar_type()==at::kBFloat16,"packed BF16 projections required");
  int h=a.size(0),d=a.size(1);TORCH_CHECK(block==0||(block>0&&h%block==0),"weight interleave block");
  auto out=at::empty(transpose?std::vector<int64_t>{d,2*h}:std::vector<int64_t>{2*h,d},a.options());
  const bf* ap=reinterpret_cast<const bf*>(a.data_ptr());
  const bf* bp=reinterpret_cast<const bf*>(b.data_ptr());
  bf* op=reinterpret_cast<bf*>(out.data_ptr());
  cudaKernel_t kernel;C10_CUDA_CHECK(cudaGetKernel(&kernel,pair_layout_k));
  void* args[]={&ap,&bp,&op,&h,&d,&block,&transpose};
  C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel),
    dim3(((2*h+31)/32)*((d+31)/32)),dim3(256),args,0,at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<at::Tensor> cast(std::vector<at::Tensor> in,std::vector<bool> fp32,bool force){
  TORCH_CHECK(!in.empty()&&in.size()<=8&&in.size()==fp32.size(),"one to eight cast tensors required");
  c10::cuda::CUDAGuard guard(in[0].device());Cast p{};std::vector<at::Tensor> out;int maxn=0;
  for(int j=0;j<(int)in.size();++j){
    check(in[j]);TORCH_CHECK(in[j].is_contiguous()&&in[j].device()==in[0].device(),"contiguous tensors on one device required");
    auto dt=fp32[j]?at::kFloat:at::kBFloat16;bool copy=force||in[j].scalar_type()!=dt;
    out.push_back(copy?at::empty(in[j].sizes(),in[j].options().dtype(dt)):in[j]);
    p.src[j]=in[j].data_ptr();p.dst[j]=out[j].data_ptr();p.n[j]=copy?in[j].numel():0;
    p.sf[j]=in[j].scalar_type()==at::kFloat;p.df[j]=fp32[j];maxn=std::max(maxn,p.n[j]);
  }
  if(maxn)cast_k<<<dim3((maxn+255)/256,in.size()),256,0,at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<at::Tensor> ln_bwd(at::Tensor dy,at::Tensor x,at::Tensor gamma,at::Tensor rs,at::Tensor c1,at::Tensor residual,int waves){
  c10::cuda::CUDAGuard guard(x.device());
  for(auto* t:{&dy,&x,&residual})TORCH_CHECK(t->is_cuda()&&t->is_contiguous()&&t->scalar_type()==at::kBFloat16&&t->sizes()==x.sizes(),"LN BF16 rows");
  int d=x.size(-1),rows=x.numel()/d;
  TORCH_CHECK((d==256||d==384||d==512||d==768)&&waves>0,"wide LN contract");
  for(auto* t:{&gamma,&rs,&c1})TORCH_CHECK(t->is_cuda()&&t->is_contiguous()&&t->scalar_type()==at::kFloat,"LN FP32 stats/affine");
  TORCH_CHECK(gamma.numel()==d&&rs.numel()==rows&&c1.numel()==rows,"LN stats shapes");
  int sms=0;C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,x.get_device()));
  int grid=sms*waves,nw=grid*4;
  auto dx=at::empty_like(x),dg=at::empty_like(gamma),db=at::empty_like(gamma),partial=at::empty({2,nw,d},gamma.options());
  auto stream=at::cuda::getCurrentCUDAStream();
  auto launch_bwd=[&](auto kernel){kernel<<<grid,128,0,stream>>>(reinterpret_cast<const bf*>(dy.data_ptr()),reinterpret_cast<const bf*>(x.data_ptr()),
    gamma.data_ptr<float>(),rs.data_ptr<float>(),c1.data_ptr<float>(),reinterpret_cast<const bf*>(residual.data_ptr()),
    reinterpret_cast<bf*>(dx.data_ptr()),partial.data_ptr<float>(),rows,d);};
  if(d>512)launch_bwd(ln_bwd_k<6>);else launch_bwd(ln_bwd_k<4>);
  ln_reduce_k<<<d,256,0,stream>>>(partial.data_ptr<float>(),dg.data_ptr<float>(),db.data_ptr<float>(),nw,d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dg,db};
}
std::vector<at::Tensor> ln_bwd_compact(at::Tensor dy,at::Tensor x,at::Tensor gamma,at::Tensor rs,at::Tensor c1,at::Tensor residual,int waves){
  c10::cuda::CUDAGuard guard(x.device());
  for(auto* t:{&dy,&x,&residual})TORCH_CHECK(t->device()==x.device()&&t->is_cuda()&&t->is_contiguous()&&t->scalar_type()==at::kBFloat16&&t->sizes()==x.sizes(),"LN BF16 rows");
  int d=x.size(-1),rows=x.numel()/d;
  TORCH_CHECK((d==512||d==384)&&waves>0,"compact D384/D512 LN contract");
  for(auto* t:{&gamma,&rs,&c1})TORCH_CHECK(t->device()==x.device()&&t->is_contiguous()&&t->scalar_type()==at::kFloat,"LN FP32 stats/affine");
  TORCH_CHECK(gamma.numel()==d&&rs.numel()==rows&&c1.numel()==rows,"LN stats shapes");
  int sms=0;C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,x.get_device()));
  int grid=sms*waves,nw=grid*4;
  auto dx=at::empty_like(x),dg=at::empty_like(gamma),db=at::empty_like(gamma),partial=at::empty({2,nw,d},gamma.options());
  auto stream=at::cuda::getCurrentCUDAStream();
  const bf *yp=reinterpret_cast<const bf*>(dy.data_ptr()),*xp=reinterpret_cast<const bf*>(x.data_ptr()),*rp=reinterpret_cast<const bf*>(residual.data_ptr());
  const float *gp=gamma.data_ptr<float>(),*rsp=rs.data_ptr<float>(),*cp=c1.data_ptr<float>();
  bf* dxp=reinterpret_cast<bf*>(dx.data_ptr());float* pp=partial.data_ptr<float>();
  cudaKernel_t kernel;
  if(d==384)C10_CUDA_CHECK(cudaGetKernel(&kernel,ln_bwd_compact384_k));
  else C10_CUDA_CHECK(cudaGetKernel(&kernel,ln_bwd_compact512_k));
  void* args[]={&yp,&xp,&gp,&rsp,&cp,&rp,&dxp,&pp,&rows};
  C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel),dim3(grid),dim3(128),args,0,stream));
  ln_reduce_k<<<d,256,0,stream>>>(pp,dg.data_ptr<float>(),db.data_ptr<float>(),nw,d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dg,db};
}
__global__ void affine_reduce_k(const float* pg,const float* pb,float* dg,float* db,int rows,int d){
  __shared__ float sg[256],sb[256];
  int lane=threadIdx.x&31,warp=threadIdx.x>>5,c=blockIdx.x*32+lane;
  float g=0.f,b=0.f;
  if(c<d)for(int r=warp;r<rows;r+=8){g=__fadd_rn(g,pg[r*d+c]);b=__fadd_rn(b,pb[r*d+c]);}
  sg[threadIdx.x]=g;sb[threadIdx.x]=b;__syncthreads();
  if(warp==0&&c<d){
    g=0.f;b=0.f;
    #pragma unroll
    for(int r=0;r<8;++r){g=__fadd_rn(g,sg[r*32+lane]);b=__fadd_rn(b,sb[r*32+lane]);}
    dg[c]=g;db[c]=b;
  }
}
std::vector<at::Tensor> reduce_affine(at::Tensor pg,at::Tensor pb){
  c10::cuda::CUDAGuard guard(pg.device());
  TORCH_CHECK(pg.is_cuda()&&pb.device()==pg.device()&&pg.dim()==2&&pb.sizes()==pg.sizes()
    &&pg.is_contiguous()&&pb.is_contiguous()&&pg.scalar_type()==at::kFloat&&pb.scalar_type()==at::kFloat,"FP32 affine partials required");
  auto dg=at::empty({pg.size(1)},pg.options()),db=at::empty_like(dg);
  affine_reduce_k<<<(pg.size(1)+31)/32,256,0,at::cuda::getCurrentCUDAStream()>>>(
    pg.data_ptr<float>(),pb.data_ptr<float>(),dg.data_ptr<float>(),db.data_ptr<float>(),pg.size(0),pg.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dg,db};
}
template<int NV> __global__ void ln_fwd_k(const bf* x,const float* gamma,const float* beta,bf* xn,float* rs,float* c1,int rows,int d,float eps){
  int lane=threadIdx.x&31,row=blockIdx.x*8+(threadIdx.x>>5);
  if(row>=rows)return;
  float v[4*NV],sum=0.f,var=0.f;
  union Vec{uint2 u;bf v[4];};
  #pragma unroll
  for(int j=0;j<NV;++j){
    int c=(j*32+lane)*4;
    Vec q;q.u=*reinterpret_cast<const uint2*>(x+row*d+c);
    #pragma unroll
    for(int k=0;k<4;++k){v[j*4+k]=__bfloat162float(q.v[k]);sum=__fadd_rn(sum,v[j*4+k]);}
  }
  float mean=wsum(sum)/d;
  #pragma unroll
  for(int k=0;k<4*NV;++k){float q=v[k]-mean;var=__fadd_rn(var,q*q);}
  float r=rsqrtf(wsum(var)/d+eps);
  #pragma unroll
  for(int j=0;j<NV;++j){
    int c=(j*32+lane)*4;Vec q;
    #pragma unroll
    for(int k=0;k<4;++k)q.v[k]=__float2bfloat16_rn((v[j*4+k]-mean)*r*gamma[c+k]+beta[c+k]);
    *reinterpret_cast<uint2*>(xn+row*d+c)=q.u;
  }
  if(!lane){rs[row]=r;c1[row]=mean*r;}
}
std::vector<at::Tensor> ln_fwd(at::Tensor x,at::Tensor gamma,at::Tensor beta,double eps){
  c10::cuda::CUDAGuard guard(x.device());
  const int d=x.size(1);
  TORCH_CHECK(x.is_cuda()&&x.dim()==2&&(d==512||d==768)&&x.is_contiguous()&&x.scalar_type()==at::kBFloat16,"D512 / D768 BF16 rows required");
  for(auto* t:{&gamma,&beta})TORCH_CHECK(t->device()==x.device()&&t->is_contiguous()&&t->scalar_type()==at::kFloat&&t->numel()==d,"FP32 affine required");
  auto xn=at::empty_like(x),rs=at::empty({x.size(0)},gamma.options()),c1=at::empty_like(rs);
  auto args_x=reinterpret_cast<const bf*>(x.data_ptr());
  if(d==768)ln_fwd_k<6><<<(x.size(0)+7)/8,256,0,at::cuda::getCurrentCUDAStream()>>>(args_x,
    gamma.data_ptr<float>(),beta.data_ptr<float>(),reinterpret_cast<bf*>(xn.data_ptr()),rs.data_ptr<float>(),c1.data_ptr<float>(),x.size(0),d,eps);
  else ln_fwd_k<4><<<(x.size(0)+7)/8,256,0,at::cuda::getCurrentCUDAStream()>>>(args_x,
    gamma.data_ptr<float>(),beta.data_ptr<float>(),reinterpret_cast<bf*>(xn.data_ptr()),rs.data_ptr<float>(),c1.data_ptr<float>(),x.size(0),d,eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {xn,rs,c1};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("ln_bwd_compact",&ln_bwd_compact);
  m.def("ln_fwd",&ln_fwd);
  m.def("reduce_affine",&reduce_affine);
  m.def("prepare",&prepare);m.def("pair_layout",&pair_layout);m.def("cast",&cast);m.def("ln_bwd",&ln_bwd);
}
#else
// Torch bindings for the fused sm_90a Transition forward and backward.
//
// The two kernels live in `transition_fused_fwd_sm90a_kernel.cu` and
// `transition_fused_bwd_sm90a_kernel.cu`; this file is only the host side: it turns torch
// tensors into the 2-D TMA descriptors they expect, allocates the outputs and the backward's
// partial buffers, and calls the launchers on torch's current stream.
//
// Two things here are not obvious and were found the hard way:
//
//   * `cuTensorMapEncodeTiled` is a driver entry point. Resolving it through
//     `cudaGetDriverEntryPoint` keeps this extension off `-lcuda`, which torch's JIT build does
//     not link by default.
//   * A descriptor is 128 B of opaque state that has to be 64-B aligned and is baked into the
//     kernel's parameter buffer at launch, so it is valid for exactly one base pointer. It is
//     cached on (pointer, shape, box) and rebuilt whenever a tensor moves -- which is what makes
//     the weight descriptors free after the first step, while activations re-encode each call.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <map>
#include <mutex>
#include <stdexcept>

// ---- the launchers, from the two kernel translation units ----------------------------------
int transition_fused_fwd_ctas();
#ifndef SUBCH
#define SUBCH 0
#endif
int transition_fused_fwd_rows();
void transition_fused_trace(unsigned long long* host);
bool transition_fused_fwd_saves_xn();
void transition_fused_fwd_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&,
                                 const CUtensorMap&, const CUtensorMap&, const float*, const float*,
                                 __nv_bfloat16*, __nv_bfloat16*, float*, float*, int, int, float,
                                 const __nv_bfloat16*, cudaStream_t);
int transition_fused_bwd_ctas();
int transition_fused_bwd_ndw();
int transition_fused_bwd_ndx();
int transition_fused_bwd_rows();
void transition_fused_bwd_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&,
                                 const CUtensorMap&, const CUtensorMap&, const CUtensorMap&,
                                 const float*, const float*, const float*, __nv_bfloat16*, float*,
                                 float*, float*, float*, int, int, cudaStream_t);
void transition_fused_reduce_launch(const float*, void*, void*, void*,
                                    const float*, float*, float*, bool, cudaStream_t);

#ifndef MW_N
#define MW_N 4
#endif

namespace {

using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                 const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                 const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    if (e != cudaSuccess || p == nullptr)
      throw std::runtime_error("cuTensorMapEncodeTiled unavailable: the driver predates TMA");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}

using Key = std::array<uint64_t, 5>;

// 2-D bf16 descriptor, dims innermost-first, 128-B swizzle -- the layout every tile load in
// both kernels assumes.
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t d_inner, uint32_t d_outer,
                            uint32_t box_inner, uint32_t box_outer) {
  static std::map<Key, CUtensorMap> cache;
  static std::mutex lock;
  const Key key{reinterpret_cast<uint64_t>(t.data_ptr()), d_inner, d_outer, box_inner, box_outer};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  CUtensorMap map{};
  const cuuint64_t dims[2] = {d_inner, d_outer};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(d_inner) * 2};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t elem[2] = {1, 1};
  // cuTensorMapEncodeTiled is a driver call and needs a current context. The autograd engine runs backward on its own thread, and when the
  // caching allocator serves every tensor from its pool no runtime call has made the primary context current there: CUDA_ERROR_INVALID_CONTEXT.
  // (cudaSetDevice binds it; cudaFree(0) would do too but is an unsafe call inside a CUDA-graph capture.)
  if (cudaSetDevice(t.get_device()) != cudaSuccess) throw std::runtime_error("cudaSetDevice failed before cuTensorMapEncodeTiled");
  CUresult r = encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides,
                         box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled failed");
  return cache.emplace(key, map).first->second;
}

void expect(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("transition fused sm90a: ") + what);
}

__nv_bfloat16* bf16_ptr(const torch::Tensor& t) {
  return reinterpret_cast<__nv_bfloat16*>(t.data_ptr());
}

}  // namespace

// x [M,128] bf16, gamma/beta [128] fp32, wa/wb [512,128] bf16, ws [128,512] bf16,
// wst = ws^T [512,128] bf16 contiguous. Returns (out, xn, rstd, c1); out = transition(x) + x.
// `save` must match the build: the FWD_SAVE=0 variant is the inference one and writes only
// `out`, so xn/rstd/c1 come back as placeholders.
std::vector<torch::Tensor> transition_fused_fwd(torch::Tensor x, torch::Tensor gamma,
                                                torch::Tensor beta, torch::Tensor wa,
                                                torch::Tensor wb, torch::Tensor wst, double eps,
                                                bool save) {
  const int64_t M = x.size(0), D = x.size(1), H = wa.size(0);
  expect(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "x must be contiguous cuda bf16");
  expect(D == 128 && H == MW_N * 128, "built for d_hidden 128 with the hidden multiple MW_N only");
  expect(M % transition_fused_fwd_rows() == 0, "row count must be a whole tile");
  expect(gamma.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32, "gamma/beta must be fp32");
  expect(wa.is_contiguous() && wb.is_contiguous() && wst.is_contiguous(), "weights must be contiguous");
  expect(wst.size(0) == H && wst.size(1) == D, "wst must be ws transposed, [H, D]");

  // Whether the forward writes what the backward needs is compiled in, not passed: the stores
  // sit inside the LayerNorm epilogue. `save` selects the build, and this catches the two
  // getting out of step -- which would otherwise be a kernel writing through a 1-element
  // placeholder.
  expect(save == transition_fused_fwd_saves_xn(), "save flag does not match this build");

  auto out = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  const int64_t saved = save ? M : 1;
  auto xn = save ? torch::empty_like(x) : torch::empty({1, D}, x.options());
  auto rstd = torch::empty({saved}, f32);
  auto c1 = torch::empty({saved}, f32);

  transition_fused_fwd_launch(
      tile_map(x, D, M, 64, 64), tile_map(wa, D, H, 64, SUBCH ? 32 : 64), tile_map(wb, D, H, 64, SUBCH ? 32 : 64),
      tile_map(wst, D, H, 64, 64), tile_map(out, D, M, 64, 64),
      gamma.data_ptr<float>(), beta.data_ptr<float>(), bf16_ptr(xn), bf16_ptr(out),
      rstd.data_ptr<float>(), c1.data_ptr<float>(), static_cast<int>(M),
      static_cast<int>(M / transition_fused_fwd_rows()), static_cast<float>(eps),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), at::cuda::getCurrentCUDAStream());
  return {out, xn, rstd, c1};
}

// dy [M,128] bf16 (gradient of the module output), plus what the forward saved.
// Returns (dx, dgamma, dbeta, dWa, dWb, dWs); dx already carries the residual branch.
std::vector<torch::Tensor> transition_fused_bwd_master(torch::Tensor dy, torch::Tensor x,
                                                torch::Tensor xn, torch::Tensor rstd,
                                                torch::Tensor c1, torch::Tensor gamma,
                                                torch::Tensor wa, torch::Tensor wb,
                                                torch::Tensor ws, bool fp32_dw) {
  const int64_t M = x.size(0), D = x.size(1), H = wa.size(0);
  expect(dy.is_cuda() && dy.is_contiguous() && dy.scalar_type() == torch::kBFloat16, "dy must be contiguous cuda bf16");
  expect(D == 128 && H == MW_N * 128, "built for d_hidden 128 with the hidden multiple MW_N only");
  expect(M % transition_fused_bwd_rows() == 0, "row count must be a whole tile");
  expect(gamma.scalar_type() == torch::kFloat32, "gamma must be fp32");
  expect(ws.is_contiguous() && ws.size(0) == D && ws.size(1) == H, "ws must be contiguous [D, H]");

  auto f32 = x.options().dtype(torch::kFloat32);
  auto dx = torch::empty_like(x);
  auto dgam = torch::zeros({D}, f32);
  auto dbeta = torch::zeros({D}, f32);
  auto partw = torch::empty({transition_fused_bwd_ndw() * 3 * 64 * D}, f32);
  auto dgbw = torch::empty({transition_fused_bwd_ndx() * 8 * 256}, f32);
  auto dWa = torch::empty(wa.sizes(), fp32_dw ? f32 : wa.options());
  auto dWb = torch::empty(wb.sizes(), fp32_dw ? f32 : wb.options());
  auto dWs = torch::empty(ws.sizes(), fp32_dw ? f32 : ws.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  transition_fused_bwd_launch(
      tile_map(dy, D, M, 64, 64), tile_map(xn, D, M, 64, 64), tile_map(x, D, M, 64, 64),
      tile_map(ws, H, D, 64, 128), tile_map(wa, D, H, 64, 64), tile_map(wb, D, H, 64, 64),
      rstd.data_ptr<float>(), c1.data_ptr<float>(), gamma.data_ptr<float>(), bf16_ptr(dx),
      dgam.data_ptr<float>(), dbeta.data_ptr<float>(), partw.data_ptr<float>(),
      dgbw.data_ptr<float>(), static_cast<int>(M),
      static_cast<int>(M / transition_fused_bwd_rows()), stream);
  transition_fused_reduce_launch(partw.data_ptr<float>(), dWa.data_ptr(), dWb.data_ptr(),
                                 dWs.data_ptr(), dgbw.data_ptr<float>(), dgam.data_ptr<float>(),
                                 dbeta.data_ptr<float>(), fp32_dw, stream);
  return {dx, dgam, dbeta, dWa, dWb, dWs};
}


// Returns (dx, dgamma, dbeta, dWa, dWb, dWs); dx already carries the residual branch.
std::vector<torch::Tensor> transition_fused_bwd_native(torch::Tensor dy, torch::Tensor x,
                                                torch::Tensor xn, torch::Tensor rstd,
                                                torch::Tensor c1, torch::Tensor gamma,
                                                torch::Tensor wa, torch::Tensor wb,
                                                torch::Tensor ws, bool fp32_dw) {
  // Autograd can call us on a fresh worker thread. Empty allocations do not
  // necessarily attach its primary context before the driver TMA encoder.
  expect(cudaSetDevice(x.get_device()) == cudaSuccess, "CUDA context initialization failed");
  const int64_t M = x.size(0), D = x.size(1), H = wa.size(0);
  expect(dy.is_cuda() && dy.is_contiguous() && dy.scalar_type() == torch::kBFloat16, "dy must be contiguous cuda bf16");
  expect(D == 128 && H == MW_N * 128, "built for d_hidden 128 with the hidden multiple MW_N only");
  expect(M % transition_fused_bwd_rows() == 0, "row count must be a whole tile");
  expect(gamma.scalar_type() == torch::kFloat32, "gamma must be fp32");
  expect(ws.is_contiguous() && ws.size(0) == D && ws.size(1) == H, "ws must be contiguous [D, H]");

  auto f32 = x.options().dtype(torch::kFloat32);
  auto dx = torch::empty_like(x);
  auto dgam = torch::empty({D}, f32);
  auto dbeta = torch::empty({D}, f32);
  auto partw = torch::empty({transition_fused_bwd_ndw() * 3 * 64 * D}, f32);
  auto dgbw = torch::empty({transition_fused_bwd_ndx() * 8 * 256}, f32);
  auto dWa = torch::empty(wa.sizes(), fp32_dw ? f32 : wa.options());
  auto dWb = torch::empty(wb.sizes(), fp32_dw ? f32 : wb.options());
  auto dWs = torch::empty(ws.sizes(), fp32_dw ? f32 : ws.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  transition_fused_bwd_launch(
      tile_map(dy, D, M, 64, 64), tile_map(xn, D, M, 64, 64), tile_map(x, D, M, 64, 64),
      tile_map(ws, H, D, 64, 128), tile_map(wa, D, H, 64, 64), tile_map(wb, D, H, 64, 64),
      rstd.data_ptr<float>(), c1.data_ptr<float>(), gamma.data_ptr<float>(), bf16_ptr(dx),
      dgam.data_ptr<float>(), dbeta.data_ptr<float>(), partw.data_ptr<float>(),
      dgbw.data_ptr<float>(), static_cast<int>(M),
      static_cast<int>(M / transition_fused_bwd_rows()), stream);
  transition_fused_reduce_launch(partw.data_ptr<float>(), dWa.data_ptr(), dWb.data_ptr(),
                                 dWs.data_ptr(), dgbw.data_ptr<float>(), dgam.data_ptr<float>(),
                                 dbeta.data_ptr<float>(), fp32_dw, stream);
  return {dx, dgam, dbeta, dWa, dWb, dWs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transition_fused_bwd_native", &transition_fused_bwd_native);
  m.def("transition_fused_fwd", &transition_fused_fwd,
        "fused sm_90a Transition forward (LN + SwiGLU expand + squeeze + residual)");
  m.def("transition_fused_bwd_master", &transition_fused_bwd_master);
  m.def("transition_fused_bwd", [](torch::Tensor dy, torch::Tensor x, torch::Tensor xn, torch::Tensor rs, torch::Tensor c1, torch::Tensor g, torch::Tensor wa, torch::Tensor wb, torch::Tensor ws) { return transition_fused_bwd_master(dy,x,xn,rs,c1,g,wa,wb,ws,false); },
        "fused sm_90a Transition backward (one kernel + a partial reduction)");
  m.def("rows_per_tile", &transition_fused_fwd_rows);
  m.def("trace", [] { std::vector<int64_t> v(64); std::vector<unsigned long long> h(64); transition_fused_trace(h.data()); for (int i = 0; i < 64; ++i) v[i] = (int64_t)h[i]; return v; });
  m.def("fwd_ctas", &transition_fused_fwd_ctas);
  m.def("bwd_ctas", &transition_fused_bwd_ctas);
  m.def("saves_xn", &transition_fused_fwd_saves_xn);
}
#endif  // MW_TRANSITION_AUX_ONLY
