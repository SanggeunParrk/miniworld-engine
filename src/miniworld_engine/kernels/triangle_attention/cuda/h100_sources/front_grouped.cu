// Hopper grouped-cuBLAS frontend alternative: CUDA LN, packed projection GEMM,
// CUDA masked bias layout. Q/K/V/G are views of one projection allocation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
__device__ float sum32(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__global__ void norm_rows(__nv_bfloat16 const* x,float const* w,float const* b,__nv_bfloat16* z,float* stats,int rows,int C,float eps){
  int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
  if(row>=rows)return;
  float val[16],mean=0.f,var=0.f;
  for(int k=0;k<C/32;++k){val[k]=__bfloat162float(x[row*C+k*32+lane]);mean+=val[k];}
  mean=sum32(mean)/C;
  for(int k=0;k<C/32;++k){float d=val[k]-mean;var+=d*d;}
  float inv=rsqrtf(sum32(var)/C+eps);
  if(lane==0){stats[row*2]=mean;stats[row*2+1]=inv;}
  for(int k=0;k<C/32;++k){int c=k*32+lane;z[row*C+c]=__float2bfloat16((val[k]-mean)*inv*w[c]+b[c]);}
}
std::vector<torch::Tensor> h100_norm(torch::Tensor x,torch::Tensor w,torch::Tensor b,double eps){
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.is_contiguous() && x.dim()==2,"BF16 x required");
  c10::cuda::CUDAGuard guard(x.device());int C=x.size(1),rows=x.size(0);
  TORCH_CHECK((C==64 || C==128 || C==256 || C==384 || C==512) && rows>0,"invalid matrix shape");
  for(auto t:{w,b})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous() && t.numel()==C,"FP32 LN affines required");
  auto z=torch::empty_like(x),s=torch::empty({rows,2},w.options());
  norm_rows<<<(rows+7)/8,256,0,at::cuda::getCurrentCUDAStream()>>>((__nv_bfloat16*)x.data_ptr(),w.data_ptr<float>(),b.data_ptr<float>(),(__nv_bfloat16*)z.data_ptr(),s.data_ptr<float>(),rows,C,float(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {z,s};
}
__global__ void bias_layout(__nv_bfloat16 const* p,bool const* mask,__nv_bfloat16* bias,int rows,int C,int L,int H){
  int idx=blockIdx.x*256+threadIdx.x;if(idx>=rows*H)return;
  int row=idx/H,h=idx%H,b=row/(L*L),q=(row/L)%L,k=row%L;
  bias[((b*H+h)*L+q)*L+k]=mask && !mask[b*L+k]?__float2bfloat16(-3.3895313892515355e38f):p[row*(4*C+64)+4*C+h];
}
torch::Tensor h100_bias(torch::Tensor p,torch::Tensor mask,int C,int L){
  TORCH_CHECK(p.is_cuda() && p.scalar_type()==torch::kBFloat16 && p.is_contiguous() && p.dim()==2 && p.size(1)==4*C+64,"packed BF16 projection required");
  c10::cuda::CUDAGuard guard(p.device());int rows=p.size(0),B=rows/(L*L),H=C/32;
  TORCH_CHECK(L>0 && L%128==0 && rows==B*L*L,"invalid pair length");
  TORCH_CHECK(mask.device()==p.device() && mask.scalar_type()==torch::kBool && mask.is_contiguous() && (mask.numel()==0 || mask.sizes()==torch::IntArrayRef({B,L})),"mask mismatch");
  auto out=torch::empty({B,H,L,L},p.options());
  bias_layout<<<(rows*H+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>((__nv_bfloat16*)p.data_ptr(),mask.numel()?mask.data_ptr<bool>():nullptr,(__nv_bfloat16*)out.data_ptr(),rows,C,L,H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

__global__ void gate_product(__nv_bfloat16 const* g,__nv_bfloat16 const* o,__nv_bfloat16* u,int rows,int C,int stride){
  int i=blockIdx.x*256+threadIdx.x;if(i>=rows*C)return;
  float gate=__bfloat162float(g[(i/C)*stride+i%C]);
  u[i]=__float2bfloat16(__bfloat162float(o[i])/(1.f+__expf(-gate)));
}
torch::Tensor h100_gate(torch::Tensor g,torch::Tensor o){
  TORCH_CHECK(o.is_cuda() && o.scalar_type()==torch::kBFloat16 && o.dim()==2 && o.is_contiguous(),"contiguous BF16 attention required");
  TORCH_CHECK(g.device()==o.device() && g.scalar_type()==o.scalar_type() && g.sizes()==o.sizes() && g.stride(1)==1,"gate mismatch");
  c10::cuda::CUDAGuard guard(o.device());auto u=torch::empty_like(o);int rows=o.size(0),C=o.size(1);
  gate_product<<<(rows*C+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>((__nv_bfloat16*)g.data_ptr(),(__nv_bfloat16*)o.data_ptr(),(__nv_bfloat16*)u.data_ptr(),rows,C,g.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return u;
}
__global__ void residual_rows(__nv_bfloat16 const* x,__nv_bfloat16 const* p,__nv_bfloat16 const* drop,__nv_bfloat16* y,int rows,int C,int L){
  int i=blockIdx.x*256+threadIdx.x;if(i>=rows*C)return;int row=i/C,c=i%C;
  float v=__bfloat162float(p[i]);
  if(drop)v=__bfloat162float(__float2bfloat16(v*__bfloat162float(drop[((row/(L*L))*L+row%L)*C+c])));
  y[i]=__float2bfloat16(__bfloat162float(x[i])+v);
}
torch::Tensor h100_residual(torch::Tensor x,torch::Tensor p,torch::Tensor drop,int L){
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.dim()==2 && x.is_contiguous(),"BF16 residual required");
  TORCH_CHECK(p.device()==x.device() && p.scalar_type()==x.scalar_type() && p.sizes()==x.sizes() && p.is_contiguous(),"projection mismatch");
  int rows=x.size(0),C=x.size(1);TORCH_CHECK(L>0 && rows%(L*L)==0,"invalid pair length");
  TORCH_CHECK(drop.device()==x.device() && drop.scalar_type()==x.scalar_type() && drop.is_contiguous() && (drop.numel()==0 || drop.numel()==rows/L*C),"dropout mismatch");
  c10::cuda::CUDAGuard guard(x.device());auto y=torch::empty_like(x);
  residual_rows<<<(rows*C+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>((__nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)p.data_ptr(),drop.numel()?(__nv_bfloat16*)drop.data_ptr():nullptr,(__nv_bfloat16*)y.data_ptr(),rows,C,L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

// Copy projection gradients a vector at a time. Compile-time widths remove
// divisions from the large Dall packing pass; only the narrow bias needs gather.
template<int C> __global__ void pack_grad_vec(__nv_bfloat16 const* dq,__nv_bfloat16 const* dk,__nv_bfloat16 const* dv,__nv_bfloat16 const* dg,__nv_bfloat16 const* db,bool const* mask,uint4* out,int rows,int L){
  constexpr int H=C/32,V=(4*C+64)/8;
  int idx=blockIdx.x*256+threadIdx.x;if(idx>=rows*V)return;int row=idx/V,col=idx%V;
  uint4 value=make_uint4(0,0,0,0);
  if(col<4*C/8){int part=col/(C/8),c=col%(C/8);auto src=part==0?dq:part==1?dk:part==2?dv:dg;value=reinterpret_cast<uint4 const*>(src)[row*(C/8)+c];}
  else{
    int b=row/(L*L),q=(row/L)%L,k=row%L,h0=(col-4*C/8)*8;
    if(!mask || mask[b*L+k]){
      auto val=reinterpret_cast<__nv_bfloat16*>(&value);
      #pragma unroll
      for(int j=0;j<8;++j)if(h0+j<H)val[j]=db[((b*H+h0+j)*L+q)*L+k];
    }
  }
  out[idx]=value;
}
torch::Tensor h100_pack_grad(torch::Tensor dq,torch::Tensor dk,torch::Tensor dv,torch::Tensor dg,torch::Tensor db,torch::Tensor mask,int L){
  TORCH_CHECK(dq.is_cuda() && dq.scalar_type()==torch::kBFloat16 && dq.dim()==2 && dq.is_contiguous(),"BF16 projection gradients required");
  c10::cuda::CUDAGuard guard(dq.device());int rows=dq.size(0),C=dq.size(1),H=C/32;
  TORCH_CHECK(L>0 && rows%(L*L)==0,"invalid pair length");
  for(auto t:{dk,dv,dg})TORCH_CHECK(t.device()==dq.device() && t.scalar_type()==dq.scalar_type() && t.sizes()==dq.sizes() && t.is_contiguous(),"gradient mismatch");
  TORCH_CHECK(db.device()==dq.device() && db.scalar_type()==dq.scalar_type() && db.is_contiguous() && db.sizes()==torch::IntArrayRef({rows/(L*L),H,L,L}),"bias gradient mismatch");
  TORCH_CHECK(mask.device()==dq.device() && mask.scalar_type()==torch::kBool && mask.is_contiguous() && (mask.numel()==0 || mask.numel()==rows/L),"mask mismatch");
  auto out=torch::empty({rows,4*C+64},dq.options());auto stream=at::cuda::getCurrentCUDAStream();int grid=(out.numel()/8+255)/256;
#define PACK(C_) case C_:pack_grad_vec<C_><<<grid,256,0,stream>>>((__nv_bfloat16*)dq.data_ptr(),(__nv_bfloat16*)dk.data_ptr(),(__nv_bfloat16*)dv.data_ptr(),(__nv_bfloat16*)dg.data_ptr(),(__nv_bfloat16*)db.data_ptr(),mask.numel()?mask.data_ptr<bool>():nullptr,(uint4*)out.data_ptr(),rows,L);break;
  switch(C){PACK(64) PACK(128) PACK(256) PACK(384) PACK(512) default:TORCH_CHECK(false,"unsupported width");}
#undef PACK
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

template<int C> __global__ void gate_bwd_rows(__nv_bfloat16 const* du,__nv_bfloat16 const* gate,__nv_bfloat16 const* o,__nv_bfloat16* dout,__nv_bfloat16* dg,__nv_bfloat16* u,float* delta,int rows,int L,int stride){
  constexpr int H=C/32;int wh=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
  if(wh>=rows*H)return;int row=wh/H,h=wh%H,col=h*32+lane,idx=row*C+col;
  float sig=1.f/(1.f+__expf(-__bfloat162float(gate[row*stride+col]))),grad=__bfloat162float(du[idx]),val=__bfloat162float(o[idx]);
  auto d=__float2bfloat16(grad*sig);dout[idx]=d;
  dg[idx]=__float2bfloat16(grad*val*sig*(1.f-sig));u[idx]=__float2bfloat16(val*sig);
  float sum=sum32(__bfloat162float(d)*val);
  if(lane==0)delta[(((row/(L*L))*H+h)*L+(row/L)%L)*L+row%L]=sum;
}
std::vector<torch::Tensor> h100_gate_bwd(torch::Tensor du,torch::Tensor gate,torch::Tensor o,int L){
  TORCH_CHECK(du.is_cuda() && du.scalar_type()==torch::kBFloat16 && du.dim()==2 && du.is_contiguous(),"BF16 gate gradient required");
  c10::cuda::CUDAGuard guard(du.device());int rows=du.size(0),C=du.size(1),H=C/32;
  TORCH_CHECK(L>0 && rows%(L*L)==0,"invalid pair length");
  for(auto t:{gate,o})TORCH_CHECK(t.device()==du.device() && t.scalar_type()==du.scalar_type() && t.sizes()==du.sizes() && t.stride(1)==1,"gate input mismatch");
  TORCH_CHECK(o.is_contiguous(),"contiguous attention required");
  auto dout=torch::empty_like(du),dg=torch::empty_like(du),u=torch::empty_like(du),delta=torch::empty({rows/(L*L),H,L,L},du.options().dtype(torch::kFloat32));
  auto stream=at::cuda::getCurrentCUDAStream();int grid=(rows*H+7)/8;
#define GATE(C_) case C_:gate_bwd_rows<C_><<<grid,256,0,stream>>>((__nv_bfloat16*)du.data_ptr(),(__nv_bfloat16*)gate.data_ptr(),(__nv_bfloat16*)o.data_ptr(),(__nv_bfloat16*)dout.data_ptr(),(__nv_bfloat16*)dg.data_ptr(),(__nv_bfloat16*)u.data_ptr(),delta.data_ptr<float>(),rows,L,gate.stride(0));break;
  switch(C){GATE(64) GATE(128) GATE(256) GATE(384) GATE(512) default:TORCH_CHECK(false,"unsupported width");}
#undef GATE
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dout,dg,u,delta};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gate_backward",&h100_gate_bwd);m.def("pack_backward",&h100_pack_grad);m.def("norm",&h100_norm);m.def("bias",&h100_bias);m.def("gate",&h100_gate);m.def("residual",&h100_residual);}

