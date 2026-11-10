// Token DiT without QK norm: natural-bias backward directly into BF16 q|k|v columns.
// The existing plain BF16 core uses raw bias; this separate build keeps token natural units.
#define AA_NO_RAWB
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "attn_bwd_dkv_sm80.cuh"
namespace {
using bf=__nv_bfloat16;
template<class G>
void dq_launch(const aa80::DqParams& p,int A) {
  static bool configured[64]={};int dev=at::cuda::current_device()&63;
  if(!configured[dev]){C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::attn_bwd_dq_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));configured[dev]=true;}
  aa80::attn_bwd_dq_kernel<G><<<dim3(p.L/128,A/G::R,p.H),256,G::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);
}
template<class G>
void dkv_launch(const aa80::DkvParams& p,int A) {
  static bool configured[64]={};int dev=at::cuda::current_device()&63;
  if(!configured[dev]){C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::attn_bwd_dkv_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));configured[dev]=true;}
  aa80::attn_bwd_dkv_kernel<G><<<dim3(p.L/128,A,p.H),256,G::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);
}
__global__ void token_db_bf16_kernel(const bf* partial,bf* output,int groups,long n8) {
  const long i=(long)blockIdx.x*256+threadIdx.x;if(i>=n8)return;
  float value[8]={};
  for(int g=0;g<groups;++g){const uint4 v=*reinterpret_cast<const uint4*>(partial+((long)g*n8+i)*8);uint32_t words[4]={v.x,v.y,v.z,v.w};
#pragma unroll
    for(int k=0;k<4;++k){value[2*k]+=aa80::bf16lo(words[k]);value[2*k+1]+=aa80::bf16hi(words[k]);}}
  *reinterpret_cast<uint4*>(output+i*8)=make_uint4(aa80::pack_bf16(value[0],value[1]),aa80::pack_bf16(value[2],value[3]),aa80::pack_bf16(value[4],value[5]),aa80::pack_bf16(value[6],value[7]));
}
// Same eight-row, four-column summation order as token qknorm_bwd with qk=false.
__global__ void token_bq_partial_kernel(const bf* qkvg,float* pbq,int M,int D,long stride) {
  const int col=threadIdx.x*4;float sum[4]={};
  for(int r=blockIdx.x*8;r<min(M,(int)blockIdx.x*8+8);++r){const uint2 v=*reinterpret_cast<const uint2*>(qkvg+(long)r*4*D+col);
    sum[0]+=aa80::bf16lo(v.x);sum[1]+=aa80::bf16hi(v.x);sum[2]+=aa80::bf16lo(v.y);sum[3]+=aa80::bf16hi(v.y);}
  *reinterpret_cast<float4*>(pbq+(long)blockIdx.x*stride+col)=make_float4(sum[0],sum[1],sum[2],sum[3]);
}
}
void backward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor dob,torch::Tensor bias,torch::Tensor lse,torch::Tensor delta,
              torch::Tensor packed,torch::Tensor db,torch::Tensor partial,torch::Tensor bias_t,torch::Tensor pbq,int64_t A,int64_t L,int64_t H,int64_t rows) {
  const int D=H*48,M=A*L;
  TORCH_CHECK(H==16 && L>=128 && L%128==0 && A>0 && A%rows==0 && (rows==1 || rows==2 || rows==4),"token native gradient geometry");
  for(auto t:{q,k,v,dob,bias,packed,db,partial,bias_t})TORCH_CHECK(t.is_cuda() && t.scalar_type()==at::kBFloat16 && t.device()==q.device(),"token native BF16 tensors");
  for(auto t:{q,k,v,dob})TORCH_CHECK(t.dim()==2 && t.size(0)==M && t.size(1)==D && t.stride(1)==1 && t.stride(0)>=D && t.stride(0)%8==0,"token native operand views");
  TORCH_CHECK(packed.is_contiguous() && packed.sizes()==torch::IntArrayRef({M,4*D}),"token native packed gradient");
  for(auto t:{bias,db,bias_t})TORCH_CHECK(t.is_contiguous() && t.numel()==H*L*L,"token native bias tensors");
  TORCH_CHECK(partial.is_contiguous() && partial.numel()==(A/rows)*H*L*L,"token native bias partials");
  for(auto t:{lse,delta})TORCH_CHECK(t.is_cuda() && t.scalar_type()==at::kFloat && t.is_contiguous() && t.numel()==A*H*L && t.device()==q.device(),"token native row terms");
  TORCH_CHECK(pbq.is_cuda() && pbq.device()==q.device() && pbq.scalar_type()==at::kFloat && pbq.dim()==2 && pbq.size(0)==(M+7)/8 && pbq.size(1)==D && pbq.stride(1)==1,"token native query bias partials");
  const at::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream();
  const double scale64=1.0/std::sqrt(48.0);
  const float scale=static_cast<float>(scale64),l2e=1.4426950408889634f;
  aa80::DqParams qp{};
  qp.q=(const bf*)q.data_ptr();qp.k=(const bf*)k.data_ptr();qp.v=(const bf*)v.data_ptr();qp.dov=(const bf*)dob.data_ptr();qp.bias=(const bf*)bias.data_ptr();
  qp.lse=lse.data_ptr<float>();qp.delta=delta.data_ptr<float>();qp.dq=packed.data_ptr();qp.dbp=(bf*)partial.data_ptr();
  qp.ldq=q.stride(0);qp.ldk=k.stride(0);qp.ldv=v.stride(0);qp.lddo=dob.stride(0);qp.lddq=4*D;qp.ss=L;qp.L=L;qp.H=H;
  qp.scl=static_cast<float>(scale64*1.4426950408889634);qp.bscl=l2e;qp.sm_scale=scale;
  if(rows==4)dq_launch<aa80::DqCfg<48,4,3,1,1,0>>(qp,A);
  else if(rows==2)dq_launch<aa80::DqCfg<48,2,4,1,1,0>>(qp,A);
  else dq_launch<aa80::DqCfg<48,1,3,2,1,0>>(qp,A);
  const long n8=H*L*L/8;
  token_db_bf16_kernel<<<(n8+255)/256,256,0,stream>>>((const bf*)partial.data_ptr(),(bf*)db.data_ptr(),A/rows,n8);
  aa80::bias_transpose_kernel<<<dim3(L/32,L/32,H),256,0,stream>>>((const bf*)bias.data_ptr(),(bf*)bias_t.data_ptr(),L);
  aa80::DkvParams kp{};
  kp.q=qp.q;kp.k=qp.k;kp.v=qp.v;kp.dov=qp.dov;kp.bias_t=(const bf*)bias_t.data_ptr();kp.lse=qp.lse;kp.delta=qp.delta;
  kp.dk=(bf*)packed.data_ptr()+D;kp.dv=(bf*)packed.data_ptr()+2*D;
  kp.ldq=qp.ldq;kp.ldk=qp.ldk;kp.ldv=qp.ldv;kp.lddo=qp.lddo;kp.lddk=4*D;kp.lddv=4*D;kp.ss=L;kp.L=L;kp.H=H;
  kp.scl=qp.scl;kp.bscl=l2e;kp.sm_scale=scale;
  dkv_launch<aa80::DkvCfg<48,3,2,1,0>>(kp,A);
  token_bq_partial_kernel<<<(M+7)/8,D/4,0,stream>>>((const bf*)packed.data_ptr(),pbq.data_ptr<float>(),M,D,pbq.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("backward",&backward);}
