// Own sm80 W x V, sigmoid gate, projection, row dropout and residual.
// All original BF16 boundaries are retained. Optional head-major O is saved
// for training; inference keeps it entirely in registers.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "sm80_common.cuh"

namespace pwa80 {
DEVI uint32_t fg128_swz64(uint32_t r,uint32_t c) {return r*64+((c^((r>>1)&3))<<4);}
struct FusedContract128Params {
  const __nv_bfloat16 *m,*w,*v,*wg,*wo,*keep;
  const float *gamma,*beta;
  __nv_bfloat16 *out,*saved_o;
  int S,L,ns,kp;
  float eps,dscale;
};
template<int MINB,bool SAVE>
__global__ void __launch_bounds__(256,MINB) pwa_contract_gate_fused128_kernel(FusedContract128Params p) {
  constexpr int BM=128,BN=64,BK=32,ST=3,AS=8192,BS=4096,MB=32768,MT=2;
  constexpr int WGB=4096,WOB=4096,RING=ST*(AS+BS);
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t base=smem_u32(smem),sm=base+RING,swg=sm+MB,swo=swg+WGB;
  float* gb=reinterpret_cast<float*>(smem+RING+MB+WGB+WOB);
  const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,wm=warp>>1,wn=warp&1;
  const int gq=lane>>2,q=lane&3,im=blockIdx.x*BM,jn=blockIdx.y*BN,sp=blockIdx.z;
  const int sample=sp*p.kp+jn/32+wn,r0=wm*32+gq,r1=r0+8,nk=p.L/BK;
#pragma unroll
  for(int it=0;it<8;++it) {
    const int g=tid+256*it,r=g/8,c=g%8,s=r/BM;
    cp_async16(sm+s*16384+swz128(r%BM,c),p.m+((long)(sp*p.kp+jn/32+s)*p.L+im+r%BM)*64+8*c);
  }
  for(int c=tid;c<64;c+=256){gb[c]=p.gamma[c];gb[64+c]=p.beta[c];}
  cp_async_commit();cp_async_wait<0>();__syncthreads();
  uint32_t ya[MT][4][4];
#pragma unroll
  for(int mt=0;mt<MT;++mt) {
    uint32_t xa[4][4];
#pragma unroll
    for(int ks=0;ks<4;++ks)ldsm_x4(xa[ks],sm+wn*16384+swz128(wm*32+mt*16+(lane&15),2*ks+(lane>>4)));
    float sum0=0.f,sum1=0.f;
#pragma unroll
    for(int ks=0;ks<4;++ks){
      sum0+=bf16lo(xa[ks][0])+bf16hi(xa[ks][0])+bf16lo(xa[ks][2])+bf16hi(xa[ks][2]);
      sum1+=bf16lo(xa[ks][1])+bf16hi(xa[ks][1])+bf16lo(xa[ks][3])+bf16hi(xa[ks][3]);
    }
    const float mean0=quad_sum(sum0)/64.f,mean1=quad_sum(sum1)/64.f;
    float v0=0.f,v1=0.f;
#pragma unroll
    for(int ks=0;ks<4;++ks){
      float d;
      d=bf16lo(xa[ks][0])-mean0;v0=fmaf(d,d,v0);d=bf16hi(xa[ks][0])-mean0;v0=fmaf(d,d,v0);
      d=bf16lo(xa[ks][2])-mean0;v0=fmaf(d,d,v0);d=bf16hi(xa[ks][2])-mean0;v0=fmaf(d,d,v0);
      d=bf16lo(xa[ks][1])-mean1;v1=fmaf(d,d,v1);d=bf16hi(xa[ks][1])-mean1;v1=fmaf(d,d,v1);
      d=bf16lo(xa[ks][3])-mean1;v1=fmaf(d,d,v1);d=bf16hi(xa[ks][3])-mean1;v1=fmaf(d,d,v1);
    }
    const float rs0=rsqrtf(quad_sum(v0)/64.f+p.eps),rs1=rsqrtf(quad_sum(v1)/64.f+p.eps);
#pragma unroll
    for(int ks=0;ks<4;++ks){
      const int c=16*ks+2*q;
      const float2 g0=*reinterpret_cast<float2*>(gb+c),g1=*reinterpret_cast<float2*>(gb+c+8);
      const float2 b0=*reinterpret_cast<float2*>(gb+64+c),b1=*reinterpret_cast<float2*>(gb+64+c+8);
      ya[mt][ks][0]=pack_bf16(fmaf((bf16lo(xa[ks][0])-mean0)*rs0,g0.x,b0.x),fmaf((bf16hi(xa[ks][0])-mean0)*rs0,g0.y,b0.y));
      ya[mt][ks][1]=pack_bf16(fmaf((bf16lo(xa[ks][1])-mean1)*rs1,g0.x,b0.x),fmaf((bf16hi(xa[ks][1])-mean1)*rs1,g0.y,b0.y));
      ya[mt][ks][2]=pack_bf16(fmaf((bf16lo(xa[ks][2])-mean0)*rs0,g1.x,b1.x),fmaf((bf16hi(xa[ks][2])-mean0)*rs0,g1.y,b1.y));
      ya[mt][ks][3]=pack_bf16(fmaf((bf16lo(xa[ks][3])-mean1)*rs1,g1.x,b1.x),fmaf((bf16hi(xa[ks][3])-mean1)*rs1,g1.y,b1.y));
    }
  }
  float outacc[MT][8][4];
#pragma unroll
  for(int mt=0;mt<MT;++mt)for(int n=0;n<8;++n)for(int e=0;e<4;++e)outacc[mt][n][e]=0.f;
#pragma unroll 1
  for(int head=0;head<8;++head){
    const __nv_bfloat16* A=p.w+(long)head*p.L*p.L;
    const __nv_bfloat16* B=p.v+(long)(head*p.ns+sp)*p.L*p.kp*32;
    const int wr=tid/8,wc=tid%8;
    cp_async16(swg+swz128(wr,wc),p.wg+(head*32+wr)*64+wc*8);
    const int orow=tid/4,och=tid%4;
    cp_async16(swo+fg128_swz64(orow,och),p.wo+orow*256+head*32+och*8);
    cp_async_commit();
    auto load=[&](int step,int slot){
      const uint32_t sa=base+slot*(AS+BS),sb=sa+AS;
      const int br=tid/8,bc=tid%8;
#pragma unroll
      for(int it=0;it<2;++it){const int g=tid+256*it,ar=g/4,ac=g%4;cp_async16(sa+fg128_swz64(ar,ac),A+(long)(im+ar)*p.L+step*BK+ac*8);}
      cp_async16(sb+swz128(br,bc),B+(long)(step*BK+br)*p.kp*32+jn+bc*8);
    };
    float oa[MT][4][4];
#pragma unroll
    for(int mt=0;mt<MT;++mt)for(int nt=0;nt<4;++nt)for(int e=0;e<4;++e)oa[mt][nt][e]=0.f;
    uint32_t af[2][MT][4],bf[2][2][4];
    auto fragments=[&](int buf,int slot,int kk){
      const uint32_t sa=base+slot*(AS+BS),sb=sa+AS;
      #pragma unroll
      for(int mt=0;mt<MT;++mt)ldsm_x4(af[buf][mt],sa+fg128_swz64(wm*32+mt*16+(lane&7)+8*((lane>>3)&1),2*kk+(lane>>4)));
#pragma unroll
      for(int np=0;np<2;++np)ldsm_x4_t(bf[buf][np],sb+swz128(16*kk+(lane&7)+8*((lane>>3)&1),wn*4+2*np+(lane>>4)));
    };
    auto multiply=[&](int buf){
#pragma unroll
      for(int mt=0;mt<MT;++mt)
#pragma unroll
      for(int np=0;np<2;++np){mma16816(oa[mt][2*np],af[buf][mt],bf[buf][np][0],bf[buf][np][1]);mma16816(oa[mt][2*np+1],af[buf][mt],bf[buf][np][2],bf[buf][np][3]);}
    };
#pragma unroll
    for(int slot=0;slot<ST-1;++slot){if(slot<nk)load(slot,slot);cp_async_commit();}
    cp_async_wait<ST-2>();__syncthreads();fragments(0,0,0);
#pragma unroll 1
    for(int step=0;step<nk;++step){
#pragma unroll
      for(int kk=0;kk<2;++kk){
        if(kk==1){
          cp_async_wait<ST-3>();__syncthreads();
          if(step+ST-1<nk)load(step+ST-1,(step+ST-1)%ST);
          cp_async_commit();if(step+1<nk)fragments(0,(step+1)%ST,0);
        }else fragments(1,step%ST,1);
        multiply(kk&1);
      }
    }
    cp_async_wait<0>();__syncthreads();
#pragma unroll
    for(int mt=0;mt<MT;++mt)
#pragma unroll
    for(int ul=0;ul<2;++ul){
      float ga[2][4];
#pragma unroll
      for(int n=0;n<2;++n)for(int e=0;e<4;++e)ga[n][e]=0.f;
#pragma unroll
      for(int ks=0;ks<4;++ks){
        uint32_t wf[4];ldsm_x4(wf,swg+swz128(16*ul+(lane&7)+8*(lane>>4),2*ks+((lane>>3)&1)));
        mma16816(ga[0],ya[mt][ks],wf[0],wf[1]);mma16816(ga[1],ya[mt][ks],wf[2],wf[3]);
      }
      uint32_t ua[4];
      ua[0]=pack_bf16(sigmoid_fast(ga[0][0])*round_bf16f(oa[mt][2*ul][0]),sigmoid_fast(ga[0][1])*round_bf16f(oa[mt][2*ul][1]));
      ua[1]=pack_bf16(sigmoid_fast(ga[0][2])*round_bf16f(oa[mt][2*ul][2]),sigmoid_fast(ga[0][3])*round_bf16f(oa[mt][2*ul][3]));
      ua[2]=pack_bf16(sigmoid_fast(ga[1][0])*round_bf16f(oa[mt][2*ul+1][0]),sigmoid_fast(ga[1][1])*round_bf16f(oa[mt][2*ul+1][1]));
      ua[3]=pack_bf16(sigmoid_fast(ga[1][2])*round_bf16f(oa[mt][2*ul+1][2]),sigmoid_fast(ga[1][3])*round_bf16f(oa[mt][2*ul+1][3]));
#pragma unroll
      for(int np=0;np<4;++np){
        uint32_t wf[4];ldsm_x4(wf,swo+fg128_swz64(16*np+(lane&7)+8*(lane>>4),2*ul+((lane>>3)&1)));
        mma16816(outacc[mt][2*np],ua,wf[0],wf[1]);mma16816(outacc[mt][2*np+1],ua,wf[2],wf[3]);
      }
    }
    if constexpr(SAVE){
#pragma unroll
      for(int mt=0;mt<MT;++mt)
#pragma unroll
      for(int nt=0;nt<4;++nt){
        sts32(base+swz128(r0+16*mt,wn*4+nt)+q*4,pack_bf16(oa[mt][nt][0],oa[mt][nt][1]));
        sts32(base+swz128(r1+16*mt,wn*4+nt)+q*4,pack_bf16(oa[mt][nt][2],oa[mt][nt][3]));
      }
      __syncthreads();
#pragma unroll
      for(int it=0;it<4;++it){const int g=tid+256*it,r=g/8,c=g%8;
        stg128(p.saved_o+(long)(head*p.ns+sp)*p.L*p.kp*32+(long)(im+r)*p.kp*32+jn+c*8,lds128(base+swz128(r,c)));
      }
    }
    __syncthreads();
  }
#pragma unroll
  for(int mt=0;mt<MT;++mt)
#pragma unroll
  for(int nt=0;nt<8;++nt){
    const int col=nt*8+q*2;
#pragma unroll
    for(int h2=0;h2<2;++h2){
      const int row=(h2==0?r0:r1)+16*mt;
      uint32_t mask=0x3f803f80;
      if(p.keep)mask=__ldg(reinterpret_cast<const uint32_t*>(p.keep+(long)(im+row)*64+col));
      const uint32_t addr=sm+wn*16384+swz128(row,col/8)+(col%8)*2;
      const uint32_t x=lds32(addr);
      float u0=round_bf16f(outacc[mt][nt][2*h2]),u1=round_bf16f(outacc[mt][nt][2*h2+1]);
      if(p.keep){u0=round_bf16f(u0*bf16lo(mask)*p.dscale);u1=round_bf16f(u1*bf16hi(mask)*p.dscale);}
      sts32(addr,pack_bf16(bf16lo(x)+u0,bf16hi(x)+u1));
    }
  }
  __syncthreads();
#pragma unroll
  for(int it=0;it<8;++it){const int g=tid+256*it,r=g/8,c=g%8,s=r/BM;
    stg128(p.out+((long)(sp*p.kp+jn/32+s)*p.L+im+r%BM)*64+c*8,lds128(sm+s*16384+swz128(r%BM,c)));
  }
}
}

template<int MINB,bool SAVE>
void fused_launch(pwa80::FusedContract128Params p){
  constexpr int SHMEM=3*12288+32768+4096+4096+512;
  static std::set<int> configured;
  if(configured.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(pwa80::pwa_contract_gate_fused128_kernel<MINB,SAVE>,cudaFuncAttributeMaxDynamicSharedMemorySize,SHMEM));
  pwa80::pwa_contract_gate_fused128_kernel<MINB,SAVE><<<dim3(p.L/128,p.kp/2,p.ns),256,SHMEM,at::cuda::getCurrentCUDAStream()>>>(p);
}
void run(torch::Tensor m,torch::Tensor w,torch::Tensor v,torch::Tensor gamma,torch::Tensor beta,torch::Tensor wg,torch::Tensor wo,c10::optional<torch::Tensor> keep,torch::Tensor out,c10::optional<torch::Tensor> saved_o,double eps,double dscale,int64_t mode){
  for(auto t:{m,w,v,wg,wo,out})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==at::kBFloat16&&t.device()==m.device(),"fused contract matching BF16 CUDA tensors");
  for(auto t:{gamma,beta})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==at::kFloat&&t.device()==m.device()&&t.numel()==64,"fused contract LN parameters");
  TORCH_CHECK(m.dim()==3&&m.size(2)==64&&out.sizes()==m.sizes()&&w.dim()==3&&w.size(0)==8&&w.size(1)==m.size(1)&&w.size(2)==m.size(1),"fused contract M/W shapes");
  TORCH_CHECK(v.dim()==3&&v.size(0)%8==0&&v.size(1)==m.size(1)&&wg.sizes()==at::IntArrayRef({256,64})&&wo.sizes()==at::IntArrayRef({64,256}),"fused contract V/weights shapes");
  const int ns=v.size(0)/8,S=m.size(0),L=m.size(1);
  TORCH_CHECK(ns>0&&S%ns==0&&v.size(2)==S/ns*32&&S/ns%2==0&&L%128==0&&L>=128,"fused contract tile dimensions");
  if(keep)TORCH_CHECK(keep->is_cuda()&&keep->is_contiguous()&&keep->scalar_type()==at::kBFloat16&&keep->device()==m.device()&&keep->sizes()==at::IntArrayRef({L,64}),"fused contract keep shape");
  if(saved_o)TORCH_CHECK(saved_o->is_cuda()&&saved_o->is_contiguous()&&saved_o->scalar_type()==at::kBFloat16&&saved_o->device()==m.device()&&saved_o->sizes()==v.sizes(),"fused contract saved O");
  TORCH_CHECK(mode==0||mode==1,"fused contract mode");
  const at::cuda::CUDAGuard guard(m.device());
  pwa80::FusedContract128Params p{(const __nv_bfloat16*)m.data_ptr(),(const __nv_bfloat16*)w.data_ptr(),(const __nv_bfloat16*)v.data_ptr(),(const __nv_bfloat16*)wg.data_ptr(),(const __nv_bfloat16*)wo.data_ptr(),keep?(const __nv_bfloat16*)keep->data_ptr():nullptr,gamma.data_ptr<float>(),beta.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),saved_o?(__nv_bfloat16*)saved_o->data_ptr():nullptr,S,L,ns,S/ns,(float)eps,(float)dscale};
  if(saved_o){if(mode==0)fused_launch<1,true>(p);else fused_launch<2,true>(p);}
  else {if(mode==0)fused_launch<1,false>(p);else fused_launch<2,false>(p);}
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
