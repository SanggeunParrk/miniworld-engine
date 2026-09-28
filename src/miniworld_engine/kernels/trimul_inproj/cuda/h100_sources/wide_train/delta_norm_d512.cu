// Frozen generated source: wide_lowreg_delta_norm.LowRegisterDeltaNorm.
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=WIDTH,H=2*D,ROWS=OUTPUT_ROWS,NT=OUTPUT_THREADS,SB=ROWS*H*2;
struct Params {CUtensorMap tri,norm;uint64_t* patches;unsigned* count;unsigned capacity;unsigned* changed;const float *gamma,*beta;float *mu,*rs;int M;};
// SPDX-License-Identifier: Apache-2.0
// Uniform CTA barriers even when the number of channel tiles is not a
// multiple of the participating warp groups (e.g. D384 / 256 threads).
TMN_DEVI uint32_t rawpos(int c,int r){uint32_t v=c*(ROWS*2)+r*2;if constexpr(ROWS==32)return sw64(v);else return v^(((v>>7)&1u)<<4);}
template<bool INVERSE> TMN_DEVI void transpose32(uint8_t* sm){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,mat=lane/8,r8=lane%8,w=warp%(ROWS/16);
 constexpr int TILES=(NT/32)/(ROWS/16);
 for(int first=0;first<H/64;first+=TILES){
  int kc=first+warp/(ROWS/16);uint32_t f[4][4],base=smem_u32(sm+kc*(ROWS*128));
  if(kc<H/64){
   #pragma unroll
   for(int q=0;q<4;++q){
    uint32_t raw=rawpos(16*q+r8+((mat&2)?8:0),16*w+((mat&1)?8:0));
    uint32_t row=swz128(16*w+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
    ldsm_x4_t(f[q],base+(INVERSE?row:raw));
   }
  }
  named_bar_sync(1,NT);
  if(kc<H/64){
   #pragma unroll
   for(int q=0;q<4;++q){
    uint32_t raw=rawpos(16*q+r8+((mat&2)?8:0),16*w+((mat&1)?8:0));
    uint32_t row=swz128(16*w+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
    stsm_x4(base+(INVERSE?raw:row),f[q][0],f[q][1],f[q][2],f[q][3]);
   }
  }
 }
 named_bar_sync(1,NT);
}

TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,q));return v;}
TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
TMN_DEVI float norm_scalar(uint8_t* sm,int pos){unsigned v;asm volatile("ld.shared.u16 %0,[%1];":"=r"(v):"r"(smem_u32(sm+pos*2)):"memory");return __uint_as_float(v<<16);}
TMN_DEVI uint32_t norm_word(uint8_t* sm,int pos){uint32_t v;asm volatile("ld.shared.b32 %0,[%1];":"=r"(v):"r"(smem_u32(sm+pos*2)):"memory");return v;}
extern "C" __global__ __launch_bounds__(NT,3)
void mw_wide_lowreg_delta_norm(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+SB);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 for(int c=tid;c<H;c+=NT){reinterpret_cast<float*>(sm+SB+128)[c]=p.gamma[c];reinterpret_cast<float*>(sm+SB+128+H*4)[c]=p.beta[c];}
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}__syncthreads();int phase=0;
 for(int row=blockIdx.x*ROWS;row<p.M;row+=gridDim.x*ROWS){
  if(tid==0){mbar_arrive_expect_tx(bar,SB);for(int c=0;c<H;c+=64)tma_load_2d(sm+(c/64)*(ROWS*128),&p.tri,bar,row,c);}
  mbar_wait(bar,phase);phase^=1;__syncthreads();transpose32<false>(sm);
  for(int r=warp;r<ROWS;r+=NT/32){
   float scalar_mu,scalar_rs;
   {
    float ss=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){ss+=norm_scalar(sm,pos(r,lane+q*32));}
    scalar_mu=sumwarp(ss)/H;ss=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){float z=norm_scalar(sm,pos(r,lane+q*32))-scalar_mu;ss+=z*z;}
    scalar_rs=rsqrtf(sumwarp(ss)/H+1e-5f);
    if(lane==0){p.mu[row+r]=scalar_mu;p.rs[row+r]=scalar_rs;}
   }
   float s=0,mu,rs;
   if constexpr(D<512){
    float values[H/32];
    #pragma unroll
    for(int q=0;q<H/32;++q){values[q]=__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,lane+q*32)]);s+=values[q];}
    mu=sumwarp(s)/H;s=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){float z=values[q]-mu;s+=z*z;}
    rs=rsqrtf(sumwarp(s)/H+1e-5f);
    #pragma unroll
    for(int q=0;q<H/32;++q){int c=lane+q*32;reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf((values[q]-mu)*rs,reinterpret_cast<float*>(sm+SB+128)[c],reinterpret_cast<float*>(sm+SB+128+H*4)[c]));}
   }else{
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){uint32_t v=norm_word(sm,pos(r,c));s+=bf16lo(v)+bf16hi(v);}
    mu=sumwarp(s)/H;s=0;
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){uint32_t v=norm_word(sm,pos(r,c));float a=bf16lo(v)-mu,b=bf16hi(v)-mu;s+=a*a+b*b;}
    rs=rsqrtf(sumwarp(s)/H+1e-5f);
    if(__float_as_uint(mu)==__float_as_uint(scalar_mu)&&__float_as_uint(rs)==__float_as_uint(scalar_rs)){
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){auto ptr=reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));uint32_t v=norm_word(sm,pos(r,c));
     *ptr=pack_bf16(fmaf((bf16lo(v)-mu)*rs,reinterpret_cast<float*>(sm+SB+128)[c],reinterpret_cast<float*>(sm+SB+128+H*4)[c]),fmaf((bf16hi(v)-mu)*rs,reinterpret_cast<float*>(sm+SB+128)[c+1],reinterpret_cast<float*>(sm+SB+128+H*4)[c+1]));
    }
    }else{
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){auto ptr=reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));uint32_t v=norm_word(sm,pos(r,c));
     float g0=reinterpret_cast<float*>(sm+SB+128)[c],g1=reinterpret_cast<float*>(sm+SB+128)[c+1],b0=reinterpret_cast<float*>(sm+SB+128+H*4)[c],b1=reinterpret_cast<float*>(sm+SB+128+H*4)[c+1];
     uint32_t packed=pack_bf16(fmaf((bf16lo(v)-mu)*rs,g0,b0),fmaf((bf16hi(v)-mu)*rs,g1,b1));
     *ptr=packed;
     if(__float_as_uint(mu)!=__float_as_uint(scalar_mu)||__float_as_uint(rs)!=__float_as_uint(scalar_rs)){
      uint32_t scalar=pack_bf16(fmaf((bf16lo(v)-scalar_mu)*scalar_rs,g0,b0),fmaf((bf16hi(v)-scalar_mu)*scalar_rs,g1,b1));
      if(packed!=scalar){
       unsigned at=atomicAdd(p.count,1u);
       if(at<p.capacity)p.patches[at]=(uint64_t(scalar)<<32)|uint32_t(((row+r)*H+c)/2);
       atomicExch(p.changed+(row+r)/64,1u);
      }
     }
    }
    }
   }
  }
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<H;c+=64){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.norm),"r"(smem_u32(sm+(c/64)*(ROWS*128))),"r"(c),"r"(row):"memory");
  }tma_store_commit();tma_store_wait_all();}__syncthreads();
 }
}

struct Apply {const uint64_t* patches;const unsigned* count;unsigned capacity;uint32_t* norm;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_apply_delta_norm(__grid_constant__ const Apply p){
 unsigned count=*p.count;if(count>p.capacity)return;
 for(unsigned i=blockIdx.x*256+threadIdx.x;i<count;i+=gridDim.x*256){uint64_t v=p.patches[i];p.norm[uint32_t(v)]=uint32_t(v>>32);}
}
