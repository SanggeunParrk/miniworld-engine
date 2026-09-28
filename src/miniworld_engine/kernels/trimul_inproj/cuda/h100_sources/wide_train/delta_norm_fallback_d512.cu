// Frozen generated source: wide_lowreg_delta_norm.LowRegisterDeltaNorm fallback.
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=WIDTH,H=2*D,ROWS=OUTPUT_ROWS,NT=OUTPUT_THREADS,SB=ROWS*H*2;
struct Params {CUtensorMap tri,norm;const float *gamma,*beta;float *mu,*rs;int M;const unsigned* count;unsigned capacity;};
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
extern "C" __global__ __launch_bounds__(NT,2)
void mw_wide_lowreg_delta_norm_fallback(__grid_constant__ const Params p){
 if(*p.count<=p.capacity)return;
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+SB);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}__syncthreads();int phase=0;
 for(int row=blockIdx.x*ROWS;row<p.M;row+=gridDim.x*ROWS){
  if(tid==0){mbar_arrive_expect_tx(bar,SB);for(int c=0;c<H;c+=64)tma_load_2d(sm+(c/64)*(ROWS*128),&p.tri,bar,row,c);}
  mbar_wait(bar,phase);phase^=1;__syncthreads();transpose32<false>(sm);
  for(int r=warp;r<ROWS;r+=NT/32){
   float s=0,mu,rs;
   if constexpr(true){
    float values[H/32];
    #pragma unroll
    for(int q=0;q<H/32;++q){values[q]=__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,lane+q*32)]);s+=values[q];}
    mu=sumwarp(s)/H;s=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){float z=values[q]-mu;s+=z*z;}
    rs=rsqrtf(sumwarp(s)/H+1e-5f);
    #pragma unroll
    for(int q=0;q<H/32;++q){int c=lane+q*32;reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf((values[q]-mu)*rs,p.gamma[c],p.beta[c]));}
   }else{
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){uint32_t v=*reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));s+=bf16lo(v)+bf16hi(v);}
    mu=sumwarp(s)/H;s=0;
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){uint32_t v=*reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));float a=bf16lo(v)-mu,b=bf16hi(v)-mu;s+=a*a+b*b;}
    rs=rsqrtf(sumwarp(s)/H+1e-5f);
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){auto ptr=reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));uint32_t v=*ptr;
     *ptr=pack_bf16(fmaf((bf16lo(v)-mu)*rs,p.gamma[c],p.beta[c]),fmaf((bf16hi(v)-mu)*rs,p.gamma[c+1],p.beta[c+1]));
    }
   }
   if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}
  }
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<H;c+=64){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.norm),"r"(smem_u32(sm+(c/64)*(ROWS*128))),"r"(c),"r"(row):"memory");
  }tma_store_commit();tma_store_wait_all();}__syncthreads();
 }
}
