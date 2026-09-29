// Frozen generated source: early_affine_init.IndependentOutputLN(wide_warp_packed_ln.WarpPackedLN).
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
// Small TMA tiles permit more resident output-LN CTAs at wide dimensions.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int H=2*WIDTH,NT=128,ROWS=LN_ROWS,SB=ROWS*H*2;
struct Params {CUtensorMap tri,dt,dnmap;const bf* dn;const float *mu,*rs,*gamma;float *dg,*db;int M;};
#ifdef LN_STATS_OFFSET   // MiniWorld: set by the single-direction build (H = WIDTH * 2 of that build)
constexpr int STATS=LN_STATS_OFFSET;
#else
constexpr int STATS=101504;
#endif
TMN_DEVI void load_stats(const Params& p,uint8_t* dst,uint64_t* bar,int row){
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(dst)),"l"(p.mu+row),"n"(ROWS*4),"r"(smem_u32(bar)):"memory");
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(dst+ROWS*4)),"l"(p.rs+row),"n"(ROWS*4),"r"(smem_u32(bar)):"memory");
}
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
TMN_DEVI uint32_t rawpos(int c,int r){uint32_t v=c*(ROWS*2)+r*2;if constexpr(ROWS==32)return sw64(v);else return v^(((v>>7)&1u)<<4);}
// One or two warps transpose each 64-channel by ROWS-row tile in place.
template<bool INVERSE> TMN_DEVI void transpose32(uint8_t* sm){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,mat=lane/8,r8=lane%8,w=warp%(ROWS/16);
 for(int kc=warp/(ROWS/16);kc<H/64;kc+=4/(ROWS/16)){
  uint32_t f[4][4],base=smem_u32(sm+kc*(ROWS*128));
  #pragma unroll
  for(int q=0;q<4;++q){
   uint32_t raw=rawpos(16*q+r8+((mat&2)?8:0),16*w+((mat&1)?8:0));
   uint32_t row=swz128(16*w+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
   ldsm_x4_t(f[q],base+(INVERSE?row:raw));
  }
  __syncwarp();
  #pragma unroll
  for(int q=0;q<4;++q){
   uint32_t raw=rawpos(16*q+r8+((mat&2)?8:0),16*w+((mat&1)?8:0));
   uint32_t row=swz128(16*w+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
   stsm_x4(base+(INVERSE?raw:row),f[q][0],f[q][1],f[q][2],f[q][3]);
  }
 }
 named_bar_sync(1,128);
}
TMN_DEVI void put_tile(const CUtensorMap* map,uint8_t* sm,int row,int c){
 asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%2,%3,0}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}

// SPDX-License-Identifier: Apache-2.0
template<int C,int NT> TMN_DEVI void aggregate_ln(float (&gg)[C/32],float (&bb)[C/32],float* sm,float* dg,float* db){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 #pragma unroll
 for(int q=0;q<C/32;++q){int c=lane+q*32;sm[warp*C+c]=gg[q];sm[(NT/32+warp)*C+c]=bb[q];}
 __syncthreads();
 for(int c=tid;c<C;c+=NT){float g=0,b=0;
  #pragma unroll
  for(int w=0;w<NT/32;++w){g+=sm[w*C+c];b+=sm[(NT/32+w)*C+c];}
  atomicAdd(dg+c,g);atomicAdd(db+c,b);
 }
 __syncthreads();
}

TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
TMN_DEVI float getdy(const Params& p,uint8_t* sm,int row,int r,int c){
 if constexpr(LN_DN_TMA)return __bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]);
 else return __bfloat162float(p.dn[size_t(row+r)*H+c]);
}
TMN_DEVI void prefetch(const Params& p,uint8_t* dst,uint8_t* stats,uint64_t* bar,int row){
 mbar_arrive_expect_tx(bar,2*SB+ROWS*8);load_stats(p,stats,bar,row);
tma_load_3d(dst,&p.tri,bar,row,0,0);tma_load_3d(dst+SB,&p.dnmap,bar,0,row,0);
}
extern "C" __global__ __launch_bounds__(NT,LN_MINBLOCKS)
void mw_independent_output_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t storage[];
 auto bar=reinterpret_cast<uint64_t*>(storage+4*SB);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;

 for(int c=tid;c<H;c+=NT)reinterpret_cast<float*>(storage+4*SB+128)[c]=p.gamma[c];
 if(tid==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}
 __syncthreads();
 float gg[H/32]={},bb[H/32]={};int phase=0;
 if(tid==0 && blockIdx.x*ROWS<p.M)prefetch(p,storage,storage+STATS,bar,blockIdx.x*ROWS);
 for(int row=blockIdx.x*ROWS,it=0;row<p.M;row+=gridDim.x*ROWS,++it){
  int slot=it%2;uint8_t* sm=storage+slot*2*SB;
  mbar_wait(bar+slot,(it/2)&1);__syncthreads();
  if(tid==0 && row+gridDim.x*ROWS<p.M){
   tma_store_wait_read<0>();
   prefetch(p,storage+(1-slot)*2*SB,storage+STATS+(1-slot)*ROWS*8,bar+1-slot,row+gridDim.x*ROWS);
  }
  transpose32<false>(sm);
  for(int r=warp;r<ROWS;r+=4){
   float mu=reinterpret_cast<const float*>(storage+STATS+slot*ROWS*8)[r],rs=reinterpret_cast<const float*>(storage+STATS+slot*ROWS*8)[ROWS+r],s0=0,s1=0;
   uint32_t tri[H/64],dn[H/64];
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=lane+q*64;
    tri[q]=uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm)[pos(r,c)]))|
           (uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm)[pos(r,c+32)]))<<16);
    dn[q]=uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]))|
          (uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm+SB)[pos(r,c+32)]))<<16);
   }
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float x=(q&1)?bf16hi(tri[q/2]):bf16lo(tri[q/2]);
    float dy=(q&1)?bf16hi(dn[q/2]):bf16lo(dn[q/2]);
    float z=(x-mu)*rs,v=dy*reinterpret_cast<float*>(storage+4*SB+128)[c];
    s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;
   }
   s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
   // Opaque, value-preserving boundaries prevent FP32 expansion from
   // surviving across the two passes. Arithmetic and channel order match.
   #pragma unroll
   for(int q=0;q<H/64;++q)asm volatile("" : "+r"(tri[q]),"+r"(dn[q]) :: "memory");
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float x=(q&1)?bf16hi(tri[q/2]):bf16lo(tri[q/2]);
    float dy=(q&1)?bf16hi(dn[q/2]):bf16lo(dn[q/2]);
    float z=(x-mu)*rs,centered=fmaf(dy,reinterpret_cast<float*>(storage+4*SB+128)[c],-s0);
    reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z,s1,centered)*rs);
   }
  }
  __syncthreads();transpose32<true>(sm);fence_proxy_async();__syncthreads();
  if(tid==0){put_tile(&p.dt,sm,row,0);tma_store_commit();}
  __syncthreads();
 }
 if(tid==0)tma_store_wait_all();__syncthreads();
 aggregate_ln<H,NT>(gg,bb,reinterpret_cast<float*>(storage),p.dg,p.db);
}
