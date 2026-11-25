// SPDX-License-Identifier: Apache-2.0
// Single-direction TriMul training: output-LN backward with the next row tile's TMA loads in flight while this tile is processed.
// Same per-row arithmetic, same row-to-CTA assignment (grid-stride over ROWS-row tiles) and same per-thread dgamma / dbeta
// accumulation order as output_ln_d512_l384.cu (its tile loop is load -> wait -> transpose -> rows -> transpose -> store -> wait,
// so loads, compute and stores of a CTA never overlap); only the order of the TMA operations changes: two tile sets (tri | dn |
// mean, rstd), the next tile is requested before the current one is processed, the dt store is not waited for until its set is
// reused. ABL (removal-ablation bits, timing experiments only): 1 no tri load, 2 no dn load, 4 no store, 8 no row compute.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
#ifndef ABL
#define ABL 0
#endif
constexpr int H=2*WIDTH,NT=128,ROWS=LN_ROWS,SB=ROWS*H*2;
constexpr int STATSZ=(ROWS*8+1023)/1024*1024,SETSZ=2*SB+STATSZ;  // tile sets stay 1024-aligned (TMA swizzle)
struct Params {CUtensorMap tri,dt,dnmap;const bf* dn;const float *mu,*rs,*gamma;float *dg,*db;int M;};
TMN_DEVI void load_stats(const Params& p,uint8_t* dst,uint64_t* bar,int row){
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(dst)),"l"(p.mu+row),"n"(ROWS*4),"r"(smem_u32(bar)):"memory");
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(dst+ROWS*4)),"l"(p.rs+row),"n"(ROWS*4),"r"(smem_u32(bar)):"memory");
}
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
TMN_DEVI uint32_t rawpos(int c,int r){uint32_t v=c*(ROWS*2)+r*2;if constexpr(ROWS==32)return sw64(v);else return v^(((v>>7)&1u)<<4);}
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
  named_bar_sync(1,128);
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
TMN_DEVI void store_wait_read(){asm volatile("cp.async.bulk.wait_group.read 0;\n":::"memory");}
TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
extern "C" __global__ __launch_bounds__(NT,LN_MINBLOCKS)
void mw_independent_output_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bars=reinterpret_cast<uint64_t*>(sm+2*SETSZ);
 float* gam=reinterpret_cast<float*>(sm+2*SETSZ+128);
 float* sums=reinterpret_cast<float*>(sm+2*SETSZ+128+4*H);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 int pair=warp/2,half=warp%2;

 if(tid==0){mbar_init(bars,1);mbar_init(bars+1,1);fence_barrier_init();}
 for(int c=tid;c<H;c+=NT)gam[c]=p.gamma[c];
 __syncthreads();
 auto issue=[&](int row,int s){
  uint8_t* b=sm+s*SETSZ;
  mbar_arrive_expect_tx(bars+s,((ABL&1)?0:SB)+((ABL&2)?0:SB)+ROWS*8);
  load_stats(p,b+2*SB,bars+s,row);
  if(!(ABL&1))tma_load_3d(b,&p.tri,bars+s,row,0,0);
  if(!(ABL&2))tma_load_3d(b+SB,&p.dnmap,bars+s,0,row,0);
 };
 const int first=blockIdx.x*ROWS,stride=gridDim.x*ROWS;
 if(tid==0&&first<p.M)issue(first,0);
 float gg[H/64]={},bb[H/64]={};
 int it=0;
 for(int row=first;row<p.M;row+=stride,++it){
  const int s=it&1;uint8_t* cur=sm+s*SETSZ;
  if(tid==0&&row+stride<p.M){store_wait_read();issue(row+stride,s^1);}  // set s ^ 1 held the previous tile: its dt store has read it
  mbar_wait(bars+s,(it>>1)&1);__syncthreads();transpose32<false>(cur);

  if(!(ABL&8))for(int r=pair;r<ROWS;r+=2){
   float mu=reinterpret_cast<const float*>(cur+2*SB)[r],rs=reinterpret_cast<const float*>(cur+2*SB)[ROWS+r];
   float z[H/64],dy[H/64],s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=lane+half*(H/2)+q*32;
    z[q]=(__bfloat162float(reinterpret_cast<bf*>(cur)[pos(r,c)])-mu)*rs;
    dy[q]=__bfloat162float(reinterpret_cast<bf*>(cur+SB)[pos(r,c)]);
    gg[q]+=dy[q]*z[q];bb[q]+=dy[q];
    if(half==0){float v=dy[q]*gam[c];s0+=v;s1+=v*z[q];}
   }
   if(half==0){sums[pair*64+lane]=s0;sums[pair*64+32+lane]=s1;}
   __syncthreads();
   if(half==1){
    s0=sums[pair*64+lane];s1=sums[pair*64+32+lane];
    #pragma unroll
    for(int q=0;q<H/64;++q){int c=lane+H/2+q*32;float v=dy[q]*gam[c];s0+=v;s1+=v*z[q];}
    s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
    if(lane==0){sums[128+pair*2]=s0;sums[128+pair*2+1]=s1;}
   }
   __syncthreads();
   s0=sums[128+pair*2];s1=sums[128+pair*2+1];
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=lane+half*(H/2)+q*32;
    float centered=fmaf(dy[q],gam[c],-s0);
    reinterpret_cast<bf*>(cur)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z[q],s1,centered)*rs);
   }
   __syncthreads();
  }
  __syncthreads();transpose32<true>(cur);fence_proxy_async();__syncthreads();
  if(tid==0){if(!(ABL&4))put_tile(&p.dt,cur,row,0);tma_store_commit();}
 }
 if(tid==0)tma_store_wait_all();
 __syncthreads();

 float* acc=reinterpret_cast<float*>(sm);
 #pragma unroll
 for(int q=0;q<H/64;++q){int c=lane+half*(H/2)+q*32;acc[pair*H+c]=gg[q];acc[(2+pair)*H+c]=bb[q];}
 __syncthreads();
 for(int c=tid;c<H;c+=NT){
  p.dg[blockIdx.x*H+c]=acc[c]+acc[H+c];p.db[blockIdx.x*H+c]=acc[2*H+c]+acc[3*H+c];
 }
}
