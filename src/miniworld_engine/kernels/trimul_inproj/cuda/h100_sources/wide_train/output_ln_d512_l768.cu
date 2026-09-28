// Frozen generated source: wide_stats_ln.StatsLN(wide_affine_output_ln.AffineOutputLN).
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int H=2*WIDTH,ROWS=16,NT=128,SB=ROWS*H*2,BAR=3*SB;
struct Params {CUtensorMap tri,dt,dnmap;const bf* dn;const float *mu,*rs,*gamma;float *dg,*db;int M;};
constexpr int STATS=102528;
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
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}

TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int row,int slot){
 mbar_arrive_expect_tx(bar+slot,2*SB+ROWS*8);load_stats(p,sm+STATS,bar+slot,row);
 for(int c=0;c<H;c+=64){
  tma_load_2d(sm+slot*2*SB+(c/64)*(ROWS*128),&p.tri,bar+slot,row,c);
  tma_load_2d(sm+slot*2*SB+SB+(c/64)*(ROWS*128),&p.dnmap,bar+slot,c,row);
 }
}
TMN_DEVI void normalizer(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;const float* gamma=reinterpret_cast<float*>(sm+BAR+128);
 int first=blockIdx.x*ROWS,step=gridDim.x*ROWS;uint8_t* out=sm+2*SB;
 if(tid==0)load(p,sm,bar,first,0);
 for(int row=first,it=0;row<p.M;row+=step,++it){
  mbar_wait(bar,it&1);named_bar_sync(1,128);transpose32<false>(sm);
  if(tid==0)mbar_arrive(bar+5);
  for(int r=warp;r<ROWS;r+=4){
   float mu=reinterpret_cast<const float*>(sm+STATS)[r],rs=reinterpret_cast<const float*>(sm+STATS)[ROWS+r],s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]),v=dy*gamma[c];s0+=v;s1+=v*z;
   }
   s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]);
    reinterpret_cast<bf*>(out)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z,s1,fmaf(dy,gamma[c],-s0))*rs);
   }
  }
  mbar_wait(bar+2,it&1);named_bar_sync(1,128);
  transpose32<true>(out);fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){for(int c=0;c<H;c+=64)put_tile(&p.dt,out+(c/64)*(ROWS*128),row,c);tma_store_commit();
   if(row+step<p.M)load(p,sm,bar,row+step,0);
   tma_store_wait_read<0>();
  }
  named_bar_sync(1,128);
 }
 if(tid==0){tma_store_wait_all();mbar_arrive(bar+4);}
}
TMN_DEVI void affine(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;float gg[H/32]={},bb[H/32]={};
 for(int row=blockIdx.x*ROWS,it=0;row<p.M;row+=gridDim.x*ROWS,++it){int slot=0;uint8_t* tile=sm;
  mbar_wait(bar+5,it&1);named_bar_sync(2,128);
  for(int r=warp;r<ROWS;r+=4){float mu=reinterpret_cast<const float*>(sm+STATS)[r],rs=reinterpret_cast<const float*>(sm+STATS)[ROWS+r];
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float z=(__bfloat162float(reinterpret_cast<bf*>(tile)[pos(r,c)])-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(tile+SB)[pos(r,c)]);gg[q]+=dy*z;bb[q]+=dy;
   }
  }
  named_bar_sync(2,128);if(tid==0)mbar_arrive(bar+2+slot);
 }
 mbar_wait(bar+4,0);named_bar_sync(2,128);float* acc=reinterpret_cast<float*>(sm);
 #pragma unroll
 for(int q=0;q<H/32;++q){int c=lane+q*32;acc[warp*H+c]=gg[q];acc[(4+warp)*H+c]=bb[q];}
 named_bar_sync(2,128);
 for(int c=tid;c<H;c+=128){float g=0,b=0;
  #pragma unroll
  for(int w=0;w<4;++w){g+=acc[w*H+c];b+=acc[(4+w)*H+c];}
  atomicAdd(p.dg+c,g);atomicAdd(p.db+c,b);
 }
}
extern "C" __global__ __launch_bounds__(256,2)
void mw_wide_stats_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);int tid=threadIdx.x;
 for(int c=blockIdx.x*256+tid;c<H;c+=gridDim.x*256){p.dg[c]=0;p.db[c]=0;}
 for(int c=tid;c<H;c+=256)reinterpret_cast<float*>(sm+BAR+128)[c]=p.gamma[c];
 if(tid==0){for(int i=0;i<7;++i)mbar_init(bar+i,1);fence_barrier_init();}
 __syncthreads();cooperative_groups::this_grid().sync();
 if(tid<128){setmaxnreg_inc<144>();normalizer(p,sm,bar);}
 else{setmaxnreg_dec<112>();affine(p,sm,bar);}
}
