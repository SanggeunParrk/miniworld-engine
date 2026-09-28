// Frozen generated source: early_affine_init.IndependentOutputLN(wide_ordered_pair_ln_safe.SafeOrderedPairLN).
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
// Small TMA tiles permit more resident output-LN CTAs at wide dimensions.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int PAIR_SUMS=69888;
constexpr int H=2*WIDTH,NT=128,ROWS=LN_ROWS,SB=ROWS*H*2;
struct Params {CUtensorMap tri,dt,dnmap;const bf* dn;const float *mu,*rs,*gamma;float *dg,*db;int M;};
constexpr int STATS=69760;
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
extern "C" __global__ __launch_bounds__(NT,LN_MINBLOCKS)
void mw_independent_output_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+SB*(1+LN_DN_TMA));
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 int pair=warp/2,half=warp%2;float* sums=reinterpret_cast<float*>(sm+PAIR_SUMS);

 if(tid==0){mbar_init(bar,1);fence_barrier_init();}
 for(int c=tid;c<H;c+=NT)reinterpret_cast<float*>(sm+SB*(1+LN_DN_TMA)+128)[c]=p.gamma[c];
 __syncthreads();
 float gg[H/64]={},bb[H/64]={};int phase=0;
 for(int row=blockIdx.x*ROWS;row<p.M;row+=gridDim.x*ROWS){
  if(tid==0){mbar_arrive_expect_tx(bar,SB*(1+LN_DN_TMA)+ROWS*8);load_stats(p,sm+STATS,bar,row);
tma_load_3d(sm,&p.tri,bar,row,0,0);tma_load_3d(sm+SB,&p.dnmap,bar,0,row,0);
  }
  mbar_wait(bar,phase);phase^=1;__syncthreads();transpose32<false>(sm);

  for(int r=pair;r<ROWS;r+=2){
   float mu=reinterpret_cast<const float*>(sm+STATS)[r],rs=reinterpret_cast<const float*>(sm+STATS)[ROWS+r];
   float z[H/64],dy[H/64],s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=lane+half*(H/2)+q*32;
    z[q]=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    dy[q]=__bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]);
    gg[q]+=dy[q]*z[q];bb[q]+=dy[q];
    if(half==0){float v=dy[q]*reinterpret_cast<float*>(sm+2*SB+128)[c];s0+=v;s1+=v*z[q];}
   }
   if(half==0){sums[pair*64+lane]=s0;sums[pair*64+32+lane]=s1;}
   __syncthreads();
   if(half==1){
    s0=sums[pair*64+lane];s1=sums[pair*64+32+lane];
    #pragma unroll
    for(int q=0;q<H/64;++q){int c=lane+H/2+q*32;float v=dy[q]*reinterpret_cast<float*>(sm+2*SB+128)[c];s0+=v;s1+=v*z[q];}
    s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
    if(lane==0){sums[128+pair*2]=s0;sums[128+pair*2+1]=s1;}
   }
   __syncthreads();
   s0=sums[128+pair*2];s1=sums[128+pair*2+1];
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=lane+half*(H/2)+q*32;
    float centered=fmaf(dy[q],reinterpret_cast<float*>(sm+2*SB+128)[c],-s0);
    reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z[q],s1,centered)*rs);
   }
   __syncthreads();
  }
  __syncthreads();transpose32<true>(sm);fence_proxy_async();__syncthreads();
  if(tid==0){put_tile(&p.dt,sm,row,0);tma_store_commit();tma_store_wait_all();}
  __syncthreads();
 }

 float* acc=reinterpret_cast<float*>(sm);
 #pragma unroll
 for(int q=0;q<H/64;++q){int c=lane+half*(H/2)+q*32;acc[pair*H+c]=gg[q];acc[(2+pair)*H+c]=bb[q];}
 __syncthreads();
 for(int c=tid;c<H;c+=NT){atomicAdd(p.dg+c,acc[c]+acc[H+c]);atomicAdd(p.db+c,acc[2*H+c]+acc[3*H+c]);}

}
