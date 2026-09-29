// Frozen generated source: early_both_affine_init.IndependentInputLN(JointInputReduce).
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
// Input LN/residual and exact-order split-weight reduction with TMA operands.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=WIDTH,NT=INPUT_THREADS,ROWS=INPUT_ROWS,SB=ROWS*D*2;
struct Params {CUtensorMap x,dn,res,dx;const float* gamma;float *dg,*db;const float* part;bf* dw[4];bf* gate;int M;};
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

TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
TMN_DEVI float rd(uint8_t* sm,int r,int c){return __bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)]);}
extern "C" __global__ __launch_bounds__(NT,INPUT_MINBLOCKS)
void mw_independent_input_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+3*SB);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;

 if(tid==0){mbar_init(bar,1);fence_barrier_init();}
 for(int c=tid;c<D;c+=NT)reinterpret_cast<float*>(sm+3*SB+128)[c]=p.gamma[c];
 __syncthreads();
 float gg[D/32]={},bb[D/32]={};int phase=0;
 for(int row=blockIdx.x*ROWS;row<p.M;row+=gridDim.x*ROWS){
  if(tid==0){mbar_arrive_expect_tx(bar,3*SB);
   for(int c=0;c<D;c+=64){
    tma_load_2d(sm+(c/64)*(ROWS*128),&p.x,bar,c,row);
    tma_load_2d(sm+SB+(c/64)*(ROWS*128),&p.dn,bar,c,row);
    tma_load_2d(sm+2*SB+(c/64)*(ROWS*128),&p.res,bar,c,row);
   }
  }
  mbar_wait(bar,phase);phase^=1;__syncthreads();
  for(int r=warp;r<ROWS;r+=NT/32){
   float xv[D/32],s=0;
   #pragma unroll
   for(int q=0;q<D/32;++q){xv[q]=rd(sm,r,lane+q*32);s+=xv[q];}
   float mu=sumwarp(s)/D;s=0;
   #pragma unroll
   for(int q=0;q<D/32;++q){float z=xv[q]-mu;s+=z*z;}
   float rs=rsqrtf(sumwarp(s)/D+1e-5f),s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<D/32;++q){int c=lane+q*32;float z=(xv[q]-mu)*rs,dy=rd(sm+SB,r,c),v=dy*reinterpret_cast<float*>(sm+3*SB+128)[c];s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;}
   s0=sumwarp(s0)/D;s1=sumwarp(s1)/D;
   #pragma unroll
   for(int q=0;q<D/32;++q){int c=lane+q*32;float z=(xv[q]-mu)*rs,v=rd(sm+SB,r,c)*reinterpret_cast<float*>(sm+3*SB+128)[c];
    float dx=(v-s0-z*s1)*rs;
    reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(dx+rd(sm+2*SB,r,c));
   }
  }
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<D;c+=64){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dx),"r"(smem_u32(sm+(c/64)*(ROWS*128))),"r"(c),"r"(row):"memory");
  }tma_store_commit();tma_store_wait_all();}
  __syncthreads();
 }
 aggregate_ln<D,NT>(gg,bb,reinterpret_cast<float*>(sm),p.dg,p.db);
#ifndef INPUT_NO_REDUCE   // MiniWorld: the single-direction path reduces its weight gradients with GEMMs
 for(int i=blockIdx.x*NT+tid;i<D*D;i+=gridDim.x*NT){float v=0;
  #pragma unroll
  for(int s=0;s<8;++s)v+=p.part[size_t(s)*11*D*D+2*D*D+i];
  p.gate[i]=__float2bfloat16_rn(v);
 }
 for(int which=0;which<4;++which){
  for(int i=blockIdx.x*NT+tid;i<2*D*D;i+=gridDim.x*NT){float v=0;
   #pragma unroll
   for(int s=0;s<8;++s)v+=p.part[size_t(s)*11*D*D+(3+2*which)*D*D+i];
   p.dw[which][i]=__float2bfloat16_rn(v);
  }
 }
#endif
}
