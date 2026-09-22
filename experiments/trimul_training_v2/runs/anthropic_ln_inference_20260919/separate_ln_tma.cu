// SPDX-License-Identifier: Apache-2.0
// Anthropic f4f62fa ln_fragment + original K1/K3 TMA/ldmatrix operand layouts.
// Standalone wrapper: one 64-row warpgroup, same saved normalized values/statistics.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
struct Params { CUtensorMap input,output; const float *gamma,*beta; __nv_bfloat16 *y; float *mean,*rstd; int m; float eps; };
extern "C" __global__ __launch_bounds__(128)
void mw_ln(__grid_constant__ const Params p) {
 __shared__ __align__(128) uint8_t sx[MW_C*64*2];
 __shared__ __align__(128) uint8_t so[8192];
 __shared__ float sg[MW_C],sb[MW_C];
 __shared__ __align__(8) uint64_t bar;
 int tid=threadIdx.x,lane=tid%32,wiw=tid/32;
 for(int c=tid;c<MW_C;c+=128){sg[c]=p.gamma[c];sb[c]=p.beta[c];}
 if(tid==0){mbar_init(&bar,1);fence_barrier_init();mbar_arrive_expect_tx(&bar,MW_C*64*2);
#pragma unroll
  for(int kc=0;kc<MW_C/64;++kc){
   if(MW_TRANSPOSE)tma_load_2d(sx+kc*8192,&p.input,&bar,blockIdx.x*64,kc*64);
   else tma_load_2d(sx+kc*8192,&p.input,&bar,kc*64,blockIdx.x*64);
  }
 }
 __syncthreads();mbar_wait(&bar,0);
 uint32_t fa[MW_C/16][4];
 if(MW_TRANSPOSE){
  int mat=lane/8,r8=lane%8,tokc=16*wiw+((mat&1)?8:0);
#pragma unroll
  for(int ks=0;ks<MW_C/16;++ks){int krow=16*ks+r8+((mat&2)?8:0);ldsm_x4_t(fa[ks],smem_u32(sx)+swz128(krow,tokc*2));}
 }else load_frag_bf16<MW_C/16,8192>(fa,smem_u32(sx),16*wiw,lane);
 LnStats st=ln_fragment<MW_C/16,MW_SERIAL>(fa,sg,sb,lane,p.eps);
 int q=lane%4,ra=blockIdx.x*64+wiw*16+lane/4,rb=ra+8;
 if(MW_BULK_STORE){
  int mat=lane/8,lrow=lane%8+8*(mat&1);
#pragma unroll
  for(int c64=0;c64<MW_C/64;++c64){
   if(lane==0)tma_store_wait_read<0>();
   __syncwarp();
#pragma unroll
   for(int slab=0;slab<4;++slab){int k=4*c64+slab;
    stsm_x4(smem_u32(so)+wiw*2048+swz128(lrow,(2*slab+(mat>>1))*16),fa[k][0],fa[k][1],fa[k][2],fa[k][3]);
   }
   fence_proxy_async();__syncwarp();
   if(lane==0){tma_store_3d(&p.output,so+wiw*2048,c64*64,blockIdx.x*64+wiw*16,0);tma_store_commit();}
  }
  if(lane==0)tma_store_wait_read<0>();
  __syncwarp();
 }else{
#pragma unroll
 for(int k=0;k<MW_C/16;++k){int c=16*k+2*q;
  if(ra<p.m){stg32(p.y+(size_t)ra*MW_C+c,fa[k][0]);stg32(p.y+(size_t)ra*MW_C+c+8,fa[k][2]);}
  if(rb<p.m){stg32(p.y+(size_t)rb*MW_C+c,fa[k][1]);stg32(p.y+(size_t)rb*MW_C+c+8,fa[k][3]);}
 }
}

}
