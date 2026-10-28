// Frozen generated source: wide_compact_projection.CompactProjection fallback.
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;
constexpr int D=WIDTH,H=2*D,STAGE=24576,BAR=2*STAGE;
struct Params {CUtensorMap norm,wp,proj;const unsigned* changed;int M;const unsigned* patch_count;unsigned patch_capacity;const unsigned* row_count;unsigned row_capacity;};
#pragma once
#include "wgmma_off.cuh"  // mma64_off / mma128_off / mma256_off / mma64 / mma128

TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int row,int col,int ki,int slot){
 mbar_arrive_expect_tx(bar+slot,STAGE);
 tma_load_2d(sm+slot*STAGE,&p.norm,bar+slot,ki,row);
 tma_load_2d(sm+slot*STAGE+8192,&p.wp,bar+slot,ki,col);
 tma_load_2d(sm+slot*STAGE+16384,&p.wp,bar+slot,ki,col+64);
}
extern "C" __global__ __launch_bounds__(128,4)
void mw_wide_compact_projection_fallback(__grid_constant__ const Params p){
 if(*p.patch_count<=p.patch_capacity&&*p.row_count<=p.row_capacity)return;
 int row=blockIdx.x*64,col=blockIdx.y*128;
 if(!p.changed[blockIdx.x])return;
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 if(tid==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}__syncthreads();
 float v[64]={};if(tid==0)load(p,sm,bar,row,col,0,0);
 for(int ki=0,it=0;ki<H;ki+=64,++it){int slot=it%2;
  mbar_wait(bar+slot,(it/2)&1);__syncthreads();
  if(tid==0&&ki+64<H)load(p,sm,bar,row,col,ki+64,1-slot);
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma128_off<q*32,q*32,0,0>(v,smem_desc(smem_u32(sm+slot*STAGE),16,1024,1),smem_desc(smem_u32(sm+slot*STAGE+8192),16,1024,1),it>0||q>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
 }
 static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  *reinterpret_cast<uint32_t*>(sm+(c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(v[j],v[j+1]);
 });fence_proxy_async();__syncthreads();
 if(tid==0){for(int c=0;c<128;c+=64){
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.proj),"r"(smem_u32(sm+(c/64)*8192)),"r"(col+c),"r"(row):"memory");
 }tma_store_commit();tma_store_wait_all();}
}
