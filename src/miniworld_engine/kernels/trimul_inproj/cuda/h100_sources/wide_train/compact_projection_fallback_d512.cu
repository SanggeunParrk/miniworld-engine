// Frozen generated source: wide_compact_projection.CompactProjection fallback.
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;
constexpr int D=WIDTH,H=2*D,STAGE=24576,BAR=2*STAGE;
struct Params {CUtensorMap norm,wp,proj;const unsigned* changed;int M;const unsigned* patch_count;unsigned patch_capacity;const unsigned* row_count;unsigned row_capacity;};
#pragma once
template<int OA,int OB,int TA,int TB> TMN_DEVI void mma64_off(float (&v)[32],uint64_t aa,uint64_t bb,int ac){
 asm volatile("{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%32,%37;add.u64 bx,%33,%38;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},ax,bx,p,1,1,%35,%36;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]) : "l"(aa),"l"(bb),"r"(ac),"n"(TA),"n"(TB),"n"(OA>>4),"n"(OB>>4));
}
template<int OA,int OB,int TA,int TB> TMN_DEVI void mma128_off(float (&v)[64],uint64_t aa,uint64_t bb,int ac){
 asm volatile("{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%64,%69;add.u64 bx,%65,%70;setp.ne.b32 p,%66,0;wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},ax,bx,p,1,1,%67,%68;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]),"+f"(v[32]),"+f"(v[33]),"+f"(v[34]),"+f"(v[35]),"+f"(v[36]),"+f"(v[37]),"+f"(v[38]),"+f"(v[39]),"+f"(v[40]),"+f"(v[41]),"+f"(v[42]),"+f"(v[43]),"+f"(v[44]),"+f"(v[45]),"+f"(v[46]),"+f"(v[47]),"+f"(v[48]),"+f"(v[49]),"+f"(v[50]),"+f"(v[51]),"+f"(v[52]),"+f"(v[53]),"+f"(v[54]),"+f"(v[55]),"+f"(v[56]),"+f"(v[57]),"+f"(v[58]),"+f"(v[59]),"+f"(v[60]),"+f"(v[61]),"+f"(v[62]),"+f"(v[63]) : "l"(aa),"l"(bb),"r"(ac),"n"(TA),"n"(TB),"n"(OA>>4),"n"(OB>>4));
}

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
