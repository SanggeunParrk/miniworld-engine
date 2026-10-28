// Frozen generated source: d256_full_spatial_contract.FullSpatialContract.
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
// Pure four-contraction kernel, fixed K, ordered WGMMA, no GP epilogue.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
#pragma once
#include "wgmma_off.cuh"  // mma64_off / mma128_off / mma256_off / mma64 / mma128


template<int OA,int OB,int TA,int TB> TMN_DEVI void mma192_off(float (&v)[96],uint64_t aa,uint64_t bb,int ac){
 asm volatile("{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%96,%101;add.u64 bx,%97,%102;setp.ne.b32 p,%98,0;wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95},ax,bx,p,1,1,%99,%100;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]),"+f"(v[32]),"+f"(v[33]),"+f"(v[34]),"+f"(v[35]),"+f"(v[36]),"+f"(v[37]),"+f"(v[38]),"+f"(v[39]),"+f"(v[40]),"+f"(v[41]),"+f"(v[42]),"+f"(v[43]),"+f"(v[44]),"+f"(v[45]),"+f"(v[46]),"+f"(v[47]),"+f"(v[48]),"+f"(v[49]),"+f"(v[50]),"+f"(v[51]),"+f"(v[52]),"+f"(v[53]),"+f"(v[54]),"+f"(v[55]),"+f"(v[56]),"+f"(v[57]),"+f"(v[58]),"+f"(v[59]),"+f"(v[60]),"+f"(v[61]),"+f"(v[62]),"+f"(v[63]),"+f"(v[64]),"+f"(v[65]),"+f"(v[66]),"+f"(v[67]),"+f"(v[68]),"+f"(v[69]),"+f"(v[70]),"+f"(v[71]),"+f"(v[72]),"+f"(v[73]),"+f"(v[74]),"+f"(v[75]),"+f"(v[76]),"+f"(v[77]),"+f"(v[78]),"+f"(v[79]),"+f"(v[80]),"+f"(v[81]),"+f"(v[82]),"+f"(v[83]),"+f"(v[84]),"+f"(v[85]),"+f"(v[86]),"+f"(v[87]),"+f"(v[88]),"+f"(v[89]),"+f"(v[90]),"+f"(v[91]),"+f"(v[92]),"+f"(v[93]),"+f"(v[94]),"+f"(v[95]) : "l"(aa),"l"(bb),"r"(ac),"n"(TA),"n"(TB),"n"(OA>>4),"n"(OB>>4));
}

constexpr int D=256,N=384,NT=128*ROW_GROUPS,INPUT=(ROW_GROUPS+6)*8192,SLOTS=CONTRACT_SLOTS,BAR=INPUT*SLOTS;
struct Params{CUtensorMap a[4],b[4],out[2];};
template<int MODE> TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni,int step){
 constexpr int TA=MODE==1,TB=MODE!=2;int slot=step%SLOTS,ki=step*64;
 mbar_arrive_expect_tx(bar+slot,INPUT);
 tma_load_3d(sm+slot*INPUT,p.a+MODE,bar+slot,TA?0:ki,TA?ch*N+ki:0,TA?mi/64:(ch*N+mi)/64);
 tma_load_3d(sm+slot*INPUT+ROW_GROUPS*8192,p.b+MODE,bar+slot,TB?0:ki,TB?ch*N+ki:0,TB?ni/64:(ch*N+ni)/64);
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr int TA=MODE==1,TB=MODE!=2,SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,wg=threadIdx.x/128;
 if(threadIdx.x==0)for(int step=0;step<SLOTS-1;++step)load<MODE>(p,sm,bar,ch,mi,ni,step);
 float v0[128]={},v1[64]={};
 #pragma unroll
 for(int step=0;step<6;++step){
  int slot=step%SLOTS;mbar_wait(bar+slot,(step/SLOTS)&1);__syncthreads();
  if(threadIdx.x==0 && step+SLOTS-1<6)load<MODE>(p,sm,bar,ch,mi,ni,step+SLOTS-1);
  fence_regs(v0);fence_regs(v1);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma256_off<k*(TA?2048:32),0+k*(TB?2048:32),TA,TB>(v0,smem_desc(smem_u32(sm+slot*INPUT+wg*8192),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+ROW_GROUPS*8192),TB?8192:16,1024,1),step>0||k>0);
   mma128_off<k*(TA?2048:32),32768+k*(TB?2048:32),TA,TB>(v1,smem_desc(smem_u32(sm+slot*INPUT+wg*8192),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+ROW_GROUPS*8192),TB?8192:16,1024,1),step>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v0);fence_regs(v1);
 }
 __syncthreads();
 static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=0+(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  *reinterpret_cast<uint32_t*>(sm+(wg*6+c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(v0[j],v0[j+1]);
 });
 static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=256+(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  *reinterpret_cast<uint32_t*>(sm+(wg*6+c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(v1[j],v1[j+1]);
 });
 fence_proxy_async();__syncthreads();
 if(threadIdx.x==0){
  #pragma unroll
  for(int wm=0;wm<ROW_GROUPS;++wm){
   asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{0,%2,%3}],[%1];"::"l"(p.out+SIDE),"r"(smem_u32(sm+wm*49152)),"r"((ch+HALF*D)*N+mi+wm*64),"r"(ni/64):"memory");
  }
  tma_store_commit();tma_store_wait_all();
 }
}
extern "C" __global__ __launch_bounds__(NT,MIN_BLOCKS)
void mw_d256_full_spatial_contract(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 constexpr int MT=N/(64*ROW_GROUPS),TILES=MT;
 int mode,ch,tile;
 #if GRID_ORDER==0
 mode=blockIdx.x/(D*TILES);ch=(blockIdx.x/TILES)%D;tile=blockIdx.x%TILES;
 #elif GRID_ORDER==1
 int half=blockIdx.x/(2*D*TILES),rem=blockIdx.x%(2*D*TILES);
 ch=rem/(2*TILES);mode=2*half+rem%2;tile=(rem/2)%TILES;
 #else
 mode=blockIdx.x/(D*TILES);int rem=blockIdx.x%(D*TILES);
 tile=(rem/8)%TILES;ch=rem%8+8*(rem/(8*TILES));
 #endif
 int mi=tile*(64*ROW_GROUPS),ni=0;
 if(threadIdx.x==0){for(int i=0;i<SLOTS;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();
 if(mode==0)run<0>(p,sm,bar,ch,mi,ni);
 else if(mode==1)run<1>(p,sm,bar,ch,mi,ni);
 else if(mode==2)run<2>(p,sm,bar,ch,mi,ni);
 else run<3>(p,sm,bar,ch,mi,ni);
}
