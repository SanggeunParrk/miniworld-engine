// Frozen generated source: d256_full_spatial_contract.FullSpatialContract.
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
// Pure four-contraction kernel, fixed K, ordered WGMMA, no GP epilogue.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
#pragma once
template<int OA,int OB,int TA,int TB> TMN_DEVI void mma64_off(float (&v)[32],uint64_t aa,uint64_t bb,int ac){
 asm volatile("{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%32,%37;add.u64 bx,%33,%38;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},ax,bx,p,1,1,%35,%36;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]) : "l"(aa),"l"(bb),"r"(ac),"n"(TA),"n"(TB),"n"(OA>>4),"n"(OB>>4));
}
template<int OA,int OB,int TA,int TB> TMN_DEVI void mma128_off(float (&v)[64],uint64_t aa,uint64_t bb,int ac){
 asm volatile("{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%64,%69;add.u64 bx,%65,%70;setp.ne.b32 p,%66,0;wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},ax,bx,p,1,1,%67,%68;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]),"+f"(v[32]),"+f"(v[33]),"+f"(v[34]),"+f"(v[35]),"+f"(v[36]),"+f"(v[37]),"+f"(v[38]),"+f"(v[39]),"+f"(v[40]),"+f"(v[41]),"+f"(v[42]),"+f"(v[43]),"+f"(v[44]),"+f"(v[45]),"+f"(v[46]),"+f"(v[47]),"+f"(v[48]),"+f"(v[49]),"+f"(v[50]),"+f"(v[51]),"+f"(v[52]),"+f"(v[53]),"+f"(v[54]),"+f"(v[55]),"+f"(v[56]),"+f"(v[57]),"+f"(v[58]),"+f"(v[59]),"+f"(v[60]),"+f"(v[61]),"+f"(v[62]),"+f"(v[63]) : "l"(aa),"l"(bb),"r"(ac),"n"(TA),"n"(TB),"n"(OA>>4),"n"(OB>>4));
}

// SPDX-License-Identifier: Apache-2.0
// Full D256 WGMMA accumulator. Callers must statically index the stores.
template<int OA,int OB,int TA,int TB> TMN_DEVI void mma256_off(float (&d)[128],uint64_t a,uint64_t b,int ac){
 asm volatile("{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%128,%133;add.u64 bx,%129,%134;setp.ne.b32 p,%130,0;wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127},ax,bx,p,1,1,%131,%132;}"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),"+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),"+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),"+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),"+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),"+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),"+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),"+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127]) : "l"(a),"l"(b),"r"(ac),"n"(TA),"n"(TB),"n"(OA>>4),"n"(OB>>4));
}

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
