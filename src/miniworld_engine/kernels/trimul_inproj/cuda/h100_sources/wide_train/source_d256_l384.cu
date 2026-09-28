// Frozen generated source: d256_register_budget_source.RegisterBudgetSource (budget 208).
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
// MiniWorld D256 B7: recompute derivatives and immediately accumulate dW.
// Adapted from our D128 producer-local dW schedule; D256 uses shared operands.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
using bf=__nv_bfloat16;
#pragma once
template<int TA,int TB> TMN_DEVI void mma64(float (&v)[32],uint64_t aa,uint64_t bb,int ac){
 asm volatile("{.reg .pred p;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,%35,%36;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]) : "l"(aa),"l"(bb),"r"(ac),"n"(TA),"n"(TB));
}
template<int TA,int TB> TMN_DEVI void mma128(float (&v)[64],uint64_t aa,uint64_t bb,int ac){
 asm volatile("{.reg .pred p;setp.ne.b32 p,%66,0;wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},%64,%65,p,1,1,%67,%68;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]),"+f"(v[32]),"+f"(v[33]),"+f"(v[34]),"+f"(v[35]),"+f"(v[36]),"+f"(v[37]),"+f"(v[38]),"+f"(v[39]),"+f"(v[40]),"+f"(v[41]),"+f"(v[42]),"+f"(v[43]),"+f"(v[44]),"+f"(v[45]),"+f"(v[46]),"+f"(v[47]),"+f"(v[48]),"+f"(v[49]),"+f"(v[50]),"+f"(v[51]),"+f"(v[52]),"+f"(v[53]),"+f"(v[54]),"+f"(v[55]),"+f"(v[56]),"+f"(v[57]),"+f"(v[58]),"+f"(v[59]),"+f"(v[60]),"+f"(v[61]),"+f"(v[62]),"+f"(v[63]) : "l"(aa),"l"(bb),"r"(ac),"n"(TA),"n"(TB));
}

constexpr int D=256,H=512,CH=32768,INPUT=36864,WEIGHT=73728,DERIV=106496;
struct Params {CUtensorMap xn,w,dl,dr,gmap[4];const bf* mask;bf *gp[4];float* part;int M;};
TMN_DEVI float rd(const bf* x,size_t i){return __bfloat162float(x[i]);}
TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int row,int rank,int slot){
 mbar_arrive_expect_tx(bar+slot,INPUT+128);
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],128,[%2];"::"r"(smem_u32(sm+114816+slot*128)),"l"(p.mask+row),"r"(smem_u32(bar+slot)):"memory");
 for(int c=0;c<4;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);
 tma_load_2d(sm+slot*INPUT+CH,rank<16?&p.dl:&p.dr,bar+slot,row,(rank%16)*32);
}

TMN_DEVI void store_gp(const CUtensorMap* map,uint8_t* sm,int row,int c){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}

// SPDX-License-Identifier: Apache-2.0
// D256 adaptation of MiniWorld D128 pair_glu: packed ldmatrix/stmatrix epilogue.
TMN_DEVI void packed_glu(float (&a)[32],uint8_t* s,uint8_t* sg,uint32_t ma,uint32_t mb){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
 #pragma unroll
 for(int q=0;q<2;++q){uint32_t dy[4],dg[4],dp[4];ldsm_x4_t(dy,smem_u32(s+32768)+swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
  #pragma unroll
  for(int j=0;j<4;++j){uint32_t masked,m=(j&1)?mb:ma;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy[j]),"r"(m));
   uint32_t gr=pack_bf16(a[q*8+j*2],a[q*8+j*2+1]),pr=pack_bf16(a[(q+2)*8+j*2],a[(q+2)*8+j*2+1]);float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
   dg[j]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));dp[j]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  }
  uint32_t addr=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);stsm_x4_t(smem_u32(sg)+addr,dg[0],dg[1],dg[2],dg[3]);stsm_x4_t(smem_u32(sg+4096)+addr,dp[0],dp[1],dp[2],dp[3]);
 }
}

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

TMN_DEVI void compute_source(const Params& p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+114688);
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,rank=blockIdx.x%32,split=blockIdx.x/32;
 int tiles=p.M/64,begin=(tiles*split)/WEIGHT_SPLITS,end=(tiles*(split+1))/WEIGHT_SPLITS;
 mbar_wait(bar+2,0);named_bar_sync(1,128);
 float dw[128]={};
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2,row=tile*64;uint8_t* xn=sm+slot*INPUT;

  mbar_wait(bar+slot,(it/2)&1);named_bar_sync(1,128);
  float pre[32]={};fence_regs(pre);wgmma_fence();
  static_for<16>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma64_off<(k/4)*8192+(k%4)*32,(k/4)*8192+(k%4)*32,0,0>(pre,smem_desc(smem_u32(xn),16,1024,1),smem_desc(smem_u32(sm+WEIGHT),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(pre);
  int ra=warp*16+lane/4;
  uint32_t ma=uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm+114816+slot*128)[ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(reinterpret_cast<bf*>(sm+114816+slot*128)[ra+8]))*0x10001u;
  packed_glu(pre,xn,sm+DERIV,ma,mb);
  named_bar_sync(1,128);fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){store_gp(p.gmap+(rank/16)*2,sm+DERIV+4096,row,(rank%16)*32);store_gp(p.gmap+(rank/16)*2+1,sm+DERIV,row,(rank%16)*32);tma_store_commit();}
  fence_regs(dw);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   uint64_t a=smem_desc(smem_u32(sm+DERIV),16,1024,1);
   mma256_off<k*32,k*2048,0,1>(dw,a,smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);
   
  });wgmma_commit();wgmma_wait<0>();fence_regs(dw);if(tid==0)tma_store_wait_all();named_bar_sync(1,128);if(tid==0)mbar_arrive(bar+3+slot);
 }
 static_for<128>([&](auto jj){constexpr int j=decltype(jj)::value;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
  int which=(rank/16)*2+(r<32?1:0),outrow=(rank%16)*32+r%32;
  size_t ix=size_t(split)*11*D*D+(3+2*which)*D*D+outrow*D+c;
  p.part[ix]=dw[j];
 });
}

extern "C" __global__ __launch_bounds__(256,1) void mw_d256_register_budget_source(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+114688);
 if(threadIdx.x==0){for(int i=0;i<5;++i)mbar_init(bar+i,1);fence_barrier_init();}
 __syncthreads();
 if(threadIdx.x<128){
  setmaxnreg_dec<32>();
  if(threadIdx.x==0){
   int rank=blockIdx.x%32,split=blockIdx.x/32,tiles=p.M/64,begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
   mbar_arrive_expect_tx(bar+2,CH);
   for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
   for(int tile=begin,it=0;tile<end;++tile,++it){int slot=it%2;if(it>=2)mbar_wait(bar+3+slot,((it/2)-1)&1);load_input(p,sm,bar,tile*64,rank,slot);}
  }
 }else{setmaxnreg_inc<208>();compute_source(p);}
}
