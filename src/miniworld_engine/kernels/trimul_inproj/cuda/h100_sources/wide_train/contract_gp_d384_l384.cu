// Frozen generated source: wide_staged_epilogue_gp.StagedEpilogueGP.
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
#pragma once
template<int OA,int OB,int TA,int TB> TMN_DEVI void mma64_off(float (&v)[32],uint64_t aa,uint64_t bb,int ac){
 asm volatile("{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%32,%37;add.u64 bx,%33,%38;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},ax,bx,p,1,1,%35,%36;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]) : "l"(aa),"l"(bb),"r"(ac),"n"(TA),"n"(TB),"n"(OA>>4),"n"(OB>>4));
}
template<int OA,int OB,int TA,int TB> TMN_DEVI void mma128_off(float (&v)[64],uint64_t aa,uint64_t bb,int ac){
 asm volatile("{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%64,%69;add.u64 bx,%65,%70;setp.ne.b32 p,%66,0;wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},ax,bx,p,1,1,%67,%68;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]),"+f"(v[32]),"+f"(v[33]),"+f"(v[34]),"+f"(v[35]),"+f"(v[36]),"+f"(v[37]),"+f"(v[38]),"+f"(v[39]),"+f"(v[40]),"+f"(v[41]),"+f"(v[42]),"+f"(v[43]),"+f"(v[44]),"+f"(v[45]),"+f"(v[46]),"+f"(v[47]),"+f"(v[48]),"+f"(v[49]),"+f"(v[50]),"+f"(v[51]),"+f"(v[52]),"+f"(v[53]),"+f"(v[54]),"+f"(v[55]),"+f"(v[56]),"+f"(v[57]),"+f"(v[58]),"+f"(v[59]),"+f"(v[60]),"+f"(v[61]),"+f"(v[62]),"+f"(v[63]) : "l"(aa),"l"(bb),"r"(ac),"n"(TA),"n"(TB),"n"(OA>>4),"n"(OB>>4));
}

constexpr int D=WIDTH,H=2*D,INPUT=32768,SLOTS=3,BAR=INPUT*SLOTS;
struct Params {CUtensorMap a[4],b[4];const bf* pre;const bf* mask;bf* gp[4];bf* dl;bf* dr;CUtensorMap premap,maskmap,outmap[4];int N;};

template<int MODE> TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni,int ki,int slot){
 constexpr int TA=MODE==1,TB=MODE!=2;
 mbar_arrive_expect_tx(bar+slot,INPUT);
 for(int g=0;g<2;++g){int m=mi+64*g,n=ni+64*g;
  tma_load_2d(sm+slot*INPUT+g*8192,p.a+MODE,bar+slot,TA?m:ki,ch*p.N+(TA?ki:m));
  tma_load_2d(sm+slot*INPUT+16384+g*8192,p.b+MODE,bar+slot,TB?n:ki,ch*p.N+(TB?ki:n));
 }
}

template<int MODE,int PLANE> TMN_DEVI void load_epi(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
 if constexpr(PLANE==0)mbar_arrive_expect_tx(bar+6,98304);
 for(int wm=0;wm<2;++wm)for(int wn=0;wn<2;++wn){
  int q=(wm*2+wn)*8192;
  if constexpr(PLANE<2)tma_load_2d(sm+PLANE*32768+q,&p.premap,bar+6,ni+64*wn,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);
  else tma_load_2d(sm+65536+q,&p.maskmap,bar+6,ni+64*wn,mi+64*wm);
 }
}
template<int MODE> TMN_DEVI void consume(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr int TA=MODE==1,TB=MODE!=2;
 const int WG=threadIdx.x/128;
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;float v[64]={};
 if(threadIdx.x==0){load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);load_input<MODE>(p,sm,bar,ch,mi,ni,64,1);}

 for(int ki=0,it=0;ki<p.N;ki+=64,++it){
  int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);__syncthreads();
  // N is divisible by 3*64: slot0/slot1 retire before the last two steps.
  if(threadIdx.x==0){
   if(ki==p.N-128)load_epi<MODE,0>(p,sm,bar,ch,mi,ni);
   if(ki==p.N-64)load_epi<MODE,1>(p,sm,bar,ch,mi,ni);
  }

  if(threadIdx.x==0 && ki+128<p.N)load_input<MODE>(p,sm,bar,ch,mi,ni,ki+128,(it+2)%SLOTS);
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma128_off<k*(TA?2048:32),k*(TB?2048:32),TA,TB>(v,smem_desc(smem_u32(sm+slot*INPUT+WG*8192),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+16384),TB?8192:16,1024,1),it>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);

 }
 // Both compute groups have now retired slot2.
 __syncthreads();
 if(threadIdx.x==0)load_epi<MODE,2>(p,sm,bar,ch,mi,ni);
 mbar_wait(bar+6,0);__syncthreads();

 static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  uint32_t off=(WG*2+c/64)*8192+swz128(r,(c%64)*2);
  uint32_t raw=pack_bf16(v[j],v[j+1]),masked,mask=*reinterpret_cast<uint32_t*>(sm+65536+off);
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
  uint32_t gr=*reinterpret_cast<uint32_t*>(sm+off),pr=*reinterpret_cast<uint32_t*>(sm+32768+off);
  float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
  uint32_t dp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  uint32_t dg=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
  *reinterpret_cast<uint32_t*>(sm+off)=dp;*reinterpret_cast<uint32_t*>(sm+32768+off)=dg;
 });
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 consume<MODE>(p,sm,bar,ch,mi,ni);
 fence_proxy_async();__syncthreads();
 if(threadIdx.x==0){
  constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);int outch=ch+HALF*D;
  for(int g=0;g<2;++g)for(int wm=0;wm<2;++wm)for(int wn=0;wn<2;++wn){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(p.outmap+2*SIDE+g),"r"(smem_u32(sm+g*32768+(wm*2+wn)*8192)),"r"(ni+wn*64),"r"(outch*p.N+mi+wm*64):"memory");
  }
  tma_store_commit();tma_store_wait_all();
 }
}
extern "C" __global__ __launch_bounds__(256,2)
void mw_wide_staged_epilogue_gp(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tiles=p.N/128;int half=blockIdx.x/(2*D*tiles*tiles),rem=blockIdx.x%(2*D*tiles*tiles),ch=rem/(2*tiles*tiles),mode=2*half+rem%2,tile=(rem/2)%(tiles*tiles);
 int mi=(tile/tiles)*128,ni=(tile%tiles)*128;
 if(threadIdx.x==0){for(int i=0;i<7;++i)mbar_init(bar+i,(i>=3&&i<6)?2:1);fence_barrier_init();}__syncthreads();
 if(mode==0)run<0>(p,sm,bar,ch,mi,ni);
 else if(mode==1)run<1>(p,sm,bar,ch,mi,ni);
 else if(mode==2)run<2>(p,sm,bar,ch,mi,ni);
 else run<3>(p,sm,bar,ch,mi,ni);
}
