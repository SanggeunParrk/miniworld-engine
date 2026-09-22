// SPDX-License-Identifier: Apache-2.0
// Training extension using Anthropic-derived Hopper WGMMA/TMA primitives.
#include <cooperative_groups.h>
#include "common_recompute.cuh"
#ifndef B7_PREFETCH
#define B7_PREFETCH 0
#endif
constexpr int RING_CHUNK=B7_RING_CHUNK,RING_SLOTS=65536/RING_CHUNK,RING_PHASES=131072/RING_CHUNK,RING_H=RING_CHUNK/8192;
constexpr int THREADS=384,SMEM=114688;
#ifndef B7_CONSUMERS
#define B7_CONSUMERS 8
#endif
#ifndef B7_REUSE_DP
#define B7_REUSE_DP 0
#endif
constexpr int SOURCES=16,CONSUMERS=B7_CONSUMERS,GROUP=SOURCES+CONSUMERS/2,RINGS=B7_RING_DEPTH;
#define allsync() named_bar_sync(1+threadIdx.x/128,128)
struct Params{CUtensorMap xn,wp,dl,dr,dg,wgate,x,res,dx,wt[4],ringstore;const __nv_bfloat16* mask;const float *gamma,*beta;__nv_bfloat16 *dxptr,*dw;float *dgam,*dbeta,*partw,*partln;unsigned int* counts;int M,tiles;uint8_t *ring,*xring;unsigned* flags;};
#include "single_wg.inc"
TMN_DEVI void store2d(const CUtensorMap* map,const void* src,int c,int r){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");}
TMN_DEVI void multicast_xn(void* dst,const CUtensorMap* map,uint64_t* bar,int c,int row){asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%3,%4}],[%2],%5;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c),"r"(row),"h"(uint16_t((1<<B7_HW_CLUSTER)-1)):"memory");}
constexpr int SG=0,LN=0,XN=32768,WGATE=49152,WDX=0,SRC_SG=16384,SRC_XN=32768;
TMN_DEVI void csync(){named_bar_sync(15,512);asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;":::"memory");}
TMN_DEVI uint32_t remote_addr(const void* ptr,int rank){uint32_t v;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(v):"r"(smem_u32(ptr)),"r"(rank));return v;}
TMN_DEVI void signal_peer(uint64_t* bar,int peer){asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote_addr(bar,peer)):"memory");}
TMN_DEVI void send_bulk(void* dst,void* src,uint64_t* bar,int bytes,int peer){asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(remote_addr(dst,peer)),"r"(smem_u32(src)),"r"(bytes),"r"(remote_addr(bar,peer)):"memory");}
TMN_DEVI void input32(float (&d)[16],uint64_t a,uint64_t b,int accumulate){asm volatile("{ .reg .pred p; setp.ne.b32 p, %18, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 1, 0; }":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]):"l"(a),"l"(b),"r"(accumulate));}
TMN_DEVI void gate32(float (&d)[16],uint64_t a,uint64_t b,int accumulate){asm volatile("{ .reg .pred p; setp.ne.b32 p, %18, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0; }":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]):"l"(a),"l"(b),"r"(accumulate));}
TMN_DEVI void multicast_cons(void* dst,const CUtensorMap* map,uint64_t* bar,int c,int row){asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%3,%4}],[%2],%5;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c),"r"(row),"h"(uint16_t(240)):"memory");}
TMN_DEVI void bulk_load(void* dst,const void* src,int bytes,uint64_t* bar){asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(dst)),"l"(src),"r"(bytes),"r"(smem_u32(bar)):"memory");}
TMN_DEVI void bulk_store(void* dst,const void* src,int bytes){asm volatile("{.reg .b64 pol;createpolicy.fractional.L2::evict_last.b64 pol,1.0;cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0],[%1],%2,pol;}"::"l"(dst),"r"(smem_u32(src)),"r"(bytes):"memory");}
TMN_DEVI void ring_wait(const unsigned* ptr,unsigned want){unsigned got;do{asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(got):"l"(ptr):"memory");if(got<want)__nanosleep(32);}while(got<want);}
TMN_DEVI void publish(unsigned* ptr,unsigned v){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(ptr),"r"(v):"memory");}

TMN_DEVI void pair_gp(float (&a)[32],uint32_t (&xn)[8][4],uint64_t desc){
 uint32_t lo=desc,hi=desc>>32;fence_regs(a);wgmma_fence();
 static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;wgmma_m64n64k16_rs_off<(k/4)*8192+(k%4)*32>(a,xn[k],lo,hi,k>0);});wgmma_commit();
}
TMN_DEVI void pair_glu(float (&a)[32],uint8_t* s,uint8_t* sg,uint32_t ma,uint32_t mb){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
 #pragma unroll
 for(int q=0;q<2;++q){uint32_t dy[4],dg[4],dp[4];ldsm_x4_t(dy,smem_u32(s+16384)+swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
  #pragma unroll
  for(int j=0;j<4;++j){uint32_t masked,m=(j&1)?mb:ma;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy[j]),"r"(m));
   uint32_t gr=pack_bf16(a[q*8+j*2],a[q*8+j*2+1]),pr=pack_bf16(a[(q+2)*8+j*2],a[(q+2)*8+j*2+1]);float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
   float dpa=bf16lo(masked)*ga,dpb=bf16hi(masked)*gb;dg[j]=pack_bf16((dpa*pa)*(1.f-ga),(dpb*pb)*(1.f-gb));dp[j]=pack_bf16(dpa,dpb);
  }
  uint32_t addr=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);stsm_x4_t(smem_u32(sg)+addr,dg[0],dg[1],dg[2],dg[3]);stsm_x4_t(smem_u32(sg+4096)+addr,dp[0],dp[1],dp[2],dp[3]);
 }
}

TMN_DEVI void pair_glu_packed(uint32_t (&a)[16],uint8_t* s,uint8_t* sg,uint32_t ma,uint32_t mb){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
 #pragma unroll
 for(int q=0;q<2;++q){uint32_t dy[4],dg[4],dp[4];ldsm_x4_t(dy,smem_u32(s+16384)+swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
  #pragma unroll
  for(int j=0;j<4;++j){uint32_t masked,m=(j&1)?mb:ma;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy[j]),"r"(m));
   uint32_t gr=a[q*4+j],pr=a[(q+2)*4+j];float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
   float dpa=bf16lo(masked)*ga,dpb=bf16hi(masked)*gb;dg[j]=pack_bf16((dpa*pa)*(1.f-ga),(dpb*pb)*(1.f-gb));dp[j]=pack_bf16(dpa,dpb);
  }
  uint32_t addr=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);stsm_x4_t(smem_u32(sg)+addr,dg[0],dg[1],dg[2],dg[3]);stsm_x4_t(smem_u32(sg+4096)+addr,dp[0],dp[1],dp[2],dp[3]);
 }
}

// Compact ring planes: left dg, left dp, right dg, right dp (32 KiB each).
TMN_DEVI void publish_tile(const Params& p,uint8_t* src,int cid,int rs,int rank){
#if B7_RING_TENSOR
 // The 3D tensor map scatters two contiguous 4 KiB shared tiles in one transfer.
 asm volatile("{.reg .b64 pol;createpolicy.fractional.L2::evict_last.b64 pol,1.0;cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%2,%3,%4}],[%1],pol;}"::"l"(&p.ringstore),"r"(smem_u32(src)),"r"(0),"r"((rank%8)*8),"r"((cid*RINGS+rs)*4+(rank/8)*2):"memory");
#else
 uint8_t* dst=p.ring+(cid*RINGS+rs)*131072+(rank/8)*65536+(rank%8)*4096;
 bulk_store(dst,src,4096);bulk_store(dst+32768,src+4096,4096);
#endif
}
TMN_DEVI void source_producer(const Params& p,uint8_t* sm,uint64_t* bar){
 if(threadIdx.x>=64)return;int rank=blockIdx.x%GROUP,cid=blockIdx.x/GROUP,groups=gridDim.x/GROUP;
 if(threadIdx.x==32){int rounds=(p.tiles-1-cid)/groups+1;
  for(int round=0;round<rounds;round+=2){bool second=round+1<rounds;mbar_wait(bar+19,(round/2)&1);
   publish_tile(p,sm+98304,cid,round%RINGS,rank);
   if(second)publish_tile(p,sm+106496,cid,(round+1)%RINGS,rank);
   tma_store_commit();tma_store_wait_all();
   publish(p.flags+(cid*RINGS+round%RINGS)*(SOURCES+2)+1+rank,round+1);
   
   mbar_arrive(bar+20);
  }return;
 }if(threadIdx.x)return;
 for(int round=0,tile=cid;tile<p.tiles;tile+=groups,++round){int slot=round%4,rs=round%RINGS;
  if(round>=4)mbar_wait(bar+4+slot,((round/4)-1)&1);
  if(round>=RINGS)ring_wait(p.flags+(cid*RINGS+rs)*(SOURCES+2)+SOURCES+1,round-RINGS+1);
  uint8_t* dst=sm+slot*20480;mbar_arrive_expect_tx(bar+slot,20480);
#if B7_MULTICAST
  signal_peer(bar+32+slot,0);
  if(rank%B7_HW_CLUSTER==0){mbar_wait(bar+32+slot,(round/4)&1);for(int c=0;c<2;++c)multicast_xn(dst+c*8192,&p.xn,bar+slot,c*64,tile*64);}
#else
  for(int c=0;c<2;++c)tma_load_2d(dst+c*8192,&p.xn,bar+slot,c*64,tile*64);
#endif
  tma_load_2d(dst+16384,rank<8?&p.dl:&p.dr,bar+slot,tile*64,(rank%8)*32);
 }
}
#if B7_TRANSPOSE_PART
TMN_DEVI void store_pair_dw(const Params& p,float (&v)[64],int part,int rank,uint8_t* sm){
 int tid=threadIdx.x%128,lane=threadIdx.x%32,w=tid/32;float* dst=p.partw+(part*SOURCES+rank)*8192;float* tmp=reinterpret_cast<float*>(sm);
 static_for<16>([&](auto qq){constexpr int q=decltype(qq)::value;int row=w*16+lane/4,c=q*8+2*(lane%4);
  tmp[(row/32)*4096+c*32+row%32]=v[q*4];tmp[(row/32)*4096+(c+1)*32+row%32]=v[q*4+1];
  tmp[(row/32)*4096+c*32+row%32+8]=v[q*4+2];tmp[(row/32)*4096+(c+1)*32+row%32+8]=v[q*4+3];
 });allsync();
 for(int i=tid;i<8192;i+=128)dst[i]=tmp[i];
}
#else
TMN_DEVI void store_pair_dw(const Params& p,float (&v)[64],int part,int rank,uint8_t* sm){
 int lane=threadIdx.x%32,w=(threadIdx.x%128)/32;float* dst=p.partw+(part*SOURCES+rank)*8192;
 static_for<16>([&](auto qq){constexpr int q=decltype(qq)::value;int row=w*16+lane/4,c=q*8+2*(lane%4);stg64f(dst+row*128+c,v[q*4],v[q*4+1]);stg64f(dst+(row+8)*128+c,v[q*4+2],v[q*4+3]);});
}
#endif
TMN_DEVI void small_dw(float (&d)[64],uint64_t a,uint64_t b,int accumulate){
 static_for<2>([&](auto hh){constexpr int h=decltype(hh)::value;auto& part=*reinterpret_cast<float(*)[32]>(d+h*32);mma_recompute(part,a,b+h*(8192>>4),accumulate);});
}
TMN_DEVI void input64(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 1, 0; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));

}

TMN_DEVI void source_compute(const Params& p,uint8_t* sm,uint64_t* bar){
 int rank=blockIdx.x%GROUP,cid=blockIdx.x/GROUP,groups=gridDim.x/GROUP,tid=threadIdx.x%128,lane=tid%32,w=tid/32;
 int rounds=(p.tiles-1-cid)/groups+1,splitpoint=((rounds+3)/4)*2;
 float dw[64]={};uint64_t desc=smem_desc(smem_u32(sm+81920),16,1024,1);
 for(int round=0;round<rounds;round+=2){int slot=round%4,tile=cid+round*groups;bool second=round+1<rounds;
  
  uint8_t* s0=sm+slot*20480;uint8_t* s1=sm+((slot+1)%4)*20480;uint8_t* sg0=sm+98304;uint8_t* sg1=sm+106496;
  mbar_wait(bar+slot,(round/4)&1);uint32_t xn[8][4];load_frag_bf16<8,8192>(xn,smem_u32(s0),w*16,lane);
  int ra=w*16+lane/4;uint32_t ma=uint32_t(__bfloat16_as_ushort(p.mask[tile*64+ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(p.mask[tile*64+ra+8]))*0x10001u;
  float a1[32]={};
#if B7_PACK_GP
  uint32_t packed_a0[16];
  {float a0[32]={};pair_gp(a0,xn,desc);wgmma_wait<0>();fence_regs(a0);
   static_for<16>([&](auto jj){constexpr int j=decltype(jj)::value;packed_a0[j]=pack_bf16(a0[j*2],a0[j*2+1]);});}
#else
  float a0[32]={};pair_gp(a0,xn,desc);wgmma_wait<0>();fence_regs(a0);
#endif
  if(second){mbar_wait(bar+slot+1,(round/4)&1);load_frag_bf16<8,8192>(xn,smem_u32(s1),w*16,lane);pair_gp(a1,xn,desc);}
  if(round>0)mbar_wait(bar+20,((round/2)-1)&1);
#if B7_PACK_GP
  pair_glu_packed(packed_a0,s0,sg0,ma,mb);
#else
  pair_glu(a0,s0,sg0,ma,mb);
#endif
  if(second){wgmma_wait<0>();fence_regs(a1);ma=uint32_t(__bfloat16_as_ushort(p.mask[(tile+groups)*64+ra]))*0x10001u;mb=uint32_t(__bfloat16_as_ushort(p.mask[(tile+groups)*64+ra+8]))*0x10001u;pair_glu(a1,s1,sg1,ma,mb);}
  fence_proxy_async();allsync();
  // Both consumers use the same derivatives. Publish while local dW executes.
  if(tid==0)mbar_arrive(bar+19);
  fence_regs(dw);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;small_dw(dw,smem_desc(smem_u32(sg0+k*32),16,1024,1),smem_desc(smem_u32(s0+k*2048),8192,1024,1),round>0||k>0);});
  if(second)static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;small_dw(dw,smem_desc(smem_u32(sg1+k*32),16,1024,1),smem_desc(smem_u32(s1+k*2048),8192,1024,1),1);});
  wgmma_commit();
  
  wgmma_wait<0>();fence_regs(dw);
  allsync();if(tid==0){mbar_arrive(bar+4+slot);if(second)mbar_arrive(bar+4+slot+1);}
 }
 store_pair_dw(p,dw,cid,rank,sm);
}
TMN_DEVI void input128(float (&d)[64],uint64_t a,uint64_t b,int accumulate){
 static_for<2>([&](auto hh){constexpr int h=decltype(hh)::value;auto& part=*reinterpret_cast<float(*)[32]>(d+h*32);input64(part,a,b+h*(8192>>4),accumulate);});
}
#include "pair_producer.inc"
TMN_DEVI void wide_gate(float (&a)[64],uint32_t (&xn)[8][4],uint8_t* wb,uint8_t* xs){
 fence_regs(a);wgmma_fence();static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;
 uint64_t b=smem_desc(smem_u32(wb+(k/4)*16384+(k%4)*32),16,1024,1),x=smem_desc(smem_u32(xs+(k/4)*8192+(k%4)*32),16,1024,1);
 asm volatile("{ .reg .pred p;setp.ne.b32 p,%66,0;wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},%64,%65,p,1,1,0,0; }":"+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]),"+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]),"+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23]),"+f"(a[24]),"+f"(a[25]),"+f"(a[26]),"+f"(a[27]),"+f"(a[28]),"+f"(a[29]),"+f"(a[30]),"+f"(a[31]),"+f"(a[32]),"+f"(a[33]),"+f"(a[34]),"+f"(a[35]),"+f"(a[36]),"+f"(a[37]),"+f"(a[38]),"+f"(a[39]),"+f"(a[40]),"+f"(a[41]),"+f"(a[42]),"+f"(a[43]),"+f"(a[44]),"+f"(a[45]),"+f"(a[46]),"+f"(a[47]),"+f"(a[48]),"+f"(a[49]),"+f"(a[50]),"+f"(a[51]),"+f"(a[52]),"+f"(a[53]),"+f"(a[54]),"+f"(a[55]),"+f"(a[56]),"+f"(a[57]),"+f"(a[58]),"+f"(a[59]),"+f"(a[60]),"+f"(a[61]),"+f"(a[62]),"+f"(a[63]):"l"(x),"l"(b),"r"(int(k>0)));
 });wgmma_commit();wgmma_wait<0>();fence_regs(a);
}

TMN_DEVI void consumer_compute(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma,const float* beta){
 int g=threadIdx.x/128-1,rank=2*(blockIdx.x%GROUP-SOURCES)+g,cid=blockIdx.x/GROUP,clusters=gridDim.x/GROUP,round=0,tid=threadIdx.x%128,lane=tid%32,w=tid/32,ra=w*16+lane/4,rb=ra+8;float run_g=0,run_b=0;
 for(int base=cid+(rank-g)*clusters;base<p.tiles;base+=CONSUMERS*clusters,++round){int tile=base+g*clusters,row=tile*64;bool valid=tile<p.tiles;
  uint32_t packed_dx[32];
  {
  float acc[64]={};fence_regs(acc);wgmma_fence();
  for(int batch=0;batch<8;++batch){
   for(int sub=0;sub<2;++sub){int stage=batch*2+sub,slot=batch%2,ws=stage%2;
    if(sub==0)mbar_wait(bar+8+g*4+slot,(batch/2)&1);
    mbar_wait(bar+ws,(stage/2)&1);
    {uint8_t* ds=sm+g*32768+slot*16384+sub*8192;
     static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
      input128(acc,smem_desc(smem_u32(ds+k*2048),16,1024,1),smem_desc(smem_u32(sm+65536+ws*16384+k*32),16,1024,1),stage>0||k>0);
     });wgmma_commit();
    }
   }
   wgmma_wait<0>();fence_regs(acc);allsync();
   if(tid==0){for(int ws=0;ws<2;++ws)mbar_arrive(bar+2+ws);mbar_arrive(bar+10+g*4+batch%2);}
  }
  mbar_wait(bar+19,round&1);
  static_for<4>([&](auto hh){constexpr int h=decltype(hh)::value;float gate[16]={};fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;
    gate32(gate,smem_desc(smem_u32(sm+g*32768+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+65536+(k/4)*16384+h*4096+(k%4)*32),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
   static_for<8>([&](auto jj){constexpr int j=decltype(jj)::value;
    packed_dx[h*8+j]=pack_bf16(acc[h*16+j*2]+math::round_bf16(gate[j*2]),acc[h*16+j*2+1]+math::round_bf16(gate[j*2+1]));
   });
  });
  }
  #define B7_ACC(i) (((i)&1)?bf16hi(packed_dx[(i)/2]):bf16lo(packed_dx[(i)/2]))
  allsync();
  // dGate is dead after WGMMA; reuse its 16 KiB for residual, not another buffer.
  if(tid==0){mbar_arrive_expect_tx(bar+20+g,16384);for(int k=0;k<2;++k)tma_load_2d(sm+g*32768+k*8192,&p.res,bar+20+g,k*64,row);}
  uint8_t* lnsm=sm+g*32768+16384;
  LnStats st;
  {uint32_t raw[8][4];load_frag_bf16<8,8192>(raw,smem_u32(lnsm),w*16,lane);st=ln_stats_only<8>(raw,gamma,beta,lane,1e-5f);}
  float mu[2]={st.mA,st.mB},rs[2]={st.rA,st.rB},s1[2][2]={},s2[2][2]={};
  static_for<8>([&](auto qq){constexpr int q=decltype(qq)::value;
   uint32_t rawq[4];ldsm_x4(rawq,smem_u32(lnsm)+(q/4)*8192+swz128(w*16+(lane&7)+((lane&8)?8:0),((q%4)*16+((lane&16)?8:0))*2));
   static_for<4>([&](auto jj){constexpr int j=decltype(jj)::value;int rr=j&1,c=q*16+2*(lane%4)+8*(j>>1);
    float xa=__fmul_rn(__fsub_rn(bf16lo(rawq[j]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(rawq[j]),mu[rr]),rs[rr]);
    float ha=B7_ACC(q*8+j*2)*gamma[c],hb=B7_ACC(q*8+j*2+1)*gamma[c+1];s1[q/4][rr]+=ha*xa+hb*xb;s2[q/4][rr]+=ha+hb;
   });
  });
  float c1[2],c2[2];
  static_for<2>([&](auto ii){constexpr int i=decltype(ii)::value;c1[i]=quad_sum(s1[0][i])/128.f+quad_sum(s1[1][i])/128.f;c2[i]=quad_sum(s2[0][i])/128.f+quad_sum(s2[1][i])/128.f;});
  allsync(); // All LN source reads finish before in-place dx stores.
  float* tmp=reinterpret_cast<float*>(sm+98304+g*4096);
  mbar_wait(bar+20+g,round&1);
  static_for<8>([&](auto qq){constexpr int q=decltype(qq)::value;
   uint32_t rawq[4];ldsm_x4(rawq,smem_u32(lnsm)+(q/4)*8192+swz128(w*16+(lane&7)+((lane&8)?8:0),((q%4)*16+((lane&16)?8:0))*2));
   static_for<2>([&](auto pp){constexpr int pair=decltype(pp)::value;int j=pair*2,c=q*16+2*(lane%4)+8*pair;
    float xaa=__fmul_rn(__fsub_rn(bf16lo(rawq[pair*2]),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(bf16hi(rawq[pair*2]),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(bf16lo(rawq[pair*2+1]),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(bf16hi(rawq[pair*2+1]),mu[1]),rs[1]);
    float da=B7_ACC(q*8+pair*4),db=B7_ACC(q*8+pair*4+1),dc=B7_ACC(q*8+pair*4+2),dd=B7_ACC(q*8+pair*4+3),ga=gamma[c],gb=gamma[c+1];
    uint32_t oa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
    uint32_t ob=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
    uint8_t* out=lnsm-16384+(q/4)*8192;uint8_t* res=lnsm-16384+(q/4)*8192;
    uint32_t resa=pair_get(res,ra,c%64),resb=pair_get(res,rb,c%64);
    *reinterpret_cast<uint32_t*>(out+swz128(ra,(c%64)*2))=pack_bf16(bf16lo(oa)+bf16lo(resa),bf16hi(oa)+bf16hi(resa));
    *reinterpret_cast<uint32_t*>(out+swz128(rb,(c%64)*2))=pack_bf16(bf16lo(ob)+bf16lo(resb),bf16hi(ob)+bf16hi(resb));
    float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
    if(lane<4){tmp[w*256+c]=dga;tmp[w*256+c+1]=dgb;tmp[w*256+128+c]=dba;tmp[w*256+128+c+1]=dbb;}
   });
  });allsync();
  run_g+=(tmp[tid]+tmp[256+tid])+(tmp[512+tid]+tmp[768+tid]);
  run_b+=(tmp[128+tid]+tmp[384+tid])+(tmp[640+tid]+tmp[896+tid]);
  fence_proxy_async();allsync();
  if(tid==0&&valid){store2d(&p.dx,lnsm-16384,0,row);store2d(&p.dx,lnsm-8192,64,row);tma_store_commit();}
   if(tid==0&&valid)tma_store_wait_all();allsync();
  
  allsync();if(tid==0)mbar_arrive(bar+18);
 }
 p.partln[(cid*CONSUMERS+rank)*256+tid]=run_g;p.partln[(cid*CONSUMERS+rank)*256+128+tid]=run_b;
}

#undef B7_ACC
extern "C" __global__ __maxnreg__(B7_REG_CAP) __cluster_dims__(B7_HW_CLUSTER,1,1)
void b7_joint(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[64];__shared__ float gamma[128],beta[128];
 int wi=threadIdx.x/128,rank=blockIdx.x%GROUP,cid=blockIdx.x/GROUP,groups=gridDim.x/GROUP;
 if(threadIdx.x<128){gamma[threadIdx.x]=p.gamma[threadIdx.x];beta[threadIdx.x]=p.beta[threadIdx.x];}
 if(threadIdx.x==0){for(int i=0;i<64;++i)mbar_init(bar+i,(rank>=SOURCES&&(i==2||i==3||i==18))?2:1);fence_barrier_init();}__syncthreads();
#if B7_MULTICAST
 cooperative_groups::this_cluster().sync();
#endif
 if(rank<SOURCES){
  if(wi==2){setmaxnreg_dec<24>();}else{
  if(threadIdx.x==0){mbar_arrive_expect_tx(bar+18,16384);for(int k=0;k<2;++k)tma_load_2d(sm+81920+k*8192,&p.wp,bar+18,k*64,rank*64);}
  mbar_wait(bar+18,0);named_bar_sync(14,256);
  if(wi==0){setmaxnreg_dec<B7_PRODUCER_REGS>();source_producer(p,sm,bar);}else{setmaxnreg_inc<216-B7_PRODUCER_REGS>();source_compute(p,sm,bar);}
 }}else{
  if(wi==0){setmaxnreg_dec<32>();pair_producer(p,sm,bar);}else{setmaxnreg_inc<B7_COMPUTE_REGS>();consumer_compute(p,sm,bar,gamma,beta);}
 }
 cooperative_groups::this_grid().sync();
 for(int i=blockIdx.x*THREADS+threadIdx.x;i<131328;i+=gridDim.x*THREADS){float v=0;
  if(i<131072){
#if B7_REDUCE_NATIVE
   int rk=i/8192,j=i%8192;
#if B7_TRANSPOSE_PART
   int half=j/4096,c=(j%4096)/32,h=j%32;
#else
   int half=j/4096,c=j%128,h=(j%4096)/128;
#endif
   int kind=(rk/8)*2+(half==0),out=kind*32768+c*256+(rk%8)*32+h;
#else
   int kind=i/32768,c=(i/256)%128,h=i%256,rk=(kind/2)*8+h/32,out=i;
#if B7_TRANSPOSE_PART
   int j=((kind&1)?0:4096)+c*32+h%32;
#else
   int j=((kind&1)?h%32:32+h%32)*128+c;
#endif
#endif
   for(int a=0;a<groups;++a)v+=reinterpret_cast<volatile float*>(p.partw)[(a*SOURCES+rk)*8192+j];p.dw[out]=__float2bfloat16_rn(v);
  }else{int c=i-131072;for(int a=0;a<groups*CONSUMERS;++a)v+=reinterpret_cast<volatile float*>(p.partln)[a*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}
 }
 for(int i=blockIdx.x*THREADS+threadIdx.x;i<groups*RINGS*(SOURCES+2);i+=gridDim.x*THREADS)p.flags[i]=0;
 // Cooperative grid barrier has no software counters to reset.
}
