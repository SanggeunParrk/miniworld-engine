#define WGRAD_SLICES 2
// SPDX-License-Identifier: Apache-2.0
// Training extension of Anthropic v5: two GLU/TMA producer WGs, one MMA WG.
#include "warp_primitives.cuh"
#ifndef UCOUNT
#define UCOUNT 132
#endif
#ifndef DW_SPLITS
#define DW_SPLITS 8
#endif
#ifndef PART_ONLY
#define PART_ONLY 2
#endif
#ifndef DW_PREFETCH
#define DW_PREFETCH 3
#endif
constexpr int DWCOUNT=8*DW_SPLITS,DXCOUNT=UCOUNT-DWCOUNT;
constexpr int DW_STAGES=5,DW_SLOT=40960,DX_SLOT=114688;
static_assert(DXCOUNT>0 && DW_PREFETCH<DW_STAGES,"invalid persistent partition");
struct Params{
 CUtensorMap dl,dr,pre,xn,dg,wlg,wl,wrg,wr,wgate,x,res,dx;
 const __nv_bfloat16* mask;const float *mean,*rs,*gamma;
 __nv_bfloat16* dw;float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;__nv_bfloat16 *debugdc,*debugxn;int M,L,tiles;
};
struct Barriers{uint64_t tx[5],ready[5],empty[5];};
TMN_DEVI void psync(){named_bar_sync(1,256);}
TMN_DEVI void csync(){named_bar_sync(2,128);}
TMN_DEVI int cid(){return int(threadIdx.x)-256;}
TMN_DEVI void ready(const Params& p,Barriers* b,int slot){fence_proxy_async();psync();if(threadIdx.x==0)mbar_arrive(b->ready+slot);}
TMN_DEVI void release(Barriers* b,int slot){fence_proxy_async();csync();if(cid()==0)mbar_arrive(b->empty+slot);}
TMN_DEVI void store2d(const CUtensorMap* map,const void* src,int c,int r){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2,%3}], [%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");
}

TMN_DEVI void glu64(const Params& p,uint8_t* s,int row){
 int tid=threadIdx.x,mr=row+(tid%32)*2;
 float ma=__bfloat162float(p.mask[mr]),mb=__bfloat162float(p.mask[mr+1]);
 uint32_t gg[8],pp[8];
 static_for<8>([&](auto qi){constexpr int q=decltype(qi)::value;int i=tid+q*256,c=i/32,r=(i%32)*2;
  uint32_t dy=pair_get(s+16384,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r);
  uint32_t masked=pack_bf16(bf16lo(dy)*ma,bf16hi(dy)*mb);
  float ga=math::sigmoid_div(bf16lo(gl)),gb=math::sigmoid_div(bf16hi(gl));
  gg[q]=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb));
  pp[q]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
 });
 psync();
 static_for<8>([&](auto qi){constexpr int q=decltype(qi)::value;int i=tid+q*256,c=i/32,r=(i%32)*2;
  *reinterpret_cast<uint32_t*>(s+swz128(c,r*2))=gg[q];
  *reinterpret_cast<uint32_t*>(s+8192+swz128(c,r*2))=pp[q];
 });
}
TMN_DEVI void load_dw(const Params& p,uint8_t* s,Barriers* b,int slot,int row,int group){
 if(threadIdx.x)return;int side=group/4,h=(group%4)*64;
 mbar_arrive_expect_tx(b->tx+slot,40960);
 tma_load_2d(s,&p.pre,b->tx+slot,row,side*512+2*h);
 tma_load_2d(s+16384,side?&p.dr:&p.dl,b->tx+slot,row,h);
 for(int c=0;c<2;++c)tma_load_2d(s+24576+c*8192,&p.xn,b->tx+slot,c*64,row);
}
TMN_DEVI void dw_producer(const Params& p,uint8_t* sm,Barriers* b){
 int group=blockIdx.x/DW_SPLITS,split=blockIdx.x%DW_SPLITS,rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS;
 for(int r=0;r<DW_PREFETCH&&r<rounds;++r)load_dw(p,sm+r*DW_SLOT,b,r,(split+r*DW_SPLITS)*64,group);
 for(int r=0;r<rounds;++r){
  int future=r+DW_PREFETCH;
  if(future<rounds){int fs=future%DW_STAGES;if(future>=DW_STAGES)mbar_wait(b->empty+fs,((future/DW_STAGES)-1)&1);load_dw(p,sm+fs*DW_SLOT,b,fs,(split+future*DW_SPLITS)*64,group);}
  int slot=r%DW_STAGES;mbar_wait(b->tx+slot,(r/DW_STAGES)&1);glu64(p,sm+slot*DW_SLOT,(split+r*DW_SPLITS)*64);ready(p,b,slot);
 }
}
TMN_DEVI void dw_consumer(const Params& p,uint8_t* sm,Barriers* b){
 int group=blockIdx.x/DW_SPLITS,split=blockIdx.x%DW_SPLITS,tid=cid(),lane=tid%32,w=tid/32;
 float acc[2][64]={};int r=0,rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,segmentRounds=(rounds+1)/2;
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){
  int slot=r%DW_STAGES;uint8_t* s=sm+slot*DW_SLOT;mbar_wait(b->ready+slot,(r/DW_STAGES)&1);
  static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;fence_regs(acc[n]);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_weight128(acc[n],smem_desc(smem_u32(s+n*8192+k*32),16,1024,1),smem_desc(smem_u32(s+24576+k*2048),8192,1024,1),r%segmentRounds>0||k>0);});
  });wgmma_commit();wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);release(b,slot);
 if((r+1)%segmentRounds==0 || tile+DW_SPLITS>=p.tiles){
 static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;float* out=p.partw+((group*DW_SPLITS+split)*2+r/segmentRounds)*16384+n*8192;
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);
   stg64f(out+rr*128+c,acc[n][q*4],acc[n][q*4+1]);stg64f(out+(rr+8)*128+c,acc[n][q*4+2],acc[n][q*4+3]);
  });
 }); }
 }
}

TMN_DEVI int dxid(){return int(threadIdx.x)-128;}
TMN_DEVI void dxsync(){named_bar_sync(2,256);}
TMN_DEVI void dxrelease(Barriers* b,int slot){fence_proxy_async();dxsync();if(dxid()==0)mbar_arrive(b->empty+slot);}
constexpr int FRONT_SLOT=65536,PGRAD=131072,PWEIGHT=196608;
TMN_DEVI void load_gate128(const Params& p,uint8_t* sm,Barriers* b,int row){
 uint8_t* s=sm+FRONT_SLOT;mbar_arrive_expect_tx(b->tx+1,65536);
 for(int m=0;m<2;++m)for(int k=0;k<2;++k)tma_load_2d(s+m*16384+k*8192,&p.dg,b->tx+1,k*64,row+m*64);
 for(int k=0;k<2;++k)for(int n=0;n<2;++n)tma_load_2d(s+32768+k*16384+n*8192,&p.wgate,b->tx+1,k*64,n*64);
}
TMN_DEVI void load_glu128(const Params& p,uint8_t* sm,Barriers* b,int h,int row,int side){
 int slot=h&1;uint8_t* s=sm+slot*FRONT_SLOT;mbar_arrive_expect_tx(b->tx+slot,65536);
 for(int m=0;m<2;++m){tma_load_2d(s+m*16384,&p.pre,b->tx+slot,row+m*64,side*512+h*128);tma_load_2d(s+32768+m*8192,side?&p.dr:&p.dl,b->tx+slot,row+m*64,h*64);}
 for(int n=0;n<2;++n)tma_load_2d(s+49152+n*8192,side?&p.wrg:&p.wlg,b->tx+slot,h*64,n*64);
}
TMN_DEVI void load_pw(const Params& p,uint8_t* sm,Barriers* b,int h,int side){
 int slot=h&1;uint8_t* s=sm+PWEIGHT+slot*16384;mbar_arrive_expect_tx(b->tx+slot+2,16384);
 for(int n=0;n<2;++n)tma_load_2d(s+n*8192,side?&p.wr:&p.wl,b->tx+slot+2,h*64,n*64);
}
TMN_DEVI void load_ln128(const Params& p,uint8_t* sm,Barriers* b,int row){
 mbar_arrive_expect_tx(b->tx,65536);
 for(int m=0;m<2;++m)for(int c=0;c<2;++c){tma_load_2d(sm+m*16384+c*8192,&p.x,b->tx,c*64,row+m*64);tma_load_2d(sm+32768+m*16384+c*8192,&p.res,b->tx,c*64,row+m*64);}
}
TMN_DEVI void dx_producer(const Params& p,uint8_t* sm,Barriers* b){
 if(threadIdx.x)return;int split=blockIdx.x-DWCOUNT,round=0,tiles=p.tiles/2;
 for(int tile=split;tile<tiles;tile+=DXCOUNT,++round){int row=tile*128,ph=round&1;
  if(round==0)load_gate128(p,sm,b,row);
  if(round)mbar_wait(b->empty,ph^1);load_glu128(p,sm,b,0,row,0);
  if(round)mbar_wait(b->empty+2,1);load_pw(p,sm,b,0,0);
  if(round)mbar_wait(b->empty+3,1);load_pw(p,sm,b,1,0);
  mbar_wait(b->empty+1,ph);load_glu128(p,sm,b,1,row,0);
  mbar_wait(b->empty,ph);load_glu128(p,sm,b,2,row,0);
  mbar_wait(b->empty+1,ph^1);load_glu128(p,sm,b,3,row,0);
  mbar_wait(b->empty,ph^1);load_glu128(p,sm,b,0,row,1);
  mbar_wait(b->empty+1,ph);load_glu128(p,sm,b,1,row,1);
  mbar_wait(b->empty+2,0);load_pw(p,sm,b,2,0);
  mbar_wait(b->empty+3,0);load_pw(p,sm,b,3,0);
  mbar_wait(b->empty+2,1);load_pw(p,sm,b,0,1);
  mbar_wait(b->empty+3,1);load_pw(p,sm,b,1,1);
  mbar_wait(b->empty,ph);load_glu128(p,sm,b,2,row,1);
  mbar_wait(b->empty+1,ph^1);load_glu128(p,sm,b,3,row,1);
  mbar_wait(b->empty,ph^1);load_ln128(p,sm,b,row);
  mbar_wait(b->empty+1,ph);if(tile+DXCOUNT<tiles)load_gate128(p,sm,b,(tile+DXCOUNT)*128);
  mbar_wait(b->empty+2,0);load_pw(p,sm,b,2,1);
  mbar_wait(b->empty+3,0);load_pw(p,sm,b,3,1);
 }
}
TMN_DEVI void glu_rows128(const Params& p,uint8_t* sm,int h,int row){
 int tid=dxid();uint8_t* s=sm+(h&1)*FRONT_SLOT;uint32_t gg[16],pp[16];
 float ma0=__bfloat162float(p.mask[row+(tid%32)*2]),mb0=__bfloat162float(p.mask[row+(tid%32)*2+1]);
 float ma1=__bfloat162float(p.mask[row+64+(tid%32)*2]),mb1=__bfloat162float(p.mask[row+64+(tid%32)*2+1]);
 static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int i=tid+q*256,m=i/2048,c=(i%2048)/32,r=(i%32)*2;
  uint32_t dy=pair_get(s+32768+m*8192,c,r),gl=pair_get(s+m*16384,c*2,r),pr=pair_get(s+m*16384,c*2+1,r);
  uint32_t masked=pack_bf16(bf16lo(dy)*(m?ma1:ma0),bf16hi(dy)*(m?mb1:mb0));float ga=math::sigmoid_div(bf16lo(gl)),gb=math::sigmoid_div(bf16hi(gl));
  gg[q]=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb));pp[q]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
 });
 dxsync();
 static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int i=tid+q*256,m=i/2048,c=(i%2048)/32,r=(i%32)*2;
  *reinterpret_cast<uint32_t*>(s+m*8192+swz128(c,r*2))=gg[q];*reinterpret_cast<uint32_t*>(sm+PGRAD+h*16384+m*8192+swz128(c,r*2))=pp[q];
 });fence_proxy_async();dxsync();
}
TMN_DEVI void dx_consumer(const Params& p,uint8_t* sm,Barriers* b,const float* gamma){
 int split=blockIdx.x-DWCOUNT,tid=dxid(),wi=tid/128,lane=tid%32,w=(tid/32)%4,ra=w*16+lane/4,rb=ra+8,round=0;float running=0;
 for(int tile=split;tile<p.tiles/2;tile+=DXCOUNT,++round){
  int row=tile*128,ph=round&1;uint32_t packed[32];float acc[64]={};mbar_wait(b->tx+1,ph);
  {float gate[64]={};uint8_t* s=sm+FRONT_SLOT;fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_gate128(gate,smem_desc(smem_u32(s+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(s+32768+(k/4)*16384+(k%4)*32),16,1024,1),k>0);});
   wgmma_commit();wgmma_wait<0>();fence_regs(gate);static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;packed[q]=pack_bf16(gate[q*2],gate[q*2+1]);});}
  dxrelease(b,1);
  for(int side=0;side<2;++side){
   for(int h=0;h<4;++h){int slot=h&1;mbar_wait(b->tx+slot,ph^(((h+1)/2)&1));glu_rows128(p,sm,h,row);uint8_t* s=sm+slot*FRONT_SLOT;
    fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(s+wi*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+49152+k*32),16,1024,1),side>0||h>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);dxrelease(b,slot);
   }
   for(int h=0;h<4;++h){int slot=h&1;mbar_wait(b->tx+slot+2,h/2);fence_regs(acc);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(sm+PGRAD+h*16384+wi*8192+k*2048),16,1024,1),smem_desc(smem_u32(sm+PWEIGHT+slot*16384+k*32),16,1024,1),1);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);dxrelease(b,slot+2);
   }
  }
  static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;acc[q*2]=__bfloat162float(__float2bfloat16_rn(acc[q*2]+bf16lo(packed[q])));acc[q*2+1]=__bfloat162float(__float2bfloat16_rn(acc[q*2+1]+bf16hi(packed[q])));});
  mbar_wait(b->tx,ph);uint8_t* ln=sm+wi*16384;
  float mu[2]={p.mean[row+wi*64+ra],p.mean[row+wi*64+rb]},rs[2]={p.rs[row+wi*64+ra],p.rs[row+wi*64+rb]},s1[2]={},s2[2]={};
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   static_for<2>([&](auto ri){constexpr int r=decltype(ri)::value;int rr=r?rb:ra;
    float xa=__fmul_rn(__fsub_rn(get(ln+base,rr,cc),mu[r]),rs[r]),xb=__fmul_rn(__fsub_rn(get(ln+base,rr,cc+1),mu[r]),rs[r]);
    float ha=acc[q*4+r*2]*gamma[c],hb=acc[q*4+r*2+1]*gamma[c+1];s1[r]+=ha*xa+hb*xb;s2[r]+=ha+hb;
   });
  });
  float c1[2]={quad_sum(s1[0])/128.f,quad_sum(s1[1])/128.f},c2[2]={quad_sum(s2[0])/128.f,quad_sum(s2[1])/128.f};float* tmp=reinterpret_cast<float*>(sm+PGRAD);
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   float xaa=__fmul_rn(__fsub_rn(get(ln+base,ra,cc),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(get(ln+base,ra,cc+1),mu[0]),rs[0]);
   float xba=__fmul_rn(__fsub_rn(get(ln+base,rb,cc),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(get(ln+base,rb,cc+1),mu[1]),rs[1]);
   float da=acc[q*4],db=acc[q*4+1],dc=acc[q*4+2],dd=acc[q*4+3],ga=gamma[c],gb=gamma[c+1];
   uint32_t outa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
   uint32_t outb=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
   put(ln+base,ra,cc,bf16lo(outa)+get(ln+32768+base,ra,cc));put(ln+base,ra,cc+1,bf16hi(outa)+get(ln+32768+base,ra,cc+1));
   put(ln+base,rb,cc,bf16lo(outb)+get(ln+32768+base,rb,cc));put(ln+base,rb,cc+1,bf16hi(outb)+get(ln+32768+base,rb,cc+1));
   float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
   for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
   if(lane<4){tmp[(wi*4+w)*256+c]=dga;tmp[(wi*4+w)*256+c+1]=dgb;tmp[(wi*4+w)*256+128+c]=dba;tmp[(wi*4+w)*256+128+c+1]=dbb;}
  });
  dxsync();float sum=0;for(int w=0;w<8;++w)sum+=tmp[w*256+tid];running+=sum;
  fence_proxy_async();dxsync();if(tid==0){for(int m=0;m<2;++m){store2d(&p.dx,sm+m*16384,0,row+m*64);store2d(&p.dx,sm+m*16384+8192,64,row+m*64);}tma_store_commit();tma_store_wait_all();}dxrelease(b,0);
 }
 p.partln[split*256+tid]=running;
}
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<131072){int group=i/16384,j=i%16384,kind=j/8192,z=j%8192;float v=0;
  for(int b=0;b<DW_SPLITS*2;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*2+b)*16384+j];
  int out=(group/4)*2+(kind==0?1:0),rr=z%128,c=(group%4)*64+z/128;p.dw[(out*128+rr)*256+c]=__float2bfloat16_rn(v);
 }else if(i<131328){int c=i-131072;float v=0;for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}
}
extern "C" __global__ __launch_bounds__(384,1)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ Barriers bars;__shared__ float gamma[128];
 if(threadIdx.x<128)gamma[threadIdx.x]=p.gamma[threadIdx.x];
 if(threadIdx.x==0){for(int i=0;i<5;++i){mbar_init(bars.tx+i,1);mbar_init(bars.ready+i,1);mbar_init(bars.empty+i,1);}fence_barrier_init();}named_bar_sync(0,384);
 int wg=__shfl_sync(0xffffffffu,threadIdx.x>>7,0);
 if(blockIdx.x<DWCOUNT){
  if(wg<2){setmaxnreg_dec<128>();dw_producer(p,sm,&bars);}else{setmaxnreg_inc<240>();dw_consumer(p,sm,&bars);}
 }else{
  if(wg==0){setmaxnreg_dec<40>();dx_producer(p,sm,&bars);}else{setmaxnreg_inc<232>();dx_consumer(p,sm,&bars,gamma);}
 }
 named_bar_sync(0,384);
#if PART_ONLY == 2
 __threadfence();named_bar_sync(0,384);
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}named_bar_sync(0,384);
 for(int i=blockIdx.x*384+threadIdx.x;i<131328;i+=UCOUNT*384)reduce_at(p,i);
 __threadfence();named_bar_sync(0,384);
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
