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
// Each pair is two adjacent rows, preserving preact's channel-major loads.
TMN_DEVI void glu128_pair(const Params& p,uint8_t* s,int i,int row,float ma,float mb,uint32_t& dg,uint32_t& dp){
 int c=i/32,r=(i%32)*2;
 uint32_t dy=pair_get(s+32768,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r);

 // The mask multiply has its own BF16 boundary in the reference.
 uint32_t masked=pack_bf16(bf16lo(dy)*ma,bf16hi(dy)*mb);
 float ga=math::sigmoid_div(bf16lo(gl)),gb=math::sigmoid_div(bf16hi(gl));
 dg=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb));
 dp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
}
TMN_DEVI void write_glu128(uint8_t* out,int i,uint32_t g,uint32_t p){
 int c=i/32,r=(i%32)*2;
 *reinterpret_cast<uint32_t*>(out+swz128(c,r*2))=g;
 *reinterpret_cast<uint32_t*>(out+16384+swz128(c,r*2))=p;
}
template<bool INPLACE> TMN_DEVI void glu128(const Params& p,uint8_t* s,int row,int group){
 int mr=row+(dxid()%32)*2;float ma=__bfloat162float(p.mask[mr]),mb=__bfloat162float(p.mask[mr+1]);
 if constexpr(INPLACE){
  uint32_t gg[16],pp[16];
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;glu128_pair(p,s,dxid()+q*256,row,ma,mb,gg[q],pp[q]);});
  // All preact reads finish before the transposed GLU stores overwrite it.
  dxsync();
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int i=dxid()+q*256;write_glu128(s,i,gg[q],pp[q]);
#if DEBUG_SAVE
   int c=i/32,r=(i%32)*2,side=group/2,h=(group%2)*128+c;
   *reinterpret_cast<uint32_t*>(p.debugdc+(size_t)(side*512+h)*p.M+row+r)=gg[q];
   *reinterpret_cast<uint32_t*>(p.debugdc+(size_t)(side*512+256+h)*p.M+row+r)=pp[q];
#endif
  });
 }else{
  for(int i=dxid();i<4096;i+=256){uint32_t g,pr;glu128_pair(p,s,i,row,ma,mb,g,pr);write_glu128(s+65536,i,g,pr);}
 }
 // Publish generic stores to WGMMA's async proxy; all producers participate.
 fence_proxy_async();dxsync();
}

TMN_DEVI void h_load_gate(const Params& p,uint8_t* sm,Barriers* b,int slot,int row){
 mbar_arrive_expect_tx(b->tx+slot,49152);
 for(int k=0;k<2;++k){tma_load_2d(sm+k*8192,&p.dg,b->tx+slot,k*64,row);
  for(int n=0;n<2;++n)tma_load_2d(sm+16384+n*16384+k*8192,&p.wgate,b->tx+slot,k*64,n*64);}
}
TMN_DEVI void h_load_front(const Params& p,uint8_t* sm,Barriers* b,int slot,int row,int group){
 uint8_t* s=sm+slot*DX_SLOT;int side=group/2,h=(group%2)*128;
 mbar_arrive_expect_tx(b->tx+slot,114688);
 for(int c=0;c<2;++c){tma_load_2d(s+c*16384,&p.pre,b->tx+slot,row,side*512+2*h+c*128);tma_load_2d(s+32768+c*8192,side?&p.dr:&p.dl,b->tx+slot,row,h+c*64);}
 for(int kind=0;kind<2;++kind)for(int n=0;n<2;++n)for(int k=0;k<2;++k)tma_load_2d(s+49152+kind*32768+n*16384+k*8192,side?(kind?&p.wr:&p.wrg):(kind?&p.wl:&p.wlg),b->tx+slot,h+k*64,n*64);
}
TMN_DEVI void dx_producer(const Params& p,uint8_t* sm,Barriers* b){
 if(threadIdx.x)return;int split=blockIdx.x-DWCOUNT,round=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int row=tile*64,a=round&1,other=a^1;
  if(round)mbar_wait(b->empty+a,1);h_load_gate(p,sm+a*DX_SLOT,b,a,row);
  if(round)mbar_wait(b->empty+other,1);h_load_front(p,sm,b,other,row,1);
  mbar_wait(b->empty+a,0);h_load_front(p,sm,b,a,row,0);
  mbar_wait(b->empty+a,1);h_load_front(p,sm,b,a,row,2);
  mbar_wait(b->empty+other,0);h_load_front(p,sm,b,other,row,3);
  mbar_wait(b->empty+a,0);mbar_arrive_expect_tx(b->tx+a,32768);
  for(int c=0;c<2;++c){tma_load_2d(sm+a*DX_SLOT+c*8192,&p.x,b->tx+a,c*64,row);tma_load_2d(sm+a*DX_SLOT+16384+c*8192,&p.res,b->tx+a,c*64,row);}
 }
}
TMN_DEVI void dx_consumer(const Params& p,uint8_t* sm,Barriers* b,const float* cached_gamma){
 int split=blockIdx.x-DWCOUNT,wi=dxid()/128,lane=dxid()%32,w=(dxid()/32)%4;
 int ra=w*16+lane/4,rb=ra+8;float running=0;int round=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){
  int row=tile*64,a=round&1;uint8_t* gs=sm+a*DX_SLOT;uint32_t gate_packed[16];float acc[32]={};
  // Per row tile: bar0 gate+2channel+LN transactions (4); bar1 2channels.
  // Both phase parities return to0 before advancing to the next row tile.
  mbar_wait(b->tx+a,0);
  {float gate[32]={};
  fence_regs(gate);wgmma_fence();
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_dgrad(gate,smem_desc(smem_u32(gs+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(gs+16384+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
  // B9 rounds separately before B10's accumulation/add.
  static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;gate_packed[j]=pack_bf16(gate[j*2],gate[j*2+1]);});}
  dxrelease(b,a);
  // Preserve baseline K order: Lg[0:256], Lp[0:256], Rg, Rp.
  for(int side=0;side<2;++side){
   for(int half=0;half<2;++half){
    int slot=half,group=side*2+half;uint8_t* s=sm+(slot^a)*DX_SLOT;
    mbar_wait(b->tx+(slot^a),(side+1-half)&1);glu128<true>(p,s,row,group);
    fence_regs(acc);wgmma_fence();
    static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
     mma_input64(acc,smem_desc(smem_u32(s+k*2048),16,1024,1),smem_desc(smem_u32(s+49152+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),side>0||half>0||k>0);
    });wgmma_commit();wgmma_wait<0>();fence_regs(acc);dxsync();
   }
   for(int half=0;half<2;++half){
    uint8_t* s=sm+(half^a)*DX_SLOT;fence_regs(acc);wgmma_fence();
    static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
     mma_input64(acc,smem_desc(smem_u32(s+16384+k*2048),16,1024,1),smem_desc(smem_u32(s+81920+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),1);
    });wgmma_commit();wgmma_wait<0>();fence_regs(acc);dxsync();
    dxrelease(b,half^a);
   }
  }
  // B10 outputs BF16 dx_n. Keep it in registers through B11/B12.
  static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;acc[j*2]=__bfloat162float(__float2bfloat16_rn(acc[j*2]+bf16lo(gate_packed[j])));acc[j*2+1]=__bfloat162float(__float2bfloat16_rn(acc[j*2+1]+bf16hi(gate_packed[j])));});
  mbar_wait(b->tx+a,1);uint8_t* ln=sm+a*DX_SLOT;
  float mu[2]={p.mean[row+ra],p.mean[row+rb]},rs[2]={p.rs[row+ra],p.rs[row+rb]},s1[2]={},s2[2]={};
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;int rr=j&1,c=q*16+2*(lane%4)+8*(j>>1),r=rr?rb:ra;
    float xa=__fmul_rn(__fsub_rn(get(ln+wi*8192,r,c),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(get(ln+wi*8192,r,c+1),mu[rr]),rs[rr]);
    float ha=acc[q*8+j*2]*cached_gamma[wi*64+c],hb=acc[q*8+j*2+1]*cached_gamma[wi*64+c+1];s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;
#if DEBUG_SAVE
    p.debugxn[(size_t)(row+r)*128+wi*64+c]=__float2bfloat16_rn(acc[q*8+j*2]);p.debugxn[(size_t)(row+r)*128+wi*64+c+1]=__float2bfloat16_rn(acc[q*8+j*2+1]);
#endif
   });
  });
  s1[0]=quad_sum(s1[0])/128.f;s1[1]=quad_sum(s1[1])/128.f;s2[0]=quad_sum(s2[0])/128.f;s2[1]=quad_sum(s2[1])/128.f;
  float* stats=reinterpret_cast<float*>(ln+36864);float* tmp=reinterpret_cast<float*>(ln+32768);
  if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}
  dxsync(); // Publish both64-channel halves before full128-channel LN reduction.
  float c1[2]={stats[ra*2]+stats[128+ra*2],stats[rb*2]+stats[128+rb*2]},c2[2]={stats[ra*2+1]+stats[128+ra*2+1],stats[rb*2+1]+stats[128+rb*2+1]};
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;int j=pair*2,c=q*16+2*(lane%4)+8*pair,gc=wi*64+c;
    float xaa=__fmul_rn(__fsub_rn(get(ln+wi*8192,ra,c),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(get(ln+wi*8192,ra,c+1),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(get(ln+wi*8192,rb,c),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(get(ln+wi*8192,rb,c+1),mu[1]),rs[1]);
    float da=acc[q*8+j*2],db=acc[q*8+j*2+1],dc=acc[q*8+j*2+2],dd=acc[q*8+j*2+3],ga=cached_gamma[gc],gb=cached_gamma[gc+1];
    uint32_t outa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
    uint32_t outb=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
    put(ln+wi*8192,ra,c,bf16lo(outa)+get(ln+16384+wi*8192,ra,c));put(ln+wi*8192,ra,c+1,bf16hi(outa)+get(ln+16384+wi*8192,ra,c+1));
    put(ln+wi*8192,rb,c,bf16lo(outb)+get(ln+16384+wi*8192,rb,c));put(ln+wi*8192,rb,c+1,bf16hi(outb)+get(ln+16384+wi*8192,rb,c+1));
    float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
    if(lane<4){tmp[w*256+gc]=dga;tmp[w*256+gc+1]=dgb;tmp[w*256+128+gc]=dba;tmp[w*256+128+gc+1]=dbb;}
   });
  });
  dxsync(); // All warp parameter partials and dx stores published.
  running+=(tmp[dxid()]+tmp[256+dxid()])+(tmp[512+dxid()]+tmp[768+dxid()]);
  fence_proxy_async();dxsync();
  if(dxid()==0){store2d(&p.dx,ln,0,row);store2d(&p.dx,ln+8192,64,row);tma_store_commit();tma_store_wait_all();}
  dxrelease(b,a); // TMA store cannot read a slot overwritten by next tile's gate loads.
 }
 p.partln[split*256+dxid()]=running;
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
  if(wg==0){setmaxnreg_dec<80>();dx_producer(p,sm,&bars);}else{setmaxnreg_inc<208>();dx_consumer(p,sm,&bars,gamma);}
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
