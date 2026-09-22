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
constexpr int DW_STAGES=5,DW_SLOT=40960,DX_SLOT=57344;
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
 float acc[2][64]={};int r=0;
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){
  int slot=r%DW_STAGES;uint8_t* s=sm+slot*DW_SLOT;mbar_wait(b->ready+slot,(r/DW_STAGES)&1);
  static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;fence_regs(acc[n]);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_weight128(acc[n],smem_desc(smem_u32(s+n*8192+k*32),16,1024,1),smem_desc(smem_u32(s+24576+k*2048),8192,1024,1),r>0||k>0);});
  });wgmma_commit();wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);release(b,slot);
 }
 static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;float* out=p.partw+(group*DW_SPLITS+split)*16384+n*8192;
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);
   stg64f(out+rr*128+c,acc[n][q*4],acc[n][q*4+1]);stg64f(out+(rr+8)*128+c,acc[n][q*4+2],acc[n][q*4+3]);
  });
 });
}
TMN_DEVI void load_gate(const Params& p,uint8_t* s,Barriers* b,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b->tx+3,49152);
 for(int k=0;k<2;++k){tma_load_2d(s+k*8192,&p.dg,b->tx+3,k*64,row);
  for(int n=0;n<2;++n)tma_load_2d(s+16384+k*16384+n*8192,&p.wgate,b->tx+3,k*64,n*64);}
}
TMN_DEVI void load_dx(const Params& p,uint8_t* sm,Barriers* b,int slot,int row,int side){
 if(threadIdx.x)return;uint8_t* s=sm+slot*DX_SLOT;int h=slot*64;
 mbar_arrive_expect_tx(b->tx+slot,57344);
 tma_load_2d(s,&p.pre,b->tx+slot,row,side*512+2*h);
 tma_load_2d(s+16384,side?&p.dr:&p.dl,b->tx+slot,row,h);
 for(int kind=0;kind<2;++kind)for(int n=0;n<2;++n)tma_load_2d(s+24576+kind*16384+n*8192,side?(kind?&p.wr:&p.wrg):(kind?&p.wl:&p.wlg),b->tx+slot,h,n*64);
}
TMN_DEVI void produce_dx(const Params& p,uint8_t* sm,Barriers* b,int slot,int row,int phase){
 mbar_wait(b->tx+slot,phase);glu64(p,sm+slot*DX_SLOT,row);ready(p,b,slot);
}
TMN_DEVI void dx_producer(const Params& p,uint8_t* sm,Barriers* b){
 int split=blockIdx.x-DWCOUNT,round=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){
  int row=tile*64,ph=round&1;
  if(round)mbar_wait(b->empty+3,ph^1);load_gate(p,sm+3*DX_SLOT,b,row);
  for(int s=1;s<=2;++s){if(round)mbar_wait(b->empty+s,1);load_dx(p,sm,b,s,row,0);}
  if(round)mbar_wait(b->empty,ph^1);load_dx(p,sm,b,0,row,0);
  mbar_wait(b->tx+3,ph);ready(p,b,3);
  produce_dx(p,sm,b,0,row,ph);
  mbar_wait(b->empty+3,ph);load_dx(p,sm,b,3,row,0);
  produce_dx(p,sm,b,1,row,0);produce_dx(p,sm,b,2,row,0);produce_dx(p,sm,b,3,row,ph^1);
  mbar_wait(b->empty,ph);load_dx(p,sm,b,0,row,1);
  mbar_wait(b->empty+1,0);load_dx(p,sm,b,1,row,1);
  produce_dx(p,sm,b,0,row,ph^1);
  mbar_wait(b->empty+2,0);load_dx(p,sm,b,2,row,1);
  produce_dx(p,sm,b,1,row,1);
  mbar_wait(b->empty+3,ph^1);load_dx(p,sm,b,3,row,1);
  produce_dx(p,sm,b,2,row,1);produce_dx(p,sm,b,3,row,ph);
  mbar_wait(b->empty,ph^1);
  if(threadIdx.x==0){mbar_arrive_expect_tx(b->tx,32768);for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.x,b->tx,c*64,row);tma_load_2d(sm+16384+c*8192,&p.res,b->tx,c*64,row);}}
  mbar_wait(b->tx,ph);ready(p,b,0);
 }
}
TMN_DEVI void dx_consumer(const Params& p,uint8_t* sm,Barriers* b,const float* gamma){
 int split=blockIdx.x-DWCOUNT,tid=cid(),lane=tid%32,w=tid/32,ra=w*16+lane/4,rb=ra+8,round=0;
 float running_g=0,running_b=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){
  int row=tile*64,ph=round&1;uint32_t packed[32];float acc[64]={};
  mbar_wait(b->ready+3,ph);
  {float gate[64]={};uint8_t* s=sm+3*DX_SLOT;fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_gate128(gate,smem_desc(smem_u32(s+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(s+16384+(k/4)*16384+(k%4)*32),16,1024,1),k>0);});
   wgmma_commit();wgmma_wait<0>();fence_regs(gate);static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;packed[q]=pack_bf16(gate[q*2],gate[q*2+1]);});}
  release(b,3);
  for(int side=0;side<2;++side){
   for(int slot=0;slot<4;++slot){int phase=(slot==0?ph:(slot==3?(ph^1):0))^side;mbar_wait(b->ready+slot,phase);uint8_t* s=sm+slot*DX_SLOT;
    fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(s+k*2048),16,1024,1),smem_desc(smem_u32(s+24576+k*32),16,1024,1),side>0||slot>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);
   }
   for(int slot=0;slot<4;++slot){uint8_t* s=sm+slot*DX_SLOT;fence_regs(acc);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(s+8192+k*2048),16,1024,1),smem_desc(smem_u32(s+40960+k*32),16,1024,1),1);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);release(b,slot);
   }
  }
  static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;acc[q*2]=__bfloat162float(__float2bfloat16_rn(acc[q*2]+bf16lo(packed[q])));acc[q*2+1]=__bfloat162float(__float2bfloat16_rn(acc[q*2+1]+bf16hi(packed[q])));});
  mbar_wait(b->ready,ph);
  float mu[2]={p.mean[row+ra],p.mean[row+rb]},rs[2]={p.rs[row+ra],p.rs[row+rb]},s1[2]={},s2[2]={};
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   static_for<2>([&](auto ri){constexpr int r=decltype(ri)::value;int rr=r?rb:ra;
    float xa=__fmul_rn(__fsub_rn(get(sm+base,rr,cc),mu[r]),rs[r]),xb=__fmul_rn(__fsub_rn(get(sm+base,rr,cc+1),mu[r]),rs[r]);
    float ha=acc[q*4+r*2]*gamma[c],hb=acc[q*4+r*2+1]*gamma[c+1];s1[r]+=ha*xa+hb*xb;s2[r]+=ha+hb;
   });
  });
  float c1[2]={quad_sum(s1[0])/128.f,quad_sum(s1[1])/128.f},c2[2]={quad_sum(s2[0])/128.f,quad_sum(s2[1])/128.f};float* tmp=reinterpret_cast<float*>(sm+32768);
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   float xaa=__fmul_rn(__fsub_rn(get(sm+base,ra,cc),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(get(sm+base,ra,cc+1),mu[0]),rs[0]);
   float xba=__fmul_rn(__fsub_rn(get(sm+base,rb,cc),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(get(sm+base,rb,cc+1),mu[1]),rs[1]);
   float da=acc[q*4],db=acc[q*4+1],dc=acc[q*4+2],dd=acc[q*4+3],ga=gamma[c],gb=gamma[c+1];
   uint32_t outa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
   uint32_t outb=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
   put(sm+base,ra,cc,bf16lo(outa)+get(sm+16384+base,ra,cc));put(sm+base,ra,cc+1,bf16hi(outa)+get(sm+16384+base,ra,cc+1));
   put(sm+base,rb,cc,bf16lo(outb)+get(sm+16384+base,rb,cc));put(sm+base,rb,cc+1,bf16hi(outb)+get(sm+16384+base,rb,cc+1));
   float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
   for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
   if(lane<4){tmp[w*256+c]=dga;tmp[w*256+c+1]=dgb;tmp[w*256+128+c]=dba;tmp[w*256+128+c+1]=dbb;}
  });
  csync();running_g+=(tmp[tid]+tmp[256+tid])+(tmp[512+tid]+tmp[768+tid]);running_b+=(tmp[128+tid]+tmp[384+tid])+(tmp[640+tid]+tmp[896+tid]);
  fence_proxy_async();csync();if(tid==0){store2d(&p.dx,sm,0,row);store2d(&p.dx,sm+8192,64,row);tma_store_commit();tma_store_wait_all();}release(b,0);
 }
 p.partln[split*256+tid]=running_g;p.partln[split*256+128+tid]=running_b;
}
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<131072){int group=i/16384,j=i%16384,kind=j/8192,z=j%8192;float v=0;
  for(int b=0;b<DW_SPLITS;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS+b)*16384+j];
  int out=(group/4)*2+(kind==0?1:0),rr=z%128,c=(group%4)*64+z/128;p.dw[(out*128+rr)*256+c]=__float2bfloat16_rn(v);
 }else if(i<131328){int c=i-131072;float v=0;for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}
}
extern "C" __global__ __launch_bounds__(384,1)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ Barriers bars;__shared__ float gamma[128];
 if(threadIdx.x<128)gamma[threadIdx.x]=p.gamma[threadIdx.x];
 if(threadIdx.x==0){for(int i=0;i<5;++i){mbar_init(bars.tx+i,1);mbar_init(bars.ready+i,1);mbar_init(bars.empty+i,1);}fence_barrier_init();}named_bar_sync(0,384);
 if(threadIdx.x==0){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));reinterpret_cast<unsigned long long*>(p.counts+2)[blockIdx.x*2]=t;}
 int wg=__shfl_sync(0xffffffffu,threadIdx.x>>7,0);
 if(wg<2){setmaxnreg_dec<128>();if(blockIdx.x<DWCOUNT)dw_producer(p,sm,&bars);else dx_producer(p,sm,&bars);}
 else{setmaxnreg_inc<240>();if(blockIdx.x<DWCOUNT)dw_consumer(p,sm,&bars);else dx_consumer(p,sm,&bars,gamma);}
 named_bar_sync(0,384);if(threadIdx.x==0){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));reinterpret_cast<unsigned long long*>(p.counts+2)[blockIdx.x*2+1]=t;}
#if PART_ONLY == 2
 __threadfence();named_bar_sync(0,384);
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}named_bar_sync(0,384);
 for(int i=blockIdx.x*384+threadIdx.x;i<131328;i+=UCOUNT*384)reduce_at(p,i);
 __threadfence();named_bar_sync(0,384);
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
