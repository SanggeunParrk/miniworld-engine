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
constexpr int DW_SLOT=57344,DX_SLOT=40960;
static_assert(DXCOUNT>0,"invalid partition");
struct Params{
 CUtensorMap dl,dr,pre,xn,dg,wlg,wl,wrg,wr,wgate,x,res,dx;
 const __nv_bfloat16* mask;const float *mean,*rs,*gamma;
 __nv_bfloat16* dw;float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;__nv_bfloat16 *debugdc,*debugxn;int M,L,tiles;
};
TMN_DEVI void mma_weight64(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
TMN_DEVI void store2d(const CUtensorMap* map,const void* src,int c,int r){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2,%3}], [%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");
}

TMN_DEVI void glu_small(const Params& p,uint8_t* s,uint8_t* gout,uint8_t* pout,int row){
 unsigned tid=threadIdx.x;uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2);
 #pragma unroll 2
 for(unsigned q=0;q<8;++q){unsigned i=tid+q*256,c=i/32,r=(i%32)*2;
  uint32_t dy=pair_get(s+16384,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r);
  uint32_t masked;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(mask));float ga=math::sigmoid(bf16lo(gl)),gb=math::sigmoid(bf16hi(gl));
  uint32_t g=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb)),pp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  *reinterpret_cast<uint32_t*>(gout+swz128(c,r*2))=g;*reinterpret_cast<uint32_t*>(pout+swz128(c,r*2))=pp;
 }fence_proxy_async();allsync();
}
TMN_DEVI void load_dw(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int group){
 if(threadIdx.x)return;uint8_t* s=sm+slot*DW_SLOT;int side=group/4,h=(group%4)*64;mbar_arrive_expect_tx(b+slot,40960);
 tma_load_2d(s,&p.pre,b+slot,row,side*512+h*2);tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,row,h);
 for(int n=0;n<2;++n)tma_load_2d(s+24576+n*8192,&p.xn,b+slot,n*64,row);
}

TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* b){
 int group=blockIdx.x/DW_SPLITS,split=blockIdx.x%DW_SPLITS,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,segmentRounds=(rounds+1)/2,r=0;float acc[64]={};
 if(split<p.tiles)load_dw(p,sm,b,0,split*64,group);if(split+DW_SPLITS<p.tiles)load_dw(p,sm,b,1,(split+DW_SPLITS)*64,group);
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){int slot=r&1;uint8_t* s=sm+slot*DW_SLOT;mbar_wait(b+slot,(r/2)&1);glu_small(p,s,s+40960,s+49152,tile*64);
  fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_weight128(acc,smem_desc(smem_u32(s+40960+wi*8192+k*32),16,1024,1),smem_desc(smem_u32(s+24576+k*2048),8192,1024,1),r%segmentRounds>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
  if(tile+2*DW_SPLITS<p.tiles)load_dw(p,sm,b,slot,(tile+2*DW_SPLITS)*64,group);
  if((r+1)%segmentRounds==0||tile+DW_SPLITS>=p.tiles){float* out=p.partw+((group*DW_SPLITS+split)*2+r/segmentRounds)*16384+wi*8192;
   static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,acc[q*4],acc[q*4+1]);stg64f(out+(rr+8)*128+c,acc[q*4+2],acc[q*4+3]);});}
 }
}

struct Bars{uint64_t tx[4],ready[2],empty[4];};
TMN_DEVI void psync(){named_bar_sync(1,128);}
TMN_DEVI void csync(){named_bar_sync(2,128);}
TMN_DEVI int cid(){return int(threadIdx.x)-128;}
TMN_DEVI void publish(Bars* b,int slot){fence_proxy_async();psync();if(threadIdx.x==0)mbar_arrive(b->ready+slot);}
TMN_DEVI void release(Bars* b,int slot){fence_proxy_async();csync();if(cid()==0)mbar_arrive(b->empty+slot);}
TMN_DEVI void load_front(const Params& p,uint8_t* sm,Bars* b,int row,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b->tx+slot,40960);
 tma_load_2d(s,&p.pre,b->tx+slot,row,side*512+h*128);tma_load_2d(s+16384,side?&p.dr:&p.dl,b->tx+slot,row,h*64);
 for(int n=0;n<2;++n)tma_load_2d(s+24576+n*8192,side?&p.wrg:&p.wlg,b->tx+slot,h*64,n*64);
}
TMN_DEVI void load_pw(const Params& p,uint8_t* sm,Bars* b,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b->tx+slot,16384);
 for(int n=0;n<2;++n)tma_load_2d(s+n*8192,side?&p.wr:&p.wl,b->tx+slot,h*64,n*64);
}
TMN_DEVI void load_gate(const Params& p,uint8_t* sm,Bars* b,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b->tx+2,49152);
 for(int k=0;k<2;++k){tma_load_2d(sm+40960+k*8192,&p.dg,b->tx+2,k*64,row);for(int n=0;n<2;++n)tma_load_2d(sm+81920+k*16384+n*8192,&p.wgate,b->tx+2,k*64,n*64);}
}
TMN_DEVI void load_ln(const Params& p,uint8_t* sm,Bars* b,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b->tx+3,32768);
 for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.x,b->tx+3,c*64,row);tma_load_2d(sm+16384+c*8192,&p.res,b->tx+3,c*64,row);}
}
TMN_DEVI void glu_prod(const Params& p,uint8_t* sm,int h,int row){
 unsigned tid=threadIdx.x;uint8_t* s=sm+(h&1)*DX_SLOT;uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2);
 #pragma unroll 2
 for(unsigned q=0;q<16;++q){unsigned i=tid+q*128,c=i/32,r=(i%32)*2;uint32_t dy=pair_get(s+16384,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r),masked;
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(mask));float ga=math::sigmoid(bf16lo(gl)),gb=math::sigmoid(bf16hi(gl));
  uint32_t g=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb)),pp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  *reinterpret_cast<uint32_t*>(s+16384+swz128(c,r*2))=g;*reinterpret_cast<uint32_t*>(sm+81920+h*8192+swz128(c,r*2))=pp;
 }
}
TMN_DEVI void dx_producer(const Params& p,uint8_t* sm,Bars* b){
 int split=blockIdx.x-DWCOUNT,round=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int row=tile*64,ph=round&1;
  if(round==0)load_gate(p,sm,b,row);
  if(threadIdx.x==0){if(round)mbar_wait(b->empty+3,ph^1);load_front(p,sm,b,row,0,0);mbar_wait(b->empty+2,ph);load_front(p,sm,b,row,0,1);}psync();
  for(int side=0;side<2;++side){
   for(int h=0;h<4;++h){int slot=h&1;mbar_wait(b->tx+slot,h/2);glu_prod(p,sm,h,row);publish(b,slot);
    if(threadIdx.x==0){mbar_wait(b->empty+slot,h/2);if(h<2)load_front(p,sm,b,row,side,h+2);}
   }
   if(threadIdx.x==0){load_pw(p,sm,b,side,0);load_pw(p,sm,b,side,1);
    mbar_wait(b->empty,0);load_pw(p,sm,b,side,2);mbar_wait(b->empty+1,0);load_pw(p,sm,b,side,3);
    mbar_wait(b->empty,1);if(side==0)load_front(p,sm,b,row,1,0);else load_ln(p,sm,b,row);
    mbar_wait(b->empty+1,1);if(side==0)load_front(p,sm,b,row,1,1);else if(tile+DXCOUNT<p.tiles)load_gate(p,sm,b,(tile+DXCOUNT)*64);
   }psync();
  }
 }
}
TMN_DEVI void dx_consumer(const Params& p,uint8_t* sm,Bars* b,const float* gamma){
 int split=blockIdx.x-DWCOUNT,tid=cid(),lane=tid%32,w=tid/32,ra=w*16+lane/4,rb=ra+8,round=0;float running=0,runningb=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int row=tile*64,ph=round&1;uint32_t packed[32];float acc[64]={};mbar_wait(b->tx+2,ph);
  {float gate[64]={};fence_regs(gate);wgmma_fence();static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_gate128(gate,smem_desc(smem_u32(sm+40960+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+81920+(k/4)*16384+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(gate);static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;packed[q]=pack_bf16(gate[q*2],gate[q*2+1]);});}release(b,2);
  for(int side=0;side<2;++side){
   for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(b->ready+slot,h/2);fence_regs(acc);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(s+16384+k*2048),16,1024,1),smem_desc(smem_u32(s+24576+k*32),16,1024,1),side>0||h>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);release(b,slot);
   }
   for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(b->tx+slot,h/2);fence_regs(acc);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(sm+81920+h*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+k*32),16,1024,1),1);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);release(b,slot);
   }
  }
  static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;acc[q*2]=__bfloat162float(__float2bfloat16_rn(acc[q*2]+bf16lo(packed[q])));acc[q*2+1]=__bfloat162float(__float2bfloat16_rn(acc[q*2+1]+bf16hi(packed[q])));});
  mbar_wait(b->tx+3,ph);uint8_t* ln=sm;
  float mu[2]={p.mean[row+ra],p.mean[row+rb]},rs[2]={p.rs[row+ra],p.rs[row+rb]},s1[2]={},s2[2]={};
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   static_for<2>([&](auto ri){constexpr int r=decltype(ri)::value;int rr=r?rb:ra;
    float xa=__fmul_rn(__fsub_rn(get(ln+base,rr,cc),mu[r]),rs[r]),xb=__fmul_rn(__fsub_rn(get(ln+base,rr,cc+1),mu[r]),rs[r]);
    float ha=acc[q*4+r*2]*gamma[c],hb=acc[q*4+r*2+1]*gamma[c+1];s1[r]+=ha*xa+hb*xb;s2[r]+=ha+hb;
   });
  });
  float c1[2]={quad_sum(s1[0])/128.f,quad_sum(s1[1])/128.f},c2[2]={quad_sum(s2[0])/128.f,quad_sum(s2[1])/128.f};float* tmp=reinterpret_cast<float*>(sm+65536);
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   float xaa=__fmul_rn(__fsub_rn(get(ln+base,ra,cc),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(get(ln+base,ra,cc+1),mu[0]),rs[0]);
   float xba=__fmul_rn(__fsub_rn(get(ln+base,rb,cc),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(get(ln+base,rb,cc+1),mu[1]),rs[1]);
   float da=acc[q*4],db=acc[q*4+1],dc=acc[q*4+2],dd=acc[q*4+3],ga=gamma[c],gb=gamma[c+1];
   uint32_t outa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
   uint32_t outb=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
   put(ln+base,ra,cc,bf16lo(outa)+get(ln+16384+base,ra,cc));put(ln+base,ra,cc+1,bf16hi(outa)+get(ln+16384+base,ra,cc+1));
   put(ln+base,rb,cc,bf16lo(outb)+get(ln+16384+base,rb,cc));put(ln+base,rb,cc+1,bf16hi(outb)+get(ln+16384+base,rb,cc+1));
   float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
   for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
   if(lane<4){tmp[w*256+c]=dga;tmp[w*256+c+1]=dgb;tmp[w*256+128+c]=dba;tmp[w*256+128+c+1]=dbb;}
  });

  csync();float sum=0,sumb=0;for(int w=0;w<4;++w){sum+=tmp[w*256+tid];sumb+=tmp[w*256+128+tid];}running+=sum;runningb+=sumb;
  fence_proxy_async();csync();if(tid==0){store2d(&p.dx,sm,0,row);store2d(&p.dx,sm+8192,64,row);tma_store_commit();tma_store_wait_all();}release(b,3);
 }
 p.partln[split*256+tid]=running;p.partln[split*256+128+tid]=runningb;
}
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<131072){int group=i/16384,j=i%16384,kind=j/8192,z=j%8192;float v=0;
  for(int b=0;b<DW_SPLITS*2;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*2+b)*16384+j];
  int out=(group/4)*2+(kind==0?1:0),rr=z%128,c=(group%4)*64+z/128;p.dw[(out*128+rr)*256+c]=__float2bfloat16_rn(v);
 }else if(i<131328){int c=i-131072;float v=0;for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}
}

extern "C" __global__ __launch_bounds__(256,2)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ Bars b;__shared__ float gamma[128];
 if(threadIdx.x<128)gamma[threadIdx.x]=p.gamma[threadIdx.x];
 if(threadIdx.x==0){for(int i=0;i<4;++i){mbar_init(b.tx+i,1);mbar_init(b.empty+i,1);if(i<2)mbar_init(b.ready+i,1);}fence_barrier_init();}allsync();
 if(blockIdx.x<DWCOUNT)weight_role(p,sm,b.tx);else{
  int wg=__shfl_sync(0xffffffff,threadIdx.x/128,0);
  if(wg==0){setmaxnreg_dec<40>();dx_producer(p,sm,&b);}else{setmaxnreg_inc<216>();dx_consumer(p,sm,&b,gamma);}
 }
 allsync();
#if PART_ONLY == 2
 __threadfence();allsync();if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();
 for(int i=blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256)reduce_at(p,i);
 __threadfence();allsync();if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
