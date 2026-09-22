#define RING_TILES 96
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
 CUtensorMap dl,dr,pre,xn,dg,wlg,wl,wrg,wr,wgate,x,res,dx,ring;
 const __nv_bfloat16* mask;const float *mean,*rs,*gamma;
 __nv_bfloat16* dw;float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;__nv_bfloat16 *debugdc,*debugxn;int M,L,tiles;
};
TMN_DEVI void tma_last(void* dst,const CUtensorMap* map,uint64_t* bar,int c0,int c1){asm volatile("{.reg .b64 policy;createpolicy.fractional.L2::evict_last.b64 policy,1.0;cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%3,%4}],[%2],policy;}"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c0),"r"(c1):"memory");}
TMN_DEVI void tma_first(void* dst,const CUtensorMap* map,uint64_t* bar,int c0,int c1){asm volatile("{.reg .b64 policy;createpolicy.fractional.L2::evict_first.b64 policy,1.0;cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%3,%4}],[%2],policy;}"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c0),"r"(c1):"memory");}
TMN_DEVI void ring_store_last(const CUtensorMap* map,const void* s,int c,int r){asm volatile("{.reg .b64 policy;createpolicy.fractional.L2::evict_last.b64 policy,1.0;cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%2,%3}],[%1],policy;}"::"l"(map),"r"(smem_u32(s)),"r"(c),"r"(r):"memory");}
TMN_DEVI void mma_weight64(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
TMN_DEVI void store2d(const CUtensorMap* map,const void* src,int c,int r){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2,%3}], [%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");
}

TMN_DEVI void glu_small(const Params& p,uint8_t* s,uint8_t* gout,uint8_t* pout,int row){
 unsigned tid=threadIdx.x;uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2);
 #pragma unroll 4
 for(unsigned q=0;q<8;++q){unsigned i=tid+q*256,c=i/32,r=(i%32)*2;
  uint32_t dy=pair_get(s+16384,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r);
  uint32_t masked;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(mask));float ga=math::sigmoid(bf16lo(gl)),gb=math::sigmoid(bf16hi(gl));
  uint32_t g=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb)),pp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  *reinterpret_cast<uint32_t*>(gout+swz128(c,r*2))=g;*reinterpret_cast<uint32_t*>(pout+swz128(c,r*2))=pp;
 }fence_proxy_async();allsync();
}
TMN_DEVI void load_dw(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int group){
 if(threadIdx.x)return;uint8_t* s=sm+slot*DW_SLOT;int side=group/4,h=(group%4)*64;mbar_arrive_expect_tx(b+slot,40960);
 tma_first(s,&p.pre,b+slot,row,side*512+h*2);tma_first(s+16384,side?&p.dr:&p.dl,b+slot,row,h);
 for(int n=0;n<2;++n)tma_load_2d(s+24576+n*8192,&p.xn,b+slot,n*64,row);
}

TMN_DEVI void ring_wait(const unsigned* ptr,unsigned want){unsigned got;do{asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(got):"l"(ptr):"memory");if(got<want)__nanosleep(32);}while(got<want);}
TMN_DEVI void ring_publish(unsigned* ptr,unsigned value){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(ptr),"r"(value):"memory");}
TMN_DEVI void ring_begin(const Params& p,uint8_t* s,int tile,int group){if(threadIdx.x)return;int slot=tile%RING_TILES;if(tile>=RING_TILES)ring_wait(p.counts+2+8*RING_TILES+slot,tile-RING_TILES+1);int h=(group/4)*512+(group%4)*64;ring_store_last(&p.ring,s+40960,slot*64,h);ring_store_last(&p.ring,s+49152,slot*64,h+256);tma_store_commit();}
TMN_DEVI void ring_finish(const Params& p,int tile,int group){if(threadIdx.x)return;tma_store_wait_all();ring_publish(p.counts+2+(tile%RING_TILES)*8+group,tile+1);}
TMN_DEVI void ring_ready(const Params& p,int tile){if(threadIdx.x<8)ring_wait(p.counts+2+(tile%RING_TILES)*8+threadIdx.x,tile+1);allsync();}
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* b){
 int group=blockIdx.x%8,split=blockIdx.x/8,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,segmentRounds=(rounds+1)/2,r=0;float acc[64]={};
 if(split<p.tiles)load_dw(p,sm,b,0,split*64,group);if(split+DW_SPLITS<p.tiles)load_dw(p,sm,b,1,(split+DW_SPLITS)*64,group);
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){int slot=r&1;uint8_t* s=sm+slot*DW_SLOT;mbar_wait(b+slot,(r/2)&1);glu_small(p,s,s+40960,s+49152,tile*64);if(r>0)ring_finish(p,tile-DW_SPLITS,group);ring_begin(p,s,tile,group);
  fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_weight128(acc,smem_desc(smem_u32(s+40960+wi*8192+k*32),16,1024,1),smem_desc(smem_u32(s+24576+k*2048),8192,1024,1),r%segmentRounds>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
  if(tile+2*DW_SPLITS<p.tiles)load_dw(p,sm,b,slot,(tile+2*DW_SPLITS)*64,group);
  if((r+1)%segmentRounds==0||tile+DW_SPLITS>=p.tiles){float* out=p.partw+((group*DW_SPLITS+split)*2+r/segmentRounds)*16384+wi*8192;
   static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,acc[q*4],acc[q*4+1]);stg64f(out+(rr+8)*128+c,acc[q*4+2],acc[q*4+3]);});}
 }
 if(rounds>0)ring_finish(p,split+(rounds-1)*DW_SPLITS,group);
}
// SPDX-License-Identifier: Apache-2.0
// Anthropic v5 training extension. One low-register TMA WG and one N128 MMA WG.
// Four 24-KiB stages; gate prefetch at64KiB overlaps the LN epilogue at0KiB.
constexpr int WS_STAGES=4,WS_SLOT=24576,WS_GATE=65536;
struct WsBarriers { uint64_t tx[6],empty[5]; };
TMN_DEVI void ws_gate(const Params& p,uint8_t* sm,WsBarriers* b,int row){
 mbar_arrive_expect_tx(b->tx+4,49152);
 for(int k=0;k<2;++k){
  tma_load_2d(sm+WS_GATE+k*8192,&p.dg,b->tx+4,k*64,row);
  tma_load_2d(sm+WS_GATE+16384+k*16384,&p.wgate,b->tx+4,k*64,0);
 }
}
TMN_DEVI void ws_producer(const Params& p,uint8_t* sm,WsBarriers* b){
 if(threadIdx.x)return;
 int split=blockIdx.x-DWCOUNT,round=0;
 if(split<p.tiles)ws_gate(p,sm,b,split*64);
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){
  int row=tile*64;
  for(int group=0;group<8;++group)ring_wait(p.counts+2+(tile%RING_TILES)*8+group,tile+1);
  // Gate MMA retires all operands in the upper shared-memory region.
  mbar_wait(b->empty+4,round&1);
  for(int step=0;step<16;++step){
   int slot=step%WS_STAGES,epoch=step/WS_STAGES;
   if(round||step>=WS_STAGES)mbar_wait(b->empty+slot,(epoch+1)&1);
   int side=step/8,kind=(step/4)%2,h=step%4;
   const CUtensorMap* weight=side?(kind?&p.wr:&p.wrg):(kind?&p.wl:&p.wlg);
   uint8_t* s=sm+slot*WS_SLOT;
   mbar_arrive_expect_tx(b->tx+slot,24576);
   tma_last(s,&p.ring,b->tx+slot,row%(RING_TILES*64),side*512+kind*256+h*64);
   tma_load_2d(s+8192,weight,b->tx+slot,h*64,0);
  }
  // P is loaded on demand. Publish the ring slot only after the final P read.
  mbar_wait(b->tx+3,1);
  ring_publish(p.counts+2+8*RING_TILES+tile%RING_TILES,tile+1);
  mbar_wait(b->empty,1);mbar_wait(b->empty+1,1);
  mbar_arrive_expect_tx(b->tx+5,32768);
  for(int c=0;c<2;++c){
   tma_load_2d(sm+c*8192,&p.x,b->tx+5,c*64,row);
   tma_load_2d(sm+16384+c*8192,&p.res,b->tx+5,c*64,row);
  }
  if(tile+DXCOUNT<p.tiles){
   mbar_wait(b->empty+2,1);mbar_wait(b->empty+3,1);
   ws_gate(p,sm,b,(tile+DXCOUNT)*64);
  }
 }
}
TMN_DEVI void ws_consumer(const Params& p,uint8_t* sm,WsBarriers* b,const float* gamma){
 int split=blockIdx.x-DWCOUNT,tid=threadIdx.x-128,lane=tid%32,w=tid/32;
 int ra=w*16+lane/4,rb=ra+8,round=0;float running_g=0,running_b=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){
  int row=tile*64;uint32_t packed[32];float acc[64]={};
  mbar_wait(b->tx+4,round&1);
  {float gate[64]={};uint8_t* s=sm+WS_GATE;
   fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
    mma_gate128(gate,smem_desc(smem_u32(s+(k/4)*8192+(k%4)*32),16,1024,1),
      smem_desc(smem_u32(s+16384+(k/4)*16384+(k%4)*32),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
   static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;packed[q]=pack_bf16(gate[q*2],gate[q*2+1]);});
  }
  if(tid==0)mbar_arrive(b->empty+4);
  for(int step=0;step<16;++step){
   int slot=step%WS_STAGES,epoch=step/WS_STAGES;uint8_t* s=sm+slot*WS_SLOT;
   mbar_wait(b->tx+slot,epoch&1);fence_regs(acc);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
    mma_front128(acc,smem_desc(smem_u32(s+k*2048),16,1024,1),
      smem_desc(smem_u32(s+8192+k*32),16,1024,1),step>0||k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(acc);
   if(tid==0)mbar_arrive(b->empty+slot);
  }
  static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;
   acc[q*2]=__bfloat162float(__float2bfloat16_rn(acc[q*2]+bf16lo(packed[q])));
   acc[q*2+1]=__bfloat162float(__float2bfloat16_rn(acc[q*2+1]+bf16hi(packed[q])));
  });
  mbar_wait(b->tx+5,round&1);
  float mu[2]={p.mean[row+ra],p.mean[row+rb]},rs[2]={p.rs[row+ra],p.rs[row+rb]};
  float s1[2][2]={},s2[2][2]={};
  // Match the original two C64 halves and their final addition exactly.
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;constexpr int half=q/8;
   int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   static_for<2>([&](auto ri){constexpr int r=decltype(ri)::value;int rr=r?rb:ra;
    float xa=__fmul_rn(__fsub_rn(get(sm+base,rr,cc),mu[r]),rs[r]);
    float xb=__fmul_rn(__fsub_rn(get(sm+base,rr,cc+1),mu[r]),rs[r]);
    float ha=acc[q*4+r*2]*gamma[c],hb=acc[q*4+r*2+1]*gamma[c+1];
    s1[half][r]+=ha*xa+hb*xb;s2[half][r]+=ha+hb;
   });
  });
  float c1[2]={quad_sum(s1[0][0])/128.f+quad_sum(s1[1][0])/128.f,
               quad_sum(s1[0][1])/128.f+quad_sum(s1[1][1])/128.f};
  float c2[2]={quad_sum(s2[0][0])/128.f+quad_sum(s2[1][0])/128.f,
               quad_sum(s2[0][1])/128.f+quad_sum(s2[1][1])/128.f};
  float* tmp=reinterpret_cast<float*>(sm+32768);
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;
   int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   float xaa=__fmul_rn(__fsub_rn(get(sm+base,ra,cc),mu[0]),rs[0]);
   float xab=__fmul_rn(__fsub_rn(get(sm+base,ra,cc+1),mu[0]),rs[0]);
   float xba=__fmul_rn(__fsub_rn(get(sm+base,rb,cc),mu[1]),rs[1]);
   float xbb=__fmul_rn(__fsub_rn(get(sm+base,rb,cc+1),mu[1]),rs[1]);
   float da=acc[q*4],db=acc[q*4+1],dc=acc[q*4+2],dd=acc[q*4+3],ga=gamma[c],gb=gamma[c+1];
   uint32_t oa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
   uint32_t ob=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
   uint32_t resa=pair_get(sm+16384+base,ra,cc),resb=pair_get(sm+16384+base,rb,cc);
   *reinterpret_cast<uint32_t*>(sm+base+swz128(ra,cc*2))=pack_bf16(bf16lo(oa)+bf16lo(resa),bf16hi(oa)+bf16hi(resa));
   *reinterpret_cast<uint32_t*>(sm+base+swz128(rb,cc*2))=pack_bf16(bf16lo(ob)+bf16lo(resb),bf16hi(ob)+bf16hi(resb));
   float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
   for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
   if(lane<4){tmp[w*256+c]=dga;tmp[w*256+c+1]=dgb;tmp[w*256+128+c]=dba;tmp[w*256+128+c+1]=dbb;}
  });
  named_bar_sync(1,128);
  running_g+=(tmp[tid]+tmp[256+tid])+(tmp[512+tid]+tmp[768+tid]);
  running_b+=(tmp[128+tid]+tmp[384+tid])+(tmp[640+tid]+tmp[896+tid]);
  fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){store2d(&p.dx,sm,0,row);store2d(&p.dx,sm+8192,64,row);tma_store_commit();tma_store_wait_all();}
  named_bar_sync(1,128);
 }
 p.partln[split*256+tid]=running_g;p.partln[split*256+128+tid]=running_b;
}

TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<131072){int group=i/16384,j=i%16384,kind=j/8192,z=j%8192;float v=0;
  for(int b=0;b<DW_SPLITS*2;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*2+b)*16384+j];
  int out=(group/4)*2+(kind==0?1:0),rr=z%128,c=(group%4)*64+z/128;p.dw[(out*128+rr)*256+c]=__float2bfloat16_rn(v);
 }else if(i<131328){int c=i-131072;float v=0;for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}
}
extern "C" __global__ __launch_bounds__(256,2)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ WsBarriers bars;uint64_t* bar=bars.tx;__shared__ float gamma[128];if(threadIdx.x<128)gamma[threadIdx.x]=p.gamma[threadIdx.x];
 if(threadIdx.x==0){for(int i=0;i<6;++i)mbar_init(bars.tx+i,1);for(int i=0;i<5;++i)mbar_init(bars.empty+i,1);fence_barrier_init();}allsync();
 if(blockIdx.x<DWCOUNT)weight_role(p,sm,bar);else {
  int wg=__shfl_sync(0xffffffff,threadIdx.x/128,0);if(wg==0){setmaxnreg_dec<32>();ws_producer(p,sm,&bars);}
  else{setmaxnreg_inc<224>();ws_consumer(p,sm,&bars,gamma);}
  allsync();if(wg==0)setmaxnreg_inc<128>();else setmaxnreg_dec<128>();
 }
#if PART_ONLY == 2
 __threadfence();allsync(); // Publish all partial writers before completion ticket.
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();
 for(int i=blockIdx.x*256+threadIdx.x;i<9*RING_TILES;i+=UCOUNT*256)p.counts[2+i]=0;
 for(int i=blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256)reduce_at(p,i);
 __threadfence();allsync();
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
