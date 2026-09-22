// Diagnostic: dual_traffic_floor. Memory-schedule replica of dual_ln_prefetch with ALL arithmetic removed:
// same roles, TMA loads, two-slot pipeline, dg (stg128 of the dy slot) and dtri (TMA store of the tri slot)
// writes, barriers and reduction; no B1, WGMMA or LN. Outputs are meaningless. Its time is the floor of
// this exact access pattern and pipeline depth, used only to bound the selected kernel from below.
// SPDX-License-Identifier: Apache-2.0
// Anthropic native v5 TMA/WGMMA/fragment primitives; MiniWorld B1-B4 training.
// Selected role ratio10:23: H100132 SMs ->40 weight CTAs,92 input/LN CTAs.
// One cooperative launch, two input stages per role. Same forward saves and
// BF16 dp/dg/dnorm rounding. No cluster/multicast, no intermediate global dp.
// Register mask cache supports up to3 phases; generic ds loads otherwise.
// A plan owns its workspace and runs on one stream, without reentrant calls.
// See BALANCED_AUDIT_20260920.md for layouts, validation and measured limits.
#define DW_RATIO 10
#define DX_RATIO 23
#include "dual_primitives.cuh"
#ifndef PART_ONLY
#define PART_ONLY 0
#endif
#ifndef UCOUNT
#define UCOUNT 132
#endif
TMN_DEVI void allsync(){named_bar_sync(0,256);}
TMN_DEVI void mma_dgrad(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
constexpr int DWCOUNT=UCOUNT*DW_RATIO/(DW_RATIO+DX_RATIO);
constexpr int DXCOUNT=UCOUNT-DWCOUNT;
static_assert(DWCOUNT>0&&DXCOUNT>0,"both CTA roles required");
// Independent CTA roles; each barrier is armed before its own TMA loads.
template<bool DW> TMN_DEVI void issue_slot(const Params& p,uint8_t* sm,uint64_t* full,int slot,int tile,bool initial){
 if(threadIdx.x!=0)return;
 uint8_t* s=sm+slot*98304;int row=tile*64;
 if constexpr(DW){
#pragma unroll
  for(int c=0;c<2;++c){tma_load_2d(s+c*8192,&p.dy,full+slot,c*64,row);tma_load_2d(s+16384+c*8192,&p.gate,full+slot,c*64,row);tma_load_2d(s+32768+c*8192,&p.proj,full+slot,c*64,row);tma_load_2d(s+49152+c*8192,&p.xn,full+slot,c*64,row);}
#pragma unroll
  for(int c=0;c<4;++c)tma_load_2d(s+65536+c*8192,&p.norm,full+slot,c*64,row);
 }else{
#pragma unroll
  for(int c=0;c<2;++c){tma_load_2d(s+c*8192,&p.dy,full+slot,c*64,row);tma_load_2d(s+16384+c*8192,&p.gate,full+slot,c*64,row);}
  tma_load_2d(s+32768,&p.tri,full+slot,row,0);
  if(initial)for(int n=0;n<4;++n)for(int k=0;k<2;++k)tma_load_2d(sm+163840+n*16384+k*8192,&p.wp,full+slot,k*64,n*64);
 }
}
struct MaskCycle{uint32_t b0,b1,b2,scale;int period;bool cached;};
template<int STRIDE> TMN_DEVI MaskCycle mask_cycle(const Params& p,int first){
 MaskCycle v={0,0,0,0,0,false};int a=64*STRIDE,b=p.L;while(b){int r=a%b;a=b;b=r;}
 v.period=p.L/a;v.cached=v.period<=3;
 if(v.cached&&first<p.tiles)for(int z=0;z<v.period;++z){uint32_t bits=0;int j0=((first+z*STRIDE)*64)%p.L;
  for(int i=threadIdx.x;i<1024;i+=256){int cb=i/512,r=(i%512)/8,c=(i%8)*8,jr=j0+r;if(jr>=p.L)jr-=p.L;
   uint4 ds=ldg128(p.ds+jr*128+cb*64+c);uint32_t* dd=reinterpret_cast<uint32_t*>(&ds);
#pragma unroll
   for(int q=0;q<4;++q){uint32_t lo=dd[q]&65535u,hi=dd[q]>>16;int bit=8*(i/256)+2*q;
    bits|=(uint32_t(lo!=0)<<bit)|(uint32_t(hi!=0)<<(bit+1));v.scale|=lo|hi;}
  }
  if(z==0)v.b0=bits;else if(z==1)v.b1=bits;else v.b2=bits;
 }
 return v;
}
template<bool DW> TMN_DEVI void gate_backward(const Params& p,uint8_t* s,int row,const MaskCycle& mask,int mi){
 if constexpr(DW){
  for(int i=threadIdx.x;i<1024;i+=256){int cb=i/512,r=(i%512)/8,c=(i%8)*8;
   uint4 y=lds128(smem_u32(s+cb*8192)+swz128(r,c*2));stg128(p.dg+(size_t)(row+r)*128+cb*64+c,y);}
 }
 fence_proxy_async();allsync();
}
// Anthropic WGMMA/ldmatrix/stmatrix fragment conventions, applied to LN backward.
// This is only for the separate DX CTA: no 192-register dW live set is present.
TMN_DEVI void dual_dgrad(const Params& p,uint8_t* sm,int m0,int slot){
 const int wi=threadIdx.x/128,tid=threadIdx.x%128;
 uint8_t* sx=sm+slot*98304+32768;
 fence_proxy_async();sync_group();
 if(tid==0){for(int ch=wi*128;ch<(wi+1)*128;ch+=16)tma_store_3d(&p.dtri,sx+ch*128,m0,ch,0);tma_store_commit();tma_store_wait_all();}
 sync_group();
}
// DW: two 96 KiB input slots; 192 FP32 dW accumulators per thread.
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* full){
 int split=blockIdx.x;
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32;
 MaskCycle mask=mask_cycle<DWCOUNT>(p,split);int mi=0,round=0;float acc[3][64]={};
 for(int it=split;it<p.tiles;it+=DWCOUNT,++round){int slot=round&1;
  // Two slots; each toggles parity only on its own reuse.
  mbar_wait(full+slot,(round/2)&1);uint8_t* s=sm+slot*98304;
  gate_backward<true>(p,s,it*64,mask,mi);
  (void)s; // No WGMMA in the traffic replica.
  bool next=it+2*DWCOUNT<p.tiles;
  if(next&&threadIdx.x==0)mbar_arrive_expect_tx(full+slot,98304);
  // CTA consumers finished before refilling this independent slot.
  allsync();
  if(next)issue_slot<true>(p,sm,full,slot,it+2*DWCOUNT,false);
  if(++mi==mask.period)mi=0;
 }
 static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*3+n,tile=t<2?0:1+(t-2)/2;float* part=p.partw+(tile*DWCOUNT+split)*16384;
#pragma unroll
  for(int q=0;q<16;++q){int rr=w*16+lane/4+(t<2?t*64:0),c=q*8+2*(lane%4)+(t<2?0:((t-2)%2)*128),stride=t<2?128:256;stg64f(part+rr*stride+c,acc[n][4*q],acc[n][4*q+1]);stg64f(part+(rr+8)*stride+c,acc[n][4*q+2],acc[n][4*q+3]);}
 });
}
TMN_DEVI void prefetch_stats(const Params& p,uint8_t* sm,int tile,int slot){
 if(threadIdx.x>=16)return;
 int r=tile*64+threadIdx.x*4;
 uint4 mu=ldg128(p.mean+r),rs=ldg128(p.rs+r);
 uint32_t dst=smem_u32(sm+73728+slot*512+threadIdx.x*16);
 asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(dst),"r"(mu.x),"r"(mu.y),"r"(mu.z),"r"(mu.w):"memory");
 asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(dst+256),"r"(rs.x),"r"(rs.y),"r"(rs.z),"r"(rs.w):"memory");
}
// DX: two64KiB dy/gate/tri slots, resident64KiB Wp,8KiB warp sums,
// and2KiB running LN partials. dnorm and tri fragments stay in registers.
// No dW accumulator is live during LN; no32KiB dnorm materialization.
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* full){
 int split=blockIdx.x-DWCOUNT;int mi=0,round=0;
 MaskCycle mask=mask_cycle<DXCOUNT>(p,split);
 for(int it=split;it<p.tiles;it+=DXCOUNT,++round){int slot=round&1;
  mbar_wait(full+slot,(round/2)&1);
  prefetch_stats(p,sm,it,slot);
  gate_backward<false>(p,sm+slot*98304,it*64,mask,mi);
  dual_dgrad(p,sm,it*64,slot);
  bool next=it+2*DXCOUNT<p.tiles;
  if(next&&threadIdx.x==0)mbar_arrive_expect_tx(full+slot,65536);
  // DX store completion protects tri reuse within this independent CTA.
  allsync();
  if(next)issue_slot<false>(p,sm,full,slot,it+2*DXCOUNT,false);
  if(++mi==mask.period)mi=0;
 }
 for(int j=threadIdx.x;j<512;j+=256)p.partln[split*512+j]=reinterpret_cast<float*>(sm+229376)[j];
}
extern "C" __global__ __launch_bounds__(256,1)
void dual_b1b4(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t full[2];
 const bool dw=blockIdx.x<DWCOUNT;int split=dw?blockIdx.x:blockIdx.x-DWCOUNT;int stride=dw?DWCOUNT:DXCOUNT;
 if(threadIdx.x==0){
  mbar_init(full,1);mbar_init(full+1,1);fence_barrier_init();
  if(split<p.tiles)mbar_arrive_expect_tx(full,dw?98304:131072);
  if(split+stride<p.tiles)mbar_arrive_expect_tx(full+1,dw?98304:65536);
 }
 for(int i=threadIdx.x;i<512;i+=256)reinterpret_cast<float*>(sm+229376)[i]=0;
 if(!dw)reinterpret_cast<float*>(sm+74752)[threadIdx.x]=p.gamma[threadIdx.x];
 // Publish initialized slot barriers and LN sums before issuing local TMA.
 allsync();
 if(dw){
  if(split<p.tiles)issue_slot<true>(p,sm,full,0,split,true);
  if(split+stride<p.tiles)issue_slot<true>(p,sm,full,1,split+stride,false);
  weight_role(p,sm,full);
 }else{
  if(split<p.tiles)issue_slot<false>(p,sm,full,0,split,true);
  if(split+stride<p.tiles)issue_slot<false>(p,sm,full,1,split+stride,false);
  input_role(p,sm,full);
 }
#if PART_ONLY == 2
 // Every thread fences its partials; CTA joins before publishing a ticket.
 __threadfence();allsync();
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}
 allsync(); // All DW and DX partials are globally visible.
 for(int i=blockIdx.x*256+threadIdx.x;i<49664;i+=UCOUNT*256){
  if(i<49152){int tile=i/16384,j=i%16384;float v=0;for(int b=0;b<DWCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*DWCOUNT+b)*16384+j];(tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);}
  else{int j=i-49152;float v=0;for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
 }
 __threadfence();allsync(); // Readers finish before completion tickets reset.
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void unified_reduce(__grid_constant__ const Params p){
 int i=blockIdx.x*256+threadIdx.x;
 if(i<49152){int tile=i/16384,j=i%16384;float v=0;for(int b=0;b<DWCOUNT;++b)v+=p.partw[(tile*DWCOUNT+b)*16384+j];(tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);}
 else if(i<49664){int j=i-49152;float v=0;for(int b=0;b<DXCOUNT;++b)v+=p.partln[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
}
