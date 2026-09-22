// Experiment: dual_ws3f_r50 (role ratio 25:41 = 50 DW CTAs)
// DX CTAs: warp specialization (CUTLASS/Anthropic style). Producer WG0: warp0 = TMA issuer (waits
// empty[slot], re-arms full[slot], issues round r+2), warps1-3 (96 threads) = stats prefetch + B1.
// Consumer WG1/WG2 = B3a WGMMA + B4 LN; they release a slot by arriving on empty[slot]. B1(r+1)
// overlaps LN(r). DW CTAs: the parent fused B1+B2/B3b schedule on WG1/WG2 (240 registers, mask words
// in shared memory); their WG0 idles at24 registers. B1 needs many warps, so DW keeps all eight.
// SPDX-License-Identifier: Apache-2.0
// Anthropic native v5 TMA/WGMMA/fragment/setmaxnreg primitives; MiniWorld B1-B4 training.
// One variable on dual_ln_prefetch: warp specialization. Each CTA has 384 threads:
//   WG0 (producer, WS_PROD_REGS registers): TMA issue, saved mean/rstd prefetch, B1 (dp/dg) into the slot.
//   WG1+WG2 (consumers, WS_CONS_REGS registers): DW = B2/B3b WGMMA accumulation; DX = B3a WGMMA + B4 LN backward.
// B1 of tile t+1 overlaps the consumers' MMA/LN of tile t. Same forward saves, BF16 dp/dg/dnorm
// rounding points, six outputs, 64-row tiles, two input slots, 40/92 CTA roles, PART1/PART2 reductions.
// Slot handshake: full[s] (TMA -> B1 warps), ready[s] (B1 warps -> consumers, 96 arrivals), empty[s] (consumers -> issuer warp, 256 arrivals).
// See SOL_AUDIT_20260920.md for the parent layout; this file keeps every shared-memory offset.
#define CTA_THREADS 384
#define DW_RATIO 25
#define DX_RATIO 41
#ifndef WS_PROD_REGS
#define WS_PROD_REGS 56
#endif
#ifndef WS_CONS_REGS
#define WS_CONS_REGS 224
#endif
#ifndef WS_B1_UNROLL
#define WS_B1_UNROLL 2
#endif
#define WS_DW_IDLE_REGS 24
#define WS_DW_REGS 240
static_assert(WS_PROD_REGS+2*WS_CONS_REGS<=3*168&&WS_DW_IDLE_REGS+2*WS_DW_REGS<=3*168,"register partitions must fit the 384-thread launch allocation");
#include "dual_primitives.cuh"
#ifndef PART_ONLY
#define PART_ONLY 0
#endif
#ifndef UCOUNT
#define UCOUNT 132
#endif
// Barrier ids: 0 = all 384 threads; 1..3 = one warpgroup each (sync_group); 4 = both consumer warpgroups.
TMN_DEVI void ctasync(){named_bar_sync(0,384);}
TMN_DEVI void consync(){named_bar_sync(4,256);}
TMN_DEVI void mma_dgrad(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
constexpr int DWCOUNT=UCOUNT*DW_RATIO/(DW_RATIO+DX_RATIO);
constexpr int DXCOUNT=UCOUNT-DWCOUNT;
static_assert(DWCOUNT>0&&DXCOUNT>0,"both CTA roles required");
// Producer thread 0 issues every TMA load of a slot after arming full[slot] itself.
template<bool DW> TMN_DEVI void issue_slot(const Params& p,uint8_t* sm,uint64_t* full,int slot,int tile,bool initial){
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
// B1 warps: 96 threads own iterations i=b1t+96k (k<=10). Lossless mask cache: 3 words per phase,
// word k/4, bit 8*(k%4)+2q; static word selection keeps the struct in registers. Periods above 3
// fall back to generic ds loads.
constexpr int B1T=96;
struct MaskCycle{uint32_t b[9];uint32_t scale;int period;bool cached;};
template<int STRIDE> TMN_DEVI MaskCycle mask_cycle(const Params& p,int first){
 MaskCycle v;v.scale=0;v.cached=false;
#pragma unroll
 for(int z=0;z<9;++z)v.b[z]=0;
 const int b1t=threadIdx.x-32;
 int a=64*STRIDE,bb=p.L;while(bb){int r=a%bb;a=bb;bb=r;}
 v.period=p.L/a;v.cached=v.period<=3;
 if(v.cached&&first<p.tiles)
#pragma unroll 1
 for(int z=0;z<v.period;++z){uint32_t w0=0,w1=0,w2=0;int j0=((first+z*STRIDE)*64)%p.L;
#pragma unroll 1
  for(int k=0;k<11;++k){int i=b1t+B1T*k;if(i>=1024)break;int cb=i/512,r=(i%512)/8,c=(i%8)*8,jr=j0+r;if(jr>=p.L)jr-=p.L;
   uint4 ds=ldg128(p.ds+jr*128+cb*64+c);uint32_t* dd=reinterpret_cast<uint32_t*>(&ds);uint32_t bits=0;
#pragma unroll
   for(int q=0;q<4;++q){uint32_t lo=dd[q]&65535u,hi=dd[q]>>16;int bit=8*(k%4)+2*q;bits|=(uint32_t(lo!=0)<<bit)|(uint32_t(hi!=0)<<(bit+1));v.scale|=lo|hi;}
   if(k<4)w0|=bits;else if(k<8)w1|=bits;else w2|=bits;
  }
  if(z==0){v.b[0]=w0;v.b[1]=w1;v.b[2]=w2;}else if(z==1){v.b[3]=w0;v.b[4]=w1;v.b[5]=w2;}else{v.b[6]=w0;v.b[7]=w1;v.b[8]=w2;}
 }
 return v;
}
// B1 by the 96 B1 threads: dp (and dg for DW) written in place over dy (and gate). Same FP32 products
// and BF16 rounding as the parent kernel. Each thread publishes its stores to the async proxy and
// arrives on ready[slot]; no producer-side barrier.
template<bool DW> TMN_DEVI void producer_b1(const Params& p,uint8_t* s,int row,const MaskCycle& mask,int mi,uint64_t* ready){
 const int b1t=threadIdx.x-32;int j0=row%p.L;
 uint32_t w0=mi==0?mask.b[0]:mi==1?mask.b[3]:mask.b[6],w1=mi==0?mask.b[1]:mi==1?mask.b[4]:mask.b[7],w2=mi==0?mask.b[2]:mi==1?mask.b[5]:mask.b[8];
#pragma unroll 1
 for(int k=0;k<11;++k){int i=b1t+B1T*k;if(i>=1024)break;int cb=i/512,r=(i%512)/8,c=(i%8)*8,jr=j0+r;if(jr>=p.L)jr-=p.L;
  uint32_t sy=smem_u32(s+cb*8192)+swz128(r,c*2),sg=smem_u32(s+16384+cb*8192)+swz128(r,c*2);
  uint4 y=lds128(sy),g=lds128(sg),v,ds,dp,dg;
  if constexpr(DW)v=lds128(smem_u32(s+32768+cb*8192)+swz128(r,c*2));
  if(mask.cached){uint32_t* dd=reinterpret_cast<uint32_t*>(&ds);uint32_t bits=k<4?w0:k<8?w1:w2;
#pragma unroll
   for(int q=0;q<4;++q){int bit=8*(k%4)+2*q;dd[q]=((bits>>bit)&1?mask.scale:0)|((bits>>(bit+1))&1?mask.scale<<16:0);}
  }else ds=ldg128(p.ds+jr*128+cb*64+c);
  uint32_t *yy=reinterpret_cast<uint32_t*>(&y),*gg=reinterpret_cast<uint32_t*>(&g),*vv=reinterpret_cast<uint32_t*>(&v),*dd=reinterpret_cast<uint32_t*>(&ds),*oo=reinterpret_cast<uint32_t*>(&dp),*zz=reinterpret_cast<uint32_t*>(&dg);
#pragma unroll
  for(int q=0;q<4;++q){float ya=bf16lo(yy[q])*bf16lo(dd[q]),yb=bf16hi(yy[q])*bf16hi(dd[q]),ga=bf16lo(gg[q]),gb=bf16hi(gg[q]);oo[q]=pack_bf16(ya*ga,yb*gb);
   if constexpr(DW)zz[q]=pack_bf16(((ya*bf16lo(vv[q]))*ga)*(1.f-ga),((yb*bf16hi(vv[q]))*gb)*(1.f-gb));
  }
  asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(sy),"r"(dp.x),"r"(dp.y),"r"(dp.z),"r"(dp.w):"memory");
  if constexpr(DW){asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(sg),"r"(dg.x),"r"(dg.y),"r"(dg.z),"r"(dg.w):"memory");stg128(p.dg+(size_t)(row+r)*128+cb*64+c,dg);}
 }
 fence_proxy_async(); // This thread's generic dp/dg stores become visible to WGMMA descriptors.
 mbar_arrive(ready); // ready[slot] counts the 96 B1 threads.
}
TMN_DEVI void prefetch_stats(const Params& p,uint8_t* sm,int tile,int slot){
 const int b1t=threadIdx.x-32;if(b1t>=16)return;
 int r=tile*64+b1t*4;
 uint4 mu=ldg128(p.mean+r),rs=ldg128(p.rs+r);
 uint32_t dst=smem_u32(sm+73728+slot*512+b1t*16);
 asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(dst),"r"(mu.x),"r"(mu.y),"r"(mu.z),"r"(mu.w):"memory");
 asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(dst+256),"r"(rs.x),"r"(rs.y),"r"(rs.z),"r"(rs.w):"memory");
}
// Producer WG. Warp0 lane0 issues rounds 0/1, then round r+2 as soon as empty[slot] reports that the
// consumers finished round r (completion number r>>1). Warps1-3 run stats prefetch + B1 per round.
template<bool DW> TMN_DEVI void producer(const Params& p,uint8_t* sm,uint64_t* full,uint64_t* ready,uint64_t* empty,int split,int stride){
 constexpr uint32_t BYTES=DW?98304u:65536u;
 if(threadIdx.x<32){
  if(threadIdx.x==0){
   if(split<p.tiles){mbar_arrive_expect_tx(full,DW?BYTES:BYTES+65536u);issue_slot<DW>(p,sm,full,0,split,true);}
   if(split+stride<p.tiles){mbar_arrive_expect_tx(full+1,BYTES);issue_slot<DW>(p,sm,full,1,split+stride,false);}
  }
  int round=2;
  for(int it=split+2*stride;it<p.tiles;it+=stride,++round){int slot=round&1;
   mbar_wait(empty+slot,((round>>1)-1)&1); // Consumers released round-2 from this slot.
   if(threadIdx.x==0){mbar_arrive_expect_tx(full+slot,BYTES);issue_slot<DW>(p,sm,full,slot,it,false);}
   __syncwarp();
  }
  return;
 }
 MaskCycle mask=mask_cycle<DW?DWCOUNT:DXCOUNT>(p,split);int mi=0,round=0;
 for(int it=split;it<p.tiles;it+=stride,++round){int slot=round&1;
  mbar_wait(full+slot,(round>>1)&1);
  if constexpr(!DW)prefetch_stats(p,sm,it,slot); // Published to consumers by this thread's ready arrive.
  producer_b1<DW>(p,sm+slot*98304,it*64,mask,mi,ready+slot);
  if(++mi==mask.period)mi=0;
 }
}
// Anthropic WGMMA/ldmatrix/stmatrix fragment conventions, applied to LN backward by the two consumer WGs.
TMN_DEVI void dual_dgrad(const Params& p,uint8_t* sm,int m0,int slot){
 const int ct=threadIdx.x-128,wi=ct/128,tid=ct%128,lane=tid%32,w=tid/32,mat=lane/8,r8=lane%8;
 uint8_t* sx=sm+slot*98304+32768;
 float* stats=reinterpret_cast<float*>(sm+slot*98304+16384);
 float* mus=reinterpret_cast<float*>(sm+73728+slot*512);
 float* rss=mus+64;float* gam=reinterpret_cast<float*>(sm+74752);
 int ra=w*16+lane/4,rb=ra+8;
 float mu[2]={mus[ra],mus[rb]},rs[2]={rss[ra],rss[rb]},s1[2]={},s2[2]={};
 uint32_t fx[2][4][4],dn[2][4][4];
 static_for<2>([&](auto ni){constexpr int nlocal=decltype(ni)::value;int n=wi*2+nlocal;
  uint8_t* sw=sm+163840+n*16384;float acc[32]={};
  fence_regs(acc);wgmma_fence();
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_dgrad(acc,smem_desc(smem_u32(sm+slot*98304+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(acc);
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   ldsm_x4_t(fx[nlocal][q],smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;
    dn[nlocal][q][j]=pack_bf16(acc[q*8+j*2],acc[q*8+j*2+1]);
    int rr=j&1,c=n*64+q*16+2*(lane%4)+8*(j>>1);
    float xa=__fmul_rn(__fsub_rn(bf16lo(fx[nlocal][q][j]),mu[rr]),rs[rr]);
    float xb=__fmul_rn(__fsub_rn(bf16hi(fx[nlocal][q][j]),mu[rr]),rs[rr]);
    float ha=bf16lo(dn[nlocal][q][j])*gam[c],hb=bf16hi(dn[nlocal][q][j])*gam[c+1];
    s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;
   });
  });
 });
 s1[0]=quad_sum(s1[0])/256.f;s1[1]=quad_sum(s1[1])/256.f;
 s2[0]=quad_sum(s2[0])/256.f;s2[1]=quad_sum(s2[1])/256.f;
 if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}
 consync(); // Both channel halves of each row are published before the LN epilogue.
 float c1a=stats[ra*2]+stats[128+ra*2],c1b=stats[rb*2]+stats[128+rb*2];
 float c2a=stats[ra*2+1]+stats[128+ra*2+1],c2b=stats[rb*2+1]+stats[128+rb*2+1];
 float* tmp=reinterpret_cast<float*>(sm+65536);
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;uint32_t out[4];
   static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;constexpr int j=pair*2;
    int c=n*64+q*16+2*(lane%4)+8*pair;
    uint32_t xa=fx[nl][q][j],xb=fx[nl][q][j+1],da=dn[nl][q][j],db=dn[nl][q][j+1];
    float xaa=__fmul_rn(__fsub_rn(bf16lo(xa),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(bf16hi(xa),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(bf16lo(xb),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(bf16hi(xb),mu[1]),rs[1]);
    float daa=bf16lo(da),dab=bf16hi(da),dba=bf16lo(db),dbb=bf16hi(db);
    float ga=gam[c],gb=gam[c+1];
    out[j]=pack_bf16(rs[0]*((daa*ga-c2a)-xaa*c1a),rs[0]*((dab*gb-c2a)-xab*c1a));
    out[j+1]=pack_bf16(rs[1]*((dba*ga-c2b)-xba*c1b),rs[1]*((dbb*gb-c2b)-xbb*c1b));
    float dga=daa*xaa+dba*xba,dgb=dab*xab+dbb*xbb,dba0=daa+dba,dbb0=dab+dbb;
#pragma unroll
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba0+=__shfl_xor_sync(0xffffffff,dba0,sh);dbb0+=__shfl_xor_sync(0xffffffff,dbb0,sh);}
    if(lane<4){tmp[w*512+c]=dga;tmp[w*512+c+1]=dgb;tmp[w*512+256+c]=dba0;tmp[w*512+256+c+1]=dbb0;}
   });
   int chq=8*(mat>>1)+r8;
   stsm_x4_t(smem_u32(sx)+swz128(n*64+q*16+chq,(w*16+8*(mat&1))*2),out[0],out[1],out[2],out[3]);
  });
 });
 consync(); // All four warps' parameter partials and all stmatrix writes are visible.
 int c=ct;float* red=reinterpret_cast<float*>(sm+229376);
 red[c]+=(tmp[c]+tmp[512+c])+(tmp[1024+c]+tmp[1536+c]);
 red[256+c]+=(tmp[256+c]+tmp[768+c])+(tmp[1280+c]+tmp[1792+c]);
 fence_proxy_async();sync_group(); // Generic dtri stores visible before each WG's TMA read.
 if(tid==0){for(int ch=wi*128;ch<(wi+1)*128;ch+=16)tma_store_3d(&p.dtri,sx+ch*128,m0,ch,0);tma_store_commit();tma_store_wait_all();}
 sync_group(); // TMA finished consuming this slot before it is released.
}
struct MaskCycle256{uint32_t b0,b1,b2,scale;int period;bool cached;};
template<int STRIDE> TMN_DEVI MaskCycle256 mask_cycle256(const Params& p,int first){
 const int ct=threadIdx.x-128; // consumer-local thread index (threads128..383)
 MaskCycle256 v={0,0,0,0,0,false};int a=64*STRIDE,b=p.L;while(b){int r=a%b;a=b;b=r;}
 v.period=p.L/a;v.cached=v.period<=3;
 if(v.cached&&first<p.tiles)for(int z=0;z<v.period;++z){uint32_t bits=0;int j0=((first+z*STRIDE)*64)%p.L;
  for(int i=ct;i<1024;i+=256){int cb=i/512,r=(i%512)/8,c=(i%8)*8,jr=j0+r;if(jr>=p.L)jr-=p.L;
   uint4 ds=ldg128(p.ds+jr*128+cb*64+c);uint32_t* dd=reinterpret_cast<uint32_t*>(&ds);
#pragma unroll
   for(int q=0;q<4;++q){uint32_t lo=dd[q]&65535u,hi=dd[q]>>16;int bit=8*(i/256)+2*q;
    bits|=(uint32_t(lo!=0)<<bit)|(uint32_t(hi!=0)<<(bit+1));v.scale|=lo|hi;}
  }
  if(z==0)v.b0=bits;else if(z==1)v.b1=bits;else v.b2=bits;
 }
 return v;
}
// Mask words live in shared memory (196608+ct*16, one word per phase) and the common scale at200704:
// this keeps the fused DW loop within its 240-register share (the parent used 252 of 255).
template<bool DW> TMN_DEVI void gate_backward_fused(const Params& p,uint8_t* sm,uint8_t* s,int row,int period,int mi){
 const int ct=threadIdx.x-128; // consumer-local thread index (threads128..383)
 int j0=row%p.L;const bool cached=period<=3;uint32_t bits=0,scale=0;
 if(cached){bits=*reinterpret_cast<const uint32_t*>(sm+196608+ct*16+mi*4);scale=*reinterpret_cast<const uint32_t*>(sm+200704);}
 for(int i=ct;i<1024;i+=256){int cb=i/512,r=(i%512)/8,c=(i%8)*8,jr=j0+r;if(jr>=p.L)jr-=p.L;
  uint32_t sy=smem_u32(s+cb*8192)+swz128(r,c*2),sg=smem_u32(s+16384+cb*8192)+swz128(r,c*2);
  uint4 y=lds128(sy),g=lds128(sg),v,ds,dp,dg;
  if constexpr(DW)v=lds128(smem_u32(s+32768+cb*8192)+swz128(r,c*2));
  if(cached){uint32_t* dd=reinterpret_cast<uint32_t*>(&ds);
#pragma unroll
   for(int q=0;q<4;++q){int bit=8*(i/256)+2*q;dd[q]=((bits>>bit)&1?scale:0)|((bits>>(bit+1))&1?scale<<16:0);}
  }else ds=ldg128(p.ds+jr*128+cb*64+c);
  uint32_t *yy=reinterpret_cast<uint32_t*>(&y),*gg=reinterpret_cast<uint32_t*>(&g),*vv=reinterpret_cast<uint32_t*>(&v),*dd=reinterpret_cast<uint32_t*>(&ds),*oo=reinterpret_cast<uint32_t*>(&dp),*zz=reinterpret_cast<uint32_t*>(&dg);
#pragma unroll
  for(int q=0;q<4;++q){float ya=bf16lo(yy[q])*bf16lo(dd[q]),yb=bf16hi(yy[q])*bf16hi(dd[q]),ga=bf16lo(gg[q]),gb=bf16hi(gg[q]);oo[q]=pack_bf16(ya*ga,yb*gb);
   if constexpr(DW)zz[q]=pack_bf16(((ya*bf16lo(vv[q]))*ga)*(1.f-ga),((yb*bf16hi(vv[q]))*gb)*(1.f-gb));
  }
  asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(sy),"r"(dp.x),"r"(dp.y),"r"(dp.z),"r"(dp.w):"memory");
  if constexpr(DW){asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(sg),"r"(dg.x),"r"(dg.y),"r"(dg.z),"r"(dg.w):"memory");stg128(p.dg+(size_t)(row+r)*128+cb*64+c,dg);}
 }
 // Publish generic B1 stores to the WGMMA proxy, then join CTA threads.
 fence_proxy_async();consync();
}
// DW (unchanged fused schedule from dual_ln_prefetch) on the two consumer WGs (threads128..383):
// B1 + B2/B3b with 192 FP32 accumulators per thread; the producer WG idles at24 registers.
TMN_DEVI void weight_fused(const Params& p,uint8_t* sm,uint64_t* full){
 const int ct=threadIdx.x-128; // consumer-local thread index (threads128..383)
 int split=blockIdx.x;
 const int wi=ct/128,tid=ct%128,lane=tid%32,w=tid/32;
 int period;{MaskCycle256 mask=mask_cycle256<DWCOUNT>(p,split);period=mask.period;
  *reinterpret_cast<uint32_t*>(sm+196608+ct*16)=mask.b0;*reinterpret_cast<uint32_t*>(sm+196608+ct*16+4)=mask.b1;*reinterpret_cast<uint32_t*>(sm+196608+ct*16+8)=mask.b2;
  if(ct==0)*reinterpret_cast<uint32_t*>(sm+200704)=mask.scale;}
 consync(); // Publish the common scale before the first B1.
 int mi=0,round=0;float acc[3][64]={};
 for(int it=split;it<p.tiles;it+=DWCOUNT,++round){int slot=round&1;
  // Two slots; each toggles parity only on its own reuse.
  mbar_wait(full+slot,(round/2)&1);uint8_t* s=sm+slot*98304;
  gate_backward_fused<true>(p,sm,s,it*64,period,mi);
  static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*3+n;uint8_t* sa=t<2?s+49152+t*8192:s+((t-2)/2)*8192;uint8_t* sb=t<2?s+16384:s+65536+((t-2)%2)*16384;
   // Fence before WGMMA, then commit/wait before refilling either operand.
   fence_regs(acc[n]);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc[n],smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),it>split||k>0);});wgmma_commit();
  });wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);fence_regs(acc[2]);
  bool next=it+2*DWCOUNT<p.tiles;
  if(next&&ct==0)mbar_arrive_expect_tx(full+slot,98304);
  // CTA consumers finished before refilling this independent slot.
  consync();
  if(next&&ct==0)issue_slot<true>(p,sm,full,slot,it+2*DWCOUNT,false); // ws3 issue_slot has no thread guard.
  if(++mi==period)mi=0;
 }
 static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*3+n,tile=t<2?0:1+(t-2)/2;float* part=p.partw+(tile*DWCOUNT+split)*16384;
#pragma unroll
  for(int q=0;q<16;++q){int rr=w*16+lane/4+(t<2?t*64:0),c=q*8+2*(lane%4)+(t<2?0:((t-2)%2)*128),stride=t<2?128:256;stg64f(part+rr*stride+c,acc[n][4*q],acc[n][4*q+1]);stg64f(part+(rr+8)*stride+c,acc[n][4*q+2],acc[n][4*q+3]);}
 });
}
TMN_DEVI void input_consumer(const Params& p,uint8_t* sm,uint64_t* ready,uint64_t* empty,int split){
 int round=0;
 for(int it=split;it<p.tiles;it+=DXCOUNT,++round){int slot=round&1;
  mbar_wait(ready+slot,(round>>1)&1);
  dual_dgrad(p,sm,it*64,slot);
  mbar_arrive(empty+slot); // After this WG's TMA store completed (dual_dgrad tail).
 }
}
extern "C" __global__ __launch_bounds__(384,1)
void dual_b1b4(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t full[2],ready[2],empty[2];
 const bool dw=blockIdx.x<DWCOUNT;int split=dw?blockIdx.x:blockIdx.x-DWCOUNT;int stride=dw?DWCOUNT:DXCOUNT;
 const bool prod=threadIdx.x<128;
 if(threadIdx.x==0){
  mbar_init(full,1);mbar_init(full+1,1);mbar_init(ready,B1T);mbar_init(ready+1,B1T);mbar_init(empty,256);mbar_init(empty+1,256);fence_barrier_init();
 }
 if(!prod){int ct=threadIdx.x-128;for(int i=ct;i<512;i+=256)reinterpret_cast<float*>(sm+229376)[i]=0;
  if(!dw)reinterpret_cast<float*>(sm+74752)[ct]=p.gamma[ct];}
 ctasync(); // Barriers, LN sums and gamma are initialized before any TMA or consumer read.
 if(dw){
  if(prod){setmaxnreg_dec<WS_DW_IDLE_REGS>();}
  else{setmaxnreg_inc<WS_DW_REGS>();int ct=threadIdx.x-128;
   if(ct==0){if(split<p.tiles){mbar_arrive_expect_tx(full,98304u);issue_slot<true>(p,sm,full,0,split,true);}
    if(split+stride<p.tiles){mbar_arrive_expect_tx(full+1,98304u);issue_slot<true>(p,sm,full,1,split+stride,false);}}
   weight_fused(p,sm,full);}
 }else if(prod){setmaxnreg_dec<WS_PROD_REGS>();producer<false>(p,sm,full,ready,empty,split,stride);}
 else{setmaxnreg_inc<WS_CONS_REGS>();input_consumer(p,sm,ready,empty,split);}
 ctasync(); // Consumers' running LN sums and dW partials are complete.
 if(!dw)for(int j=threadIdx.x;j<512;j+=384)p.partln[split*512+j]=reinterpret_cast<float*>(sm+229376)[j];
#if PART_ONLY == 2
 __threadfence();ctasync();
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}
 ctasync(); // All DW and DX partials are globally visible.
 for(int i=blockIdx.x*384+threadIdx.x;i<49664;i+=UCOUNT*384){
  if(i<49152){int tile=i/16384,j=i%16384;float v=0;for(int b=0;b<DWCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*DWCOUNT+b)*16384+j];(tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);}
  else{int j=i-49152;float v=0;for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
 }
 __threadfence();ctasync(); // Readers finish before completion tickets reset.
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void unified_reduce(__grid_constant__ const Params p){
 int i=blockIdx.x*256+threadIdx.x;
 if(i<49152){int tile=i/16384,j=i%16384;float v=0;for(int b=0;b<DWCOUNT;++b)v+=p.partw[(tile*DWCOUNT+b)*16384+j];(tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);}
 else if(i<49664){int j=i-49152;float v=0;for(int b=0;b<DXCOUNT;++b)v+=p.partln[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
}
