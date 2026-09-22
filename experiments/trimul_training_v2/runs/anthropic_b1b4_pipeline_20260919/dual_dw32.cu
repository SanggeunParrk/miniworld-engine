// Experiment: dual_dw32
// One variable on dual_ln_prefetch: the DW role streams 32-row tiles through FOUR 48KiB slots
// (0,49152,98304,147456) instead of two 96KiB slots, so up to three tiles (144KiB) are in flight per
// SM instead of one (96KiB). Same B1 arithmetic and rounding, same WGMMA operands with K=32 (two
// k16 steps), same FP32 accumulators and partials. DX is unchanged (64-row tiles, two slots).
// SPDX-License-Identifier: Apache-2.0
// Anthropic native v5 TMA/WGMMA/fragment primitives; MiniWorld B1-B4 training.
// Selected role ratio10:23: H100132 SMs ->40 weight CTAs,92 input/LN CTAs.
// One cooperative launch, two input stages per role. Same forward saves and
// BF16 dp/dg/dnorm rounding. No cluster/multicast, no intermediate global dp.
// Register mask cache supports up to3 phases; generic ds loads otherwise.
// A plan owns its workspace and runs on one stream, without reentrant calls.
// See SOL_AUDIT_20260920.md for the parent layout, validation and limits.
#define DW_RATIO 10
#define DX_RATIO 23
// SPDX-License-Identifier: Apache-2.0
// Inlined copy of dual_primitives.cuh (hashed with this source) with five extra tensor maps.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
struct Params {
 CUtensorMap dy,gate,proj,xn,norm,tri,wp,dtri;
 const __nv_bfloat16* ds;
 const float *mean,*rs,*gamma;
 __nv_bfloat16 *dg,*dwg,*dwp;
 float *dgam,*dbeta,*partw,*partln,*groupln;
 unsigned int *counts;
 int M,L,tiles,groups;
 CUtensorMap dy32,gate32,proj32,xn32,norm32; // dual_dw32: 32-row boxes for the DW role (appended, 64B aligned).
};
TMN_DEVI void sync_group(){named_bar_sync(1+threadIdx.x/128,128);}
TMN_DEVI float get(const uint8_t* s,int r,int c){return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(s+swz128(r,c*2)));}
TMN_DEVI void put(uint8_t* s,int r,int c,float v){*reinterpret_cast<__nv_bfloat16*>(s+swz128(r,c*2))=__float2bfloat16_rn(v);}
TMN_DEVI void load_trans(uint32_t (&f)[4][4],uint8_t* s){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8,r8=lane%8;
#pragma unroll
 for(int k=0;k<4;++k)ldsm_x4_t(f[k],smem_u32(s)+swz128(16*k+r8+8*((mat>>1)&1),(16*w+8*(mat&1))*2));
}
TMN_DEVI uint32_t pair_get(const uint8_t* s,int r,int c){return *reinterpret_cast<const uint32_t*>(s+swz128(r,c*2));}
TMN_DEVI void pair_put(uint8_t* s,int r,int c,uint32_t v){*reinterpret_cast<uint32_t*>(s+swz128(r,c*2))=v;}
TMN_DEVI float warp_sum(float x){
#pragma unroll
 for(int k=16;k;k>>=1)x+=__shfl_xor_sync(0xffffffff,x,k);
 return x;
}
// No CTA waits for another CTA. Last-arriver reduces published partials and resets
// its counter. Workspace belongs to one plan/stream; stream completion permits reuse.
TMN_DEVI bool ticket(unsigned int* p,unsigned int total,int* last){
 __threadfence();sync_group();
 if(threadIdx.x%128==0)*last=(atomicAdd(p,1u)==total-1);
 sync_group();return *last;
}
// NVIDIA WGMMA SS N128 operand contract; raw TMA buffers stay MN-major.
TMN_DEVI void mma_ss128(float (&d)[64],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %66, 0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 1, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]) : "l"(a),"l"(b),"r"(accumulate));
}
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
 if constexpr(DW){ // 32-row tile: dy 0..8K, gate 8..16K, proj 16..24K, xn 24..32K, norm 32..48K (4KiB column blocks).
  uint8_t* s=sm+slot*49152;int row=tile*32;
#pragma unroll
  for(int c=0;c<2;++c){tma_load_2d(s+c*4096,&p.dy32,full+slot,c*64,row);tma_load_2d(s+8192+c*4096,&p.gate32,full+slot,c*64,row);tma_load_2d(s+16384+c*4096,&p.proj32,full+slot,c*64,row);tma_load_2d(s+24576+c*4096,&p.xn32,full+slot,c*64,row);}
#pragma unroll
  for(int c=0;c<4;++c)tma_load_2d(s+32768+c*4096,&p.norm32,full+slot,c*64,row);
 }else{uint8_t* s=sm+slot*98304;int row=tile*64;
#pragma unroll
  for(int c=0;c<2;++c){tma_load_2d(s+c*8192,&p.dy,full+slot,c*64,row);tma_load_2d(s+16384+c*8192,&p.gate,full+slot,c*64,row);}
  tma_load_2d(s+32768,&p.tri,full+slot,row,0);
  if(initial)for(int n=0;n<4;++n)for(int k=0;k<2;++k)tma_load_2d(sm+163840+n*16384+k*8192,&p.wp,full+slot,k*64,n*64);
 }
}
struct MaskCycle{uint32_t b0,b1,b2,scale;int period;bool cached;};
// ROWS=64 (DX) or32 (DW): iteration i covers 8 channels of row (i%(8*ROWS))/8, column block i/(8*ROWS).
template<int STRIDE,int ROWS> TMN_DEVI MaskCycle mask_cycle(const Params& p,int first,int ntiles){
 MaskCycle v={0,0,0,0,0,false};int a=ROWS*STRIDE,b=p.L;while(b){int r=a%b;a=b;b=r;}
 v.period=p.L/a;v.cached=v.period<=3;
 if(v.cached&&first<ntiles)for(int z=0;z<v.period;++z){uint32_t bits=0;int j0=((first+z*STRIDE)*ROWS)%p.L;
  for(int i=threadIdx.x;i<ROWS*16;i+=256){int cb=i/(ROWS*8),r=(i%(ROWS*8))/8,c=(i%8)*8,jr=j0+r;if(jr>=p.L)jr-=p.L;
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
 constexpr int ROWS=DW?32:64,BLK=ROWS*128; // bytes per 64-channel column block
 int j0=row%p.L;uint32_t bits=mi==0?mask.b0:mi==1?mask.b1:mask.b2;
 for(int i=threadIdx.x;i<ROWS*16;i+=256){int cb=i/(ROWS*8),r=(i%(ROWS*8))/8,c=(i%8)*8,jr=j0+r;if(jr>=p.L)jr-=p.L;
  uint32_t sy=smem_u32(s+cb*BLK)+swz128(r,c*2),sg=smem_u32(s+2*BLK+cb*BLK)+swz128(r,c*2);
  uint4 y=lds128(sy),g=lds128(sg),v,ds,dp,dg;
  if constexpr(DW)v=lds128(smem_u32(s+4*BLK+cb*BLK)+swz128(r,c*2));
  if(mask.cached){uint32_t* dd=reinterpret_cast<uint32_t*>(&ds);
#pragma unroll
   for(int q=0;q<4;++q){int bit=8*(i/256)+2*q;dd[q]=((bits>>bit)&1?mask.scale:0)|((bits>>(bit+1))&1?mask.scale<<16:0);} // i/256 in 0..3 (DX) or 0..1 (DW)
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
 fence_proxy_async();allsync();
}
// Anthropic WGMMA/ldmatrix/stmatrix fragment conventions, applied to LN backward.
// This is only for the separate DX CTA: no 192-register dW live set is present.
TMN_DEVI void dual_dgrad(const Params& p,uint8_t* sm,int m0,int slot){
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32,mat=lane/8,r8=lane%8;
 uint8_t* sx=sm+slot*98304+32768;
 float* stats=reinterpret_cast<float*>(sm+slot*98304+16384);
 // 73728..74752: double-buffered saved mean/rstd. Gamma is resident
 // at74752..75776. B1's existing CTA barrier publishes the row statistics.
 float* mus=reinterpret_cast<float*>(sm+73728+slot*512);
 float* rss=mus+64;float* gam=reinterpret_cast<float*>(sm+74752);
 int ra=w*16+lane/4,rb=ra+8;
 float mu[2]={mus[ra],mus[rb]},rs[2]={rss[ra],rss[rb]},s1[2]={},s2[2]={};
 uint32_t fx[2][4][4],dn[2][4][4];
 static_for<2>([&](auto ni){constexpr int nlocal=decltype(ni)::value;int n=wi*2+nlocal;
  uint8_t* sw=sm+163840+n*16384;float acc[32]={};
  fence_regs(acc);wgmma_fence(); // Order initialized accumulators before async MMA.
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_dgrad(acc,smem_desc(smem_u32(sm+slot*98304+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(acc); // dnorm ready for BF16 rounding.
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
 allsync(); // Both channel halves of each row must be published before LN epilogue.
 float c1a=stats[ra*2]+stats[128+ra*2],c1b=stats[rb*2]+stats[128+rb*2];
 float c2a=stats[ra*2+1]+stats[128+ra*2+1],c2b=stats[rb*2+1]+stats[128+rb*2+1];
 // 65536..73728 holds8KiB per-warp LN partials; no dnorm shared round trip.
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
    // Same columns, eight row groups in the warp (each contributes two rows).
#pragma unroll
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba0+=__shfl_xor_sync(0xffffffff,dba0,sh);dbb0+=__shfl_xor_sync(0xffffffff,dbb0,sh);}
    if(lane<4){tmp[w*512+c]=dga;tmp[w*512+c+1]=dgb;tmp[w*512+256+c]=dba0;tmp[w*512+256+c+1]=dbb0;}
   });
   // All raw tri values were loaded before this point. Store dtri in the same
   // transposed matrix layout, without materializing or rereading dnorm.
   int chq=8*(mat>>1)+r8;
   stsm_x4_t(smem_u32(sx)+swz128(n*64+q*16+chq,(w*16+8*(mat&1))*2),out[0],out[1],out[2],out[3]);
  });
 });
 allsync(); // Publish all four warps' parameter partials (and all stmatrix writes).
 int c=threadIdx.x;float* red=reinterpret_cast<float*>(sm+229376);
 red[c]+=(tmp[c]+tmp[512+c])+(tmp[1024+c]+tmp[1536+c]);
 red[256+c]+=(tmp[256+c]+tmp[768+c])+(tmp[1280+c]+tmp[1792+c]);
 fence_proxy_async();sync_group(); // Generic dtri stores visible before each WG's TMA read.
 if(tid==0){for(int ch=wi*128;ch<(wi+1)*128;ch+=16)tma_store_3d(&p.dtri,sx+ch*128,m0,ch,0);tma_store_commit();tma_store_wait_all();}
 sync_group(); // TMA finished consuming this slot before cluster reuse.
}

// DW: four 48 KiB input slots of 32-row tiles; 192 FP32 dW accumulators per thread.
// Round r uses slot r&3 (parity (r>>2)&1); the slot is refilled with tile it+4*DWCOUNT after the MMAs.
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* full){
 int split=blockIdx.x;const int ntiles=2*p.tiles;
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32;
 MaskCycle mask=mask_cycle<DWCOUNT,32>(p,split,ntiles);int mi=0,round=0;float acc[3][64]={};
 for(int it=split;it<ntiles;it+=DWCOUNT,++round){int slot=round&3;
  mbar_wait(full+slot,(round>>2)&1);uint8_t* s=sm+slot*49152;
  gate_backward<true>(p,s,it*32,mask,mi);
  // A = xn (dWg) or dp (dWp) column block, MN-major; B = dg or norm, two 4KiB column blocks (LBO 4096).
  static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*3+n;uint8_t* sa=t<2?s+24576+t*4096:s+((t-2)/2)*4096;uint8_t* sb=t<2?s+8192:s+32768+((t-2)%2)*8192;
   fence_regs(acc[n]);wgmma_fence();
   static_for<2>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc[n],smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),4096,1024,1),it>split||k>0);});wgmma_commit();
  });wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);fence_regs(acc[2]);
  bool next=it+4*DWCOUNT<ntiles;
  if(next&&threadIdx.x==0)mbar_arrive_expect_tx(full+slot,49152);
  allsync(); // Both WGs finished reading this slot.
  if(next)issue_slot<true>(p,sm,full,slot,it+4*DWCOUNT,false);
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
 MaskCycle mask=mask_cycle<DXCOUNT,64>(p,split,p.tiles);
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
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t full[4];
 const bool dw=blockIdx.x<DWCOUNT;int split=dw?blockIdx.x:blockIdx.x-DWCOUNT;int stride=dw?DWCOUNT:DXCOUNT;
 const int ntiles=dw?2*p.tiles:p.tiles; // DW counts 32-row tiles.
 if(threadIdx.x==0){
  mbar_init(full,1);mbar_init(full+1,1);mbar_init(full+2,1);mbar_init(full+3,1);fence_barrier_init();
  if(dw){for(int q=0;q<4;++q)if(split+q*stride<ntiles)mbar_arrive_expect_tx(full+q,49152);}
  else{if(split<ntiles)mbar_arrive_expect_tx(full,131072);if(split+stride<ntiles)mbar_arrive_expect_tx(full+1,65536);}
 }
 for(int i=threadIdx.x;i<512;i+=256)reinterpret_cast<float*>(sm+229376)[i]=0;
 if(!dw)reinterpret_cast<float*>(sm+74752)[threadIdx.x]=p.gamma[threadIdx.x];
 // Publish initialized slot barriers and LN sums before issuing local TMA.
 allsync();
 if(dw){
  for(int q=0;q<4;++q)if(split+q*stride<ntiles)issue_slot<true>(p,sm,full,q,split+q*stride,q==0);
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
