// On-chip full recomputation; no global xn/pre/LN-stat reads.
#ifndef WGRAD_SLICES
#define WGRAD_SLICES 2
#endif
// SPDX-License-Identifier: Apache-2.0
// Anthropic-derived training extension: one TMA producer WG and two dW consumers.
// dX roles reuse both gate/projection weights; all activations stay on chip.
#include "common_recompute.cuh"
#include "rs_recompute.cuh"
#ifndef SPLIT_ROLE
#define SPLIT_ROLE 0
#endif
#ifndef PIPE_CONSUMERS
#define PIPE_CONSUMERS 2
#endif
#ifndef DX_CTAS
#define DX_CTAS 132
#endif
#ifndef DW_PROD_REGS
#define DW_PROD_REGS 40
#endif
#ifndef DW_CONS_REGS
#define DW_CONS_REGS 232
#endif
constexpr int DW_THREADS=128*(1+PIPE_CONSUMERS);
constexpr int THREADS=SPLIT_ROLE==2?256:DW_THREADS;
constexpr int PIPE_SG_BASE=PIPE_CONSUMERS*40960,PIPE_W_BASE=PIPE_SG_BASE+PIPE_CONSUMERS*8192;
#ifndef PIPE_HIDDEN64
#define PIPE_HIDDEN64 0
#endif
#ifndef B7_LN_SERIAL
#define B7_LN_SERIAL 1
#endif
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
#if PIPE_HIDDEN64
constexpr int PIPE_GROUPS=8;
#else
constexpr int PIPE_GROUPS=16;
#endif
constexpr int DWCOUNT=PIPE_GROUPS*DW_SPLITS,DXCOUNT=SPLIT_ROLE?DX_CTAS:UCOUNT-DWCOUNT;
constexpr int DW_SLOT=57344,DX_SLOT=40960;
static_assert(DXCOUNT>0,"invalid partition");
struct Params{
 CUtensorMap dl,dr,pre,xn,dg,wlg,wl,wrg,wr,wgate,x,res,dx;
 const __nv_bfloat16* mask;const float *mean,*rs,*gamma;
 __nv_bfloat16* dw;float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;__nv_bfloat16 *debugdc,*debugxn;int M,L,tiles;const float* beta;
};
TMN_DEVI void mma_weight64(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
TMN_DEVI void store2d(const CUtensorMap* map,const void* src,int c,int r){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2,%3}], [%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");
}

TMN_DEVI void load_g(const Params& p,uint8_t* sm,uint64_t* b,int row,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b+slot,40960);
 for(int wg=0;wg<2;++wg)for(int k=0;k<2;++k)tma_load_2d(s+wg*16384+k*8192,&p.pre,b+slot,k*64,(side*4+h)*128+wg*64);
 tma_load_2d(s+32768,side?&p.dr:&p.dl,b+slot,row,h*64);tma_load_2d(s+36864,side?&p.dr:&p.dl,b+slot,row,h*64+32);
}
TMN_DEVI void load_p(const Params& p,uint8_t* sm,uint64_t* b,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b+slot,16384);
 for(int n=0;n<2;++n)tma_load_2d(s+n*8192,side?&p.wr:&p.wl,b+slot,h*64,n*64);
}
TMN_DEVI void issue_gate(const Params& p,uint8_t* sm,uint64_t* bar,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(bar,49152);
#if B7_XN_EARLY
 mbar_arrive_expect_tx(bar+2,16384);for(int c=0;c<2;++c)tma_load_2d(sm+81920+c*8192,USE_SAVED_XN?&p.xn:&p.x,bar+2,c*64,row);
#endif

 for(int k=0;k<2;++k){tma_load_2d(sm+k*8192,&p.dg,bar,k*64,row);
  for(int n=0;n<2;++n)tma_load_2d(sm+16384+n*16384+k*8192,&p.wgate,bar,k*64,n*64);}
}
TMN_DEVI void load_ln_next(const Params& p,uint8_t* sm,uint64_t* bar,int row){if(threadIdx.x)return;mbar_arrive_expect_tx(bar,32768);for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.x,bar,c*64,row);tma_load_2d(sm+16384+c*8192,&p.res,bar,c*64,row);}}
#include "b7_packed_dx.inc"
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma,const float* beta){
 int split=blockIdx.x-(SPLIT_ROLE==2?0:DWCOUNT),wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int ra=w*16+lane/4,rb=ra+8;float running=0;
 if(split<p.tiles)issue_gate(p,sm,bar+2,split*64);int round=0;for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){
  int row=tile*64;uint32_t gate_packed[16];float acc[32]={};
#if B7_MASK_HOIST
  uint32_t maskA=__bfloat16_as_ushort(p.mask[row+ra]),maskB=__bfloat16_as_ushort(p.mask[row+rb]);maskA|=maskA<<16;maskB|=maskB<<16;
#else
  uint32_t maskA=0,maskB=0;
#endif
  // Per row tile: bar0 gate+2channel+LN transactions (4); bar1 2channels.
  // Both phase parities return to0 before advancing to the next row tile.
  mbar_wait(bar+2,round&1);
  {float gate[32]={};
  fence_regs(gate);wgmma_fence();
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_dgrad(gate,smem_desc(smem_u32(sm+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+16384+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
  // B9 rounds separately before B10's accumulation/add.
  static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;gate_packed[j]=pack_bf16(gate[j*2],gate[j*2+1]);});}

  allsync();
  
#if !B7_XN_EARLY
  if(threadIdx.x==0){mbar_arrive_expect_tx(bar+4,16384);for(int c=0;c<2;++c)tma_load_2d(sm+81920+c*8192,USE_SAVED_XN?&p.xn:&p.x,bar+4,c*64,row);}
#endif

  mbar_wait(bar+4,round&1);
  uint32_t xn_frag[8][4];load_frag_bf16<8,8192>(xn_frag,smem_u32(sm+81920),w*16,lane);
#if !USE_SAVED_XN
  ln_recompute_fragment<8,B7_LN_SERIAL>(xn_frag,gamma,beta,lane,1e-5f);
#endif
  allsync();
  if(threadIdx.x==0&&round>0)tma_store_wait_all();allsync();
#if DX_RESIDENT
  for(int side=0;side<2;++side){
   if(threadIdx.x==0){mbar_arrive_expect_tx(bar+7,131072);for(int h=0;h<4;++h)for(int wg=0;wg<2;++wg)for(int k=0;k<2;++k)tma_load_2d(sm+h*32768+wg*16384+k*8192,&p.pre,bar+7,k*64,(side*4+h)*128+wg*64);}
   auto load_dy=[&](int h){if(threadIdx.x)return;int slot=h&1;uint8_t* dst=sm+131072+slot*8192;mbar_arrive_expect_tx(bar+5+slot,8192);tma_load_2d(dst,side?&p.dr:&p.dl,bar+5+slot,row,h*64);tma_load_2d(dst+4096,side?&p.dr:&p.dl,bar+5+slot,row,h*64+32);};
   load_dy(0);load_dy(1);mbar_wait(bar+7,side);
   for(int h=0;h<4;++h){int slot=h&1;uint8_t* wb=sm+h*32768;uint8_t* dy=sm+131072+slot*8192;mbar_wait(bar+5+slot,h/2);packed_derivatives(p,xn_frag,wb,dy,dy,sm+147456+h*8192,row,maskA,maskB);
    fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_dxpacked(acc,smem_desc(smem_u32(dy+k*2048),16,1024,1),smem_desc(smem_u32(wb+(k/2)*16384+wi*8192+(k%2)*2048),8192,1024,1),side>0||h>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)load_dy(h+2);
   }
   if(side==1)load_ln_next(p,sm+180224,bar+3,row);
   fence_regs(acc);wgmma_fence();
   static_for<4>([&](auto hh){constexpr int h=decltype(hh)::value;static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
    mma_dxpacked(acc,smem_desc(smem_u32(sm+147456+h*8192+k*2048),16,1024,1),smem_desc(smem_u32(sm+h*32768+(k/2)*16384+wi*8192+(k%2)*2048+4096),8192,1024,1),1);
   });});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
  }
#else
  for(int side=0;side<2;++side){
   load_g(p,sm,bar,row,side,0);load_g(p,sm,bar,row,side,1);
   for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,h/2);packed_derivatives(p,xn_frag,s,s+32768,s+32768,sm+81920+h*8192,row,maskA,maskB);
    fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_dxpacked(acc,smem_desc(smem_u32(s+32768+k*2048),16,1024,1),smem_desc(smem_u32(s+(k/2)*16384+wi*8192+(k%2)*2048),8192,1024,1),side>0||h>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)load_g(p,sm,bar,row,side,h+2);
   }
   load_p(p,sm,bar,side,0);load_p(p,sm,bar,side,1);
   for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,h/2);fence_regs(acc);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_input64(acc,smem_desc(smem_u32(sm+81920+h*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+wi*8192+k*32),16,1024,1),1);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)load_p(p,sm,bar,side,h+2);if(side==1&&h==1)load_ln_next(p,sm+65536,bar+3,row);
   }
  }
#endif
  if(tile+DXCOUNT<p.tiles)issue_gate(p,sm,bar+2,(tile+DXCOUNT)*64);
  // B10 outputs BF16 dx_n. Keep it in registers through B11/B12.
  static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;acc[j*2]=__bfloat162float(__float2bfloat16_rn(acc[j*2]+bf16lo(gate_packed[j])));acc[j*2+1]=__bfloat162float(__float2bfloat16_rn(acc[j*2+1]+bf16hi(gate_packed[j])));});
  uint8_t* lnsm=sm+
#if DX_RESIDENT
   180224;
#else
   65536;
#endif
mbar_wait(bar+3,round&1);
#if B7_LN_MODE
  uint32_t raw_all[8][4];load_frag_bf16<8,8192>(raw_all,smem_u32(lnsm),w*16,lane);
  LnStats recstats=ln_stats_only<8>(raw_all,gamma,beta,lane,1e-5f);
#if B7_LN_MODE == 2
  uint32_t raw_half[4][4];
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;raw_half[q][j]=wi==0?raw_all[q][j]:raw_all[q+4][j];});});
#endif
#else
  LnStats recstats=normalize_tile<8,false,false,false>(lnsm,nullptr,gamma,beta);
#endif
  float mu[2]={recstats.mA,recstats.mB},rs[2]={recstats.rA,recstats.rB},s1[2]={},s2[2]={};
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;int rr=j&1,c=q*16+2*(lane%4)+8*(j>>1),r=rr?rb:ra;
    #if B7_LN_MODE == 2
    float xa=__fmul_rn(__fsub_rn(bf16lo(raw_half[q][j]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(raw_half[q][j]),mu[rr]),rs[rr]);
#else
    float xa=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,r,c),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,r,c+1),mu[rr]),rs[rr]);
#endif
    float ha=acc[q*8+j*2]*gamma[wi*64+c],hb=acc[q*8+j*2+1]*gamma[wi*64+c+1];s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;
#if DEBUG_SAVE
    p.debugxn[(size_t)(row+r)*128+wi*64+c]=__float2bfloat16_rn(acc[q*8+j*2]);p.debugxn[(size_t)(row+r)*128+wi*64+c+1]=__float2bfloat16_rn(acc[q*8+j*2+1]);
#endif
   });
  });
  s1[0]=quad_sum(s1[0])/128.f;s1[1]=quad_sum(s1[1])/128.f;s2[0]=quad_sum(s2[0])/128.f;s2[1]=quad_sum(s2[1])/128.f;
  float* stats=reinterpret_cast<float*>(lnsm+36864);float* tmp=reinterpret_cast<float*>(lnsm+32768);
  if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}
  allsync(); // Publish both64-channel halves before full128-channel LN reduction.
  float c1[2]={stats[ra*2]+stats[128+ra*2],stats[rb*2]+stats[128+rb*2]},c2[2]={stats[ra*2+1]+stats[128+ra*2+1],stats[rb*2+1]+stats[128+rb*2+1]};
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;int j=pair*2,c=q*16+2*(lane%4)+8*pair,gc=wi*64+c;
    #if B7_LN_MODE == 2
    float xaa=__fmul_rn(__fsub_rn(bf16lo(raw_half[q][j]),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(bf16hi(raw_half[q][j]),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(bf16lo(raw_half[q][j+1]),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(bf16hi(raw_half[q][j+1]),mu[1]),rs[1]);
#else
    float xaa=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,ra,c),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,ra,c+1),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,rb,c),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(get(lnsm+wi*8192,rb,c+1),mu[1]),rs[1]);
#endif
    float da=acc[q*8+j*2],db=acc[q*8+j*2+1],dc=acc[q*8+j*2+2],dd=acc[q*8+j*2+3],ga=gamma[gc],gb=gamma[gc+1];
    uint32_t outa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
    uint32_t outb=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
    uint32_t resa=pair_get(lnsm+16384+wi*8192,ra,c),resb=pair_get(lnsm+16384+wi*8192,rb,c);
    *reinterpret_cast<uint32_t*>(lnsm+wi*8192+swz128(ra,c*2))=pack_bf16(bf16lo(outa)+bf16lo(resa),bf16hi(outa)+bf16hi(resa));
    *reinterpret_cast<uint32_t*>(lnsm+wi*8192+swz128(rb,c*2))=pack_bf16(bf16lo(outb)+bf16lo(resb),bf16hi(outb)+bf16hi(resb));
    float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
    if(lane<4){tmp[w*256+gc]=dga;tmp[w*256+gc+1]=dgb;tmp[w*256+128+gc]=dba;tmp[w*256+128+gc+1]=dbb;}
   });
  });
  allsync(); // All warp parameter partials and dx stores published.
  running+=(tmp[threadIdx.x]+tmp[256+threadIdx.x])+(tmp[512+threadIdx.x]+tmp[768+threadIdx.x]);
  fence_proxy_async();allsync();
  if(threadIdx.x==0){store2d(&p.dx,lnsm,0,row);store2d(&p.dx,lnsm+8192,64,row);tma_store_commit();}
  // dx store overlaps the next gate contraction; drained before front buffers refill.
 }
 if(threadIdx.x==0)tma_store_wait_all();allsync();
 p.partln[split*256+threadIdx.x]=running;
}
#include "b7_pipe_dw.inc"
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<131072){int group=i/8192,j=i%8192;float v=0;
  #if PIPE_HIDDEN64
  for(int b=0;b<DW_SPLITS*WGRAD_SLICES;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(((group/2)*DW_SPLITS+b/WGRAD_SLICES)*2*WGRAD_SLICES+(group%2)*WGRAD_SLICES+b%WGRAD_SLICES)*8192+j];
#else
  for(int b=0;b<DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES+b)*8192+j];
#endif
  int row=j/128,out=(group/8)*2+(row<32?1:0),h=(group%8)*32+row%32;p.dw[(out*128+j%128)*256+h]=__float2bfloat16_rn(v);
 }else if(i<131328){int c=i-131072;float v=0;for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}
}
#if DW_REDUCE_TMA
TMN_DEVI void reduce_tile(const Params& p,uint8_t* sm){
 // 128 CTAs cover 4 matrices x 8 input tiles x 4 hidden tiles.
 int kind=blockIdx.x/32,c0=((blockIdx.x%32)/4)*16,h0=(blockIdx.x%4)*64;
 if(blockIdx.x<128){
  for(int i=threadIdx.x;i<1024;i+=384){int h=h0+i/16,c=c0+i%16,group=(kind/2)*8+h/32,j=((kind&1)?0:32)*128+(h%32)*128+c;float v=0;
#if PIPE_HIDDEN64
   for(int b=0;b<DW_SPLITS*WGRAD_SLICES;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(((group/2)*DW_SPLITS+b/WGRAD_SLICES)*2*WGRAD_SLICES+(group%2)*WGRAD_SLICES+b%WGRAD_SLICES)*8192+j];
#else
   for(int b=0;b<DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES+b)*8192+j];
#endif
   *reinterpret_cast<__nv_bfloat16*>(sm+swz128(i%16,(i/16)*2))=__float2bfloat16_rn(v);
  }
  fence_proxy_async();named_bar_sync(15,384);if(threadIdx.x==0){store2d(&p.xn,sm,h0,kind*128+c0);tma_store_commit();tma_store_wait_all();}
 }
 if(blockIdx.x==0&&threadIdx.x<256)reduce_at(p,131072+threadIdx.x);
}
#endif

// Independent launches specialize resource budgets and grid-wide reductions.
extern "C" __global__ __launch_bounds__(THREADS,(PIPE_CONSUMERS==1 && SPLIT_ROLE==1)?2:1)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[10];__shared__ float gamma[128],beta[128];
 if(threadIdx.x<128){gamma[threadIdx.x]=p.gamma[threadIdx.x];beta[threadIdx.x]=p.beta[threadIdx.x];}
 if(threadIdx.x==0){for(int i=0;i<10;++i)mbar_init(bar+i,1);fence_barrier_init();}named_bar_sync(0,THREADS);
#if SPLIT_ROLE != 2
 if(SPLIT_ROLE==1 || blockIdx.x<DWCOUNT){
  if(threadIdx.x<128){setmaxnreg_dec<DW_PROD_REGS>();
   if(threadIdx.x==0){mbar_arrive_expect_tx(bar+4*PIPE_CONSUMERS,16384);for(int k=0;k<2;++k)tma_load_2d(sm+PIPE_W_BASE+k*8192,&p.pre,bar+4*PIPE_CONSUMERS,k*64,(blockIdx.x%PIPE_GROUPS)*64);}
   pipe_producer(p,sm,bar);
  }else{setmaxnreg_inc<DW_CONS_REGS>();pipe_consumer(p,sm,bar,gamma,beta);}
 }
#endif
#if SPLIT_ROLE != 1
 if(SPLIT_ROLE==2 || blockIdx.x>=DWCOUNT){
#if SPLIT_ROLE == 0
  if(threadIdx.x>=256)setmaxnreg_dec<32>();
  else{setmaxnreg_inc<232>();input_role(p,sm,bar,gamma,beta);}
#else
  input_role(p,sm,bar,gamma,beta);
#endif
 }
#endif
 named_bar_sync(15,THREADS);
 __threadfence();named_bar_sync(15,THREADS);
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}named_bar_sync(15,THREADS);
#if SPLIT_ROLE == 1
 for(int i=blockIdx.x*THREADS+threadIdx.x;i<131072;i+=UCOUNT*THREADS)reduce_at(p,i);
#elif SPLIT_ROLE == 2
 for(int i=blockIdx.x*THREADS+threadIdx.x;i<256;i+=UCOUNT*THREADS)reduce_at(p,131072+i);
#else
 for(int i=blockIdx.x*THREADS+threadIdx.x;i<131328;i+=UCOUNT*THREADS)reduce_at(p,i);
#endif
 __threadfence();named_bar_sync(15,THREADS);
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
}
