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
constexpr int THREADS=SPLIT_ROLE==2?(DX_PC?256:128):DW_THREADS;
constexpr int PIPE_SG_BASE=DW_PAIR?81920:PIPE_CONSUMERS*40960,PIPE_W_BASE=PIPE_SG_BASE+(DW_PAIR?16384:PIPE_CONSUMERS*8192);
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
#if SPLIT_ROLE == 2
#define allsync() named_bar_sync(0,128)
#endif
#include "single_wg.inc"
#include "single_pc.inc"
#include "single_input.inc"
#if DW_PAIR
#include "dw_pair.inc"
#else
#include "b7_pipe_dw.inc"
#endif
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<131072){int group=i/8192,j=i%8192;float v=0;
  #if PIPE_HIDDEN64
  for(int b=0;b<DW_SPLITS*WGRAD_SLICES;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(((group/2)*DW_SPLITS+b/WGRAD_SLICES)*2*WGRAD_SLICES+(group%2)*WGRAD_SLICES+b%WGRAD_SLICES)*8192+j];
#else
  float v0=0,v1=0,v2=0,v3=0;
  for(int b=0;b<DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES;b+=4){
   const volatile float* q=p.partw+(group*DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES+b)*8192+j;
   v0+=q[0];if(b+1<DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES)v1+=q[8192];if(b+2<DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES)v2+=q[16384];if(b+3<DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES)v3+=q[24576];
  }v=(v0+v1)+(v2+v3);
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
   float v0=0,v1=0,v2=0,v3=0;
  for(int b=0;b<DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES;b+=4){
   const volatile float* q=p.partw+(group*DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES+b)*8192+j;
   v0+=q[0];if(b+1<DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES)v1+=q[8192];if(b+2<DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES)v2+=q[16384];if(b+3<DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES)v3+=q[24576];
  }v=(v0+v1)+(v2+v3);
#endif
   *reinterpret_cast<__nv_bfloat16*>(sm+swz128(i%16,(i/16)*2))=__float2bfloat16_rn(v);
  }
  fence_proxy_async();named_bar_sync(15,384);if(threadIdx.x==0){store2d(&p.xn,sm,h0,kind*128+c0);tma_store_commit();tma_store_wait_all();}
 }
 if(blockIdx.x==0&&threadIdx.x<256)reduce_at(p,131072+threadIdx.x);
}
#endif

// Independent launches specialize resource budgets and grid-wide reductions.
extern "C" __global__ __launch_bounds__(THREADS,(SPLIT_ROLE==2 || (PIPE_CONSUMERS==1 && SPLIT_ROLE==1))?2:1)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[10];__shared__ float gamma[128],beta[128];
 if(threadIdx.x<128){gamma[threadIdx.x]=p.gamma[threadIdx.x];beta[threadIdx.x]=p.beta[threadIdx.x];}
 if(threadIdx.x==0){for(int i=0;i<10;++i)mbar_init(bar+i,1);fence_barrier_init();}named_bar_sync(0,THREADS);
#if SPLIT_ROLE != 2
 if(SPLIT_ROLE==1 || blockIdx.x<DWCOUNT){
  if(threadIdx.x<128){setmaxnreg_dec<DW_PROD_REGS>();
   if(threadIdx.x==0){mbar_arrive_expect_tx(bar+(DW_PAIR?8:4*PIPE_CONSUMERS),16384);for(int k=0;k<2;++k)tma_load_2d(sm+PIPE_W_BASE+k*8192,&p.pre,bar+(DW_PAIR?8:4*PIPE_CONSUMERS),k*64,(blockIdx.x%PIPE_GROUPS)*64);}
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
#if DX_PC
  if(threadIdx.x<128){setmaxnreg_dec<32>();single_producer(p,sm,bar);}
  else{setmaxnreg_inc<224>();input_role(p,sm,bar,gamma,beta);}
#else
  input_role(p,sm,bar,gamma,beta);
#endif
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
