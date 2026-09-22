// SPDX-License-Identifier: Apache-2.0
// Anthropic TMA/WGMMA/LN primitives; streamed on-chip training derivatives.
#include "common_recompute.cuh"
#ifndef PRODUCER
#define PRODUCER 0
#endif
#ifndef B1_CONS_REGS
#define B1_CONS_REGS 240
#endif
#ifndef B1_PROD_REGS
#define B1_PROD_REGS 24
#endif
#define THREADS (PRODUCER?384:256)
static_assert(!PRODUCER || 256*B1_CONS_REGS+128*B1_PROD_REGS <= 384*168, "register redistribution exceeds CTA pool");
#ifndef UCOUNT
#define UCOUNT 132
#endif
#ifndef PART_ONLY
#define PART_ONLY 2
#endif
#ifndef B1_LN_SERIAL
#define B1_LN_SERIAL 1
#endif

#ifndef TRAIN_L
#define TRAIN_L p.L
#endif

struct Params {
 CUtensorMap dy,x,xhat,wp,wg,dtri,dgmap;
 const __nv_bfloat16* ds;const float *gi,*bi,*gamma,*bo,*saved_rs;
 __nv_bfloat16 *dg,*dwg,*dwp;float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;int M,L,tiles;
};
#include "b1_pipeline_math.inc"
#include "xhat_read.inc"
#include "b1_lowreg.inc"
#include "b1_stream_ln.inc"
TMN_DEVI void dg_store(const CUtensorMap* map,const void* src,int c,int r){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2,%3}], [%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");}
struct FragmentMask{uint32_t b0,b1,b2,scale;
#if B1_EXT_MASK
 uint32_t b3,b4,b5,b6,b7,b8,b9,b10,b11;
#endif
 int period;bool cached;};
template<int STRIDE> TMN_DEVI FragmentMask fragment_mask(const Params& p,int first,int half=-1){
 FragmentMask m={};int a=64*STRIDE,b=TRAIN_L;while(b){int r=a%b;a=b;b=r;}m.period=TRAIN_L/a;m.cached=m.period<=
#if B1_EXT_MASK
 12;
#else
 3;
#endif
 int wi=half<0?threadIdx.x/128:half,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 if(m.cached && first<p.tiles)for(int z=0;z<m.period;++z){int row=((first+z*STRIDE)*64)%TRAIN_L;uint32_t bits=0;
  #pragma unroll
  for(int q=0;q<4;++q){
   #pragma unroll
   for(int j=0;j<4;++j){int r=row+w*16+lane/4+8*(j&1);if(r>=TRAIN_L)r-=TRAIN_L;int c=wi*64+q*16+2*(lane%4)+8*(j>>1);
    uint32_t ds=*reinterpret_cast<const uint32_t*>(p.ds+r*128+c);bits|=(uint32_t((ds&65535)!=0)<<(q*8+j*2))|(uint32_t((ds>>16)!=0)<<(q*8+j*2+1));m.scale|=(ds&65535)|(ds>>16);
   }
  }
  if(z==0)m.b0=bits;else if(z==1)m.b1=bits;else if(z==2)m.b2=bits;
#if B1_EXT_MASK
  else if(z==3)m.b3=bits;else if(z==4)m.b4=bits;else if(z==5)m.b5=bits;else if(z==6)m.b6=bits;else if(z==7)m.b7=bits;else if(z==8)m.b8=bits;else if(z==9)m.b9=bits;else if(z==10)m.b10=bits;else m.b11=bits;
#endif
 }
 return m;
}
TMN_DEVI uint32_t mask_bits(const FragmentMask& m,int mi){
#if B1_EXT_MASK
 return mi==0?m.b0:mi==1?m.b1:mi==2?m.b2:mi==3?m.b3:mi==4?m.b4:mi==5?m.b5:mi==6?m.b6:mi==7?m.b7:mi==8?m.b8:mi==9?m.b9:mi==10?m.b10:m.b11;
#else
 return mi==0?m.b0:mi==1?m.b1:m.b2;
#endif
}
TMN_DEVI void load_raw(const Params& p,uint8_t* sm,uint64_t* bar,int slot,int row){
 if(threadIdx.x)return;uint8_t* dst=sm+16384+slot*49152;mbar_arrive_expect_tx(bar+slot,49152);
 tma_load_2d(dst,&p.x,bar+slot,0,row);tma_load_2d(dst+8192,&p.x,bar+slot,64,row);
#if XHAT_FP32
 for(int c=0;c<2;++c)for(int k=0;k<2;++k)tma_load_2d(sm+114688+k*16384+c*8192,&p.wg,bar+slot,c*64,k*64);
#else
 tma_load_2d(dst+16384,&p.xhat,bar+slot,row,0);
#endif

}
TMN_DEVI void load_dy(const Params& p,uint8_t* sm,uint64_t* bar,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(bar,16384);tma_load_2d(sm,&p.dy,bar,0,row);tma_load_2d(sm+8192,&p.dy,bar,64,row);
}
#ifndef GATE_PHASE
#define GATE_PHASE 0
#endif
// Each CTA owns rows and all three weight-gradient accumulators.
// The opposite raw slot becomes normalized output + dProj staging until
// weight consumers finish. It is then reused by the next asynchronous TMA load.
TMN_DEVI void prepare_shared(const Params& p,uint8_t* sm,uint64_t* bars,int slot,int row,int phase,const FragmentMask& mask,int mi){
 const int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
 uint8_t* x=sm+16384+slot*49152;
 uint8_t* norm=sm+16384+(1-slot)*49152;
 float* ps=reinterpret_cast<float*>(sm+212992);
 // Input x_n is already saved by forward.
 float acc[32];uint32_t gate[16];
 recompute_gemm<128,16384>(acc,x,sm+114688+wi*8192);
 #pragma unroll
 for(int j=0;j<16;++j)gate[j]=pack_bf16(math::sigmoid(math::round_bf16(acc[j*2])),math::sigmoid(math::round_bf16(acc[j*2+1])));
 fence_proxy_async();allsync();
#if XHAT_FP32
 // Wg has finished: reuse its 32 KiB tile for the second half of xhat.
 if(threadIdx.x==0){mbar_arrive_expect_tx(bars+4,65536);tma_load_2d(x+16384,&p.xhat,bars+4,row,0);tma_load_2d(sm+114688,&p.xhat,bars+4,row+32,0);}
 mbar_wait(bars+4,phase);
#endif
 if(wi==1){
  int ra=w*16+lane/4,rb=ra+8;
  if(lane%4==0){float* rs=reinterpret_cast<float*>(sm+225280)+64;rs[ra]=p.saved_rs[row+ra];rs[rb]=p.saved_rs[row+rb];}
  #pragma unroll 4
  for(int k=0;k<16;++k){uint32_t f[4];int c=16*k+2*(lane&3);
   #pragma unroll
   for(int j=0;j<4;++j){int r=(j&1)?rb:ra,cc=c+8*(j>>1);float a=xhat_at(sm,slot,r,cc),b=xhat_at(sm,slot,r,cc+1);f[j]=pack_bf16(__fmaf_rn(a,ps[256+cc],ps[512+cc]),__fmaf_rn(b,ps[257+cc],ps[513+cc]));}
   stsm_x4(smem_u32(norm+(k/4)*8192)+swz128(w*16+lane%8+8*(mat&1),(2*(k%4)+(mat>>1))*16),f[0],f[1],f[2],f[3]);
  }
 }
 fence_proxy_async();allsync();
 recompute_gemm<256,16384>(acc,norm,sm+147456+wi*8192);
 mbar_wait(bars+2,phase);
 #pragma unroll
 for(int q=0;q<4;++q){uint32_t dy[4],dg[4],dp[4];
  uint32_t off=swz128(w*16+lane%8+8*(mat&1),(2*q+(mat>>1))*16);
  ldsm_x4(dy,smem_u32(sm+wi*8192)+off);
  #pragma unroll
  for(int j=0;j<4;++j){int r=w*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j>>1),jr=(row+r)%TRAIN_L;uint32_t ds;
   if(mask.cached){int bit=q*8+j*2;uint32_t bits=mask_bits(mask,mi);ds=((bits>>bit)&1?mask.scale:0)|((bits>>(bit+1))&1?mask.scale<<16:0);}
   else ds=*reinterpret_cast<const uint32_t*>(p.ds+jr*128+wi*64+c);
   float a=bf16lo(dy[j])*bf16lo(ds),b=bf16hi(dy[j])*bf16hi(ds),ga=bf16lo(gate[q*4+j]),gb=bf16hi(gate[q*4+j]);
   dp[j]=pack_bf16(a*ga,b*gb);
   dg[j]=pack_bf16(((a*math::round_bf16(acc[q*8+j*2]))*ga)*(1.f-ga),((b*math::round_bf16(acc[q*8+j*2+1]))*gb)*(1.f-gb));
  }
  stsm_x4(smem_u32(sm+wi*8192)+off,dg[0],dg[1],dg[2],dg[3]);
  stsm_x4(smem_u32(norm+32768+wi*8192)+off,dp[0],dp[1],dp[2],dp[3]);
 }
 fence_proxy_async();allsync();
 if(threadIdx.x==0){dg_store(&p.dgmap,sm,0,row);dg_store(&p.dgmap,sm+8192,64,row);tma_store_commit();}
}
TMN_DEVI void weight_gemm(float (&acc)[64],uint8_t* sa,uint8_t* sb,bool add){
 fence_regs(acc);wgmma_fence();
 static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc,smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),add||k>0);});
 wgmma_commit();wgmma_wait<0>();fence_regs(acc);
}

TMN_DEVI void weight_pair(float (&a)[64],float (&b)[64],uint8_t* norm,bool add){
 int wi=threadIdx.x/128;uint8_t* sa=norm+32768;uint8_t* sb=norm+wi*16384;
 fence_regs(a);fence_regs(b);wgmma_fence();
 static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
  uint64_t bd=smem_desc(smem_u32(sb+k*2048),8192,1024,1);
  mma_ss128(a,smem_desc(smem_u32(sa+k*2048),16,1024,1),bd,add||k>0);
  mma_ss128(b,smem_desc(smem_u32(sa+8192+k*2048),16,1024,1),bd,add||k>0);
 });wgmma_commit();wgmma_wait<0>();fence_regs(a);fence_regs(b);
}
TMN_DEVI void store_weight(const Params& p,float (&acc)[64],int kind){
 int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 float* out=p.partw+blockIdx.x*49152+kind*16384;
 #pragma unroll
 for(int q=0;q<16;++q){int rr=w*16+lane/4+(kind==0?wi*64:0),c=q*8+2*(lane%4)+(kind==0?0:wi*128),stride=kind==0?128:256;
  stg64f(out+rr*stride+c,acc[4*q],acc[4*q+1]);stg64f(out+(rr+8)*stride+c,acc[4*q+2],acc[4*q+3]);}
}
TMN_DEVI void shared_role(const Params& p,uint8_t* sm,uint64_t* bars){
 int wi=threadIdx.x/128,round=0,mi=0;
 FragmentMask mask=fragment_mask<UCOUNT>(p,blockIdx.x);
 float wp0[64]={},wp1[64]={};
#if !GATE_PHASE
 float wg[64]={};
#endif
#if !PRODUCER
 if(blockIdx.x<p.tiles)load_raw(p,sm,bars,0,blockIdx.x*64);
#endif
 for(int tile=blockIdx.x;tile<p.tiles;tile+=UCOUNT,++round){int slot=round&1;
  mbar_wait(bars+slot,(round/2)&1);
#if !PRODUCER
  load_dy(p,sm,bars+2,tile*64);
#endif
  prepare_shared(p,sm,bars,slot,tile*64,round&1,mask,mi);
  uint8_t* x=sm+16384+slot*49152;
  uint8_t* norm=sm+16384+(1-slot)*49152;
#if !GATE_PHASE
  weight_gemm(wg,x+wi*8192,sm,round>0);
#endif
#if PAIR_WP
  weight_pair(wp0,wp1,norm,round>0);
#else
  weight_gemm(wp0,norm+32768,norm+wi*16384,round>0);
  weight_gemm(wp1,norm+40960,norm+wi*16384,round>0);
#endif
  if(threadIdx.x==0)tma_store_wait_all();allsync();
  for(int j=threadIdx.x;j<4096;j+=256)reinterpret_cast<uint32_t*>(sm)[j]=reinterpret_cast<uint32_t*>(norm+32768)[j];
  fence_proxy_async();allsync();
  // All consumers have released the opposite slot; overlap its next TMA load
  // with dTri/LN backward on the current raw tri tile.
#if LOWREG
  lowreg_dgrad(p,sm,tile*64,slot);allsync();
  if(tile+UCOUNT<p.tiles){
#if PRODUCER
   if(threadIdx.x==0)mbar_arrive(bars+4);
#else
   load_raw(p,sm,bars,1-slot,(tile+UCOUNT)*64);
#endif
  }
#else
  if(tile+UCOUNT<p.tiles){
#if PRODUCER
   if(threadIdx.x==0)mbar_arrive(bars+4);
#else
   load_raw(p,sm,bars,1-slot,(tile+UCOUNT)*64);
#endif
  }
  pipeline_dgrad(p,sm,tile*64,slot);allsync();
#endif
#if PRODUCER
  if(tile+UCOUNT<p.tiles && threadIdx.x==0)mbar_arrive(bars+5);
#endif
  if(++mi==mask.period)mi=0;
 }
#if !GATE_PHASE
 store_weight(p,wg,0);
#endif
 store_weight(p,wp0,1);store_weight(p,wp1,2);
 for(int j=threadIdx.x;j<512;j+=256)p.partln[blockIdx.x*512+j]=reinterpret_cast<float*>(sm+225792)[j];
}

#if GATE_PHASE
TMN_DEVI void load_gate_operands(const Params& p,uint8_t* sm,uint64_t* bars,int slot,int row){
 if(threadIdx.x)return;uint8_t* dst=sm+slot*32768;mbar_arrive_expect_tx(bars+slot,32768);
 tma_load_2d(dst,&p.dgmap,bars+slot,0,row);tma_load_2d(dst+8192,&p.dgmap,bars+slot,64,row);
 tma_load_2d(dst+16384,&p.x,bars+slot,0,row);tma_load_2d(dst+24576,&p.x,bars+slot,64,row);
}
TMN_DEVI void gate_phase(const Params& p,uint8_t* sm,uint64_t* bars){
 // All dGate tiles are already required outputs consumed later by B7.
 // Reuse that buffer rather than spill three live weight accumulators.
 int wi=threadIdx.x/128,round=0;float acc[64]={};
 if(blockIdx.x<p.tiles)load_gate_operands(p,sm,bars,0,blockIdx.x*64);
 for(int tile=blockIdx.x;tile<p.tiles;tile+=UCOUNT,++round){int slot=round&1;
  mbar_wait(bars+slot,(round/2)&1);
  if(tile+UCOUNT<p.tiles)load_gate_operands(p,sm,bars,1-slot,(tile+UCOUNT)*64);
  weight_gemm(acc,sm+slot*32768+16384+wi*8192,sm+slot*32768,round>0);allsync();
 }
 store_weight(p,acc,0);
}
#endif
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<49152){float v=0;for(int b=0;b<UCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partw)[b*49152+i];
  (i<16384?p.dwg:p.dwp-16384)[i]=__float2bfloat16_rn(v);
 }else if(i<49664){int j=i-49152;float v=0;for(int b=0;b<UCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
}

#if PRODUCER
TMN_DEVI void load_producer(const Params& p,uint8_t* sm,uint64_t* bars){
 if(threadIdx.x!=256)return;
 int round=0;
 for(int tile=blockIdx.x;tile<p.tiles;tile+=UCOUNT,++round){int slot=round&1;
  if(round)mbar_wait(bars+4,(round-1)&1);
  uint8_t* dst=sm+16384+slot*49152;mbar_arrive_expect_tx(bars+slot,49152);
  tma_load_2d(dst,&p.x,bars+slot,0,tile*64);tma_load_2d(dst+8192,&p.x,bars+slot,64,tile*64);tma_load_2d(dst+16384,&p.xhat,bars+slot,tile*64,0);
  if(round)mbar_wait(bars+5,(round-1)&1);
  mbar_arrive_expect_tx(bars+2,16384);tma_load_2d(sm,&p.dy,bars+2,0,tile*64);tma_load_2d(sm+8192,&p.dy,bars+2,64,tile*64);
 }
}
#endif

extern "C" __global__ __launch_bounds__(THREADS,1) void b1_fused(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bars[6];
 if(threadIdx.x==0){for(int b=0;b<6;++b)mbar_init(bars+b,1);fence_barrier_init();}
 named_bar_sync(15,THREADS);
#if PRODUCER
 if(threadIdx.x>=256){
  setmaxnreg_dec<B1_PROD_REGS>();load_producer(p,sm,bars);
 }else {
  setmaxnreg_inc<B1_CONS_REGS>();
#else
 {
#endif
 if(threadIdx.x==0){mbar_arrive_expect_tx(bars+3,98304);
  for(int c=0;c<2;++c)for(int k=0;k<2;++k)tma_load_2d(sm+114688+k*16384+c*8192,&p.wg,bars+3,c*64,k*64);
  for(int c=0;c<2;++c)for(int k=0;k<4;++k)tma_load_2d(sm+147456+k*16384+c*8192,&p.wp,bars+3,c*64,k*64);
 }
 float* ps=reinterpret_cast<float*>(sm+212992);for(int c=threadIdx.x;c<768;c+=256)ps[c]=c<128?p.gi[c]:c<256?p.bi[c-128]:c<512?p.gamma[c-256]:p.bo[c-512];
 for(int j=threadIdx.x;j<512;j+=256)reinterpret_cast<float*>(sm+225792)[j]=0.f;allsync();mbar_wait(bars+3,0);
 shared_role(p,sm,bars);
#if GATE_PHASE
 __threadfence();allsync();if(threadIdx.x==0){atomicAdd(p.counts+2,1u);while(atomicAdd(p.counts+2,0u)!=UCOUNT)__nanosleep(32);}allsync();
 // Every producer finished its final load before the corresponding consumers
 // finish; no producer can still touch bars0/1 after this grid barrier.
 if(threadIdx.x==0){for(int b=0;b<2;++b)mbar_init(bars+b,1);fence_barrier_init();}allsync();
 gate_phase(p,sm,bars);
#endif
#if PART_ONLY==2
 __threadfence();allsync();if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();
 for(int i=blockIdx.x*256+threadIdx.x;i<49664;i+=UCOUNT*256)reduce_at(p,i);
 __threadfence();allsync();if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+2,0u);atomicExch(p.counts+1,0u);}
#endif
 }
 named_bar_sync(15,THREADS);
}
extern "C" __global__ void b1_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
