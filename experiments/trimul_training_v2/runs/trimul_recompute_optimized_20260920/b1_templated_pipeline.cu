// SPDX-License-Identifier: Apache-2.0
// Anthropic TMA/WGMMA/LN primitives; streamed on-chip training derivatives.
#include "common_recompute.cuh"
#ifndef UCOUNT
#define UCOUNT 132
#endif
#ifndef DW_SPLITS
#define DW_SPLITS 28
#endif
#ifndef PART_ONLY
#define PART_ONLY 2
#endif
#ifndef B1_LN_SERIAL
#define B1_LN_SERIAL 1
#endif
constexpr int DWCOUNT=3*DW_SPLITS,DXCOUNT=UCOUNT-DWCOUNT;
static_assert(DXCOUNT>0,"need DX roles");
struct Params {
 CUtensorMap dy,x,tri,wp,wg,dtri;
 const __nv_bfloat16* ds;const float *gi,*bi,*gamma,*bo;
 __nv_bfloat16 *dg,*dwg,*dwp;float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;int M,L,tiles;
};
#include "b1_pipeline_math.inc"
struct FragmentMask{uint32_t b0,b1,b2,scale;int period;bool cached;};
template<int STRIDE> TMN_DEVI FragmentMask fragment_mask(const Params& p,int first){
 FragmentMask m={0,0,0,0,0,false};int a=64*STRIDE,b=p.L;while(b){int r=a%b;a=b;b=r;}m.period=p.L/a;m.cached=m.period<=3;
 int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 if(m.cached && first<p.tiles)for(int z=0;z<m.period;++z){int row=((first+z*STRIDE)*64)%p.L;uint32_t bits=0;
  #pragma unroll
  for(int q=0;q<4;++q){
   #pragma unroll
   for(int j=0;j<4;++j){int r=row+w*16+lane/4+8*(j&1);if(r>=p.L)r-=p.L;int c=wi*64+q*16+2*(lane%4)+8*(j>>1);
    uint32_t ds=*reinterpret_cast<const uint32_t*>(p.ds+r*128+c);bits|=(uint32_t((ds&65535)!=0)<<(q*8+j*2))|(uint32_t((ds>>16)!=0)<<(q*8+j*2+1));m.scale|=(ds&65535)|(ds>>16);
   }
  }
  if(z==0)m.b0=bits;else if(z==1)m.b1=bits;else m.b2=bits;
 }
 return m;
}
TMN_DEVI void load_raw(const Params& p,uint8_t* sm,uint64_t* bar,int slot,int row){
 if(threadIdx.x)return;uint8_t* dst=sm+16384+slot*49152;mbar_arrive_expect_tx(bar+slot,49152);
 tma_load_2d(dst,&p.x,bar+slot,0,row);tma_load_2d(dst+8192,&p.x,bar+slot,64,row);tma_load_2d(dst+16384,&p.tri,bar+slot,row,0);
}
TMN_DEVI void load_dy(const Params& p,uint8_t* sm,uint64_t* bar,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(bar,16384);tma_load_2d(sm,&p.dy,bar,0,row);tma_load_2d(sm+8192,&p.dy,bar,64,row);
}
template<bool DW> TMN_DEVI void normalize(const Params& p,uint8_t* sm,int slot){
 uint8_t* x=sm+16384+slot*49152;float* ps=reinterpret_cast<float*>(sm+212992);
 if(threadIdx.x<128)normalize_tile<8>(x,x,ps,ps+128);
 else {
  LnStats st=normalize_tile<16,true,B1_LN_SERIAL,DW>(x+16384,x+16384,ps+256,ps+512);
  if constexpr(!DW){int lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
   if(lane%4==0){int ra=w*16+lane/4,rb=ra+8;float* mu=reinterpret_cast<float*>(sm+225280);mu[ra]=st.mA;mu[rb]=st.mB;mu[64+ra]=st.rA;mu[64+rb]=st.rB;}
  }
 }
 fence_proxy_async();allsync();
}
template<bool DG> TMN_DEVI void derivative(const Params& p,uint8_t* sm,uint64_t* dybar,int slot,int row,int phase,const FragmentMask& mask,int mi){
 int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
 uint8_t* x=sm+16384+slot*49152;float acc[32];uint32_t gate[16];
 recompute_gemm<128,16384>(acc,x,sm+114688+wi*8192);
 #pragma unroll
 for(int j=0;j<16;++j)gate[j]=pack_bf16(math::sigmoid(math::round_bf16(acc[j*2])),math::sigmoid(math::round_bf16(acc[j*2+1])));
 if constexpr(DG)recompute_gemm<256,16384>(acc,x+16384,sm+147456+wi*8192);
 mbar_wait(dybar,phase);
 #pragma unroll
 for(int q=0;q<4;++q){uint32_t dy[4],out[4];
  ldsm_x4(dy,smem_u32(sm+wi*8192)+swz128(w*16+lane%8+8*(mat&1),(2*q+(mat>>1))*16));
  #pragma unroll
  for(int j=0;j<4;++j){int r=w*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j>>1),jr=(row+r)%p.L;uint32_t ds;
   if(mask.cached){int bit=q*8+j*2;uint32_t bits=mi==0?mask.b0:mi==1?mask.b1:mask.b2;ds=((bits>>bit)&1?mask.scale:0)|((bits>>(bit+1))&1?mask.scale<<16:0);}
   else ds=*reinterpret_cast<const uint32_t*>(p.ds+jr*128+wi*64+c);
   float a=bf16lo(dy[j])*bf16lo(ds),b=bf16hi(dy[j])*bf16hi(ds),ga=bf16lo(gate[q*4+j]),gb=bf16hi(gate[q*4+j]);
   if constexpr(DG){a=((a*math::round_bf16(acc[q*8+j*2]))*ga)*(1.f-ga);b=((b*math::round_bf16(acc[q*8+j*2+1]))*gb)*(1.f-gb);}else {a*=ga;b*=gb;}
   out[j]=pack_bf16(a,b);if constexpr(DG)*reinterpret_cast<uint32_t*>(p.dg+(size_t)(row+r)*128+wi*64+c)=out[j];
  }
  stsm_x4(smem_u32(sm+wi*8192)+swz128(w*16+lane%8+8*(mat&1),(2*q+(mat>>1))*16),out[0],out[1],out[2],out[3]);
 }
 fence_proxy_async();allsync();
}
template<int GROUP> TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* bars){
 int split=blockIdx.x/3,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int t=GROUP==0?wi:2+(GROUP-1)*2+wi,round=0,mi=0;float acc[64]={};FragmentMask mask=fragment_mask<DW_SPLITS>(p,split);
 if(split<p.tiles)load_raw(p,sm,bars,0,split*64);
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++round){int slot=round&1;
  mbar_wait(bars+slot,(round/2)&1);load_dy(p,sm,bars+2,tile*64);
  if(tile+DW_SPLITS<p.tiles)load_raw(p,sm,bars,1-slot,(tile+DW_SPLITS)*64);
  normalize<true>(p,sm,slot);derivative<GROUP==0>(p,sm,bars+2,slot,tile*64,round&1,mask,mi);
  uint8_t* x=sm+16384+slot*49152;uint8_t* sa=GROUP==0?x+wi*8192:sm+(GROUP-1)*8192;uint8_t* sb=GROUP==0?sm:x+16384+wi*16384;
  fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc,smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),round>0||k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(++mi==mask.period)mi=0;
 }
 float* out=p.partw+(GROUP*DW_SPLITS+split)*16384;
 #pragma unroll
 for(int q=0;q<16;++q){int rr=w*16+lane/4+(t<2?t*64:0),c=q*8+2*(lane%4)+(t<2?0:((t-2)%2)*128),stride=t<2?128:256;stg64f(out+rr*stride+c,acc[4*q],acc[4*q+1]);stg64f(out+(rr+8)*stride+c,acc[4*q+2],acc[4*q+3]);}
}
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* bars){
 int split=blockIdx.x-DWCOUNT,round=0,mi=0;FragmentMask mask=fragment_mask<DXCOUNT>(p,split);
 if(split<p.tiles)load_raw(p,sm,bars,0,split*64);
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int slot=round&1;
  mbar_wait(bars+slot,(round/2)&1);load_dy(p,sm,bars+2,tile*64);if(tile+DXCOUNT<p.tiles)load_raw(p,sm,bars,1-slot,(tile+DXCOUNT)*64);
  normalize<false>(p,sm,slot);derivative<false>(p,sm,bars+2,slot,tile*64,round&1,mask,mi);pipeline_dgrad(p,sm,tile*64,slot);allsync();if(++mi==mask.period)mi=0;
 }
 for(int j=threadIdx.x;j<512;j+=256)p.partln[split*512+j]=reinterpret_cast<float*>(sm+225792)[j];
}
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<49152){int tile=i/16384,j=i%16384;float v=0;for(int b=0;b<DW_SPLITS;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*DW_SPLITS+b)*16384+j];(tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);}
 else if(i<49664){int j=i-49152;float v=0;for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
}
extern "C" __global__ __launch_bounds__(256,1) void b1_fused(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bars[4];
 if(threadIdx.x==0){for(int b=0;b<4;++b)mbar_init(bars+b,1);fence_barrier_init();mbar_arrive_expect_tx(bars+3,98304);
  for(int c=0;c<2;++c)for(int k=0;k<2;++k)tma_load_2d(sm+114688+k*16384+c*8192,&p.wg,bars+3,c*64,k*64);
  for(int c=0;c<2;++c)for(int k=0;k<4;++k)tma_load_2d(sm+147456+k*16384+c*8192,&p.wp,bars+3,c*64,k*64);
 }
 float* ps=reinterpret_cast<float*>(sm+212992);for(int c=threadIdx.x;c<768;c+=256)ps[c]=c<128?p.gi[c]:c<256?p.bi[c-128]:c<512?p.gamma[c-256]:p.bo[c-512];
 for(int j=threadIdx.x;j<512;j+=256)reinterpret_cast<float*>(sm+225792)[j]=0.f;allsync();mbar_wait(bars+3,0);
 if(blockIdx.x<DWCOUNT){if(blockIdx.x%3==0)weight_role<0>(p,sm,bars);else if(blockIdx.x%3==1)weight_role<1>(p,sm,bars);else weight_role<2>(p,sm,bars);}else input_role(p,sm,bars);
#if PART_ONLY==2
 __threadfence();allsync();if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();
 for(int i=blockIdx.x*256+threadIdx.x;i<49664;i+=UCOUNT*256)reduce_at(p,i);
 __threadfence();allsync();if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void b1_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
