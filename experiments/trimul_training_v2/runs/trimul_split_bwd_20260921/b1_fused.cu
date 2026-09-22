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

#ifndef TRAIN_L
#define TRAIN_L p.L
#endif
#ifndef GATE_SPLITS
#define GATE_SPLITS DW_SPLITS
#endif
constexpr int DWCOUNT=GATE_SPLITS+2*DW_SPLITS,DXCOUNT=UCOUNT-DWCOUNT;
static_assert(DXCOUNT>0,"need DX roles");
struct Params {
 CUtensorMap dy,x,tri,wp,wg,dtri,dgmap;
 const __nv_bfloat16* ds;const float *gi,*bi,*gamma,*bo;
 __nv_bfloat16 *dg,*dwg,*dwp;float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;int M,L,tiles;
};
#include "b1_pipeline_math.inc"
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
 tma_load_2d(dst,&p.x,bar+slot,0,row);tma_load_2d(dst+8192,&p.x,bar+slot,64,row);tma_load_2d(dst+16384,&p.tri,bar+slot,row,0);
}
TMN_DEVI void load_dy(const Params& p,uint8_t* sm,uint64_t* bar,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(bar,16384);tma_load_2d(sm,&p.dy,bar,0,row);tma_load_2d(sm+8192,&p.dy,bar,64,row);
}
template<bool DW> TMN_DEVI void normalize(const Params& p,uint8_t* sm,int slot){
 uint8_t* x=sm+16384+slot*49152;float* ps=reinterpret_cast<float*>(sm+212992);
 if(threadIdx.x<128){if(!USE_SAVED_XN)normalize_tile<8>(x,x,ps,ps+128);}
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
  for(int j=0;j<4;++j){int r=w*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j>>1),jr=(row+r)%TRAIN_L;uint32_t ds;
   if(mask.cached){int bit=q*8+j*2;uint32_t bits=mask_bits(mask,mi);ds=((bits>>bit)&1?mask.scale:0)|((bits>>(bit+1))&1?mask.scale<<16:0);}
   else ds=*reinterpret_cast<const uint32_t*>(p.ds+jr*128+wi*64+c);
   float a=bf16lo(dy[j])*bf16lo(ds),b=bf16hi(dy[j])*bf16hi(ds),ga=bf16lo(gate[q*4+j]),gb=bf16hi(gate[q*4+j]);
   if constexpr(DG){a=((a*math::round_bf16(acc[q*8+j*2]))*ga)*(1.f-ga);b=((b*math::round_bf16(acc[q*8+j*2+1]))*gb)*(1.f-gb);}else {a*=ga;b*=gb;}
   out[j]=pack_bf16(a,b);
  }
  stsm_x4(smem_u32(sm+wi*8192)+swz128(w*16+lane%8+8*(mat&1),(2*q+(mat>>1))*16),out[0],out[1],out[2],out[3]);
 }
 fence_proxy_async();allsync();
 if constexpr(DG){if(threadIdx.x==0){dg_store(&p.dgmap,sm,0,row);dg_store(&p.dgmap,sm+8192,64,row);tma_store_commit();}}
}
#if B1_DWG_PIPE
template<int K> TMN_DEVI void recompute128(float (&acc)[64],uint8_t* a,uint8_t* b){
 #pragma unroll
 for(int j=0;j<64;++j)acc[j]=0.f;
 fence_regs(acc);wgmma_fence();static_for<K/16>([&](auto ki){constexpr int k=decltype(ki)::value;
  mma_weight128(acc,smem_desc(smem_u32(a+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(b+(k/4)*16384+(k%4)*2048),8192,1024,1),k>0);
 });wgmma_commit();wgmma_wait<0>();fence_regs(acc);
}
TMN_DEVI void prepare_gate_weight(const Params& p,uint8_t* sm,uint64_t* dybar,int slot,int row,int phase,const FragmentMask& mask,int mi){
 int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;uint8_t* x=sm+16384+slot*49152;float* ps=reinterpret_cast<float*>(sm+212992);float acc[64];uint32_t own[16];
 if(wi==0){if(!USE_SAVED_XN)normalize_tile<8>(x,x,ps,ps+128);fence_proxy_async();sync_group();recompute128<128>(acc,x,sm+114688);}
 else {normalize_tile<16,true,B1_LN_SERIAL>(x+16384,x+16384,ps+256,ps+512);fence_proxy_async();sync_group();recompute128<256>(acc,x+16384,sm+147456);}
 allsync(); // Both recomputation GEMMs finished; output-LN tile is now dead.
 static_for<8>([&](auto qi){constexpr int q=decltype(qi)::value;uint32_t f[4];
  static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;float a=math::round_bf16(acc[q*8+j*2]),b=math::round_bf16(acc[q*8+j*2+1]);if(wi==0){a=math::sigmoid(a);b=math::sigmoid(b);}f[j]=pack_bf16(a,b);});
  if(q/4==wi){static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;own[(q%4)*4+j]=f[j];});}
  else stsm_x4(smem_u32(x+16384+wi*8192)+swz128(w*16+lane%8+8*(mat&1),(2*(q%4)+(mat>>1))*16),f[0],f[1],f[2],f[3]);
 });
 fence_proxy_async();allsync();mbar_wait(dybar,phase);
 static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;uint32_t dy[4],other[4],out[4];uint32_t off=swz128(w*16+lane%8+8*(mat&1),(2*q+(mat>>1))*16);
  ldsm_x4(dy,smem_u32(sm+wi*8192)+off);ldsm_x4(other,smem_u32(x+16384+(1-wi)*8192)+off);
  static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;int rr=w*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j>>1),jr=(row+rr)%TRAIN_L;uint32_t ds;
   if(mask.cached){constexpr int bit=q*8+j*2;uint32_t bits=mask_bits(mask,mi);ds=((bits>>bit)&1?mask.scale:0)|((bits>>(bit+1))&1?mask.scale<<16:0);}else ds=*reinterpret_cast<const uint32_t*>(p.ds+jr*128+wi*64+c);
   uint32_t g=wi==0?own[q*4+j]:other[j],pr=wi==0?other[j]:own[q*4+j];float ga=bf16lo(g),gb=bf16hi(g);
   out[j]=pack_bf16((((bf16lo(dy[j])*bf16lo(ds))*bf16lo(pr))*ga)*(1.f-ga),(((bf16hi(dy[j])*bf16hi(ds))*bf16hi(pr))*gb)*(1.f-gb));
  });stsm_x4(smem_u32(sm+wi*8192)+off,out[0],out[1],out[2],out[3]);
 });fence_proxy_async();allsync();if(threadIdx.x==0){dg_store(&p.dgmap,sm,0,row);dg_store(&p.dgmap,sm+8192,64,row);tma_store_commit();}
}
#endif
#if B1_DWPROJ_PIPE
TMN_DEVI void prepare_projection(const Params& p,uint8_t* sm,uint64_t* dybar,int slot,int row,int phase,const FragmentMask& mask,int mi,int half){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;uint8_t* x=sm+16384+slot*49152;float* ps=reinterpret_cast<float*>(sm+212992);
 if(threadIdx.x<128){
  if(!USE_SAVED_XN)normalize_tile<8>(x,x,ps,ps+128);fence_proxy_async();sync_group();
  float acc[32];recompute_gemm<128,16384>(acc,x,sm+114688+half*8192);
  mbar_wait(dybar,phase);
  #pragma unroll
  for(int q=0;q<4;++q){uint32_t dy[4],out[4];ldsm_x4(dy,smem_u32(sm+half*8192)+swz128(w*16+lane%8+8*(mat&1),(2*q+(mat>>1))*16));
   #pragma unroll
   for(int j=0;j<4;++j){int rr=w*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j>>1),jr=(row+rr)%TRAIN_L;uint32_t ds;
    if(mask.cached){int bit=q*8+j*2;uint32_t bits=mask_bits(mask,mi);ds=((bits>>bit)&1?mask.scale:0)|((bits>>(bit+1))&1?mask.scale<<16:0);}else ds=*reinterpret_cast<const uint32_t*>(p.ds+jr*128+half*64+c);
    float ga=math::round_bf16(math::sigmoid(math::round_bf16(acc[q*8+j*2]))),gb=math::round_bf16(math::sigmoid(math::round_bf16(acc[q*8+j*2+1])));
    out[j]=pack_bf16((bf16lo(dy[j])*bf16lo(ds))*ga,(bf16hi(dy[j])*bf16hi(ds))*gb);
   }
   stsm_x4(smem_u32(sm+half*8192)+swz128(w*16+lane%8+8*(mat&1),(2*q+(mat>>1))*16),out[0],out[1],out[2],out[3]);
  }
 }else normalize_tile<16,true,B1_LN_SERIAL>(x+16384,x+16384,ps+256,ps+512);
 fence_proxy_async();allsync();
}
#endif
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* bars){
 int GROUP=blockIdx.x<GATE_SPLITS?0:blockIdx.x<GATE_SPLITS+DW_SPLITS?1:2,split=blockIdx.x-(GROUP==0?0:GATE_SPLITS+(GROUP-1)*DW_SPLITS),stride=GROUP==0?GATE_SPLITS:DW_SPLITS,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int t=GROUP==0?wi:2+(GROUP-1)*2+wi,round=0,mi=0;float acc[64]={};FragmentMask mask=GROUP==0?fragment_mask<GATE_SPLITS>(p,split):fragment_mask<DW_SPLITS>(p,split,
#if B1_DWPROJ_PIPE
 GROUP-1
#else
 -1
#endif
 );
 if(split<p.tiles)load_raw(p,sm,bars,0,split*64);
 for(int tile=split;tile<p.tiles;tile+=stride,++round){int slot=round&1;
  mbar_wait(bars+slot,(round/2)&1);load_dy(p,sm,bars+2,tile*64);
  if(tile+stride<p.tiles)load_raw(p,sm,bars,1-slot,(tile+stride)*64);
#if B1_DWPROJ_PIPE
  if(GROUP!=0)prepare_projection(p,sm,bars+2,slot,tile*64,round&1,mask,mi,GROUP-1);else
#endif
  {
#if B1_DWG_PIPE
   if(GROUP==0)prepare_gate_weight(p,sm,bars+2,slot,tile*64,round&1,mask,mi);else
#endif
   {normalize<true>(p,sm,slot);if(GROUP==0)derivative<true>(p,sm,bars+2,slot,tile*64,round&1,mask,mi);else derivative<false>(p,sm,bars+2,slot,tile*64,round&1,mask,mi);}
  }
  uint8_t* x=sm+16384+slot*49152;uint8_t* sa=GROUP==0?x+wi*8192:sm+(GROUP-1)*8192;uint8_t* sb=GROUP==0?sm:x+16384+wi*16384;
  fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc,smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),round>0||k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(GROUP==0&&threadIdx.x==0)tma_store_wait_all();allsync();if(++mi==mask.period)mi=0;
 }
 float* out=p.partw+((GROUP==0?0:GATE_SPLITS+(GROUP-1)*DW_SPLITS)+split)*16384;
 #pragma unroll
 for(int q=0;q<16;++q){int rr=w*16+lane/4+(t<2?t*64:0),c=q*8+2*(lane%4)+(t<2?0:((t-2)%2)*128),stride=t<2?128:256;stg64f(out+rr*stride+c,acc[4*q],acc[4*q+1]);stg64f(out+(rr+8)*stride+c,acc[4*q+2],acc[4*q+3]);}
}
#if B1_DX_PIPE
TMN_DEVI void prepare_input(const Params& p,uint8_t* sm,uint64_t* dybar,int slot,int row,int phase,const FragmentMask& mask0,const FragmentMask& mask1,int mi){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;uint8_t* x=sm+16384+slot*49152;float* ps=reinterpret_cast<float*>(sm+212992);
 if(threadIdx.x<128){
  if(!USE_SAVED_XN)normalize_tile<8>(x,x,ps,ps+128);fence_proxy_async();sync_group();
  static_for<2>([&](auto hi){constexpr int half=decltype(hi)::value;const FragmentMask& mask=half==0?mask0:mask1;
  float acc[32];recompute_gemm<128,16384>(acc,x,sm+114688+half*8192);
  mbar_wait(dybar,phase);
  #pragma unroll
  for(int q=0;q<4;++q){uint32_t dy[4],out[4];ldsm_x4(dy,smem_u32(sm+half*8192)+swz128(w*16+lane%8+8*(mat&1),(2*q+(mat>>1))*16));
   #pragma unroll
   for(int j=0;j<4;++j){int rr=w*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j>>1),jr=(row+rr)%TRAIN_L;uint32_t ds;
    if(mask.cached){int bit=q*8+j*2;uint32_t bits=mask_bits(mask,mi);ds=((bits>>bit)&1?mask.scale:0)|((bits>>(bit+1))&1?mask.scale<<16:0);}else ds=*reinterpret_cast<const uint32_t*>(p.ds+jr*128+half*64+c);
    float ga=math::round_bf16(math::sigmoid(math::round_bf16(acc[q*8+j*2]))),gb=math::round_bf16(math::sigmoid(math::round_bf16(acc[q*8+j*2+1])));
    out[j]=pack_bf16((bf16lo(dy[j])*bf16lo(ds))*ga,(bf16hi(dy[j])*bf16hi(ds))*gb);
   }
   stsm_x4(smem_u32(sm+half*8192)+swz128(w*16+lane%8+8*(mat&1),(2*q+(mat>>1))*16),out[0],out[1],out[2],out[3]);
  }
  });
 }else {
  LnStats st=normalize_tile<16,true,false,false>(x+16384,nullptr,ps+256,ps+512);
  if(lane%4==0){int ra=w*16+lane/4,rb=ra+8;float* mu=reinterpret_cast<float*>(sm+225280);mu[ra]=st.mA;mu[rb]=st.mB;mu[64+ra]=st.rA;mu[64+rb]=st.rB;}
 }
 fence_proxy_async();allsync();
}
#endif
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* bars){
 int split=blockIdx.x-DWCOUNT,round=0,mi=0;FragmentMask mask=fragment_mask<DXCOUNT>(p,split,
#if B1_DX_PIPE
 0
#else
 -1
#endif
 );
#if B1_DX_PIPE
 FragmentMask mask1=fragment_mask<DXCOUNT>(p,split,1);
#endif
 if(split<p.tiles)load_raw(p,sm,bars,0,split*64);
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int slot=round&1;
  mbar_wait(bars+slot,(round/2)&1);load_dy(p,sm,bars+2,tile*64);if(tile+DXCOUNT<p.tiles)load_raw(p,sm,bars,1-slot,(tile+DXCOUNT)*64);
#if B1_DX_PIPE
  prepare_input(p,sm,bars+2,slot,tile*64,round&1,mask,mask1,mi);
#else
  normalize<false>(p,sm,slot);derivative<false>(p,sm,bars+2,slot,tile*64,round&1,mask,mi);
#endif
  pipeline_dgrad(p,sm,tile*64,slot);allsync();if(++mi==mask.period)mi=0;
 }
 for(int j=threadIdx.x;j<512;j+=256)p.partln[split*512+j]=reinterpret_cast<float*>(sm+225792)[j];
}
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<49152){int tile=i/16384,j=i%16384;float v=0;for(int b=0;b<(tile==0?GATE_SPLITS:DW_SPLITS);++b)v+=reinterpret_cast<volatile float*>(p.partw)[((tile==0?0:GATE_SPLITS+(tile-1)*DW_SPLITS)+b)*16384+j];(tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);}
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
#if PROFILE_ROLES
 unsigned long long role_begin=clock64();
#endif
 if(blockIdx.x<DWCOUNT)weight_role(p,sm,bars);else input_role(p,sm,bars);
#if PROFILE_ROLES
 if(threadIdx.x==0)p.counts[2+blockIdx.x]=unsigned(clock64()-role_begin);
#endif
#if PART_ONLY==2
 __threadfence();allsync();if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();
 for(int i=blockIdx.x*256+threadIdx.x;i<49664;i+=UCOUNT*256)reduce_at(p,i);
 __threadfence();allsync();if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void b1_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
