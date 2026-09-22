// SPDX-License-Identifier: Apache-2.0
// Anthropic TMA/WGMMA/LN fragments + Miniworld B1-B4 math, recomputed on chip.
#include "common_recompute.cuh"
#ifndef UCOUNT
#define UCOUNT 132
#endif
#ifndef DW_SPLITS
#define DW_SPLITS 16
#endif
#ifndef PART_ONLY
#define PART_ONLY 2
#endif
constexpr int DWCOUNT=3*DW_SPLITS,DXCOUNT=UCOUNT-DWCOUNT;
static_assert(DXCOUNT>0,"need DX roles");
struct Params {
 CUtensorMap dy,x,tri,wp,wg,dtri;
 const __nv_bfloat16* ds;
 const float *gi,*bi,*gamma,*bo;
 __nv_bfloat16 *dg,*dwg,*dwp;
 float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;
 int M,L,tiles;
};
#include "b1_saved_math.inc"

template<bool DW> TMN_DEVI void load_rows(const Params& p,uint8_t* sm,uint64_t* full,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(full,65536);
 for(int c=0;c<2;++c){
  tma_load_2d(sm+c*8192,&p.dy,full,c*64,row);
  tma_load_2d(sm+(DW?49152:98304)+c*8192,&p.x,full,c*64,row);
 }
 tma_load_2d(sm+(DW?98304:32768),&p.tri,full,row,0);
}
template<bool DW> TMN_DEVI void recreate(const Params& p,uint8_t* sm,int group){
 const int tid=threadIdx.x,wi=tid/128,lane=tid%32,w=(tid/32)%4;
 float* params=reinterpret_cast<float*>(sm+(DW?32768:65536));
 for(int c=tid;c<768;c+=256)params[c]=c<128?p.gi[c]:c<256?p.bi[c-128]:c<512?p.gamma[c-256]:p.bo[c-512];
 allsync();
 uint8_t* xn=sm+(DW?49152:98304);uint8_t* tri=sm+(DW?98304:32768);
 if(wi==0)normalize_tile<8>(xn,xn,params,params+128);
 else {
  LnStats st=normalize_tile<16,true,true,DW>(tri,DW?sm+65536:nullptr,params+256,params+512);
  if constexpr(!DW){if(lane%4==0){int ra=w*16+lane/4,rb=ra+8;
   float* mu=reinterpret_cast<float*>(sm+73728);float* rs=mu+64;
   mu[ra]=st.mA;mu[rb]=st.mB;rs[ra]=st.rA;rs[rb]=st.rB;}}
 }
 fence_proxy_async();allsync();
 // Wg is K-major transposed input weight; each WG owns64 output channels.
 float acc[32];recompute_gemm<128,16384>(acc,xn,sm+131072+wi*8192);
 store_recomputed<true>(acc,sm+16384+wi*8192);
 if constexpr(DW){if(group==0){
  // Only the output-gate weight derivative needs output projection values.
  // Other DW groups and DX skip this forward GEMM entirely.
  recompute_gemm<256,16384>(acc,sm+65536,sm+163840+wi*8192);
  store_recomputed<false>(acc,sm+32768+wi*8192);
 }}
 fence_proxy_async();allsync();
}

TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* full){
 int group=blockIdx.x%3,split=blockIdx.x/3,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int t=group==0?wi:2+(group-1)*2+wi;
 float acc[64]={};MaskCycle mask=mask_cycle<DW_SPLITS>(p,split);int mi=0,round=0;
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++round){
  load_rows<true>(p,sm,full,tile*64);mbar_wait(full,round&1);
  recreate<true>(p,sm,group);
  if(group==0)gate_backward<true>(p,sm,tile*64,mask,mi);else gate_backward<false>(p,sm,tile*64,mask,mi);
  uint8_t* sa=t<2?sm+49152+t*8192:sm+((t-2)/2)*8192;
  uint8_t* sb=t<2?sm+16384:sm+65536+((t-2)%2)*16384;
  fence_regs(acc);wgmma_fence();
  static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_ss128(acc,smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),round>0||k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
  if(++mi==mask.period)mi=0;
 }
 float* out=p.partw+(group*DW_SPLITS+split)*16384;
#pragma unroll
 for(int q=0;q<16;++q){int rr=w*16+lane/4+(t<2?t*64:0),c=q*8+2*(lane%4)+(t<2?0:((t-2)%2)*128),stride=t<2?128:256;
  stg64f(out+rr*stride+c,acc[4*q],acc[4*q+1]);stg64f(out+(rr+8)*stride+c,acc[4*q+2],acc[4*q+3]);}
}
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* full){
 int split=blockIdx.x-DWCOUNT,round=0,mi=0;MaskCycle mask=mask_cycle<DXCOUNT>(p,split);
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){
  load_rows<false>(p,sm,full,tile*64);mbar_wait(full,round&1);
  recreate<false>(p,sm,0);gate_backward<false>(p,sm,tile*64,mask,mi);
  dual_dgrad(p,sm,tile*64,0);
  // The original math used alternating slots. This reconstruction reuses one
  // slot: BOTH WGs must finish their TMA dtri stores before thread 0 reloads it.
  allsync();
  if(++mi==mask.period)mi=0;
 }
 for(int j=threadIdx.x;j<512;j+=256)p.partln[split*512+j]=reinterpret_cast<float*>(sm+229376)[j];
}
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<49152){int tile=i/16384,j=i%16384;float v=0;
  for(int b=0;b<DW_SPLITS;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*DW_SPLITS+b)*16384+j];
  (tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);
 }else if(i<49664){int j=i-49152;float v=0;for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
}
extern "C" __global__ __launch_bounds__(256,1)
void b1_fused(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t full[2];
 if(threadIdx.x==0){mbar_init(full,1);mbar_init(full+1,1);fence_barrier_init();mbar_arrive_expect_tx(full,98304);
  for(int c=0;c<2;++c)for(int k=0;k<2;++k)tma_load_2d(sm+131072+k*16384+c*8192,&p.wg,full,c*64,k*64);
  for(int c=0;c<2;++c)for(int k=0;k<4;++k)tma_load_2d(sm+163840+k*16384+c*8192,&p.wp,full,c*64,k*64);
 }
 for(int i=threadIdx.x;i<512;i+=256)reinterpret_cast<float*>(sm+229376)[i]=0.f;
 if(blockIdx.x>=DWCOUNT)reinterpret_cast<float*>(sm+74752)[threadIdx.x]=p.gamma[threadIdx.x];
 allsync();mbar_wait(full,0);
 if(blockIdx.x<DWCOUNT)weight_role(p,sm,full+1);else input_role(p,sm,full+1);
#if PART_ONLY==2
 __threadfence();allsync();
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();
 for(int i=blockIdx.x*256+threadIdx.x;i<49664;i+=UCOUNT*256)reduce_at(p,i);
 __threadfence();allsync();
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void b1_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
