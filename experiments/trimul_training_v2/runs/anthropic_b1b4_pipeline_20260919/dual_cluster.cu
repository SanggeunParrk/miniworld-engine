// Experimental two-CTA TMA multicast + double-buffer pipeline.
// Experimental single-CTA ownership of B1, dX/LN and ALL dW tiles.
// Preserves FP32 dW accumulation and BF16 intermediate rounding.
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
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
static_assert(UCOUNT%2==0,"two CTAs per cluster");
constexpr int PAIRS=UCOUNT/2;
// Upstream has no multicast wrapper. Identical shared/mbarrier offsets in
// both CTAs are targeted by this mask. PTX 8.8, CUDA 12.9:
// https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html
TMN_DEVI void multicast_load(void* dst,const CUtensorMap* map,uint64_t* bar,int c,int r){
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3,%4}], [%2], %5;"
 ::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c),"r"(r),"h"(uint16_t(3)):"memory");
}
// Barriers in both CTAs are armed before the issuing cluster barrier.
template<bool DW> TMN_DEVI void cluster_issue(const Params& p,uint8_t* sm,uint64_t* full,int slot,int tile,bool initial){
 if(threadIdx.x!=0)return;
 uint8_t* s=sm+slot*98304;int row=tile*64;
 if constexpr(DW){
#pragma unroll
  for(int c=0;c<2;++c){multicast_load(s+c*8192,&p.dy,full+slot,c*64,row);multicast_load(s+16384+c*8192,&p.gate,full+slot,c*64,row);tma_load_2d(s+32768+c*8192,&p.proj,full+slot,c*64,row);tma_load_2d(s+49152+c*8192,&p.xn,full+slot,c*64,row);}
#pragma unroll
  for(int c=0;c<4;++c)tma_load_2d(s+65536+c*8192,&p.norm,full+slot,c*64,row);
 }else{
  tma_load_2d(s+32768,&p.tri,full+slot,row,0);
  if(initial)for(int n=0;n<4;++n)for(int k=0;k<2;++k)tma_load_2d(sm+163840+n*16384+k*8192,&p.wp,full+slot,k*64,n*64);
 }
}
struct MaskCycle{uint32_t b0,b1,b2,scale;int period;bool cached;};
TMN_DEVI MaskCycle mask_cycle(const Params& p,int first){
 MaskCycle v={0,0,0,0,0,false};int a=64*PAIRS,b=p.L;while(b){int r=a%b;a=b;b=r;}
 v.period=p.L/a;v.cached=v.period<=3;
 if(v.cached&&first<p.tiles)for(int z=0;z<v.period;++z){uint32_t bits=0;int j0=((first+z*PAIRS)*64)%p.L;
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
template<bool DW> TMN_DEVI void cluster_b1(const Params& p,uint8_t* s,int row,const MaskCycle& mask,int mi){
 int j0=row%p.L;uint32_t bits=mi==0?mask.b0:mi==1?mask.b1:mask.b2;
 for(int i=threadIdx.x;i<1024;i+=256){int cb=i/512,r=(i%512)/8,c=(i%8)*8,jr=j0+r;if(jr>=p.L)jr-=p.L;
  uint32_t sy=smem_u32(s+cb*8192)+swz128(r,c*2),sg=smem_u32(s+16384+cb*8192)+swz128(r,c*2);
  uint4 y=lds128(sy),g=lds128(sg),v,ds,dp,dg;
  if constexpr(DW)v=lds128(smem_u32(s+32768+cb*8192)+swz128(r,c*2));
  if(mask.cached){uint32_t* dd=reinterpret_cast<uint32_t*>(&ds);
#pragma unroll
   for(int q=0;q<4;++q){int bit=8*(i/256)+2*q;dd[q]=((bits>>bit)&1?mask.scale:0)|((bits>>(bit+1))&1?mask.scale<<16:0);}
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
TMN_DEVI void dual_dgrad(const Params& p,uint8_t* sm,int m0,int slot){
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32,mat=lane/8,r8=lane%8;
 uint8_t *sn=sm+65536,*sx=sm+slot*98304+32768;float* stats=reinterpret_cast<float*>(sm+slot*98304+16384);
 float* mus=stats+256,*rss=mus+64,*gam=rss+64;
 if(threadIdx.x<64){mus[tid]=p.mean[m0+tid];rss[tid]=p.rs[m0+tid];}gam[threadIdx.x]=p.gamma[threadIdx.x];allsync();
 int ra=w*16+lane/4,rb=ra+8;float mu[2]={mus[ra],mus[rb]},rs[2]={rss[ra],rss[rb]},s1[2]={},s2[2]={};
#pragma unroll
 for(int qn=0;qn<2;++qn){int n=wi*2+qn;uint8_t* sw=sm+163840+n*16384;
  float acc[32]={};fence_regs(acc);wgmma_fence();
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_dgrad(acc,smem_desc(smem_u32(sm+slot*98304+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);
#pragma unroll
  for(int q=0;q<4;++q){uint32_t fx[4],dn[4];ldsm_x4_t(fx,smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
#pragma unroll
   for(int j=0;j<4;++j){dn[j]=pack_bf16(acc[q*8+j*2],acc[q*8+j*2+1]);int rr=j&1,c=n*64+q*16+2*(lane%4)+8*(j>>1);float xa=__fmul_rn(__fsub_rn(bf16lo(fx[j]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(fx[j]),mu[rr]),rs[rr]);float ha=bf16lo(dn[j])*gam[c],hb=bf16hi(dn[j])*gam[c+1];s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;}
   int chq=8*(mat>>1)+r8;stsm_x4_t(smem_u32(sn)+swz128(n*64+q*16+chq,(w*16+8*(mat&1))*2),dn[0],dn[1],dn[2],dn[3]);
  }
  sync_group();
 }
 s1[0]=quad_sum(s1[0])/256.f;s1[1]=quad_sum(s1[1])/256.f;s2[0]=quad_sum(s2[0])/256.f;s2[1]=quad_sum(s2[1])/256.f;
 if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}allsync();
 float* red=reinterpret_cast<float*>(sm+229376);
#pragma unroll 1
 for(int b=0;b<8;++b){int c=wi*128+b*16+w*4+lane/8;float dg=0,db=0,gamma=gam[c];
#pragma unroll
  for(int k=0;k<4;++k){int r=2*(lane%8)+16*k;uint32_t x=pair_get(sx,c,r),dn=pair_get(sn,c,r);float mua=mus[r],mub=mus[r+1],rsa=rss[r],rsb=rss[r+1];float xa=__fmul_rn(__fsub_rn(bf16lo(x),mua),rsa),xb=__fmul_rn(__fsub_rn(bf16hi(x),mub),rsb),da=bf16lo(dn),dd=bf16hi(dn);dg+=da*xa+dd*xb;db+=da+dd;pair_put(sx,c,r,pack_bf16(rsa*((da*gamma-(stats[r*2+1]+stats[128+r*2+1]))-xa*(stats[r*2]+stats[128+r*2])),rsb*((dd*gamma-(stats[(r+1)*2+1]+stats[128+(r+1)*2+1]))-xb*(stats[(r+1)*2]+stats[128+(r+1)*2]))));}
#pragma unroll
  for(int sh=1;sh<8;sh*=2){dg+=__shfl_xor_sync(0xffffffff,dg,sh);db+=__shfl_xor_sync(0xffffffff,db,sh);}
  if(lane%8==0){red[c]+=dg;red[256+c]+=db;}
 }
 sync_group();fence_proxy_async();sync_group();if(tid==0){for(int c=wi*128;c<(wi+1)*128;c+=16)tma_store_3d(&p.dtri,sx+c*128,m0,c,0);tma_store_commit();tma_store_wait_all();}sync_group();
}


// DW: two 96 KiB input slots; 192 FP32 dW accumulators per thread.
TMN_DEVI void cluster_dw(const Params& p,uint8_t* sm,uint64_t* full){
 auto cluster=cg::this_cluster();int split=blockIdx.x/2;
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32;
 MaskCycle mask=mask_cycle(p,split);int mi=0,round=0;float acc[3][64]={};
 for(int it=split;it<p.tiles;it+=PAIRS,++round){int slot=round&1;
  // Two slots; each toggles parity only on its own reuse.
  mbar_wait(full+slot,(round/2)&1);uint8_t* s=sm+slot*98304;
  cluster_b1<true>(p,s,it*64,mask,mi);
  static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*3+n;uint8_t* sa=t<2?s+49152+t*8192:s+((t-2)/2)*8192;uint8_t* sb=t<2?s+16384:s+65536+((t-2)%2)*16384;
   // Fence before WGMMA, then commit/wait before refilling either operand.
   fence_regs(acc[n]);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc[n],smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),it>split||k>0);});wgmma_commit();
  });wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);fence_regs(acc[2]);
  bool next=it+2*PAIRS<p.tiles;
  if(next&&threadIdx.x==0)mbar_arrive_expect_tx(full+slot,98304);
  // Both consumers finished and armed barriers before multicast reuse.
  cluster.sync();
  if(next)cluster_issue<true>(p,sm,full,slot,it+2*PAIRS,false);
  if(++mi==mask.period)mi=0;
 }
 static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*3+n,tile=t<2?0:1+(t-2)/2;float* part=p.partw+(tile*PAIRS+split)*16384;
#pragma unroll
  for(int q=0;q<16;++q){int rr=w*16+lane/4+(t<2?t*64:0),c=q*8+2*(lane%4)+(t<2?0:((t-2)%2)*128),stride=t<2?128:256;stg64f(part+rr*stride+c,acc[n][4*q],acc[n][4*q+1]);stg64f(part+(rr+8)*stride+c,acc[n][4*q+2],acc[n][4*q+3]);}
 });
}
// DX: two 64 KiB dy/gate/tri slots, resident 64 KiB Wp, 32 KiB dnorm,
// and 2 KiB LN partials. No dW accumulator is live during LN.
TMN_DEVI void cluster_dx(const Params& p,uint8_t* sm,uint64_t* full){
 auto cluster=cg::this_cluster();int split=blockIdx.x/2;int mi=0,round=0;
 MaskCycle mask=mask_cycle(p,split);
 for(int it=split;it<p.tiles;it+=PAIRS,++round){int slot=round&1;
  mbar_wait(full+slot,(round/2)&1);
  cluster_b1<false>(p,sm+slot*98304,it*64,mask,mi);
  dual_dgrad(p,sm,it*64,slot);
  bool next=it+2*PAIRS<p.tiles;
  if(next&&threadIdx.x==0)mbar_arrive_expect_tx(full+slot,65536);
  // DX store completion protects tri reuse; join DW before multicast.
  cluster.sync();
  if(next)cluster_issue<false>(p,sm,full,slot,it+2*PAIRS,false);
  if(++mi==mask.period)mi=0;
 }
 for(int j=threadIdx.x;j<512;j+=256)p.partln[split*512+j]=reinterpret_cast<float*>(sm+229376)[j];
}
extern "C" __global__ __cluster_dims__(2,1,1) __launch_bounds__(256,1)
void dual_b1b4(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t full[2];
 auto cluster=cg::this_cluster();const bool dw=(blockIdx.x&1)==0;int split=blockIdx.x/2;
 if(threadIdx.x==0){
  mbar_init(full,1);mbar_init(full+1,1);fence_barrier_init();
  if(split<p.tiles)mbar_arrive_expect_tx(full,dw?98304:131072);
  if(split+PAIRS<p.tiles)mbar_arrive_expect_tx(full+1,dw?98304:65536);
 }
 for(int i=threadIdx.x;i<512;i+=256)reinterpret_cast<float*>(sm+229376)[i]=0;
 // Both destinations install byte counts before any multicast completion.
 cluster.sync();
 if(dw){
  if(split<p.tiles)cluster_issue<true>(p,sm,full,0,split,true);
  if(split+PAIRS<p.tiles)cluster_issue<true>(p,sm,full,1,split+PAIRS,false);
  cluster_dw(p,sm,full);
 }else{
  if(split<p.tiles)cluster_issue<false>(p,sm,full,0,split,true);
  if(split+PAIRS<p.tiles)cluster_issue<false>(p,sm,full,1,split+PAIRS,false);
  cluster_dx(p,sm,full);
 }
#if PART_ONLY == 2
 // Every thread fences its partials; CTA joins before publishing a ticket.
 __threadfence();allsync();
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}
 allsync(); // All DW and DX partials are globally visible.
 for(int i=blockIdx.x*256+threadIdx.x;i<49664;i+=UCOUNT*256){
  if(i<49152){int tile=i/16384,j=i%16384;float v=0;for(int b=0;b<PAIRS;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*PAIRS+b)*16384+j];(tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);}
  else{int j=i-49152;float v=0;for(int b=0;b<PAIRS;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
 }
 __threadfence();allsync(); // Readers finish before completion tickets reset.
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void unified_reduce(__grid_constant__ const Params p){
 int i=blockIdx.x*256+threadIdx.x;
 if(i<49152){int tile=i/16384,j=i%16384;float v=0;for(int b=0;b<PAIRS;++b)v+=p.partw[(tile*PAIRS+b)*16384+j];(tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);}
 else if(i<49664){int j=i-49152;float v=0;for(int b=0;b<PAIRS;++b)v+=p.partln[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
}
