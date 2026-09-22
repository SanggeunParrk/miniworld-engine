// SPDX-License-Identifier: Apache-2.0
// Training extension using Anthropic-derived Hopper WGMMA/TMA primitives.
#include <cooperative_groups.h>
#include "common_recompute.cuh"
constexpr int THREADS=384,SMEM=180224;
#define allsync() named_bar_sync(0,128)
struct Params{CUtensorMap xn,wp,dl,dr,dg,wgate,x,res,dx;const __nv_bfloat16* mask;const float *gamma,*beta;__nv_bfloat16 *dw;float *dgam,*dbeta,*partw,*partln;unsigned int* counts;int M,tiles;};
#include "single_wg.inc"
TMN_DEVI void store2d(const CUtensorMap* map,const void* src,int c,int r){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");}
TMN_DEVI void multicast_xn(void* dst,const CUtensorMap* map,uint64_t* bar,int c,int row){asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%3,%4}],[%2],%5;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c),"r"(row),"h"(uint16_t(255)):"memory");}
TMN_DEVI void dw_step(float (&g)[64],float (&p)[64],uint8_t* sm,int half,int accumulated){
 fence_regs(g);fence_regs(p);wgmma_fence();
 static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
  mma_weight128(g,smem_desc(smem_u32(sm+81920+half*8192+k*32),16,1024,1),smem_desc(smem_u32(sm+65536+k*2048),8192,1024,1),accumulated||k>0);
  mma_weight128(p,smem_desc(smem_u32(sm+98304+half*8192+k*32),16,1024,1),smem_desc(smem_u32(sm+65536+k*2048),8192,1024,1),accumulated||k>0);
 });wgmma_commit();wgmma_wait<0>();fence_regs(g);fence_regs(p);
}
TMN_DEVI void joint_dx(float (&acc)[64],uint8_t* sm){
 fence_regs(acc);wgmma_fence();
 static_for<1>([&](auto cc){constexpr int chunk=decltype(cc)::value;static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
  single_dx(acc,smem_desc(smem_u32(sm+81920+chunk*8192+k*2048),16,1024,1),smem_desc(smem_u32(sm+chunk*32768+(k/2)*16384+(k%2)*2048),8192,1024,1),chunk>0||k>0);
 });});wgmma_commit();
 static_for<1>([&](auto cc){constexpr int chunk=decltype(cc)::value;static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
  single_dx(acc,smem_desc(smem_u32(sm+98304+chunk*8192+k*2048),16,1024,1),smem_desc(smem_u32(sm+chunk*32768+4096+(k/2)*16384+(k%2)*2048),8192,1024,1),1);
 });});wgmma_commit();wgmma_wait<0>();fence_regs(acc);
}
TMN_DEVI void store_dw(const Params& p,float (&v)[64],int cluster,int rank,int half,int kind){
 int lane=threadIdx.x%32,w=(threadIdx.x%128)/32;float* dst=p.partw+((((cluster*8+rank)*1+half)*2+kind)*8192);
 static_for<16>([&](auto qq){constexpr int q=decltype(qq)::value;int row=w*16+lane/4,c=q*8+2*(lane%4);stg64f(dst+row*128+c,v[q*4],v[q*4+1]);stg64f(dst+(row+8)*128+c,v[q*4+2],v[q*4+3]);});
}
TMN_DEVI void csync(){named_bar_sync(15,384);asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;":::"memory");}
TMN_DEVI void producer_loop(const Params& p,uint8_t* sm,uint64_t* bar){
 int rank=blockIdx.x%8,cid=blockIdx.x/8,clusters=gridDim.x/8,round=0;
 for(int tile=cid;tile<p.tiles;tile+=clusters,++round){int row=tile*64;
  if(threadIdx.x==0)mbar_arrive_expect_tx(bar,24576);csync();
  if(threadIdx.x==0){if(rank==0)for(int k=0;k<2;++k)multicast_xn(sm+65536+k*8192,&p.xn,bar,k*64,row);
   for(int h=0;h<2;++h)tma_load_2d(sm+81920+h*4096,rank<4?&p.dl:&p.dr,bar,row,(rank%4)*64+h*32);
  }
  csync();
  if(rank==0&&threadIdx.x==0){mbar_arrive_expect_tx(bar+5,49152);for(int k=0;k<2;++k){tma_load_2d(sm+65536+k*8192,&p.x,bar+5,k*64,row);tma_load_2d(sm+81920+k*8192,&p.res,bar+5,k*64,row);tma_load_2d(sm+98304+k*8192,&p.dg,bar+5,k*64,row);}}
  csync();
 }
}
TMN_DEVI void dw_loop(const Params& p,uint8_t* sm,uint64_t* bar){
 int rank=blockIdx.x%8,cid=blockIdx.x/8,clusters=gridDim.x/8,half=0,round=0;float dwg[64]={},dwp[64]={};
 for(int tile=cid;tile<p.tiles;tile+=clusters,++round){csync();mbar_wait(bar+3,round&1);dw_step(dwg,dwp,sm,half,round>0);csync();csync();}
 store_dw(p,dwg,cid,rank,half,1);store_dw(p,dwp,cid,rank,half,0);
}
TMN_DEVI void compute_loop(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma,const float* beta){
 int rank=blockIdx.x%8,cid=blockIdx.x/8,clusters=gridDim.x/8,round=0;
 int tid=threadIdx.x%128,lane=tid%32,w=tid/32,ra=w*16+lane/4,rb=ra+8;float run_g=0,run_b=0;
 for(int tile=cid;tile<p.tiles;tile+=clusters,++round){int row=tile*64;csync();mbar_wait(bar,round&1);float acc[64]={};
  {uint32_t xn[8][4];load_frag_bf16<8,8192>(xn,smem_u32(sm+65536),w*16,lane);allsync();
   uint32_t ma=__bfloat16_as_ushort(p.mask[row+ra]),mb=__bfloat16_as_ushort(p.mask[row+rb]);ma|=ma<<16;mb|=mb<<16;
   for(int h=0;h<1;++h)single_derivatives(p,xn,sm+h*32768,sm+81920+h*8192,sm+81920+h*8192,sm+98304+h*8192,row,ma,mb);
  }
  if(tid==0)mbar_arrive(bar+3);joint_dx(acc,sm);
  static_for<16>([&](auto qq){constexpr int q=decltype(qq)::value;int r=w*16+lane/4,c=q*8+2*(lane%4);float* z=reinterpret_cast<float*>(sm+114688);
   z[r*128+c]=acc[q*4];z[r*128+c+1]=acc[q*4+1];z[(r+8)*128+c]=acc[q*4+2];z[(r+8)*128+c+1]=acc[q*4+3];});
  csync();
  if(rank==0){

   for(int other=1;other<8;++other){const float* remote=cooperative_groups::this_cluster().map_shared_rank(reinterpret_cast<float*>(sm+114688),other);
    static_for<16>([&](auto qq){constexpr int q=decltype(qq)::value;int r=w*16+lane/4,c=q*8+2*(lane%4);
     acc[q*4]+=remote[r*128+c];acc[q*4+1]+=remote[r*128+c+1];acc[q*4+2]+=remote[(r+8)*128+c];acc[q*4+3]+=remote[(r+8)*128+c+1];});
   }mbar_wait(bar+5,round&1);
   {float gate[64]={};fence_regs(gate);wgmma_fence();static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;
    gate128(gate,smem_desc(smem_u32(sm+98304+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+147456+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(gate);static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;acc[j]=math::round_bf16(acc[j]+math::round_bf16(gate[j]));});}  uint8_t* lnsm=sm+65536;
  uint32_t raw[8][4];load_frag_bf16<8,8192>(raw,smem_u32(lnsm),w*16,lane);
  LnStats st=ln_stats_only<8>(raw,gamma,beta,lane,1e-5f);
  float mu[2]={st.mA,st.mB},rs[2]={st.rA,st.rB},s1[2][2]={},s2[2][2]={};
  static_for<8>([&](auto qq){constexpr int q=decltype(qq)::value;
   static_for<4>([&](auto jj){constexpr int j=decltype(jj)::value;int rr=j&1,c=q*16+2*(lane%4)+8*(j>>1);
    float xa=__fmul_rn(__fsub_rn(bf16lo(raw[q][j]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(raw[q][j]),mu[rr]),rs[rr]);
    float ha=acc[q*8+j*2]*gamma[c],hb=acc[q*8+j*2+1]*gamma[c+1];s1[q/4][rr]+=ha*xa+hb*xb;s2[q/4][rr]+=ha+hb;
   });
  });
  float c1[2],c2[2];
  static_for<2>([&](auto ii){constexpr int i=decltype(ii)::value;c1[i]=quad_sum(s1[0][i])/128.f+quad_sum(s1[1][i])/128.f;c2[i]=quad_sum(s2[0][i])/128.f+quad_sum(s2[1][i])/128.f;});
  allsync(); // All LN source reads finish before in-place dx stores.
  float* tmp=reinterpret_cast<float*>(lnsm+32768);
  static_for<8>([&](auto qq){constexpr int q=decltype(qq)::value;
   static_for<2>([&](auto pp){constexpr int pair=decltype(pp)::value;int j=pair*2,c=q*16+2*(lane%4)+8*pair;
    float xaa=__fmul_rn(__fsub_rn(bf16lo(raw[q][pair*2]),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(bf16hi(raw[q][pair*2]),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(bf16lo(raw[q][pair*2+1]),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(bf16hi(raw[q][pair*2+1]),mu[1]),rs[1]);
    float da=acc[q*8+pair*4],db=acc[q*8+pair*4+1],dc=acc[q*8+pair*4+2],dd=acc[q*8+pair*4+3],ga=gamma[c],gb=gamma[c+1];
    uint32_t oa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
    uint32_t ob=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
    uint8_t* out=lnsm+(q/4)*8192;uint8_t* res=lnsm+16384+(q/4)*8192;
    uint32_t resa=pair_get(res,ra,c%64),resb=pair_get(res,rb,c%64);
    *reinterpret_cast<uint32_t*>(out+swz128(ra,(c%64)*2))=pack_bf16(bf16lo(oa)+bf16lo(resa),bf16hi(oa)+bf16hi(resa));
    *reinterpret_cast<uint32_t*>(out+swz128(rb,(c%64)*2))=pack_bf16(bf16lo(ob)+bf16lo(resb),bf16hi(ob)+bf16hi(resb));
    float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
    if(lane<4){tmp[w*256+c]=dga;tmp[w*256+c+1]=dgb;tmp[w*256+128+c]=dba;tmp[w*256+128+c+1]=dbb;}
   });
  });allsync();
  run_g+=(tmp[tid]+tmp[256+tid])+(tmp[512+tid]+tmp[768+tid]);
  run_b+=(tmp[128+tid]+tmp[384+tid])+(tmp[640+tid]+tmp[896+tid]);
  fence_proxy_async();allsync();
  if(tid==0){store2d(&p.dx,lnsm,0,row);store2d(&p.dx,lnsm+8192,64,row);tma_store_commit();}
   if(tid==0)tma_store_wait_all();allsync();
  }csync();
 }
 if(rank==0){p.partln[cid*256+tid]=run_g;p.partln[cid*256+128+tid]=run_b;}
}
extern "C" __global__ __cluster_dims__(8,1,1) __launch_bounds__(384,1)
void b7_joint(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[6];__shared__ float gamma[128],beta[128];
 int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=threadIdx.x%32,w=tid/32,ra=w*16+lane/4,rb=ra+8;
 auto cl=cooperative_groups::this_cluster();int rank=blockIdx.x%8,cid=blockIdx.x/8,clusters=gridDim.x/8;
 if(threadIdx.x<128){gamma[threadIdx.x]=p.gamma[threadIdx.x];beta[threadIdx.x]=p.beta[threadIdx.x];}
 if(threadIdx.x==0){for(int i=0;i<6;++i)mbar_init(bar+i,1);fence_barrier_init();}
 __syncthreads();cl.sync();
 if(threadIdx.x==0){mbar_arrive_expect_tx(bar+1,rank==0?65536:32768);
  for(int half=0;half<2;++half)for(int k=0;k<2;++k)tma_load_2d(sm+half*16384+k*8192,&p.wp,bar+1,k*64,rank*128+half*64);
  if(rank==0)for(int n=0;n<2;++n)for(int k=0;k<2;++k)tma_load_2d(sm+147456+n*16384+k*8192,&p.wgate,bar+1,k*64,n*64);
 }mbar_wait(bar+1,0);__syncthreads();
 if(wi==0){setmaxnreg_dec<32>();producer_loop(p,sm,bar);}
 else if(wi==1){setmaxnreg_inc<224>();compute_loop(p,sm,bar,gamma,beta);}
 else{setmaxnreg_inc<248>();dw_loop(p,sm,bar);}

 __threadfence();__syncthreads();
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=gridDim.x)__nanosleep(32);}__syncthreads();
 for(int i=blockIdx.x*THREADS+threadIdx.x;i<131328;i+=gridDim.x*THREADS){float v=0;
  if(i<131072){int kind=i/32768,c=(i/256)%128,h=i%256,rk=(kind/2)*4+h/64,half=0,j=(h%64)*128+c;
   for(int a=0;a<clusters;++a)v+=reinterpret_cast<volatile float*>(p.partw)[((((a*8+rk)*1+half)*2+(kind&1))*8192+j)];p.dw[i]=__float2bfloat16_rn(v);
  }else{int c=i-131072;for(int a=0;a<clusters;++a)v+=reinterpret_cast<volatile float*>(p.partln)[a*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}
 }
 __threadfence();__syncthreads();if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==gridDim.x-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
}
