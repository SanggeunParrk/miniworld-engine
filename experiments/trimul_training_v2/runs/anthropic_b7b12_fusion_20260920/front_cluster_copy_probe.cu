// SPDX-License-Identifier: Apache-2.0
// Anthropic v5 primitives, MiniWorld training extension: single GLU producer via Hopper DSM.
#define DIRECT_FOUR_WEIGHTS 1
#include "front_mn_primitives.cuh"
#include <cooperative_groups.h>
#ifndef UCOUNT
#define UCOUNT 120
#endif
#ifndef DW_SPLITS
#define DW_SPLITS 15
#endif
constexpr int CLUSTERS=UCOUNT/8;
static_assert(UCOUNT%8==0&&DW_SPLITS==CLUSTERS,"cluster geometry mismatch");
struct Params{
 CUtensorMap dl,dr,pre,xn,dg,wlg,wl,wrg,wr,wgate,x,res,dx;
 const __nv_bfloat16* mask;const float *mean,*rs,*gamma;
 __nv_bfloat16* dw;float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;__nv_bfloat16 *debugdc,*debugxn;int M,L,tiles;
};
TMN_DEVI void store2d(const CUtensorMap* map,const void* src,int c,int r){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2,%3}], [%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");
}

TMN_DEVI void issue_gate(const Params& p,uint8_t* sm,uint64_t* bar,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(bar,49152);
 for(int k=0;k<2;++k){tma_load_2d(sm+k*8192,&p.dg,bar,k*64,row);
  for(int n=0;n<2;++n)tma_load_2d(sm+16384+n*16384+k*8192,&p.wgate,bar,k*64,n*64);}
}

struct Bars{uint64_t ready,empty[4],copy,io,weight[2];};
TMN_DEVI uint32_t remote_addr(const void* ptr,int rank){uint32_t out;asm("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(out):"r"(smem_u32(ptr)),"r"(rank));return out;}
TMN_DEVI void remote_arrive(const uint64_t* ptr,int rank){asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"::"r"(remote_addr(ptr,rank)):"memory");}
TMN_DEVI void cluster_copy(uint32_t dst,const void* src,uint32_t bar){asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0],[%1],32768,[%2];"::"r"(dst),"r"(smem_u32(src)),"r"(bar):"memory");}

extern "C" __global__ __cluster_dims__(8,1,1) __launch_bounds__(256,1) void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ Bars b;int rank=blockIdx.x%8;for(int i=threadIdx.x;i<131072/4;i+=256)reinterpret_cast<unsigned*>(sm)[i]=blockIdx.x;
 if(threadIdx.x==0){for(int q=0;q<4;++q)mbar_init(b.empty+q,1);mbar_init(&b.copy,1);if(rank>=4)mbar_arrive_expect_tx(&b.copy,131072);fence_barrier_init();}fence_proxy_async();allsync();auto c=cooperative_groups::this_cluster();c.sync();
 for(int r=0;r<128;++r){
  if(rank<4){if(threadIdx.x==0){for(int q=0;q<4;++q){if(r)mbar_wait(b.empty+q,(r-1)&1);cluster_copy(remote_addr(sm+rank*32768,4+q),sm,remote_addr(&b.copy,4+q));}}allsync();}
  else{mbar_wait(&b.copy,r&1);allsync();if(threadIdx.x==0){if(r+1<128)mbar_arrive_expect_tx(&b.copy,131072);for(int q=0;q<4;++q)remote_arrive(b.empty+rank-4,q);}}
 }
 c.sync();if(rank>=4&&threadIdx.x<4)p.partln[(blockIdx.x/8*4+rank-4)*256+threadIdx.x]=float(reinterpret_cast<unsigned*>(sm)[threadIdx.x*8192]);
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){}
