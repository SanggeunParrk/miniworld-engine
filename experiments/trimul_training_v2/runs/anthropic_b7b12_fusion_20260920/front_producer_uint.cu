#define DIRECT_FOUR_WEIGHTS 1
// SPDX-License-Identifier: Apache-2.0
// Anthropic native v5 primitives; B7-B12 fixed-saves training extension.
#include "front_mn_primitives.cuh"
#ifndef UCOUNT
#define UCOUNT 132
#endif
#ifndef DW_SPLITS
#define DW_SPLITS 12
#endif
#ifndef PART_ONLY
#define PART_ONLY 2
#endif
#ifndef DEBUG_SAVE
#define DEBUG_SAVE 0
#endif
constexpr int DWCOUNT=4*DW_SPLITS,DXCOUNT=UCOUNT-DWCOUNT;
static_assert(DXCOUNT>0&&DW_SPLITS>0,"Both roles required");
constexpr int DX_SLOT=114688,DW_SLOT=98304;
struct Params{
 CUtensorMap dl,dr,pre,xn,dg,wlg,wl,wrg,wr,wgate,x,res,dx;
 const __nv_bfloat16* mask;
 const float *mean,*rs,*gamma;
 __nv_bfloat16* dw;
 float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;
 __nv_bfloat16 *debugdc,*debugxn;
 int M,L,tiles;
};
TMN_DEVI void store2d(const CUtensorMap* map,const void* src,int c,int r){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2,%3}], [%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");
}
// Each pair is two adjacent rows, preserving preact's channel-major loads.
TMN_DEVI void glu_pair(const Params& p,uint8_t* s,unsigned i,int row,float ma,float mb,uint32_t& dg,uint32_t& dp){
 unsigned c=i/32,r=(i%32)*2;
 uint32_t dy=pair_get(s+32768,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r);

 // The mask multiply has its own BF16 boundary in the reference.
 uint32_t masked=pack_bf16(bf16lo(dy)*ma,bf16hi(dy)*mb);
 float ga=math::sigmoid_div(bf16lo(gl)),gb=math::sigmoid_div(bf16hi(gl));
 dg=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb));
 dp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
}

TMN_DEVI void load_prod(const Params& p,uint8_t* sm,uint64_t* b,int group,int row){
 if(threadIdx.x)return;int slot=group&1;uint8_t* s=sm+slot*49152;mbar_arrive_expect_tx(b+slot,49152);
 tma_load_2d(s,&p.pre,b+slot,row,(group/2)*512+(group%2)*256);
 tma_load_2d(s+32768,group>=2?&p.dr:&p.dl,b+slot,row,(group%2)*128);
}
TMN_DEVI void producer(const Params& p,uint8_t* sm,uint64_t* b){
 for(int tile=blockIdx.x;tile<p.tiles;tile+=UCOUNT){int row=tile*64;load_prod(p,sm,b,0,row);load_prod(p,sm,b,1,row);
  float ma=__bfloat162float(p.mask[row+(threadIdx.x%32)*2]),mb=__bfloat162float(p.mask[row+(threadIdx.x%32)*2+1]);
  for(int group=0;group<4;++group){int slot=group&1;mbar_wait(b+slot,group/2);uint8_t* s=sm+slot*49152;
   #pragma unroll 4
   for(int q=0;q<16;++q){unsigned i=threadIdx.x+q*256,c=i/32,r=(i%32)*2;uint32_t g,pr;glu_pair(p,s,i,row,ma,mb,g,pr);
    int h=(group%2)*128+c,side=group/2;
    *reinterpret_cast<uint32_t*>(p.debugdc+(size_t)(side*512+h)*p.M+row+r)=g;
    *reinterpret_cast<uint32_t*>(p.debugdc+(size_t)(side*512+256+h)*p.M+row+r)=pr;
   }allsync();if(group<2)load_prod(p,sm,b,group+2,row);
  }
 }
}
extern "C" __global__ __launch_bounds__(256,1)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[2];
 if(threadIdx.x==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}allsync();producer(p,sm,bar);
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){}
