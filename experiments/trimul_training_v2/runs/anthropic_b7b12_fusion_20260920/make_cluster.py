from pathlib import Path
p=Path(__file__).resolve().parent
kind=(p/'front_twocta_kindwg.cu').read_text();params=kind[kind.index('struct Params'):kind.index('TMN_DEVI void mma_weight64')]
store=kind[kind.index('TMN_DEVI void store2d'):kind.index('TMN_DEVI void glu_small')]
gate=kind[kind.index('TMN_DEVI void issue_gate'):kind.index('TMN_DEVI void input_role')]
ln=kind[kind.index('  // B10 outputs BF16 dx_n.'):kind.index('\n p.partln[split*256+threadIdx.x]=running;')]
# Remove the final loop close: this fragment belongs inside the new consumer loop.
assert ln.endswith('\n }');ln=ln[:-3]+'''
  if(threadIdx.x==0){if(tile+CLUSTERS*4<p.tiles)mbar_arrive_expect_tx(&b->copy,131072);for(int group=0;group<4;++group)remote_arrive(b->empty+slot,group);}
'''
head='''// SPDX-License-Identifier: Apache-2.0
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
'''
code=r'''
struct Bars{uint64_t ready,empty[4],copy,io,weight[2];};
TMN_DEVI uint32_t remote_addr(const void* ptr,int rank){uint32_t out;asm("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(out):"r"(smem_u32(ptr)),"r"(rank));return out;}
TMN_DEVI void remote_arrive(const uint64_t* ptr,int rank){asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"::"r"(remote_addr(ptr,rank)):"memory");}
TMN_DEVI void cluster_copy(uint32_t dst,const void* src,uint32_t bar){asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0],[%1],32768,[%2];"::"r"(dst),"r"(smem_u32(src)),"r"(bar):"memory");}
TMN_DEVI void producer(const Params& p,uint8_t* sm,Bars* b){
 int cid=blockIdx.x/8,group=blockIdx.x%8,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 float acc[2][64]={};int total=0;for(int base=cid*4;base<p.tiles;base+=CLUSTERS*4)total+=min(4,p.tiles-base);int segment=(total+1)/2,r=0,round=0;
 for(int base=cid*4;base<p.tiles;base+=CLUSTERS*4,++round){
  for(int q=0;q<4&&base+q<p.tiles;++q,++r){int row=(base+q)*64;uint8_t* out=sm+q*32768;uint8_t* inp=sm+131072;
   if(threadIdx.x==0){if(round)mbar_wait(b->empty+q,(round-1)&1);mbar_arrive_expect_tx(&b->io,65536);
    tma_load_2d(inp,&p.pre,&b->io,row,(group/2)*512+(group%2)*256);tma_load_2d(inp+32768,group>=2?&p.dr:&p.dl,&b->io,row,(group%2)*128);
    for(int c=0;c<2;++c)tma_load_2d(inp+49152+c*8192,&p.xn,&b->io,c*64,row);
   }mbar_wait(&b->io,r&1);
   uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(threadIdx.x%32)*2);
   #pragma unroll 2
   for(unsigned j=0;j<16;++j){unsigned i=threadIdx.x+j*256,c=i/32,rr=(i%32)*2;
    uint32_t dy=pair_get(inp+32768,c,rr),gl=pair_get(inp,c*2,rr),pr=pair_get(inp,c*2+1,rr),masked;
    asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(mask));float ga=math::sigmoid(bf16lo(gl)),gb=math::sigmoid(bf16hi(gl));
    uint32_t gg=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb));
    uint32_t pp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
    *reinterpret_cast<uint32_t*>(out+swz128(c,rr*2))=gg;*reinterpret_cast<uint32_t*>(out+16384+swz128(c,rr*2))=pp;
   }fence_proxy_async();allsync();if(threadIdx.x==0)cluster_copy(remote_addr(sm+group*32768,4+q),out,remote_addr(&b->copy,4+q));
   static_for<2>([&](auto hi){constexpr int half=decltype(hi)::value;fence_regs(acc[half]);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_weight128(acc[half],smem_desc(smem_u32(out+wi*16384+half*8192+k*32),16,1024,1),smem_desc(smem_u32(inp+49152+k*2048),8192,1024,1),r%segment>0||k>0);});
    wgmma_commit();wgmma_wait<0>();fence_regs(acc[half]);
   });allsync();
   if((r+1)%segment==0||r+1==total){float* dest=p.partw+((group*CLUSTERS+cid)*2+r/segment)*32768+wi*16384;
    static_for<2>([&](auto hi){constexpr int half=decltype(hi)::value;static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;int rr=half*64+w*16+lane/4,c=j*8+2*(lane%4);stg64f(dest+rr*128+c,acc[half][j*4],acc[half][j*4+1]);stg64f(dest+(rr+8)*128+c,acc[half][j*4+2],acc[half][j*4+3]);});});
   }
  }
 }
}
TMN_DEVI void load_weight(const Params& p,uint8_t* sm,Bars* b,int side,int kind,int half){
 if(threadIdx.x)return;uint8_t* s=sm+131072+half*32768;const CUtensorMap* map=kind?(side?&p.wr:&p.wl):(side?&p.wrg:&p.wlg);mbar_arrive_expect_tx(b->weight+half,32768);
 for(int c=0;c<2;++c)for(int k=0;k<2;++k)tma_load_2d(s+c*16384+k*8192,map,b->weight+half,half*128+k*64,c*64);
}
TMN_DEVI void consumer(const Params& p,uint8_t* sm,Bars* b,const float* gamma){
 int cid=blockIdx.x/8,slot=blockIdx.x%8-4,split=cid*4+slot,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int ra=w*16+lane/4,rb=ra+8;float running=0;uint64_t* bar=&b->io;int round=0;
 for(int tile=split;tile<p.tiles;tile+=CLUSTERS*4,++round){int row=tile*64;uint32_t gate_packed[16];float acc[32]={};
  issue_gate(p,sm+131072,bar,row);
  mbar_wait(bar,0);
  {float gate[32]={};fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_dgrad(gate,smem_desc(smem_u32(sm+131072+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+131072+16384+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(gate);
   static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;gate_packed[j]=pack_bf16(gate[j*2],gate[j*2+1]);});
  }allsync();mbar_wait(&b->copy,round&1);
  for(int side=0;side<2;++side){
   for(int kind=0;kind<2;++kind){load_weight(p,sm,b,side,kind,0);load_weight(p,sm,b,side,kind,1);
    for(int half=0;half<2;++half){mbar_wait(b->weight+half,kind);uint8_t* as=sm+(side*2+half)*32768+kind*16384;uint8_t* ws=sm+131072+half*32768+wi*16384;
     fence_regs(acc);wgmma_fence();static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_input64(acc,smem_desc(smem_u32(as+k*2048),16,1024,1),smem_desc(smem_u32(ws+(k/4)*8192+(k%4)*32),16,1024,1),side>0||kind>0||half>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
    }
   }
  }
'''
reduce=r'''
 }
 p.partln[split*256+threadIdx.x]=running;
}
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<131072){int group=i/32768,j=i%32768,kind=j/16384,z=j%16384;float v=0;for(int k=0;k<2*CLUSTERS;++k)v+=reinterpret_cast<volatile float*>(p.partw)[(group*CLUSTERS*2+k)*32768+j];int out=(group/2)*2+(kind==0?1:0),rr=z%128,c=(group%2)*128+z/128;p.dw[(out*128+rr)*256+c]=__float2bfloat16_rn(v);
 }else if(i<131328){int c=i-131072;float v=0;for(int k=0;k<CLUSTERS*4;++k)v+=reinterpret_cast<volatile float*>(p.partln)[k*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}
}
extern "C" __global__ __cluster_dims__(8,1,1) __launch_bounds__(256,1)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ Bars b;__shared__ float gamma[128];
 if(threadIdx.x<128)gamma[threadIdx.x]=p.gamma[threadIdx.x];if(threadIdx.x==0){mbar_init(&b.ready,4);for(int i=0;i<4;++i)mbar_init(b.empty+i,1);mbar_init(&b.copy,1);if(blockIdx.x%8>=4)mbar_arrive_expect_tx(&b.copy,131072);mbar_init(&b.io,1);mbar_init(b.weight,1);mbar_init(b.weight+1,1);fence_barrier_init();}allsync();auto cluster=cooperative_groups::this_cluster();cluster.sync();
 if(blockIdx.x%8<4)producer(p,sm,&b);else consumer(p,sm,&b,gamma);
 cluster.sync();__threadfence();allsync();if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();for(int i=blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256)reduce_at(p,i);__threadfence();allsync();if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
'''
(p/'front_cluster.cu').write_text(head+params+store+gate+code+ln+reduce)
(p/'front_cluster.launch.json').write_text('{"direct_weights":true,"wgrad_slices":2}\n')
