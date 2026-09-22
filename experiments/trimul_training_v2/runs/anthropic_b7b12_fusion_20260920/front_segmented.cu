#define WGRAD_SLICES 2
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
TMN_DEVI void glu_pair(const Params& p,uint8_t* s,int i,int row,float ma,float mb,uint32_t& dg,uint32_t& dp){
 int c=i/32,r=(i%32)*2;
 uint32_t dy=pair_get(s+32768,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r);

 // The mask multiply has its own BF16 boundary in the reference.
 uint32_t masked=pack_bf16(bf16lo(dy)*ma,bf16hi(dy)*mb);
 float ga=math::sigmoid_div(bf16lo(gl)),gb=math::sigmoid_div(bf16hi(gl));
 dg=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb));
 dp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
}
TMN_DEVI void write_glu(uint8_t* out,int i,uint32_t g,uint32_t p){
 int c=i/32,r=(i%32)*2;
 *reinterpret_cast<uint32_t*>(out+swz128(c,r*2))=g;
 *reinterpret_cast<uint32_t*>(out+16384+swz128(c,r*2))=p;
}
template<bool INPLACE> TMN_DEVI void glu(const Params& p,uint8_t* s,int row,int group){
 int mr=row+(threadIdx.x%32)*2;float ma=__bfloat162float(p.mask[mr]),mb=__bfloat162float(p.mask[mr+1]);
 if constexpr(INPLACE){
  uint32_t gg[16],pp[16];
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;glu_pair(p,s,threadIdx.x+q*256,row,ma,mb,gg[q],pp[q]);});
  // All preact reads finish before the transposed GLU stores overwrite it.
  allsync();
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int i=threadIdx.x+q*256;write_glu(s,i,gg[q],pp[q]);
#if DEBUG_SAVE
   int c=i/32,r=(i%32)*2,side=group/2,h=(group%2)*128+c;
   *reinterpret_cast<uint32_t*>(p.debugdc+(size_t)(side*512+h)*p.M+row+r)=gg[q];
   *reinterpret_cast<uint32_t*>(p.debugdc+(size_t)(side*512+256+h)*p.M+row+r)=pp[q];
#endif
  });
 }else{
  for(int i=threadIdx.x;i<4096;i+=256){uint32_t g,pr;glu_pair(p,s,i,row,ma,mb,g,pr);write_glu(s+65536,i,g,pr);}
 }
 // Publish generic stores to WGMMA's async proxy; all producers participate.
 fence_proxy_async();allsync();
}
TMN_DEVI void issue_dw(const Params& p,uint8_t* sm,uint64_t* bar,int slot,int tile,int group){
 if(threadIdx.x)return;uint8_t* s=sm+slot*DW_SLOT;int side=group/2,h=(group%2)*128,row=tile*64;
 mbar_arrive_expect_tx(bar+slot,65536);
 tma_load_2d(s,&p.pre,bar+slot,row,side*512+2*h);
 tma_load_2d(s+32768,side?&p.dr:&p.dl,bar+slot,row,h);
 for(int c=0;c<2;++c)tma_load_2d(s+49152+c*8192,&p.xn,bar+slot,c*64,row);
}
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* bar){
 int group=blockIdx.x/DW_SPLITS,physical_split=blockIdx.x%DW_SPLITS,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 #pragma unroll 1
 for(int segment=0;segment<WGRAD_SLICES;++segment){
 int split=physical_split+segment*DW_SPLITS;
 if(segment>0){allsync();if(threadIdx.x==0){
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(smem_u32(bar)):"memory");
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(smem_u32(bar+1)):"memory");
  mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}allsync();}
 float acc[2][64]={};
 if(split<p.tiles)issue_dw(p,sm,bar,0,split,group);
 if(split+DW_SPLITS*WGRAD_SLICES<p.tiles)issue_dw(p,sm,bar,1,split+DW_SPLITS*WGRAD_SLICES,group);
 int round=0;
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS*WGRAD_SLICES,++round){
  int slot=round&1;uint8_t* s=sm+slot*DW_SLOT;mbar_wait(bar+slot,(round/2)&1);
  glu<false>(p,s,tile*64,group);
  static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;fence_regs(acc[n]);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
    mma_weight128(acc[n],smem_desc(smem_u32(s+65536+n*16384+wi*8192+k*32),16,1024,1),smem_desc(smem_u32(s+49152+k*2048),8192,1024,1),round>0||k>0);
   });wgmma_commit();
  });wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);
  // WGMMA consumers completed before a producer refills this stage.
  allsync();if(tile+2*DW_SPLITS*WGRAD_SLICES<p.tiles)issue_dw(p,sm,bar,slot,tile+2*DW_SPLITS*WGRAD_SLICES,group);
 }
 static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;float* out=p.partw+(group*DW_SPLITS*WGRAD_SLICES+split)*32768+n*16384;
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=wi*64+w*16+lane/4,c=q*8+2*(lane%4);
   stg64f(out+rr*128+c,acc[n][q*4],acc[n][q*4+1]);stg64f(out+(rr+8)*128+c,acc[n][q*4+2],acc[n][q*4+3]);
  });
 });
 }
}
TMN_DEVI void issue_gate(const Params& p,uint8_t* sm,uint64_t* bar,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(bar,49152);
 for(int k=0;k<2;++k){tma_load_2d(sm+k*8192,&p.dg,bar,k*64,row);
  for(int n=0;n<2;++n)tma_load_2d(sm+16384+n*16384+k*8192,&p.wgate,bar,k*64,n*64);}
}
TMN_DEVI void issue_dx(const Params& p,uint8_t* sm,uint64_t* bar,int slot,int row,int group){
 if(threadIdx.x)return;uint8_t* s=sm+slot*DX_SLOT;int side=group/2,h=(group%2)*128;
 mbar_arrive_expect_tx(bar+slot,114688);
 tma_load_2d(s,&p.pre,bar+slot,row,side*512+2*h);
 tma_load_2d(s+32768,side?&p.dr:&p.dl,bar+slot,row,h);
 for(int kind=0;kind<2;++kind)for(int n=0;n<2;++n)for(int k=0;k<2;++k)
  tma_load_2d(s+49152+kind*32768+n*16384+k*8192,side?(kind?&p.wr:&p.wrg):(kind?&p.wl:&p.wlg),bar+slot,h+k*64,n*64);
}
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* bar){
 int split=blockIdx.x-DWCOUNT,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int ra=w*16+lane/4,rb=ra+8;float running=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT){
  int row=tile*64;uint32_t gate_packed[16];float acc[32]={};
  // Per row tile: bar0 gate+2channel+LN transactions (4); bar1 2channels.
  // Both phase parities return to0 before advancing to the next row tile.
  issue_gate(p,sm,bar,row);mbar_wait(bar,0);
  {float gate[32]={};
  fence_regs(gate);wgmma_fence();
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_dgrad(gate,smem_desc(smem_u32(sm+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+16384+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
  // B9 rounds separately before B10's accumulation/add.
  static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;gate_packed[j]=pack_bf16(gate[j*2],gate[j*2+1]);});}
  allsync();issue_dx(p,sm,bar,0,row,0);issue_dx(p,sm,bar,1,row,1);
  // Preserve baseline K order: Lg[0:256], Lp[0:256], Rg, Rp.
  for(int side=0;side<2;++side){
   for(int half=0;half<2;++half){
    int slot=half,group=side*2+half;uint8_t* s=sm+slot*DX_SLOT;
    mbar_wait(bar+slot,(side+1-half)&1);glu<true>(p,s,row,group);
    fence_regs(acc);wgmma_fence();
    static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
     mma_input64(acc,smem_desc(smem_u32(s+k*2048),16,1024,1),smem_desc(smem_u32(s+49152+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),side>0||half>0||k>0);
    });wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
   }
   for(int half=0;half<2;++half){
    uint8_t* s=sm+half*DX_SLOT;fence_regs(acc);wgmma_fence();
    static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
     mma_input64(acc,smem_desc(smem_u32(s+16384+k*2048),16,1024,1),smem_desc(smem_u32(s+81920+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),1);
    });wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
    if(side==0)issue_dx(p,sm,bar,half,row,2+half);
   }
  }
  // B10 outputs BF16 dx_n. Keep it in registers through B11/B12.
  static_for<16>([&](auto ji){constexpr int j=decltype(ji)::value;acc[j*2]=__bfloat162float(__float2bfloat16_rn(acc[j*2]+bf16lo(gate_packed[j])));acc[j*2+1]=__bfloat162float(__float2bfloat16_rn(acc[j*2+1]+bf16hi(gate_packed[j])));});
  if(threadIdx.x==0){mbar_arrive_expect_tx(bar,32768);
   for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.x,bar,c*64,row);tma_load_2d(sm+16384+c*8192,&p.res,bar,c*64,row);}}
  mbar_wait(bar,1);
  float mu[2]={p.mean[row+ra],p.mean[row+rb]},rs[2]={p.rs[row+ra],p.rs[row+rb]},s1[2]={},s2[2]={};
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;int rr=j&1,c=q*16+2*(lane%4)+8*(j>>1),r=rr?rb:ra;
    float xa=__fmul_rn(__fsub_rn(get(sm+wi*8192,r,c),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(get(sm+wi*8192,r,c+1),mu[rr]),rs[rr]);
    float ha=acc[q*8+j*2]*p.gamma[wi*64+c],hb=acc[q*8+j*2+1]*p.gamma[wi*64+c+1];s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;
#if DEBUG_SAVE
    p.debugxn[(size_t)(row+r)*128+wi*64+c]=__float2bfloat16_rn(acc[q*8+j*2]);p.debugxn[(size_t)(row+r)*128+wi*64+c+1]=__float2bfloat16_rn(acc[q*8+j*2+1]);
#endif
   });
  });
  s1[0]=quad_sum(s1[0])/128.f;s1[1]=quad_sum(s1[1])/128.f;s2[0]=quad_sum(s2[0])/128.f;s2[1]=quad_sum(s2[1])/128.f;
  float* stats=reinterpret_cast<float*>(sm+36864);float* tmp=reinterpret_cast<float*>(sm+32768);
  if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}
  allsync(); // Publish both64-channel halves before full128-channel LN reduction.
  float c1[2]={stats[ra*2]+stats[128+ra*2],stats[rb*2]+stats[128+rb*2]},c2[2]={stats[ra*2+1]+stats[128+ra*2+1],stats[rb*2+1]+stats[128+rb*2+1]};
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;int j=pair*2,c=q*16+2*(lane%4)+8*pair,gc=wi*64+c;
    float xaa=__fmul_rn(__fsub_rn(get(sm+wi*8192,ra,c),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(get(sm+wi*8192,ra,c+1),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(get(sm+wi*8192,rb,c),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(get(sm+wi*8192,rb,c+1),mu[1]),rs[1]);
    float da=acc[q*8+j*2],db=acc[q*8+j*2+1],dc=acc[q*8+j*2+2],dd=acc[q*8+j*2+3],ga=p.gamma[gc],gb=p.gamma[gc+1];
    uint32_t outa=pack_bf16((da*ga-fmaf(xaa,c1[0],c2[0]))*rs[0],(db*gb-fmaf(xab,c1[0],c2[0]))*rs[0]);
    uint32_t outb=pack_bf16((dc*ga-fmaf(xba,c1[1],c2[1]))*rs[1],(dd*gb-fmaf(xbb,c1[1],c2[1]))*rs[1]);
    put(sm+wi*8192,ra,c,bf16lo(outa)+get(sm+16384+wi*8192,ra,c));put(sm+wi*8192,ra,c+1,bf16hi(outa)+get(sm+16384+wi*8192,ra,c+1));
    put(sm+wi*8192,rb,c,bf16lo(outb)+get(sm+16384+wi*8192,rb,c));put(sm+wi*8192,rb,c+1,bf16hi(outb)+get(sm+16384+wi*8192,rb,c+1));
    float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
    if(lane<4){tmp[w*256+gc]=dga;tmp[w*256+gc+1]=dgb;tmp[w*256+128+gc]=dba;tmp[w*256+128+gc+1]=dbb;}
   });
  });
  allsync(); // All warp parameter partials and dx stores published.
  running+=(tmp[threadIdx.x]+tmp[256+threadIdx.x])+(tmp[512+threadIdx.x]+tmp[768+threadIdx.x]);
  fence_proxy_async();allsync();
  if(threadIdx.x==0){store2d(&p.dx,sm,0,row);store2d(&p.dx,sm+8192,64,row);tma_store_commit();tma_store_wait_all();}
  allsync(); // TMA store cannot read a slot overwritten by next tile's gate loads.
 }
 p.partln[split*256+threadIdx.x]=running;
}
TMN_DEVI void reduce_at(const Params& p,int i){
 if(i<131072){int group=i/32768,j=i%32768,kind=j/16384,z=j%16384;float v=0;
  for(int b=0;b<DW_SPLITS*WGRAD_SLICES;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*WGRAD_SLICES+b)*32768+j];
  int out=(group/2)*2+(kind==0?1:0),rr=z%128,c=(group%2)*128+z/128;p.dw[(out*128+rr)*256+c]=__float2bfloat16_rn(v);
 }else if(i<131328){int c=i-131072;float v=0;for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}
}
extern "C" __global__ __launch_bounds__(256,1)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[2];
 if(threadIdx.x==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}allsync();
 if(blockIdx.x<DWCOUNT)weight_role(p,sm,bar);else input_role(p,sm,bar);
#if PART_ONLY == 2
 __threadfence();allsync(); // Publish all partial writers before completion ticket.
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();
 for(int i=blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256)reduce_at(p,i);
 __threadfence();allsync();
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
