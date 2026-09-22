// Diagnostic: dual_stream_floor. Pattern-faithful memory floor of B1-B4 with NO arithmetic and NO role split:
// every CTA streams its cyclic 32-row tiles of dy, gate, proj, xn, norm, tri (TMA, 64KiB per tile) through
// THREE slots, writes dg (stg128 of the dy tile) and dtri (TMA store of the tri tile). Same bytes as the
// real kernel (dy/gate read once), balanced by construction, 3-deep pipeline. Outputs are meaningless.
// SPDX-License-Identifier: Apache-2.0
// Inlined copy of dual_primitives.cuh (hashed with this source) with seven extra tensor maps.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
struct Params {
 CUtensorMap dy,gate,proj,xn,norm,tri,wp,dtri;
 const __nv_bfloat16* ds;
 const float *mean,*rs,*gamma;
 __nv_bfloat16 *dg,*dwg,*dwp;
 float *dgam,*dbeta,*partw,*partln,*groupln;
 unsigned int *counts;
 int M,L,tiles,groups;
 CUtensorMap dy32,gate32,proj32,xn32,norm32,tri32,dtri32; // dual_stream_floor: 32-row boxes (appended).
};
TMN_DEVI void sync_group(){named_bar_sync(1+threadIdx.x/128,128);}
TMN_DEVI float get(const uint8_t* s,int r,int c){return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(s+swz128(r,c*2)));}
TMN_DEVI void put(uint8_t* s,int r,int c,float v){*reinterpret_cast<__nv_bfloat16*>(s+swz128(r,c*2))=__float2bfloat16_rn(v);}
TMN_DEVI void load_trans(uint32_t (&f)[4][4],uint8_t* s){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8,r8=lane%8;
#pragma unroll
 for(int k=0;k<4;++k)ldsm_x4_t(f[k],smem_u32(s)+swz128(16*k+r8+8*((mat>>1)&1),(16*w+8*(mat&1))*2));
}
TMN_DEVI uint32_t pair_get(const uint8_t* s,int r,int c){return *reinterpret_cast<const uint32_t*>(s+swz128(r,c*2));}
TMN_DEVI void pair_put(uint8_t* s,int r,int c,uint32_t v){*reinterpret_cast<uint32_t*>(s+swz128(r,c*2))=v;}
TMN_DEVI float warp_sum(float x){
#pragma unroll
 for(int k=16;k;k>>=1)x+=__shfl_xor_sync(0xffffffff,x,k);
 return x;
}
// No CTA waits for another CTA. Last-arriver reduces published partials and resets
// its counter. Workspace belongs to one plan/stream; stream completion permits reuse.
TMN_DEVI bool ticket(unsigned int* p,unsigned int total,int* last){
 __threadfence();sync_group();
 if(threadIdx.x%128==0)*last=(atomicAdd(p,1u)==total-1);
 sync_group();return *last;
}
// NVIDIA WGMMA SS N128 operand contract; raw TMA buffers stay MN-major.
TMN_DEVI void mma_ss128(float (&d)[64],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %66, 0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 1, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]) : "l"(a),"l"(b),"r"(accumulate));
}
#ifndef PART_ONLY
#define PART_ONLY 0
#endif
#ifndef UCOUNT
#define UCOUNT 132
#endif
TMN_DEVI void allsync(){named_bar_sync(0,256);}
// slot layout (32 rows): dy 0..8K (2 x 4KiB col blocks), gate 8..16K, proj 16..24K, xn 24..32K, norm 32..48K, tri 48..64K
TMN_DEVI void issue(const Params& p,uint8_t* sm,uint64_t* full,int slot,int tile){
 uint8_t* s=sm+slot*65536;int row=tile*32;
#pragma unroll
 for(int c=0;c<2;++c){tma_load_2d(s+c*4096,&p.dy32,full+slot,c*64,row);tma_load_2d(s+8192+c*4096,&p.gate32,full+slot,c*64,row);tma_load_2d(s+16384+c*4096,&p.proj32,full+slot,c*64,row);tma_load_2d(s+24576+c*4096,&p.xn32,full+slot,c*64,row);}
#pragma unroll
 for(int c=0;c<4;++c)tma_load_2d(s+32768+c*4096,&p.norm32,full+slot,c*64,row);
 tma_load_2d(s+49152,&p.tri32,full+slot,row,0);
}
extern "C" __global__ __launch_bounds__(256,1)
void dual_b1b4(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t full[3];
 const int ntiles=2*p.tiles,split=blockIdx.x,wi=threadIdx.x/128,tid=threadIdx.x%128;
 if(threadIdx.x==0){mbar_init(full,1);mbar_init(full+1,1);mbar_init(full+2,1);fence_barrier_init();
  for(int q=0;q<3;++q)if(split+q*UCOUNT<ntiles){mbar_arrive_expect_tx(full+q,65536);issue(p,sm,full,q,split+q*UCOUNT);}}
 allsync();
 int round=0;
 for(int it=split;it<ntiles;it+=UCOUNT,++round){int slot=round%3;uint32_t par=(round/3)&1;
  mbar_wait(full+slot,par);uint8_t* s=sm+slot*65536;int row=it*32;
  for(int i=threadIdx.x;i<512;i+=256){int cb=i/256,r=(i%256)/8,c=(i%8)*8;uint4 y=lds128(smem_u32(s+cb*4096)+swz128(r,c*2));stg128(p.dg+(size_t)(row+r)*128+cb*64+c,y);}
  fence_proxy_async();sync_group();
  if(tid==0){for(int ch=wi*128;ch<(wi+1)*128;ch+=16)tma_store_3d(&p.dtri32,s+49152+ch*64,row,ch,0);tma_store_commit();tma_store_wait_all();}
  sync_group();
  bool next=it+3*UCOUNT<ntiles;
  if(next&&threadIdx.x==0)mbar_arrive_expect_tx(full+slot,65536);
  allsync();
  if(next&&threadIdx.x==0)issue(p,sm,full,slot,it+3*UCOUNT);
 }
}
extern "C" __global__ void unified_reduce(__grid_constant__ const Params p){}
