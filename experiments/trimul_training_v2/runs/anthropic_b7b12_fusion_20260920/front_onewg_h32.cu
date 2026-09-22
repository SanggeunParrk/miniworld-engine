#define WGRAD_SLICES 2
// SPDX-License-Identifier: Apache-2.0
// Training extension of Anthropic v5: two GLU/TMA producer WGs, one MMA WG.
#include "warp_primitives.cuh"
#ifndef UCOUNT
#define UCOUNT 132
#endif
#ifndef DW_SPLITS
#define DW_SPLITS 8
#endif
#ifndef PART_ONLY
#define PART_ONLY 2
#endif
#ifndef DW_PREFETCH
#define DW_PREFETCH 3
#endif
constexpr int DWCOUNT=16*DW_SPLITS,DXCOUNT=UCOUNT-DWCOUNT;
constexpr int DW_SLOT=36864,DX_SLOT=20480;
TMN_DEVI void sync128(){named_bar_sync(0,128);}
#define allsync sync128
static_assert(DXCOUNT>0,"invalid partition");
template<int AO,int BO> TMN_DEVI void mma_weight_off(float (&d)[64],uint32_t al,uint32_t ah,uint32_t bl,uint32_t bh,int scale){
 asm volatile("{.reg .pred p;.reg .b32 la,lb;.reg .b64 ad,bd;setp.ne.b32 p,%68,0;add.u32 la,%64,%69;add.u32 lb,%66,%70;mov.b64 ad,{la,%65};mov.b64 bd,{lb,%67};wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},ad,bd,p,1,1,0,1;}" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]) : "r"(al),"r"(ah),"r"(bl),"r"(bh),"r"(scale),"n"(AO>>4),"n"(BO>>4));
}
template<int OFF> TMN_DEVI void store_off(float* base,float a,float b){asm volatile("{.reg .b64 p;add.u64 p,%0,%3;st.global.v2.f32 [p],{%1,%2};}"::"l"(base),"f"(a),"f"(b),"n"(OFF):"memory");}
struct Params{
 CUtensorMap dl,dr,pre,xn,dg,wlg,wl,wrg,wr,wgate,x,res,dx;
 const __nv_bfloat16* mask;const float *mean,*rs,*gamma;
 __nv_bfloat16* dw;float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;__nv_bfloat16 *debugdc,*debugxn;int M,L,tiles;
};
TMN_DEVI void mma_weight64(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
TMN_DEVI void store2d(const CUtensorMap* map,const void* src,int c,int r){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2,%3}], [%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");
}

TMN_DEVI void mma_w32(float (&d)[16],uint64_t a,uint64_t b,int scale){asm volatile("{.reg .pred p;setp.ne.b32 p,%18,0;wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},%16,%17,p,1,1,1,0;}" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]) : "l"(a),"l"(b),"r"(scale));}

TMN_DEVI void glu32(const Params& p,uint8_t* s,uint8_t* g,uint8_t* pp,int row){
 unsigned tid=threadIdx.x;uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2);
 #pragma unroll 2
 for(unsigned q=0;q<8;++q){unsigned i=tid+q*128,c=i/32,r=(i%32)*2;uint32_t dy=pair_get(s+8192,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r),masked;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(mask));float ga=math::sigmoid(bf16lo(gl)),gb=math::sigmoid(bf16hi(gl));*reinterpret_cast<uint32_t*>(g+swz128(c,r*2))=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb));*reinterpret_cast<uint32_t*>(pp+swz128(c,r*2))=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);}
 fence_proxy_async();allsync();
}
TMN_DEVI void load_dw32(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int group){if(threadIdx.x)return;uint8_t* s=sm+slot*DW_SLOT;int side=group/8,h=(group%8)*32;mbar_arrive_expect_tx(b+slot,28672);tma_load_2d(s,&p.pre,b+slot,row,side*512+h*2);tma_load_2d(s+8192,side?&p.dr:&p.dl,b+slot,row,h);for(int c=0;c<2;++c)tma_load_2d(s+12288+c*8192,&p.xn,b+slot,c*64,row);}
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* bar){
 int group=blockIdx.x%16,split=blockIdx.x/16,tid=threadIdx.x,lane=tid%32,w=tid/32,r=0;int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,seg=(rounds+3)/4;float acc[4][16]={};
 // A short validation shape still writes all four segments, including empty ones.
 for(int i=tid;i<4*8192;i+=128)p.partw[(group*DW_SPLITS+split)*4*8192+i]=0.f;
 if(split<p.tiles)load_dw32(p,sm,bar,0,split*64,group);if(split+DW_SPLITS<p.tiles)load_dw32(p,sm,bar,1,(split+DW_SPLITS)*64,group);
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){int slot=r&1;uint8_t* s=sm+slot*DW_SLOT;mbar_wait(bar+slot,(r/2)&1);glu32(p,s,s+28672,s+32768,tile*64);
  static_for<4>([&](auto qi){constexpr int part=decltype(qi)::value;constexpr int kind=part/2,cb=part%2;fence_regs(acc[part]);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_w32(acc[part],smem_desc(smem_u32(s+12288+cb*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+28672+kind*4096+k*32),16,1024,1),r%seg>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc[part]);});allsync();if(tile+2*DW_SPLITS<p.tiles)load_dw32(p,sm,bar,slot,(tile+2*DW_SPLITS)*64,group);
  if((r+1)%seg==0||tile+DW_SPLITS>=p.tiles){static_for<4>([&](auto qi){constexpr int part=decltype(qi)::value;float* out=p.partw+((group*DW_SPLITS+split)*4+r/seg)*8192+(part/2)*4096+(part%2)*2048;static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;int rr=w*16+lane/4,c=j*8+2*(lane%4);stg64f(out+rr*32+c,acc[part][j*4],acc[part][j*4+1]);stg64f(out+(rr+8)*32+c,acc[part][j*4+2],acc[part][j*4+3]);});});}
 }
}
TMN_DEVI void load_g32(const Params& p,uint8_t* sm,uint64_t* bar,int row,int side,int h){if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(bar+slot,20480);tma_load_2d(s,&p.pre,bar+slot,row,side*512+h*64);tma_load_2d(s+8192,side?&p.dr:&p.dl,bar+slot,row,h*32);for(int c=0;c<2;++c)tma_load_2d(s+12288+c*4096,side?&p.wrg:&p.wlg,bar+slot,h*32,c*64);}
TMN_DEVI void load_p32(const Params& p,uint8_t* sm,uint64_t* bar,int side,int h){if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(bar+slot,8192);for(int c=0;c<2;++c)tma_load_2d(s+c*4096,side?&p.wr:&p.wl,bar+slot,h*32,c*64);}
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma){
 int split=blockIdx.x-DWCOUNT,tid=threadIdx.x,lane=tid%32,w=tid/32,ra=w*16+lane/4,rb=ra+8,round=0;float running_g=0,running_b=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int row=tile*64,ph=round&1;uint32_t packed[32];float acc[64]={};
  if(tid==0){mbar_arrive_expect_tx(bar+2,49152);for(int k=0;k<2;++k){tma_load_2d(sm+k*8192,&p.dg,bar+2,k*64,row);for(int c=0;c<2;++c)tma_load_2d(sm+16384+k*16384+c*8192,&p.wgate,bar+2,k*64,c*64);}}
  mbar_wait(bar+2,ph);{float gate[64]={};fence_regs(gate);wgmma_fence();static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_gate128(gate,smem_desc(smem_u32(sm+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+16384+(k/4)*16384+(k%4)*32),16,1024,1),k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(gate);static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;packed[q]=pack_bf16(gate[q*2],gate[q*2+1]);});}allsync();
  for(int side=0;side<2;++side){load_g32(p,sm,bar,row,side,0);load_g32(p,sm,bar,row,side,1);
   for(int h=0;h<8;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,(h/2)&1);glu32(p,s,s+8192,sm+40960+h*4096,row);fence_regs(acc);wgmma_fence();static_for<2>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(s+8192+k*2048),16,1024,1),smem_desc(smem_u32(s+12288+k*32),16,512,2),side>0||h>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h+2<8)load_g32(p,sm,bar,row,side,h+2);}
   load_p32(p,sm,bar,side,0);load_p32(p,sm,bar,side,1);
   for(int h=0;h<8;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,(h/2)&1);fence_regs(acc);wgmma_fence();static_for<2>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(sm+40960+h*4096+k*2048),16,1024,1),smem_desc(smem_u32(s+k*32),16,512,2),1);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();if(h+2<8)load_p32(p,sm,bar,side,h+2);}
  }
  static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;acc[q*2]=__bfloat162float(__float2bfloat16_rn(acc[q*2]+bf16lo(packed[q])));acc[q*2+1]=__bfloat162float(__float2bfloat16_rn(acc[q*2+1]+bf16hi(packed[q])));});
  if(tid==0){mbar_arrive_expect_tx(bar+3,32768);for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.x,bar+3,c*64,row);tma_load_2d(sm+16384+c*8192,&p.res,bar+3,c*64,row);}}mbar_wait(bar+3,ph);
  float mu[2]={p.mean[row+ra],p.mean[row+rb]},rs[2]={p.rs[row+ra],p.rs[row+rb]},s1[2]={},s2[2]={};
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   static_for<2>([&](auto ri){constexpr int r=decltype(ri)::value;int rr=r?rb:ra;
    float xa=__fmul_rn(__fsub_rn(get(sm+base,rr,cc),mu[r]),rs[r]),xb=__fmul_rn(__fsub_rn(get(sm+base,rr,cc+1),mu[r]),rs[r]);
    float ha=acc[q*4+r*2]*gamma[c],hb=acc[q*4+r*2+1]*gamma[c+1];s1[r]+=ha*xa+hb*xb;s2[r]+=ha+hb;
   });
  });
  float c1[2]={quad_sum(s1[0])/128.f,quad_sum(s1[1])/128.f},c2[2]={quad_sum(s2[0])/128.f,quad_sum(s2[1])/128.f};float* tmp=reinterpret_cast<float*>(sm+32768);
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   float xaa=__fmul_rn(__fsub_rn(get(sm+base,ra,cc),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(get(sm+base,ra,cc+1),mu[0]),rs[0]);
   float xba=__fmul_rn(__fsub_rn(get(sm+base,rb,cc),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(get(sm+base,rb,cc+1),mu[1]),rs[1]);
   float da=acc[q*4],db=acc[q*4+1],dc=acc[q*4+2],dd=acc[q*4+3],ga=gamma[c],gb=gamma[c+1];
   uint32_t outa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
   uint32_t outb=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
   put(sm+base,ra,cc,bf16lo(outa)+get(sm+16384+base,ra,cc));put(sm+base,ra,cc+1,bf16hi(outa)+get(sm+16384+base,ra,cc+1));
   put(sm+base,rb,cc,bf16lo(outb)+get(sm+16384+base,rb,cc));put(sm+base,rb,cc+1,bf16hi(outb)+get(sm+16384+base,rb,cc+1));
   float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
   for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
   if(lane<4){tmp[w*256+c]=dga;tmp[w*256+c+1]=dgb;tmp[w*256+128+c]=dba;tmp[w*256+128+c+1]=dbb;}
  });
  allsync();running_g+=(tmp[tid]+tmp[256+tid])+(tmp[512+tid]+tmp[768+tid]);running_b+=(tmp[128+tid]+tmp[384+tid])+(tmp[640+tid]+tmp[896+tid]);
  fence_proxy_async();allsync();if(tid==0){store2d(&p.dx,sm,0,row);store2d(&p.dx,sm+8192,64,row);tma_store_commit();tma_store_wait_all();}allsync();
 }
 p.partln[split*256+tid]=running_g;p.partln[split*256+128+tid]=running_b;
}

TMN_DEVI void reduce_at(const Params& p,int i){if(i<131072){int group=i/8192,j=i%8192,kind=j/4096,z=j%4096;float v=0;for(int q=0;q<DW_SPLITS*4;++q)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*4+q)*8192+j];int out=(group/8)*2+(kind==0?1:0),c=z/32,h=(group%8)*32+z%32;p.dw[(out*128+c)*256+h]=__float2bfloat16_rn(v);}else if(i<131328){int c=i-131072;float v=0;for(int q=0;q<DXCOUNT;++q)v+=reinterpret_cast<volatile float*>(p.partln)[q*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}}
extern "C" __global__ __launch_bounds__(128,3) void front_b7b12(__grid_constant__ const Params p){extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[4];__shared__ float gamma[128];gamma[threadIdx.x]=p.gamma[threadIdx.x];if(threadIdx.x==0){for(int i=0;i<4;++i)mbar_init(bar+i,1);fence_barrier_init();}allsync();if(blockIdx.x<DWCOUNT)weight_role(p,sm,bar);else input_role(p,sm,bar,gamma);__threadfence();allsync();if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();for(int i=blockIdx.x*128+threadIdx.x;i<131328;i+=UCOUNT*128)reduce_at(p,i);__threadfence();allsync();if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
