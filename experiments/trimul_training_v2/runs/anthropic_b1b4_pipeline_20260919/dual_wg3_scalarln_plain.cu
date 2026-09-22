// 3 warpgroups:128 dW registers/thread,12 warps/CTA; same64-row fusion.
#define CTA_THREADS 384
// One variable on dual_maskbits: dnorm output tile N32.
// One variable on dual_cyclic: lossless reuse of periodic dropout masks.
// One variable on dual_vec2: cyclic rather than contiguous row tile ownership.
// One-variable experiment: pair adjacent FP32 partial stores using Anthropic stg64f.
// No layout, reduction order, arithmetic, TMA schedule or buffer change.
// Experimental single-CTA ownership of B1, dX/LN and ALL dW tiles.
// Preserves FP32 dW accumulation and BF16 intermediate rounding.
#include "dual_primitives.cuh"
#ifndef PART_ONLY
#define PART_ONLY 0
#endif
#ifndef UCOUNT
#define UCOUNT 132
#endif
TMN_DEVI void allsync(){named_bar_sync(0,384);}
// N32 has the same K order and BF16 result, with 16 rather than 32 FP32
// temporary accumulators. This trades more MMA issue/waits for register space.
TMN_DEVI void mma_dgrad(float (&d)[16],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %18, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]) : "l"(a),"l"(b),"r"(accumulate));
}
// LN row reductions in the GEMM register layout; channel-contiguous epilogue.
// Materialize BF16 dnorm, then reduce contiguous token pairs with12 warps.
// Extra32KiB shared read buys lower peak register pressure for384 threads.
TMN_DEVI void dual_dgrad(const Params& p,uint8_t* sm,uint64_t* bar,int m0,bool next){
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32,mat=lane/8,r8=lane%8;
 uint8_t* sn=sm+196608;uint8_t* sx=sm+98304;float* stats=reinterpret_cast<float*>(sm+32768);
 float* mus=stats+1536,*rss=mus+64,*gam=rss+64;
 if(threadIdx.x<64){mus[tid]=p.mean[m0+tid];rss[tid]=p.rs[m0+tid];}
 if(threadIdx.x<256)gam[threadIdx.x]=p.gamma[threadIdx.x];
 allsync(); // Publish saved row statistics and gamma; dW operands are complete.
#pragma unroll
 for(int qn=0;qn<3;++qn){int n=wi+qn*3;if(n>=8)continue;
  uint8_t* sw=sm+131072+(n/2)*16384+(n%2)*4096;
  float acc[16]={};fence_regs(acc);wgmma_fence();
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_dgrad(acc,smem_desc(smem_u32(sm+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(acc); // Ready before BF16 packing.
#pragma unroll
  for(int q=0;q<2;++q){uint32_t dn[4];
#pragma unroll
   for(int j=0;j<4;++j)dn[j]=pack_bf16(acc[q*8+j*2],acc[q*8+j*2+1]);
   stsm_x4_t(smem_u32(sn)+swz128(n*32+q*16+8*(mat>>1)+r8,(w*16+8*(mat&1))*2),dn[0],dn[1],dn[2],dn[3]);
  }
 }
 allsync(); // Publish dnorm from all three WGs before any LN row reads.
 // WGMMA has released dy/gate/xn/norm. Prefetch80KiB while all LN work runs.
 if(next&&threadIdx.x==0){mbar_arrive_expect_tx(bar,131072);
#pragma unroll
  for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.dy,bar,c*64,m0+64*UCOUNT);tma_load_2d(sm+16384+c*8192,&p.gate,bar,c*64,m0+64*UCOUNT);tma_load_2d(sm+49152+c*8192,&p.xn,bar,c*64,m0+64*UCOUNT);}
#pragma unroll
  for(int c=0;c<4;++c)tma_load_2d(sm+65536+c*8192,&p.norm,bar,c*64,m0+64*UCOUNT);
 }
 int row=2*lane,cg=threadIdx.x/32;
 float mu0=mus[row],mu1=mus[row+1],rs0=rss[row],rs1=rss[row+1];
 float a=0,b=0,aa=0,bb=0;
#pragma unroll 1
 for(int c=cg;c<256;c+=12){uint32_t x=pair_get(sx,c,row),dn=pair_get(sn,c,row);float g=gam[c];
  float xa=__fmul_rn(__fsub_rn(bf16lo(x),mu0),rs0),xb=__fmul_rn(__fsub_rn(bf16hi(x),mu1),rs1);
  float ha=bf16lo(dn)*g,hb=bf16hi(dn)*g;a+=ha*xa;b+=hb*xb;aa+=ha;bb+=hb;
 }
 stats[cg*128+row*2]=a;stats[cg*128+row*2+1]=aa;
 stats[cg*128+row*2+2]=b;stats[cg*128+row*2+3]=bb;
 allsync(); // Publish12 channel partials for each of64 rows.
 if(threadIdx.x<64){int rr=threadIdx.x;float v1=0,v2=0;
#pragma unroll 1
  for(int k=0;k<12;++k){v1+=stats[k*128+rr*2];v2+=stats[k*128+rr*2+1];}
  // Each owner overwrites only its own row's consumed partials.
  stats[rr*2]=v1/256.f;stats[rr*2+1]=v2/256.f;
  stats[128+rr*2]=0;stats[128+rr*2+1]=0;stats[256+rr*2]=0;stats[256+rr*2+1]=0;
 }
 allsync(); // Publish combined full-channel row statistics.
 float* red=reinterpret_cast<float*>(sm+229376);
#pragma unroll 1
 for(int b=0;b<6;++b){int c=wi*16+b*48+w*4+lane/8;if(c>=256)continue;float dg=0,db=0,gamma=gam[c];
#pragma unroll 1
  for(int k=0;k<4;++k){int r=2*(lane%8)+16*k;uint32_t x=pair_get(sx,c,r),dn=pair_get(sn,c,r);float mua=mus[r],mub=mus[r+1],rsa=rss[r],rsb=rss[r+1];float xa=__fmul_rn(__fsub_rn(bf16lo(x),mua),rsa),xb=__fmul_rn(__fsub_rn(bf16hi(x),mub),rsb),da=bf16lo(dn),dd=bf16hi(dn);dg+=da*xa+dd*xb;db+=da+dd;pair_put(sx,c,r,pack_bf16(rsa*((da*gamma-stats[r*2+1])-xa*stats[r*2]),rsb*((dd*gamma-stats[(r+1)*2+1])-xb*stats[(r+1)*2])));}
#pragma unroll
  for(int sh=1;sh<8;sh*=2){dg+=__shfl_xor_sync(0xffffffff,dg,sh);db+=__shfl_xor_sync(0xffffffff,db,sh);}
  if(lane%8==0){red[c]+=dg;red[256+c]+=db;}
 }
 // All generic writes precede the TMA proxy reads; caller joins stores.
 sync_group();fence_proxy_async();sync_group();if(tid==0){for(int c=wi*16;c<256;c+=48)tma_store_3d(&p.dtri,sx+c*128,m0,c,0);tma_store_commit();tma_store_wait_all();}sync_group();
}

TMN_DEVI void dual_load(const Params& p,uint8_t* sm,uint64_t* bar,int row,int phase,bool first,uint32_t mask_bits,uint32_t mask_scale,bool mask_reuse){
 if(threadIdx.x==0){if(first)mbar_arrive_expect_tx(bar,196608);
#pragma unroll
 for(int c=0;c<2;++c){if(first){tma_load_2d(sm+c*8192,&p.dy,bar,c*64,row);tma_load_2d(sm+16384+c*8192,&p.gate,bar,c*64,row);}tma_load_2d(sm+32768+c*8192,&p.proj,bar,c*64,row);if(first)tma_load_2d(sm+49152+c*8192,&p.xn,bar,c*64,row);}
#pragma unroll
 if(first)for(int c=0;c<4;++c)tma_load_2d(sm+65536+c*8192,&p.norm,bar,c*64,row);
 tma_load_2d(sm+98304,&p.tri,bar,row,0);
 if(first)for(int n=0;n<4;++n)for(int k=0;k<2;++k)tma_load_2d(sm+131072+n*16384+k*8192,&p.wp,bar,k*64,n*64);
 }
 allsync();mbar_wait(bar,phase);int j0=row%p.L;
 // Pairwise contiguous lanes minimize B1 temporaries under the168-register cap.
 for(int i=threadIdx.x;i<4096;i+=384){int cb=i/2048,r=(i%2048)/32,c=2*(i%32),jr=j0+r;if(jr>=p.L)jr-=p.L;
  uint32_t sy=smem_u32(sm+cb*8192)+swz128(r,c*2),sg=smem_u32(sm+16384+cb*8192)+swz128(r,c*2),sp=smem_u32(sm+32768+cb*8192)+swz128(r,c*2);
  uint32_t yy,gg,vv,dd;
  asm volatile("ld.shared.b32 %0,[%1];":"=r"(yy):"r"(sy));
  asm volatile("ld.shared.b32 %0,[%1];":"=r"(gg):"r"(sg));
  asm volatile("ld.shared.b32 %0,[%1];":"=r"(vv):"r"(sp));
  if(mask_reuse){int bit=2*(i/384);dd=((mask_bits>>bit)&1?mask_scale:0)|((mask_bits>>(bit+1))&1?mask_scale<<16:0);}
  else dd=ldg32(p.ds+jr*128+cb*64+c);
  float ya=bf16lo(yy)*bf16lo(dd),yb=bf16hi(yy)*bf16hi(dd),ga=bf16lo(gg),gb=bf16hi(gg);
  uint32_t dg=pack_bf16(((ya*bf16lo(vv))*ga)*(1.f-ga),((yb*bf16hi(vv))*gb)*(1.f-gb));
  uint32_t dp=pack_bf16(ya*ga,yb*gb);
  sts32(sy,dp);sts32(sg,dg);stg32(p.dg+(size_t)(row+r)*128+cb*64+c,dg);
 }
 fence_proxy_async();allsync();
}
extern "C" __global__ __launch_bounds__(384,1) void dual_b1b4(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar;
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32,split=blockIdx.x;
 if(threadIdx.x==0){mbar_init(&bar,1);fence_barrier_init();}
 for(int i=threadIdx.x;i<512;i+=384)reinterpret_cast<float*>(sm+229376)[i]=0;allsync();
 // Cyclic CTA ownership preserves all rows and handles any ragged/empty tail.
 int first=split,end=p.tiles,phase=0;
 // ds is a dropout mask (zero or one common positive BF16 scale). If the
 // cyclic row stride wraps exactly, each thread reuses its same 32 entries.
 // Lossless binary encoding uses two registers, not 16 mask registers. Other
 // lengths retain the ordinary ds loads, without any benchmark-shape branch.
 const bool mask_reuse=false;
 uint32_t mask_bits=0,mask_scale=0;
 if(mask_reuse&&first<end){int j0=(first*64)%p.L;
  for(int i=threadIdx.x;i<4096;i+=384){int cb=i/2048,r=(i%2048)/32,c=2*(i%32),jr=j0+r;if(jr>=p.L)jr-=p.L;
   uint32_t ds=ldg32(p.ds+jr*128+cb*64+c),lo=ds&65535u,hi=ds>>16;int bit=2*(i/384);
   if(lo){mask_bits|=1u<<bit;mask_scale=lo;}if(hi){mask_bits|=1u<<(bit+1);mask_scale=hi;}
  }
 }
 float acc[2][64]={};
 for(int it=first;it<end;it+=UCOUNT){dual_load(p,sm,&bar,it*64,phase,it==first,mask_bits,mask_scale,mask_reuse);phase^=1;
  static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*2+n;uint8_t* sa=t<2?sm+49152+t*8192:sm+((t-2)/2)*8192;uint8_t* sb=t<2?sm+16384:sm+65536+((t-2)%2)*16384;fence_regs(acc[n]);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc[n],smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),it>first||k>0);});wgmma_commit();
  });wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);
  dual_dgrad(p,sm,&bar,it*64,it+UCOUNT<end);allsync();
 }
 static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*2+n,tile=t<2?0:1+(t-2)/2;float* part=p.partw+(tile*UCOUNT+split)*16384;
#pragma unroll
 for(int q=0;q<16;++q){int rr=w*16+lane/4+(t<2?t*64:0),c=q*8+2*(lane%4)+(t<2?0:((t-2)%2)*128),stride=t<2?128:256;stg64f(part+rr*stride+c,acc[n][4*q],acc[n][4*q+1]);stg64f(part+(rr+8)*stride+c,acc[n][4*q+2],acc[n][4*q+3]);}});
 for(int j=threadIdx.x;j<512;j+=384)p.partln[split*512+j]=reinterpret_cast<float*>(sm+229376)[j];
#if PART_ONLY == 2
 __threadfence();allsync();
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}
 allsync();
 for(int i=blockIdx.x*384+threadIdx.x;i<49664;i+=UCOUNT*384){
  if(i<49152){int tile=i/16384,j=i%16384;float v=0;for(int b=0;b<UCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*UCOUNT+b)*16384+j];(tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);}
  else{int j=i-49152;float v=0;for(int b=0;b<UCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
 }
 __threadfence();allsync();
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif

}
extern "C" __global__ void unified_reduce(__grid_constant__ const Params p){
 int i=blockIdx.x*256+threadIdx.x;
 if(i<49152){int tile=i/16384,j=i%16384;float v=0;for(int b=0;b<UCOUNT;++b)v+=p.partw[(tile*UCOUNT+b)*16384+j];(tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);}
 else if(i<49664){int j=i-49152;float v=0;for(int b=0;b<UCOUNT;++b)v+=p.partln[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
}
