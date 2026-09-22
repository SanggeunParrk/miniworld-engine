// Experimental single-CTA ownership of B1, dX/LN and ALL dW tiles.
// Preserves FP32 dW accumulation and BF16 intermediate rounding.
#include "fused.cu"
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
TMN_DEVI void udgrad(const Params& p,uint8_t* sm,uint64_t* bar,int m0){
 const int tid=threadIdx.x,lane=tid%32,w=tid/32;uint8_t* sw=sm+131072,*sn=sm+147456,*sx=sm+98304;
#pragma unroll
 for(int qn=0;qn<4;++qn){int n=(qn+3)%4;sw=sm+(n==0?131072:n==1?65536:n==2?81920:147456);
  float acc[32]={};fence_regs(acc);wgmma_fence();
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_dgrad(acc,smem_desc(smem_u32(sm+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(acc);
#pragma unroll
  for(int q=0;q<4;++q){int mat=lane/8,rr=lane%8+8*(mat&1),cc=q*16+8*(mat>>1);stsm_x4(smem_u32(sn+n*8192)+swz128(w*16+rr,cc*2),pack_bf16(acc[q*8],acc[q*8+1]),pack_bf16(acc[q*8+2],acc[q*8+3]),pack_bf16(acc[q*8+4],acc[q*8+5]),pack_bf16(acc[q*8+6],acc[q*8+7]));}
  fence_proxy_async();sync_group();
 }
 float dga[8]={},dbe[8]={};
 for(int r=w;r<64;r+=4){float mu=p.mean[m0+r],rs=p.rs[m0+r];float dn[8],xh[8],dh[8],s1=0,s2=0;
#pragma unroll
  for(int k=0;k<8;++k){int c=lane+32*k;dn[k]=get(sn+(c/64)*8192,r,c%64);xh[k]=__fmul_rn(__fsub_rn(get(sx,c,r),mu),rs);dh[k]=dn[k]*p.gamma[c];s1+=dh[k]*xh[k];s2+=dh[k];dga[k]+=dn[k]*xh[k];dbe[k]+=dn[k];}
  s1=warp_sum(s1)/256.f;s2=warp_sum(s2)/256.f;
#pragma unroll
  for(int k=0;k<8;++k){int c=lane+32*k;put(sx,c,r,rs*((dh[k]-s2)-xh[k]*s1));}
 }
 sync_group();fence_proxy_async();sync_group();if(tid==0){for(int c=0;c<256;c+=16)tma_store_3d(&p.dtri,sx+c*128,m0,c,0);tma_store_commit();tma_store_wait_all();}
 float* red=reinterpret_cast<float*>(sm+180224);
#pragma unroll
 for(int k=0;k<8;++k){red[w*512+lane+32*k]+=dga[k];red[w*512+256+lane+32*k]+=dbe[k];}
 sync_group();
}

#include "dgrad_stats.cuh"
TMN_DEVI void flat_load(const Params& p,uint8_t* sm,uint64_t* bar,int row,int phase,bool wp,bool first){
 if(threadIdx.x==0){mbar_arrive_expect_tx(bar,wp?65536:(first?163840:114688));
#pragma unroll
  for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.dy,bar,c*64,row);tma_load_2d(sm+16384+c*8192,&p.gate,bar,c*64,row);
   if(!wp){tma_load_2d(sm+32768+c*8192,&p.proj,bar,c*64,row);tma_load_2d(sm+49152+c*8192,&p.xn,bar,c*64,row);}}
  if(wp){for(int c=0;c<4;++c)tma_load_2d(sm+65536+c*8192,&p.norm,bar,c*64,row);}
  else{tma_load_2d(sm+98304,&p.tri,bar,row,0);
#pragma unroll
   for(int n=0;n<4;++n){if(first||n==3){uint8_t* sw=sm+(n==0?131072:n==1?65536:n==2?81920:147456);for(int k=0;k<2;++k)tma_load_2d(sw+k*8192,&p.wp,bar,k*64,n*64);}}
  }
 }
 allsync();mbar_wait(bar,phase);int j0=row%p.L;
 for(int i=threadIdx.x;i<4096;i+=256){int c=i/2048*64+2*(i%32),rr=i%2048/32,jr=j0+rr;if(jr>=p.L)jr-=p.L;uint8_t* sy=sm+(c/64)*8192,*sg=sm+16384+(c/64)*8192,*sp=sm+32768+(c/64)*8192;
  uint32_t y=pair_get(sy,rr,c%64),g=pair_get(sg,rr,c%64),ds=ldg32(p.ds+jr*128+c);float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);
  if(!wp){uint32_t v=pair_get(sp,rr,c%64);uint32_t vdg=pack_bf16(((ya*bf16lo(v))*ga)*(1.f-ga),((yb*bf16hi(v))*gb)*(1.f-gb));pair_put(sg,rr,c%64,vdg);stg32(p.dg+(size_t)(row+rr)*128+c,vdg);}
  pair_put(sy,rr,c%64,pack_bf16(ya*ga,yb*gb));
 }
 fence_proxy_async();allsync();
}
extern "C" __global__ __launch_bounds__(256,1) void flat_b1b4(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar;
 const int split=blockIdx.x/2,role=blockIdx.x%2,wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32;
 if(threadIdx.x==0){mbar_init(&bar,1);fence_barrier_init();}
 for(int i=threadIdx.x;i<2048;i+=256)reinterpret_cast<float*>(sm+180224)[i]=0;allsync();
 int first=p.tiles*split/UCOUNT,end=p.tiles*(split+1)/UCOUNT,phase=0;
 float acc[2][64]={};
 for(int it=first;it<end;++it){flat_load(p,sm,&bar,it*64,phase,role==1,it==first);phase^=1;
  if(role==0&&wi==0){udgrad_stats(p,sm,&bar,it*64);}else{
   static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;uint8_t* sa=role==0?sm+49152+n*8192:sm+wi*8192;uint8_t* sb=role==0?sm+16384:sm+65536+n*16384;
    fence_regs(acc[n]);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc[n],smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),it>first||k>0);});wgmma_commit();
   });wgmma_wait<0>();fence_regs(acc[0]);fence_regs(acc[1]);
  }
  allsync();
 }
 if(role==0&&wi==0){float* red=reinterpret_cast<float*>(sm+180224);for(int j=tid;j<512;j+=128){float v=0;for(int w=0;w<4;++w)v+=red[w*512+j];p.partln[split*512+j]=v;}}
 else{
  int tile=role==0?0:1+wi;float* part=p.partw+(tile*UCOUNT+split)*16384;
  static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;
#pragma unroll
   for(int q=0;q<16;++q){int rr=w*16+lane/4+(role==0?n*64:0),c=q*8+2*(lane%4)+(role==0?0:n*128),stride=role==0?128:256;part[rr*stride+c]=acc[n][4*q];part[rr*stride+c+1]=acc[n][4*q+1];part[(rr+8)*stride+c]=acc[n][4*q+2];part[(rr+8)*stride+c+1]=acc[n][4*q+3];}
  });
 }
}
extern "C" __global__ void unified_reduce(__grid_constant__ const Params p){
 int i=blockIdx.x*256+threadIdx.x;
 if(i<49152){int tile=i/16384,j=i%16384;float v=0;for(int b=0;b<UCOUNT;++b)v+=p.partw[(tile*UCOUNT+b)*16384+j];(tile==0?p.dwg:p.dwp+(tile-1)*16384)[j]=__float2bfloat16_rn(v);}
 else if(i<49664){int j=i-49152;float v=0;for(int b=0;b<UCOUNT;++b)v+=p.partln[b*512+j];(j<256?p.dgam:p.dbeta)[j%256]=v;}
}
