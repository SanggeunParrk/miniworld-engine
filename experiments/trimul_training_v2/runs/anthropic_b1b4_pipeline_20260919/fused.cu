// SPDX-License-Identifier: Apache-2.0
// Anthropic native v5 f4f62fa: TMA/ldmatrix/WGMMA RS primitives and fragment layouts.
// New backward: gate/dropout + dgrad/LN backward + split-K wgrad, one launch.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#ifndef WSTAGES
#define WSTAGES 2
#endif
#ifndef WSS
#define WSS 0
#endif
#ifndef DFAST
#define DFAST 0
#endif
#ifndef PREFETCH
#define PREFETCH 0
#endif
#ifndef LNFRAG
#define LNFRAG 0
#endif
#ifndef FULLW
#define FULLW 0
#endif
#ifndef WGROUPS
#define WGROUPS 1
#endif
#ifndef COMPACT
#define COMPACT 0
#endif
#ifndef ACCS
#define ACCS 2
#endif
#ifndef SPLITS
#define SPLITS 64
#endif
#ifndef RED_GROUP
#define RED_GROUP 32
#endif
struct Params {
 CUtensorMap dy,gate,proj,xn,norm,tri,wp,dtri;
 const __nv_bfloat16* ds;
 const float *mean,*rs,*gamma;
 __nv_bfloat16 *dg,*dwg,*dwp;
 float *dgam,*dbeta,*partw,*partln,*groupln;
 unsigned int *counts;
 int M,L,tiles,groups;
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
TMN_DEVI void ln_finish(const Params& p,float* red,int* last){
 int tid=threadIdx.x%128,tile=blockIdx.x,group=tile/RED_GROUP;
 for(int j=tid;j<512;j+=128){float v=0;for(int w=0;w<4;++w)v+=red[w*512+j];p.partln[tile*512+j]=v;}
 unsigned int total=min(RED_GROUP,p.tiles-group*RED_GROUP);
 if(ticket(p.counts+12+group,total,last)){
  for(int j=tid;j<512;j+=128){float v=0;for(int t=0;t<total;++t)v+=reinterpret_cast<volatile float*>(p.partln)[(group*RED_GROUP+t)*512+j];p.groupln[group*512+j]=v;}
  sync_group();if(tid==0)atomicExch(p.counts+12+group,0u);
  if(ticket(p.counts+12+p.groups,p.groups,last)){
   for(int j=tid;j<512;j+=128){float v=0;for(int g=0;g<p.groups;++g)v+=reinterpret_cast<volatile float*>(p.groupln)[g*512+j];if(j<256)p.dgam[j]=v;else p.dbeta[j-256]=v;}
   sync_group();if(tid==0)atomicExch(p.counts+12+p.groups,0u);
  }
 }
}
TMN_DEVI void dgrad(const Params& p,uint8_t* sm,uint64_t* bar,int* last){
 const int tid=threadIdx.x%128,lane=tid%32,w=tid/32,m0=blockIdx.x*64;
 uint8_t *sy=sm,*sg=sm+(COMPACT?8192:16384),*sp=sm+(COMPACT?16384:32768),*sw=sm+(COMPACT?32768:49152),*sx=sm+(COMPACT?49152:114688);
#if COMPACT
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}sync_group();
 uint32_t a[8][4];int j0=m0%p.L;
#pragma unroll
 for(int kc=0;kc<2;++kc){
  if(tid==0){mbar_arrive_expect_tx(bar,24576+(kc==0?49152:0));tma_load_2d(sy,&p.dy,bar,kc*64,m0);tma_load_2d(sg,&p.gate,bar,kc*64,m0);tma_load_2d(sp,&p.proj,bar,kc*64,m0);
   if(kc==0){tma_load_2d(sx,&p.tri,bar,m0,0);for(int k=0;k<2;++k)tma_load_2d(sw+k*8192,&p.wp,bar,k*64,0);}
  }
  sync_group();mbar_wait(bar,kc&1);
  for(int i=tid;i<2048;i+=128){int r=i/32,c=2*(i%32),jr=j0+r;if(jr>=p.L)jr-=p.L;uint32_t y=pair_get(sy,r,c),g=pair_get(sg,r,c),v=pair_get(sp,r,c),ds=ldg32(p.ds+jr*128+kc*64+c);float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);
   stg32(p.dg+(size_t)(m0+r)*128+kc*64+c,pack_bf16(((ya*bf16lo(v))*ga)*(1.f-ga),((yb*bf16hi(v))*gb)*(1.f-gb)));pair_put(sy,r,c,pack_bf16(ya*ga,yb*gb));
  }
  sync_group();uint32_t f[4][4];load_frag_bf16<4,8192>(f,smem_u32(sy),w*16,lane);
#pragma unroll
  for(int k=0;k<4;++k)for(int q=0;q<4;++q)a[kc*4+k][q]=f[k][q];
  fence_proxy_async();sync_group();
 }
 uint8_t* sn=sy;
#pragma unroll
 for(int n=0;n<4;++n){
  if(n>0){if(tid==0){mbar_arrive_expect_tx(bar,16384);for(int k=0;k<2;++k)tma_load_2d(sw+k*8192,&p.wp,bar,k*64,n*64);}sync_group();mbar_wait(bar,(n+1)&1);}
  float acc[32]={};uint64_t de=smem_desc(smem_u32(sw),16,1024,1);uint32_t lo[1]={(uint32_t)de},hi[1]={(uint32_t)(de>>32)};fence_regs(acc);wgmma_fence();mma_chain<2,8,1>(acc,a,lo,hi);wgmma_commit();wgmma_wait<0>();fence_regs(acc);
#pragma unroll
  for(int q=0;q<4;++q){int mat=lane/8,rr=lane%8+8*(mat&1),cc=q*16+8*(mat>>1);stsm_x4(smem_u32(sn+n*8192)+swz128(w*16+rr,cc*2),pack_bf16(acc[q*8],acc[q*8+1]),pack_bf16(acc[q*8+2],acc[q*8+3]),pack_bf16(acc[q*8+4],acc[q*8+5]),pack_bf16(acc[q*8+6],acc[q*8+7]));}
  fence_proxy_async();sync_group();
 }
#else
 if(tid==0){mbar_init(bar,1);fence_barrier_init();mbar_arrive_expect_tx(bar,147456);
#pragma unroll
  for(int c=0;c<2;++c){tma_load_2d(sy+c*8192,&p.dy,bar,c*64,m0);tma_load_2d(sg+c*8192,&p.gate,bar,c*64,m0);tma_load_2d(sp+c*8192,&p.proj,bar,c*64,m0);}
#pragma unroll
  for(int n=0;n<4;++n)for(int k=0;k<2;++k)tma_load_2d(sw+(n*2+k)*8192,&p.wp,bar,k*64,n*64);
  tma_load_2d(sx,&p.tri,bar,m0,0);
 }
 sync_group();mbar_wait(bar,0);
 int j0=m0%p.L;
 for(int i=tid;i<4096;i+=128){int r=i/64,c=2*(i%64),jr=j0+r;if(jr>=p.L)jr-=p.L;uint8_t* yy=sy+(c/64)*8192;
  uint32_t y=pair_get(yy,r,c%64),g=pair_get(sg+(c/64)*8192,r,c%64),v=pair_get(sp+(c/64)*8192,r,c%64),ds=ldg32(p.ds+jr*128+c);
  float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);
  stg32(p.dg+(size_t)(m0+r)*128+c,pack_bf16(((ya*bf16lo(v))*ga)*(1.f-ga),((yb*bf16hi(v))*gb)*(1.f-gb)));
  pair_put(yy,r,c%64,pack_bf16(ya*ga,yb*gb));
 }
 sync_group();uint32_t a[8][4];load_frag_bf16<8,8192>(a,smem_u32(sy),w*16,lane);
 uint8_t* sn=sy;
#pragma unroll
 for(int nb=0;nb<4;nb+=ACCS){
  float acc[ACCS][32];
#pragma unroll
  for(int jn=0;jn<ACCS;++jn){int n=nb+jn;for(int j=0;j<32;++j)acc[jn][j]=0;uint64_t de=smem_desc(smem_u32(sw+n*16384),16,1024,1);uint32_t lo[1]={(uint32_t)de},hi[1]={(uint32_t)(de>>32)};fence_regs(acc[jn]);wgmma_fence();mma_chain<2,8,1>(acc[jn],a,lo,hi);wgmma_commit();}
  wgmma_wait<0>();
#pragma unroll
  for(int jn=0;jn<ACCS;++jn){int n=nb+jn;fence_regs(acc[jn]);
#pragma unroll
   for(int q=0;q<4;++q){int mat=lane/8,rr=lane%8+8*(mat&1),cc=q*16+8*(mat>>1);
    stsm_x4(smem_u32(sn+n*8192)+swz128(w*16+rr,cc*2),pack_bf16(acc[jn][q*8],acc[jn][q*8+1]),pack_bf16(acc[jn][q*8+2],acc[jn][q*8+3]),pack_bf16(acc[jn][q*8+4],acc[jn][q*8+5]),pack_bf16(acc[jn][q*8+6],acc[jn][q*8+7]));
   }
  }
 }
#endif
 sync_group();
#if LNFRAG
 float* red=reinterpret_cast<float*>(sw);float s1[2]={},s2[2]={};
 int ra=m0+w*16+lane/4,rb=ra+8;float mu[2]={p.mean[ra],p.mean[rb]},rs[2]={p.rs[ra],p.rs[rb]};
 int mat=lane/8,r8=lane%8;
#pragma unroll
 for(int k=0;k<16;++k){uint32_t fd[4],fx[4];
  ldsm_x4(fd,smem_u32(sn)+(k/4)*8192+swz128(w*16+r8+8*(mat&1),((k%4)*16+8*(mat>>1))*2));
  ldsm_x4_t(fx,smem_u32(sx)+swz128(k*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
  float gd[4]={},bd[4]={};
#pragma unroll
  for(int q=0;q<4;++q){int rr=q&1,cc=k*16+2*(lane%4)+8*(q>>1);float da=bf16lo(fd[q]),db=bf16hi(fd[q]);float xa=__fmul_rn(__fsub_rn(bf16lo(fx[q]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(fx[q]),mu[rr]),rs[rr]);float ha=da*p.gamma[cc],hb=db*p.gamma[cc+1];s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;gd[2*(q>>1)]+=da*xa;gd[2*(q>>1)+1]+=db*xb;bd[2*(q>>1)]+=da;bd[2*(q>>1)+1]+=db;}
#pragma unroll
  for(int q=0;q<4;++q){
#pragma unroll
   for(int sh=4;sh<=16;sh*=2){gd[q]+=__shfl_xor_sync(0xffffffff,gd[q],sh);bd[q]+=__shfl_xor_sync(0xffffffff,bd[q],sh);}
   if(lane<4){int c=k*16+2*lane+8*(q/2)+q%2;red[w*512+c]=gd[q];red[w*512+256+c]=bd[q];}
  }
 }
 s1[0]=quad_sum(s1[0])/256.f;s1[1]=quad_sum(s1[1])/256.f;s2[0]=quad_sum(s2[0])/256.f;s2[1]=quad_sum(s2[1])/256.f;
#pragma unroll
 for(int k=0;k<16;++k){uint32_t fd[4],fx[4],fo[4];
  ldsm_x4(fd,smem_u32(sn)+(k/4)*8192+swz128(w*16+r8+8*(mat&1),((k%4)*16+8*(mat>>1))*2));
  ldsm_x4_t(fx,smem_u32(sx)+swz128(k*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
#pragma unroll
  for(int q=0;q<4;++q){int rr=q&1,cc=k*16+2*(lane%4)+8*(q>>1);float xa=__fmul_rn(__fsub_rn(bf16lo(fx[q]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(fx[q]),mu[rr]),rs[rr]);fo[q]=pack_bf16(rs[rr]*((bf16lo(fd[q])*p.gamma[cc]-s2[rr])-xa*s1[rr]),rs[rr]*((bf16hi(fd[q])*p.gamma[cc+1]-s2[rr])-xb*s1[rr]));}
  int chq=8*(mat>>1)+r8;uint32_t off=chq*128+(((2*w+(mat&1))^(chq&7))*16)+k*16*128;stsm_x4_t(smem_u32(sx)+off,fo[0],fo[1],fo[2],fo[3]);
 }
 sync_group();fence_proxy_async();sync_group();
 if(tid==0){for(int c=0;c<256;c+=16)tma_store_3d(&p.dtri,sx+c*128,m0,c,0);tma_store_commit();tma_store_wait_all();}
 sync_group();ln_finish(p,red,last);
#else
 sync_group();float dga[8]={},dbe[8]={};
 // Warp owns rows; 8 channels per lane. Input x is channel-major in sx.
 for(int r=w;r<64;r+=4){float mu=p.mean[m0+r],rs=p.rs[m0+r];float dn[8],xh[8],dh[8],s1=0,s2=0;
#pragma unroll
  for(int k=0;k<8;++k){int c=lane+32*k;dn[k]=get(sn+(c/64)*8192,r,c%64);xh[k]=__fmul_rn(__fsub_rn(get(sx,c,r),mu),rs);dh[k]=dn[k]*p.gamma[c];s1+=dh[k]*xh[k];s2+=dh[k];dga[k]+=dn[k]*xh[k];dbe[k]+=dn[k];}
  s1=warp_sum(s1)/256.f;s2=warp_sum(s2)/256.f;
#pragma unroll
  for(int k=0;k<8;++k){int c=lane+32*k;put(sx,c,r,rs*((dh[k]-s2)-xh[k]*s1));}
 }
 sync_group();fence_proxy_async();sync_group();
 if(tid==0){for(int c=0;c<256;c+=16)tma_store_3d(&p.dtri,sx+c*128,m0,c,0);tma_store_commit();tma_store_wait_all();}
 float* red=reinterpret_cast<float*>(sw);
#pragma unroll
 for(int k=0;k<8;++k){red[w*512+lane+32*k]=dga[k];red[w*512+256+lane+32*k]=dbe[k];}
 sync_group();ln_finish(p,red,last);
#endif
}
TMN_DEVI void wgrad(const Params& p,uint8_t* sm,uint64_t* bar,int* last){
 int tid=threadIdx.x%128,lane=tid%32,w=tid/32,b=(blockIdx.x-p.tiles)*WGROUPS+threadIdx.x/128,tile=b/SPLITS,split=b%SPLITS;
 bool wg=tile<4;int t=wg?tile:tile-4,ncols=wg?2:4,mch=(t/ncols)*64,nch=(t%ncols)*64;
 uint8_t *sy=sm,*sg=sm+8192,*sp=sm+16384,*sx=sm+24576,*sb=sm+32768;
 float acc[32]={};int chunks=(p.M+63)/64,first=(chunks*split)/SPLITS,end=(chunks*(split+1))/SPLITS;
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}sync_group();
 auto load_tile = [&](int it){int row=it*64,cc=wg?nch:mch;
  if(tid==0){mbar_arrive_expect_tx(bar,wg?32768:24576);tma_load_2d(sy,&p.dy,bar,cc,row);tma_load_2d(sg,&p.gate,bar,cc,row);if(wg)tma_load_2d(sp,&p.proj,bar,cc,row);tma_load_2d(sx,wg?&p.xn:&p.norm,bar,wg?mch:nch,row);}
 };
 auto transform_tile = [&](int it){int row=it*64,cc=wg?nch:mch;
  int j0=row%p.L;
  for(int i=tid;i<2048;i+=128){int r=i/32,c=2*(i%32),jr=j0+r;if(jr>=p.L)jr-=p.L;
   uint32_t y=pair_get(sy,r,c),g=pair_get(sg,r,c),v=wg?pair_get(sp,r,c):0,ds=ldg32(p.ds+jr*128+cc+c);float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);
   pair_put(sy,r,c,pack_bf16(wg?((ya*bf16lo(v))*ga)*(1.f-ga):ya*ga,wg?((yb*bf16hi(v))*gb)*(1.f-gb):yb*gb));
  }
 };
 if(PREFETCH && first<end)load_tile(first);
 for(int it=first;it<end;++it){int row=it*64,cc=wg?nch:mch;
  if(!PREFETCH)load_tile(it);
  sync_group();mbar_wait(bar,(it-first)&1);
  if(PREFETCH != 3 || it==first)transform_tile(it);
  sync_group();uint32_t a[4][4],fb[4][4];load_trans(a,wg?sx:sy);load_frag_bf16<4,8192>(fb,smem_u32(wg?sy:sx),w*16,lane);
#if PREFETCH >= 2
  // Raw operands are now in registers; refill while transposing and issuing MMA.
  fence_proxy_async();sync_group();
  if(it+1<end)load_tile(it+1);
#endif
#pragma unroll
  for(int k=0;k<4;++k){int idx=lane>>3,chq=8*(idx>>1)+(lane&7);uint32_t off=chq*128+(((2*w+(idx&1))^(chq&7))*16)+k*16*128;
   stsm_x4_t(smem_u32(sb)+off,fb[k][0],fb[k][1],fb[k][2],fb[k][3]);
  }
  sync_group();fence_proxy_async();sync_group();uint64_t de=smem_desc(smem_u32(sb),16,1024,1);fence_regs(acc);wgmma_fence();
  static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;wgmma_m64n64k16_rs_off<k*32>(acc,a[k],(uint32_t)de,(uint32_t)(de>>32),it>first||k>0);});
  wgmma_commit();
  // Raw readers finished: MMA now uses registers and the separate sb buffer.
  if(PREFETCH == 1 && it+1<end)load_tile(it+1);
  if(PREFETCH == 3 && it+1<end){
   mbar_wait(bar,(it+1-first)&1);
   transform_tile(it+1);
  }
  wgmma_wait<0>();fence_regs(acc);fence_proxy_async();sync_group();
 }
 float* part=p.partw+(tile*SPLITS+split)*4096;
#pragma unroll
 for(int q=0;q<8;++q){int r=w*16+lane/4,c=q*8+2*(lane%4);part[r*64+c]=acc[4*q];part[r*64+c+1]=acc[4*q+1];part[(r+8)*64+c]=acc[4*q+2];part[(r+8)*64+c+1]=acc[4*q+3];}
 if(ticket(p.counts+tile,SPLITS,last)){
  for(int i=tid;i<4096;i+=128){float v=0;for(int s=0;s<SPLITS;++s)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*SPLITS+s)*4096+i];int r=mch+i/64,c=nch+i%64;(wg?p.dwg:p.dwp)[r*(wg?128:256)+c]=__float2bfloat16_rn(v);}
  sync_group();if(tid==0)atomicExch(p.counts+tile,0u);
 }
}
TMN_DEVI void wgrad_full(const Params& p,uint8_t* sm,uint64_t* bar,int* last){
 int tid=threadIdx.x%128,lane=tid%32,w=tid/32,b=(blockIdx.x-p.tiles)*WGROUPS+threadIdx.x/128,tile=b/SPLITS,split=b%SPLITS;
 bool wg=tile<2;int mch=(tile%2)*64,NC=wg?128:256,NB=NC/64;
 uint8_t *sy=sm,*sg=sm+8192,*sp=sm+16384,*sx=sm+24576,*sb=sm+32768;
 float acc[4][32]={};int chunks=p.M/64,first=(chunks*split)/SPLITS,end=(chunks*(split+1))/SPLITS,phase=0;
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}sync_group();
 for(int it=first;it<end;++it){int row=it*64,j0=row%p.L;
  uint32_t a[4][4];
  if(wg){
   if(tid==0){mbar_arrive_expect_tx(bar,8192);tma_load_2d(sx,&p.xn,bar,mch,row);}sync_group();mbar_wait(bar,(phase++)&1);load_trans(a,sx);fence_proxy_async();sync_group();
  }else{
   if(tid==0){mbar_arrive_expect_tx(bar,16384);tma_load_2d(sy,&p.dy,bar,mch,row);tma_load_2d(sg,&p.gate,bar,mch,row);}sync_group();mbar_wait(bar,(phase++)&1);
   for(int i=tid;i<2048;i+=128){int r=i/32,c=2*(i%32),jr=j0+r;if(jr>=p.L)jr-=p.L;uint32_t y=pair_get(sy,r,c),g=pair_get(sg,r,c),ds=ldg32(p.ds+jr*128+mch+c);pair_put(sy,r,c,pack_bf16((bf16lo(y)*bf16lo(ds))*bf16lo(g),(bf16hi(y)*bf16hi(ds))*bf16hi(g)));}
   sync_group();load_trans(a,sy);fence_proxy_async();sync_group();
  }
  static_for<4>([&](auto nbi){constexpr int nb=decltype(nbi)::value;if(nb<NB){
   if(tid==0){mbar_arrive_expect_tx(bar,wg?24576:8192);if(wg){tma_load_2d(sy,&p.dy,bar,nb*64,row);tma_load_2d(sg,&p.gate,bar,nb*64,row);tma_load_2d(sp,&p.proj,bar,nb*64,row);}else tma_load_2d(sx,&p.norm,bar,nb*64,row);}
   sync_group();mbar_wait(bar,(phase++)&1);
   if(wg){for(int i=tid;i<2048;i+=128){int r=i/32,c=2*(i%32),jr=j0+r;if(jr>=p.L)jr-=p.L;uint32_t y=pair_get(sy,r,c),g=pair_get(sg,r,c),v=wg?pair_get(sp,r,c):0,ds=ldg32(p.ds+jr*128+nb*64+c);float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);pair_put(sy,r,c,pack_bf16(((ya*bf16lo(v))*ga)*(1.f-ga),((yb*bf16hi(v))*gb)*(1.f-gb)));}sync_group();}
   uint32_t fb[4][4];load_frag_bf16<4,8192>(fb,smem_u32(wg?sy:sx),w*16,lane);
#pragma unroll
   for(int k=0;k<4;++k){int idx=lane>>3,chq=8*(idx>>1)+(lane&7);uint32_t off=chq*128+(((2*w+(idx&1))^(chq&7))*16)+k*16*128;stsm_x4_t(smem_u32(sb)+off,fb[k][0],fb[k][1],fb[k][2],fb[k][3]);}
   sync_group();fence_proxy_async();sync_group();uint64_t de=smem_desc(smem_u32(sb),16,1024,1);fence_regs(acc[nb]);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;wgmma_m64n64k16_rs_off<k*32>(acc[nb],a[k],(uint32_t)de,(uint32_t)(de>>32),it>first||k>0);});
   wgmma_commit();wgmma_wait<0>();fence_regs(acc[nb]);fence_proxy_async();sync_group();
  }});
 }
 int ne=64*NC,base=wg?tile*SPLITS*8192:2*SPLITS*8192+(tile-2)*SPLITS*16384;float* part=p.partw+base+split*ne;
 static_for<4>([&](auto nbi){constexpr int nb=decltype(nbi)::value;if(nb<NB){
#pragma unroll
  for(int q=0;q<8;++q){int r=w*16+lane/4,c=nb*64+q*8+2*(lane%4);part[r*NC+c]=acc[nb][4*q];part[r*NC+c+1]=acc[nb][4*q+1];part[(r+8)*NC+c]=acc[nb][4*q+2];part[(r+8)*NC+c+1]=acc[nb][4*q+3];}
 }});
 if(ticket(p.counts+tile,SPLITS,last)){
  for(int i=tid;i<ne;i+=128){float v=0;for(int s=0;s<SPLITS;++s)v+=reinterpret_cast<volatile float*>(p.partw)[base+s*ne+i];(wg?p.dwg:p.dwp)[mch*NC+i]=__float2bfloat16_rn(v);}
  sync_group();if(tid==0)atomicExch(p.counts+tile,0u);
 }
}

#include "dgrad_wide.cuh"
#include "wgrad_ss.cuh"
#include "wgrad_ws.cuh"
#include "wgrad_ws128.cuh"

extern "C" __global__ __launch_bounds__(128*WGROUPS,1)
void fused_b1b4(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 __shared__ __align__(8) uint64_t bar[16];__shared__ int last[WGROUPS];
 int wg=threadIdx.x/128;
 if(blockIdx.x<p.tiles){if(wg==0&&ROLE!=2){if(DFAST)dgrad_wide(p,sm,&bar[0],&last[0]);else dgrad(p,sm,&bar[0],&last[0]);}}else{if(ROLE!=1){if(WSS==4)wgrad_ws128(p,sm,bar,&last[1]);else if(WSS==3)wgrad_ws(p,sm,bar,&last[1]);else if(WSS)wgrad_ss(p,sm+wg*(WSS>=2?65536:32768),&bar[wg*2],&last[wg]);else if(FULLW)wgrad_full(p,sm+wg*40960,&bar[wg],&last[wg]);else wgrad(p,sm+wg*40960,&bar[wg],&last[wg]);}}
}
