// SPDX-License-Identifier: Apache-2.0
// Single-direction training extension of Anthropic Hopper TMA/WGMMA primitives.
// A CTA owns 64 rows and accumulates both parameter gradients on chip.
// Only dGate (B7 consumer) and dTri (contraction consumer) leave the CTA.
#include "common_recompute.cuh"
using bf=__nv_bfloat16;
constexpr int XN=0,TRI=16384,NORM=32768,DY=49152,DP=65536,DG=81920,DN=98304,WP=114688,WG=147456;
// second input set (xn, tri, dy of the next tile, TMA-prefetched while the current tile computes): 49152 B behind mu / rs
#ifndef B1_LNI
#define B1_LNI 2
#endif
#ifndef B1_VEC
#define B1_VEC 1
#endif
#ifndef B1_PK
#define B1_PK 1
#endif
constexpr int XN2=181248,TRI2=XN2+16384,DY2=TRI2+16384;
constexpr int SMEM=DY2+16384;
#if MASTER_FP32
using MasterDW=float;
TMN_DEVI float master_value(float v){return v;}
#else
using MasterDW=bf;
TMN_DEVI bf master_value(float v){return __float2bfloat16_rn(v);}
#endif
struct Params {
 CUtensorMap xn,tri,dy,wp,wg,dg,dt;
 const bf* x;const bf* ds;const float* gamma;const float* beta;
 bf* y;MasterDW* dwg;MasterDW* dwp;float* dgamma;float* dbeta;float* partial;int M,L;
};
TMN_DEVI int pos(int r,int c){return (c/64)*8192+swz128(r,(c%64)*2);}
TMN_DEVI float val(uint8_t* sm,int r,int c){return __bfloat162float(*reinterpret_cast<bf*>(sm+pos(r,c)));}
TMN_DEVI void set(uint8_t* sm,int r,int c,float v){*reinterpret_cast<bf*>(sm+pos(r,c))=__float2bfloat16_rn(v);}
// two adjacent columns (c even) as one 32-bit shared access: the same bytes as two scalar accesses
TMN_DEVI uint32_t val2(uint8_t* sm,int r,int c){return *reinterpret_cast<uint32_t*>(sm+pos(r,c));}
TMN_DEVI void set2(uint8_t* sm,int r,int c,float a,float b){__nv_bfloat162 v=__floats2bfloat162_rn(a,b);*reinterpret_cast<__nv_bfloat162*>(sm+pos(r,c))=v;}
template<int N> TMN_DEVI void sum_n(float (&a)[N]){
#pragma unroll
 for(int k=16;k;k>>=1){
#pragma unroll
  for(int q=0;q<N;++q)a[q]+=__shfl_xor_sync(0xffffffff,a[q],k);}}
TMN_DEVI float sum(float v){for(int k=16;k;k>>=1)v+=__shfl_xor_sync(0xffffffff,v,k);return v;}
TMN_DEVI int rr(int j){return (threadIdx.x/32%4)*16+threadIdx.x%32/4+8*((j/2)&1);}
TMN_DEVI int cc(int j){return j/8*16+2*(threadIdx.x%4)+8*(j/2%4/2)+j%2;}
TMN_DEVI void project(float (&v)[32],uint8_t* a,uint8_t* b){
 int wi=threadIdx.x/128;fence_regs(v);wgmma_fence();
 static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;
 mma_dgrad(v,smem_desc(smem_u32(a+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(b+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
 });wgmma_commit();wgmma_wait<0>();fence_regs(v);
}
TMN_DEVI void weight(float (&v)[64],uint8_t* a,uint8_t* b,bool add){
 int wi=threadIdx.x/128;fence_regs(v);wgmma_fence();
 static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
 mma_ss128(v,smem_desc(smem_u32(a+wi*8192+k*2048),16,1024,1),smem_desc(smem_u32(b+k*2048),8192,1024,1),add||k>0);
 });wgmma_commit();wgmma_wait<0>();fence_regs(v);
}
TMN_DEVI void stmap(const CUtensorMap* map,const void* src,int c,int r){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r):"memory");}
template<bool FWD> TMN_DEVI void body(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x,wi=tid/128,lane=tid%32,warp=tid/32;
 float* mu=reinterpret_cast<float*>(sm+180224);float* rs=mu+64;
 float dwg[64]={},dwp[64]={},gg[4]={},bb[4]={};
 if(tid==0){mbar_init(bar,1);mbar_init(bar+1,1);mbar_init(bar+2,1);fence_barrier_init();}__syncthreads();
 if(tid==0){mbar_arrive_expect_tx(bar,65536);for(int n=0;n<2;++n)for(int k=0;k<2;++k){
 tma_load_2d(sm+WP+n*16384+k*8192,&p.wp,bar,k*64,n*64);
 tma_load_2d(sm+WG+n*16384+k*8192,&p.wg,bar,k*64,n*64);}}
 mbar_wait(bar,0);__syncthreads();
 // first tile of this CTA into input set 0
 if(tid==0&&(int)blockIdx.x<p.M/64){mbar_arrive_expect_tx(bar+1,FWD?32768:49152);
  for(int k=0;k<2;++k){tma_load_2d(sm+XN+k*8192,&p.xn,bar+1,k*64,blockIdx.x*64);if constexpr(!FWD)tma_load_2d(sm+DY+k*8192,&p.dy,bar+1,k*64,blockIdx.x*64);}
  tma_load_2d(sm+TRI,&p.tri,bar+1,blockIdx.x*64,0);}
 int iteration=0;
 for(int tile=blockIdx.x;tile<p.M/64;tile+=gridDim.x,++iteration){int row=tile*64;
  uint8_t* const sXN=sm+((iteration&1)?XN2:XN);uint8_t* const sTRI=sm+((iteration&1)?TRI2:TRI);uint8_t* const sDY=sm+((iteration&1)?DY2:DY);
  // the next tile's xn / tri / dy go to the other input set now; its previous user (tile iteration - 1) finished (end of its loop body)
  if(tid==0&&tile+(int)gridDim.x<p.M/64){const int nb=(iteration+1)&1,nrow=(tile+gridDim.x)*64;uint64_t* nbar=bar+1+nb;
   mbar_arrive_expect_tx(nbar,FWD?32768:49152);
   for(int k=0;k<2;++k){tma_load_2d(sm+(nb?XN2:XN)+k*8192,&p.xn,nbar,k*64,nrow);if constexpr(!FWD)tma_load_2d(sm+(nb?DY2:DY)+k*8192,&p.dy,nbar,k*64,nrow);}
   tma_load_2d(sm+(nb?TRI2:TRI),&p.tri,nbar,nrow,0);}
  mbar_wait(bar+1+(iteration&1),(iteration>>1)&1);__syncthreads();
  // batched samples (M = B L L, tokens b-major): the 64-row tile lies in sample s = row / L^2, whose dropout scale is rows s L .. s L + L - 1
  const bf* const ds_rows=p.ds+((size_t)(row/(p.L*p.L))*p.L+(row%p.L))*128;  // (row + r) % L = row % L + r: L % 64 == 0, r < 64
#if B1_LNI>1
  // B1_LNI independent rows (r, r + 8, ...) of this warp per pass: the same per-row arithmetic and shuffle order, the chains interleaved
  for(int r0=warp;r0<64;r0+=8*B1_LNI){float v[B1_LNI][4],s[B1_LNI],mean[B1_LNI],vv[B1_LNI],inv[B1_LNI];
   #pragma unroll
   for(int q=0;q<B1_LNI;++q){s[q]=0;
    #pragma unroll
    for(int k=0;k<4;++k){v[q][k]=get(sTRI,lane+k*32,r0+8*q);s[q]+=v[q][k];}}
   sum_n<B1_LNI>(s);
   #pragma unroll
   for(int q=0;q<B1_LNI;++q){mean[q]=s[q]/128.f;vv[q]=0;
    #pragma unroll
    for(int k=0;k<4;++k){float z=v[q][k]-mean[q];vv[q]+=z*z;}}
   sum_n<B1_LNI>(vv);
   #pragma unroll
   for(int q=0;q<B1_LNI;++q){inv[q]=rsqrtf(vv[q]/128.f+1e-5f);if(lane==0){mu[r0+8*q]=mean[q];rs[r0+8*q]=inv[q];}
    #pragma unroll
    for(int k=0;k<4;++k){int c=lane+k*32;set(sm+NORM,r0+8*q,c,(v[q][k]-mean[q])*inv[q]*p.gamma[c]+p.beta[c]);}}
  }
  __syncthreads();fence_proxy_async();__syncthreads();
#else
  for(int r=warp;r<64;r+=8){float v[4],s=0;
   #pragma unroll
   for(int k=0;k<4;++k){v[k]=get(sTRI,lane+k*32,r);s+=v[k];}
   float mean=sum(s)/128.f;float vv=0;
   #pragma unroll
   for(int k=0;k<4;++k){float z=v[k]-mean;vv+=z*z;}
   float inv=rsqrtf(sum(vv)/128.f+1e-5f);if(lane==0){mu[r]=mean;rs[r]=inv;}
   #pragma unroll
   for(int k=0;k<4;++k){int c=lane+k*32;set(sm+NORM,r,c,(v[k]-mean)*inv*p.gamma[c]+p.beta[c]);}
  }__syncthreads();fence_proxy_async();__syncthreads();
#endif
  float gate[32]={},proj[32]={};project(gate,sXN,sm+WG);project(proj,sm+NORM,sm+WP);
#if B1_VEC
  if constexpr(!FWD){
   // the same per-element arithmetic, two adjacent columns per shared access and per dropout-scale load
   #pragma unroll
   for(int j=0;j<32;j+=2){int r=rr(j),c=wi*64+cc(j);
#if B1_PK
    // packed: one cvt.rn.bf16x2 per pair of roundings, one bf16x2 multiply for dy * dropout scale (the product of two bf16 is exact in fp32: one rounding, as before)
    const __nv_bfloat162 gb=__floats2bfloat162_rn(gate[j],gate[j+1]),vb=__floats2bfloat162_rn(proj[j],proj[j+1]);
    float g0=math::sigmoid(__bfloat162float(gb.x)),g1=math::sigmoid(__bfloat162float(gb.y));
    float v0=__bfloat162float(vb.x),v1=__bfloat162float(vb.y);
    __nv_bfloat162 dd=*reinterpret_cast<const __nv_bfloat162*>(ds_rows+r*128+c);const uint32_t yy=val2(sDY,r,c);
    const __nv_bfloat162 dyp=__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&yy),dd);
    float dy0=__bfloat162float(dyp.x),dy1=__bfloat162float(dyp.y);
#else
    float g0=math::sigmoid(math::round_bf16(gate[j])),v0=math::round_bf16(proj[j]),g1=math::sigmoid(math::round_bf16(gate[j+1])),v1=math::round_bf16(proj[j+1]);
    const __nv_bfloat162 dd=*reinterpret_cast<const __nv_bfloat162*>(ds_rows+r*128+c);const uint32_t yy=val2(sDY,r,c);
    float dy0=math::round_bf16(__bfloat162float(__ushort_as_bfloat16(yy&0xffffu))*__bfloat162float(dd.x));
    float dy1=math::round_bf16(__bfloat162float(__ushort_as_bfloat16(yy>>16))*__bfloat162float(dd.y));
#endif
    set2(sm+DP,r,c,dy0*g0,dy1*g1);set2(sm+DG,r,c,((dy0*v0)*g0)*(1-g0),((dy1*v1)*g1)*(1-g1));
   }
  } else
#endif
  {
  #pragma unroll
  for(int j=0;j<32;++j){int r=rr(j),c=wi*64+cc(j);float g=math::sigmoid(math::round_bf16(gate[j])),v=math::round_bf16(proj[j]);float ds=__bfloat162float(ds_rows[(r)*128+(c)]);
   if constexpr(FWD){size_t ix=size_t(row+r)*128+c;p.y[ix]=__float2bfloat16_rn(__bfloat162float(p.x[ix])+v*g*ds);}
   else {float dy=math::round_bf16(val(sDY,r,c)*ds);set(sm+DP,r,c,dy*g);set(sm+DG,r,c,((dy*v)*g)*(1-g));}
  }}
  __syncthreads();
  if constexpr(!FWD){
   fence_proxy_async();__syncthreads();
   // Full-precision dW accumulators persist over the CTA's row tiles.
   weight(dwg,sm+DG,sXN,iteration>0);weight(dwp,sm+DP,sm+NORM,iteration>0);
   float dn[32];recompute_gemm<128,16384>(dn,sm+DP,sm+WP+wi*8192);
   #pragma unroll
#if B1_VEC && !SINGLE_DEBUG
   for(int j=0;j<32;j+=2)set2(sm+DN,rr(j),wi*64+cc(j),dn[j],dn[j+1]);
#else
   for(int j=0;j<32;++j){set(sm+DN,rr(j),wi*64+cc(j),dn[j]);
#if SINGLE_DEBUG
    p.y[size_t(row+rr(j))*128+wi*64+cc(j)]=__float2bfloat16_rn(dn[j]);
#endif
   }
#endif
   __syncthreads();
#if B1_LNI>1
   for(int r0=warp;r0<64;r0+=8*B1_LNI){float z[B1_LNI][4],v[B1_LNI][4],grad[B1_LNI][4],s0[B1_LNI],s1[B1_LNI];
    #pragma unroll
    for(int q=0;q<B1_LNI;++q){int r=r0+8*q;s0[q]=0;s1[q]=0;
     #pragma unroll
     for(int k=0;k<4;++k){int c=lane+k*32;z[q][k]=(get(sTRI,c,r)-mu[r])*rs[r];grad[q][k]=val(sm+DN,r,c);v[q][k]=grad[q][k]*p.gamma[c];s0[q]+=v[q][k];s1[q]+=v[q][k]*z[q][k];gg[k]+=grad[q][k]*z[q][k];bb[k]+=grad[q][k];}}
    sum_n<B1_LNI>(s0);sum_n<B1_LNI>(s1);
    #pragma unroll
    for(int q=0;q<B1_LNI;++q){int r=r0+8*q;float a0=s0[q]/128.f,a1=s1[q]/128.f;
     #pragma unroll
     for(int k=0;k<4;++k)put(sTRI,lane+k*32,r,(v[q][k]-a0-z[q][k]*a1)*rs[r]);}
   }
   __syncthreads();fence_proxy_async();__syncthreads();
#else
   for(int r=warp;r<64;r+=8){float z[4],v[4],grad[4],s0=0,s1=0;
    #pragma unroll
    for(int k=0;k<4;++k){int c=lane+k*32;z[k]=(get(sTRI,c,r)-mu[r])*rs[r];grad[k]=val(sm+DN,r,c);v[k]=grad[k]*p.gamma[c];s0+=v[k];s1+=v[k]*z[k];gg[k]+=grad[k]*z[k];bb[k]+=grad[k];}
    s0=sum(s0)/128.f;s1=sum(s1)/128.f;
    #pragma unroll
    for(int k=0;k<4;++k)put(sTRI,lane+k*32,r,(v[k]-s0-z[k]*s1)*rs[r]);
   }__syncthreads();fence_proxy_async();__syncthreads();
#endif
   if(tid==0){for(int k=0;k<2;++k)stmap(&p.dg,sm+DG+k*8192,k*64,row);for(int k=0;k<8;++k)stmap(&p.dt,sTRI+k*2048,row,k*16);tma_store_commit();tma_store_wait_all();}
   __syncthreads();
  }
 }
 if constexpr(!FWD){
  // One compact FP32 partial per CTA; final reduction runs after this kernel.
  #pragma unroll
  for(int q=0;q<16;++q){int r=(tid/32%4)*16+lane/4+wi*64,c=q*8+2*(lane%4);float* base=p.partial+blockIdx.x*33024;
   stg64f(base+r*128+c,dwg[q*4],dwg[q*4+1]);stg64f(base+(r+8)*128+c,dwg[q*4+2],dwg[q*4+3]);
   stg64f(base+16384+r*128+c,dwp[q*4],dwp[q*4+1]);stg64f(base+16384+(r+8)*128+c,dwp[q*4+2],dwp[q*4+3]);
  }
  float* tmp=reinterpret_cast<float*>(sm);
  #pragma unroll
  for(int k=0;k<4;++k){tmp[warp*256+lane+k*32]=gg[k];tmp[warp*256+128+lane+k*32]=bb[k];}
  __syncthreads();float s=0;for(int w=0;w<8;++w)s+=tmp[w*256+tid];p.partial[blockIdx.x*33024+32768+tid]=s;
 }
}
extern "C" __global__ __launch_bounds__(256,1) void single_output(__grid_constant__ const Params p){extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[3];body<true>(p,sm,bar);}
extern "C" __global__ __launch_bounds__(256,1) void single_b1(__grid_constant__ const Params p){extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[3];body<false>(p,sm,bar);}
extern "C" __global__ void single_reduce(__grid_constant__ const Params p,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=33024)return;float s=0;for(int c=0;c<count;++c)s+=p.partial[c*33024+i];
 if(i<16384)p.dwg[i]=master_value(s);else if(i<32768)p.dwp[i-16384]=master_value(s);else if(i<32896)p.dgamma[i-32768]=s;else p.dbeta[i-32896]=s;
}
