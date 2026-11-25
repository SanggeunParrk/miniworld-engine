// Frozen generated source: wide_dense_output.DenseOutput.
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
// MiniWorld wide training forward. Uses the engine's TMA/WGMMA primitives.
// Keep the output-LN tile on chip; stream weight K64 chunks instead of
// allocating whole gate/projection blocks. No global normalized-triangle buffer.
#include "tmn_kernels.cuh"
using namespace tmn;
using namespace tmn::sm90;
using bf = __nv_bfloat16;
constexpr int D=WIDTH,H=2*D,G=GROUPS,NT=128*G;
constexpr int KC=KCHUNK;
constexpr int XBYTES=H*128,STAGE=8192*(G+1)*KC;
struct Params {CUtensorMap tri,norm;const float *gamma,*beta;float *mu,*rs;int M;};
TMN_DEVI float rd(const bf* p,int i){return __bfloat162float(p[i]);}
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,q));return v;}
TMN_DEVI void mma(float (&v)[32],uint64_t a,uint64_t b,int ac){
 asm volatile("{.reg .pred p;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]) : "l"(a),"l"(b),"r"(ac));
}
TMN_DEVI uint32_t load_shared_word(uint32_t a){uint32_t v;asm("ld.shared.b32 %0,[%1];":"=r"(v):"r"(a));return v;}
TMN_DEVI void store_shared_word(uint32_t a,uint32_t v){asm volatile("st.shared.b32 [%0],%1;"::"r"(a),"r"(v):"memory");}
extern "C" __global__ __launch_bounds__(NT,1) void mw_wide_dense_norm(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 uint8_t* sx=sm;uint64_t* bar=reinterpret_cast<uint64_t*>(sm+XBYTES);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 if(tid==0){mbar_init(bar,1);mbar_init(bar+1,1);mbar_init(bar+2,1);fence_barrier_init();}
 __syncthreads();int phase=0,xphase=0;
 for(int row=blockIdx.x*64;row<p.M;row+=gridDim.x*64){
  if(tid==0){mbar_arrive_expect_tx(bar+2,XBYTES);for(int c=0;c<H;c+=64)tma_load_2d(sx+c*128,&p.tri,bar+2,row,c);}
  mbar_wait(bar+2,xphase);xphase^=1;__syncthreads();
  // Warpgroup-local in-place transpose, one independent K64 chunk per group.
  // ldmatrix/stmatrix avoid the scalar channel-strided shared-memory accesses.
  for(int kc=tid/128;kc<H/64;kc+=G){
   uint32_t f[4][4];uint32_t base=smem_u32(sx+kc*8192);
   int mat=lane>>3,r8=lane&7,wiw=warp%4;
   #pragma unroll
   for(int q=0;q<4;++q)ldsm_x4_t(f[q],base+swz128(16*q+r8+((mat&2)?8:0),(16*wiw+((mat&1)?8:0))*2));
   named_bar_sync(1+tid/128,128);
   #pragma unroll
   for(int q=0;q<4;++q){int r=16*wiw+r8+((mat&1)?8:0),c=16*q+((mat&2)?8:0);stsm_x4(base+swz128(r,c*2),f[q][0],f[q][1],f[q][2],f[q][3]);}
  }
  __syncthreads();
#if WIDTH < 512
  // Each warp owns complete rows; values stay in registers through both LN passes.
  for(int r=warp;r<64;r+=NT/32){
   float values[H/32];float s=0;
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;values[q]=rd(reinterpret_cast<bf*>(sx+(c/64)*8192),swz128(r,(c%64)*2)/2);s+=values[q];}
   float mu=sumwarp(s)/H;s=0;
   #pragma unroll
   for(int q=0;q<H/32;++q){float z=values[q]-mu;s+=z*z;}
   float rs=rsqrtf(sumwarp(s)/H+1e-5f);if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;reinterpret_cast<bf*>(sx+(c/64)*8192)[swz128(r,(c%64)*2)/2]=__float2bfloat16_rn(fmaf((values[q]-mu)*rs,p.gamma[c],p.beta[c]));}
  }
#else
  // Keep the wide LN live set bounded: packed coalesced row passes.
  for(int r=warp;r<64;r+=NT/32){
   float s=0;
   #pragma unroll 1
   for(int c=2*lane;c<H;c+=64){uint32_t v=load_shared_word(smem_u32(sx+(c/64)*8192)+swz128(r,(c%64)*2));s+=bf16lo(v)+bf16hi(v);}
   float mu=sumwarp(s)/H;s=0;
   #pragma unroll 1
   for(int c=2*lane;c<H;c+=64){uint32_t v=load_shared_word(smem_u32(sx+(c/64)*8192)+swz128(r,(c%64)*2));float a=bf16lo(v)-mu,b=bf16hi(v)-mu;s+=a*a+b*b;}
   float rs=rsqrtf(sumwarp(s)/H+1e-5f);if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}
   #pragma unroll 1
   for(int c=2*lane;c<H;c+=64){uint32_t addr=smem_u32(sx+(c/64)*8192)+swz128(r,(c%64)*2);uint32_t v=load_shared_word(addr);float a=fmaf((bf16lo(v)-mu)*rs,p.gamma[c],p.beta[c]),b=fmaf((bf16hi(v)-mu)*rs,p.gamma[c+1],p.beta[c+1]);store_shared_word(addr,pack_bf16(a,b));}
  }
#endif
  fence_proxy_async();__syncthreads();

  if(tid==0){for(int c=0;c<H;c+=64){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.norm),"r"(smem_u32(sx+c*128)),"r"(c),"r"(row):"memory");
  }tma_store_commit();tma_store_wait_all();}
  __syncthreads();
 }
}
struct Epi {const bf *proj,*gate,*x,*ds;bf *y;int elements,period;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_dense_output_epi(__grid_constant__ const Epi p){
 for(int i=blockIdx.x*256+threadIdx.x;i<p.elements/2;i+=gridDim.x*256){
  uint32_t pr=reinterpret_cast<const uint32_t*>(p.proj)[i],ga=reinterpret_cast<const uint32_t*>(p.gate)[i];
  uint32_t xx=reinterpret_cast<const uint32_t*>(p.x)[i],ds=reinterpret_cast<const uint32_t*>(p.ds)[i%(p.period/2)];
  float a=bf16lo(pr)*math::sigmoid(bf16lo(ga)),b=bf16hi(pr)*math::sigmoid(bf16hi(ga));
  reinterpret_cast<uint32_t*>(p.y)[i]=pack_bf16(fmaf(a,bf16lo(ds),bf16lo(xx)),fmaf(b,bf16hi(ds),bf16hi(xx)));
 }
}
