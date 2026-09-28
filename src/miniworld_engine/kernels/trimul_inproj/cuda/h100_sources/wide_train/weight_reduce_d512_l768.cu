// Frozen generated source: wide_split_dwp.SplitProjectionWeight reduce (widths.cu parameters).
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
// Width-scalable training extension of Anthropic Hopper TMA/WGMMA primitives.
// D128 policies remain available. Workspace is transient, not forward activations.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;
using bf=__nv_bfloat16;
#ifndef HIDDEN
#define HIDDEN (2*WIDTH)
#endif
constexpr int D=WIDTH,H=HIDDEN,SPLITS=WEIGHT_SPLITS,GROUPS=(D==64?1:2),THREADS=128*GROUPS,NT=64*GROUPS,STAGE=8192*(1+GROUPS);
struct Params{CUtensorMap map[16];bf* t[24];float* f[13];int M,L;};
// t: x,tri,dy,ds,y,xn,norm,dp,dg,dn,dxn,dx,dtri,gp0..3,dw0..3,dwp,dwg,unused
// f: gi,bi,go,bo,mask,mu,rs,partw,ln0,ln1,ln2,ln3
TMN_DEVI void stamp(const Params& p,int i){
#if PROFILE_STAGE
 cooperative_groups::this_grid().sync();if(blockIdx.x==0&&threadIdx.x==0)reinterpret_cast<unsigned long long*>(p.f[12])[i]=clock64();
#endif
}
TMN_DEVI float rd(const bf* x,size_t i){return __bfloat162float(x[i]);}
TMN_DEVI bf cv(float v){return __float2bfloat16_rn(v);}
TMN_DEVI float rn(float v){return __bfloat162float(cv(v));}
TMN_DEVI float wsum(float x){for(int k=16;k;k>>=1)x+=__shfl_xor_sync(0xffffffff,x,k);return x;}
TMN_DEVI void syncvis(){__syncthreads();fence_proxy_async();__syncthreads();}
template<int TA,int TB> TMN_DEVI void mma(float (&d)[32],uint64_t a,uint64_t b,int ac){
 asm volatile("{.reg .pred p;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,%35,%36;}" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(ac),"n"(TA),"n"(TB));
}
// C[64,64] = op(A)*op(B), B's non-transposed representation is N,K.
// TA=0 A[M,K], TA=1 A[K,M]; TB=0 B[N,K], TB=1 B[K,N].
template<int TA,int TB> TMN_DEVI void gemm(float (&acc)[32],const CUtensorMap* A,const CUtensorMap* B,int row,int col,int begin,int end,uint8_t* sm,uint64_t* bar,int& phase,bool add=false){
 if(!add)for(int j=0;j<32;++j)acc[j]=0;
 if(threadIdx.x==0){mbar_arrive_expect_tx(bar,STAGE);tma_load_2d(sm,A,bar,TA?row:begin,TA?begin:row);for(int g=0;g<GROUPS;++g)tma_load_2d(sm+8192*(1+g),B,bar,TB?col+g*64:begin,TB?begin:col+g*64);}
 for(int k=begin,iter=0;k<end;k+=64,++iter){
  int slot=iter%2;uint8_t* buf=sm+slot*STAGE;
  if(k+64<end && threadIdx.x==0){int nx=k+64;uint8_t* nxt=sm+(1-slot)*STAGE;mbar_arrive_expect_tx(bar+1-slot,STAGE);tma_load_2d(nxt,A,bar+1-slot,TA?row:nx,TA?nx:row);for(int g=0;g<GROUPS;++g)tma_load_2d(nxt+8192*(1+g),B,bar+1-slot,TB?col+g*64:nx,TB?nx:col+g*64);}
  mbar_wait(bar+slot,(phase>>slot)&1);phase^=1<<slot;__syncthreads();fence_regs(acc);wgmma_fence();
  #pragma unroll
  for(int q=0;q<4;++q)mma<TA,TB>(acc,smem_desc(smem_u32(buf+(TA?q*2048:q*32)),16,1024,1),smem_desc(smem_u32(buf+8192*(1+threadIdx.x/128)+(TB?q*2048:q*32)),16,1024,1),add||k>begin||q>0);
  wgmma_commit();wgmma_wait<0>();fence_regs(acc);__syncthreads();
 }

}
TMN_DEVI int rr(int j){return ((threadIdx.x/32)%4)*16+(threadIdx.x%32)/4+8*((j/2)&1);}
TMN_DEVI int cc(int j){return (j/8)*16+2*(threadIdx.x%4)+8*((j/2)%4/2)+(j%2);}
TMN_DEVI void store(bf* out,float (&v)[32],int row,int col,int cols){
 for(int j=0;j<32;++j)out[size_t(row+rr(j))*cols+col+cc(j)]=cv(v[j]);
}
// Per-warp LN. Channel-first tri loads, row-major normalized output.
TMN_DEVI int si(int r,int c){return r*H+(c^(r*8));}
TMN_DEVI void norm(const Params& p,uint8_t* sm){
 int lane=threadIdx.x%32,warp=threadIdx.x/32;bf* tile=reinterpret_cast<bf*>(sm);
 for(int first=blockIdx.x*16;first<p.M;first+=gridDim.x*16){
  for(int i=threadIdx.x;i<16*H;i+=THREADS){int r=i%16,c=i/16;tile[si(r,c)]=p.t[1][size_t(c)*p.M+first+r];}__syncthreads();
  for(int r=warp;r<16;r+=THREADS/32){int row=first+r;float sum=0;for(int c=lane;c<H;c+=32)sum+=rd(tile,si(r,c));float mu=wsum(sum)/H;
   float var=0;for(int c=lane;c<H;c+=32){float z=rd(tile,si(r,c))-mu;var+=z*z;}float rs=rsqrtf(wsum(var)/H+1e-5f);
   if(lane==0){p.f[5][row]=mu;p.f[6][row]=rs;}
   for(int c=lane;c<H;c+=32)p.t[6][size_t(row)*H+c]=cv((rd(tile,si(r,c))-mu)*rs*p.f[2][c]+p.f[3][c]);
  }__syncthreads();
 }
}
template<bool FWD> TMN_DEVI void out_gp(const Params& p,uint8_t* sm,uint64_t* bar,int& phase){
 for(int tile=blockIdx.x;tile<(p.M/64)*(D/NT);tile+=gridDim.x){
  int row=(tile/(D/NT))*64,col=(tile%(D/NT))*NT+(threadIdx.x/128)*64;float proj[32],gate[32];
  gemm<0,0>(proj,&p.map[3],&p.map[1],row,col,0,H,sm,bar,phase);
  gemm<0,0>(gate,&p.map[0],&p.map[2],row,col,0,D,sm,bar,phase);
  for(int j=0;j<32;++j){int r=row+rr(j),c=col+cc(j);size_t ix=size_t(r)*D+c;float g=math::sigmoid(rn(gate[j])),v=rn(proj[j]),ds=rd(p.t[3],(r%p.L)*D+c);
   if constexpr(FWD)p.t[4][ix]=cv(rd(p.t[0],ix)+v*g*ds);
   else {float dy=rn(rd(p.t[2],ix)*ds);p.t[7][ix]=cv(dy*g);p.t[8][ix]=cv(((dy*v)*g)*(1-g));}
  }
 }
}
// dW deterministic split-K. Transposed A is dp/dg in row-major M,C form.
TMN_DEVI void weight(const Params& p,int amap,int bmap,int N,int K,int offset,uint8_t* sm,uint64_t* bar,int& phase){
 int nt=N/64,kt=K/NT,tiles=nt*kt,step=p.M/SPLITS;
 for(int t=blockIdx.x;t<tiles*SPLITS;t+=gridDim.x){int s=t/tiles,u=t%tiles,row=(u/kt)*64,col=(u%kt)*NT+(threadIdx.x/128)*64;float v[32];gemm<1,1>(v,&p.map[amap],&p.map[bmap],row,col,s*step,(s+1)*step,sm,bar,phase);
  for(int j=0;j<32;++j)p.f[7][size_t(s)*(5*H*D+D*D)+offset+size_t(row+rr(j))*K+col+cc(j)]=v[j];
 }
}
TMN_DEVI void reduce_w(const Params& p,int offset,int count,bf* out){
 for(int i=blockIdx.x*THREADS+threadIdx.x;i<count;i+=gridDim.x*THREADS){float v=0;for(int s=0;s<SPLITS;++s)v+=p.f[7][size_t(s)*(5*H*D+D*D)+offset+i];out[i]=cv(v);}
}

extern "C" __global__ __launch_bounds__(THREADS,2)
void mw_wide512_dwp(__grid_constant__ const Params p){
 extern __shared__ __align__(128) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+2*STAGE);
 if(threadIdx.x==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}
 __syncthreads();int phase=0;
 weight(p,4,3,D,H,0,sm,bar,phase);
}
extern "C" __global__ __launch_bounds__(THREADS,2)
void mw_wide512_dwp_reduce(__grid_constant__ const Params p){reduce_w(p,0,D*H,p.t[21]);}
