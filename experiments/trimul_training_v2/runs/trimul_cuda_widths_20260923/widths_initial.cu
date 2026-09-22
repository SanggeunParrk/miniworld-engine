// SPDX-License-Identifier: Apache-2.0
// Width-scalable training extension of Anthropic Hopper TMA/WGMMA primitives.
// D128 policies remain available. Workspace is transient, not forward activations.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;
using bf=__nv_bfloat16;
constexpr int D=WIDTH,H=2*D,SPLITS=8;
struct Params{CUtensorMap map[16];bf* t[24];float* f[12];int M,L;};
// t: x,tri,dy,ds,y,xn,norm,dp,dg,dn,dxn,dx,dtri,gp0..3,dw0..3,dwp,dwg,unused
// f: gi,bi,go,bo,mask,mu,rs,partw,ln0,ln1,ln2,ln3
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
 for(int k=begin;k<end;k+=64){
  if(threadIdx.x==0){mbar_arrive_expect_tx(bar,16384);tma_load_2d(sm,A,bar,TA?row:k,TA?k:row);tma_load_2d(sm+8192,B,bar,TB?col:k,TB?k:col);}
  mbar_wait(bar,phase);phase^=1;__syncthreads();fence_regs(acc);wgmma_fence();
  #pragma unroll
  for(int q=0;q<4;++q)mma<TA,TB>(acc,smem_desc(smem_u32(sm+(TA?q*2048:q*32)),16,1024,1),smem_desc(smem_u32(sm+8192+(TB?q*2048:q*32)),16,1024,1),add||k>begin||q>0);
  wgmma_commit();wgmma_wait<0>();fence_regs(acc);__syncthreads();
 }
}
TMN_DEVI int rr(int j){return (threadIdx.x/32)*16+(threadIdx.x%32)/4+8*((j/2)&1);}
TMN_DEVI int cc(int j){return (j/8)*16+2*(threadIdx.x%4)+8*((j/2)%4/2)+(j%2);}
TMN_DEVI void store(bf* out,float (&v)[32],int row,int col,int cols){
 for(int j=0;j<32;++j)out[size_t(row+rr(j))*cols+col+cc(j)]=cv(v[j]);
}
// Per-warp LN. Channel-first tri loads, row-major normalized output.
TMN_DEVI void norm(const Params& p){
 int lane=threadIdx.x%32,warp=threadIdx.x/32;
 for(int row=blockIdx.x*4+warp;row<p.M;row+=gridDim.x*4){
  float sum=0;for(int c=lane;c<H;c+=32)sum+=rd(p.t[1],size_t(c)*p.M+row);float mu=wsum(sum)/H;
  float var=0;for(int c=lane;c<H;c+=32){float z=rd(p.t[1],size_t(c)*p.M+row)-mu;var+=z*z;}float rs=rsqrtf(wsum(var)/H+1e-5f);
  if(lane==0){p.f[5][row]=mu;p.f[6][row]=rs;}
  for(int c=lane;c<H;c+=32)p.t[6][size_t(row)*H+c]=cv((rd(p.t[1],size_t(c)*p.M+row)-mu)*rs*p.f[2][c]+p.f[3][c]);
 }
}
template<bool FWD> TMN_DEVI void out_gp(const Params& p,uint8_t* sm,uint64_t* bar,int& phase){
 for(int tile=blockIdx.x;tile<(p.M/64)*(D/64);tile+=gridDim.x){
  int row=(tile/(D/64))*64,col=(tile%(D/64))*64;float proj[32],gate[32];
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
 int nt=N/64,kt=K/64,tiles=nt*kt,step=p.M/SPLITS;
 for(int t=blockIdx.x;t<tiles*SPLITS;t+=gridDim.x){int s=t/tiles,u=t%tiles,row=(u/kt)*64,col=(u%kt)*64;float v[32];gemm<1,1>(v,&p.map[amap],&p.map[bmap],row,col,s*step,(s+1)*step,sm,bar,phase);
  for(int j=0;j<32;++j)p.f[7][size_t(s)*11*D*D+offset+size_t(row+rr(j))*K+col+cc(j)]=v[j];
 }
}
TMN_DEVI void reduce_w(const Params& p,int offset,int count,bf* out){
 for(int i=blockIdx.x*128+threadIdx.x;i<count;i+=gridDim.x*128){float v=0;for(int s=0;s<SPLITS;++s)v+=p.f[7][size_t(s)*11*D*D+offset+i];out[i]=cv(v);}
}
template<bool INPUT> TMN_DEVI void ln_bwd(const Params& p){
 constexpr int C=INPUT?D:H;int lane=threadIdx.x%32,warp=threadIdx.x/32;float gg[C/32]={},bb[C/32]={};
 for(int row=blockIdx.x*4+warp;row<p.M;row+=gridDim.x*4){
  float mu,rs;
  if constexpr(INPUT){float z=0;for(int c=lane;c<C;c+=32)z+=rd(p.t[0],size_t(row)*C+c);mu=wsum(z)/C;z=0;for(int c=lane;c<C;c+=32){float v=rd(p.t[0],size_t(row)*C+c)-mu;z+=v*v;}rs=rsqrtf(wsum(z)/C+1e-5f);}
  else{mu=p.f[5][row];rs=p.f[6][row];}
  float s0=0,s1=0;
  for(int c=lane;c<C;c+=32){float z=(rd(INPUT?p.t[0]:p.t[1],INPUT?size_t(row)*C+c:size_t(c)*p.M+row)-mu)*rs,dy=rd(INPUT?p.t[10]:p.t[9],size_t(row)*C+c);float v=dy*p.f[INPUT?0:2][c];s0+=v;s1+=v*z;gg[c/32]+=dy*z;bb[c/32]+=dy;}
  s0=wsum(s0)/C;s1=wsum(s1)/C;
  for(int c=lane;c<C;c+=32){float z=(rd(INPUT?p.t[0]:p.t[1],INPUT?size_t(row)*C+c:size_t(c)*p.M+row)-mu)*rs,v=rd(INPUT?p.t[10]:p.t[9],size_t(row)*C+c)*p.f[INPUT?0:2][c];float dx=(v-s0-z*s1)*rs;
   if constexpr(INPUT)p.t[11][size_t(row)*C+c]=cv(dx+rd(p.t[2],size_t(row)*C+c));else p.t[12][size_t(c)*p.M+row]=cv(dx);
  }
 }
 for(int c=lane;c<C;c+=32){atomicAdd(p.f[INPUT?8:10]+c,gg[c/32]);atomicAdd(p.f[INPUT?9:11]+c,bb[c/32]);}
}
TMN_DEVI void front_gp(const Params& p,uint8_t* sm,uint64_t* bar,int& phase){
 for(int t=blockIdx.x;t<(p.M/64)*(H/64)*2;t+=gridDim.x){int side=t/((p.M/64)*(H/64)),u=t%((p.M/64)*(H/64)),row=(u/(H/64))*64,col=(u%(H/64))*64;float proj[32],gate[32];
  gemm<0,0>(proj,&p.map[0],&p.map[10+2*side],row,col,0,D,sm,bar,phase);gemm<0,0>(gate,&p.map[0],&p.map[11+2*side],row,col,0,D,sm,bar,phase);
  // t[22] and t[23] are channel-first dleft/dright in B7; dWgate uses t[22] only in B1.
  for(int j=0;j<32;++j){int r=row+rr(j),c=col+cc(j);float da=rn(rd(p.t[22+side],size_t(c)*p.M+r)*p.f[4][r]),g=math::sigmoid(rn(gate[j])),v=rn(proj[j]);p.t[13+side*2][size_t(r)*H+c]=cv(da*g);p.t[14+side*2][size_t(r)*H+c]=cv(((da*v)*g)*(1-g));}
 }
}
TMN_DEVI void front_dx(const Params& p,uint8_t* sm,uint64_t* bar,int& phase){
 for(int t=blockIdx.x;t<(p.M/64)*(D/64);t+=gridDim.x){int row=(t/(D/64))*64,col=(t%(D/64))*64;float v[32];
  gemm<0,1>(v,&p.map[5],&p.map[2],row,col,0,D,sm,bar,phase);
  for(int i=0;i<4;++i)gemm<0,1>(v,&p.map[6+i],&p.map[10+i],row,col,0,H,sm,bar,phase,true);
  store(p.t[10],v,row,col,D);
 }
}
extern "C" __global__ __launch_bounds__(128,2) void width_forward(__grid_constant__ const Params p){
 extern __shared__ __align__(128) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+16384);if(threadIdx.x==0)mbar_init(bar,1);__syncthreads();int phase=0;auto grid=cooperative_groups::this_grid();norm(p);__threadfence();asm volatile("fence.proxy.async.global;":::"memory");grid.sync();out_gp<true>(p,sm,bar,phase);
}
extern "C" __global__ __launch_bounds__(128,2) void width_b1(__grid_constant__ const Params p){
 extern __shared__ __align__(128) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+16384);if(threadIdx.x==0)mbar_init(bar,1);__syncthreads();int phase=0;auto grid=cooperative_groups::this_grid();
 for(int c=blockIdx.x*128+threadIdx.x;c<H;c+=gridDim.x*128){p.f[10][c]=0;p.f[11][c]=0;}norm(p);__threadfence();asm volatile("fence.proxy.async.global;":::"memory");grid.sync();out_gp<false>(p,sm,bar,phase);__threadfence();asm volatile("fence.proxy.async.global;":::"memory");grid.sync();
 for(int t=blockIdx.x;t<(p.M/64)*(H/64);t+=gridDim.x){int row=(t/(H/64))*64,col=(t%(H/64))*64;float v[32];gemm<0,1>(v,&p.map[4],&p.map[1],row,col,0,D,sm,bar,phase);store(p.t[9],v,row,col,H);}
 weight(p,4,3,D,H,0,sm,bar,phase);weight(p,5,0,D,D,2*D*D,sm,bar,phase);__threadfence();asm volatile("fence.proxy.async.global;":::"memory");grid.sync();ln_bwd<false>(p);reduce_w(p,0,D*H,p.t[21]);reduce_w(p,2*D*D,D*D,p.t[22]);
}
extern "C" __global__ __launch_bounds__(128,2) void width_b7(__grid_constant__ const Params p){
 extern __shared__ __align__(128) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+16384);if(threadIdx.x==0)mbar_init(bar,1);__syncthreads();int phase=0;auto grid=cooperative_groups::this_grid();
 for(int c=blockIdx.x*128+threadIdx.x;c<D;c+=gridDim.x*128){p.f[8][c]=0;p.f[9][c]=0;}front_gp(p,sm,bar,phase);__threadfence();asm volatile("fence.proxy.async.global;":::"memory");grid.sync();front_dx(p,sm,bar,phase);
 for(int i=0;i<4;++i)weight(p,6+i,0,H,D,(3+2*i)*D*D,sm,bar,phase);__threadfence();asm volatile("fence.proxy.async.global;":::"memory");grid.sync();ln_bwd<true>(p);for(int i=0;i<4;++i)reduce_w(p,(3+2*i)*D*D,H*D,p.t[17+i]);
}
// Unit probe: all transpose cases are compared against torch matmul before integration.
extern "C" __global__ __launch_bounds__(128,2) void width_probe(__grid_constant__ const Params p){
 extern __shared__ __align__(128) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+16384);if(threadIdx.x==0)mbar_init(bar,1);__syncthreads();int phase=0;float v[32];int ta=p.L/2,tb=p.L%2;
 if(ta==0&&tb==0)gemm<0,0>(v,&p.map[0],&p.map[1],0,0,0,128,sm,bar,phase);
 if(ta==0&&tb==1)gemm<0,1>(v,&p.map[0],&p.map[1],0,0,0,128,sm,bar,phase);
 if(ta==1&&tb==0)gemm<1,0>(v,&p.map[0],&p.map[1],0,0,0,128,sm,bar,phase);
 if(ta==1&&tb==1)gemm<1,1>(v,&p.map[0],&p.map[1],0,0,0,128,sm,bar,phase);
 store(p.t[0],v,0,0,64);
}
