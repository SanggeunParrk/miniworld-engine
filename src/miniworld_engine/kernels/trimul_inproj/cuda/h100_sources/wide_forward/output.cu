#if defined(MW_AUX_ONLY)
// Live preparation for the Hopper K1 / contraction / K3 pipeline. No cached
// weight values: every launch reads the original leaves on the current stream.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using bf = __nv_bfloat16;
__device__ float aux_read(const void* p, int i, int fp32) {
  return fp32 ? static_cast<const float*>(p)[i]
              : __bfloat162float(static_cast<const bf*>(p)[i]);
}
__device__ float aux_sum(float v) {
  __shared__ float sums[8];
  for (int q=16;q;q>>=1) v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,q));
  if (!(threadIdx.x&31)) sums[threadIdx.x>>5]=v;
  __syncthreads();
  v=threadIdx.x<8?sums[threadIdx.x]:0.f;
  if (threadIdx.x<32) {
    for (int q=16;q;q>>=1) v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,q));
    if (!threadIdx.x) sums[0]=v;
  }
  __syncthreads();
  v=sums[0];
  __syncthreads();
  return v;
}
struct AuxCast {
  const void* src[6]; bf* dst[6];
  int count[6], cols[6], s0[6], s1[6], fp32[6];
};
struct AuxCopyN {
  const void* src[6]; void* dst[6];
  int count[6], cols[6], s0[6], s1[6], src32[6], dst32[6];
};
// Up to six dense copies (strided source, dtype conversion) in one launch; grid.y selects the matrix.
extern "C" __global__ void mw_aux_copy_n(AuxCopyN p) {
  const int k=blockIdx.y,i=blockIdx.x*256+threadIdx.x;
  if(i>=p.count[k])return;
  const int j=(i/p.cols[k])*p.s0[k]+(i%p.cols[k])*p.s1[k];
  if(p.src32[k]==p.dst32[k]) {
    if(p.src32[k])static_cast<unsigned int*>(p.dst[k])[i]=static_cast<const unsigned int*>(p.src[k])[j];
    else static_cast<unsigned short*>(p.dst[k])[i]=static_cast<const unsigned short*>(p.src[k])[j];
  } else if(p.dst32[k])static_cast<float*>(p.dst[k])[i]=__bfloat162float(static_cast<const bf*>(p.src[k])[j]);
  else static_cast<bf*>(p.dst[k])[i]=__float2bfloat16_rn(static_cast<const float*>(p.src[k])[j]);
}
extern "C" __global__ void mw_aux_dense_copy(const void* src,void* dst,
    int rows,int cols,int s0,int s1,int src32,int dst32) {
  int i=blockIdx.x*256+threadIdx.x;if(i>=rows*cols)return;
  int j=(i/cols)*s0+(i%cols)*s1;
  if(src32==dst32) {
    if(src32)static_cast<unsigned int*>(dst)[i]=static_cast<const unsigned int*>(src)[j];
    else static_cast<unsigned short*>(dst)[i]=static_cast<const unsigned short*>(src)[j];
  } else if(dst32)static_cast<float*>(dst)[i]=__bfloat162float(static_cast<const bf*>(src)[j]);
  else static_cast<bf*>(dst)[i]=__float2bfloat16_rn(static_cast<const float*>(src)[j]);
}
extern "C" __global__ void mw_aux_zero_bytes(unsigned char* dst,int n) {
  for(int i=blockIdx.x*256+threadIdx.x;i<n;i+=gridDim.x*256)dst[i]=0;
}
extern "C" __global__ void mw_aux_cast(AuxCast p) {
  int leaf=blockIdx.y,i=blockIdx.x*256+threadIdx.x;
  if (i<p.count[leaf]) {
    int j=(i/p.cols[leaf])*p.s0[leaf]+(i%p.cols[leaf])*p.s1[leaf];
    p.dst[leaf][i]=__float2bfloat16_rn(aux_read(p.src[leaf],j,p.fp32[leaf]));
  }
}
struct AuxPack {
  bf* dst; const void* w[4]; int rows,cols;
  int s0[4],s1[4],fp32[4];
};
// Autograd steals a dense gradient only when it has its parameter's layout.
// D128 front matrices are column-major leaves; keep this copy in owned CUDA.
struct AuxGradLayout {
  const void* src[6]; void* dst[6];
  int rows[6],cols[6],s0[6],s1[6],fp32[6];
};
extern "C" __global__ void mw_aux_grad_layout(AuxGradLayout p) {
  int leaf=blockIdx.y,i=blockIdx.x*256+threadIdx.x;
  if(i>=p.rows[leaf]*p.cols[leaf])return;
  int j=(i/p.cols[leaf])*p.s0[leaf]+(i%p.cols[leaf])*p.s1[leaf];
  if(p.fp32[leaf])static_cast<unsigned int*>(p.dst[leaf])[j]=static_cast<const unsigned int*>(p.src[leaf])[i];
  else static_cast<unsigned short*>(p.dst[leaf])[j]=static_cast<const unsigned short*>(p.src[leaf])[i];
}
extern "C" __global__ void mw_aux_pack(AuxPack p) {
  int i=blockIdx.x*256+threadIdx.x;
  if(i>=4*p.rows*p.cols)return;
  int row=i/p.cols,col=i%p.cols,source=(row/64)*32+row%32;
  int leaf=2*(source>=p.rows)+(row%64<32),r=source%p.rows;
  p.dst[i]=__float2bfloat16_rn(aux_read(p.w[leaf],r*p.s0[leaf]+col*p.s1[leaf],p.fp32[leaf]));
}
// Column-major bf16 leaves (unit row stride): a 32 x 32 tile is read along the leaf rows and written along the packed columns, so both
// sides are coalesced. Same element mapping as mw_aux_pack; rows and cols are multiples of 32.
extern "C" __global__ void mw_aux_pack_t(AuxPack p) {
  __shared__ unsigned short tile[32][34];
  const int row0=blockIdx.x*32,ct=blockIdx.y,tx=threadIdx.x,ty=threadIdx.y;
  const int base=(row0/64)*32,leaf=2*(base>=p.rows)+(row0%64<32),r0=base%p.rows;
  const unsigned short* src=static_cast<const unsigned short*>(p.w[leaf]);
  const int ld=p.s1[leaf];
#pragma unroll
  for(int k=0;k<4;++k)tile[tx][ty+8*k]=src[(size_t)(ct*32+ty+8*k)*ld+(r0+tx)];
  __syncthreads();
  unsigned short* dst=reinterpret_cast<unsigned short*>(p.dst);
#pragma unroll
  for(int k=0;k<4;++k)dst[(size_t)(row0+ty+8*k)*p.cols+ct*32+tx]=tile[ty+8*k][tx];
}
extern "C" __global__ void mw_aux_pack_vec(AuxPack p) {
  int i=blockIdx.x*256+threadIdx.x,cols=p.cols/8;
  if(i>=4*p.rows*cols)return;
  int row=i/cols,col=i%cols,source=(row/64)*32+row%32;
  int leaf=2*(source>=p.rows)+(row%64<32),r=source%p.rows;
  reinterpret_cast<uint4*>(p.dst)[i]=reinterpret_cast<const uint4*>(p.w[leaf])[r*cols+col];
}
extern "C" __global__ void mw_aux_pack_training_pair(bf* packed,bf* stacked,
    const bf* wg,const bf* wl,const bf* wlg,const bf* wr,const bf* wrg,int rows,int width){
  const int i=blockIdx.x*256+threadIdx.x,cols=width/8,gate=width*cols;
  if(i<gate){reinterpret_cast<uint4*>(stacked)[i]=reinterpret_cast<const uint4*>(wg)[i];return;}
  const int q=i-gate;
  if(q>=4*rows*cols)return;
  const int leaf=q/(rows*cols),r=(q/cols)%rows,c=q%cols;
  const bf* src=leaf==0?wl:leaf==1?wlg:leaf==2?wr:wrg;
  const uint4 value=reinterpret_cast<const uint4*>(src)[r*cols+c];
  reinterpret_cast<uint4*>(stacked)[i]=value;
  const int source=(leaf/2)*rows+r;
  const int outrow=(source/32)*64+source%32+(leaf%2?0:32);
  reinterpret_cast<uint4*>(packed)[outrow*cols+c]=value;
}
extern "C" __global__ void mw_aux_norm(const bf* x,const float* g,const float* b,
                                        bf* y,float* stats,int width) {
  int row=blockIdx.x; float sum=0.f;
  for(int c=threadIdx.x;c<width;c+=256)sum=__fadd_rn(sum,__bfloat162float(x[row*width+c]));
  float mean=aux_sum(sum)/width,var=0.f;
  for(int c=threadIdx.x;c<width;c+=256){
    float z=__bfloat162float(x[row*width+c])-mean;
    var=__fadd_rn(var,__fmul_rn(z,z));
  }
  float rs=rsqrtf(aux_sum(var)/width+1e-5f);
  if(!threadIdx.x&&stats){stats[2*row]=mean;stats[2*row+1]=rs;}
  for(int c=threadIdx.x;c<width;c+=256)
    y[row*width+c]=__float2bfloat16_rn(fmaf((__bfloat162float(x[row*width+c])-mean)*rs,g[c],b[c]));
}
// Eight rows per CTA, with inputs retained through both warp reductions.
// Keep mw_aux_norm above as the frozen CTA-per-row comparator.
__device__ float aux_warp_sum(float v) {
  for(int q=16;q;q>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,q));
  return v;
}
extern "C" __global__ void mw_aux_norm_warp(const bf* x,const float* g,const float* b,
                                             bf* y,float* stats,int width,int rows) {
  const int row=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  float values[16],sum=0.f;
  #pragma unroll
  for(int j=0;j<16;++j){
    int c=lane+32*j;
    values[j]=c<width?__bfloat162float(x[row*width+c]):0.f;
    sum=__fadd_rn(sum,values[j]);
  }
  const float mean=aux_warp_sum(sum)/width;
  float var=0.f;
  #pragma unroll
  for(int j=0;j<16;++j)if(lane+32*j<width){
    float z=values[j]-mean;var=__fadd_rn(var,__fmul_rn(z,z));
  }
  const float rs=rsqrtf(aux_warp_sum(var)/width+1e-5f);
  if(!lane&&stats){stats[2*row]=mean;stats[2*row+1]=rs;}
  #pragma unroll
  for(int j=0;j<16;++j){
    int c=lane+32*j;
    if(c<width)y[row*width+c]=__float2bfloat16_rn(fmaf((values[j]-mean)*rs,g[c],b[c]));
  }
}
extern "C" __global__ void mw_aux_fold(const bf* wp,const float* go,const float* bo,
                                        bf* wpf,const bf* wg,const float* gi,const float* bi,
                                        bf* wgf,float* uv,int width,int hidden) {
  int c=blockIdx.x; bool gate=blockIdx.y!=0;
  int kmax=gate?width:hidden; float u=0.f,v=0.f;
  const bf* w=gate?wg:wp; const float* gamma=gate?gi:go; const float* beta=gate?bi:bo;
  bf* folded=gate?wgf:wpf;
  for(int k=threadIdx.x;k<kmax;k+=256){
    float a=__bfloat162float(w[c*kmax+k]);
    bf f=__float2bfloat16_rn(__fmul_rn(a,gamma[k]));
    folded[c*kmax+k]=f;
    u=__fadd_rn(u,__bfloat162float(f)); v=__fadd_rn(v,__fmul_rn(a,beta[k]));
  }
  float us=aux_sum(u),vs=aux_sum(v);
  if(!threadIdx.x){uv[(gate?2:0)*width+c]=us;uv[(gate?3:1)*width+c]=vs;}
}
extern "C" __global__ void mw_aux_mask(const bool* token,bf* pair,int length) {
  int i=blockIdx.x*256+threadIdx.x;
  if(i<length*length)pair[i]=__float2bfloat16_rn(!token||(token[i/length]&&token[i%length])?1.f:0.f);
}
__device__ float aux_mask_read(const void* p,int i,int kind) {
  return kind==1?float(static_cast<const bool*>(p)[i]):
         kind==2?__bfloat162float(static_cast<const bf*>(p)[i]):static_cast<const float*>(p)[i];
}
// Batched samples (MiniWorld templates): `samples` pair masks [B][L][L], b-major; a token mask is [B L] (element i = (s L + a) L + b reads
// tokens s L + a and s L + b), a pair-mask input is [B L L] and read in place. samples = 1 is the single-sample contract.
__device__ __forceinline__ float aux_pair_mask(const void* mask,int i,int length,int kind,int token,int samples) {
  return kind==0?1.f:token?
    float(aux_mask_read(mask,i/length,kind)!=0.f&&aux_mask_read(mask,(samples>1?i/(length*length)*length:0)+i%length,kind)!=0.f):
    aux_mask_read(mask,i,kind);
}
extern "C" __global__ void mw_aux_train_mask(const void* mask,float* pair,int length,int kind,int token,int samples) {
  int i=blockIdx.x*256+threadIdx.x;
  if(i>=samples*length*length)return;
  pair[i]=aux_pair_mask(mask,i,length,kind,token,samples);
}
extern "C" __global__ void mw_aux_unit_scale(bf* scale,int count) {
  int i=blockIdx.x*256+threadIdx.x;
  if(i<count)scale[i]=__float2bfloat16_rn(1.f);
}
struct AuxD64TrainPrepare {
  const void* w[6]; bf *packed,*gate,*output; const void* mask; float* pair;
  int s0[6],s1[6],fp32[6],hidden,length,mask_kind,token,samples;   // samples: B pair masks [B][L][L] (see aux_pair_mask)
};
extern "C" __global__ void mw_aux_d64_training_prepare(AuxD64TrainPrepare p) {
  const int i=blockIdx.x*256+threadIdx.x;
  if(i<4*p.hidden*64) {
    const int row=i/64,col=i%64,source=(row/64)*32+row%32;
    const int leaf=2*(source>=p.hidden)+(row%64<32),r=source%p.hidden;
    p.packed[i]=__float2bfloat16_rn(aux_read(p.w[leaf],r*p.s0[leaf]+col*p.s1[leaf],p.fp32[leaf]));
  }
  if(i<64*64)
    p.gate[i]=__float2bfloat16_rn(aux_read(p.w[4],(i/64)*p.s0[4]+(i%64)*p.s1[4],p.fp32[4]));
  if(i<64*p.hidden)
    p.output[i]=__float2bfloat16_rn(aux_read(p.w[5],(i/p.hidden)*p.s0[5]+(i%p.hidden)*p.s1[5],p.fp32[5]));
  if(i<p.samples*p.length*p.length)
    p.pair[i]=aux_pair_mask(p.mask,i,p.length,p.mask_kind,p.token,p.samples);
}
// Deterministic normalization backward shared by the general Hopper training
// buckets. Two bounded reductions replace atomic affine updates. The output
// operand stays channel-major; no full-size transpose or affine partial exists.
extern "C" __global__ void mw_aux_ln_backward(const bf* x,const bf* dy,
    const float* gamma,const bf* residual,bf* dx,float* mean,float* rstd,
    int rows,int width,int planes,int saved_stats) {
  int row=blockIdx.x; float mu=0.f,rs=0.f;
  if(saved_stats){mu=mean[row];rs=rstd[row];}
  else {
    float v=0.f;
    for(int c=threadIdx.x;c<width;c+=256)v+=__bfloat162float(x[row*width+c]);
    mu=aux_sum(v)/width;v=0.f;
    for(int c=threadIdx.x;c<width;c+=256){float z=__bfloat162float(x[row*width+c])-mu;v+=z*z;}
    rs=rsqrtf(aux_sum(v)/width+1e-5f);
    if(!threadIdx.x){mean[row]=mu;rstd[row]=rs;}
  }
  float s1=0.f,s2=0.f;
  for(int c=threadIdx.x;c<width;c+=256){
    int i=planes?c*rows+row:row*width+c;
    float z=(__bfloat162float(x[i])-mu)*rs,dd=__bfloat162float(dy[row*width+c])*gamma[c];
    s1+=dd;s2+=dd*z;
  }
  s1=aux_sum(s1)/width;s2=aux_sum(s2)/width;
  for(int c=threadIdx.x;c<width;c+=256){
    int i=planes?c*rows+row:row*width+c;
    float z=(__bfloat162float(x[i])-mu)*rs,dd=__bfloat162float(dy[row*width+c])*gamma[c];
    float v=rs*(dd-s1-z*s2);
    if(residual)v+=__bfloat162float(residual[row*width+c]);
    dx[i]=__float2bfloat16_rn(v);
  }
}
extern "C" __global__ void mw_aux_ln_affine_partial(const bf* x,const bf* dy,
    const float* mean,const float* rstd,float* partial,int rows,int width,int planes){
  const int c=blockIdx.x,start=blockIdx.y*4096,end=min(start+4096,rows);
  float dg=0.f,db=0.f;
  for(int r=start+threadIdx.x;r<end;r+=256){
    float v=__bfloat162float(dy[r*width+c]);
    float z=(__bfloat162float(x[planes?c*rows+r:r*width+c])-mean[r])*rstd[r];
    dg+=v*z;db+=v;
  }
  dg=aux_sum(dg);db=aux_sum(db);
  if(!threadIdx.x){partial[(blockIdx.y*2)*width+c]=dg;partial[(blockIdx.y*2+1)*width+c]=db;}
}
extern "C" __global__ void mw_aux_ln_affine_finish(const float* partial,float* dg,float* db,int width,int blocks){
  int c=blockIdx.x*256+threadIdx.x;if(c>=width)return;
  float g=0.f,b=0.f;
  for(int k=0;k<blocks;++k){g+=partial[2*k*width+c];b+=partial[(2*k+1)*width+c];}
  dg[c]=g;db[c]=b;
}
struct AuxStack {const bf* w[5];bf* dst;int rows[5],s0[5],s1[5],width;};
// Fixed CTA-index reduction of the selected TMA LayerNorm kernels' affine sums.
extern "C" __global__ void mw_aux_affine_cta_finish(const float* pg,const float* pb,
    float* dg,float* db,int width,int blocks){
  const int lane=threadIdx.x%32,warp=threadIdx.x/32,c=blockIdx.x*32+lane;
  __shared__ float sums[2][4][32];
  float g=0.f,b=0.f;
  if(c<width)for(int k=warp;k<blocks;k+=4){g+=pg[k*width+c];b+=pb[k*width+c];}
  sums[0][warp][lane]=g;sums[1][warp][lane]=b;__syncthreads();
  if(warp==0&&c<width){
    g=0.f;b=0.f;
    #pragma unroll
    for(int w=0;w<4;++w){g+=sums[0][w][lane];b+=sums[1][w][lane];}
    dg[c]=g;db[c]=b;
  }
}
extern "C" __global__ void mw_aux_stack(AuxStack p){
  int leaf=blockIdx.y,i=blockIdx.x*256+threadIdx.x;
  if(i>=p.rows[leaf]*p.width)return;
  int offset=0;for(int k=0;k<leaf;++k)offset+=p.rows[k]*p.width;
  p.dst[offset+i]=p.w[leaf][(i/p.width)*p.s0[leaf]+(i%p.width)*p.s1[leaf]];
}
extern "C" __global__ void mw_aux_stack_vec(AuxStack p){
  int leaf=blockIdx.y,i=blockIdx.x*256+threadIdx.x;
  if(i>=p.rows[leaf]*(p.width/8))return;
  int offset=0;for(int k=0;k<leaf;++k)offset+=p.rows[k]*(p.width/8);
  reinterpret_cast<uint4*>(p.dst)[offset+i]=reinterpret_cast<const uint4*>(p.w[leaf])[i];
}
extern "C" __global__ void mw_aux_affine_cta_finish_parallel(const float* pg,const float* pb,
    float* dg,float* db,int width,int blocks){
  const int tid=threadIdx.x,lane=tid%32,warp=tid/32,c=blockIdx.x*4+(tid%4);
  __shared__ float sums[2][4][4];
  float g=0.f,b=0.f;
  if(c<width)for(int k=tid/4;k<blocks;k+=32){g+=pg[k*width+c];b+=pb[k*width+c];}
  #pragma unroll
  for(int off=16;off>=4;off/=2){g+=__shfl_down_sync(0xffffffff,g,off);b+=__shfl_down_sync(0xffffffff,b,off);}
  if(lane<4){sums[0][warp][lane]=g;sums[1][warp][lane]=b;}
  __syncthreads();
  if(tid<4&&c<width){
    g=0.f;b=0.f;
    #pragma unroll
    for(int w=0;w<4;++w){g+=sums[0][w][tid];b+=sums[1][w][tid];}
    dg[c]=g;db[c]=b;
  }
}
extern "C" __global__ void mw_aux_affine_pair_finish(
    const float* pg0,const float* pb0,float* dg0,float* db0,
    const float* pg1,const float* pb1,float* dg1,float* db1,
    int width0,int blocks0,int width1,int blocks1){
  const bool second=int(blockIdx.x)>=width0/4;
  const int width=second?width1:width0,blocks=second?blocks1:blocks0;
  const float* pg=second?pg1:pg0;const float* pb=second?pb1:pb0;
  float* dg=second?dg1:dg0;float* db=second?db1:db0;
  const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
  const int c=(int(blockIdx.x)-(second?width0/4:0))*4+tid%4;
  __shared__ float sums[2][4][4];float g=0.f,b=0.f;
  for(int k=tid/4;k<blocks;k+=32){g+=pg[k*width+c];b+=pb[k*width+c];}
  #pragma unroll
  for(int off=16;off>=4;off/=2){g+=__shfl_down_sync(0xffffffff,g,off);b+=__shfl_down_sync(0xffffffff,b,off);}
  if(lane<4){sums[0][warp][lane]=g;sums[1][warp][lane]=b;}
  __syncthreads();
  if(tid<4){g=0.f;b=0.f;
    #pragma unroll
    for(int w=0;w<4;++w){g+=sums[0][w][tid];b+=sums[1][w][tid];}
    dg[c]=g;db[c]=b;
  }
}
// samples: B b-major pair masks [B n][n] from a token mask [B n] or a pair mask [B n n] (aux_pair_mask); 1 = one sample.
extern "C" __global__ void mw_aux_owned_mask(const void* src,bf* dst,int n,int kind,int token,int samples){
  int i=blockIdx.x*256+threadIdx.x;if(i>=samples*n*n)return;
  dst[i]=__float2bfloat16_rn(aux_pair_mask(src,i,n,kind,token,samples));
}
#else
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
struct Params {CUtensorMap tri,xn,wp,wg; const bf *x,*ds; bf *y; const float *gamma,*beta; int M,L;};
TMN_DEVI float rd(const bf* p,int i){return __bfloat162float(p[i]);}
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,q));return v;}
TMN_DEVI void mma(float (&v)[32],uint64_t a,uint64_t b,int ac){
 asm volatile("{.reg .pred p;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]) : "l"(a),"l"(b),"r"(ac));
}
TMN_DEVI uint32_t load_shared_word(uint32_t a){uint32_t v;asm("ld.shared.b32 %0,[%1];":"=r"(v):"r"(a));return v;}
TMN_DEVI void store_shared_word(uint32_t a,uint32_t v){asm volatile("st.shared.b32 [%0],%1;"::"r"(a),"r"(v):"memory");}
template<bool GATE> TMN_DEVI void product(const Params& p,float (&v)[32],int row,int col,uint8_t* sx,uint8_t* buf,uint64_t* bars,int& phase){
 constexpr int K=GATE?D:H;
 auto load=[&](int k,int slot){
  mbar_arrive_expect_tx(bars+slot,8192*KC*(G+(GATE?1:0)));
  for(int g=0;g<G;++g)for(int q=0;q<KC;++q)tma_load_2d(buf+slot*STAGE+8192*(g*KC+q),GATE?&p.wg:&p.wp,bars+slot,k+64*q,col+64*g);
  if constexpr(GATE)for(int q=0;q<KC;++q)tma_load_2d(buf+slot*STAGE+8192*(G*KC+q),&p.xn,bars+slot,k+64*q,row);
 };
 for(int j=0;j<32;++j)v[j]=0;
 if(threadIdx.x==0)load(0,0);
 for(int k=0,it=0;k<K;k+=64*KC,++it){
  int slot=it&1;
  if(k+64*KC<K&&threadIdx.x==0)load(k+64*KC,1-slot);
  mbar_wait(bars+slot,(phase>>slot)&1);phase^=1<<slot;__syncthreads();
  uint8_t* a=GATE?buf+slot*STAGE+8192*G*KC:sx+(k/64)*8192;
  uint8_t* b=buf+slot*STAGE+8192*KC*(threadIdx.x/128);
  fence_regs(v);wgmma_fence();
  #pragma unroll
  for(int q=0;q<4*KC;++q)mma(v,smem_desc(smem_u32(a+8192*(q/4)+32*(q%4)),16,1024,1),smem_desc(smem_u32(b+8192*(q/4)+32*(q%4)),16,1024,1),k>0||q>0);
  wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
 }
}
extern "C" __global__ __launch_bounds__(NT,1) void mw_wide_output(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 uint8_t* sx=sm;uint8_t* buf=sm+XBYTES;
 float* stats=reinterpret_cast<float*>(buf+2*STAGE);uint64_t* bar=reinterpret_cast<uint64_t*>(stats+128);
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
   float rs=rsqrtf(sumwarp(s)/H+1e-5f);
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
   float rs=rsqrtf(sumwarp(s)/H+1e-5f);
   #pragma unroll 1
   for(int c=2*lane;c<H;c+=64){uint32_t addr=smem_u32(sx+(c/64)*8192)+swz128(r,(c%64)*2);uint32_t v=load_shared_word(addr);float a=fmaf((bf16lo(v)-mu)*rs,p.gamma[c],p.beta[c]),b=fmaf((bf16hi(v)-mu)*rs,p.gamma[c+1],p.beta[c+1]);store_shared_word(addr,pack_bf16(a,b));}
  }
#endif
  fence_proxy_async();__syncthreads();
  for(int col=0;col<D;col+=64*G){
   float proj[32],gate[32];product<false>(p,proj,row,col,sx,buf,bar,phase);product<true>(p,gate,row,col,sx,buf,bar,phase);
   #pragma unroll
   for(int j=0;j<32;++j){int r=((tid/32)%4)*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2)+col+64*(tid/128);if(c<D){size_t ix=size_t(row+r)*D+c;float v=math::round_bf16(proj[j])*math::sigmoid(math::round_bf16(gate[j]));p.y[ix]=__float2bfloat16_rn(fmaf(v,rd(p.ds,((row+r)%p.L)*D+c),__bfloat162float(p.x[ix])));}}
  }
  __syncthreads();
 }
}
#endif
