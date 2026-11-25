#if defined(MW_AUX_ONLY)
// Live preparation for the Hopper K1 / contraction / K3 pipeline. No cached
// weight values: every launch reads the original leaves on the current stream.
//
// Address arithmetic is 64-bit throughout: element offsets / counts are int64_t, products are formed
// after the widening cast. Elementwise kernels that split a flat index with a division (copy, mask)
// pick the index type per launch: a uniform branch runs the 32-bit body while the whole range fits
// in int32 (no 64-bit division in the common case) and the int64 body past it.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
using bf = __nv_bfloat16;
__device__ __forceinline__ int64_t aux_tid(int per_block=256) {
  return static_cast<int64_t>(blockIdx.x)*per_block+threadIdx.x;
}
__device__ float aux_read(const void* p, int64_t i, int fp32) {
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
// One strided element copy with dtype conversion: destination i (dense, row-major [., cols]) <- source (i / cols, i % cols).
template<class I> __device__ __forceinline__ void aux_copy_one(const void* src,void* dst,I i,int cols,int s0,int s1,int src32,int dst32) {
  const I r=i/cols;
  const int64_t j=static_cast<int64_t>(r)*s0+static_cast<int64_t>(i-r*cols)*s1;
  if(src32==dst32) {
    if(src32)static_cast<unsigned int*>(dst)[i]=static_cast<const unsigned int*>(src)[j];
    else static_cast<unsigned short*>(dst)[i]=static_cast<const unsigned short*>(src)[j];
  } else if(dst32)static_cast<float*>(dst)[i]=__bfloat162float(static_cast<const bf*>(src)[j]);
  else static_cast<bf*>(dst)[i]=__float2bfloat16_rn(static_cast<const float*>(src)[j]);
}
struct AuxCopyN {
  const void* src[6]; void* dst[6];
  int64_t count[6];                                   // elements of each matrix (int64: the launcher wraps them as i64)
  int cols[6], s0[6], s1[6], src32[6], dst32[6];
};
// Up to six dense copies (strided source, dtype conversion) in one launch; grid.y selects the matrix.
extern "C" __global__ void mw_aux_copy_n(AuxCopyN p) {
  const int k=blockIdx.y;const int64_t i=aux_tid();
  if(i>=p.count[k])return;
  if(p.count[k]<=INT32_MAX)aux_copy_one<int>(p.src[k],p.dst[k],static_cast<int>(i),p.cols[k],p.s0[k],p.s1[k],p.src32[k],p.dst32[k]);
  else aux_copy_one<int64_t>(p.src[k],p.dst[k],i,p.cols[k],p.s0[k],p.s1[k],p.src32[k],p.dst32[k]);
}
extern "C" __global__ void mw_aux_dense_copy(const void* src,void* dst,
    int rows,int cols,int s0,int s1,int src32,int dst32) {
  const int64_t i=aux_tid(),count=static_cast<int64_t>(rows)*cols;if(i>=count)return;
  if(count<=INT32_MAX)aux_copy_one<int>(src,dst,static_cast<int>(i),cols,s0,s1,src32,dst32);
  else aux_copy_one<int64_t>(src,dst,i,cols,s0,s1,src32,dst32);
}
// Two builds: the 32-bit grid-stride loop below 2**31 bytes (an int64 loop counter cost 4-7 % at 64 MB), the int64 one past it
// (the launcher, `_h100_pack.zero_native_`, picks by size).
extern "C" __global__ void mw_aux_zero_bytes(unsigned char* dst,int n) {
  for(int i=blockIdx.x*256+threadIdx.x;i<n;i+=gridDim.x*256)dst[i]=0;
}
extern "C" __global__ void mw_aux_zero_bytes_wide(unsigned char* dst,int64_t n) {
  for(int64_t i=aux_tid();i<n;i+=static_cast<int64_t>(gridDim.x)*256)dst[i]=0;
}
struct AuxPack {
  bf* dst; const void* w[4]; int rows,cols;
  int s0[4],s1[4],fp32[4];
};
extern "C" __global__ void mw_aux_pack(AuxPack p) {
  const int64_t i=aux_tid();
  if(i>=4*static_cast<int64_t>(p.rows)*p.cols)return;
  const int64_t row=i/p.cols;const int col=static_cast<int>(i-row*p.cols);
  const int64_t source=(row/64)*32+row%32;
  const int leaf=2*(source>=p.rows)+(row%64<32);const int64_t r=source%p.rows;
  p.dst[i]=__float2bfloat16_rn(aux_read(p.w[leaf],r*p.s0[leaf]+static_cast<int64_t>(col)*p.s1[leaf],p.fp32[leaf]));
}
extern "C" __global__ void mw_aux_pack_vec(AuxPack p) {
  const int64_t i=aux_tid();const int cols=p.cols/8;
  if(i>=4*static_cast<int64_t>(p.rows)*cols)return;
  const int64_t row=i/cols;const int col=static_cast<int>(i-row*cols);
  const int64_t source=(row/64)*32+row%32;
  const int leaf=2*(source>=p.rows)+(row%64<32);const int64_t r=source%p.rows;
  reinterpret_cast<uint4*>(p.dst)[i]=reinterpret_cast<const uint4*>(p.w[leaf])[r*cols+col];
}
extern "C" __global__ void mw_aux_pack_training_pair(bf* packed,bf* stacked,
    const bf* wg,const bf* wl,const bf* wlg,const bf* wr,const bf* wrg,int rows,int width){
  const int64_t i=aux_tid();const int cols=width/8;const int64_t gate=static_cast<int64_t>(width)*cols;
  if(i<gate){reinterpret_cast<uint4*>(stacked)[i]=reinterpret_cast<const uint4*>(wg)[i];return;}
  const int64_t q=i-gate,plane=static_cast<int64_t>(rows)*cols;
  if(q>=4*plane)return;
  const int leaf=static_cast<int>(q/plane);const int64_t r=(q/cols)%rows,c=q%cols;
  const bf* src=leaf==0?wl:leaf==1?wlg:leaf==2?wr:wrg;
  const uint4 value=reinterpret_cast<const uint4*>(src)[r*cols+c];
  reinterpret_cast<uint4*>(stacked)[i]=value;
  const int64_t source=(leaf/2)*static_cast<int64_t>(rows)+r;
  const int64_t outrow=(source/32)*64+source%32+(leaf%2?0:32);
  reinterpret_cast<uint4*>(packed)[outrow*cols+c]=value;
}
// Eight rows per CTA, with inputs retained through both warp reductions.
__device__ float aux_warp_sum(float v) {
  for(int q=16;q;q>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,q));
  return v;
}
// A 64-bit row base (rows * width reaches 2**31 at L2048 D512). Measured at the served widths: no cost; a templated 32 / 64-bit
// body behind a uniform branch cost 3-15 % (register pressure of the two inlined bodies), so it stays one 64-bit body.
extern "C" __global__ void mw_aux_norm_warp(const bf* x,const float* g,const float* b,
                                             bf* y,float* stats,int width,int rows) {
  const int row=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  const int64_t ro=static_cast<int64_t>(row)*width;     // 64-bit row base: rows * width reaches 2^31 at L2048 D512
  const bf* xr=x+ro;bf* yr=y+ro;
  float values[16],sum=0.f;
  #pragma unroll
  for(int j=0;j<16;++j){
    int c=lane+32*j;
    values[j]=c<width?__bfloat162float(xr[c]):0.f;
    sum=__fadd_rn(sum,values[j]);
  }
  const float mean=aux_warp_sum(sum)/width;
  float var=0.f;
  #pragma unroll
  for(int j=0;j<16;++j)if(lane+32*j<width){
    float z=values[j]-mean;var=__fadd_rn(var,__fmul_rn(z,z));
  }
  const float rs=rsqrtf(aux_warp_sum(var)/width+1e-5f);
  if(!lane&&stats){stats[2*static_cast<int64_t>(row)]=mean;stats[2*static_cast<int64_t>(row)+1]=rs;}
  #pragma unroll
  for(int j=0;j<16;++j){
    int c=lane+32*j;
    if(c<width)yr[c]=__float2bfloat16_rn(fmaf((values[j]-mean)*rs,g[c],b[c]));
  }
}
extern "C" __global__ void mw_aux_fold(const bf* wp,const float* go,const float* bo,
                                        bf* wpf,const bf* wg,const float* gi,const float* bi,
                                        bf* wgf,float* uv,int width,int hidden) {
  int c=blockIdx.x; bool gate=blockIdx.y!=0;
  int kmax=gate?width:hidden; float u=0.f,v=0.f;
  const bf* w=gate?wg:wp; const float* gamma=gate?gi:go; const float* beta=gate?bi:bo;
  bf* folded=gate?wgf:wpf;
  const int64_t base=static_cast<int64_t>(c)*kmax;
  for(int k=threadIdx.x;k<kmax;k+=256){
    float a=__bfloat162float(w[base+k]);
    bf f=__float2bfloat16_rn(__fmul_rn(a,gamma[k]));
    folded[base+k]=f;
    u=__fadd_rn(u,__bfloat162float(f)); v=__fadd_rn(v,__fmul_rn(a,beta[k]));
  }
  float us=aux_sum(u),vs=aux_sum(v);
  if(!threadIdx.x){uv[(gate?2:0)*static_cast<int64_t>(width)+c]=us;uv[(gate?3:1)*static_cast<int64_t>(width)+c]=vs;}
}
template<class I> __device__ __forceinline__ float aux_token_pair(const bool* token,I i,int length) {
  const I a=i/length;
  return !token||(token[a]&&token[i-a*length])?1.f:0.f;
}
extern "C" __global__ void mw_aux_mask(const bool* token,bf* pair,int length) {
  const int64_t i=aux_tid(),count=static_cast<int64_t>(length)*length;
  if(i>=count)return;
  pair[i]=__float2bfloat16_rn(count<=INT32_MAX?aux_token_pair<int>(token,static_cast<int>(i),length)
                                               :aux_token_pair<int64_t>(token,i,length));
}
template<class I> __device__ float aux_mask_read(const void* p,I i,int kind) {
  return kind==1?float(static_cast<const bool*>(p)[i]):
         kind==2?__bfloat162float(static_cast<const bf*>(p)[i]):static_cast<const float*>(p)[i];
}
// Batched samples (MiniWorld templates): `samples` pair masks [B][L][L], b-major; a token mask is [B L] (element i = (s L + a) L + b reads
// tokens s L + a and s L + b), a pair-mask input is [B L L] and read in place. samples = 1 is the single-sample contract.
// I = int while samples * L * L fits in int32 (the caller's uniform branch), int64_t past it.
template<class I> __device__ __forceinline__ float aux_pair_mask(const void* mask,I i,int length,int kind,int token,int samples) {
  if(kind==0)return 1.f;
  if(!token)return aux_mask_read<I>(mask,i,kind);
  const I row=i/length,col=i-row*length;              // row = s L + a
  const I first=samples>1?(row/length)*length:I(0);   // s L
  return float(aux_mask_read<I>(mask,row,kind)!=0.f&&aux_mask_read<I>(mask,first+col,kind)!=0.f);
}
extern "C" __global__ void mw_aux_unit_scale(bf* scale,int64_t count) {
  const int64_t i=aux_tid();
  if(i<count)scale[i]=__float2bfloat16_rn(1.f);
}
struct AuxD64TrainPrepare {
  const void* w[6]; bf *packed,*gate,*output; const void* mask; float* pair;
  int s0[6],s1[6],fp32[6],hidden,length,mask_kind,token,samples;   // samples: B pair masks [B][L][L] (see aux_pair_mask)
};
extern "C" __global__ void mw_aux_d64_training_prepare(AuxD64TrainPrepare p) {
  const int64_t i=aux_tid();
  if(i<4*static_cast<int64_t>(p.hidden)*64) {
    const int row=static_cast<int>(i/64),col=static_cast<int>(i%64),source=(row/64)*32+row%32;
    const int leaf=2*(source>=p.hidden)+(row%64<32),r=source%p.hidden;
    p.packed[i]=__float2bfloat16_rn(aux_read(p.w[leaf],static_cast<int64_t>(r)*p.s0[leaf]+static_cast<int64_t>(col)*p.s1[leaf],p.fp32[leaf]));
  }
  if(i<64*64)
    p.gate[i]=__float2bfloat16_rn(aux_read(p.w[4],(i/64)*p.s0[4]+(i%64)*p.s1[4],p.fp32[4]));
  if(i<64*static_cast<int64_t>(p.hidden))
    p.output[i]=__float2bfloat16_rn(aux_read(p.w[5],(i/p.hidden)*p.s0[5]+(i%p.hidden)*p.s1[5],p.fp32[5]));
  const int64_t pairs=static_cast<int64_t>(p.samples)*p.length*p.length;
  if(i<pairs)
    p.pair[i]=pairs<=INT32_MAX?aux_pair_mask<int>(p.mask,static_cast<int>(i),p.length,p.mask_kind,p.token,p.samples)
                              :aux_pair_mask<int64_t>(p.mask,i,p.length,p.mask_kind,p.token,p.samples);
}
// Deterministic normalization backward shared by the general Hopper training
// buckets. Two bounded reductions replace atomic affine updates. The output
// operand stays channel-major; no full-size transpose or affine partial exists.
// (No in-engine caller today: `_h100_pack.deterministic_ln_backward` is defined but not called.)
extern "C" __global__ void mw_aux_ln_backward(const bf* x,const bf* dy,
    const float* gamma,const bf* residual,bf* dx,float* mean,float* rstd,
    int rows,int width,int planes,int saved_stats) {
  const int row=blockIdx.x; float mu=0.f,rs=0.f;
  const int64_t ro=static_cast<int64_t>(row)*width;
  if(saved_stats){mu=mean[row];rs=rstd[row];}
  else {
    float v=0.f;
    for(int c=threadIdx.x;c<width;c+=256)v+=__bfloat162float(x[ro+c]);
    mu=aux_sum(v)/width;v=0.f;
    for(int c=threadIdx.x;c<width;c+=256){float z=__bfloat162float(x[ro+c])-mu;v+=z*z;}
    rs=rsqrtf(aux_sum(v)/width+1e-5f);
    if(!threadIdx.x){mean[row]=mu;rstd[row]=rs;}
  }
  float s1=0.f,s2=0.f;
  for(int c=threadIdx.x;c<width;c+=256){
    const int64_t i=planes?static_cast<int64_t>(c)*rows+row:ro+c;
    float z=(__bfloat162float(x[i])-mu)*rs,dd=__bfloat162float(dy[ro+c])*gamma[c];
    s1+=dd;s2+=dd*z;
  }
  s1=aux_sum(s1)/width;s2=aux_sum(s2)/width;
  for(int c=threadIdx.x;c<width;c+=256){
    const int64_t i=planes?static_cast<int64_t>(c)*rows+row:ro+c;
    float z=(__bfloat162float(x[i])-mu)*rs,dd=__bfloat162float(dy[ro+c])*gamma[c];
    float v=rs*(dd-s1-z*s2);
    if(residual)v+=__bfloat162float(residual[ro+c]);
    dx[i]=__float2bfloat16_rn(v);
  }
}
// (No in-engine caller today, like mw_aux_ln_backward.)
extern "C" __global__ void mw_aux_ln_affine_partial(const bf* x,const bf* dy,
    const float* mean,const float* rstd,float* partial,int rows,int width,int planes){
  const int c=blockIdx.x,start=blockIdx.y*4096,end=min(start+4096,rows);
  float dg=0.f,db=0.f;
  for(int r=start+threadIdx.x;r<end;r+=256){
    const int64_t ro=static_cast<int64_t>(r)*width+c;
    float v=__bfloat162float(dy[ro]);
    float z=(__bfloat162float(x[planes?static_cast<int64_t>(c)*rows+r:ro])-mean[r])*rstd[r];
    dg+=v*z;db+=v;
  }
  dg=aux_sum(dg);db=aux_sum(db);
  if(!threadIdx.x){
    partial[(static_cast<int64_t>(blockIdx.y)*2)*width+c]=dg;partial[(static_cast<int64_t>(blockIdx.y)*2+1)*width+c]=db;
  }
}
extern "C" __global__ void mw_aux_ln_affine_finish(const float* partial,float* dg,float* db,int width,int blocks){
  int c=blockIdx.x*256+threadIdx.x;if(c>=width)return;
  float g=0.f,b=0.f;
  for(int k=0;k<blocks;++k){g+=partial[2*static_cast<int64_t>(k)*width+c];b+=partial[(2*static_cast<int64_t>(k)+1)*width+c];}
  dg[c]=g;db[c]=b;
}
struct AuxStack {const bf* w[5];bf* dst;int rows[5],s0[5],s1[5],width;};
extern "C" __global__ void mw_aux_stack_vec(AuxStack p){
  const int leaf=blockIdx.y;const int64_t i=aux_tid();
  if(i>=static_cast<int64_t>(p.rows[leaf])*(p.width/8))return;
  int64_t offset=0;for(int k=0;k<leaf;++k)offset+=static_cast<int64_t>(p.rows[k])*(p.width/8);
  reinterpret_cast<uint4*>(p.dst)[offset+i]=reinterpret_cast<const uint4*>(p.w[leaf])[i];
}
extern "C" __global__ void mw_aux_affine_cta_finish_parallel(const float* pg,const float* pb,
    float* dg,float* db,int width,int blocks){
  const int tid=threadIdx.x,lane=tid%32,warp=tid/32,c=blockIdx.x*4+(tid%4);
  __shared__ float sums[2][4][4];
  float g=0.f,b=0.f;
  if(c<width)for(int k=tid/4;k<blocks;k+=32){const int64_t o=static_cast<int64_t>(k)*width+c;g+=pg[o];b+=pb[o];}
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
  for(int k=tid/4;k<blocks;k+=32){const int64_t o=static_cast<int64_t>(k)*width+c;g+=pg[o];b+=pb[o];}
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
  const int64_t i=aux_tid(),count=static_cast<int64_t>(samples)*n*n;if(i>=count)return;
  dst[i]=__float2bfloat16_rn(count<=INT32_MAX?aux_pair_mask<int>(src,static_cast<int>(i),n,kind,token,samples)
                                             :aux_pair_mask<int64_t>(src,i,n,kind,token,samples));
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
