// E122v2: retain failed v1 and reserve output drain scratch as well as stage ring.
// E122: retain normalized tile across all column products. Original is preserved.
// E117 independent D256 front: original BF16 LN, folded weight NT product,
// bias planes in the drain. No cached input or weights; no cuBLAS call.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "sm80_common.cuh"
namespace triattn_front_stream2_80 {
using namespace a100;
constexpr int D=256, KS=320, DH4=1024, H=8;
DEVI float bf16_min_f() { return -3.3895313892515355e38f; }
template<int CPR> DEVI uint32_t off(int r,int c) {
 return r*(CPR*16u)+((c^(CPR==2?(r&1):CPR==4?((r>>1)&3):(r&7)))<<4);
}
DEVI float sum32(float v) {
 for(int s=16;s>0;s>>=1)v+=__shfl_xor_sync(0xffffffffu,v,s);
 return v;
}
DEVI size_t row(size_t t,int L,int transposed) {
 if(!transposed)return t;
 size_t LL=(size_t)L*L,z=t/LL,rem=t-z*LL,a=rem/L,b=rem-a*L;
 return z*LL+b*L+a;
}
struct Params {
 const __nv_bfloat16 *x,*w;const uint8_t *mask;
 __nv_bfloat16 *y,*bias;
 int T,N,K,L,transposed;float eps;
};
template<int BM_,int BN_,int BK_,int ST_,int THREADS_=128,int MINB_=0>struct Config {
 static constexpr int BM=BM_,BN=BN_,BK=BK_,ST=ST_,THREADS=THREADS_,MT=BM/(THREADS/4),NT=BN/16;
 static constexpr int NORMAL=BM*KS*2,BS=BN*BK*2,BIAS=8*BK*2,STAGE=BS+BIAS,ARENA=ST*STAGE>BM*BN*2?ST*STAGE:BM*BN*2,SMEM=NORMAL+ARENA;
 static constexpr int MINB=MINB_?MINB_:(SMEM>83000||THREADS==256?1:2);
};
template<class C>DEVI void tile(const Params&p,int ti,int tj,uint8_t*storage,bool normalize) {
 const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,wm=warp>>1,wn=warp&1;
 const int i0=ti*C::BM,j0=tj*C::BN;
 const bool head=tj==0;
 const uint32_t s0=smem_u32(storage),bs0=s0+C::NORMAL;
 // Identical lane grouping, pair sums, sequential variance FMA and BF rounding
 // to rows_sm80.ln_rows_kernel<256>. A is kept in shared memory for this tile.
 if(normalize) {
 for(int r=warp;r<C::BM;r+=C::THREADS/32) {
  const size_t t=(size_t)i0+r;
  const uint4 v=ldg128(p.x+row(t<p.T?t:0,p.L,p.transposed)*D+lane*8);
  float s=(bf16lo(v.x)+bf16hi(v.x))+(bf16lo(v.y)+bf16hi(v.y))+(bf16lo(v.z)+bf16hi(v.z))+(bf16lo(v.w)+bf16hi(v.w));
  const float mean=sum32(s)*(1.f/D);
  float q=0.f,d;
  d=bf16lo(v.x)-mean;q=fmaf(d,d,q);d=bf16hi(v.x)-mean;q=fmaf(d,d,q);
  d=bf16lo(v.y)-mean;q=fmaf(d,d,q);d=bf16hi(v.y)-mean;q=fmaf(d,d,q);
  d=bf16lo(v.z)-mean;q=fmaf(d,d,q);d=bf16hi(v.z)-mean;q=fmaf(d,d,q);
  d=bf16lo(v.w)-mean;q=fmaf(d,d,q);d=bf16hi(v.w)-mean;q=fmaf(d,d,q);
  const float inv=rsqrtf(sum32(q)*(1.f/D)+p.eps);
  uint4 norm=make_uint4(pack_bf16((bf16lo(v.x)-mean)*inv,(bf16hi(v.x)-mean)*inv),pack_bf16((bf16lo(v.y)-mean)*inv,(bf16hi(v.y)-mean)*inv),pack_bf16((bf16lo(v.z)-mean)*inv,(bf16hi(v.z)-mean)*inv),pack_bf16((bf16lo(v.w)-mean)*inv,(bf16hi(v.w)-mean)*inv));
  if(t>=p.T)norm=make_uint4(0u,0u,0u,0u);
  sts128(s0+off<KS/8>(r,lane),norm);
  if(lane<8)sts128(s0+off<KS/8>(r,32+lane),make_uint4(t<p.T&&lane==0?0x3f803f80u:0u,0u,0u,0u));
 }
 }
 __syncthreads();
 constexpr int CB=C::BK/8,NB=C::BN*C::BK/8/C::THREADS;
 auto issue=[&](int kt,int slot) {
  const uint32_t base=bs0+slot*C::STAGE;
#pragma unroll
  for(int z=0;z<NB;++z){int ix=tid+z*C::THREADS,r=ix/CB,c=ix%CB;int k=kt*C::BK+c*8;const __nv_bfloat16*src=p.w+(size_t)(j0+r)*p.K+k;cp_async16(base+off<CB>(r,c),src,j0+r<DH4&&k<p.K?16:0);}
  if(head) {
   for(int ix=tid;ix<8*CB;ix+=C::THREADS){int r=ix/CB,c=ix%CB,k=kt*C::BK+c*8;const __nv_bfloat16*src=p.w+(size_t)(DH4+(r<8?r:0))*p.K+k;cp_async16(base+C::BS+off<CB>(r,c),src,r<8&&k<p.K?16:0);}
  }
 };
 float acc[C::MT][C::NT][4],biasacc[C::MT][4];
#pragma unroll
 for(int mt=0;mt<C::MT;++mt){
#pragma unroll
  for(int n=0;n<C::NT;++n)for(int e=0;e<4;++e)acc[mt][n][e]=0.f;
  for(int e=0;e<4;++e)biasacc[mt][e]=0.f;
 }
 constexpr int NST=(KS+C::BK-1)/C::BK,NKS=C::BK/16;
 uint32_t af[2][C::MT][4],bb[2][2];
 auto fragments=[&](int buf,int slot,int kt,int ksub) {
  const uint32_t base=bs0+slot*C::STAGE;
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)ldsm_x4(af[buf][mt],s0+off<KS/8>(wm*C::MT*16+mt*16+(lane&15),(kt*C::BK)/8+ksub*2+(lane>>4)));
  if(head&&wn==0)ldsm_x2(bb[buf],base+C::BS+off<CB>((lane&7)+8*(lane>>4),ksub*2+((lane>>3)&1)));
 };
#pragma unroll
 for(int slot=0;slot<C::ST-1;++slot){if(slot<NST)issue(slot,slot);cp_async_commit();}
 cp_async_wait<C::ST-2>();__syncthreads();fragments(0,0,0,0);
 int rd=0,wr=C::ST-1;
#pragma unroll 1
 for(int kt=0;kt<NST;++kt) {
#pragma unroll
  for(int ks=0;ks<NKS;++ks) {
   const uint32_t current_bs=bs0+rd*C::STAGE;
   if(ks+1<NKS)fragments(((kt*NKS+ks)&1)^1,rd,kt,ks+1);
   else {
    cp_async_wait<C::ST-3>();__syncthreads();
    if(kt+C::ST-1<NST)issue(kt+C::ST-1,wr);
    cp_async_commit();rd=rd+1==C::ST?0:rd+1;wr=wr+1==C::ST?0:wr+1;
    if(kt+1<NST)fragments(((kt*NKS+ks)&1)^1,rd,kt+1,0);
   }
#pragma unroll
   for(int mt=0;mt<C::MT;++mt){
    if(head&&wn==0)mma16816(biasacc[mt],af[(kt*NKS+ks)&1][mt],bb[(kt*NKS+ks)&1][0],bb[(kt*NKS+ks)&1][1]);
#pragma unroll
    for(int _pairs=0;_pairs<1;++_pairs) {
     uint32_t streamed[4];
     ldsm_x4(streamed,current_bs+off<CB>(wn*C::NT*8+(lane&7)+8*(lane>>4),ks*2+((lane>>3)&1)));
#pragma unroll
     for(int np=0;np<C::NT/2;++np) {
      const uint32_t previous[4]={streamed[0],streamed[1],streamed[2],streamed[3]};
      if(np+1<C::NT/2)ldsm_x4(streamed,current_bs+off<CB>(wn*C::NT*8+(np+1)*16+(lane&7)+8*(lane>>4),ks*2+((lane>>3)&1)));
      mma16816(acc[mt][np*2],af[(kt*NKS+ks)&1][mt],previous[0],previous[1]);
      mma16816(acc[mt][np*2+1],af[(kt*NKS+ks)&1][mt],previous[2],previous[3]);
     }
    }
   }
  }
 }
 cp_async_wait<0>();__syncthreads();
 // B scratch is now free for the coalesced result drain.
 constexpr int CO=C::BN/8;
#pragma unroll
 for(int mt=0;mt<C::MT;++mt){
#pragma unroll
  for(int n=0;n<C::NT;++n)for(int hh=0;hh<2;++hh){int r=wm*C::MT*16+mt*16+8*hh+(lane>>2),c=wn*C::NT+n;sts32(bs0+off<CO>(r,c)+4*(lane&3),pack_bf16(acc[mt][n][2*hh],acc[mt][n][2*hh+1]));}
  if(head&&wn==0) {
#pragma unroll
   for(int hh=0;hh<2;++hh){int r=i0+wm*C::MT*16+mt*16+8*hh+(lane>>2),h=(lane&3)*2;size_t LL=(size_t)p.L*p.L,z=(size_t)r/LL,rem=(size_t)r-z*LL;bool keep=p.mask==nullptr||p.mask[z*p.L+rem%p.L]!=0;uint32_t packed=pack_bf16(biasacc[mt][2*hh],biasacc[mt][2*hh+1]);if(r<p.T){stg32(p.y+(size_t)r*p.N+DH4+h,packed);auto v0=__ushort_as_bfloat16((unsigned short)packed),v1=__ushort_as_bfloat16((unsigned short)(packed>>16));p.bias[(z*H+h)*LL+rem]=keep?v0:__float2bfloat16_rn(bf16_min_f());p.bias[(z*H+h+1)*LL+rem]=keep?v1:__float2bfloat16_rn(bf16_min_f());}}
  }
 }
 __syncthreads();
#pragma unroll
 for(int q=0;q<C::BM*CO/C::THREADS;++q){int ix=tid+q*C::THREADS,r=ix/CO,c=ix%CO;if(i0+r<p.T)stg128(p.y+(size_t)(i0+r)*p.N+j0+c*8,lds128(bs0+off<CO>(r,c)));}
 if(head)for(int ix=tid;ix<C::BM*(p.N-DH4-H);ix+=C::THREADS){int r=ix/(p.N-DH4-H),c=ix%(p.N-DH4-H);if(i0+r<p.T)p.y[(size_t)(i0+r)*p.N+DH4+H+c]=__float2bfloat16_rn(0.f);}
 __syncthreads();
}
template<class C>__global__ void __launch_bounds__(C::THREADS,C::MINB) kernel(Params p) {
 extern __shared__ __align__(128)uint8_t storage[];
 constexpr int NC=DH4/C::BN;int tasks=(p.T+C::BM-1)/C::BM;
 for(int ix=blockIdx.x;ix<tasks;ix+=gridDim.x) {
#pragma unroll 1
  for(int tj=0;tj<NC;++tj)tile<C>(p,ix,tj,storage,tj==0);
 }
}
template<class C>void launch(const Params&p,int grid) {
 static bool ready=false;if(!ready){C10_CUDA_CHECK(cudaFuncSetAttribute(kernel<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));ready=true;}
 int tasks=(p.T+C::BM-1)/C::BM;kernel<C><<<grid>0?grid:tasks,C::THREADS,C::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);
}
void run(torch::Tensor x,torch::Tensor wp,torch::Tensor mask,torch::Tensor y,torch::Tensor bias,int64_t L,bool transposed,double eps,int64_t cfg,int64_t grid) {
 TORCH_CHECK(x.is_cuda()&&wp.is_cuda()&&y.is_cuda()&&bias.is_cuda(),"CUDA inputs");const c10::cuda::CUDAGuard guard(x.device());TORCH_CHECK(x.device()==wp.device()&&x.device()==y.device()&&x.device()==bias.device(),"same device");
 TORCH_CHECK(x.scalar_type()==at::kBFloat16&&wp.scalar_type()==at::kBFloat16&&y.scalar_type()==at::kBFloat16&&bias.scalar_type()==at::kBFloat16,"BF16 required");
 TORCH_CHECK(x.dim()==2&&x.size(1)==D&&x.is_contiguous()&&wp.dim()==2&&wp.is_contiguous()&&wp.size(0)>=DH4+H&&(wp.size(1)==D+8||wp.size(1)==D+32),"D256 original folded front shapes");
 TORCH_CHECK(y.dim()==2&&y.is_contiguous()&&y.size(0)==x.size(0)&&y.size(1)==wp.size(0)&&L>0&&x.size(0)%(L*L)==0,"square stack/y");
 TORCH_CHECK(bias.is_contiguous()&&bias.numel()==(x.size(0)/L/L)*H*L*L,"bias planes");TORCH_CHECK(!mask.numel()||(mask.is_cuda()&&mask.device()==x.device()&&mask.is_contiguous()&&mask.scalar_type()==at::kByte&&mask.numel()==x.size(0)/L),"uint8 mask");
 Params p{reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(wp.data_ptr()),mask.numel()?mask.data_ptr<uint8_t>():nullptr,reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()),(int)x.size(0),(int)y.size(1),(int)wp.size(1),(int)L,(int)transposed,(float)eps};
 switch(cfg){
  case 0:launch<Config<32,128,32,3,128,3>>(p,grid);break;
  case 1:launch<Config<32,128,64,3,128,2>>(p,grid);break;
  case 2:launch<Config<64,128,32,3,256,2>>(p,grid);break;
  case 3:launch<Config<64,128,64,3,256,1>>(p,grid);break;
  case 4:launch<Config<128,64,32,3,256,1>>(p,grid);break;
  case 5:launch<Config<128,128,32,3,256,1>>(p,grid);break;
  case 6:launch<Config<16,128,16,3,64,0>>(p,grid);break;
  case 7:launch<Config<32,128,16,4,128,4>>(p,grid);break;
  case 8:launch<Config<32,256,16,4,128,3>>(p,grid);break;
  case 9:launch<Config<64,64,16,4,128,3>>(p,grid);break;
  case 10:launch<Config<128,128,16,4,256,1>>(p,grid);break;
  default:TORCH_CHECK(false,"unknown fused front stream tile");
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace triattn_front_stream2_80
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&triattn_front_stream2_80::run);}
