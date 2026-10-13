// E178 independent512thread square/tall/wide WG. Physical operands/workspace unchanged; predicated zero-fill+stores.
// E174 private input-copy/MMA overlap; original E161 all12 configs and final drain preserved.
// E143 independent single-fragment/on-demand-address adaptation. Original sources retained.
// E054: rectangular adaptation of our own contract_tuned.cu; original preserved.
// A100 plane GEMM candidate. Separate from the established path until full-module measurements pass.
// NT/TN/NN/TT layouts share a cp.async ring; operand pointers and fragment offsets are hoisted out of K.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "../../trimul_inproj/cuda/sm80/sm80_common.cuh"

namespace ct_atom_wgrad_sixteen178 {
using namespace a100;

template<int BM_, int BN_, int BK_, int ST_, int THREADS_=128,int MINB_=1>
struct Config {
  static constexpr int BM=BM_, BN=BN_, BK=BK_, ST=ST_, NT=BN/16, MT=BM/(THREADS_/4);
  static constexpr int THREADS=THREADS_, AS=BM*BK*2, BS=BN*BK*2, STAGE=AS+BS, SMEM=ST*STAGE;
  // Eight-warp tiles need more than 128 registers per thread. Requiring two CTAs
  // would force the compiler to spill under the 64K-register SM budget.
  static constexpr int MINB=MINB_;
};
template<int CPR>
DEVI uint32_t offset(int row, int chunk) {
  return row*(CPR*16u) + ((chunk ^ (CPR==4 ? ((row>>1)&3) : (row&7)))<<4);
}
struct Params { const __nv_bfloat16 *a,*b; float *out; int m,n,k,splits; };

template<class C,bool AM,bool BMN>
DEVI void tile(const Params &p,int ti,int tj,uint8_t *storage) {
  constexpr int CA=AM ? C::BM/8 : C::BK/8, CB=BMN ? C::BN/8 : C::BK/8;
  constexpr int NA=C::BM*C::BK/8/C::THREADS, NB=C::BN*C::BK/8/C::THREADS;
  const int tid=threadIdx.x, warp=tid>>5, lane=tid&31, wm=warp>>1, wn=warp&1;
  const int i0=ti*C::BM,j0=tj*C::BN,steps=(p.k+C::BK-1)/C::BK;

  const uint32_t s0=smem_u32(storage);
  auto issue=[&](int kt,int slot) {
    const uint32_t base=s0+slot*C::STAGE;
#pragma unroll
    for(int q=0;q<NA;++q){
      const int idx=tid+q*C::THREADS,r=idx/CA,c=idx%CA;
      const auto* pointer=p.a+(AM ? (size_t)(kt*C::BK+r)*p.m+i0+c*8 : (size_t)(i0+r)*p.k+kt*C::BK+c*8);
      const bool valid=AM ? (kt*C::BK+r<p.k && i0+c*8<p.m) : (i0+r<p.m && kt*C::BK+c*8<p.k);
      cp_async16(base+offset<CA>(r,c),pointer,valid?16:0);
    }
#pragma unroll
    for(int q=0;q<NB;++q){
      const int idx=tid+q*C::THREADS,r=idx/CB,c=idx%CB;
      const auto* pointer=p.b+(BMN ? (size_t)(kt*C::BK+r)*p.n+j0+c*8 : (size_t)(j0+r)*p.k+kt*C::BK+c*8);
      const bool valid=BMN ? (kt*C::BK+r<p.k && j0+c*8<p.n) : (j0+r<p.n && kt*C::BK+c*8<p.k);
      cp_async16(base+C::AS+offset<CB>(r,c),pointer,valid?16:0);
    }
  };
  float acc[C::MT][C::NT][4];
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int nt=0;nt<C::NT;++nt)
#pragma unroll
      for(int e=0;e<4;++e)acc[mt][nt][e]=0.f;
  static_assert(C::ST>=2,"free wr and current rd must differ");
  constexpr int KS=C::BK/16;
  uint32_t af[C::MT][4],bf[C::NT/2][4];
  auto fragments=[&](int slot,int ks) {
    const uint32_t base=s0+slot*C::STAGE;
#pragma unroll
    for(int mt=0;mt<C::MT;++mt) {
      const uint32_t ao=AM ? offset<CA>(ks*16+(lane&7)+8*(lane>>4),2*(wm*C::MT+mt)+((lane>>3)&1))
                          : offset<CA>(wm*C::MT*16+mt*16+(lane&15),ks*2+(lane>>4));
      if constexpr(AM)ldsm_x4_t(af[mt],base+ao);
      else ldsm_x4(af[mt],base+ao);
    }
#pragma unroll
    for(int np=0;np<C::NT/2;++np) {
      const uint32_t bo=C::AS+(BMN ? offset<CB>(ks*16+(lane&7)+8*((lane>>3)&1),wn*C::NT+np*2+(lane>>4))
                                 : offset<CB>(wn*C::NT*8+np*16+(lane&7)+8*(lane>>4),ks*2+((lane>>3)&1)));
      if constexpr(BMN)ldsm_x4_t(bf[np],base+bo);
      else ldsm_x4(bf[np],base+bo);
    }
  };
#pragma unroll
  for(int slot=0;slot<C::ST-1;++slot){if(slot<steps)issue(slot,slot);cp_async_commit();}
  cp_async_wait<C::ST-2>();__syncthreads();
  int rd=0,wr=C::ST-1;
#pragma unroll 1
  for(int kt=0;kt<steps;++kt){
    // E174: rd=kt mod ST, wr=(kt+ST-1) mod ST; ST>=2 => distinct.
    // Prior iteration barrier completes every read before an old rd slot is reused.
    // Issue into free wr before MMA reads rd, then retain the original wait/barrier.
    if(kt+C::ST-1<steps)issue(kt+C::ST-1,wr);
    cp_async_commit();
#pragma unroll
    for(int ks=0;ks<KS;++ks){
      fragments(rd,ks);
#pragma unroll
      for(int mt=0;mt<C::MT;++mt)
#pragma unroll
        for(int np=0;np<C::NT/2;++np){
          mma16816(acc[mt][np*2],af[mt],bf[np][0],bf[np][1]);
          mma16816(acc[mt][np*2+1],af[mt],bf[np][2],bf[np][3]);
        }
    }
    cp_async_wait<C::ST-2>();__syncthreads();
    rd=rd+1==C::ST ? 0 : rd+1;wr=wr+1==C::ST ? 0 : wr+1;
  }
  cp_async_wait<0>();__syncthreads();
#pragma unroll
  for(int mt=0;mt<C::MT;++mt)
#pragma unroll
    for(int nt=0;nt<C::NT;++nt)
#pragma unroll
      for(int hh=0;hh<2;++hh){
        const int row=i0+wm*C::MT*16+mt*16+8*hh+(lane>>2);
        const int col=j0+(wn*C::NT+nt)*8+2*(lane&3);
        if(row<p.m && col<p.n) {
          p.out[(size_t)row*p.n+col]=acc[mt][nt][2*hh];
          p.out[(size_t)row*p.n+col+1]=acc[mt][nt][2*hh+1];
        }
      }

}
template<class C,int MODE>
__global__ void __launch_bounds__(C::THREADS,C::MINB) kernel(Params p) {
  p.a+=(size_t)blockIdx.y*p.k*p.m;
  p.b+=(size_t)blockIdx.y*p.k*p.n;
  p.out+=(size_t)blockIdx.y*p.m*p.n;
  extern __shared__ __align__(128) uint8_t storage[];
  const int cols=(p.n+C::BN-1)/C::BN,tpc=((p.m+C::BM-1)/C::BM)*cols;
  for(int idx=blockIdx.x;idx<tpc;idx+=gridDim.x) {
    const int ti=idx/cols,tj=idx-ti*cols;
    if constexpr(MODE==0)tile<C,false,false>(p,ti,tj,storage);
    else if constexpr(MODE==1)tile<C,true,true>(p,ti,tj,storage);
    else if constexpr(MODE==2)tile<C,false,true>(p,ti,tj,storage);
    else tile<C,true,false>(p,ti,tj,storage);
  }
}


__global__ void reduce_kernel(const float *partial,float *out,int elements,int splits){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=elements)return;
 float sum=0.f;
#pragma unroll 1
 for(int s=0;s<splits;++s)sum+=partial[(size_t)s*elements+i];
 out[i]=sum;
}
template<class C>void launch(Params p){
 TORCH_CHECK(p.m>0&&p.n>0&&p.m%8==0&&p.n%8==0&&p.k>0,"physical width multiples8; CUDA zero-fill handles tile/K tails");
 static bool configured=false;if(!configured){C10_CUDA_CHECK(cudaFuncSetAttribute(kernel<C,1>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));configured=true;}
 const int blocks=((p.m+C::BM-1)/C::BM)*((p.n+C::BN-1)/C::BN);
 kernel<C,1><<<dim3(blocks,p.splits),C::THREADS,C::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(torch::Tensor a,torch::Tensor b,torch::Tensor partial,torch::Tensor out,int64_t cfg,int64_t splits){
 TORCH_CHECK(a.is_cuda()&&b.device()==a.device()&&partial.device()==a.device()&&out.device()==a.device(),"same CUDA device");
 TORCH_CHECK(a.scalar_type()==at::kBFloat16&&b.scalar_type()==at::kBFloat16&&partial.scalar_type()==at::kFloat&&out.scalar_type()==at::kFloat,"BF16 operands and unrounded FP32 outputs");
 TORCH_CHECK(a.is_contiguous()&&b.is_contiguous()&&partial.is_contiguous()&&out.is_contiguous()&&a.dim()==2&&b.dim()==2&&out.dim()==2,"contiguous matrices");
 const int K=a.size(0),M=a.size(1),N=b.size(1);
 TORCH_CHECK(b.size(0)==K&&splits>0&&splits<=128&&K%splits==0&&out.sizes()==at::IntArrayRef({M,N})&&partial.sizes()==at::IntArrayRef({splits,M,N}),"physical dimensions and split workspace");
 at::cuda::CUDAGuard guard(a.device());Params p{(const __nv_bfloat16*)a.data_ptr(),(const __nv_bfloat16*)b.data_ptr(),(float*)partial.data_ptr(),M,N,K/(int)splits,(int)splits};
 switch(cfg){
  case 0:launch<Config<128,128,32,3,512,1>>(p);break;
  case 1:launch<Config<128,128,32,3,512,2>>(p);break;
  case 2:launch<Config<128,128,64,2,512,1>>(p);break;
  case 3:launch<Config<128,128,64,2,512,2>>(p);break;
  case 4:launch<Config<128,128,64,3,512,1>>(p);break;
  case 5:launch<Config<128,128,128,2,512,1>>(p);break;
  case 6:launch<Config<256,128,32,3,512,1>>(p);break;
  case 7:launch<Config<256,128,64,2,512,1>>(p);break;
  case 8:launch<Config<256,128,32,3,512,2>>(p);break;
  case 9:launch<Config<128,256,32,3,512,1>>(p);break;
  case 10:launch<Config<128,256,64,2,512,1>>(p);break;
  case 11:launch<Config<128,256,32,3,512,2>>(p);break;
  default:TORCH_CHECK(false,"cfg0..11");
 }
 reduce_kernel<<<(M*N+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>((const float*)partial.data_ptr(),(float*)out.data_ptr(),M*N,splits);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<class C>std::vector<int64_t> resource(){
 cudaFuncAttributes attrs;int blocks=0;
 C10_CUDA_CHECK(cudaFuncSetAttribute(kernel<C,1>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs,kernel<C,1>));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel<C,1>,C::THREADS,C::SMEM));
 return {attrs.numRegs,(int64_t)attrs.localSizeBytes,(int64_t)attrs.sharedSizeBytes,C::SMEM,C::THREADS,blocks,blocks*C::THREADS/32,at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor,at::cuda::getCurrentDeviceProperties()->multiProcessorCount,attrs.binaryVersion,attrs.ptxVersion,attrs.maxThreadsPerBlock};
}
std::vector<int64_t> resources(int64_t cfg){switch(cfg){
case 0:return resource<Config<128,128,32,3,512,1>>();
  case 1:return resource<Config<128,128,32,3,512,2>>();
  case 2:return resource<Config<128,128,64,2,512,1>>();
  case 3:return resource<Config<128,128,64,2,512,2>>();
  case 4:return resource<Config<128,128,64,3,512,1>>();
  case 5:return resource<Config<128,128,128,2,512,1>>();
  case 6:return resource<Config<256,128,32,3,512,1>>();
  case 7:return resource<Config<256,128,64,2,512,1>>();
  case 8:return resource<Config<256,128,32,3,512,2>>();
  case 9:return resource<Config<128,256,32,3,512,1>>();
  case 10:return resource<Config<128,256,64,2,512,1>>();
  case 11:return resource<Config<128,256,32,3,512,2>>();
default:TORCH_CHECK(false,"largek cfg0..11");}}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&ct_atom_wgrad_sixteen178::run);m.def("resources",&ct_atom_wgrad_sixteen178::resources);}
