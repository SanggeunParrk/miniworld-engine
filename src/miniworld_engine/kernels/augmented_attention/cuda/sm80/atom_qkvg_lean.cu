// E057 keeps channel/K loops rolled to reduce measured REG255/stack32 pressure.
// E056: atom-width packed Q/K/V/gate projection, preserving addmm's BF16 output boundary.
// Four independent projection tiles keep one 128x128 weight resident in shared memory.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "adaln_mma.cuh"

namespace atom_qkvg_lean80 {
using namespace adl;
struct Params { const bf *x,*w,*bias; bf *out; int64_t M,xstride; };
template<int NW,int MINB>
__global__ void __launch_bounds__(NW*32,MINB) atom_qkvg_kernel(Params p){
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t base=smem_u32(smem);
  const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,g8=lane>>2,q4=lane&3,proj=blockIdx.y;
  load_weight128(base,p.w+proj*128*128,tid,NW*32);
  cp_async_commit();cp_async_wait<0>();__syncthreads();
  const int64_t ntile=(p.M+15)/16;
  for(int64_t tile=(int64_t)warp*gridDim.x+blockIdx.x;tile<ntile;tile+=(int64_t)gridDim.x*NW){
    int64_t rows[2];bool ok[2];uint4 xv[2][4];
#pragma unroll
    for(int hh=0;hh<2;++hh){rows[hh]=tile*16+g8+8*hh;ok[hh]=rows[hh]<p.M;
#pragma unroll
      for(int gq=0;gq<4;++gq)xv[hh][gq]=ok[hh]?ldg128(p.x+rows[hh]*p.xstride+32*gq+8*q4):make_uint4(0,0,0,0);
    }
    uint32_t a[8][4];a_from_vec(a,xv);
#pragma unroll 1
    for(int gq=0;gq<4;++gq){
      float acc[4][4];
#pragma unroll
      for(int t=0;t<4;++t)acc[t][0]=acc[t][1]=acc[t][2]=acc[t][3]=0.f;
#pragma unroll 1
      for(int kb=0;kb<4;++kb)
#pragma unroll
        for(int t=0;t<4;++t){const uint4 b=lds128_ro(base+(4*gq+t)*8*256+qf_woff(g8,4*kb+q4));
          mma16816(acc[t],a[2*kb],b.x,b.y);mma16816(acc[t],a[2*kb+1],b.z,b.w);
        }
      float bv[8];unpack8(ldg128(p.bias+proj*128+32*gq+8*q4),bv);
#pragma unroll
      for(int hh=0;hh<2;++hh){float out[8];
#pragma unroll
        for(int i=0;i<8;++i)out[i]=acc[i>>1][2*hh+(i&1)]+bv[i];
        if(ok[hh])stg128(p.out+rows[hh]*512+proj*128+32*gq+8*q4,pack8(out));
      }
    }
  }
}
template<int NW,int MINB>void launch(Params p,cudaStream_t stream){
  const int nsm=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid=std::min<int64_t>((p.M+16*NW-1)/(16*NW),nsm*MINB);
  atom_qkvg_kernel<NW,MINB><<<dim3(grid,4),NW*32,32768,stream>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(torch::Tensor x,torch::Tensor w,torch::Tensor bias,torch::Tensor out,int64_t cfg){
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.dim()==2&&x.size(0)>0&&x.size(1)==128&&x.stride(1)==1&&x.stride(0)%8==0,"x BF16 CUDA [M,128] aligned row stride");
  auto check=[&](const torch::Tensor&t){TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.is_contiguous()&&t.scalar_type()==at::kBFloat16,"same-device contiguous BF16 operand");};
  check(w);check(bias);check(out);TORCH_CHECK(w.sizes()==at::IntArrayRef({512,128})&&bias.numel()==512&&out.sizes()==at::IntArrayRef({x.size(0),512}),"packed QKVG dimensions");
  at::cuda::CUDAGuard guard(x.device());Params p{(const bf*)x.data_ptr(),(const bf*)w.data_ptr(),(const bf*)bias.data_ptr(),(bf*)out.data_ptr(),x.size(0),x.stride(0)};auto stream=at::cuda::getCurrentCUDAStream();
  if(cfg==0)launch<4,2>(p,stream);else if(cfg==1)launch<4,4>(p,stream);else if(cfg==2)launch<8,2>(p,stream);else if(cfg==3)launch<8,1>(p,stream);else if(cfg==4)launch<2,4>(p,stream);else if(cfg==5)launch<4,1>(p,stream);else TORCH_CHECK(false,"cfg0..5");
}
} // namespace atom_qkvg_lean80
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&atom_qkvg_lean80::run);}
