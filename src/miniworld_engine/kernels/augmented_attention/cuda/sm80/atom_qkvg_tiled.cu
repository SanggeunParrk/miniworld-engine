// E058: allfour projectionweightsresident, inputreadonce, rolledchannels/staticK MMA operand indexing.
// E056: atom-width packed Q/K/V/gate projection, preserving addmm's BF16 output boundary.
// Four independent projection tiles keep one 128x128 weight resident in shared memory.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "adaln_mma.cuh"

namespace atom_qkvg_tiled80 {
using namespace adl;
struct Params { const bf *x,*w,*bias; bf *out; int64_t M,xstride; };
// Small raw edge inputs can straddle a BF16 rounding midpoint after an FP32
// tensor-core dot. Keep full physical module tiles on the measured MMA path;
// accumulate the small edge dot exactly enough to preserve the original band.
__global__ void atom_qkvg_small_precise_kernel(Params p){
  const int64_t index=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=p.M*512)return;
  const int64_t row=index/512;const int col=index%512;
  double acc=(double)__bfloat162float(p.bias[col]);
#pragma unroll 1
  for(int k=0;k<128;++k)
    acc+=(double)__bfloat162float(p.x[row*p.xstride+k])*(double)__bfloat162float(p.w[col*128+k]);
  float rounded=__double2float_rn(acc);
  // Avoid a double-rounding tie when FP32 lands exactly on a BF16 midpoint.
  if((__float_as_uint(rounded)&0xffffu)==0x8000u&&(double)rounded!=acc)
    rounded=nextafterf(rounded,acc>(double)rounded?INFINITY:-INFINITY);
  p.out[index]=__float2bfloat16_rn(rounded);
}
template<int NW,int MINB,int NC>
__global__ void __launch_bounds__(NW*32,MINB) atom_qkvg_kernel(Params p){
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t base=smem_u32(smem);
  const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,g8=lane>>2,q4=lane&3;
  // E072: narrower weight tiles improve SM coverage and shorten cp.async chains.
  for(int i=tid;i<NC*16;i+=NW*32){
    const int row=i>>4;
    cp_async16(base+qf_woff(row,i&15),p.w+(size_t)(blockIdx.y*NC+qf_channel(row))*128+(i&15)*8);
  }
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
    for(int proj=0;proj<1;++proj){
#pragma unroll 1
    for(int gq=0;gq<NC/32;++gq){
      float acc[4][4];
#pragma unroll
      for(int t=0;t<4;++t)acc[t][0]=acc[t][1]=acc[t][2]=acc[t][3]=0.f;
#pragma unroll
      for(int kb=0;kb<4;++kb)
#pragma unroll
        for(int t=0;t<4;++t){const uint4 b=lds128_ro(base+proj*32768+(4*gq+t)*8*256+qf_woff(g8,4*kb+q4));
          mma16816(acc[t],a[2*kb],b.x,b.y);mma16816(acc[t],a[2*kb+1],b.z,b.w);
        }
      float bv[8];unpack8(ldg128(p.bias+blockIdx.y*NC+32*gq+8*q4),bv);
#pragma unroll
      for(int hh=0;hh<2;++hh){float out[8];
#pragma unroll
        for(int i=0;i<8;++i)out[i]=acc[i>>1][2*hh+(i&1)]+bv[i];
        if(ok[hh])stg128(p.out+rows[hh]*512+blockIdx.y*NC+32*gq+8*q4,pack8(out));
      }
    }
    }
  }
}
template<int NW,int MINB,int NC>void launch(Params p,cudaStream_t stream){
  const int nsm=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid=std::min<int64_t>((p.M+16*NW-1)/(16*NW),nsm*MINB);
  auto kernel=atom_qkvg_kernel<NW,MINB,NC>;static std::set<int> devices;
  if(devices.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,NC*256));
  kernel<<<dim3(grid,512/NC),NW*32,NC*256,stream>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(torch::Tensor x,torch::Tensor w,torch::Tensor bias,torch::Tensor out,int64_t cfg){
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.dim()==2&&x.size(0)>0&&x.size(1)==128&&x.stride(1)==1&&x.stride(0)%8==0,"x BF16 CUDA [M,128] aligned row stride");
  auto check=[&](const torch::Tensor&t){TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.is_contiguous()&&t.scalar_type()==at::kBFloat16,"same-device contiguous BF16 operand");};
  check(w);check(bias);check(out);TORCH_CHECK(w.sizes()==at::IntArrayRef({512,128})&&bias.numel()==512&&out.sizes()==at::IntArrayRef({x.size(0),512}),"packed QKVG dimensions");
  at::cuda::CUDAGuard guard(x.device());Params p{(const bf*)x.data_ptr(),(const bf*)w.data_ptr(),(const bf*)bias.data_ptr(),(bf*)out.data_ptr(),x.size(0),x.stride(0)};auto stream=at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(cfg>=0&&cfg<=11,"cfg0..11");
  if(p.M<128){atom_qkvg_small_precise_kernel<<<(p.M*512+255)/256,256,0,stream>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();return;}
  switch(cfg){case 0:launch<1,4,32>(p,stream);break;case 1:launch<2,4,32>(p,stream);break;case 2:launch<4,4,32>(p,stream);break;case 3:launch<8,2,32>(p,stream);break;case 4:launch<1,4,64>(p,stream);break;case 5:launch<2,4,64>(p,stream);break;case 6:launch<4,4,64>(p,stream);break;case 7:launch<8,2,64>(p,stream);break;case 8:launch<1,4,128>(p,stream);break;case 9:launch<2,4,128>(p,stream);break;case 10:launch<4,4,128>(p,stream);break;case 11:launch<8,2,128>(p,stream);break;default:TORCH_CHECK(false,"cfg0..11");}
}
} // namespace atom_qkvg_tiled80
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&atom_qkvg_tiled80::run);}
