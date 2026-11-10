// E205 private E168 exact original BF do plus sequential32-FMA delta carry-shuffle fusion.
// E168: private unroll1 prologue/output-gq resource/liveness experiment. Parent E165: original A100 dog=BF(dy@Wo), then FP32 gate product chains/BF final stores.
// The physical Wo is transposed directly into the CTA's resident shared weight.
// No global transpose/pack cache or global dog is used by the timed path.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "adaln_mma.cuh"

namespace atom_post_grad205 {
using namespace adl;
struct Params {
  const bf *dy,*wo,*o,*g;
  bf *dob,*dg,*dog;
  float *delta;
  int64_t M,dys,os,gs,dobs,dgs,L;
};
ADL_DEVI void sts128(uint32_t a,uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a),"r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w) : "memory");
}
ADL_DEVI float original_sigmoid(float x) {
  float y;asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(.5f*x));
  return fmaf(y,.5f,.5f);
}
ADL_DEVI void physical_transposed_weight(uint32_t dst,const bf *wo,int tid,int nthr) {
#pragma unroll 1
  for(int i=tid;i<128*16;i+=nthr) {
    const int r=i>>4,c=i&15,ch=qf_channel(r);
    float v[8];
#pragma unroll
    for(int e=0;e<8;++e)v[e]=__bfloat162float(__ldg(wo+(c*8+e)*128+ch));
    sts128(dst+qf_woff(r,c),pack8(v));
  }
}
template<int NW,int MINB,bool DIAG>
__global__ void __launch_bounds__(NW*32,MINB) kernel(Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t sw=smem_u32(smem);
  const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,g8=lane>>2,q4=lane&3;
  physical_transposed_weight(sw,p.wo,tid,NW*32);__syncthreads();
  const int64_t ntile=(p.M+15)/16;
  for(int64_t tile=(int64_t)blockIdx.x*NW+warp;tile<ntile;tile+=(int64_t)gridDim.x*NW) {
    int64_t rows[2];bool ok[2];uint4 xv[2][4];
#pragma unroll
    for(int hh=0;hh<2;++hh) {
      rows[hh]=tile*16+g8+8*hh;ok[hh]=rows[hh]<p.M;
#pragma unroll
      for(int gq=0;gq<4;++gq)xv[hh][gq]=ok[hh]?ldg128(p.dy+rows[hh]*p.dys+32*gq+8*q4):make_uint4(0,0,0,0);
    }
    uint32_t af[8][4];a_from_vec(af,xv);
#pragma unroll 1
    for(int gq=0;gq<4;++gq) {
      float acc[4][4];
#pragma unroll
      for(int t=0;t<4;++t)acc[t][0]=acc[t][1]=acc[t][2]=acc[t][3]=0.f;
#pragma unroll
      for(int kb=0;kb<4;++kb)
#pragma unroll
        for(int t=0;t<4;++t) {
          const uint32_t off=(4*gq+t)*8*256+qf_woff(g8,4*kb+q4);
          const uint4 w=lds128_ro(sw+off);
          mma16816(acc[t],af[2*kb],w.x,w.y);mma16816(acc[t],af[2*kb+1],w.z,w.w);
        }
#pragma unroll
      for(int hh=0;hh<2;++hh) {
        if(!ok[hh])continue;
        const int64_t col=32*gq+8*q4;
        float ov[8],gv[8],dv[8],dobv[8],dgv[8];
        unpack8(ldg128(p.o+rows[hh]*p.os+col),ov);unpack8(ldg128(p.g+rows[hh]*p.gs+col),gv);
#pragma unroll
        for(int e=0;e<8;++e) {
          dv[e]=rn(acc[e>>1][2*hh+(e&1)]); // Preserve original GEMM BF boundary before gate arithmetic.
          const float s=original_sigmoid(gv[e]);
          dobv[e]=dv[e]*s;
          dgv[e]=dv[e]*ov[e]*s*(1.f-s); // Same FP32 left association as glue_sm80.cuh.
        }
        stg128(p.dob+rows[hh]*p.dobs+col,pack8(dobv));stg128(p.dg+rows[hh]*p.dgs+col,pack8(dgv));
        if constexpr(DIAG)stg128(p.dog+rows[hh]*128+col,pack8(dv));
        // Same original aa80::attn_delta_kernel<32,BF> 32-channel sequential FMA.
        // The four lanes of this row each own eight adjacent channels; pass
        // the accumulator across q4=0,1,2,3 without reassociating the products.
        // rn(dobv) is the BF do boundary before the original delta re-read.
        float rounded_do[8];
#pragma unroll
        for(int e=0;e<8;++e)rounded_do[e]=rn(dobv[e]);
        const unsigned active=__activemask(); // all four lanes of each valid tail row are present.
        float carry=0.f;
#pragma unroll
        for(int owner=0;owner<4;++owner) {
          if(q4==owner) {
#pragma unroll
            for(int e=0;e<8;++e)carry=fmaf(rounded_do[e],ov[e],carry);
          }
          carry=__shfl_sync(active,carry,owner,4);
        }
        if(q4==0) {
          const int64_t row=rows[hh],sample=row/p.L,query=row%p.L;
          p.delta[(sample*4+gq)*p.L+query]=carry;
        }
      }
    }
  }
}
template<int NW,int MINB,bool DIAG>void launch(Params p,cudaStream_t stream) {
  const int nsm=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid=std::min<int64_t>((p.M+16*NW-1)/(16*NW),nsm*MINB);
  kernel<NW,MINB,DIAG><<<grid,NW*32,32768,stream>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<bool DIAG>void dispatch(Params p,cudaStream_t stream,int cfg) {
  switch(cfg) {
    case 0:launch<4,1,DIAG>(p,stream);break;
    case 1:launch<4,2,DIAG>(p,stream);break;
    case 2:launch<8,1,DIAG>(p,stream);break;
    case 3:launch<8,2,DIAG>(p,stream);break;
    case 4:launch<2,2,DIAG>(p,stream);break;
    case 5:launch<2,4,DIAG>(p,stream);break;
    case 6:launch<4,3,DIAG>(p,stream);break;
    default:TORCH_CHECK(false,"E168 cfg0..6");
  }
}
template<int NW,int MINB>std::vector<int64_t> resource() {
  cudaFuncAttributes a;int blocks;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel<NW,MINB,false>));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel<NW,MINB,false>,NW*32,32768));
  const auto *d=at::cuda::getCurrentDeviceProperties();
  return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,32768,NW*32,blocks,blocks*NW,d->maxThreadsPerMultiProcessor,d->multiProcessorCount,a.binaryVersion,a.ptxVersion,a.maxThreadsPerBlock};
}
std::vector<int64_t> resources(int64_t cfg) {
  switch(cfg) {
    case 0:return resource<4,1>();case 1:return resource<4,2>();case 2:return resource<8,1>();case 3:return resource<8,2>();case 4:return resource<2,2>();case 5:return resource<2,4>();case 6:return resource<4,3>();default:TORCH_CHECK(false,"E168 cfg0..6");
  }
}
void run(torch::Tensor dy,torch::Tensor wo,torch::Tensor o,torch::Tensor gate,torch::Tensor dob,torch::Tensor dg,torch::Tensor dog,torch::Tensor delta,int64_t length,int64_t cfg) {
  TORCH_CHECK(dy.is_cuda()&&dy.scalar_type()==at::kBFloat16&&dy.dim()==2&&dy.size(1)==128&&dy.size(0)>0,"BF CUDA dy[M,128]");
  const int64_t M=dy.size(0);
  auto matrix=[&](const torch::Tensor &t){TORCH_CHECK(t.device()==dy.device()&&t.scalar_type()==at::kBFloat16&&t.dim()==2&&t.sizes()==dy.sizes()&&t.stride(1)==1&&t.stride(0)%8==0,"same CUDA BF[M,128], aligned physical row stride");};
  matrix(dy);matrix(o);matrix(gate);matrix(dob);matrix(dg);
  TORCH_CHECK(wo.device()==dy.device()&&wo.scalar_type()==at::kBFloat16&&wo.is_contiguous()&&wo.sizes()==at::IntArrayRef({128,128}),"physical contiguous BF Wo[128,128]");
  const bool diag=dog.numel()!=0;if(diag){matrix(dog);TORCH_CHECK(dog.is_contiguous(),"diagnostic dog contiguous");}
  TORCH_CHECK(length>0&&M%length==0&&delta.device()==dy.device()&&delta.scalar_type()==at::kFloat&&delta.is_contiguous()&&delta.sizes()==at::IntArrayRef({M/length,4,length}),"delta F32 CUDA[A,4,L], M=A*L");
  at::cuda::CUDAGuard guard(dy.device());
  Params p{(const bf*)dy.data_ptr(),(const bf*)wo.data_ptr(),(const bf*)o.data_ptr(),(const bf*)gate.data_ptr(),(bf*)dob.data_ptr(),(bf*)dg.data_ptr(),diag?(bf*)dog.data_ptr():nullptr,delta.data_ptr<float>(),M,dy.stride(0),o.stride(0),gate.stride(0),dob.stride(0),dg.stride(0),length};
  if(diag)dispatch<true>(p,at::cuda::getCurrentCUDAStream(),cfg);else dispatch<false>(p,at::cuda::getCurrentCUDAStream(),cfg);
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&atom_post_grad205::run);m.def("resources",&atom_post_grad205::resources);}
