// E050: atom output projection + conditioning projection + rounded sigmoid/residual.
// Original BF16 boundaries: rn(og Wo^T), rn(cond Ws^T + bs), rn(res + sigmoid(g2) y).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "adaln_mma.cuh"

namespace atom_output_tiled_rolled80 {
using namespace adl;
struct Params {
  const bf *og,*cond,*wo,*ws,*bs,*res;
  bf *out,*saved_y,*saved_g2;
  int64_t M,P,ostride;
};
ADL_DEVI float gate_sigmoid(float x){float y;asm("tanh.approx.f32 %0, %1;":"=f"(y):"f"(.5f*x));return fmaf(y,.5f,.5f);}
template<int NW,int MINB,int NC,bool SAVE,bool RES>
__global__ void __launch_bounds__(NW*32,MINB) atom_output_fused_kernel(Params p){
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t base=smem_u32(smem),swo=base,sws=base+NC*256;
  const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,g8=lane>>2,q4=lane&3;
  for(int i=tid;i<NC*16;i+=NW*32){
    const int row=i>>4,col=blockIdx.y*NC+qf_channel(row);
    cp_async16(swo+qf_woff(row,i&15),p.wo+(size_t)col*128+(i&15)*8);
    cp_async16(sws+qf_woff(row,i&15),p.ws+(size_t)col*128+(i&15)*8);
  }
  cp_async_commit();cp_async_wait<0>();__syncthreads();
  const int64_t ntile=(p.M+15)/16;
  for(int64_t tile=(int64_t)warp*gridDim.x+blockIdx.x;tile<ntile;tile+=(int64_t)gridDim.x*NW){
    int64_t rows[2];bool ok[2];uint4 xo[2][4],xc[2][4];
#pragma unroll
    for(int hh=0;hh<2;++hh){rows[hh]=tile*16+g8+8*hh;ok[hh]=rows[hh]<p.M;
#pragma unroll
      for(int gq=0;gq<4;++gq){const int c=32*gq+8*q4;
        xo[hh][gq]=ok[hh]?ldg128(p.og+rows[hh]*p.ostride+c):make_uint4(0,0,0,0);
        xc[hh][gq]=ok[hh]?ldg128(p.cond+(rows[hh]%p.P)*128+c):make_uint4(0,0,0,0);
      }
    }
    uint32_t oa[8][4],ca[8][4];a_from_vec(oa,xo);a_from_vec(ca,xc);
#pragma unroll
    for(int gq=0;gq<NC/32;++gq){
      float ya[4][4],ga[4][4];
#pragma unroll
      for(int t=0;t<4;++t){ya[t][0]=ya[t][1]=ya[t][2]=ya[t][3]=0;ga[t][0]=ga[t][1]=ga[t][2]=ga[t][3]=0;}
#pragma unroll 1
      for(int kb=0;kb<4;++kb)
#pragma unroll
        for(int t=0;t<4;++t){const uint32_t off=(4*gq+t)*8*256+qf_woff(g8,4*kb+q4);
          uint4 by=lds128_ro(swo+off),bg=lds128_ro(sws+off);
          mma16816(ya[t],oa[2*kb],by.x,by.y);mma16816(ya[t],oa[2*kb+1],by.z,by.w);
          mma16816(ga[t],ca[2*kb],bg.x,bg.y);mma16816(ga[t],ca[2*kb+1],bg.z,bg.w);
        }
      float bias[8];unpack8(ldg128(p.bs+blockIdx.y*NC+32*gq+8*q4),bias);
#pragma unroll
      for(int hh=0;hh<2;++hh){float rv[8],yv[8],gv[8],ov[8];
        if constexpr(RES)unpack8(ok[hh]?ldg128(p.res+rows[hh]*128+blockIdx.y*NC+32*gq+8*q4):make_uint4(0,0,0,0),rv);
#pragma unroll
        for(int i=0;i<8;++i){yv[i]=rn(ya[i>>1][2*hh+(i&1)]);gv[i]=rn(ga[i>>1][2*hh+(i&1)]+bias[i]);
          const float gate=gate_sigmoid(gv[i]);ov[i]=RES?fmaf(gate,yv[i],rv[i]):gate*yv[i];
        }
        if(ok[hh]){const int64_t pos=rows[hh]*128+blockIdx.y*NC+32*gq+8*q4;stg128(p.out+pos,pack8(ov));
          if constexpr(SAVE){stg128(p.saved_y+pos,pack8(yv));stg128(p.saved_g2+pos,pack8(gv));}
        }
      }
    }
  }
}
template<int NW,int MINB,int NC,bool SAVE,bool RES> void launch(Params p,cudaStream_t stream){
  auto kernel=atom_output_fused_kernel<NW,MINB,NC,SAVE,RES>;static std::set<int> devices;
  if(devices.insert(at::cuda::current_device()).second)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,NC*512));
  const int nsm=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid=std::min<int64_t>((p.M+16*NW-1)/(16*NW),nsm*MINB);
  kernel<<<dim3(grid,128/NC),NW*32,NC*512,stream>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<bool SAVE,bool RES> void dispatch(Params p,cudaStream_t stream,int cfg){
  if(cfg==0)launch<1,4,32,SAVE,RES>(p,stream);
  else if(cfg==1)launch<2,4,32,SAVE,RES>(p,stream);
  else if(cfg==2)launch<4,4,32,SAVE,RES>(p,stream);
  else if(cfg==3)launch<8,2,32,SAVE,RES>(p,stream);
  else if(cfg==4)launch<1,4,64,SAVE,RES>(p,stream);
  else if(cfg==5)launch<2,4,64,SAVE,RES>(p,stream);
  else if(cfg==6)launch<4,4,64,SAVE,RES>(p,stream);
  else if(cfg==7)launch<8,2,64,SAVE,RES>(p,stream);
  else if(cfg==8)launch<1,2,128,SAVE,RES>(p,stream);
  else if(cfg==9)launch<2,2,128,SAVE,RES>(p,stream);
  else if(cfg==10)launch<4,2,128,SAVE,RES>(p,stream);
  else if(cfg==11)launch<8,2,128,SAVE,RES>(p,stream);
  else TORCH_CHECK(false,"cfg0..11");
}
} // namespace atom_output_tiled_rolled80
std::vector<torch::Tensor> result(torch::Tensor og,torch::Tensor cond,torch::Tensor wo,torch::Tensor ws,torch::Tensor bs,torch::Tensor residual,bool save,int64_t cfg){
  TORCH_CHECK(og.is_cuda()&&og.scalar_type()==at::kBFloat16&&og.dim()==2&&og.size(1)==128&&og.stride(1)==1&&og.stride(0)%8==0,"og BF16 CUDA [M,128] aligned row stride");
  auto check=[&](const torch::Tensor& t){TORCH_CHECK(t.is_cuda()&&t.device()==og.device()&&t.is_contiguous()&&t.scalar_type()==at::kBFloat16,"contiguous BF16 CUDA operand");};
  check(cond);check(wo);check(ws);check(bs);const int64_t M=og.size(0),P=cond.size(0);TORCH_CHECK(M>0&&P>0&&cond.dim()==2&&cond.size(1)==128,"cond [P,128]");
  TORCH_CHECK(wo.sizes()==at::IntArrayRef({128,128})&&ws.sizes()==wo.sizes()&&bs.numel()==128,"weights128");
  const bool res=residual.numel()!=0;if(res){check(residual);TORCH_CHECK(residual.sizes()==og.sizes(),"residual shape");}
  at::cuda::CUDAGuard guard(og.device());auto options=og.options();auto out=torch::empty({M,128},options),y=torch::empty({save?M:0,128},options),g=torch::empty({save?M:0,128},options);
  atom_output_tiled_rolled80::Params p{(const adl::bf*)og.data_ptr(),(const adl::bf*)cond.data_ptr(),(const adl::bf*)wo.data_ptr(),(const adl::bf*)ws.data_ptr(),(const adl::bf*)bs.data_ptr(),res?(const adl::bf*)residual.data_ptr():nullptr,(adl::bf*)out.data_ptr(),save?(adl::bf*)y.data_ptr():nullptr,save?(adl::bf*)g.data_ptr():nullptr,M,P,og.stride(0)};
  auto stream=at::cuda::getCurrentCUDAStream();
  if(save){if(res)atom_output_tiled_rolled80::dispatch<true,true>(p,stream,cfg);else atom_output_tiled_rolled80::dispatch<true,false>(p,stream,cfg);}
  else {if(res)atom_output_tiled_rolled80::dispatch<false,true>(p,stream,cfg);else atom_output_tiled_rolled80::dispatch<false,false>(p,stream,cfg);}
  return {out,y,g};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("result",&result);}
