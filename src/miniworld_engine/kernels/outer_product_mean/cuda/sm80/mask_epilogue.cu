// E049 own-CUDA mask bit packing + projection epilogue, no separate count kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <set>
#include "opm_epilogue_mask_sm80.cuh"
template<int COLS>
__global__ void opm_mask_epilogue_pack_kernel(const uint8_t* __restrict__ mask,uint32_t* __restrict__ packed,uint8_t* __restrict__ valid,int S,int L){
  constexpr int NT=128,WG=NT/COLS;
  __shared__ uint32_t invalid_warp[4];
  const int tid=threadIdx.x,column=blockIdx.x*COLS+tid%COLS,worker=tid/COLS,nk=(S+31)/32;
  uint32_t any=0;
  for(int chunk=worker;chunk<nk;chunk+=WG){
    uint32_t word=0;
#pragma unroll
    for(int bit=0;bit<32;++bit){const int depth=chunk*32+bit;const bool exists=depth<S;
      const bool value=exists&&column<L&&mask[(long)depth*L+column]!=0;
      word|=uint32_t(value)<<bit;if(exists&&column<L&&!value)any=1;
    }
    if(column<L)packed[(long)chunk*L+column]=word;
  }
  const uint32_t bits=__ballot_sync(0xffffffffu,any!=0);
  if((tid&31)==0)invalid_warp[tid>>5]=bits;__syncthreads();
  if(tid<COLS&&column<L){
    constexpr uint32_t PATTERN=uint32_t(0xffffffffull/((1ull<<COLS)-1));
    const uint32_t bits=invalid_warp[0]|invalid_warp[1]|invalid_warp[2]|invalid_warp[3];
    valid[column]=(bits&(PATTERN<<tid))==0;
  }
}

template<int CZ> void launch(opm_mask80::EpilogueParams p, cudaStream_t stream) {
  using C=opm_mask80::EpCfg<CZ,2,2,false>;
  auto kernel=opm_mask80::opm_mask_epilogue_kernel<CZ,2,2,false>;
  static std::set<int> devices;
  if(devices.insert(at::cuda::current_device()).second)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));
  dim3 grid(((p.L+C::BJ-1)/C::BJ)*p.ntile,(p.L+C::BI-1)/C::BI);
  kernel<<<grid,C::NTHR,C::SMEM,stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<torch::Tensor> result(torch::Tensor O,torch::Tensor mask,torch::Tensor wo,torch::Tensor bias,torch::Tensor residual,bool norm_first,bool save_norm,int64_t cfg){
  TORCH_CHECK(O.is_cuda()&&O.is_contiguous()&&O.scalar_type()==at::kBFloat16&&O.dim()==2,"O BF16 CUDA matrix");
  TORCH_CHECK(mask.is_cuda()&&mask.device()==O.device()&&mask.is_contiguous()&&(mask.scalar_type()==at::kBool||mask.scalar_type()==at::kByte)&&mask.dim()==2,"mask CUDA bool/byte [S,L]");
  const int S=mask.size(0),L=mask.size(1),CZ=wo.size(0);
  TORCH_CHECK(S>0&&L>0&&O.sizes()==at::IntArrayRef({(int64_t)L*32,(int64_t)L*32}),"O/mask sizes");
  TORCH_CHECK(wo.is_cuda()&&wo.device()==O.device()&&wo.is_contiguous()&&wo.scalar_type()==at::kBFloat16&&wo.sizes()==at::IntArrayRef({CZ,1024})&&(CZ==128||CZ==256||CZ==384),"wo BF16 [CZ,1024]");
  TORCH_CHECK(bias.is_cuda()&&bias.device()==O.device()&&bias.is_contiguous()&&bias.scalar_type()==at::kFloat&&bias.numel()==CZ,"bias FP32");
  TORCH_CHECK(residual.numel()==0||(residual.is_cuda()&&residual.device()==O.device()&&residual.is_contiguous()&&residual.scalar_type()==at::kBFloat16&&residual.sizes()==at::IntArrayRef({L,L,CZ})),"residual BF16");
  at::cuda::CUDAGuard guard(O.device());auto stream=at::cuda::getCurrentCUDAStream();
  auto bits=torch::empty({(S+31)/32,L},mask.options().dtype(at::kInt));
  auto valid=torch::empty({L},mask.options().dtype(at::kByte));
  auto out=torch::empty({L,L,CZ},O.options());
  auto norm=torch::empty({save_norm?L:0,L},O.options().dtype(at::kFloat));
  auto* mp=(const uint8_t*)mask.data_ptr();auto* bp=(uint32_t*)bits.data_ptr();auto* vp=valid.data_ptr<uint8_t>();
  if(cfg==0)opm_mask_epilogue_pack_kernel<1><<<L,128,0,stream>>>(mp,bp,vp,S,L);
  else if(cfg==1)opm_mask_epilogue_pack_kernel<2><<<(L+1)/2,128,0,stream>>>(mp,bp,vp,S,L);
  else if(cfg==2)opm_mask_epilogue_pack_kernel<4><<<(L+3)/4,128,0,stream>>>(mp,bp,vp,S,L);
  else if(cfg==3)opm_mask_epilogue_pack_kernel<8><<<(L+7)/8,128,0,stream>>>(mp,bp,vp,S,L);
  else TORCH_CHECK(false,"cfg 0..3");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  opm_mask80::EpilogueParams p{};
  p.O=(const __nv_bfloat16*)O.data_ptr();p.wo=(const __nv_bfloat16*)wo.data_ptr();p.bias=bias.data_ptr<float>();
  p.out=(__nv_bfloat16*)out.data_ptr();p.residual=residual.numel()?(const __nv_bfloat16*)residual.data_ptr():nullptr;
  p.norm_const=1;p.norm=nullptr;p.norm_out=save_norm?norm.data_ptr<float>():nullptr;p.bits=bp;p.valid=vp;p.S=S;p.nk=(S+31)/32;
  p.ldo=O.stride(0);p.L=L;p.norm_first=norm_first;p.cz=CZ;p.ntile=CZ/128;
  launch<128>(p,stream);
  return {out,norm};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("result",&result);}
