// Exact bool mask counts: bit-pack depths, then AND/popcount per token pair.
// All-valid columns are detected from the actual mask and take an exact S path.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

template<int COLS>
__global__ void opm_mask_ballot_pack_kernel(const uint8_t* __restrict__ mask,uint32_t* __restrict__ packed,uint8_t* __restrict__ valid,int S,int L){
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
template<bool CLAMP>
__global__ void opm_mask_ballot_count_kernel(const uint32_t* __restrict__ packed,const uint8_t* __restrict__ valid,float* __restrict__ norm,int S,int L){
  const long pos=(long)blockIdx.x*256+threadIdx.x;if(pos>=(long)L*L)return;
  const int i=pos/L,j=pos%L;
  if(valid[i]&&valid[j]){norm[pos]=float(S);return;}
  int count=0,nk=(S+31)/32;
  for(int k=0;k<nk;++k)count+=__popc(packed[(long)k*L+i]&packed[(long)k*L+j]);
  norm[pos]=float(CLAMP&&count==0?1:count);
}
void run(torch::Tensor mask,torch::Tensor packed,torch::Tensor valid,torch::Tensor norm,bool clamp,int64_t cfg){
  TORCH_CHECK(mask.is_cuda()&&mask.is_contiguous()&&(mask.scalar_type()==at::kBool||mask.scalar_type()==at::kByte)&&mask.dim()==2,"mask_count bool/byte CUDA [S,L]");
  const int S=mask.size(0),L=mask.size(1);TORCH_CHECK(S>0&&L>0,"mask_count dimensions");
  TORCH_CHECK(packed.is_cuda()&&packed.is_contiguous()&&packed.scalar_type()==at::kInt&&packed.device()==mask.device()&&packed.sizes()==at::IntArrayRef({(S+31)/32,L}),"mask_count packed");
  TORCH_CHECK(valid.is_cuda()&&valid.is_contiguous()&&valid.scalar_type()==at::kByte&&valid.device()==mask.device()&&valid.numel()==L,"mask_count valid");
  TORCH_CHECK(norm.is_cuda()&&norm.is_contiguous()&&norm.scalar_type()==at::kFloat&&norm.device()==mask.device()&&norm.sizes()==at::IntArrayRef({L,L}),"mask_count output");
  const at::cuda::CUDAGuard guard(mask.device());const auto stream=at::cuda::getCurrentCUDAStream();
  const uint8_t* m=(const uint8_t*)mask.data_ptr();uint32_t* p=(uint32_t*)packed.data_ptr();uint8_t* v=valid.data_ptr<uint8_t>();
  if(cfg==0)opm_mask_ballot_pack_kernel<1><<<L,128,0,stream>>>(m,p,v,S,L);
  else if(cfg==1)opm_mask_ballot_pack_kernel<2><<<(L+1)/2,128,0,stream>>>(m,p,v,S,L);
  else if(cfg==2)opm_mask_ballot_pack_kernel<4><<<(L+3)/4,128,0,stream>>>(m,p,v,S,L);
  else if(cfg==3)opm_mask_ballot_pack_kernel<8><<<(L+7)/8,128,0,stream>>>(m,p,v,S,L);
  else TORCH_CHECK(false,"mask_count cfg");
  if(clamp)opm_mask_ballot_count_kernel<true><<<((long)L*L+255)/256,256,0,stream>>>(p,v,norm.data_ptr<float>(),S,L);
  else opm_mask_ballot_count_kernel<false><<<((long)L*L+255)/256,256,0,stream>>>(p,v,norm.data_ptr<float>(),S,L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
torch::Tensor result(torch::Tensor mask,bool clamp,int64_t cfg){
  TORCH_CHECK(mask.dim()==2,"mask_count [S,L]");const int S=mask.size(0),L=mask.size(1);
  auto packed=torch::empty({(S+31)/32,L},mask.options().dtype(at::kInt));
  auto valid=torch::empty({L},mask.options().dtype(at::kByte));
  auto norm=torch::empty({L,L},mask.options().dtype(at::kFloat));
  run(mask,packed,valid,norm,clamp,cfg);return norm;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("result",&result);}
