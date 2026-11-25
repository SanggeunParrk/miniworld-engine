// Independent private E442: original outer gate plus first residual.
// Preserve every Cu GEMM and exact original BF sigmoid/product/residual stores.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <tuple>

namespace af3_local_outer_gate_residual_sm80 {
using bf=__nv_bfloat16;
union Bits {uint4 words;uint16_t halves[8];};
__device__ __forceinline__ bf original_sigmoid(bf gate) {
  // Literal original sm80_atom_local/glue.cu:20; no fast-math flags.
  return __float2bfloat16_rn(1.f/(1.f+expf(-__bfloat162float(gate))));
}
__device__ __forceinline__ bf original_product(bf sigmoid,bf projection) {
  // Literal original glue.cu:22 product rounding, BEFORE residual addition.
  return __float2bfloat16_rn(__bfloat162float(sigmoid)*__bfloat162float(projection));
}
__device__ __forceinline__ bf original_residual(bf single,bf delta) {
  return __float2bfloat16_rn(__fadd_rn(__bfloat162float(single),__bfloat162float(delta)));
}
template<int Threads,bool Save,bool Aligned>
__global__ void local_outer_gate_residual(const bf *single,const bf *gate,const bf *projection,
    bf *current,bf *sigmoid,int64_t count) {
  constexpr int Vec=Aligned?8:1;
  const int64_t base=((int64_t)blockIdx.x*Threads+threadIdx.x)*Vec;
  if constexpr (Aligned) {
    if(base+8<=count) {
      Bits a,g,p,c,s;
      a.words=*reinterpret_cast<const uint4*>(single+base);
      g.words=*reinterpret_cast<const uint4*>(gate+base);
      p.words=*reinterpret_cast<const uint4*>(projection+base);
#pragma unroll
      for(int i=0;i<8;++i) {
        const bf sig=original_sigmoid(__ushort_as_bfloat16(g.halves[i]));
        const bf delta=original_product(sig,__ushort_as_bfloat16(p.halves[i]));
        c.halves[i]=__bfloat16_as_ushort(original_residual(__ushort_as_bfloat16(a.halves[i]),delta));
        if constexpr (Save)s.halves[i]=__bfloat16_as_ushort(sig);
      }
      *reinterpret_cast<uint4*>(current+base)=c.words;
      if constexpr (Save)*reinterpret_cast<uint4*>(sigmoid+base)=s.words;
      return;
    }
  }
#pragma unroll
  for(int i=0;i<Vec;++i)if(base+i<count) {
    const bf sig=original_sigmoid(gate[base+i]);
    const bf delta=original_product(sig,projection[base+i]);
    current[base+i]=original_residual(single[base+i],delta);
    if constexpr (Save)sigmoid[base+i]=sig;
  }
}
template<int Threads,bool Save>
void launch(const bf*a,const bf*g,const bf*p,bf*c,bf*s,int64_t count,cudaStream_t stream) {
  const bool aligned=((reinterpret_cast<uintptr_t>(a)|reinterpret_cast<uintptr_t>(g)|
      reinterpret_cast<uintptr_t>(p)|reinterpret_cast<uintptr_t>(c)|
      (Save?reinterpret_cast<uintptr_t>(s):0))&15)==0;
  if(aligned)local_outer_gate_residual<Threads,Save,true>
    <<<(count+Threads*8-1)/(Threads*8),Threads,0,stream>>>(a,g,p,c,s,count);
  else local_outer_gate_residual<Threads,Save,false>
    <<<(count+Threads-1)/Threads,Threads,0,stream>>>(a,g,p,c,s,count);
}
} // namespace af3_local_outer_gate_residual_sm80

std::tuple<torch::Tensor,torch::Tensor> gate_residual(
    torch::Tensor single,torch::Tensor gate,torch::Tensor projection,bool save,int64_t cfg) {
  TORCH_CHECK(cfg==0||cfg==1,"private E442 block128 or block256 cfg");
  TORCH_CHECK(single.is_cuda()&&gate.is_cuda()&&projection.is_cuda(),"CUDA operands");
  TORCH_CHECK(single.device()==gate.device()&&single.device()==projection.device(),"same CUDA device");
  TORCH_CHECK(single.sizes()==gate.sizes()&&single.sizes()==projection.sizes(),"matching original shape");
  TORCH_CHECK(single.is_contiguous()&&gate.is_contiguous()&&projection.is_contiguous(),"contiguous original operands");
  TORCH_CHECK(single.scalar_type()==at::kBFloat16&&gate.scalar_type()==at::kBFloat16&&projection.scalar_type()==at::kBFloat16,"original BF16 operands");
  const c10::cuda::CUDAGuard guard(single.device());
  auto current=torch::empty_like(single);
  auto sigmoid=save?torch::empty_like(single):torch::empty({0},single.options());
  const int64_t count=single.numel();
  if(count) {
    using bf=__nv_bfloat16;
    const auto *a=reinterpret_cast<const bf*>(single.data_ptr());
    const auto *g=reinterpret_cast<const bf*>(gate.data_ptr());
    const auto *p=reinterpret_cast<const bf*>(projection.data_ptr());
    auto *c=reinterpret_cast<bf*>(current.data_ptr());
    auto *s=save?reinterpret_cast<bf*>(sigmoid.data_ptr()):nullptr;
    const auto stream=at::cuda::getCurrentCUDAStream();
    using namespace af3_local_outer_gate_residual_sm80;
    if(cfg==0) {if(save)launch<128,true>(a,g,p,c,s,count,stream);else launch<128,false>(a,g,p,c,s,count,stream);}
    else {if(save)launch<256,true>(a,g,p,c,s,count,stream);else launch<256,false>(a,g,p,c,s,count,stream);}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {current,sigmoid};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gate_residual",&gate_residual);}
