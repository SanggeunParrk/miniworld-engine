// A100 row/column dropout scale. Own Philox CUDA kernel, using torch's graph-aware generator state.
// Match native torch.rand's float4 mapping, BF16 rounding and upper-bound wrap, then threshold/scale in the same launch.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <curand_kernel.h>
#include <mutex>

template<class T>
__global__ void __launch_bounds__(256,4) row_scale_kernel(T *out,int64_t count,at::PhiloxCudaState state,T threshold,float inverse) {
  auto [seed,offset]=at::cuda::philox::unpack(state);
  const int64_t thread=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t stride=(int64_t)blockDim.x*gridDim.x;
  curandStatePhilox4_32_10_t generator;
  curand_init(seed,thread,offset,&generator);
  const int64_t rounded=((count-1)/(stride*4)+1)*stride*4;
  for(int64_t start=thread;start<rounded;start+=stride*4) {
    const float4 samples=curand_uniform4(&generator);
    const float values[4]={samples.x,samples.y,samples.z,samples.w};
#pragma unroll
    for(int i=0;i<4;++i) {
      const int64_t index=start+stride*i;
      if(index<count) {
        T value=T(values[i]);
        if(value==T(1.f))value=T(0.f);
        out[index]=T(value>threshold ? inverse : 0.f);
      }
    }
  }
}

torch::Tensor row_scale(torch::Tensor reference,double probability,int64_t dimension) {
  TORCH_CHECK(reference.is_cuda(),"CUDA tensor required");
  TORCH_CHECK(reference.scalar_type()==at::kBFloat16 || reference.scalar_type()==at::kFloat,"BF16/FP32 required");
  TORCH_CHECK(probability>=0. && probability<1.,"dropout probability must be in [0,1)");
  TORCH_CHECK(dimension>=0 && dimension<reference.dim(),"invalid broadcast dimension");
  const c10::cuda::CUDAGuard guard(reference.device());
  std::vector<int64_t> sizes(reference.sizes().begin(),reference.sizes().end());
  sizes[dimension]=1;
  auto out=at::empty(sizes,reference.options());
  const int64_t count=out.numel();
  if(count==0)return out;
  const auto *properties=at::cuda::getCurrentDeviceProperties();
  const int blocks=(int)std::min<int64_t>((count+255)/256,properties->multiProcessorCount*(properties->maxThreadsPerMultiProcessor/256));
  const uint64_t increment=((count-1)/(256ll*blocks*4)+1)*4;
  auto *generator=at::get_generator_or_default<at::CUDAGeneratorImpl>(std::nullopt,at::cuda::detail::getDefaultCUDAGenerator(reference.get_device()));
  at::PhiloxCudaState state;
  {std::lock_guard<std::mutex> lock(generator->mutex_);state=generator->philox_cuda_state(increment);}
  const float inverse=(float)(1./(1.-probability));
  const auto stream=at::cuda::getCurrentCUDAStream();
  if(reference.scalar_type()==at::kBFloat16)
    row_scale_kernel<at::BFloat16><<<blocks,256,0,stream>>>(out.data_ptr<at::BFloat16>(),count,state,at::BFloat16(probability),inverse);
  else row_scale_kernel<float><<<blocks,256,0,stream>>>(out.data_ptr<float>(),count,state,(float)probability,inverse);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("row_scale",&row_scale);}
