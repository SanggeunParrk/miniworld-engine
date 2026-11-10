// Generic A100 tails: FP32/BF16 elementwise gates and stable softmax.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>

template<class T> struct alignas(16) Pack { T v[16 / sizeof(T)]; };

template<class T, int MODE, bool BACKWARD>
__global__ void binary_kernel(const T* a, const T* b, const T* dy, T* y, T* db, int64_t n, bool aligned) {
  constexpr int V = 16 / sizeof(T);
  const int64_t first = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * V;
  if (first >= n) return;
  Pack<T> aa, bb, gg, yy, zz;
  const bool full = first + V <= n;
  if (aligned && full) {
    aa = *reinterpret_cast<const Pack<T>*>(a + first);
    bb = *reinterpret_cast<const Pack<T>*>(b + first);
    if constexpr (BACKWARD) gg = *reinterpret_cast<const Pack<T>*>(dy + first);
  } else {
#pragma unroll
    for (int j = 0; j < V; ++j) if (first + j < n) {
      aa.v[j] = a[first+j]; bb.v[j] = b[first+j];
      if constexpr (BACKWARD) gg.v[j] = dy[first+j];
    }
  }
#pragma unroll
  for (int j = 0; j < V; ++j) if (first + j < n) {
    float x = float(aa.v[j]), v = float(bb.v[j]);
    float s = 0.f;
    if constexpr (MODE >= 2) s = 1.f / (1.f + expf(-x));
    if constexpr (!BACKWARD) {
      yy.v[j] = T(MODE == 0 ? x+v : MODE == 1 ? x*v : MODE == 2 ? s*v : x*s*v);
    } else {
      const float g = float(gg.v[j]);
      yy.v[j] = T(g * (MODE == 0 ? 1.f : MODE == 1 ? v : MODE == 2 ? s*(1.f-s)*v : s*(1.f+x*(1.f-s))*v));
      zz.v[j] = T(g * (MODE == 0 ? 1.f : MODE == 1 ? x : MODE == 2 ? s : x*s));
    }
  }
  if (full) {
    *reinterpret_cast<Pack<T>*>(y + first) = yy;
    if constexpr (BACKWARD) *reinterpret_cast<Pack<T>*>(db + first) = zz;
  } else {
#pragma unroll
    for (int j = 0; j < V; ++j) if (first+j < n) {
      y[first+j] = yy.v[j];
      if constexpr (BACKWARD) db[first+j] = zz.v[j];
    }
  }
}

template<class T, int MODE>
void launch_binary(torch::Tensor a, torch::Tensor b, torch::Tensor dy, torch::Tensor y, torch::Tensor db) {
  constexpr int V = 16 / sizeof(T);
  const int blocks = (a.numel()+256*V-1)/(256*V);
  const bool aligned = ((uintptr_t)a.data_ptr() | (uintptr_t)b.data_ptr() | (dy.numel() ? (uintptr_t)dy.data_ptr() : 0)) % 16 == 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (dy.numel()) binary_kernel<T,MODE,true><<<blocks,256,0,stream>>>(a.data_ptr<T>(),b.data_ptr<T>(),dy.data_ptr<T>(),y.data_ptr<T>(),db.data_ptr<T>(),a.numel(),aligned);
  else binary_kernel<T,MODE,false><<<blocks,256,0,stream>>>(a.data_ptr<T>(),b.data_ptr<T>(),nullptr,y.data_ptr<T>(),nullptr,a.numel(),aligned);
}

std::vector<torch::Tensor> binary(torch::Tensor a, torch::Tensor b, torch::Tensor dy, int64_t mode) {
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && b.is_contiguous() && a.sizes()==b.sizes() && a.device()==b.device() && a.scalar_type()==b.scalar_type(), "binary: matching contiguous CUDA operands");
  TORCH_CHECK(mode >= 0 && mode <= 3, "binary: mode");
  const c10::cuda::CUDAGuard guard(a.device());
  auto y = torch::empty_like(a), db = dy.numel() ? torch::empty_like(a) : torch::empty({0}, a.options());
  if (a.numel()) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, a.scalar_type(), "a100_binary", [&] {
      switch (mode) {
        case 0: launch_binary<scalar_t,0>(a,b,dy,y,db); break;
        case 1: launch_binary<scalar_t,1>(a,b,dy,y,db); break;
        case 2: launch_binary<scalar_t,2>(a,b,dy,y,db); break;
        case 3: launch_binary<scalar_t,3>(a,b,dy,y,db); break;
      }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {y,db};
}

__global__ void softmax_kernel(const float* x, float* p, int64_t rows, int n) {
  const int64_t row = blockIdx.x;
  __shared__ float scratch[256];
  float hi=-INFINITY;
  for(int j=threadIdx.x;j<n;j+=256) hi=fmaxf(hi,x[row*n+j]);
  scratch[threadIdx.x]=hi; __syncthreads();
  for(int s=128;s;s>>=1) { if(threadIdx.x<s) scratch[threadIdx.x]=fmaxf(scratch[threadIdx.x],scratch[threadIdx.x+s]); __syncthreads(); }
  hi=scratch[0]; float sum=0.f;
  for(int j=threadIdx.x;j<n;j+=256) sum+=isfinite(hi) ? expf(x[row*n+j]-hi) : 0.f;
  scratch[threadIdx.x]=sum; __syncthreads();
  for(int s=128;s;s>>=1) { if(threadIdx.x<s) scratch[threadIdx.x]+=scratch[threadIdx.x+s]; __syncthreads(); }
  sum=scratch[0];
  for(int j=threadIdx.x;j<n;j+=256) p[row*n+j]=sum>0.f ? expf(x[row*n+j]-hi)/sum : 0.f;
}

torch::Tensor softmax(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kFloat && x.is_contiguous() && x.size(-1)>0, "softmax: contiguous FP32 CUDA rows");
  const c10::cuda::CUDAGuard guard(x.device());
  auto p=torch::empty_like(x);
  if(x.numel()) softmax_kernel<<<x.numel()/x.size(-1),256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),p.data_ptr<float>(),x.numel()/x.size(-1),x.size(-1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return p;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("binary", &binary); m.def("softmax", &softmax); }
