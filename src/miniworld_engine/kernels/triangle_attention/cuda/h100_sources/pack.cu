// One CUDA launch stages either BF16 weights or FP32 master weights.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <ATen/ops/fill_native.h>
#include <torch/library.h>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>

// PyTorch's graph replay updates Philox seed/offset scalars with aten::fill_.
// Keep its generator, counter reservation and stream ordering unchanged, but
// use our CUDA writer for the two scalars owned by a graph containing this op.
// All other tensors go directly to the original native fill implementation.
// CUDA graph user objects bound the pointer registration to graph/exec lifetime;
// a destroyed graph cannot leave an allocator address registered indefinitely.
namespace {
struct GraphRngRegistry {
  std::mutex mutex;
  std::unordered_map<void*, size_t> references;
};
GraphRngRegistry& graph_rng_registry() {
  // CUDA may run user-object destructors during runtime shutdown. Keep this
  // small host registry alive until process exit instead of destructing early.
  static auto* registry = new GraphRngRegistry;
  return *registry;
}
struct GraphRngPointers {
  void* seed;
  void* offset;
  GraphRngPointers(void* s, void* o) : seed(s), offset(o) {
    auto& r = graph_rng_registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    ++r.references[seed]; ++r.references[offset];
  }
  ~GraphRngPointers() {
    auto& r = graph_rng_registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (auto* pointer : {seed, offset}) {
      auto it = r.references.find(pointer);
      if (it != r.references.end() && --it->second == 0) r.references.erase(it);
    }
  }
};
__global__ void graph_rng_scalar_write(int64_t* destination, int64_t value) {
  *destination = value;
}
at::Tensor& graph_rng_fill(at::Tensor& tensor, const at::Scalar& value) {
  bool owned = false;
  if (tensor.is_cuda() && tensor.scalar_type() == at::kLong && tensor.numel() == 1) {
    auto& r = graph_rng_registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    owned = r.references.count(tensor.data_ptr()) != 0;
  }
  if (!owned) return at::native::fill_(tensor, value);
  c10::cuda::CUDAGuard guard(tensor.device());
  auto* destination = tensor.data_ptr<int64_t>();
  int64_t scalar = value.toLong();
  cudaKernel_t kernel;
  C10_CUDA_CHECK(cudaGetKernel(&kernel, graph_rng_scalar_write));
  void* args[] = {&destination, &scalar};
  C10_CUDA_CHECK(cudaLaunchKernel((void*)kernel, dim3(1), dim3(1), args, 0,
                                at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tensor;
}
void enable_graph_rng_writer() {
  static auto registration = [] {
    auto library = std::make_unique<torch::Library>(
        torch::Library::IMPL, "aten", c10::DispatchKey::CUDA, __FILE__, __LINE__);
    library->impl("fill_.Scalar", &graph_rng_fill);
    return library;
  }();
}
void CUDART_CB release_graph_rng_pointers(void* pointers) {
  delete static_cast<GraphRngPointers*>(pointers);
}
void register_graph_rng(const at::PhiloxCudaState& state) {
  if (!state.captured_) return;
  cudaStreamCaptureStatus status;
  cudaGraph_t graph = nullptr;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(at::cuda::getCurrentCUDAStream(),
                                        &status, nullptr, &graph, nullptr, nullptr));
  TORCH_CHECK(status == cudaStreamCaptureStatusActive && graph,
              "Philox graph state requires an active graph");
  auto pointers = std::make_unique<GraphRngPointers>(state.seed_.ptr, state.offset_.ptr);
  cudaUserObject_t object;
  C10_CUDA_CHECK(cudaUserObjectCreate(&object, pointers.get(), release_graph_rng_pointers,
                                    1, cudaUserObjectNoDestructorSync));
  pointers.release();
  auto retained = cudaGraphRetainUserObject(graph, object, 1, cudaGraphUserObjectMove);
  if (retained != cudaSuccess) {
    C10_CUDA_CHECK(cudaUserObjectRelease(object, 1));
    C10_CUDA_CHECK(retained);
  }
}
size_t graph_rng_pointer_count() {
  auto& r = graph_rng_registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  return r.references.size();
}
}  // namespace
struct Pack {void const* ptr[6];bool fp32[6];};
__global__ void pack_parameters(Pack p,__nv_bfloat16* wall,__nv_bfloat16* wo,int C,int H) {
  int64_t idx=int64_t(blockIdx.x)*256+threadIdx.x,n=int64_t(4*C+64)*C;
  if(idx<n+int64_t(C)*C) {
    int which;int64_t off;__nv_bfloat16* dst;
    if(idx<n) {
      dst=wall+idx;which=idx/(int64_t(C)*C);off=idx%(int64_t(C)*C);
      if(which==4 && off>=H*C) {*dst=__float2bfloat16(0.f);return;}
    } else {which=5;off=idx-n;dst=wo+off;}
    *dst=p.fp32[which]?__float2bfloat16(reinterpret_cast<float const*>(p.ptr[which])[off]):reinterpret_cast<__nv_bfloat16 const*>(p.ptr[which])[off];
  }
}
std::vector<torch::Tensor> pack(std::vector<torch::Tensor> w) {
  TORCH_CHECK(w.size()==6,"six projection weights required");
  TORCH_CHECK(w[0].is_cuda() && w[0].dim()==2,"CUDA weights required");
  c10::cuda::CUDAGuard guard(w[0].device());int C=w[0].size(1),H=C/32;
  TORCH_CHECK(C>=64 && C<=512 && C%64==0,"C64..512 required");
  Pack p;
  for(int j=0;j<6;++j) {
    TORCH_CHECK(w[j].device()==w[0].device() && w[j].is_contiguous() && w[j].sizes()==torch::IntArrayRef({j==4?H:C,C}),"weight shape/device/layout mismatch");
    TORCH_CHECK(w[j].scalar_type()==torch::kBFloat16 || w[j].scalar_type()==torch::kFloat32,"BF16 or FP32 weights required");
    p.ptr[j]=w[j].data_ptr();p.fp32[j]=w[j].scalar_type()==torch::kFloat32;
  }
  auto wall=torch::empty({4*C+64,C},w[0].options().dtype(torch::kBFloat16));
  auto wo=torch::empty({C,C},wall.options());
  int64_t total=wall.numel()+wo.numel();
  // Resolve the kernel handle explicitly: CUDA12.9 otherwise probes the host
  // symbol with cuKernelGetFunction and logs INVALID_HANDLE before falling back.
  cudaKernel_t kernel;C10_CUDA_CHECK(cudaGetKernel(&kernel,pack_parameters));
  auto wp=(__nv_bfloat16*)wall.data_ptr(),op=(__nv_bfloat16*)wo.data_ptr();
  void* args[]={&p,&wp,&op,&C,&H};
  C10_CUDA_CHECK(cudaLaunchKernel((void*)kernel,dim3((total+255)/256),dim3(256),args,0,at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {wall,wo};
}
struct Finish {void* ptr[8];bool fp32[8];};
__global__ void finish_parameters(float const* g,float const* wo,float const* norm,Finish out,int C,int H) {
  int64_t idx=int64_t(blockIdx.x)*256+threadIdx.x;
  int64_t n0=2*C,n1=n0+int64_t(4)*C*C,n2=n1+H*C,total=n2+int64_t(C)*C;
  if(idx>=total)return;
  int which;int64_t off;float val;
  if(idx<n0){which=idx/C;off=idx%C;val=norm[idx];}
  else if(idx<n1){int64_t pos=idx-n0;which=2+pos/(int64_t(C)*C);off=pos%(int64_t(C)*C);val=g[pos];}
  else if(idx<n2){which=6;off=idx-n1;val=g[int64_t(4)*C*C+off];}
  else {which=7;off=idx-n2;val=wo[off];}
  if(out.fp32[which])reinterpret_cast<float*>(out.ptr[which])[off]=val;
  else reinterpret_cast<__nv_bfloat16*>(out.ptr[which])[off]=__float2bfloat16(val);
}
std::vector<torch::Tensor> finish(torch::Tensor g,torch::Tensor wo,torch::Tensor norm,std::vector<bool> fp32) {
  TORCH_CHECK(g.is_cuda() && g.dim()==2 && g.scalar_type()==torch::kFloat32 && g.is_contiguous(),"FP32 gradient matrix required");
  c10::cuda::CUDAGuard guard(g.device());int C=g.size(1),H=C/32;
  TORCH_CHECK(C>=64 && C<=512 && C%64==0 && g.size(0)==4*C+64 && fp32.size()==8,"gradient metadata mismatch");
  TORCH_CHECK(wo.device()==g.device() && wo.scalar_type()==torch::kFloat32 && wo.is_contiguous() && wo.sizes()==torch::IntArrayRef({C,C}),"dWo mismatch");
  TORCH_CHECK(norm.device()==g.device() && norm.scalar_type()==torch::kFloat32 && norm.is_contiguous() && norm.sizes()==torch::IntArrayRef({2,C}),"LN gradient mismatch");
  std::vector<torch::Tensor> result;Finish out;
  for(int j=0;j<8;++j) {
    auto opt=g.options().dtype(fp32[j]?torch::kFloat32:torch::kBFloat16);
    result.push_back(j<2?torch::empty({C},opt):torch::empty({j==6?H:C,C},opt));
    out.ptr[j]=result.back().data_ptr();out.fp32[j]=fp32[j];
  }
  int64_t n=2*C+int64_t(5)*C*C+H*C;
  finish_parameters<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(g.data_ptr<float>(),wo.data_ptr<float>(),norm.data_ptr<float>(),out,C,H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
// Fixed-order partial reduction for split-K weight and LN affine gradients.
// Each output element has exactly one owner; no atomics or framework kernels.
__global__ void reduce_partials_kernel(float const* input,float* output,int64_t width,int count) {
  const int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;
  if(i>=width)return;
  float value=0.f;
  for(int p=0;p<count;++p)value+=input[int64_t(p)*width+i];
  output[i]=value;
}
torch::Tensor reduce_partials(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==torch::kFloat32 && input.is_contiguous()
    && input.dim()>=2 && input.size(0)>0,"contiguous nonempty FP32 partials required");
  c10::cuda::CUDAGuard guard(input.device());
  auto output=torch::empty(input.sizes().slice(1),input.options());
  const auto* src=input.data_ptr<float>();auto* dst=output.data_ptr<float>();
  int64_t width=output.numel();int count=input.size(0);
  cudaKernel_t kernel;C10_CUDA_CHECK(cudaGetKernel(&kernel,reduce_partials_kernel));
  void* args[]={&src,&dst,&width,&count};
  C10_CUDA_CHECK(cudaLaunchKernel((void*)kernel,dim3((width+255)/256),dim3(256),args,0,at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
// Same Philox subsequences, BF16 uniform rounding and counter advance as
// torch.rand(B,1,L,C,dtype=BF16), with threshold and scale in this one launch.
// PhiloxCudaState carries graph-owned seed/offset pointers during capture.
__global__ void drop_scale_kernel(__nv_bfloat16* out,int64_t n,at::PhiloxCudaState rng,float p,float inverse) {
  auto seeds=at::cuda::philox::unpack(rng);
  int64_t idx=int64_t(blockIdx.x)*256+threadIdx.x,stride=int64_t(gridDim.x)*256;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),idx,std::get<1>(seeds),&state);
  for(int64_t base=idx;base<n;base+=4*stride) {
    float4 values=curand_uniform4(&state);
    #pragma unroll
    for(int j=0;j<4;++j) {
      int64_t pos=base+j*stride;
      if(pos<n) {
        float value=__bfloat162float(__float2bfloat16_rn((&values.x)[j]));
        if(value==1.f)value=0.f;
        out[pos]=__float2bfloat16_rn(value>p?inverse:0.f);
      }
    }
  }
}
torch::Tensor drop_scale(torch::Tensor pair,double probability) {
  TORCH_CHECK(pair.is_cuda() && pair.scalar_type()==torch::kBFloat16 && pair.dim()==4
    && pair.size(1)==pair.size(2) && probability>0. && probability<1.,"BF16 square pair and 0<p<1 required");
  c10::cuda::CUDAGuard guard(pair.device());
  const char* native_rng_env = std::getenv("MINIWORLD_TRIATTN_H100_NATIVE_RNG");
  const bool native_graph_rng = !native_rng_env || std::string(native_rng_env) != "0";
  if (native_graph_rng) enable_graph_rng_writer();
  auto out=torch::empty({pair.size(0),pair.size(1),pair.size(3)},pair.options());
  int64_t n=out.numel();auto* prop=at::cuda::getCurrentDeviceProperties();
  int blocks=std::min<int64_t>((n+255)/256,prop->multiProcessorCount*(prop->maxThreadsPerMultiProcessor/256));
  uint64_t increment=((n-1)/(int64_t(blocks)*256*4)+1)*4;
  auto* gen=at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState state;
  {std::lock_guard<std::mutex> lock(gen->mutex_);state=gen->philox_cuda_state(increment);}
  if (native_graph_rng) register_graph_rng(state);
  auto* dst=(__nv_bfloat16*)out.data_ptr();float p=probability,inverse=1.f/(1.f-p);
  cudaKernel_t kernel;C10_CUDA_CHECK(cudaGetKernel(&kernel,drop_scale_kernel));
  void* args[]={&dst,&n,&state,&p,&inverse};
  C10_CUDA_CHECK(cudaLaunchKernel((void*)kernel,dim3(blocks),dim3(256),args,0,at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("pack",&pack);m.def("finish",&finish);m.def("reduce_partials",&reduce_partials);m.def("drop_scale",&drop_scale);m.def("graph_rng_pointer_count",&graph_rng_pointer_count);}
