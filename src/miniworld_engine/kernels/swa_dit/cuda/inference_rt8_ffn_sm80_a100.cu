#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <vector>
#include "inference_rt8_ffn_sm80_a100.cuh"
namespace swa_a100_infer_rt8_ffn80 {
struct Init { int active, set_attribute_calls, occupancy_calls, cold_capture; };
std::mutex cache_mutex;
std::map<std::pair<int,int>,Init> device_kernel_cache;
template<class G> int initialize(int cfg) {
 int device=-1;C10_CUDA_CHECK(cudaGetDevice(&device));
 std::lock_guard<std::mutex> lock(cache_mutex);
 const auto key=std::make_pair(device,cfg);
 auto found=device_kernel_cache.find(key);
 if(found!=device_kernel_cache.end())return found->second.active;
 cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
 C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(),&status));
 // Exact original cold attribute API; no cold-capture rejection or event query.
 // Fresh public-like kernel attributes never depend on input/weight contents.
 C10_CUDA_CHECK(cudaFuncSetAttribute(ffn_fwd_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
 int active=0;
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,ffn_fwd_kernel<G>,G::NTHR,G::SMEM));
 TORCH_CHECK(active>=1,"No valid resident CTA");
 device_kernel_cache.emplace(key,Init{active,1,1,status!=cudaStreamCaptureStatusNone});
 return active;
}
template<class G> void launch(const FfnFwdParams& p,int cfg) {
 const int active=initialize<G>(cfg);
 const int ntile=((p.S+G::ROW_TILE-1)/G::ROW_TILE)*(p.M/p.S);
 const int ctas=std::min(at::cuda::getCurrentDeviceProperties()->multiProcessorCount*active,(ntile+G::NW-1)/G::NW);
 ffn_fwd_kernel<G><<<ctas,G::NTHR,G::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
torch::Tensor compute(torch::Tensor qi,torch::Tensor g,torch::Tensor o,torch::Tensor mod,
 torch::Tensor wo,torch::Tensor wu,torch::Tensor wd,int64_t S,int64_t B,double eps,int64_t cfg) {
 TORCH_CHECK(qi.is_cuda()&&qi.dim()==2&&S>0&&B>0,"CUDA operands/positive physical dimensions");
 const auto device=qi.device();c10::cuda::CUDAGuard guard(device);
 const int M=qi.size(0);
 for(auto t:{qi,g,o,wo,wu,wd})TORCH_CHECK(t.device()==device&&t.scalar_type()==at::kBFloat16&&t.is_contiguous(),"Original contiguous BF storage");
 for(auto t:{qi,g,o})TORCH_CHECK(t.sizes()==at::IntArrayRef({M,128}),"Full actual row storage");
 TORCH_CHECK(mod.device()==device&&mod.scalar_type()==at::kFloat&&mod.is_contiguous()&&mod.numel()==B*S*768,"Physical FP32 modulation");
 TORCH_CHECK(M%S==0&&(M/S)%B==0&&wo.sizes()==at::IntArrayRef({128,128})&&wu.sizes()==at::IntArrayRef({512,128})&&wd.sizes()==at::IntArrayRef({128,256}),"Original full dimensions");
 auto out=torch::empty_like(qi);
 auto bp=[](torch::Tensor t){return (const __nv_bfloat16*)t.data_ptr();};
 FfnFwdParams p{bp(qi),bp(g),bp(o),mod.data_ptr<float>(),bp(wo),bp(wu),bp(wd),
  (__nv_bfloat16*)out.data_ptr(),nullptr,nullptr,nullptr,nullptr,M,(int)S,(int)B,(float)eps,nullptr};
 if(cfg==3)launch<FfnCfg<4,2,1,32,8>>(p,3);
 else if(cfg==4)launch<FfnCfg<2,2,1,32,8>>(p,4);
 else TORCH_CHECK(false,"E679 private primitive exposes only original RT8 cfg3/cfg4; public RT16 controls remain unchanged");
 return out;
}
std::vector<std::vector<int>> cache_snapshot() {
 std::lock_guard<std::mutex> lock(cache_mutex);
 std::vector<std::vector<int>> result;
 for(const auto& entry:device_kernel_cache)result.push_back({entry.first.first,entry.first.second,
  entry.second.active,entry.second.set_attribute_calls,entry.second.occupancy_calls,entry.second.cold_capture});
 return result;
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("compute",&swa_a100_infer_rt8_ffn80::compute,pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("cache_snapshot",&swa_a100_infer_rt8_ffn80::cache_snapshot);
}
