#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <vector>
#include "inference_qkvg_singlehead_sm80.cuh"
namespace swa_a100_infer_single80 {
struct Init {int active,attributes,occupancy,cold_capture;};
std::mutex init_mutex;
std::map<std::pair<int,int>,Init> init_cache;
template<class G> int initialize(int cfg) {
 int device=-1;C10_CUDA_CHECK(cudaGetDevice(&device));
 std::lock_guard<std::mutex> lock(init_mutex);
 auto key=std::make_pair(device,cfg);auto found=init_cache.find(key);
 if(found!=init_cache.end())return found->second.active;
 cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
 C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(),&status));
 C10_CUDA_CHECK(cudaFuncSetAttribute(qkvg_fwd_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
 int active=0;
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,qkvg_fwd_kernel<G>,G::NTHR,G::SMEM));
 TORCH_CHECK(active>=1,"No resident head-group CTA");
 init_cache.emplace(key,Init{active,1,1,status!=cudaStreamCaptureStatusNone});return active;
}
template<class G> void launch(const QkvgFwdParams& p,int cfg) {
 initialize<G>(cfg);
 const int ntile=(p.M+G::ROWS-1)/G::ROWS;
 const int ctas=std::min(at::cuda::getCurrentDeviceProperties()->multiProcessorCount,ntile);
 // Each head group receives the exact original row tile allocation. Group0 and
 // group1 have disjoint head/channel stores and jointly one original weight set.
 const dim3 grid(ctas,4,1);
 qkvg_fwd_kernel<G><<<grid,G::NTHR,G::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<torch::Tensor> compute(torch::Tensor x,torch::Tensor mod,torch::Tensor cos,torch::Tensor sin,
 torch::Tensor wq,torch::Tensor wg,int64_t S,int64_t B,double eps,double qk_eps,bool save,int64_t cfg) {
 TORCH_CHECK(x.is_cuda()&&x.dim()==2&&S>0&&B>0&&x.size(1)==128&&x.size(0)%S==0,"Physical x/S/B");
 const auto dev=x.device();c10::cuda::CUDAGuard guard(dev);
 for(const auto& t:{x,wq,wg})TORCH_CHECK(t.device()==dev&&t.scalar_type()==torch::kBFloat16&&t.is_contiguous(),"CUDA BF contiguous x/weights");
 for(const auto& t:{mod,cos,sin})TORCH_CHECK(t.device()==dev&&t.scalar_type()==torch::kFloat32&&t.is_contiguous(),"CUDA F32 contiguous mod/RoPE");
 TORCH_CHECK(wq.dim()==2&&wq.size(0)==384&&wq.size(1)==128&&wg.dim()==2&&wg.size(0)==128&&wg.size(1)==128,"Original weight dimensions");
 TORCH_CHECK(mod.numel()==B*S*768&&cos.numel()==B*S*16&&sin.numel()==B*S*16,"Original physical conditioning/RoPE");
 const int64_t M=x.size(0),N=M/S;
 auto qh=torch::empty({N,4,S,32},x.options());auto kh=torch::empty_like(qh);auto vh=torch::empty_like(qh);
 auto g=torch::empty_like(x);
 torch::Tensor xs,pqs,pks;
 if(save){xs=torch::empty_like(x);pqs=torch::empty_like(x);pks=torch::empty_like(x);}
 auto bf=[](const torch::Tensor& t){return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr());};
 auto bfw=[](torch::Tensor& t){return reinterpret_cast<__nv_bfloat16*>(t.data_ptr());};
 QkvgFwdParams p{bf(x),mod.data_ptr<float>(),cos.data_ptr<float>(),sin.data_ptr<float>(),bf(wq),bf(wg),
  bfw(qh),bfw(kh),bfw(vh),bfw(g),save?bfw(xs):nullptr,save?bfw(pqs):nullptr,save?bfw(pks):nullptr,
  (int)M,(int)S,(int)B,(float)eps,(float)qk_eps};
 switch(cfg){
  case 0:launch<QkvgCfg<4,1>>(p,cfg);break;
  case 1:launch<QkvgCfg<8,1>>(p,cfg);break;
  case 2:launch<QkvgCfg<4,2>>(p,cfg);break;
  case 3:launch<QkvgCfg<8,2>>(p,cfg);break;
  case 4:launch<QkvgCfg<2,1>>(p,cfg);break;
  case 5:launch<QkvgCfg<2,2>>(p,cfg);break;
  default:TORCH_CHECK(false,"Unknown fixed head-group configuration");
 }
 std::vector<torch::Tensor> out{qh,kh,vh,g};if(save){out.push_back(xs);out.push_back(pqs);out.push_back(pks);}return out;
}
std::vector<std::vector<int>> cache_snapshot() {
 std::lock_guard<std::mutex> lock(init_mutex);std::vector<std::vector<int>> out;
 for(const auto& kv:init_cache)out.push_back({kv.first.first,kv.first.second,kv.second.active,kv.second.attributes,kv.second.occupancy,kv.second.cold_capture});return out;
}
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <vector>
#include "inference_qkvg_headgroup_sm80.cuh"
namespace swa_a100_infer_group80 {
struct Init {int active,attributes,occupancy,cold_capture;};
std::mutex init_mutex;
std::map<std::pair<int,int>,Init> init_cache;
template<class G> int initialize(int cfg) {
 int device=-1;C10_CUDA_CHECK(cudaGetDevice(&device));
 std::lock_guard<std::mutex> lock(init_mutex);
 auto key=std::make_pair(device,cfg);auto found=init_cache.find(key);
 if(found!=init_cache.end())return found->second.active;
 cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
 C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(),&status));
 C10_CUDA_CHECK(cudaFuncSetAttribute(qkvg_fwd_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
 int active=0;
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,qkvg_fwd_kernel<G>,G::NTHR,G::SMEM));
 TORCH_CHECK(active>=1,"No resident head-group CTA");
 init_cache.emplace(key,Init{active,1,1,status!=cudaStreamCaptureStatusNone});return active;
}
template<class G> void launch(const QkvgFwdParams& p,int cfg) {
 initialize<G>(cfg);
 const int ntile=(p.M+G::ROWS-1)/G::ROWS;
 const int ctas=std::min(at::cuda::getCurrentDeviceProperties()->multiProcessorCount,ntile);
 // Each head group receives the exact original row tile allocation. Group0 and
 // group1 have disjoint head/channel stores and jointly one original weight set.
 const dim3 grid(ctas,2,1);
 qkvg_fwd_kernel<G><<<grid,G::NTHR,G::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<torch::Tensor> compute(torch::Tensor x,torch::Tensor mod,torch::Tensor cos,torch::Tensor sin,
 torch::Tensor wq,torch::Tensor wg,int64_t S,int64_t B,double eps,double qk_eps,bool save,int64_t cfg) {
 TORCH_CHECK(x.is_cuda()&&x.dim()==2&&S>0&&B>0&&x.size(1)==128&&x.size(0)%S==0,"Physical x/S/B");
 const auto dev=x.device();c10::cuda::CUDAGuard guard(dev);
 for(const auto& t:{x,wq,wg})TORCH_CHECK(t.device()==dev&&t.scalar_type()==torch::kBFloat16&&t.is_contiguous(),"CUDA BF contiguous x/weights");
 for(const auto& t:{mod,cos,sin})TORCH_CHECK(t.device()==dev&&t.scalar_type()==torch::kFloat32&&t.is_contiguous(),"CUDA F32 contiguous mod/RoPE");
 TORCH_CHECK(wq.dim()==2&&wq.size(0)==384&&wq.size(1)==128&&wg.dim()==2&&wg.size(0)==128&&wg.size(1)==128,"Original weight dimensions");
 TORCH_CHECK(mod.numel()==B*S*768&&cos.numel()==B*S*16&&sin.numel()==B*S*16,"Original physical conditioning/RoPE");
 const int64_t M=x.size(0),N=M/S;
 auto qh=torch::empty({N,4,S,32},x.options());auto kh=torch::empty_like(qh);auto vh=torch::empty_like(qh);
 auto g=torch::empty_like(x);
 torch::Tensor xs,pqs,pks;
 if(save){xs=torch::empty_like(x);pqs=torch::empty_like(x);pks=torch::empty_like(x);}
 auto bf=[](const torch::Tensor& t){return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr());};
 auto bfw=[](torch::Tensor& t){return reinterpret_cast<__nv_bfloat16*>(t.data_ptr());};
 QkvgFwdParams p{bf(x),mod.data_ptr<float>(),cos.data_ptr<float>(),sin.data_ptr<float>(),bf(wq),bf(wg),
  bfw(qh),bfw(kh),bfw(vh),bfw(g),save?bfw(xs):nullptr,save?bfw(pqs):nullptr,save?bfw(pks):nullptr,
  (int)M,(int)S,(int)B,(float)eps,(float)qk_eps};
 switch(cfg){
  case 0:launch<QkvgCfg<4,1>>(p,cfg);break;
  case 1:launch<QkvgCfg<8,1>>(p,cfg);break;
  case 2:launch<QkvgCfg<4,2>>(p,cfg);break;
  case 3:launch<QkvgCfg<8,2>>(p,cfg);break;
  case 4:launch<QkvgCfg<2,1>>(p,cfg);break;
  case 5:launch<QkvgCfg<2,2>>(p,cfg);break;
  default:TORCH_CHECK(false,"Unknown fixed head-group configuration");
 }
 std::vector<torch::Tensor> out{qh,kh,vh,g};if(save){out.push_back(xs);out.push_back(pqs);out.push_back(pks);}return out;
}
std::vector<std::vector<int>> cache_snapshot() {
 std::lock_guard<std::mutex> lock(init_mutex);std::vector<std::vector<int>> out;
 for(const auto& kv:init_cache)out.push_back({kv.first.first,kv.first.second,kv.second.active,kv.second.attributes,kv.second.occupancy,kv.second.cold_capture});return out;
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("singlehead",&swa_a100_infer_single80::compute,pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("headgroup",&swa_a100_infer_group80::compute,pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("singlehead_cache",&swa_a100_infer_single80::cache_snapshot);
 m.def("headgroup_cache",&swa_a100_infer_group80::cache_snapshot);
}
