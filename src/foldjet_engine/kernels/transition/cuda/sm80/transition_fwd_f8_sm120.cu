// transition_fwd_f8_sm120.cu -- torch binding of the OPT-IN e4m3 Transition inference forward on the RTX 5090 (tr_fwd_f8_sm120.cuh; fp8_sm120.py), current stream:
//   fwd8  x [T, D] bf16, w [NCHUNK * SLOT] uint8 (fp8_sm120._pack), gb [2 D] f32 slots, sc [>= 5] f32, peak [1] f32 -> out [T, D] bf16
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

#include "tr_fwd_f8_sm120.cuh"

namespace {
int num_sms() {
  static int n = 0;
  if (n == 0) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, at::cuda::current_device());
  return n;
}
template <class K>
void set_smem(K kernel, int bytes) {
  static std::vector<const void*> done;
  const void* k = reinterpret_cast<const void*>(kernel);
  if (std::find(done.begin(), done.end(), k) == done.end()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    done.push_back(k);
  }
}

template <int D, int H>
torch::Tensor fwd_t(torch::Tensor x, torch::Tensor w, torch::Tensor gb, torch::Tensor sc, torch::Tensor peak, double eps) {
  using G = a100::TrF8Cfg<D, H>;
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.size(1) == D && x.scalar_type() == at::kBFloat16, "x: [T, D] bf16");
  TORCH_CHECK(w.scalar_type() == at::kByte && w.is_contiguous() && w.numel() == (int64_t)G::NCHUNK * G::SLOT, "packed e4m3 weights");
  TORCH_CHECK(gb.scalar_type() == at::kFloat && gb.numel() == 2 * D && sc.scalar_type() == at::kFloat && sc.numel() >= 5, "affine / scales");
  TORCH_CHECK(peak.scalar_type() == at::kFloat && peak.numel() >= 1, "peak");
  const int64_t T = x.size(0);
  auto out = torch::empty_like(x);
  if (T == 0) return out;
  a100::Tr8Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.w = reinterpret_cast<const uint8_t*>(w.data_ptr());
  p.gb = reinterpret_cast<const float4*>(gb.data_ptr());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.sc = sc.data_ptr<float>();
  p.peak = peak.data_ptr<float>();
  TORCH_CHECK(T <= (int64_t)1 << 28, "fwd8: rows");
  p.T = (int)T; p.num_tiles = (p.T + G::BM - 1) / G::BM; p.eps = (float)eps;
  const int g = std::min(p.num_tiles, num_sms() * G::MINB);
  TORCH_CHECK(p.num_tiles / g + 2 <= G::MAX_ITEMS, "fwd8: too many work items per CTA (split the call)");
  set_smem(a100::tr_fwd_f8_kernel<G>, G::SMEM);
  a100::tr_fwd_f8_kernel<G><<<g, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}  // namespace

torch::Tensor fwd8(int64_t D, int64_t H, torch::Tensor x, torch::Tensor w, torch::Tensor gb, torch::Tensor sc, torch::Tensor peak, double eps) {
#define F8_CASE(DV, HV) \
  if (D == DV && H == HV) return fwd_t<DV, HV>(x, w, gb, sc, peak, eps);
  F8_CASE(128, 256) F8_CASE(128, 512) F8_CASE(256, 512) F8_CASE(256, 1024)
#undef F8_CASE
  TORCH_CHECK(false, "e4m3 forward: no build for D = ", D, ", H = ", H);
  return {};
}

int64_t slot_bytes(int64_t D) { return D == 128 ? a100::TrF8Cfg<128, 256>::SLOT : a100::TrF8Cfg<256, 512>::SLOT; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd8", &fwd8);
  m.def("slot_bytes", &slot_bytes);
}
