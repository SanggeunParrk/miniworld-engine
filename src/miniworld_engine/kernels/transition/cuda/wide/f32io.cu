// Torch binding of the fp32 inference Transition forward (kernels/transition_fwd_f32io.cu) at one channel width F32_D (64 / 128) and
// hidden multiple F32_N (2 / 4). The builds of the kernel (tile layout per CTA, CTAs per SM) are template instances; ``variant`` picks one.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <array>
#include <map>
#include <mutex>
#include <stdexcept>

#include "kernels/transition_fwd_f32io.cu"
#include "launch_util.cuh"

#ifndef F32_D
#error "F32_D (64 / 128) must be defined"
#endif
#ifndef F32_N
#define F32_N 4
#endif

namespace {

using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    if (e != cudaSuccess || p == nullptr) throw std::runtime_error("cuTensorMapEncodeTiled unavailable: the driver predates TMA");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}

// 2-D row-major contiguous [rows][cols] descriptor (fp32 or fp16), 128-byte swizzle, box innermost-first; cached on (pointer, shape, box)
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t box_inner, uint32_t box_outer) {
  using Key = std::array<uint64_t, 6>;
  static std::map<Key, CUtensorMap> cache;
  static std::mutex lock;
  const bool f32 = t.scalar_type() == torch::kFloat32;
  const uint64_t inner = (uint64_t)t.size(1), outer = (uint64_t)t.size(0);
  const Key key{reinterpret_cast<uint64_t>(t.data_ptr()), inner, outer, box_inner, box_outer, (uint64_t)f32};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {inner * (f32 ? 4 : 2)};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t elem[2] = {1, 1};
  if (cudaSetDevice(t.get_device()) != cudaSuccess) throw std::runtime_error("cudaSetDevice failed before cuTensorMapEncodeTiled");
  CUresult r = encoder()(&map, f32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, t.data_ptr(), dims, strides, box,
                         elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: result=", int(r), " inner=", inner, " outer=", outer,
              " box=", box_inner, "x", box_outer, " pointer_mod_128=", reinterpret_cast<uintptr_t>(t.data_ptr()) % 128);
  return cache.emplace(key, map).first->second;
}

void expect(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("transition f32io sm90a: ") + what);
}
void check_2d(const torch::Tensor& t, torch::ScalarType dt, int64_t rows, int64_t cols, const char* what) {
  expect(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dt && t.dim() == 2 && t.size(0) == rows && t.size(1) == cols, what);
}

constexpr int64_t D = F32_D, H = F32_N * F32_D;

template <int HS, int NWG, int NBUF, int CPS>
void launch(const torch::Tensor& x, const torch::Tensor& gamma, const torch::Tensor& beta, const torch::Tensor& wa, const torch::Tensor& wb,
            const torch::Tensor& wst, const torch::Tensor& sc, const torch::Tensor& out, float eps, int sms) {
  using C = F32Cfg<F32_D, F32_N, HS, NWG, NBUF, CPS>;
  auto fn = transition_fwd_f32io<F32_D, F32_N, HS, NWG, NBUF, CPS>;
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(fn), C::SMEM, "f32io fwd"), true);
  (void)ready;
  const int64_t M = x.size(0);
  const int tiles = (int)((M + C::ROWS - 1) / C::ROWS), grid = tiles < CPS * sms ? tiles : CPS * sms;
  fn<<<grid, 128 * NWG, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      tile_map(x, 32, 64), tile_map(wa, 64, HS), tile_map(wb, 64, HS), tile_map(wst, 64, HS), tile_map(out, 32, 64),
      gamma.data_ptr<float>(), beta.data_ptr<float>(), sc.data_ptr<float>(), tiles, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// x fp32 [M, D]; gamma / beta fp32 [D]; wa16, wb16 fp16 [H, D] (scaled); wst16 = (Ws / s_ws)^T fp16 [H, D]; sc fp32 [4]. Returns x + transition(x).
torch::Tensor fwd(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa, torch::Tensor wb, torch::Tensor wst,
                  torch::Tensor sc, double eps, int64_t variant, int64_t sms) {
  const int64_t M = x.size(0);
  check_2d(x, torch::kFloat32, M, D, "x must be contiguous fp32 [M, D]");
  expect(M > 0, "empty input");
  expect(gamma.is_cuda() && gamma.is_contiguous() && gamma.scalar_type() == torch::kFloat32 && gamma.numel() == D, "gamma must be fp32 [D]");
  expect(beta.is_cuda() && beta.is_contiguous() && beta.scalar_type() == torch::kFloat32 && beta.numel() == D, "beta must be fp32 [D]");
  check_2d(wa, torch::kHalf, H, D, "wa must be fp16 [H, D]");
  check_2d(wb, torch::kHalf, H, D, "wb must be fp16 [H, D]");
  check_2d(wst, torch::kHalf, H, D, "wst must be fp16 Ws^T [H, D]");
  expect(sc.is_cuda() && sc.scalar_type() == torch::kFloat32 && sc.numel() == 4 && sc.is_contiguous(), "sc must be fp32 [4]");
  auto out = torch::empty_like(x);
  const float e = (float)eps;
  const int s = (int)sms;
#if F32_D == 64
  switch (variant) {
    case 0: launch<64, 1, 2, 2>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
    case 1: launch<64, 1, 1, 3>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
    case 2: launch<64, 2, 2, 1>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
    default: expect(false, "variant must be 0 .. 2 at D = 64");
  }
#elif F32_D == 128
  switch (variant) {
    case 0: launch<32, 1, 1, 2>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
    case 1: launch<64, 2, 2, 1>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
    case 2: launch<64, 2, 1, 1>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
    default: expect(false, "variant must be 0 .. 2 at D = 128");
  }
#else
#error "F32_D must be 64 or 128"
#endif
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fwd, "fp32 inference Transition forward (+ residual), fp16 tensor cores at TF32 rounding", py::arg("x"), py::arg("gamma"),
        py::arg("beta"), py::arg("wa"), py::arg("wb"), py::arg("wst"), py::arg("sc"), py::arg("eps"), py::arg("variant"), py::arg("sms"));
  m.attr("width") = F32_D;
  m.attr("n") = F32_N;
}
