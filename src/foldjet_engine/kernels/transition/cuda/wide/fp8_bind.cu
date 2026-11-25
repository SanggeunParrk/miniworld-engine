// OPT-IN FP8 (docs/fp8.md): torch bindings of the e4m3-h wide forward (D 384 / 512): ln_swiglu_gemm_f8 (bf16 expand, h8 out) and
// squeeze_gemm_f8 (e4m3 squeeze + residual). Built per width (-DWIDE_D, -DWIDE_N, -DWIDE_SMS) by kernels/transition/cuda/fp8_sm90a.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <array>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#ifndef F8G1
#define F8G1 0   // 1: the expand in e4m3 too (w1p e4m3, scales[0] = sx, scales[1] = 1 / (sx s1))
#endif

void wide_lnsg_f8_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const float*, const float*, __nv_bfloat16*, float*, float*, int,
                         int, float, const float*, unsigned int*, int, cudaStream_t);
int wide_squeeze_f8_bn();
void wide_squeeze_f8_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, int, int, int, const float*, int,
                            cudaStream_t);

namespace {
constexpr int D = WIDE_D, H = WIDE_N * WIDE_D;
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                 const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    if (e != cudaSuccess || p == nullptr) throw std::runtime_error("cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}
// contiguous 2-D [rows][cols] tensor, element size from its dtype (bf16 or uint8), swizzle 64 / 128 B
const CUtensorMap& rm(const torch::Tensor& t, uint32_t box_inner, uint32_t box_outer, int swizzle = 128) {
  using Key = std::array<uint64_t, 6>;
  static std::map<Key, CUtensorMap> cache;
  static std::mutex lock;
  const int esz = (int)t.element_size();
  const Key key{reinterpret_cast<uint64_t>(t.data_ptr()), (uint64_t)t.size(1), (uint64_t)t.size(0), box_inner, box_outer,
                (uint64_t)(swizzle * 4 + esz)};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {(cuuint64_t)t.size(1), (cuuint64_t)t.size(0)};
  const cuuint64_t strides[1] = {(cuuint64_t)t.size(1) * esz};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t elem[2] = {1, 1};
  if (cudaSetDevice(t.get_device()) != cudaSuccess) throw std::runtime_error("cudaSetDevice failed");
  CUresult r = encoder()(&map, esz == 1 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, elem,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(r));
  return cache.emplace(key, map).first->second;
}
void expect(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("transition wide fp8 sm90a: ") + what);
}
}  // namespace

// x [M, D] bf16 (M a multiple of 128); w1p [2H, D] bf16 (64-row blocks [Wa 64 | Wb 64]); ws8 [D, H] uint8 (e4m3, K in the h8 column order);
// scales float[>= 4] (scales[2] = sh, scales[3] = 1 / (sh s2)); peak float[1] (atomicMax of max |h sh| as bits). Returns out [M, D] bf16.
torch::Tensor fwd_f8h(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor w1p, torch::Tensor ws8, torch::Tensor scales,
                      torch::Tensor peak, double eps) {
  const int64_t M = x.size(0);
  expect(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.size(1) == D, "x must be bf16 [M, D]");
  expect(M % 128 == 0, "row count must be a whole number of 128-row tiles");
#if F8G1
  expect(w1p.scalar_type() == torch::kUInt8 && w1p.is_contiguous() && w1p.size(0) == 2 * H && w1p.size(1) == D, "w1p must be uint8 (e4m3) [2H, D]");
#else
  expect(w1p.scalar_type() == torch::kBFloat16 && w1p.is_contiguous() && w1p.size(0) == 2 * H && w1p.size(1) == D, "w1p must be bf16 [2H, D]");
#endif
  expect(ws8.scalar_type() == torch::kUInt8 && ws8.is_contiguous() && ws8.size(0) == D && ws8.size(1) == H, "ws8 must be uint8 [D, H]");
  expect(gamma.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32, "gamma / beta fp32");
  expect(scales.scalar_type() == torch::kFloat32 && scales.numel() >= 4 && peak.scalar_type() == torch::kFloat32, "scales / peak fp32");
  auto out = torch::empty_like(x);
  auto h8 = torch::empty({M, H}, x.options().dtype(torch::kUInt8));
  auto f32 = x.options().dtype(torch::kFloat32);
  auto xn = torch::empty({1, D}, x.options());
  auto rstd = torch::empty({1}, f32), c1 = torch::empty({1}, f32);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto pk = reinterpret_cast<unsigned int*>(peak.data_ptr<float>());
  wide_lnsg_f8_launch(rm(x, 64, 64), rm(w1p, F8G1 ? 128 : 64, 128), rm(h8, 64, 64, 64), gamma.data_ptr<float>(), beta.data_ptr<float>(),
                      reinterpret_cast<__nv_bfloat16*>(xn.data_ptr()), rstd.data_ptr<float>(), c1.data_ptr<float>(), (int)M, (int)H, (float)eps,
                      scales.data_ptr<float>(), pk, WIDE_SMS, stream);
  const int bn = wide_squeeze_f8_bn();
  wide_squeeze_f8_launch(rm(h8, 128, 64), rm(ws8, 128, bn), rm(x, 64, 64), rm(out, 64, 64), (int)M, (int)H, (int)D, scales.data_ptr<float>(),
                         WIDE_SMS, stream);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fwd_f8h", &fwd_f8h, "OPT-IN FP8 wide Transition forward (h e4m3, e4m3 squeeze)"); }
