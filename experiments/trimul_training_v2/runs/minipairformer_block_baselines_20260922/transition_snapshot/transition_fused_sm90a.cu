// Torch bindings for the fused sm_90a Transition forward and backward.
//
// The two kernels live in `transition_fused_fwd_sm90a_kernel.cu` and
// `transition_fused_bwd_sm90a_kernel.cu`; this file is only the host side: it turns torch
// tensors into the 2-D TMA descriptors they expect, allocates the outputs and the backward's
// partial buffers, and calls the launchers on torch's current stream.
//
// Two things here are not obvious and were found the hard way:
//
//   * `cuTensorMapEncodeTiled` is a driver entry point. Resolving it through
//     `cudaGetDriverEntryPoint` keeps this extension off `-lcuda`, which torch's JIT build does
//     not link by default.
//   * A descriptor is 128 B of opaque state that has to be 64-B aligned and is baked into the
//     kernel's parameter buffer at launch, so it is valid for exactly one base pointer. It is
//     cached on (pointer, shape, box) and rebuilt whenever a tensor moves -- which is what makes
//     the weight descriptors free after the first step, while activations re-encode each call.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <map>
#include <mutex>
#include <stdexcept>

// ---- the launchers, from the two kernel translation units ----------------------------------
int transition_fused_fwd_ctas();
int transition_fused_fwd_rows();
bool transition_fused_fwd_saves_xn();
void transition_fused_fwd_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&,
                                 const CUtensorMap&, const CUtensorMap&, const float*, const float*,
                                 __nv_bfloat16*, __nv_bfloat16*, float*, float*, int, int, float,
                                 cudaStream_t);
int transition_fused_bwd_ctas();
int transition_fused_bwd_ndw();
int transition_fused_bwd_ndx();
int transition_fused_bwd_rows();
void transition_fused_bwd_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&,
                                 const CUtensorMap&, const CUtensorMap&, const CUtensorMap&,
                                 const float*, const float*, const float*, __nv_bfloat16*, float*,
                                 float*, float*, float*, int, int, cudaStream_t);
void transition_fused_reduce_launch(const float*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
                                    const float*, float*, float*, cudaStream_t);

namespace {

using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                 const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                 const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    if (e != cudaSuccess || p == nullptr)
      throw std::runtime_error("cuTensorMapEncodeTiled unavailable: the driver predates TMA");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}

using Key = std::array<uint64_t, 5>;

// 2-D bf16 descriptor, dims innermost-first, 128-B swizzle -- the layout every tile load in
// both kernels assumes.
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t d_inner, uint32_t d_outer,
                            uint32_t box_inner, uint32_t box_outer) {
  static std::map<Key, CUtensorMap> cache;
  static std::mutex lock;
  const Key key{reinterpret_cast<uint64_t>(t.data_ptr()), d_inner, d_outer, box_inner, box_outer};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  CUtensorMap map{};
  const cuuint64_t dims[2] = {d_inner, d_outer};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(d_inner) * 2};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t elem[2] = {1, 1};
  CUresult r = encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides,
                         box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled failed");
  return cache.emplace(key, map).first->second;
}

void expect(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("transition fused sm90a: ") + what);
}

__nv_bfloat16* bf16_ptr(const torch::Tensor& t) {
  return reinterpret_cast<__nv_bfloat16*>(t.data_ptr());
}

}  // namespace

// x [M,128] bf16, gamma/beta [128] fp32, wa/wb [512,128] bf16, ws [128,512] bf16,
// wst = ws^T [512,128] bf16 contiguous. Returns (out, xn, rstd, c1); out = transition(x) + x.
// `save` must match the build: the FWD_SAVE=0 variant is the inference one and writes only
// `out`, so xn/rstd/c1 come back as placeholders.
std::vector<torch::Tensor> transition_fused_fwd(torch::Tensor x, torch::Tensor gamma,
                                                torch::Tensor beta, torch::Tensor wa,
                                                torch::Tensor wb, torch::Tensor wst, double eps,
                                                bool save) {
  const int64_t M = x.size(0), D = x.size(1), H = wa.size(0);
  expect(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "x must be contiguous cuda bf16");
  expect(D == 128 && H == 512, "only d_hidden 128 with n 4 is built");
  expect(M % transition_fused_fwd_rows() == 0, "row count must be a whole tile");
  expect(gamma.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32, "gamma/beta must be fp32");
  expect(wa.is_contiguous() && wb.is_contiguous() && wst.is_contiguous(), "weights must be contiguous");
  expect(wst.size(0) == H && wst.size(1) == D, "wst must be ws transposed, [H, D]");

  // Whether the forward writes what the backward needs is compiled in, not passed: the stores
  // sit inside the LayerNorm epilogue. `save` selects the build, and this catches the two
  // getting out of step -- which would otherwise be a kernel writing through a 1-element
  // placeholder.
  expect(save == transition_fused_fwd_saves_xn(), "save flag does not match this build");

  auto out = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  const int64_t saved = save ? M : 1;
  auto xn = save ? torch::empty_like(x) : torch::empty({1, D}, x.options());
  auto rstd = torch::empty({saved}, f32);
  auto c1 = torch::empty({saved}, f32);

  transition_fused_fwd_launch(
      tile_map(x, D, M, 64, 64), tile_map(wa, D, H, 64, 64), tile_map(wb, D, H, 64, 64),
      tile_map(wst, D, H, 64, 64), tile_map(out, D, M, 64, 64),
      gamma.data_ptr<float>(), beta.data_ptr<float>(), bf16_ptr(xn), bf16_ptr(out),
      rstd.data_ptr<float>(), c1.data_ptr<float>(), static_cast<int>(M),
      static_cast<int>(M / transition_fused_fwd_rows()), static_cast<float>(eps),
      at::cuda::getCurrentCUDAStream());
  return {out, xn, rstd, c1};
}

// dy [M,128] bf16 (gradient of the module output), plus what the forward saved.
// Returns (dx, dgamma, dbeta, dWa, dWb, dWs); dx already carries the residual branch.
std::vector<torch::Tensor> transition_fused_bwd(torch::Tensor dy, torch::Tensor x,
                                                torch::Tensor xn, torch::Tensor rstd,
                                                torch::Tensor c1, torch::Tensor gamma,
                                                torch::Tensor wa, torch::Tensor wb,
                                                torch::Tensor ws) {
  const int64_t M = x.size(0), D = x.size(1), H = wa.size(0);
  expect(dy.is_cuda() && dy.is_contiguous() && dy.scalar_type() == torch::kBFloat16, "dy must be contiguous cuda bf16");
  expect(D == 128 && H == 512, "only d_hidden 128 with n 4 is built");
  expect(M % transition_fused_bwd_rows() == 0, "row count must be a whole tile");
  expect(gamma.scalar_type() == torch::kFloat32, "gamma must be fp32");
  expect(ws.is_contiguous() && ws.size(0) == D && ws.size(1) == H, "ws must be contiguous [D, H]");

  auto f32 = x.options().dtype(torch::kFloat32);
  auto dx = torch::empty_like(x);
  auto dgam = torch::zeros({D}, f32);
  auto dbeta = torch::zeros({D}, f32);
  auto partw = torch::empty({transition_fused_bwd_ndw() * 3 * 64 * D}, f32);
  auto dgbw = torch::empty({transition_fused_bwd_ndx() * 8 * 256}, f32);
  auto dWa = torch::empty_like(wa);
  auto dWb = torch::empty_like(wb);
  auto dWs = torch::empty_like(ws);

  auto stream = at::cuda::getCurrentCUDAStream();
  transition_fused_bwd_launch(
      tile_map(dy, D, M, 64, 64), tile_map(xn, D, M, 64, 64), tile_map(x, D, M, 64, 64),
      tile_map(ws, H, D, 64, 128), tile_map(wa, D, H, 64, 64), tile_map(wb, D, H, 64, 64),
      rstd.data_ptr<float>(), c1.data_ptr<float>(), gamma.data_ptr<float>(), bf16_ptr(dx),
      dgam.data_ptr<float>(), dbeta.data_ptr<float>(), partw.data_ptr<float>(),
      dgbw.data_ptr<float>(), static_cast<int>(M),
      static_cast<int>(M / transition_fused_bwd_rows()), stream);
  transition_fused_reduce_launch(partw.data_ptr<float>(), bf16_ptr(dWa), bf16_ptr(dWb),
                                 bf16_ptr(dWs), dgbw.data_ptr<float>(), dgam.data_ptr<float>(),
                                 dbeta.data_ptr<float>(), stream);
  return {dx, dgam, dbeta, dWa, dWb, dWs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transition_fused_fwd", &transition_fused_fwd,
        "fused sm_90a Transition forward (LN + SwiGLU expand + squeeze + residual)");
  m.def("transition_fused_bwd", &transition_fused_bwd,
        "fused sm_90a Transition backward (one kernel + a partial reduction)");
  m.def("rows_per_tile", &transition_fused_fwd_rows);
  m.def("fwd_ctas", &transition_fused_fwd_ctas);
  m.def("bwd_ctas", &transition_fused_bwd_ctas);
  m.def("saves_xn", &transition_fused_fwd_saves_xn);
}
