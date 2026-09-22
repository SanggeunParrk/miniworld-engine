// Torch bindings for the wide-width (D != 128) Transition kernels on sm_90a. One extension per channel width: WIDE_D picks
// which launchers exist, WIDE_SMS is the device's multiprocessor count (the persistent grids are sized from it at build time).
//
//   D = 64          fwd: one fused kernel              bwd: one fused two-role kernel + reduction
//   D = 256         fwd: one fused kernel              bwd: gate kernel -> (torch) fp32 dW GEMMs -> d_xn + LN-bwd kernel
//   D = 384, 512    fwd: ln_swiglu_gemm + squeeze_gemm bwd: gate kernel -> (torch) fp32 dW GEMMs, d_xn GEMM, LN bwd
//
// Descriptors are cached on (pointer, dims, stride, box, swizzle) exactly like the D = 128 extension: 128 opaque bytes that
// are valid for one base pointer, so weights encode once and activations re-encode per call.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <array>
#include <map>
#include <mutex>
#include <stdexcept>

#if WIDE_D == 64
int wide_d64_fwd_ctas();
void wide_d64_fwd_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&,
                         const float*, const float*, __nv_bfloat16*, __nv_bfloat16*, float*, float*, int, int, float, int, cudaStream_t);
int wide_d64_bwd_ndw();
int wide_d64_bwd_ndx();
void wide_d64_bwd_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&,
                         const CUtensorMap&, const float*, const float*, const float*, __nv_bfloat16*, float*, float*, float*, float*,
                         __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, cudaStream_t);
#endif
#if WIDE_D == 256
void wide_d256_fwd_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&,
                          const float*, const float*, __nv_bfloat16*, __nv_bfloat16*, float*, float*, int, int, float, int, cudaStream_t);
void wide_dxln_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
                      const float*, const float*, const float*, float*, float*, int, int, cudaStream_t);
#endif
#if WIDE_D == 384 || WIDE_D == 512
int wide_squeeze_bn();
void wide_lnsg_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const float*, const float*, __nv_bfloat16*,
                      float*, float*, int, int, float, int, int, cudaStream_t);
void wide_squeeze_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, int, int, int, int,
                         cudaStream_t);
void wide_gate_noh_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&,
                          const CUtensorMap&, int, int, int, int, cudaStream_t);
#endif
#if WIDE_D >= 256
void wide_gate_launch(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, const CUtensorMap&,
                      const CUtensorMap&, int, int, int, int, cudaStream_t);
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

// 2-D bf16 row-major [outer][inner] descriptor over a (possibly row-strided) tensor; box innermost-first.
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t inner, uint32_t outer, uint64_t row_stride_elems,
                            uint32_t box_inner, uint32_t box_outer, int swizzle = 128) {
  using Key = std::array<uint64_t, 7>;
  static std::map<Key, CUtensorMap> cache;
  static std::mutex lock;
  const Key key{reinterpret_cast<uint64_t>(t.data_ptr()), inner, outer, row_stride_elems, box_inner, box_outer, (uint64_t)swizzle};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {row_stride_elems * 2};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t elem[2] = {1, 1};
  const CUtensorMapSwizzle sw = swizzle == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
  CUresult r = encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, elem,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled failed");
  return cache.emplace(key, map).first->second;
}
// the common case: a contiguous [rows][cols] tensor
const CUtensorMap& rm(const torch::Tensor& t, uint32_t box_inner, uint32_t box_outer, int swizzle = 128) {
  return tile_map(t, t.size(1), t.size(0), t.size(1), box_inner, box_outer, swizzle);
}

void expect(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("transition wide sm90a: ") + what);
}
void bf16_2d(const torch::Tensor& t, int64_t rows, int64_t cols, const char* what) {
  expect(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kBFloat16 && t.dim() == 2 && t.size(0) == rows && t.size(1) == cols, what);
}
void f32_1d(const torch::Tensor& t, int64_t n, const char* what) {
  expect(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat32 && t.numel() == n, what);
}
__nv_bfloat16* bp(const torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); }
float* fp(const torch::Tensor& t) { return t.data_ptr<float>(); }
constexpr int64_t D = WIDE_D, H = 4 * WIDE_D, ROWS = 128;

}  // namespace

#if WIDE_D == 64 || WIDE_D == 256
// x [M,D]; gamma/beta fp32 [D]; wa, wb [H,D]; wst = ws^T [H,D]. Returns (out, xn, rstd, c1); out = transition(x) + x.
// save = false: inference -- xn / rstd / c1 come back as 1-element placeholders and are never written.
std::vector<torch::Tensor> fwd(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa, torch::Tensor wb,
                               torch::Tensor wst, double eps, bool save) {
  const int64_t M = x.size(0);
  bf16_2d(x, M, D, "x must be contiguous bf16 [M, D]");
  expect(M % ROWS == 0, "row count must be a whole number of 128-row tiles");
  f32_1d(gamma, D, "gamma must be fp32 [D]"); f32_1d(beta, D, "beta must be fp32 [D]");
  bf16_2d(wa, H, D, "wa must be [H, D]"); bf16_2d(wb, H, D, "wb must be [H, D]"); bf16_2d(wst, H, D, "wst must be ws^T [H, D]");
  auto out = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto xn = save ? torch::empty_like(x) : torch::empty({1, D}, x.options());
  auto rstd = torch::empty({save ? M : 1}, f32), c1 = torch::empty({save ? M : 1}, f32);
  auto stream = at::cuda::getCurrentCUDAStream();
#if WIDE_D == 64
  wide_d64_fwd_launch(rm(x, 64, 64), rm(wa, 64, 64), rm(wb, 64, 64), rm(wst, 64, 64), rm(out, 64, 64), fp(gamma), fp(beta),
                      bp(xn), bp(out), fp(rstd), fp(c1), (int)M, (int)(M / ROWS), (float)eps, save ? 1 : 0, stream);
#else
  wide_d256_fwd_launch(rm(x, 64, 64), rm(wa, 64, 32), rm(wb, 64, 32), rm(wst, 64, 32), rm(out, 64, 64), fp(gamma), fp(beta),
                       bp(xn), bp(out), fp(rstd), fp(c1), (int)M, (int)(M / ROWS), (float)eps, save ? 1 : 0, stream);
#endif
  return {out, xn, rstd, c1};
}
#endif

#if WIDE_D == 384 || WIDE_D == 512
// x [M,D]; gamma/beta fp32; w1p = [Wa;Wb] packed in 64-row blocks [Wa 64 | Wb 64] [2H,D]; ws [D,H]. Returns (out, xn, rstd, c1, h):
// h [M,H] is the SwiGLU intermediate the two kernels pass through HBM anyway -- the backward may keep it (save_h).
std::vector<torch::Tensor> fwd(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor w1p, torch::Tensor ws,
                               double eps, bool save) {
  const int64_t M = x.size(0);
  bf16_2d(x, M, D, "x must be contiguous bf16 [M, D]");
  expect(M % ROWS == 0, "row count must be a whole number of 128-row tiles");
  f32_1d(gamma, D, "gamma must be fp32 [D]"); f32_1d(beta, D, "beta must be fp32 [D]");
  bf16_2d(w1p, 2 * H, D, "w1p must be [2H, D]"); bf16_2d(ws, D, H, "ws must be [D, H]");
  auto out = torch::empty_like(x);
  auto h = torch::empty({M, H}, x.options());
  auto f32 = x.options().dtype(torch::kFloat32);
  auto xn = save ? torch::empty_like(x) : torch::empty({1, D}, x.options());
  auto rstd = torch::empty({save ? M : 1}, f32), c1 = torch::empty({save ? M : 1}, f32);
  auto stream = at::cuda::getCurrentCUDAStream();
  wide_lnsg_launch(rm(x, 64, 64), rm(w1p, 64, 128), rm(h, 64, 64), fp(gamma), fp(beta), bp(xn), fp(rstd), fp(c1), (int)M, (int)H,
                   (float)eps, save ? 1 : 0, WIDE_SMS, stream);
  const int bn = wide_squeeze_bn();
  wide_squeeze_launch(rm(h, 64, 64), rm(ws, 64, bn), rm(x, 64, 64), rm(out, 64, 64), (int)M, (int)H, (int)D, WIDE_SMS, stream);
  return {out, xn, rstd, c1, h};
}
#endif

#if WIDE_D == 64
// Returns (dx, dgamma, dbeta, dWa, dWb, dWs); dx carries the residual branch.
std::vector<torch::Tensor> bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor xn, torch::Tensor rstd, torch::Tensor c1,
                               torch::Tensor gamma, torch::Tensor wa, torch::Tensor wb, torch::Tensor ws) {
  const int64_t M = x.size(0);
  bf16_2d(dy, M, D, "dy must be contiguous bf16 [M, D]"); bf16_2d(x, M, D, "x"); bf16_2d(xn, M, D, "xn");
  f32_1d(rstd, M, "rstd"); f32_1d(c1, M, "c1"); f32_1d(gamma, D, "gamma must be fp32 [D]");
  bf16_2d(wa, H, D, "wa"); bf16_2d(wb, H, D, "wb"); bf16_2d(ws, D, H, "ws must be [D, H]");
  expect(M % ROWS == 0, "row count must be a whole number of 128-row tiles");
  auto f32 = x.options().dtype(torch::kFloat32);
  auto dx = torch::empty_like(x);
  auto dgam = torch::empty({D}, f32), dbeta = torch::empty({D}, f32);
  auto partw = torch::empty({wide_d64_bwd_ndw() * 4 * 64 * D}, f32);
  auto dgbw = torch::empty({wide_d64_bwd_ndx() * 8 * 2 * D}, f32);
  auto dWa = torch::empty_like(wa), dWb = torch::empty_like(wb), dWs = torch::empty_like(ws);
  wide_d64_bwd_launch(rm(dy, 64, 64), rm(xn, 64, 64), rm(x, 64, 64), rm(ws, 64, D), rm(wa, 64, 64), rm(wb, 64, 64), fp(rstd), fp(c1),
                      fp(gamma), bp(dx), fp(dgam), fp(dbeta), fp(partw), fp(dgbw), bp(dWa), bp(dWb), bp(dWs), (int)M,
                      (int)(M / ROWS), at::cuda::getCurrentCUDAStream());
  return {dx, dgam, dbeta, dWa, dWb, dWs};
}
#endif

#if WIDE_D >= 256
// Gate stage: xn, dy [M,D]; w1p = [Wa;Wb] packed in 128-row blocks [Wa 128 | Wb 128] [2H,D]; wst = ws^T [H,D].
// Returns (h [M,H], dab [M,2H]) -- dA in columns 0..H-1, dB in H..2H-1. write_h = false (D >= 384): h is a placeholder.
std::vector<torch::Tensor> gate(torch::Tensor xn, torch::Tensor dy, torch::Tensor w1p, torch::Tensor wst, bool write_h) {
  const int64_t M = xn.size(0);
  bf16_2d(xn, M, D, "xn"); bf16_2d(dy, M, D, "dy must be contiguous bf16 [M, D]");
  bf16_2d(w1p, 2 * H, D, "w1p must be [2H, D]"); bf16_2d(wst, H, D, "wst must be ws^T [H, D]");
  expect(M % ROWS == 0, "row count must be a whole number of 128-row tiles");
#if WIDE_D == 256
  expect(write_h, "D = 256 has no forward h to reuse");
#endif
  auto h = write_h ? torch::empty({M, H}, xn.options()) : torch::empty({ROWS, 64}, xn.options());   // placeholder: never written
  auto dab = torch::empty({M, 2 * H}, xn.options());
#if WIDE_D != 256
  if (!write_h) {
    wide_gate_noh_launch(rm(xn, 32, 64, 64), rm(dy, 32, 64, 64), rm(w1p, 32, 128, 64), rm(wst, 32, 128, 64), rm(h, 64, 64), rm(dab, 64, 64),
                         (int)M, (int)D, (int)H, WIDE_SMS, at::cuda::getCurrentCUDAStream());
    return {h, dab};
  }
#endif
  wide_gate_launch(rm(xn, 32, 64, 64), rm(dy, 32, 64, 64), rm(w1p, 32, 128, 64), rm(wst, 32, 128, 64), rm(h, 64, 64), rm(dab, 64, 64),
                   (int)M, (int)D, (int)H, WIDE_SMS, at::cuda::getCurrentCUDAStream());
  return {h, dab};
}
#endif

#if WIDE_D == 256
// d_xn + LayerNorm backward + residual: dab [M,2H]; wabt = [Wa;Wb]^T [D,2H]. Returns (dx, dgamma, dbeta).
std::vector<torch::Tensor> dxln(torch::Tensor dab, torch::Tensor wabt, torch::Tensor x, torch::Tensor dy, torch::Tensor gamma,
                                torch::Tensor rstd, torch::Tensor c1) {
  const int64_t M = x.size(0);
  bf16_2d(dab, M, 2 * H, "dab"); bf16_2d(wabt, D, 2 * H, "wabt must be [D, 2H]"); bf16_2d(x, M, D, "x"); bf16_2d(dy, M, D, "dy");
  f32_1d(gamma, D, "gamma must be fp32 [D]"); f32_1d(rstd, M, "rstd"); f32_1d(c1, M, "c1");
  auto dx = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto pdg = torch::empty({WIDE_SMS, D}, f32), pdb = torch::empty({WIDE_SMS, D}, f32);
  wide_dxln_launch(rm(dab, 64, 64), rm(wabt, 64, D / 2), rm(x, 64, 64), rm(dx, 64, 64), bp(x), bp(dy), bp(dx), fp(gamma), fp(rstd), fp(c1), fp(pdg), fp(pdb), (int)M,
                   WIDE_SMS, at::cuda::getCurrentCUDAStream());
  return {dx, pdg.sum(0), pdb.sum(0)};
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fwd, "wide-width fused Transition forward (+ residual)");
#if WIDE_D == 64
  m.def("bwd", &bwd, "D = 64 fused Transition backward");
#endif
#if WIDE_D >= 256
  m.def("gate", &gate, "backward gate stage: h, [dA|dB]");
#endif
#if WIDE_D == 256
  m.def("dxln", &dxln, "d_xn GEMM + LayerNorm backward + residual");
#endif
  m.attr("width") = WIDE_D;
}
