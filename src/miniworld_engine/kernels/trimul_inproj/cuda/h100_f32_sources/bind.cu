// Torch bindings of the fp32 / TF32 bidirectional TriMul kernels (tmf32.cuh).  Tensor maps are cached on (pointer, dims, box): 128 opaque
// bytes valid for one base pointer; they travel by value in the kernel parameters, which is what CUDA graphs capture.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <array>
#include <map>
#include <mutex>
#include <stdexcept>

#include "tmf32.cuh"
#ifdef TMF_BWD
#include "tmf32_bwd.cuh"
#endif

using namespace tmf;

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

void expect(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("trimul fp32 h100: ") + what);
}

// 2-D row-major fp32 [rows][cols] tensor, box = [32 columns (128 B)][box_rows], 128 B swizzle (K-major MMA operand slabs).
const CUtensorMap& slab_map(const torch::Tensor& t, int box_rows) {
  using Key = std::array<uint64_t, 5>;
  static std::map<Key, CUtensorMap> cache;
  static std::mutex lock;
  const uint64_t rows = t.size(0), cols = t.size(1);
  const Key key{reinterpret_cast<uint64_t>(t.data_ptr()), rows, cols, (uint64_t)box_rows, (uint64_t)t.stride(0)};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {cols, rows};
  const cuuint64_t strides[1] = {(cuuint64_t)t.stride(0) * 4};
  const cuuint32_t box[2] = {32, (cuuint32_t)box_rows};
  const cuuint32_t elem[2] = {1, 1};
  if (cudaSetDevice(t.get_device()) != cudaSuccess) throw std::runtime_error("cudaSetDevice failed before cuTensorMapEncodeTiled");
  CUresult r = encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, t.data_ptr(), dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled failed");
  return cache.emplace(key, map).first->second;
}

int sm_count() {
  static std::map<int, int> cache;
  static std::mutex lock;
  int dev = 0;
  cudaGetDevice(&dev);
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(dev);
  if (it != cache.end()) return it->second;
  int n = 0;
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  return cache.emplace(dev, n).first->second;
}

template <typename K>
void optin(K kernel, int bytes) {
  cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (e != cudaSuccess) throw std::runtime_error(std::string("smem opt-in: ") + cudaGetErrorString(e));
}

template <typename K, typename... Args>
void launch384(K kernel, int grid, int smem, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(384, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, args...);
  if (e != cudaSuccess) throw std::runtime_error(std::string("GEMM kernel launch: ") + cudaGetErrorString(e));
}

const float* cf(const torch::Tensor& t) { return t.data_ptr<float>(); }
float* mf(torch::Tensor& t) { return t.data_ptr<float>(); }

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
std::vector<torch::Tensor> pack_weights(torch::Tensor wlg, torch::Tensor wl, torch::Tensor wrg, torch::Tensor wr, torch::Tensor wo,
                                        torch::Tensor wg) {
  expect(wlg.sizes() == torch::IntArrayRef({256, 128}) && wo.sizes() == torch::IntArrayRef({128, 256}) &&
             wg.sizes() == torch::IntArrayRef({128, 128}), "weight shapes");
  auto opt = wlg.options();
  auto wpk = torch::empty({1024, 128}, opt), woo = torch::empty({128, 256}, opt), wgg = torch::empty({128, 128}, opt);
  PackArgs a{};
  a.src[0] = cf(wlg); a.src[1] = cf(wl); a.src[2] = cf(wrg); a.src[3] = cf(wr); a.src[4] = cf(wo); a.src[5] = cf(wg);
  a.wpk = mf(wpk); a.wo = mf(woo); a.wg = mf(wgg);
  pack_kernel<<<dim3(32, 6), 256, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  return {wpk, woo, wgg};
}

std::vector<torch::Tensor> ln_in(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save) {
  expect(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.size(1) == 128, "ln_in: x must be contiguous [T, 128]");
  const int64_t T = x.size(0);
  auto xn = torch::empty_like(x);
  auto mean = save ? torch::empty({T}, x.options()) : torch::empty({0}, x.options());
  auto rstd = save ? torch::empty({T}, x.options()) : torch::empty({0}, x.options());
  ln_in_kernel<<<(T + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(cf(x), cf(w), cf(b), mf(xn), save ? mf(mean) : nullptr,
                                                                           save ? mf(rstd) : nullptr, T, (float)eps);
  return {xn, mean, rstd};
}

// planes [512][T]; pre [8][T] x 256?  -> [4][256][T] when save
std::vector<torch::Tensor> front(torch::Tensor xn, torch::Tensor wpk, c10::optional<torch::Tensor> tmask, int64_t L, bool save) {
  expect(xn.is_cuda() && xn.is_contiguous() && xn.size(1) == 128, "front: xn must be contiguous [T, 128]");
  const int64_t T = xn.size(0);
  auto planes = torch::empty({512, T}, xn.options());
  auto pre = save ? torch::empty({4, 256, T}, xn.options()) : torch::empty({0}, xn.options());
  const uint8_t* tm = tmask.has_value() ? reinterpret_cast<const uint8_t*>(tmask->data_ptr()) : nullptr;
  const CUtensorMap mA = slab_map(xn, BM), mB = slab_map(wpk, 128);
  const int64_t tiles = ((T + BM - 1) / BM) * F_NT;
  const int grid = (int)std::min<int64_t>(tiles, sm_count());
  if (save) {
    static const bool ready = (optin(front_kernel<true>, F_SMEM), true); (void)ready;
    launch384(front_kernel<true>, grid, F_SMEM, mA, mB, T, (int)L, tm, mf(planes), mf(pre));
  } else {
    static const bool ready = (optin(front_kernel<false>, F_SMEM), true); (void)ready;
    launch384(front_kernel<false>, grid, F_SMEM, mA, mB, T, (int)L, tm, mf(planes), (float*)nullptr);
  }
  return {planes, pre};
}

std::vector<torch::Tensor> ln_out_t(torch::Tensor tri, torch::Tensor w, torch::Tensor b, double eps, bool save) {
  expect(tri.is_cuda() && tri.is_contiguous() && tri.dim() == 2 && tri.size(0) == 256, "ln_out_t: tri must be contiguous [256, T]");
  const int64_t T = tri.size(1);
  auto on = torch::empty({T, 256}, tri.options());
  auto mean = save ? torch::empty({T}, tri.options()) : torch::empty({0}, tri.options());
  auto rstd = save ? torch::empty({T}, tri.options()) : torch::empty({0}, tri.options());
  const int smem = 256 * 65 * 4;
  static const bool ready = (optin(ln_out_t_kernel, smem), true); (void)ready;
  ln_out_t_kernel<<<(T + 63) / 64, 256, smem, at::cuda::getCurrentCUDAStream()>>>(cf(tri), cf(w), cf(b), mf(on), save ? mf(mean) : nullptr,
                                                                                     save ? mf(rstd) : nullptr, T, (float)eps);
  return {on, mean, rstd};
}

std::vector<torch::Tensor> back(torch::Tensor on, torch::Tensor xn, torch::Tensor wo, torch::Tensor wg, torch::Tensor x,
                                c10::optional<torch::Tensor> ds, int64_t L, bool save) {
  expect(on.is_cuda() && on.size(1) == 256 && xn.size(1) == 128 && x.is_contiguous(), "back: shapes");
  const int64_t T = x.size(0);
  auto y = torch::empty_like(x);
  auto gs = save ? torch::empty_like(x) : torch::empty({0}, x.options());
  auto ps = save ? torch::empty_like(x) : torch::empty({0}, x.options());
  const CUtensorMap mA1 = slab_map(on, BM), mB1 = slab_map(wo, 128), mA2 = slab_map(xn, BM), mB2 = slab_map(wg, 128);
  const int64_t tiles = (T + BM - 1) / BM;
  const int grid = (int)std::min<int64_t>(tiles, sm_count());
  const float* dsp = ds.has_value() ? ds->data_ptr<float>() : nullptr;
  if (save) {
    static const bool ready = (optin(back_kernel<true>, B_SMEM), true); (void)ready;
    launch384(back_kernel<true>, grid, B_SMEM, mA1, mB1, mA2, mB2, T, (int)L, cf(x), dsp, mf(y), mf(gs), mf(ps));
  } else {
    static const bool ready = (optin(back_kernel<false>, B_SMEM), true); (void)ready;
    launch384(back_kernel<false>, grid, B_SMEM, mA1, mB1, mA2, mB2, T, (int)L, cf(x), dsp, mf(y), (float*)nullptr, (float*)nullptr);
  }
  return {y, gs, ps};
}

#ifdef TMF_BWD
// ---------------------------------------------------------------------------------------------------------------------------------- backward
// [woT [256][128] (TF32-rounded Wo^T), uv [256] = (Wo w_out, Wo b_out)]
std::vector<torch::Tensor> wo_prep(torch::Tensor wo, torch::Tensor w, torch::Tensor b) {
  expect(wo.sizes() == torch::IntArrayRef({128, 256}), "wo_prep: Wo [128, 256]");
  auto woT = torch::empty({256, 128}, wo.options()), uv = torch::empty({256}, wo.options());
  wo_prep_kernel<<<128, 256, 0, at::cuda::getCurrentCUDAStream()>>>(cf(wo), cf(w), cf(b), mf(woT), mf(uv));
  return {woT, uv};
}

// [d_proj, d_glogit, s12 [T][2]]
std::vector<torch::Tensor> gate_bwd(torch::Tensor dy, torch::Tensor gs, torch::Tensor ps, c10::optional<torch::Tensor> ds, torch::Tensor uv,
                                    int64_t L) {
  const int64_t T = dy.size(0);
  auto dp = torch::empty_like(dy), dg = torch::empty_like(dy), s12 = torch::empty({T, 2}, dy.options());
  gate_bwd_kernel<<<(T + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      cf(dy), cf(gs), cf(ps), ds.has_value() ? ds->data_ptr<float>() : nullptr, cf(uv), mf(dp), mf(dg),
      reinterpret_cast<float2*>(s12.data_ptr<float>()), T, (int)L);
  return {dp, dg, s12};
}

// [d_tri [256][T], pdg [grid][256], pdb [grid][256]]
std::vector<torch::Tensor> lnout_bwd(torch::Tensor dproj, torch::Tensor woT, torch::Tensor tri, torch::Tensor mean, torch::Tensor rstd,
                                     torch::Tensor s12, torch::Tensor gamma) {
  const int64_t T = dproj.size(0);
  auto dtri = torch::empty_like(tri);
  const int64_t tiles = (T + BM - 1) / BM;
  const int grid = (int)std::min<int64_t>(tiles, sm_count());
  auto pdg = torch::empty({grid, 256}, tri.options()), pdb = torch::empty({grid, 256}, tri.options());
  const CUtensorMap mAw = slab_map(woT, 256), mBd = slab_map(dproj, 128);
  static const bool ready = (optin(lnout_bwd_kernel, LB_SMEM), true); (void)ready;
  launch384(lnout_bwd_kernel, grid, LB_SMEM, mAw, mBd, T, cf(tri), cf(mean), cf(rstd), reinterpret_cast<const float2*>(s12.data_ptr<float>()),
            cf(gamma), mf(dtri), mf(pdg), mf(pdb));
  return {dtri, pdg, pdb};
}

torch::Tensor front_bwd(torch::Tensor xn, torch::Tensor wpk, c10::optional<torch::Tensor> tmask, torch::Tensor dpl, int64_t L) {
  const int64_t T = xn.size(0);
  auto out = torch::empty({4, 256, T}, xn.options());
  const uint8_t* tm = tmask.has_value() ? reinterpret_cast<const uint8_t*>(tmask->data_ptr()) : nullptr;
  const CUtensorMap mA = slab_map(xn, BM), mB = slab_map(wpk, 128);
  const int64_t tiles = ((T + BM - 1) / BM) * F_NT;
  const int grid = (int)std::min<int64_t>(tiles, sm_count());
  static const bool ready = (optin(front_bwd_kernel, F_SMEM), true); (void)ready;
  launch384(front_bwd_kernel, grid, F_SMEM, mA, mB, T, (int)L, tm, cf(dpl), mf(out));
  return out;
}

// [dx, pdg [grid][128], pdb [grid][128]]
std::vector<torch::Tensor> ln_in_bwd(torch::Tensor dxn, torch::Tensor x, torch::Tensor mean, torch::Tensor rstd, torch::Tensor dy,
                                     torch::Tensor w) {
  const int64_t T = x.size(0);
  auto dx = torch::empty_like(x);
  const int grid = sm_count() * 4;
  auto pdg = torch::empty({grid, 128}, x.options()), pdb = torch::empty({grid, 128}, x.options());
  ln_in_bwd_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(cf(dxn), cf(x), cf(mean), cf(rstd), cf(dy), cf(w), mf(dx), mf(pdg),
                                                                           mf(pdb), T);
  return {dx, pdg, pdb};
}

#endif  // TMF_BWD

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef TMF_BWD
  m.def("wo_prep", &wo_prep);
  m.def("gate_bwd", &gate_bwd);
  m.def("lnout_bwd", &lnout_bwd);
  m.def("front_bwd", &front_bwd);
  m.def("ln_in_bwd", &ln_in_bwd);
#endif
  m.def("pack_weights", &pack_weights);
  m.def("ln_in", &ln_in);
  m.def("front", &front);
  m.def("ln_out_t", &ln_out_t);
  m.def("back", &back);
}
