// Torch bindings of the generic sm_90a Transition kernels (tr90.cuh). One extension for every (D, hidden, dtype): the kernels are
// instantiated per tile width / dtype and picked at run time.
//
// Tensor maps are cached on (pointer, dims, box, dtype): 128 opaque bytes valid for one base pointer, so weights encode once and
// activations re-encode per call (host work only; the maps travel by value in the kernel parameters, which is what CUDA graphs capture).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <array>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>

#include "tr90.cuh"

using namespace tr90;

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
  if (!ok) throw std::invalid_argument(std::string("transition generic sm90a: ") + what);
}

bool is_bf16(const torch::Tensor& t) { return t.scalar_type() == torch::kBFloat16; }
bool is_f16(const torch::Tensor& t) { return t.scalar_type() == torch::kHalf; }

// 2-D row-major [rows][cols] tensor, box = [128 B of columns][box_rows], 128 B swizzle (K-major MMA operand slabs).
const CUtensorMap& slab_map(const torch::Tensor& t, int box_rows) {
  using Key = std::array<uint64_t, 6>;
  static std::map<Key, CUtensorMap> cache;
  static std::mutex lock;
  const uint64_t rows = t.size(0), cols = t.size(1);
  const int es = (is_bf16(t) || is_f16(t)) ? 2 : 4;
  const Key key{reinterpret_cast<uint64_t>(t.data_ptr()), rows, cols, (uint64_t)box_rows, (uint64_t)es, (uint64_t)t.stride(0)};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {cols, rows};
  const cuuint64_t strides[1] = {(cuuint64_t)t.stride(0) * es};
  const cuuint32_t box[2] = {(cuuint32_t)(128 / es), (cuuint32_t)box_rows};
  const cuuint32_t elem[2] = {1, 1};
  // cuTensorMapEncodeTiled is a driver call and needs a current context. The autograd engine runs backward on its own thread, and when the
  // caching allocator serves every tensor from its pool no runtime call has made the primary context current there: CUDA_ERROR_INVALID_CONTEXT.
  // (cudaSetDevice binds it; cudaFree(0) would do too but is an unsafe call inside a CUDA-graph capture.)
  if (cudaSetDevice(t.get_device()) != cudaSuccess) throw std::runtime_error("cudaSetDevice failed before cuTensorMapEncodeTiled");
  CUresult r = encoder()(&map, is_bf16(t) ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : is_f16(t) ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, t.data_ptr(), dims,
                         strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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

// grid / smem launch of a 384-thread GEMM kernel.
template <typename K, typename... Args>
void launch_gemm_kernel(K kernel, int grid, int smem, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(384, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;       // the kernel waits (griddepcontrol.wait) before reading
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, args...);
  if (e != cudaSuccess) throw std::runtime_error(std::string("GEMM kernel launch: ") + cudaGetErrorString(e));
}

// persistent grid for `mtiles` x `nt` work units
int gemm_grid(int64_t mtiles, int64_t nt) { return (int)std::min<int64_t>(mtiles * nt, (int64_t)sm_count()); }

template <typename K>
void optin(K kernel, int bytes) {
  cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (e != cudaSuccess) throw std::runtime_error(std::string("smem opt-in: ") + cudaGetErrorString(e));
}

template <typename T> const T* cp(const torch::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <typename T> T* mp(torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

void check_act(const torch::Tensor& t, int64_t cols, const char* what) {
  expect(t.is_cuda() && t.dim() == 2 && t.size(1) == cols && t.stride(1) == 1 && t.stride(0) == cols &&
             (is_bf16(t) || is_f16(t) || t.scalar_type() == torch::kFloat32), what);
}

template <typename T, int BN, int MODE, typename TO = T>
void launch_gemm(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& B2, const torch::Tensor* Xt, torch::Tensor* Ot,
                 int64_t M, int64_t N, int64_t K, const void* X, void* O, const float* SC) {
  using C = GemmCfg<BN, Tt<T>::ES, MODE, Tt<TO>::ES>;
  static const bool ready = (optin(gemm_kernel<T, BN, MODE, false, TO>, C::SMEM), true);
  (void)ready;
  const int NW = C::NW;
  const int64_t mtiles = (M + BM - 1) / BM, nt = N / NW;
  const int grid = gemm_grid(mtiles, nt);
  const torch::Tensor& bt2 = MODE == MODE_SWIGLU ? B2 : B;
  const CUtensorMap& mx = (C::STAGED && Xt) ? slab_map(*Xt, 64) : slab_map(A, BM);
  const CUtensorMap& mo = (C::STAGED && Ot) ? slab_map(*Ot, 64) : slab_map(A, BM);
  const int brows = MODE == MODE_SWIGLU ? BN / 2 : BN;
  const CUtensorMap mb = slab_map(B, brows), mb2 = slab_map(bt2, brows), mA = slab_map(A, BM);
  launch_gemm_kernel(gemm_kernel<T, BN, MODE, false, TO>, grid, C::SMEM, mA, mb, mb2, mx, mo, mo, (int)M, (int)N, (int)K, static_cast<const TO*>(X), O, SC);
}

template <typename T, int MODE, typename TO = T>
void gemm_bn(int bn, const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& B2, const torch::Tensor* Xt,
             torch::Tensor* Ot, int64_t M, int64_t N, int64_t K, const void* X, void* O, const float* SC = nullptr) {
  switch (bn) {
    case 64: launch_gemm<T, 64, MODE, TO>(A, B, B2, Xt, Ot, M, N, K, X, O, SC); break;
    case 128: launch_gemm<T, 128, MODE, TO>(A, B, B2, Xt, Ot, M, N, K, X, O, SC); break;
    case 192: launch_gemm<T, 192, MODE, TO>(A, B, B2, Xt, Ot, M, N, K, X, O, SC); break;
    case 256: launch_gemm<T, 256, MODE, TO>(A, B, B2, Xt, Ot, M, N, K, X, O, SC); break;
    default: throw std::invalid_argument("transition generic sm90a: unsupported tile width");
  }
}

// Persistent grid: a tile width w costs ceil(tiles / SMs) waves of (w columns of MMA work + a fixed per-tile overhead of 32 columns);
// pick the width with the least total.  (The widest tile that fills the SMs left 144 tiles on 132 SMs = two waves at M = 3072, D = 768.)
// `cols` = MMA columns per output column of the tile (2 for SwiGLU: a and b).  Ties go to the wider tile.
int pick_width(int64_t M, int64_t N, std::initializer_list<int> widths, int cols = 1) {
  const int64_t mtiles = (M + BM - 1) / BM;
  int best = 0;
  int64_t best_cost = 0;
  for (int w : widths) {
    if (N % w) continue;
    const int64_t tiles = mtiles * (N / w), waves = (tiles + sm_count() - 1) / sm_count();
    const int64_t cost = waves * (w * cols + 32);
    if (!best || cost < best_cost) { best = w; best_cost = cost; }
  }
  if (!best) throw std::invalid_argument("transition generic sm90a: width must be a multiple of 64");
  return best;
}
int pick_bn(int64_t M, int64_t N) { return pick_width(M, N, {256, 192, 128, 64}); }

template <typename T, int PER>
void run_ln_fwd(const torch::Tensor& x, const torch::Tensor& g, const torch::Tensor& b, torch::Tensor& xn, torch::Tensor& rstd,
                torch::Tensor& c1, double eps, bool save) {
  const int64_t M = x.size(0), D = x.size(1);
  ln_fwd_kernel<T, PER><<<(M + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      cp<T>(x), g.data_ptr<float>(), b.data_ptr<float>(), mp<T>(xn), rstd.data_ptr<float>(), c1.data_ptr<float>(), (int)M, (int)D,
      (float)eps, save ? 1 : 0);
}

template <typename T, int PER>
void run_ln_bwd(const torch::Tensor& dxn, const torch::Tensor& x, const torch::Tensor& dy, const torch::Tensor& rstd,
                const torch::Tensor& c1, const torch::Tensor& g, torch::Tensor& dx, torch::Tensor& pdg, torch::Tensor& pdb, int grid) {
  ln_bwd_kernel<T, PER><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      cp<T>(dxn), cp<T>(x), cp<T>(dy), rstd.data_ptr<float>(), c1.data_ptr<float>(), g.data_ptr<float>(), mp<T>(dx),
      pdg.data_ptr<float>(), pdb.data_ptr<float>(), (int)x.size(0), (int)x.size(1));
}

#define PER_SWITCH(PER_EXPR, FN, ...)                                    \
  switch (PER_EXPR) {                                                    \
    case 2: FN<T, 2>(__VA_ARGS__); break;                              \
    case 4: FN<T, 4>(__VA_ARGS__); break;                              \
    case 6: FN<T, 6>(__VA_ARGS__); break;                              \
    case 8: FN<T, 8>(__VA_ARGS__); break;                              \
    case 10: FN<T, 10>(__VA_ARGS__); break;                              \
    case 12: FN<T, 12>(__VA_ARGS__); break;                              \
    case 14: FN<T, 14>(__VA_ARGS__); break;                              \
    case 16: FN<T, 16>(__VA_ARGS__); break;                              \
    case 18: FN<T, 18>(__VA_ARGS__); break;                              \
    case 20: FN<T, 20>(__VA_ARGS__); break;                              \
    case 22: FN<T, 22>(__VA_ARGS__); break;                              \
    case 24: FN<T, 24>(__VA_ARGS__); break;                              \
    case 26: FN<T, 26>(__VA_ARGS__); break;                              \
    case 28: FN<T, 28>(__VA_ARGS__); break;                              \
    case 30: FN<T, 30>(__VA_ARGS__); break;                              \
    case 32: FN<T, 32>(__VA_ARGS__); break;                              \
    default: throw std::invalid_argument("transition generic sm90a: unsupported width for the row kernels"); \
  }

template <typename T>
void ln_fwd_t(const torch::Tensor& x, const torch::Tensor& g, const torch::Tensor& b, torch::Tensor& xn, torch::Tensor& rstd,
              torch::Tensor& c1, double eps, bool save) {
  PER_SWITCH((int)(x.size(1) / 32), run_ln_fwd, x, g, b, xn, rstd, c1, eps, save);
}
template <typename T>
void ln_bwd_t(const torch::Tensor& dxn, const torch::Tensor& x, const torch::Tensor& dy, const torch::Tensor& rstd,
              const torch::Tensor& c1, const torch::Tensor& g, torch::Tensor& dx, torch::Tensor& pdg, torch::Tensor& pdb, int grid) {
  PER_SWITCH((int)(x.size(1) / 32), run_ln_bwd, dxn, x, dy, rstd, c1, g, dx, pdg, pdb, grid);
}

template <typename T>
void gate_t(const torch::Tensor& xn, const torch::Tensor& dy, const torch::Tensor& wa, const torch::Tensor& wb, const torch::Tensor& wst,
            torch::Tensor& h, torch::Tensor& dab) {
  using C = GateCfg<T>;
  static const bool ready = (optin(gate_kernel<T>, C::SMEM), true);
  (void)ready;
  const int64_t M = xn.size(0), D = xn.size(1), HID = wa.size(0);
  const int64_t tiles = ((M + BM - 1) / BM) * (HID / 64);
  const int grid = (int)std::min<int64_t>(tiles, sm_count());
  gate_kernel<T><<<grid, 384, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      slab_map(xn, BM), slab_map(dy, BM), slab_map(wa, 64), slab_map(wb, 64), slab_map(wst, 64), slab_map(h, 64), slab_map(dab, 64), (int)M, (int)D, (int)HID, mp<T>(h), mp<T>(dab));
}

// bf16 saved-a/b path.  swiglu_save: h [M, HID] and ab [M, 2 HID] (a | b); gate_ab: dh = dy Ws with the SwiGLU backward in the
// epilogue from the saved ab -> h, dab.
template <int BN>
void swiglu_save_bn(const torch::Tensor& xn, const torch::Tensor& wa, const torch::Tensor& wb, torch::Tensor& h, torch::Tensor& ab) {
  using C = GemmCfg<BN, 2, MODE_SWIGLU_SAVE>;
  static const bool ready = (optin(gemm_kernel<__nv_bfloat16, BN, MODE_SWIGLU_SAVE>, C::SMEM), true);
  (void)ready;
  const int64_t M = xn.size(0), D = xn.size(1), HID = wa.size(0);
  const int grid = gemm_grid((M + BM - 1) / BM, HID / C::NW);
  launch_gemm_kernel(gemm_kernel<__nv_bfloat16, BN, MODE_SWIGLU_SAVE>, grid, C::SMEM,
      slab_map(xn, BM), slab_map(wa, BN / 2), slab_map(wb, BN / 2), slab_map(h, 64), slab_map(h, 64), slab_map(ab, 64), (int)M, (int)HID,
      (int)D, (const __nv_bfloat16*)nullptr, (void*)nullptr, (const float*)nullptr);
}

template <int BN>
void gate_ab_bn(const torch::Tensor& dy, const torch::Tensor& ws, const torch::Tensor& ab, torch::Tensor& dab) {
  using C = GemmCfg<BN, 2, MODE_GATE>;
  static const bool ready = (optin(gemm_kernel<__nv_bfloat16, BN, MODE_GATE, true>, C::SMEM), true);
  (void)ready;
  const int64_t M = dy.size(0), D = dy.size(1), HID = ws.size(1);
  const int grid = gemm_grid((M + BM - 1) / BM, HID / C::NW);
  launch_gemm_kernel(gemm_kernel<__nv_bfloat16, BN, MODE_GATE, true>, grid, C::SMEM,
      slab_map(dy, BM), slab_map(ws, 64), slab_map(ws, 64), slab_map(ab, 64), slab_map(dab, 64), slab_map(dab, 64), (int)M, (int)HID,
      (int)D, (const __nv_bfloat16*)nullptr, (void*)nullptr, (const float*)nullptr);
}

template <int BN>
void dxn_bt_bn(const torch::Tensor& dab, const torch::Tensor& wa, const torch::Tensor& wb, torch::Tensor& out) {
  using C = GemmCfg<BN, 2, MODE_OUT>;
  static const bool ready = (optin(gemm_kernel<__nv_bfloat16, BN, MODE_OUT, true>, C::SMEM), true);
  (void)ready;
  const int64_t M = dab.size(0), K = dab.size(1), D = wa.size(1);
  const int grid = gemm_grid((M + BM - 1) / BM, D / C::NW);
  launch_gemm_kernel(gemm_kernel<__nv_bfloat16, BN, MODE_OUT, true>, grid, C::SMEM,
      slab_map(dab, BM), slab_map(wa, 64), slab_map(wb, 64), slab_map(out, 64), slab_map(out, 64), slab_map(out, 64), (int)M, (int)D,
      (int)K, (const __nv_bfloat16*)nullptr, (void*)nullptr, (const float*)nullptr);
}

}  // namespace

// x [M, D]; gamma / beta fp32 [D]. Returns (xn, rstd, c1); rstd / c1 are 1-element placeholders unless save.
std::vector<torch::Tensor> ln_fwd(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps, bool save) {
  const int64_t M = x.size(0), D = x.size(1);
  check_act(x, D, "x must be contiguous bf16 / fp32 [M, D]");
  expect(D % 64 == 0 && D <= 1024, "D must be a multiple of 64, at most 1024");
  expect(gamma.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32 && gamma.numel() == D && beta.numel() == D,
         "gamma / beta must be fp32 [D]");
  auto xn = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto rstd = torch::empty({save ? M : 1}, f32), c1 = torch::empty({save ? M : 1}, f32);
  if (M > 0) {
    if (is_bf16(x)) ln_fwd_t<__nv_bfloat16>(x, gamma, beta, xn, rstd, c1, eps, save);
    else ln_fwd_t<float>(x, gamma, beta, xn, rstd, c1, eps, save);
  }
  return {xn, rstd, c1};
}

// h = silu(xn Wa^T) * (xn Wb^T); xn [M, D], wa / wb [HID, D]. HID must be a multiple of 128.
torch::Tensor swiglu(torch::Tensor xn, torch::Tensor wa, torch::Tensor wb) {
  const int64_t M = xn.size(0), D = xn.size(1), HID = wa.size(0);
  check_act(xn, D, "xn"); check_act(wa, D, "wa"); check_act(wb, D, "wb");
  expect(wa.size(0) == wb.size(0) && HID % 128 == 0 && D % 64 == 0, "hidden width must be a multiple of 128");
  expect(xn.scalar_type() == wa.scalar_type() && xn.scalar_type() == wb.scalar_type(), "dtype mismatch");
  auto h = torch::empty({M, HID}, xn.options());
  if (M == 0) return h;
  const int bn = 2 * pick_width(M, HID, {128, 64}, 2);                 // hidden units per tile = bn / 2
  if (is_bf16(xn)) gemm_bn<__nv_bfloat16, MODE_SWIGLU>(bn, xn, wa, wb, nullptr, &h, M, HID, D, nullptr, h.data_ptr());
  else gemm_bn<float, MODE_SWIGLU>(bn, xn, wa, wb, nullptr, &h, M, HID, D, nullptr, h.data_ptr());
  return h;
}

// bf16: h = silu(a) b as `swiglu`, plus ab [M, 2 HID] = [a | b] saved for `gate_ab`.
std::vector<torch::Tensor> swiglu_save(torch::Tensor xn, torch::Tensor wa, torch::Tensor wb) {
  const int64_t M = xn.size(0), D = xn.size(1), HID = wa.size(0);
  check_act(xn, D, "xn"); check_act(wa, D, "wa"); check_act(wb, D, "wb");
  expect(is_bf16(xn) && is_bf16(wa) && is_bf16(wb), "saved a / b path is bf16 only");
  expect(wa.size(0) == wb.size(0) && HID % 128 == 0 && D % 64 == 0, "hidden width must be a multiple of 128");
  auto h = torch::empty({M, HID}, xn.options());
  auto ab = torch::empty({M, 2 * HID}, xn.options());
  if (M == 0) return {h, ab};
  const int bn = 2 * pick_width(M, HID, {128, 64}, 2);
  if (bn == 256) swiglu_save_bn<256>(xn, wa, wb, h, ab);
  else swiglu_save_bn<128>(xn, wa, wb, h, ab);
  return {h, ab};
}

// bf16 backward gate from the saved a | b: returns dab [M, 2 HID] (dA | dB).  ws [D, HID] is read as stored (MN-major operand).
torch::Tensor gate_ab(torch::Tensor dy, torch::Tensor ws, torch::Tensor ab) {
  const int64_t M = dy.size(0), D = dy.size(1), HID = ws.size(1);
  check_act(dy, D, "dy"); check_act(ws, HID, "ws"); check_act(ab, 2 * HID, "ab");
  expect(is_bf16(dy) && is_bf16(ws) && is_bf16(ab) && ab.size(0) == M && ws.size(0) == D && HID % 128 == 0 && D % 64 == 0, "shapes / dtype");
  auto dab = torch::empty({M, 2 * HID}, dy.options());
  if (M == 0) return dab;
  const int bn = pick_width(M, HID, {128, 64});
  if (bn == 128) gate_ab_bn<128>(dy, ws, ab, dab);
  else gate_ab_bn<64>(dy, ws, ab, dab);
  return dab;
}

// out = h Ws^T + x; h [M, HID], ws [D, HID], x [M, D].
torch::Tensor squeeze_res(torch::Tensor h, torch::Tensor ws, torch::Tensor x) {
  const int64_t M = h.size(0), HID = h.size(1), D = ws.size(0);
  check_act(h, HID, "h"); check_act(ws, HID, "ws"); check_act(x, D, "x");
  expect(x.size(0) == M && HID % 64 == 0, "shapes");
  expect(h.scalar_type() == ws.scalar_type() && h.scalar_type() == x.scalar_type(), "dtype mismatch");
  auto out = torch::empty_like(x);
  if (M == 0) return out;
  const int bn = pick_bn(M, D);
  if (is_bf16(h)) gemm_bn<__nv_bfloat16, MODE_RES>(bn, h, ws, ws, &x, &out, M, D, HID, cp<__nv_bfloat16>(x), out.data_ptr());
  else gemm_bn<float, MODE_RES>(bn, h, ws, ws, &x, &out, M, D, HID, cp<float>(x), out.data_ptr());
  return out;
}


// Range scales of the fp16 intermediate path, in two launches (a memset and these) instead of the ~20 small torch kernels they replace.
// st = [max|gamma|, max|beta|, max row L1 norm of Wa / Wb, max|Ws|] (non-negative floats: atomicMax on the bit pattern orders them).
__global__ void __launch_bounds__(256)
h16_absmax_kernel(const float* __restrict__ gamma, const float* __restrict__ beta, const float* __restrict__ wa, const float* __restrict__ wb,
                  const float* __restrict__ ws, float* __restrict__ st, int D, int HID) {
  const int lane = threadIdx.x & 31;
  const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5, nwarps = ((long long)gridDim.x * blockDim.x) >> 5;
  float rl1 = 0.f, wmax = 0.f, gmax = 0.f, bmax = 0.f;
  for (long long r = warp; r < 2LL * HID; r += nwarps) {                         // one warp per row of Wa (then Wb)
    const float* row = (r < HID ? wa + r * D : wb + (r - HID) * D);
    float s = 0.f;
    for (int c = lane; c < D; c += 32) s += fabsf(row[c]);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    rl1 = fmaxf(rl1, s);
  }
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, nt = (long long)gridDim.x * blockDim.x;
  for (long long i = tid; i < (long long)D * HID; i += nt) wmax = fmaxf(wmax, fabsf(ws[i]));
  for (long long i = tid; i < D; i += nt) { gmax = fmaxf(gmax, fabsf(gamma[i])); bmax = fmaxf(bmax, fabsf(beta[i])); }
  for (int o = 16; o > 0; o >>= 1) {
    rl1 = fmaxf(rl1, __shfl_xor_sync(0xffffffffu, rl1, o)); wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffffu, wmax, o));
    gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffffu, gmax, o)); bmax = fmaxf(bmax, __shfl_xor_sync(0xffffffffu, bmax, o));
  }
  if (lane == 0) {
    int* p = reinterpret_cast<int*>(st);
    atomicMax(p + 0, __float_as_int(gmax)); atomicMax(p + 1, __float_as_int(bmax));
    atomicMax(p + 2, __float_as_int(rl1)); atomicMax(p + 3, __float_as_int(wmax));
  }
}

// inv_sh = 1 / s_h, s = s_h * s_w, ws16 = half(Ws / s_w): s_h = max(1, (sqrt(D) max|gamma| + max|beta|)^2 max-row-L1^2 / 60000), s_w = max(1, max|Ws| / 60000)
__global__ void __launch_bounds__(256)
h16_apply_kernel(const float* __restrict__ ws, const float* __restrict__ st, __half* __restrict__ ws16, float* __restrict__ inv_sh,
                 float* __restrict__ s_out, long long n, int D) {
  const float xn_max = sqrtf((float)D) * st[0] + st[1];
  const float a_max = xn_max * st[2];
  const float s_h = fmaxf(a_max * a_max * (1.0f / 60000.0f), 1.0f);
  const float s_w = fmaxf(st[3] * (1.0f / 60000.0f), 1.0f);
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) { inv_sh[0] = 1.0f / s_h; s_out[0] = s_h * s_w; }
  for (long long i = tid; i < n; i += (long long)gridDim.x * blockDim.x) ws16[i] = __float2half_rn(ws[i] / s_w);
}

std::vector<torch::Tensor> h16_scales(torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa, torch::Tensor wb, torch::Tensor ws) {
  const int64_t HID = wa.size(0), D = wa.size(1);
  check_act(wa, D, "wa"); check_act(wb, D, "wb"); check_act(ws, HID, "ws");
  expect(wb.size(0) == HID && ws.size(0) == D && gamma.numel() == D && beta.numel() == D, "weight shapes");
  expect(wa.scalar_type() == torch::kFloat32 && wb.scalar_type() == torch::kFloat32 && ws.scalar_type() == torch::kFloat32 &&
         gamma.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32 && gamma.is_contiguous() && beta.is_contiguous(), "fp32 operands");
  auto f32 = wa.options().dtype(torch::kFloat32);
  auto st = torch::empty({4}, f32), inv_sh = torch::empty({1}, f32), s = torch::empty({1}, f32);
  auto ws16 = torch::empty({D, HID}, wa.options().dtype(torch::kHalf));
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(st.data_ptr(), 0, 4 * sizeof(float), stream);
  h16_absmax_kernel<<<128, 256, 0, stream>>>(gamma.data_ptr<float>(), beta.data_ptr<float>(), cp<float>(wa), cp<float>(wb), cp<float>(ws), st.data_ptr<float>(), (int)D, (int)HID);
  const long long n = D * HID;
  const unsigned blocks = (unsigned)std::min<long long>(1024, (n + 255) / 256);
  h16_apply_kernel<<<blocks, 256, 0, stream>>>(cp<float>(ws), st.data_ptr<float>(), reinterpret_cast<__half*>(ws16.data_ptr()), inv_sh.data_ptr<float>(), s.data_ptr<float>(), n, (int)D);
  return {inv_sh, s, ws16};
}

// fp32 activations on TF32 tensor cores, with the (M, HID) intermediate in fp16: the same 10-bit mantissa as TF32 in half the bytes, which
// are the dominant traffic of the two GEMMs (written by one, read by the other).  `inv_s` / `s` are fp32 device scalars that keep h inside
// fp16's range (the caller bounds |h|); the squeeze reads Ws as fp16 too (also divided by its own scale, folded into `s`).
torch::Tensor swiglu_h16(torch::Tensor xn, torch::Tensor wa, torch::Tensor wb, torch::Tensor inv_s) {
  const int64_t M = xn.size(0), D = xn.size(1), HID = wa.size(0);
  check_act(xn, D, "xn"); check_act(wa, D, "wa"); check_act(wb, D, "wb");
  expect(wa.size(0) == wb.size(0) && HID % 128 == 0 && D % 64 == 0, "hidden width must be a multiple of 128");
  expect(xn.scalar_type() == torch::kFloat32 && wa.scalar_type() == torch::kFloat32 && wb.scalar_type() == torch::kFloat32, "fp32 operands");
  expect(inv_s.is_cuda() && inv_s.scalar_type() == torch::kFloat32 && inv_s.numel() == 1, "scale");
  auto h = torch::empty({M, HID}, xn.options().dtype(torch::kHalf));
  if (M == 0) return h;
  const int bn = 2 * pick_width(M, HID, {128, 64}, 2);
  gemm_bn<float, MODE_SWIGLU, __half>(bn, xn, wa, wb, nullptr, &h, M, HID, D, nullptr, h.data_ptr(), inv_s.data_ptr<float>());
  return h;
}

// out = (h Ws^T) * s + x; h fp16 [M, HID], ws fp16 [D, HID], x / out fp32 [M, D].
torch::Tensor squeeze_res_h16(torch::Tensor h, torch::Tensor ws, torch::Tensor x, torch::Tensor s) {
  const int64_t M = h.size(0), HID = h.size(1), D = ws.size(0);
  check_act(h, HID, "h"); check_act(ws, HID, "ws"); check_act(x, D, "x");
  expect(x.size(0) == M && HID % 64 == 0, "shapes");
  expect(is_f16(h) && is_f16(ws) && x.scalar_type() == torch::kFloat32, "fp16 h / ws, fp32 x");
  expect(s.is_cuda() && s.scalar_type() == torch::kFloat32 && s.numel() == 1, "scale");
  auto out = torch::empty_like(x);
  if (M == 0) return out;
  const int bn = pick_bn(M, D);
  gemm_bn<__half, MODE_RES, float>(bn, h, ws, ws, &x, &out, M, D, HID, cp<float>(x), out.data_ptr(), s.data_ptr<float>());
  return out;
}

// dxn = dab @ Wab (activation dtype), with wabt = [Wa;Wb]^T [D, 2 HID]; dab [M, 2 HID].
torch::Tensor dxn_gemm(torch::Tensor dab, torch::Tensor wabt) {
  const int64_t M = dab.size(0), K = dab.size(1), D = wabt.size(0);
  check_act(dab, K, "dab"); check_act(wabt, K, "wabt");
  expect(dab.scalar_type() == wabt.scalar_type() && K % 64 == 0, "shapes / dtype");
  auto out = torch::empty({M, D}, dab.options());
  if (M == 0) return out;
  const int bn = pick_bn(M, D);
  if (is_bf16(dab)) gemm_bn<__nv_bfloat16, MODE_OUT>(bn, dab, wabt, wabt, nullptr, &out, M, D, K, nullptr, out.data_ptr());
  else gemm_bn<float, MODE_OUT>(bn, dab, wabt, wabt, nullptr, nullptr, M, D, K, nullptr, out.data_ptr());
  return out;
}

// bf16 SwiGLU backward from a cuBLAS dh [M, HID] and the saved ab [M, 2 HID]: returns dab [M, 2 HID] (dA | dB).
torch::Tensor swiglu_bwd(torch::Tensor dh, torch::Tensor ab) {
  const int64_t M = dh.size(0), HID = dh.size(1);
  check_act(dh, HID, "dh"); check_act(ab, 2 * HID, "ab");
  expect(is_bf16(dh) && is_bf16(ab) && ab.size(0) == M && HID % 8 == 0, "shapes / dtype");
  auto dab = torch::empty({M, 2 * HID}, dh.options());
  if (M == 0) return dab;
  const long long nvec = (long long)M * (HID / 8);
  const int grid = (int)std::min<long long>((nvec + 255) / 256, 8LL * sm_count());
  swiglu_bwd_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(cp<__nv_bfloat16>(dh), cp<__nv_bfloat16>(ab), mp<__nv_bfloat16>(dab), nvec,
                                                                     (int)HID);
  return dab;
}

// bf16 dxn = dab [Wa; Wb] with the weights read as stored (MN-major operand); dab [M, 2 HID], wa / wb [HID, D].
torch::Tensor dxn_gemm_t(torch::Tensor dab, torch::Tensor wa, torch::Tensor wb) {
  const int64_t M = dab.size(0), D = wa.size(1), HID = wa.size(0);
  check_act(dab, 2 * HID, "dab"); check_act(wa, D, "wa"); check_act(wb, D, "wb");
  expect(is_bf16(dab) && is_bf16(wa) && is_bf16(wb) && wb.size(0) == HID && D % 64 == 0 && HID % 64 == 0, "shapes / dtype");
  auto out = torch::empty({M, D}, dab.options());
  if (M == 0) return out;
  switch (pick_bn(M, D)) {
    case 64: dxn_bt_bn<64>(dab, wa, wb, out); break;
    case 128: dxn_bt_bn<128>(dab, wa, wb, out); break;
    case 192: dxn_bt_bn<192>(dab, wa, wb, out); break;
    default: dxn_bt_bn<256>(dab, wa, wb, out); break;
  }
  return out;
}

// Backward gate: returns (h [M, HID], dab [M, 2 HID]).  wst = Ws^T [HID, D].
std::vector<torch::Tensor> gate(torch::Tensor xn, torch::Tensor dy, torch::Tensor wa, torch::Tensor wb, torch::Tensor wst) {
  const int64_t M = xn.size(0), D = xn.size(1), HID = wa.size(0);
  check_act(xn, D, "xn"); check_act(dy, D, "dy"); check_act(wa, D, "wa"); check_act(wb, D, "wb"); check_act(wst, D, "wst");
  expect(dy.size(0) == M && wst.size(0) == HID && HID % 64 == 0 && D % 64 == 0, "shapes");
  auto h = torch::empty({M, HID}, xn.options());
  auto dab = torch::empty({M, 2 * HID}, xn.options());
  if (M == 0) return {h, dab};
  if (is_bf16(xn)) gate_t<__nv_bfloat16>(xn, dy, wa, wb, wst, h, dab);
  else gate_t<float>(xn, dy, wa, wb, wst, h, dab);
  return {h, dab};
}

// LayerNorm backward + residual: returns (dx, dgamma, dbeta).
std::vector<torch::Tensor> ln_bwd(torch::Tensor dxn, torch::Tensor x, torch::Tensor dy, torch::Tensor rstd, torch::Tensor c1,
                                  torch::Tensor gamma) {
  const int64_t M = x.size(0), D = x.size(1);
  check_act(x, D, "x"); check_act(dy, D, "dy");
  expect(dxn.scalar_type() == x.scalar_type() && dxn.is_contiguous() && dxn.size(0) == M && dxn.size(1) == D, "dxn must be [M, D] in the activation dtype");
  expect(rstd.numel() == M && c1.numel() == M && gamma.numel() == D && D % 64 == 0 && D <= 1024, "shapes");
  auto dx = torch::empty_like(x);
  const int grid = (int)std::max<int64_t>(1, std::min<int64_t>(2 * (int64_t)sm_count(), (M + 7) / 8));   // two CTAs per SM
  auto f32 = x.options().dtype(torch::kFloat32);
  auto pdg = torch::empty({grid, D}, f32), pdb = torch::empty({grid, D}, f32);
  if (is_bf16(x)) ln_bwd_t<__nv_bfloat16>(dxn, x, dy, rstd, c1, gamma, dx, pdg, pdb, grid);
  else ln_bwd_t<float>(dxn, x, dy, rstd, c1, gamma, dx, pdg, pdb, grid);
  auto dg = torch::empty({D}, f32), db = torch::empty({D}, f32);
  {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((D + 31) / 32, 1, 1);
    cfg.blockDim = dim3(1024, 1, 1);
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, reduce_cols_kernel, (const float*)pdg.data_ptr<float>(), (const float*)pdb.data_ptr<float>(),
                       dg.data_ptr<float>(), db.data_ptr<float>(), grid, (int)D);
  }
  return {dx, dg, db};
}

// Weight prep in one launch: returns (wst [HID, D], wabt [D, 2 HID], wa_r, wb_r, ws_r) -- bit 0..5 of `mask` selects Ws^T, the Wa^T / Wb^T halves of
// [Wa;Wb]^T, and copies of Wa / Wb / Ws; with `round` (fp32 activations) every output is rounded to nearest TF32. Outputs not selected are
// 1-element placeholders.
std::vector<torch::Tensor> prep(torch::Tensor wa, torch::Tensor wb, torch::Tensor ws, int64_t mask, bool round) {
  const int64_t HID = wa.size(0), D = wa.size(1);
  check_act(wa, D, "wa"); check_act(wb, D, "wb"); check_act(ws, HID, "ws");
  expect(wb.size(0) == HID && ws.size(0) == D && D % 32 == 0 && HID % 32 == 0, "weight shapes");
  expect(wa.scalar_type() == wb.scalar_type() && wa.scalar_type() == ws.scalar_type(), "dtype mismatch");
  auto slot = [&](int bit, std::initializer_list<int64_t> shape) {
    return (mask >> bit) & 1 ? torch::empty(shape, wa.options()) : torch::empty({1, 1}, wa.options());
  };
  auto wst = slot(0, {HID, D}), wabt = ((mask & 6) ? torch::empty({D, 2 * HID}, wa.options()) : torch::empty({1, 1}, wa.options()));
  auto war = slot(3, {HID, D}), wbr = slot(4, {HID, D}), wsr = slot(5, {D, HID});
  const unsigned tiles = (unsigned)(std::max(HID, D) / 32);
  const dim3 grid(tiles, tiles, 6), block(32, 8);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int r = round ? 1 : 0, mk = (int)mask;
  if (is_bf16(wa)) prep_bwd_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(cp<__nv_bfloat16>(wa), cp<__nv_bfloat16>(wb), cp<__nv_bfloat16>(ws), mp<__nv_bfloat16>(wst), mp<__nv_bfloat16>(wabt), mp<__nv_bfloat16>(war), mp<__nv_bfloat16>(wbr), mp<__nv_bfloat16>(wsr), (int)D, (int)HID, mk, r);
  else prep_bwd_kernel<float><<<grid, block, 0, stream>>>(cp<float>(wa), cp<float>(wb), cp<float>(ws), mp<float>(wst), mp<float>(wabt), mp<float>(war), mp<float>(wbr), mp<float>(wsr), (int)D, (int)HID, mk, r);
  return {wst, wabt, war, wbr, wsr};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_fwd", &ln_fwd);
  m.def("swiglu", &swiglu);
  m.def("squeeze_res", &squeeze_res);
  m.def("swiglu_h16", &swiglu_h16);
  m.def("squeeze_res_h16", &squeeze_res_h16);
  m.def("dxn_gemm", &dxn_gemm);
  m.def("gate", &gate);
  m.def("swiglu_save", &swiglu_save);
  m.def("gate_ab", &gate_ab);
  m.def("dxn_gemm_t", &dxn_gemm_t);
  m.def("swiglu_bwd", &swiglu_bwd);
  m.def("ln_bwd", &ln_bwd);
  m.def("prep", &prep);
  m.def("h16_scales", &h16_scales);
}
