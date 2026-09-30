// Torch bindings for the fused sm_100a Transition forward and backward.
//
// The kernels (`tr_fwd_sm100.cu`: 2-CTA forward; `tr_bwd_sm100.cu`: two-role backward + partial reduction) are NOT compiled into this
// extension. `fused_sm100a.py` builds them into two cubins with the newest nvcc on the machine that knows sm_100a, and this file loads
// those through the driver API and launches them. The reason is measured, not stylistic: the same backward source built by CUDA 12.9's
// ptxas (the toolkit torch cu129 pins) runs ~8 % slower than built by 13.1's (L384 backward 250 vs 230 us, L768 990 vs 885 us), and a
// cubin is independent of the runtime this extension links against.
//
// As in the sm_90a binding, driver entry points are resolved through `cudaGetDriverEntryPoint` (no `-lcuda` in torch's JIT build), and
// TMA descriptors are cached on (pointer, shape): a descriptor is valid for exactly one base pointer.
//
// The other channel widths (`fused_wide_sm100a.py`: D = 64 / 256 / 384 / 512) use the generic surface at the end of this file:
// `load_cubin` (a named cubin, loaded per device), `tmap` (a tensor map, cached like the D = 128 ones) and `launch_kernel` (grid /
// block / cluster and a list of arguments: tensor maps by value, tensors as device pointers, ints as int32, floats as fp32). The
// sequence of kernels per width is assembled in Python.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t D = 128, H = 512, ROWS = 128;

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument("transition fused sm100a: " + what);
}

// ---------------------------------------------------------------------------------------------------------------- driver API
template <typename F>
F entry(const char* name) {
  void* p = nullptr;
  cudaDriverEntryPointQueryResult q{};
  cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
  if (e != cudaSuccess || p == nullptr) throw std::runtime_error(std::string("driver entry point unavailable: ") + name);
  return reinterpret_cast<F>(p);
}

struct Driver {
  CUresult (*encode)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                     const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  CUresult (*load)(CUmodule*, const void*);
  CUresult (*function)(CUfunction*, CUmodule, const char*);
  CUresult (*global)(CUdeviceptr*, size_t*, CUmodule, const char*);
  CUresult (*dtoh)(void*, CUdeviceptr, size_t);
  CUresult (*attribute)(CUfunction, CUfunction_attribute, int);
  CUresult (*launch)(const CUlaunchConfig*, CUfunction, void**, void**);
  CUresult (*error)(CUresult, const char**);
  CUresult (*clusters)(int*, CUfunction, const CUlaunchConfig*);
};

const Driver& drv() {
  static const Driver d = [] {
    Driver x{};
    x.encode = entry<decltype(x.encode)>("cuTensorMapEncodeTiled");
    x.load = entry<decltype(x.load)>("cuModuleLoadData");
    x.function = entry<decltype(x.function)>("cuModuleGetFunction");
    x.global = entry<decltype(x.global)>("cuModuleGetGlobal");
    x.dtoh = entry<decltype(x.dtoh)>("cuMemcpyDtoH");
    x.attribute = entry<decltype(x.attribute)>("cuFuncSetAttribute");
    x.launch = entry<decltype(x.launch)>("cuLaunchKernelEx");
    x.error = entry<decltype(x.error)>("cuGetErrorString");
    x.clusters = entry<decltype(x.clusters)>("cuOccupancyMaxActiveClusters");
    return x;
  }();
  return d;
}

void check(CUresult r, const std::string& what) {
  if (r == CUDA_SUCCESS) return;
  const char* msg = nullptr;
  drv().error(r, &msg);
  throw std::runtime_error("transition fused sm100a: " + what + ": " + (msg ? msg : "unknown driver error"));
}

// ---------------------------------------------------------------------------------------------------------------- kernels
struct Kernels {
  CUfunction fwd = nullptr, bwd = nullptr, reduce = nullptr;
  int fwd_smem = 0, bwd_smem = 0;
};

std::string g_fwd_cubin, g_bwd_cubin;
std::map<int, Kernels> g_kernels;                           // per device: a module lives in one context
std::mutex g_lock;

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  expect(f.good(), "cannot read " + path);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

CUmodule load_module(const std::string& path) {
  const std::vector<char> image = read_file(path);
  CUmodule m = nullptr;
  check(drv().load(&m, image.data()), "cuModuleLoadData(" + path + ")");
  return m;
}

int read_int(CUmodule m, const char* name) {
  CUdeviceptr p = 0;
  size_t bytes = 0;
  check(drv().global(&p, &bytes, m, name), std::string("cuModuleGetGlobal(") + name + ")");
  int v = 0;
  check(drv().dtoh(&v, p, sizeof(v)), std::string("read ") + name);
  return v;
}

const Kernels& kernels() {
  int dev = 0;
  cudaGetDevice(&dev);
  std::lock_guard<std::mutex> guard(g_lock);
  auto it = g_kernels.find(dev);
  if (it != g_kernels.end()) return it->second;
  expect(!g_fwd_cubin.empty() && !g_bwd_cubin.empty(), "kernels not loaded (call load() first)");
  cudaFree(nullptr);                                         // the runtime's primary context is current before any module load
  Kernels k;
  CUmodule mf = load_module(g_fwd_cubin), mb = load_module(g_bwd_cubin);
  check(drv().function(&k.fwd, mf, "transition_fwd2_sm100"), "cuModuleGetFunction(transition_fwd2_sm100)");
  check(drv().function(&k.bwd, mb, "transition_bwd_sm100"), "cuModuleGetFunction(transition_bwd_sm100)");
  check(drv().function(&k.reduce, mb, "transition_bwd_reduce"), "cuModuleGetFunction(transition_bwd_reduce)");
  k.fwd_smem = read_int(mf, "transition_sm100_fwd_smem_bytes");
  k.bwd_smem = read_int(mb, "transition_sm100_bwd_smem_bytes");
  check(drv().attribute(k.fwd, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.fwd_smem), "forward smem opt-in");
  check(drv().attribute(k.bwd, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.bwd_smem), "backward smem opt-in");
  return g_kernels.emplace(dev, k).first->second;
}

void launch(CUfunction f, unsigned grid, unsigned block, unsigned smem, unsigned cluster, cudaStream_t stream, std::vector<void*> args,
            const char* what) {
  CUlaunchConfig cfg{};
  cfg.gridDimX = grid; cfg.gridDimY = 1; cfg.gridDimZ = 1;
  cfg.blockDimX = block; cfg.blockDimY = 1; cfg.blockDimZ = 1;
  cfg.sharedMemBytes = smem;
  cfg.hStream = reinterpret_cast<CUstream>(stream);
  CUlaunchAttribute attr[1];
  if (cluster > 1) {
    attr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attr[0].value.clusterDim.x = cluster; attr[0].value.clusterDim.y = 1; attr[0].value.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
  }
  check(drv().launch(&cfg, f, args.data(), nullptr), what);
}

// ---------------------------------------------------------------------------------------------------------------- tensor maps
using Key = std::array<uint64_t, 3>;

// 2-D bf16 descriptor over a contiguous [outer][inner] tensor, 64 x 64 boxes, 128-B swizzle -- every tile load and store in both kernels
// uses this shape.
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t inner, uint32_t outer) {
  static std::map<Key, CUtensorMap> cache;
  static std::mutex lock;
  const Key key{reinterpret_cast<uint64_t>(t.data_ptr()), inner, outer};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(inner) * 2};
  const cuuint32_t box[2] = {64, 64};
  const cuuint32_t elem[2] = {1, 1};
  check(drv().encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled");
  return cache.emplace(key, map).first->second;
}

int sm_count(const torch::Tensor& t) { return at::cuda::getDeviceProperties(t.device().index())->multiProcessorCount; }

void expect_weights(const torch::Tensor& wa, const torch::Tensor& wb, const torch::Tensor& ws) {
  for (const auto* w : {&wa, &wb, &ws})
    expect(w->is_cuda() && w->is_contiguous() && w->scalar_type() == torch::kBFloat16, "weights must be contiguous cuda bf16");
  expect(wa.size(0) == H && wa.size(1) == D && wb.sizes() == wa.sizes(), "wa / wb must be [512, 128]");
  expect(ws.size(0) == D && ws.size(1) == H, "ws must be [128, 512]");
}

}  // namespace

// The two cubins built by `fused_sm100a.py`. Loaded here on the current device, so a cubin the driver rejects fails the build step
// (and the caller keeps its other path); other devices load theirs on first use.
void load(const std::string& fwd_cubin, const std::string& bwd_cubin) {
  {
    std::lock_guard<std::mutex> guard(g_lock);
    if (fwd_cubin != g_fwd_cubin || bwd_cubin != g_bwd_cubin) g_kernels.clear();
    g_fwd_cubin = fwd_cubin;
    g_bwd_cubin = bwd_cubin;
  }
  kernels();
}

// x [M,128] bf16, gamma / beta [128] fp32, wa / wb [512,128] bf16, ws [128,512] bf16 (the module layouts).
// Returns (out, xn, rstd, c1) with out = transition(x) + x. With save = false only `out` is written (xn / rstd / c1 are 1-element
// placeholders): the kernel skips those stores at run time.
std::vector<torch::Tensor> transition_fused_fwd(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa,
                                                torch::Tensor wb, torch::Tensor ws, double eps, bool save) {
  const int64_t M = x.size(0);
  expect(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && x.size(1) == D, "x must be contiguous cuda bf16 [M, 128]");
  expect(M > 0 && M % ROWS == 0, "row count must be a whole number of 128-row tiles");
  expect(gamma.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32 && gamma.is_contiguous() && beta.is_contiguous(),
         "gamma / beta must be contiguous fp32");
  expect_weights(wa, wb, ws);
  const Kernels& k = kernels();
  auto out = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto xn = save ? torch::empty_like(x) : torch::empty({1, D}, x.options());
  auto rstd = torch::empty({save ? M : 1}, f32);
  auto c1 = torch::empty({save ? M : 1}, f32);
  // the inference call passes a valid map in xn's slot (the kernel never stores through it when save = 0)
  const CUtensorMap& mx = tile_map(x, D, M);
  const CUtensorMap& mwa = tile_map(wa, D, H);
  const CUtensorMap& mwb = tile_map(wb, D, H);
  const CUtensorMap& mws = tile_map(ws, H, D);
  const CUtensorMap& mout = tile_map(out, D, M);
  const CUtensorMap& mxn = save ? tile_map(xn, D, M) : mout;
  const float* g = gamma.data_ptr<float>();
  const float* b = beta.data_ptr<float>();
  float* rs = rstd.data_ptr<float>();
  float* c = c1.data_ptr<float>();
  int tiles = static_cast<int>(M / ROWS), sv = save ? 1 : 0;
  float e = static_cast<float>(eps);
  // persistent grid of 2-CTA clusters: min(SMs, tiles) rounded down to the pair; a pair walks the leader's tile count
  int grid = std::min<int>(sm_count(x), tiles);
  grid -= grid % 2;
  if (grid < 2) grid = 2;
  launch(k.fwd, grid, 512, k.fwd_smem, 2, at::cuda::getCurrentCUDAStream(),
         {(void*)&mx, (void*)&mwa, (void*)&mwb, (void*)&mws, (void*)&mout, (void*)&mxn, (void*)&g, (void*)&b, (void*)&rs, (void*)&c,
          (void*)&tiles, (void*)&e, (void*)&sv},
         "forward launch");
  return {out, xn, rstd, c1};
}

// dy [M,128] bf16 (gradient of the module output) plus what the forward saved; `repl` = hidden-slice replicas of the weight role
// (8 * repl CTAs). Returns (dx, dgamma, dbeta, dWa, dWb, dWs); dx already carries the residual branch, dgamma / dbeta are fp32 and the
// weight gradients bf16 (fp32 sums rounded once).
std::vector<torch::Tensor> transition_fused_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor xn, torch::Tensor rstd, torch::Tensor c1,
                                                torch::Tensor gamma, torch::Tensor wa, torch::Tensor wb, torch::Tensor ws, int64_t repl) {
  const int64_t M = x.size(0);
  expect(dy.is_cuda() && dy.is_contiguous() && dy.scalar_type() == torch::kBFloat16 && dy.sizes() == x.sizes(),
         "dy must be contiguous cuda bf16 shaped like x");
  expect(x.is_contiguous() && xn.is_contiguous() && xn.sizes() == x.sizes() && x.size(1) == D, "x / xn must be contiguous [M, 128]");
  expect(M > 0 && M % ROWS == 0, "row count must be a whole number of 128-row tiles");
  expect(rstd.numel() == M && c1.numel() == M && rstd.scalar_type() == torch::kFloat32 && c1.scalar_type() == torch::kFloat32,
         "rstd / c1 must be the forward's fp32 statistics");
  expect(gamma.scalar_type() == torch::kFloat32 && gamma.is_contiguous(), "gamma must be contiguous fp32");
  expect_weights(wa, wb, ws);
  const Kernels& k = kernels();
  const int sms = sm_count(x);
  int ndw = static_cast<int>(8 * repl), ndx = sms - ndw;
  expect(repl >= 1 && sms % 2 == 0 && ndx >= 2 && ndx % 2 == 0, "SM count / replica count do not form whole 2-CTA clusters");
  auto f32 = x.options().dtype(torch::kFloat32);
  auto dx = torch::empty_like(x);
  auto partab = torch::empty({ndw, 128, 128}, f32);
  auto parts = torch::empty({ndw, 128, 64}, f32);
  auto dgbw = torch::empty({ndx * 4, 256}, f32);
  auto dwa = torch::empty_like(wa), dwb = torch::empty_like(wb), dws = torch::empty_like(ws);
  auto dgam = torch::empty({D}, f32), dbeta = torch::empty({D}, f32);
  const CUtensorMap& mdy = tile_map(dy, D, M);
  const CUtensorMap& mxn = tile_map(xn, D, M);
  const CUtensorMap& mx = tile_map(x, D, M);
  const CUtensorMap& mws = tile_map(ws, H, D);
  const CUtensorMap& mwa = tile_map(wa, D, H);
  const CUtensorMap& mwb = tile_map(wb, D, H);
  const CUtensorMap& mdx = tile_map(dx, D, M);
  const float* rs = rstd.data_ptr<float>();
  const float* c = c1.data_ptr<float>();
  const float* g = gamma.data_ptr<float>();
  const void* xg = x.data_ptr();
  float* pab = partab.data_ptr<float>();
  float* ps = parts.data_ptr<float>();
  float* pg = dgbw.data_ptr<float>();
  void* pwa = dwa.data_ptr();
  void* pwb = dwb.data_ptr();
  void* pws = dws.data_ptr();
  float* pdg = dgam.data_ptr<float>();
  float* pdb = dbeta.data_ptr<float>();
  int tiles = static_cast<int>(M / ROWS), nrows_dg = ndx * 4;
  auto stream = at::cuda::getCurrentCUDAStream();
  launch(k.bwd, sms, 512, k.bwd_smem, 2, stream,
         {(void*)&mdy, (void*)&mxn, (void*)&mx, (void*)&mws, (void*)&mwa, (void*)&mwb, (void*)&mdx, (void*)&rs, (void*)&c, (void*)&g,
          (void*)&xg, (void*)&pab, (void*)&ps, (void*)&pg, (void*)&tiles, (void*)&ndw},
         "backward launch");
  const int nred = static_cast<int>(3 * H * D + 256);
  launch(k.reduce, (nred + 255) / 256, 256, 0, 1, stream,
         {(void*)&pab, (void*)&ps, (void*)&pg, (void*)&pwa, (void*)&pwb, (void*)&pws, (void*)&pdg, (void*)&pdb, (void*)&ndw,
          (void*)&nrows_dg},
         "reduction launch");
  return {dx, dgam, dbeta, dwa, dwb, dws};
}

// ---------------------------------------------------------------------------------------------------------------- generic surface
namespace {

// Make `index`'s primary context current on this thread. The backward runs on autograd's device thread, whose first driver call
// (cuTensorMapEncodeTiled) otherwise fails with "invalid device context"; cudaSetDevice (not cudaFree(nullptr)) is safe under
// CUDA-graph capture.
void make_current(int index) {
  int dev = -1;
  cudaGetDevice(&dev);
  if (index < 0) index = dev;
  cudaSetDevice(index);
}

struct TMap {
  CUtensorMap map;
};

struct Cubin {
  CUmodule module = nullptr;
  int smem = 0;                                              // `transition_smem_bytes` of the cubin, 0 if it has none (static smem)
  std::map<std::string, CUfunction> functions;
};

std::map<std::string, std::string> g_cubin_paths;           // name -> path
std::map<std::pair<int, std::string>, Cubin> g_cubins;       // (device, name) -> module

Cubin& cubin(const std::string& name) {
  int dev = 0;
  cudaGetDevice(&dev);
  auto key = std::make_pair(dev, name);
  auto it = g_cubins.find(key);
  if (it != g_cubins.end()) return it->second;
  auto p = g_cubin_paths.find(name);
  expect(p != g_cubin_paths.end(), "cubin " + name + " not loaded (call load_cubin() first)");
  cudaFree(nullptr);
  Cubin c;
  c.module = load_module(p->second);
  CUdeviceptr ptr = 0;
  size_t bytes = 0;
  // the widths kernels export transition_smem_bytes; the D = 128 ones (also built at n = 2) keep their own names
  for (const char* sym : {"transition_smem_bytes", "transition_sm100_fwd_smem_bytes", "transition_sm100_bwd_smem_bytes"})
    if (drv().global(&ptr, &bytes, c.module, sym) == CUDA_SUCCESS) { c.smem = read_int(c.module, sym); break; }
  return g_cubins.emplace(key, std::move(c)).first->second;
}

CUfunction function(Cubin& c, const std::string& name) {
  auto it = c.functions.find(name);
  if (it != c.functions.end()) return it->second;
  CUfunction f = nullptr;
  check(drv().function(&f, c.module, name.c_str()), "cuModuleGetFunction(" + name + ")");
  if (c.smem > 0) check(drv().attribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, c.smem), name + " smem opt-in");
  check(drv().attribute(f, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1), name + " cluster size opt-in");   // 12-CTA clusters
  return c.functions.emplace(name, f).first->second;
}

}  // namespace

void load_cubin(const std::string& name, const std::string& path) {
  std::lock_guard<std::mutex> guard(g_lock);
  auto it = g_cubin_paths.find(name);
  if (it != g_cubin_paths.end() && it->second != path)
    for (auto c = g_cubins.begin(); c != g_cubins.end();) c = c->first.second == name ? g_cubins.erase(c) : std::next(c);
  g_cubin_paths[name] = path;
  cubin(name);                                               // load on the current device now: a rejected cubin fails here
}

// 2-D bf16 (or fp32, from the tensor) descriptor over [outer][inner] elements with rows `row_stride` elements apart (-1: inner, i.e. contiguous), a
// (box_inner x box_outer) box and 128-B swizzle. A row stride wider than `inner` describes a column slice of a wider matrix.
TMap tmap(const torch::Tensor& t, int64_t inner, int64_t outer, int64_t box_inner, int64_t box_outer, int64_t row_stride) {
  static std::map<std::array<uint64_t, 6>, CUtensorMap> cache;
  static std::mutex lock;
  if (row_stride < 0) row_stride = inner;
  make_current(t.device().index());
  const bool f32 = t.scalar_type() == torch::kFloat32;
  const int64_t esz = f32 ? 4 : 2;
  expect(t.is_cuda() && (f32 || t.scalar_type() == torch::kBFloat16) && row_stride >= inner && (row_stride * esz) % 16 == 0,
         "tmap: cuda bf16 or fp32 tensor, row stride >= inner and a multiple of 16 bytes");
  expect(row_stride == inner ? (t.is_contiguous() && t.numel() == inner * outer)
                             : (t.dim() == 2 && t.size(0) == outer && t.size(1) == inner && t.stride(0) == row_stride && t.stride(1) == 1),
         "tmap: a contiguous tensor of inner x outer elements, or a [outer, inner] view with that row stride");
  const std::array<uint64_t, 6> key{reinterpret_cast<uint64_t>(t.data_ptr()), static_cast<uint64_t>(inner), static_cast<uint64_t>(outer),
                                    static_cast<uint64_t>(box_inner), static_cast<uint64_t>(box_outer),
                                    static_cast<uint64_t>(row_stride) | (f32 ? (1ull << 62) : 0ull)};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it == cache.end()) {
    if (cache.size() > 4096) cache.clear();                  // pointers recur through the caching allocator; bound the rest
    CUtensorMap map{};
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(inner), static_cast<cuuint64_t>(outer)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_stride * esz)};
    const cuuint32_t box[2] = {static_cast<cuuint32_t>(box_inner), static_cast<cuuint32_t>(box_outer)};
    const cuuint32_t elem[2] = {1, 1};
    check(drv().encode(&map, f32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides,
                       box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                       CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled");
    it = cache.emplace(key, map).first;
  }
  return TMap{it->second};
}

// Launch `kernel` of cubin `name` on the current stream with the cubin's dynamic shared memory. `args` in the kernel's parameter order.
void launch_kernel(const std::string& name, const std::string& kernel, int64_t grid, int64_t block, int64_t cluster, py::list args) {
  make_current(-1);
  CUfunction f = nullptr;
  int smem = 0;
  {
    std::lock_guard<std::mutex> guard(g_lock);
    Cubin& c = cubin(name);
    f = function(c, kernel);
    smem = c.smem;
  }
  const size_t n = args.size();
  std::vector<CUtensorMap> maps;
  std::vector<void*> ptrs;
  std::vector<int> ints;
  std::vector<float> floats;
  maps.reserve(n); ptrs.reserve(n); ints.reserve(n); floats.reserve(n);
  std::vector<void*> params;
  params.reserve(n);
  for (const auto& a : args) {
    if (py::isinstance<TMap>(a)) {
      maps.push_back(a.cast<const TMap&>().map);
      params.push_back(&maps.back());
    } else if (THPVariable_Check(a.ptr())) {
      ptrs.push_back(THPVariable_Unpack(a.ptr()).data_ptr());
      params.push_back(&ptrs.back());
    } else if (py::isinstance<py::bool_>(a) || py::isinstance<py::int_>(a)) {
      ints.push_back(static_cast<int>(a.cast<int64_t>()));
      params.push_back(&ints.back());
    } else if (py::isinstance<py::float_>(a)) {
      floats.push_back(static_cast<float>(a.cast<double>()));
      params.push_back(&floats.back());
    } else {
      throw std::invalid_argument("transition fused sm100a: launch_kernel argument of unsupported type");
    }
  }
  launch(f, static_cast<unsigned>(grid), static_cast<unsigned>(block), static_cast<unsigned>(smem), static_cast<unsigned>(cluster),
         at::cuda::getCurrentCUDAStream(), params, kernel.c_str());
}

// How many `cluster`-CTA clusters of `kernel` can be resident at once on the current device (one wave).
int64_t max_active_clusters(const std::string& name, const std::string& kernel, int64_t block, int64_t cluster) {
  make_current(-1);
  CUfunction f = nullptr;
  int smem = 0;
  {
    std::lock_guard<std::mutex> guard(g_lock);
    Cubin& c = cubin(name);
    f = function(c, kernel);
    smem = c.smem;
  }
  CUlaunchConfig cfg{};
  cfg.gridDimX = static_cast<unsigned>(cluster); cfg.gridDimY = 1; cfg.gridDimZ = 1;
  cfg.blockDimX = static_cast<unsigned>(block); cfg.blockDimY = 1; cfg.blockDimZ = 1;
  cfg.sharedMemBytes = static_cast<unsigned>(smem);
  CUlaunchAttribute attr[1];
  attr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attr[0].value.clusterDim.x = static_cast<unsigned>(cluster); attr[0].value.clusterDim.y = 1; attr[0].value.clusterDim.z = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  int n = 0;
  check(drv().clusters(&n, f, &cfg), "cuOccupancyMaxActiveClusters(" + kernel + ")");
  return n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_active_clusters", &max_active_clusters, "resident clusters of a loaded kernel at this cluster size (one wave)");
  py::class_<TMap>(m, "TMap");
  m.def("load_cubin", &load_cubin, "register a named cubin and load it on the current device");
  m.def("tmap", &tmap, "2-D bf16 128-B-swizzled tensor map (inner, outer, box_inner, box_outer, row_stride = -1: contiguous)",
        py::arg("t"), py::arg("inner"), py::arg("outer"), py::arg("box_inner"), py::arg("box_outer"), py::arg("row_stride") = -1);
  m.def("launch_kernel", &launch_kernel, "launch a kernel of a loaded cubin on the current stream");
  m.def("load", &load, "set the forward / backward cubins (loaded lazily per device)");
  m.def("transition_fused_fwd", &transition_fused_fwd, "fused sm_100a Transition forward (LN + SwiGLU expand + squeeze + residual)");
  m.def("transition_fused_bwd", &transition_fused_bwd, "fused sm_100a Transition backward (one kernel + a partial reduction)");
}
