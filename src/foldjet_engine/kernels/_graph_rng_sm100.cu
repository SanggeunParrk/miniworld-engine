// The B200 graph RNG prologue writer: ONE aten::fill_.Scalar (CUDA) override for every family whose kernels draw Philox numbers
// inside a CUDA graph (TriMul's forms launch, wide_aux.cu; TriangleAttention's drop_scale, h100_sources/pack.cu built for sm_100a).
//
// A graph replay that holds a Philox draw first refreshes the default generator's two graph-owned int64 scalars with two aten::fill_
// launches (CUDAGeneratorState::replay_prologue: seed, then offset), each an ordinary launch after the previous stream work. For the
// scalars of a graph that holds a registered draw (registered at capture, released with the graph through a CUDA user object) the fill
// is served here: the seed fill is held and written by the offset fill's ONE launch, a programmatic dependent (griddepcontrol.wait
// first: the previous stream work complete and visible, as for an ordinary launch) whose launch overlaps the previous kernel. Same
// values, same stream order before the graph; every other fill_ goes to ATen's.
//
// Before (2026-11-17 part 6) TriMul (wide_aux.cu) and TriangleAttention (pack.cu) each registered their own override with their own
// registry: the last registration served the fills and the other family's scalars fell back to ATen's two launches. Here one
// registration serves one registry; the families' extensions register their graphs' scalars through `api()` (a C table of function
// pointers handed over by Python once per extension load: the extensions are separate shared objects).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/fill_native.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {
struct Reg {
  std::mutex m;
  std::unordered_map<void*, size_t> refs;
  std::unordered_map<void*, std::pair<void*, int>> role;   // pointer -> (partner, 0 seed / 1 offset)
  std::unordered_map<void*, int64_t> held;                 // seed pointer -> value waiting for its offset fill
  int mode = 2;   // 0: ATen's fills; 1: one ordinary launch; 2: one programmatic launch
};
Reg& reg() {
  static auto* r = new Reg;   // CUDA may run user-object destructors at shutdown: never destroyed
  return *r;
}
struct Ptrs {
  void* s;
  void* o;
  Ptrs(void* s_, void* o_) : s(s_), o(o_) {
    auto& r = reg();
    std::lock_guard<std::mutex> lk(r.m);
    ++r.refs[s]; ++r.refs[o];
    r.role[s] = {o, 0}; r.role[o] = {s, 1};
  }
  ~Ptrs() {
    auto& r = reg();
    std::lock_guard<std::mutex> lk(r.m);
    for (void* p : {s, o}) {
      auto it = r.refs.find(p);
      if (it != r.refs.end() && --it->second == 0) { r.refs.erase(it); r.role.erase(p); r.held.erase(p); }
    }
  }
};
__global__ void rng_scalar_write(int64_t* a, int64_t va, int64_t* b, int64_t vb) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (a) *a = va;
  *b = vb;
}
at::Tensor& rng_fill(at::Tensor& t, const at::Scalar& v) {
  if (t.is_cuda() && t.scalar_type() == at::kLong && t.numel() == 1) {
    auto& r = reg();
    std::unique_lock<std::mutex> lk(r.m);
    auto it = r.role.find(t.data_ptr());
    if (r.mode && it != r.role.end()) {
      const int64_t val = v.toLong();
      if (it->second.second == 0) { r.held[t.data_ptr()] = val; return t; }   // the seed: written by its offset's launch
      int64_t* a = nullptr;
      int64_t va = 0;
      auto h = r.held.find(it->second.first);
      if (h != r.held.end()) { a = static_cast<int64_t*>(h->first); va = h->second; r.held.erase(h); }
      const int pdl = r.mode == 2;
      lk.unlock();
      c10::cuda::CUDAGuard guard(t.device());
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(1); cfg.blockDim = dim3(32); cfg.dynamicSmemBytes = 0; cfg.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = attr; cfg.numAttrs = pdl;
      int64_t* b = t.data_ptr<int64_t>();
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, rng_scalar_write, a, va, b, val));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return t;
    }
  }
  return at::native::fill_(t, v);
}
void ensure_registered() {
  static auto lib = [] {
    auto l = std::make_unique<torch::Library>(torch::Library::IMPL, "aten", c10::DispatchKey::CUDA, __FILE__, __LINE__);
    l->impl("fill_.Scalar", &rng_fill);
    return l;
  }();
  (void)lib;
}
void CUDART_CB release(void* p) { delete static_cast<Ptrs*>(p); }
// The scalars (seed, offset) of a captured PhiloxCudaState, inside the capture of the current stream: served by the writer while the
// graph (or an executable instantiated from it) lives.
void register_state(void* seed, void* offset) {
  ensure_registered();
  cudaStreamCaptureStatus st;
  cudaGraph_t graph = nullptr;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(at::cuda::getCurrentCUDAStream(), &st, nullptr, &graph, nullptr, nullptr));
  TORCH_CHECK(st == cudaStreamCaptureStatusActive && graph, "graph RNG writer: captured Philox state without an active capture");
  auto ptrs = std::make_unique<Ptrs>(seed, offset);
  cudaUserObject_t obj;
  C10_CUDA_CHECK(cudaUserObjectCreate(&obj, ptrs.get(), release, 1, cudaUserObjectNoDestructorSync));
  ptrs.release();
  const auto rc = cudaGraphRetainUserObject(graph, obj, 1, cudaGraphUserObjectMove);
  if (rc != cudaSuccess) { C10_CUDA_CHECK(cudaUserObjectRelease(obj, 1)); C10_CUDA_CHECK(rc); }
}
// The table the families' extensions call (abi 1). Kept in sync with the struct their sources declare.
struct Api {
  int64_t abi;
  void (*register_state)(void*, void*);
  at::Tensor& (*fill)(at::Tensor&, const at::Scalar&);   // the writer's fill_ (for a family whose own override must forward here)
};
const Api kApi = {1, &register_state, &rng_fill};
}  // namespace

int64_t api() { return reinterpret_cast<int64_t>(&kApi); }
// 2 (default): one programmatic launch; 1: one ordinary launch; 0: ATen's two fills (A / B, tests)
void set_mode(int64_t mode) {
  auto& r = reg();
  std::lock_guard<std::mutex> lk(r.m);
  r.mode = (int)mode;
  r.held.clear();
}
int64_t pointer_count() {
  auto& r = reg();
  std::lock_guard<std::mutex> lk(r.m);
  return (int64_t)r.refs.size();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("api", &api);
  m.def("set_mode", &set_mode);
  m.def("pointer_count", &pointer_count);
}
