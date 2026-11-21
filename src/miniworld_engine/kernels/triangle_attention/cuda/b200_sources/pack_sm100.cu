// The B200 (sm_100a) build of the H100 pack source (../h100_sources/pack.cu, included unchanged: the shared source stays byte-identical)
// with its graph RNG prologue routed to the ONE B200 writer (kernels/_graph_rng_sm100.cu, shared with TriMul's forms draw).
//
// pack.cu registers its own aten::fill_.Scalar override (graph_rng_fill) and keeps its own registry of the captured scalars; next to
// TriMul's registration the last one served the fills and the other family's graphs fell back to ATen. Here, without touching its text:
//   - `&graph_rng_fill` (the function its override registers) resolves to the forwarder below, which hands every fill to the shared
//     writer: whichever of the two registrations is last, the same registry serves both families. pack.cu's own fill body is compiled
//     under another name (graph_rng_fill followed by "(" is a function-like macro: its definition is renamed, the address-of is not).
//   - its register_graph_rng(state) calls (state a non-const PhiloxCudaState) resolve to the overload below taking a non-const
//     reference (an exact match beats pack.cu's const-reference definition, renamed by the same macro), which registers the captured
//     scalars with the shared writer.
// The writer's function table comes in once per load (torch.ops.mw_pack_sm100.set_graph_rng_api, b200_triattn._pack).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <ATen/ops/fill_native.h>
#include <torch/library.h>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {
struct SharedRngApi {
  int64_t abi;
  void (*register_state)(void*, void*);
  at::Tensor& (*fill)(at::Tensor&, const at::Scalar&);
};
const SharedRngApi* shared_rng_api = nullptr;
at::Tensor& graph_rng_fill(at::Tensor& tensor, const at::Scalar& value) {
  if (shared_rng_api) return shared_rng_api->fill(tensor, value);
  return at::native::fill_(tensor, value);
}
void mw_pack_register_graph_rng(at::PhiloxCudaState& state) {
  if (!state.captured_) return;
  TORCH_CHECK(shared_rng_api && shared_rng_api->abi == 1, "pack_sm100: captured Philox draw without the graph RNG writer (set_graph_rng_api)");
  shared_rng_api->register_state(state.seed_.ptr, state.offset_.ptr);
}
}  // namespace

#define graph_rng_fill(...) mw_pack_h100_graph_rng_fill(__VA_ARGS__)
#define register_graph_rng(...) mw_pack_register_graph_rng(__VA_ARGS__)
#include "../h100_sources/pack.cu"
#undef register_graph_rng
#undef graph_rng_fill

namespace {
void set_graph_rng_api(int64_t address) {
  const auto* a = reinterpret_cast<const SharedRngApi*>(address);
  TORCH_CHECK(a && a->abi == 1 && a->fill, "set_graph_rng_api: unknown writer table");
  shared_rng_api = a;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(mw_pack_sm100, m) { m.def("set_graph_rng_api(int address) -> ()", &set_graph_rng_api); }
