// Pin a byte range in L2 for the kernels launched on torch's current stream (Ampere+ persisting access policy window).
// The token DiT step streams ~50 MB of activations through each block, so the residual x is evicted between the two row
// passes that read and write it. A persisting window keeps it (and xa, y) in L2 instead of HBM.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// One window per stream, so the range must be contiguous: allocate x, xa and y as slices of one buffer.
void set_window(torch::Tensor t, double hit_ratio) {
  size_t bytes = t.numel() * t.element_size();
  size_t cap = 0;
  cudaDeviceGetLimit(&cap, cudaLimitPersistingL2CacheSize);
  int max_persist = 0;
  cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, t.device().index());
  const size_t want = bytes < (size_t)max_persist ? bytes : (size_t)max_persist;
  if (cap < want) TORCH_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want) == cudaSuccess, "set limit");
  cudaStreamAttrValue v{};
  v.accessPolicyWindow.base_ptr = t.data_ptr();
  v.accessPolicyWindow.num_bytes = want;
  v.accessPolicyWindow.hitRatio = (float)hit_ratio;
  v.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  v.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  TORCH_CHECK(cudaStreamSetAttribute(at::cuda::getCurrentCUDAStream(), cudaStreamAttributeAccessPolicyWindow, &v) == cudaSuccess,
              "cudaStreamSetAttribute failed");
}

void clear_window() {
  cudaStreamAttrValue v{};
  v.accessPolicyWindow.num_bytes = 0;
  cudaStreamSetAttribute(at::cuda::getCurrentCUDAStream(), cudaStreamAttributeAccessPolicyWindow, &v);
  cudaCtxResetPersistingL2Cache();
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
}

int64_t max_persist_bytes() {
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxPersistingL2CacheSize, 0);
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_window", &set_window);
  m.def("clear_window", &clear_window);
  m.def("max_persist_bytes", &max_persist_bytes);
}
