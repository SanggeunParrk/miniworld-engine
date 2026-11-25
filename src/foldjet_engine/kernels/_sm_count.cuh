// Device facts by the cudaDeviceGetAttribute query.  Never read cudaDeviceProp through at::cuda::getCurrentDeviceProperties() / cudaGetDeviceProperties() in an extension: libtorch fills the struct by
// the layout of the CUDA it was built with (12.9 here), the extension reads it by the layout of the toolkit nvcc compiles against (12.8 / 13.x: fields were removed and moved), so a field read lands
// on the wrong member (multiProcessorCount read as 1 on the RTX 5090 server).  The attribute query has no struct: the answer is the same for every toolkit.
#pragma once
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

namespace mw {

inline int device_attr(cudaDeviceAttr attr, int dev) {
  int v = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&v, attr, dev));
  return v;
}

inline int current_dev() {
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  return dev;
}

// streaming multiprocessors of device ``dev`` (the current device when dev < 0); read once per device
inline int sm_count(int dev = -1) {
  static int cache[64] = {0};
  if (dev < 0) dev = current_dev();
  if (dev >= 64) return device_attr(cudaDevAttrMultiProcessorCount, dev);
  if (cache[dev] == 0) cache[dev] = device_attr(cudaDevAttrMultiProcessorCount, dev);
  return cache[dev];
}

inline int max_threads_per_sm(int dev = -1) { return device_attr(cudaDevAttrMaxThreadsPerMultiProcessor, dev < 0 ? current_dev() : dev); }
inline bool cooperative_launch(int dev = -1) { return device_attr(cudaDevAttrCooperativeLaunch, dev < 0 ? current_dev() : dev) != 0; }

}  // namespace mw
