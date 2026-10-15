// Programmatic dependent launch (Hopper): a kernel launched with ``pdl_launch`` may be scheduled while the previous kernel of its
// stream is still running (under CUDA-graph capture too); it does its prologue -- weight copies into shared memory -- and then calls
// ``pdl_gate()``, which blocks until the previous grid has completed and its writes are visible. Everything a predecessor
// produces must be read (and everything it reads must be overwritten) AFTER the gate; what the prologue touches must be data no
// kernel of the stream writes (parameters, saved weights). A kernel launched without the attribute (``pdl`` false) gets the gate as a
// no-op and the ordinary stream order.
#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ void pdl_gate() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

template <typename... KArgs, typename... Args>
inline cudaError_t pdl_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream, bool pdl, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
}
