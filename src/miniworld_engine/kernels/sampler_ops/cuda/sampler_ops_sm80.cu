// sampler_ops_sm80.cu -- the A100 (sm_80) build of sampler_ops.cu: the same kernels, launched plainly (no programmatic dependent launch, which needs
// sm_90). sampler_ops.cu is compiled as it is (the H100 build's source stays byte-identical); this wrapper only turns its two PDL pieces into no-ops:
//   * the device side: ``asm volatile("griddepcontrol...")`` in pdl_wait / pdl_trigger becomes an empty asm statement (every header is included
//     first, so the macro reaches sampler_ops.cu's own two statements only);
//   * the host side: the launch attribute becomes cudaLaunchAttributeIgnore, so each kernel waits for the previous one by stream order.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <climits>
#include <cuda_bf16.h>

#define volatile(...) volatile("")
#define cudaLaunchAttributeProgrammaticStreamSerialization cudaLaunchAttributeIgnore
#include "sampler_ops.cu"
#undef volatile
#undef cudaLaunchAttributeProgrammaticStreamSerialization
