// tmap.h -- host-side CUtensorMap encoding through the runtime's driver entry point (no -lcuda link needed).
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

namespace tmap {

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

inline EncodeFn encoder() {
  static EncodeFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p,
                "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<EncodeFn>(p);
  }
  return fn;
}

// dims / box innermost-first; strides_bytes for dims 1.. (innermost stride = element size).
inline CUtensorMap make(void* base, CUtensorMapDataType dt, std::vector<uint64_t> dims, std::vector<uint64_t> strides_bytes,
                        std::vector<uint32_t> box, CUtensorMapSwizzle swz,
                        CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_256B) {
  CUtensorMap m;
  std::vector<uint32_t> es(dims.size(), 1);
  CUresult r = encoder()(&m, dt, (cuuint32_t)dims.size(), base, dims.data(), strides_bytes.data(), box.data(), es.data(),
                         CU_TENSOR_MAP_INTERLEAVE_NONE, swz, l2, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

}  // namespace tmap
