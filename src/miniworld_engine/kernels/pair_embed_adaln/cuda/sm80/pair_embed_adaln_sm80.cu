// pair_embed_adaln_sm80.cu -- the A100 (sm_80) build of sm90/pair_embed_adaln.cu: the same kernels and host entry points, the source compiled
// unchanged (the H100 build's file stays byte-identical); pe_sm80.cuh (pair_embed_geom's) supplies the sm_80 forms of its sm_90-only helpers
// (the table gradient's warpgroup product as mma.sync on the same swizzled shared-memory operands).
//
// The backward raises the dynamic shared-memory limit of both table-gradient kernels before picking one; the shared-memory variant's budget
// (4 x NT x 64 fp32 = 187 KiB for the 187-row table) is above sm_80's 163 KiB, and that kernel is not the launched one (impl 1, tbl4 at 89 KiB):
// its refused attribute call is cleared here instead of surfacing as the next launch check's error. A kernel launched past its limit still fails.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../../../pair_embed_geom/cuda/sm80/pe_sm80.cuh"

template <class K> cudaError_t pe80_func_set_attribute(K kernel, cudaFuncAttribute attr, int value) {
  const cudaError_t e = cudaFuncSetAttribute(kernel, attr, value);
  if (e != cudaSuccess) (void)cudaGetLastError();
  return e;
}
#define cudaFuncSetAttribute pe80_func_set_attribute
#include "../sm90/pair_embed_adaln.cu"
#undef cudaFuncSetAttribute
