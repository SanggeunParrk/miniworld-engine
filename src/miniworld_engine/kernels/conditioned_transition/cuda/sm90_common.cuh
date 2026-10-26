// sm90_common.cuh -- what the AdaLN / ConditionedTransition sm_90a GEMM kernels share: the dtype of an activation operand, TMA tensor-map
// encoding, programmatic dependent launch (PDL) helpers and the launcher that sets its attribute.
//
// Every GEMM here is the structure of ``gemm_swiglu_sm90.cu`` (one producer warpgroup streaming 128-B-swizzled A / B slabs by TMA through an
// mbarrier ring, two consumer warpgroups of m64nN wgmma, a staged epilogue and TMA stores), with a different epilogue. bf16 operands use
// wgmma k16, fp32 operands the TF32 tensor cores (k8); both accumulate in fp32.
#pragma once
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "ct_wgmma.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace ctk {

template <bool F32> using T = std::conditional_t<F32, float, __nv_bfloat16>;

// ---------------------------------------------------------------------------------------------------- PDL
// A kernel launched with the programmatic-serialization attribute may start while its predecessor still runs; ``pdl_wait`` blocks until the
// predecessor has completed and its writes are visible. The predecessor calls ``pdl_trigger`` to let the dependent grid be scheduled early.
TMN_DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
TMN_DEVI void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

template <typename... KArgs, typename... Args>
void launch(void (*kernel)(KArgs...), unsigned grid, unsigned block, size_t smem, bool pdl, Args&&... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...) == cudaSuccess, "kernel launch failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------------- small device helpers
TMN_DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}
// round an fp32 to the TF32 grid, to nearest (ties away): the value the tensor cores then read exactly (a TMA-fed fp32 operand is truncated)
TMN_DEVI float tf32_rna(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
TMN_DEVI float sigmoid_exact(float a) { return rcpf(1.f + __expf(-a)); }   // ex2.approx + rcp.approx: ~1 ulp

// ---------------------------------------------------------------------------------------------------- host: TMA descriptors
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
inline EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p, "no TMA");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}
// a [rows, cols] row-major matrix (row stride ``rs`` elements), boxes of ``br`` rows x ``bc`` columns (bc * elem = 128 B), 128-B swizzle
inline CUtensorMap map2d(const void* p, bool f32, uint64_t rows, uint64_t cols, uint64_t rs, uint32_t bc, uint32_t br) {
  CUtensorMap map{};
  const cuuint64_t dims[2] = {cols, rows}, strides[1] = {rs * (f32 ? 4 : 2)};
  const cuuint32_t box[2] = {bc, br}, elem[2] = {1, 1};
  TORCH_CHECK(encoder()(&map, f32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), dims, strides,
                        box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "tensor map encode failed");
  return map;
}
inline int num_sms(const at::Tensor& t) { return at::cuda::getDeviceProperties(t.get_device())->multiProcessorCount; }

}  // namespace ctk

// ---------------------------------------------------------------------------------------------------- tracing (development: -DCT_TRACE)
// Per CTA: the SM clock at a few points of the kernel (slots 0..5, relative to the CTA's start in slot 0 by the reader), the global timer at its
// start (slot 6) and end (slot 7). Read back with ``trace()``.
#ifdef CT_TRACE
__device__ unsigned long long g_trace[512][16];
TMN_DEVI unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TRACE(slot) do { g_trace[blockIdx.x][slot] = clock64(); } while (0)
#define TRACE_T(slot) do { g_trace[blockIdx.x][slot] = gtimer(); } while (0)
inline at::Tensor trace_read() {
  at::Tensor t = at::empty({512, 16}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  cudaDeviceSynchronize();
  cudaMemcpyFromSymbol(t.data_ptr(), g_trace, sizeof(unsigned long long) * 512 * 16);
  return t;
}
#else
#define TRACE(slot) do {} while (0)
#define TRACE_T(slot) do {} while (0)
#endif
#ifdef CT_TRACE
#define TRACE_TILE(tl, k) do { if (blockIdx.x == 0 && (tl) < 12) g_trace[256 + (tl)][k] = clock64(); } while (0)
#else
#define TRACE_TILE(tl, k) do {} while (0)
#endif
