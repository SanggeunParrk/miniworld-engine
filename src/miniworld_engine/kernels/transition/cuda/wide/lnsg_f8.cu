// OPT-IN FP8 (docs/fp8.md), D 384 / 512 forward, first half: ln_swiglu_gemm with h written as e4m3. See kernels/ln_swiglu_gemm_f8.cu.
#define KD WIDE_D
#define SWP 1
#ifndef F8G1
#define F8G1 0
#endif
#define ln_swiglu_gemm_f8 wide_lnsg_f8_kernel
#include "kernels/ln_swiglu_gemm_f8.cu"
#include "launch_util.cuh"

void wide_lnsg_f8_launch(const CUtensorMap& mx, const CUtensorMap& mw1p, const CUtensorMap& mh8, const float* gamma, const float* beta,
                         __nv_bfloat16* xn, float* rstd, float* c1, int M, int H, float eps, const float* scales, unsigned int* peak, int ctas,
                         cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_lnsg_f8_kernel), SMEM_BYTES, "ln_swiglu_gemm_f8"), true);
  (void)ready;
  wide_lnsg_f8_kernel<<<ctas, 384, SMEM_BYTES, stream>>>(mx, mw1p, mh8, gamma, beta, xn, rstd, c1, M, H, eps, 0, scales, peak);
}
