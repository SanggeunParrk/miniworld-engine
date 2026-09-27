// D >= 384 forward, first half: LayerNorm inside the SwiGLU expand GEMM (h = silu(xn Wa^T) * xn Wb^T), xn / rstd / c1 saved
// behind a runtime flag. See kernels/ln_swiglu_gemm.cu. KD is the channel width.
#define KD WIDE_D
#define SWP 1                                             // software-pipelined epilogue: -6 % (D384) / -8 % (D512) at fixed clocks
#define ln_swiglu_gemm wide_lnsg_kernel
#include "kernels/ln_swiglu_gemm.cu"
#include "launch_util.cuh"

void wide_lnsg_launch(const CUtensorMap& mx, const CUtensorMap& mw1p, const CUtensorMap& mh, const float* gamma, const float* beta,
                      __nv_bfloat16* xn, float* rstd, float* c1, int M, int H, float eps, int save, int ctas, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_lnsg_kernel), SMEM_BYTES, "ln_swiglu_gemm"), true);
  (void)ready;
  wide_lnsg_kernel<<<ctas, 384, SMEM_BYTES, stream>>>(mx, mw1p, mh, gamma, beta, xn, rstd, c1, M, H, eps, save);
}
