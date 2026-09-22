// D = 256 forward: one fused kernel (producer warpgroup + two consumer warpgroups). See kernels/transition_fwd_d256.cu.
#define NCTA WIDE_SMS
#define PP 1                                              // ping-pong the two consumer warpgroups: -2.7 %
#define transition_fwd_fused wide_d256_fwd_kernel
#include "kernels/transition_fwd_d256.cu"
#include "launch_util.cuh"

void wide_d256_fwd_launch(const CUtensorMap& mx, const CUtensorMap& mwa, const CUtensorMap& mwb, const CUtensorMap& mwst,
                          const CUtensorMap& mout, const float* gamma, const float* beta, __nv_bfloat16* xn, __nv_bfloat16* out,
                          float* rstd, float* c1, int M, int tiles, float eps, int save, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_d256_fwd_kernel), SMEM_BYTES, "d256 fwd"), true);
  (void)ready;
  wide_d256_fwd_kernel<<<NCTA, 384, SMEM_BYTES, stream>>>(mx, mwa, mwb, mwst, mout, gamma, beta, xn, out, rstd, c1, M, tiles, eps, save);
}
