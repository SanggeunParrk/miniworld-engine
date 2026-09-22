// D = 64 forward: one fused kernel, two CTAs per SM (112 KB of shared memory each). See kernels/transition_fwd_d64.cu.
#define NCTA (2 * WIDE_SMS)
#define CTAS_PER_SM 2
#define FWD_SAVE 0                                         // the xn / rstd / c1 stores behind the RUNTIME save flag (no_grad passes placeholders)
#define transition_fwd_fused wide_d64_fwd_kernel
#include "kernels/transition_fwd_d64.cu"
#include "launch_util.cuh"

int wide_d64_fwd_ctas() { return NCTA; }
void wide_d64_fwd_launch(const CUtensorMap& mx, const CUtensorMap& mwa, const CUtensorMap& mwb, const CUtensorMap& mwst,
                         const CUtensorMap& mout, const float* gamma, const float* beta, __nv_bfloat16* xn, __nv_bfloat16* out,
                         float* rstd, float* c1, int M, int tiles, float eps, int save, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_d64_fwd_kernel), SMEM_BYTES, "d64 fwd"), true);
  (void)ready;
  wide_d64_fwd_kernel<<<NCTA, 256, SMEM_BYTES, stream>>>(mx, mwa, mwb, mwst, mout, gamma, beta, xn, out, rstd, c1, M, tiles, eps, save);
}
