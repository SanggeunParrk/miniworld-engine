// D = 64 forward, 64-row tiles: the same kernel as d64_fwd.cu with ONE consumer warpgroup per CTA (NWG = 1), so a call of 96 .. 260
// row tiles of 64 rows fills the SMs in one wave instead of leaving a 128-row tile per CTA to run its whole chain serially.
// Two CTAs per SM (82 KB of shared memory each).
#define NWG 1
#define NCTA (2 * WIDE_SMS)
#define CTAS_PER_SM 2
#define FWD_SAVE 0                                         // inference: xn / rstd / c1 behind the RUNTIME save flag
#define transition_fwd_fused wide_d64_fwd_small_kernel
#include "kernels/transition_fwd_d64.cu"
#include "launch_util.cuh"

void wide_d64_fwd_small_launch(const CUtensorMap& mx, const CUtensorMap& mwa, const CUtensorMap& mwb, const CUtensorMap& mwst,
                               const CUtensorMap& mout, const float* gamma, const float* beta, __nv_bfloat16* xn, __nv_bfloat16* out,
                               float* rstd, float* c1, int M, int tiles, float eps, int save, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_d64_fwd_small_kernel), SMEM_BYTES, "d64 fwd small"), true);
  (void)ready;
  wide_d64_fwd_small_kernel<<<(tiles < NCTA ? tiles : NCTA), 128, SMEM_BYTES, stream>>>(mx, mwa, mwb, mwst, mout, gamma, beta, xn, out, rstd, c1, M, tiles, eps, save);
}
