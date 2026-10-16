// D = 64 forward, 64-row tiles, 3 CTAs per SM, LayerNorm in the wgmma A-fragment layout (LNFRAG = 1): xn never touches shared memory.
#define NWG 1
#define NBUF 1
#define LNFRAG 1
#define NCTA (3 * WIDE_SMS)
#define CTAS_PER_SM 3
#define FWD_SAVE 0
#define transition_fwd_fused wide_d64_fwd_small_f3_kernel
#include "kernels/transition_fwd_d64.cu"
#include "launch_util.cuh"

void wide_d64_fwd_small_f3_launch(const CUtensorMap& mx, const CUtensorMap& mwa, const CUtensorMap& mwb, const CUtensorMap& mwst,
                                 const CUtensorMap& mout, const float* gamma, const float* beta, __nv_bfloat16* xn, __nv_bfloat16* out,
                                 float* rstd, float* c1, int M, int tiles, float eps, int save, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_d64_fwd_small_f3_kernel), SMEM_BYTES, "d64 fwd small f3"), true);
  (void)ready;
  wide_d64_fwd_small_f3_kernel<<<(tiles < NCTA ? tiles : NCTA), 128, SMEM_BYTES, stream>>>(mx, mwa, mwb, mwst, mout, gamma, beta, xn, out, rstd, c1, M, tiles, eps, save);
}
