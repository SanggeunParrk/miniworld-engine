// D = 64 forward, 64-row tiles, THREE CTAs per SM: one consumer warpgroup per CTA (NWG = 1), x / xn single-buffered (NBUF = 1), 66 KB of
// shared memory each.  The per-tile chain (x load -> LayerNorm -> hidden chunks -> epilogue -> store) is latency bound; three resident CTAs
// per SM interleave three chains where the 128-row build runs one tile per CTA with two CTAs per SM.
#define NWG 1
#define NBUF 1
#define NCTA (3 * WIDE_SMS)
#define CTAS_PER_SM 3
#define FWD_SAVE 0
#define transition_fwd_fused wide_d64_fwd_small3_kernel
#include "kernels/transition_fwd_d64.cu"
#include "launch_util.cuh"

void wide_d64_fwd_small3_launch(const CUtensorMap& mx, const CUtensorMap& mwa, const CUtensorMap& mwb, const CUtensorMap& mwst,
                                const CUtensorMap& mout, const float* gamma, const float* beta, __nv_bfloat16* xn, __nv_bfloat16* out,
                                float* rstd, float* c1, int M, int tiles, float eps, int save, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_d64_fwd_small3_kernel), SMEM_BYTES, "d64 fwd small3"), true);
  (void)ready;
  wide_d64_fwd_small3_kernel<<<(tiles < NCTA ? tiles : NCTA), 128, SMEM_BYTES, stream>>>(mx, mwa, mwb, mwst, mout, gamma, beta, xn, out, rstd, c1, M, tiles, eps, save);
}
