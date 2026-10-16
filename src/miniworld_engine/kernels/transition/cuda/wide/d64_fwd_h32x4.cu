// D = 64 forward, 64-row tiles, hidden chunks of 32 units (HSZ = 32): the AB accumulator is m64n64 (32 registers instead of 64) and a weight slot
// is 12 KB instead of 24 KB, so 4 CTAs of one consumer warpgroup fit an SM (42 KB of shared memory, <= 128 registers per thread).
#define NWG 1
#define NBUF 1
#define HSZ 32
#define NCTA (4 * WIDE_SMS)
#define CTAS_PER_SM 4
#define FWD_SAVE 0
#define transition_fwd_fused wide_d64_fwd_h32x4_kernel
#include "kernels/transition_fwd_d64.cu"
#include "launch_util.cuh"

void wide_d64_fwd_h32x4_launch(const CUtensorMap& mx, const CUtensorMap& mwa, const CUtensorMap& mwb, const CUtensorMap& mwst,
                                 const CUtensorMap& mout, const float* gamma, const float* beta, __nv_bfloat16* xn, __nv_bfloat16* out,
                                 float* rstd, float* c1, int M, int tiles, float eps, int save, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_d64_fwd_h32x4_kernel), SMEM_BYTES, "d64 fwd h32x4"), true);
  (void)ready;
  wide_d64_fwd_h32x4_kernel<<<(tiles < NCTA ? tiles : NCTA), 128, SMEM_BYTES, stream>>>(mx, mwa, mwb, mwst, mout, gamma, beta, xn, out, rstd, c1, M, tiles, eps, save);
}
