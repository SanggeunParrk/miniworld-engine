// D = 64 backward: one fused two-role kernel (64 weight CTAs = 4 hidden slices x DW_REPL 16, the rest input CTAs) and the
// partial-sum reduction. See kernels/transition_bwd_d64.cu.
#define NCTA WIDE_SMS
#define DW_REPL 16
#define transition_bwd_fused wide_d64_bwd_kernel
#define reduce_partials wide_d64_reduce_kernel
#include "kernels/transition_bwd_d64.cu"
#include "launch_util.cuh"

int wide_d64_bwd_ndw() { return NDW; }
int wide_d64_bwd_ndx() { return NDX; }
void wide_d64_bwd_launch(const CUtensorMap& mdy, const CUtensorMap& mxn, const CUtensorMap& mx, const CUtensorMap& mws,
                         const CUtensorMap& mwa, const CUtensorMap& mwb, const float* rstd, const float* c1, const float* gamma,
                         __nv_bfloat16* dx, float* dgam, float* dbeta, float* partw, float* dgbw, __nv_bfloat16* dWa,
                         __nv_bfloat16* dWb, __nv_bfloat16* dWs, int M, int tiles, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_d64_bwd_kernel), SMEM_BYTES, "d64 bwd"), true);
  (void)ready;
  wide_d64_bwd_kernel<<<NCTA, 256, SMEM_BYTES, stream>>>(mdy, mxn, mx, mws, mwa, mwb, rstd, c1, gamma, dx, dgam, dbeta, partw, dgbw, M, tiles);
  const int n = 3 * NSL * HS * D_ + 2 * D_;
  wide_d64_reduce_kernel<<<(n + 255) / 256, 256, 0, stream>>>(partw, dWa, dWb, dWs, dgbw, dgam, dbeta);
}
