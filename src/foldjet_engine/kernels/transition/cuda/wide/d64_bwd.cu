// D = 64 backward: one fused two-role kernel (DW_REPL replicas of each hidden slice in the weight role, the rest input CTAs) and
// the partial-sum reduction. See kernels/transition_bwd_d64.cu.
// Weight-role CTAs: 4 slices x 16 for the AF3 Transition (n = 4), 2 slices x 28 for n = 2 (measured L384 / L768 backward, repl 24: 126 / 398 us, 32: 123 / 437 us, 40: 166 / 568 us); with n = 2 the input role has half the hidden chunks to stream
// per tile, so it takes fewer CTAs and the weight role more (WIDE_REPL, MINIWORLD_TRANSITION_WIDE_REPL override).
#ifndef WIDE_N
#define WIDE_N 4
#endif
#ifndef WIDE_REPL
#define WIDE_REPL (WIDE_N == 4 ? 16 : 28)
#endif
#define NCTA WIDE_SMS
#ifndef MW_D64_DW_REPL
#define MW_D64_DW_REPL 16
#endif
#define DW_REPL MW_D64_DW_REPL
#define transition_bwd_fused wide_d64_bwd_kernel
#define reduce_partials wide_d64_reduce_kernel
#include "kernels/transition_bwd_d64.cu"
#include "launch_util.cuh"

int wide_d64_bwd_ndw() { return NDW; }
int wide_d64_bwd_ndx() { return NDX; }
void wide_d64_bwd_launch(const CUtensorMap& mdy, const CUtensorMap& mxn, const CUtensorMap& mx, const CUtensorMap& mws,
                         const CUtensorMap& mwa, const CUtensorMap& mwb, const float* rstd, const float* c1, const float* gamma,
                         __nv_bfloat16* dx, float* dgam, float* dbeta, float* partw, float* dgbw, void* dWa,
                         void* dWb, void* dWs, int M, int tiles, bool fp32_dw, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_d64_bwd_kernel), SMEM_BYTES, "d64 bwd"), true);
  (void)ready;
  wide_d64_bwd_kernel<<<NCTA, 256, SMEM_BYTES, stream>>>(mdy, mxn, mx, mws, mwa, mwb, rstd, c1, gamma, dx, dgam, dbeta, partw, dgbw, M, tiles);
#if MW_D64_AFFINE_PARALLEL
  const int n = 3 * NSL * HS * D_ + ((D_+31)/32)*256;
#else
  const int n = 3 * NSL * HS * D_ + 2 * D_;
#endif
  if(fp32_dw) wide_d64_reduce_kernel<float><<<(n + 255) / 256, 256, 0, stream>>>(partw, (float*)dWa, (float*)dWb, (float*)dWs, dgbw, dgam, dbeta);
  else wide_d64_reduce_kernel<__nv_bfloat16><<<(n + 255) / 256, 256, 0, stream>>>(partw, (__nv_bfloat16*)dWa, (__nv_bfloat16*)dWb, (__nv_bfloat16*)dWs, dgbw, dgam, dbeta);
}
