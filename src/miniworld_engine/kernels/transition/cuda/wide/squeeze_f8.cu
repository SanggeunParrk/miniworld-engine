// OPT-IN FP8 (docs/fp8.md), D 384 / 512 forward, second half: out = (h8 Ws8^T) co + x. See kernels/squeeze_gemm_f8.cu.
#if WIDE_D % 256 == 0
#define SQ_BN 256
#else
#define SQ_BN 192
#endif
#define WAIT0 1
#define squeeze_gemm_f8 wide_squeeze_f8_kernel
#include "kernels/squeeze_gemm_f8.cu"
#include "launch_util.cuh"

int wide_squeeze_f8_bn() { return SQ_BN; }
void wide_squeeze_f8_launch(const CUtensorMap& mh8, const CUtensorMap& mws8, const CUtensorMap& mx, const CUtensorMap& mout, int M, int H, int D,
                            const float* scales, int ctas, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_squeeze_f8_kernel), SMEM_BYTES, "squeeze_gemm_f8"), true);
  (void)ready;
  wide_squeeze_f8_kernel<<<ctas, 384, SMEM_BYTES, stream>>>(mh8, mws8, mx, mout, M, H, D, scales);
}
