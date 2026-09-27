// D >= 384 forward, second half: out = h Ws^T + x. See kernels/squeeze_gemm.cu. Output tile width 256 where D allows it.
#if WIDE_D % 256 == 0                                    // a literal: squeeze_gemm token-pastes it into mma_n<SQ_BN>
#define SQ_BN 256
#else
#define SQ_BN 192
#endif
#define WAIT0 1                                           // slab released as its MMAs retire: -1.3 % / -2.7 %
#define squeeze_gemm wide_squeeze_kernel
#include "kernels/squeeze_gemm.cu"
#include "launch_util.cuh"

int wide_squeeze_bn() { return SQ_BN; }
void wide_squeeze_launch(const CUtensorMap& mh, const CUtensorMap& mws, const CUtensorMap& mx, const CUtensorMap& mout,
                         int M, int H, int D, int ctas, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_squeeze_kernel), SMEM_BYTES, "squeeze_gemm"), true);
  (void)ready;
  wide_squeeze_kernel<<<ctas, 384, SMEM_BYTES, stream>>>(mh, mws, mx, mout, M, H, D);
}
