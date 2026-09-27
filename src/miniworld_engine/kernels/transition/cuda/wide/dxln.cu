// D = 256 backward tail: d_xn = [dA|dB] [Wa;Wb] with the LayerNorm backward and the residual in the epilogue (d_xn stays fp32).
// Both consumer warpgroups on 64 rows, 128 columns each. See kernels/dxn_lnbwd.cu.
#define DW WIDE_D
#define COLS 2
#define TBK 64
#define NSTAGE 4
#define WAIT0 1
#define STGDX 1                                           // dx staged with stmatrix + TMA store: -1.5 %
#define dxn_lnbwd wide_dxln_kernel
#include "kernels/dxn_lnbwd.cu"
#include "launch_util.cuh"

void wide_dxln_launch(const CUtensorMap& mdab, const CUtensorMap& mwabt, const CUtensorMap& mx, const CUtensorMap& mdx, const __nv_bfloat16* x, const __nv_bfloat16* dy,
                      __nv_bfloat16* dx, const float* gamma, const float* rstd, const float* c1, float* pdg, float* pdb,
                      int M, int ctas, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_dxln_kernel), SMEM_BYTES, "dxn_lnbwd"), true);
  (void)ready;
  wide_dxln_kernel<<<ctas, 384, SMEM_BYTES, stream>>>(mdab, mwabt, mx, mdx, x, dy, dx, gamma, rstd, c1, pdg, pdb, M);
}
