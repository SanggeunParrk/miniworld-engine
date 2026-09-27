// D >= 256 backward, gate stage: dh = dy Ws, [a|b] = xn [Wa;Wb]^T and the SwiGLU backward in one kernel, writing h [M][H] and
// [dA|dB] [M][2H]. 128 rows x 128 hidden tile, 32-wide K slabs, 4 stages, two-pass epilogue, slab released as its MMAs retire.
// See kernels/gate_gemm2.cu.
#define TBK 32
#define NSTAGE 4
#define STG_HALF 1
#define WAIT0 1
#define gate_gemm wide_gate_kernel
#include "kernels/gate_gemm2.cu"
#include "launch_util.cuh"

void wide_gate_launch(const CUtensorMap& mxn, const CUtensorMap& mdy, const CUtensorMap& mw1p, const CUtensorMap& mwst,
                      const CUtensorMap& mh, const CUtensorMap& mdab, int M, int K, int H, int ctas, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_gate_kernel), SMEM_BYTES, "gate"), true);
  (void)ready;
  wide_gate_kernel<<<ctas, 384, SMEM_BYTES, stream>>>(mxn, mdy, mw1p, mwst, mh, mdab, M, K, H);
}
