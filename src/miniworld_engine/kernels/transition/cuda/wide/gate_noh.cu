// D >= 384 backward, gate stage WITHOUT the h store (MINIWORLD_TRANSITION_WIDE_SAVE_H=1: h comes from the forward): dh = dy Ws, [a|b] = xn [Wa;Wb]^T and the SwiGLU backward in one kernel, writing only
// [dA|dB] [M][2H]. 128 rows x 128 hidden tile, 32-wide K slabs, 4 stages, two-pass epilogue, slab released as its MMAs retire.
// See kernels/gate_gemm2.cu.
#define TBK 32
#define NSTAGE 4
#define STG_HALF 1
#define WAIT0 1
#define NO_H 1                                            // h is the forward's saved activation: not stored again
#define gate_gemm wide_gate_noh_kernel
#include "kernels/gate_gemm2.cu"
#include "launch_util.cuh"

void wide_gate_noh_launch(const CUtensorMap& mxn, const CUtensorMap& mdy, const CUtensorMap& mw1p, const CUtensorMap& mwst,
                      const CUtensorMap& mh, const CUtensorMap& mdab, int M, int K, int H, int ctas, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_gate_noh_kernel), SMEM_BYTES, "gate (no h)"), true);
  (void)ready;
  wide_gate_noh_kernel<<<ctas, 384, SMEM_BYTES, stream>>>(mxn, mdy, mw1p, mwst, mh, mdab, M, K, H);
}
