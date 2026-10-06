// D >= 384 backward, gate stage WITHOUT the h store (settings.transition_wide_save_h=1: h comes from the forward): dh = dy Ws, [a|b] = xn [Wa;Wb]^T and the SwiGLU backward in one kernel, writing only
// [dA|dB] [M][2H]. 128 rows x 128 hidden tile, 32-wide K slabs, 4 stages, two-pass epilogue, slab released as its MMAs retire.
// See kernels/gate_gemm2.cu.
#if MW_GATE_SAVED_TMA && MW_SAVE_AB
#define TBK 32
#define NSTAGE 3
#define STG_HALF 1
#elif MW_GATE_FULL_STAGE && !MW_SAVE_AB
#define TBK 32
#define NSTAGE 3
#define STG_HALF 0
#elif MW_GATE_K64 && !MW_SAVE_AB
#define TBK 64
#define NSTAGE 2
#else
#define TBK 32
#define NSTAGE 4
#endif
#ifndef STG_HALF
#define STG_HALF 1
#endif
#if MW_GATE_K64 == 2 && !MW_SAVE_AB
#define WAIT0 0
#else
#define WAIT0 1
#endif
#if MW_GATE_MULTICAST && !MW_SAVE_AB
#define MCW 1
#endif
#define NO_H 1                                            // h is the forward's saved activation: not stored again
#if MW_SAVE_AB
#define MW_GATE_SAVED 1
#define REG_C 112
#define gate_gemm wide_gate_saved_kernel
#else
#define gate_gemm wide_gate_noh_kernel
#endif
#include "kernels/gate_gemm2.cu"
#include "launch_util.cuh"

#if MW_SAVE_AB
void wide_gate_saved_launch(const CUtensorMap& mdy,const CUtensorMap& mwst,const CUtensorMap& mdab,
                           const CUtensorMap& ma,const CUtensorMap& mb,
                           const __nv_bfloat16* a,const __nv_bfloat16* b,int M,int K,int H,int ctas,cudaStream_t stream){
  static const bool ready=(wide_smem_optin(reinterpret_cast<const void*>(wide_gate_saved_kernel),SMEM_BYTES,"saved gate"),true);
  (void)ready;
  wide_gate_saved_kernel<<<ctas,384,SMEM_BYTES,stream>>>(ma,mdy,mb,mwst,mdab,mdab,M,K,H,a,b);
}
#else
void wide_gate_noh_launch(const CUtensorMap& mxn, const CUtensorMap& mdy, const CUtensorMap& mw1p, const CUtensorMap& mwst,
                      const CUtensorMap& mh, const CUtensorMap& mdab, int M, int K, int H, int ctas, cudaStream_t stream) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_gate_noh_kernel), SMEM_BYTES, "gate (no h)"), true);
  (void)ready;
  wide_gate_noh_kernel<<<ctas, 384, SMEM_BYTES, stream>>>(mxn, mdy, mw1p, mwst, mh, mdab, M, K, H);
}
#endif
