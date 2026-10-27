// D >= 384 forward, first half: LayerNorm inside the SwiGLU expand GEMM (h = silu(xn Wa^T) * xn Wb^T), xn / rstd / c1 saved
// behind a runtime flag. See kernels/ln_swiglu_gemm.cu. KD is the channel width.
#if WIDE_D > 512                                           // D768: the headers the shared pipelines expect at global scope (ln_swiglu_gemm.cu brings them otherwise)
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma_n.cuh"
#include "launch_util.cuh"
#endif
#if WIDE_D <= 512                                          // D768: the resident LN tile does not fit; only the split forward below exists
#define KD WIDE_D
#define SWP 1                                             // software-pipelined epilogue: -6 % (D384) / -8 % (D512) at fixed clocks
#define ln_swiglu_gemm wide_lnsg_kernel
#include "kernels/ln_swiglu_gemm.cu"
#include "launch_util.cuh"

void wide_lnsg_launch(const CUtensorMap& mx, const CUtensorMap& mw1p, const CUtensorMap& mh, const float* gamma, const float* beta,
                      __nv_bfloat16* xn, float* rstd, float* c1, int M, int H, float eps, int save, int ctas, cudaStream_t stream
#if MW_SAVE_AB
                      ,__nv_bfloat16* saved_a,__nv_bfloat16* saved_b
#endif
                      ) {
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(wide_lnsg_kernel), SMEM_BYTES, "ln_swiglu_gemm"), true);
  (void)ready;
  wide_lnsg_kernel<<<ctas, 384, SMEM_BYTES, stream>>>(mx, mw1p, mh, gamma, beta, xn, rstd, c1, M, H, eps, save
#if MW_SAVE_AB
    ,saved_a,saved_b
#endif
    );
}
#endif  // WIDE_D <= 512

#if MW_SPLIT_FWD
// B200 D512 boundary: LN materializes xn; expand writes h/a/b with TMA stores.
// Namespace isolates this second instance of the shared WGMMA pipeline.
namespace mw_split {
#undef WAIT0
#define WAIT0 1
#if WIDE_D == 512                                         // 64-wide K slabs, 3 stages: -1.6 % of the D512 training step against 32 x 4 (384 x 384 pair)
#define MW_EXP_TBK 64
#define MW_EXP_NSTAGE 3
#else
#define MW_EXP_TBK 32
#define MW_EXP_NSTAGE 4
#endif
#define MW_EXP_STGHALF 1
#define TBK MW_EXP_TBK
#define NSTAGE MW_EXP_NSTAGE
#define STG_HALF MW_EXP_STGHALF
#define REG_C 200
#define MW_EXPAND_ONLY 1
#define gate_gemm wide_expand_saved_kernel
#define tma_store_2d split_tma_store_2d
#include "kernels/gate_gemm2.cu"
#undef tma_store_2d
}
void wide_expand_saved_launch(const CUtensorMap& mxn,const CUtensorMap& mw1,
                             const CUtensorMap& mh,const CUtensorMap& ma,const CUtensorMap& mb,
                             int M,int D,int H,int ctas,cudaStream_t stream){
  static const bool ready=(wide_smem_optin(reinterpret_cast<const void*>(mw_split::wide_expand_saved_kernel),
                                          mw_split::SMEM_BYTES,"expand saved"),true);
  (void)ready;
  mw_split::wide_expand_saved_kernel<<<ctas,384,mw_split::SMEM_BYTES,stream>>>(mxn,mxn,mw1,mw1,mh,mh,M,D,H,ma,mb);
}
#endif
