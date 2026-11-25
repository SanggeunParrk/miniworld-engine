// Frozen generated source: d256_pool_checkpoint: saved_input_stats.SavedInputFront.
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
#include "tmn_kernels.cuh"
using C=tmn::K1Cfg<256,512,false,2,64,4,4,-1>;
extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB) void mw_d256_front_save_stats(__grid_constant__ const tmn::K1Params p){tmn::sm90::k1_body<C,true,1,true,true,1>(p);}