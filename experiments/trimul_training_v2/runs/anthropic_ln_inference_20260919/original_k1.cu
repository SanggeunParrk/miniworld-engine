// SPDX-License-Identifier: Apache-2.0
// Unmodified Anthropic native v5 body and shipped C128/H256 BF16 configuration.
#include "tmn_kernels.cuh"
using Cfg=tmn::K1Cfg<128,256,false,2,64,8,2>;
extern "C" __global__ __launch_bounds__(Cfg::NTHR,Cfg::MINB)
void infer_k1(__grid_constant__ const tmn::K1Params p){tmn::sm90::k1_body<Cfg,true,1,false,false>(p);}
