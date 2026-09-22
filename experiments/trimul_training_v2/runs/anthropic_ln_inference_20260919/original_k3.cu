// SPDX-License-Identifier: Apache-2.0
// Unmodified Anthropic native v5 body and shipped C128/H256 BF16 configuration.
#include "tmn_kernels.cuh"
using Cfg=tmn::K3Cfg<128,256,0,2,64,4,1>;
extern "C" __global__ __launch_bounds__(384,1)
void infer_k3(__grid_constant__ const tmn::K3Params p){tmn::sm90::k3_body<Cfg,1>(p);}
