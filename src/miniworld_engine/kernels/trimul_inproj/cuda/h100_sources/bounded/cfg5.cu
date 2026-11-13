// SPDX-License-Identifier: Apache-2.0
// bounded contraction config 5: tile Shape<cute::_128,cute::_128,cute::_64>, cluster Shape<cute::_2,cute::_1,cute::_1>, pingpong
#include "gemm.cuh"
using mwe_cfg5_tile = cute::Shape<cute::_128,cute::_128,cute::_64>;
using mwe_cfg5_cluster = cute::Shape<cute::_2,cute::_1,cute::_1>;
MWE_BOUNDED_CONFIG(mwe_bounded_cfg5, mwe_cfg5_tile, mwe_cfg5_cluster, true)
