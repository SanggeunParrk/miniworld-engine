// SPDX-License-Identifier: Apache-2.0
// bounded contraction config 3: tile Shape<cute::_128,cute::_192,cute::_64>, cluster Shape<cute::_2,cute::_1,cute::_1>, pingpong
#include "gemm.cuh"
using mwe_cfg3_tile = cute::Shape<cute::_128,cute::_192,cute::_64>;
using mwe_cfg3_cluster = cute::Shape<cute::_2,cute::_1,cute::_1>;
MWE_BOUNDED_CONFIG(mwe_bounded_cfg3, mwe_cfg3_tile, mwe_cfg3_cluster, true)
