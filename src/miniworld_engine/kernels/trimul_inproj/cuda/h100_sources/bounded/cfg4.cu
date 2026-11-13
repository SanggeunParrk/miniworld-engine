// SPDX-License-Identifier: Apache-2.0
// bounded contraction config 4: tile Shape<cute::_128,cute::_128,cute::_64>, cluster Shape<cute::_1,cute::_2,cute::_1>, pingpong
#include "gemm.cuh"
using mwe_cfg4_tile = cute::Shape<cute::_128,cute::_128,cute::_64>;
using mwe_cfg4_cluster = cute::Shape<cute::_1,cute::_2,cute::_1>;
MWE_BOUNDED_CONFIG(mwe_bounded_cfg4, mwe_cfg4_tile, mwe_cfg4_cluster, true)
