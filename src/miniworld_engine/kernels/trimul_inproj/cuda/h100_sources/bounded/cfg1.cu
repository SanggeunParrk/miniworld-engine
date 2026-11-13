// SPDX-License-Identifier: Apache-2.0
// bounded contraction config 1: tile Shape<cute::_128,cute::_256,cute::_64>, cluster Shape<cute::_1,cute::_2,cute::_1>, cooperative
#include "gemm.cuh"
using mwe_cfg1_tile = cute::Shape<cute::_128,cute::_256,cute::_64>;
using mwe_cfg1_cluster = cute::Shape<cute::_1,cute::_2,cute::_1>;
MWE_BOUNDED_CONFIG(mwe_bounded_cfg1, mwe_cfg1_tile, mwe_cfg1_cluster, false)
