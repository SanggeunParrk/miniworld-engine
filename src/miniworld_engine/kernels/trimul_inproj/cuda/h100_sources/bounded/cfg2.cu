// SPDX-License-Identifier: Apache-2.0
// bounded contraction config 2: tile Shape<cute::_128,cute::_256,cute::_64>, cluster Shape<cute::_1,cute::_1,cute::_1>, cooperative
#include "gemm.cuh"
using mwe_cfg2_tile = cute::Shape<cute::_128,cute::_256,cute::_64>;
using mwe_cfg2_cluster = cute::Shape<cute::_1,cute::_1,cute::_1>;
MWE_BOUNDED_CONFIG(mwe_bounded_cfg2, mwe_cfg2_tile, mwe_cfg2_cluster, false)
