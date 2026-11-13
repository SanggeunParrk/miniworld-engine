// SPDX-License-Identifier: Apache-2.0
// bounded contraction config 0: tile Shape<cute::_128,cute::_256,cute::_64>, cluster Shape<cute::_2,cute::_1,cute::_1>, cooperative
#include "gemm.cuh"
using mwe_cfg0_tile = cute::Shape<cute::_128,cute::_256,cute::_64>;
using mwe_cfg0_cluster = cute::Shape<cute::_2,cute::_1,cute::_1>;
MWE_BOUNDED_CONFIG(mwe_bounded_cfg0, mwe_cfg0_tile, mwe_cfg0_cluster, false)
