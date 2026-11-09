// rms90.cuh — the conditioning helpers of the forward row kernels (cond90, pre90, post90), on top of row90.cuh: AlphaFold 3's LayerNorms
// and sigmoids.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "row90.cuh"

namespace r90 {
// the statistics of the activation's AdaLN: LayerNorm
DEVI void norm_stats(const Tile& c, float eps, float& m0, float& r0, float& m1, float& r1) {
  ln_stats(c, eps, m0, r0, m1, r1);
}
// a conditioning table entry from its product and bias: sigmoid(v + b)
DEVI float cond_act(float v) {
  return sigm(v);
}
}  // namespace r90
