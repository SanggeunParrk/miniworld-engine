// rms90.cuh — the conditioning variants of the forward row kernels (cond90, pre90, post90), on top of row90.cuh.
//
// -DRMS=1: ESMFold2's conditioning. Every AdaLN normalises with an affine-free RMSNorm (``norm_stats``: mean 0), the conditioning tables are
// products of silu(c) (no LayerNorm of it), the scales are (1 + s) and the output gates raw (``cond_act``: cond90 adds the bias and applies
// no sigmoid; pre90 likewise for the K / V scale). Otherwise AlphaFold 3's: LayerNorms and sigmoids.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "row90.cuh"

#ifndef RMS
#define RMS 0
#endif

namespace r90 {
// affine-free RMSNorm statistics in the LayerNorm's form (mean 0)
DEVI void rms_stats(const Tile& c, float eps, float& m0, float& r0, float& m1, float& r1) {
  float v0 = 0.f, v1 = 0.f;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    v0 += c.f[nt][0] * c.f[nt][0] + c.f[nt][1] * c.f[nt][1];
    v1 += c.f[nt][2] * c.f[nt][2] + c.f[nt][3] * c.f[nt][3];
  }
  m0 = m1 = 0.f;
  r0 = rsqrtf(quad_sum(v0) * (1.f / 128.f) + eps);
  r1 = rsqrtf(quad_sum(v1) * (1.f / 128.f) + eps);
}
// the statistics of the activation's AdaLN: LayerNorm, or (RMS) RMSNorm
DEVI void norm_stats(const Tile& c, float eps, float& m0, float& r0, float& m1, float& r1) {
#if RMS
  rms_stats(c, eps, m0, r0, m1, r1);
#else
  ln_stats(c, eps, m0, r0, m1, r1);
#endif
}
// a conditioning table entry from its product and bias: sigmoid(v + b), or (RMS) v + b
DEVI float cond_act(float v) {
#if RMS
  return v;
#else
  return sigm(v);
#endif
}
}  // namespace r90
