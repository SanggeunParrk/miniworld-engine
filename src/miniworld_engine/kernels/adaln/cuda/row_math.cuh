// row_math.cuh -- the elementary pieces every AdaLN-shaped row pass of the engine shares (one definition, behaviour of each user unchanged):
//   warp_sum    xor-butterfly sum over the 32 lanes (all lanes get the total)
//   sigmoid     1 / (1 + exp(-v)) with __expf: the exact-division form (the gate of the row kernels)
//   sigmoid_fast  ex2.approx + rcp.approx form (the GEMM epilogues of adaln_sm90.cu / ct_sm90.cu: ~1 ulp, no division)
//   modulate    y = gate * ((x - mean) * rstd) + shift: the adaptive LayerNorm output (adaln_rows.cu ``adaln_fwd``, adaln_sm90.cu epilogue)
// Users: adaln/cuda/adaln_rows.cu, adaln/cuda/adaln_sm90.cu, conditioned_transition/cuda/token_dit_common.cuh (hence the token DiT, bias-only DiT,
// DiT-variant and AugmentedAttention row kernels). The token DiT's own ``adaln_a`` pass computes gate * d * rstd + shift (the product in another
// order: not bit-identical to ``modulate``), so it keeps its expression and shares only warp_sum and sigmoid.
#pragma once

namespace rowmath {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float sigmoid(float v) { return 1.f / (1.f + __expf(-v)); }
__device__ __forceinline__ float sigmoid_fast(float v) {
  float e, r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(-v * 1.4426950408889634f));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(1.f + e));
  return r;
}
__device__ __forceinline__ float modulate(float x, float mean, float rstd, float gate, float shift) { return gate * ((x - mean) * rstd) + shift; }

}  // namespace rowmath
