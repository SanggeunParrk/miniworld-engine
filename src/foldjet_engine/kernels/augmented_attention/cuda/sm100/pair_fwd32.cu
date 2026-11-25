// pair_fwd32.cu -- AttentionPairBias fp32 pair -> bias on sm_100a (tcgen05 kind::tf32), inference: the cubin build of
// ../apb/apb_pair_fwd32_sm100.cuh for one (heads, d_pair): -DPF_NH -DPF_DPW (``sm100.apb_pair_fwd32``). Block 192 threads, one per SM.
// SPDX-License-Identifier: Apache-2.0
#include "../apb/apb_pair_fwd32_sm100.cuh"
#ifndef PF_NH
#define PF_NH 8
#endif
#ifndef PF_DPW
#define PF_DPW 384
#endif
extern "C" __global__ void __launch_bounds__(192, 1) apb_pair_fwd32_sm100(const __grid_constant__ CUtensorMap tmz, const float* __restrict__ WF,
    const bool* __restrict__ MASK, float* __restrict__ OUT, int L, float eps, float neg, int perm) {
  pf32::pair_fwd32_body<PF_NH, PF_DPW>(tmz, WF, MASK, OUT, L, eps, neg, perm);
}
