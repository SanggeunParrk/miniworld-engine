// pair_fwd.cu -- AttentionPairBias pair -> bias on sm_100a (tcgen05): the cubin build of ../apb/apb_pair_fwd_sm100.cuh for one
// (heads, d_pair): -DPF_NH=8 | 12 | 16 | 24 -DPF_DPW=128 | 256 | 384 | 512 (``sm100.apb_pair_fwd``). Block 192 threads, one per SM.
// SPDX-License-Identifier: Apache-2.0
#include "../apb/apb_pair_fwd_sm100.cuh"
#ifndef PF_NH
#define PF_NH 16
#endif
#ifndef PF_DPW
#define PF_DPW 128
#endif
extern "C" __global__ void __launch_bounds__(192, 1) apb_pair_fwd_sm100(const __grid_constant__ CUtensorMap tmz, const void* __restrict__ WSRC,
    const float* __restrict__ LPW, float ws, int wfp32, const bool* __restrict__ MASK, __nv_bfloat16* __restrict__ OUT, float2* __restrict__ ST,
    float* __restrict__ BU, int L, float eps, float neg, int NB) {
  pf100::pair_fwd_tc_body<PF_NH, PF_DPW>(tmz, WSRC, LPW, ws, wfp32, MASK, OUT, ST, BU, L, eps, neg, NB);
}
