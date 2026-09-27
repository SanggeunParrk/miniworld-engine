// pwa_compact.cuh -- key compaction for the PWA split path: idx[k] = k-th unmasked key j, cnt = {n, n_pad = round_up(n, 16)}.
// Masked keys only contribute exp(min - max) = 0 to a softmax row, so the contraction over j can run over the n valid keys alone.
// If every key is masked (the module's softmax is then uniform over all L), all keys are listed: the pair kernel's mask test turns
// every logit into the same minimum, which reproduces the uniform row. One CTA; everything stays on the device (graph-safe).
#pragma once
#include "sm80_common.cuh"

namespace a100 {

// posinv (optional, the backward): posinv[j] = k with idx[k] = j, or -1 for a masked key.
__global__ void __launch_bounds__(1024) pwa_compact_kernel(const uint8_t* __restrict__ mask, int L, int* __restrict__ idx,
                                                          int* __restrict__ cnt, int* __restrict__ posinv) {
  __shared__ int wsum[32];
  __shared__ int total;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const bool m = t < L && (!mask || mask[t]);
  const uint32_t b = __ballot_sync(0xffffffffu, m);
  if (lane == 0) wsum[warp] = __popc(b);
  __syncthreads();
  if (warp == 0) {
    int v = wsum[lane], x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
    wsum[lane] = x - v;                                   // exclusive prefix over warps
    if (lane == 31) total = x;
  }
  __syncthreads();
  const int n = total;
  if (n == 0) {
    if (t < L) { idx[t] = t; if (posinv) posinv[t] = t; }
  } else if (t < L) {
    const int k = wsum[warp] + __popc(b & ((1u << lane) - 1u));
    if (m) idx[k] = t;
    if (posinv) posinv[t] = m ? k : -1;
  }
  if (t == 0) {
    const int nn = n == 0 ? L : n;
    cnt[0] = nn;
#ifndef PWA_PAD
#define PWA_PAD 16
#endif
    cnt[1] = (nn + PWA_PAD - 1) / PWA_PAD * PWA_PAD;     // 16: one k16 step (pwa_ctr runs a partial last chunk)
  }
}

}  // namespace a100
