// pack_sm80.cuh -- every weight layout of the A100 TriMul kernels in one launch (called per forward / backward: no weight cache, so a
// captured CUDA graph repacks after an optimizer step).  One warp per item, fixed-order warp reductions (bit-identical replays):
//   items [0, 4 CH)          K1 row r: w1 (granule-major 64-row blocks, 0.5 W) and, in training, wdx row r (W unscaled, row-major)
//   items [4 CH, 4 CH + 128) W_og row c: wg3 = bf16(0.5 W_og diag gamma_in), sg / bg; training: wdx row 4 CH + c, sg_b1 / bg_b1
//   items [.. + 128)         W_o row c:  wo3 = bf16(0.5 W_o diag gamma_out), so / bo; training: wob1 = bf16(W_o diag gamma_out), so_b1 / bo_b1
// K1 row order: r = 64 step + 16 nw + n  ->  plane channel oc = 32 step + 8 nw + (n % 8), gate row for n < 8 (left = oc < CH).
#pragma once
#include <cuda_bf16.h>

namespace a100 {

struct PackParams {
  const __nv_bfloat16 *wl, *wlg, *wr, *wrg, *wg, *wo;   // [CH, 128] x 4 (strides below), [128, 128], [128, CH]
  int rs[4], ks[4];                                      // element strides of wl, wlg, wr, wrg: output channel, input channel
                                                         // (row-major: 128, 1; the bidirectional module stores them as [in, out]: 1, CH)
  const float *gi, *bi, *go, *bo;                        // LayerNorm affine: in [128], out [CH]
  __nv_bfloat16 *w1, *wg3, *wo3;                         // [4 CH * 128], [128 * 128], [128 * CH]
  __nv_bfloat16 *wdx, *wob1;                             // training only (else nullptr): [(4 CH + 128) * 128], [128 * CH]
  float* vec;                                            // [8][128]: so, bo, sg, bg, so_b1, bo_b1, sg_b1, bg_b1
  int CH;
};

__device__ __forceinline__ float pack_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__global__ void __launch_bounds__(256) pack_kernel(const PackParams p) {
  const int item = (int)(blockIdx.x * 8 + (threadIdx.x >> 5)), lane = threadIdx.x & 31, CH = p.CH;
  if (item < 4 * CH) {
    const int r = item, step = r >> 6, rr = r & 63, n = rr & 15, oc = 32 * step + 8 * (rr >> 4) + (n & 7);
    const int which = n < 8 ? (oc < CH ? 1 : 3) : (oc < CH ? 0 : 2);                  // wl, wlg, wr, wrg
    const __nv_bfloat16* src = (which == 0 ? p.wl : which == 1 ? p.wlg : which == 2 ? p.wr : p.wrg) +
                               (size_t)(oc < CH ? oc : oc - CH) * p.rs[which];
    const size_t ks = (size_t)p.ks[which];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int k = 4 * lane + e;
      const __nv_bfloat16 w = src[k * ks];
      p.w1[((size_t)(step * 16 + (k >> 3)) * 64 + rr) * 8 + (k & 7)] = __float2bfloat16(0.5f * __bfloat162float(w));
      if (p.wdx) p.wdx[(size_t)r * 128 + k] = w;
    }
  } else if (item < 4 * CH + 128) {
    const int c = item - 4 * CH;
    float s = 0.f, b = 0.f, s1 = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int k = 4 * lane + e;
      const __nv_bfloat16 w = p.wg[(size_t)c * 128 + k];
      const float wf = __bfloat162float(w);
      const __nv_bfloat16 h = __float2bfloat16(0.5f * wf * p.gi[k]);
      p.wg3[(size_t)c * 128 + k] = h;
      s += __bfloat162float(h);
      s1 += __bfloat162float(__float2bfloat16(wf * p.gi[k]));
      b = fmaf(wf, p.bi[k], b);
      if (p.wdx) p.wdx[(size_t)(4 * CH + c) * 128 + k] = w;
    }
    s = pack_warp_sum(s); s1 = pack_warp_sum(s1); b = pack_warp_sum(b);
    if (lane == 0) {
      p.vec[2 * 128 + c] = s; p.vec[3 * 128 + c] = 0.5f * b;
      p.vec[6 * 128 + c] = s1; p.vec[7 * 128 + c] = b;
    }
  } else if (item < 4 * CH + 256) {
    const int c = item - 4 * CH - 128;
    float s = 0.f, b = 0.f, s1 = 0.f;
    for (int k0 = 0; k0 < CH; k0 += 128)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int k = k0 + 4 * lane + e;
        const float wf = __bfloat162float(p.wo[(size_t)c * CH + k]);
        const __nv_bfloat16 h = __float2bfloat16(0.5f * wf * p.go[k]), h1 = __float2bfloat16(wf * p.go[k]);
        p.wo3[(size_t)c * CH + k] = h;
        if (p.wob1) p.wob1[(size_t)c * CH + k] = h1;
        s += __bfloat162float(h);
        s1 += __bfloat162float(h1);
        b = fmaf(wf, p.bo[k], b);
      }
    s = pack_warp_sum(s); s1 = pack_warp_sum(s1); b = pack_warp_sum(b);
    if (lane == 0) {
      p.vec[0 * 128 + c] = s; p.vec[1 * 128 + c] = 0.5f * b;
      p.vec[4 * 128 + c] = s1; p.vec[5 * 128 + c] = b;
    }
  }
}

}  // namespace a100
