// wgrad_sm80.cuh -- the small kernels around the front's / back's backward: the weights in the backward kernels' layouts, and the parameters' gradients from
// G = D^T [xh | 1].  Both replace a dozen tiny PyTorch launches (casts, transposes, concatenations, elementwise products) by one launch each.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

// ---- the weights of the backward: wt[p][c][o] = W_p[o][c] (p = q, k, v, g: the front's input gradient is D W, so the B operand is W^T), wbt[c][h] = Wb[h][c],
// wot[c][o] = Wo[o][c] (the back's backward da = dy Wo), gamma32 = the LayerNorm scale as fp32.  One launch, one thread per element.
struct BwdPackParams {
  const __nv_bfloat16* w[7];    // Wq, Wk, Wv, Wg: [128][128]; Wb: [4][128]; Wo: [128][128]; (unused)
  const void* gamma;            // [128] bf16 or fp32
  int gamma_bf16;
  __nv_bfloat16* wt;            // [4][128][128]
  __nv_bfloat16* wbt;           // [128][4]
  __nv_bfloat16* wot;           // [128][128]
  float* gamma32;               // [128]
};

__global__ void __launch_bounds__(256) bwd_pack_kernel(const BwdPackParams p) {
  const int i = blockIdx.x * 256 + threadIdx.x;                // 0 .. 6 x 16384 + 512 + 128
  if (i < 4 * 16384) {
    const int blk = i >> 14, o = (i >> 7) & 127, c = i & 127;   // read W_blk[o][c] (coalesced), write wt[blk][c][o]
    p.wt[(size_t)blk * 16384 + c * 128 + o] = p.w[blk][o * 128 + c];
  } else if (i < 5 * 16384) {
    const int j = i - 4 * 16384, o = j >> 7, c = j & 127;
    p.wot[c * 128 + o] = p.w[5][o * 128 + c];
  } else if (i < 5 * 16384 + 512) {
    const int j = i - 5 * 16384, h = j >> 7, c = j & 127;
    p.wbt[c * 4 + h] = p.w[4][h * 128 + c];
  } else if (i < 5 * 16384 + 512 + 128) {
    const int c = i - (5 * 16384 + 512);
    p.gamma32[c] = p.gamma_bf16 ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p.gamma)[c]) : reinterpret_cast<const float*>(p.gamma)[c];
  }
}

// ---- the parameters' gradients from G = D^T [xh | 1] (fp32 [516][136]; D = [dq | dk | dv | dg | db], column 128 = the column sums s):
//   dW[o][c] = G[o][c] gamma_c + s_o beta_c      (rows 0 .. 515 of the concatenated Wq | Wk | Wv | Wg | Wb, cast to the weights' dtype)
//   dgamma_c = sum_o W[o][c] G[o][c]              dbeta_c = sum_o W[o][c] s_o
// Blocks 0 .. 515: one row of dW each; blocks 516 .. 643: one column each of dgamma / dbeta (a fixed-order tree over o: deterministic).
struct WgradParams {
  const float* g;               // [516][136]
  const __nv_bfloat16* w[5];    // Wq, Wk, Wv, Wg [128][128], Wb [4][128]
  __nv_bfloat16* dw[5];         // gradients, same shapes (bf16: the parameters' dtype)
  const void* gamma;            // [128] bf16 or fp32 (the parameters' dtype)
  const void* beta;
  void* dgamma;                 // [128] in the same dtype
  void* dbeta;
  int ln_bf16;
};

DEVI float ld_param(const void* p, int bf16, int i) { return bf16 ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]) : reinterpret_cast<const float*>(p)[i]; }
DEVI void st_param(void* p, int bf16, int i, float v) {
  if (bf16) reinterpret_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v); else reinterpret_cast<float*>(p)[i] = v;
}

__global__ void __launch_bounds__(128) wgrad_finalize_kernel(const WgradParams p) {
  const int b = blockIdx.x, c = threadIdx.x;
  if (b < 516) {
    const int w = b < 512 ? b >> 7 : 4, r = b < 512 ? b & 127 : b - 512;
    p.dw[w][r * 128 + c] = __float2bfloat16_rn(p.g[(size_t)b * 136 + c] * ld_param(p.gamma, p.ln_bf16, c) + p.g[(size_t)b * 136 + 128] * ld_param(p.beta, p.ln_bf16, c));
    return;
  }
  const int col = b - 516;                                     // dgamma / dbeta of channel `col`: thread t sums rows t, t + 128, ...
  __shared__ float red[2][128];
  float a = 0.f, s = 0.f;
  for (int o = c; o < 516; o += 128) {
    const float wv = __bfloat162float(p.w[o < 512 ? o >> 7 : 4][(o < 512 ? o & 127 : o - 512) * 128 + col]);
    a = fmaf(wv, p.g[(size_t)o * 136 + col], a);
    s = fmaf(wv, p.g[(size_t)o * 136 + 128], s);
  }
  red[0][c] = a;
  red[1][c] = s;
  __syncthreads();
#pragma unroll
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (c < stride) { red[0][c] += red[0][c + stride]; red[1][c] += red[1][c + stride]; }
    __syncthreads();
  }
  if (c == 0) {
    st_param(p.dgamma, p.ln_bf16, col, red[0][0]);
    st_param(p.dbeta, p.ln_bf16, col, red[1][0]);
  }
}

}  // namespace a100
