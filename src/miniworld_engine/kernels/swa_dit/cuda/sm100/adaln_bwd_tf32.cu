// adaln_bwd_tf32.cu — the RMS-adaLN backward of the fp32 SWA atom block's two norms on sm_100a, after the GEMM that gives dy (the FFN's
// dy = [da | db] Wu, the projections' dx = dP W): the Triton kernels' elementwise tails, one CUDA kernel for both:
//   xh = x rstd, rstd = 1 / sqrt(mean(x^2) + eps);  dxh = dy (1 + scale);  dx_out = res + rstd (dxh - xh mean(dxh xh))
//   d shift = sum_aug dy,  d scale = sum_aug dy xh,  [d gate = sum_aug res gin   (the FFN's gate_f: res = dq2, gin = ffn)],
//   [d gate_a = sum_aug out att   (the out-projection's gate: out = this kernel's dq1, att = the saved raw out-projection; column att_col)]
// -> dmod columns out_col (shift), out_col + C (scale), out_col + 2 C (gate, only when gin != null); scale is read from the same column
// out_col + C of mod. (FFN: out_col = 3C, x = q1, res = dq2; qkvg: out_col = 0, x = q, res = dq1.)
// One block per (batch element, atom): its eight warps walk the A augments (row n = a B + b), a warp = one 128-channel row (lane = four
// channels, 512 contiguous bytes per load), the row sums by warp shuffles; the augment sums stay in registers and meet in shared memory,
// so every dmod element is written once, by one thread, in a fixed order: no atomics, bit-reproducible.
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>

constexpr int C = 128, MODW = 6 * C, NWARP = 8;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

extern "C" __global__ void __launch_bounds__(NWARP * 32)
swa_adaln_bwd_tf32_sm100(const float* __restrict__ X, const float* __restrict__ DY, const float* __restrict__ RES,
                         const float* __restrict__ GIN, const float* __restrict__ MOD, float* __restrict__ OUT,
                         float* __restrict__ DMOD, int S, int A, int Bn, float eps, int out_col,
                         const float* __restrict__ ATT, int att_col) {
  __shared__ float red[NWARP][4][C];
  const int bs = (int)blockIdx.x, b = bs / S, s = bs % S;                  // the modulation row of this atom: b S + s = bs
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float4 sc4 = *reinterpret_cast<const float4*>(MOD + (size_t)bs * MODW + out_col + C + 4 * lane);
  float4 dsh = make_float4(0.f, 0.f, 0.f, 0.f), dsc = dsh, dga = dsh, dat = dsh;
  for (int a = warp; a < A; a += NWARP) {
    const size_t o = (((size_t)a * Bn + b) * S + s) * C + 4 * lane;
    const float4 q = *reinterpret_cast<const float4*>(X + o), dy = *reinterpret_cast<const float4*>(DY + o);
    const float4 d2 = *reinterpret_cast<const float4*>(RES + o);
    const float rstd = 1.f / sqrtf(warp_sum(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) * (1.f / C) + eps);
    const float4 xh = make_float4(q.x * rstd, q.y * rstd, q.z * rstd, q.w * rstd);
    const float4 dxh = make_float4(dy.x * (1.f + sc4.x), dy.y * (1.f + sc4.y), dy.z * (1.f + sc4.z), dy.w * (1.f + sc4.w));
    const float m = warp_sum(dxh.x * xh.x + dxh.y * xh.y + dxh.z * xh.z + dxh.w * xh.w) * (1.f / C);
    float4 r;
    r.x = d2.x + rstd * (dxh.x - xh.x * m); r.y = d2.y + rstd * (dxh.y - xh.y * m);
    r.z = d2.z + rstd * (dxh.z - xh.z * m); r.w = d2.w + rstd * (dxh.w - xh.w * m);
    *reinterpret_cast<float4*>(OUT + o) = r;
    dsh.x += dy.x; dsh.y += dy.y; dsh.z += dy.z; dsh.w += dy.w;
    dsc.x += dy.x * xh.x; dsc.y += dy.y * xh.y; dsc.z += dy.z * xh.z; dsc.w += dy.w * xh.w;
    if (GIN != nullptr) {
      const float4 ff = *reinterpret_cast<const float4*>(GIN + o);
      dga.x += d2.x * ff.x; dga.y += d2.y * ff.y; dga.z += d2.z * ff.z; dga.w += d2.w * ff.w;
    }
    if (ATT != nullptr) {
      const float4 at = *reinterpret_cast<const float4*>(ATT + o);
      dat.x += r.x * at.x; dat.y += r.y * at.y; dat.z += r.z * at.z; dat.w += r.w * at.w;
    }
  }
  *reinterpret_cast<float4*>(&red[warp][0][4 * lane]) = dsh;
  *reinterpret_cast<float4*>(&red[warp][1][4 * lane]) = dsc;
  *reinterpret_cast<float4*>(&red[warp][2][4 * lane]) = dga;
  *reinterpret_cast<float4*>(&red[warp][3][4 * lane]) = dat;
  __syncthreads();
  const int nm = GIN != nullptr ? 3 : 2;
  for (int i = threadIdx.x; i < nm * C; i += NWARP * 32) {                 // i = 128 m + c: shift, scale, gate
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < NWARP; ++w) t += red[w][i / C][i % C];
    DMOD[(size_t)bs * MODW + out_col + i] = t;
  }
  if (ATT != nullptr) {
    for (int c = threadIdx.x; c < C; c += NWARP * 32) {
      float t = 0.f;
#pragma unroll
      for (int w = 0; w < NWARP; ++w) t += red[w][3][c];
      DMOD[(size_t)bs * MODW + att_col + c] = t;
    }
  }
}
